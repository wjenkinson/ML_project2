"""Train a GNN with relative-position edge features and displacement output.

The model predicts particle DISPLACEMENTS (not absolute positions) using:
- Node features: atom type only (translation-invariant)
- Edge features: relative position vectors [Δx, Δy, Δz, ||r||]

At inference: pos(t+1) = pos(t) + predicted_displacement

Training loss combines:
- Per-node MSE on displacements and velocities for non-fluid atoms (solid/wall/piston).
- Grid-based density and velocity field losses for fluid atoms.
- An optional floor constraint.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing

from .graph_dataset import LammpsGraphDataset, NODE_FEATURE_DIM
from .grid_fields import grid_density_loss, grid_velocity_loss, make_grid
from .preprocess_data import NEIGHBOR_RADIUS


class EdgeConvLayer(MessagePassing):
    """Message passing layer that uses edge features (relative positions).

    For each edge (i -> j), the message is computed from:
    - Source node embedding
    - Edge attributes (relative position + distance)

    Messages are summed at each receiver node and combined with a residual.
    """

    def __init__(self, hidden_channels: int, edge_channels: int = 4) -> None:
        super().__init__(aggr="mean")
        # MLP that processes [src_embedding, edge_attr] -> message
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_channels + edge_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        # Update MLP after aggregation
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

    def forward(self, x, edge_index, edge_attr):
        # x: (N, hidden), edge_attr: (E, edge_channels)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        # Combine with original embedding (residual-style)
        out = self.update_mlp(torch.cat([x, out], dim=-1))
        return x + out  # residual connection

    def message(self, x_j, edge_attr):
        # x_j: source node features, edge_attr: relative positions
        return self.edge_mlp(torch.cat([x_j, edge_attr], dim=-1))


class SimpleGnnPredictor(nn.Module):
    """GNN for predicting particle displacements and velocities.

    Node features: [type_id] (1 channel) - position-independent
    Edge features: [Δx, Δy, Δz, ||r||] (4 channels) - relative geometry
    Output: displacement (dx, dy, dz) + velocity (vx, vy) per particle

    Architecture:
    1. Embed type_id to hidden dimension
    2. Apply EdgeConvLayers that use relative positions
    3. Two heads: displacement (3D) and velocity (2D)
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 128,
        edge_channels: int = 4,
        num_layers: int = 2,
    ) -> None:
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        # Embed node features (type_id) to hidden dimension
        self.node_embed = nn.Linear(in_channels, hidden_channels)

        # Message passing layers with edge features
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(EdgeConvLayer(hidden_channels, edge_channels))

        # Head 1: predict displacement (dx, dy, dz)
        self.out_lin = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 3),
        )

        # Head 2: predict next-step velocity (vx, vy)
        self.vel_lin = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 2),
        )

    def forward(self, data):  # type: ignore[override]
        x = data.x  # (N, 1) - type_id
        edge_index = data.edge_index
        edge_attr = data.edge_attr  # (E, 4) - relative positions + distance

        # Embed node features
        x = self.node_embed(x)
        x = F.relu(x)

        # Message passing with edge features
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)

        # Predict displacement and velocity from shared backbone
        displacement = self.out_lin(x)
        velocity = self.vel_lin(x)
        return displacement, velocity


# Floor constraint: rigid horizontal plate at this y-coordinate.
FLOOR_Y = -0.07


def floor_constraint_loss(
    displacement_preds: torch.Tensor,
    batch,
    floor_y: float = FLOOR_Y,
) -> torch.Tensor:
    """Penalize fluid nodes (type 1) that fall below the floor.

    There is a rigid horizontal plate at y = floor_y; nothing should go below it.
    """

    # Compute absolute positions from displacement
    abs_pos = batch.pos + displacement_preds
    y_pos = abs_pos[:, 1]

    # Get fluid mask (type 1)
    atom_type = batch.atom_type
    fluid_mask = atom_type == 1

    # Penalize fluid nodes below the floor
    y_fluid = y_pos[fluid_mask]
    violation = torch.relu(floor_y - y_fluid)  # positive if below floor

    if violation.numel() == 0:
        return torch.zeros((), dtype=displacement_preds.dtype, device=displacement_preds.device)

    return violation.pow(2).mean()


def boundary_node_weights(
    atom_type: torch.Tensor,
    lambda_boundary: float = 5.0,
) -> torch.Tensor:
    """Per-node weights that upweight solid (type 2) and wall (type 3) nodes.

    Solid and wall atoms receive weight ``lambda_boundary``; all others
    (fluid, piston) receive weight ``1.0``.
    """

    weights = torch.ones(atom_type.shape[0], device=atom_type.device)
    weights[(atom_type == 2) | (atom_type == 3) | (atom_type == 4)] = lambda_boundary
    return weights


def weighted_mse(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """MSE loss with per-node weights.  *weights* shape is (N,)."""

    se = (pred - target).pow(2).mean(dim=-1)  # (N,)
    return (se * weights).mean()


def train(
    epochs: int = 5,
    batch_size: int = 1,
    learning_rate: float = 3e-3,
    radius: float = NEIGHBOR_RADIUS,
    lambda_floor: float = 0.0,
    lambda_vel: float = 1.0,
    lambda_boundary: float = 5.0,
    hidden_channels: int = 128,
    num_layers: int = 2,
    experiment_name: str | None = None,
) -> float:
    project_root = Path(__file__).parent.parent
    print(f"Project root: {project_root}")

    device = torch.device("cpu")
    print(f"Using device: {device}")

    torch.manual_seed(0)

    train_dataset = LammpsGraphDataset(split="train", radius=radius)
    val_dataset = LammpsGraphDataset(split="val", radius=radius)

    if len(train_dataset) == 0:
        print("Train dataset is empty; nothing to train on.")
        return float("inf")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model_hparams = {
        "in_channels": NODE_FEATURE_DIM,
        "hidden_channels": hidden_channels,
        "edge_channels": 4,
        "num_layers": num_layers,
    }
    model = SimpleGnnPredictor(**model_hparams).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True)

    if experiment_name:
        model_path = output_dir / f"simple_pinn_predictor_{experiment_name}.pt"
    else:
        model_path = output_dir / "simple_pinn_predictor.pt"

    # Pre-compute grid centres for field losses
    grid_x, grid_y = make_grid(device)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)

            disp_preds, vel_preds = model(batch)
            disp_targets = batch.y.to(device)
            vel_targets = batch.vel_target.to(device)
            atom_type = batch.atom_type

            # Predicted and GT next-step positions
            pred_pos = batch.pos + disp_preds
            gt_pos = batch.pos + disp_targets

            # --- Non-fluid per-node loss (solid / wall / piston) ---
            non_fluid = (atom_type == 2) | (atom_type == 3) | (atom_type == 4)
            if non_fluid.any():
                nf_disp_loss = criterion(disp_preds[non_fluid], disp_targets[non_fluid])
                nf_vel_loss = criterion(vel_preds[non_fluid], vel_targets[non_fluid])
            else:
                nf_disp_loss = torch.zeros((), device=device)
                nf_vel_loss = torch.zeros((), device=device)

            # --- Fluid grid field losses ---
            density_loss = grid_density_loss(pred_pos, gt_pos, atom_type, grid_x, grid_y)
            velocity_loss = grid_velocity_loss(
                pred_pos, vel_preds, gt_pos, vel_targets, atom_type, grid_x, grid_y,
            )

            floor_loss = floor_constraint_loss(disp_preds, batch)

            loss = (
                lambda_boundary * (nf_disp_loss + lambda_vel * nf_vel_loss)
                + density_loss + lambda_vel * velocity_loss
                + lambda_floor * floor_loss
            )

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        avg_train_loss = train_loss / max(num_batches, 1)

        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                disp_preds, vel_preds = model(batch)
                disp_targets = batch.y.to(device)
                vel_targets = batch.vel_target.to(device)
                atom_type = batch.atom_type

                pred_pos = batch.pos + disp_preds
                gt_pos = batch.pos + disp_targets

                non_fluid = (atom_type == 2) | (atom_type == 3) | (atom_type == 4)
                if non_fluid.any():
                    nf_disp_loss = criterion(disp_preds[non_fluid], disp_targets[non_fluid])
                    nf_vel_loss = criterion(vel_preds[non_fluid], vel_targets[non_fluid])
                else:
                    nf_disp_loss = torch.zeros((), device=device)
                    nf_vel_loss = torch.zeros((), device=device)

                density_loss = grid_density_loss(pred_pos, gt_pos, atom_type, grid_x, grid_y)
                velocity_loss = grid_velocity_loss(
                    pred_pos, vel_preds, gt_pos, vel_targets, atom_type, grid_x, grid_y,
                )

                floor_loss = floor_constraint_loss(disp_preds, batch)

                loss = (
                    lambda_boundary * (nf_disp_loss + lambda_vel * nf_vel_loss)
                    + density_loss + lambda_vel * velocity_loss
                    + lambda_floor * floor_loss
                )

                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={avg_train_loss:.6f} | "
            f"val_loss={avg_val_loss:.6f} | "
            f"lambda_floor={lambda_floor} | "
            f"lambda_vel={lambda_vel} | "
            f"lambda_boundary={lambda_boundary}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_hparams": model_hparams,
                },
                model_path,
            )
            print(f"  Saved new best PINN model to {model_path}")

    print("Training complete.")
    print(f"Best validation loss: {best_val_loss:.6f}")
    return best_val_loss
def main() -> None:
    """Run one or more training configurations for comparison.

    By default, both configurations are trained. You can restrict this using
    CLI flags, for example:

    - ``python -m src.train_pinn -c vanilla``
    - ``python -m src.train_pinn -c density``
    """

    all_configs = [
        {"name": "vanilla", "lambda_floor": 0.0, "lambda_boundary": 0.0},
        {"name": "floor", "lambda_floor": 10.0, "lambda_boundary": 0.0},
        {"name": "boundary", "lambda_floor": 0.0, "lambda_boundary": 5.0},
    ]

    parser = argparse.ArgumentParser(description="Train physics-informed GNN configurations.")
    parser.add_argument(
        "-c",
        "--config",
        dest="configs",
        nargs="+",
        choices=[cfg["name"] for cfg in all_configs],
        help=(
            "Name(s) of configurations to train. "
            "If omitted, both configurations are trained. "
            "Choices: vanilla, density."
        ),
    )

    args = parser.parse_args()

    if args.configs:
        configs = [cfg for cfg in all_configs if cfg["name"] in args.configs]
    else:
        configs = all_configs

    if not configs:
        print("No matching configurations to train (check --config names).")
        return

    for cfg in configs:
        print("\n" + "=" * 80)
        print(f"Training configuration: {cfg['name']}")
        print("=" * 80)

        train(
            lambda_floor=cfg.get("lambda_floor", 0.0),
            lambda_vel=cfg.get("lambda_vel", 1.0),
            lambda_boundary=cfg.get("lambda_boundary", 5.0),
            hidden_channels=cfg.get("hidden_channels", 128),
            num_layers=cfg.get("num_layers", 2),
            experiment_name=cfg["name"],
        )


if __name__ == "__main__":  # pragma: no cover - CLI guard
    main()
