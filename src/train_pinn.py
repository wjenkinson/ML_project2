"""Train a simple GNN baseline with a mass-conservation style constraint.

This adapts the Project 1 ``train_GNN`` script to the current repo layout and
2D SPH data. The model predicts next-step particle positions given current
positions and atom types, and the training loss combines:

- A data term (MSE on positions).
- A simple "mass" term based on center-of-mass consistency between predicted
  and target positions (treating each particle as unit mass).
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv

from .graph_dataset import LammpsGraphDataset
from .preprocess_data import NEIGHBOR_RADIUS


class SimpleGnnPredictor(nn.Module):
    """Simple 2-layer GNN for next-step particle position prediction.

    Node features: [x, y, z, type_id]
    Target: positions at t+1 for each particle (x, y, z).
    """

    def __init__(
        self,
        in_channels: int = 4,
        hidden_channels: int = 64,
        num_layers: int = 2,
    ) -> None:
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        self.out_lin = nn.Linear(hidden_channels, 3)  # predict (x, y, z) at t+1

    def forward(self, data):  # type: ignore[override]
        x, edge_index = data.x, data.edge_index

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)

        pred = self.out_lin(x)
        return pred


def mass_center_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Center-of-mass consistency loss as a simple mass proxy.

    With unit mass per particle, the center of mass is the mean position. This
    loss encourages the predicted center of mass to match the target one.
    """

    com_pred = preds.mean(dim=0)
    com_true = targets.mean(dim=0)
    return F.mse_loss(com_pred, com_true)


def rigid_wall_velocity_loss(preds: torch.Tensor, batch) -> torch.Tensor:
    """Stub for rigid-wall velocity constraint.

    Rigid walls (type 3) should ideally have zero velocity. This stub returns
    zero for now and will be implemented once velocity information is threaded
    through the dataset.
    """

    # Placeholder: no contribution yet.
    return torch.zeros((), dtype=preds.dtype, device=preds.device)


def interface_sharpness_loss(preds: torch.Tensor, batch) -> torch.Tensor:
    """Stub for interface sharpness constraint.

    Intended to penalise overly mixed interfaces between phases (types 1 and 2).
    This stub returns zero for now and will be implemented later.
    """

    # Placeholder: no contribution yet.
    return torch.zeros((), dtype=preds.dtype, device=preds.device)


def neighbor_constraint_loss(
    preds: torch.Tensor,
    batch,
    max_neighbors: float = 8.0,
    max_diff_type_neighbors: float = 8.0,
    radius: float | None = None,
) -> torch.Tensor:
    edge_index = batch.edge_index
    atom_type = batch.atom_type.to(preds.device)
    row = edge_index[0]
    col = edge_index[1]

    pos_i = preds[row]
    pos_j = preds[col]
    d2 = torch.sum((pos_i - pos_j) ** 2, dim=-1)

    if radius is not None:
        r = torch.as_tensor(radius, dtype=preds.dtype, device=preds.device)
    else:
        r = torch.sqrt(d2.mean() + 1e-8)

    sigma2 = (0.5 * r) ** 2 + 1e-12
    weights = torch.exp(-d2 / (2.0 * sigma2))

    same_type = atom_type[row] == atom_type[col]
    diff_mask = (~same_type).to(preds.dtype)

    N = preds.shape[0]
    soft_all = torch.zeros(N, device=preds.device, dtype=preds.dtype)
    soft_diff = torch.zeros(N, device=preds.device, dtype=preds.dtype)
    soft_all.scatter_add_(0, row, weights)
    soft_diff.scatter_add_(0, row, weights * diff_mask)

    over_all = torch.relu(soft_all - max_neighbors)
    over_diff = torch.relu(soft_diff - max_diff_type_neighbors)

    return over_all.pow(2).mean() + over_diff.pow(2).mean()


def train(
    epochs: int = 10,
    batch_size: int = 1,
    learning_rate: float = 1e-3,
    radius: float = NEIGHBOR_RADIUS,
    lambda_mass: float = 0.0,
    lambda_neighbors: float = 0.0,
    lambda_rigid: float = 0.0,
    lambda_interface: float = 0.0,
    experiment_name: str | None = None,
) -> None:
    project_root = Path(__file__).parent.parent
    print(f"Project root: {project_root}")

    device = torch.device("cpu")
    print(f"Using device: {device}")

    torch.manual_seed(0)

    train_dataset = LammpsGraphDataset(split="train", radius=radius)
    val_dataset = LammpsGraphDataset(split="val", radius=radius)

    if len(train_dataset) == 0:
        print("Train dataset is empty; nothing to train on.")
        return

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = SimpleGnnPredictor(in_channels=4, hidden_channels=64, num_layers=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True)

    if experiment_name:
        model_path = output_dir / f"simple_pinn_predictor_{experiment_name}.pt"
    else:
        model_path = output_dir / "simple_pinn_predictor.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)

            preds = model(batch)  # (total_nodes_in_batch, 3)
            targets = batch.y.to(device)  # same shape

            data_loss = criterion(preds, targets)
            mass_loss = mass_center_loss(preds, targets)
            neighbor_loss = neighbor_constraint_loss(
                preds,
                batch,
                max_neighbors=8.0,
                max_diff_type_neighbors=8.0,
                radius=radius,
            )
            rigid_loss = rigid_wall_velocity_loss(preds, batch)
            interface_loss = interface_sharpness_loss(preds, batch)

            loss = (
                data_loss
                + lambda_mass * mass_loss
                + lambda_neighbors * neighbor_loss
                + lambda_rigid * rigid_loss
                + lambda_interface * interface_loss
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
                preds = model(batch)
                targets = batch.y.to(device)

                data_loss = criterion(preds, targets)
                mass_loss = mass_center_loss(preds, targets)
                neighbor_loss = neighbor_constraint_loss(
                    preds,
                    batch,
                    max_neighbors=8.0,
                    max_diff_type_neighbors=8.0,
                    radius=radius,
                )
                rigid_loss = rigid_wall_velocity_loss(preds, batch)
                interface_loss = interface_sharpness_loss(preds, batch)

                loss = (
                    data_loss
                    + lambda_mass * mass_loss
                    + lambda_neighbors * neighbor_loss
                    + lambda_rigid * rigid_loss
                    + lambda_interface * interface_loss
                )

                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={avg_train_loss:.6f} | "
            f"val_loss={avg_val_loss:.6f} | "
            f"lambda_mass={lambda_mass} | "
            f"lambda_neighbors={lambda_neighbors} | "
            f"lambda_rigid={lambda_rigid} | "
            f"lambda_interface={lambda_interface}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_path)
            print(f"  Saved new best PINN model to {model_path}")

    print("Training complete.")
    print(f"Best validation loss: {best_val_loss:.6f}")
def main() -> None:
    """Run a sequence of training configurations for comparison.

    Configurations:
    - vanilla: data loss only.
    - mass: data + mass-center loss.
    - neighbors: data + neighbour loss.
    - rigid: data + (stub) rigid-wall loss.
    - interface: data + (stub) interface-sharpness loss.
    - all: data + all available physics terms.
    """

    configs = [
        {"name": "vanilla", "lambda_mass": 0.0, "lambda_neighbors": 0.0, "lambda_rigid": 0.0, "lambda_interface": 0.0},
        {"name": "mass", "lambda_mass": 1e-3, "lambda_neighbors": 0.0, "lambda_rigid": 0.0, "lambda_interface": 0.0},
        {"name": "neighbors", "lambda_mass": 0.0, "lambda_neighbors": 1e-4, "lambda_rigid": 0.0, "lambda_interface": 0.0},
        {"name": "rigid", "lambda_mass": 0.0, "lambda_neighbors": 0.0, "lambda_rigid": 1e-4, "lambda_interface": 0.0},
        {"name": "interface", "lambda_mass": 0.0, "lambda_neighbors": 0.0, "lambda_rigid": 0.0, "lambda_interface": 1e-4},
        {"name": "all", "lambda_mass": 1e-3, "lambda_neighbors": 1e-4, "lambda_rigid": 1e-4, "lambda_interface": 1e-4},
    ]

    for cfg in configs:
        print("\n" + "=" * 80)
        print(f"Training configuration: {cfg['name']}")
        print("=" * 80)

        train(
            lambda_mass=cfg["lambda_mass"],
            lambda_neighbors=cfg["lambda_neighbors"],
            lambda_rigid=cfg["lambda_rigid"],
            lambda_interface=cfg["lambda_interface"],
            experiment_name=cfg["name"],
        )


if __name__ == "__main__":  # pragma: no cover - CLI guard
    main()
