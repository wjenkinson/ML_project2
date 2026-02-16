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
import argparse
import math

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
    """Rigid-wall velocity constraint based on predicted displacement.

    Rigid walls (type 3) should ideally have zero velocity. We approximate
    velocity using the displacement between positions at t (``batch.pos``)
    and predicted positions at t+1 (``preds``), and penalise motion of type-3
    atoms.
    """

    atom_type = batch.atom_type.to(preds.device)
    pos_t = batch.pos.to(preds.device)

    wall_mask = atom_type == 3
    if not torch.any(wall_mask):
        return torch.zeros((), dtype=preds.dtype, device=preds.device)

    disp = preds - pos_t
    wall_disp = disp[wall_mask]

    if wall_disp.numel() == 0:
        return torch.zeros((), dtype=preds.dtype, device=preds.device)

    return wall_disp.pow(2).sum(dim=-1).mean()


def interface_sharpness_loss(preds: torch.Tensor, batch) -> torch.Tensor:
    """Stub for interface sharpness constraint.

    Intended to penalise overly mixed interfaces between phases (types 1 and 2).
    This stub returns zero for now and will be implemented later.
    """

    # Placeholder: no contribution yet.
    return torch.zeros((), dtype=preds.dtype, device=preds.device)


def density_constraint_loss(
    preds: torch.Tensor,
    batch,
    radius: float,
    rho_min: float = 950.0,
    rho_max: float = 1050.0,
) -> torch.Tensor:
    """SPH-style density constraint using a 2D cubic spline kernel.

    Uses predicted positions ``preds`` to compute per-particle densities and
    penalises values outside [rho_min, rho_max].
    """

    pos = preds[:, :2]
    edge_index = batch.edge_index
    row = edge_index[0]
    col = edge_index[1]

    h = radius / 2.0
    if h <= 0.0:
        return torch.zeros((), dtype=preds.dtype, device=preds.device)

    # Particle mass consistent with physics_checks: rho0 * L0^2 with
    # rho0=1000, L0=0.001.
    mass_per_particle = 0.001

    # Base density from self-contribution W(0, h).
    sigma = 10.0 / (7.0 * math.pi * h * h)
    w0 = sigma * 1.0

    N = pos.shape[0]
    rho = torch.full((N,), mass_per_particle * w0, dtype=preds.dtype, device=preds.device)

    # Distances between neighbours from predicted positions.
    diff = pos[row] - pos[col]
    dist = torch.sqrt(torch.sum(diff * diff, dim=-1) + 1e-12)

    q = dist / h
    w = torch.zeros_like(dist)

    mask1 = (q >= 0.0) & (q < 1.0)
    if mask1.any():
        q1 = q[mask1]
        w[mask1] = 1.0 - 1.5 * q1 * q1 + 0.75 * q1 * q1 * q1

    mask2 = (q >= 1.0) & (q < 2.0)
    if mask2.any():
        q2 = q[mask2]
        w[mask2] = 0.25 * (2.0 - q2) ** 3

    w = sigma * w

    contrib = mass_per_particle * w
    rho.scatter_add_(0, row, contrib)

    under = torch.relu(torch.as_tensor(rho_min, dtype=preds.dtype, device=preds.device) - rho)
    over = torch.relu(rho - torch.as_tensor(rho_max, dtype=preds.dtype, device=preds.device))

    return (under.pow(2) + over.pow(2)).mean()


def train(
    epochs: int = 10,
    batch_size: int = 1,
    learning_rate: float = 1e-3,
    radius: float = NEIGHBOR_RADIUS,
    lambda_mass: float = 0.0,
    lambda_rigid: float = 0.0,
    lambda_interface: float = 0.0,
    lambda_density: float = 0.0,
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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

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
            interface_loss = interface_sharpness_loss(preds, batch)
            density_loss = density_constraint_loss(
                preds,
                batch,
                radius=radius,
                rho_min=950.0,
                rho_max=1050.0,
            )

            loss = (
                data_loss
                + lambda_mass * mass_loss
                + lambda_interface * interface_loss
                + lambda_density * density_loss
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
                interface_loss = interface_sharpness_loss(preds, batch)
                density_loss = density_constraint_loss(
                    preds,
                    batch,
                    radius=radius,
                    rho_min=950.0,
                    rho_max=1050.0,
                )

                loss = (
                    data_loss
                    + lambda_mass * mass_loss
                    + lambda_interface * interface_loss
                    + lambda_density * density_loss
                )

                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={avg_train_loss:.6f} | "
            f"val_loss={avg_val_loss:.6f} | "
            f"lambda_mass={lambda_mass} | "
            f"lambda_rigid={lambda_rigid} | "
            f"lambda_interface={lambda_interface} | "
            f"lambda_density={lambda_density}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_path)
            print(f"  Saved new best PINN model to {model_path}")

    print("Training complete.")
    print(f"Best validation loss: {best_val_loss:.6f}")
def main() -> None:
    """Run one or more training configurations for comparison.

    By default, all configurations are trained. You can restrict this using
    CLI flags, for example:

    - ``python -m src.train_pinn -c vanilla``
    - ``python -m src.train_pinn -c vanilla mass density``
    """

    all_configs = [
        {"name": "vanilla", "lambda_mass": 0.0, "lambda_rigid": 0.0, "lambda_interface": 0.0, "lambda_density": 0.0},
        {"name": "mass", "lambda_mass": 1e-3, "lambda_rigid": 0.0, "lambda_interface": 0.0, "lambda_density": 0.0},
        {"name": "rigid", "lambda_mass": 0.0, "lambda_rigid": 0.0, "lambda_interface": 0.0, "lambda_density": 0.0},
        {"name": "interface", "lambda_mass": 0.0, "lambda_rigid": 0.0, "lambda_interface": 1e-4, "lambda_density": 0.0},
        {"name": "density", "lambda_mass": 0.0, "lambda_rigid": 0.0, "lambda_interface": 0.0, "lambda_density": 1e-5},
        {"name": "all", "lambda_mass": 1e-3, "lambda_rigid": 0.0, "lambda_interface": 1e-4, "lambda_density": 1e-5},
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
            "If omitted, all configurations are trained. "
            "Choices: vanilla, mass, rigid, interface, density, all."
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
            lambda_mass=cfg["lambda_mass"],
            lambda_rigid=cfg["lambda_rigid"],
            lambda_interface=cfg["lambda_interface"],
            lambda_density=cfg["lambda_density"],
            experiment_name=cfg["name"],
        )


if __name__ == "__main__":  # pragma: no cover - CLI guard
    main()
