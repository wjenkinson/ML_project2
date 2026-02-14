"""Generate next-step sequence predictions using the trained GNN model.

This script:

- Loads the best ``SimpleGnnPredictor`` checkpoint from ``output/``.
- Runs it over the validation ``LammpsGraphDataset`` to predict positions at
  t+1 given positions at t.
- Saves ground-truth and predicted positions for each (t, t+1) pair to a
  torch file in ``output/``, for consumption by ``post_videos.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import argparse

import torch
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data

from .graph_dataset import LammpsGraphDataset
from .preprocess_data import NEIGHBOR_RADIUS
from .train_pinn import SimpleGnnPredictor


def generate_gnn_predictions(
    project_root: Path,
    experiment_name: str | None = None,
    radius: float = NEIGHBOR_RADIUS,
) -> Path | None:
    """Run the trained GNN on the validation split and save predictions.

    Returns the path to the saved predictions file, or None if no predictions
    could be generated.
    """

    device = torch.device("cpu")

    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True)

    if experiment_name:
        ckpt_path = output_dir / f"simple_pinn_predictor_{experiment_name}.pt"
    else:
        ckpt_path = output_dir / "simple_pinn_predictor.pt"

    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}. Run train_pinn.py for this configuration first.")
        return None

    # Validation dataset
    val_dataset = LammpsGraphDataset(split="val", radius=radius)
    if len(val_dataset) == 0:
        print("Validation dataset is empty; nothing to predict.")
        return None

    if not val_dataset.pairs:
        print("Validation dataset has no frame pairs; nothing to predict.")
        return None

    # Load model
    model = SimpleGnnPredictor(in_channels=4, hidden_channels=64, num_layers=2).to(device)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Reconstruct ordered list of frame names from the stored pairs.
    # pairs[i] = (name_t, name_tp1) with names sorted by timestep.
    first_name = val_dataset.pairs[0][0]
    frame_names = [first_name] + [tp1 for (_t, tp1) in val_dataset.pairs]

    # Assume atom types are constant across frames; take from first frame.
    first_frame = val_dataset.frames[first_name]
    atom_type = first_frame["atom_type"].to(device)

    # Start autoregressive rollout from the ground-truth positions of the first frame.
    pos_current = first_frame["pos"].to(device)

    pairs: List[Dict[str, torch.Tensor | str | int]] = []

    with torch.no_grad():
        for step in range(len(frame_names) - 1):
            name_t = frame_names[step]
            name_tp1 = frame_names[step + 1]

            # Ground-truth next-frame positions for benchmarking.
            frame_tp1 = val_dataset.frames[name_tp1]
            pos_true_next = frame_tp1["pos"].to(device)

            # Build node features from current (possibly predicted) positions.
            x = torch.cat(
                [
                    pos_current,
                    atom_type.to(dtype=torch.float32).unsqueeze(-1),
                ],
                dim=-1,
            )

            # Build radius-based neighbourhood graph on the current positions.
            coords = pos_current.cpu().numpy()
            nbrs = NearestNeighbors(radius=radius, algorithm="ball_tree")
            nbrs.fit(coords)
            indices = nbrs.radius_neighbors(return_distance=False)

            row: List[int] = []
            col: List[int] = []
            for i, neigh in enumerate(indices):
                for j in neigh:
                    if i == j:
                        continue  # no self-loops
                    row.append(i)
                    col.append(j)

            edge_index = torch.tensor([row, col], dtype=torch.long, device=device)

            data_step = Data(x=x, edge_index=edge_index)

            # One-step prediction from current positions.
            preds_next = model(data_step)  # (N, 3)

            # Store CPU copies for serialization.
            pos_t_cpu = pos_current.detach().cpu()
            pos_tp1_cpu = pos_true_next.detach().cpu()
            preds_cpu = preds_next.detach().cpu()
            atom_type_cpu = atom_type.detach().cpu()

            pairs.append(
                {
                    "name_t": name_t,
                    "name_tp1": name_tp1,
                    "pos_t": pos_t_cpu,
                    "pos_tp1_true": pos_tp1_cpu,
                    "pos_tp1_pred": preds_cpu,
                    "atom_type": atom_type_cpu,
                }
            )

            # Use the prediction as the starting point for the next timestep.
            pos_current = preds_next

    if experiment_name:
        pred_path = output_dir / f"pred_sequences_pinn_{experiment_name}.pt"
    else:
        pred_path = output_dir / "pred_sequences_pinn.pt"

    torch.save({"pairs": pairs}, pred_path)

    if experiment_name:
        print(f"Saved PINN prediction pairs for {experiment_name} to {pred_path}")
    else:
        print(f"Saved PINN prediction pairs to {pred_path}")

    return pred_path


def main() -> None:
    project_root = Path(__file__).parent.parent

    all_configs = [
        "vanilla",
        "mass",
        "neighbors",
        "rigid",
        "interface",
        "all",
    ]

    parser = argparse.ArgumentParser(description="Generate validation predictions for selected configurations.")
    parser.add_argument(
        "-c",
        "--config",
        dest="configs",
        nargs="+",
        choices=all_configs,
        help=(
            "Name(s) of configurations to generate predictions for. "
            "If omitted, predictions are generated for all configurations. "
            "Choices: vanilla, mass, neighbors, rigid, interface, all."
        ),
    )

    args = parser.parse_args()

    if args.configs:
        configs = [name for name in all_configs if name in args.configs]
    else:
        configs = all_configs

    if not configs:
        print("No matching configurations to predict for (check --config names).")
        return

    for name in configs:
        print("\n" + "-" * 80)
        print(f"Generating predictions for configuration: {name}")
        print("-" * 80)
        generate_gnn_predictions(project_root, experiment_name=name)


if __name__ == "__main__":  # pragma: no cover - CLI guard
    main()
