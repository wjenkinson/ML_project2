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

import torch

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

    # Load model
    model = SimpleGnnPredictor(in_channels=4, hidden_channels=64, num_layers=2).to(device)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    pairs: List[Dict[str, torch.Tensor | str | int]] = []

    for idx in range(len(val_dataset)):
        data = val_dataset[idx]
        data = data.to(device)

        with torch.no_grad():
            preds = model(data)  # (N, 3)

        # Store on CPU for serialization
        preds_cpu = preds.detach().cpu()
        pos_t_cpu = data.pos.detach().cpu()
        pos_tp1_cpu = data.y.detach().cpu()
        atom_type_cpu = data.atom_type.detach().cpu()

        name_t = getattr(data, "name_t", f"val_pair_{idx}_t")
        name_tp1 = getattr(data, "name_tp1", f"val_pair_{idx}_tp1")

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
    configs = [
        "vanilla",
        "mass",
        "neighbors",
        "rigid",
        "interface",
        "all",
    ]

    for name in configs:
        print("\n" + "-" * 80)
        print(f"Generating predictions for configuration: {name}")
        print("-" * 80)
        generate_gnn_predictions(project_root, experiment_name=name)


if __name__ == "__main__":  # pragma: no cover - CLI guard
    main()
