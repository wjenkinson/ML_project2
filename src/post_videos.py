"""Create videos / GIFs comparing ground truth and predicted sequences.

Current implementation focuses on the PINN/GNN predictions produced by
``predict_sequence.py``. It loads ``output/pred_sequences_pinn_*.pt`` files
(one per experiment configuration) and creates animated GIFs showing
side-by-side scatter plots of:

- Left: ground-truth particle positions at t+1.
- Right: GNN-predicted positions at t+1.

Atom type colors:
- Type 1 (fluid): blue
- Type 2 (solid): orange
- Type 3 (wall): gray
- Type 4 (piston): red
"""

from __future__ import annotations

from pathlib import Path
from typing import List
import argparse

import imageio.v2 as imageio
import matplotlib
import numpy as np
import torch


def create_gnn_comparison_gif(
    project_root: Path,
    experiment_name: str | None = None,
    output_name: str | None = None,
    max_frames: int | None = None,
) -> Path | None:
    """Create a side-by-side GT vs prediction GIF from saved GNN sequences."""

    output_dir = project_root / "output"

    if experiment_name:
        pred_path = output_dir / f"pred_sequences_pinn_{experiment_name}.pt"
    else:
        pred_path = output_dir / "pred_sequences_pinn.pt"

    if not pred_path.exists():
        print(f"Prediction file not found: {pred_path}. Run predict_sequence.py for this configuration first.")
        return None

    payload = torch.load(pred_path, map_location="cpu")
    pairs = payload.get("pairs", [])
    if not pairs:
        print(f"No prediction pairs found in {pred_path.name}")
        return None

    if max_frames is not None:
        pairs = pairs[:max_frames]

    # Use a non-interactive backend suitable for headless environments.
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Compute global bounds across all frames and both GT and predictions.
    x_min, x_max = float("inf"), float("-inf")
    y_min, y_max = float("inf"), float("-inf")

    for item in pairs:
        pos_true = item["pos_tp1_true"].numpy()
        pos_pred = item["pos_tp1_pred"].numpy()

        x_min = min(x_min, pos_true[:, 0].min(), pos_pred[:, 0].min())
        x_max = max(x_max, pos_true[:, 0].max(), pos_pred[:, 0].max())
        y_min = min(y_min, pos_true[:, 1].min(), pos_pred[:, 1].min())
        y_max = max(y_max, pos_true[:, 1].max(), pos_pred[:, 1].max())

    # Centreline positions in world coordinates (used both for plots and GIFs).
    cx = 0.5 * (x_min + x_max)
    cy = y_min + 0.25 * (y_max - y_min)

    frames: List[np.ndarray] = []

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    ax_gt, ax_pred = axes

    for item in pairs:
        pos_true = item["pos_tp1_true"].numpy()
        pos_pred = item["pos_tp1_pred"].numpy()
        atom_type = item["atom_type"].numpy().astype(int)
        name_tp1 = item.get("name_tp1", "frame")

        for ax in axes:
            ax.clear()
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xticks([])
            ax.set_yticks([])

        # Map atom types to colors: 1=blue, 2=orange, 3=gray, 4=red
        type_colors = {1: "tab:blue", 2: "tab:orange", 3: "gray", 4: "tab:red"}
        colors = [type_colors.get(t, "black") for t in atom_type]

        # Ground truth panel with centreline crosshair
        sc_gt = ax_gt.scatter(pos_true[:, 0], pos_true[:, 1], c=colors, s=4)
        ax_gt.axvline(cx, color="black", linestyle="--", linewidth=1.7, alpha=0.7)
        ax_gt.axhline(cy, color="black", linestyle="--", linewidth=1.7, alpha=0.7)
        ax_gt.set_title("Ground truth")

        # Prediction panel with the same centreline crosshair for consistency
        sc_pred = ax_pred.scatter(pos_pred[:, 0], pos_pred[:, 1], c=colors, s=4)
        ax_pred.axvline(cx, color="black", linestyle="--", linewidth=1.7, alpha=0.7)
        ax_pred.axhline(cy, color="black", linestyle="--", linewidth=1.7, alpha=0.7)
        ax_pred.set_title("GNN prediction")

        fig.suptitle(f"{name_tp1}")

        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        image = image[..., :3]
        frames.append(image.copy())

    plt.close(fig)

    if output_name is None:
        if experiment_name:
            output_name = f"prediction_vs_gt_pinn_{experiment_name}.gif"
        else:
            output_name = "prediction_vs_gt_pinn.gif"

    output_path = output_dir / output_name
    imageio.mimsave(output_path, frames, fps=4)
    print(f"Saved GT vs prediction GIF to {output_path}")

    return output_path


def main() -> None:
    project_root = Path(__file__).parent.parent

    all_configs = [
        "vanilla",
        "density",
        "floor",
        "density_floor",
    ]

    parser = argparse.ArgumentParser(description="Create GT vs prediction GIFs for selected configurations.")
    parser.add_argument(
        "-c",
        "--config",
        dest="configs",
        nargs="+",
        choices=all_configs,
        help=(
            "Name(s) of configurations to render GIFs for. "
            "If omitted, GIFs are generated for both configurations. "
            "Choices: vanilla, density."
        ),
    )

    args = parser.parse_args()

    if args.configs:
        configs = [name for name in all_configs if name in args.configs]
    else:
        configs = all_configs

    if not configs:
        print("No matching configurations to render (check --config names).")
        return

    for name in configs:
        print("\n" + "-" * 80)
        print(f"Creating GIF for configuration: {name}")
        print("-" * 80)
        create_gnn_comparison_gif(project_root, experiment_name=name)


if __name__ == "__main__":  # pragma: no cover - CLI guard
    main()
