"""Generate static figures for quantitative and qualitative evaluation.

Current implementation:

- Rasterise particle positions from ``pred_sequences_pinn_*.pt`` prediction
  files onto a regular 2D grid.
- Extract vertical and horizontal centreline profiles from the last frame.
- Compare these profiles across different experiment configurations
  (e.g. vanilla, mass, neighbours, all) using a shared rasterisation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
import numpy as np
import torch

# Use a non-interactive backend suitable for headless environments.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def rasterize_points_to_grid(
    xy: np.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    nx: int = 128,
    ny: int = 128,
) -> np.ndarray:
    """Rasterise 2D points onto a regular (ny, nx) grid as counts per cell.

    Parameters
    ----------
    xy:
        Array of shape ``(N, 2)`` containing particle positions ``(x, y)``.
    x_min, x_max, y_min, y_max:
        Global bounds used to define the grid.
    nx, ny:
        Number of grid cells in x and y.
    """

    if xy.size == 0:
        return np.zeros((ny, nx), dtype=np.float32)

    eps = 1e-8
    xs = (xy[:, 0] - x_min) / max(x_max - x_min, eps)
    ys = (xy[:, 1] - y_min) / max(y_max - y_min, eps)

    xs = np.clip(xs, 0.0, 1.0 - 1e-7)
    ys = np.clip(ys, 0.0, 1.0 - 1e-7)

    ix = (xs * nx).astype(int)
    iy = (ys * ny).astype(int)

    grid = np.zeros((ny, nx), dtype=np.float32)
    for j, i in zip(ix, iy):  # j -> x (column), i -> y (row)
        grid[i, j] += 1.0

    return grid


def compute_centerlines_from_pairs(
    pairs: List[Dict[str, torch.Tensor | str | int]],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    nx: int = 128,
    ny: int = 128,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """From a list of (GT, pred) pairs, build rasters and extract centre-lines.

    Uses the last pair in the list as representative for visualisation.
    Returns ``(gt_vert, gt_horiz, pred_vert, pred_horiz, mse)`` where ``mse``
    is the mean-squared error between the normalised rasters.
    """

    if not pairs:
        raise ValueError("No pairs provided to compute_centerlines_from_pairs.")

    last = pairs[-1]
    pos_true = last["pos_tp1_true"].detach().cpu().numpy()
    pos_pred = last["pos_tp1_pred"].detach().cpu().numpy()

    true_xy = pos_true[:, :2]
    pred_xy = pos_pred[:, :2]

    gt_grid = rasterize_points_to_grid(true_xy, x_min, x_max, y_min, y_max, nx=nx, ny=ny)
    pred_grid = rasterize_points_to_grid(pred_xy, x_min, x_max, y_min, y_max, nx=nx, ny=ny)

    if gt_grid.shape != pred_grid.shape:
        raise ValueError(f"Raster shapes differ: {gt_grid.shape} vs {pred_grid.shape}")

    max_val = max(gt_grid.max(), pred_grid.max(), 1e-8)
    gt_norm = gt_grid / max_val
    pred_norm = pred_grid / max_val

    mse = float(np.mean((pred_norm - gt_norm) ** 2))

    ny_grid, nx_grid = gt_norm.shape
    cx = nx_grid // 2
    cy = int(0.25 * ny_grid)

    gt_vert = gt_norm[:, cx]
    pred_vert = pred_norm[:, cx]

    gt_horiz = gt_norm[cy, :]
    pred_horiz = pred_norm[cy, :]

    return gt_vert, gt_horiz, pred_vert, pred_horiz, mse


def main() -> None:
    """Create rasterised centreline comparison plots across experiments."""

    project_root = Path(__file__).parent.parent
    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True)

    # Collect prediction files from the multi-experiment pipeline.
    seq_paths = sorted(output_dir.glob("pred_sequences_pinn_*.pt"))
    if not seq_paths:
        seq_paths = list(output_dir.glob("pred_sequences_pinn.pt"))

    if not seq_paths:
        print("No prediction sequence files found; run predict_sequence.py first.")
        return

    experiments: List[Tuple[str, List[Dict[str, torch.Tensor | str | int]]]] = []
    x_min = float("inf")
    x_max = float("-inf")
    y_min = float("inf")
    y_max = float("-inf")

    for seq_path in seq_paths:
        payload = torch.load(seq_path, map_location="cpu")
        pairs = payload.get("pairs", [])
        if not pairs:
            print(f"{seq_path.name} has no 'pairs'; skipping.")
            continue

        stem = seq_path.stem
        if stem.startswith("pred_sequences_pinn_"):
            tag = stem[len("pred_sequences_pinn_") :]
        elif stem == "pred_sequences_pinn":
            tag = "default"
        else:
            tag = stem

        for item in pairs:
            pos_true = item["pos_tp1_true"].detach().cpu().numpy()
            xy = pos_true[:, :2]
            x_min = min(x_min, float(xy[:, 0].min()))
            x_max = max(x_max, float(xy[:, 0].max()))
            y_min = min(y_min, float(xy[:, 1].min()))
            y_max = max(y_max, float(xy[:, 1].max()))

        experiments.append((tag, pairs))

    if not experiments or not np.isfinite([x_min, x_max, y_min, y_max]).all():
        print("No valid prediction data found; nothing to plot.")
        return

    nx = 128
    ny = 128

    gt_vert_ref: np.ndarray | None = None
    gt_horiz_ref: np.ndarray | None = None
    y_idx_ref: np.ndarray | None = None
    x_idx_ref: np.ndarray | None = None
    model_curves: List[Tuple[str, np.ndarray, np.ndarray, float]] = []

    for tag, pairs in experiments:
        try:
            gt_vert, gt_horiz, pred_vert, pred_horiz, mse = compute_centerlines_from_pairs(
                pairs,
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                nx=nx,
                ny=ny,
            )
        except Exception as exc:  # pragma: no cover - robustness over strictness
            print(f"Failed to compute centre-lines for {tag}: {exc}")
            continue

        if gt_vert_ref is None:
            gt_vert_ref = gt_vert
            gt_horiz_ref = gt_horiz
            y_idx_ref = np.arange(len(gt_vert))
            x_idx_ref = np.arange(len(gt_horiz))

        model_curves.append((tag, pred_vert, pred_horiz, mse))

    if gt_vert_ref is None or gt_horiz_ref is None or y_idx_ref is None or x_idx_ref is None or not model_curves:
        print("No valid centreline data collected; nothing to plot.")
        return

    fig, (ax_vert, ax_horiz) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax_vert.plot(y_idx_ref, gt_vert_ref, label="Ground truth (rasterised)", linewidth=2, color="black")
    for tag, pred_vert, _pred_horiz, mse in model_curves:
        ax_vert.plot(
            y_idx_ref,
            pred_vert,
            label=f"{tag} (MSE={mse:.4f})",
            linewidth=1.0,
        )
    ax_vert.set_xlabel("Grid y index (vertical centreline)")
    ax_vert.set_ylabel("Normalised count")
    ax_vert.set_title("Vertical centreline (last frame, rasterised)")
    ax_vert.grid(True, alpha=0.3)
    ax_vert.legend()

    ax_horiz.plot(x_idx_ref, gt_horiz_ref, label="Ground truth (rasterised)", linewidth=2, color="black")
    for tag, _pred_vert, pred_horiz, mse in model_curves:
        ax_horiz.plot(
            x_idx_ref,
            pred_horiz,
            label=f"{tag} (MSE={mse:.4f})",
            linewidth=1.0,
        )
    ax_horiz.set_xlabel("Grid x index (horizontal centreline)")
    ax_horiz.set_ylabel("Normalised count")
    ax_horiz.set_title("Horizontal centreline (last frame, rasterised)")
    ax_horiz.grid(True, alpha=0.3)
    ax_horiz.legend()

    out_path = output_dir / "model_sensitivity_centerlines.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved rasterised centreline sensitivity plot to {out_path}")


if __name__ == "__main__":  # pragma: no cover - CLI guard
    main()
