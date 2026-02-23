"""Phase-7 diagnostics: animated histogram GIFs comparing GT vs model rollout.

Produces two animated GIFs per experiment configuration:

1. **Velocity histogram** – Per-particle velocity magnitudes (fluid only, 2D).
   Each GIF frame = one rollout timestep, with overlaid GT (blue) and model
   (orange) histograms so you can see when/how the model diverges.

2. **Neighbor-distance histogram** – Pairwise distances within the neighbour
   radius (fluid only).  Same overlay format.

Usage::

    python -m src.post_diagnostics                 # all configs, val split
    python -m src.post_diagnostics --split train   # training split
    python -m src.post_diagnostics -c vanilla      # single config
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import argparse

import imageio.v2 as imageio
import matplotlib
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

from .preprocess_data import NEIGHBOR_RADIUS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_pred_pairs(
    output_dir: Path,
    experiment_name: str,
    split: str = "val",
) -> list | None:
    """Load saved prediction pairs for the given config and split."""

    split_suffix = f"_{split}" if split != "val" else ""
    pred_path = output_dir / f"pred_sequences_pinn_{experiment_name}{split_suffix}.pt"

    if not pred_path.exists():
        print(f"Prediction file not found: {pred_path}")
        return None

    payload = torch.load(pred_path, map_location="cpu")
    pairs = payload.get("pairs", [])
    if not pairs:
        print(f"No prediction pairs in {pred_path.name}")
        return None

    return pairs


def _fluid_velocity_magnitudes(
    vel: np.ndarray,
    atom_type: np.ndarray,
) -> np.ndarray:
    """Compute 2D velocity magnitudes for fluid particles (type 1)."""

    fluid_mask = atom_type == 1
    return np.linalg.norm(vel[fluid_mask], axis=1)


def _fluid_neighbor_distances(
    positions: np.ndarray,
    atom_type: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Collect all pairwise distances within *radius* among fluid particles."""

    fluid_mask = atom_type == 1
    coords = positions[fluid_mask, :2]
    if len(coords) < 2:
        return np.array([], dtype=np.float64)

    nn = NearestNeighbors(radius=radius, algorithm="ball_tree")
    nn.fit(coords)
    dist_list, _ = nn.radius_neighbors(coords, return_distance=True)

    all_dists: List[np.ndarray] = []
    for dists in dist_list:
        if dists.size > 0:
            all_dists.append(dists[dists > 0])  # exclude self (dist == 0)

    if not all_dists:
        return np.array([], dtype=np.float64)

    return np.concatenate(all_dists)




# ---------------------------------------------------------------------------
# GIF renderers
# ---------------------------------------------------------------------------

def _render_to_frame(fig) -> np.ndarray:
    """Render a matplotlib figure to an RGB numpy array."""

    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    return image[..., :3].copy()


def create_velocity_histogram_gif(
    pairs: list,
    output_path: Path,
    vel_range: Tuple[float, float] = (0.0, 2.0),
    n_bins: int = 60,
) -> Path:
    """Animated histogram GIF of fluid velocity magnitudes: GT vs model."""

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    atom_type = pairs[0]["atom_type"].numpy().astype(int)

    frames: List[np.ndarray] = []
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(vel_range[0], vel_range[1], n_bins + 1)

    for step, item in enumerate(pairs):
        # GT and model velocities are stored directly in the prediction pairs
        vel_gt = _fluid_velocity_magnitudes(item["vel_gt"].numpy(), atom_type)
        vel_model = _fluid_velocity_magnitudes(item["vel_pred"].numpy(), atom_type)

        ax.clear()
        ax.hist(vel_gt, bins=bins, alpha=0.55, color="tab:blue", label="GT", density=True)
        ax.hist(vel_model, bins=bins, alpha=0.55, color="tab:orange", label="Model", density=True)
        ax.set_xlim(vel_range)
        ax.set_xlabel("Velocity magnitude (fluid)")
        ax.set_ylabel("Density")
        ax.set_title(f"Step {step}  ({item.get('name_tp1', '')})")
        ax.legend(loc="upper right")

        frames.append(_render_to_frame(fig))

    plt.close(fig)

    imageio.mimsave(str(output_path), frames, fps=4)
    print(f"Saved velocity histogram GIF to {output_path}")
    return output_path


def create_neighbor_distance_histogram_gif(
    pairs: list,
    output_path: Path,
    radius: float = NEIGHBOR_RADIUS,
    dist_range: Tuple[float, float] = (0.0, 0.003),
    n_bins: int = 60,
) -> Path:
    """Animated histogram GIF of fluid neighbor distances: GT vs model."""

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    atom_type = pairs[0]["atom_type"].numpy().astype(int)

    frames: List[np.ndarray] = []
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(dist_range[0], dist_range[1], n_bins + 1)

    for step, item in enumerate(pairs):
        gt_pos = item["pos_tp1_true"].numpy()
        pred_pos = item["pos_tp1_pred"].numpy()

        dists_gt = _fluid_neighbor_distances(gt_pos, atom_type, radius)
        dists_model = _fluid_neighbor_distances(pred_pos, atom_type, radius)

        ax.clear()
        if dists_gt.size > 0:
            ax.hist(dists_gt, bins=bins, alpha=0.55, color="tab:blue", label="GT", density=True)
        if dists_model.size > 0:
            ax.hist(dists_model, bins=bins, alpha=0.55, color="tab:orange", label="Model", density=True)
        ax.set_xlim(dist_range)
        ax.set_xlabel("Neighbor distance (fluid)")
        ax.set_ylabel("Density")
        ax.set_title(f"Step {step}  ({item.get('name_tp1', '')})")
        ax.legend(loc="upper right")

        frames.append(_render_to_frame(fig))

    plt.close(fig)

    imageio.mimsave(str(output_path), frames, fps=4)
    print(f"Saved neighbor-distance histogram GIF to {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run_diagnostics(
    project_root: Path,
    experiment_name: str,
    split: str = "val",
) -> None:
    """Generate all diagnostic GIFs for a single experiment + split."""

    output_dir = project_root / "output"
    pairs = _load_pred_pairs(output_dir, experiment_name, split)
    if pairs is None:
        return

    split_suffix = f"_{split}" if split != "val" else ""
    tag = f"{experiment_name}{split_suffix}"

    create_velocity_histogram_gif(
        pairs,
        output_dir / f"velocity_hist_{tag}.gif",
    )

    create_neighbor_distance_histogram_gif(
        pairs,
        output_dir / f"neighbor_dist_hist_{tag}.gif",
    )


def main() -> None:
    project_root = Path(__file__).parent.parent

    all_configs = [
        "vanilla",
        "density",
        "floor",
        "density_floor",
    ]

    parser = argparse.ArgumentParser(
        description="Create diagnostic histogram GIFs (velocity, neighbor distance) for GT vs model."
    )
    parser.add_argument(
        "-c",
        "--config",
        dest="configs",
        nargs="+",
        choices=all_configs,
        help="Configuration name(s). Defaults to all.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val"],
        default="val",
        help="Which data split to visualise (default: val).",
    )

    args = parser.parse_args()

    if args.configs:
        configs = [name for name in all_configs if name in args.configs]
    else:
        configs = all_configs

    if not configs:
        print("No matching configurations (check --config names).")
        return

    for name in configs:
        print("\n" + "-" * 80)
        print(f"Diagnostics for {name} (split={args.split})")
        print("-" * 80)
        run_diagnostics(project_root, name, split=args.split)


if __name__ == "__main__":  # pragma: no cover - CLI guard
    main()
