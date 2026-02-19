"""Generate next-step sequence predictions using the trained GNN model.

This script:

- Loads the best ``SimpleGnnPredictor`` checkpoint from ``output/``.
- Runs autoregressive rollout over the validation split.
- The model predicts DISPLACEMENTS; positions are computed as:
  pos(t+1) = pos(t) + displacement_pred
- Saves ground-truth and predicted positions to a torch file for post_videos.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import argparse

import torch
from torch_geometric.data import Data

from .graph_dataset import (
    LammpsGraphDataset,
    NODE_FEATURE_DIM,
    build_radius_graph_pbc_x,
    minimum_image_rel_pos,
    BOX_LX,
)
from .preprocess_data import NEIGHBOR_RADIUS
from .train_pinn import SimpleGnnPredictor


def generate_gnn_predictions(
    project_root: Path,
    experiment_name: str | None = None,
    radius: float = NEIGHBOR_RADIUS,
    max_steps: int | None = None,
    neighbor_overflow_limit: int | None = 80,
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

    # Load model (new architecture with edge features)
    model = SimpleGnnPredictor(
        in_channels=NODE_FEATURE_DIM,  # [is_fluid, is_solid, is_wall, vx, vy]
        hidden_channels=64,
        edge_channels=4,  # [dx, dy, dz, dist]
        num_layers=3,
    ).to(device)
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

    # Fixed wall positions (type 3) taken from the first frame.
    wall_mask = atom_type == 3
    wall_pos_fixed = first_frame["pos"].to(device)[wall_mask] if torch.any(wall_mask) else None

    # Piston (type 4): moves down 0.001 per frame from initial position.
    piston_mask = atom_type == 4
    piston_pos_initial = first_frame["pos"].to(device)[piston_mask].clone() if torch.any(piston_mask) else None
    piston_dy_per_frame = -0.0015  # moves down

    # One-hot type encoding (constant across rollout)
    is_fluid = (atom_type == 1).to(torch.float32).unsqueeze(-1)
    is_solid = (atom_type == 2).to(torch.float32).unsqueeze(-1)
    is_wall = (atom_type == 3).to(torch.float32).unsqueeze(-1)
    is_piston = (atom_type == 4).to(torch.float32).unsqueeze(-1)

    # Start autoregressive rollout from the ground-truth positions of the first frame.
    pos_current = first_frame["pos"].to(device)
    vel_current = first_frame["vel"].to(device)  # initial velocities

    pairs: List[Dict[str, torch.Tensor | str | int]] = []

    num_steps = len(frame_names) - 1
    if max_steps is not None:
        num_steps = min(num_steps, max_steps)

    with torch.no_grad():
        for step in range(num_steps):
            print("Step", step, flush=True)
            name_t = frame_names[step]
            name_tp1 = frame_names[step + 1]

            # Ground-truth next-frame positions for benchmarking.
            frame_tp1 = val_dataset.frames[name_tp1]
            pos_true_next = frame_tp1["pos"].to(device)

            # Build node features: [is_fluid, is_solid, is_wall, is_piston, vx, vy]
            x = torch.cat([is_fluid, is_solid, is_wall, is_piston, vel_current], dim=-1)  # (N, 6)

            # Build radius-based neighbourhood graph with periodic BC in x
            coords = pos_current.cpu().numpy()
            row, col = build_radius_graph_pbc_x(coords, radius)

            # Check neighbor overflow (approximate via edge count per node)
            if neighbor_overflow_limit is not None:
                from collections import Counter
                neigh_counts = Counter(row)
                max_neigh = max(neigh_counts.values(), default=0)
                if max_neigh > neighbor_overflow_limit:
                    print(
                        f"Neighbor overflow at step {step}: max neighbors per particle = {max_neigh} "
                        f"> limit {neighbor_overflow_limit}. Aborting rollout to avoid instability.",
                        flush=True,
                    )
                    break

            edge_index = torch.tensor([row, col], dtype=torch.long, device=device)

            # Compute edge attributes: relative positions + distance (with PBC in x)
            if edge_index.numel() > 0:
                src, dst = edge_index[0], edge_index[1]
                rel_pos = minimum_image_rel_pos(pos_current[src], pos_current[dst], Lx=BOX_LX)
                dist = torch.norm(rel_pos, dim=-1, keepdim=True)
                edge_attr = torch.cat([rel_pos, dist], dim=-1)
            else:
                edge_attr = torch.zeros((0, 4), dtype=torch.float32, device=device)

            data_step = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos_current)

            # Model predicts DISPLACEMENT, not absolute position
            displacement_pred = model(data_step)  # (N, 3)

            # Compute next positions from displacement
            preds_next = pos_current + displacement_pred  # (N, 3)

            # Clamp rigid-wall atoms (type 3) so they remain fixed in space.
            if wall_pos_fixed is not None:
                preds_next = preds_next.clone()
                preds_next[wall_mask] = wall_pos_fixed

            # Clamp piston atoms (type 4) to prescribed trajectory: initial + step * dy.
            if piston_pos_initial is not None:
                piston_pos_now = piston_pos_initial.clone()
                piston_pos_now[:, 1] += piston_dy_per_frame * (step + 1)
                preds_next[piston_mask] = piston_pos_now

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

            # Update velocity estimate from displacement (assumes dt is implicit)
            # displacement ≈ velocity * dt, so we use displacement[:, :2] as velocity proxy
            vel_current = displacement_pred[:, :2].clone()
            # Walls have zero velocity
            if wall_pos_fixed is not None:
                vel_current[wall_mask] = 0.0
            # Piston has prescribed downward velocity
            if piston_pos_initial is not None:
                vel_current[piston_mask, 0] = 0.0
                vel_current[piston_mask, 1] = piston_dy_per_frame

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
        "density",
        "floor",
        "density_floor",
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
            "If omitted, predictions are generated for both configurations. "
            "Choices: vanilla, density."
        ),
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum number of autoregressive steps to roll out (per configuration).",
    )
    parser.add_argument(
        "--neighbor-overflow-limit",
        type=int,
        default=80,
        help="Abort rollout if any particle has more than this many neighbors in the radius graph.",
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
        generate_gnn_predictions(
            project_root,
            experiment_name=name,
            radius=NEIGHBOR_RADIUS,
            max_steps=args.max_steps,
            neighbor_overflow_limit=args.neighbor_overflow_limit,
        )


if __name__ == "__main__":  # pragma: no cover - CLI guard
    main()
