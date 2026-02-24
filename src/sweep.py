"""Phase-8 hyperparameter sweep runner.

Performs a grid search over model and training hyperparameters, generating
all diagnostic artifacts for each configuration.  Results are organised in
``output/sweep/<run_name>/`` with a CSV leaderboard.

Usage::

    python -m src.sweep                    # full Stage A sweep
    python -m src.sweep --dry-run          # print configs without training
    python -m src.sweep --stages A         # only Stage A
    python -m src.sweep --stages B         # only Stage B (needs Stage A results)
"""

from __future__ import annotations

import csv
import itertools
import shutil
from pathlib import Path
from typing import Dict, List

from .train_pinn import train
from .predict_sequence import generate_gnn_predictions
from .post_videos import create_gnn_comparison_gif
from .post_diagnostics import run_diagnostics
from .preprocess_data import NEIGHBOR_RADIUS


# ---------------------------------------------------------------------------
# Sweep grids
# ---------------------------------------------------------------------------

STAGE_A_GRID = {
    "learning_rate": [3e-4, 1e-3, 3e-3],
    "hidden_channels": [64, 128],
    "num_layers": [2, 3],
}

STAGE_A_FIXED = {
    "epochs": 8,
    "batch_size": 1,
    "lambda_floor": 0.0,
    "lambda_vel": 1.0,
    "lambda_boundary": 0.0,
}

STAGE_B_BOUNDARY_VALUES = [0.0, 2.0, 5.0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_name(lr: float, hc: int, nl: int, lb: float | None = None) -> str:
    """Generate a descriptive experiment name from hyperparameters."""
    base = f"lr{lr:.0e}_hc{hc}_nl{nl}".replace("+", "")
    if lb is not None and lb > 0:
        base += f"_lb{lb:.1f}"
    return base


def _collect_artifacts(experiment_name: str, output_dir: Path, dest_dir: Path) -> List[Path]:
    """Move all files matching *experiment_name* from output_dir into dest_dir."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    moved: List[Path] = []
    for f in sorted(output_dir.iterdir()):
        if f.is_file() and experiment_name in f.name:
            dest = dest_dir / f.name
            shutil.move(str(f), str(dest))
            moved.append(dest)
    return moved


def _run_single_config(
    project_root: Path,
    output_dir: Path,
    sweep_dir: Path,
    name: str,
    train_kwargs: dict,
) -> float:
    """Train, predict, generate all artifacts, collect into sweep subdir."""

    # Train
    best_val_loss = train(experiment_name=name, **train_kwargs)

    # Predict (val + train)
    for split in ("val", "train"):
        generate_gnn_predictions(
            project_root,
            experiment_name=name,
            radius=NEIGHBOR_RADIUS,
            split=split,
        )

    # Comparison GIFs (val + train)
    for split in ("val", "train"):
        create_gnn_comparison_gif(
            project_root,
            experiment_name=name,
            split=split,
        )

    # Diagnostic histograms (val + train)
    for split in ("val", "train"):
        run_diagnostics(
            project_root,
            experiment_name=name,
            split=split,
        )

    # Collect artifacts into sweep subdirectory
    run_dir = sweep_dir / name
    moved = _collect_artifacts(name, output_dir, run_dir)
    print(f"  Collected {len(moved)} artifacts → {run_dir}")

    return best_val_loss


def _write_leaderboard(leaderboard: List[Dict], csv_path: Path) -> None:
    """Write sorted leaderboard to CSV and print to stdout."""
    leaderboard.sort(key=lambda x: x["best_val_loss"])

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=leaderboard[0].keys())
        writer.writeheader()
        writer.writerows(leaderboard)

    print("\n" + "=" * 80)
    print("LEADERBOARD")
    print("=" * 80)
    for rank, entry in enumerate(leaderboard, 1):
        print(f"  #{rank}: {entry['name']}  val_loss={entry['best_val_loss']:.6f}")
    print(f"\nSaved to {csv_path}")


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------

def run_stage_a(
    project_root: Path,
    output_dir: Path,
    sweep_dir: Path,
    dry_run: bool = False,
) -> List[Dict]:
    """Stage A: sweep lr × hidden_channels × num_layers."""

    combos = list(itertools.product(
        STAGE_A_GRID["learning_rate"],
        STAGE_A_GRID["hidden_channels"],
        STAGE_A_GRID["num_layers"],
    ))

    print(f"\n{'='*80}")
    print(f"STAGE A: {len(combos)} configurations")
    print(f"{'='*80}")

    leaderboard: List[Dict] = []

    for i, (lr, hc, nl) in enumerate(combos, 1):
        name = _run_name(lr, hc, nl)
        print(f"\n[{i}/{len(combos)}] {name}")
        print(f"  lr={lr}, hidden_channels={hc}, num_layers={nl}")

        if dry_run:
            leaderboard.append({
                "name": name,
                "learning_rate": lr,
                "hidden_channels": hc,
                "num_layers": nl,
                "lambda_boundary": 0.0,
                "best_val_loss": float("nan"),
            })
            continue

        train_kwargs = {
            **STAGE_A_FIXED,
            "learning_rate": lr,
            "hidden_channels": hc,
            "num_layers": nl,
        }

        best_val_loss = _run_single_config(
            project_root, output_dir, sweep_dir, name, train_kwargs,
        )

        leaderboard.append({
            "name": name,
            "learning_rate": lr,
            "hidden_channels": hc,
            "num_layers": nl,
            "lambda_boundary": 0.0,
            "best_val_loss": best_val_loss,
        })

    return leaderboard


def run_stage_b(
    project_root: Path,
    output_dir: Path,
    sweep_dir: Path,
    stage_a_leaderboard: List[Dict],
    dry_run: bool = False,
    top_n: int = 2,
) -> List[Dict]:
    """Stage B: sweep lambda_boundary on top-N Stage A configs."""

    # Sort and pick top N from Stage A
    ranked = sorted(stage_a_leaderboard, key=lambda x: x["best_val_loss"])
    top_configs = ranked[:top_n]

    combos = list(itertools.product(top_configs, STAGE_B_BOUNDARY_VALUES))

    print(f"\n{'='*80}")
    print(f"STAGE B: {len(combos)} configurations (top {top_n} × {len(STAGE_B_BOUNDARY_VALUES)} boundary values)")
    print(f"{'='*80}")

    leaderboard: List[Dict] = []

    for i, (base_cfg, lb) in enumerate(combos, 1):
        lr = base_cfg["learning_rate"]
        hc = int(base_cfg["hidden_channels"])
        nl = int(base_cfg["num_layers"])
        name = _run_name(lr, hc, nl, lb=lb)

        print(f"\n[{i}/{len(combos)}] {name}")
        print(f"  lr={lr}, hidden_channels={hc}, num_layers={nl}, lambda_boundary={lb}")

        if dry_run:
            leaderboard.append({
                "name": name,
                "learning_rate": lr,
                "hidden_channels": hc,
                "num_layers": nl,
                "lambda_boundary": lb,
                "best_val_loss": float("nan"),
            })
            continue

        train_kwargs = {
            **STAGE_A_FIXED,
            "learning_rate": lr,
            "hidden_channels": hc,
            "num_layers": nl,
            "lambda_boundary": lb,
        }

        best_val_loss = _run_single_config(
            project_root, output_dir, sweep_dir, name, train_kwargs,
        )

        leaderboard.append({
            "name": name,
            "learning_rate": lr,
            "hidden_channels": hc,
            "num_layers": nl,
            "lambda_boundary": lb,
            "best_val_loss": best_val_loss,
        })

    return leaderboard


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_sweep(stages: str = "AB", dry_run: bool = False) -> None:
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "output"
    sweep_dir = output_dir / "sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    all_results: List[Dict] = []

    if "A" in stages.upper():
        stage_a_results = run_stage_a(project_root, output_dir, sweep_dir, dry_run=dry_run)
        all_results.extend(stage_a_results)

        if not dry_run and stage_a_results:
            _write_leaderboard(
                list(stage_a_results),
                sweep_dir / "leaderboard_stage_a.csv",
            )
    else:
        # Load Stage A results from existing CSV for Stage B
        csv_path = sweep_dir / "leaderboard_stage_a.csv"
        if csv_path.exists():
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                stage_a_results = []
                for row in reader:
                    row["learning_rate"] = float(row["learning_rate"])
                    row["hidden_channels"] = int(row["hidden_channels"])
                    row["num_layers"] = int(row["num_layers"])
                    row["lambda_boundary"] = float(row["lambda_boundary"])
                    row["best_val_loss"] = float(row["best_val_loss"])
                    stage_a_results.append(row)
        else:
            print("No Stage A leaderboard found. Run --stages A first.")
            return

    if "B" in stages.upper():
        stage_b_results = run_stage_b(
            project_root, output_dir, sweep_dir,
            stage_a_results, dry_run=dry_run,
        )
        all_results.extend(stage_b_results)

        if not dry_run and stage_b_results:
            _write_leaderboard(
                list(stage_b_results),
                sweep_dir / "leaderboard_stage_b.csv",
            )

    # Combined leaderboard
    if not dry_run and all_results:
        _write_leaderboard(all_results, sweep_dir / "leaderboard.csv")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Phase-8 hyperparameter sweep.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configurations without training.",
    )
    parser.add_argument(
        "--stages",
        default="AB",
        help="Which stages to run: A, B, or AB (default: AB).",
    )
    args = parser.parse_args()

    run_sweep(stages=args.stages, dry_run=args.dry_run)


if __name__ == "__main__":  # pragma: no cover - CLI guard
    main()
