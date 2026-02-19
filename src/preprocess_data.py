"""Preprocess raw simulation data into train/validation splits and metadata.

Responsibilities:
- Discover available ``dump.*.LAMMPS`` files under ``data/``.
- Create a contiguous 80/20 train/validation split based on timestep order.
- Persist the split as file lists under ``data/splits``.
- Write lightweight metadata about the split and timestep ranges.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence


DATA_DIR_NAME = "data"
SPLITS_SUBDIR = "splits"
NEIGHBOR_RADIUS = 0.003


@dataclass
class SplitResult:
    """Container for the outcome of a train/validation split."""

    train_files: List[Path]
    val_files: List[Path]


def timestep_from_name(path: Path) -> int:
    """Extract numeric timestep from filenames like ``dump.12345.LAMMPS``.

    Falls back to 0 if parsing fails so that such files sort to the front.
    """

    name = path.name
    try:
        parts = name.split(".")
        return int(parts[1])
    except (IndexError, ValueError):
        return 0


def list_dump_files(data_dir: Path) -> List[Path]:
    """Return all ``dump.*.LAMMPS`` files under ``data_dir`` sorted by timestep."""

    return sorted(data_dir.glob("dump.*.LAMMPS"), key=timestep_from_name)


def train_val_split(
    dump_files: Sequence[Path], train_fraction: float = 0.8
) -> SplitResult:
    """Split a sequence of dump files into contiguous train/val segments.

    The first ``train_fraction`` of frames (floor) is used for training, the
    remainder for validation.
    """

    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")

    num_files = len(dump_files)
    num_train = int(train_fraction * num_files)

    train_files = list(dump_files[:num_train])
    val_files = list(dump_files[num_train:])

    return SplitResult(train_files=train_files, val_files=val_files)


def write_file_list(path: Path, files: Iterable[Path]) -> None:
    """Write basenames of ``files`` to ``path``, one per line."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for p in files:
            f.write(p.name + "\n")


def compute_split_metadata(split: SplitResult) -> dict:
    """Compute simple metadata about the train/validation split."""

    def _timesteps(paths: Sequence[Path]) -> List[int]:
        return [timestep_from_name(p) for p in paths]

    train_ts = _timesteps(split.train_files)
    val_ts = _timesteps(split.val_files)

    meta = {
        "num_train": len(split.train_files),
        "num_val": len(split.val_files),
    }

    if train_ts:
        meta["train_timestep_min"] = min(train_ts)
        meta["train_timestep_max"] = max(train_ts)
    if val_ts:
        meta["val_timestep_min"] = min(val_ts)
        meta["val_timestep_max"] = max(val_ts)

    return meta


def write_metadata(data_dir: Path, split: SplitResult) -> Path:
    """Write split metadata JSON into ``data/metadata.json`` and return its path."""

    meta = compute_split_metadata(split)
    output_path = data_dir / "metadata.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    return output_path


def main() -> None:
    """Entry point for preprocessing.

    - Locates the project ``data/`` directory.
    - Lists and splits ``dump.*.LAMMPS`` files into train/val sets.
    - Writes file lists under ``data/splits``.
    - Writes minimal metadata to ``data/metadata.json``.
    """

    project_root = Path(__file__).parent.parent
    data_dir = project_root / DATA_DIR_NAME

    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        return

    dump_files = list_dump_files(data_dir)
    if not dump_files:
        print(f"No LAMMPS dump files found in {data_dir}")
        return

    split = train_val_split(dump_files, train_fraction=0.8)

    splits_dir = data_dir / SPLITS_SUBDIR
    train_list_path = splits_dir / "train_files.txt"
    val_list_path = splits_dir / "val_files.txt"

    write_file_list(train_list_path, split.train_files)
    write_file_list(val_list_path, split.val_files)

    meta_path = write_metadata(data_dir, split)

    num_files = len(dump_files)
    print(f"Found {num_files} dump files in {data_dir}")
    print(f"Training files: {len(split.train_files)} -> {train_list_path}")
    print(f"Validation files: {len(split.val_files)} -> {val_list_path}")
    print(f"Metadata written to: {meta_path}")

    if split.train_files:
        print("First train file:", split.train_files[0].name)
        print("Last train file:", split.train_files[-1].name)
    if split.val_files:
        print("First val file:", split.val_files[0].name)
        print("Last val file:", split.val_files[-1].name)


if __name__ == "__main__":  # pragma: no cover - CLI guard
    main()
