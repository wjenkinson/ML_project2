"""Preprocess raw simulation data into train/validation splits and metadata.

Current responsibilities (minimal version):
- Discover available ``dump.*.LAMMPS`` files under ``data/``.
- Create a contiguous 80/20 train/validation split based on timestep order.
- Persist the split as file lists under ``data/splits``.
- Write lightweight metadata about the split and timestep ranges.

Planned extensions (stubs only for now):
- Caching neighbour lists / edge indices for graph-based models.
- Caching per-frame scalars (e.g. mass, conservation diagnostics).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
from sklearn.neighbors import NearestNeighbors


DATA_DIR_NAME = "data"
SPLITS_SUBDIR = "splits"
NEIGHBOR_CACHE_SUBDIR = "neighbors"
NEIGHBOR_RADIUS = 0.0015


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


def neighbor_cache_dir(data_dir: Path) -> Path:
    """Return the directory where cached neighbour lists will be stored.

    This is a shared convention that other modules can use to discover
    precomputed neighbour lists / edge indices.
    """

    return data_dir / NEIGHBOR_CACHE_SUBDIR


def neighbor_cache_exists(data_dir: Path) -> bool:
    """Return True if a neighbour cache directory already exists.

    The current stub implementation does not create this directory; a future
    implementation that actually computes neighbour lists is expected to
    populate it. Other modules can use this helper to decide whether to
    rely on the cache or emit a message such as:

        "Neighbour list not cached, run preprocess_data.py".
    """

    return neighbor_cache_dir(data_dir).exists()


def cache_neighbor_lists_stub(data_dir: Path, split: SplitResult) -> None:
    """Compute and cache simple neighbour statistics for each split frame.

    For each dump file in the train/validation split, this function:
    - Loads atom positions and types.
    - Builds a radius-based neighbour graph using ``NEIGHBOR_RADIUS``.
    - Computes, per atom:
      - Total neighbour count (excluding self).
      - Number of neighbours of a different type.
    - Saves these statistics into ``data/neighbors/<dump_name>.npz``.

    The cached data is intended for later use in physics checks (e.g.
    neighbour-count constraints) and, potentially, for defining graph
    structures for GNNs.
    """

    cache_dir = neighbor_cache_dir(data_dir)
    # Always refresh the cache so changes in NEIGHBOR_RADIUS or data are
    # reflected immediately. Remove existing .npz files and recreate the
    # directory if needed.
    if cache_dir.exists():
        for p in cache_dir.glob("*.npz"):
            p.unlink()
    else:
        cache_dir.mkdir(parents=True, exist_ok=True)

    def _load_types_and_coords(path: Path) -> tuple[np.ndarray, np.ndarray]:
        """Load atom types and 2D coordinates (x, y) from a dump file."""

        lines = path.read_text().splitlines(keepends=True)

        timestep = None
        num_atoms = None
        atoms_header_idx = None

        for i, line in enumerate(lines):
            s = line.strip()
            if s.startswith("ITEM: TIMESTEP") and i + 1 < len(lines):
                timestep = int(lines[i + 1].strip())
            elif s.startswith("ITEM: NUMBER OF ATOMS") and i + 1 < len(lines):
                num_atoms = int(lines[i + 1].strip())
            elif s.startswith("ITEM: ATOMS"):
                atoms_header_idx = i

        if timestep is None or num_atoms is None or atoms_header_idx is None:
            raise ValueError(f"File {path} does not look like a valid LAMMPS dump")

        header_tokens = lines[atoms_header_idx].strip().split()
        columns = header_tokens[2:]
        col_indices = {name: idx for idx, name in enumerate(columns)}

        if "type" not in col_indices or "x" not in col_indices or "y" not in col_indices:
            raise ValueError(
                f"Dump {path} must contain at least 'id type x y' columns; got {columns}"
            )

        data_start = atoms_header_idx + 1
        data_end = data_start + num_atoms
        data_lines = lines[data_start:data_end]

        raw = np.loadtxt(StringIO("".join(data_lines)))
        if raw.ndim == 1:
            raw = raw[np.newaxis, :]

        types = raw[:, col_indices["type"]].astype(int)
        xs = raw[:, col_indices["x"]]
        ys = raw[:, col_indices["y"]]
        coords = np.stack([xs, ys], axis=1)

        return types, coords

    def _compute_neighbor_stats(types: np.ndarray, coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute per-atom neighbour counts and different-type neighbour counts."""

        nn = NearestNeighbors(radius=NEIGHBOR_RADIUS)
        nn.fit(coords)
        indices_list = nn.radius_neighbors(coords, return_distance=False)

        neighbor_counts = np.zeros(len(coords), dtype=int)
        diff_type_counts = np.zeros(len(coords), dtype=int)

        for i, neigh_idx in enumerate(indices_list):
            # Exclude self from neighbour count if present.
            mask = neigh_idx != i
            neigh_idx = neigh_idx[mask]

            neighbor_counts[i] = neigh_idx.size
            if neigh_idx.size > 0:
                diff_type_counts[i] = np.count_nonzero(types[neigh_idx] != types[i])

        return neighbor_counts, diff_type_counts

    all_files: list[Path] = []
    all_files.extend(split.train_files)
    all_files.extend(split.val_files)

    for dump_path in all_files:
        cache_path = cache_dir / f"{dump_path.name}.npz"
        if cache_path.exists():
            continue

        types, coords = _load_types_and_coords(dump_path)
        neighbor_counts, diff_type_counts = _compute_neighbor_stats(types, coords)

        np.savez_compressed(
            cache_path,
            neighbor_counts=neighbor_counts,
            diff_type_counts=diff_type_counts,
            types=types,
            radius=NEIGHBOR_RADIUS,
        )

    print(f"Cached neighbor lists: Yes (radius={NEIGHBOR_RADIUS}) in {cache_dir}")


def cache_frame_scalars_stub(data_dir: Path, split: SplitResult) -> None:
    """Placeholder for caching per-frame scalars and conservation diagnostics.

    In a future iteration, this function could compute quantities such as:
    - Approximate mass per frame.
    - Other scalar diagnostics used for physics-based checks.
    """

    print("Cached frame scalars: No (stub implementation).")


def main() -> None:
    """Entry point for preprocessing.

    - Locates the project ``data/`` directory.
    - Lists and splits ``dump.*.LAMMPS`` files into train/val sets.
    - Writes file lists under ``data/splits``.
    - Writes minimal metadata to ``data/metadata.json``.
    - Invokes stub hooks for future caching steps.
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

    # Stub hooks for future, more advanced preprocessing steps.
    cache_neighbor_lists_stub(data_dir, split)
    cache_frame_scalars_stub(data_dir, split)


if __name__ == "__main__":  # pragma: no cover - CLI guard
    main()
