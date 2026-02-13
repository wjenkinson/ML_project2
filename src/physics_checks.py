"""Physics checks utilities.

Initial focus: mass/atom-count conservation on sequences of LAMMPS dump files.

This module is intended to be both importable (for tests) and executable as a
script. When run as a script, it will, by default:

- Read the training split list from ``data/splits/train_files.txt`` if it
  exists, otherwise fall back to scanning all ``dump.*.LAMMPS`` files in
  ``data/``.
- Compute per-frame atom-count deviations relative to a reference frame.
- Print human-readable diagnostics such as:

    Frame dump.0000.LAMMPS (t = 0): 0 atom(s) lost
    Frame dump.1000.LAMMPS (t = 1000): 0 atom(s) gained

The underlying functions can also be reused later for model evaluation and for
physics-informed loss design.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np

from .preprocess_data import (
    DATA_DIR_NAME,
    SPLITS_SUBDIR,
    list_dump_files,
    neighbor_cache_dir,
    neighbor_cache_exists,
)
from .preview_data import infer_metadata_from_lines


@dataclass
class MassDiagnostic:
    """Per-frame mass / atom-count diagnostic."""

    path: Path
    timestep: int
    n_atoms: int
    delta_atoms: int  # relative to reference frame


def count_atoms(path: Path) -> MassDiagnostic:
    """Count atoms in a single dump file and return a MassDiagnostic.

    The reference for ``delta_atoms`` is not set here; it will be filled in
    by higher-level routines once a reference frame is chosen.
    """

    lines = path.read_text().splitlines(keepends=True)
    meta = infer_metadata_from_lines(path, lines)
    return MassDiagnostic(path=path, timestep=meta.timestep, n_atoms=meta.num_atoms, delta_atoms=0)


def mass_conservation_over_sequence(dump_files: Sequence[Path], ref_index: int = 0) -> List[MassDiagnostic]:
    """Compute mass-conservation diagnostics over a sequence of dump files.

    Parameters
    ----------
    dump_files:
        Ordered sequence of dump file paths.
    ref_index:
        Index of the reference frame in ``dump_files``. By default, the first
        frame is used.
    """

    if not dump_files:
        return []

    diagnostics = [count_atoms(p) for p in dump_files]

    if not (0 <= ref_index < len(diagnostics)):
        raise IndexError("ref_index out of range for diagnostics list")

    ref_n = diagnostics[ref_index].n_atoms

    for diag in diagnostics:
        diag.delta_atoms = diag.n_atoms - ref_n

    return diagnostics


def _load_training_dump_files(project_root: Path) -> List[Path]:
    """Load training dump files based on data/splits, or fall back to all dumps."""

    data_dir = project_root / DATA_DIR_NAME
    splits_dir = data_dir / SPLITS_SUBDIR
    train_list_path = splits_dir / "train_files.txt"

    if train_list_path.exists():
        # Use the explicit training split list.
        with train_list_path.open("r", encoding="utf-8") as f:
            names = [line.strip() for line in f if line.strip()]
        dump_files = [data_dir / name for name in names]
    else:
        # Fallback: use all dumps in data/.
        dump_files = list_dump_files(data_dir)

    return dump_files


def _load_neighbor_stats_for_file(data_dir: Path, dump_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Load cached neighbour statistics for a single dump file, if present.

    Returns ``(neighbor_counts, diff_type_counts, types)`` or ``None`` if no
    cache file is found for ``dump_path``.
    """

    cache_path = neighbor_cache_dir(data_dir) / f"{dump_path.name}.npz"
    if not cache_path.exists():
        return None

    data = np.load(cache_path)
    return data["neighbor_counts"], data["diff_type_counts"], data["types"]


def format_neighbor_constraints(
    data_dir: Path,
    dump_files: Sequence[Path],
    max_neighbors: int = 8,
    min_diff_type_neighbors_for_violation: int = 8,
) -> str:
    """Format neighbour-count constraint diagnostics.

    Uses cached neighbour statistics if available. Frames without cache files
    are skipped with a note.
    """

    if not dump_files:
        return "No dump files to analyse for neighbour constraints."

    lines: List[str] = []

    for dump_path in dump_files:
        stats = _load_neighbor_stats_for_file(data_dir, dump_path)
        if stats is None:
            lines.append(f"Frame {dump_path.name}: neighbour cache missing; run preprocess_data.py.")
            continue

        neighbor_counts, diff_type_counts, types = stats

        # Re-use metadata parsing to recover timestep (optional but nice).
        lines_raw = dump_path.read_text().splitlines(keepends=True)
        meta = infer_metadata_from_lines(dump_path, lines_raw)

        too_many_neighbors = int(np.count_nonzero(neighbor_counts > max_neighbors))
        too_many_diff_type = int(
            np.count_nonzero(diff_type_counts >= min_diff_type_neighbors_for_violation)
        )

        avg_neighbors = float(neighbor_counts.mean()) if neighbor_counts.size > 0 else 0.0
        avg_diff_type = (
            float(diff_type_counts.mean()) if diff_type_counts.size > 0 else 0.0
        )

        lines.append(
            f"Frame {dump_path.name} (t = {meta.timestep}): "
            f"{too_many_neighbors} atom(s) with > {max_neighbors} neighbours (ave = {avg_neighbors:.2f}); "
            f"{too_many_diff_type} atom(s) with >= {min_diff_type_neighbors_for_violation} "
            f"neighbours of different type.(ave = {avg_diff_type:.2f})"
        )

    return "\n".join(lines)


def format_mass_diagnostics(diagnostics: Sequence[MassDiagnostic]) -> str:
    """Format per-frame mass diagnostics into a human-readable report."""

    if not diagnostics:
        return "No dump files to analyse."

    lines: List[str] = []
    ref_n = diagnostics[0].n_atoms

    lines.append(f"Reference frame: {diagnostics[0].path.name} (t = {diagnostics[0].timestep}), N = {ref_n}")
    lines.append("")

    for diag in diagnostics:
        if diag.delta_atoms == 0:
            change_str = "0 atom(s) lost"
        elif diag.delta_atoms < 0:
            change_str = f"{-diag.delta_atoms} atom(s) lost"
        else:
            change_str = f"{diag.delta_atoms} atom(s) gained"

        lines.append(
            f"Frame {diag.path.name} (t = {diag.timestep}): N = {diag.n_atoms} ({change_str})"
        )

    # Aggregate statistics
    deltas = [d.delta_atoms for d in diagnostics]
    max_abs_delta = max(abs(d) for d in deltas)
    frac = max_abs_delta / float(ref_n) if ref_n > 0 else 0.0

    lines.append("")
    lines.append(f"Max |ΔN| across frames: {max_abs_delta} atoms ({frac:.3%} of reference)")

    return "\n".join(lines)


def main() -> None:
    """CLI entry point for running mass-conservation checks on training data."""

    project_root = Path(__file__).parent.parent
    data_dir = project_root / DATA_DIR_NAME

    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        return

    dump_files = _load_training_dump_files(project_root)
    if not dump_files:
        print(f"No dump.*.LAMMPS files found under {data_dir}")
        return

    diagnostics = mass_conservation_over_sequence(dump_files, ref_index=0)
    report = format_mass_diagnostics(diagnostics)
    print(report)

    print("\n=== Neighbour constraints ===")
    if neighbor_cache_exists(data_dir):
        neighbor_report = format_neighbor_constraints(data_dir, dump_files)
        print(neighbor_report)
    else:
        print("Neighbour list not cached, run preprocess_data.py to build neighbour statistics.")


if __name__ == "__main__":  # pragma: no cover - CLI guard
    main()
