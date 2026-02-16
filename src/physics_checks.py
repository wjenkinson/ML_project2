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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
from sklearn.neighbors import NearestNeighbors

from .preprocess_data import (
    DATA_DIR_NAME,
    SPLITS_SUBDIR,
    list_dump_files,
    neighbor_cache_dir,
    neighbor_cache_exists,
    NEIGHBOR_RADIUS,
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
    min_neighbors: int = 4,
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
        too_few_neighbors = int(np.count_nonzero(neighbor_counts < min_neighbors))
        too_many_diff_type = int(
            np.count_nonzero(diff_type_counts >= min_diff_type_neighbors_for_violation)
        )

        avg_neighbors = float(neighbor_counts.mean()) if neighbor_counts.size > 0 else 0.0
        avg_diff_type = (
            float(diff_type_counts.mean()) if diff_type_counts.size > 0 else 0.0
        )

        lines.append(
            f"Frame {dump_path.name} (t = {meta.timestep}): "
            f"{too_many_neighbors} atom(s) with > {max_neighbors} neighbours; "
            f"{too_few_neighbors} atom(s) with < {min_neighbors} neighbours (ave = {avg_neighbors:.2f}); "
            f"{too_many_diff_type} atom(s) with >= {min_diff_type_neighbors_for_violation} "
            f"neighbours of different type.(ave = {avg_diff_type:.2f})"
        )

    return "\n".join(lines)

def _load_types_and_coords_for_density(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load atom types and 2D coordinates (x, y) from a dump file.

    This duplicates the minimal parsing logic from ``preprocess_data`` so that
    density checks can operate on raw frames without going through the cache.
    """

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

    # Use NumPy's text loader for robustness.
    raw = np.loadtxt("".join(data_lines).splitlines())
    if raw.ndim == 1:
        raw = raw[np.newaxis, :]

    types = raw[:, col_indices["type"]].astype(int)
    xs = raw[:, col_indices["x"]]
    ys = raw[:, col_indices["y"]]
    coords = np.stack([xs, ys], axis=1)

    return types, coords


def _sph_cubic_spline_kernel_2d(distances: np.ndarray, h: float) -> np.ndarray:
    """2D cubic spline SPH kernel with compact support 2h.

    Returns W(r, h) evaluated elementwise for ``distances``.
    """

    if h <= 0.0:
        raise ValueError("Smoothing length h must be positive")

    q = distances / h
    w = np.zeros_like(distances, dtype=float)

    # Normalisation constant for 2D cubic spline.
    sigma = 10.0 / (7.0 * np.pi * h * h)

    mask1 = (q >= 0.0) & (q < 1.0)
    if np.any(mask1):
        q1 = q[mask1]
        w[mask1] = 1.0 - 1.5 * q1 * q1 + 0.75 * q1 * q1 * q1

    mask2 = (q >= 1.0) & (q < 2.0)
    if np.any(mask2):
        q2 = q[mask2]
        w[mask2] = 0.25 * (2.0 - q2) ** 3

    return sigma * w


def compute_sph_densities_for_dump(dump_path: Path, h: float) -> np.ndarray:
    """Compute SPH kernel density estimates for all particles in a dump frame."""

    _types, coords = _load_types_and_coords_for_density(dump_path)

    if coords.size == 0:
        return np.zeros(0, dtype=float)

    # Particle mass based on reference density rho0=1000 and lattice spacing
    # L0=0.001: m = rho0 * L0^2 = 1000 * (0.001)^2 = 0.001.
    mass_per_particle = 0.001

    # Use neighbours within 2h, which is the support of the cubic spline kernel.
    nn = NearestNeighbors(radius=2.0 * h)
    nn.fit(coords)
    # sklearn returns (distances, indices) when return_distance=True.
    dist_list, indices_list = nn.radius_neighbors(coords, return_distance=True)

    densities = np.zeros(len(coords), dtype=float)

    for i, (dists, idxs) in enumerate(zip(dist_list, indices_list)):
        if dists.size == 0:
            continue

        # Include self in the density estimate; for self, distance is zero and
        # the kernel takes its maximum value.
        w = _sph_cubic_spline_kernel_2d(dists, h)
        densities[i] = mass_per_particle * w.sum()

    return densities

def format_sph_density_diagnostics(
    dump_files: Sequence[Path],
    ref_density: float = 1000.0,
    h: float | None = None,
) -> str:
    """Format SPH-style density diagnostics over a sequence of dump files.

    The per-particle density is estimated as ``rho_i = sum_j m_j W(r_ij, h)``
    using the kernel and particle mass as defined in ``compute_sph_densities_for_dump``.
    No additional calibration or rescaling is applied here; densities are
    reported in the natural units implied by ``W`` and ``m_j``. ``ref_density``
    is only used as a reference value when computing fractions below/above a
    tolerance band (e.g. 0.9 * ref_density).
    """

    if not dump_files:
        return "No dump files to analyse for SPH density."

    if h is None:
        # Use half the neighbour radius as smoothing length, so that the kernel
        # support 2h matches the neighbour cutoff.
        h = NEIGHBOR_RADIUS / 2.0

    lines: List[str] = []

    for idx, dump_path in enumerate(dump_files):
        try:
            densities = compute_sph_densities_for_dump(dump_path, h)
        except Exception as exc:  # pragma: no cover - diagnostic only
            lines.append(f"Frame {dump_path.name}: SPH density computation failed: {exc}")
            continue

        if densities.size == 0:
            lines.append(f"Frame {dump_path.name}: no particles for SPH density.")
            continue

        rho_avg = float(densities.mean())
        rho_min = float(densities.min())
        rho_max = float(densities.max())

        frac_low = float(np.mean(densities < 0.9 * ref_density))
        frac_high = float(np.mean(densities > 1.1 * ref_density))

        # Re-use metadata parsing to recover timestep (optional but nice).
        lines_raw = dump_path.read_text().splitlines(keepends=True)
        meta = infer_metadata_from_lines(dump_path, lines_raw)

        lines.append(
            f"Frame {dump_path.name} (t = {meta.timestep}): "
            f"rho_avg={rho_avg:.1f}, rho_min={rho_min:.1f}, rho_max={rho_max:.1f}, "
            f"frac(rho<0.9*rho0)={frac_low:.3f}, frac(rho>1.1*rho0)={frac_high:.3f}"
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

    print("\n=== SPH density diagnostics (kernel-based) ===")
    try:
        density_report = format_sph_density_diagnostics(dump_files, ref_density=1000.0)
        print(density_report)
    except Exception as exc:  # pragma: no cover - diagnostic only
        print(f"SPH density diagnostics failed: {exc}")


if __name__ == "__main__":  # pragma: no cover - CLI guard
    main()
