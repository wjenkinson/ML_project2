"""Preview raw LAMMPS simulation data for basic sanity checks.

This module is intended to be both importable (for tests) and executable as a
script. It provides utilities to:

- Inspect one or more dump files and infer dimensionality (2D vs 3D).
- Extract the list of per-atom columns from the ATOMS section.
- Print a lightweight textual summary to stdout.
- Generate an N-frame GIF of atom positions coloured by atom type.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


@dataclass
class DumpMetadata:
    """Lightweight description of a single LAMMPS dump file."""

    file: Path
    timestep: int
    num_atoms: int
    dimension: int
    columns: List[str]


def _read_lines(path: Path) -> List[str]:
    with path.open("r") as f:
        return f.readlines()


def infer_metadata_from_lines(path: Path, lines: Sequence[str]) -> DumpMetadata:
    """Infer timestep, atom count, dimensionality, and columns from dump lines.

    Assumes a typical LAMMPS custom dump of the form::

        ITEM: TIMESTEP
        0
        ITEM: NUMBER OF ATOMS
        1234
        ITEM: BOX BOUNDS ...
        ...
        ITEM: ATOMS id type x y z ...
        <per-atom data rows>
    """

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
    # 'ITEM:', 'ATOMS', '<col1>', '<col2>', ...
    columns = header_tokens[2:]

    # Infer dimensionality from presence of 'z' in the coordinate columns.
    dimension = 3 if "z" in columns else 2

    return DumpMetadata(
        file=path,
        timestep=timestep,
        num_atoms=num_atoms,
        dimension=dimension,
        columns=list(columns),
    )


def load_atoms_2d(path: Path, lines: Sequence[str] | None = None) -> Tuple[np.ndarray, np.ndarray, DumpMetadata]:
    """Load atom types and 2D positions (x, y) from a dump file.

    If the dump is 3D, the z-coordinate is ignored for visualisation purposes.
    Returns (types, coords_xy, metadata).
    """

    if lines is None:
        lines = _read_lines(path)

    meta = infer_metadata_from_lines(path, lines)

    # Locate ATOMS section again to slice out the numerical rows.
    atoms_header_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("ITEM: ATOMS"):
            atoms_header_idx = i
            break

    if atoms_header_idx is None:
        raise ValueError(f"Could not find ATOMS section in {path}")

    data_start = atoms_header_idx + 1
    data_end = data_start + meta.num_atoms
    data_lines = lines[data_start:data_end]

    from io import StringIO

    raw = np.loadtxt(StringIO("".join(data_lines)))
    if raw.ndim == 1:
        raw = raw[np.newaxis, :]

    # Column indices
    col_indices: Dict[str, int] = {name: idx for idx, name in enumerate(meta.columns)}

    if "type" not in col_indices or "x" not in col_indices or "y" not in col_indices:
        raise ValueError(
            f"Dump {path} must contain at least 'id type x y' columns; got {meta.columns}"
        )

    types = raw[:, col_indices["type"]].astype(int)
    xs = raw[:, col_indices["x"]]
    ys = raw[:, col_indices["y"]]
    coords = np.stack([xs, ys], axis=1)

    return types, coords, meta


def summarize_dataset(dump_files: Iterable[Path]) -> str:
    """Produce a short, human-readable summary string for the dataset."""

    dump_files = list(dump_files)
    if not dump_files:
        return "No dump files found. Expected files like dump.0.LAMMPS in data/."

    metas = [infer_metadata_from_lines(p, _read_lines(p)) for p in dump_files]

    timesteps = [m.timestep for m in metas]
    dimensions = {m.dimension for m in metas}
    columns_sets = {tuple(m.columns) for m in metas}
    n_atoms = {m.num_atoms for m in metas}

    lines = []
    lines.append(f"Files analysed: {len(dump_files)}")
    lines.append(f"Timestep range: {min(timesteps)} .. {max(timesteps)}")
    lines.append(f"Dimensionality: {', '.join(str(d) + 'D' for d in sorted(dimensions))}")
    lines.append(f"Unique atom counts across files: {sorted(n_atoms)}")
    lines.append("")
    lines.append("Column layout(s):")
    for cols in sorted(columns_sets):
        lines.append("  - " + ", ".join(cols))

    return "\n".join(lines)


def _pick_sample_files(dump_files: Sequence[Path], n_frames: int) -> List[Path]:
    if not dump_files:
        return []
    n_frames = min(n_frames, len(dump_files))
    if n_frames <= 0:
        return []
    indices = np.linspace(0, len(dump_files) - 1, num=n_frames, dtype=int)
    unique_indices = sorted(set(int(i) for i in indices))
    return [dump_files[i] for i in unique_indices]


def generate_atom_type_gif(
    dump_files: Sequence[Path],
    output_path: Path,
    n_frames: int = 8,
) -> Path:
    """Generate a GIF of atom positions coloured by type.

    The function samples up to ``n_frames`` dump files from ``dump_files``,
    projects to (x, y), and writes a GIF to ``output_path``. It returns the
    output path for convenience.
    """

    sample_files = _pick_sample_files(dump_files, n_frames)
    if not sample_files:
        raise ValueError("No dump files provided for GIF generation")

    import matplotlib

    # Use a non-interactive backend suitable for headless environments and
    # GIF generation. This avoids issues with GUI canvases that do not
    # implement ``tostring_rgb``.
    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import imageio.v2 as imageio

    frames: List[np.ndarray] = []
    x_min, x_max = float("inf"), float("-inf")
    y_min, y_max = float("inf"), float("-inf")

    # First pass: load data and track global bounds.
    loaded: List[Tuple[np.ndarray, np.ndarray]] = []
    for path in sample_files:
        types, coords, _ = load_atoms_2d(path)
        loaded.append((types, coords))
        x_min = min(x_min, coords[:, 0].min())
        x_max = max(x_max, coords[:, 0].max())
        y_min = min(y_min, coords[:, 1].min())
        y_max = max(y_max, coords[:, 1].max())

    fig, ax = plt.subplots(figsize=(5, 5))
    for types, coords in loaded:
        ax.clear()
        ax.scatter(coords[:, 0], coords[:, 1], c=types, s=5, cmap="tab10")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])

        fig.canvas.draw()
        # Use the backend-agnostic buffer_rgba() API and drop the alpha
        # channel to obtain an RGB image array.
        image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        image = image[..., :3]
        frames.append(image.copy())

    plt.close(fig)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, frames, fps=4, loop=0)
    return output_path


def main() -> None:
    """Entry point for CLI usage.

    - Scans ``data/`` for files named ``dump.*.LAMMPS``.
    - Prints a brief summary of dimensionality and columns.
    - Writes an 8-frame GIF to ``output/preview_atoms.gif``.
    """

    data_dir = Path("data")
    if not data_dir.exists():
        print("data/ directory not found. Nothing to preview.")
        return

    dump_files = sorted(data_dir.glob("dump.*.LAMMPS"))
    if not dump_files:
        print("No dump.*.LAMMPS files found in data/.")
        return

    print("=== Dataset summary ===")
    print(summarize_dataset(dump_files))

    output_path = Path("output") / "preview_atoms.gif"
    try:
        generate_atom_type_gif(dump_files, output_path, n_frames=8)
    except Exception as exc:  # pragma: no cover - defensive path
        print(f"Failed to generate GIF: {exc}")
    else:
        print("")
        print(f"Preview GIF written to: {output_path}")


if __name__ == "__main__":  # pragma: no cover - CLI guard
    main()
