from pathlib import Path

from src.physics_checks import MassDiagnostic, format_mass_diagnostics, mass_conservation_over_sequence


def _write_fake_dump(path: Path, timestep: int, num_atoms: int) -> None:
    """Write a minimal LAMMPS-style dump header with the given metadata.

    Only the TIMESTEP, NUMBER OF ATOMS, and ATOMS header lines are required
    for infer_metadata_from_lines; no per-atom rows are needed here.
    """

    content = f"""\
ITEM: TIMESTEP
{timestep}
ITEM: NUMBER OF ATOMS
{num_atoms}
ITEM: BOX BOUNDS pp pp pp
0 1
0 1
0 1
ITEM: ATOMS id type x y vx vy
"""
    path.write_text(content)


def test_mass_conservation_over_sequence_computes_deltas_correctly(tmp_path: Path) -> None:
    paths = [tmp_path / f"dump.{i}.LAMMPS" for i in range(3)]

    # Create three dumps with known atom counts 10, 11, 9
    _write_fake_dump(paths[0], timestep=0, num_atoms=10)
    _write_fake_dump(paths[1], timestep=1, num_atoms=11)
    _write_fake_dump(paths[2], timestep=2, num_atoms=9)

    recomputed = mass_conservation_over_sequence(paths, ref_index=0)

    assert [d.n_atoms for d in recomputed] == [10, 11, 9]
    assert [d.delta_atoms for d in recomputed] == [0, 1, -1]


def test_format_mass_diagnostics_includes_frame_messages(tmp_path: Path) -> None:
    paths = [tmp_path / "dump.0.LAMMPS", tmp_path / "dump.1.LAMMPS"]
    for p in paths:
        _write_fake_dump(p, timestep=int(p.stem.split(".")[-1]), num_atoms=10)

    diagnostics = [
        MassDiagnostic(path=paths[0], timestep=0, n_atoms=10, delta_atoms=0),
        MassDiagnostic(path=paths[1], timestep=1, n_atoms=12, delta_atoms=2),
    ]

    report = format_mass_diagnostics(diagnostics)

    assert "Frame dump.0.LAMMPS (t = 0): N = 10 (0 atom(s) lost)" in report
    assert "Frame dump.1.LAMMPS (t = 1): N = 12 (2 atom(s) gained)" in report
    assert "Max |ΔN| across frames: 2 atoms" in report
