import textwrap
from pathlib import Path

from src.preview_data import infer_metadata_from_lines


def _make_fake_dump(tmp_path: Path, *, dim: int = 2) -> Path:
    columns = "id type x y" if dim == 2 else "id type x y z"
    content = f"""\
ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS pp pp pp
0 1
0 1
0 1
ITEM: ATOMS {columns}
1 1 0.1 0.2{(' 0.3' if dim == 3 else '')}
2 2 0.4 0.5{(' 0.6' if dim == 3 else '')}
"""
    path = tmp_path / ("fake2d.dump" if dim == 2 else "fake3d.dump")
    path.write_text(textwrap.dedent(content))
    return path


def test_infer_metadata_2d(tmp_path: Path) -> None:
    dump_path = _make_fake_dump(tmp_path, dim=2)
    lines = dump_path.read_text().splitlines(keepends=True)

    meta = infer_metadata_from_lines(dump_path, lines)

    assert meta.timestep == 0
    assert meta.num_atoms == 2
    assert meta.dimension == 2
    assert meta.columns == ["id", "type", "x", "y"]


def test_infer_metadata_3d(tmp_path: Path) -> None:
    dump_path = _make_fake_dump(tmp_path, dim=3)
    lines = dump_path.read_text().splitlines(keepends=True)

    meta = infer_metadata_from_lines(dump_path, lines)

    assert meta.dimension == 3
    assert meta.columns == ["id", "type", "x", "y", "z"]
