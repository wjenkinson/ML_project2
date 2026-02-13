from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset
from torch_geometric.data import Data


class LammpsGraphDataset(Dataset):
    """Graph dataset for particle-based SPH simulations using all particles.

    For a given split (train/val), this dataset:
    - Reads dump filenames from ``data/splits/{split}_files.txt``.
    - Sorts them by numeric timestep inferred from the filename.
    - Loads all frames into memory (positions and types), sorted by atom ID so
      that nodes are consistently ordered across time.
    - Builds consecutive frame pairs (t, t+1).

    Each sample is a torch_geometric.data.Data object with fields:
    - x: node features [x, y, z, type_id] at time t, shape (N, 4)
         (z is zero for this 2D simulation.)
    - pos: positions at time t, shape (N, 3)
    - edge_index: graph connectivity from radius-based neighborhood at time t
    - y: target positions at time t+1, shape (N, 3)
    - atom_type: integer atom type IDs, shape (N,)
    """

    def __init__(
        self,
        split: str,
        radius: float = 0.002,
    ) -> None:
        if split not in {"train", "val"}:
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")

        self.radius = radius

        project_root = Path(__file__).parent.parent
        self.data_dir = project_root / "data"
        splits_dir = self.data_dir / "splits"
        list_path = splits_dir / f"{split}_files.txt"

        if not list_path.exists():
            raise FileNotFoundError(f"Split file not found: {list_path}")

        with list_path.open("r", encoding="utf-8") as f:
            names = [line.strip() for line in f if line.strip()]

        def timestep_from_name(name: str) -> int:
            try:
                parts = name.split(".")
                return int(parts[1])
            except (IndexError, ValueError):
                return 0

        names_sorted = sorted(names, key=timestep_from_name)

        # Load all frames into memory, keyed by filename
        self.frames: Dict[str, Dict[str, torch.Tensor]] = {}

        for name in names_sorted:
            path = self.data_dir / name
            atoms = self._read_atoms(path)

            if atoms.shape[1] < 6:
                raise ValueError(
                    f"Expected at least 6 columns (id, type, x, y, vx, vy) in atoms for {name}, got {atoms.shape[1]}"
                )

            ids = atoms[:, 0].astype(int)
            sort_idx = np.argsort(ids)
            atoms_sorted = atoms[sort_idx]

            # Positions: (x, y, 0.0) for 2D simulation
            pos = np.zeros((atoms_sorted.shape[0], 3), dtype=np.float32)
            pos[:, 0] = atoms_sorted[:, 2].astype(np.float32)  # x
            pos[:, 1] = atoms_sorted[:, 3].astype(np.float32)  # y

            pos_t = torch.from_numpy(pos)  # (N, 3)
            atom_type = torch.from_numpy(atoms_sorted[:, 1].astype(np.int64))  # (N,)

            self.frames[name] = {
                "pos": pos_t,
                "atom_type": atom_type,
            }

        # Build consecutive pairs (t, t+1)
        self.pairs: List[Tuple[str, str]] = []
        for i in range(len(names_sorted) - 1):
            self.pairs.append((names_sorted[i], names_sorted[i + 1]))

    def _read_atoms(self, path: Path) -> np.ndarray:
        """Read per-atom data from a LAMMPS dump file as a NumPy array.

        Assumes columns include at least: id, type, x, y, vx, vy.
        """

        lines = path.read_text().splitlines(keepends=True)

        num_atoms = None
        atoms_header_idx = None

        for i, line in enumerate(lines):
            s = line.strip()
            if s.startswith("ITEM: NUMBER OF ATOMS") and i + 1 < len(lines):
                num_atoms = int(lines[i + 1].strip())
            elif s.startswith("ITEM: ATOMS"):
                atoms_header_idx = i

        if num_atoms is None or atoms_header_idx is None:
            raise ValueError(f"File {path} does not look like a valid LAMMPS dump")

        header_tokens = lines[atoms_header_idx].strip().split()
        columns = header_tokens[2:]
        col_indices = {name: idx for idx, name in enumerate(columns)}

        required = {"id", "type", "x", "y", "vx", "vy"}
        missing = required - set(columns)
        if missing:
            raise ValueError(f"Dump {path} is missing required columns: {missing}. Got {columns}")

        data_start = atoms_header_idx + 1
        data_end = data_start + num_atoms
        data_lines = lines[data_start:data_end]

        raw = np.loadtxt("".join(data_lines).splitlines())
        if raw.ndim == 1:
            raw = raw[np.newaxis, :]

        return raw

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Data:  # type: ignore[override]
        name_t, name_tp1 = self.pairs[idx]

        frame_t = self.frames[name_t]
        frame_tp1 = self.frames[name_tp1]

        pos_t = frame_t["pos"]  # (N, 3)
        atom_type = frame_t["atom_type"]  # (N,)

        pos_tp1 = frame_tp1["pos"]  # (N, 3)

        if pos_t.shape != pos_tp1.shape:
            raise ValueError(
                f"Mismatched shapes between {name_t} and {name_tp1}: "
                f"{pos_t.shape} vs {pos_tp1.shape}"
            )

        # Node features: [x, y, z, type_id]
        x = torch.cat([
            pos_t,
            atom_type.to(dtype=torch.float32).unsqueeze(-1),
        ], dim=-1)

        # Radius-based neighborhood graph (first neighbors) using scikit-learn
        coords = pos_t.cpu().numpy()  # (N, 3)
        nbrs = NearestNeighbors(radius=self.radius, algorithm="ball_tree")
        nbrs.fit(coords)
        indices = nbrs.radius_neighbors(return_distance=False)

        row: List[int] = []
        col: List[int] = []
        for i, neigh in enumerate(indices):
            for j in neigh:
                if i == j:
                    continue  # no self-loops
                row.append(i)
                col.append(j)

        edge_index = torch.tensor([row, col], dtype=torch.long)

        data = Data(
            x=x,
            pos=pos_t,
            edge_index=edge_index,
            y=pos_tp1,
            atom_type=atom_type,
        )

        return data
