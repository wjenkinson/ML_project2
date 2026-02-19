from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset
from torch_geometric.data import Data

from .preprocess_data import timestep_from_name


# Node feature channels: [is_fluid, is_solid, is_wall, is_piston, vx, vy]
NODE_FEATURE_DIM = 6

# Piston identification: type 1 atoms with y > this threshold at frame 0
PISTON_Y_THRESHOLD = 0.101

# Periodic box bounds in x (from LAMMPS BOX BOUNDS)
BOX_X_MIN = -0.1
BOX_X_MAX = 0.1
BOX_LX = BOX_X_MAX - BOX_X_MIN  # 0.2


def build_radius_graph_pbc_x(
    coords: np.ndarray,
    radius: float,
    x_min: float = BOX_X_MIN,
    x_max: float = BOX_X_MAX,
) -> Tuple[List[int], List[int]]:
    """Build radius-based neighbor graph with periodic boundary conditions in x.

    Uses ghost particles near the x boundaries to capture neighbors across
    the periodic boundary.

    Returns:
        row, col: edge lists (source, destination) for the graph
    """
    N = coords.shape[0]
    Lx = x_max - x_min

    x_coords = coords[:, 0]

    # Find particles near boundaries that need ghosts
    near_left = x_coords < (x_min + radius)
    near_right = x_coords > (x_max - radius)

    # Create ghost coordinates
    ghost_coords_list = [coords]  # start with real particles
    ghost_map = list(range(N))  # ghost_map[ghost_idx] = real_idx

    # Ghosts for left-boundary particles (appear on the right)
    if np.any(near_left):
        left_ghosts = coords[near_left].copy()
        left_ghosts[:, 0] += Lx
        ghost_coords_list.append(left_ghosts)
        ghost_map.extend(np.where(near_left)[0].tolist())

    # Ghosts for right-boundary particles (appear on the left)
    if np.any(near_right):
        right_ghosts = coords[near_right].copy()
        right_ghosts[:, 0] -= Lx
        ghost_coords_list.append(right_ghosts)
        ghost_map.extend(np.where(near_right)[0].tolist())

    all_coords = np.vstack(ghost_coords_list)  # (N + num_ghosts, 3)

    # Build neighbor list on extended coordinates
    nbrs = NearestNeighbors(radius=radius, algorithm="ball_tree")
    nbrs.fit(all_coords)
    indices = nbrs.radius_neighbors(coords, return_distance=False)  # query only real particles

    row: List[int] = []
    col: List[int] = []
    for i, neigh in enumerate(indices):
        for j in neigh:
            if j == i:
                continue  # no self-loops
            # Map ghost indices back to real particle indices
            real_j = ghost_map[j]
            if real_j == i:
                continue  # don't connect particle to its own ghost
            row.append(i)
            col.append(real_j)

    return row, col


def minimum_image_rel_pos(
    pos_src: torch.Tensor,
    pos_dst: torch.Tensor,
    Lx: float = BOX_LX,
) -> torch.Tensor:
    """Compute relative position with minimum image convention in x.

    Returns pos_dst - pos_src, but wraps the x-component to [-Lx/2, Lx/2].
    """
    rel = pos_dst - pos_src  # (E, 3)
    # Wrap x-component
    rel_x = rel[:, 0]
    rel_x = rel_x - Lx * torch.round(rel_x / Lx)
    rel = rel.clone()
    rel[:, 0] = rel_x
    return rel


class LammpsGraphDataset(Dataset):
    """Graph dataset for particle-based SPH simulations using all particles.

    For a given split (train/val), this dataset:
    - Reads dump filenames from ``data/splits/{split}_files.txt``.
    - Sorts them by numeric timestep inferred from the filename.
    - Loads all frames into memory (positions, types, velocities), sorted by
      atom ID so that nodes are consistently ordered across time.
    - Builds consecutive frame pairs (t, t+1).

    Each sample is a torch_geometric.data.Data object with fields:
    - x: node features [is_fluid, is_solid, is_wall, is_piston, vx, vy], shape (N, 6)
    - pos: positions at time t, shape (N, 3)
    - edge_index: graph connectivity from radius-based neighborhood at time t
    - edge_attr: relative position vectors [Δx, Δy, Δz, ||r||] per edge, shape (E, 4)
    - y: target DISPLACEMENT (pos_tp1 - pos_t), shape (N, 3)
    - atom_type: integer atom type IDs (1=fluid, 2=solid, 3=wall, 4=piston), shape (N,)

    Piston atoms are identified at frame 0 as type 1 atoms with y > PISTON_Y_THRESHOLD,
    then relabeled as type 4 across all frames.
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

        names_sorted = sorted(names, key=lambda n: timestep_from_name(Path(n)))

        # Load all frames into memory, keyed by filename
        self.frames: Dict[str, Dict[str, torch.Tensor]] = {}

        # Identify piston atom IDs once from the earliest training frame so
        # that both train and validation splits share the same piston set.
        piston_ids = self._compute_piston_ids()
        if piston_ids:
            print(f"Identified {len(piston_ids)} piston atoms (type 1, y > {PISTON_Y_THRESHOLD})")

        for name in names_sorted:
            path = self.data_dir / name
            atoms = self._read_atoms(path)

            # Sort by atom ID for consistent ordering across frames
            sort_idx = np.argsort(atoms["id"])
            sorted_ids = atoms["id"][sort_idx]

            # Positions: (x, y, 0.0) for 2D simulation
            pos = np.zeros((len(sort_idx), 3), dtype=np.float32)
            pos[:, 0] = atoms["x"][sort_idx]
            pos[:, 1] = atoms["y"][sort_idx]

            # Velocities: (vx, vy)
            vel = np.zeros((len(sort_idx), 2), dtype=np.float32)
            vel[:, 0] = atoms["vx"][sort_idx]
            vel[:, 1] = atoms["vy"][sort_idx]

            atom_type = atoms["type"][sort_idx].copy()

            # Relabel piston atoms as type 4 across all frames
            if piston_ids:
                for i, aid in enumerate(sorted_ids):
                    if aid in piston_ids:
                        atom_type[i] = 4

            self.frames[name] = {
                "pos": torch.from_numpy(pos),
                "vel": torch.from_numpy(vel),
                "atom_type": torch.from_numpy(atom_type.astype(np.int64)),
            }

        # Build consecutive pairs (t, t+1)
        self.pairs: List[Tuple[str, str]] = []
        for i in range(len(names_sorted) - 1):
            self.pairs.append((names_sorted[i], names_sorted[i + 1]))

    def _compute_piston_ids(self) -> set[int]:
        """Identify piston atom IDs from the earliest training frame."""

        splits_dir = self.data_dir / "splits"
        train_list_path = splits_dir / "train_files.txt"

        if not train_list_path.exists():
            return set()

        with train_list_path.open("r", encoding="utf-8") as f:
            names = [line.strip() for line in f if line.strip()]

        if not names:
            return set()

        # Use globally earliest training frame.
        names_sorted = sorted(names, key=lambda n: timestep_from_name(Path(n)))
        first_name = names_sorted[0]
        path = self.data_dir / first_name

        atoms = self._read_atoms(path)
        ids = atoms["id"]
        y = atoms["y"]
        types = atoms["type"]

        mask = (types == 1) & (y > PISTON_Y_THRESHOLD)
        return set(ids[mask])

    def _read_atoms(self, path: Path) -> dict:
        """Read per-atom data from a LAMMPS dump file.

        Returns a dict with keys: id, type, x, y, vx, vy (all as numpy arrays).
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

        # Extract columns by name (handles any column order)
        return {
            "id": raw[:, col_indices["id"]].astype(int),
            "type": raw[:, col_indices["type"]].astype(int),
            "x": raw[:, col_indices["x"]].astype(np.float32),
            "y": raw[:, col_indices["y"]].astype(np.float32),
            "vx": raw[:, col_indices["vx"]].astype(np.float32),
            "vy": raw[:, col_indices["vy"]].astype(np.float32),
        }

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Data:  # type: ignore[override]
        name_t, name_tp1 = self.pairs[idx]

        frame_t = self.frames[name_t]
        frame_tp1 = self.frames[name_tp1]

        pos_t = frame_t["pos"]  # (N, 3)
        vel_t = frame_t["vel"]  # (N, 2)
        atom_type = frame_t["atom_type"]  # (N,)

        pos_tp1 = frame_tp1["pos"]  # (N, 3)

        if pos_t.shape != pos_tp1.shape:
            raise ValueError(
                f"Mismatched shapes between {name_t} and {name_tp1}: "
                f"{pos_t.shape} vs {pos_tp1.shape}"
            )

        # Target: displacement instead of absolute position
        displacement = pos_tp1 - pos_t  # (N, 3)

        # Node features: one-hot type encoding + velocities
        # Types: 1=fluid, 2=solid, 3=wall, 4=piston
        N = pos_t.shape[0]
        is_fluid = (atom_type == 1).to(torch.float32).unsqueeze(-1)   # (N, 1)
        is_solid = (atom_type == 2).to(torch.float32).unsqueeze(-1)   # (N, 1)
        is_wall = (atom_type == 3).to(torch.float32).unsqueeze(-1)    # (N, 1)
        is_piston = (atom_type == 4).to(torch.float32).unsqueeze(-1)  # (N, 1)

        # Combine: [is_fluid, is_solid, is_wall, is_piston, vx, vy]
        x = torch.cat([is_fluid, is_solid, is_wall, is_piston, vel_t], dim=-1)  # (N, 6)

        # Radius-based neighborhood graph with periodic BC in x
        coords = pos_t.cpu().numpy()  # (N, 3)
        row, col = build_radius_graph_pbc_x(coords, self.radius)
        edge_index = torch.tensor([row, col], dtype=torch.long)

        # Edge attributes: relative position vectors + distance (with PBC in x)
        if edge_index.numel() > 0:
            src, dst = edge_index[0], edge_index[1]
            rel_pos = minimum_image_rel_pos(pos_t[src], pos_t[dst])  # (E, 3)
            dist = torch.norm(rel_pos, dim=-1, keepdim=True)  # (E, 1)
            edge_attr = torch.cat([rel_pos, dist], dim=-1)  # (E, 4)
        else:
            edge_attr = torch.zeros((0, 4), dtype=torch.float32)

        data = Data(
            x=x,
            pos=pos_t,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=displacement,  # target is now displacement
            atom_type=atom_type,
        )

        return data
