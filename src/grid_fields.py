"""Differentiable grid-based field computations for density and velocity.

Particles are splatted onto a regular 2D grid using a smooth SPH cubic spline
kernel.  Because the kernel is differentiable w.r.t. particle positions, the
resulting field losses back-propagate gradients to the displacement head.

Grid fields replace per-node losses for fluid atoms, allowing fungible
particles to be evaluated on collective behaviour rather than individual
identity.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import Tensor

from .graph_dataset import BOX_X_MIN, BOX_X_MAX, BOX_LX
from .preprocess_data import NEIGHBOR_RADIUS


# ---------------------------------------------------------------------------
# Domain / grid defaults
# ---------------------------------------------------------------------------

GRID_X_MIN = BOX_X_MIN   # -0.1
GRID_X_MAX = BOX_X_MAX   #  0.1
GRID_Y_MIN = -0.08
GRID_Y_MAX = 0.13

GRID_CELL_SIZE = NEIGHBOR_RADIUS   # 0.003
GRID_H = NEIGHBOR_RADIUS           # smoothing length (support = 2h = 0.006)

# Particle mass: rho0=1000, L0=0.001 -> m = rho0 * L0^2 = 0.001
PARTICLE_MASS = 0.001
RHO0 = 1000.0


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------

def _cubic_spline_2d(dist: Tensor, h: float) -> Tensor:
    """Differentiable 2D cubic spline kernel with compact support 2h."""

    sigma = 10.0 / (7.0 * math.pi * h * h)
    q = dist / h

    w = torch.zeros_like(q)

    mask1 = q < 1.0
    if mask1.any():
        q1 = q[mask1]
        w[mask1] = 1.0 - 1.5 * q1.pow(2) + 0.75 * q1.pow(3)

    mask2 = (q >= 1.0) & (q < 2.0)
    if mask2.any():
        q2 = q[mask2]
        w[mask2] = 0.25 * (2.0 - q2).pow(3)

    return sigma * w


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

def _grid_centers(lo: float, hi: float, cell: float, device: torch.device) -> Tensor:
    """Return 1-D cell centres covering [lo, hi) with spacing *cell*."""

    n = max(1, int(math.ceil((hi - lo) / cell)))
    half = cell / 2.0
    return torch.linspace(lo + half, lo + half + (n - 1) * cell, n, device=device)


def make_grid(device: torch.device = torch.device("cpu")) -> Tuple[Tensor, Tensor]:
    """Create default grid centres for the simulation domain."""

    grid_x = _grid_centers(GRID_X_MIN, GRID_X_MAX, GRID_CELL_SIZE, device)
    grid_y = _grid_centers(GRID_Y_MIN, GRID_Y_MAX, GRID_CELL_SIZE, device)
    return grid_x, grid_y


def _local_patch_weights(
    pos_2d: Tensor,
    grid_x: Tensor,
    grid_y: Tensor,
    h: float,
    periodic_lx: float | None = BOX_LX,
) -> Tuple[Tensor, Tensor]:
    """Compute SPH kernel weights using only local grid patches (within 2h).

    Instead of the dense O(N * Gy * Gx) approach, each particle only
    evaluates the kernel at the ~(2*ceil(2h/cell)+1)^2 nearby cells,
    reducing memory by ~100-200x.

    Returns:
        w:        (N, P)  kernel weights, P = patch_y * patch_x
        flat_idx: (N, P)  flattened (row-major) grid indices into (Gy, Gx)
    """

    N = pos_2d.shape[0]
    device = pos_2d.device
    Gx = grid_x.shape[0]
    Gy = grid_y.shape[0]

    cell_x = (grid_x[-1] - grid_x[0]).item() / max(Gx - 1, 1)
    cell_y = (grid_y[-1] - grid_y[0]).item() / max(Gy - 1, 1)

    # Half-width of the local patch in cells (covers full 2h support)
    hw_x = int(math.ceil(2.0 * h / cell_x))
    hw_y = int(math.ceil(2.0 * h / cell_y))

    # Offsets from centre cell
    ox = torch.arange(-hw_x, hw_x + 1, device=device)           # (pw_x,)
    oy = torch.arange(-hw_y, hw_y + 1, device=device)           # (pw_y,)

    # Nearest cell index for each particle (detached — indices not differentiable)
    cx = (pos_2d[:, 0] - grid_x[0]).div(cell_x).round().long()  # (N,)
    cy = (pos_2d[:, 1] - grid_y[0]).div(cell_y).round().long()  # (N,)

    # Cell indices for each particle's local patch: (N, pw_y, pw_x)
    ix = cx[:, None, None] + ox[None, None, :]
    iy = cy[:, None, None] + oy[None, :, None]

    # Periodic wrap in x; y stays clamped
    if periodic_lx is not None:
        ix = ix % Gx
        valid = (iy >= 0) & (iy < Gy)
    else:
        valid = (ix >= 0) & (ix < Gx) & (iy >= 0) & (iy < Gy)

    ix = ix.clamp(0, Gx - 1)
    iy = iy.clamp(0, Gy - 1)

    # Cell centres for the patch (no grad needed)
    gx_p = grid_x[ix]                                            # (N, pw_y, pw_x)
    gy_p = grid_y[iy]                                            # (N, pw_y, pw_x)

    # Displacement from particle to cell centre (differentiable w.r.t. pos_2d)
    dx = pos_2d[:, 0, None, None] - gx_p
    dy = pos_2d[:, 1, None, None] - gy_p

    if periodic_lx is not None:
        dx = dx - periodic_lx * torch.round(dx / periodic_lx)

    dist = (dx * dx + dy * dy + 1e-12).sqrt()                    # (N, pw_y, pw_x)

    w = _cubic_spline_2d(dist, h) * valid.float()                # (N, pw_y, pw_x)

    flat_idx = iy * Gx + ix                                      # (N, pw_y, pw_x)

    return w.reshape(N, -1), flat_idx.reshape(N, -1)


# ---------------------------------------------------------------------------
# Splatting
# ---------------------------------------------------------------------------

def splat_density(
    pos_2d: Tensor,
    grid_x: Tensor,
    grid_y: Tensor,
    h: float = GRID_H,
    mass: float = PARTICLE_MASS,
    periodic_lx: float | None = BOX_LX,
) -> Tensor:
    """Splat particle mass onto a 2D grid -> density field (Gy, Gx).

    Returns density **normalised by RHO0** so typical values are ~1.0.
    """

    Gx_n, Gy_n = grid_x.shape[0], grid_y.shape[0]
    w, idx = _local_patch_weights(pos_2d, grid_x, grid_y, h, periodic_lx)

    rho_flat = torch.zeros(Gy_n * Gx_n, device=pos_2d.device, dtype=pos_2d.dtype)
    rho_flat.scatter_add_(0, idx.reshape(-1), (mass * w).reshape(-1))

    return rho_flat.reshape(Gy_n, Gx_n) / RHO0


def splat_velocity(
    pos_2d: Tensor,
    vel_2d: Tensor,
    grid_x: Tensor,
    grid_y: Tensor,
    h: float = GRID_H,
    mass: float = PARTICLE_MASS,
    periodic_lx: float | None = BOX_LX,
    eps: float = 1e-8,
) -> Tensor:
    """Splat particle velocity onto a 2D grid -> velocity field (Gy, Gx, 2).

    Uses SPH interpolation: v_grid = sum(m * v * W) / sum(m * W).
    Cells with negligible density are zeroed out.
    """

    Gx_n, Gy_n = grid_x.shape[0], grid_y.shape[0]
    G = Gy_n * Gx_n
    D = vel_2d.shape[1]

    w, idx = _local_patch_weights(pos_2d, grid_x, grid_y, h, periodic_lx)

    # Density for normalisation  (Gy, Gx)
    rho_flat = torch.zeros(G, device=pos_2d.device, dtype=pos_2d.dtype)
    rho_flat.scatter_add_(0, idx.reshape(-1), (mass * w).reshape(-1))
    rho = rho_flat.reshape(Gy_n, Gx_n)

    # Momentum: mass * v_p * W(p, cell)  ->  (Gy, Gx, D)
    # w: (N, P), vel_2d: (N, D) -> weighted: (N, P, D)
    weighted = mass * w.unsqueeze(-1) * vel_2d[:, None, :]         # (N, P, D)
    idx_d = idx.unsqueeze(-1).expand(-1, -1, D)                    # (N, P, D)

    mom_flat = torch.zeros(G, D, device=pos_2d.device, dtype=pos_2d.dtype)
    mom_flat.scatter_add_(0, idx_d.reshape(-1, D), weighted.reshape(-1, D))
    momentum = mom_flat.reshape(Gy_n, Gx_n, D)

    vel_field = momentum / (rho.unsqueeze(-1) + eps)

    # Zero out where density is negligible
    low_density = rho < eps * 10
    vel_field[low_density] = 0.0

    return vel_field


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def grid_density_loss(
    pred_pos: Tensor,
    gt_pos: Tensor,
    atom_type: Tensor,
    grid_x: Tensor,
    grid_y: Tensor,
    h: float = GRID_H,
) -> Tensor:
    """MSE between predicted and GT normalised density grids (fluid only)."""

    fluid_mask = atom_type == 1
    pred_fluid = pred_pos[fluid_mask, :2]
    gt_fluid = gt_pos[fluid_mask, :2]

    if pred_fluid.shape[0] == 0:
        return torch.zeros((), device=pred_pos.device)

    rho_pred = splat_density(pred_fluid, grid_x, grid_y, h)

    with torch.no_grad():
        rho_gt = splat_density(gt_fluid, grid_x, grid_y, h)

    return (rho_pred - rho_gt).pow(2).mean()


def grid_velocity_loss(
    pred_pos: Tensor,
    pred_vel: Tensor,
    gt_pos: Tensor,
    gt_vel: Tensor,
    atom_type: Tensor,
    grid_x: Tensor,
    grid_y: Tensor,
    h: float = GRID_H,
) -> Tensor:
    """MSE between predicted and GT velocity fields on grid (fluid only)."""

    fluid_mask = atom_type == 1
    pred_fluid_pos = pred_pos[fluid_mask, :2]
    pred_fluid_vel = pred_vel[fluid_mask]
    gt_fluid_pos = gt_pos[fluid_mask, :2]
    gt_fluid_vel = gt_vel[fluid_mask]

    if pred_fluid_pos.shape[0] == 0:
        return torch.zeros((), device=pred_pos.device)

    vel_pred = splat_velocity(pred_fluid_pos, pred_fluid_vel, grid_x, grid_y, h)

    with torch.no_grad():
        vel_gt = splat_velocity(gt_fluid_pos, gt_fluid_vel, grid_x, grid_y, h)

    return (vel_pred - vel_gt).pow(2).mean()
