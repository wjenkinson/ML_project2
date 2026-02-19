# Physics Notes

## System Description

- **Scenario:** 2D extrusion of a Newtonian liquid with a solid inclusion.
- **Phases / atom types:**
  - `type = 1`: water (liquid phase)
  - `type = 2`: solid inclusion
  - `type = 3`: rigid walls
- **Assumptions:**
  - Solid and liquid are weakly compressible in the numerical scheme but are
    treated as effectively incompressible for analysis.
  - Simulation is represented as per-atom data with columns:
    `id, type, x, y, vx, vy`.

## Active Physics Constraint

**SPH Density Constraint** — the only physics-informed loss currently in use.

- Per-particle density is estimated using a 2D cubic spline SPH kernel:
  - `rho_i = sum_j m_j * W(r_ij, h)`
  - Smoothing length `h = NEIGHBOR_RADIUS / 2 = 0.001`
  - Particle mass `m = rho0 * L0^2 = 0.001` (derived from reference density
    1000 and lattice spacing 0.001)
- Loss penalises particles outside the band `[950, 1050]` kg/m³.
- Kernel support is `2h`, matching the neighbour cutoff used for graph edges.

## Removed Constraints (see `postmortem.md`)

The following constraints were explored but removed to simplify the codebase:

| Constraint | Reason for removal |
|------------|-------------------|
| Mass / atom-count conservation | Already enforced by GNN architecture (node count fixed) |
| Neighbour count | Caused clumping artifacts; hard to tune |
| Rigid-wall velocity | Handled by clamping in prediction loop instead of loss |
| Interface sharpness | Never fully implemented; low leverage |

## Diagnostics

`src/physics_checks.py` provides SPH density diagnostics on raw LAMMPS dumps:

```bash
python -m src.physics_checks
```

Reports per-frame `rho_avg`, `rho_min`, `rho_max`, and fraction of particles
outside ±10% of reference density (1000 kg/m³).

## Future Directions

- Incorporate **relative positions** in node features.
- Add **velocities** (`vx`, `vy`) to feature vector.
- Re-enable density constraint once core GNN training is improved.
