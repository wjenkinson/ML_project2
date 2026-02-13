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

## Core Physical Constraints

1. **Mass / atom-count conservation**
   - The total number of atoms `N_total = N1 + N2 + N3` should remain
     constant over time.
   - For a fixed simulation domain, this implies bulk density is constant.

2. **Neighbour count constraint**
   - Each atom should have **no more than 8 neighbours** within a chosen
     cutoff radius (packing constraint).
   - Phases are allowed to mix at the neighbour level, but:
     - Atoms that have **8 or more neighbours of a *different* type** are
       considered strongly mixed and should be penalized.

3. **Phase separation (no phase penetration)**
   - Water (type 1) and solid inclusion (type 2) should remain separated
     except at interfaces.
   - Strong cross-phase mixing at the per-atom neighbour level is a proxy for
     phase penetration.

4. **Rigid-wall velocity constraint**
   - Rigid wall atoms (type 3) should have zero velocity:
     - `vx = 0` and `vy = 0` (within numerical tolerance) at all times.

5. **Non-negativity of density**
   - Any future density-like quantity (e.g. particle-to-grid density) must be
     non-negative everywhere.

6. **Interface thickness / sharpness (future work)**
   - The transition between phases should be no more than approximately one
     atom diameter in thickness.
   - In future, this may be quantified by:
     - Counting cross-phase neighbours per atom, or
     - Rasterising to a grid and measuring the width of mixed-phase bands.

## Planned Physics Checks

The following checks are planned as reusable functions that operate on raw
frames (or sequences of frames) and can be used for:

- **Data sanity checks** on the ground-truth LAMMPS dumps.
- **Evaluation metrics** for model predictions.
- **Physics-informed loss terms** during training.

1. **Mass-conservation check (first implementation)**
   - For a sequence of frames, compute:
     - `N_total(t)` for each frame `t`.
     - `ΔN_total(t) = N_total(t) - N_total(t_ref)` relative to a reference
       frame (typically the first training frame).
   - Report per-frame diagnostics such as:
     - `Frame dump.XXXX.LAMMPS (t = T): 0 atom(s) lost.`
     - `Frame dump.YYYY.LAMMPS (t = T'): 2 atom(s) gained.`
   - Provide aggregated statistics, e.g. maximum absolute deviation as a
     fraction of `N_total(t_ref)`.

2. **Neighbour-based phase-mixing check (future)**
   - For each atom, count:
     - Total neighbours within a cutoff radius.
     - Number of neighbours of a **different type**.
   - Define a per-frame phase-mixing score based on atoms that have
     `>= 8` neighbours of a different type.

3. **Rigid-wall velocity check (future)**
   - For atoms with `type = 3`, measure deviations of `(vx, vy)` from zero.
   - Report per-frame maximum and typical wall-speed magnitudes.

4. **Interface sharpness check (future)**
   - Use neighbour statistics or a low-resolution grid to approximate
     interface thickness between phases 1 and 2.
   - Penalise overly thick interfaces in model predictions.

These notes will guide the design of `physics_checks` utilities and the
physics-informed loss terms in later phases of the project.
