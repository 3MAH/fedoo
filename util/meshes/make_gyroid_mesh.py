"""Generate a clean periodic gyroid sheet mesh (tet4) for the IPC examples.

Pipeline (CAD -> periodic gmsh mesh -> mmg remesh), with microgen::

    microgen Tpms (structured-grid clip of the level set)
      -> [snap]  move grid values away from the iso-levels (periodic-safe)
      -> CAD shell (OCC, planar faces on the cell sides) -> STEP
      -> microgen.mesh_periodic: gmsh volume mesh with ``setPeriodic`` on
         opposite cell faces
      -> microgen.remesh.remesh_keeping_boundaries_for_fem: mmg3d remesh with
         the cell-face triangles *required* (kept untouched, so the periodic
         node pairing survives)

Why the ``snap`` step: microgen builds the sheet by clipping a structured
grid with the two iso-levels ``f = +-t/2``. When a grid value is very close to
an iso-level, the cut point falls almost on the grid node and the clip creates
triangles with edges down to ~1e-6 (the former ``gyroid_per.vtk``). Every one
of these triangles becomes a CAD face that gmsh must honour, so gmsh cannot
remove them. Snapping ``|f -+ t/2| >= alpha * h * |grad f|`` at every node
bounds the distance between a cut point and a grid node by ``~alpha * h``
(``h`` = grid spacing), so no edge shorter than ``~alpha * h`` is created. The
grid values are made exactly periodic before snapping, so the cell faces stay
periodic.

The remesh step then coarsens / regularizes the gyroid surface and the volume
(``hmin``, ``hmax``, ``hausd``, ``hgrad``) while keeping the cell-face
triangles required, which preserves the node pairing needed by
``fd.constraint.PeriodicBC``. The remesh alone (without ``snap``) is not
enough: the short edges that lie *on* the cell faces are required, hence kept.
mmg's level-set adaptation is not deterministic: two runs give slightly
different meshes.

What is left: with ``dhat = 5e-3`` x bbox diagonal, the IPC pairs found at
rest with ``excluded_rings=1`` or ``2`` all touch a cell-face vertex. They sit
in the sharp wedge where the gyroid sheet meets a cell face at a shallow
angle (geometry of the cut cell, not a mesh defect). ``excluded_rings=3``
removes them all.

Tested with microgen 2.0.0b1, gmsh 4.15.2, mmg 5.8.0: about 1 min.

Usage (from this folder)::

    python make_gyroid_mesh.py                     # gyroid_per_clean.vtk
    python make_gyroid_mesh.py --no-remesh         # snap + gmsh only
    python make_gyroid_mesh.py --alpha 0 --no-remesh --out raw.vtk  # no snap

Requirements: ``microgen`` with its ``[cad]`` extra (``cadquery-ocp-novtk``),
``gmsh``, ``pyvista``, ``scipy`` and, unless ``--no-remesh``, the ``mmg3d_O3``
executable on the ``PATH`` (``conda install -c conda-forge mmg``).
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import numpy as np
import pyvista as pv
from microgen import Phase, Rve, Tpms, mesh_periodic
from microgen.cad import import_step
from microgen.remesh import remesh_keeping_boundaries_for_fem
from microgen.shape.surface_functions import gyroid

HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# 1. Level set on the structured grid, with periodic-safe snapping
# ---------------------------------------------------------------------------
def _grid_indices(tpms: Tpms) -> tuple[np.ndarray, tuple[int, int, int], float]:
    """Return (i, j, k) integer index of every grid point, grid shape, spacing."""
    pts = np.asarray(tpms.grid.points)
    lo = pts.min(axis=0)
    n = np.asarray(tpms.resolution * tpms.repeat_cell, dtype=int)
    h = (pts.max(axis=0) - lo) / (n - 1)
    ijk = np.rint((pts - lo) / h).astype(int)
    return ijk, tuple(n), float(h.min())


def _snap(values: np.ndarray, ijk: np.ndarray, shape, h: float, alpha: float):
    """Make ``values`` exactly periodic, then push them away from 0."""
    v = np.empty(shape)
    v[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = values
    # exact periodicity: the max face copies the min face (they only differ
    # by round-off, sin(pi) != sin(-pi))
    v[-1, :, :] = v[0, :, :]
    v[:, -1, :] = v[:, 0, :]
    v[:, :, -1] = v[:, :, 0]
    if alpha > 0:
        grad = np.linalg.norm(np.stack(np.gradient(v, h)), axis=0)
        eps = alpha * h * grad
        close = np.abs(v) < eps
        sign = np.where(v >= 0.0, 1.0, -1.0)
        v[close] = (sign * eps)[close]
        # gradient is periodic up to round-off too: re-impose the copy
        v[-1, :, :] = v[0, :, :]
        v[:, -1, :] = v[:, 0, :]
        v[:, :, -1] = v[:, :, 0]
        n_snapped = int(close.sum())
    else:
        n_snapped = 0
    return v[ijk[:, 0], ijk[:, 1], ijk[:, 2]], n_snapped


def build_tpms(density: float, resolution: int, alpha: float) -> Tpms:
    tpms = Tpms(
        surface_function=gyroid,
        density=density,
        cell_size=1.0,
        repeat_cell=(1, 1, 1),
        resolution=resolution,
    )
    offset = tpms._compute_offset_to_fit_density(part_type="sheet")
    # freeze the offset so that generate_cad() does not recompute it (and
    # does not overwrite the snapped grid arrays)
    tpms.density = None
    tpms.offset = offset
    ijk, shape, h = _grid_indices(tpms)
    n_tot = 0
    for name in ("lower_surface", "upper_surface"):
        tpms.grid[name], n = _snap(np.asarray(tpms.grid[name]), ijk, shape, h, alpha)
        n_tot += n
    tpms._grid_sheet = None  # force the clip to be recomputed
    print(f"[tpms] offset={offset:.5f}  h={h:.4f}  snapped nodes={n_tot}")
    return tpms


# ---------------------------------------------------------------------------
# 2. CAD + periodic gmsh mesh (microgen.mesh_periodic)
# ---------------------------------------------------------------------------
def periodic_mesh(tpms: Tpms, size: float, work: Path) -> pv.UnstructuredGrid:
    """CAD -> STEP -> periodic gmsh tet4 mesh."""
    step = work / "gyroid.step"
    tpms.generate_cad(type_part="sheet", smoothing=0).export_step(step)
    msh = work / "gyroid_periodic.msh"
    mesh_periodic(
        str(step),
        Rve(dim=1.0, center=(0.0, 0.0, 0.0)),
        [Phase(shape=import_step(step))],
        size=size,
        order=1,
        output_file=str(msh),
        msh_file_version=2,  # meshio cannot read the gmsh 4.1 entities
    )
    return _tet4(pv.read(msh).extract_cells_by_type(pv.CellType.TETRA).clean())


def _tet4(grid) -> pv.UnstructuredGrid:
    """Plain tet4 grid without cell or point data."""
    tets = grid.cells_dict[pv.CellType.TETRA]
    return pv.UnstructuredGrid({pv.CellType.TETRA: tets}, np.asarray(grid.points))


# ---------------------------------------------------------------------------
# 3. Periodicity check (node pairing on opposite cell faces)
# ---------------------------------------------------------------------------
def periodic_pairing_error(points: np.ndarray, tol: float = 1e-8) -> float:
    """Max distance between a face node and its translated twin (inf if unpaired)."""
    from scipy.spatial import cKDTree

    lo, hi = points.min(axis=0), points.max(axis=0)
    worst = 0.0
    for a in range(3):
        m = np.flatnonzero(np.abs(points[:, a] - lo[a]) < tol)
        p = np.flatnonzero(np.abs(points[:, a] - hi[a]) < tol)
        if len(m) != len(p):
            return np.inf
        shifted = points[m].copy()
        shifted[:, a] += hi[a] - lo[a]
        d, _ = cKDTree(points[p]).query(shifted)
        worst = max(worst, float(d.max()))
    return worst


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--density", type=float, default=0.3)
    ap.add_argument("--resolution", type=int, default=40)
    ap.add_argument("--size", type=float, default=0.06, help="gmsh mesh size")
    ap.add_argument(
        "--alpha",
        type=float,
        default=0.25,
        help="snap distance in grid spacings (0 = no snapping, old mesh)",
    )
    ap.add_argument(
        "--no-remesh",
        dest="remesh",
        action="store_false",
        help="skip the mmg remesh",
    )
    ap.add_argument("--hmin", type=float, default=0.012)
    ap.add_argument("--hmax", type=float, default=0.06)
    ap.add_argument("--hausd", type=float, default=0.003)
    ap.add_argument("--hgrad", type=float, default=1.3)
    ap.add_argument("--out", type=Path, default=HERE / "gyroid_per_clean.vtk")
    ap.add_argument("--keep", type=Path, default=None, help="keep work files here")
    args = ap.parse_args()

    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        work = args.keep or Path(tmp)
        work.mkdir(parents=True, exist_ok=True)
        tpms = build_tpms(args.density, args.resolution, args.alpha)
        grid = periodic_mesh(tpms, args.size, work)
        print(
            f"[gmsh] {grid.n_points} nodes, {grid.n_cells} tets "
            f"({time.perf_counter() - t0:.0f} s)"
        )
    if args.remesh:
        grid = _tet4(
            remesh_keeping_boundaries_for_fem(
                grid,
                periodic=True,
                hmin=args.hmin,
                hmax=args.hmax,
                hausd=args.hausd,
                hgrad=args.hgrad,
            )
        )
        print(f"[mmg] {grid.n_points} nodes, {grid.n_cells} tets")

    err = periodic_pairing_error(np.asarray(grid.points))
    print(f"periodic pairing error: {err:.2e}   volume fraction: {grid.volume:.4f}")
    grid.save(args.out, binary=True)
    print(f"wrote {args.out}  ({time.perf_counter() - t0:.0f} s)")


if __name__ == "__main__":
    main()
