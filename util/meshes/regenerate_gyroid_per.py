"""Regenerate util/meshes/gyroid_per.vtk from microgen.

Periodic gyroid sheet (density 0.3, unit cell, 1×1×1) meshed with gmsh
via microgen's ``mesh_periodic`` pipeline. The resulting mesh has nodes
exactly aligned across opposite faces (gmsh ``setPeriodic`` constraint)
and ``mesh.is_periodic()`` passes at any tolerance.

Run this script to refresh ``util/meshes/gyroid_per.vtk`` whenever the
upstream microgen API changes or the desired mesh density changes.

Requires the ``[cad]`` extra of microgen (cadquery-ocp-novtk).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from microgen import Phase, Rve, Tpms, mesh_periodic
from microgen.shape.surface_functions import gyroid

OUT = Path(__file__).resolve().parent / "gyroid_per.vtk"


def main():
    tpms = Tpms(
        surface_function=gyroid,
        density=0.3,
        cell_size=1.0,
        repeat_cell=(1, 1, 1),
        resolution=40,
    )
    cad = tpms.generate(type_part="sheet", smoothing=0)

    with tempfile.TemporaryDirectory() as work:
        step_file = Path(work) / "gyroid.step"
        # microgen's CAD wrapper API has shifted across versions: try the
        # current method name first, then the legacy one.
        if hasattr(cad, "save_step"):
            cad.save_step(str(step_file))
        else:
            cad.exportStep(str(step_file))

        phase = Phase(shape=cad.shape if hasattr(cad, "shape") else cad)
        rve = Rve(dim=1.0, center=(0.0, 0.0, 0.0))
        mesh_periodic(
            mesh_file=str(step_file),
            rve=rve,
            list_phases=[phase],
            size=0.06,
            order=1,
            output_file=str(OUT),
        )

    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
