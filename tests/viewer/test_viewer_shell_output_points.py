"""Shell thickness extraction, saved interpolation and mixed-element recovery."""

import numpy as np
import pytest
import fedoo as fd
from fedoo.util.voigt_tensors import StressTensorList
from fedoo.weakform.plate import _ShellComponentList


def shell_problem(section=None, nx=2):
    fd.ModelingSpace("3D")
    if section is None:
        section = fd.constitutivelaw.ShellHomogeneous(
            fd.constitutivelaw.ElasticIsotrop(1000, 0.25), 2.0, k=5 / 6
        )
    mesh = fd.mesh.rectangle_mesh(nx=nx, ny=2, elm_type="quad4", ndim=3)
    assembly = fd.Assembly.create(fd.weakform.PlateEquilibrium(section), mesh)
    problem = fd.problem.Linear(assembly)
    n = assembly.n_gauss_points
    assembly.sv["ShellStrain"] = _ShellComponentList(
        [
            np.full(n, v)
            for v in (0.01, -0.003, 0.004, 0.02, 0.005, -0.002, 0.008, -0.006)
        ]
    )
    if section._recovery_model.endswith("nonlinear"):
        section.initialize(assembly, problem)
    section.update(assembly, problem)
    return problem, assembly, section


def test_viewer_lists_shell_components_with_thickness_reduction():
    from types import SimpleNamespace

    viewer = pytest.importorskip("fedoo.util.viewer")
    pb, assembly, _ = shell_problem()
    result = pb.get_results(assembly, ["Stress", "Strain"])
    result.shell_options.update(reduction="abs_max")
    dock = SimpleNamespace(data=result)
    # Component discovery must not request a full tensor under a scalar
    # thickness reduction. Call the method without creating a Qt window.
    for field in ("Stress", "Stress_local", "Stress_global", "Strain"):
        components = viewer.PlotDock.get_components(dock, field)
        assert "XX" in components
        assert "XY" in components
        if field.startswith("Stress"):
            assert "vm" in components
    assert result.shell_options["reduction"] == "abs_max"
