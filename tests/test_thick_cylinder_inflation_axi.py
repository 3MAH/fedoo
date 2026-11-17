"""Thick cylinder inflation: F_theta-theta verification (2Daxi, updated Lagrangian).

Regression test for the hoop component of the deformation gradient in a
finite-strain axisymmetric problem, :math:`F_{\\theta\\theta} = r / R`
(Bonet & Wood, Box 8.3; Holzapfel Sec. 2.5).

A thick-walled annulus :math:`r \\in [A, B]` is expanded by prescribing a
radial displacement ``delta`` on the outer boundary (inner boundary clamped
radially, top and bottom clamped axially). The relation is a pure kinematic
identity, so linear elasticity is sufficient.

Formerly ``examples/models/axisymmetric/thick_cylinder_inflation_axi.py``.
"""

import numpy as np
import pytest

import fedoo as fd


@pytest.fixture(autouse=True)
def _clear_assembly_cache():
    fd.Assembly.delete_memory()
    yield
    fd.Assembly.delete_memory()


def test_thick_cylinder_inflation_axi_hoop_stretch():
    # r in [A, B], z in [0, H]: in 2Daxi, r is the X column and z the Y column.
    A, B = 1.0, 2.0
    H = 0.5
    delta = 0.6  # 30 % radial expansion at r = B

    fd.ModelingSpace("2Daxi")
    mesh = fd.mesh.rectangle_mesh(nx=21, ny=11, x_min=A, x_max=B, y_min=0.0, y_max=H)

    material = fd.constitutivelaw.ElasticIsotrop(1.0e3, 0.3)
    wf = fd.weakform.StressEquilibrium(material)
    assembly = fd.Assembly.create(wf, mesh)

    pb = fd.problem.NonLinear(assembly, nlgeom="UL")
    pb.set_nr_criterion("Displacement", tol=1e-6, max_subiter=15)

    pb.bc.add("Dirichlet", mesh.node_sets["left"], "DispX", 0.0)
    pb.bc.add("Dirichlet", mesh.node_sets["bottom"], "DispY", 0.0)
    pb.bc.add("Dirichlet", mesh.node_sets["top"], "DispY", 0.0)
    pb.bc.add("Dirichlet", mesh.node_sets["right"], "DispX", delta)

    pb.nlsolve(dt=0.1, tmax=1.0, update_dt=True, print_info=0)

    # F is stored as (3, 3, n_gauss_points); component [2, 2] is the hoop one.
    F_tt = assembly.sv["F"][2, 2, :]
    # Reference radius at the Gauss points, captured at problem initialize.
    R0 = assembly.sv["_R0_gausspoints"]
    # Current radius at the Gauss points, interpolated on the deformed mesh.
    r_current = assembly.current.mesh.convert_data(
        assembly.current.mesh.nodes[:, 0],
        "Node",
        "GaussPoint",
        n_elm_gp=assembly.n_elm_gp,
    )

    # F_theta-theta is computed exactly from u_r and R0 by _comp_grad_disp:
    # any discrepancy beyond round-off is a regression.
    np.testing.assert_allclose(F_tt, r_current / R0, rtol=0, atol=1e-10)

    # The final boundary motion is reached: right nodes at r = B + delta.
    np.testing.assert_allclose(
        assembly.current.mesh.nodes[mesh.node_sets["right"], 0], B + delta, rtol=1e-12
    )
    np.testing.assert_allclose(
        assembly.current.mesh.nodes[mesh.node_sets["left"], 0], A, rtol=1e-12
    )

    # Reference radii lie inside the annulus and the hoop stretch is bounded by
    # its values at the inner (1) and outer ((B + delta) / B = 1.3) radii.
    assert A < R0.min() and R0.max() < B
    assert 1.0 < F_tt.min() < 1.05
    assert 1.25 < F_tt.max() < 1.3
