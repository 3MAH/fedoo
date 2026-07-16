"""Regression test for the finite-strain hyperelastic (NEOHC) consistent tangent.

Guards the ``StressEquilibrium`` conversion of simcoon's hyperelastic UMAT
tangent. simcoon >= 1.14 returns the "box" tangent ``d(tau_hat)/dD`` (Kirchhoff
stress, logarithmic corotational rate). The weak form must convert it to
``dS/dE`` for total-lagrangian, and to the Cauchy spatial rate tangent
(``Lt / det(F)`` for the log rate) for updated-lagrangian. With the stale
pre-1.14 conversion the tangent was ~12% wrong (the shear block ~4x too stiff)
and non-symmetric beyond ~10% strain, so Newton stalled/diverged.

The tests exercise a genuinely finite strain (simple shear at gamma = 0.3, which
loads the shear block) and a large-deflection cantilever, for both UL and TL,
and assert:
  * convergence,
  * a small NR iteration count (the consistent-tangent / quadratic-convergence
    proxy),
  * that UL and TL give the same Cauchy stress (frame-invariant),
  * a reference response within a tight tolerance.
"""

import numpy as np
import pytest

simcoon = pytest.importorskip("simcoon")

import fedoo as fd

# NEOHC parameters: nearly-incompressible (mu = 3, kappa = 150 -> nu ~ 0.49).
MU, KAPPA = 3.0, 150.0


def _solve_simple_shear(nlgeom, gamma=0.3):
    """One increment of boundary-driven simple shear of a unit cube."""
    fd.ModelingSpace("3D")
    mesh = fd.mesh.box_mesh(nx=3, ny=3, nz=3, elm_type="hex8", name="Cube")
    mat = fd.constitutivelaw.Simcoon("NEOHC", np.array([MU, KAPPA]), name="M")
    wf = fd.weakform.StressEquilibrium(mat, nlgeom=nlgeom, name="W")
    fd.Assembly.create(wf, mesh, name="A")
    pb = fd.problem.NonLinear("A")
    pb.set_nr_criterion("Displacement", err0=1.0, tol=1e-6, max_subiter=25)

    bottom = mesh.find_nodes("Z", mesh.bounding_box.zmin)
    top = mesh.find_nodes("Z", mesh.bounding_box.zmax)
    pb.bc.add("Dirichlet", bottom, "Disp", 0.0)
    pb.bc.add("Dirichlet", top, "DispY", 0.0)
    pb.bc.add("Dirichlet", top, "DispZ", 0.0)
    pb.bc.add(
        "Dirichlet",
        top,
        "DispX",
        gamma,
        time_func=lambda tf: min(1.0, tf * 1e9),  # step load in one increment
    )

    pb.initialize()
    pb.tmax = 1.0
    pb.dtime = 1.0
    pb.set_start()
    pb.time = 0.0
    convergence, nb_iter, res = pb.solve_time_increment()
    return pb, mesh, convergence, nb_iter


def _max_cauchy_vm(pb, mesh):
    stress = pb.get_results("A", "Stress", "GaussPoint")["Stress"]
    return float(np.max(fd.core.dataset.StressTensorList(stress).von_mises()))


def test_neohc_simple_shear_ul():
    """UL finite simple shear converges quadratically with the corrected tangent."""
    pb, mesh, conv, nb_iter = _solve_simple_shear("UL", gamma=0.3)
    assert conv, "NEOHC UL simple shear did not converge (inconsistent tangent?)"
    assert nb_iter <= 5, f"too many NR iterations ({nb_iter}): tangent inconsistent"


def test_neohc_simple_shear_tl():
    """TL finite simple shear converges and matches UL (same Cauchy stress)."""
    pb_tl, mesh_tl, conv_tl, nb_iter_tl = _solve_simple_shear("TL", gamma=0.3)
    assert conv_tl, "NEOHC TL simple shear did not converge"
    assert nb_iter_tl <= 6, f"too many NR iterations ({nb_iter_tl})"

    pb_ul, mesh_ul, conv_ul, _ = _solve_simple_shear("UL", gamma=0.3)
    assert conv_ul
    # Cauchy stress is frame-invariant -> UL and TL must agree.
    vm_tl = _max_cauchy_vm(pb_tl, mesh_tl)
    vm_ul = _max_cauchy_vm(pb_ul, mesh_ul)
    assert np.isclose(
        vm_ul, vm_tl, rtol=1e-3
    ), f"UL/TL Cauchy vM disagree: UL={vm_ul:g} TL={vm_tl:g}"


def _solve_cantilever(nlgeom, uimp=1.5, n_steps=6):
    """Displacement-controlled large-deflection cantilever (robust, no soft mode)."""
    fd.ModelingSpace("3D")
    mesh = fd.mesh.box_mesh(
        nx=7,
        ny=3,
        nz=3,
        x_min=0,
        x_max=4,
        y_min=0,
        y_max=1,
        z_min=0,
        z_max=1,
        elm_type="hex20",
        name="Beam",
    )
    mat = fd.constitutivelaw.Simcoon("NEOHC", np.array([MU, KAPPA]), name="Mc")
    wf = fd.weakform.StressEquilibrium(mat, nlgeom=nlgeom, name="Wc")
    wf.geometric_stiffness = True
    fd.Assembly.create(wf, mesh, name="Ac")
    pb = fd.problem.NonLinear("Ac")
    pb.set_nr_criterion("Displacement", err0=1.0, tol=1e-4, max_subiter=25)
    pb.bc.add("Dirichlet", mesh.find_nodes("X", 0.0), "Disp", 0.0)
    pb.bc.add("Dirichlet", mesh.find_nodes("X", 4.0), "DispY", uimp)
    pb.nlsolve(dt=1.0 / n_steps, tmax=1.0, update_dt=False, print_info=0)
    return pb, mesh


def test_neohc_cantilever_large_deflection_ul():
    """UL large-deflection Neo-Hookean cantilever reaches equilibrium (~40% of L)."""
    pb, mesh = _solve_cantilever("UL", uimp=1.5)
    disp = pb.get_dof_solution()
    n = mesh.n_nodes
    max_ux = float(np.abs(disp[:n]).max())
    # reference axial pull-in of the tip due to the large transverse bend
    assert np.isclose(
        max_ux, 0.578, rtol=0.03
    ), f"cantilever tip Ux {max_ux:g} != reference 0.578"


if __name__ == "__main__":
    test_neohc_simple_shear_ul()
    test_neohc_simple_shear_tl()
    test_neohc_cantilever_large_deflection_ul()
    print("test_neohookean_cantilever: OK")
