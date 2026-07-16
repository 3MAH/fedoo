"""Regression test for StressEquilibriumMixed in the nearly-incompressible limit.

Guards the bulk-modulus scaling of the pressure-constraint equation. The mixed
(u, Pressure) formulation scales the constraint by 1/K_scale while the momentum
coupling is unscaled, so the assembled tangent is symmetric AND consistent with
the residual (under the inherited assume_sym=True) only when K_scale equals the
true bulk modulus K. When ``bulk_modulus`` is left to its default (None), K_scale
must be read from the tangent as (1/9) sum_{i,j<3} H[i][j] = K (NOT trH/9, which
is off by ~2-3x and made Newton diverge to a wrong solution).

Test: uniaxial-stress compression of a nearly-incompressible block (nu = 0.499),
solved with the DEFAULT bulk_modulus=None. It must converge and reproduce the
incompressible Poisson response (lateral expansion = nu * axial strain), and
match the reference run with bulk_modulus = K.
"""

import numpy as np

import fedoo as fd


def _solve_compression(bulk_modulus, E=1.0e3, nu=0.499, dz=-1.0e-2):
    fd.ModelingSpace("3D")
    mesh = fd.mesh.box_mesh(
        nx=4,
        ny=4,
        nz=4,
        x_min=0,
        x_max=1,
        y_min=0,
        y_max=1,
        z_min=0,
        z_max=1,
        elm_type="hex8",
        name="Cube",
    )
    mat = fd.constitutivelaw.ElasticIsotrop(E, nu, name="M")
    wf = fd.weakform.StressEquilibriumMixed(
        mat, bulk_modulus=bulk_modulus, nlgeom=False, name="W"
    )
    fd.Assembly.create(wf, mesh, name="A")
    pb = fd.problem.NonLinear("A")
    pb.set_nr_criterion("Displacement", tol=1e-6, max_subiter=40)

    # quarter symmetry + imposed top compression, lateral faces free (uniaxial stress)
    pb.bc.add("Dirichlet", mesh.find_nodes("X", 0.0), "DispX", 0.0)
    pb.bc.add("Dirichlet", mesh.find_nodes("Y", 0.0), "DispY", 0.0)
    pb.bc.add("Dirichlet", mesh.find_nodes("Z", 0.0), "DispZ", 0.0)
    pb.bc.add(
        "Dirichlet",
        mesh.find_nodes("Z", 1.0),
        "DispZ",
        dz,
        time_func=lambda tf: min(1.0, tf * 1e9),  # step load
    )
    pb.initialize()
    pb.tmax = 1.0
    pb.dtime = 1.0
    pb.set_start()
    pb.time = 0.0
    convergence, nb_iter, res = pb.solve_time_increment()
    return pb, mesh, convergence, nb_iter, res


def test_mixed_nearly_incompressible_default_bulk():
    """Default bulk_modulus=None must converge and give the incompressible response."""
    E, nu, dz = 1.0e3, 0.499, -1.0e-2

    pb, mesh, conv, nb_iter, res = _solve_compression(None, E=E, nu=nu, dz=dz)

    # 1. Must converge (the trH/9 scaling made this diverge: conv=0).
    assert conv, f"mixed (bulk=None) did not converge: res={res:g}, nb_iter={nb_iter}"
    # linear problem with a consistent tangent -> very few iterations
    assert nb_iter <= 4, f"too many NR iterations ({nb_iter}): tangent inconsistent?"

    # 2. Incompressible Poisson response: lateral expansion = nu * |axial strain|.
    #    Uniaxial stress, axial strain = dz / height = -1e-2 over a unit cube.
    dispx = pb.get_dof_solution("DispX")
    lateral_expected = nu * abs(dz)  # DispX at x=1
    assert np.isclose(np.max(np.abs(dispx)), lateral_expected, rtol=0.03), (
        f"lateral expansion {np.max(np.abs(dispx)):g} != incompressible "
        f"{lateral_expected:g} (would lock or be wrong without the K_scale fix)"
    )


def test_mixed_bulk_none_matches_explicit_bulk():
    """bulk_modulus=None must give the same solution as bulk_modulus=K."""
    E, nu = 1.0e3, 0.499
    K = E / (3 * (1 - 2 * nu))

    pb_none, _, conv_none, _, _ = _solve_compression(None, E=E, nu=nu)
    pb_K, _, conv_K, _, _ = _solve_compression(K, E=E, nu=nu)

    assert conv_none and conv_K
    sol_none = pb_none.get_dof_solution()
    sol_K = pb_K.get_dof_solution()
    diff = np.max(np.abs(sol_none - sol_K))
    scale = np.max(np.abs(sol_K))
    assert (
        diff < 1e-3 * scale
    ), f"bulk=None and bulk=K solutions differ by {diff:g} (scale {scale:g})"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
