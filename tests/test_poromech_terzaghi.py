"""1D Terzaghi-style consolidation smoke test for the poromechanics module.

Sets up a thin saturated elastic column under a step compressive displacement
imposed on the drained top surface. Checks that:

  * the coupled (u, p, PorePressure) system assembles and converges;
  * the pore pressure builds up under the load and decays over time
    (consolidation);
  * the displacement at the top evolves monotonically toward the drained
    elastic solution.

This test deliberately stays in small-strain, linear-elastic regime: the
analytical Terzaghi 1D solution is the reference benchmark, but here we use
permissive sanity assertions to keep the test robust against discretization
choices. Quantitative Terzaghi comparison lives in a separate example.
"""

import numpy as np
import pytest

import fedoo as fd


def _build_column_problem(
    L=1.0,
    nx=2,
    ny=2,
    nz=10,
    E=1.0e6,
    nu=0.3,
    biot_M=1.0e8,
    permeability=1.0e-7,
    fluid_viscosity=1.0,
):
    """Build a thin saturated column problem ready to solve.

    Returns
    -------
    pb : fedoo.problem.NonLinear
    mesh : fedoo.Mesh
    top_nodes, bottom_nodes, side_x_nodes, side_y_nodes : ndarray
    """
    fd.ModelingSpace("3D")

    mesh = fd.mesh.box_mesh(
        nx=nx + 1,
        ny=ny + 1,
        nz=nz + 1,
        x_min=0,
        x_max=0.1,
        y_min=0,
        y_max=0.1,
        z_min=0,
        z_max=L,
        elm_type="hex8",
        name="Column",
    )

    skel = fd.constitutivelaw.ElasticIsotrop(E, nu, name="Skeleton")
    fluid = fd.constitutivelaw.PoroFluidProperties(
        permeability=permeability,
        fluid_viscosity=fluid_viscosity,
        biot_coefficient=1.0,
        biot_modulus=biot_M,
        initial_porosity=0.5,
        name="Fluid",
    )

    K_bulk = E / (3.0 * (1.0 - 2.0 * nu))
    wf = fd.weakform.PoroMechanics(
        skel, fluid, bulk_modulus=K_bulk, nlgeom=False, name="Poro"
    )
    fd.Assembly.create(wf, mesh, name="PoroAssembly")
    pb = fd.problem.NonLinear("PoroAssembly")
    pb.set_nr_criterion("Displacement", tol=1e-3, max_subiter=10, err0=1.0)

    bottom = mesh.find_nodes("Z", 0.0)
    top = mesh.find_nodes("Z", L)
    side_x = np.unique(
        np.concatenate([mesh.find_nodes("X", 0.0), mesh.find_nodes("X", 0.1)])
    )
    side_y = np.unique(
        np.concatenate([mesh.find_nodes("Y", 0.0), mesh.find_nodes("Y", 0.1)])
    )
    return pb, mesh, top, bottom, side_x, side_y


def test_terzaghi_consolidation_smoke():
    """Coupled poromechanics: solve, drain, check monotonic consolidation."""
    L = 1.0
    delta = -1.0e-4  # compressive top displacement (1e-4 m)

    pb, mesh, top, bottom, side_x, side_y = _build_column_problem(L=L)

    # Lateral confinement (no x, no y displacement on side faces)
    pb.bc.add("Dirichlet", side_x, "DispX", 0.0)
    pb.bc.add("Dirichlet", side_y, "DispY", 0.0)

    # Bottom fully fixed in z, free in p (no drainage at the bottom)
    pb.bc.add("Dirichlet", bottom, "DispZ", 0.0)

    # Top: imposed compressive displacement, drainage open (p = 0)
    pb.bc.add("Dirichlet", top, "DispZ", delta)
    pb.bc.add("Dirichlet", top, "PorePressure", 0.0)

    dt = 0.5
    nb_steps = 40

    p_bottom_history = []
    uz_top_history = []

    pb.initialize()
    pb.tmax = dt * nb_steps
    pb.dtime = dt
    pb.set_start()

    for step in range(nb_steps):
        pb.time = step * dt
        convergence, nb_nr, res = pb.solve_time_increment()
        assert convergence, f"Step {step}: poromechanics Newton failed (res={res:g})"
        pb.set_start()

        dof = pb.get_dof_solution()
        p_field = pb.get_dof_solution("PorePressure")
        uz_field = pb.get_dof_solution("DispZ")
        p_bottom_history.append(float(np.mean(p_field[bottom])))
        uz_top_history.append(float(np.mean(uz_field[top])))

    p_history = np.asarray(p_bottom_history)
    uz_history = np.asarray(uz_top_history)

    # Sanity 1: pore pressure built up under the load
    assert (
        p_history[0] > 0.0
    ), f"Pore pressure must be positive after the load step, got {p_history[0]}"

    # Sanity 2: pore pressure decays (consolidation) and approaches zero
    assert p_history[-1] < p_history[0], (
        "Pore pressure must decay over time at the closed bottom node, "
        f"got start={p_history[0]:g}, end={p_history[-1]:g}"
    )
    assert abs(p_history[-1]) < 0.05 * p_history[0], (
        "Pore pressure should be largely dissipated at the end of the run, "
        f"got |p_end|={abs(p_history[-1]):g} vs p_start={p_history[0]:g}"
    )

    # Sanity 3: top displacement reached the imposed value (Dirichlet)
    assert np.allclose(uz_history[-1], delta, rtol=1e-6)

    # Sanity 4: monotonic decay of bottom pore pressure (consolidation profile)
    decreasing = np.all(np.diff(p_history[1:]) <= 1e-9)
    assert decreasing, (
        "Bottom pore pressure must decrease monotonically after the first step; "
        f"diffs = {np.diff(p_history[1:])}"
    )


if __name__ == "__main__":
    test_terzaghi_consolidation_smoke()
    print("test_poromech_terzaghi: OK")
