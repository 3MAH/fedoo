"""Quantitative Terzaghi 1D consolidation benchmark against the analytical series.

A saturated linear-elastic column is loaded by a constant compressive surface
pressure (step load) on its drained top face; the bottom is sealed and the
lateral displacement is confined (1D / oedometric consolidation).

The numerical pore-pressure dissipation at the sealed bottom is compared to the
classical Terzaghi series solution, and the initial undrained pressure and the
final drained settlement are compared to their closed-form values. This pins the
Biot coupling, the storage term, the Darcy diffusion and the consolidation
coefficient ``c_v`` together (the companion ``test_poromech_terzaghi`` only
checks qualitative signs/decay).

Closed form (single drainage, drainage path H = L):
    M_oed = K + 4G/3
    c_v   = (k / mu_f) / (alpha**2 / M_oed + 1 / M)
    p0    = alpha * M * sigma0 / (M_oed + alpha**2 * M)          (undrained)
    p(z=0, t) / p0 = sum_m (2 / Mm) (-1)**m exp(-Mm**2 T),  Mm = (pi/2)(2m+1)
    settlement(t -> inf) = sigma0 * L / M_oed                    (drained)
"""

import numpy as np

import fedoo as fd


def _terzaghi_series(T, nterms=300):
    """Normalized excess pore pressure at the sealed face, p(0, t) / p0."""
    m = np.arange(nterms)
    Mm = (np.pi / 2.0) * (2 * m + 1)
    return np.sum((2.0 / Mm) * ((-1.0) ** m) * np.exp(-(Mm**2) * T))


def test_terzaghi_consolidation_vs_analytical():
    """Stress-controlled 1D consolidation must match the Terzaghi series."""
    L = 1.0
    E, nu = 1.0e6, 0.3
    M = 1.0e8  # Biot modulus
    alpha = 1.0
    k = 1.0e-7  # permeability
    mu_f = 1.0
    sigma0 = 1.0e3  # applied surface load [Pa]

    G = E / (2.0 * (1.0 + nu))
    K = E / (3.0 * (1.0 - 2.0 * nu))
    M_oed = K + 4.0 * G / 3.0
    c_v = (k / mu_f) / (alpha**2 / M_oed + 1.0 / M)
    p0_analytic = alpha * M * sigma0 / (M_oed + alpha**2 * M)
    settlement_drained = -sigma0 * L / M_oed

    fd.ModelingSpace("3D")
    nz = 20
    mesh = fd.mesh.box_mesh(
        nx=2,
        ny=2,
        nz=nz + 1,
        x_min=0,
        x_max=0.1,
        y_min=0,
        y_max=0.1,
        z_min=0,
        z_max=L,
        elm_type="hex8",
        name="Col",
    )
    skel = fd.constitutivelaw.ElasticIsotrop(E, nu, name="Sk")
    fluid = fd.constitutivelaw.PoroFluidProperties(
        permeability=k,
        fluid_viscosity=mu_f,
        biot_coefficient=alpha,
        biot_modulus=M,
        initial_porosity=0.5,
        name="Fl",
    )
    wf = fd.weakform.PoroMechanics(skel, fluid, bulk_modulus=K, nlgeom=False, name="P")
    poro_asm = fd.Assembly.create(wf, mesh, name="A")

    top = mesh.find_nodes("Z", L)
    bottom = mesh.find_nodes("Z", 0.0)
    side_x = np.unique(
        np.concatenate([mesh.find_nodes("X", 0.0), mesh.find_nodes("X", 0.1)])
    )
    side_y = np.unique(
        np.concatenate([mesh.find_nodes("Y", 0.0), mesh.find_nodes("Y", 0.1)])
    )

    # Constant surface load applied as a STEP (initial_pressure == pressure ->
    # no incremental ramp), so the classical Terzaghi step response is exercised.
    press_asm = fd.constraint.Pressure.from_nodes(
        mesh, top, sigma0, initial_pressure=sigma0
    )
    asm = poro_asm + press_asm

    pb = fd.problem.NonLinear(asm)
    pb.set_nr_criterion("Displacement", tol=1e-4, max_subiter=30)
    pb.bc.add("Dirichlet", side_x, "DispX", 0.0)
    pb.bc.add("Dirichlet", side_y, "DispY", 0.0)
    pb.bc.add("Dirichlet", bottom, "DispZ", 0.0)
    pb.bc.add("Dirichlet", top, "PorePressure", 0.0)  # drained top

    dt = 0.2
    nb_steps = 60
    pb.initialize()
    pb.tmax = dt * nb_steps
    pb.dtime = dt
    pb.set_start()

    p_bottom = np.zeros(nb_steps)
    uz_top = np.zeros(nb_steps)
    for s in range(nb_steps):
        pb.time = s * dt
        convergence, nb_nr, res = pb.solve_time_increment()
        assert convergence, f"Step {s}: consolidation Newton failed (res={res:g})"
        pb.set_start()
        p_bottom[s] = float(np.mean(pb.get_dof_solution("PorePressure")[bottom]))
        uz_top[s] = float(np.mean(pb.get_dof_solution("DispZ")[top]))

    # 1. Undrained initial bottom pressure matches the closed form (< 3 %).
    assert (
        abs(p_bottom[0] - p0_analytic) < 0.03 * p0_analytic
    ), f"Undrained p0: numerical {p_bottom[0]:g} vs analytical {p0_analytic:g}"

    # 2. The whole dissipation curve matches the Terzaghi series (< 5 % abs on
    #    the normalized pressure, accounting for FE discretization).
    t = (np.arange(nb_steps) + 1) * dt
    T = c_v * t / L**2
    p_norm = p_bottom / p0_analytic
    series = np.array([_terzaghi_series(Ti) for Ti in T])
    max_err = np.max(np.abs(p_norm - series))
    assert max_err < 0.05, (
        f"Dissipation curve deviates from Terzaghi series by {max_err:.3f} " f"(> 0.05)"
    )

    # 3. Final settlement approaches the drained elastic value (< 5 %).
    assert abs(uz_top[-1] - settlement_drained) < 0.05 * abs(settlement_drained), (
        f"Drained settlement: numerical {uz_top[-1]:g} vs analytical "
        f"{settlement_drained:g}"
    )

    # 4. Pore pressure has largely dissipated by the end of the run.
    assert abs(p_bottom[-1]) < 0.05 * p0_analytic


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
