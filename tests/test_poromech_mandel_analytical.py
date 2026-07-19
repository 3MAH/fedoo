"""Quantitative Mandel benchmark against the Abousleiman/Cheng-Detournay series.

Force-controlled rigid, frictionless, impermeable plate (Mandel's setup): a
constant (step) total load is applied on the top while the plate is kept flat,
the lateral face drains and is traction-free. This is the 2D benchmark that
exhibits the Mandel-Cryer effect (a non-monotonic pore-pressure overshoot at the
centre) which no 1D consolidation can reproduce.

The centre pore pressure history is compared to the analytical series

    p(x, t) = 2 p0 * sum_n [sin(a_n) / (a_n - sin a_n cos a_n)]
                         * [cos(a_n x/a) - cos a_n] * exp(-a_n^2 c t / a^2)

with a_n the roots of  tan(a_n) = (1 - nu) / (nu_u - nu) * a_n,
p0 = B (1 + nu_u) F / (3 a),  and the consolidation coefficient

    c = (k / mu) * M * M_oed / (alpha^2 M + M_oed),   M_oed = K + 4 G / 3.

The rigid plate is realised with a constant surface pressure applied as a STEP
(``initial_pressure == pressure`` to avoid the t_fact ramp) plus an MPC tying all
top vertical displacements together (kept flat).
"""

import numpy as np
from scipy.optimize import brentq

import fedoo as fd


def _mandel_constants(E, nu, M, alpha, k, mu):
    G = E / (2 * (1 + nu))
    K = E / (3 * (1 - 2 * nu))
    M_oed = K + 4 * G / 3
    K_u = K + alpha**2 * M
    nu_u = (3 * K_u - 2 * G) / (2 * (3 * K_u + G))
    B = alpha * M / K_u
    c = (k / mu) * M * M_oed / (alpha**2 * M + M_oed)
    return G, K, M_oed, nu_u, B, c


def _mandel_roots(nu, nu_u, n_roots=60):
    ratio = (1 - nu) / (nu_u - nu)
    roots = []
    for n in range(1, n_roots + 1):
        lo, hi = (n - 1) * np.pi + 1e-9, (n - 0.5) * np.pi - 1e-9
        roots.append(brentq(lambda x: np.tan(x) - ratio * x, lo, hi))
    return np.array(roots)


def test_mandel_vs_abousleiman():
    """Force-controlled Mandel must match the analytical series and overshoot."""
    E, nu, M, alpha, k, mu = 1.0e6, 0.3, 1.0e8, 1.0, 1.0e-7, 1.0
    a = b = 1.0
    thin_y = 0.1
    sigma0 = 1.0e3

    G, K, M_oed, nu_u, B, c = _mandel_constants(E, nu, M, alpha, k, mu)
    roots = _mandel_roots(nu, nu_u)
    # F (force per unit thickness over the half-width) = sigma0 * a
    p0 = B * (1 + nu_u) * (sigma0 * a) / (3 * a)

    def p_centre(t):
        s = sum(
            np.sin(an)
            / (an - np.sin(an) * np.cos(an))
            * (1 - np.cos(an))
            * np.exp(-(an**2) * c * t / a**2)
            for an in roots
        )
        return 2 * p0 * s

    fd.ModelingSpace("3D")
    mesh = fd.mesh.box_mesh(
        nx=21,
        ny=2,
        nz=6,
        x_min=0,
        x_max=a,
        y_min=0,
        y_max=thin_y,
        z_min=0,
        z_max=b,
        elm_type="hex8",
        name="MQ",
    )
    skel = fd.constitutivelaw.ElasticIsotrop(E, nu, name="Sk")
    fluid = fd.constitutivelaw.PoroFluidProperties(
        permeability=k,
        fluid_viscosity=mu,
        biot_coefficient=alpha,
        biot_modulus=M,
        initial_porosity=0.5,
        name="Fl",
    )
    wf = fd.weakform.poro_mechanics_simple(skel, fluid, nlgeom=False, name="P")
    poro = fd.Assembly.create(wf, mesh, name="A")

    top = mesh.find_nodes("Z", b)
    # Constant (step) load -> initial_pressure == pressure (no t_fact ramp).
    press = fd.constraint.Pressure.from_nodes(
        mesh, top, sigma0, initial_pressure=sigma0
    )
    asm = poro + press

    pb = fd.problem.NonLinear(asm)
    pb.set_nr_criterion("Displacement", tol=1e-4, max_subiter=40)

    sym_x = mesh.find_nodes("X", 0.0)
    pb.bc.add("Dirichlet", sym_x, "DispX", 0.0)
    pb.bc.add("Dirichlet", mesh.find_nodes("Z", 0.0), "DispZ", 0.0)
    pb.bc.add("Dirichlet", mesh.find_nodes("Y", 0.0), "DispY", 0.0)
    pb.bc.add("Dirichlet", mesh.find_nodes("Y", thin_y), "DispY", 0.0)
    pb.bc.add("Dirichlet", mesh.find_nodes("X", a), "PorePressure", 0.0)

    # Rigid (flat) plate: tie all top DispZ to the first top node.
    # NOTE: the multi-MPC path needs array factors (length N) and constant=None
    # (constant=0.0 hits a missing-`start_value` bug in this fedoo version).
    master = top[0]
    slaves = top[top != master]
    nsl = len(slaves)
    pb.bc.mpc(
        [slaves, np.full(nsl, master)],
        ["DispZ", "DispZ"],
        [np.ones(nsl), -np.ones(nsl)],
        None,
    )

    dt = 0.05
    nb_steps = 80
    pb.initialize()
    pb.tmax = dt * nb_steps
    pb.dtime = dt
    pb.set_start()

    hist = np.zeros(nb_steps)
    for s in range(nb_steps):
        pb.time = s * dt
        convergence, nb_nr, res = pb.solve_time_increment()
        assert convergence, f"Step {s}: Mandel Newton failed (res={res:g})"
        pb.set_start()
        hist[s] = float(np.mean(pb.get_dof_solution("PorePressure")[sym_x]))

    t = (np.arange(nb_steps) + 1) * dt
    analytic = np.array([p_centre(ti) for ti in t])

    # 1. Positive centre pressure under compression.
    assert hist[0] > 0.0

    # 2. Mandel-Cryer overshoot: the maximum occurs AFTER the first step.
    idx_max = int(np.argmax(hist))
    assert idx_max >= 1, f"Mandel-Cryer peak at first step: {hist[:5]}"
    assert hist[idx_max] > hist[0], "no pore-pressure overshoot"

    # 3. Quantitative match to the Abousleiman series (< 5 % relative, max over
    #    the whole run; numerically ~1 % on this mesh).
    rel = np.abs(hist - analytic) / np.maximum(np.abs(analytic), 1.0)
    assert np.max(rel) < 0.05, (
        f"Mandel centre pressure deviates from the Abousleiman series by "
        f"{np.max(rel) * 100:.1f}% (> 5%)"
    )

    # 4. The analytical peak time T ~ 0.06 is reproduced (within a few steps).
    T_peak_num = c * t[idx_max] / a**2
    assert 0.02 < T_peak_num < 0.15, f"peak at T={T_peak_num:.3f} (expected ~0.06)"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
