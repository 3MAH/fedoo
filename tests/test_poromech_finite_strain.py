"""Finite-strain poromechanics with a simcoon hyperelastic skeleton.

Confined compression (Terzaghi-like) consolidation of a saturated column
with a compressible Neo-Hookean skeleton (simcoon ``NEOHC``) in updated
Lagrangian (log_R corotational) mode, with the Holmes-Mow
deformation-dependent permeability.

Checks that:

  * the coupled finite-strain (u, PorePressure) system assembles and
    converges with ``PoroMechanicsSimple`` + ``nlgeom="UL"``;
  * ``lnJ`` is computed at gauss points (regression: plain
    ``StressEquilibrium`` does not provide it, ``PoroMomentumSimple`` must);
  * the drained final state matches the exact confined kinematics
    ``lnJ = ln(1 + delta/L)`` (F = diag(1, 1, 1 + delta/L));
  * pore pressure builds up under compression (p > 0) then dissipates;
  * the Holmes-Mow permeability actually receives J: k(J_final) < k0.

The test stays at 10% compression: beyond ~12% the UL Newton loop is
unstable with hyperelastic (``_Lt_from_F``) skeletons — a fedoo-core
tangent limitation, not a poro one. See the Limitations section of
docs/poromechanics.rst for the full analysis.
"""

import numpy as np
import pytest

import fedoo as fd

try:
    from tests.test_poromech_terzaghi import _build_column_problem
except ImportError:  # direct run: python tests/test_poromech_finite_strain.py
    from test_poromech_terzaghi import _build_column_problem

try:
    from simcoon import simmit  # noqa: F401

    USE_SIMCOON = True
except ImportError:
    USE_SIMCOON = False


@pytest.mark.skipif(not USE_SIMCOON, reason="simcoon is required for NEOHC")
def test_finite_strain_consolidation_neohc():
    L = 1.0
    stretch = 0.9  # 10% confined compression (see module docstring)
    delta = (stretch - 1.0) * L

    E = 1.0e6
    nu = 0.3
    mu = E / (2.0 * (1.0 + nu))
    kappa = E / (3.0 * (1.0 - 2.0 * nu))

    phi0 = 0.5
    k0 = 1.0e-6  # reference mobility (mu_f = 1)

    skel = fd.constitutivelaw.Simcoon("NEOHC", np.array([mu, kappa]), name="SkelFS")
    perm = fd.constitutivelaw.HolmesMowPermeability(k0=k0, n0=1.0 - phi0)

    pb, mesh, assembly, top, bottom, side_x, side_y = _build_column_problem(
        L=L,
        nx=1,
        ny=1,
        nz=8,
        permeability=perm,
        biot_M=1.0e8,
        skeleton=skel,
        use_simple=True,
        nlgeom="UL",
        initial_porosity=phi0,
        name_suffix="FS",
    )
    pb.set_nr_criterion("Displacement", tol=1e-3, max_subiter=50)

    # Confined compression: no lateral displacement, fixed bottom (undrained),
    # drained top loaded by an imposed displacement ramped over the first
    # steps then held so consolidation can complete.
    pb.bc.add("Dirichlet", side_x, "DispX", 0.0)
    pb.bc.add("Dirichlet", side_y, "DispY", 0.0)
    pb.bc.add("Dirichlet", bottom, "DispZ", 0.0)

    dt = 0.5
    nb_steps = 60
    tmax = dt * nb_steps
    n_ramp = 8  # load reaches its final value after n_ramp steps

    pb.bc.add(
        "Dirichlet",
        top,
        "DispZ",
        delta,
        time_func=lambda tf: min(1.0, tf * tmax / (n_ramp * dt)),
    )
    pb.bc.add("Dirichlet", top, "PorePressure", 0.0)

    p_bottom_history = []

    pb.initialize()
    pb.tmax = tmax
    pb.dtime = dt
    pb.set_start()

    for step in range(nb_steps):
        pb.time = step * dt
        convergence, nb_nr, res = pb.solve_time_increment()
        assert (
            convergence
        ), f"Step {step}: finite-strain poromechanics Newton failed (res={res:g})"
        pb.set_start()
        p_field = pb.get_dof_solution("PorePressure")
        p_bottom_history.append(float(np.mean(p_field[bottom])))

    p_history = np.asarray(p_bottom_history)

    # 1. lnJ present at gauss points (the PoroMomentumSimple fix) and equal
    # to the exact confined kinematics at the drained state.
    lnJ = assembly.sv["lnJ"]
    assert isinstance(lnJ, np.ndarray), "lnJ must be computed in nlgeom mode"
    assert np.allclose(lnJ, np.log(stretch), rtol=1e-2), (
        f"Drained confined compression must give lnJ = ln({stretch}) = "
        f"{np.log(stretch):.5f} uniformly, got "
        f"[{lnJ.min():.5f}, {lnJ.max():.5f}]"
    )

    # 2. Pore pressure builds up (positive in compression) then dissipates.
    p_peak = p_history.max()
    assert (
        p_peak > 0.0
    ), f"Pore pressure must be positive under compression, got {p_peak}"
    assert abs(p_history[-1]) < 0.05 * p_peak, (
        "Pore pressure should be largely dissipated at the end of the run, "
        f"got |p_end|={abs(p_history[-1]):g} vs peak={p_peak:g}"
    )

    # 3. Holmes-Mow permeability received the true J (drops in compression).
    J_final = np.exp(lnJ)
    k_final = perm(J_final)
    assert np.all(k_final < 0.6 * k0), (
        "Holmes-Mow permeability must decrease markedly at 10% compression, "
        f"got k/k0 in [{k_final.min() / k0:.3f}, {k_final.max() / k0:.3f}]"
    )

    # 4. Undrained short-time check: right after the ramp the excess pore
    # pressure at the sealed bottom must still carry a significant part of
    # the applied stress (drainage cannot be instantaneous through the
    # low-permeability column).
    p_end_ramp = p_history[n_ramp - 1]
    assert p_end_ramp > 0.2 * p_peak, (
        "Excess pore pressure at the closed bottom should persist right "
        f"after the ramp, got {p_end_ramp:g} vs peak {p_peak:g}"
    )


if __name__ == "__main__":
    test_finite_strain_consolidation_neohc()
    print("test_poromech_finite_strain: OK")
