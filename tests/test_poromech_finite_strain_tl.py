"""Total Lagrangian poromechanics is the exact image of the updated Lagrangian one.

In UL the fluid mass balance is integrated on the current volume with
spatial gradients. In TL it is integrated on the reference volume, so the
storage and coupling terms carry J and the mobility is pulled back,
``J F^{-1} (k/mu) F^{-T}``. The two formulations then assemble the same
residual and must give the same consolidation.

The second test checks the TL mass-balance tangent (storage and coupling,
including the variation of the J factor) against a central finite
difference of the assembled residual. Darcy is left out: its dependence on
the displacement is not linearized (frozen kinematics, in UL and TL alike).
"""

import numpy as np
import pytest

import fedoo as fd

try:
    from simcoon import simmit  # noqa: F401

    USE_SIMCOON = True
except ImportError:
    USE_SIMCOON = False

E, NU = 1.0e6, 0.3
MU = E / (2.0 * (1.0 + NU))
KAPPA = E / (3.0 * (1.0 - 2.0 * NU))


def _column_side_nodes(mesh, width):
    side_x = np.unique(
        np.concatenate([mesh.find_nodes("X", 0.0), mesh.find_nodes("X", width)])
    )
    side_y = np.unique(
        np.concatenate([mesh.find_nodes("Y", 0.0), mesh.find_nodes("Y", width)])
    )
    return side_x, side_y


def _consolidation(nlgeom, stretch=0.95, nb_steps=20, dt=0.5, n_ramp=4):
    """Confined consolidation history of the bottom pore pressure and lnJ."""
    fd.Assembly.delete_memory()
    fd.ModelingSpace("3D")
    L, width = 1.0, 0.1
    mesh = fd.mesh.box_mesh(
        nx=2,
        ny=2,
        nz=7,
        x_min=0,
        x_max=width,
        y_min=0,
        y_max=width,
        z_min=0,
        z_max=L,
        elm_type="hex8",
    )
    skeleton = fd.constitutivelaw.Simcoon("NEOHC", np.array([MU, KAPPA]))
    permeability = fd.constitutivelaw.HolmesMowPermeability(k0=1.0e-6, n0=0.5)
    fluid = fd.constitutivelaw.PoroFluidProperties(
        permeability=permeability,
        fluid_viscosity=1.0,
        biot_coefficient=1.0,
        biot_modulus=1.0e7,
        initial_porosity=0.5,
    )
    wf = fd.weakform.PoroMechanicsSimple(skeleton, fluid, nlgeom=nlgeom)
    assembly = fd.Assembly.create(wf, mesh)
    pb = fd.problem.NonLinear(assembly)
    pb.set_nr_criterion("Displacement", tol=1e-6, max_subiter=50)

    top = mesh.find_nodes("Z", L)
    bottom = mesh.find_nodes("Z", 0.0)
    side_x, side_y = _column_side_nodes(mesh, width)
    tmax = dt * nb_steps
    pb.bc.add("Dirichlet", side_x, "DispX", 0.0)
    pb.bc.add("Dirichlet", side_y, "DispY", 0.0)
    pb.bc.add("Dirichlet", bottom, "DispZ", 0.0)
    pb.bc.add(
        "Dirichlet",
        top,
        "DispZ",
        (stretch - 1.0) * L,
        time_func=lambda tf: min(1.0, tf * tmax / (n_ramp * dt)),
    )
    pb.bc.add("Dirichlet", top, "PorePressure", 0.0)

    pb.initialize()
    pb.tmax = tmax
    pb.dtime = dt
    pb.set_start()
    p_bottom = np.zeros(nb_steps)
    for step in range(nb_steps):
        pb.time = step * dt
        convergence, nb_nr, res = pb.solve_time_increment()
        assert convergence, f"{nlgeom} step {step}: Newton failed (res={res:g})"
        pb.set_start()
        p_bottom[step] = np.mean(pb.get_dof_solution("PorePressure")[bottom])
    return p_bottom, np.asarray(assembly.sv["lnJ"]).copy()


@pytest.mark.skipif(not USE_SIMCOON, reason="simcoon is required for NEOHC")
def test_tl_consolidation_matches_ul():
    p_ul, lnJ_ul = _consolidation("UL")
    p_tl, lnJ_tl = _consolidation("TL")

    p_peak = np.abs(p_ul).max()
    assert p_peak > 0.0
    err = np.abs(p_tl - p_ul).max() / p_peak
    assert err < 1e-3, (
        f"TL and UL pore pressure histories differ by {err:.2e} of the peak "
        f"(peak {p_peak:g} Pa)"
    )
    assert np.allclose(np.sort(lnJ_tl), np.sort(lnJ_ul), atol=1e-6)


def _tl_storage_fd_error(eps=1e-7, n_dir=3):
    """Max rel. error of the mass-balance rows of K vs a central FD."""
    fd.Assembly.delete_memory()
    space = fd.ModelingSpace("3D")
    width = 0.1
    mesh = fd.mesh.box_mesh(
        nx=2,
        ny=2,
        nz=3,
        x_min=0,
        x_max=width,
        y_min=0,
        y_max=width,
        z_min=0,
        z_max=0.2,
        elm_type="hex8",
    )
    skeleton = fd.constitutivelaw.Simcoon("NEOHC", np.array([MU, KAPPA]))
    fluid = fd.constitutivelaw.PoroFluidProperties(
        permeability=1.0e-6,
        fluid_viscosity=1.0,
        biot_coefficient=0.8,
        biot_modulus=1.0e7,
        initial_porosity=0.5,
    )
    # Momentum + storage only (see module docstring).
    wf = fd.WeakFormSum(
        [
            fd.weakform.PoroMomentumSimple(skeleton, fluid, nlgeom="TL"),
            fd.weakform.PoroMassStorage(fluid),
        ]
    )
    assembly = fd.Assembly.create(wf, mesh)
    pb = fd.problem.NonLinear(assembly)
    pb.set_nr_criterion("Displacement", tol=1e-10, max_subiter=30)

    top = mesh.find_nodes("Z", 0.2)
    bottom = mesh.find_nodes("Z", 0.0)
    side_x, side_y = _column_side_nodes(mesh, width)
    pb.bc.add("Dirichlet", side_x, "DispX", 0.0)
    pb.bc.add("Dirichlet", side_y, "DispY", 0.0)
    pb.bc.add("Dirichlet", bottom, "DispZ", 0.0)
    pb.bc.add("Dirichlet", top, "DispZ", -0.02)  # 10 % compression

    # One uncommitted increment: the start state stays at rest, so the
    # mass-balance rate (and the variation of J it multiplies) is non zero.
    pb.initialize()
    pb.tmax = 1.0
    pb.dtime = 1.0
    pb.set_start()
    pb.time = 0.0
    convergence, nb_nr, res = pb.solve_time_increment()
    assert convergence, f"TL storage increment did not converge (res={res:g})"

    n_nodes = mesh.n_nodes
    rank_p = space.variable_rank("PorePressure")
    rows_p = np.arange(rank_p * n_nodes, (rank_p + 1) * n_nodes)
    blocked = set()
    for nodes, rank in (
        (side_x, space.variable_rank("DispX")),
        (side_y, space.variable_rank("DispY")),
        (np.concatenate([bottom, top]), space.variable_rank("DispZ")),
    ):
        blocked.update(int(n) + rank * n_nodes for n in nodes)
    free = np.array([i for i in range(pb.n_dof) if i not in blocked])

    U0 = np.asarray(pb._U + (0 if np.isscalar(pb._dU) else pb._dU)).copy()
    pb._U = U0.copy()
    pb._dU = 0
    assembly.update(pb, "all")
    K = assembly.current.get_global_matrix().tocsr()

    rng = np.random.default_rng(0)
    errors = []
    for _ in range(n_dir):
        delta = np.zeros(K.shape[0])
        delta[free] = rng.standard_normal(len(free))
        # Scale the displacement and pressure parts separately.
        delta[: rows_p[0]] *= 1e-3
        delta[rows_p] *= 1e3
        r = []
        for sgn in (1, -1):
            pb._U = U0 + sgn * eps * delta
            pb._dU = 0
            assembly.update(pb, "vector")
            r.append(np.asarray(assembly.current.get_global_vector()).ravel().copy())
        fd_dir = (r[0] - r[1]) / (2 * eps)
        Kd = np.asarray(K @ delta).ravel()
        # fedoo assembles -residual in the global vector
        errors.append(
            np.linalg.norm(fd_dir[rows_p] + Kd[rows_p]) / np.linalg.norm(Kd[rows_p])
        )
    return max(errors)


@pytest.mark.skipif(not USE_SIMCOON, reason="simcoon is required for NEOHC")
def test_tl_mass_balance_tangent_is_fd_consistent():
    err = _tl_storage_fd_error()
    assert err < 1e-5, f"TL mass-balance tangent vs FD: rel. error {err:.2e}"
