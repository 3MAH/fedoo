"""Geometrically exact (Simo-Reissner) beam: tangent, analytical solutions."""

from copy import deepcopy

import numpy as np
import pytest
from scipy.integrate import solve_bvp
from scipy.spatial.transform import Rotation

import fedoo as fd
from fedoo.weakform._beam_exact import exact_beam_kinematics


def _deformed_beam(dim):
    space = fd.ModelingSpace("2D" if dim == 2 else "3D")
    nodes = np.array([[0.0, 0.0, 0.0], [2.0, 0.3, 0.2], [3.0, 1.0, 0.5]])[:, :dim]
    mesh = fd.Mesh(nodes, np.array([[0, 1], [1, 2]]), "lin2")
    wf = fd.weakform.BeamEquilibrium(
        fd.constitutivelaw.ElasticIsotrop(1200.0, 0.25),
        A=0.4,
        Jx=0.03,
        Iyy=0.02,
        Izz=0.05,
        k=0.8,
        nlgeom=True,
        space=space,
        kinematics="exact",
    )
    assembly = fd.Assembly.create(wf, mesh)
    pb = fd.problem.NonLinear(assembly)
    assembly.initialize(pb)
    u = np.zeros((space.nvar, mesh.n_nodes))
    u[space.variable_rank("DispX"), 1:] = 0.12
    u[space.variable_rank("DispY"), 1:] = 0.65
    if dim == 2:
        u[space.variable_rank("RotZ")] = np.linspace(0.25, 0.65, mesh.n_nodes)
    else:
        u[space.variable_rank("DispZ"), 1:] = -0.25
        for name, values in zip(
            ["RotX", "RotY", "RotZ"], [[0.2, 0.45], [-0.3, 0.15], [0.3, 0.7]]
        ):
            u[space.variable_rank(name)] = np.linspace(*values, mesh.n_nodes)
    pb._dU = u.ravel()
    pb.set_X(pb._dU.copy())
    assembly.update(pb)
    # Linearization is with respect to the next spatial increment.
    pb.set_X(np.zeros_like(pb._dU))
    assembly.update(pb)
    return wf, assembly, pb


def _numerical_tangent(assembly, pb, h=1e-6):
    state = deepcopy(assembly.sv)
    base = pb._dU.copy()
    pb._line_search_update = True

    def residual(delta):
        assembly.sv = assembly.current.sv = deepcopy(state)
        pb._dU = base + delta
        pb.set_X(delta)
        assembly.update(pb, compute="vector")
        return -assembly.current.global_vector.copy()

    numerical = np.empty((len(base), len(base)))
    for col in range(len(base)):
        delta = np.zeros_like(base)
        delta[col] = h
        numerical[:, col] = (residual(delta) - residual(-delta)) / (2 * h)
    return numerical


@pytest.mark.parametrize("dim", [2, 3])
def test_tangent_matches_assembled_internal_force(dim):
    _, assembly, pb = _deformed_beam(dim)
    tangent = assembly.current.global_matrix.toarray()
    numerical = _numerical_tangent(assembly, pb)
    scale = np.abs(numerical).max()
    np.testing.assert_allclose(tangent, numerical, rtol=0, atol=1e-8 * scale)
    if dim == 2:
        # Additive planar rotations: the residual is a potential gradient.
        np.testing.assert_allclose(tangent, tangent.T, rtol=0, atol=1e-12 * scale)


def test_strains_are_invariant_under_a_superposed_rigid_rotation():
    space = fd.ModelingSpace("3D")
    # defines the Disp and Rot variables of the space
    fd.weakform.BeamEquilibrium(
        fd.constitutivelaw.ElasticIsotrop(1.0, 0.3), 1, 1, 1, 1, k=1, space=space
    )
    rng = np.random.default_rng(0)
    length = np.array([1.3, 0.7])
    u_local = rng.normal(scale=0.2, size=(2, 2, 3))
    triads = Rotation.from_rotvec(rng.normal(scale=0.8, size=(4, 3))).as_matrix()
    triads = triads.reshape(2, 2, 3, 3)
    reference = exact_beam_kinematics(space, length, u_local, triads)
    q = Rotation.from_rotvec([0.4, -2.2, 1.1]).as_matrix()
    # Positions x = X + u rotate with q; in the element frame X = (s, 0, 0).
    positions = u_local.copy()
    positions[:, 1, 0] += length
    rotated = np.einsum("ij,enj->eni", q, positions)
    rotated[:, 1, 0] -= length
    rotated_triads = q @ triads
    moved = exact_beam_kinematics(space, length, rotated, rotated_triads)
    np.testing.assert_allclose(moved.strain, reference.strain, atol=1e-12)


def _cantilever(dim, n, material, section_factory, nlgeom="TL"):
    fd.ModelingSpace("2D" if dim == 2 else "3D")
    length = 10.0
    mesh = fd.mesh.line_mesh(n + 1, 0, length, "lin2", ndim=dim)
    section = section_factory(material)
    wf = fd.weakform.BeamEquilibrium(section, nlgeom=nlgeom, kinematics="exact")
    pb = fd.problem.NonLinear(fd.Assembly.create(wf, mesh))
    tip = mesh.find_nodes("X", length)
    pb.bc.add("Dirichlet", [0], "Disp", 0)
    pb.bc.add("Dirichlet", [0], "Rot", 0)
    pb.set_nr_criterion("Displacement", tol=1e-10, max_subiter=20)
    return pb, mesh, tip, length


@pytest.mark.parametrize("dim", [2, 3])
def test_end_moment_rolls_the_beam_into_a_closed_circle(dim):
    # Simo & Vu-Quoc (1986): M = 2 pi EI / L bends the beam into a full
    # circle. N = 0, so the discrete solution is the inscribed polygon and
    # the free end returns exactly to the clamp.
    material = fd.constitutivelaw.ElasticIsotrop(1.0e3, 0.3)
    pb, mesh, tip, length = _cantilever(
        dim, 20, material, lambda m: fd.constitutivelaw.BeamRectangular(m, 0.5, 0.5)
    )
    inertia = 0.5**4 / 12
    pb.bc.add("Neumann", tip, "RotZ", 2 * np.pi * 1.0e3 * inertia / length)
    pb.nlsolve(dt=0.1, update_dt=False, print_info=0)
    end = mesh.nodes[tip[0]] + pb.get_disp()[:, tip[0]]
    np.testing.assert_allclose(end, 0, atol=1e-10)
    rotation = pb.get_rot()[..., tip[0]]
    assert np.ravel(rotation)[-1] == pytest.approx(2 * np.pi, abs=1e-10)


def _reissner_elastica(load, axial, shear, bending, length):
    """Tip of a shearable, extensible cantilever under a dead transverse load."""

    def rhs(s, y):
        _, _, theta, kappa = y
        c, s_ = np.cos(theta), np.sin(theta)
        gamma_1 = load * s_ / axial
        gamma_2 = load * c / shear
        dx = c * (1 + gamma_1) - s_ * gamma_2
        dy = s_ * (1 + gamma_1) + c * gamma_2
        return np.vstack([dx, dy, kappa, -load * dx / bending])

    s = np.linspace(0, length, 200)
    guess = np.zeros((4, s.size))
    guess[0] = s
    solution = solve_bvp(
        rhs,
        lambda a, b: np.array([a[0], a[1], a[2], b[3]]),
        s,
        guess,
        tol=1e-10,
        max_nodes=100000,
    )
    return solution.sol(length)[:2]


def test_cantilever_under_dead_load_converges_to_the_reissner_elastica():
    E, nu, radius, k = 1.0e3, 0.3, 0.25, 0.9
    material = fd.constitutivelaw.ElasticIsotrop(E, nu)
    area, inertia = np.pi * radius**2, np.pi * radius**4 / 4
    load = 10 * E * inertia / 10.0**2
    reference = _reissner_elastica(
        load, E * area, k * E / (2 * (1 + nu)) * area, E * inertia, 10.0
    )
    errors = []
    for n in (16, 32):
        pb, mesh, tip, length = _cantilever(
            2,
            n,
            material,
            lambda m: fd.constitutivelaw.BeamCircular(m, radius, k=k),
        )
        pb.bc.add("Neumann", tip, "DispY", load)
        pb.nlsolve(dt=0.2, update_dt=True, print_info=0)
        end = mesh.nodes[tip[0]] + pb.get_disp()[:, tip[0]]
        errors.append(np.linalg.norm(end - reference) / length)
    assert errors[1] < 2e-4
    # one-point two-node elements converge at second order
    assert errors[0] / errors[1] == pytest.approx(4, rel=0.1)


def test_fixed_spatial_end_moment_gives_the_exact_helix():
    # Circular section with nu = 0, so GJ = EI: under a fixed spatial end
    # moment m, n = 0 and Lambda(s) = exp(s [m] / EI).
    material = fd.constitutivelaw.ElasticIsotrop(1.0e3, 0.0)
    radius = 0.25
    pb, mesh, tip, length = _cantilever(
        3, 40, material, lambda m: fd.constitutivelaw.BeamCircular(m, radius, k=0.9)
    )
    bending = 1.0e3 * np.pi * radius**4 / 4
    moment = np.array([0.6, 0.0, 1.0]) * np.pi * bending / length
    for name, value in zip(("RotX", "RotZ"), moment[[0, 2]]):
        pb.bc.add("Neumann", tip, name, value)
    pb.nlsolve(dt=0.1, update_dt=True, print_info=0)
    s = np.linspace(0, length, 4001)
    directors = Rotation.from_rotvec(np.outer(s, moment / bending)).apply([1, 0, 0])
    reference = np.trapezoid(directors, s, axis=0)
    end = mesh.nodes[tip[0]] + pb.get_disp()[:, tip[0]]
    np.testing.assert_allclose(end, reference, atol=3e-4 * length)


def test_closed_loop_of_nodal_rotations_leaves_no_strain():
    _, assembly, pb = _deformed_beam(3)
    space = assembly.space
    n = assembly.mesh.n_nodes
    strain = np.array(assembly.sv["BeamStrain"]).copy()
    first, second = (
        Rotation.from_rotvec([0.7, 0, 0]),
        Rotation.from_rotvec([0, -0.9, 0]),
    )
    # Two spins about different axes, then the single spin that undoes both:
    # the nodal rotation returns to its start along a non-trivial path.
    closing = (second * first).inv().as_rotvec()
    for spin in ([0.7, 0, 0], [0, -0.9, 0], closing):
        delta = np.zeros((space.nvar, n))
        delta[space.get_rank_vector("Rot"), 1] = spin
        pb._dU = pb._dU + delta.ravel()
        pb.set_X(delta.ravel())
        assembly.update(pb)
    np.testing.assert_allclose(np.array(assembly.sv["BeamStrain"]), strain, atol=1e-12)


@pytest.mark.parametrize("dim", [2, 3])
def test_rollback_restores_force_and_tangent(dim):
    _, assembly, pb = _deformed_beam(dim)
    assembly.set_start(pb)
    tangent = assembly.current.global_matrix.toarray().copy()
    force = assembly.current.global_vector.copy()
    base = pb._dU.copy()
    delta = np.full_like(base, 0.025)
    pb._dU = base + delta
    pb.set_X(delta)
    assembly.update(pb)
    pb._dU = base
    pb.set_X(np.zeros_like(base))
    assembly.to_start(pb)
    assembly.current.assemble_global_mat()
    np.testing.assert_allclose(
        assembly.current.global_matrix.toarray(), tangent, atol=1e-10
    )
    np.testing.assert_allclose(assembly.current.global_vector, force, atol=1e-10)


def test_invalid_configurations_are_rejected():
    space = fd.ModelingSpace("2D")
    material = fd.constitutivelaw.ElasticIsotrop(1.0e3, 0.3)
    mesh = fd.mesh.line_mesh(3, 0, 1, "lin2", ndim=2)
    with pytest.raises(ValueError, match="kinematics"):
        fd.weakform.BeamEquilibrium(material, 1, 1, 1, 1, k=1, kinematics="foo")
    with pytest.raises(ValueError, match="consistent_tangent"):
        fd.weakform.BeamEquilibrium(
            material, 1, 1, 1, 1, k=1, kinematics="exact", consistent_tangent=True
        )
    for wf, error in [
        # no shear stiffness
        (
            fd.weakform.BeamEquilibrium(
                material, 1, 1, 1, 1, nlgeom="TL", kinematics="exact", space=space
            ),
            ValueError,
        ),
        # linear problem
        (
            fd.weakform.BeamEquilibrium(
                material, 1, 1, 1, 1, k=1, nlgeom=False, kinematics="exact"
            ),
            ValueError,
        ),
        # the corotational beam has no total Lagrangian version
        (
            fd.weakform.BeamEquilibrium(material, 1, 1, 1, 1, nlgeom="TL"),
            NotImplementedError,
        ),
    ]:
        assembly = fd.Assembly.create(wf, mesh)
        with pytest.raises(error):
            fd.problem.NonLinear(assembly).initialize()


@pytest.mark.parametrize("kinematics", ["corotational", "exact"])
def test_ccd_step_filter_converges_with_prescribed_rotations(kinematics):
    # A pinned beam is pushed onto a clamped one with IPC contact and CCD.
    # The rotvec rewrite of the Rot DOFs used to leave 1e-24 on the
    # prescribed rotations, so the "Dirichlet applied" test never passed
    # once CCD shortened a step: every increment failed at contact onset.
    pytest.importorskip("ipctk")
    fd.Assembly.delete_memory()
    fd.ModelingSpace("3D")
    radius, gap, push, dhat, n = 0.05, 0.2, 0.15, 0.02, 10
    s = np.linspace(-1, 1, n)
    zero = np.zeros(n)
    nodes = np.vstack([np.c_[s, zero, zero], np.c_[zero, s, zero + gap]])
    elements = np.c_[np.arange(n - 1), np.arange(1, n)]
    mesh = fd.Mesh(nodes, np.vstack([elements, elements + n]), "lin2")
    section = fd.constitutivelaw.BeamCircular(
        fd.constitutivelaw.ElasticIsotrop(1e5, 0.3), radius, k=0.9
    )
    wf = fd.weakform.BeamEquilibrium(section, nlgeom=True, kinematics=kinematics)
    contact = fd.constraint.IPCContact(
        mesh,
        surface_mesh=mesh,
        dhat=dhat,
        dhat_is_relative=False,
        dmin=2 * radius,
        use_ccd=True,
    )
    pb = fd.problem.NonLinear(fd.Assembly.sum(fd.Assembly.create(wf, mesh), contact))
    lower, upper = [0, n - 1], [n, 2 * n - 1]
    pb.bc.add("Dirichlet", lower, "Disp", 0)
    pb.bc.add("Dirichlet", lower, "Rot", 0)
    for name, value in (("DispX", 0), ("DispY", 0), ("DispZ", -push), ("RotY", 0)):
        pb.bc.add("Dirichlet", upper, name, value)
    pb.set_nr_criterion("Displacement", tol=5e-3, max_subiter=15)
    pb.nlsolve(dt=0.1, tmax=1.0, update_dt=True, print_info=0)
    vertices = contact._get_current_vertices(pb)
    assert (
        2 * radius * (1 - 1e-6)
        <= contact._minimum_distance(vertices)
        < 2 * radius + dhat
    )
    fd.Assembly.delete_memory()


def test_so3_coefficients_are_accurate_across_the_series_threshold():
    mp = pytest.importorskip("mpmath")
    from fedoo.util.so3 import jacobian_coefficients

    mp.mp.dps = 40
    closed = {
        "a": lambda x: (1 - x / 2 * mp.cot(x / 2)) / x**2,
        "b": lambda x: (1 - mp.cos(x)) / x**2,
        "c": lambda x: (x - mp.sin(x)) / x**3,
    }
    angles = np.array([1e-3, 0.05, 0.051, 0.3, 0.99, 1.0, 1.01, 2.0, 3.0])
    a, alpha, b, beta, c, gamma = jacobian_coefficients(angles)
    computed = {"a": a, "alpha": alpha, "b": b, "beta": beta, "c": c, "gamma": gamma}
    for name, derivative in (("a", "alpha"), ("b", "beta"), ("c", "gamma")):
        f = closed[name]
        exact = [float(f(mp.mpf(x))) for x in angles]
        exact_d = [float(mp.diff(f, mp.mpf(x)) / x) for x in angles]
        np.testing.assert_allclose(computed[name], exact, rtol=1e-13)
        np.testing.assert_allclose(computed[derivative], exact_d, rtol=1e-13)


def test_line_search_trial_evaluates_the_scaled_rotation_step():
    # A trial at alpha must see the state the solver reaches when it accepts
    # alpha: rotations included, although they are composed on stored triads.
    from fedoo.problem.line_search import _evaluate_residual_norm

    material = fd.constitutivelaw.ElasticIsotrop(1.0e3, 0.3)
    pb, mesh, tip, length = _cantilever(
        3, 10, material, lambda m: fd.constitutivelaw.BeamCircular(m, 0.25, k=0.9)
    )
    bending = 1.0e3 * np.pi * 0.25**4 / 4
    pb.bc.add("Neumann", tip, "RotY", 0.4 * np.pi * bending / length)
    pb.bc.add("Neumann", tip, "RotZ", 0.6 * np.pi * bending / length)
    trials, mismatch = [], []

    def half_steps(pb, dX):
        if trials and len(mismatch) < len(trials):
            reached = pb._get_free_dof_residual()
            mismatch.append(
                np.linalg.norm(reached - trials[-1]) / np.linalg.norm(trials[-1])
            )
        if len(trials) == 3:
            return 1.0
        trials.append(_evaluate_residual_norm(pb, dX, 0.5)[1])
        return 0.5

    pb.add_line_search(half_steps)
    pb.nlsolve(dt=0.25, update_dt=True, print_info=0)
    assert len(mismatch) == 3
    assert max(mismatch) < 1e-10
