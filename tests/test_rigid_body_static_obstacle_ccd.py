"""CCD line search for a rigid body against a static IPC obstacle."""

import numpy as np
import pytest

import fedoo as fd

ipctk = pytest.importorskip("ipctk")
pv = pytest.importorskip("pyvista")

G = 9.81
RADIUS = 0.1
Z0 = 0.5
DHAT = 0.01


def _bounce_problem(use_ccd):
    space = fd.ModelingSpace("3D")
    space.new_variable("DispX")
    space.new_variable("DispY")
    space.new_variable("DispZ")
    space.new_vector("Disp", ("DispX", "DispY", "DispZ"))

    ball = fd.Mesh.from_pyvista(
        pv.Sphere(
            radius=RADIUS, center=(0, 0, Z0), theta_resolution=8, phi_resolution=8
        )
    )
    floor = fd.Mesh.from_pyvista(
        pv.Plane(
            center=(0, 0, 0),
            direction=(0, 0, 1),
            i_size=1.5,
            j_size=1.5,
            i_resolution=4,
            j_resolution=4,
        ).triangulate()
    )
    body = fd.constraint.RigidBody(
        ball,
        mass=1.0,
        inertia_tensor=0.4 * RADIUS**2 * np.eye(3),
        center_of_mass=np.array([0.0, 0.0, Z0]),
    )
    body.set_force([0.0, 0.0, -G])
    body.set_rayleigh_damping(1.0)
    body.set_static_obstacle(floor, dhat=DHAT, kappa=1e8, use_ccd=use_ccd)

    pb = fd.problem.NonLinear(body.assembly)
    pb.set_time_integrator(fd.time.SECOND_ORDER, fd.time.Newmark())

    # Record the lowest point of the ball at every contact evaluation.
    asm = body.assembly
    z_min = []
    compute_contact = asm.compute_contact

    def recording_compute_contact(q, rt, compute="all"):
        z_min.append(asm._ipc_vertices(q, rt)[: asm._ipc_n_body, 2].min())
        return compute_contact(q, rt, compute)

    asm.compute_contact = recording_compute_contact
    return pb, body, z_min


def test_static_obstacle_registers_ccd_line_search():
    pb, body, _ = _bounce_problem(use_ccd=True)
    pb.nlsolve(dt=1e-3, tmax=2e-3, update_dt=False, print_info=0)
    assert f"ccd_{body.assembly.name}" in pb._ls_callbacks


def test_static_obstacle_ccd_can_be_disabled():
    pb, body, _ = _bounce_problem(use_ccd=False)
    pb.nlsolve(dt=1e-3, tmax=2e-3, update_dt=False, print_info=0)
    assert not pb._ls_callbacks


@pytest.mark.parametrize("dt", [5e-3, 2e-2])
def test_large_time_step_does_not_tunnel_through_obstacle(dt):
    # Impact speed is about 2.8 m/s, so both steps travel further than dhat.
    assert np.sqrt(2 * G * (Z0 - RADIUS)) * dt > DHAT

    pb, _, z_min = _bounce_problem(use_ccd=True)
    pb.nlsolve(dt=dt, dt_max=dt, tmax=0.6, update_dt=True, print_info=0)

    assert pb.time == pytest.approx(0.6)
    assert min(z_min) > 0.0
    # The ball reached the barrier zone, so the contact was actually exercised.
    assert min(z_min) < DHAT


class _FixedStateProblem:
    """Minimal problem exposing a rigid DOF state to the CCD line search."""

    def __init__(self, q):
        self._q = np.asarray(q, dtype=float)

    def get_dof_solution(self):
        return self._q


def test_ccd_accounts_for_curved_paths_of_a_rotating_body():
    space = fd.ModelingSpace("3D")
    space.new_variable("DispX")
    space.new_variable("DispY")
    space.new_variable("DispZ")
    space.new_vector("Disp", ("DispX", "DispY", "DispZ"))

    height = 0.3
    bar = fd.Mesh.from_pyvista(
        pv.Cube(center=(0, 0, height), x_length=0.4, y_length=0.05, z_length=0.05)
        .triangulate()
        .clean()
    )
    body = fd.constraint.RigidBody(
        bar,
        mass=1.0,
        inertia_tensor=np.diag([4e-4, 1.35e-2, 1.35e-2]),
        center_of_mass=np.array([0.0, 0.0, height]),
    )
    body.set_static_plane(dhat=DHAT, kappa=1e8)
    asm = body.assembly
    asm._dof_indices = np.arange(6)

    # Steps with a 1 rad rotation: the vertices follow arcs, so a step scaled
    # by a straight-line CCD would end below the floor in many of these cases.
    rng = np.random.default_rng(1)
    n_limited = 0
    while n_limited < 25:
        q = np.zeros(6)
        q[2] = -rng.uniform(0.0, 0.2)
        q[3:] = 0.3 * rng.normal(size=3)
        if asm._ipc_vertices(q, asm.rigid_tie)[:, 2].min() < 1.2 * DHAT:
            continue
        axis = rng.normal(size=3)
        dq = np.zeros(6)
        dq[2] = -rng.uniform(0.0, 0.05)
        dq[3:] = axis / np.linalg.norm(axis)

        alpha = asm._ccd_line_search(_FixedStateProblem(q), dq)
        assert 0.0 <= alpha <= 1.0
        if alpha == 1.0:
            continue
        n_limited += 1
        # The whole accepted path stays above the floor.
        for fraction in np.linspace(0.0, 1.0, 21):
            vertices = asm._ipc_vertices(q + fraction * alpha * dq, asm.rigid_tie)
            assert vertices[:, 2].min() > 0.0
