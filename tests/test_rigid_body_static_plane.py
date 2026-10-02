"""Rigid body IPC contact against an analytic static plane."""

import numpy as np
import pytest

import fedoo as fd

ipctk = pytest.importorskip("ipctk")
pv = pytest.importorskip("pyvista")

G = 9.81
RADIUS = 0.1
Z0 = 0.5
DHAT = 0.01


def _ball():
    space = fd.ModelingSpace("3D")
    space.new_variable("DispX")
    space.new_variable("DispY")
    space.new_variable("DispZ")
    space.new_vector("Disp", ("DispX", "DispY", "DispZ"))

    mesh = fd.Mesh.from_pyvista(
        pv.Sphere(
            radius=RADIUS, center=(0, 0, Z0), theta_resolution=8, phi_resolution=8
        )
    )
    return fd.constraint.RigidBody(
        mesh,
        mass=1.0,
        inertia_tensor=0.4 * RADIUS**2 * np.eye(3),
        center_of_mass=np.array([0.0, 0.0, Z0]),
    )


def _contact_force(body, gap, x=0.0, y=0.0):
    asm = body.assembly
    q = np.array([x, y, -(Z0 - RADIUS) + gap, 0.0, 0.0, 0.0])
    return asm.compute_contact(q, asm.rigid_tie, compute="force")[0].copy()


def test_plane_force_is_independent_of_lateral_position():
    body = _ball()
    body.set_static_plane(dhat=DHAT, kappa=1e8)

    reference = _contact_force(body, 0.3 * DHAT)
    assert reference[2] > 0
    for x, y in [(1e-3, 0.0), (0.05, 0.02), (0.37, -0.21)]:
        force = _contact_force(body, 0.3 * DHAT, x, y)
        assert force[2] == pytest.approx(reference[2], rel=1e-10)
        # frictionless flat floor under a symmetric ball: purely normal force
        assert np.abs(force[[0, 1, 3, 4, 5]]).max() < 1e-9 * reference[2]


def test_plane_barrier_activates_continuously_at_dhat():
    body = _ball()
    body.set_static_plane(dhat=DHAT, kappa=1e8)

    assert _contact_force(body, 1.01 * DHAT)[2] == 0
    just_inside = _contact_force(body, 0.99 * DHAT)[2]
    assert 0 < just_inside < 1.0
    # ipctk 1.6 activates plane-vertex pairs at half the build distance:
    # the force must not jump there.
    above = _contact_force(body, 0.501 * DHAT)[2]
    below = _contact_force(body, 0.499 * DHAT)[2]
    assert below == pytest.approx(above, rel=0.02)


@pytest.mark.parametrize("dt", [5e-3, 1e-2])
def test_bounce_on_plane_stays_vertical_at_large_time_step(dt):
    body = _ball()
    body.set_force([0.0, 0.0, -G])
    body.set_rayleigh_damping(1.0)
    body.set_static_plane(dhat=DHAT, kappa=1e8)

    pb = fd.problem.NonLinear(body.assembly)
    pb.set_time_integrator(fd.time.SECOND_ORDER, fd.time.Newmark())

    asm = body.assembly
    z_min = []
    compute_contact = asm.compute_contact

    def recording_compute_contact(q, rt, compute="all"):
        z_min.append(asm._ipc_vertices(q, rt)[:, 2].min())
        return compute_contact(q, rt, compute)

    asm.compute_contact = recording_compute_contact

    # update_dt=False: any non-converged increment raises.
    pb.nlsolve(dt=dt, tmax=0.8, update_dt=False, print_info=0)

    assert 0.0 < min(z_min) < DHAT
    q = pb.get_dof_solution()[asm.dof_indices]
    assert np.abs(q[[0, 1, 3, 4, 5]]).max() < 1e-8


def test_several_planes_and_argument_checks():
    body = _ball()
    body.set_static_plane(dhat=DHAT, kappa=1e8)
    body.set_static_plane(normal=(1, 0, 0), point=(-0.5, 0, 0), dhat=DHAT, kappa=1e8)
    assert len(body.assembly._ipc_collision_mesh.planes) == 2

    with pytest.raises(ValueError, match="same dhat and kappa"):
        body.set_static_plane(normal=(0, 1, 0), dhat=2 * DHAT, kappa=1e8)
    with pytest.raises(ValueError, match="non-zero"):
        body.set_static_plane(normal=(0, 0, 0), dhat=DHAT, kappa=1e8)

    floor = fd.Mesh.from_pyvista(pv.Plane(direction=(0, 0, 1)).triangulate())
    other = _ball()
    other.set_static_obstacle(floor, dhat=DHAT, kappa=1e8)
    with pytest.raises(ValueError, match="set_static_obstacle"):
        other.set_static_plane(dhat=DHAT, kappa=1e8)
