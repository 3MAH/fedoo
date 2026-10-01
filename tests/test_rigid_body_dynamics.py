"""Rotational inertia tests for dynamic rigid bodies."""

import numpy as np
import pytest

import fedoo as fd
from fedoo.constraint.rigid_body import RigidBodyAssembly
from simcoon import Rotation


def _make_assembly(inertia, rotvec):
    space = fd.ModelingSpace("3D")
    tie = fd.constraint.RigidTie(np.array([0]), center=np.zeros(3), use_quaternion=True)
    tie._Q_base = Rotation.identity()
    tie._angles_at_base = np.zeros(3)
    state = {"rotvec": np.asarray(rotvec, dtype=float).copy()}

    def get_dof_ref(_problem):
        return np.concatenate([np.zeros(3), state["rotvec"]]), None

    tie._get_dof_ref = get_dof_ref
    assembly = RigidBodyAssembly(
        mass=2.0,
        inertia_tensor=np.asarray(inertia, dtype=float),
        rigid_tie=tie,
        space=space,
    )
    assembly._dof_indices = np.arange(6)
    return assembly, state


def test_non_spherical_inertia_adds_gyroscopic_force():
    inertia = np.diag([2.0, 3.0, 5.0])
    assembly, _ = _make_assembly(inertia, np.zeros(3))
    acceleration = np.array([0.1, -0.2, 0.3, 0.4, -0.5, 0.6])
    velocity = np.array([1.0, 2.0, -1.0, 0.7, -0.4, 1.2])

    force = assembly.get_time_inertia_force(object(), acceleration, velocity)

    angular_velocity = velocity[3:]
    expected_moment = inertia @ acceleration[3:] + np.cross(
        angular_velocity, inertia @ angular_velocity
    )
    np.testing.assert_allclose(force[:3], 2.0 * acceleration[:3])
    np.testing.assert_allclose(force[3:], expected_moment)
    assert not np.allclose(np.cross(angular_velocity, inertia @ angular_velocity), 0.0)


def test_spherical_inertia_has_no_gyroscopic_force():
    inertia = 3.0 * np.eye(3)
    assembly, _ = _make_assembly(inertia, np.zeros(3))
    acceleration = np.array([0.0, 0.0, 0.0, 0.4, -0.5, 0.6])
    velocity = np.array([0.0, 0.0, 0.0, 0.7, -0.4, 1.2])

    force = assembly.get_time_inertia_force(object(), acceleration, velocity)

    np.testing.assert_allclose(force[3:], inertia @ acceleration[3:])


def test_gyroscopic_inertia_tangent_matches_finite_difference():
    inertia = np.diag([2.0, 3.0, 5.0])
    rotvec = np.array([0.2, -0.15, 0.1])
    assembly, state = _make_assembly(inertia, rotvec)
    acceleration = np.array([0.1, -0.2, 0.3, 0.4, -0.5, 0.6])
    velocity = np.array([1.0, 2.0, -1.0, 0.7, -0.4, 1.2])
    acceleration_factor = 8.0
    velocity_factor = 2.5

    tangent = assembly.get_time_inertia_tangent(
        object(),
        acceleration,
        velocity,
        acceleration_factor,
        velocity_factor,
    )

    h = 1e-7
    numerical = np.zeros((3, 3))
    reference = state["rotvec"].copy()
    for axis in range(3):
        perturbation = np.zeros(3)
        perturbation[axis] = h

        state["rotvec"] = reference + perturbation
        acceleration_plus = acceleration.copy()
        velocity_plus = velocity.copy()
        acceleration_plus[3 + axis] += acceleration_factor * h
        velocity_plus[3 + axis] += velocity_factor * h
        force_plus = assembly.get_time_inertia_force(
            object(), acceleration_plus, velocity_plus
        )[3:]

        state["rotvec"] = reference - perturbation
        acceleration_minus = acceleration.copy()
        velocity_minus = velocity.copy()
        acceleration_minus[3 + axis] -= acceleration_factor * h
        velocity_minus[3 + axis] -= velocity_factor * h
        force_minus = assembly.get_time_inertia_force(
            object(), acceleration_minus, velocity_minus
        )[3:]
        numerical[:, axis] = (force_plus - force_minus) / (2.0 * h)

    state["rotvec"] = reference
    np.testing.assert_allclose(tangent[3:, 3:], numerical, rtol=1e-6, atol=1e-6)


def test_principal_axis_torque_matches_constant_angular_acceleration():
    space = fd.ModelingSpace("3D")
    space.new_variable("DispX")
    space.new_variable("DispY")
    space.new_variable("DispZ")
    space.new_vector("Disp", ("DispX", "DispY", "DispZ"))
    nodes = np.array(
        [
            [-0.5, -0.5, 0.0],
            [0.5, -0.5, 0.0],
            [0.5, 0.5, 0.0],
            [-0.5, 0.5, 0.0],
        ]
    )
    mesh = fd.Mesh(nodes, np.array([[0, 1, 2, 3]]), "quad4")
    inertia = np.diag([2.0, 3.0, 5.0])
    body = fd.constraint.RigidBody(
        mesh,
        mass=1.0,
        inertia_tensor=inertia,
        center_of_mass=np.zeros(3),
    )
    torque = 4.0
    body.set_torque([torque, 0.0, 0.0])

    end_time = 0.02
    body.solve(
        dt=1e-3,
        tmax=end_time,
        print_info=0,
        solver="direct_scipy",
    )

    expected_angle = 0.5 * torque / inertia[0, 0] * end_time**2
    rotation_vector = body.constraint.Q_total.as_rotvec()
    np.testing.assert_allclose(rotation_vector, [expected_angle, 0.0, 0.0], atol=1e-10)


@pytest.mark.parametrize("scheme", ["newmark", "generalized_alpha"])
def test_rigid_body_neumann_loads_in_assembly_sum(scheme):
    """Each body's load enters B once and remains visible in recovered forces."""
    space = fd.ModelingSpace("3D")
    space.new_variable("DispX")
    space.new_variable("DispY")
    space.new_variable("DispZ")
    space.new_vector("Disp", ("DispX", "DispY", "DispZ"))
    square = np.array([[-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, 0.5, 0], [-0.5, 0.5, 0]])
    mesh = fd.Mesh(
        np.vstack([square, square + [3, 0, 0]]),
        np.array([[0, 1, 2, 3], [4, 5, 6, 7]]),
        "quad4",
    )
    bodies = [
        fd.constraint.RigidBody(
            mesh.extract_elements([i]),
            mass=i + 1,
            inertia_tensor=np.eye(3),
            name=f"body_{i}",
        )
        for i in range(2)
    ]
    assembly = fd.Assembly.sum(*(body.assembly for body in bodies))
    dt, end_time = 0.01, 0.02
    pb = fd.problem.NonLinear(assembly)
    integrator = (
        fd.time.Newmark()
        if scheme == "newmark"
        else fd.time.GeneralizedAlpha(alpha_m=0.1, alpha_f=0.1)
    )
    pb.set_time_integrator(fd.time.SECOND_ORDER, integrator)
    pb.set_nr_criterion("Force", tol=1e-10)
    pb.set_solver("direct_scipy")
    original_bc_count = len(pb.bc)
    for body in bodies:
        body.add_to_problem(pb)
    assert len(pb.bc) == original_bc_count

    # These setters run after registration: the generated loads must stay live.
    bodies[0].set_force([0, 0, -3])
    bodies[0].set_torque([0.5, 0, 0])
    bodies[1].set_generalized_force([4, 0, 0, 0, 0, 0])
    pb.apply_boundary_conditions(t_fact=0)
    pb.initialize()
    for body in bodies:
        indices = body.assembly.dof_indices
        np.testing.assert_array_equal(pb.get_B()[indices], body.assembly.force)
        np.testing.assert_array_equal(
            body.assembly.get_time_initial_force(), np.zeros(6)
        )
        np.testing.assert_array_equal(
            body.assembly._time_integrator.force_start, np.zeros(6)
        )
        expected_acceleration = body.assembly.force.copy()
        expected_acceleration[:3] /= body.mass
        np.testing.assert_allclose(
            body.assembly.sv["Acceleration"], expected_acceleration
        )

    pb.nlsolve(dt=dt, tmax=end_time, update_dt=False, print_info=0)
    for body in bodies:
        indices = body.assembly.dof_indices
        np.testing.assert_allclose(
            pb.get_ext_forces(include_mpc=False)[indices],
            body.assembly.force,
            atol=1e-10,
        )
        expected_displacement = 0.5 * end_time**2 * body.assembly.force[:3] / body.mass
        np.testing.assert_allclose(
            pb.get_dof_solution()[indices[:3]], expected_displacement, atol=1e-12
        )
