"""Unit tests for rigid body kinematics and the IPC contact Jacobian.

Avoids ipctk dependencies by testing the RigidBodyAssembly geometry paths
(`_ipc_vertices`, `_build_ipc_jacobian`) against manual formulas.
"""

import numpy as np
import pytest

import fedoo as fd
from fedoo.constraint.rigid_body import RigidBodyAssembly

from simcoon import Rotation  # required dependency (fedoo imports it at load)


@pytest.fixture
def space():
    space = fd.ModelingSpace("3D")
    space.new_variable("DispX")
    space.new_variable("DispY")
    space.new_variable("DispZ")
    space.new_vector("Disp", ("DispX", "DispY", "DispZ"))
    return space


def _make_assembly(rest_positions, obstacle_positions, center):
    rt = fd.constraint.RigidTie(np.arange(len(rest_positions)), center=center)
    rt.center = np.asarray(center, dtype=float)
    asm = RigidBodyAssembly(mass=1.0, inertia_tensor=np.eye(3), rigid_tie=rt)
    asm._ipc_rest_positions = np.asarray(rest_positions, dtype=float)
    asm._ipc_obstacle_nodes = np.asarray(obstacle_positions, dtype=float)
    asm._ipc_n_body = len(rest_positions)
    return asm, rt


def test_ipc_vertices_pure_translation(space):
    rest = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    obst = np.array([[0.0, 0.0, -1.0]])
    center = np.array([0.0, 0.0, 0.0])
    asm, rt = _make_assembly(rest, obst, center)

    q = np.array([0.1, -0.2, 0.3, 0.0, 0.0, 0.0])
    verts = asm._ipc_vertices(q, rt)

    expected_body = rest + q[:3]
    assert np.allclose(verts[:3], expected_body)
    assert np.allclose(verts[3:], obst)


def test_ipc_vertices_pure_rotation_about_center(space):
    rest = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    obst = np.zeros((0, 3))
    center = np.array([0.0, 0.0, 0.0])
    asm, rt = _make_assembly(rest, obst, center)

    # 90° about z — should rotate x into y, y into -x
    q = np.array([0.0, 0.0, 0.0, 0.0, 0.0, np.pi / 2])
    verts = asm._ipc_vertices(q, rt)

    R_expected = Rotation.from_rotvec([0, 0, np.pi / 2]).as_matrix()
    expected = rest @ R_expected.T
    assert np.allclose(verts, expected, atol=1e-12)


def test_ipc_vertices_rotation_off_center(space):
    rest = np.array([[1.0, 0.0, 0.0]])
    obst = np.zeros((0, 3))
    center = np.array([0.5, 0.0, 0.0])
    asm, rt = _make_assembly(rest, obst, center)

    # 180° about z around (0.5,0,0) maps (1,0,0) -> (0,0,0)
    q = np.array([0.0, 0.0, 0.0, 0.0, 0.0, np.pi])
    verts = asm._ipc_vertices(q, rt)
    assert np.allclose(verts[0], [0.0, 0.0, 0.0], atol=1e-12)


def test_build_ipc_jacobian_structure(space):
    rest = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    obst = np.array([[0.0, 0.0, -1.0], [1.0, 0.0, -1.0]])
    center = np.array([0.0, 0.0, 0.0])
    asm, rt = _make_assembly(rest, obst, center)

    J = asm._build_ipc_jacobian(rt, np.zeros(3))

    assert J.shape == (12, 6)
    # Translation block: body rows are identity per vertex; obstacle rows zero.
    for i in range(2):
        for d in range(3):
            assert J[i * 3 + d, d] == 1.0
    assert np.all(J[6:, :3] == 0.0)  # obstacle rows zero in translation cols
    assert np.all(J[6:, 3:] == 0.0)  # obstacle rows zero in rotation cols


def test_build_ipc_jacobian_matches_finite_difference(space):
    rest = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
    )
    obst = np.zeros((0, 3))
    center = np.array([0.2, -0.1, 0.0])
    asm, rt = _make_assembly(rest, obst, center)

    angles0 = np.array([0.1, -0.15, 0.2])
    J = asm._build_ipc_jacobian(rt, angles0)

    # Numerically differentiate vertex positions w.r.t. q = [dx,dy,dz,rx,ry,rz]
    def vertex_positions(q):
        return asm._ipc_vertices(q, rt).ravel()

    q0 = np.concatenate([np.zeros(3), angles0])
    h = 1e-6
    for k in range(6):
        eps = np.zeros(6)
        eps[k] = h
        du_dq = (vertex_positions(q0 + eps) - vertex_positions(q0 - eps)) / (2 * h)
        assert np.allclose(J[:, k], du_dq, atol=1e-6), f"column {k} mismatch"


def _make_rigid_body(dynamic=True):
    nodes = np.array(
        [
            [-0.5, -0.5, 0.0],
            [0.5, -0.5, 0.0],
            [0.5, 0.5, 0.0],
            [-0.5, 0.5, 0.0],
        ]
    )
    mesh = fd.Mesh(nodes, np.array([[0, 1, 2, 3]]), "quad4")
    kwargs = {}
    if dynamic:
        kwargs = {"mass": 1.0, "inertia_tensor": np.eye(3)}
    return fd.constraint.RigidBody(mesh, dynamic=dynamic, **kwargs)


def test_rigid_body_rayleigh_damping_accepts_mass_and_stiffness_terms(space):
    body = _make_rigid_body()

    body.set_rayleigh_damping(alpha=0.7, beta=0.2)

    assert body.assembly.dissipation == fd.time.RayleighDamping(
        alpha=0.7, beta=0.2
    )


def test_static_body_is_not_compiled_by_second_order_integrator(space):
    body = _make_rigid_body(dynamic=False)
    pb = fd.problem.NonLinear(body.assembly)

    assert body.constraint in pb.bc
    assert pb.n_global_dof == 6

    pb.set_time_integrator(fd.time.SECOND_ORDER, fd.time.Newmark())

    pb.initialize()

    assert body.assembly.time_evolution is None
    assert body.assembly.storage is None
    assert body.assembly._time_integrator is None
    assert not body.assembly._fedoo_time_integrated


def test_rigid_body_integrator_is_recompiled_for_each_problem(space):
    body = _make_rigid_body()

    first = fd.problem.NonLinear(body.assembly)
    first.set_time_integrator(
        fd.time.SECOND_ORDER, fd.time.Newmark(beta=0.2, gamma=0.55)
    )
    first.initialize()
    first_adapter = body.assembly._time_integrator
    assert first_adapter.beta == pytest.approx(0.2)

    second = fd.problem.NonLinear(body.assembly)
    second.set_time_integrator(
        fd.time.SECOND_ORDER, fd.time.Newmark(beta=0.3, gamma=0.6)
    )
    second.initialize()
    assert body.assembly._time_integrator is not first_adapter
    assert body.assembly._time_integrator.beta == pytest.approx(0.3)

    static = fd.problem.NonLinear(body.assembly)
    with pytest.warns(UserWarning, match="Assembly provider"):
        static.initialize()
    assert body.assembly._time_integrator is None
    assert not body.assembly._fedoo_time_integrated


def test_problem_set_start_commits_quaternion_increment(space):
    body = _make_rigid_body(dynamic=False)
    pb = fd.problem.NonLinear(body.assembly)
    pb.initialize()
    rotation_indices = body.assembly.dof_indices[3:]

    first = np.array([0.2, 0.0, 0.0])
    pb._dU = np.zeros(pb.n_dof)
    pb._dU[rotation_indices] = first
    pb.set_start()

    second = np.array([0.0, -0.3, 0.1])
    pb._dU = np.zeros(pb.n_dof)
    pb._dU[rotation_indices] = second
    pb.set_start()

    expected = Rotation.from_rotvec(second) * Rotation.from_rotvec(first)
    quaternion = body.constraint.Q_total.as_quat()
    expected_quaternion = expected.as_quat()
    assert np.allclose(quaternion, expected_quaternion) or np.allclose(
        quaternion, -expected_quaternion
    )


if __name__ == "__main__":
    pytest.main([__file__])
