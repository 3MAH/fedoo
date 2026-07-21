"""Lifecycle tests for multiplicative quaternion rotation updates."""

import numpy as np

from fedoo.constraint.rigid_tie import RigidTie
from simcoon import Rotation


def _make_tie(angles):
    tie = RigidTie(np.array([0, 1]), use_quaternion=True)
    tie._Q_base = Rotation.identity()
    tie._angles_at_base = np.zeros(3)
    calls = iter(np.asarray(value, dtype=float) for value in angles)

    def get_dof_ref(_problem):
        return np.concatenate([np.zeros(3), next(calls)]), None

    tie._get_dof_ref = get_dof_ref
    return tie


def _same_rotation(left, right, atol=1e-14):
    left_quaternion = left.as_quat()
    right_quaternion = right.as_quat()
    return np.allclose(left_quaternion, right_quaternion, atol=atol) or np.allclose(
        left_quaternion, -right_quaternion, atol=atol
    )


def test_successive_increments_compose_multiplicatively():
    first = np.array([0.1, 0.0, 0.0])
    second = np.array([0.1, 0.2, 0.0])
    tie = _make_tie([first, second])

    tie.set_start(None)
    tie.set_start(None)

    expected = Rotation.from_rotvec(second - first) * Rotation.from_rotvec(first)
    assert _same_rotation(tie.Q_total, expected)
    np.testing.assert_array_equal(tie._angles_at_base, second)


def test_failed_increment_keeps_last_converged_orientation():
    converged = np.array([0.3, -0.4, 0.5])
    tie = _make_tie([converged])
    tie.set_start(None)
    orientation = Rotation.from_quat(tie.Q_total.as_quat())

    tie.to_start(None)
    tie.to_start(None)

    assert _same_rotation(tie.Q_total, orientation)


def test_total_rotvec_mode_does_not_create_quaternion_state():
    tie = RigidTie(np.array([0, 1]), use_quaternion=False)
    rotation, _, _, _ = tie._compute_rotation(np.array([0.2, -0.1, 0.4]))

    expected = Rotation.from_rotvec([0.2, -0.1, 0.4]).as_matrix()
    np.testing.assert_allclose(rotation, expected)
    assert tie.Q_total is None


def test_non_finite_increment_does_not_mutate_orientation():
    tie = _make_tie([[np.nan, 0.0, 0.0]])
    tie.set_start(None)
    assert _same_rotation(tie.Q_total, Rotation.identity())
