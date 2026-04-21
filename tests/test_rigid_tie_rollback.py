"""Quaternion rollback: to_start_bc must exactly restore the pre-set_start state."""

import numpy as np
import pytest

from fedoo.constraint.rigid_tie import RigidTie

try:
    from simcoon import Rotation
except ImportError:
    from scipy.spatial.transform import Rotation


def _make_tie_with_fake_problem(angles_seq):
    """Build a RigidTie wired to a fake problem that returns `angles_seq[i]`
    on the i-th call to `_get_dof_ref`."""
    rt = RigidTie([0, 1], use_quaternion=True)
    rt._Q_base = Rotation.identity()
    rt._Q_base_backup = Rotation.identity()
    rt._angles_at_base = np.zeros(3)

    calls = {"i": 0}

    def fake_get_dof_ref(_problem):
        i = calls["i"]
        calls["i"] += 1
        angles = np.asarray(angles_seq[i], dtype=float)
        return np.concatenate([np.zeros(3), angles]), None

    rt._get_dof_ref = fake_get_dof_ref
    return rt


def _quat_equal(a, b, atol=1e-14):
    qa = a.as_quat()
    qb = b.as_quat()
    return np.allclose(qa, qb, atol=atol) or np.allclose(qa, -qb, atol=atol)


def test_to_start_bc_restores_identity_after_failed_increment():
    rt = _make_tie_with_fake_problem([[0.3, -0.4, 0.5]])
    rt.set_start(problem=None)
    assert not _quat_equal(rt._Q_base, Rotation.identity())

    rt.to_start_bc(problem=None)
    assert _quat_equal(rt._Q_base, Rotation.identity())


def test_two_successful_increments_compose_multiplicatively():
    a1 = np.array([0.1, 0.0, 0.0])
    a2 = np.array([0.1, 0.2, 0.0])
    rt = _make_tie_with_fake_problem([a1, a2])

    rt.set_start(problem=None)
    rt.set_start(problem=None)

    # Q_total should be R(a2 - a1) * R(a1) = composed rotations
    expected = Rotation.from_rotvec(a2 - a1) * Rotation.from_rotvec(a1)
    assert _quat_equal(rt._Q_base, expected)
    assert np.array_equal(rt._angles_at_base, a2)


def test_rollback_after_converged_increment_reverts_to_prior_base():
    a1 = np.array([0.3, 0.0, 0.0])
    a2 = np.array([0.3, 0.4, 0.0])
    rt = _make_tie_with_fake_problem([a1, a2])

    rt.set_start(problem=None)
    q_after_first = rt._Q_base

    rt.set_start(problem=None)
    rt.to_start_bc(problem=None)

    assert _quat_equal(rt._Q_base, q_after_first)


def test_nan_angles_do_not_mutate_base():
    rt = _make_tie_with_fake_problem([[np.nan, 0.0, 0.0]])
    q_before = rt._Q_base
    rt.set_start(problem=None)
    assert _quat_equal(rt._Q_base, q_before)


if __name__ == "__main__":
    pytest.main([__file__])
