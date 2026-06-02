"""Quaternion base lifecycle: set_start advances, to_start_bc never reverts.

These tests model the *real* solver sequence (see ``nlsolve``):

* ``set_start`` is called only for a **converged** increment, at the top of
  the following increment — this is the only place ``_Q_base`` advances.
* ``to_start_bc`` is called only for a **failed** increment, which never ran
  ``set_start`` — so ``_Q_base`` is already at the last-converged state and
  rollback must be a no-op.

The earlier version of this constraint reverted ``_Q_base`` to a backup
captured *before* the last converged advance, which discarded the last
converged rotation on any ``dt`` reduction. The tests below pin the correct
behaviour.
"""

import numpy as np
import pytest

from fedoo.constraint.rigid_tie import RigidTie

try:
    from simcoon import Rotation
except ImportError:
    from scipy.spatial.transform import Rotation


def _make_tie_with_fake_problem(angles_seq):
    """Build a RigidTie wired to a fake problem that returns ``angles_seq[i]``
    on the i-th call to ``_get_dof_ref``."""
    rt = RigidTie([0, 1], use_quaternion=True)
    rt._Q_base = Rotation.identity()
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


def test_to_start_bc_is_a_noop_on_the_base():
    """Rollback must leave the quaternion base untouched."""
    rt = _make_tie_with_fake_problem([[0.2, 0.1, -0.3]])
    rt.set_start(problem=None)  # accept a converged increment
    q = rt._Q_base
    assert not _quat_equal(q, Rotation.identity())

    rt.to_start_bc(problem=None)
    rt.to_start_bc(problem=None)  # idempotent
    assert _quat_equal(rt._Q_base, q)


def test_failed_increment_does_not_discard_converged_rotation():
    """The bug this fixes: converge A (set_start), then B fails (to_start_bc).

    In the solver, A's ``set_start`` runs at the top of increment B. When B
    fails, the rollback must keep the base at A's rotation — *not* revert it
    further (which previously sent it all the way back to identity).
    """
    a_converged = [0.3, -0.4, 0.5]
    rt = _make_tie_with_fake_problem([a_converged])  # one set_start: accept A
    rt.set_start(problem=None)
    q_after_A = rt._Q_base
    assert not _quat_equal(q_after_A, Rotation.identity())

    # Increment B now runs and FAILS; its set_start is never called. The
    # solver rolls back via to_start_bc — the base must remain at A.
    rt.to_start_bc(problem=None)
    assert _quat_equal(rt._Q_base, q_after_A)


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


def test_converge_fail_retry_keeps_first_increment():
    """Full sequence: A converges, B fails and rolls back, B retried converges.

    After the retry, the base must equal R(a_B - a_A) * R(a_A) — i.e. the
    converged A rotation is preserved through the failed attempt.
    """
    a_A = np.array([0.3, 0.0, 0.0])
    a_B = np.array([0.3, 0.4, 0.0])
    # set_start consumes one angle per call: accept A, then accept B (retry).
    rt = _make_tie_with_fake_problem([a_A, a_B])

    rt.set_start(problem=None)  # accept converged A
    q_after_A = rt._Q_base

    rt.to_start_bc(problem=None)  # increment B fails -> rollback (no-op)
    assert _quat_equal(rt._Q_base, q_after_A)

    rt.set_start(problem=None)  # accept converged B (retried)
    expected = Rotation.from_rotvec(a_B - a_A) * Rotation.from_rotvec(a_A)
    assert _quat_equal(rt._Q_base, expected)
    assert np.array_equal(rt._angles_at_base, a_B)


def test_nan_angles_do_not_mutate_base():
    rt = _make_tie_with_fake_problem([[np.nan, 0.0, 0.0]])
    q_before = rt._Q_base
    rt.set_start(problem=None)
    assert _quat_equal(rt._Q_base, q_before)


if __name__ == "__main__":
    pytest.main([__file__])
