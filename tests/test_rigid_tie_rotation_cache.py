"""Cache of the last rotation computed by RigidTie._compute_rotation."""

import numpy as np
import pytest
from simcoon import Rotation

import fedoo as fd


def _fresh(tie, rotvec):
    """Result of _compute_rotation without the cache."""
    tie._rotation_cache = None
    return tie._compute_rotation(rotvec)


def _assert_same(res1, res2):
    for a, b in zip(res1, res2):
        assert np.array_equal(a, b)


@pytest.fixture
def tie():
    tie = fd.constraint.RigidTie(np.arange(4), use_quaternion=True)
    tie._Q_base = Rotation.identity()
    tie._angles_at_base = np.zeros(3)
    return tie


def test_cache_reused_and_read_only(tie):
    rotvec = np.array([0.1, -0.2, 0.3])
    res1 = tie._compute_rotation(rotvec)
    res2 = tie._compute_rotation(rotvec.copy())
    assert all(a is b for a, b in zip(res1, res2))
    for a in res1:
        assert not a.flags.writeable
    _assert_same(res1, _fresh(tie, rotvec))


def test_cache_invalidated_by_rotvec(tie):
    res1 = tie._compute_rotation(np.array([0.1, -0.2, 0.3]))
    rotvec = np.array([0.1, -0.2, 0.3 + 1e-12])
    res2 = tie._compute_rotation(rotvec)
    assert not np.array_equal(res1[0], res2[0])
    _assert_same(res2, _fresh(tie, rotvec))


def test_cache_invalidated_by_converged_orientation(tie):
    rotvec = np.array([0.2, 0.1, -0.1])
    tie._compute_rotation(rotvec)
    # new converged orientation (as done by set_start)
    tie._Q_base = Rotation.from_rotvec([0.3, -0.2, 0.1])
    res = tie._compute_rotation(rotvec)
    _assert_same(res, _fresh(tie, rotvec))
    # new angles at base only
    tie._compute_rotation(rotvec)
    tie._angles_at_base = np.array([0.05, 0.0, 0.0])
    res = tie._compute_rotation(rotvec)
    _assert_same(res, _fresh(tie, rotvec))


def test_cache_invalidated_by_rotation_mode(tie):
    rotvec = np.array([0.2, 0.1, -0.1])
    tie._angles_at_base = np.array([0.1, 0.0, 0.0])
    res_quat = tie._compute_rotation(rotvec)
    tie.use_quaternion = False
    res_total = tie._compute_rotation(rotvec)
    assert not np.array_equal(res_quat[0], res_total[0])
    _assert_same(res_total, _fresh(tie, rotvec))


def test_cache_reused_in_total_rotation_mode():
    tie = fd.constraint.RigidTie(np.arange(4), use_quaternion=False)
    rotvec = np.array([0.3, 0.1, -0.2])
    res1 = tie._compute_rotation(rotvec)
    res2 = tie._compute_rotation(rotvec.copy())
    assert all(a is b for a, b in zip(res1, res2))
    _assert_same(res1, _fresh(tie, rotvec))


def test_non_finite_rotvec_gives_the_converged_orientation(tie):
    tie._Q_base = Rotation.from_rotvec([0.3, -0.2, 0.1])
    rotvec = np.array([np.nan, 0.0, 0.0])
    res = tie._compute_rotation(rotvec)
    assert np.array_equal(res[0], tie._Q_base.as_matrix())
    _assert_same(tie._compute_rotation(rotvec), _fresh(tie, rotvec))
