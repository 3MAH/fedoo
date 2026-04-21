"""Finite-difference validation of RigidTie rotation derivatives."""

import numpy as np
import pytest

from fedoo.constraint.rigid_tie import RigidTie


def _R(omega):
    """Rotation matrix via the tie's own forward pass (non-quaternion path)."""
    rt = RigidTie([0, 1], use_quaternion=False)
    R, _, _, _ = rt._compute_rotation(np.asarray(omega, dtype=float))
    return R


def _dR_analytic(omega):
    dR = RigidTie._dR_drotvec(np.asarray(omega, dtype=float))
    return np.stack(dR, axis=0)


def _dR_fd(omega, h=1e-6):
    omega = np.asarray(omega, dtype=float)
    out = np.empty((3, 3, 3))
    for k in range(3):
        eps = np.zeros(3)
        eps[k] = h
        out[k] = (_R(omega + eps) - _R(omega - eps)) / (2 * h)
    return out


@pytest.mark.parametrize(
    "omega",
    [
        np.zeros(3),
        np.array([1e-12, 0.0, 0.0]),
        np.array([0.1, 0.0, 0.0]),
        np.array([0.0, 0.2, 0.0]),
        np.array([0.0, 0.0, 0.3]),
        np.array([0.3, -0.4, 0.5]),
        np.array([np.pi / 4, np.pi / 6, -np.pi / 8]),
    ],
)
def test_dR_drotvec_matches_finite_difference(omega):
    analytic = _dR_analytic(omega)
    numeric = _dR_fd(omega)
    # FD error ~ O(h^2) with h=1e-6 → ~1e-8 in smooth regime
    assert np.allclose(analytic, numeric, atol=1e-6)


def test_dR_drotvec_zero_limit_is_skew_basis():
    dR = RigidTie._dR_drotvec(np.zeros(3))
    expected = (
        np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]]),
        np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]]),
        np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]]),
    )
    for got, exp in zip(dR, expected):
        assert np.array_equal(got, exp)


def test_compute_rotation_identity_at_zero():
    rt = RigidTie([0, 1], use_quaternion=False)
    R, _, _, _ = rt._compute_rotation(np.zeros(3))
    assert np.allclose(R, np.eye(3))


def test_compute_rotation_is_orthogonal():
    rt = RigidTie([0, 1], use_quaternion=False)
    for omega in [
        np.array([0.3, -0.4, 0.5]),
        np.array([1.2, 0.0, 0.0]),
        np.array([0.0, 2.5, -1.5]),
    ]:
        R, _, _, _ = rt._compute_rotation(omega)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-12)


if __name__ == "__main__":
    pytest.main([__file__])
