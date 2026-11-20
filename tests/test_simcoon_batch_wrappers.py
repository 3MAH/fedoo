"""fedoo.core._simcoon kinematics and rotation wrappers (simcoon 2.6 batch calls)
against numpy / scipy references."""

import numpy as np
from scipy.spatial.transform import Rotation

from fedoo.core import _simcoon


def test_deformation_gradient():
    rng = np.random.default_rng(0)
    grad = 0.2 * rng.standard_normal((3, 3, 150))
    as_lists = [[grad[i, j].copy() for j in range(3)] for i in range(3)]
    F, J = _simcoon.deformation_gradient(as_lists)
    F_ref = grad + np.eye(3)[:, :, None]
    assert F.shape == (3, 3, 150) and F.flags.f_contiguous and J.shape == (150,)
    np.testing.assert_array_equal(F, F_ref)
    np.testing.assert_allclose(
        J, np.linalg.det(F_ref.transpose(2, 0, 1)), rtol=1e-13, atol=1e-15
    )


def test_rotation_conversions():
    rng = np.random.default_rng(1)
    v = rng.standard_normal((150, 3))
    v *= (np.pi * rng.random(150) / np.linalg.norm(v, axis=1))[:, None]
    R = _simcoon.rotvec_to_matrix(v)
    assert R.shape == (150, 3, 3)
    np.testing.assert_allclose(R, Rotation.from_rotvec(v).as_matrix(), atol=1e-14)
    w = _simcoon.matrix_to_rotvec(R)
    assert w.shape == (150, 3)
    np.testing.assert_allclose(w, v, atol=1e-10)
    # the transposed views of the call sites (rotation vectors stored (3, N))
    np.testing.assert_allclose(_simcoon.rotvec_to_matrix(v.T.copy().T), R, atol=0)


def test_dexp_inv():
    rng = np.random.default_rng(2)
    v = rng.standard_normal((4, 25, 3))  # arbitrary leading dimensions
    v[0, 0] = 1e-9
    T_inv = _simcoon.dexp_inv(v)
    assert T_inv.shape == (4, 25, 3, 3)
    # T^-1 maps the spin of exp to the rotation vector increment: T^-1 w = w on the axis
    np.testing.assert_allclose(np.einsum("...ij,...j->...i", T_inv, v), v, atol=1e-14)
    angle = np.linalg.norm(v, axis=-1)[..., None, None]
    x, y, z = np.moveaxis(v, -1, 0)
    zero = np.zeros_like(x)
    s = np.stack([zero, -z, y, z, zero, -x, -y, x, zero], axis=-1).reshape(
        v.shape + (3,)
    )
    # first-order expansion I - s/2 holds to O(angle^2)
    assert np.all(np.abs(T_inv - (np.eye(3) - 0.5 * s)) <= angle**2)
