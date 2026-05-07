"""Tests for fedoo.util.recovery (nodal gradient and Hessian recovery)."""

import numpy as np
import pytest

import fedoo as fd


def _interior_mask(mesh, margin=0.15):
    """Boolean mask of nodes that are at least `margin` (relative bbox)
    away from the bounding box, i.e. away from the mesh boundary where the
    lumped GP-to-Node averaging is one-sided.
    """
    crd = mesh.nodes
    lo = crd.min(axis=0)
    hi = crd.max(axis=0)
    span = hi - lo
    eps = margin * span
    return np.all((crd > lo + eps) & (crd < hi - eps), axis=1)


def _eval_field(mesh, f):
    return f(*mesh.nodes.T)


@pytest.mark.parametrize(
    "elm_type, nx, ny",
    [("quad4", 21, 21), ("tri3", 21, 21), ("tri6", 21, 21)],
)
def test_recover_hessian_2d_quadratic(elm_type, nx, ny):
    """Quadratic 2D field: H is constant; recovery must hit it on interior nodes."""
    mesh = fd.mesh.rectangle_mesh(nx=nx, ny=ny, elm_type=elm_type)

    # f(x,y) = x^2 + x*y + 2*y^2  =>  H = [[2, 1], [1, 4]]
    field = _eval_field(mesh, lambda x, y: x * x + x * y + 2 * y * y)
    H = fd.recover_hessian(mesh, field)

    assert H.shape == (mesh.n_nodes, 2, 2)
    # Symmetric by construction (post-symmetrize step).
    np.testing.assert_allclose(H[:, 0, 1], H[:, 1, 0], atol=1e-12)

    H_expected = np.array([[2.0, 1.0], [1.0, 4.0]])
    interior = _interior_mask(mesh)
    np.testing.assert_allclose(
        H[interior],
        np.broadcast_to(H_expected, (interior.sum(), 2, 2)),
        atol=1e-9,
    )


def test_recover_hessian_2d_linear_field_is_zero():
    """Linear field has zero Hessian everywhere (boundary too)."""
    mesh = fd.mesh.rectangle_mesh(nx=11, ny=11, elm_type="tri3")
    field = _eval_field(mesh, lambda x, y: 2.0 * x + 3.0 * y - 0.5)
    H = fd.recover_hessian(mesh, field)
    np.testing.assert_allclose(H, 0.0, atol=1e-10)


def test_recover_hessian_3d_quadratic_hex8():
    """Quadratic 3D field on hex8: H constant on interior nodes."""
    mesh = fd.mesh.box_mesh(nx=11, ny=11, nz=11, elm_type="hex8")
    # f(x,y,z) = x^2 + 2*y^2 + 3*z^2 + x*y
    # H = [[2, 1, 0], [1, 4, 0], [0, 0, 6]]
    field = _eval_field(mesh, lambda x, y, z: x * x + 2 * y * y + 3 * z * z + x * y)
    H = fd.recover_hessian(mesh, field)

    assert H.shape == (mesh.n_nodes, 3, 3)
    np.testing.assert_allclose(H, H.swapaxes(-1, -2), atol=1e-12)

    H_expected = np.array([[2.0, 1.0, 0.0], [1.0, 4.0, 0.0], [0.0, 0.0, 6.0]])
    interior = _interior_mask(mesh)
    np.testing.assert_allclose(
        H[interior],
        np.broadcast_to(H_expected, (interior.sum(), 3, 3)),
        atol=1e-9,
    )


def test_recover_gradient_2d():
    """Gradient of f(x,y) = 3x + 2y is the constant vector [3, 2]."""
    mesh = fd.mesh.rectangle_mesh(nx=11, ny=11, elm_type="quad4")
    field = _eval_field(mesh, lambda x, y: 3.0 * x + 2.0 * y)
    g = fd.recover_gradient(mesh, field)

    assert g.shape == (mesh.n_nodes, 2)
    np.testing.assert_allclose(g, np.broadcast_to([3.0, 2.0], g.shape), atol=1e-10)


def test_to_upper_diagonal_3d_round_trip():
    """to_upper_diagonal must recover the (n, 6) packing in mmg's order."""
    rng = np.random.default_rng(0)
    n = 7
    H = rng.normal(size=(n, 3, 3))
    H = 0.5 * (H + H.swapaxes(-1, -2))

    packed = fd.to_upper_diagonal(H)
    assert packed.shape == (n, 6)
    # Order [m11, m12, m13, m22, m23, m33].
    np.testing.assert_array_equal(packed[:, 0], H[:, 0, 0])
    np.testing.assert_array_equal(packed[:, 1], H[:, 0, 1])
    np.testing.assert_array_equal(packed[:, 2], H[:, 0, 2])
    np.testing.assert_array_equal(packed[:, 3], H[:, 1, 1])
    np.testing.assert_array_equal(packed[:, 4], H[:, 1, 2])
    np.testing.assert_array_equal(packed[:, 5], H[:, 2, 2])


def test_to_upper_diagonal_2d():
    rng = np.random.default_rng(1)
    n = 5
    H = rng.normal(size=(n, 2, 2))
    H = 0.5 * (H + H.swapaxes(-1, -2))

    packed = fd.to_upper_diagonal(H)
    assert packed.shape == (n, 3)
    np.testing.assert_array_equal(packed[:, 0], H[:, 0, 0])
    np.testing.assert_array_equal(packed[:, 1], H[:, 0, 1])
    np.testing.assert_array_equal(packed[:, 2], H[:, 1, 1])


def test_to_voigt_3d():
    """to_voigt must produce (6, n) in fedoo Voigt order [XX,YY,ZZ,XY,XZ,YZ]."""
    rng = np.random.default_rng(2)
    n = 4
    H = rng.normal(size=(n, 3, 3))
    H = 0.5 * (H + H.swapaxes(-1, -2))

    voigt = fd.to_voigt(H)
    assert voigt.shape == (6, n)
    np.testing.assert_array_equal(voigt[0], H[:, 0, 0])
    np.testing.assert_array_equal(voigt[1], H[:, 1, 1])
    np.testing.assert_array_equal(voigt[2], H[:, 2, 2])
    np.testing.assert_array_equal(voigt[3], H[:, 0, 1])
    np.testing.assert_array_equal(voigt[4], H[:, 0, 2])
    np.testing.assert_array_equal(voigt[5], H[:, 1, 2])


def test_recover_hessian_validates_shape():
    mesh = fd.mesh.rectangle_mesh(nx=5, ny=5, elm_type="quad4")
    with pytest.raises(ValueError, match="shape"):
        fd.recover_hessian(mesh, np.zeros(mesh.n_nodes - 1))
