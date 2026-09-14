"""Gauss-point to node extrapolation must reproduce constant (and linear)
fields for every element type and every number of Gauss points, including
reduced integration where there are fewer Gauss points than nodes."""

import numpy as np
import pytest

import fedoo as fd
from fedoo.lib_elements.element_base import gausspoint_extrapolation_matrix
from fedoo.lib_elements.element_list import get_element

CASES = [
    ("lin2", 1),
    ("lin2", 2),
    ("lin3", 1),
    ("lin3", 2),
    ("lin3", 3),
    ("tri3", 1),
    ("tri3", 3),
    ("tri6", 1),
    ("tri6", 3),
    ("tri6", 4),
    ("tri6", 7),
    ("quad4", 1),
    ("quad4", 4),
    ("quad8", 1),
    ("quad8", 4),
    ("quad8", 9),
    ("quad9", 4),
    ("quad9", 9),
    ("tet4", 1),
    ("tet4", 4),
    ("tet10", 1),
    ("tet10", 4),
    ("tet10", 5),
    ("tet10", 15),
    ("hex8", 1),
    ("hex8", 8),
    ("hex20", 1),
    ("hex20", 8),
    ("hex20", 27),
    ("wed6", 1),
    ("wed6", 6),
    ("wed15", 6),
    ("wed15", 8),
    ("wed15", 21),
    ("wed18", 6),
    ("wed18", 8),
    ("wed18", 21),
]


def _element(elm_type, n_gp):
    elm = get_element(elm_type)
    if hasattr(elm, "geometry_elm"):
        elm = elm.geometry_elm
    return elm(n_gp)


@pytest.mark.parametrize("elm_type, n_gp", CASES)
def test_extrapolation_reproduces_constant_field(elm_type, n_gp):
    elm = _element(elm_type, n_gp)
    E = gausspoint_extrapolation_matrix(elm.xi_pg, elm.xi_nd, elm.shape_function_gp)
    assert E.shape == (elm.n_nodes, n_gp)
    np.testing.assert_allclose(E.sum(axis=1), 1.0, atol=1e-10)


@pytest.mark.parametrize("elm_type, n_gp", CASES)
def test_extrapolation_reproduces_linear_field(elm_type, n_gp):
    elm = _element(elm_type, n_gp)
    xi_gp = np.asarray(elm.xi_pg).reshape(n_gp, -1)
    xi_nd = np.asarray(elm.xi_nd).reshape(elm.n_nodes, -1)
    ndim = xi_gp.shape[1]
    if n_gp < ndim + 1:
        pytest.skip("quadrature cannot resolve a linear field")
    coef = np.arange(1, ndim + 1, dtype=float)
    E = gausspoint_extrapolation_matrix(xi_gp, xi_nd, elm.shape_function_gp)
    np.testing.assert_allclose(E @ (xi_gp @ coef), xi_nd @ coef, atol=1e-10)


def test_extrapolation_least_squares_when_enough_gauss_points():
    """With at least as many Gauss points as nodes the historical
    pseudo-inverse is kept unchanged."""
    elm = _element("tet10", 15)
    E = gausspoint_extrapolation_matrix(elm.xi_pg, elm.xi_nd, elm.shape_function_gp)
    np.testing.assert_allclose(E, np.linalg.pinv(elm.shape_function_gp))


def test_tet10_four_gauss_points_is_linear_subelement():
    """Hinton & Campbell: 4 Gauss points of a tet10 span a linear basis."""
    elm = _element("tet10", 4)
    E = gausspoint_extrapolation_matrix(elm.xi_pg, elm.xi_nd, elm.shape_function_gp)
    # the historical pseudo-inverse gave -0.2 at corners and 0.8 at mid-edges
    bad = np.linalg.pinv(elm.shape_function_gp).sum(axis=1)
    assert bad.min() < 0 and bad.max() < 1
    # exact for any linear field, mid-edge value = mean of the corner values
    xi_gp = elm.xi_pg
    f_gp = 1.0 + 2 * xi_gp[:, 0] - 3 * xi_gp[:, 1] + 0.5 * xi_gp[:, 2]
    xi_nd = elm.xi_nd
    f_nd = 1.0 + 2 * xi_nd[:, 0] - 3 * xi_nd[:, 1] + 0.5 * xi_nd[:, 2]
    np.testing.assert_allclose(E @ f_gp, f_nd, atol=1e-12)


@pytest.mark.parametrize(
    "mesh_fn, elm_type, n_gp",
    [
        (lambda: fd.mesh.box_mesh(2, 2, 2, elm_type="hex20"), "hex20", 8),
        (lambda: fd.mesh.rectangle_mesh(3, 3, elm_type="tri6"), "tri6", 3),
        (lambda: fd.mesh.rectangle_mesh(3, 3, elm_type="quad8"), "quad8", 4),
    ],
)
def test_mesh_convert_data_reduced_integration_constant_field(mesh_fn, elm_type, n_gp):
    mesh = mesh_fn()
    assert mesh.elm_type == elm_type
    field_gp = np.full(mesh.n_elements * n_gp, 3.5)
    field_nd = mesh.convert_data(
        field_gp, "GaussPoint", "Node", n_elm_gp=n_gp, method="mean"
    )
    np.testing.assert_allclose(field_nd, 3.5, atol=1e-10)
