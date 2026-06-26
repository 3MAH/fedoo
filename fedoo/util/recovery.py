"""Vectorized FE-aware nodal gradient and Hessian recovery on fedoo meshes.

Useful as a fast input for metric-based adaptive remeshing (e.g.
``mmgpy.metrics.create_metric_from_hessian``). The recovery uses a double
projection: per-element gradient at Gauss points -> nodal values via
``Mesh.convert_data`` -> repeat for the Hessian. All work is a few einsums plus
a handful of sparse matvecs; no per-vertex Python loop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from fedoo.core.mesh import Mesh


def _physical_shape_derivatives(mesh: Mesh, n_elm_gp: int | None = None):
    """Return (dNdx, n_gp) for the requested integration order.

    dNdx[el, gp, n, i] is dN_n/dx_i at gauss point gp of element el.
    """
    if n_elm_gp is None:
        from fedoo.lib_elements.element_list import get_default_n_gp

        n_elm_gp = get_default_n_gp(mesh.elm_type)

    # Triggers init_interpolation + compute_jacobian_with_inverse, populating
    # inv_jacobian_matrix on the cached element interpolation object.
    mesh._get_gaussian_quadrature_mat(n_elm_gp)

    elm = mesh._elm_interpolation[n_elm_gp]
    inv_J = elm.inv_jacobian_matrix  # (n_el, n_gp, ndim_phys, ndim_xi)

    if inv_J.shape[-1] != inv_J.shape[-2]:
        raise NotImplementedError(
            "recover_gradient/recover_hessian require a square Jacobian "
            "(volumetric or planar elements). Got inv_J shape "
            f"{inv_J.shape} on element type '{mesh.elm_type}'."
        )

    # Reuse the per-element-class cache populated in __init__
    # (e.g. ElementTriangle.__init__ at fedoo/lib_elements/triangle.py:15).
    # dN_xi[g, x, n] = dN_n/dxi_x at gp g
    dN_xi = np.stack(elm.shape_function_derivative_gp, axis=0)

    # dNdx[e, g, n, i] = sum_x dN_xi[g, x, n] * inv_J[e, g, i, x]
    dNdx = np.einsum("gxn, egix -> egni", dN_xi, inv_J)

    return dNdx, n_elm_gp


def _validate_scalar_field(mesh: Mesh, field: np.ndarray) -> np.ndarray:
    field = np.asarray(field, dtype=np.float64)
    if field.shape != (mesh.n_nodes,):
        raise ValueError(
            f"field must have shape (n_nodes={mesh.n_nodes},), got {field.shape}"
        )
    return field


def recover_gradient(
    mesh: Mesh,
    field: np.ndarray,
    n_elm_gp: int | None = None,
    method: str | None = "l2",
) -> np.ndarray:
    """Recover the nodal gradient of a scalar field on a fedoo mesh.

    Computes the gradient at every Gauss point via the FE shape function
    derivatives, then converts back to nodes with ``mesh.convert_data``.

    Restricted to volumetric or planar elements (square Jacobian); raises
    ``NotImplementedError`` for shell, beam, or surface-in-3D elements.
    Boundary nodes use one-sided element averaging and are therefore less
    accurate than interior nodes.

    Parameters
    ----------
    mesh : fedoo.Mesh
        The mesh on which the field is defined.
    field : (n_nodes,) ndarray
        Scalar nodal values.
    n_elm_gp : int, optional
        Number of Gauss points per element. Defaults to the element default.
    method : {'mean', 'l2'}, default = 'l2'
        GaussPoint-to-Node conversion method passed to ``mesh.convert_data``.

    Returns
    -------
    (n_nodes, ndim) ndarray
        Recovered nodal gradient.
    """
    field = _validate_scalar_field(mesh, field)
    dNdx, n_elm_gp = _physical_shape_derivatives(mesh, n_elm_gp)

    grad_gp = np.einsum("en, egni -> egi", field[mesh.elements], dNdx)
    grad_node = mesh.convert_data(
        grad_gp.transpose(2, 1, 0).reshape(mesh.ndim, -1),
        convert_from="GaussPoint",
        convert_to="Node",
        n_elm_gp=n_elm_gp,
        method=method,
    )
    return grad_node.T


def recover_hessian(
    mesh: Mesh,
    field: np.ndarray,
    n_elm_gp: int | None = None,
    method: str | None = "l2",
) -> np.ndarray:
    """Recover the nodal Hessian of a scalar field via double projection.

    Step 1 recovers the gradient as a continuous P1 nodal vector field.
    Step 2 takes the gradient of each component and projects back to nodes,
    giving a full (possibly non-symmetric) tensor; the result is then
    symmetrized.

    Restricted to volumetric or planar elements (square Jacobian); raises
    ``NotImplementedError`` for shell, beam, or surface-in-3D elements.
    Boundary nodes use one-sided element averaging and are therefore less
    accurate than interior nodes - acceptable when feeding an mmg metric,
    which clips eigenvalues by ``hmin``/``hmax``.

    Pack the result with :func:`to_voigt` for the fedoo convention
    (``[XX, YY, ZZ, XY, XZ, YZ]``) or :func:`to_upper_diagonal` for the
    row-major upper triangle (``[m11, m12, m13, m22, m23, m33]``), the layout
    consumed by mmg's ``MMG{2,3}D_Set_tensorSols``.

    Parameters
    ----------
    mesh : fedoo.Mesh
    field : (n_nodes,) ndarray
        Scalar nodal values.
    n_elm_gp : int, optional
        Number of Gauss points per element. Defaults to the element default.
    method : {'mean', 'l2'}, default='l2'
        GaussPoint-to-Node conversion method passed to ``mesh.convert_data``.

    Returns
    -------
    (n_nodes, ndim, ndim) ndarray
        Symmetric Hessian tensor at every node.
    """
    field = _validate_scalar_field(mesh, field)
    dNdx, n_elm_gp = _physical_shape_derivatives(mesh, n_elm_gp)

    grad_gp = np.einsum("en, egni -> egi", field[mesh.elements], dNdx)
    grad_node = mesh.convert_data(
        grad_gp.transpose(2, 1, 0).reshape(mesh.ndim, -1),
        convert_from="GaussPoint",
        convert_to="Node",
        n_elm_gp=n_elm_gp,
        method=method,
    ).T

    # hessian_gp[e, g, j, i] = sum_n grad_node[e, n, j] * dNdx[e, g, n, i]
    hessian_gp = np.einsum("enj, egni -> egji", grad_node[mesh.elements], dNdx)
    hessian = (
        mesh.convert_data(
            hessian_gp.transpose(2, 3, 1, 0).reshape(mesh.ndim * mesh.ndim, -1),
            convert_from="GaussPoint",
            convert_to="Node",
            n_elm_gp=n_elm_gp,
            method=method,
        )
        .reshape(mesh.ndim, mesh.ndim, mesh.n_nodes)
        .transpose(2, 0, 1)
    )

    return 0.5 * (hessian + hessian.swapaxes(-1, -2))


def to_voigt(T: np.ndarray) -> np.ndarray:
    """Pack a symmetric tensor field to fedoo Voigt order.

    3D ``(n, 3, 3)`` -> ``(6, n)`` in order ``[XX, YY, ZZ, XY, XZ, YZ]``.
    2D ``(n, 2, 2)`` -> ``(3, n)`` in order ``[XX, YY, XY]``.

    Matches the storage convention of
    :class:`fedoo.util.voigt_tensors._SymmetricTensorList`.
    """
    T = np.asarray(T)
    if T.ndim != 3 or T.shape[-1] != T.shape[-2]:
        raise ValueError(f"expected (n, d, d) tensor field, got {T.shape}")
    d = T.shape[-1]
    if d == 3:
        return np.stack(
            [T[:, 0, 0], T[:, 1, 1], T[:, 2, 2], T[:, 0, 1], T[:, 0, 2], T[:, 1, 2]],
            axis=0,
        )
    if d == 2:
        return np.stack([T[:, 0, 0], T[:, 1, 1], T[:, 0, 1]], axis=0)
    raise ValueError(f"unsupported tensor dimension {d}; expected 2 or 3")


def to_upper_diagonal(T: np.ndarray) -> np.ndarray:
    """Pack a symmetric tensor field to its row-major upper triangle.

    3D ``(n, 3, 3)`` -> ``(n, 6)`` in order ``[m11, m12, m13, m22, m23, m33]``.
    2D ``(n, 2, 2)`` -> ``(n, 3)`` in order ``[m11, m12, m22]``.

    This is the layout consumed by mmg's ``MMG3D_Set_tensorSols`` /
    ``MMG2D_Set_tensorSols`` (and therefore by mmgpy), but the helper is
    consumer-agnostic and applies to any symmetric tensor field.
    """
    T = np.asarray(T)
    if T.ndim != 3 or T.shape[-1] != T.shape[-2]:
        raise ValueError(f"expected (n, d, d) tensor field, got {T.shape}")
    d = T.shape[-1]
    if d == 3:
        return np.stack(
            [T[:, 0, 0], T[:, 0, 1], T[:, 0, 2], T[:, 1, 1], T[:, 1, 2], T[:, 2, 2]],
            axis=1,
        )
    if d == 2:
        return np.stack([T[:, 0, 0], T[:, 0, 1], T[:, 1, 1]], axis=1)
    raise ValueError(f"unsupported tensor dimension {d}; expected 2 or 3")
