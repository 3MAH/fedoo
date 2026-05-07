"""Vectorized FE-aware nodal gradient and Hessian recovery on fedoo meshes.

Useful as a fast input for metric-based adaptive remeshing (e.g.
``mmgpy.metrics.create_metric_from_hessian``). The recovery uses a double
Galerkin L2-projection: per-element gradient at Gauss points -> lumped
GP-to-Node averaging via fedoo's cached projection matrix -> repeat for the
Hessian. All work is a few einsums plus a handful of sparse matvecs; no
per-vertex Python loop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from fedoo.core.mesh import Mesh


def _physical_shape_derivatives(mesh: Mesh, n_elm_gp: int | None = None):
    """Return (dNdx, proj, n_gp) for the requested integration order.

    dNdx[el, gp, n, i] is dN_n/dx_i at gauss point gp of element el.
    proj is the cached sparse GP-to-Node lumped averaging matrix.
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

    # shape_function_derivative returns a list of length n_gp, each (ndim_xi, n_nodes_per_elm)
    dN_xi = np.stack(elm.shape_function_derivative(elm.xi_pg), axis=0)
    # dN_xi[g, x, n] = dN_n/dxi_x at gp g

    # dNdx[e, g, n, i] = sum_x dN_xi[g, x, n] * inv_J[e, g, i, x]
    dNdx = np.einsum("gxn, egix -> egni", dN_xi, inv_J, optimize=True)

    proj = mesh._get_gausspoint2node_mat(n_elm_gp)  # sparse (n_nodes, n_el * n_gp)
    return dNdx, proj, n_elm_gp


def _project_gp_field(field_gp: np.ndarray, proj) -> np.ndarray:
    """Project a GP field of shape (n_el, n_gp, ...) to nodes via proj.

    Returns a node field with leading shape (n_nodes, ...).
    """
    n_el, n_gp = field_gp.shape[:2]
    tail = field_gp.shape[2:]
    # GP layout in proj: column index = el + gp*n_el (mesh.py:1138-1142),
    # i.e. elements vary fastest. Transposing axes 0 and 1 then ravelling
    # in C order produces exactly that layout.
    flat = field_gp.transpose(1, 0, *range(2, field_gp.ndim)).reshape(n_el * n_gp, -1)
    out = proj @ flat  # (n_nodes, prod(tail))
    return out.reshape(out.shape[0], *tail)


def recover_gradient(
    mesh: Mesh,
    field: np.ndarray,
    n_elm_gp: int | None = None,
) -> np.ndarray:
    """Recover the nodal gradient of a scalar field on a fedoo mesh.

    Computes the gradient at every Gauss point via the FE shape function
    derivatives, then averages back to nodes through fedoo's cached lumped
    GP-to-Node projection (``mesh._get_gausspoint2node_mat``).

    Parameters
    ----------
    mesh : fedoo.Mesh
        The mesh on which the field is defined.
    field : (n_nodes,) ndarray
        Scalar nodal values.
    n_elm_gp : int, optional
        Number of Gauss points per element. Defaults to the element default.

    Returns
    -------
    (n_nodes, ndim) ndarray
        Recovered nodal gradient.
    """
    field = np.asarray(field, dtype=np.float64)
    if field.shape != (mesh.n_nodes,):
        raise ValueError(
            f"field must have shape (n_nodes={mesh.n_nodes},), got {field.shape}"
        )

    dNdx, proj, _ = _physical_shape_derivatives(mesh, n_elm_gp)
    f_elem = field[mesh.elements]  # (n_el, n_elm_nd)
    g_gp = np.einsum("en, egni -> egi", f_elem, dNdx, optimize=True)
    return _project_gp_field(g_gp, proj)


def recover_hessian(
    mesh: Mesh,
    field: np.ndarray,
    n_elm_gp: int | None = None,
) -> np.ndarray:
    """Recover the nodal Hessian of a scalar field via double L2-projection.

    Step 1 recovers the gradient as a continuous P1 nodal vector field.
    Step 2 takes the gradient of each component and projects back to nodes,
    giving a full (possibly non-symmetric) tensor; the result is then
    symmetrized.

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

    Returns
    -------
    (n_nodes, ndim, ndim) ndarray
        Symmetric Hessian tensor at every node.
    """
    field = np.asarray(field, dtype=np.float64)
    if field.shape != (mesh.n_nodes,):
        raise ValueError(
            f"field must have shape (n_nodes={mesh.n_nodes},), got {field.shape}"
        )

    dNdx, proj, _ = _physical_shape_derivatives(mesh, n_elm_gp)

    f_elem = field[mesh.elements]
    g_gp = np.einsum("en, egni -> egi", f_elem, dNdx, optimize=True)
    g_node = _project_gp_field(g_gp, proj)  # (n_nodes, ndim)

    # Step 2: gradient of g, component by component, all in one einsum.
    g_elem = g_node[mesh.elements]  # (n_el, n_elm_nd, ndim)
    # H_gp[e, g, j, i] = sum_n g_elem[e, n, j] * dNdx[e, g, n, i]
    H_gp = np.einsum("enj, egni -> egji", g_elem, dNdx, optimize=True)
    H = _project_gp_field(H_gp, proj)  # (n_nodes, ndim, ndim)

    return 0.5 * (H + H.swapaxes(-1, -2))


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
