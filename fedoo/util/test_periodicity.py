"""Periodicity test and face matching utilities.

The :func:`match_opposing_faces` helper pairs nodes on the two faces
normal to a given axis using a KDTree on the perpendicular coordinates.
``tol`` is the maximum accepted distance between paired nodes — it is
used both to gather nodes considered "on the boundary plane" and to
validate the pairing distance afterwards. Pairing failures raise
``ValueError`` with a precise message.

:func:`is_periodic` is a thin wrapper around the helper that returns
``True``/``False`` for the full mesh.
"""

import numpy as np
from scipy.spatial import cKDTree


def pair_node_sets(crd, idx_minus, idx_plus, free_axes, tol):
    """Reorder ``idx_plus`` so that ``crd[idx_minus][i]`` and
    ``crd[idx_plus][i]`` are paired by Euclidean distance on
    ``free_axes``.

    Parameters
    ----------
    crd : np.ndarray, shape (n_nodes, ndim)
        Node coordinates.
    idx_minus, idx_plus : np.ndarray
        Node-index arrays for the two opposing sets.
    free_axes : sequence of int
        Axes along which the two sets share coordinates (perpendicular
        to the matching direction). Empty for 1D case (single-node
        opposing sets).
    tol : float
        Maximum accepted distance between paired nodes.

    Returns
    -------
    idx_minus, idx_plus : np.ndarray
        Same nodes; ``idx_plus`` reordered to match ``idx_minus``.

    Raises
    ------
    ValueError
        If the two sets have different sizes, or if any pairing
        distance exceeds ``tol``.
    """
    if len(idx_minus) != len(idx_plus):
        raise ValueError(
            f"opposing node sets have {len(idx_minus)} vs "
            f"{len(idx_plus)} nodes; cannot pair."
        )
    if len(idx_minus) == 0 or not list(free_axes):
        return idx_minus, idx_plus

    free_axes = list(free_axes)
    tree = cKDTree(crd[idx_plus][:, free_axes])
    dist, order = tree.query(crd[idx_minus][:, free_axes])
    if dist.max() > tol:
        raise ValueError(
            f"max pairing distance {dist.max():.3e} exceeds "
            f"tol={tol:.3e}; nodes on opposite sets do not align."
        )
    return idx_minus, idx_plus[order]


def match_opposing_faces(crd, axis, tol):
    """Pair nodes on the two faces normal to ``axis``.

    Parameters
    ----------
    crd : np.ndarray, shape (n_nodes, ndim)
        Node coordinates.
    axis : int
        Axis (0, 1, 2) along which to match opposing faces.
    tol : float
        Maximum accepted distance between paired nodes (also used to
        identify nodes lying on the boundary plane). Must be smaller
        than the mesh element size.

    Returns
    -------
    idx_minus, idx_plus : np.ndarray
        Node index arrays such that ``crd[idx_minus][i]`` and
        ``crd[idx_plus][i]`` are paired with perpendicular distance
        ``≤ tol`` for every ``i``.

    Raises
    ------
    ValueError
        If the two faces have different node counts, or if any pairing
        distance exceeds ``tol``.
    """
    bmin = crd[:, axis].min()
    bmax = crd[:, axis].max()

    minus = np.where(np.abs(crd[:, axis] - bmin) < tol)[0]
    plus = np.where(np.abs(crd[:, axis] - bmax) < tol)[0]

    free_axes = [i for i in range(crd.shape[1]) if i != axis]
    try:
        return pair_node_sets(crd, minus, plus, free_axes, tol)
    except ValueError as e:
        raise ValueError(
            f"axis {axis}: faces at {bmin:.4g} and {bmax:.4g}: {e}"
        ) from None


def is_periodic(crd, tol=1e-8, dim=3):
    """Test if node coordinates form a periodic mesh.

    Pairs nodes on opposite faces (axis 0; also 1 and 2 if ``dim``
    allows) using :func:`match_opposing_faces`. Returns ``True`` if all
    requested axes pair successfully within ``tol``.

    Parameters
    ----------
    crd : np.ndarray, shape (n_nodes, ndim)
        Node coordinates.
    tol : float, default 1e-8
        Maximum accepted distance between paired nodes. Must be smaller
        than the mesh element size; otherwise interior nodes will be
        captured into face sets and pairing will fail.
    dim : int in {1, 2, 3}, default 3
        Number of axes along which to test periodicity.

    Returns
    -------
    bool
        ``True`` if the mesh is periodic at this tolerance, ``False``
        otherwise.
    """
    for axis in range(dim):
        try:
            match_opposing_faces(crd, axis, tol)
        except ValueError:
            return False
    return True
