import numpy as np


def is_periodic(crd, tol=1e-8, dim=3):
    """
    Test if a list of node coordinates is periodic (have nodes at the same positions on adjacent faces)

    Parameters
    ----------
    crd: numpy array with shape = [n_nodes, ndim]
        list of node coordinates associated to a mesh
    tol : float (default = 1e-8)
        Tolerance used to test the nodes positions.
    dim : 1,2 or 3 (default = 3)
        Dimension of the periodicity. If dim = 1, the periodicity is tested only over the 1st axis (x axis).
        if dim = 2, the periodicity is tested on the 2 first axis (x and y axis).
        if dim = 3, the periodicity is tested in 3 directions (x,y,z).

    Returns
    -------
    True if the mesh is periodic else return False.
    """

    # bounding box
    xmax = np.max(crd[:, 0])
    xmin = np.min(crd[:, 0])
    ymax = np.max(crd[:, 1])
    ymin = np.min(crd[:, 1])
    if dim == 3:
        zmax = np.max(crd[:, 2])
        zmin = np.min(crd[:, 2])

    # extract face nodes
    left = np.where(np.abs(crd[:, 0] - xmin) < tol)[0]
    right = np.where(np.abs(crd[:, 0] - xmax) < tol)[0]

    if dim > 1:
        bottom = np.where(np.abs(crd[:, 1] - ymin) < tol)[0]
        top = np.where(np.abs(crd[:, 1] - ymax) < tol)[0]

    if dim > 2:  # or dim == 3
        back = np.where(np.abs(crd[:, 2] - zmin) < tol)[0]
        front = np.where(np.abs(crd[:, 2] - zmax) < tol)[0]

    # sort adjacent faces to ensure node correspondance
    if crd.shape[1] == 2:  # 2D mesh
        left = left[np.argsort(crd[left, 1])]
        right = right[np.argsort(crd[right, 1])]
        if dim > 1:
            bottom = bottom[np.argsort(crd[bottom, 0])]
            top = top[np.argsort(crd[top, 0])]

    elif crd.shape[1] > 2:
        decimal_round = int(-np.log10(tol) - 1)
        left = left[np.lexsort((crd[left, 1], crd[left, 2].round(decimal_round)))]
        right = right[np.lexsort((crd[right, 1], crd[right, 2].round(decimal_round)))]
        if dim > 1:
            bottom = bottom[
                np.lexsort((crd[bottom, 0], crd[bottom, 2].round(decimal_round)))
            ]
            top = top[np.lexsort((crd[top, 0], crd[top, 2].round(decimal_round)))]
        if dim > 2:
            back = back[np.lexsort((crd[back, 0], crd[back, 1].round(decimal_round)))]
            front = front[
                np.lexsort((crd[front, 0], crd[front, 1].round(decimal_round)))
            ]

    # ==========================
    # test if mesh is periodic:
    # ==========================

    # test if same number of nodes in adjacent faces
    if len(left) != len(right):
        return False
    if dim > 1 and len(bottom) != len(top):
        return False
    if dim > 2 and (len(back) != len(front)):
        return False

    # check nodes position
    if (crd[right, 1:] - crd[left, 1:] > tol).any():
        return False
    if dim > 1 and (crd[top, ::2] - crd[bottom, ::2] > tol).any():
        return False
    if dim > 2 and (crd[front, :2] - crd[back, :2] > tol).any():
        return False

    return True
