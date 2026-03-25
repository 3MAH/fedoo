from fedoo.core.mesh import Mesh
from simcoon import Rotation

import numpy as np


class LocalFrame(np.ndarray):
    """Array of local coordinate frames.

    A LocalFrame object is an (N, 3, 3) shaped array in 3D or (N, 2, 2) in 2D,
    where N is the number of points (nodes, gauss points, elements, ...).

    If ``LF`` is a LocalFrame object:
        - ``LF[i]`` gives the local frame at the i-th point.
        - ``LF[i][0]``, ``LF[i][1]`` and ``LF[i][2]`` give the 3 unit
          orthogonal vectors defining the local frame.
    """

    def __new__(cls, localFrame):
        return np.asarray(localFrame).view(cls)

    def Rotate(self, angle, axis="Z"):
        """Rotate all local frames by a given angle around a global axis.

        Parameters
        ----------
        angle : float
            Rotation angle in degrees.
        axis : str, optional
            Axis of rotation: 'X', 'Y' or 'Z' (default 'Z').

        Returns
        -------
        self
        """
        if angle != 0:
            rot = Rotation.from_euler(axis.upper(), angle, degrees=True)
            self[:] = np.matmul(self, rot.as_matrix().T)
            return self

    def as_rotation(self):
        """Convert local frames to a batch simcoon.Rotation object.

        Returns
        -------
        simcoon.Rotation
            Batch rotation from local to global frame.
        """
        return Rotation.from_matrix(np.asarray(self))

    def __getitem__(self, index):
        new = super(LocalFrame, self).__getitem__(index)
        if new.ndim == 3 and new.shape[1] in [2, 3] and new.shape[2] in [2, 3]:
            return new
        else:
            return np.asarray(new)


def global_local_frame(n_points):
    """Return identity local frames (global = local) for n_points."""
    return LocalFrame(np.tile(np.eye(3), (n_points, 1, 1)))


def GenerateCylindricalLocalFrame(crd, axis=2, origin=[0, 0, 0], dim=3):
    """Generate cylindrical local frames (er, etheta, ez) at each node.

    Parameters
    ----------
    crd : array_like or Mesh
        Node coordinates, shape (N, 3) or (N, 2).
    axis : int, optional
        Cylinder axis: 0=X, 1=Y, 2=Z (default 2).
    origin : array_like, optional
        Origin of the cylindrical coordinate system.
    dim : int, optional
        Spatial dimension: 2 or 3 (default 3).

    Returns
    -------
    LocalFrame
    """
    if isinstance(crd, Mesh):
        crd = Mesh.nodes

    localFrame = np.zeros((len(crd), dim, dim))
    if dim == 3:
        plane = [0, 1, 2]
        plane.pop(axis)
        localFrame[:, 2, axis] = 1.0  # ez
    else:
        plane = [0, 1]
        crd = crd[:, 0:2]
        origin = np.array(origin)[0:2]

    crd = crd - np.array(origin).reshape(1, -1)

    localFrame[:, 0, plane] = crd[:, plane] / np.sqrt(
        crd[:, plane[0]] ** 2 + crd[:, plane[1]] ** 2
    ).reshape(-1, 1)  # er

    if dim == 3:
        localFrame[:, 1] = np.cross(
            localFrame[:, 2], localFrame[:, 0]
        )  # etheta = ez x er
    else:
        localFrame[:, 1, 0] = -localFrame[:, 0, 1]
        localFrame[:, 1, 1] = localFrame[:, 0, 0]
    return localFrame.view(LocalFrame)
