from fedoo.util.deprecation import deprecated_alias
from simcoon import Rotation
from scipy.spatial.transform import Rotation as ScipyRotation

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

    def rotate(self, angle, axis="Z"):
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

    Rotate = deprecated_alias(rotate, "Rotate")

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


def as_local_frame(local_frame, dimension=None):
    """Return rotation matrices as a :class:`LocalFrame` array.

    ``local_frame`` may be an array whose last two axes contain 2x2 or 3x3
    matrices, or any rotation object exposing an ``as_matrix()`` method (in
    particular SciPy and simcoon ``Rotation`` objects). Leading array axes are
    retained so element-by-Gauss-point arrays can be interpreted later.
    """
    if hasattr(local_frame, "as_matrix"):
        local_frame = local_frame.as_matrix()

    matrices = np.asarray(local_frame, dtype=float)
    if matrices.ndim == 2:
        matrices = matrices[np.newaxis]
    elif (
        matrices.ndim == 3
        and matrices.shape[:2] in ((2, 2), (3, 3))
        and matrices.shape[-2:] not in ((2, 2), (3, 3))
    ):
        matrices = np.moveaxis(matrices, -1, 0)

    if matrices.ndim < 3 or matrices.shape[-1] != matrices.shape[-2]:
        raise ValueError(
            "local_frame must contain square rotation matrices on its last two axes"
        )
    if matrices.shape[-1] not in (2, 3):
        raise ValueError("local-frame matrices must be 2x2 or 3x3")
    if dimension is not None and matrices.shape[-1] != dimension:
        raise ValueError(
            f"expected {dimension}x{dimension} local-frame matrices, got "
            f"{matrices.shape[-2]}x{matrices.shape[-1]}"
        )
    if not np.all(np.isfinite(matrices)):
        raise ValueError("local-frame matrices must only contain finite values")

    flat = matrices.reshape(-1, matrices.shape[-1], matrices.shape[-1])
    identity = np.eye(matrices.shape[-1])
    if not np.allclose(flat @ flat.transpose(0, 2, 1), identity, atol=1e-7):
        raise ValueError("local-frame matrices must be orthonormal")
    if not np.allclose(np.linalg.det(flat), 1.0, atol=1e-7):
        raise ValueError(
            "local-frame matrices must be proper rotations (determinant +1)"
        )
    return matrices.view(LocalFrame)


def _normalize_location(location):
    if location is None:
        return None
    if location not in ("Node", "Element", "GaussPoint"):
        raise ValueError(
            "local-frame location must be 'Node', 'Element' or 'GaussPoint'"
        )
    return location


def _interpolate_nodal_frames(local_frame, assembly):
    """Interpolate nodal rotations to Gauss points using a local log map."""
    mesh = assembly.mesh
    if assembly.n_elm_gp not in mesh._elm_interpolation:
        mesh.init_interpolation(assembly.n_elm_gp)
    element_nodes = mesh._elements_geom
    shape_functions = mesh._elm_interpolation[assembly.n_elm_gp].shape_function_gp
    nodal_frames = np.asarray(local_frame)[element_nodes]

    # Express each element's nodal rotations relative to its first node. The
    # weighted rotation vectors form a smooth FE interpolation while the
    # exponential map guarantees an orthonormal result at every Gauss point.
    reference = nodal_frames[:, 0]
    relative = reference[:, None].transpose(0, 1, 3, 2) @ nodal_frames
    relative_vectors = ScipyRotation.from_matrix(relative.reshape(-1, 3, 3))
    relative_vectors = relative_vectors.as_rotvec().reshape(
        mesh.n_elements, element_nodes.shape[1], 3
    )
    gp_vectors = np.einsum("gn,end->egd", shape_functions, relative_vectors)
    increments = ScipyRotation.from_rotvec(gp_vectors.reshape(-1, 3)).as_matrix()
    increments = increments.reshape(mesh.n_elements, assembly.n_elm_gp, 3, 3)
    gp_frames = reference[:, None] @ increments

    # Fedoo stores fields in Gauss-point-major, element-minor order.
    return LocalFrame(gp_frames.transpose(1, 0, 2, 3).reshape(-1, 3, 3))


def local_frame_to_gausspoints(local_frame, assembly, location=None):
    """Resolve material frames to an assembly's Gauss points.

    Frames may be uniform, nodal, elemental, or already defined at every
    Gauss point. With ``location=None`` the storage is inferred from the number
    of matrices. Use an explicit location when mesh counts are ambiguous.
    """
    frames = as_local_frame(local_frame, dimension=3)
    n_elements = assembly.mesh.n_elements
    n_nodes = assembly.mesh.n_nodes
    n_elm_gp = assembly.n_elm_gp
    n_gauss_points = n_elements * n_elm_gp

    if frames.ndim == 4:
        if frames.shape[:2] == (n_elements, n_elm_gp):
            frames = frames.transpose(1, 0, 2, 3).reshape(-1, 3, 3)
        elif frames.shape[:2] == (n_elm_gp, n_elements):
            frames = frames.reshape(-1, 3, 3)
        else:
            frames = frames.reshape(-1, 3, 3)
    else:
        frames = frames.reshape(-1, 3, 3)

    location = _normalize_location(location)
    n_frames = len(frames)
    expected = {
        "Node": n_nodes,
        "Element": n_elements,
        "GaussPoint": n_gauss_points,
    }

    if n_frames == 1 and location is None:
        return LocalFrame(frames)
    if location is None:
        matches = [kind for kind, count in expected.items() if count == n_frames]
        if not matches:
            raise ValueError(
                f"got {n_frames} local frames; expected 1, {n_nodes} nodes, "
                f"{n_elements} elements, or {n_gauss_points} Gauss points"
            )
        if len(matches) > 1:
            # Element and Gauss-point storage are identical with one point per
            # element, so this ambiguity has no effect on the resolved data.
            if set(matches) == {"Element", "GaussPoint"} and n_elm_gp == 1:
                location = "GaussPoint"
            else:
                raise ValueError(
                    "the number of local frames is ambiguous for this mesh; "
                    "specify location='Node', 'Element' or 'GaussPoint'"
                )
        else:
            location = matches[0]
    elif n_frames != expected[location]:
        raise ValueError(
            f"{location} local frames require {expected[location]} matrices, "
            f"got {n_frames}"
        )

    if location == "Node":
        return _interpolate_nodal_frames(frames, assembly)
    if location == "Element":
        return LocalFrame(np.tile(frames, (n_elm_gp, 1, 1)))
    return LocalFrame(frames)


def global_local_frame(n_points):
    """Return identity local frames (global = local) for n_points."""
    return LocalFrame(np.tile(np.eye(3), (n_points, 1, 1)))


def generate_cylindrical_local_frame(crd, axis=2, origin=None, dim=None):
    """Generate cylindrical local frames (er, etheta, ez) at each node.

    Parameters
    ----------
    crd : array_like or Mesh
        Node coordinates, shape (N, 3) or (N, 2).
    axis : int, optional
        Cylinder axis: 0=X, 1=Y, 2=Z (default 2).
    origin : array_like, optional
        Origin of the cylindrical coordinate system. The coordinate-system
        origin is used by default.
    dim : {2, 3}, optional
        Spatial dimension. By default it is inferred from ``crd`` or the
        supplied mesh.

    Returns
    -------
    LocalFrame
    """
    if hasattr(crd, "nodes") and hasattr(crd, "ndim"):
        if dim is None:
            dim = crd.ndim
        crd = crd.nodes

    crd = np.asarray(crd, dtype=float)
    if crd.ndim != 2:
        raise ValueError("coordinates must be a two-dimensional array")
    if dim is None:
        dim = crd.shape[1]
    if dim not in (2, 3):
        raise ValueError("cylindrical local frames are only defined in 2D or 3D")
    if crd.shape[1] < dim:
        crd = np.pad(crd, ((0, 0), (0, dim - crd.shape[1])))
    elif crd.shape[1] > dim:
        crd = crd[:, :dim]

    if origin is None:
        origin = np.zeros(dim)
    else:
        origin = np.asarray(origin, dtype=float).reshape(-1)
        if len(origin) < dim:
            origin = np.pad(origin, (0, dim - len(origin)))
        elif len(origin) > dim:
            origin = origin[:dim]

    localFrame = np.zeros((len(crd), dim, dim))
    if dim == 3:
        plane = [0, 1, 2]
        plane.pop(axis)
        localFrame[:, 2, axis] = 1.0  # ez
    else:
        plane = [0, 1]
        crd = crd[:, 0:2]
        origin = origin[0:2]

    crd = crd - origin.reshape(1, -1)

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
