"""Portable local-frame field semantics shared by output, FDH5 and the viewer.

Fields are component-first. Matrices contain local basis vectors in their rows;
quaternions (x,y,z,w) and rotation vectors (radians) encode that same row matrix.
"""

import numpy as np
from scipy.spatial.transform import Rotation

FRAME_NAMES = {"ShellLocalFrame", "BeamLocalFrame", "MaterialLocalFrame"}
FRAME_COMPONENTS = ["Whole frame", "Local 1", "Local 2", "Local 3"]
FRAME_MATRIX_COMPONENTS = ["XX", "XY", "XZ", "YX", "YY", "YZ", "ZX", "ZY", "ZZ"]


def frame_axis_mask(axis):
    """Accept one-based axis numbers or a sequence; None selects all axes."""
    if axis is None:
        return (True, True, True)
    axes = [axis] if np.isscalar(axis) else list(axis)
    if any(
        isinstance(item, (bool, np.bool_))
        or not isinstance(item, (int, np.integer))
        or item not in (1, 2, 3)
        for item in axes
    ):
        raise ValueError(
            "local_frame_axis must be 1, 2, 3, a sequence of these, or None"
        )
    return tuple(index in axes for index in (1, 2, 3))


FRAME_ENCODINGS = {
    "matrix": "matrix-rows",
    "quaternion": "quaternion-xyzw-rows",
    "rotvec": "rotvec-radians-rows",
}


def is_frame_field(name, metadata=None):
    if name and name.lstrip("_") in FRAME_NAMES:
        return True
    description = (metadata or {}).get(name, {})
    return description.get("type") == "local_frame" or any(
        item.get("type") == "local_frame"
        for item in description.get("submeshes", {}).values()
    )


def frame_field_names(data):
    return sorted(
        {
            name
            for values in (data.node_data, data.element_data, data.gausspoint_data)
            for name in values
            if is_frame_field(name, data.field_metadata)
        }
    )


def frame_matrices(values):
    """Decode component-first 9/4/3 arrays to (n_locations, 3, 3) row matrices."""
    array = np.asarray(values)
    if array.ndim != 2 or array.shape[0] not in (9, 4, 3):
        raise ValueError("Local-frame fields must have shape (9, n), (4, n) or (3, n)")
    if not array.shape[1]:
        return np.empty((0, 3, 3), dtype=float)
    if array.shape[0] == 9:
        return array.T.reshape(-1, 3, 3)
    values = np.asarray(array.T, dtype=float)
    rotation = (
        Rotation.from_quat(values)
        if array.shape[0] == 4
        else Rotation.from_rotvec(values)
    )
    return rotation.as_matrix()


def encode_frames(matrices, encoding):
    if encoding == "matrix":
        return matrices.reshape(-1, 9).T
    if not len(matrices):
        return np.empty((4 if encoding == "quaternion" else 3, 0))
    rotation = Rotation.from_matrix(matrices)
    return (
        rotation.as_quat(canonical=True)
        if encoding == "quaternion"
        else rotation.as_rotvec()
    ).T
