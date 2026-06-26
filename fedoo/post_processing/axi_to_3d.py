import numpy as np
from fedoo.mesh.functions import axisymmetric_extrusion
from fedoo.core.dataset import DataSet, MultiFrameDataSet, read_data
import os
from zipfile import Path


def _revolve_axi_field(field, theta, n_nodes_2d, kind=None, field_name=None):
    """Revolve a 2D axisymmetric field around theta to a 3D field.

    The 2D (r, z) plane is interpreted using fedoo's 2Daxi conventions:
    r = column 0, z = column 1, hoop component in Voigt slot 2 of any
    6-vector. The output 3D Cartesian frame uses (X, Y, Z) with the
    symmetry axis along Z, matching ``axisymmetric_extrusion``.

    Parameters
    ----------
    field : ndarray, shape (n_components, n_nodes_2d) or (n_nodes_2d,)
        Field on the 2D mesh.
    theta : ndarray, shape (n_theta,)
        Angular sample positions.
    n_nodes_2d : int
        Number of nodes in the 2D mesh.
    kind : {"stress", "strain"}, optional
        For 6-tensor fields, declares whether the off-diagonal Voigt slot
        stores a tensor component (``"stress"`` -> ``sigma_ij``) or an
        engineering shear (``"strain"`` -> ``gamma_ij = 2 * eps_ij``). A
        strain field needs an extra factor of 2 in the rotated slot 3
        (``gamma_xy``) because the hoop / radial diagonals are *not*
        engineering quantities. ``None`` falls back to the
        ``field_name`` heuristic below.
    field_name : str, optional
        Used only when ``kind`` is None: any name containing ``"Strain"``
        is treated as ``kind="strain"``, otherwise ``kind="stress"``.

    Returns
    -------
    ndarray, shape (n_components_out, n_nodes_2d * n_theta)
        Revolved field. ``n_components_out`` is 3 for a 2-vector input,
        6 for a 6-tensor input, and equal to ``n_components`` otherwise
        (scalars and unrecognised shapes are tiled along theta).
    """
    field = np.asarray(field)
    n_theta = theta.shape[0]
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # Promote a 1-D scalar field to 2-D for uniform handling.
    if field.ndim == 1:
        field = field.reshape(1, -1)

    n_comp = field.shape[0]

    # Convention: the extruded 3D mesh has nodes laid out node-major,
    # i.e. the 2D node index varies slowest and theta varies fastest.
    # All per-node arrays we build below have shape (n_nodes_2d, n_theta)
    # so that ``.ravel()`` produces [n0t0, n0t1, ..., n1t0, n1t1, ...].
    cos_row = cos_t[None, :]  # (1, n_theta)
    sin_row = sin_t[None, :]

    if n_comp == 2:
        # Cylindrical (dr, dz) vector -> Cartesian (X, Y, Z)
        # at each angle theta_k: (dr*cos, dr*sin, dz).
        dr = field[0][:, None]  # (n_nodes_2d, 1)
        dz = field[1][:, None]
        out = np.empty((3, n_nodes_2d * n_theta), dtype=field.dtype)
        out[0] = (dr * cos_row).ravel()
        out[1] = (dr * sin_row).ravel()
        out[2] = np.broadcast_to(dz, (n_nodes_2d, n_theta)).ravel()
        return out

    if n_comp == 6:
        # Symmetric second-order tensor in fedoo's 2Daxi slot ordering
        # (s_rr, s_zz, s_tt, s_rz, 0, 0) -> Cartesian Voigt (xx, yy, zz, xy, xz, yz).
        # Rotation about the symmetry axis Z by theta gives, for the underlying
        # tensor: s_xy_tensor = sin*cos (s_rr - s_tt). For strain fields whose
        # slot 3 stores an engineering shear gamma = 2*eps, the output slot 3
        # gamma_xy equals 2 * sin*cos * (s_rr - s_tt) = sin(2 theta)*(...).
        # Slots 4 and 5 (rz -> xz/yz) carry the same convention as slot 3 of the
        # input on both sides, so no extra factor is needed there.
        if kind is None:
            kind = "strain" if (field_name and "Strain" in field_name) else "stress"
        if kind not in ("stress", "strain"):
            raise ValueError(f"kind must be 'stress' or 'strain', got {kind!r}.")
        diag_to_xy = 2.0 if kind == "strain" else 1.0

        s_rr = field[0][:, None]  # (n_nodes_2d, 1)
        s_zz = field[1][:, None]
        s_tt = field[2][:, None]
        s_rz = field[3][:, None]
        c2 = (cos_t**2)[None, :]
        s2 = (sin_t**2)[None, :]
        sc = (sin_t * cos_t)[None, :]
        out = np.empty((6, n_nodes_2d * n_theta), dtype=field.dtype)
        out[0] = (c2 * s_rr + s2 * s_tt).ravel()
        out[1] = (s2 * s_rr + c2 * s_tt).ravel()
        out[2] = np.broadcast_to(s_zz, (n_nodes_2d, n_theta)).ravel()
        out[3] = (diag_to_xy * sc * (s_rr - s_tt)).ravel()
        out[4] = (cos_row * s_rz).ravel()
        out[5] = (sin_row * s_rz).ravel()
        return out

    # Scalar fields and any unrecognised shape: tile along theta unchanged
    # (node-major ordering: each 2D node value repeated n_theta times).
    return (field.reshape(n_comp, n_nodes_2d, 1) * np.ones((1, 1, n_theta))).reshape(
        n_comp, n_nodes_2d * n_theta
    )


def axi_to_3d(axi_data: DataSet | MultiFrameDataSet, n_theta: int = 41, filename=None):
    """Convert axisymmetric data into a full 3D representation.

    Generate a new mesh by revolving the 2d geometry around the symmetry axis.

    This function accepts either:
      • a single-frame axisymmetric dataset (`DataSet`), or
      • a multi-frame axisymmetric dataset (`MultiFrameDataSet`).

    Behavior depends on the input type and whether an output filename is provided:

    - If `axi_data` is a `MultiFrameDataSet`:
        * When `filename` is provided, creates a new 3D multi-frame dataset on
          disk (an `fdz` file) and returns the corresponding 3D
          `MultiFrameDataSet`.
        * When `filename` is not provided, returns a lightweight,
          memory-efficient wrapper (`AxiMultiFrameDataSet`) that exposes a
          3D-like view without materializing all revolved data in memory.

    - If `axi_data` is a `DataSet`:
        * Computes the revolved 3D dataset in memory. If `filename` is
          provided, the resulting dataset is also saved to disk.

    Parameters
    ----------
    axi_data : DataSet or MultiFrameDataSet
        Axisymmetric input data to revolve into 3D. The data is assumed to be
        axisymmetric about the Y-axis.
    n_theta : int, default=41
        Number of azimuthal samples used in the revolution. Angular samples
        are distributed uniformly over [0, 2π].
    filename : str, optional
        filename used to save the generated data on disk.

    Returns
    -------
    DataSet or MultiFrameDataSet or AxiMultiFrameDataSet
    """
    if isinstance(axi_data, str):
        axi_data = read_data(axi_data)
    if isinstance(axi_data, MultiFrameDataSet):
        if filename:
            # write a new fdz file and open the corresponding MultiFrameDataSet
            return axi_to_3d_multi(filename, axi_data, n_theta)
        else:
            # simple wrapper to the existing data (memory efficient)
            return AxiMultiFrameDataSet(axi_data)
    elif isinstance(axi_data, DataSet):
        axi_data = axi_to_3d_dataset(axi_data, n_theta)
        if filename:
            axi_data.save(filename, True, False)


def axi_to_3d_dataset(axi_data: DataSet, n_theta: int = 41):
    """Convert a 2D axisymmetric DataSet to a full 3D DataSet.

    A 3D mesh is built from a 2D axisymmetric mesh and all field data (node
    data, element data and gausspoint data) are converted to node in the new
    DataSet object.

    Parameters
    ----------
    axi_data: DataSet
        DataSet object containing 2D mesh (axi_data.mesh.ndim == 2) interpreted
        as axisymmetric data.
    n_theta: int, default=41
        Number of nodes used to build the 3D mesh along the theta direction in
        cylindrical coordinates.
    """
    mesh = axi_data.mesh
    full_mesh = axisymmetric_extrusion(mesh, n_theta, merge_nodes=False)
    theta = np.linspace(0, 2 * np.pi, n_theta)

    res3d = DataSet(full_mesh)
    for field in [
        *axi_data.node_data,
        *axi_data.gausspoint_data,
        *axi_data.element_data,
    ]:
        if field == "Disp":
            data2d = axi_data.node_data[field]
        else:
            data2d = axi_data.get_data(field, data_type="Node")
        res3d.node_data[field] = _revolve_axi_field(
            data2d, theta, mesh.n_nodes, field_name=field
        )

    res3d.scalar_data = axi_data.scalar_data
    return res3d


def axi_to_3d_multi(filename: str, axi_data: MultiFrameDataSet, n_theta: int = 41):
    """Convert a 2D axisymmetric MultiFrameDataSet to a full 3D one.

    A 3D mesh is built from a 2D axisymmetric mesh and all field data (node
    data, element data and gausspoint data) are converted to node in the new
    MultiFrameDataSet object. The 3D data is saved on disk using the filename
    and the fdz format.

    Parameters
    ----------
    filename: str
        Name of the str file (with path) to save the data.
        The fdz format is used.
    axi_data: MultiFrameDataSet
        DataSet object containing 2D mesh (axi_data.mesh.ndim == 2) interpreted
        as axisymmetric data.
    n_theta: int, default=41
        Number of nodes used to build the 3D mesh along the theta direction in
        cylindrical coordinates.
    """
    name, ext = os.path.splitext(filename)
    if ext == "":
        filename = filename + ".fdz"

    for i in range(0, axi_data.n_iter):
        axi_data.load(i)
        res = axi_to_3d_dataset(axi_data, n_theta)
        if i == 0:
            res3d = MultiFrameDataSet(res.mesh)
            res.to_fdz(filename, True)
        else:
            res.to_fdz(filename, False, i)

        res3d.list_data.append(Path(filename, "iter_" + str(i) + ".npz"))
    return res3d


class AxiMultiFrameDataSet(MultiFrameDataSet):
    def __init__(self, axi_data: MultiFrameDataSet, n_theta: int = 41):
        """MultiFrameDataSet wrapper to read axisymmetric data in 3d.

        This class allow to plot and read 3d data from 2d data by building an
        axisymmetric mesh. The data are converted to 3d only when requested,
        which is memory efficient.

        Parameters
        ----------
        axi_data: MultiFrameDataSet
            MultiFrameDataSet associated to a 2D mesh (axi_data.mesh.ndim == 2)
            interpreted as axisymmetric data.
        n_theta: int, default=41
            Number of nodes used to build the 3D mesh along the theta direction in
            cylindrical coordinates.
        """
        if isinstance(axi_data, str):
            axi_data = read_data(axi_data)
            if not isinstance(axi_data, MultiFrameDataSet):
                raise ValueError("Data not compatible with AxiMultiFrameDataSet.")
        self.axi_data = axi_data
        self.n_theta = n_theta

        mesh = axisymmetric_extrusion(self.mesh2d, n_theta, merge_nodes=False)
        # Precompute the angular sample positions once; reused on every
        # frame load by _revolve_axi_field.
        self._theta = np.linspace(0, 2 * np.pi, n_theta)

        DataSet.__init__(self, mesh)
        if self.loaded_iter is None:
            self.load(-1)
        else:
            self.load(self.loaded_iter)

    @property
    def mesh2d(self):
        return self.axi_data.mesh

    @property
    def list_data(self):
        return self.axi_data.list_data

    @property
    def loaded_iter(self):
        return self.axi_data.loaded_iter

    @property
    def scalar_data(self):
        return self.axi_data.scalar_data

    @scalar_data.setter
    def scalar_data(self, value):
        self.axi_data.scalar_data = value

    def field_names(self):
        return self.axi_data.field_names()

    def _import_disp_to_3d(self):
        if "Disp" not in self.node_data and "Disp" in self.axi_data.node_data:
            self.node_data["Disp"] = _revolve_axi_field(
                self.axi_data.node_data["Disp"], self._theta, self.mesh2d.n_nodes
            )

    def get_data(self, field, component=None, data_type=None, return_data_type=False):
        if field not in self.node_data:
            # import field as 3d node data if not already present
            data2d = self.axi_data.get_data(field, data_type="Node")
            self.node_data[field] = _revolve_axi_field(
                data2d, self._theta, self.mesh2d.n_nodes, field_name=field
            )

        return DataSet.get_data(self, field, component, data_type, return_data_type)

    def load(self, data=-1, load_mesh=False):
        if load_mesh:
            return NotImplemented
        self.axi_data.load(data, load_mesh)
        self.node_data = {}
        self.element_data = {}
        self.gausspoint_data = {}
        self._import_disp_to_3d()
