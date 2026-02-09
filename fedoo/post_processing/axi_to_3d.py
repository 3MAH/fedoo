import numpy as np
from fedoo.mesh.functions import axisymmetric_extrusion
from fedoo.core.dataset import DataSet, MultiFrameDataSet, read_data
import os
from zipfile import Path


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
    e_r = np.tile(np.column_stack((np.cos(theta), np.sin(theta))), (mesh.n_nodes, 1))

    res3d = DataSet(full_mesh)
    for field in [
        *axi_data.node_data,
        *axi_data.gausspoint_data,
        *axi_data.element_data,
    ]:
        if field == "Disp":
            # dr,dz = np.tile(res.node_data[field],n_theta)
            dr, dz = (
                axi_data.node_data[field].reshape(2, mesh.n_nodes, 1)
                * np.ones((1, 1, n_theta))
            ).reshape(2, -1)
            res3d.node_data[field] = np.column_stack((dr[:, np.newaxis] * e_r, dz)).T
        else:
            res3d.node_data[field] = (
                axi_data.get_data(field, data_type="Node").reshape(-1, mesh.n_nodes, 1)
                * np.ones((1, 1, n_theta))
            ).reshape(-1, mesh.n_nodes * n_theta)

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
        theta = np.linspace(0, 2 * np.pi, n_theta)
        self._er = np.tile(
            np.column_stack((np.cos(theta), np.sin(theta))), (self.mesh2d.n_nodes, 1)
        )

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
            dr, dz = (
                self.axi_data.node_data["Disp"].reshape(2, self.mesh2d.n_nodes, 1)
                * np.ones((1, 1, self.n_theta))
            ).reshape(2, -1)
            self.node_data["Disp"] = np.column_stack(
                (dr[:, np.newaxis] * self._er, dz)
            ).T

    def get_data(self, field, component=None, data_type=None, return_data_type=False):
        if field not in self.node_data:
            # import field as 3d node data if not already present
            data2d = self.axi_data.get_data(field, data_type="Node")
            self.node_data[field] = (
                data2d.reshape(-1, self.mesh2d.n_nodes, 1)
                * np.ones((1, 1, self.n_theta))
            ).reshape(-1, self.mesh2d.n_nodes * self.n_theta)

        return DataSet.get_data(self, field, component, data_type, return_data_type)

    def load(self, data=-1, load_mesh=False):
        if load_mesh:
            return NotImplemented
        self.axi_data.load(data, load_mesh)
        self.node_data = {}
        self.element_data = {}
        self.gausspoint_data = {}
        self._import_disp_to_3d()
