import numpy as np
from fedoo.mesh.functions import extrude
from fedoo.core.dataset import DataSet, MultiFrameDataSet
import os
from zipfile import Path


def axi_to_3d(axi_data: DataSet, n_theta: int = 41):
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
    full_mesh = extrude(mesh, 2 * np.pi, n_theta)
    r, z, theta = full_mesh.nodes.T
    full_mesh.nodes = np.c_[r * np.cos(theta), r * np.sin(theta), z]
    # fd.DataSet(full_mesh).plot()

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
            res3d.node_data[field] = np.vstack(
                (dr * np.cos(theta), dr * np.sin(theta), dz)
            )
        else:
            res3d.node_data[field] = (
                axi_data.get_data(field, data_type="Node").reshape(-1, mesh.n_nodes, 1)
                * np.ones((1, 1, n_theta))
            ).reshape(-1, mesh.n_nodes * n_theta)

    res3d.scalar_data = axi_data.scalar_data
    return res3d


def axi_to_3d_multi(filename: str, axi_data: DataSet, n_theta: int = 41):
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
    axi_data: DataSet
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
        res = axi_to_3d(axi_data, n_theta)
        if i == 0:
            res3d = MultiFrameDataSet(res.mesh)
            res.to_fdz(filename, True)
        else:
            res.to_fdz(filename, False, i)

        res3d.list_data.append(Path(filename, "iter_" + str(i) + ".npz"))
    return res3d
