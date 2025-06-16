import numpy as np
from fedoo.mesh.functions import extrude
from fedoo.core.dataset import DataSet, MultiFrameDataSet

def axi_to_3d(axi_data, n_theta):
    mesh = axi_data.mesh
    full_mesh = extrude(mesh,2*np.pi,n_theta)
    r,z,theta = full_mesh.nodes.T 
    full_mesh.nodes = np.c_[r*np.cos(theta), r*np.sin(theta), z]
    # fd.DataSet(full_mesh).plot()

    res3d = DataSet(full_mesh)
    for field in axi_data.node_data:
        if field == 'Disp':
            # dr,dz = np.tile(res.node_data[field],n_theta)
            dr,dz = (axi_data.node_data[field].reshape(2,-1,1) * np.ones((1,1,n_theta))).reshape(2,-1)
            res3d.node_data[field] = np.vstack( (dr*np.cos(theta), dr*np.sin(theta), dz))
        else:
            res3d.node_data[field] = np.tile(axi_data.node_data[field],n_theta)
    return res3d

def axi_to_3d_multi(filename, axi_data, n_theta):
    from zipfile import Path
    for i in range(0,axi_data.n_iter):
        axi_data.load(i)
        res = axi_to_3d(axi_data, n_theta)
        if i == 0:
            res3d = MultiFrameDataSet(res.mesh)
            res.to_fdz(filename, True)
        else:
            res.to_fdz(filename, False, i)

        res3d.list_data.append(Path(filename,'iter_'+str(i)+'.npz'))
    return res3d