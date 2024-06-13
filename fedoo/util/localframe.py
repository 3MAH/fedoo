# from fedoo.pgd.SeparatedArray import SeparatedArray
from fedoo.core.mesh import Mesh

# from fedoo.pgd.MeshPGD import MeshPGD
import numpy as np


class LocalFrame(np.ndarray):
    """
    The class LocalFrame is derived from the np.ndarray class including some additional function to treat local frame.
    A LocalFrame object should be a (N,3,3) shaped array in 3D or (N,2,2) shaped array in 2D,
    where N is the number of points (nodes, gauss point, elements, ...) in which local frames are defined.
    If LF is a LocalFrame object:
        - LF[i] gives the local frame associated to the ith point.
        - LF[i][0], LF[i][1] and LF[i][2]  gives respectively the 3 vectors defining the local frame.
          These 3 vectors are assumed unit and orthogonal.
    """

    def __new__(self, localFrame):
        return np.asarray(localFrame).view(self)

    def Rotate(self, angle, axis="Z"):
        # angle in degree
        if angle is not 0:
            angle = angle / 180.0 * np.pi

            if axis.upper() == "X":
                axis = (1, 0, 0)
            elif axis.upper() == "Y":
                axis = (0, 1, 0)
            elif axis.upper() == "Z":
                axis = (0, 0, 1)

            s = np.sin(angle)
            c = np.cos(angle)
            x = axis[0]
            y = axis[1]
            z = axis[2]
            axis = np.array([axis])

            RotMatrix = np.array(
                [
                    [
                        (1 - c) * x**2 + c,
                        (1 - c) * x * y - z * s,
                        (1 - c) * x * z + y * s,
                    ],
                    [
                        (1 - c) * x * y + z * s,
                        (1 - c) * y**2 + c,
                        (1 - c) * y * z - x * s,
                    ],
                    [
                        (1 - c) * x * z - y * s,
                        (1 - c) * y * z + x * s,
                        (1 - c) * z**2 + c,
                    ],
                ]
            )

            if len(RotMatrix.shape) == 3:
                RotMatrix = np.transpose(RotMatrix, [2, 1, 0])
            elif len(RotMatrix.shape) == 2:
                RotMatrix = RotMatrix.T

            self[:] = np.matmul(self, RotMatrix)
            return self

    #            np.dot(axis.T,axis)*(1-c) \
    #                 + np.array([[ c        ,-axis[2]*s, axis[1]*s],
    #                             [ axis[2]*s, c        ,-axis[0]*s],
    #                         [-axis[2]*s, axis[0]*s, c        ]])

    #            if len(RotMatrix.shape) == 3:
    #                RotMatrix = np.transpose(R_epsilon,[2,0,1])
    #            else: R_sigma_inv = R_epsilon.T
    #            if len(H.shape) == 3: H = np.rollaxis(H,2,0)
    #            H = np.matmul(R_sigma_inv, np.matmul(H,R_epsilon))
    #            if len(H.shape) == 3: H = np.rollaxis(H,0,3)
    #
    #        return H

    def __getitem__(self, index):
        new = super(LocalFrame, self).__getitem__(index)
        if new.ndim == 3 and new.shape[1] in [2, 3] and new.shape[2] in [2, 3]:
            return new
        else:
            return np.asarray(new)


def global_local_frame(n_points):
    return LocalFrame([np.eye(3) for i in range(n_points)])


def GenerateCylindricalLocalFrame(crd, axis=2, origin=[0, 0, 0], dim=3):
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

    crd = crd - np.array(origin).reshape(1, -1)  # changement of origin

    localFrame[:, 0, plane] = crd[:, plane] / np.sqrt(
        crd[:, plane[0]] ** 2 + crd[:, plane[1]] ** 2
    ).reshape(-1, 1)  # er

    if dim == 3:
        localFrame[:, 1] = np.cross(
            localFrame[:, 2], localFrame[:, 0]
        )  # etheta is the cross product
    else:
        localFrame[:, 1, 0] = -localFrame[:, 0, 1]
        localFrame[:, 1, 1] = localFrame[:, 0, 0]
    return localFrame.view(LocalFrame)


# MOVE separated_local_frame to the pgd folder

# def separated_local_frame(localFrame, mesh, dimensions = ('X','Y','Z')):
#     """
#     Permit to automatically assign the localFrame to the appropriate submesh of the mesh object
#     Generate a local frame under the form of the (3,3) shaped array dedicated of SeparatedArray objects
#     This functions work only if the local frame is restricted to one subspace.
#     """

#     dim = localFrame.shape[-1]
#     idmesh = mesh.FindCoordinatename(dimensions[0])
#     if idmesh != mesh.FindCoordinatename(dimensions[1]): raise NameError("'{}' and '{}' coordinates should be associated to the same subMesh".format(dimensions[0], dimensions[1]))
#     if dim == 3:
#         if idmesh != mesh.FindCoordinatename(dimensions[3]):  raise NameError("'{}' and '{}' coordinates should be associated to the same subMesh. Consider using a 2D local frame.".format(dimensions[0], dimensions[2]))

#     id_crd = []
#     for label in dimensions:
#         if label == 'X': id_crd.append(0)
#         elif label == 'Y': id_crd.append(1)
#         elif label == 'Z': id_crd.append(2)
#         else: raise NameError("Coordinates for local frame should be 'X', 'Y' or 'Z'. '{}' unknown.".format(label))

#     newLocalFrame = np.zeros((3, 3), dtype =object) #the resulting local frame is always in dim = 3

#     for j in range(dim):
#         for i in range(dim):
#             newLocalFrame[i,id_crd[j]] = SeparatedArray([np.c_[localFrame[:,i,j]] if k==idmesh else np.array([[1.]]) for k in range(mesh.get_dimension())])
#     return newLocalFrame


#   TODO: Not finished
#   if isinstance(crd, MeshPGD):
#        mesh = crd
#        crd_all = [] #list containing the values of coordinates for each nodes of the separated mesh using 3 SeparatedArray objects
#        for ii, namecrd in enumerate(['X','Y','Z']):
#            idmesh = mesh.FindCoordinatename(namecrd)
#            subMesh = mesh.GetListMesh()[idmesh]
#            crd = subMesh.nodes[:, subMesh.crd_name.index(namecrd)]
#            crd_all.append(SeparatedArray([np.c_[crd] if i == idmesh else np.array([[1.]]) for i in range(mesh.get_dimension())]))
#
#        localFrame = np.zeros((dim, dim), dtype =object)
#
#        plane = [0,1,2] ; plane.pop(axis)
#        localFrame[2, axis] =  SeparatedArray([np.array([[1.]]) for i in range(mesh.get_dimension())])#ez
#
#        crd_all[0] = crd_all[0] - origin[0] #changement of origin
#        crd_all[1] = crd_all[1] - origin[1] #changement of origin
#        crd_all[2] = crd_all[2] - origin[2] #changement of origin
#
#
#
#        localFrame[:, 0, plane] = crd[:,plane]/sqrt(crd_all[plane[0]]**2+crd_all[plane[1]]**2) #er
#
#        if dim == 3:
#            localFrame[:, 1] = np.cross(localFrame[:,2], localFrame[:,0]) #etheta is the cross product
#        else:
#            localFrame[:, 1, 0] = -localFrame[:,0, 1]
#            localFrame[:, 1, 1] = localFrame[:,0, 0]
