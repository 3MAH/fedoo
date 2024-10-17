# derive de ConstitutiveLaw
# simcoon compatible

from fedoo.core.mechanical3d import Mechanical3D
from fedoo.constitutivelaw.elastic_anisotropic import ElasticAnisotropic

import numpy as np


class ElasticOrthotropic(ElasticAnisotropic):
    """
    Linear Orthotropic constitutive law defined from the engineering coefficients in local material coordinates.

    The constitutive Law should be associated with :mod:`fedoo.weakform.InternalForce`

    Parameters
    ----------
    EX: scalars or arrays of gauss point values
        Young modulus along the X direction
    EY: scalars or arrays of gauss point values
        Young modulus along the Y direction
    EZ: scalars or arrays of gauss point values
        Young modulus along the Z direction
    GYZ, GXZ, GXY: scalars or arrays of gauss point values
        Shear modulus
    nuYZ, nuXZ, nuXY: scalars or arrays of gauss point values
        Poisson's ratio
    """

    def __init__(self, Ex, Ey, Ez, Gyz, Gxz, Gxy, nuyz, nuxz, nuxy, name=""):
        Mechanical3D.__init__(self, name)  # heritage

        self.Ex = Ex
        self.Ey = Ey
        self.Ez = Ez
        self.Gyz = Gyz
        self.Gxz = Gxz
        self.Gxy = Gxy
        self.nuyz = nuyz
        self.nuxz = nuxz
        self.nuxy = nuxy

    def get_tangent_matrix(self, assembly, dimension=None):
        if dimension is None:
            dimension = assembly.space.get_dimension()

        EX = self.Ex
        EY = self.Ey
        EZ = self.Ez
        GYZ = self.Gyz
        GXZ = self.Gxz
        GXY = self.Gxy
        nuYZ = self.nuyz
        nuXZ = self.nuxz
        nuXY = self.nuxy

        #        S = np.array([[1/EX    , -nuXY/EX, -nuXZ/EX, 0    , 0    , 0    ], \
        #                      [-nuXY/EX, 1/EY    , -nuYZ/EY, 0    , 0    , 0    ], \
        #                      [-nuXZ/EX, -nuYZ/EY, 1/EZ    , 0    , 0    , 0    ], \
        #                      [0       , 0       , 0       , 1/GXY, 0    , 0    ], \
        #                      [0       , 0       , 0       , 0    , 1/GXZ, 0    ], \
        #                      [0       , 0       , 0       , 0    , 0    , 1/GYZ]])
        #        H = linalg.inv(S) #H  = np.zeros((6,6), dtype='object')

        if np.isscalar(EX):
            H = np.zeros((6, 6))
        elif isinstance(EX, (np.ndarray, list)):
            H = np.zeros((6, 6, len(EX)))
        else:
            H = np.zeros((6, 6), dtype="object")

        nuYX = nuXY * EY / EX
        nuZX = nuXZ * EZ / EX
        nuZY = nuYZ * EZ / EY
        k = (
            1
            - nuYZ * nuZY
            - nuXY * nuYX
            - nuXZ * nuZX
            - nuXY * nuYZ * nuZX
            - nuYX * nuZY * nuXZ
        )
        H[0, 0] = EX * (1 - nuYZ * nuZY) / k
        H[1, 1] = EY * (1 - nuXZ * nuZX) / k
        H[2, 2] = EZ * (1 - nuXY * nuYX) / k
        H[0, 1] = H[1, 0] = EX * (nuYZ * nuZX + nuYX) / k
        H[0, 2] = H[2, 0] = EX * (nuYX * nuZY + nuZX) / k
        H[1, 2] = H[2, 1] = EY * (nuXY * nuZX + nuZY) / k
        H[3, 3] = GXY
        H[4, 4] = GXZ
        H[5, 5] = GYZ

        H = self.local2global_H(H)
        if dimension == "2Dstress":
            return self.get_H_plane_stress(H)
        else:
            return H
