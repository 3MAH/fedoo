"""Not intended for public use, excepted to derive new mechanical constitutivelaw"""

# baseclass
import numpy as np
from fedoo.core.base import ConstitutiveLaw


class Mechanical3D(ConstitutiveLaw):
    """Base class for mechanical constitutive laws."""

    # model of constitutive law for InternalForce Weakform

    def __init__(self, name=""):
        ConstitutiveLaw.__init__(self, name)
        self._Lt_from_F = False
        # _Lt_from_F attribute is True if the tangent matrix is related
        # to F instead of log epsilonn, ie for hyper elastic materials

    def initialize(self, assembly, pb):
        pass
        # #function called to initialize the constutive law
        # assembly.sv['Strain'] = 0
        # assembly.sv['Stress'] = 0
        # assembly.sv['DispGradient'] = 0
        # assembly.sv['TangentMatrix'] = self.get_tangent_matrix(assembly)

    def update(self, assembly, pb):
        pass
        # function called to update the state of constitutive law
        # assembly.sv['TangentMatrix'] = self.get_tangent_matrix(assembly)

    # def get_tangent_matrix(self, assembly, dimension=None): #Tangent Matrix in lobal coordinate system (no change of basis)
    #     return NotImplemented

    # def get_H(self, assembly, dimension = None): #Tangent Matrix in global coordinate system (apply change of basis) + account for dimension of the problem
    #     if dimension is None: dimension = assembly.space.get_dimension()
    #     if dimension == "2Dstress":
    #         H = self.get_tangent_matrix_2Dstress()
    #         if H is NotImplemented:
    #             H = self.local2global_H(self.get_tangent_matrix())
    #             return self.get_H_plane_stress(H)
    #         else:
    #             return self.local2global_H(H)

    #     return self.local2global_H(self.get_tangent_matrix())

    def get_H_plane_stress(self, H):
        """
        Convert a full 3D tangent matrix H in an equivalent behavior in 2D with the plane stress assumption.

        Parameters
        ----------
        H : TYPE
            Full 3D tangent matrix

        Returns
        -------
        H_plane_stress

        """
        return [
            [
                H[i][j] - H[i][2] * H[j][2] / H[2][2] if j in [0, 1, 3] else 0
                for j in range(6)
            ]
            if i in [0, 1, 3]
            else [0, 0, 0, 0, 0, 0]
            for i in range(6)
        ]

    def local2global_H(self, H_global):
        # Change of basis capability for laws on the form : StressTensor = H * StrainTensor
        # StressTensor and StrainTensor are column vectors based on the voigt notation
        if self.local_frame is not None:
            # building the matrix to change the basis of the stress and the strain
            #            theta = np.pi/8
            #            np.array([[np.cos(theta),np.sin(theta),0], [-np.sin(theta),np.cos(theta),0], [0,0,1]])
            if len(self.local_frame.shape) == 2:
                self.local_frame = self.local_frame.reshape(1, 3, 3)
            R_epsilon = np.empty((len(self.local_frame), 6, 6))
            R_epsilon[:, :3, :3] = self.local_frame**2
            R_epsilon[:, :3, 3:6] = (
                self.local_frame[:, :, [0, 2, 1]] * self.local_frame[:, :, [1, 0, 2]]
            )
            R_epsilon[:, 3:6, :3] = (
                2 * self.local_frame[:, [0, 2, 1]] * self.local_frame[:, [1, 0, 2]]
            )
            R_epsilon[:, 3:6, 3:6] = (
                self.local_frame[:, [[0], [2], [1]], [0, 2, 1]]
                * self.local_frame[:, [[1], [0], [2]], [1, 0, 2]]
                + self.local_frame[:, [[1], [0], [2]], [0, 2, 1]]
                * self.local_frame[:, [[0], [2], [1]], [1, 0, 2]]
            )
            R_sigma_inv = R_epsilon.transpose(
                0, 2, 1
            )  # np.transpose(R_epsilon,[0,2,1])

            if len(H_global.shape) == 3:
                H_global = np.rollaxis(H_global, 2, 0)
            H_local = np.matmul(R_sigma_inv, np.matmul(H_global, R_epsilon))
            if len(H_local.shape) == 3:
                if H_local.shape[0] == 1:
                    return H_local[0, :, :]
                H_local = np.rollaxis(H_local, 0, 3)

            return H_local

        else:
            return H_global
