"""Not intended for public use, excepted to derive new mechanical constitutivelaw"""

# baseclass
import numpy as np
from simcoon import Rotation
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

    def local2global_H(self, H):
        """Rotate stiffness matrix from local material frame to global frame.

        Uses simcoon.Rotation to build the 6x6 Voigt stress rotation matrix
        QS from the local frame, then computes H_global = QS @ H @ QS^T.
        """
        if self.local_frame is not None:
            local_frame = np.asarray(self.local_frame)
            if local_frame.ndim == 2:
                local_frame = local_frame[np.newaxis]
            rot = Rotation.from_matrix(local_frame)
            QS = rot.as_voigt_stress_rotation()  # (N, 6, 6)

            H = np.asarray(H, dtype=float)
            if H.ndim == 3:
                H = np.rollaxis(H, 2, 0)  # (6,6,M) -> (M,6,6)
            H = np.matmul(QS, np.matmul(H, QS.transpose(0, 2, 1)))
            if H.ndim == 3:
                if H.shape[0] == 1:
                    return H[0]
                return np.rollaxis(H, 0, 3)  # (N,6,6) -> (6,6,N)
            return H

        return H
