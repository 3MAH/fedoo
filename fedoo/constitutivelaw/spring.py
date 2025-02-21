# derive de ConstitutiveLaw

from fedoo.core.base import ConstitutiveLaw
from fedoo.core.base import AssemblyBase
import numpy as np
from numpy import linalg


class Spring(ConstitutiveLaw):
    """
    Simple directional spring connector between nodes or surfaces

    This constitutive Law should be associated with :mod:`fedoo.weakform.InterfaceForce`

    Parameters
    ----------
    Kx: scalar
        the rigidity along the X direction in material coordinates
    Ky: scalar
        the rigidity along the Y direction in material coordinates
    Kz: scalar
        the rigidity along the Z direction in material coordinates
    name: str, optional
        The name of the constitutive law
    """

    # Similar to CohesiveLaw but with different rigidity axis and without damage variable
    # Use with WeakForm.InterfaceForce
    def __init__(self, Kx=0, Ky=0, Kz=0, name=""):
        ConstitutiveLaw.__init__(self, name)  # heritage
        self.parameters = {"Kx": Kx, "Ky": Ky, "Kz": Kz}
        # self._InterfaceStress = 0

    def initialize(self, assembly, pb):
        assembly.sv["InterfaceStress"] = 0  # Interface Stress
        assembly.sv["TangentMatrix"] = self.get_K()

    # def GetRelativeDisp(self):
    #     return self.__Delta

    # def GetInterfaceStress(self):
    #     return self._InterfaceStress

    def get_tangent_matrix(self):
        return [
            [self.parameters["Kx"], 0, 0],
            [0, self.parameters["Ky"], 0],
            [0, 0, self.parameters["Kz"]],
        ]

    def get_K(self):
        return self.local2global_K(self.get_tangent_matrix())

    def local2global_K(self, K):
        # Change of basis capability for spring type laws on the form : ForceVector = K * DispVector
        if self.local_frame is not None:
            # building the matrix to change the basis of the stress and the strain
            B = self.local_frame

            if len(B.shape) == 3:
                Binv = np.transpose(B, [2, 1, 0])
                B = np.transpose(B, [1, 2, 0])

            elif len(B.shape) == 2:
                Binv = B.T

            dim = len(B)

            KB = [
                [sum([K[i][j] * B[j][k] for j in range(dim)]) for k in range(dim)]
                for i in range(dim)
            ]
            K = [
                [sum([Binv[i][j] * KB[j][k] for j in range(dim)]) for k in range(dim)]
                for i in range(dim)
            ]

            if dim == 2:
                K[0].append(0)
                K[1].append(0)
                K.append([0, 0, 0])

        return K

    def update(self, assembly, pb):
        displacement = pb.get_dof_solution()
        K = self.get_K()
        assembly.sv["TangentMatrix"] = K
        if np.isscalar(displacement) and displacement == 0:
            assembly.sv["InterfaceStress"] = assembly.sv["RelativeDisp"] = 0
        else:
            op_delta = (
                assembly.space.op_disp()
            )  # relative displacement = disp if used with cohesive element
            delta = [assembly.get_gp_results(op, displacement) for op in op_delta]
            assembly.sv["RelativeDisp"] = delta

            # Compute interface stress
            dim = len(delta)
            assembly.sv["InterfaceStress"] = [
                sum([delta[j] * K[i][j] for j in range(dim)]) for i in range(dim)
            ]  # list of 3 objects
