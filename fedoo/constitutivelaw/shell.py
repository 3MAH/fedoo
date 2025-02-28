# derive de ConstitutiveLaw
# compatible with the simcoon strain and stress notation

from fedoo.core.base import ConstitutiveLaw
from fedoo.util.voigt_tensors import StressTensorList, StrainTensorList

import numpy as np


class ShellBase(ConstitutiveLaw):
    # base model class that should derive any other shell constitutive laws
    def __init__(self, thickness, k=1, name=""):
        # assert get_Dimension() == '3D', "No 2D model for a shell kinematic. Choose '3D' problem dimension."

        ConstitutiveLaw.__init__(self, name)  # heritage

        self.__thickness = thickness
        self.__k = k
        self.__GeneralizedStrain = None
        self.__GeneralizedStress = None

    def GetThickness(self):
        return self.__thickness

    def Get_k(self):
        return self.__k

    def GetShellRigidityMatrix(self):
        raise NameError('"GetShellRigidityMatrix" not implemented, contact developer.')

    def GetShellRigidityMatrix_RI(self):
        raise NameError(
            '"GetShellRigidityMatrix_RI" not implemented, contact developer.'
        )

    def GetShellRigidityMatrix_FI(self):
        raise NameError(
            '"GetShellRigidityMatrix_FI" not implemented, contact developer.'
        )

    def update(self, assembly, pb):
        # disp = pb.get_disp()
        # rot = pb.get_rot()
        U = pb.get_dof_solution()
        if np.isscalar(U) and U == 0:
            assembly.sv["GeneralizedStrain"] = 0
            assembly.sv["GeneralizedStress"] = 0
        else:
            GeneralizedStrainOp = assembly.weakform.GetGeneralizedStrainOperator()
            GeneralizedStrain = [
                op if np.isscalar(op) else assembly.get_gp_results(op, U)
                for op in GeneralizedStrainOp
            ]

            H = self.GetShellRigidityMatrix()

            # all terms are computed. Perhaps could be optimized by computed only the termes related to the associated weak form (eg shear for selective integration)
            assembly.sv["GeneralizedStress"] = [
                sum(
                    [
                        GeneralizedStrain[j] * assembly.convert_data(H[i][j])
                        for j in range(8)
                    ]
                )
                for i in range(8)
            ]  # H[i][j] are converted to gauss point excepted if scalar
            assembly.sv["GeneralizedStrain"] = GeneralizedStrain

    def get_strain(self, assembly, **kargs):
        """
        Return the last computed strain associated to the given assembly

        Parameters
        ------------------------
        assembly: Assembly

        position : float (optional)
            Position in the thickness, given as a fraction of the demi thickness :
            z = position * total_thickness/2
            position = 1 for the top face (default)
            position = -1 for the bottom face
            position = 0 for the mid-plane

        Returns
        -------
        StrainTensorList object containing the strain at integration point
        """
        position = kargs.get("position", 1)
        z = position * self.GetThickness() / 2

        Strain = StrainTensorList([0 for i in range(6)])
        GeneralizedStrain = assembly.sv["GeneralizedStrain"]
        Strain[0] = (
            GeneralizedStrain[0] + z * GeneralizedStrain[4]
        )  # epsXX -> membrane and bending
        Strain[1] = (
            GeneralizedStrain[1] - z * GeneralizedStrain[3]
        )  # epsYY -> membrane and bending
        Strain[3] = GeneralizedStrain[2]  # 2epsXY
        Strain[4:6] = GeneralizedStrain[6:8]  # 2epsXZ and 2epsYZ -> shear

        return Strain

    def get_stress(self, **kargs):
        raise NameError('"GetStress" not implemented, contact developer.')


class ShellHomogeneous(ShellBase):
    def __init__(self, MatConstitutiveLaw, thickness, k=1, name=""):
        # assert get_Dimension() == '3D', "No 2D model for a shell kinematic. Choose '3D' problem dimension."

        if isinstance(MatConstitutiveLaw, str):
            MatConstitutiveLaw = ConstitutiveLaw.get_all()[MatConstitutiveLaw]

        ShellBase.__init__(self, thickness, k, name)  # heritage

        self.__material = MatConstitutiveLaw

    def GetMaterial(self):
        return self.__material

    def GetShellRigidityMatrix(self):
        Hplane = self.__material.get_elastic_matrix(
            "2Dstress"
        )  # membrane rigidity matrix with plane stress assumption
        Hplane = np.array(
            [[Hplane[i][j] for j in [0, 1, 3]] for i in [0, 1, 3]], dtype="object"
        )
        Hshear = self.__material.get_elastic_matrix()
        Hshear = np.array(
            [[Hshear[i][j] for j in [4, 5]] for i in [4, 5]], dtype="object"
        )

        H = np.zeros((8, 8), dtype="object")
        H[:3, :3] = self.GetThickness() * Hplane  # Membrane
        H[3:6, 3:6] = (
            self.GetThickness() ** 3 / 12
        ) * Hplane  # Flexual rigidity matrix
        H[6:8, 6:8] = (self.Get_k() * self.GetThickness()) * Hshear

        return H

    def GetShellRigidityMatrix_RI(self):
        # only shear component are given for reduce integration part
        Hshear = self.__material.get_elastic_matrix()
        Hshear = np.array(
            [[Hshear[i][j] for j in [4, 5]] for i in [4, 5]], dtype="object"
        )

        return (self.Get_k() * self.GetThickness()) * Hshear

    def GetShellRigidityMatrix_FI(self):
        # membrane and flexural component are given for full integration part
        Hplane = self.__material.get_elastic_matrix(
            "2Dstress"
        )  # membrane rigidity matrix with plane stress assumption
        Hplane = np.array(
            [[Hplane[i][j] for j in [0, 1, 3]] for i in [0, 1, 3]], dtype="object"
        )

        H = np.zeros((6, 6), dtype="object")
        H[:3, :3] = self.GetThickness() * Hplane  # Membrane
        H[3:6, 3:6] = (
            self.GetThickness() ** 3 / 12
        ) * Hplane  # Flexual rigidity matrix

        return H

    def get_stress(self, assembly, **kargs):
        Strain = self.get_strain(assembly, **kargs)
        Hplane = self.__material.get_elastic_matrix(
            "2Dstress"
        )  # membrane rigidity matrix with plane stress assumption
        Stress = [
            sum(
                [
                    (
                        0
                        if (np.isscalar(Strain[j]) and Strain[j] == 0)
                        else Strain[j] * Hplane[i][j]
                    )
                    for j in range(4)
                ]
            )
            for i in range(4)
        ]  # SXX, SYY, SXY (SZZ should be = 0)
        Hshear = self.__material.get_elastic_matrix()
        Stress += [
            sum(
                [
                    (
                        0
                        if (np.isscalar(Strain[j]) and Strain[j] == 0)
                        else Strain[j] * Hshear[i][j]
                    )
                    for j in [4, 5]
                ]
            )
            for i in [4, 5]
        ]  # SXX, SYY, SXY (SZZ should be = 0)

        return StressTensorList(Stress)

    def GetStressDistribution(self, assembly, pg, resolution=100):
        h = self.GetThickness()
        z = np.arange(-h / 2, h / 2, h / resolution)

        Strain = StrainTensorList([0 for i in range(6)])
        GeneralizedStrain = assembly.sv["GeneralizedStrain"]
        Strain[0] = (
            GeneralizedStrain[0][pg] + z * GeneralizedStrain[4][pg]
        )  # epsXX -> membrane and bending
        Strain[1] = (
            GeneralizedStrain[1][pg] - z * GeneralizedStrain[3][pg]
        )  # epsYY -> membrane and bending
        Strain[3] = GeneralizedStrain[2][pg] * np.ones_like(z)  # 2epsXY
        Strain[4] = GeneralizedStrain[6][pg] * np.ones_like(z)  # 2epsXZ -> shear
        Strain[5] = GeneralizedStrain[6][pg] * np.ones_like(z)  # 2epsYZ -> shear

        Hplane = self.__material.get_elastic_matrix(
            "2Dstress"
        )  # membrane rigidity matrix with plane stress assumption
        Stress = [
            sum(
                [
                    (
                        0
                        if (np.isscalar(Strain[j]) and Strain[j] == 0)
                        else Strain[j] * Hplane[i][j]
                    )
                    for j in range(4)
                ]
            )
            for i in range(4)
        ]  # SXX, SYY, SXY (SZZ should be = 0)
        Hshear = self.__material.get_elastic_matrix()
        Stress += [
            sum(
                [
                    (
                        0
                        if (np.isscalar(Strain[j]) and Strain[j] == 0)
                        else Strain[j] * Hshear[i][j]
                    )
                    for j in [4, 5]
                ]
            )
            for i in [4, 5]
        ]  # SXX, SYY, SXY (SZZ should be = 0)

        return z, Stress


class ShellLaminate(ShellBase):
    def __init__(self, listMat, listThickness, k=1, name=""):
        # assert get_Dimension() == '3D', "No 2D model for a shell kinematic. Choose '3D' problem dimension."

        self.__listMat = [
            ConstitutiveLaw.get_all()[mat] if isinstance(mat, str) else mat
            for mat in listMat
        ]
        thickness = sum(listThickness)  # total thickness

        self.__layer = (
            np.hstack((0, np.cumsum(listThickness))) - np.sum(listThickness) / 2
        )  # z coord of layers interfaces
        self.__listThickness = listThickness

        ShellBase.__init__(self, thickness, k, name)  # heritage

    def GetShellRigidityMatrix(self):
        H = np.zeros((8, 8), dtype="object")
        for i in range(len(self.__listThickness)):
            Hplane = self.__listMat[i].get_elastic_matrix(
                "2Dstress"
            )  # membrane rigidity matrix with plane stress assumption
            Hplane = np.array(
                [[Hplane[i][j] for j in [0, 1, 3]] for i in [0, 1, 3]], dtype="object"
            )
            Hshear = self.__listMat[i].get_elastic_matrix()
            Hshear = np.array(
                [[Hshear[i][j] for j in [4, 5]] for i in [4, 5]], dtype="object"
            )

            H[0:3, 0:3] += self.__listThickness[i] * Hplane  # Membrane
            H[0:3, 3:6] += (
                0.5 * (self.__layer[i + 1] ** 2 - self.__layer[i] ** 2) * Hplane
            )
            H[3:6, 0:3] += (
                0.5 * (self.__layer[i + 1] ** 2 - self.__layer[i] ** 2) * Hplane
            )
            H[3:6, 3:6] += (
                (1 / 3) * (self.__layer[i + 1] ** 3 - self.__layer[i] ** 3) * Hplane
            )  # Flexual rigidity matrix
            H[6:8, 6:8] += (self.Get_k() * self.__listThickness[i]) * Hshear

        return H

    def GetShellRigidityMatrix_RI(self):
        # only shear component are given for reduce integration part
        H = np.zeros((2, 2), dtype="object")
        for i in range(len(self.__listThickness)):
            Hshear = self.__listMat[i].get_elastic_matrix()
            Hshear = np.array(
                [[Hshear[i][j] for j in [4, 5]] for i in [4, 5]], dtype="object"
            )
            H += (self.Get_k() * self.__listThickness[i]) * Hshear

        return H

    def GetShellRigidityMatrix_FI(self):
        # membrane and flexural component are given for full integration part
        H = np.zeros((6, 6), dtype="object")
        for i in range(len(self.__listThickness)):
            Hplane = self.__listMat[i].get_elastic_matrix(
                "2Dstress"
            )  # membrane rigidity matrix with plane stress assumption
            Hplane = np.array(
                [[Hplane[i][j] for j in [0, 1, 3]] for i in [0, 1, 3]], dtype="object"
            )

            H[0:3, 0:3] += self.__listThickness[i] * Hplane  # Membrane
            H[0:3, 3:6] += (
                0.5 * (self.__layer[i + 1] ** 2 - self.__layer[i] ** 2) * Hplane
            )
            H[3:6, 0:3] += (
                0.5 * (self.__layer[i + 1] ** 2 - self.__layer[i] ** 2) * Hplane
            )
            H[3:6, 3:6] += (
                (1 / 3) * (self.__layer[i + 1] ** 3 - self.__layer[i] ** 3) * Hplane
            )  # Flexual rigidity matrix

        return H

    def get_stress(self, assembly, **kargs):
        Strain = self.get_strain(assembly, **kargs)
        position = kargs.get("position", 1)
        layer = self.find_layer(
            position
        )  # find the layer corresponding to the specified position

        Hplane = self.__listMat[layer].get_elastic_matrix(
            "2Dstress"
        )  # membrane rigidity matrix with plane stress assumption
        Stress = [
            sum(
                [
                    (
                        0
                        if (np.isscalar(Strain[j]) and Strain[j] == 0)
                        else Strain[j] * Hplane[i][j]
                    )
                    for j in range(4)
                ]
            )
            for i in range(4)
        ]  # SXX, SYY, SXY (SZZ should be = 0)
        Hshear = self.__listMat[layer].get_elastic_matrix()
        Stress += [
            sum(
                [
                    (
                        0
                        if (np.isscalar(Strain[j]) and Strain[j] == 0)
                        else Strain[j] * Hshear[i][j]
                    )
                    for j in [4, 5]
                ]
            )
            for i in [4, 5]
        ]  # SXX, SYY, SXY (SZZ should be = 0)

        return StressTensorList(Stress)

    def GetStressDistribution(self, assembly, pg, resolution=100):
        h = self.GetThickness()
        z = np.linspace(-h / 2, h / 2, resolution)

        Strain = StrainTensorList([0 for i in range(6)])
        GeneralizedStrain = assembly.sv["GeneralizedStrain"]

        Strain[0] = (
            GeneralizedStrain[0][pg] + z * GeneralizedStrain[4][pg]
        )  # epsXX -> membrane and bending
        Strain[1] = (
            GeneralizedStrain[1][pg] - z * GeneralizedStrain[3][pg]
        )  # epsYY -> membrane and bending
        Strain[3] = GeneralizedStrain[2][pg] * np.ones_like(z)  # 2epsXY
        Strain[4] = GeneralizedStrain[6][pg] * np.ones_like(z)  # 2epsXZ -> shear
        Strain[5] = GeneralizedStrain[6][pg] * np.ones_like(z)  # 2epsYZ -> shear

        layer_z = [
            list((pos - self.__layer) <= 0).index(True) - 1 for pos in z
        ]  # find the layer corresponding to all positions in z -> could be improved as z have increasing values
        layer_z[0] = 0  # to avoid -1 value for 1st layer

        Hplane = [
            mat.get_elastic_matrix("2Dstress") for mat in self.__listMat
        ]  # membrane rigidity matrix with plane stress assumption
        Hshear = [mat.get_elastic_matrix() for mat in self.__listMat]
        Hplane = [
            [
                [
                    0 if np.array_equal(Hplane[layer][i][j], 0) else Hplane[layer][i][j]
                    for layer in layer_z
                ]
                for j in range(4)
            ]
            for i in range(4)
        ]
        Hshear = [
            [
                [
                    0 if np.array_equal(Hshear[layer][i][j], 0) else Hshear[layer][i][j]
                    for layer in layer_z
                ]
                for j in [4, 5]
            ]
            for i in [4, 5]
        ]

        Stress = [
            sum(
                [
                    (
                        0
                        if (np.isscalar(Strain[j]) and Strain[j] == 0)
                        else Strain[j] * np.array(Hplane[i][j])
                    )
                    for j in range(4)
                ]
            )
            for i in range(4)
        ]  # SXX, SYY, SXY (SZZ should be = 0)
        Stress += [
            sum(
                [
                    (
                        0
                        if (np.isscalar(Strain[4 + j]) and Strain[4 + j] == 0)
                        else Strain[4 + j] * np.array(Hshear[i][j])
                    )
                    for j in range(2)
                ]
            )
            for i in range(2)
        ]  # SXX, SYY, SXY (SZZ should be = 0)
        return z, Stress

    def find_layer(self, position=1):
        """
        Returns the num of layer corresponding to the given position in the thickness

        Parameters
        ----------
        position : float
            Position in the thickness, given as a fraction of the demi thickness :
            z = position * total_thickness/2
            position = 1 for the top face (default)
            position = -1 for the bottom face
            position = 0 for the mid-plane

        Returns
        -------
        layer_id (int) : id of the layer at given position
        """
        assert (
            position >= -1 and position <= 1
        ), "position should be a float with value in [-1,1]"
        if position == -1:
            return 0  # 1st layer = bottom layer
        z = position * self.GetThickness() / 2
        return list((z - self.__layer) <= 0).index(True) - 1
