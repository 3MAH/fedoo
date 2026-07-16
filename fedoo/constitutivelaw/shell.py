# derive de ConstitutiveLaw
# compatible with the simcoon strain and stress notation

from fedoo.core.base import ConstitutiveLaw
from fedoo.util.voigt_tensors import StressTensorList, StrainTensorList

import numpy as np


class ShellBase(ConstitutiveLaw):
    # base model class that should derive any other shell constitutive laws
    def __init__(self, thickness, k=1, name=""):
        ConstitutiveLaw.__init__(self, name)  # heritage

        self.thickness = thickness
        """Shell thickness."""
        self.k = k
        """Shear shape factor."""

    def get_shell_stiffness_matrix(self):
        raise NameError(
            '"get_shell_stiffness_matrix" not implemented, contact developer.'
        )

    @property
    def area_density(self):
        """Mass per unit midsurface area."""
        if not hasattr(self, "_area_density"):
            self._area_density = self.compute_area_density()
        return self._area_density

    @property
    def rotary_density(self):
        """Mass moment of inertia per unit midsurface area."""
        if not hasattr(self, "_rotary_density"):
            self._rotary_density = self.compute_rotary_density()
        return self._rotary_density

    def compute_area_density(self):
        raise NotImplementedError(
            f"{type(self).__name__} does not define shell area density."
        )

    def compute_rotary_density(self):
        raise NotImplementedError(
            f"{type(self).__name__} does not define shell rotary density."
        )

    @staticmethod
    def _material_density(material, owner_name):
        density = getattr(material, "density", None)
        if density is None:
            material_name = getattr(material, "name", type(material).__name__)
            raise ValueError(
                f"Shell law {owner_name!r} needs density on material "
                f"{material_name!r} for dynamic analysis. Set it with "
                "material.set_density(rho), or attach storage explicitly "
                "with weakform.set_inertia(...)."
            )
        return density

    def update(self, assembly, pb):
        """Update shell stress from the strain computed by the weak form."""
        shell_strain = assembly.sv["ShellStrain"]
        if np.isscalar(shell_strain) and shell_strain == 0:
            assembly.sv["ShellStress"] = 0
            return

        H = self.get_shell_stiffness_matrix()
        assembly.sv["_ShellStiffnessMatrix"] = H
        stress = [
            sum(
                [
                    (
                        shell_strain[j] * assembly.convert_data(H[i][j])
                        if not np.array_equal(shell_strain[j], 0)
                        else 0
                    )
                    for j in range(8)
                ]
            )
            for i in range(8)
        ]
        assembly.sv["ShellStress"] = type(shell_strain)(stress)

    def get_strain(self, assembly, **kargs):
        """Return the last computed strain associated to the given assembly.

        Parameters
        ----------
        assembly: Assembly

        position : float (optional)
            Position in the thickness, given as a fraction of the demi
            thickness:
            z = position * total_thickness/2
            position = 1 for the top face (default)
            position = -1 for the bottom face
            position = 0 for the mid-plane

        Returns
        -------
        StrainTensorList object containing the strain at integration point
        """
        position = kargs.get("position", 1)
        z = position * self.thickness / 2

        Strain = StrainTensorList([0 for i in range(6)])
        ShellStrain = assembly.sv["ShellStrain"]
        if np.isscalar(ShellStrain) and ShellStrain == 0:
            zeros = np.zeros(assembly.n_gauss_points)
            return StrainTensorList([zeros.copy() for _ in range(6)])
        Strain[0] = ShellStrain[0] + z * ShellStrain[4]  # epsXX -> membrane and bending
        Strain[1] = ShellStrain[1] - z * ShellStrain[3]  # epsYY -> membrane and bending
        Strain[3] = ShellStrain[2]  # 2epsXY
        Strain[4:6] = ShellStrain[6:8]  # 2epsXZ and 2epsYZ -> shear

        return Strain

    def get_stress(self, **kargs):
        raise NameError('"GetStress" not implemented, contact developer.')


class ShellHomogeneous(ShellBase):
    def __init__(self, material, thickness, k=1, name=""):
        # k: shear shape factor

        if isinstance(material, str):
            material = ConstitutiveLaw.get_all()[material]

        ShellBase.__init__(self, thickness, k, name)  # heritage

        self.material = material

    def compute_area_density(self):
        density = self._material_density(self.material, self.name)
        return density * self.thickness

    def compute_rotary_density(self):
        density = self._material_density(self.material, self.name)
        return density * self.thickness**3 / 12.0

    def get_shell_stiffness_matrix(self):
        Hplane = self.material.get_elastic_matrix(
            "2Dstress"
        )  # membrane rigidity matrix with plane stress assumption
        Hplane = np.array(
            [[Hplane[i][j] for j in [0, 1, 3]] for i in [0, 1, 3]], dtype="object"
        )
        Hshear = self.material.get_elastic_matrix()
        Hshear = np.array(
            [[Hshear[i][j] for j in [4, 5]] for i in [4, 5]], dtype="object"
        )

        H = np.zeros((8, 8), dtype="object")
        H[:3, :3] = self.thickness * Hplane  # Membrane
        H[3:6, 3:6] = (self.thickness**3 / 12) * Hplane  # Flexual rigidity matrix
        H[6:8, 6:8] = (self.k * self.thickness) * Hshear

        return H

    def get_stress(self, assembly, **kargs):
        Strain = self.get_strain(assembly, **kargs)
        Hplane = self.material.get_elastic_matrix(
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
        Hshear = self.material.get_elastic_matrix()
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
        h = self.thickness
        z = np.arange(-h / 2, h / 2, h / resolution)

        Strain = StrainTensorList([0 for i in range(6)])
        ShellStrain = assembly.sv["ShellStrain"]
        Strain[0] = (
            ShellStrain[0][pg] + z * ShellStrain[4][pg]
        )  # epsXX -> membrane and bending
        Strain[1] = (
            ShellStrain[1][pg] - z * ShellStrain[3][pg]
        )  # epsYY -> membrane and bending
        Strain[3] = ShellStrain[2][pg] * np.ones_like(z)  # 2epsXY
        Strain[4] = ShellStrain[6][pg] * np.ones_like(z)  # 2epsXZ -> shear
        Strain[5] = ShellStrain[6][pg] * np.ones_like(z)  # 2epsYZ -> shear

        Hplane = self.material.get_elastic_matrix(
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
        Hshear = self.material.get_elastic_matrix()
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
    def __init__(self, listMat, list_thickness, k=1, name=""):
        # assert get_Dimension() == '3D', "No 2D model for a shell kinematic. Choose '3D' problem dimension."

        self.__listMat = [
            ConstitutiveLaw.get_all()[mat] if isinstance(mat, str) else mat
            for mat in listMat
        ]
        thickness = sum(list_thickness)  # total thickness

        self.__layer = (
            np.hstack((0, np.cumsum(list_thickness))) - np.sum(list_thickness) / 2
        )  # z coord of layers interfaces
        self.list_thickness = list_thickness

        ShellBase.__init__(self, thickness, k, name)  # heritage

    def compute_area_density(self):
        return sum(
            self._material_density(material, self.name) * thickness
            for material, thickness in zip(self.__listMat, self.list_thickness)
        )

    def compute_rotary_density(self):
        return sum(
            self._material_density(material, self.name)
            * (self.__layer[i + 1] ** 3 - self.__layer[i] ** 3)
            / 3.0
            for i, material in enumerate(self.__listMat)
        )

    def get_shell_stiffness_matrix(self):
        H = np.zeros((8, 8), dtype="object")
        for i in range(len(self.list_thickness)):
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

            H[0:3, 0:3] += self.list_thickness[i] * Hplane  # Membrane
            H[0:3, 3:6] += (
                0.5 * (self.__layer[i + 1] ** 2 - self.__layer[i] ** 2) * Hplane
            )
            H[3:6, 0:3] += (
                0.5 * (self.__layer[i + 1] ** 2 - self.__layer[i] ** 2) * Hplane
            )
            H[3:6, 3:6] += (
                (1 / 3) * (self.__layer[i + 1] ** 3 - self.__layer[i] ** 3) * Hplane
            )  # Flexual rigidity matrix
            H[6:8, 6:8] += (self.k * self.list_thickness[i]) * Hshear

        return H

    def get_shell_stiffness_matrix_RI(self):
        # only shear component are given for reduce integration part
        H = np.zeros((2, 2), dtype="object")
        for i in range(len(self.list_thickness)):
            Hshear = self.__listMat[i].get_elastic_matrix()
            Hshear = np.array(
                [[Hshear[i][j] for j in [4, 5]] for i in [4, 5]], dtype="object"
            )
            H += (self.k * self.list_thickness[i]) * Hshear

        return H

    def get_shell_stiffness_matrix_FI(self):
        # membrane and flexural component are given for full integration part
        H = np.zeros((6, 6), dtype="object")
        for i in range(len(self.list_thickness)):
            Hplane = self.__listMat[i].get_elastic_matrix(
                "2Dstress"
            )  # membrane rigidity matrix with plane stress assumption
            Hplane = np.array(
                [[Hplane[i][j] for j in [0, 1, 3]] for i in [0, 1, 3]], dtype="object"
            )

            H[0:3, 0:3] += self.list_thickness[i] * Hplane  # Membrane
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
        h = self.thickness
        z = np.linspace(-h / 2, h / 2, resolution)

        Strain = StrainTensorList([0 for i in range(6)])
        ShellStrain = assembly.sv["ShellStrain"]

        Strain[0] = (
            ShellStrain[0][pg] + z * ShellStrain[4][pg]
        )  # epsXX -> membrane and bending
        Strain[1] = (
            ShellStrain[1][pg] - z * ShellStrain[3][pg]
        )  # epsYY -> membrane and bending
        Strain[3] = ShellStrain[2][pg] * np.ones_like(z)  # 2epsXY
        Strain[4] = ShellStrain[6][pg] * np.ones_like(z)  # 2epsXZ -> shear
        Strain[5] = ShellStrain[6][pg] * np.ones_like(z)  # 2epsYZ -> shear

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
        """Return the id of layer at a given position in the thickness.

        Parameters
        ----------
        position : float
            Position in the thickness, given as a fraction of the demi
            thickness :
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
        z = position * self.thickness / 2
        return list((z - self.__layer) <= 0).index(True) - 1
