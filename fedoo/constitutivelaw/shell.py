from fedoo.util.shell_recovery import (
    sample_positions,
    validate_positions,
    interpolate_tensors,
    generalized_tensors,
)

# derive de ConstitutiveLaw
# compatible with the simcoon strain and stress notation

from fedoo.core.base import ConstitutiveLaw
from fedoo.util.voigt_tensors import StressTensorList, StrainTensorList

import copy
import numpy as np


class ShellBase(ConstitutiveLaw):
    # base model class that should derive any other shell constitutive laws
    def __init__(self, thickness, k=1, name="", *, output_points=None, n_points=3):
        ConstitutiveLaw.__init__(self, name)  # heritage

        self.thickness = thickness
        """Shell thickness."""
        self.k = k
        """Shear shape factor."""
        self.n_points = n_points
        self.output_points = output_points

    @property
    def output_points(self):
        """Normalized thickness positions saved when output position is None.

        Nonlinear shells default to their actual material integration points;
        linear laminates include both sides of every layer interface.
        """
        if self._output_points is not None:
            return self._output_points.copy()
        if self._recovery_model != "homogeneous_linear":
            return self._integration_sampling()[0]

        return sample_positions(self.section_description(), self.n_points)[0]

    @output_points.setter
    def output_points(self, points):
        self._output_points = None if points is None else validate_positions(points)

    def _integration_sampling(self):
        if hasattr(self, "_layer_z"):
            points = np.concatenate(self._layer_z) * 2 / self.thickness
            layers = np.concatenate(
                [np.full(len(z), i) for i, z in enumerate(self._layer_z)]
            )
            interfaces = self._interfaces * 2 / self.thickness
            interfaces[[0, -1]] = [-1, 1]
            return points, layers, interfaces
        if hasattr(self, "_z"):
            return (
                self._z * 2 / self.thickness,
                np.zeros(len(self._z), dtype=int),
                np.array([-1.0, 1.0]),
            )
        # Linear laminates need the two one-sided limits at every interface.
        interfaces = (
            np.concatenate(([0.0], np.cumsum(self.list_thickness))) * 2 / self.thickness
            - 1
        )
        interfaces[[0, -1]] = [-1, 1]
        points = np.column_stack((interfaces[:-1], interfaces[1:])).ravel()
        return points, np.repeat(np.arange(len(self.list_thickness)), 2), interfaces

    def section_description(self, assembly=None):
        """Numeric, material-independent information for saved shell recovery."""
        thickness = np.asarray(self.thickness)
        if assembly is not None and thickness.ndim:
            thickness = assembly.convert_data(thickness)
        description = dict(
            family="shell",
            version=1,
            model=self._recovery_model,
            thickness=thickness,
            k=self.k,
            n_points=self.n_points,
        )
        if assembly is not None:
            description["n_elm_gp"] = assembly.n_elm_gp
        if self._recovery_model != "homogeneous_linear":
            points, layers, interfaces = self._integration_sampling()
            description.update(
                integration_points=points,
                integration_layers=layers,
                interfaces=interfaces,
            )
        return description

    def integration_stresses(self, assembly):
        """Actual thickness stresses, including elastic transverse shear."""
        if self._recovery_model == "laminate_linear":
            points, layers, _ = self._integration_sampling()
            tensors = []
            for point, layer in zip(points, layers):
                # Use each layer's own material, including at shared interfaces.
                strain = self.get_strain(assembly, position=float(point)).asarray()
                material = self._ShellLaminate__list_mat[layer]
                tensors.append(_elastic_shell_stress(material, assembly, strain))
            return np.stack(tensors, axis=1)
        if hasattr(self, "_material_assemblies"):
            if self._material_assemblies is None:
                raise RuntimeError("The shell law has not been initialized")
            blocks = []
            for layer, material_assembly in enumerate(self._material_assemblies):
                block = (
                    material_assembly.sv["Stress"]
                    .asarray()
                    .reshape(6, -1, assembly.n_gauss_points)
                    .copy()
                )
                shear = self._shear_matrices[layer]
                blocks.append((block, shear))
        else:
            if self._material_assembly is None:
                raise RuntimeError("The shell law has not been initialized")
            block = (
                self._material_assembly.sv["Stress"]
                .asarray()
                .reshape(6, -1, assembly.n_gauss_points)
                .copy()
            )
            blocks = [(block, self._elastic_shear_matrix())]
        strain = self.get_strain(assembly, position=0).asarray()
        for block, shear in blocks:
            block[4:6] = (shear @ strain[4:6])[:, None, :]
        return np.concatenate([block for block, _ in blocks], axis=1)

    def get_output_tensors(self, assembly, kind, position=None):
        """Return (6, thickness point, shell GP) tensors and point metadata."""

        points = (
            self.output_points if position is None else validate_positions(position)
        )
        metadata = self.section_description(assembly)
        metadata["stored_points"] = points
        if self._recovery_model != "homogeneous_linear":
            native, layers, interfaces = self._integration_sampling()
            metadata["point_layers"] = (
                layers
                if position is None and self._output_points is None
                else np.clip(
                    np.searchsorted(interfaces, points, side="left") - 1,
                    0,
                    len(interfaces) - 2,
                )
            )
            if (
                position is None
                and self._output_points is None
                and self._recovery_model.endswith("nonlinear")
            ):
                metadata["sampling"] = "integration"
        if kind == "Strain":
            tensors = self.get_strain(assembly, position=points).asarray()
        elif self._recovery_model == "homogeneous_linear":
            tensors = self.get_stress(assembly, position=points).asarray()
        else:
            raw = self.integration_stresses(assembly)
            native, layers, interfaces = self._integration_sampling()
            if position is None and self._output_points is None:
                tensors = raw
                metadata["point_layers"] = layers
            else:
                tensors = interpolate_tensors(
                    raw, native, points, layers, interfaces, allow_constant=True
                )
                metadata["point_layers"] = np.clip(
                    np.searchsorted(interfaces, points, side="left") - 1,
                    0,
                    len(interfaces) - 2,
                )
        cls = StrainTensorList if kind == "Strain" else StressTensorList
        return cls(tensors), metadata

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
        if position is None:
            position = self.output_points
        values = assembly.sv["ShellStrain"]
        values = np.array(
            [
                np.broadcast_to(c, (assembly.n_gauss_points,))
                for c in ([0] * 8 if np.isscalar(values) else values)
            ]
        )
        tensors = generalized_tensors(
            values, self.section_description(assembly), "Strain", position
        )
        return StrainTensorList(tensors[:, 0] if np.isscalar(position) else tensors)

    def get_stress(self, assembly, **kargs):
        position = kargs.get("position", 1)
        tensors, _ = self.get_output_tensors(assembly, "Stress", position)
        return (
            StressTensorList(tensors.asarray()[:, 0])
            if np.isscalar(position)
            else tensors
        )


def _elastic_shell_stress(material, assembly, strain):
    plane = material.get_elastic_matrix("2Dstress")
    shear = material.get_elastic_matrix()
    result = np.zeros_like(strain)
    for i in (0, 1, 3, 4, 5):
        matrix = shear if i >= 4 else plane
        indices = (4, 5) if i >= 4 else (0, 1, 3)
        result[i] = sum(
            strain[j] * assembly.convert_data(matrix[i][j]) for j in indices
        )
    return result


class ShellHomogeneous(ShellBase):
    _recovery_model = "homogeneous_linear"

    def __init__(
        self, material, thickness, k=1, name="", *, output_points=None, n_points=3
    ):
        # k: shear shape factor

        if isinstance(material, str):
            material = ConstitutiveLaw.get_all()[material]

        ShellBase.__init__(
            self, thickness, k, name, output_points=output_points, n_points=n_points
        )

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
        strain = self.get_strain(assembly, **kargs).asarray()
        return StressTensorList(_elastic_shell_stress(self.material, assembly, strain))

    def get_stress_distribution(self, assembly, pg, resolution=100):
        h = self.thickness
        z = np.arange(-h / 2, h / 2, h / resolution)

        Strain = StrainTensorList([0 for i in range(6)])
        ShellStrain = assembly.sv["ShellStrain"]
        Strain[0] = (
            ShellStrain[0][pg] + z * ShellStrain[3][pg]
        )  # epsXX -> membrane and bending
        Strain[1] = (
            ShellStrain[1][pg] + z * ShellStrain[4][pg]
        )  # epsYY -> membrane and bending
        Strain[3] = (
            ShellStrain[2][pg] + z * ShellStrain[5][pg]
        )  # 2epsXY -> membrane and twist
        Strain[4] = ShellStrain[6][pg] * np.ones_like(z)  # 2epsXZ -> shear
        Strain[5] = ShellStrain[7][pg] * np.ones_like(z)  # 2epsYZ -> shear

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


class _ShellMaterialPointAssembly:
    """Minimal assembly interface used by through-thickness material points."""

    class _MaterialSpace:
        def __init__(self, dimension):
            self._dimension = dimension

        def get_dimension(self):
            return self._dimension

        @property
        def ndim(self):
            return 2 if self._dimension == "2Dstress" else 3

        @staticmethod
        def list_variables():
            return ()

    def __init__(self, n_gauss_points, dimension="2Dstress"):
        self.n_gauss_points = n_gauss_points
        self.space = self._MaterialSpace(dimension)
        self.sv = {
            "Strain": StrainTensorList(np.zeros((6, n_gauss_points))),
            "Stress": StressTensorList(np.zeros((6, n_gauss_points))),
        }
        self.sv_start = dict(self.sv)
        self.sv_component = {}
        self._nlgeom = False

    def convert_data(self, data, *args, **kwargs):
        """Return pointwise material data without mesh interpolation."""
        return data


def _copy_state(state):
    """Deep-ish copy of a state-variable dict (shared by the nonlinear shells)."""
    copied = {}
    for key, value in state.items():
        if hasattr(value, "copy"):
            copied[key] = value.copy()
        else:
            copied[key] = copy.deepcopy(value)
    return copied


def _tangent_array(tangent, n_points):
    """Normalize scalar/pointwise tangent entries to a dense (6, 6, n) array."""
    dense = np.empty((6, 6, n_points), dtype=float)
    for row in range(6):
        for column in range(6):
            dense[row, column] = tangent[row][column]
    return dense


def _strain_matrix(z):
    """Membrane+bending strain-interpolation matrix at through-thickness ``z``."""
    matrix = np.zeros((6, 8))
    matrix[0, 0] = 1
    matrix[0, 3] = z
    matrix[1, 1] = 1
    matrix[1, 4] = z
    matrix[3, 2] = 1
    matrix[3, 5] = z
    return matrix


class ShellHomogeneousNonLinear(ShellBase):
    """Homogeneous shell integrated from nonlinear plane-stress material points.

    The membrane and bending response is obtained by calling a private copy
    of ``material`` at Gauss--Legendre points through the thickness. The
    material must support the standard Fedoo ``"2Dstress"`` constitutive-law
    interface. Transverse shear retains the elastic Reissner--Mindlin
    treatment used by :class:`ShellHomogeneous`.

    This implementation is restricted to small-strain material laws. For
    plasticity (or any law needing a plane-stress return mapping) use a Simcoon
    law, e.g. ``fedoo.constitutivelaw.Simcoon("EPICP", props)``; the pedagogical
    :class:`ElastoPlasticity` is 3D-only and does not support ``"2Dstress"``.

    Parameters
    ----------
    material : ConstitutiveLaw or str
        Plane-stress-compatible material law or its registered name.
    thickness : float
        Shell thickness.
    n_thickness_points : int, default=5
        Number of Gauss--Legendre points through the thickness.
    k : float, default=1
        Transverse shear correction factor.
    name : str, optional
        Name of the shell constitutive law.
    """

    _recovery_model = "homogeneous_nonlinear"

    def __init__(
        self,
        material,
        thickness,
        n_thickness_points=5,
        k=1,
        name="",
        *,
        output_points=None,
        n_points=3,
    ):
        if isinstance(material, str):
            material = ConstitutiveLaw.get_all()[material]
        if n_thickness_points < 1:
            raise ValueError("n_thickness_points must be at least one.")

        super().__init__(
            thickness, k, name, output_points=output_points, n_points=n_points
        )
        self.material = copy.deepcopy(material)
        self._shear_material = copy.deepcopy(material)
        self.n_thickness_points = n_thickness_points
        points, weights = np.polynomial.legendre.leggauss(n_thickness_points)
        self._z = points * thickness / 2
        self._weights = weights * thickness / 2
        self._material_assembly = None
        self._n_shell_points = None
        self._shear_matrix = None

    def compute_area_density(self):
        density = self._material_density(self.material, self.name)
        return density * self.thickness

    def compute_rotary_density(self):
        density = self._material_density(self.material, self.name)
        return density * self.thickness**3 / 12.0

    def _elastic_shear_matrix(self):
        if self._shear_matrix is not None:
            return self._shear_matrix
        if not hasattr(self.material, "get_elastic_matrix"):
            raise RuntimeError(
                "The shell law must be initialized before its transverse "
                "shear stiffness can be requested."
            )
        elastic = np.asarray(self.material.get_elastic_matrix("3D"), dtype=float)
        return elastic[np.ix_([4, 5], [4, 5])]

    def get_shell_stiffness_matrix(self):
        if self._material_assembly is not None:
            return self._shell_tangent

        plane = np.asarray(self.material.get_elastic_matrix("2Dstress"), dtype=float)
        tangent = np.zeros((8, 8))
        for z, weight in zip(self._z, self._weights):
            strain_matrix = _strain_matrix(z)
            tangent += weight * strain_matrix.T @ plane @ strain_matrix
        tangent[6:8, 6:8] = self.k * self.thickness * self._elastic_shear_matrix()
        return tangent

    def initialize(self, assembly, pb):
        self._n_shell_points = assembly.n_gauss_points
        n_material_points = self.n_thickness_points * self._n_shell_points
        self._material_assembly = _ShellMaterialPointAssembly(n_material_points)
        shear_assembly = _ShellMaterialPointAssembly(1, dimension="3D")
        self.material.reset()
        self._shear_material.reset()
        self.material.initialize(self._material_assembly, pb)
        self._shear_material.initialize(shear_assembly, pb)
        shear_tangent = _tangent_array(shear_assembly.sv["TangentMatrix"], 1)
        shear_tangent = shear_tangent[:, :, 0]
        self._shear_matrix = shear_tangent[np.ix_([4, 5], [4, 5])]
        self._material_assembly.sv_start = _copy_state(self._material_assembly.sv)

        assembly.sv["_ShellStiffnessMatrix"] = self._integrate_tangent()
        assembly.sv["ShellStress"] = 0

    def _material_strain(self, shell_strain):
        shell_array = np.asarray(
            [
                (
                    np.zeros(self._n_shell_points)
                    if np.isscalar(component)
                    else component
                )
                for component in shell_strain
            ]
        )
        strains = [_strain_matrix(z) @ shell_array for z in self._z]
        return StrainTensorList(np.concatenate(strains, axis=1))

    def _integrate_tangent(self):
        material_tangent = _tangent_array(
            self._material_assembly.sv["TangentMatrix"],
            self.n_thickness_points * self._n_shell_points,
        )

        shell_tangent = np.zeros((8, 8, self._n_shell_points))
        for index, (z, weight) in enumerate(zip(self._z, self._weights)):
            point_slice = slice(
                index * self._n_shell_points,
                (index + 1) * self._n_shell_points,
            )
            strain_matrix = _strain_matrix(z)
            shell_tangent += weight * np.einsum(
                "ia,ijp,jb->abp",
                strain_matrix,
                material_tangent[:, :, point_slice],
                strain_matrix,
            )

        shell_tangent[6:8, 6:8] = (
            self.k * self.thickness * self._elastic_shear_matrix()[:, :, None]
        )
        self._shell_tangent = shell_tangent
        return shell_tangent

    def _integrate_stress(self, shell_strain):
        material_stress = self._material_assembly.sv["Stress"].asarray()
        resultants = np.zeros((8, self._n_shell_points))
        for index, (z, weight) in enumerate(zip(self._z, self._weights)):
            point_slice = slice(
                index * self._n_shell_points,
                (index + 1) * self._n_shell_points,
            )
            resultants += weight * (
                _strain_matrix(z).T @ material_stress[:, point_slice]
            )

        shear_strain = np.asarray(
            [
                (
                    np.zeros(self._n_shell_points)
                    if np.isscalar(shell_strain[index])
                    else shell_strain[index]
                )
                for index in [6, 7]
            ]
        )
        resultants[6:8] = (
            self.k * self.thickness * self._elastic_shear_matrix() @ shear_strain
        )
        return type(shell_strain)(resultants)

    def update(self, assembly, pb):
        shell_strain = assembly.sv["ShellStrain"]
        if np.isscalar(shell_strain) and shell_strain == 0:
            shell_strain = [np.zeros(self._n_shell_points) for _ in range(8)]

        self._material_assembly.sv["Strain"] = self._material_strain(shell_strain)
        self.material.update(self._material_assembly, pb)
        assembly.sv["_ShellStiffnessMatrix"] = self._integrate_tangent()
        assembly.sv["ShellStress"] = self._integrate_stress(shell_strain)

    def set_start(self, assembly, pb):
        self.material.set_start(self._material_assembly, pb)
        self._material_assembly.sv_start = _copy_state(self._material_assembly.sv)

    def to_start(self, assembly, pb):
        self.material.to_start(self._material_assembly, pb)
        self._material_assembly.sv = _copy_state(self._material_assembly.sv_start)

    def get_stress_distribution(self, assembly, pg, resolution=None):
        """Return stresses through the thickness at one shell Gauss point.

        By default, stresses are returned at the constitutive integration
        points. If ``resolution`` is given, every stress component is
        interpolated onto that many equally spaced positions for
        visualization.
        """
        if self._material_assembly is None:
            raise RuntimeError("The shell law has not been initialized.")
        if not 0 <= pg < self._n_shell_points:
            raise IndexError("pg is outside the shell Gauss-point range.")

        stress = (
            self._material_assembly.sv["Stress"]
            .asarray()[
                :,
                pg :: self._n_shell_points,
            ]
            .copy()
        )
        shell_strain = assembly.sv["ShellStrain"]
        shear_strain = np.array(
            [
                (0.0 if np.isscalar(shell_strain[index]) else shell_strain[index][pg])
                for index in [6, 7]
            ]
        )
        stress[4:6] = self._elastic_shear_matrix() @ shear_strain[:, None]

        z = self._z.copy()
        if resolution is not None:
            if resolution < 2:
                raise ValueError("resolution must be at least two.")
            output_z = np.linspace(
                -self.thickness / 2,
                self.thickness / 2,
                resolution,
            )
            stress = np.array(
                [np.interp(output_z, z, component) for component in stress]
            )
            z = output_z
        return z, StressTensorList(stress)


class ShellLaminateNonLinear(ShellBase):
    """Layered shell integrated from nonlinear plane-stress material points.

    Each layer owns an independent copy of its material law and its internal
    variables. Membrane and bending stresses and tangents are integrated at
    Gauss--Legendre points in every layer. Transverse shear is treated
    elastically using the three-dimensional initial tangent of each material
    and the shell correction factor ``k``.

    This implementation is restricted to small-strain material laws. For
    plasticity (or any law needing a plane-stress return mapping) use a Simcoon
    law, e.g. ``fedoo.constitutivelaw.Simcoon("EPICP", props)``; the pedagogical
    :class:`ElastoPlasticity` is 3D-only and does not support ``"2Dstress"``.

    Parameters
    ----------
    list_mat : sequence of ConstitutiveLaw or str
        Material law, or registered material name, for every layer, ordered
        from the bottom to the top surface.
    list_thickness : sequence of float
        Thickness of every layer.
    n_thickness_points : int or sequence of int, default=3
        Number of Gauss--Legendre points in each layer.
    k : float, default=1
        Transverse shear correction factor.
    name : str, optional
        Name of the shell constitutive law.
    """

    _recovery_model = "laminate_nonlinear"

    def __init__(
        self,
        list_mat,
        list_thickness,
        n_thickness_points=3,
        k=1,
        name="",
        *,
        output_points=None,
        n_points=3,
    ):
        if len(list_mat) != len(list_thickness):
            raise ValueError("list_mat and list_thickness must have the same length.")
        if len(list_mat) == 0:
            raise ValueError("At least one laminate layer is required.")
        if np.any(np.asarray(list_thickness) <= 0):
            raise ValueError("Every layer thickness must be positive.")

        materials = [
            (
                ConstitutiveLaw.get_all()[material]
                if isinstance(material, str)
                else material
            )
            for material in list_mat
        ]
        if np.isscalar(n_thickness_points):
            point_counts = [int(n_thickness_points)] * len(materials)
        else:
            point_counts = [int(value) for value in n_thickness_points]
            if len(point_counts) != len(materials):
                raise ValueError(
                    "n_thickness_points must be an integer or contain one "
                    "value per layer."
                )
        if any(value < 1 for value in point_counts):
            raise ValueError("Every layer must have at least one thickness point.")

        thickness = float(np.sum(list_thickness))
        super().__init__(
            thickness, k, name, output_points=output_points, n_points=n_points
        )
        self.materials = [copy.deepcopy(material) for material in materials]
        self._shear_materials = [copy.deepcopy(material) for material in materials]
        self.list_thickness = np.asarray(list_thickness, dtype=float)
        self.n_thickness_points = point_counts
        self._interfaces = (
            np.concatenate(([0.0], np.cumsum(self.list_thickness))) - thickness / 2
        )
        self._layer_z = []
        self._layer_weights = []
        for index, point_count in enumerate(point_counts):
            points, weights = np.polynomial.legendre.leggauss(point_count)
            lower = self._interfaces[index]
            upper = self._interfaces[index + 1]
            self._layer_z.append((lower + upper) / 2 + points * (upper - lower) / 2)
            self._layer_weights.append(weights * (upper - lower) / 2)

        self._material_assemblies = None
        self._shear_matrices = None
        self._n_shell_points = None

    def compute_area_density(self):
        return sum(
            self._material_density(material, self.name) * thickness
            for material, thickness in zip(self.materials, self.list_thickness)
        )

    def compute_rotary_density(self):
        return sum(
            self._material_density(material, self.name)
            * (self._interfaces[index + 1] ** 3 - self._interfaces[index] ** 3)
            / 3
            for index, material in enumerate(self.materials)
        )

    def _initial_shear_matrix(self, index):
        if self._shear_matrices is not None:
            return self._shear_matrices[index]
        material = self.materials[index]
        if not hasattr(material, "get_elastic_matrix"):
            raise RuntimeError(
                "The laminate must be initialized before its transverse "
                "shear stiffness can be requested."
            )
        elastic = np.asarray(material.get_elastic_matrix("3D"), dtype=float)
        return elastic[np.ix_([4, 5], [4, 5])]

    def get_shell_stiffness_matrix(self):
        if self._material_assemblies is not None:
            return self._shell_tangent

        tangent = np.zeros((8, 8))
        for index, material in enumerate(self.materials):
            plane = np.asarray(material.get_elastic_matrix("2Dstress"), dtype=float)
            for z, weight in zip(self._layer_z[index], self._layer_weights[index]):
                strain_matrix = _strain_matrix(z)
                tangent += weight * strain_matrix.T @ plane @ strain_matrix
            tangent[6:8, 6:8] += (
                self.k * self.list_thickness[index] * self._initial_shear_matrix(index)
            )
        return tangent

    def initialize(self, assembly, pb):
        self._n_shell_points = assembly.n_gauss_points
        self._material_assemblies = []
        self._shear_matrices = []

        for material, shear_material, point_count in zip(
            self.materials,
            self._shear_materials,
            self.n_thickness_points,
        ):
            material_assembly = _ShellMaterialPointAssembly(
                point_count * self._n_shell_points
            )
            shear_assembly = _ShellMaterialPointAssembly(1, dimension="3D")
            material.reset()
            shear_material.reset()
            material.initialize(material_assembly, pb)
            shear_material.initialize(shear_assembly, pb)

            shear_tangent = _tangent_array(shear_assembly.sv["TangentMatrix"], 1)
            shear_tangent = shear_tangent[:, :, 0]
            self._shear_matrices.append(shear_tangent[np.ix_([4, 5], [4, 5])])
            material_assembly.sv_start = _copy_state(material_assembly.sv)
            self._material_assemblies.append(material_assembly)

        assembly.sv["_ShellStiffnessMatrix"] = self._integrate_tangent()
        assembly.sv["ShellStress"] = 0

    def _material_strain(self, shell_strain, layer):
        shell_array = np.asarray(
            [
                (
                    np.zeros(self._n_shell_points)
                    if np.isscalar(component)
                    else component
                )
                for component in shell_strain
            ]
        )
        strains = [_strain_matrix(z) @ shell_array for z in self._layer_z[layer]]
        return StrainTensorList(np.concatenate(strains, axis=1))

    def _integrate_tangent(self):
        shell_tangent = np.zeros((8, 8, self._n_shell_points))
        for layer, material_assembly in enumerate(self._material_assemblies):
            material_tangent = _tangent_array(
                material_assembly.sv["TangentMatrix"],
                self.n_thickness_points[layer] * self._n_shell_points,
            )

            for point, (z, weight) in enumerate(
                zip(
                    self._layer_z[layer],
                    self._layer_weights[layer],
                )
            ):
                point_slice = slice(
                    point * self._n_shell_points,
                    (point + 1) * self._n_shell_points,
                )
                strain_matrix = _strain_matrix(z)
                shell_tangent += weight * np.einsum(
                    "ia,ijp,jb->abp",
                    strain_matrix,
                    material_tangent[:, :, point_slice],
                    strain_matrix,
                )

            shell_tangent[6:8, 6:8] += (
                self.k
                * self.list_thickness[layer]
                * self._shear_matrices[layer][:, :, None]
            )

        self._shell_tangent = shell_tangent
        return shell_tangent

    def _integrate_stress(self, shell_strain):
        resultants = np.zeros((8, self._n_shell_points))
        for layer, material_assembly in enumerate(self._material_assemblies):
            material_stress = material_assembly.sv["Stress"].asarray()
            for point, (z, weight) in enumerate(
                zip(
                    self._layer_z[layer],
                    self._layer_weights[layer],
                )
            ):
                point_slice = slice(
                    point * self._n_shell_points,
                    (point + 1) * self._n_shell_points,
                )
                resultants += weight * (
                    _strain_matrix(z).T @ material_stress[:, point_slice]
                )

        shear_strain = np.asarray(
            [
                (
                    np.zeros(self._n_shell_points)
                    if np.isscalar(shell_strain[index])
                    else shell_strain[index]
                )
                for index in [6, 7]
            ]
        )
        shear_stiffness = sum(
            thickness * matrix
            for thickness, matrix in zip(self.list_thickness, self._shear_matrices)
        )
        resultants[6:8] = self.k * shear_stiffness @ shear_strain
        return type(shell_strain)(resultants)

    def update(self, assembly, pb):
        shell_strain = assembly.sv["ShellStrain"]
        if np.isscalar(shell_strain) and shell_strain == 0:
            shell_strain = [np.zeros(self._n_shell_points) for _ in range(8)]

        for layer, (material, material_assembly) in enumerate(
            zip(self.materials, self._material_assemblies)
        ):
            material_assembly.sv["Strain"] = self._material_strain(shell_strain, layer)
            material.update(material_assembly, pb)

        assembly.sv["_ShellStiffnessMatrix"] = self._integrate_tangent()
        assembly.sv["ShellStress"] = self._integrate_stress(shell_strain)

    def set_start(self, assembly, pb):
        for material, material_assembly in zip(
            self.materials, self._material_assemblies
        ):
            material.set_start(material_assembly, pb)
            material_assembly.sv_start = _copy_state(material_assembly.sv)

    def to_start(self, assembly, pb):
        for material, material_assembly in zip(
            self.materials, self._material_assemblies
        ):
            material.to_start(material_assembly, pb)
            material_assembly.sv = _copy_state(material_assembly.sv_start)

    def get_stress_distribution(self, assembly, pg, resolution=None):
        """Return layerwise stresses at one shell Gauss point.

        Without ``resolution``, values are returned at the actual material
        integration points. With ``resolution``, each layer is sampled
        independently so stress jumps at material interfaces are preserved.
        """
        if self._material_assemblies is None:
            raise RuntimeError("The shell law has not been initialized.")
        if not 0 <= pg < self._n_shell_points:
            raise IndexError("pg is outside the shell Gauss-point range.")
        if resolution is not None and resolution < 2:
            raise ValueError("resolution must be at least two.")

        shell_strain = assembly.sv["ShellStrain"]
        shear_strain = np.array(
            [
                (0.0 if np.isscalar(shell_strain[index]) else shell_strain[index][pg])
                for index in [6, 7]
            ]
        )
        z_values = []
        stress_values = []
        for layer, material_assembly in enumerate(self._material_assemblies):
            layer_stress = (
                material_assembly.sv["Stress"]
                .asarray()[
                    :,
                    pg :: self._n_shell_points,
                ]
                .copy()
            )
            layer_stress[4:6] = self._shear_matrices[layer] @ shear_strain[:, None]
            layer_z = self._layer_z[layer]

            if resolution is not None:
                output_z = np.linspace(
                    self._interfaces[layer],
                    self._interfaces[layer + 1],
                    resolution,
                )
                layer_stress = np.array(
                    [
                        np.interp(output_z, layer_z, component)
                        for component in layer_stress
                    ]
                )
                layer_z = output_z

            z_values.append(layer_z)
            stress_values.append(layer_stress)

        return (
            np.concatenate(z_values),
            StressTensorList(np.concatenate(stress_values, axis=1)),
        )


class ShellLaminate(ShellBase):
    _recovery_model = "laminate_linear"

    def __init__(
        self, list_mat, list_thickness, k=1, name="", *, output_points=None, n_points=3
    ):
        # assert get_Dimension() == '3D', "No 2D model for a shell kinematic. Choose '3D' problem dimension."

        self.__list_mat = [
            ConstitutiveLaw.get_all()[mat] if isinstance(mat, str) else mat
            for mat in list_mat
        ]
        thickness = sum(list_thickness)  # total thickness

        self.__layer = (
            np.hstack((0, np.cumsum(list_thickness))) - np.sum(list_thickness) / 2
        )  # z coord of layers interfaces
        self.list_thickness = list_thickness

        ShellBase.__init__(
            self, thickness, k, name, output_points=output_points, n_points=n_points
        )

    def compute_area_density(self):
        return sum(
            self._material_density(material, self.name) * thickness
            for material, thickness in zip(self.__list_mat, self.list_thickness)
        )

    def compute_rotary_density(self):
        return sum(
            self._material_density(material, self.name)
            * (self.__layer[i + 1] ** 3 - self.__layer[i] ** 3)
            / 3.0
            for i, material in enumerate(self.__list_mat)
        )

    def get_shell_stiffness_matrix(self):
        H = np.zeros((8, 8), dtype="object")
        for i in range(len(self.list_thickness)):
            Hplane = self.__list_mat[i].get_elastic_matrix(
                "2Dstress"
            )  # membrane rigidity matrix with plane stress assumption
            Hplane = np.array(
                [[Hplane[i][j] for j in [0, 1, 3]] for i in [0, 1, 3]], dtype="object"
            )
            Hshear = self.__list_mat[i].get_elastic_matrix()
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
            Hshear = self.__list_mat[i].get_elastic_matrix()
            Hshear = np.array(
                [[Hshear[i][j] for j in [4, 5]] for i in [4, 5]], dtype="object"
            )
            H += (self.k * self.list_thickness[i]) * Hshear

        return H

    def get_shell_stiffness_matrix_FI(self):
        # membrane and flexural component are given for full integration part
        H = np.zeros((6, 6), dtype="object")
        for i in range(len(self.list_thickness)):
            Hplane = self.__list_mat[i].get_elastic_matrix(
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
        return super().get_stress(assembly, **kargs)

    def get_stress_distribution(self, assembly, pg, resolution=100):
        h = self.thickness
        z = np.linspace(-h / 2, h / 2, resolution)

        Strain = StrainTensorList([0 for i in range(6)])
        ShellStrain = assembly.sv["ShellStrain"]

        Strain[0] = (
            ShellStrain[0][pg] + z * ShellStrain[3][pg]
        )  # epsXX -> membrane and bending
        Strain[1] = (
            ShellStrain[1][pg] + z * ShellStrain[4][pg]
        )  # epsYY -> membrane and bending
        Strain[3] = (
            ShellStrain[2][pg] + z * ShellStrain[5][pg]
        )  # 2epsXY -> membrane and twist
        Strain[4] = ShellStrain[6][pg] * np.ones_like(z)  # 2epsXZ -> shear
        Strain[5] = ShellStrain[7][pg] * np.ones_like(z)  # 2epsYZ -> shear

        layer_z = [
            list((pos - self.__layer) <= 0).index(True) - 1 for pos in z
        ]  # find the layer corresponding to all positions in z -> could be improved as z have increasing values
        layer_z[0] = 0  # to avoid -1 value for 1st layer

        Hplane = [
            mat.get_elastic_matrix("2Dstress") for mat in self.__list_mat
        ]  # membrane rigidity matrix with plane stress assumption
        Hshear = [mat.get_elastic_matrix() for mat in self.__list_mat]
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
