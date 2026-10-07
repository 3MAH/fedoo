"""Beam constitutive laws."""

from types import SimpleNamespace
from fedoo.util.beam_recovery import validate_positions, _polar_sampling_grid
from fedoo.mesh import disk_mesh, hollow_disk_mesh, rectangle_mesh


from fedoo.core.base import ConstitutiveLaw
from fedoo.util.beam_tensors import BeamStressList, BeamStrainList
import numpy as np


class BeamProperties(ConstitutiveLaw):
    section_type = "generic"
    section_dimensions = ()
    section_registry = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "section_type" in cls.__dict__:
            BeamProperties.section_registry[cls.section_type] = cls

    def section_description(self):
        """Portable section schema; arrays retain their beam data association.

        Subclasses declare ``section_type`` and ``section_dimensions`` and
        implement section meshing/sampling without changes to FDH5 or the viewer.
        Coordinates are centroid-relative, never bounding-box-centred.
        """
        names = ("A", "Jx", "Iyy", "Izz", "k") + self.section_dimensions
        description = {
            "version": 3,
            "geometry": self.section_type,
            "properties": {
                name: np.asarray(getattr(self, name)).tolist() for name in names
            },
        }
        if hasattr(self, "reference_bounds"):
            description["reference_bounds"] = list(self.reference_bounds)
        if hasattr(self, "_section_points"):
            description["section_points"] = self._section_points.tolist()
            lower, upper = (
                self._section_points.min(axis=0),
                self._section_points.max(axis=0),
            )
            description["reference_bounds"] = [lower[0], upper[0], lower[1], upper[1]]
            description["properties"].update(
                scale_y=np.asarray(self.scale_y).tolist(),
                scale_z=np.asarray(self.scale_z).tolist(),
            )
        if hasattr(self, "section_property_associations"):
            description["associations"] = dict(self.section_property_associations)
        description["n_points"] = getattr(self, "n_points", 32)
        if getattr(self, "_output_points", None) is not None:
            description["output_points"] = self._output_points.tolist()
        if hasattr(self, "_bounds"):
            for name, value in zip(
                ("bound_ymin", "bound_ymax", "bound_zmin", "bound_zmax"), self._bounds
            ):
                description["properties"][name] = np.asarray(value).tolist()
        return description

    @classmethod
    def from_section_description(cls, description):
        """Restore a recovery-only section without a material/solver registry."""

        kind = description["geometry"]
        section_class = cls if kind == "generic" else cls.section_registry.get(kind)
        if section_class is None:
            raise ValueError(f"Beam section type {kind!r} is not registered")
        section = section_class.__new__(section_class)
        for name, value in description["properties"].items():
            setattr(section, name, np.asarray(value))
        # Version 1 called the circular radius r_ext.
        if kind == "circle" and not hasattr(section, "r"):
            section.r = section.r_ext
        section.material = SimpleNamespace()
        if "section_points" in description:
            section._section_points = np.asarray(description["section_points"])
        section.n_points = description.get("n_points", 32)
        section._output_points = (
            np.asarray(description["output_points"])
            if "output_points" in description
            else None
        )
        if hasattr(section, "bound_ymin"):
            section._bounds = tuple(
                getattr(section, name)
                for name in ("bound_ymin", "bound_ymax", "bound_zmin", "bound_zmax")
            )
        return section

    @property
    def output_points(self):
        """Configured normalized output coordinates, or None without sampling."""
        try:
            return self.normalized_output_points()
        except (ValueError, NotImplementedError):
            return None

    @output_points.setter
    def output_points(self, points):
        self._output_points = (
            None if points is None else validate_positions(points).copy()
        )

    def normalized_output_points(self, n_points=None):
        """Sampling coordinates; varying pipe radii may yield (points, 2, GP)."""
        if getattr(self, "_output_points", None) is not None:
            return self._output_points.copy()
        physical = list(
            self.scale_section_points(
                self.sample_points(self.n_points if n_points is None else n_points)
            )
        )
        return np.asarray(
            [self.normalize_section_coordinates(point) for point in physical]
        )

    def normalize_section_coordinates(self, position):
        """Inverse centroid-preserving map, including heterogeneous dimensions."""
        ymin, ymax, zmin, zmax = self.section_bounds()
        y, z = position

        def normalize(value, lower, upper):
            divisor = np.where(np.asarray(value) >= 0, upper, -np.asarray(lower))
            return np.divide(
                value,
                divisor,
                out=np.zeros(np.broadcast_shapes(np.shape(value), np.shape(divisor))),
                where=divisor != 0,
            )

        return np.broadcast_arrays(normalize(y, ymin, ymax), normalize(z, zmin, zmax))

    def section_mesh(self, n_points=32, **kwargs):
        """Mesh in normalized centroid-relative (y,z) coordinates.

        ``n_points`` is an approximate node budget, shared with sampling.
        Mesher-specific keyword arguments override the inferred discretization.
        """
        if hasattr(self, "_section_mesh"):
            return self._section_mesh.copy()
        raise NotImplementedError("This section has no section mesh")

    def sample_points(self, n_points=32):
        """Normalized section points; default to section-mesh nodes."""
        if hasattr(self, "_section_points"):
            return self._section_points.copy()
        if not isinstance(n_points, (int, np.integer)) or n_points < 8:
            raise ValueError("Use an integer budget of at least eight section samples")
        try:
            return self.section_mesh(n_points).nodes[:, :2]
        except NotImplementedError as error:
            raise ValueError(
                "A section envelope requires saved section geometry"
            ) from error

    def scale_section_points(self, points):
        """Yield physical points, broadcasting varying dimensions along beams."""
        if hasattr(self, "_section_points"):
            for y, z in points:
                yield y * self.scale_y, z * self.scale_z
            return
        raise ValueError("A section envelope requires saved section geometry")

    def section_bounds(self):
        """Physical bounding-box bounds relative to the section centroid."""
        if hasattr(self, "_bounds"):
            return self._bounds
        return (-1.0, 1.0, -1.0, 1.0)

    def normalized_section_coordinates(self, position):
        """Map normalized (y,z) toward each bounding face about the centroid.

        Each negative/positive unit coordinate selects the corresponding lower/
        upper bounding-box face. The origin remains the area centroid even for
        an asymmetric section. A bounding-box point need not lie in the material.
        """
        if len(position) != 2:
            raise ValueError("Normalized section position must be a finite (y,z) pair")
        values = np.asarray(np.broadcast_arrays(*position), dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("Normalized section position must be a finite (y,z) pair")
        ymin, ymax, zmin, zmax = self.section_bounds()
        y, z = values
        return y * np.where(y >= 0, ymax, -np.asarray(ymin)), z * np.where(
            z >= 0, zmax, -np.asarray(zmin)
        )

    def __init__(
        self,
        material,
        A,
        Jx,
        Iyy,
        Izz,
        k=0,
        name="",
        *,
        section_points=None,
        section_mesh=None,
        section_scale=(1.0, 1.0),
        output_points=None,
        n_points=32,
        section_bounds=None,
    ):
        """Beam properties constitutive law.

        This constitutive law should be associated with the weakform
        :mod:`fedoo.weakform.BeamEquilibrium`

        Parameters
        ----------
        material: ConstitutiveLaw name (str) or ConstitutiveLaw object
            Material Constitutive Law used to get the elastic modulus and
            shear modulus. The ConstitutiveLaw object should have attributes
            E and G that gives the young modulus and the shear modulus
            (as :mod:`fedoo.constitutivelaw.ElasticIsotrop`).
        A: scalar or arrays of gauss point values
            Beam section area.
        Jx: scalar or arrays of gauss point values
            Torsion constant.
        Iyy: scalar or arrays of gauss point values
            Second moment of area with respect to y (beam local coordinate
            system).
        Izz: scalar or arrays of gauss point values
            Second moment of area with respect to z (beam local coordinate
            system).
        k: scalar or arrays of gauss point values, default: 0
            Shear shape factor. If k=0 (*default*) the beam use the bernoulli
            hypothesis.
        name: str
            name of the WeakForm.

        Notes
        -----
        The current beam formulation supports isotropic linear elastic
        materials only. Material orientations are ignored; the beam
        cross-section orientation is defined by the assembly element frame.
        """
        if isinstance(material, str):
            material = ConstitutiveLaw[material]

        if name == "":
            name = material.name

        self.material = material
        self.n_points = n_points
        self.output_points = output_points
        if section_bounds is not None:
            if len(section_bounds) != 4:
                raise ValueError("section_bounds must be (ymin, ymax, zmin, zmax)")
            bounds = tuple(np.asarray(value, dtype=float) for value in section_bounds)
            if (
                any(not np.all(np.isfinite(value)) for value in bounds)
                or np.any(bounds[0] > 0)
                or np.any(bounds[1] < 0)
                or np.any(bounds[2] > 0)
                or np.any(bounds[3] < 0)
                or np.any(bounds[0] >= bounds[1])
                or np.any(bounds[2] >= bounds[3])
            ):
                raise ValueError(
                    "Section bounds must be finite, ordered and contain the section origin"
                )
            self._bounds = bounds
        if section_mesh is not None:
            self._section_mesh = section_mesh
            if section_points is None:
                section_points = section_mesh.nodes[:, :2]
        if section_points is not None:
            points = np.asarray(section_points, dtype=float)
            if (
                points.ndim != 2
                or points.shape[1] != 2
                or not len(points)
                or not np.all(np.isfinite(points))
            ):
                raise ValueError(
                    "section_points must be finite centroid-relative (y,z) pairs"
                )
            self._section_points = points.copy()
            self.scale_y, self.scale_z = section_scale

        self.A = A
        """Section surface"""

        self.Jx = Jx
        """Torsion constant"""

        self.Iyy = Iyy
        """Second moment of area with respect to y"""

        self.Izz = Izz
        """Second moment of area with respect to z"""

        self.k = k
        """Shear shape factor.

        if k=0, the bernoulli hypothesis is considered."""

        ConstitutiveLaw.__init__(self, name)  # heritage

    @property
    def linear_density(self):
        """Mass per unit beam length."""
        if not hasattr(self, "_linear_density"):
            self._linear_density = self.compute_linear_density()
        return self._linear_density

    @property
    def rotary_density(self):
        """Mass moments of inertia per unit beam length."""
        if not hasattr(self, "_rotary_density"):
            self._rotary_density = self.compute_rotary_density()
        return self._rotary_density

    def compute_linear_density(self):
        density = self._material_density()
        return density * self.A

    def compute_rotary_density(self):
        density = self._material_density()
        return [density * self.Jx, density * self.Iyy, density * self.Izz]

    def _material_density(self):
        density = getattr(self.material, "density", None)
        if density is None:
            material_name = getattr(self.material, "name", type(self.material).__name__)
            raise ValueError(
                f"Beam properties {self.name!r} need density on material "
                f"{material_name!r} for dynamic analysis. Set it with "
                "material.set_density(rho), or attach storage explicitly "
                "with weakform.set_inertia(...)."
            )
        return density

    def get_beam_rigidity(self):
        E = self.material.E
        G = self.material.G

        shear_stiffness = self.k * G * self.A

        return [
            E * self.A,
            shear_stiffness,
            shear_stiffness,
            G * self.Jx,
            E * self.Iyy,
            E * self.Izz,
        ]

    def section_coordinates(self, position):
        """Resolve normalized coordinates; scalar positions are ambiguous."""
        if np.isscalar(position) or position is None:
            raise ValueError(
                "Beam position must be normalized (y, z) coordinates, not a scalar"
            )
        if isinstance(position, tuple) and len(position) == 2:
            return self.normalized_section_coordinates(position)
        values = np.asarray(position, dtype=float)
        if values.shape == (2,):
            return self.normalized_section_coordinates(values)
        if values.ndim == 2 and values.shape[1] == 2:
            return self.normalized_section_coordinates(
                (values[:, 0, None], values[:, 1, None])
            )
        raise ValueError("Beam position must have shape (2,) or (n_points, 2)")

    def shear_stress(self, force_y, force_z, y, z):
        """Average transverse shear; subclasses may supply section distributions."""
        return force_y / self.A, force_z / self.A

    def torsion_stress(self, torque, y, z):
        """Circular-section torsion, or no torsion for generic sections."""
        if hasattr(self, "r") or hasattr(self, "r_ext"):
            return -z * torque / self.Jx, y * torque / self.Jx
        if np.any(np.asarray(torque) != 0):
            raise NotImplementedError(
                "torsional section stress recovery is supported only for circular and pipe sections"
            )
        return 0.0, 0.0

    def torsion_strain(self, twist, y, z):
        """Circular-section torsion kinematics, independent of material stiffness."""
        if hasattr(self, "r") or hasattr(self, "r_ext"):
            return -z * twist, y * twist
        if np.any(np.asarray(twist) != 0):
            raise NotImplementedError(
                "torsional section strain recovery requires a warping model "
                "for noncircular sections"
            )
        return 0.0, 0.0

    def get_local_frame(self, assembly):
        """Current local basis rows in global coordinates, in Gauss-point order."""
        frames = assembly.current.get_element_local_frame()
        if frames.shape[-1] == 2:
            frame3d = np.broadcast_to(np.eye(3), (len(frames), 3, 3)).copy()
            frame3d[:, :2, :2] = frames
            frames = frame3d
        return np.tile(frames, (assembly.n_elm_gp, 1, 1))

    def get_stress(self, assembly, position=None):
        """Recover local stress at normalized points, defaulting to output_points.

        Returns a tensor with shape (6, n_section_points, n_beam_gauss_points).
        """

        points = (
            self.normalized_output_points()
            if position is None
            else validate_positions(position)
        )
        coordinates = (
            (points[:, 0, None], points[:, 1, None])
            if points.ndim == 2
            else tuple(np.moveaxis(points, 1, 0))
        )
        beamstress = assembly.sv["BeamStress"]
        if np.isscalar(beamstress) and beamstress == 0:
            beamstress = np.zeros((6, assembly.n_gauss_points))
        return BeamStressList(beamstress).get_stress(self, coordinates)

    def get_strain(self, assembly, position=None):
        """Recover local kinematic section strain without elastic properties."""

        points = (
            self.normalized_output_points()
            if position is None
            else validate_positions(position)
        )
        coordinates = (
            (points[:, 0, None], points[:, 1, None])
            if points.ndim == 2
            else tuple(np.moveaxis(points, 1, 0))
        )
        beamstrain = assembly.sv["BeamStrain"]
        if np.isscalar(beamstrain) and beamstrain == 0:
            beamstrain = np.zeros((6, assembly.n_gauss_points))
        return BeamStrainList(beamstrain).get_strain(self, coordinates)


class BeamCircular(BeamProperties):
    section_type = "circle"
    section_dimensions = ("r",)
    reference_bounds = (-1.0, 1.0, -1.0, 1.0)

    def shear_stress(self, force_y, force_z, y, z):
        """Jourawski chord-average approximation, not a full 2D stress field."""
        return (
            4 * force_y / (3 * self.A) * (1 - (y / self.r) ** 2),
            4 * force_z / (3 * self.A) * (1 - (z / self.r) ** 2),
        )

    def section_mesh(self, n_points=32, **kwargs):
        count = max(2, int(np.sqrt(n_points / 5)) + 1)
        return disk_mesh(**({"radius": 1.0, "nr": count, "nt": count} | kwargs))

    def sample_points(self, n_points=32):
        if not isinstance(n_points, (int, np.integer)) or n_points < 8:
            raise ValueError("Use an integer budget of at least eight section samples")
        intervals, angles = _polar_sampling_grid(n_points, False)
        points = [(0.0, 0.0)]
        for radius in np.linspace(0, 1, intervals + 1)[1:]:
            for angle in np.linspace(0, 2 * np.pi, angles, endpoint=False):
                points.append((radius * np.cos(angle), radius * np.sin(angle)))
        return np.asarray(points)

    def scale_section_points(self, points):
        for y, z in points:
            yield y * self.r, z * self.r

    def section_bounds(self):
        return (-self.r, self.r, -self.r, self.r)

    def __init__(self, material, r, k=0.9, name="", *, output_points=None, n_points=32):
        """Properties for a beam with circular cross section.

        Parameters
        ----------
        material: ConstitutiveLaw name (str) or ConstitutiveLaw object
            Material Constitutive Law used to get the elastic modulus and shear
            modulus. The ConstitutiveLaw object should have attributes
            E and G that gives the young and the shear modulus
            (as :mod:`fedoo.constitutivelaw.ElasticIsotrop`).
        r: scalar or arrays of gauss point values
            Radius of the beam section.
        k: scalar or arrays of gauss point values, default: 0.9
            Shear shape factor. If k=0 the beam use the bernoulli hypothesis.
            Default is set to 0.9 (usual value for cylindrical beam).
        name: str
            Name of the WeakForm.

        Notes
        -----
        The current beam formulation supports isotropic linear elastic
        materials only. Material orientations are ignored; the beam
        cross-section orientation is defined by the assembly element frame.
        """
        self.r = r
        """Radius of the beam section."""
        A = np.pi * r**2
        Jx = np.pi / 2 * r**4
        Iyy = Izz = np.pi / 4 * r**4
        BeamProperties.__init__(
            self,
            material,
            A,
            Jx,
            Iyy,
            Izz,
            k,
            name,
            output_points=output_points,
            n_points=n_points,
        )


class BeamPipe(BeamProperties):
    section_type = "pipe"
    section_dimensions = ("r_int", "r_ext")
    reference_bounds = (-1.0, 1.0, -1.0, 1.0)

    def section_mesh(self, n_points=32, **kwargs):
        # Normalized radial fraction, mapped between local inner/outer radii.
        nr = max(2, int(np.sqrt(n_points / 4)))
        nt = max(9, 4 * int(n_points / nr / 4) + 1)
        return hollow_disk_mesh(
            **({"radius": 1.0, "thickness": 0.5, "nr": nr, "nt": nt} | kwargs)
        )

    def sample_points(self, n_points=32):
        if not isinstance(n_points, (int, np.integer)) or n_points < 8:
            raise ValueError("Use an integer budget of at least eight section samples")
        intervals, angles = _polar_sampling_grid(n_points, True)
        return np.asarray(
            [
                (radius * np.cos(angle), radius * np.sin(angle))
                for radius in np.linspace(0.5, 1, intervals + 1)
                for angle in np.linspace(0, 2 * np.pi, angles, endpoint=False)
            ]
        )

    def scale_section_points(self, points):
        for y, z in points:
            radius = np.hypot(y, z)
            physical = self.r_int + (2 * radius - 1) * (self.r_ext - self.r_int)
            yield physical * y / radius, physical * z / radius

    def section_bounds(self):
        return (-self.r_ext, self.r_ext, -self.r_ext, self.r_ext)

    def __init__(
        self, material, r_int, r_ext, k=0.5, name="", *, output_points=None, n_points=32
    ):
        """Properties for a beam with pipe cross section.

        Parameters
        ----------
        material: ConstitutiveLaw name (str) or ConstitutiveLaw object
            Material Constitutive Law used to get the elastic modulus and shear
            modulus. The ConstitutiveLaw object should have attributes
            E and G that gives the young and the shear modulus
            (as :mod:`fedoo.constitutivelaw.ElasticIsotrop`).
        r_int: scalar or arrays of gauss point values
            Internal radius.
        r_ext: scalar or arrays of gauss point values
            External radius.
        k: scalar or arrays of gauss point values, default: 0.5
            Shear shape factor. If k=0 the beam use the bernoulli hypothesis.
            Default is set to 0.5 (usual value for thin tube)
        name: str
            Name of the WeakForm

        Notes
        -----
        The current beam formulation supports isotropic linear elastic
        materials only. Material orientations are ignored; the beam
        cross-section orientation is defined by the assembly element frame.
        """
        self.r_int = r_int
        """Internal radius of the beam section."""

        self.r_ext = r_ext
        """External radius of the beam section."""

        A = np.pi * (r_ext**2 - r_int**2)
        Jx = np.pi / 2 * (r_ext**4 - r_int**4)
        Izz = Iyy = np.pi / 4 * (r_ext**4 - r_int**4)
        BeamProperties.__init__(
            self,
            material,
            A,
            Jx,
            Iyy,
            Izz,
            k,
            name,
            output_points=output_points,
            n_points=n_points,
        )


class BeamRectangular(BeamProperties):
    section_type = "rectangle"
    section_dimensions = ("a", "b")
    reference_bounds = (-0.5, 0.5, -0.5, 0.5)

    def shear_stress(self, force_y, force_z, y, z):
        """Parabolic Jourawski transverse-shear approximation."""
        return (
            3 * force_y / (2 * self.A) * (1 - (2 * y / self.a) ** 2),
            3 * force_z / (2 * self.A) * (1 - (2 * z / self.b) ** 2),
        )

    @staticmethod
    def _torsion_constant(a, b):
        """Saint-Venant constant evaluated with 128 odd Fourier harmonics."""
        h, w = np.maximum(a, b), np.minimum(a, b)
        series = sum(np.tanh(n * np.pi * h / (2 * w)) / n**5 for n in range(1, 256, 2))
        return h * w**3 / 3 * (1 - 192 * w / (np.pi**5 * h) * series)

    def torsion_strain(self, twist, y, z):
        """Free-warping Saint-Venant engineering shear, using 128 harmonics.

        Dimensions and coordinates broadcast across Gauss points. Exponential
        ratios avoid overflow for thin rectangles. No material data is needed.
        """
        if np.all(np.asarray(twist) == 0):
            return 0.0, 0.0
        swap = np.asarray(self.a) < np.asarray(self.b)
        h = np.maximum(self.a, self.b) / 2
        c = np.minimum(self.a, self.b) / 2
        u, v = np.where(swap, z, y), np.where(swap, y, z)
        shear_u, shear_v = -2 * v, np.zeros_like(u, dtype=float)
        for index, n in enumerate(range(1, 256, 2)):
            wave = n * np.pi / (2 * c)
            positive = np.exp(wave * (u - h))
            negative = np.exp(wave * (-u - h))
            denominator = 1 + np.exp(-2 * wave * h)
            coefficient = 16 * c / np.pi**2 * (-1) ** index / n**2
            shear_u = (
                shear_u
                + coefficient * np.sin(wave * v) * (positive + negative) / denominator
            )
            shear_v = (
                shear_v
                + coefficient * np.cos(wave * v) * (positive - negative) / denominator
            )
        # The exact normal traction is zero on each face. Remove the small
        # Fourier truncation residual there, including at the corners.
        shear_u = np.where(np.isclose(np.abs(u / h), 1, rtol=0, atol=1e-14), 0, shear_u)
        shear_v = np.where(np.isclose(np.abs(v / c), 1, rtol=0, atol=1e-14), 0, shear_v)
        return (
            twist * np.where(swap, -shear_v, shear_u),
            twist * np.where(swap, -shear_u, shear_v),
        )

    def torsion_stress(self, torque, y, z):
        """Recover torsional shear from torque, independently of material."""
        return self.torsion_strain(
            torque / self._torsion_constant(self.a, self.b), y, z
        )

    def section_mesh(self, n_points=32, **kwargs):
        count = max(3, 2 * int(np.sqrt(n_points) / 2) + 1)
        return rectangle_mesh(
            **(
                {
                    "nx": count,
                    "ny": count,
                    "x_min": -0.5,
                    "x_max": 0.5,
                    "y_min": -0.5,
                    "y_max": 0.5,
                }
                | kwargs
            )
        )

    def scale_section_points(self, points):
        for y, z in points:
            yield y * self.a, z * self.b

    def section_bounds(self):
        return (-self.a / 2, self.a / 2, -self.b / 2, self.b / 2)

    def __init__(
        self, material, a, b=None, k=5 / 6, name="", *, output_points=None, n_points=32
    ):
        """Properties for a beam with rectangular cross section.

        Parameters
        ----------
        material: ConstitutiveLaw name (str) or ConstitutiveLaw object
            Material Constitutive Law used to get the elastic modulus and shear
            modulus. The ConstitutiveLaw object should have attributes
            E and G that gives the young and the shear modulus
            (as :mod:`fedoo.constitutivelaw.ElasticIsotrop`).
        a: scalar or arrays of gauss point values
            Dimension of the beam section along the local y axis.
        b: scalar or arrays of gauss point values, optional
            Dimension of the beam section along the local z axis.
            If b is not specified, a square section is assumed.
        k: scalar or arrays of gauss point values, default: 5/6
            Shear shape factor. If k=0 the beam use the bernoulli hypothesis.
            Default is set to 5/6 (usual value for rectangular beam).
        name: str
            Name of the WeakForm.

        Notes
        -----
        The current beam formulation supports isotropic linear elastic
        materials only. Material orientations are ignored; the beam
        cross-section orientation is defined by the assembly element frame.
        """
        self.a = a
        """Dimension of the beam section along the y axis."""
        if b is None:
            b = a
        self.b = b
        """Dimension of the beam section along the z axis."""

        A = a * b
        Jx = self._torsion_constant(a, b)
        Iyy = a * b**3 / 12
        Izz = b * a**3 / 12

        BeamProperties.__init__(
            self,
            material,
            A,
            Jx,
            Iyy,
            Izz,
            k,
            name,
            output_points=output_points,
            n_points=n_points,
        )
