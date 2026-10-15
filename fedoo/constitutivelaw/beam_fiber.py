"""Nonlinear multifiber beam section."""

import copy
from types import SimpleNamespace

import numpy as np

from fedoo.constitutivelaw._material_points import (
    _copy_state,
    _MaterialPointAssembly,
    _tangent_array,
)
from fedoo.constitutivelaw.beam import BeamProperties
from fedoo.core.base import ConstitutiveLaw
from fedoo.util.beam_recovery import SectionRecoveryUnavailable
from fedoo.util.beam_tensors import BeamStressList
from fedoo.util.voigt_tensors import StrainTensorList

# Generalized strains [epsX, gammaY, gammaZ, kX, kY, kZ] carried by the fibers.
_AXIAL = [0, 4, 5]
# Generalized strains kept elastic: transverse shear and torsion.
_ELASTIC = [1, 2, 3]


def _gauss_layers(lower, upper, n_layers):
    """Two-point Gauss rule on each of ``n_layers`` equal layers."""
    edges = np.linspace(lower, upper, n_layers + 1)
    half = 0.5 * np.diff(edges)
    center = 0.5 * (edges[:-1] + edges[1:])
    offset = half / np.sqrt(3)
    points = np.column_stack((center - offset, center + offset)).ravel()
    return points, np.repeat(half, 2)


def _polar_rule(r_int, r_ext, n_r, n_theta):
    """Gauss-Legendre in the radius (weight r dr) times mid-sector angles."""
    xi, weights = np.polynomial.legendre.leggauss(n_r)
    radius = 0.5 * (r_ext + r_int) + 0.5 * (r_ext - r_int) * xi
    radial_weights = 0.5 * (r_ext - r_int) * weights * radius
    angle = (np.arange(n_theta) + 0.5) * 2 * np.pi / n_theta
    rho, theta = np.meshgrid(radius, angle, indexing="ij")
    w = np.repeat(radial_weights, n_theta) * (2 * np.pi / n_theta)
    return (rho * np.cos(theta)).ravel(), (rho * np.sin(theta)).ravel(), w


def _section_fibers(section, n_fibers):
    """Default fiber quadrature (y, z, weight) of a built-in section."""
    kind = section.section_type
    dimensions = [getattr(section, name) for name in section.section_dimensions]
    if any(np.ndim(value) for value in dimensions):
        raise ValueError(
            "BeamFiberSection requires uniform section dimensions; "
            "pass explicit fibers for a varying section"
        )
    if kind == "rectangle":
        n_y, n_z = (4, 4) if n_fibers is None else n_fibers
        y, wy = _gauss_layers(-section.a / 2, section.a / 2, n_y)
        z, wz = _gauss_layers(-section.b / 2, section.b / 2, n_z)
        y, z = np.meshgrid(y, z, indexing="ij")
        return y.ravel(), z.ravel(), np.outer(wy, wz).ravel()
    if kind == "circle":
        n_r, n_theta = (3, 16) if n_fibers is None else n_fibers
        return _polar_rule(0.0, section.r, n_r, n_theta)
    if kind == "pipe":
        n_r, n_theta = (2, 16) if n_fibers is None else n_fibers
        return _polar_rule(section.r_int, section.r_ext, n_r, n_theta)
    raise ValueError(
        f"No default fiber quadrature for section type {kind!r}; "
        "pass explicit fibers=(y, z, weights)"
    )


def _elastic_constants(material):
    """Initial elastic E, G and nu along the fiber axis of ``material``."""
    if hasattr(material, "E") and hasattr(material, "G"):
        E, G = material.E, material.G
    else:
        probe = copy.deepcopy(material)
        probe.reset()
        points = _MaterialPointAssembly(1, "3D")
        probe.initialize(points, None)
        H = _tangent_array(points.sv["TangentMatrix"], 1)[:, :, 0]
        E = 1 / np.linalg.inv(H)[0, 0]
        G = 0.5 * (H[3, 3] + H[4, 4])
    # nu is only used with G through E / (2 (1 + nu)) by the beam element.
    return E, G, E / (2 * G) - 1


class BeamFiberSection(BeamProperties):
    """Beam section integrated from nonlinear material points (fibers).

    The axial strain of a fiber at :math:`(y, z)` in the local section frame
    is :math:`\\varepsilon_{11} = \\varepsilon_X + z\\,\\kappa_Y - y\\,\\kappa_Z`.
    A private copy of ``material`` is evaluated at every fiber of every beam
    Gauss point under uniaxial stress (``"1Dstress"``). The normal force and
    bending moments are the fiber integrals :math:`N = \\int \\sigma_{11}`,
    :math:`M_Y = \\int z\\,\\sigma_{11}`, :math:`M_Z = -\\int y\\,\\sigma_{11}`,
    and the section tangent couples them once the section yields.

    Transverse shear and torsion keep the elastic treatment of the
    underlying section (:math:`kGA` and :math:`GJ_x`, with the initial
    elastic shear modulus of the material), as the nonlinear shells do.

    Parameters
    ----------
    section : BeamCircular, BeamPipe or BeamRectangular
        Section geometry. Its material is the fiber material unless
        ``material`` is given. Section dimensions must be uniform.
    n_fibers : tuple of int, optional
        Section quadrature: ``(n_y, n_z)`` layers of a rectangle, each with
        a two-point Gauss rule (default ``(4, 4)``; even counts integrate the
        fully plastic moment exactly); ``(n_r, n_theta)`` radial Gauss points
        and angular sectors of a circle (default ``(3, 16)``) or a pipe
        (default ``(2, 16)``).
    material : Mechanical3D or str, optional
        Fiber material, for instance ``fedoo.constitutivelaw.Simcoon``. It
        must support the ``"1Dstress"`` stress state. Defaults to the
        section material.
    fibers : tuple of array_like, optional
        Explicit fibers ``(y, z, weights)`` in physical centroid-relative
        coordinates, overriding ``n_fibers``.
    name : str, optional
        Name of the constitutive law.

    Notes
    -----
    The fiber material sees a small strain in the corotated element frame,
    so geometric nonlinearities follow the corotational beam kinematics.
    Shape functions, inertia and the elastic shear and torsion stiffness use
    the initial elastic constants of the material. Stresses are only
    available at the fibers: saved ``Stress`` fields use the fibers as
    section points, with elastic shear and torsion stresses added.

    The elastic constants and the density are read when the section is
    built: set the material density before. A section instance holds the
    material state of one assembly.

    Examples
    --------
    .. code-block:: python

        import fedoo as fd

        fd.ModelingSpace("3D")
        steel = fd.constitutivelaw.Simcoon(
            "EPICP", [210e3, 0.3, 0.0, 355.0, 500.0, 0.3]
        )
        section = fd.constitutivelaw.BeamFiberSection(
            fd.constitutivelaw.BeamRectangular(steel, 20.0, 10.0), n_fibers=(8, 4)
        )
        wf = fd.weakform.BeamEquilibrium(section, nlgeom=True)
    """

    is_section_law = True
    recovery = "stored_only"

    def __init__(self, section, n_fibers=None, *, material=None, fibers=None, name=""):
        if not isinstance(section, BeamProperties) or isinstance(
            section, BeamFiberSection
        ):
            raise TypeError("section must be a beam section, e.g. BeamRectangular")
        if material is None:
            material = section.material
        elif isinstance(material, str):
            material = ConstitutiveLaw[material]

        self._section = copy.copy(section)
        self.section_type = section.section_type
        self.section_dimensions = section.section_dimensions
        if hasattr(section, "reference_bounds"):
            self.reference_bounds = section.reference_bounds
        for attribute in ("A", "Jx", "Iyy", "Izz", "k") + section.section_dimensions:
            setattr(self, attribute, getattr(section, attribute))
        self._n_points = section.n_points

        if fibers is None:
            y, z, weights = _section_fibers(section, n_fibers)
        else:
            y, z, weights = (np.asarray(value, dtype=float).ravel() for value in fibers)
            if not len(y) == len(z) == len(weights):
                raise ValueError("fibers must be three arrays of equal length")
        self._y, self._z, self._weights = y, z, weights
        self._output_points = np.column_stack(
            section.normalize_section_coordinates((y, z))
        )

        self.fiber_material = copy.deepcopy(material)
        """Private copy of the fiber material law."""
        E, G, nu = _elastic_constants(self.fiber_material)
        self.material = SimpleNamespace(
            E=E,
            G=G,
            nu=nu,
            density=getattr(material, "density", None),
            name=getattr(material, "name", ""),
        )
        """Initial elastic constants, read by the beam element and the
        elastic shear and torsion terms."""
        self._material_assembly = None
        self._owner = None
        ConstitutiveLaw.__init__(self, name)

    @property
    def n_fibers(self):
        """Number of fibers per beam Gauss point."""
        return len(self._weights)

    @property
    def fibers(self):
        """Fiber coordinates and weights ``(y, z, weights)``."""
        return self._y.copy(), self._z.copy(), self._weights.copy()

    # Section geometry of the underlying section
    def section_bounds(self):
        return self._section.section_bounds()

    def scale_section_points(self, points):
        return self._section.scale_section_points(points)

    def sample_points(self, n_points=32):
        return self._section.sample_points(n_points)

    def section_mesh(self, n_points=32, **kwargs):
        return self._section.section_mesh(n_points, **kwargs)

    def shear_stress(self, force_y, force_z, y, z):
        return self._section.shear_stress(force_y, force_z, y, z)

    def torsion_stress(self, torque, y, z):
        return self._section.torsion_stress(torque, y, z)

    def torsion_strain(self, twist, y, z):
        return self._section.torsion_strain(twist, y, z)

    @BeamProperties.output_points.setter
    def output_points(self, points):
        raise AttributeError("The output points of a fiber section are its fibers")

    # Lifecycle
    def initialize(self, assembly, pb):
        if self._owner is not None and self._owner is not assembly:
            raise RuntimeError(
                "A BeamFiberSection holds the state of one assembly; "
                "create one section per assembly"
            )
        self._owner = assembly
        self._n_gp = assembly.n_gauss_points
        points = _MaterialPointAssembly(self.n_fibers * self._n_gp, "1Dstress")
        material = self.fiber_material
        material.reset()
        material.initialize(points, pb)
        points.sv_start = _copy_state(points.sv)
        self._material_assembly = points
        self._store(assembly, np.zeros((6, self._n_gp)))
        assembly.sv["BeamStress"] = 0
        self.is_initialized = True

    def update(self, assembly, pb):
        strain = self._generalized_strain(assembly)
        fiber_strain = np.zeros((6, self.n_fibers * self._n_gp))
        fiber_strain[0] = (
            strain[0] + self._z[:, None] * strain[4] - self._y[:, None] * strain[5]
        ).ravel()
        self._material_assembly.sv["Strain"] = StrainTensorList(fiber_strain)
        self.fiber_material.update(self._material_assembly, pb)
        self._store(assembly, strain)

    def set_start(self, assembly, pb):
        self.fiber_material.set_start(self._material_assembly, pb)
        self._material_assembly.sv_start = _copy_state(self._material_assembly.sv)

    def to_start(self, assembly, pb):
        self.fiber_material.to_start(self._material_assembly, pb)
        self._material_assembly.sv = _copy_state(self._material_assembly.sv_start)

    def reset(self):
        self.fiber_material.reset()
        self._material_assembly = None
        self._owner = None
        ConstitutiveLaw.reset(self)

    def _generalized_strain(self, assembly):
        values = assembly.sv["BeamStrain"]
        if np.isscalar(values) and values == 0:
            return np.zeros((6, self._n_gp))
        return np.array([np.broadcast_to(c, (self._n_gp,)) for c in values], float)

    def _fiber_axial(self, key):
        """Fiber field ``sigma_11`` or ``dsigma_11/deps_11`` as (fiber, GP)."""
        points = self._material_assembly
        n = self.n_fibers * self._n_gp
        if key == "Stress":
            stress = points.sv["Stress"]
            if np.isscalar(stress) and stress == 0:
                return np.zeros((self.n_fibers, self._n_gp))
            values = np.broadcast_to(np.asarray(stress[0], dtype=float), (n,))
        else:
            values = _tangent_array(points.sv["TangentMatrix"], n)[0, 0]
        return values.reshape(self.n_fibers, self._n_gp)

    def _store(self, assembly, strain):
        """Integrate the section resultants and tangent from the fibers."""
        b = np.array([np.ones_like(self._y), self._z, -self._y])
        weighted = b * self._weights
        stress = np.zeros((6, self._n_gp))
        stress[_AXIAL] = weighted @ self._fiber_axial("Stress")
        tangent = np.zeros((6, 6, self._n_gp))
        axial = np.einsum(
            "af,bf,fg->abg", weighted, b, self._fiber_axial("TangentMatrix")
        )
        # Drop round-off couplings of a symmetric section, so an elastic
        # section assembles the same diagonal terms as a plain one.
        diagonal = np.abs(np.einsum("aag->ag", axial)).max(axis=1)
        axial[
            np.abs(axial) <= 1e-13 * np.sqrt(np.outer(diagonal, diagonal))[..., None]
        ] = 0
        tangent[np.ix_(_AXIAL, _AXIAL)] = axial
        rigidity = self.get_beam_rigidity()
        components = list(stress)
        for i in _ELASTIC:
            if np.isscalar(rigidity[i]) and rigidity[i] == 0:
                # Bernoulli beam: no shear term, as for a plain section.
                components[i] = 0
                continue
            tangent[i, i] = rigidity[i]
            components[i] = rigidity[i] * strain[i]
        assembly.sv["BeamStress"] = BeamStressList(components)
        assembly.sv["_BeamStiffnessMatrix"] = tangent

    # Outputs
    def integration_stresses(self, assembly):
        """Local fiber stresses, shape (6, n_fibers, n_beam_gauss_points).

        The normal stress comes from the fiber material; transverse shear
        and torsion stresses are the elastic distributions of the section.
        """
        if self._material_assembly is None:
            raise RuntimeError("The fiber section has not been initialized")
        resultants = assembly.sv["BeamStress"]
        if np.isscalar(resultants) and resultants == 0:
            resultants = np.zeros((6, self._n_gp))
        resultants = np.asarray(BeamStressList(resultants).asarray(), dtype=float)
        y, z = self._y[:, None], self._z[:, None]
        shear_y, shear_z = self.shear_stress(resultants[1], resultants[2], y, z)
        torsion_y, torsion_z = self.torsion_stress(resultants[3], y, z)
        stress = np.zeros((6, self.n_fibers, self._n_gp))
        stress[0] = self._fiber_axial("Stress")
        stress[3] = shear_y + torsion_y
        stress[4] = shear_z + torsion_z
        return stress

    def get_stress(self, assembly, position=None):
        """Fiber stresses (6, n_fibers, n_beam_gauss_points) in local axes.

        Stresses are only known at the fibers: ``position`` must be None.
        """
        if position is not None:
            raise SectionRecoveryUnavailable(
                "Fiber section stresses are only available at the fibers "
                "(position=None)"
            )
        return self.integration_stresses(assembly)

    def get_stress_distribution(self, assembly, pg):
        """Return ``(y, z, stress)`` over the fibers at one beam Gauss point."""
        if not 0 <= pg < assembly.n_gauss_points:
            raise IndexError("pg is outside the beam Gauss-point range.")
        return (
            self._y.copy(),
            self._z.copy(),
            self.integration_stresses(assembly)[:, :, pg],
        )
