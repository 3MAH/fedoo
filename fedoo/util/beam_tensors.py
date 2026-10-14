"""Section recovery from local generalized beam forces and strains."""

import numpy as np
from .voigt_tensors import StressTensorList, StrainTensorList


def _to_global(values, local_frame, engineering_shear=False):
    if local_frame is None:
        return values
    data = values.asarray()
    frame = np.asarray(local_frame)
    # BeamLocalFrame output stores flattened row-wise matrices as (9, n_points).
    if frame.ndim == 2 and frame.shape[0] == 9:
        frame = frame.T.reshape(-1, 3, 3)
    if frame.shape[-2:] != (3, 3):
        raise ValueError("local_frame must contain 3-by-3 matrices")
    shear = 0.5 if engineering_shear else 1.0
    tensor = np.array(
        [
            [data[0], shear * data[3], shear * data[4]],
            [shear * data[3], data[1], shear * data[5]],
            [shear * data[4], shear * data[5], data[2]],
        ]
    )
    tensor = np.moveaxis(tensor, (0, 1), (-2, -1))
    # Frame rows are local basis vectors expressed in global coordinates.
    tensor = np.einsum("...ia,...ij,...jb->...ab", frame, tensor, frame)
    factor = 2.0 if engineering_shear else 1.0
    return type(values)(
        np.array(
            [
                tensor[..., 0, 0],
                tensor[..., 1, 1],
                tensor[..., 2, 2],
                factor * tensor[..., 0, 1],
                factor * tensor[..., 0, 2],
                factor * tensor[..., 1, 2],
            ]
        )
    )


class _BeamComponentList(list):
    components = ()

    def __init__(self, values):
        if len(values) != 6:
            raise ValueError("beam fields must contain six components")
        super().__init__(values)

    def __getitem__(self, item):
        if isinstance(item, str):
            try:
                item = self.components.index(item)
            except ValueError as exc:
                raise KeyError(item) from exc
        return super().__getitem__(item)

    def asarray(self):
        return np.array(np.broadcast_arrays(*self))


class BeamStressList(_BeamComponentList):
    """Local resultants [N, QY, QZ, MX, MY, MZ], not a stress tensor.

    ``get_stress(section, position=(y, z))`` recovers local section stress.
    Coordinates are normalized against bounds about the section centroid.
    Pass ``local_frame`` to return global components instead. Transverse shear
    uses the section recovery model: parabolic approximations for rectangles
    and disks, or average Q/A by default. Circular, pipe and rectangle sections
    include Saint-Venant torsion; other sections reject nonzero torsion because
    their torsional stress distribution is not determined by Jx alone.
    """

    components = ("N", "QY", "QZ", "MX", "MY", "MZ")

    def get_stress(self, section, position=(0.0, 0.0), local_frame=None):
        y, z = section.section_coordinates(position)
        force = self.asarray()
        axial = (
            force[0] / section.A
            + z * force[4] / section.Iyy
            - y * force[5] / section.Izz
        )
        tau_y, tau_z = section.torsion_stress(force[3], y, z)
        shear_y, shear_z = section.shear_stress(force[1], force[2], y, z)
        axial, tau_y, tau_z = np.broadcast_arrays(
            axial, shear_y + tau_y, shear_z + tau_z
        )
        zero = np.zeros_like(axial)
        return _to_global(
            StressTensorList([axial, zero, zero, tau_y, tau_z, zero]), local_frame
        )

    def get_strain(self, section, position=(0.0, 0.0), local_frame=None):
        """Recover elastic strain, including Poisson contraction and engineering shear."""
        stress = self.get_stress(section, position)
        axial = stress[0] / section.material.E
        nu = getattr(
            section.material, "nu", section.material.E / (2 * section.material.G) - 1
        )
        transverse = -nu * axial
        strain = StrainTensorList(
            [
                axial,
                transverse,
                transverse,
                stress[3] / section.material.G,
                stress[4] / section.material.G,
                np.zeros_like(axial),
            ]
        )
        return _to_global(strain, local_frame, engineering_shear=True)


class BeamStrainList(_BeamComponentList):
    """Local generalized strains [epsX, gammaY, gammaZ, kappaX, kappaY, kappaZ].

    Strain recovery is purely kinematic: axial strain includes bending and
    engineering shear uses the generalized gamma components without a shear
    correction factor. Transverse normal strains and gammaYZ are zero under
    the beam kinematics; isotropic Poisson contraction is not added.
    Circular and pipe sections include torsion without warping; rectangles
    use the analytical free-warping solution. Other shapes
    require a section-specific warping model for nonzero twist.
    """

    components = ("epsX", "gammaY", "gammaZ", "kappaX", "kappaY", "kappaZ")

    def _resultants(self, section):
        return BeamStressList(
            [
                value * rigidity
                for value, rigidity in zip(self, section.get_beam_rigidity())
            ]
        )

    def get_stress(self, section, position=(0.0, 0.0), local_frame=None):
        return self._resultants(section).get_stress(section, position, local_frame)

    def get_strain(self, section, position=(0.0, 0.0), local_frame=None):
        """Recover kinematic strain without accessing elastic properties."""
        y, z = section.section_coordinates(position)
        values = self.asarray()
        axial = values[0] + z * values[4] - y * values[5]
        torsion_y, torsion_z = section.torsion_strain(values[3], y, z)
        axial, gamma_y, gamma_z = np.broadcast_arrays(
            axial, values[1] + torsion_y, values[2] + torsion_z
        )
        zero = np.zeros_like(axial)
        strain = StrainTensorList([axial, zero, zero, gamma_y, gamma_z, zero])
        return _to_global(strain, local_frame, engineering_shear=True)
