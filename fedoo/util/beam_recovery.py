"""Self-contained section descriptions and lazy recovery of saved beam fields."""

from .voigt_tensors import StressTensorList, StrainTensorList
from .beam_tensors import BeamStressList, BeamStrainList, _to_global


from types import SimpleNamespace
import numpy as np


def validate_positions(position):
    """Normalize an explicit position argument to an (n_points, 2) array."""
    values = np.asarray(position, dtype=float)
    if values.shape == (2,):
        values = values[None, :]
    if (
        values.ndim != 2
        or values.shape[1] != 2
        or len(values) == 0
        or not np.all(np.isfinite(values))
    ):
        raise ValueError(
            "Beam position must be finite normalized (y, z) coordinates with shape (2,) or (n_points, 2); scalar positions are not accepted"
        )
    return values


class BeamOptions(dict):
    """Atomic selection updates; coordinates and point indices are exclusive."""

    def __init__(self, values=None):
        super().__init__(
            source="auto",
            reduction="abs_max",
            samples=32,
            position=None,
            point_index=None,
        )
        if values:
            self.update(values)

    def update(self, *args, **kwargs):
        changed = dict(*args, **kwargs)
        if (
            changed.get("position") is not None
            and changed.get("point_index") is not None
        ):
            raise ValueError("Provide position or point_index, not both")
        if "position" in changed and changed["position"] is not None:
            positions = validate_positions(changed["position"])
            if len(positions) != 1:
                raise ValueError("Select one section position during post-processing")
            changed["position"] = tuple(positions[0])
            changed["point_index"] = None
        elif changed.get("point_index") is not None:
            index = changed["point_index"]
            if not isinstance(index, (int, np.integer)) or index < 0:
                raise ValueError("point_index must be a nonnegative integer")
            changed["position"] = None
        if "source" in changed and changed["source"] not in (
            "auto",
            "stored",
            "recomputed",
        ):
            raise ValueError("source must be 'auto', 'stored' or 'recomputed'")
        if "reduction" in changed and changed["reduction"] not in (
            None,
            "max",
            "min",
            "abs_max",
        ):
            raise ValueError("reduction must be None, 'max', 'min' or 'abs_max'")
        if (
            "position_coordinates" in changed
            and changed["position_coordinates"] != "normalized"
        ):
            raise ValueError("Beam positions are always normalized")
        super().update(changed)

    def __setitem__(self, key, value):
        self.update({key: value})


def reduce_section_tensors(tensors, kind, component, reduction):
    """Compute each point's invariant before reducing, preserving abs_max signs."""

    tensor = (StressTensorList if kind == "Stress" else StrainTensorList)(tensors)
    if isinstance(component, str) and component.isdigit():
        component = int(component)
    if reduction is not None and component is None:
        raise ValueError("Select a scalar component for a section reduction")
    # Principal-value routines expect one point axis. Flatten then restore it.
    if component is None:
        value = tensors
    else:
        flat = type(tensor)(tensors.reshape(6, -1))
        value = np.asarray(flat[component]).reshape(tensors.shape[1:])
    if reduction is None:
        return value[:, 0] if component is None else value[0]
    if reduction == "max":
        return np.max(value, axis=0)
    if reduction == "min":
        return np.min(value, axis=0)
    indices = np.argmax(np.abs(value), axis=0)
    return np.take_along_axis(value, indices[None, :], axis=0)[0]


def _polar_sampling_grid(samples, pipe):
    """Choose the nearest nested polar grid to a total point budget."""
    intervals, angles = 1, 4 if pipe else 8
    previous = None
    level = 0
    while True:
        count = (intervals + 1) * angles if pipe else intervals * angles + 1
        if count >= samples:
            if previous is not None and samples - previous[2] < count - samples:
                return previous[:2]
            return intervals, angles
        previous = intervals, angles, count
        # Alternate radial and angular refinement, preserving previous points.
        if (level % 2 == 0) != pipe:
            intervals *= 2
        else:
            angles *= 2
        level += 1


class SavedBeamSection:
    """Numeric section properties reconstructed without a solver or material registry."""

    def __init__(self, description):
        self.geometry = description["geometry"]
        self.description = description
        # BeamProperties imports these recovery helpers; avoid that import cycle.
        from fedoo.constitutivelaw.beam import BeamProperties

        self.section = BeamProperties.from_section_description(description)
        for name, value in description["properties"].items():
            setattr(self, name, np.asarray(value))
        if self.geometry == "circle" and hasattr(self, "r"):
            self.r_ext = self.r
        # Older descriptors may contain isotropic properties. They are optional
        # for the native BeamStress -> Stress and BeamStrain -> Strain paths.
        self.material = SimpleNamespace(
            **{
                name: getattr(self, name)
                for name in ("E", "G", "nu")
                if hasattr(self, name)
            }
        )

    def section_coordinates(self, position):
        return self.section.section_coordinates(position)

    def normalized_section_coordinates(self, position):
        return self.section.normalized_section_coordinates(position)

    def get_beam_rigidity(self):
        return [
            self.E * self.A,
            self.k * self.G * self.A,
            self.k * self.G * self.A,
            self.G * self.Jx,
            self.E * self.Iyy,
            self.E * self.Izz,
        ]

    def shear_stress(self, force_y, force_z, y, z):
        return self.section.shear_stress(force_y, force_z, y, z)

    def torsion_stress(self, torque, y, z):
        return self.section.torsion_stress(torque, y, z)

    def torsion_strain(self, twist, y, z):
        return self.section.torsion_strain(twist, y, z)

    def points(self, samples):
        """Physical coordinates of the configured or generated sampling points."""
        for point in self.section.normalized_output_points(samples):
            yield self.normalized_section_coordinates(tuple(point))


def section_description(section, assembly=None, element_set=None):
    """Typed section description with nonuniform properties in GP order."""
    if isinstance(element_set, str):
        element_set = assembly.mesh.element_sets[element_set]
    description = section.section_description()
    if assembly is not None and hasattr(assembly, "n_elm_gp"):
        description["n_elm_gp"] = assembly.n_elm_gp
    values = description["properties"]
    properties = {}
    for name, value in values.items():
        array = np.asarray(value)
        association = description.get("associations", {}).get(name, "GaussPoint")
        if array.ndim:
            if (
                association == "GaussPoint"
                and assembly is not None
                and array.shape[-1] != assembly.n_gauss_points
            ):
                array = assembly.convert_data(array, convert_to="GaussPoint")
            if element_set is not None:
                if association == "Node":
                    raise ValueError(
                        "Node section properties cannot be saved with an element subset"
                    )
                array = array.reshape(-1, assembly.mesh.n_elements)[
                    :, element_set
                ].ravel()
        properties[name] = array.tolist()
    description["properties"] = properties
    return description


def recover_beam_field(
    values,
    description,
    kind,
    component,
    options,
    local_frame=None,
    stored=None,
    stored_points=None,
):
    """Apply the same section selection to saved tensors and recomputed tensors."""
    section = SavedBeamSection(description)
    options = BeamOptions(options)
    reduction = options["reduction"]
    use_stored = options["source"] == "stored" or (
        options["source"] == "auto" and stored is not None
    )
    if use_stored:
        if stored is None or stored_points is None:
            raise ValueError(
                f"Stored beam {kind} and its section coordinates are unavailable"
            )
        tensors = np.asarray(stored)
        if tensors.ndim == 2:
            tensors = tensors[:, None, :]
        points = np.asarray(stored_points)
    else:
        if values is None:
            raise ValueError(
                f"Recomputed {kind} requires {'BeamStress' if kind == 'Stress' else 'BeamStrain'}"
            )
        if reduction is None and options["position"] is not None:
            points = validate_positions(options["position"])
        else:
            points = section.section.normalized_output_points(options["samples"])
        field = BeamStressList(values) if kind == "Stress" else BeamStrainList(values)
        method = field.get_stress if kind == "Stress" else field.get_strain
        tensors = None
    if reduction is None:
        index = options["point_index"]
        if index is not None:
            if index >= len(points):
                raise ValueError(
                    f"Section point index must be between 0 and {len(points) - 1}"
                )
            indices = [index]
        elif options["position"] is not None:
            if use_stored:
                target = np.asarray(options["position"])
                if points.ndim == 3:
                    target = target[:, None]
                matches = np.all(
                    np.isclose(points, target, rtol=0, atol=1e-12),
                    axis=tuple(range(1, points.ndim)),
                )
                if not np.any(matches):
                    raise ValueError(
                        "Position does not match a stored section point; select recomputed data"
                    )
                indices = [int(np.flatnonzero(matches)[0])]
            else:
                indices = [0]
        else:
            raise ValueError("With reduction=None, provide position or point_index")
        points = points[indices]
        if use_stored:
            tensors = tensors[:, indices, :]
    if not use_stored:
        tensors = np.stack(
            [method(section, tuple(point), local_frame).asarray() for point in points],
            axis=1,
        )
    elif local_frame is not None:
        cls = StressTensorList if kind == "Stress" else StrainTensorList
        tensors = _to_global(
            cls(tensors), local_frame, engineering_shear=kind == "Strain"
        ).asarray()
    return reduce_section_tensors(tensors, kind, component, reduction)
