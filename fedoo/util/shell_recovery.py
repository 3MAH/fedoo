"""Thickness sampling and recovery of saved shell tensors."""

import numpy as np

from .beam_recovery import BeamOptions, reduce_section_tensors, use_stored_beam_field
from .beam_tensors import _to_global
from .voigt_tensors import StressTensorList, StrainTensorList


def validate_positions(position):
    """Return a nonempty vector of normalized thickness coordinates."""
    points = np.atleast_1d(np.asarray(position, dtype=float))
    if (
        points.ndim != 1
        or not points.size
        or not np.all(np.isfinite(points))
        or np.any(np.abs(points) > 1)
    ):
        raise ValueError("Shell position must be a scalar or a vector in [-1, 1]")
    return points


class ShellOptions(BeamOptions):
    """Atomic thickness selection with the same sources and reductions as beams."""

    def __init__(self, values=None):
        super().__init__()
        dict.update(self, samples=3)
        dict.update(self, stored_position="interpolated", reference_point_index=None)
        if values:
            self.update(values)

    def update(self, *args, **kwargs):
        changed = dict(*args, **kwargs)
        stored_position = changed.pop(
            "stored_position", self.get("stored_position", "interpolated")
        )
        if stored_position not in ("nearest", "exact", "interpolated"):
            raise ValueError(
                "Shell stored_position must be 'nearest', 'exact' or 'interpolated'"
            )
        has_position = "position" in changed
        position = changed.pop("position", None)
        if (
            has_position or "point_index" in changed
        ) and "reference_point_index" not in changed:
            changed["reference_point_index"] = None
        if position is not None:
            if changed.get("point_index") is not None:
                raise ValueError("Provide position or point_index, not both")
            points = validate_positions(position)
            if len(points) != 1:
                raise ValueError("Select one thickness position during post-processing")
            changed["point_index"] = None
        if "samples" in changed:
            count = changed["samples"]
            if not isinstance(count, (int, np.integer)) or count < 2:
                raise ValueError("Shell samples must be an integer of at least two")
        if changed.get("reference_point_index") is not None:
            index = changed["reference_point_index"]
            if not isinstance(index, (int, np.integer)) or index < 0:
                raise ValueError(
                    "reference_point_index must be a nonnegative integer or None"
                )
        super().update(changed)
        dict.__setitem__(self, "stored_position", stored_position)
        if has_position:
            dict.__setitem__(
                self, "position", None if position is None else float(points[0])
            )


def reconstructs_stored_shell(kind, description, stored_description):
    """Two distinct saved points determine a linear thickness distribution."""
    return bool(
        stored_description is not None
        and (kind == "Strain" or description["model"] == "homogeneous_linear")
        and len(np.unique(stored_description.get("stored_points", []))) >= 2
    )


def can_recompute_shell(
    kind, description, has_generalized, has_integration, stored_description=None
):
    return bool(
        has_generalized
        and (kind == "Strain" or description["model"] == "homogeneous_linear")
        or kind == "Stress"
        and has_integration
        or reconstructs_stored_shell(kind, description, stored_description)
    )


def stored_shell_point_index(points, position, description, mode="nearest"):
    """Select a saved thickness point on the requested side of an interface."""
    points = validate_positions(points)
    candidates = np.arange(len(points))
    if "point_layers" in description:
        interfaces = np.asarray(description.get("interfaces", [-1, 1]))
        layer = np.clip(
            np.searchsorted(interfaces, position, side="left") - 1,
            0,
            len(interfaces) - 2,
        )
        candidates = candidates[np.asarray(description["point_layers"]) == layer]
    if not len(candidates):
        raise ValueError("No stored thickness points in the requested layer")
    if mode == "exact":
        matches = candidates[
            np.isclose(points[candidates], position, rtol=0, atol=1e-12)
        ]
        if not len(matches):
            raise ValueError("Position does not match a stored thickness point")
        return int(matches[0])
    return int(candidates[np.argmin(np.abs(points[candidates] - position))])


def generalized_tensors(values, description, kind, points):
    """Homogeneous elastic resultants, or Reissner--Mindlin strain kinematics."""
    values = np.asarray(values, dtype=float)
    points = validate_positions(points)
    h = np.asarray(description["thickness"], dtype=float)
    z = points[:, None] * h / 2
    result = np.zeros((6, len(points), values.shape[-1]))
    if kind == "Stress":
        if description["model"] != "homogeneous_linear":
            raise ValueError(
                "Only homogeneous linear shells recover Stress from ShellStress"
            )
        result[[0, 1, 3]] = (
            values[:3, None, :] / h + 12 * z * values[3:6, None, :] / h**3
        )
        result[4:6] = values[6:8, None, :] / (h * description["k"])
    else:
        result[[0, 1, 3]] = values[:3, None, :] + z * values[3:6, None, :]
        result[4:6] = values[6:8, None, :]
    return result


def sample_positions(description, samples):
    """Uniform samples per layer, including both sides of every interface."""
    if not isinstance(samples, (int, np.integer)) or samples < 2:
        raise ValueError("Shell samples must be an integer of at least two")
    interfaces = np.asarray(description.get("interfaces", [-1, 1]))
    native = np.asarray(description.get("integration_points", []))
    native_layers = np.asarray(description.get("integration_layers", []))
    points = [
        np.unique(
            np.concatenate((np.linspace(a, b, samples), native[native_layers == i]))
        )
        for i, (a, b) in enumerate(zip(interfaces[:-1], interfaces[1:]))
    ]
    layers = np.concatenate([np.full(len(p), i) for i, p in enumerate(points)])
    return np.concatenate(points), layers


def interpolate_tensors(
    values,
    points,
    targets,
    layers=None,
    interfaces=None,
    target_layers=None,
    *,
    allow_constant=False,
    extrapolate=False,
):
    """Piecewise linear interpolation, independently within each material layer.

    Outside a layer's integration points the nearest value is retained, as in
    the shell laws' existing distribution plots. Exact interfaces select the
    lower layer unless target_layers explicitly selects the other side.
    """
    values = np.asarray(values)
    points = validate_positions(points)
    targets = validate_positions(targets)
    layers = np.zeros(len(points), dtype=int) if layers is None else np.asarray(layers)
    if target_layers is None:
        interfaces = np.asarray([-1, 1] if interfaces is None else interfaces)
        target_layers = np.clip(
            np.searchsorted(interfaces, targets, side="left") - 1,
            0,
            len(interfaces) - 2,
        )
    result = np.empty((6, len(targets), values.shape[-1]))
    for layer in np.unique(target_layers):
        selected = np.flatnonzero(layers == layer)
        if not len(selected):
            raise ValueError(f"No saved thickness points for layer {layer}")
        selected = selected[np.argsort(points[selected], kind="stable")]
        _, unique = np.unique(points[selected], return_index=True)
        selected = selected[unique]
        xp = points[selected]
        output = np.flatnonzero(target_layers == layer)
        x = targets[output]
        if len(xp) == 1:
            if not allow_constant and not np.allclose(x, xp[0], rtol=0, atol=1e-12):
                raise ValueError(
                    "One saved thickness point cannot determine a distribution; save the integration points or recompute"
                )
            result[:, output] = values[:, selected]
            continue
        right = np.clip(np.searchsorted(xp, x, side="right"), 1, len(xp) - 1)
        left = right - 1
        weight = (x - xp[left]) / (xp[right] - xp[left])
        if not extrapolate:
            weight = np.clip(weight, 0, 1)
        result[:, output] = (
            values[:, selected[left]] * (1 - weight)[None, :, None]
            + values[:, selected[right]] * weight[None, :, None]
        )
    return result


def recover_shell_field(
    values,
    description,
    kind,
    component,
    options,
    frame=None,
    *,
    stored=None,
    stored_description=None,
    integration_values=None,
):
    """Select stored tensors or recover/interpolate through the thickness."""
    use_stored = use_stored_beam_field(
        options,
        stored is not None,
        can_recompute_shell(
            kind,
            description,
            values is not None,
            integration_values is not None,
            stored_description if stored is not None else None,
        ),
    )
    reduction = options["reduction"]
    index = options["point_index"]
    if use_stored:
        if stored is None:
            raise ValueError(f"No stored {kind} thickness field is available")
        tensors = np.asarray(stored)
        if tensors.ndim == 2:
            tensors = tensors[:, None, :]
        sampling = stored_description
        points = np.asarray(sampling["stored_points"])
        layers = sampling.get("point_layers")
    else:
        points, layers = sample_positions(description, options["samples"])
        analytic = kind == "Strain" or description["model"] == "homogeneous_linear"
        from_stored = (
            analytic
            and values is None
            and stored is not None
            and reconstructs_stored_shell(kind, description, stored_description)
        )
        if analytic:
            if from_stored:
                raw = np.asarray(stored)
                if raw.ndim == 2:
                    raw = raw[:, None, :]
                tensors = interpolate_tensors(
                    raw, stored_description["stored_points"], points, extrapolate=True
                )
            elif values is None:
                raise ValueError(f"Recomputed {kind} requires Shell{kind}")
            else:
                tensors = None
        else:
            if integration_values is None:
                raise ValueError(
                    "ShellStress alone cannot recover nonlinear or laminate stresses; save the thickness integration stresses"
                )
            tensors = interpolate_tensors(
                integration_values,
                description["integration_points"],
                points,
                description.get("integration_layers"),
                description.get("interfaces"),
                layers,
                allow_constant=True,
            )
    if reduction is None:
        if index is not None:
            if index >= len(points):
                raise ValueError(
                    f"Thickness point index must be between 0 and {len(points) - 1}"
                )
            points = points[index : index + 1]
            if tensors is not None:
                tensors = tensors[:, index : index + 1]
        elif options["position"] is not None:
            target = [options["position"]]
            if use_stored:
                mode = options.get("stored_position", "interpolated")
                if mode in ("nearest", "exact"):
                    selected = stored_shell_point_index(
                        points, target[0], sampling, mode
                    )
                    tensors = tensors[:, selected : selected + 1]
                else:
                    tensors = interpolate_tensors(
                        tensors,
                        points,
                        target,
                        layers,
                        description.get("interfaces"),
                        allow_constant=sampling.get("sampling") == "integration",
                    )
            elif from_stored:
                tensors = interpolate_tensors(
                    raw, stored_description["stored_points"], target, extrapolate=True
                )
            elif not analytic:
                tensors = interpolate_tensors(
                    integration_values,
                    description["integration_points"],
                    target,
                    description.get("integration_layers"),
                    description.get("interfaces"),
                    allow_constant=True,
                )
            points = target
        else:
            raise ValueError("With reduction=None, provide position or point_index")
    if not use_stored and analytic and not from_stored:
        tensors = generalized_tensors(values, description, kind, points)
    if frame is not None:
        cls = StressTensorList if kind == "Stress" else StrainTensorList
        tensors = _to_global(
            cls(tensors), frame, engineering_shear=kind == "Strain"
        ).asarray()
    return reduce_section_tensors(tensors, kind, component, reduction)
