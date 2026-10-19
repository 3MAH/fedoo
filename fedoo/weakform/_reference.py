"""Reference operators and volume measure on a shared TL/UL solid assembly."""

import numpy as np

from fedoo.core.base import InvalidKinematicStateError


def deformation_gradient(assembly):
    F = assembly.sv.get("F")
    if F is None:
        F = np.broadcast_to(np.eye(3)[:, :, None], (3, 3, assembly.n_gauss_points))
    F = np.asarray(F, dtype=float)
    matrices = F.transpose(2, 0, 1)
    J = np.linalg.det(matrices)
    if np.any(~np.isfinite(J)) or np.any(J <= 1e-12):
        raise InvalidKinematicStateError("Thermal geometry requires positive det F.")
    inverse = np.linalg.inv(matrices).transpose(1, 2, 0)
    return F, inverse, J


def reference_volume_weight(assembly):
    """dV0/dv when assembling on the UL mesh, otherwise unity."""
    reference = getattr(assembly, "_reference_assembly", None)
    if reference is None:
        return 1.0
    current_volume = assembly._get_gaussian_quadrature_mat().data
    if np.any(~np.isfinite(current_volume)) or np.any(current_volume <= 0):
        raise InvalidKinematicStateError(
            "Reference integration requires positive element volumes."
        )
    return reference._get_gaussian_quadrature_mat().data / current_volume


def reference_derivatives(assembly, variable, F):
    space = assembly.space
    derivatives = [space.derivative(variable, x) for x in "XYZ"[: space.ndim]]
    if getattr(assembly, "_reference_assembly", None) is None:
        return derivatives + [0] * (3 - space.ndim)
    # Grad_X = F^T grad_x. This lets a TL thermal form share an assembly
    # with UL mechanics without a second material evaluation or history.
    return [sum(derivatives[i] * F[i, A] for i in range(space.ndim)) for A in range(3)]
