"""Reference operators and volume measure on a shared TL/UL solid assembly."""

import numpy as np

from fedoo.core.base import InvalidKinematicStateError


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
