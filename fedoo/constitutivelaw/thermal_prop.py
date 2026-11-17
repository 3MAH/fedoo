# baseclass
import numpy as np
from fedoo.core.base import ConstitutiveLaw


class ThermalProperties(ConstitutiveLaw):
    """Heat transport and optional thermoelastic material properties.

    Attach this object to a native elastic solid with ``thermal=properties``
    or ``material.set_thermal_properties(properties)`` before constructing the
    weakforms. ``alpha`` is a scalar isotropic expansion coefficient, three
    normal coefficients, or six engineering-strain components in initial
    global axes. ``specific_heat`` is the capacity per mass at fixed strain;
    ``density`` is the reference density in finite strain.

    Small-strain feedback is linearized at ``reference_temperature``; finite
    strain uses the actual absolute temperature and the logarithmic strain.
    ``coupling=False`` retains thermal expansion and disables feedback.
    Conductivity is the spatial tensor in current global axes; ``heat_source``
    is supplied per reference volume in finite strain.
    """

    def __init__(
        self,
        thermal_conductivity,
        specific_heat,
        density,
        name="",
        *,
        alpha=0.0,
        reference_temperature=293.15,
        coupling=True,
        heat_source=0.0,
    ):
        ConstitutiveLaw.__init__(self, name)
        if np.isscalar(thermal_conductivity):
            self.thermal_conductivity = [
                [thermal_conductivity, 0, 0],
                [0, thermal_conductivity, 0],
                [0, 0, thermal_conductivity],
            ]
        else:
            self.thermal_conductivity = thermal_conductivity

        self.specific_heat = specific_heat
        self.density = density
        expansion = np.asarray(alpha, dtype=float)
        if expansion.shape not in ((), (3,), (6,)) or not np.all(
            np.isfinite(expansion)
        ):
            raise ValueError(
                "alpha must be a finite scalar or a 3- or 6-component vector."
            )
        if not np.isscalar(reference_temperature) or not np.isfinite(
            reference_temperature
        ):
            raise ValueError("reference_temperature must be a finite scalar.")
        self.alpha = float(expansion) if expansion.ndim == 0 else expansion.copy()
        self.reference_temperature = float(reference_temperature)
        self.coupling = bool(coupling)
        self.heat_source = heat_source

    def _expansion_vector(self):
        expansion = np.asarray(self.alpha, dtype=float)
        if expansion.ndim == 0:
            return np.array([float(expansion)] * 3 + [0.0] * 3)
        if expansion.shape == (3,):
            return np.concatenate((expansion, np.zeros(3)))
        return expansion
