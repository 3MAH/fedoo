# baseclass
import numpy as np
from fedoo.core.base import ConstitutiveLaw


class ThermalProperties(ConstitutiveLaw):
    def __init__(self, thermal_conductivity, specific_heat, density, name=""):
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
