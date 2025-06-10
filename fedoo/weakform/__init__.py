"""WeakForm module.

=============================================
Weak Formulation (:mod:`fedoo.weakform`)
=============================================

In fedoo, the differential equations related to the problem to solve are written
using weak formulations. The weak formations are defined in the WeakForm objects.

  * The WeakForm object include the differential operators (with virtual fields) and can be automatically updated at each time step
    for non linear weak formulations.

  * When created, a WeakForm object doesn't in general know the domain of integration.
    The domain of integration is defined using a :py:class:`Mesh <fedoo.Mesh>`, and is only introduced when creating the corresponding :py:class:`Assembly <fedoo.Assembly>`.

The weakform library of Fedoo includes a few classical weak formulations. Each weak formulation is a class
deriving from the WeakForm class. The developpement
of new weak formulation is easy by copying and modifying an existing class.

The WeakForm library contains the following classes:

.. autosummary::
   :toctree: generated/
   :template: custom-class-template.rst

   StressEquilibrium
   ImplicitDynamic
   StressEquilibriumRI
   SteadyHeatEquation
   HeatEquation
   BeamEquilibrium
   SpringEquilibrium
   PlateEquilibrium
   PlateEquilibriumFI
   PlateEquilibriumSI
   Inertia
   InterfaceForce
   DistributedLoad
   ExternalPressure
   HourglassStiffness
   StressEquilibriumFbar
"""

from .beam import BeamEquilibrium
from .spring import SpringEquilibrium
from .beam_parametric import ParametricBeam, ParametricBernoulliBeam
from .heat_equation import (
    HeatEquation,
    SteadyHeatEquation,
    TemperatureTimeDerivative,
)
from .implicit_dynamic import ImplicitDynamic, ImplicitDynamic2
from .inertia import Inertia
from .interface_force import InterfaceForce
from .plate import (
    PlateEquilibrium,
    PlateEquilibriumFI,
    PlateEquilibriumSI,
    PlateKirchhoffLoveEquilibrium,
    PlateShearEquilibrium,
)
from .stress_equilibrium import StressEquilibrium
from .stress_equilibrium_bbar import (
    StressEquilibriumFbar,
    StressEquilibriumRI,
    HourglassStiffness,
)

__all__ = [
    "BeamEquilibrium",
    "SpringEquilibrium",
    "Inertia",
    "InterfaceForce",
    "PlateEquilibrium",
    "PlateEquilibriumFI",
    "PlateEquilibriumSI",
    "PlateKirchhoffLoveEquilibrium",
    "PlateShearEquilibrium",
    "StressEquilibrium",
    "StressEquilibriumFbar",
    "StressEquilibriumRI",
    "HourglassStiffness",
    "HeatEquation",
    "SteadyHeatEquation",
    "TemperatureTimeDerivative",
    "ImplicitDynamic",
    "ImplicitDynamic2",
    "ParametricBeam",
    "ParametricBernoulliBeam",
]
