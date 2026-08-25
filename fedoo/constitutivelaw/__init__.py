"""Constitutive Law module.

=========================================================
Constitutive Law (:mod:`fedoo.constitutivelaw`)
=========================================================

.. currentmodule:: fedoo.constitutivelaw

The constitutive law module include several classical mechancical
constitutive laws. These laws are required to create some weak formulations.

The ConstitutiveLaw library contains the following classes:

Solid mechanical constitutive laws
======================================

These laws should be associated with :py:class:`fedoo.weakform.StressEquilibrium`

.. autosummary::
   :toctree: generated/
   :template: custom-class-template.rst

   ElasticIsotrop
   ElasticOrthotropic
   ElasticAnisotropic
   CompositeUD
   ElastoPlasticity
   FE2
   Simcoon

Interface mechanical constitutive laws
======================================

These laws should be associated with :py:class:`fedoo.weakform.StressEquilibrium`

.. autosummary::
   :toctree: generated/
   :template: custom-class-template.rst

   CohesiveLaw
   Spring

.. _beam_constitutive_laws:

Beam constitutive laws
======================================

These laws should be associated with :py:class:`fedoo.weakform.InterfaceForce`

.. autosummary::
   :toctree: generated/
   :template: custom-class-template.rst

   BeamProperties
   BeamCircular
   BeamPipe
   BeamRectangular

Shell constitutive laws
======================================

These laws should be associated with :py:class:`fedoo.weakform.PlateEquilibrium`

.. autosummary::
   :toctree: generated/
   :template: custom-class-template.rst

   ShellLaminate
   ShellLaminateNonLinear
   ShellHomogeneous
   ShellHomogeneousNonLinear

Thermal constitutive law
======================================

These laws should be associated with :py:class:`fedoo.weakform.HeatEquation`

.. autosummary::
   :toctree: generated/
   :template: custom-class-template.rst

   ThermalProperties

"""

from .beam import BeamCircular, BeamPipe, BeamProperties, BeamRectangular
from .cohesivelaw import CohesiveLaw
from .composite_ud import CompositeUD
from .elastic_anisotropic import ElasticAnisotropic
from .elastic_isotrop import ElasticIsotrop
from .elastic_orthotropic import ElasticOrthotropic
from .elasto_plasticity import ElastoPlasticity
from .fe2 import FE2
from .heterogeneous import Heterogeneous
from .shell import (
    ShellBase,
    ShellHomogeneous,
    ShellHomogeneousNonLinear,
    ShellLaminate,
    ShellLaminateNonLinear,
)
from .permeability import HolmesMowPermeability, KozenyCarmanPermeability
from .poro_fluid import PoroFluidProperties
from .simcoon_umat import Simcoon
from .spring import Spring
from .thermal_prop import ThermalProperties

__all__ = [
    "BeamCircular",
    "BeamPipe",
    "BeamProperties",
    "BeamRectangular",
    "CohesiveLaw",
    "CompositeUD",
    "ElasticAnisotropic",
    "ElasticIsotrop",
    "ElasticOrthotropic",
    "ElastoPlasticity",
    "FE2",
    "Heterogeneous",
    "ShellBase",
    "ShellHomogeneous",
    "ShellHomogeneousNonLinear",
    "ShellLaminate",
    "ShellLaminateNonLinear",
    "HolmesMowPermeability",
    "KozenyCarmanPermeability",
    "PoroFluidProperties",
    "Simcoon",
    "Spring",
    "ThermalProperties",
]
