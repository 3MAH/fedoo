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

Manage local frames
===================

Three-dimensional mechanical constitutive laws can be defined in a material
coordinate system with :py:meth:`fedoo.ConstitutiveLaw.set_local_frame`. A
frame is a proper orthogonal rotation matrix whose rows are the material axes
expressed in global coordinates. SciPy and Simcoon rotation objects are also
accepted directly.

For example, one uniform initial material orientation can be assigned with:

.. code-block:: python

   from scipy.spatial.transform import Rotation

   frame = Rotation.from_euler("z", 30, degrees=True)
   material.set_local_frame(frame)

Several frames can be supplied at nodes, elements, or Gauss points. Use the
canonical location names ``"Node"``, ``"Element"``, and ``"GaussPoint"``:

.. code-block:: python

   material.set_local_frame(nodal_frames, location="Node")
   material.set_local_frame(element_frames, location="Element")
   material.set_local_frame(gauss_point_frames, location="GaussPoint")

Nodal rotations are smoothly interpolated to the Gauss points using the
finite-element shape functions and a local rotation-vector interpolation.
Element frames are repeated at every Gauss point of their element. A
Gauss-point field must contain exactly the number of points used by the
associated assembly. When ``location`` is omitted, Fedoo infers it from the
number of frames if the choice is unambiguous.

For small-strain analyses, the prescribed material orientation remains fixed.
In finite-strain analyses, it defines the initial material orientation and the
constitutive-law machinery updates the current orientation with the material
rotation. Isotropic laws do not perform unnecessary frame transformations.

Beam and shell frames
---------------------

For beams and shells, the element frame is also a geometrical coordinate
system used by the weak form. This frame should be defined on the assembly:

.. code-block:: python

   assembly.set_element_local_frame(guide=guide, location="Element")

For a beam, the first axis is projected onto the element tangent and the guide
sets the cross-section orientation. For a shell, the third axis remains the
surface normal and the guide sets the in-plane orientation. Guides may be
uniform, elemental, or nodal. The assembly element-frame API is intentionally
limited to beams and shells; anisotropic solid orientations belong to the
constitutive law.

For shell constitutive laws that support an additional material orientation,
that orientation is interpreted relative to the geometrical shell frame
defined on the assembly. Beam constitutive laws currently use isotropic
materials only, so material orientations are ignored. The assembly element
frame defines the beam cross-section orientation.

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
