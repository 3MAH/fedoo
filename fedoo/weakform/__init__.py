"""
Weak Formulation (:mod:`fedoo.weakform`)
=============================================

In fedoo, the differential equations related to the problem to solve are written 
using weak formulations. The weak formations are defined in the WeakForm objects.
 
  * The WeakForm object include the differential operators (with virtual fields) and can be automatically updated at each time step 
    for non linear weak formulations. 
  
  * When created, a WeakForm object doesn't in general know the domain of integration. 
    The domain of integration is defined using a :py:class:`Mesh <fedoo.Mesh>`, and is only introduced when creating the corresponding :py:class:`Assembly <fedoo.Assembly>`.
  
  * A WeakForm should only be associated to one Assembly and to one Problem. The WeakForm is often modified by the Problem,
    For instance, an incremental Problem will probably modify initial displacements and initial stresses fields at each time step. 
    To associate the same WeakForm with different Assembly, several indentical WeakForm needs to be created. 

The weakform library of Fedoo includes a few classical weak formulations. Each weak formulation is a class
deriving from the WeakForm class. The developpement 
of new weak formulation is easy by copying and modifying and existing class. 

The WeakForm library contains the following classes: 

.. autosummary::   
   :toctree: generated/
   :template: custom-class-template.rst

   StressEquilibrium
   SteadyHeatEquation
   HeatEquation
   BeamEquilibrium
   PlateEquilibrium
   PlateEquilibriumFI
   PlateEquilibriumSI
   Inertia
   InterfaceForce
"""


import pkgutil

for loader, module_name, is_pkg in  pkgutil.walk_packages(__path__):
    module = loader.find_module(module_name).load_module(module_name)
    exec('from .'+module_name+' import *')

    
