"""
Weak Formulation (:mod:`fedoo.libWeakForm`)
=============================================

In fedoo, the differential equations related to the problem to solve are written 
using weak formulations. 
The WeakForm library of Fedoo includes a few classical weak formulation. Each weak formulation is a class
deriving from the WeakForm class. The developpement 
of new weak formulation is easy by copying and modifying and existing class. 

The WeakForm library contains the following classes: 

.. autosummary::   
   :toctree: generated/
   :template: custom-class-template.rst

   InternalForce
   InterfaceForce
   Beam
   BernoulliBeam
   Plate
   Inertia
   ParametricBernoulliBeam
"""


import pkgutil

for loader, module_name, is_pkg in  pkgutil.walk_packages(__path__):
    module = loader.find_module(module_name).load_module(module_name)
    exec('from .'+module_name+' import *')

    
