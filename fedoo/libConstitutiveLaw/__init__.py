"""
=========================================================
Constitutive Law (:mod:`fedoo.libConstitutiveLaw`)
=========================================================

.. currentmodule:: fedoo.libConstitutiveLaw

The constitutive law module inclued several classical mechancical classical 
constitutive laws. This laws are required to create some weak formulations. 

The ConstitutiveLaw library contains the following classes: 

Solid mechanical constitutive laws
======================================

These laws should be associated with :mod:`WeakForm.InternalForce`

.. autosummary::   
   :toctree: generated/
   :template: custom-class-template.rst

   ElasticIsotrop
   ElasticOrthotropic
   ElasticAnisotropic
   CompositeUD
   ElastoPlasticity

Interface mechanical constitutive laws
======================================

These laws should be associated with :mod:`WeakForm.InterfaceForce`

.. autosummary::   
   :toctree: generated/
   :template: custom-class-template.rst

   CohesiveLaw
   Spring

Shell constitutive laws
======================================

These laws should be associated with :mod:`WeakForm.Plate`

.. autosummary::   
   :toctree: generated/
   :template: custom-class-template.rst

   ShellLaminate
   ShellHomogeneous 

"""


import pkgutil

for loader, module_name, is_pkg in  pkgutil.walk_packages(__path__):
    module = loader.find_module(module_name).load_module(module_name)
    exec('from .'+module_name+' import *')

    
