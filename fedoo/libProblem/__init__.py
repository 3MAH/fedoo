"""
Problem (:mod:`fedoo.libProblem`)
=============================================

Fedoo allow to solve some different kind of Problems that are defined in the Problem library. 

The Problem library contains the following classes: 

.. autosummary::   
   :toctree: generated/
   :template: custom-class-template.rst

   Static
   NonLinearStatic
   Newmark
   NonLinearNewmark
   ExplicitDynamic   

"""

import pkgutil

for loader, module_name, is_pkg in  pkgutil.walk_packages(__path__):
    module = loader.find_module(module_name).load_module(module_name)
    exec('from .'+module_name+' import *')

    
