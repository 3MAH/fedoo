"""
=============================================
Problem (:mod:`fedoo.problem`)
=============================================

Fedoo allow to solve several kinds of Problems that are defined in the Problem library. 

To create a new Problem, use one of the following function: 

.. autosummary::   
   :toctree: generated/
   :template: custom-class-template.rst

   Linear   
   NonLinear
   Newmark
   NonLinearNewmark
   ExplicitDynamic   

Each of these functions creates an object that is derived from a base class "ProblemBase".

.. currentmodule:: fedoo.core.base

.. autosummary::   
   :toctree: generated/
   :template: custom-class-template.rst

   ProblemBase   
"""

import pkgutil

for loader, module_name, is_pkg in  pkgutil.walk_packages(__path__):
    module = loader.find_module(module_name).load_module(module_name)
    exec('from .'+module_name+' import *')

# from .Problem_NonLinearStatic import NonLinearStatic
