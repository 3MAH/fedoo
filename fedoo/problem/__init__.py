"""
=============================================
Problem (:mod:`fedoo.problem`)
=============================================

Fedoo allow to solve several kinds of Problems that are defined in the Problem library. 

To create a new Problem, use one of the following function: 

.. autosummary::   
   :toctree: generated/
   :template: custom-class-template.rst

   Static
   NonLinearStatic
   Newmark
   NonLinearNewmark
   ExplicitDynamic   

Each of these functions creates an object that is derived from a base class "ProblemBase".
The ProblemBase Class contains all the methods that may be used depending on the kind of Problems.

.. currentmodule:: fedoo.problem.ProblemBase

.. autosummary::   
   :toctree: generated/
   :template: custom-class-template.rst

   ProblemBase   
   
All the ProblemBase methods have an alias function at the root of the Problem Libary which allow 
to run the methods on the active problem. 
"""

import pkgutil

for loader, module_name, is_pkg in  pkgutil.walk_packages(__path__):
    module = loader.find_module(module_name).load_module(module_name)
    exec('from .'+module_name+' import *')

    
