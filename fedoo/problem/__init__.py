"""Problem module.

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

Each of these functions creates an object that is derived from the \
   base classes "ProblemBase" or "Problem".

.. currentmodule:: fedoo

.. autosummary::
   :toctree: generated/
   :template: custom-class-template.rst

   fedoo.core.base.ProblemBase
   fedoo.Problem
"""

from .explicit_dynamic import ExplicitDynamic
from .linear import Linear
from .newmark import Newmark
from .nl_newmark import NonLinearNewmark
from .non_linear import NonLinear

__all__ = [
    "Linear",
    "NonLinear",
    "Newmark",
    "NonLinearNewmark",
    "ExplicitDynamic",
]
