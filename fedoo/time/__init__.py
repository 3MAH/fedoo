r"""Problem-level time integration helpers.

=====================================
Time Integration (:mod:`fedoo.time`)
=====================================

The :mod:`fedoo.time` module contains time integrators that are attached to a
problem rather than embedded directly in a weak form. Weak forms describe the
static equilibrium, storage, and dissipation terms, while the problem selects
which temporal scheme advances each evolution category.

For example, a mechanical weak form can expose a second-order storage term
through a mass or inertia weak form, and the problem can advance it with a
Newmark or generalized-alpha integrator:

.. code-block:: python

   pb = fd.problem.NonLinear(assembly)
   pb.set_time_integrator(fd.time.SECOND_ORDER, fd.time.Newmark())

Thermal diffusion is a first-order evolution and can be advanced independently:

.. code-block:: python

   pb.set_time_integrator(fd.time.FIRST_ORDER, fd.time.BackwardEuler())

Static analysis
===============

A problem remains static as long as no time integrator is attached. Weak forms
may still expose storage metadata, such as a density or heat capacity, but this
metadata is ignored until a compatible integrator is selected:

.. code-block:: python

   pb = fd.problem.NonLinear(assembly)
   pb.nlsolve()

An integrator can also be removed before the problem is initialized by passing
``None``:

.. code-block:: python

   pb.set_time_integrator(fd.time.SECOND_ORDER, fd.time.Newmark())
   pb.set_time_integrator(fd.time.SECOND_ORDER, None)

Once the problem has been initialized, compatible weak forms may already have
been compiled into transient weak forms. At that point the integrator cannot be
removed safely from the same problem object; create a new static problem, or
change the assembly before removing the integrator.

Evolution categories
====================

The predefined categories are :data:`FIRST_ORDER` and :data:`SECOND_ORDER`.
They are instances of :class:`TimeEvolution` and are used by
:meth:`fedoo.problem.NonLinear.set_time_integrator` to match weak forms with
compatible schemes.

Integrators
===========

.. autosummary::
   :toctree: generated/
   :template: custom-class-template.rst

   BackwardEuler
   GeneralizedAlpha
   Newmark
   TimeEvolution

Second-order schemes
====================

:class:`GeneralizedAlpha` is the general second-order integrator. In fedoo's
convention, ``alpha_m = 0`` and ``alpha_f = 0`` gives the classical endpoint
Newmark scheme. The default values of ``beta`` and ``gamma`` are chosen from

.. math::

   \gamma = \frac{1}{2} - \alpha_m + \alpha_f,
   \qquad
   \beta = \frac{1}{4}(1 - \alpha_m + \alpha_f)^2.

The :class:`Newmark` constructor is kept as the compact interface for the
common case:

.. code-block:: python

   fd.time.Newmark(beta=0.25, gamma=0.5)

which is equivalent to:

.. code-block:: python

   fd.time.GeneralizedAlpha(alpha_m=0.0, alpha_f=0.0,
                            beta=0.25, gamma=0.5)

Dissipation
===========

Physical damping is defined on the weak form so that only selected parts of a
multi-assembly problem are damped:

.. code-block:: python

   wf = fd.weakform.StressEquilibrium(material)
   wf.set_dissipation(alpha=0.1, beta=0.0)

The keyword form defines Rayleigh damping
:math:`\mathbf{C} = \alpha\mathbf{M} + \beta\mathbf{K}`. A dedicated
dissipative weak form can also be passed to ``set_dissipation``.
"""

from fedoo.core.time_evolution import FIRST_ORDER, SECOND_ORDER, TimeEvolution

from .backward_euler import BackwardEuler
from .common import RayleighDamping
from .generalized_alpha import GeneralizedAlpha
from .newmark import Newmark

__all__ = [
    "BackwardEuler",
    "FIRST_ORDER",
    "GeneralizedAlpha",
    "Newmark",
    "RayleighDamping",
    "SECOND_ORDER",
    "TimeEvolution",
]
