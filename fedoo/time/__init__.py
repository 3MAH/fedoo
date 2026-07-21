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
   CentralDifference
   ExplicitGeneralizedAlpha
   ExplicitNewmark
   GeneralizedAlpha
   Newmark
   RayleighDamping
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

Explicit second-order schemes
=============================

Explicit problems use the same evolution-category attachment API, but keep
the static-force and lumped-mass operations separate. Central difference can
be selected with:

.. code-block:: python

   pb = fd.problem.ExplicitDynamic(assembly, time_step=dt)
   pb.set_time_integrator(fd.time.SECOND_ORDER,
                          fd.time.CentralDifference())

The primary interface is a user-controlled increment loop. The fixed linear
path reuses stiffness and lumped mass cached during initialization and does
not update the constitutive law implicitly:

.. code-block:: python

   pb.initialize()
   pb.apply_boundary_conditions()
   while pb.time < tmax:
       pb.solve()
       pb.update()

Call ``apply_boundary_conditions()`` again only when boundary conditions,
loads or MPCs change. For a managed nonlinear increment, use
``pb.solve_time_increment(update_weakform=True, set_start=True)``. For a
complete history, ``pb.solve_history(update_weakform=False)`` reuses the
fixed linear operators, while ``update_weakform=True`` updates and commits the
constitutive state every increment. Its ``interval_output`` argument selects
exact output times; increments are temporarily shortened when necessary and
the nominal time step is restored afterwards.

Finite-element mass is row-sum lumped and cached by default. Pass
``mass_lumping=False`` to retain the consistent matrix, and request a mass
refresh explicitly only when storage changes. Configuration-dependent
assembly providers can refresh their small storage blocks independently of
the cached finite-element mass.

For controllable high-frequency dissipation, use
:class:`ExplicitGeneralizedAlpha`, also available under the
:class:`ExplicitNewmark` alias. Both explicit schemes use the inertia declared
by the static weak form, for example through material density or
``weakform.set_inertia(...)``.

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

Different Rayleigh coefficients may be assigned to different leaf assemblies
in an ``AssemblySum``. Weakforms sharing one leaf assembly must use the same
coefficient pair because they also share one assembled stiffness matrix.
"""

from fedoo.core.time_evolution import FIRST_ORDER, SECOND_ORDER, TimeEvolution

from .backward_euler import BackwardEuler
from .common import RayleighDamping
from .explicit import (
    CentralDifference,
    ExplicitGeneralizedAlpha,
    ExplicitNewmark,
    ExplicitSecondOrderIntegrator,
)
from .generalized_alpha import GeneralizedAlpha
from .newmark import Newmark

__all__ = [
    "BackwardEuler",
    "CentralDifference",
    "ExplicitGeneralizedAlpha",
    "ExplicitNewmark",
    "ExplicitSecondOrderIntegrator",
    "FIRST_ORDER",
    "GeneralizedAlpha",
    "Newmark",
    "RayleighDamping",
    "SECOND_ORDER",
    "TimeEvolution",
]
