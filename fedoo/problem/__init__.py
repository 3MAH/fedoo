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

.. _stabilization_strategies:

    
Non-Linear Solver
=================

Fedoo provides a robust framework for solving non-linear system, arising from structural
mechanics or similar field problems, using a modified Newton-Raphson (NR) approach.
The class :class:`NonLinear` is deticated to the resolution of such complicated
problems.

Newton-Raphson Algorithm
------------------------

The Newton-Raphson algorithm is an iterative process used to find the equilibrium state
of a non-linear system. Starting from an initial guess from :meth:`nonlinear.elastic_prediction`,
the solver iteratively updates the displacement increment until the internal and
external forces are balanced within a specified tolerance.

Available Convergence Criteria
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can define how the solver determines convergence via :meth:`nonlinear.set_nr_criterion`.
Three criteria are available:

* **Force** (default): Measures the residual (out-of-balance) forces relative to the
  external applied forces.
* **Displacement**: Measures the norm of the displacement increment relative to the
  initial increment of the step.
* **Work**: Evaluates the energy (dot product of displacement and residual) relative to
  the initial work increment.

Key Newton-Raphson Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following parameters can be adjusted via :meth:`nonlinear.set_nr_criterion` or the 
``nr_parameters`` dictionary:

* **tol**: The convergence tolerance (default: ``5e-3``). A smaller value increases 
  precision but may require more iterations.
* **max_subiter**: The maximum number of NR iterations allowed per time step 
  before failure. The default is 10, but for contact or highly non-linear 
  behavior, it is recommended to increase this value.
* **err0**: The reference error used for normalization. If ``None`` (default), 
  it is assessed based on the criterion:
  
  * **Displacement**: Calculated from the current increment norm, adjusted by 
    1% of total displacement for numerical stability.
  * **Force**: Calculated from the norm of the current external forces.
  * **Work**: Initialized once at the start of the increment and stored.
  
  If the calculated reference is zero, it defaults to ``1`` to prevent 
  division by zero.
* **dt_increase_niter**: Threshold for "easy" convergence. If a step converges 
  in fewer iterations than this value, the time step ``dt`` is increased.
* **check_early_divergence**: If ``True``, the solver aborts the increment if the 
  error spikes to 100 times the previous iteration error (or becomes ``NaN`` 
  or ``inf``), or if the error increases/stagnates (less than 0.1% decrease) for 4 
  consecutive iterations.
* **norm_type**: Defines the mathematical norm used to compute the error.
  (default: ``2`` for Euclidean) 

The ``nlsolve`` Method
----------------------

The :meth:`nonlinear.nlsolve` method manages the time-steering logic, calling the NR 
loop for each increment and handling time-step adaptations.

Time-Steering Parameters
~~~~~~~~~~~~~~~~~~~~~~~~

* **dt**: The initial time increment (default: ``0.1``).
* **update_dt**: If ``True``, the solver automatically shrinks ``dt`` (by 
  0.25x) on failure or expands it (by 1.25x) on quick convergence. If ``False``
  the solver fail since it doesn't reach convergence within the allowed max_subiter
  iterations.
* **dt_increase_niter**: Threshold for increasing the time step. When 
  ``update_dt`` is ``True``, the time increment is multiplied by 1.25 if the 
  Newton–Raphson loop converges in strictly fewer than this many iterations. 
  Defaults to ``max_subiter // 3``.
* **dt_min**: The safety floor for time-stepping; if ``dt`` falls below this 
  value, the solver raises a ``RuntimeError``.
* **tmax / t0**: Define the initial and final time for the simulation.

Output and Callback Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The solver provides flexible ways to save data or execute custom logic:

* **save_at_exact_time**: If ``True``, the solver adjusts the final ``dtime`` 
  of a sequence to land exactly on an output time interval defined by 
  ``interval_output``.
* **interval_output**: 
    * If ``save_at_exact_time`` is ``True``: Defines the time frequency of 
      outputs.
    * If ``False``: Defines the number of increments between outputs.
    * If ``-1`` (default): Automatically set to ``dt`` or ``1`` iteration.
* **callback**: A user-defined function executed during the resolution.
* **exec_callback_at_each_iter**: If ``True``, the callback runs after every 
  successful time step; otherwise, it only runs when an output is requested.


Overcoming Simulation Instabilities
===================================

Non-linear finite element simulations often encounter convergence difficulties due to 
material softening, contact transitions, or geometric instabilities. **fedoo** provides
several strategies to stabilize these problems.

Numerical Stabilization Techniques
----------------------------------

1. Handling Incompressibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
For nearly incompressible materials encountered in solid mechanics (Poisson's 
ratio :math:`\nu \approx 0.5`), standard elements may suffer from 
**volumetric locking**. Fedoo offers three primary solutions:

* **F-bar Method**: 
    * Set the ``fbar`` attribute to ``True`` in the 
      :class:`fedoo.weakform.StressEquilibrium` weak form.
    * For higher accuracy in the consistent tangent matrix, use the specialized 
      :class:`fedoo.weakform.StressEquilibriumFbar` weak form.
* **Reduced Integration**: Use the :class:`fedoo.weakform.StressEquilibriumRI` 
  weak form. This evaluates volumetric terms at fewer integration points to 
  relax the incompressibility constraint. Reduced integration is known to be 
  prone to **hourglass** instabilities; this weak form includes an hourglass 
  control stiffness to mitigate this.
* **Mixed Displacement/Pressure**: For hybrid strategies introducing 
  additional "Pressure" DOFs, use the 
  :class:`fedoo.weakform.StressEquilibriumMixed` weak form. The field 
  interpolations must satisfy the **LBB conditions** (Ladyzhenskaya-Babuška-Brezzi), 
  stating the displacement interpolation should be richer than the pressure 
  one. To define a different interpolation for the Pressure field, a 
  ``CombinedElement`` must be defined:

  .. code-block:: python

      import fedoo as fd
      # Define a quadratic element with linear pressure (Taylor-Hood)
      new_elm = fd.lib_elements.element_list.CombinedElement("quad8lbb", "quad8")
      new_elm.set_variable_interpolation("Pressure", "quad4")
      wf = fd.weakform.StressEquilibriumMixed(material, bulk_modulus=kappa)
      assembly = fd.Assembly.create(wf, my_mesh, elm_type="quad8lbb")


2. Line Search Algorithms
~~~~~~~~~~~~~~~~~~~~~~~~~
Line search prevents the solver from taking steps that are too large, which can 
lead to non-physical states or divergence.

* **Usage**: Enable via the :meth:`nonlinear.add_line_search` method.
* **Mechanism**: Scales the displacement increment :math:`d\mathbf{U}` by a 
  factor :math:`\eta \in (0, 1]` to minimize the residual norm along the 
  search direction.

3. Adaptive Stiffness (Blending)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Useful for elastoplasticity or damage modeling where a sudden change in 
direction requires a "safe" search direction.

* **Mechanism**: Blends the tangent matrix :math:`\mathbf{K_T}` with the 
  initial elastic matrix :math:`\mathbf{K_E}` using a scalar :math:`\xi`:
  
  .. math:: \mathbf{K} = (1-\xi)\mathbf{K_T} + \xi\mathbf{K_E}

* **Usage**: Enable ``adaptive_stiffness=True`` in the Newton-Raphson 
  parameters via :meth:`nonlinear.set_nr_criterion`.
* **Benefit**: If the tangent prediction diverges, the solver automatically 
  increases :math:`\xi` (shifting toward the elastic matrix) to restore 
  stability. It then attempts to decrease :math:`\xi` back to 0 as 
  convergence improves.

4. Eigenvalue Shifting
~~~~~~~~~~~~~~~~~~~~~~
For strong instabilities (such as buckling or snap-through) where the tangent 
stiffness matrix becomes non-positive definite, the **Eigenvalue Shift** technique might
be effective. Note that this method is computationally 
expensive with limited results, and should only be used as a last resort.

* **Mechanism**: Adds a scaled identity matrix :math:`\alpha \mathbf{I}` to the 
  tangent stiffness :math:`\mathbf{K_T}`.
* **Usage**: Enable ``eigenvalue_shift=True`` in the Newton-Raphson parameters. 
* **Benefit**: Forces the matrix to remain positive definite, ensuring the 
  solver always finds a descent direction.

5. Static vs. Dynamic Fallback
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If a static simulation fails despite the above techniques, the problem may be 
inherently dynamic (e.g., rapid snap-through or post-buckling).

* **Strategy**: Switch to a **Dynamic Solver**. 
* **Stabilization**: Add **Rayleigh Damping** (:math:`\mathbf{C} = a\mathbf{M} + b\mathbf{K}`) to dissipate high-frequency 
  numerical noise and stabilize the inertial response during sudden transitions.
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
