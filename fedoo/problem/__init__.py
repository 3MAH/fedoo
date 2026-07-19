r"""Problem module.

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

Transient analyses are controlled by problem-level integrators from
:mod:`fedoo.time`. They are attached to a nonlinear problem with
:meth:`NonLinear.set_time_integrator`, which keeps weak forms focused on their
static, storage, and dissipation contributions. Without a time integrator, or
after removing a pending one with ``set_time_integrator(..., None)`` before
initialization, the problem remains static.

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
of a non-linear system. Starting from an initial guess from
:meth:`NonLinear.elastic_prediction`,
the solver iteratively updates the displacement increment until the internal and
external forces are balanced within a specified tolerance.

Available Convergence Criteria
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can define how the solver determines convergence via :meth:`NonLinear.set_nr_criterion`.
Three criteria are available:

* **Force** (default): Measures the residual (out-of-balance) forces relative to the
  external applied forces.
* **Displacement**: Measures the norm of the displacement increment relative to the
  initial increment of the step.
* **Work**: Evaluates the energy (dot product of displacement and residual) relative to
  the initial work increment.

Key Newton-Raphson Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following parameters can be adjusted via :meth:`NonLinear.set_nr_criterion` or the 
``nr_parameters`` dictionary:

* **tol**: The convergence tolerance (default: ``5e-3``). A smaller value increases 
  precision but may require more iterations.
* **max_subiter**: The maximum number of NR iterations allowed per time step 
  before failure. The default is 16, but for contact or highly non-linear 
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

The :meth:`NonLinear.nlsolve` method manages the time-steering logic, calling the NR 
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
* **Reduced Integration**: Use the :class:`fedoo.weakform.stress_equilibrium_ri` 
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


2. Numerical Damping (Artificial Viscosity)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
One of the most effective ways to stabilize a problem is through numerical damping. 
This introduces an artificial viscous force that regularizes the system when the 
stiffness matrix is singular or non-positive definite.

* **Mechanism**: Introduces a velocity-dependent damping force:
  
  .. math:: F_{stab} = c_{stab} \cdot M^* \cdot \frac{\Delta u}{\Delta t}

  where :math:`M^*` is an artificial mass matrix (unit-density volume integrator).
* **Usage**: Add the :class:`fedoo.weakform.ArtificialDamping` weak form to your 
  equilibrium equation:

  .. code-block:: python

      wf = fd.weakform.StressEquilibrium(material)
      # Add 5% energy-based stabilization
      wf += fd.weakform.ArtificialDamping(c_stab=0.05, energy_fraction=True)

* **Energy-Based Adaptation**: If ``energy_fraction=True``, the coefficient 
  ``c_stab`` is automatically scaled at each increment to maintain a target 
  ratio of dissipated stabilization energy to external work. This ensures the 
  damping remains "invisible" to the final physical results.
* **Important Considerations**:
    * **Lumping**: Setting ``mat_lumping=True`` (default) is recommended to 
      diagonalize the stabilization matrix, improving numerical robustness.
    * **Load Jumps**: If external loads change abruptly between iterations, 
      energy-based damping may become unadapted as it relies on the previous 
      converged state's work.
    * **Initial Step**: The very first iteration should be stable enough to 
      converge with minimal damping. If it diverges immediately, consider 
      setting ``energy_fraction=False`` to provide a constant damping floor.

3. Line Search Algorithms
~~~~~~~~~~~~~~~~~~~~~~~~~
Line search prevents the solver from taking steps that are too large, which can 
lead to non-physical states or divergence.

* **Usage**: Enable via the :meth:`NonLinear.add_line_search` method.
* **Mechanism**: Scales the displacement increment :math:`d\mathbf{U}` by a 
  factor :math:`\eta \in (0, 1]` to minimize the residual norm along the 
  search direction.

4. Stiffness Strategies (Blending & Elastic Overrides)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Useful for elastoplasticity or damage modeling where a sudden change in 
direction requires a "safe" search direction.

* **Adaptive Stiffness (Blending)**:
    * **Mechanism**: Blends the tangent matrix :math:`\mathbf{K_T}` with the 
      initial elastic matrix :math:`\mathbf{K_E}` using a scalar :math:`\xi`:
      
      .. math:: \mathbf{K} = (1-\xi)\mathbf{K_T} + \xi\mathbf{K_E}
    
    * **Usage**: Enable ``adaptive_stiffness=True`` in the Newton-Raphson 
      parameters via :meth:`NonLinear.set_nr_criterion`.
    * **Benefit**: If the tangent prediction diverges, the solver automatically 
      increases :math:`\xi` (shifting toward the elastic matrix) to restore 
      stability. It then attempts to decrease :math:`\xi` back to 0 as 
      convergence improves.

* **Forced Elastic Stiffness**:
    * **force_elastic_stiffness**: If ``True``, the solver performs a
      Modified Newton-Raphson (Initial Stiffness) solution, using
      :math:`\mathbf{K_E}` for all iterations. This is slower to converge (linear)
      but highly robust against tangent singularities.
    * **elastic_initial_guess**: If ``True``, forces the use of :math:`\mathbf{K_E}`
      only for the very first iteration of every time increment to provide a stable
      initial direction.

* **One-Time Manual Override**:
    * :meth:`NonLinear.force_elastic_matrix_next_iter`: Use this method to flag the
      solver to recompute and use the elastic stiffness matrix for the **very next**
      Newton-Raphson initial guess only. This is ideal for manually
      "restarting" a stalled increment.

5. Eigenvalue Shifting
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

6. Static vs. Dynamic Fallback
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If a static simulation fails despite the above techniques, the problem may be 
inherently dynamic (e.g., rapid snap-through or post-buckling).

* **Strategy**: Switch to a **Dynamic Solver** by adding a second-order time
  integrator from :mod:`fedoo.time` to a :class:`NonLinear` problem, for
  instance :class:`fedoo.time.Newmark` or
  :class:`fedoo.time.GeneralizedAlpha`.
* **Stabilization**: Add **Rayleigh Damping**
  (:math:`\mathbf{C} = a\mathbf{M} + b\mathbf{K}`) to dissipate high-frequency 
  numerical noise and stabilize the inertial response during sudden transitions.
  The legacy :class:`fedoo.weakform.implicit_dynamic` weak form is kept for
  pedagogical purposes, but new transient models should prefer
  problem-level time integrators.
"""

from .explicit_dynamic import ExplicitDynamic
from .linear import Linear
from .newmark import Newmark
from .non_linear import NonLinear, NonLinearNewmark

__all__ = [
    "Linear",
    "NonLinear",
    "Newmark",
    "NonLinearNewmark",
    "ExplicitDynamic",
]
