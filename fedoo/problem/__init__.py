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
   Modal
   LinearBuckling
   LinearNewmark
   NonLinear
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


Result output
=============

For every problem type, calling :meth:`~fedoo.core.problem.Problem.add_output`
enables automatic result writing during the corresponding high-level solution
procedure (``solve``, ``solve_history`` or ``nlsolve``). With no registered
output, these procedures do not write files automatically. Output registrations
can be removed between analysis stages with
:meth:`~fedoo.core.problem.Problem.clear_outputs`; existing files and in-memory
frames are retained. :meth:`~fedoo.core.problem.Problem.save_results` remains
available for user-managed solution loops.

For the multiframe ``fdh5`` format, ``add_output`` accepts a ``write_mode``.
Its default, ``"overwrite"``, starts a new file; ``"append"`` continues an
existing file after its highest stored iteration, including when it was
produced by another problem; and ``"error"`` refuses to reuse an existing
file. Other formats, including ``fdz``, retain their existing overwrite
behavior. This makes staged output explicit, for example:

.. code-block:: python

   preload.add_output("history.fdh5", ["Disp", "Stress"])
   preload.solve()

   continuation.add_output(
       "history.fdh5", ["Disp", "Stress"], write_mode="append"
   )
   continuation.nlsolve()


Eigenvalue analyses
===================

Eigenvalue problems support solver choices tailored to symmetric generalized 
eigenproblems:

* ``"auto"``: Automatically selects between a dense or sparse solver based on
  problem size.
* ``"eigsh"``: Forces the SciPy sparse solver (ARPACK).
* ``"eigh"``: Forces the SciPy dense solver.

Solver options (such as tolerance or workspace settings) are configured using 
:meth:`set_solver` and persistent across solves. Solve-specific parameters, such as 
the number of requested modes (``n_modes``) and the optional shift target (``sigma``), 
are specified directly during the solve phase.

.. code-block:: python

    modal.set_solver("eigsh", tol=1e-9, ncv=30)
    modal.solve(n_modes=10, sigma=0.0)

Modal analysis
--------------

:class:`Modal` computes the natural frequencies and mass-normalized mode
shapes of an undamped linear mechanical model. The stiffness and consistent
mass matrices are obtained from the same assembly/storage declarations as
linear transient dynamics. For example:

.. code-block:: python

   modal = fd.problem.Modal(assembly)
   modal.bc.add("Dirichlet", fixed_nodes, "Disp", 0.0)
   modal.add_output("modes.fdh5", assembly, ["Disp", "Stress"])
   modal.solve(n_modes=10)

   print(modal.frequencies)
   modal.set_mode(0)  # expose the first mode through get_disp/get_results

Dirichlet and multi-point constraints are applied to both operators. Prescribed
displacements must be homogeneous. Registered outputs are written automatically
by :meth:`Modal.solve`, with one frame per mode and its mode number, eigenvalue
and frequency included as scalar metadata. To defer output, solve before
calling :meth:`Modal.add_output`, then register the desired fields and call
:meth:`Modal.save_modes` explicitly.

Linear buckling analysis
------------------------

:class:`LinearBuckling` computes critical load factors and buckling modes from
the converged state of a prestressed assembly. It separates the material and
initial-stress stiffnesses by assembling shallow weak-form copies with
geometric stiffness disabled and enabled, respectively. When possible, the
matrix already assembled by the preload problem is reused:

.. code-block:: python

   preload = fd.problem.NonLinear(assembly, nlgeom="TL")
   preload.bc.add("Dirichlet", fixed_nodes, "Disp", 0.0)
   preload.nlsolve()

   buckling = fd.problem.LinearBuckling(preload)
   buckling.bc.add("Dirichlet", fixed_nodes, "Disp", 0.0)
   buckling.solve(n_modes=5)

   print(buckling.load_factors)

Set ``reuse_assembled_matrix=False`` to rebuild both stiffness matrices. The
first implementation supports fixed-geometry and total-Lagrangian preload
states; updated-Lagrangian states are rejected.


Time integration
================

Fedoo separates the spatial formulation from the time-integration scheme.
A weak form defines its static contribution and may declare storage and
dissipation terms, while the problem selects an integrator from
:mod:`fedoo.time`. This allows the same assembly to be used for a static
analysis or with different transient schemes.

Time-dependent weak forms are classified by an evolution category:

* :data:`fedoo.time.FIRST_ORDER` is used for equations such as heat diffusion
  and is integrated with schemes such as
  :class:`fedoo.time.BackwardEuler`.
* :data:`fedoo.time.SECOND_ORDER` is used for structural dynamics and is
  integrated with an implicit or explicit second-order scheme.

Storage must be declared by the physical model. For structural dynamics this
usually comes from ``material.set_density(rho)`` or
``weakform.set_inertia(rho)``. Fedoo does not silently introduce a default
mass when no inertia is provided. Rayleigh damping may be attached with
``weakform.set_damping(alpha=..., beta=...)``; different leaf assemblies may
use different damping coefficients.

Without a compatible time integrator, :class:`Linear` and :class:`NonLinear`
remain static and storage or dissipation declarations are not included in the
equations. :class:`ExplicitDynamic` is always transient and creates a
:class:`fedoo.time.CentralDifference` integrator by default.

Linear transient problems
-------------------------

:class:`Linear` provides an implicit second-order path for genuinely linear
problems. It accepts :class:`fedoo.time.Newmark` and
:class:`fedoo.time.GeneralizedAlpha`. The stiffness, mass and damping matrices
are assembled and cached by :meth:`Linear.initialize`; subsequent increments
normally rebuild only the effective right-hand side. Consequently, changing
material state, geometry, contact or tangent stiffness requires
:class:`NonLinear` instead.

The integrator can be passed to the constructor:

.. code-block:: python

   import fedoo as fd

   pb = fd.problem.Linear(
       assembly,
       time_step=dt,
       integrator=fd.time.Newmark(beta=0.25, gamma=0.5),
   )
   pb.solve_history(tmax, interval_output=output_dt,
                    update_weakform=True)

or attached before initialization:

.. code-block:: python

   pb = fd.problem.Linear(assembly, time_step=dt)
   pb.set_time_integrator(fd.time.SECOND_ORDER, fd.time.Newmark())

:meth:`Linear.solve_time_increment` advances one increment, while
:meth:`Linear.solve_history` manages a complete history. Set
``update_weakform=True`` when stresses, strains or other assembly-derived
fields must be refreshed at every increment. This option does not rebuild the
cached linear operators. :class:`LinearNewmark` is an alias of
:class:`Linear` provided for convenience.

Nonlinear transient problems
----------------------------

:class:`NonLinear` integrates time-dependent terms inside the nonlinear weak
form and evaluates them during the Newton--Raphson iterations. It may attach
one integrator per evolution category, which also permits coupled problems
containing both first- and second-order fields:

.. code-block:: python

   pb = fd.problem.NonLinear(assembly, nlgeom=True)
   pb.set_time_integrator(fd.time.FIRST_ORDER,
                          fd.time.BackwardEuler())
   pb.set_time_integrator(fd.time.SECOND_ORDER,
                          fd.time.Newmark())
   pb.nlsolve(dt=dt, tmax=tmax, interval_output=output_dt)

Compatible storage and dissipation weak forms are compiled when the problem
is initialized. Once transient weak forms have been compiled, their
integrator cannot safely be replaced or removed on the same problem object;
create a new problem if another scheme is required. An integrator that has
only been attached, but not yet compiled, can be removed by passing ``None``
to :meth:`NonLinear.set_time_integrator`.

:meth:`NonLinear.solve_time_increment` performs the Newton--Raphson loop for
one increment. :meth:`NonLinear.nlsolve` manages a complete history, including
adaptive time-step reduction and retry after failed increments.
:class:`NonLinearNewmark` is a convenience factory that attaches Newmark and,
when required, Backward Euler integrators.

Explicit transient problems
---------------------------

:class:`ExplicitDynamic` accepts explicit second-order integrators only. Its
default is :class:`fedoo.time.CentralDifference`; generalized-alpha numerical
dissipation is available through
:class:`fedoo.time.ExplicitGeneralizedAlpha` or its
:class:`fedoo.time.ExplicitNewmark` alias:

.. code-block:: python

   pb = fd.problem.ExplicitDynamic(
       assembly,
       time_step=dt,
       integrator=fd.time.CentralDifference(),
       mass_lumping=True,
   )
   pb.solve_history(tmax, interval_output=output_dt,
                    update_weakform=False)

The finite-element mass is row-sum lumped and cached by default. Use
``mass_lumping=False`` for a consistent mass matrix. The integrator must be
selected before :meth:`ExplicitDynamic.initialize` and cannot be changed
afterwards.

For a linear problem with fixed geometry, state-independent external loads
and fixed contact state,
``update_weakform=False`` reuses the cached stiffness and is the inexpensive
path. Use ``update_weakform=True`` for nonlinear constitutive laws, geometric
nonlinearity, evolving contact or other state-dependent forces. The managed
path then evaluates and commits the constitutive state every increment.
``update_mass`` independently controls whether mass contributions are
refreshed.

:meth:`ExplicitDynamic.solve_time_increment` advances one complete increment,
and :meth:`ExplicitDynamic.solve_history` manages a complete history without
Newton iterations. Advanced user loops may call
:meth:`ExplicitDynamic.prepare_time_increment`,
:meth:`ExplicitDynamic.apply_boundary_conditions`,
:meth:`ExplicitDynamic.solve`, :meth:`ExplicitDynamic.update` and
:meth:`ExplicitDynamic.set_start` separately.

See :mod:`fedoo.time` for the definitions and parameters of the individual
integration schemes.

.. _stabilization_strategies:

.. currentmodule:: fedoo.problem

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
**volumetric locking**. Fedoo offers the following solutions:

* **Element-level methods**: pass ``incompressibility`` to
  :class:`fedoo.weakform.StressEquilibrium` (default: ``None``, standard
  displacement formulation). These methods introduce no additional global
  unknowns.

  .. code-block:: python

      wf = fd.weakform.StressEquilibrium(material, incompressibility="auto")

  * ``"auto"``: select ``"mean_dilatation"`` where it is available and retain
    the standard formulation otherwise.
  * ``"mean_dilatation"``: recommended B-bar method, based on a
    volume-weighted projection of the element dilatation. It gives a symmetric
    tangent matrix and supports both low- and high-order elements.
    One pressure value per element is used for ``quad4``, ``hex8``, ``wed6``,
    ``tri6`` and ``tet10`` elements, a linear pressure for ``quad8``, ``quad9``,
    ``hex20``, ``wed15`` and ``wed18`` elements. It is available for small
    strain and updated-Lagrangian analyses in ``3D`` and ``2Dplane``. In
    finite strain, set ``geometric_stiffness = True`` to get the consistent
    tangent matrix.
  * ``"sri"``: legacy low-order B-bar method, with the dilatation evaluated at
    the element center. It is available for ``quad4`` and ``hex8``. It is most
    appropriate for small-strain analyses. Its finite-strain formulation is
    not consistent and emits a warning with updated Lagrangian; prefer
    ``"mean_dilatation"`` or ``"fbar"``. The total-Lagrangian formulation is
    not supported.
  * ``"fbar"``: F-bar method with the volume change evaluated at the element
    center. For ``quad4`` and ``hex8`` in ``3D`` or
    ``2Dplane``, its consistent tangent matrix is nonsymmetric. In small
    strain, it reduces to a center-based B-bar correction. Other elements,
    ``2Daxi``, and total-Lagrangian analyses retain the F-bar kinematic
    correction but use the standard tangent, with a warning.

  In plane stress (``2Dstress``), these options are ignored and the standard
  plane-stress formulation is used. They require full integration and do not
  improve constant-strain elements such as ``tri3`` and ``tet4``; use
  quadratic elements or the mixed formulation instead. With one pressure
  value per ``quad4`` or ``hex8`` element, displacement results are generally
  accurate but the recovered pressure may exhibit checkerboard oscillations.
* **Reduced Integration**: Use the :class:`fedoo.weakform.StressEquilibriumRI`
  weak form for one-point integration of ``quad4`` or ``hex8`` elements. It
  includes hourglass control to suppress the zero-energy modes introduced by
  reduced integration.
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
  factor :math:`\eta \in (0, 1]`. The ``natural`` mode combines residual
  decrease with an affine-invariant correction test, ``minimize`` provides
  the residual-descent methods, and ``safeguard`` only rejects invalid
  kinematics. If geometric backtracking finds no valid trial, the current
  increment fails and follows the solver's normal time-step reduction path.
* **Dirichlet increments**: By default, ``apply_to_bc=True`` also scales a
  prescribed displacement increment when required to keep its trial state
  kinematically valid. Its unapplied part is retained for the following Newton
  corrections, and convergence is accepted only once the complete prescribed
  increment has been applied. Residual and natural acceptance tests start only
  after that remainder reaches zero. Set ``apply_to_bc=False`` to apply
  prescribed displacements in one full step without the validity safeguard.

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

5. Static vs. Dynamic Fallback
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
  The legacy :class:`fedoo.weakform.ImplicitDynamic` weak form is kept for
  pedagogical purposes, but new transient models should prefer
  problem-level time integrators.
"""

from .explicit_dynamic import ExplicitDynamic
from .linear import Linear, LinearNewmark
from .linear_buckling import LinearBuckling
from .modal import Modal
from .non_linear import NonLinear, NonLinearNewmark

__all__ = [
    "Linear",
    "LinearBuckling",
    "Modal",
    "LinearNewmark",
    "NonLinear",
    "NonLinearNewmark",
    "ExplicitDynamic",
]
