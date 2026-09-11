from __future__ import annotations
import numpy as np
from fedoo.core.assembly import Assembly
from fedoo.core.assembly_sum import AssemblySum
from fedoo.core.problem import Problem
from fedoo.core.time_evolution import normalize_time_evolution
from fedoo.problem.line_search import line_search, _line_search_manager
from fedoo.core.base import InvalidKinematicStateError
import warnings
from typing import Callable


class NonLinear(Problem):
    def __init__(self, assembly, nlgeom=False, name="MainProblem"):
        if isinstance(assembly, str):
            assembly = Assembly.get_all()[assembly]

        # A = assembling.current.get_global_matrix() #tangent stiffness matrix
        A = 0  # tangent stiffness matrix - will be initialized only when required
        B = 0
        # D = assembling.get_global_vector() #initial stress vector
        D = 0  # initial stress vector #will be initialized later
        self.print_info = 1  # print info of NR convergence during solve
        self._U = 0  # displacement at the end of the previous converged increment
        self._dU = 0  # displacement increment
        self._initialized = False

        self._err0 = None  # initial error for NR error estimation
        self._alpha = 1  # line search current parameter

        self.nr_parameters = {
            "err0": None,  # default error for NR error estimation
            "criterion": "Force",
            "tol": 5e-3,
            "max_subiter": 16,
            "dt_increase_niter": None,
            "norm_type": 2,
            "force_elastic_stiffness": False,
            "elastic_initial_guess": False,
            "adaptive_stiffness": False,
            "assume_cvg_at_max_subiter": False,
            "check_early_divergence": True,
        }
        """Parameters to set the Newton-Raphson algorithm.

        The available parameters can be modified with the method
        :py:meth:`fedoo.problem.non_linear.set_nr_criterion`.
        """

        # attributes used for line search algorithm or contact management
        self._step_size_callback = None
        self._ls_callbacks = {}  # dict of line search functions
        self._line_search_update = False  # tag used during line_search update
        self._step_filter_callback = None  # OGC per-vertex filter
        self._nr_min_subiter = (
            0  # SDI: minimum NR sub-iterations before accepting convergence
        )
        self._t_fact_inc = None  # frozen t_fact for the current increment
        self._force_elastic_next_iter = False  # flag to force the use of elastic matrix

        self.__assembly = assembly
        super().__init__(A, B, D, assembly.mesh, name, assembly.space)
        self.nlgeom = nlgeom
        self.__assembly.register_global_dofs(self)
        self.time_integrators = {}
        self._time_integrators_compiled = False
        self.t0 = 0
        self.tmax = 1
        self.time = 0
        self.dtime = 0
        self._dtime_prev = 0  # dt of the last completed increment
        # True while set_start closes an increment that has been solved
        # (read by the time integrators, see set_start)
        self._increment_solved = False
        self._dU_old = 0  # last improving iterate, kept by adaptive_stiffness
        self.__iter = 0
        self.__compteurOutput = 0

        self.interval_output = -1  # save results every self.interval_output iter or time step if self.save_at_exact_time = True
        self.save_at_exact_time = True
        self.exec_callback_at_each_iter = False
        self.err_num = 1e-8  # numerical error

    def _run_constraint_hook(self, name):
        for bc in self.bc:
            hook = getattr(bc, name, None)
            if hook is not None:
                hook(self)

    @property
    def n_iter(self):
        """Return the number of iterations made to solve the problem."""
        return self.__iter

    def get_disp(self, name="Disp"):
        """Return the displacement components.


        Parameters
        ----------
        name : str, optional
            Name of the variable to return. For instance, if name == 'DispX'
            return only the X component of displacement.

        Returns
        -------
        numpy.ndarray
        """
        if np.isscalar(self._dU) and self._dU == 0:
            return self._get_vect_component(self._U, name)
        return self._get_vect_component(self._U + self._dU, name)

    def get_rot(self, name="Rot"):
        """Return the rotation components.


        Parameters
        ----------
        name : str, optional
            Name of the variable to return. For instance, if name == 'RotX'
            return only the X component of rotation.

        Returns
        -------
        numpy.ndarray
        """
        if np.isscalar(self._dU) and self._dU == 0:
            return self._get_vect_component(self._U, name)
        return self._get_vect_component(self._U + self._dU, name)

    def get_temp(self):
        """Return the nodal temperature field."""
        if np.isscalar(self._dU) and self._dU == 0:
            return self._get_vect_component(self._U, "Temp")
        return self._get_vect_component(self._U + self._dU, "Temp")

    # Return all the dof for every variable under a vector form
    def get_dof_solution(self, name="all"):
        if np.isscalar(self._dU) and self._dU == 0:
            return self._get_vect_component(self._U, name)
        return self._get_vect_component(self._U + self._dU, name)

    def set_dof(self, name, value):
        """Set the initial converged degree-of-freedom field.

        Nonlinear constitutive state is initialized from this field, so it
        must be prescribed before :meth:`initialize`.
        """
        if self._initialized:
            raise RuntimeError(
                "NonLinear.set_dof() must be called before problem initialization."
            )
        if np.isscalar(self._U):
            vector = self._new_vect_dof()
        else:
            vector = np.asarray(self._U).copy()
        self._set_vect_component(vector, name, value)
        self._U = vector
        self._dU = 0
        return self

    def _update_a(self):
        # dt not used for static problem
        self.set_A(self.__assembly.current.get_global_matrix())

    def _update_d(self, start=False):
        # start not used for static problem
        self.set_D(self.__assembly.current.get_global_vector())

    def set_time_integrator(self, evolution, integrator):
        """Attach or remove a problem-level time integrator.

        The integrator compiles compatible static weakforms before assembly
        initialization. For example::

            pb.set_time_integrator(fd.time.SECOND_ORDER, fd.time.Newmark())

        Passing ``None`` removes the integrator associated with the evolution
        category. This can only be used before a compatible weakform has been
        compiled into its transient form.
        """
        evolution = normalize_time_evolution(evolution)
        if self._time_integrators_compiled:
            for assembly in self.__assembly.iter_leaf():
                weakform = getattr(assembly, "weakform", None)
                if weakform is not None and any(
                    getattr(wf, "_fedoo_time_integrated", False)
                    for wf in weakform.iter_leaf()
                ):
                    # Once compiled, the transient weakforms carry the previous
                    # integrator's coefficients: replacing or removing an
                    # integrator would silently be a no-op, so fail loudly.
                    raise RuntimeError(
                        "Cannot change time integrators after the assembly has been "
                        "compiled for transient analysis. Create a new problem or "
                        "change the assembly before modifying the integrators."
                    )
        if integrator is None:
            self.time_integrators.pop(evolution, None)
            self._time_integrators_compiled = False
            return None

        integrator_evolution = normalize_time_evolution(
            getattr(integrator, "evolution", evolution)
        )
        if integrator_evolution != evolution:
            raise ValueError(
                f"Time integrator {integrator!r} is not compatible with evolution "
                f"{evolution!r}."
            )

        self.time_integrators[evolution] = integrator
        self._time_integrators_compiled = False
        return integrator

    def _has_time_integrated_weakform(self, assembly):
        if getattr(assembly, "_fedoo_time_integrated", False):
            return True
        if isinstance(assembly, AssemblySum):
            return any(
                self._has_time_integrated_weakform(child)
                for child in assembly.list_assembly
            )
        weakform = getattr(assembly, "weakform", None)
        if getattr(weakform, "_fedoo_time_integrated", False):
            return True
        return any(
            getattr(wf, "_fedoo_time_integrated", False)
            for wf in getattr(weakform, "list_weakform", [])
        )

    def _compile_time_integrators(self):
        if self._time_integrators_compiled:
            return
        for evolution, integrator in self.time_integrators.items():
            self.__assembly = integrator.compile_assembly(self.__assembly, evolution)
        self._time_integrators_compiled = True
        self._warn_ignored_storage()

    def _iter_leaf_weakforms(self, assembly):
        if isinstance(assembly, AssemblySum):
            for child in assembly.list_assembly:
                yield from self._iter_leaf_weakforms(child)
            return
        weakform = getattr(assembly, "weakform", None)
        if weakform is None:
            return
        for wf in getattr(weakform, "list_weakform", [weakform]):
            yield wf

    def _iter_assembly_time_providers(self, assembly):
        if isinstance(assembly, AssemblySum):
            for child in assembly.list_assembly:
                yield from self._iter_assembly_time_providers(child)
            return
        if getattr(assembly, "weakform", None) is None and (
            getattr(assembly, "storage", None) is not None
            or getattr(assembly, "dissipation", None) is not None
        ):
            yield assembly

    def _warn_ignored_storage(self):
        """Warn when declared storage/dissipation terms are silently ignored.

        Weakforms and assembly-level providers may declare transient metadata;
        without a matching problem-level integrator the analysis is steady/static
        and these terms are dropped.
        This is legitimate for a deliberately steady analysis, but silent for
        a user migrating from the former transient-by-default weakforms, so a
        warning is emitted once at compile time.
        """
        for assembly in self.__assembly.iter_leaf():
            weakform = getattr(assembly, "weakform", None)
            if weakform is None:
                continue
            for wf in weakform.iter_leaf():
                if getattr(wf, "_fedoo_time_integrated", False):
                    continue
                evolution = getattr(wf, "time_evolution", None)
                if evolution is None or evolution in self.time_integrators:
                    continue
                if (
                    getattr(wf, "storage", None) is not None
                    or getattr(wf, "dissipation", None) is not None
                ):
                    warnings.warn(
                        f"Weakform '{wf.name}' declares storage or dissipation "
                        f"terms for the '{evolution.kind}' time evolution, but no "
                        "matching time integrator is attached to the problem: "
                        "these terms are ignored and the analysis is treated as "
                        "steady/static. For a transient analysis, attach an "
                        "integrator, e.g. pb.set_time_integrator("
                        "fd.time.FIRST_ORDER, fd.time.BackwardEuler()) or "
                        "pb.set_time_integrator(fd.time.SECOND_ORDER, "
                        "fd.time.Newmark()).",
                        UserWarning,
                        stacklevel=3,
                    )

        for assembly in self._iter_assembly_time_providers(self.__assembly):
            evolution = getattr(assembly, "time_evolution", None)
            if evolution is None or evolution in self.time_integrators:
                continue
            warnings.warn(
                f"Assembly provider '{assembly.name}' declares storage or "
                f"dissipation terms for the '{evolution.kind}' time evolution, "
                "but no matching time integrator is attached to the problem: "
                "these terms are ignored and the analysis is treated as "
                "steady/static. Attach a matching integrator with "
                "pb.set_time_integrator.",
                UserWarning,
                stacklevel=3,
            )

    def initialize(self):
        self._compile_time_integrators()
        self.__assembly.initialize(self)
        self.set_A(0)
        self._initialized = True

    def set_start(self, save_results=False, callback=None):
        # dt not used for static problem
        self._nr_min_subiter = 0  # reset SDI for new increment
        self._err0 = self.nr_parameters["err0"]  # initial error for NR error estimation
        # Tell the time integrators whether this call closes an increment that
        # was actually solved: on an empty increment their recurrence would
        # still consume one time step (for Newmark it negates the velocity).
        self._increment_solved = not (np.isscalar(self._dU) and self._dU == 0)
        if self._increment_solved:
            self._U += self._dU
            self._dU = 0
            self.__assembly.set_start(self)
            self._run_constraint_hook("set_start")

            # Save results
            if save_results:
                self.save_results(self.__compteurOutput)
                self.__compteurOutput += 1

            if callback is not None:
                if self.exec_callback_at_each_iter or save_results:
                    callback(self)
        else:
            self.__assembly.set_start(self)
            self._run_constraint_hook("set_start")

    def to_start(self):
        self._dU = 0
        self._nr_min_subiter = 0
        self._t_fact_inc = None
        self._err0 = self.nr_parameters["err0"]  # initial error for NR error estimation
        self.__assembly.to_start(self)
        # NB: the problem deliberately keeps the tangent of the diverged
        # iterate here; the retry's elastic prediction reuses it (A != 0).
        # Refreshing it from the restored assembly was tried and reverted: it
        # drops the IPC contact block and breaks the punch benchmarks.
        self._run_constraint_hook("to_start")

    def update(self, compute="all", updateWeakForm=True):
        """
        Assemble the matrix including the following modification:
            - New initial Stress
            - New initial Displacement
            - Modification of the mesh
            - Change in constitutive law (internal variable)
        Don't Update the problem with the new assembled global matrix and global vector -> use UpdateA and UpdateD method for this purpose
        """
        if self.bc._update_during_inc:
            for bc in self.bc:
                pre_update = getattr(bc, "pre_update", None)
                if pre_update is not None:
                    pre_update(self)

        if updateWeakForm == True:
            self.__assembly.update(self, compute)
        else:
            self.__assembly.current.assemble_global_mat(compute)

        if self.bc._update_during_inc:
            self.update_boundary_conditions()

    def reset(self):
        self.__assembly.reset()

        self.set_A(0)  # tangent stiffness
        self.set_D(0)
        # self.set_A(self.__assembly.current.get_global_matrix()) #tangent stiffness
        # self.set_D(self.__assembly.current.get_global_vector())

        self._U = 0
        self._dU = 0
        self._initialized = False

        self._err0 = self.nr_parameters["err0"]  # initial error for NR error estimation
        self._t_fact_inc = None
        self.t0 = 0
        self.tmax = 1
        self.__iter = 0

    def change_assembly(self, assembling, update=True):
        """
        Modify the assembly associated to the problem and update the problem (see Assembly.update for more information)
        """
        if isinstance(assembling, str):
            assembling = Assembly[assembling]

        self.__assembly = assembling
        self._time_integrators_compiled = False
        if update:
            self.update()

    def _update_step_size_callback(self):
        """Update the line search callback from 'ls_callbacks' attribute."""
        if not self._ls_callbacks:
            self._step_size_callback = None

        elif len(self._ls_callbacks) == 1:
            self._step_size_callback = next(iter(self._ls_callbacks.values()))

        else:
            # Multiple constraints exist: use the manager
            self._step_size_callback = _line_search_manager

    def add_line_search(
        self, method="Quadratic", mode="natural", apply_to_bc=True, name=None
    ):
        r"""Add line search algorithm for the Newton-Raphson solver.

        Line search improves global convergence by scaling the displacement
        increment :math:`dX` by a step size :math:`\alpha \in (0, 1]`. This is
        particularly useful for problems with sharp non-linearities or when
        the initial guess is far from the equilibrium.

        Parameters
        ----------
        method : {'Armijo', 'Residual', 'Energy', 'Quadratic'} or callable, default 'Quadratic'
            The residual-descent strategy used when ``mode='minimize'``:

            * **'Armijo'**: Ensures a "sufficient decrease" in the residual
              using a least-square assumption. Standard for most nonlinear applications.
            * **'Residual'**: Simple backtracking that accepts any step reducing
              the residual norm. Fast but less robust.
            * **'Energy'**: Minimizes the out-of-balance work (residual projected
              onto the search direction). Ideal for snap-through/buckling.
            * **'Quadratic'**: Performs a parabolic interpolation of the objective
              function to jump directly to the estimated minimum.
            * **callable**: If a function is provided, it must follow the signature
              ``user_line_search(pb, dX) -> float`` and will be assigned directly
              as the line search callback (``mode`` is then ignored).

              A custom callback may scale a pending Dirichlet increment. A
              returned ``alpha < 1`` defers its unapplied part to subsequent
              Newton corrections, and convergence is declared only after the
              complete prescribed increment has been applied. The callback
              must therefore eventually accept the remaining increment.
        mode : {'natural', 'minimize', 'safeguard'}, default 'natural'
            The overall line search policy:

            * **'natural'** (default): validity filter, then the step is
              accepted when EITHER the classical Armijo test on
              :math:`\|R\|` OR Deuflhard's affine-invariant test on the
              simplified Newton correction :math:`K^{-1} R(u + \alpha dX)`
              passes (see :func:`fedoo.problem.line_search._natural_test`).
              The first one throttles the overshoot of penalty contact and
              elastic-plastic transitions, the second one lets the
              legitimate large steps of soft modes under force control
              through. ``method`` is ignored. Factorization reuse
              (:meth:`set_reuse_factorization`) is enabled automatically
              with a direct solver so that each trial costs one
              back-substitution.
            * **'minimize'**: classical residual-descent line search using
              ``method`` below. Throttles the step against overshoot but
              may strangle Newton on soft modes under force control.
            * **'safeguard'**: pure validity filter. The full Newton step
              is accepted whenever the trial state is kinematically valid;
              geometric backtracking is applied only on a degenerated
              trial state (:math:`\det F \le 0`). This never throttles the
              legitimate large steps of soft modes (whose quadratic
              remainder inflates the residual norm as :math:`\alpha^2`
              while remaining excellent steps), which any residual-monotone
              rule would strangle -- the right choice for force control of
              soft (bending) modes. ``method`` is ignored.
        apply_to_bc : bool, default True
            If True, the built-in line search applies its kinematic-validity
            filter to a pending Dirichlet increment. An invalid advance is
            scaled, its unapplied part is carried by the next Newton correction,
            and convergence is accepted only after the complete prescribed
            increment has been applied. Residual and natural acceptance tests
            start after that remainder reaches zero. If False, the built-in line
            search applies the complete Dirichlet increment without a validity
            check. This option does not alter custom callbacks.
        name : str, optional
            A unique identifier for the line search. If not provided, it defaults
            to 'standard' for built-in methods, or the function's name for callables.

        Notes
        -----
        * **Implementation**: This method sets the `_step_size_callback` attribute
          of the problem instance. Parameters `ls_mode`, `ls_method`,
          `ls_max_iter` and `ls_apply_to_bc` are stored within the
          `self.nr_parameters` dictionary (they may also be set through
          :meth:`set_nr_criterion`).
        * **Objective Function**: For 'Armijo' and 'Quadratic' methods, the solver
          minimizes the squared L2-norm of the residual:

          .. math:: \phi(\alpha) = \frac{1}{2} \|R(u + \alpha dX)\|^2

        * **Work Criterion**: The 'Energy' method minimizes the directional
          derivative of the potential energy (the external work).
        * **Safeguards**: To prevent solver stagnation, interpolated values are
          clipped such that :math:`\alpha_{new} \in [0.1\alpha, 0.5\alpha]`.
          Invalid kinematic trials are reduced geometrically. If no valid
          trial exists down to the minimum step, the complete increment is
          rejected through the normal time-step reduction machinery. If the
          natural-mode acceptance tests are exhausted, its last valid trial
          is used and the Newton solver decides whether the increment converges.

        Example
        -------
        >>> # Default: natural monotonicity test
        >>> my_problem.add_line_search()
        >>> # Classical residual-descent line search
        >>> my_problem.add_line_search(mode="minimize", method="Quadratic")
        >>> # Validity safeguard only (soft-mode force control)
        >>> my_problem.add_line_search(mode="safeguard")
        >>> # Apply prescribed displacement in one full step
        >>> my_problem.add_line_search(apply_to_bc=False)
        >>> # Using a custom function
        >>> def my_ls(pb, dX): return 0.5
        >>> my_problem.add_line_search(method=my_ls)
        """
        if mode not in ["natural", "minimize", "safeguard"]:
            raise ValueError(
                "Line search mode should be 'natural', 'minimize' or 'safeguard'"
            )
        if callable(method):
            cb_name = name or getattr(method, "__name__", "custom_ls")
            self._ls_callbacks[cb_name] = method
        elif method not in ["Residual", "Energy", "Armijo", "Quadratic"]:
            raise ValueError(
                "Line search method should be either 'Residual', "
                "'Energy', 'Armijo' or 'Quadratic'"
            )
        else:
            # Remove any existing entry that uses the standard line_search function
            self._ls_callbacks = {
                k: v for k, v in self._ls_callbacks.items() if v != line_search
            }
            if mode != "natural":
                self._disable_line_search_factorization_reuse()

            self.nr_parameters["ls_mode"] = mode
            self.nr_parameters["ls_method"] = method
            self.nr_parameters["ls_apply_to_bc"] = bool(apply_to_bc)
            self._ls_callbacks[name or "standard"] = line_search

        self._update_step_size_callback()

    def remove_line_search(self, name=None):
        """Remove a line search algorithm by its name.

        If no name provided, remove all defined algorithm.
        """
        if name is None:
            self._ls_callbacks.clear()
        else:
            removed = self._ls_callbacks.pop(name, None)
            if removed is None:
                warnings.warn(
                    f"Line search '{name}' not found. No action taken.", UserWarning
                )

        if line_search not in self._ls_callbacks.values():
            # solve_time_increment reads ls_mode to decide whether to enable
            # factorization reuse: it must not outlive the line search
            self._disable_line_search_factorization_reuse()
            self.nr_parameters.pop("ls_mode", None)
            self.nr_parameters.pop("ls_apply_to_bc", None)
        self._update_step_size_callback()

    def _get_free_dof_residual(self):
        if not hasattr(self, "_MatCB"):
            dof_free = self._dof_free
            if np.isscalar(self.get_D()) and self.get_D() == 0:
                return self.get_B()[dof_free]
            else:
                return self.get_B()[dof_free] + self.get_D()[dof_free]
        else:
            if np.isscalar(self.get_D()) and self.get_D() == 0:
                return self._MatCB.T @ self.get_B()
            else:
                return self._MatCB.T @ (self.get_B() + self.get_D())

    def compute_nr_error(self):
        """Compute the error of the Newton-Raphson algorithm.

        For Force and Work error criterion, the problem must be updated
        (update method).
        """
        norm_type = self.nr_parameters["norm_type"]
        dof_free = self._dof_free
        if len(dof_free) == 0:
            return 0
        if self.nr_parameters["criterion"] == "Displacement":
            if self._err0 is None:  # assess err0 from current state
                # Normalize by the current increment (not the total
                # accumulated displacement) so the criterion does not
                # become progressively looser as loading proceeds.
                err0 = np.linalg.norm(self._dU, norm_type)
                if not np.array_equal(self._U, 0):
                    # to avoid numerical error related to high U values
                    # compared to dU
                    err0 += 0.01 * np.linalg.norm(self._U, norm_type)
                # err0 += 1e-8*np.max(self.mesh.bounding_box.size)
                if err0 == 0:
                    err0 = 1
                    return 1
            else:
                err0 = self._err0
            return np.linalg.norm(self.get_X()[dof_free], norm_type) / (
                err0 * self._alpha
            )
        elif self.nr_parameters["criterion"] == "Force":
            if self._err0 is None:
                # Normalize by external force
                err0 = np.linalg.norm(
                    self.get_ext_forces(include_mpc=False),
                    norm_type,
                )
                # err0 += 1e-8  # to avoid divizion by 0
                if err0 == 0:
                    err0 = 1
                    return 1
            else:
                err0 = self._err0
            return np.linalg.norm(self._get_free_dof_residual(), norm_type) / err0
        else:  # self.nr_parameters['criterion'] == 'Work': work criterion
            # initialize the value of self._err0
            if self._err0 is None:
                self._err0 = 1
                self._err0 = self.compute_nr_error()
                if self._err0 == 0:  # zero-work increment: keep a finite ref
                    self._err0 = 1
                return 1
            else:
                if np.isscalar(self.get_D()) and self.get_D() == 0:
                    return (
                        np.linalg.norm(
                            self.get_X()[dof_free] * self.get_B()[dof_free],
                            norm_type,
                        )
                        / self._err0
                    )
                else:
                    return (
                        np.linalg.norm(
                            self.get_X()[dof_free]
                            * (self.get_B()[dof_free] + self.get_D()[dof_free]),
                            norm_type,
                        )
                        / self._err0
                    )

    def force_elastic_matrix_next_iter(self):
        """
        Flags the solver to use the elastic stiffness matrix at the begining of the
        next new iteration.
        """
        self._force_elastic_next_iter = True

    def set_nr_criterion(self, criterion="Force", **kargs):
        """Define the convergence criterion of the newton raphson algorithm.

        For a problem pb, the newton raphson parameters can also be directly set in the
        pb.nr_parameters dict.

        Parameter:
            * criterion: str in ['Displacement', 'Force', 'Work'],
              default = "Displacement".
              Type of convergence test.

        Optional parameters that can be set as kargs:
            * 'err0': float or None, default = None.
              The reference error.
              If None (default), err0 is automatically computed.
            * 'tol': float, default is 5e-3.
              Error tolerance for convergence.
            * 'max_subiter': int, default = 16.
              Number of nr iterations before returning a convergence error.
            * 'dt_increase_niter': int or None, default = None.
              Number of nr iterations threshold that define an easy convergence.
              In problem allowing automatic convergence, if the Newton–Raphson
              loop converges in at most that many iterations, the time step is
              increased.
              If None, defaults to max_subiter//3.
            * 'norm_type': int or numpy.inf, default = 2.
              Define the norm used to test the criterion.
            * 'force_elastic_stiffness': bool, default = False.
              If True, always forces the use of the elastic stiffness matrix (KE)
              for all NR iterations (Quasi-Newton / Initial Stiffness method).
            * 'elastic_initial_guess': bool, default = False.
              If True, forces the use of the elastic matrix for the initial
              guess (first iteration) of a time increment.
              Ignored if force_elastic_stiffness is True.
            * 'adaptive_stiffness': bool, default = False.
              Enable adaptive stiffness algorithm with xi-blending between safe
              (often elastic) and tangent stiffness matrices to enhance convergence
              robustness. Only works if the weak form provides an elastic stiffness
              matrix at the beginning of the increment, when set_start is triggered
              (e.g. not for contact problems).
              Ignored if force_elastic_stiffness is True.
            * 'assume_cvg_at_max_subiter': bool, default = False.
              WARNING: DANGEROUS PARAMETER. If True, assumes convergence when
              max_subiter iterations are reached, regardless of tolerance criterion.
              Use only for special cases requiring forced convergence. Skips
              convergence tolerance check when iteration limit is reached.
            * 'check_early_divergence': bool, default = True.
              If True, aborts the time step early if convergence trends are poor.
              A step is considered diverging if:
              1. The error remains "unproductive" (new_error > 0.999 * previous_error)
                 for 4 consecutive Newton-Raphson iterations.
              2. The error spikes to more than 100 times the previous error.
        """
        if criterion not in ["Displacement", "Force", "Work"]:
            raise NameError(
                'criterion must be set to "Displacement", "Force" or "Work"'
            )
        self.nr_parameters["criterion"] = criterion

        allowed_keys = [
            "err0",
            "tol",
            "max_subiter",
            "dt_increase_niter",
            "norm_type",
            "adaptive_stiffness",
            "assume_cvg_at_max_subiter",
            "check_early_divergence",
            "force_elastic_stiffness",
            "elastic_initial_guess",
            "ls_mode",
            "ls_method",
            "ls_max_iter",
            "ls_apply_to_bc",
        ]

        for key in kargs:
            if key not in allowed_keys:
                raise NameError(
                    f"Newton Raphson parameters should be in {allowed_keys}"
                )

            self.nr_parameters[key] = kargs[key]

        if "ls_mode" in kargs and kargs["ls_mode"] != "natural":
            self._disable_line_search_factorization_reuse()

    def _enable_factorization_reuse(self):
        """Natural line search: make K^-1 R at trial states a back-substitution.

        No-op with an iterative (or petsc) solver, which is left untouched:
        the trials then re-solve the reduced system. NB: the reuse backend
        follows the default direct priority (pypardiso > python-mumps >
        petsc4py) whatever direct backend was forced with set_solver.
        """
        # only the default dispatch: a solver the user picked explicitly must
        # not be silently swapped for the reuse backend
        if self._factor_context is not None or self._solver_type != "direct":
            return
        try:
            self.set_reuse_factorization(True)
            self._line_search_factor_context = self._factor_context
        except RuntimeError:  # no reuse backend installed (scipy only)
            warnings.warn(
                "natural line search: no factorization-reuse backend "
                "(pypardiso, python-mumps or petsc4py); each line-search trial "
                "will re-solve the tangent system.",
                stacklevel=2,
            )

    def _disable_line_search_factorization_reuse(self):
        """Release factorization reuse only when natural line search owns it."""
        auto_context = getattr(self, "_line_search_factor_context", None)
        if auto_context is not None and self._factor_context is auto_context:
            self.set_reuse_factorization(False)
        self._line_search_factor_context = None

    def _elastic_reference_matrix(self, assemble=True):
        """Return the "safe" fallback matrix of adaptive_stiffness.

        Read from the CURRENT assembly: under ``nlgeom="UL"`` the reference
        one holds the matrix of the undeformed configuration (and, for an
        assembly sum, a state without the contact block), which would stay
        frozen for the whole run.
        """
        if assemble:
            self.assembly.current.assemble_global_mat("matrix")
        return self.assembly.current.get_global_matrix()

    def _xbc_is_applied(self):
        """True when no Dirichlet increment remains to be applied.

        Under pure force control (or once the line search has fully applied
        the scaled Dirichlet values), _Xbc is identically zero: convergence
        may then be declared even if the line search kept returning
        alpha < 1 (which leaves the _boundary_is_0 flag False forever).
        """
        xbc = self._Xbc
        if np.isscalar(xbc):
            return xbc == 0
        return not np.any(xbc)

    def elastic_prediction(self, keep_tangent=False):
        """Solve the elastic prediction of a new increment.

        keep_tangent: do not touch the tangent matrix. Used by the
        adaptive-stiffness restart, which has just installed the "safe"
        elastic matrix on purpose.
        """
        # update the boundary conditions with the time variation
        self._alpha = 1
        self.apply_boundary_conditions(self.t_fact, self.t_fact_old)

        # For the elastic prediction, it is more efficient to reuse the tangent
        # matrix from the last converged iteration of the previous time step.
        # A new tangent matrix is computed only for the very first increment,
        # where the matrix has its initial value of 0.
        if keep_tangent:
            pass
        elif np.isscalar(self.get_A()) and self.get_A() == 0:
            self._update_a()
        elif self.time_integrators and self.dtime != self._dtime_prev:
            # A transient tangent carries the 1/(beta dt^2) inertia term, so
            # the matrix of the previous increment is wrong by (dt_prev/dt)^2
            # once the time step changed -- x16 after the standard x0.25 cut,
            # which wrecks the prediction exactly when the solver is already
            # struggling. set_start/to_start have just re-assembled at the new
            # dt, so this only installs that matrix. NB: gated on an actual
            # integrator (the "compiled" flag is set even with none attached).
            # Other dt-dependent tangents on a plain NonLinear -- the legacy
            # ImplicitDynamic weak form, poromechanics, damping stabilization
            # -- keep the previous behavior; covering them would need a
            # `dt_dependent` property carried by the weak forms.
            self._update_a()

        self._update_d(
            start=True
        )  # not modified in principle if dt is not modified, except the very first iteration. May be optimized by testing the change of dt
        self.solve()

        self._boundary_is_0 = False
        # OGC per-vertex filter or CCD scalar line search
        if self._step_filter_callback is not None:
            dX = self.get_X()
            self._step_filter_callback(self, dX, is_elastic_prediction=True)
        elif self._step_size_callback is not None:
            dX = self.get_X()
            alpha = self._step_size_callback(self, dX)
            if alpha < 1.0:
                # Scale only free DOFs; preserve prescribed Dirichlet values
                # self.set_X(dX * alpha + self._Xbc * (1 - alpha))
                dX *= alpha
                self._alpha = alpha
        if self._alpha == 1:
            self._boundary_is_0 = True

        if self._boundary_is_0:
            # set the increment Dirichlet boundray conditions to 0 (i.e. will not change during the NR interations)
            self._Xbc *= 0
        else:
            self._Xbc *= 1 - alpha

        # update displacement increment
        self._dU += self.get_X()

    def solve_nr_increment(self):
        # solve and update total displacement. A and D should up to date
        self.solve()
        if self._step_filter_callback is not None:
            dX = self.get_X()
            self._step_filter_callback(self, dX, is_elastic_prediction=False)
        elif self._step_size_callback is not None:
            dX = self.get_X()
            alpha = self._step_size_callback(self, dX)
            self._alpha = alpha
            if alpha < 1.0:
                self.set_X(dX * alpha)
            if not self._boundary_is_0:
                if alpha < 1.0:
                    self._Xbc *= 1 - alpha
                else:
                    self._Xbc *= 0
                    self._boundary_is_0 = True

        self._dU += self.get_X()

    def solve_time_increment(self, max_subiter=None, tol_nr=None):
        if max_subiter is None:
            max_subiter = self.nr_parameters["max_subiter"]
        if tol_nr is None:
            tol_nr = self.nr_parameters["tol"]
        assume_cvg_at_max_subiter = self.nr_parameters.get(
            "assume_cvg_at_max_subiter", False
        )
        check_early_divergence = self.nr_parameters.get("check_early_divergence", True)
        force_elastic_stiffness = self.nr_parameters.get(
            "force_elastic_stiffness", False
        )
        if force_elastic_stiffness:
            adaptive_stiffness = False
            elastic_initial_guess = True
        else:
            adaptive_stiffness = self.nr_parameters.get("adaptive_stiffness", False)
            elastic_initial_guess = self.nr_parameters.get(
                "elastic_initial_guess", False
            )

        subiter = 1
        error = float("inf")
        self._t_fact_inc = self.t_fact
        if self.nr_parameters.get("ls_mode") == "natural":
            self._enable_factorization_reuse()  # before the first solve
        if elastic_initial_guess or self._force_elastic_next_iter:
            # udpate assembly to the initial siffness given by the constitutive law
            self.assembly.current.assemble_global_mat("matrix")
            # ... and hand it to the elastic prediction, which otherwise
            # keeps the tangent of the previous iteration (A != 0)
            self._update_a()
            if self._force_elastic_next_iter:
                self._force_elastic_next_iter = (
                    False  # next iteration will be recomputed.
                )

        try:
            self.elastic_prediction()
        except InvalidKinematicStateError:
            # No kinematically valid line-search trial exists, even for its
            # smallest step: use the normal failed-increment/time-step-cut path.
            self._t_fact_inc = None
            return 0, subiter, error

        if adaptive_stiffness or force_elastic_stiffness:
            # we take the assembled matrix computed after set_start and before
            # update. This should be the elastic or "safe" stiffness.
            # If not, adaptive_stiffness algorithm will not work as expected.
            KE = self._elastic_reference_matrix(assemble=not elastic_initial_guess)
            xi = 0.0
            xi_increased_this_step = False

        consecutive_decreases = 0
        consecutive_increases = 0
        while subiter < max_subiter:
            # update Stress and initial displacement and Update stiffness matrix
            try:
                self.update(compute="vector")  # out of balance force vector
            except InvalidKinematicStateError:
                # the current iterate inverts elements: failed increment ->
                # the caller cuts the time step and restores the state
                return 0, subiter, error
            self._update_d()  # required to compute the NR error

            # Check convergence
            prev_error = error
            error = self.compute_nr_error()
            if (
                error < tol_nr
                and subiter >= self._nr_min_subiter
                # NB: _boundary_is_0 True implies _Xbc was zeroed, so testing
                # the actual _Xbc content subsumes the flag
                and self._xbc_is_applied()
                # constraints such as MeanMotion publish the out-of-balance on
                # their own nonlinear equations here; gate convergence on it so
                # an unsatisfied constraint cannot report a converged increment.
                and getattr(self, "_bc_residual_norm", 0.0) < 10 * tol_nr
            ):
                self._t_fact_inc = None
                return 1, subiter, error
            elif check_early_divergence and (
                not np.isfinite(error) or error > 100 * prev_error
            ):
                # suspected ill-conditioned matrix
                return 0, subiter, error

            # Track convergence trend
            if (
                error > 0.999 * prev_error
            ):  # if decrease non significant, considered as increase
                consecutive_increases += 1
                consecutive_decreases = 0
                if consecutive_increases >= 4 and check_early_divergence:
                    return 0, subiter, error
            else:
                consecutive_increases = 0
                consecutive_decreases += 1

            if self.print_info > 1:
                print_str = "     Subiter {} - Time: {:.5f} - Err: {:.5f}".format(
                    subiter, self.time + self.dtime, error
                )
                if adaptive_stiffness:
                    print_str += " - xi: {:.4f}".format(xi)
                if self._step_size_callback:
                    print_str += " - alpha: {:.4f}".format(self._alpha)
                print(print_str)

            if adaptive_stiffness:
                if consecutive_increases == 0:
                    # rolling backup of the last iterate that IMPROVED the
                    # error: saving it once the error has already risen would
                    # store the bad iterate the rollback below is meant to undo
                    self._dU_old = self._dU.copy()
                if consecutive_increases >= 2 and xi < 1.0:
                    # Diverging - switch to elastic stiffness
                    if self.print_info > 1:
                        print("     Diverging. Switching to elastic matrix (xi=1.0).")
                    xi = 1.0
                    consecutive_increases = 0
                    xi_increased_this_step = True
                    if subiter == 3:
                        # restart the iteration with the elastic stiffness
                        self.to_start()
                        self._t_fact_inc = self.t_fact  # re-freeze (cleared above)
                        subiter = 1
                        error = float("inf")
                        self.set_A(KE)
                        try:
                            self.elastic_prediction(keep_tangent=True)
                        except InvalidKinematicStateError:
                            self._t_fact_inc = None
                            return 0, subiter, error
                        continue
                    else:
                        # redo last iteration (copy: _dU is updated in place)
                        self._dU = self._dU_old.copy()
                        subiter = max(subiter - 2, 1)
                        continue
                elif consecutive_decreases > 0:
                    # Improving - try to reduce xi
                    if xi == 1.0:
                        threshold = 2 if xi_increased_this_step else 3
                        if consecutive_decreases >= threshold:
                            xi = 0.25
                            consecutive_decreases = 0
                    elif xi > 0:
                        xi /= 4.0
                        if xi < 0.0156:
                            xi = 0.0

            # --------------- Solve --------------------------------------------------------
            subiter += 1  # start a new nr iteration
            self.update(
                compute="matrix", updateWeakForm=False
            )  # assemble the tangeant matrix

            if adaptive_stiffness:
                KT = self.__assembly.current.get_global_matrix()
                A = xi * KE + (1 - xi) * KT
            elif force_elastic_stiffness:
                A = KE
            else:
                A = self.__assembly.current.get_global_matrix()

            self.set_A(A)

            try:
                self.solve_nr_increment()
            except InvalidKinematicStateError:
                self._t_fact_inc = None
                return 0, subiter, error

        if assume_cvg_at_max_subiter:
            # the last correction was added to _dU without an update: bring
            # the state variables (hence the outputs) to that iterate, with
            # the load factor still frozen at the increment's target
            try:
                self.update(compute="vector")
                self._update_d()
            except InvalidKinematicStateError:
                self._t_fact_inc = None
                return 0, subiter, error
            self._t_fact_inc = None
            return 1, subiter, error
        self._t_fact_inc = None
        return 0, subiter, error

    def nlsolve(
        self,
        dt: float = 0.1,
        update_dt: bool = True,
        tmax: float | None = None,
        t0: float | None = None,
        dt_min: float = 1e-6,
        max_subiter: int | None = None,
        dt_increase_niter: int | None = None,
        tol_nr: float | None = None,
        print_info: int | None = None,
        save_at_exact_time: bool | None = None,
        interval_output: int | float | None = None,
        callback: Callable[[Problem, ...], None] | None = None,
        exec_callback_at_each_iter: bool | None = None,
        dt_max: float | None = None,
    ) -> None:
        """Solve the non linear problem using the newton-raphson algorithm.

        Parameters
        ----------
        dt: float, default=0.1
            Initial time increment
        update_dt: bool, default = True
            If True, the time increment may be modified during resolution:
            * decrease if the solver has not converged
            * increase if the solver has converged quickly (see
              ``dt_increase_niter``).
        tmax: float, optional.
            Time at the end of the time step.
            If omitted, the attribute tmax is considered (default = 1.)
            else, the attribute tmax is modified.
        t0: float, optional.
            Time at the start of the time step.
            If omitted, the attribute t0 is considered (default = 0.)
            else, the attribute t0 is modified.
        dt_min: float, default = 1e-6
            Minimal time increment
        dt_max: float, optional
            Maximum time increment. The initial ``dt`` and subsequent
            adaptive increases are capped to this value. If omitted, the time
            increment has no upper bound other than output and end times.
        max_subiter: int, optional
            Maximal number of newton raphson iteration allowed for each time
            increment, after the initial linear guess.
            If omitted, the 'max_subiter' field in the nr_parameters
            attribute (ie nr_parameters['max_subiter']) is considered
            (default = 16).
        dt_increase_niter: int, optional
            When ``update_dt`` is ``True``, the time increment is multiplied
            by 1.25 if the Newton–Raphson loop converges in at most
            ``dt_increase_niter`` iterations. If omitted, the
            'dt_increase_niter' field in the nr_parameters attribute
            (ie nr_parameters['dt_increase_niter']) is considered
            (default = ``max_subiter // 3``).
            For contact problems where NR typically needs several iterations,
            a higher value (e.g. ``max_subiter // 2``) allows dt to recover
            after an earlier reduction.
        tol_nr: float, optional
            Tolerance of the newton-raphson algorithm.
            If omitted, the 'tol' field in the nr_parameters attribute
            (ie nr_parameters['tol']) is considered (default = 5e-3).
        print_info : int, optional
            Level of information printed to console.
            If 0, nothing is printed
            If 1, iterations info are printed
            If 2, iterations and newton-raphson sub iterations info are printed.
            If omitted, the print_info attribute is considered (default = 1).
        save_at_exact_time: bool, optional
            If True, the time increment is modified to stop at times defined by
            interval_output and allow to save results. If omitted, the
            save_at_exact_time attribute is considered (default = True).
            The given value is stored in the save_at_exact_time attribute.
        interval_output: int|float, optional
            Time step for output if save_at_exact_time is True (default) else
            number of iter increments between 2 output. If
            interval_output == -1, the results is saved at each initial
            time_step intervals or each increment depending on the
            save_at_exact_time value. If omitted, the interval_output attribute
            is considred (default -1)
        callback: function, optional
            The callback function is executed automatically during the non
            linear resolution. By default, the callback function is executed
            when output is requested (defined by the interval_output argument).
            If exec_callback_at_each_iter is True, the callback function is
            excuted at each time iteration.
        exec_callback_at_each_iter, bool, default = False
            If True, the callback function is executed after each time
            iteration.
        """
        # parameters
        if tmax is not None:
            self.tmax = tmax
        if t0 is not None:
            self.t0 = t0  # time at the start of the time step
        if dt_max is not None:
            if dt_max <= 0:
                raise ValueError("dt_max should be strictly positive.")
            if dt_max < dt_min:
                raise ValueError("dt_max should be greater than or equal to dt_min.")
            dt = min(dt, dt_max)
        if max_subiter is None:
            max_subiter = self.nr_parameters["max_subiter"]
        if dt_increase_niter is None:
            dt_increase_niter = self.nr_parameters["dt_increase_niter"]
            if dt_increase_niter is None:
                dt_increase_niter = max_subiter // 3
        if tol_nr is None:
            tol_nr = self.nr_parameters["tol"]
        if print_info is not None:
            self.print_info = print_info
        if save_at_exact_time is not None:
            self.save_at_exact_time = save_at_exact_time
        if exec_callback_at_each_iter is not None:
            self.exec_callback_at_each_iter = exec_callback_at_each_iter
        if interval_output is None:
            interval_output = self.interval_output  # time step for output if save_at_exact_time == 'True' (default) or  number of iter increments between 2 output

        if interval_output == -1:
            if self.save_at_exact_time:
                interval_output = dt
            else:
                interval_output = 1

        if self.save_at_exact_time:
            next_time = self.t0 + interval_output
        else:
            next_time = self.tmax  # next_time is the next exact time where the algorithm have to stop for output purpose

        self.init_bc_start_value()

        self.time = self.t0  # time at the begining of the iteration

        if not self._initialized:
            self.initialize()
        elif not self._time_integrators_compiled and self.time_integrators:
            # Integrators attached (or the assembly changed) after the first
            # solved stage: initialize() will not run again, so the transient
            # weakforms would never be compiled and the stage would silently
            # run static. Fail loudly instead.
            raise RuntimeError(
                "Time integrators were attached or modified after the first "
                "solve, but they are only compiled when the problem "
                "initializes. Create a new problem for the transient stage "
                "(the assembly can be reused)."
            )

        restart = False  # bool to know if the iteration is another attempt

        while self.time < self.tmax - self.err_num:
            save_results = (self.time == next_time) or (
                self.save_at_exact_time == False and self.__iter % interval_output == 0
            )

            # update next_time
            if self.time == next_time:
                # self.save_at_exact_time should be True
                next_time = next_time + interval_output
                if next_time > self.tmax - self.err_num:
                    next_time = self.tmax

            # keep the time step of the increment that has just been completed:
            # set_start finalizes state (e.g. Newmark velocity/acceleration)
            # over that increment, while self.dtime below is the NEXT step.
            self._dtime_prev = self.dtime

            if (
                self.time + dt > next_time - self.err_num
            ):  # if dt is too high, it is reduced to reach next_time
                self.dtime = next_time - self.time
            else:
                self.dtime = dt

            if restart:
                # reset internal variables, update Stress, initial displacement and assemble global matrix at previous time
                self.to_start()
                restart = False
            else:
                self.set_start(save_results, callback)

            # self.solve_time_increment = Newton Raphson loop
            convergence, nb_nr_iter, error = self.solve_time_increment(
                max_subiter, tol_nr
            )

            if convergence:
                self.time = self.time + self.dtime  # update time value
                self.__iter += 1

                if self.print_info > 0:
                    print(
                        "Iter {} - Time: {:.5f} - dt {:.5f} - NR iter: {} - Err: {:.5f}".format(
                            self.__iter, self.time, dt, nb_nr_iter, error
                        )
                    )

                # Check if dt can be increased
                if update_dt and nb_nr_iter <= dt_increase_niter and dt == self.dtime:
                    dt *= 1.25
                    if dt_max is not None:
                        dt = min(dt, dt_max)

            else:
                if self.print_info > 0:
                    print(
                        "Convergence Failed - dt {:.5f} - NR iter: {} - Err: {:.5f}".format(
                            dt, nb_nr_iter, error
                        )
                    )
                if update_dt:
                    dt *= 0.25

                    if dt < dt_min:
                        # roll back to the last converged increment so accessors
                        # return a clean state after the abort
                        self.to_start()
                        raise RuntimeError(
                            "Current time step is inferior to the specified minimal time step (dt_min)"
                        )

                    # Roll back the failed increment (to_start at the top of the loop)
                    restart = True
                    continue
                else:
                    self.to_start()
                    raise RuntimeError(
                        "Newton Raphson iteration has not converged - Reduce the time step or use update_dt = True"
                    )

        # the last increment is finalized with its own dt (the loop-top value
        # is that of the previous one, or of a failed attempt)
        self._dtime_prev = self.dtime
        self.set_start(True, callback)

    @property
    def assembly(self):
        return self.__assembly

    @property
    def t_fact(self):
        """Adimensional time used for boundary conditions.

        Frozen during ``solve_time_increment`` so that modifications to
        ``self.dtime`` (e.g. by a CCD step-size callback) do not alter
        the target time factor mid-increment.
        """
        if self._t_fact_inc is not None:
            return self._t_fact_inc
        return (self.time + self.dtime - self.t0) / (self.tmax - self.t0)

    @property
    def t_fact_old(self):
        """Previous adimensional time for boundary conditions."""
        return (self.time - self.t0) / (self.tmax - self.t0)


def NonLinearNewmark(
    assembly,
    beta=0.25,
    gamma=0.5,
    nlgeom=False,
    first_order_integrator=None,
    name="MainProblem",
):
    """Create a nonlinear Newmark problem with default time integrators.

    This is a convenience factory around :class:`NonLinear`. It attaches a
    Newmark integrator for second-order evolutions and a Backward-Euler
    integrator for first-order evolutions.

    Parameters
    ----------
    assembly : Assembly-like object or str
        Assembly used by the nonlinear problem, or its registered name.
    beta : float, default=0.25
        Newmark acceleration parameter.
    gamma : float, default=0.5
        Newmark velocity parameter.
    nlgeom : bool or str, default=False
        Geometric-nonlinearity option forwarded to :class:`NonLinear`.
    first_order_integrator : fedoo.time.BackwardEuler, optional
        Integrator for first-order evolution terms. A default Backward-Euler
        integrator is created when omitted.
    name : str, default="MainProblem"
        Name of the problem.

    .. note::
        The signature changed: the mass is now derived from the material
        density (``material.set_density(rho)``) or ``weakform.set_inertia(...)``,
        so a separate mass assembly is no longer passed. The legacy
        ``NonLinearNewmark(stiffness, mass, beta, gamma)`` form is rejected with
        an explicit error.
    """
    import numbers

    from fedoo import time

    if not isinstance(beta, numbers.Number) or not isinstance(gamma, numbers.Number):
        raise TypeError(
            "NonLinearNewmark(assembly, beta, gamma) no longer takes a separate "
            "mass assembly: the mass is derived from the material density "
            "(material.set_density(rho)) or from weakform.set_inertia(...). "
            "Replace NonLinearNewmark(stiffness, mass, beta, gamma) with "
            "NonLinearNewmark(stiffness, beta, gamma). See fedoo.time for the "
            "new time-integration API."
        )

    pb = NonLinear(assembly, nlgeom=nlgeom, name=name)
    pb.set_time_integrator(time.SECOND_ORDER, time.Newmark(beta, gamma))
    if first_order_integrator is None:
        first_order_integrator = time.BackwardEuler()
    pb.set_time_integrator(time.FIRST_ORDER, first_order_integrator)
    return pb
