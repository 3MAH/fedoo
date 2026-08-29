"""Explicit structural-dynamics problem."""

import numpy as np
from scipy import sparse

from fedoo.core.assembly import Assembly
from fedoo.core.base import AssemblyBase
from fedoo.core.matrix import as_global_csr
from fedoo.core.problem import Problem
from fedoo.core.time_evolution import SECOND_ORDER, normalize_time_evolution
from fedoo.time.common import (
    RayleighDamping,
    assemble_rayleigh_damping_matrix,
    build_storage_assembly,
)
from fedoo.time.explicit import (
    CentralDifference,
    ExplicitDynamicState,
    ExplicitSecondOrderIntegrator,
)


class ExplicitDynamic(Problem):
    """Explicit structural dynamics with a user-controlled increment loop.

    The low-level workflow intentionally mirrors the other Fedoo problems::

        pb.initialize()
        pb.apply_boundary_conditions()
        while ...:
            pb.solve()
            pb.update()

    ``solve()`` only solves the effective system already prepared for the
    current increment. ``update()`` advances displacement, velocity and
    acceleration. Updating the weakform and constitutive law is optional; the
    default linear path uses the stiffness matrix cached by ``initialize()``.

    For nonlinear work, :meth:`solve_time_increment` manages the integrator's
    force-evaluation state, weakform update, solve, kinematic update and
    optional state commit. :meth:`evaluate` exposes the prediction/evaluation
    stage for advanced user loops. That stage mainly matters for explicit
    generalized-alpha schemes; centered difference evaluates forces at the
    current state. :meth:`solve_history` provides the same linear/nonlinear
    choice for a complete time history.

    Parameters
    ----------
    assembly : AssemblyBase or str
        Static structural assembly, or its registered name.
    time_step : float
        Constant time increment used by the manual loop.
    integrator : ExplicitSecondOrderIntegrator, optional
        Explicit time integrator. Defaults to :class:`CentralDifference`.
    mass_lumping : bool, default=True
        Row-sum lump finite-element storage. Assembly-level storage providers
        keep their declared matrix blocks and are never lumped implicitly.
    name : str, default="MainProblem"
        Problem name.

    Notes
    -----
    Finite-element stiffness and mass are cached after initialization. A mass
    refresh is therefore opt-in through ``update_mass=True``. Assembly-level
    providers are conservatively treated as configuration-dependent unless
    they define ``storage_matrix_is_constant = True``.

    ``update_weakform=False`` is appropriate only when the cached stiffness
    describes the internal force, for example linear elasticity with fixed
    geometry. Use ``update_weakform=True`` for nonlinear constitutive laws,
    changing geometry, contact, or other state-dependent contributions.
    """

    def __init__(
        self,
        assembly,
        time_step,
        integrator=None,
        mass_lumping=True,
        name="MainProblem",
    ):
        if isinstance(assembly, str):
            assembly = AssemblyBase.get_all()[assembly]
        if not isinstance(assembly, AssemblyBase):
            raise TypeError("assembly must be an Assembly-like object or its name.")
        if time_step <= 0:
            raise ValueError("time_step must be strictly positive.")

        super().__init__(
            A=0,
            B=0,
            D=0,
            mesh=assembly.mesh,
            name=name,
            space=assembly.space,
        )
        self.__assembly = assembly
        self.__assembly.register_global_dofs(self)
        self.time_step = float(time_step)
        self.dtime = self.time_step
        self.time = 0.0
        self.mass_lumping = bool(mass_lumping)
        self.time_integrators = {}

        self._initialized = False
        self._step_prepared = False
        self._step_solved = False
        self._predicted = None
        self._evaluation = None
        self._force_is_assembled = False
        self._assembled_internal_force = None
        self._mass_assembly = None
        self._storage_data = None
        self._mass = None
        self._mass_matrix = None
        self._fe_mass_matrix = None
        self._provider_mass_matrices = {}
        self._assembly_storage_providers = []
        self._stiffness_matrix = None
        self._constant_internal_force = None
        self._rayleigh_damping = None
        self._rayleigh_damping_matrix = None
        self._output_counter = 0

        self._state = ExplicitDynamicState(
            displacement=self._new_vect_dof(),
            velocity=self._new_vect_dof(),
            acceleration=self._new_vect_dof(),
        )
        self._state_start = self._copy_state(self._state)
        self._time_start = self.time
        self.set_X(self._state.displacement.copy())
        self.set_time_integrator(
            SECOND_ORDER, integrator if integrator is not None else CentralDifference()
        )

    @staticmethod
    def _copy_state(state):
        return ExplicitDynamicState(
            state.displacement.copy(),
            state.velocity.copy(),
            state.acceleration.copy(),
        )

    def _global_matrix(self):
        return as_global_csr(self.__assembly.current.get_global_matrix(), self.n_dof)

    @property
    def assembly(self):
        return self.__assembly

    @property
    def time_integrator(self):
        return self.time_integrators[SECOND_ORDER]

    def set_time_integrator(self, evolution, integrator):
        """Attach an explicit second-order time integrator."""
        evolution = normalize_time_evolution(evolution)
        if evolution != SECOND_ORDER:
            raise ValueError("ExplicitDynamic only supports SECOND_ORDER evolution.")
        if self._initialized:
            raise RuntimeError(
                "The time integrator cannot be changed after initialize()."
            )
        if not isinstance(integrator, ExplicitSecondOrderIntegrator):
            raise TypeError(
                "ExplicitDynamic requires an ExplicitSecondOrderIntegrator, "
                "for example fd.time.CentralDifference()."
            )
        self.time_integrators[evolution] = integrator
        return integrator

    def _iter_assembly_storage_providers(self, assembly):
        for leaf in assembly.iter_leaf():
            if not isinstance(leaf, Assembly) and (
                getattr(leaf, "time_evolution", None) == SECOND_ORDER
            ):
                yield leaf

    def _assemble_fe_mass(self):
        if self._mass_assembly is None:
            self._mass = np.zeros(self.n_dof)
            self._fe_mass_matrix = sparse.csr_matrix((self.n_dof, self.n_dof))
            return

        self._mass_assembly.assemble_global_mat("matrix")
        matrix = as_global_csr(
            self._mass_assembly.get_global_matrix(), self.n_dof, copy=False
        )

        if self.mass_lumping:
            self._mass = np.asarray(matrix.sum(axis=1)).ravel()
            self._fe_mass_matrix = sparse.diags(self._mass, format="csr")
        else:
            self._mass = None
            self._fe_mass_matrix = matrix

    def _current_mass_matrix(self, refresh_providers=False):
        mass_matrix = self._fe_mass_matrix
        for provider in self._assembly_storage_providers:
            is_constant = getattr(provider, "storage_matrix_is_constant", False)
            if (
                refresh_providers
                or not is_constant
                or provider not in self._provider_mass_matrices
            ):
                self._provider_mass_matrices[provider] = provider.get_storage_matrix(
                    self
                )
            local_mass = self._provider_mass_matrices[provider]
            indices = np.asarray(provider.time_dof_indices, dtype=int)
            if sparse.issparse(local_mass):
                local_mass = local_mass.tocoo()
                rows = indices[local_mass.row]
                cols = indices[local_mass.col]
                values = local_mass.data
            else:
                local_mass = np.asarray(local_mass, dtype=float)
                expected_shape = (len(indices), len(indices))
                if local_mass.shape != expected_shape:
                    raise ValueError(
                        f"Assembly provider {provider.name!r} returned storage "
                        f"with shape {local_mass.shape}; expected {expected_shape}."
                    )
                rows = np.repeat(indices, len(indices))
                cols = np.tile(indices, len(indices))
                values = local_mass.ravel()
            mass_matrix = mass_matrix + sparse.csr_matrix(
                (values, (rows, cols)), shape=(self.n_dof, self.n_dof)
            )
        return mass_matrix

    def _mass_action(self, vector):
        return np.asarray(self._mass_matrix @ vector).ravel()

    def _end_step_acceleration(self, velocity):
        """Return ``M^-1 f_int(u)`` at the current displacement on free DOFs.

        This closes the symplectic central-difference velocity update. The
        acceleration is defined by the internal (and constant external) force
        only; Rayleigh damping is applied explicitly in the drift, so it is
        intentionally excluded here. Storage providers may however carry a
        velocity-proportional term inside ``get_time_inertia_force`` (the same
        bias subtracted by ``_update_d``): it is removed at ``velocity``, the
        end-step velocity estimate used by the solve, so the balance
        ``M a + f_v(v) = f`` holds for the reported acceleration. Blocked DOFs
        are set by the driver from the prescribed motion.
        """
        if self._assembled_internal_force is not None:
            internal_force = self._assembled_internal_force
        else:
            internal_force = self._linear_internal_force(self._state.displacement)
        if self._assembly_storage_providers:
            internal_force = internal_force - self._assembly_inertia_bias(velocity)
        acceleration = self._new_vect_dof()
        free = self._dof_free
        if len(free) == 0:
            return acceleration
        if (
            self.mass_lumping
            and not self._assembly_storage_providers
            and self._mass is not None
        ):
            acceleration[free] = internal_force[free] / self._mass[free]
        else:
            reduced_mass = self._MatCB.T @ self._mass_matrix @ self._MatCB
            reduced_force = self._MatCB.T @ internal_force
            acceleration = np.asarray(
                self._MatCB @ self._solve(reduced_mass, reduced_force)
            ).ravel()
        return acceleration

    def _default_mass_refresh(self):
        return any(
            not getattr(provider, "storage_matrix_is_constant", False)
            for provider in self._assembly_storage_providers
        )

    def _update_a(self, update_mass=False, update_providers=False):
        """Refresh the effective mass operator used by ``Problem.solve``."""
        if update_mass:
            self._assemble_fe_mass()
            self._refresh_rayleigh_damping_matrix()
        if update_mass or update_providers:
            self._mass_matrix = self._current_mass_matrix(refresh_providers=True)
        factor = self.time_integrator.effective_mass_factor(self.time_step)
        if (
            self.mass_lumping
            and not self._assembly_storage_providers
            and self._mass is not None
        ):
            self.set_A(factor * self._mass)
        else:
            self.set_A(factor * self._mass_matrix)

    def _assembly_inertia_bias(self, velocity):
        force = self._new_vect_dof()
        for provider in self._assembly_storage_providers:
            inertia_force = getattr(provider, "get_time_inertia_force", None)
            if inertia_force is None:
                continue
            indices = np.asarray(provider.time_dof_indices, dtype=int)
            local_force = inertia_force(
                self,
                np.zeros(len(indices)),
                np.asarray(velocity)[indices],
            )
            force[indices] += np.asarray(local_force, dtype=float)
        return force

    def _refresh_rayleigh_damping_matrix(self):
        if self._storage_data is None:
            return
        self._rayleigh_damping_matrix = assemble_rayleigh_damping_matrix(
            self._storage_data,
            self.n_dof,
            mass_lumping=self.mass_lumping,
        )

    def _damping_force(self, velocity):
        force = self._new_vect_dof()
        if self._rayleigh_damping is not None:
            alpha, beta = self._rayleigh_damping
            force += alpha * self._mass_action(velocity)
            if beta != 0.0:
                force += beta * np.asarray(self._stiffness_matrix @ velocity).ravel()
        elif self._rayleigh_damping_matrix is not None:
            force += np.asarray(self._rayleigh_damping_matrix @ velocity).ravel()

        for provider in self._assembly_storage_providers:
            damping = getattr(provider, "dissipation", None)
            if damping is None:
                continue
            if not isinstance(damping, RayleighDamping):
                raise NotImplementedError(
                    "Assembly-level explicit dynamics currently supports only "
                    "RayleighDamping."
                )
            indices = np.asarray(provider.time_dof_indices, dtype=int)
            local_velocity = np.asarray(velocity)[indices]
            local_mass = provider.get_storage_matrix(self)
            local_force = (
                damping.alpha * np.asarray(local_mass @ local_velocity).ravel()
            )
            if damping.beta != 0.0:
                stiffness_provider = getattr(
                    provider, "get_time_stiffness_matrix", None
                )
                if stiffness_provider is not None:
                    local_stiffness = stiffness_provider(self)
                else:
                    matrix = provider.get_global_matrix()
                    local_stiffness = matrix[indices][:, indices]
                local_force += (
                    damping.beta * np.asarray(local_stiffness @ local_velocity).ravel()
                )
            force[indices] += local_force
        return force

    def _linear_internal_force(self, displacement):
        force = -np.asarray(self._stiffness_matrix @ displacement).ravel()
        return force + self._constant_internal_force

    def _update_d(self):
        """Build the effective right-hand vector for the prepared increment."""
        evaluation_displacement, evaluation_velocity = self._evaluation
        if self._force_is_assembled:
            internal_force = self._assembled_internal_force.copy()
        else:
            internal_force = self._linear_internal_force(evaluation_displacement)

        history = self.time_integrator.mass_history(
            self._state, self._predicted, self.time_step
        )
        effective_force = internal_force + self._mass_action(history)
        effective_force -= self._assembly_inertia_bias(evaluation_velocity)
        effective_force -= self._damping_force(evaluation_velocity)
        self.set_D(effective_force)

    def initialize(self):
        """Initialize constitutive data and cache stiffness and mass."""
        if self._initialized:
            return

        self.__assembly = self.time_integrator.compile_assembly(
            self.__assembly, SECOND_ORDER
        )
        self.__assembly.initialize(self)
        self._assembly_storage_providers = list(
            self._iter_assembly_storage_providers(self.__assembly)
        )

        self.set_X(self._state.displacement.copy())
        self.__assembly.update(self, "all")
        self._stiffness_matrix = self._global_matrix()
        initial_force = self.__assembly.current.get_global_vector()
        if np.isscalar(initial_force):
            initial_force = self._new_vect_dof()
        else:
            initial_force = np.asarray(initial_force).ravel()
        self._constant_internal_force = (
            initial_force
            + np.asarray(self._stiffness_matrix @ self._state.displacement).ravel()
        )
        self._assembled_internal_force = initial_force.copy()

        self._storage_data = build_storage_assembly(self.__assembly, SECOND_ORDER)
        self._mass_assembly = self._storage_data.assembly
        if self._mass_assembly is not None:
            self._mass_assembly.initialize(self)
        self._assemble_fe_mass()
        self._refresh_rayleigh_damping_matrix()
        if self._fe_mass_matrix.nnz == 0 and not self._assembly_storage_providers:
            raise ValueError(
                "No second-order storage was found. Set material density, "
                "attach inertia to the structural weakform, or provide an "
                "assembly-level storage matrix."
            )
        self._provider_mass_matrices = {}
        self._mass_matrix = self._current_mass_matrix(refresh_providers=True)
        self._update_a(update_mass=False)
        self.set_D(self._new_vect_dof())
        self._initialized = True
        self._state_start = self._copy_state(self._state)
        self._time_start = self.time

    def evaluate(self, update_weakform=False, compute="all"):
        """Prepare the integrator's force-evaluation state.

        Parameters
        ----------
        update_weakform : bool, default=False
            If true, update the weakform and constitutive law at the evaluation
            state. Otherwise the cached linear stiffness is used.
        compute : {"all", "matrix", "vector", "none"}, default="all"
            Assembly contribution requested when ``update_weakform`` is true.
        """
        if not self._initialized:
            self.initialize()
        self._predicted, self._evaluation = self.time_integrator.prepare_step(
            self._state, self.time_step
        )
        self.set_X(self._evaluation[0].copy())
        if update_weakform:
            self.__assembly.update(self, compute)
            if compute != "vector":
                self._stiffness_matrix = self._global_matrix()
                self._refresh_rayleigh_damping_matrix()
            if compute != "matrix":
                force = self.__assembly.current.get_global_vector()
                self._assembled_internal_force = (
                    self._new_vect_dof()
                    if np.isscalar(force)
                    else np.asarray(force).ravel().copy()
                )
            else:
                self._assembled_internal_force = None
            self._force_is_assembled = compute != "matrix"
        else:
            evaluation_is_current = np.array_equal(
                self._evaluation[0], self._state.displacement
            )
            self._force_is_assembled = (
                evaluation_is_current and self._assembled_internal_force is not None
            )
        return self._evaluation

    def prepare_time_increment(
        self,
        update_weakform=False,
        update_mass=None,
        compute="all",
    ):
        """Prepare ``A`` and ``D`` for one subsequent call to ``solve()``."""
        self.evaluate(update_weakform=update_weakform, compute=compute)
        if update_mass is None:
            update_providers = self._default_mass_refresh()
            update_mass = False
        else:
            update_providers = bool(update_mass)
        self._update_a(
            update_mass=bool(update_mass),
            update_providers=update_providers,
        )
        self._update_d()
        self._step_prepared = True
        self._step_solved = False

    def solve(self, **kwargs):
        """Solve the current effective system using :class:`Problem`.

        No weakform or constitutive-law update is performed here. Call
        :meth:`prepare_time_increment` explicitly for an advanced loop, or use
        :meth:`solve_time_increment` for the managed workflow.
        """
        if kwargs:
            raise TypeError(
                f"Unexpected ExplicitDynamic.solve keyword: {next(iter(kwargs))}"
            )
        if not self._step_prepared:
            self.prepare_time_increment(update_weakform=False)
        if np.isscalar(self._Xbc):
            self.apply_boundary_conditions()

        operator = self.get_A()
        if getattr(operator, "ndim", 0) == 1:
            self.set_X(self._Xbc.copy())
            if self._MFext is not None:
                self.set_A(sparse.diags(operator, format="csr"))
                try:
                    super().solve()
                finally:
                    self.set_A(operator)
            else:
                super().solve()
        else:
            super().solve()
        self._step_solved = True
        return self.get_X()

    def update(self, compute="all", update_weakform=False, update_mass=False):
        """Advance kinematics and optionally update the assembly.

        This method does not call :meth:`set_start` and does not save results.
        Those operations remain explicit so linear manual loops avoid the
        constitutive commit and reassembly cost.
        """
        if not self._step_solved:
            raise RuntimeError("solve() must be called before update().")
        previous_state = self._copy_state(self._state)
        self._state = self.time_integrator.state_from_displacement(
            self._state,
            self._predicted,
            np.asarray(self.get_X()).copy(),
            self.time_step,
        )
        self.set_X(self._state.displacement.copy())
        self.time += self.time_step

        if update_weakform:
            self.__assembly.update(self, compute)
            if compute != "vector":
                self._stiffness_matrix = self._global_matrix()
                self._refresh_rayleigh_damping_matrix()
            if compute != "matrix":
                force = self.__assembly.current.get_global_vector()
                self._assembled_internal_force = (
                    self._new_vect_dof()
                    if np.isscalar(force)
                    else np.asarray(force).ravel().copy()
                )
            else:
                self._assembled_internal_force = None
        else:
            self._assembled_internal_force = None
        if update_mass:
            self._assemble_fe_mass()
            self._mass_matrix = self._current_mass_matrix(refresh_providers=True)

        if self.time_integrator.needs_end_step_acceleration:
            # Close velocity Verlet: v_{n+1} = v_n + 0.5*dt*(a_n + a_{n+1}) with
            # a_{n+1} = M^-1 f(u_{n+1}) evaluated at the just-updated state. The
            # end-step acceleration also becomes a_n for the next drift, keeping
            # the mass-history predictor consistent with the current geometry.
            dt = self.time_step
            # Velocity estimate for velocity-proportional provider forces: the
            # one the solve used (alpha/predictor evaluation), still available
            # here since _evaluation is cleared below.
            if self._evaluation is not None:
                evaluation_velocity = self._evaluation[1]
            else:
                evaluation_velocity = previous_state.velocity
            end_acceleration = self._end_step_acceleration(evaluation_velocity)
            velocity = previous_state.velocity + 0.5 * dt * (
                previous_state.acceleration + end_acceleration
            )
            blocked = getattr(self, "_dof_blocked", None)
            if blocked is not None and len(blocked):
                displacement = self._state.displacement
                velocity[blocked] = (
                    displacement[blocked] - previous_state.displacement[blocked]
                ) / dt
                end_acceleration[blocked] = (
                    velocity[blocked] - previous_state.velocity[blocked]
                ) / dt
            self._state.velocity = velocity
            self._state.acceleration = end_acceleration

        self._step_prepared = False
        self._step_solved = False
        self._predicted = None
        self._evaluation = None
        self._force_is_assembled = False
        return self._state

    def set_start(self, save_results=False, callback=None):
        """Commit assembly history and optionally save the accepted state."""
        self.__assembly.set_start(self)
        self._state_start = self._copy_state(self._state)
        self._time_start = self.time
        if save_results:
            self.save_results(self._output_counter)
            self._output_counter += 1
        if callback is not None:
            callback(self)

    def to_start(self):
        """Restore the last state committed by :meth:`set_start`."""
        self._state = self._copy_state(self._state_start)
        self.time = self._time_start
        self.set_X(self._state.displacement.copy())
        self.__assembly.to_start(self)
        self._step_prepared = False
        self._step_solved = False

    def solve_time_increment(
        self,
        update_weakform=False,
        update_mass=None,
        set_start=False,
        save_results=False,
        callback=None,
        apply_boundary_conditions=True,
        t_fact=1.0,
    ):
        """Prepare, solve and update one complete explicit increment."""
        self.prepare_time_increment(
            update_weakform=update_weakform,
            update_mass=update_mass,
        )
        if apply_boundary_conditions:
            self.apply_boundary_conditions(t_fact=t_fact)
        self.solve()
        self.update(update_weakform=update_weakform)
        if set_start:
            self.set_start(save_results=save_results, callback=callback)
        else:
            if save_results:
                self.save_results(self._output_counter)
                self._output_counter += 1
            if callback is not None:
                callback(self)
        return self.get_X()

    def _refresh_output_state(self):
        """Update assembly-derived fields without rebuilding the stiffness."""
        self.__assembly.update(self, "vector")
        force = self.__assembly.current.get_global_vector()
        self._assembled_internal_force = (
            self._new_vect_dof()
            if np.isscalar(force)
            else np.asarray(force).ravel().copy()
        )

    def solve_history(
        self,
        tmax,
        dt=None,
        update_weakform=False,
        update_mass=None,
        save_results=False,
        interval_output=None,
        callback=None,
    ):
        """Solve a complete explicit-dynamic time history.

        The nominal time increment is constant. It is shortened temporarily
        when required to reach an exact output time or ``tmax``, then restored
        when this method returns. This method does not perform Newton
        iterations; ``update_weakform`` controls whether the explicitly
        evaluated forces and constitutive state are refreshed.

        Parameters
        ----------
        tmax : float
            Final physical time. It must not be lower than the current problem
            time, so the method can also continue an existing history.
        dt : float, optional
            Nominal time increment. If omitted, the ``time_step`` supplied to
            :class:`ExplicitDynamic` is used.
        update_weakform : bool, default=False
            Select the physical-model update strategy.

            If false, stiffness and the linear internal-force relation cached
            by :meth:`initialize` are reused. No constitutive history is
            committed between increments. This is the efficient choice for a
            genuinely linear problem with fixed geometry, fixed contact state
            and state-independent loads. At requested output times only, a
            vector assembly update is performed so fields such as stress and
            strain correspond to the saved displacement; stiffness and mass
            remain cached.

            If true, the assembly and constitutive law are updated at the time
            integrator's force-evaluation state and again at the accepted
            end-of-increment state. :meth:`set_start` then commits the accepted
            constitutive history. This is required for nonlinear materials,
            geometric nonlinearity, evolving contact, or any contribution
            that depends on the current solution. It is more expensive, but
            using false in those cases would evaluate incorrect forces even
            if the explicit algebraic solve itself succeeds.
        update_mass : bool or None, default=None
            If true, rebuild both finite-element mass and assembly-provider
            storage each increment. If false, reuse all cached mass terms. If
            none, keep finite-element mass cached while refreshing only
            assembly-level storage providers that do not declare
            ``storage_matrix_is_constant = True``.
        save_results : bool, default=False
            Save registered outputs. With no ``interval_output``, save after
            every nominal time interval. Supplying ``interval_output`` also
            enables saving, so this flag need not be set in that case.
        interval_output : float, optional
            Physical-time interval between saved results. Output times are
            reached exactly, and the final state at ``tmax`` is always saved.
            Use ``-1`` to select the nominal time increment. An iteration-count
            interval is intentionally not supported by this explicit solver.
        callback : callable, optional
            Function called as ``callback(problem)`` after every accepted time
            increment, independently of the output interval.

        Returns
        -------
        ExplicitDynamic
            This problem instance.

        Notes
        -----
        Boundary conditions are reapplied for every increment, which allows
        user code to modify loads, prescribed values, node sets or MPCs while
        a history is running.
        """
        if dt is not None:
            if dt <= 0:
                raise ValueError("dt must be strictly positive.")
            self.time_step = float(dt)
            self.dtime = self.time_step
            self._step_prepared = False
        if tmax < self.time:
            raise ValueError("tmax must not be lower than the current time.")
        if not self._initialized:
            self.initialize()

        t0 = self.time
        duration = tmax - t0
        nominal_time_step = self.time_step
        tolerance = np.finfo(float).eps * max(1.0, abs(t0), abs(tmax))

        if interval_output is not None:
            if interval_output == -1:
                interval_output = nominal_time_step
            elif interval_output <= 0:
                raise ValueError("interval_output must be strictly positive or -1.")
            save_results = True
        elif save_results:
            interval_output = nominal_time_step

        output_index = 1
        next_output_time = None
        if save_results and duration > tolerance:
            next_output_time = min(t0 + float(interval_output), tmax)

        try:
            while self.time < tmax - tolerance:
                step_end = min(self.time + nominal_time_step, tmax)
                if (
                    next_output_time is not None
                    and step_end >= next_output_time - tolerance
                ):
                    step_end = next_output_time

                self.time_step = step_end - self.time
                self.dtime = self.time_step
                self._step_prepared = False
                save_increment = (
                    next_output_time is not None
                    and abs(step_end - next_output_time) <= tolerance
                )
                t_fact = 1.0 if duration == 0 else (step_end - t0) / duration
                self.solve_time_increment(
                    update_weakform=update_weakform,
                    update_mass=update_mass,
                    set_start=update_weakform,
                    save_results=False,
                    callback=callback,
                    apply_boundary_conditions=True,
                    t_fact=t_fact,
                )

                if save_increment:
                    if not update_weakform:
                        self._refresh_output_state()
                    self.save_results(self._output_counter)
                    self._output_counter += 1
                    output_index += 1
                    next_regular_output = t0 + output_index * float(interval_output)
                    next_output_time = (
                        tmax
                        if next_regular_output >= tmax - tolerance
                        else next_regular_output
                    )
        finally:
            self.time_step = nominal_time_step
            self.dtime = nominal_time_step
            self._step_prepared = False
        return self

    def set_initial_displacement(self, name, value):
        self._set_vect_component(self._state.displacement, name, value)
        self.set_X(self._state.displacement.copy())

    def set_initial_velocity(self, name, value):
        self._set_vect_component(self._state.velocity, name, value)

    def set_initial_acceleration(self, name, value):
        self._set_vect_component(self._state.acceleration, name, value)

    def set_rayleigh_damping(self, alpha, beta):
        """Override weakform damping with ``C = alpha*M + beta*K``."""
        self._rayleigh_damping = (float(alpha), float(beta))

    def get_disp(self, name="Disp"):
        return self.get_dof_solution(name)

    def get_velocity(self, name="all"):
        return self._get_vect_component(self._state.velocity, name)

    def get_acceleration(self, name="all"):
        return self._get_vect_component(self._state.acceleration, name)

    def reset(self):
        self.__assembly.reset()
        if self._mass_assembly is not None:
            self._mass_assembly.reset()
        self._state = ExplicitDynamicState(
            displacement=self._new_vect_dof(),
            velocity=self._new_vect_dof(),
            acceleration=self._new_vect_dof(),
        )
        self._state_start = self._copy_state(self._state)
        self._initialized = False
        self._step_prepared = False
        self._step_solved = False
        self.time = 0.0
        self._time_start = 0.0
        self.set_X(self._state.displacement.copy())

    def get_elastic_energy(self):
        displacement = self.get_dof_solution("all")
        return 0.5 * np.dot(displacement, self._stiffness_matrix @ displacement)

    def get_kinetic_energy(self):
        return 0.5 * np.dot(
            self._state.velocity, self._mass_matrix @ self._state.velocity
        )

    def get_damping_power(self):
        force = self._damping_force(self._state.velocity)
        return np.dot(self._state.velocity, force)
