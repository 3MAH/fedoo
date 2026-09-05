import numpy as np

from fedoo.core.assembly import Assembly
from fedoo.core.matrix import as_global_csr
from fedoo.core.problem import Problem
from fedoo.core.time_evolution import SECOND_ORDER, normalize_time_evolution
from fedoo.time.common import (
    assemble_rayleigh_damping_matrix,
    build_storage_assembly,
    newmark_acceleration_velocity,
)
from fedoo.time.generalized_alpha import GeneralizedAlpha
from fedoo.util.deprecation import deprecated_alias


class Linear(Problem):
    """Class that defines linear problems.

    This simple class allows a linear problem to be built from an assembly
    object. The discretized problem is written as ``A * X = B + D``, where
    ``A`` is the matrix built by the assembly, ``X`` is the solution vector,
    ``B`` contains the Neumann boundary conditions, and ``D`` is the vector
    built by the assembly.

    Parameters
    ----------
    assembly : Assembly-like object or str
        Assembly used to construct the matrix ``A`` and vector ``D``, or the
        name of a registered assembly.
    name : str, default="MainProblem"
        Name of the problem.
    time_step : float, optional
        Constant time increment used when a second-order ``integrator`` is
        attached. Omit it for the original static behavior.
    integrator : fedoo.time.Newmark or fedoo.time.GeneralizedAlpha, optional
        Implicit second-order time integrator. The stiffness and consistent
        mass matrices are assembled once by :meth:`initialize` and reused at
        every constant-size increment. Only the right-hand side is rebuilt.
    """

    def __init__(
        self,
        assembly: Assembly,
        name: str = "MainProblem",
        time_step: float | None = None,
        integrator: GeneralizedAlpha | None = None,
    ):
        if isinstance(assembly, str):
            assembly = Assembly.get_all()[assembly]

        super().__init__(mesh=assembly.mesh, name=name)

        self.nlgeom = False
        assembly.register_global_dofs(self)
        assembly.initialize(self)
        self.time = self.dtime = 0
        # self.set_A(assembly.get_global_matrix())
        # self.set_D(assembly.get_global_vector())
        self.__assembly = assembly

        # Optional linear-transient state. Static Linear behavior remains the
        # default and continues through the original solve/update paths below.
        self.time_step = None if time_step is None else float(time_step)
        if self.time_step is not None and self.time_step <= 0:
            raise ValueError("time_step must be strictly positive.")
        self.time_integrators = {}
        self._dynamic_initialized = False
        self._dynamic_step_solved = False
        self._dynamic_stiffness = None
        self._dynamic_mass = None
        self._dynamic_damping = None
        self._dynamic_constant_force = None
        self._dynamic_mass_assembly = None
        self._dynamic_storage_data = None
        self._output_counter = 0
        self._dynamic_displacement = self._new_vect_dof()
        self._dynamic_velocity = self._new_vect_dof()
        self._dynamic_acceleration = self._new_vect_dof()
        if integrator is not None:
            self.set_time_integrator(SECOND_ORDER, integrator)

    @property
    def time_integrator(self):
        """Return the attached second-order integrator, or ``None``."""
        return self.time_integrators.get(SECOND_ORDER)

    @property
    def is_dynamic(self):
        """Whether this linear problem has a time integrator attached."""
        return self.time_integrator is not None

    def set_time_integrator(self, evolution, integrator):
        """Attach a linear implicit second-order time integrator.

        The static assembly is kept unchanged. Its storage and dissipation
        declarations are assembled separately when :meth:`initialize` starts
        the transient analysis.
        """
        evolution = normalize_time_evolution(evolution)
        if evolution != SECOND_ORDER:
            raise ValueError("Linear currently supports only SECOND_ORDER dynamics.")
        if self._dynamic_initialized:
            raise RuntimeError(
                "The time integrator cannot be changed after initialize()."
            )
        if integrator is None:
            self.time_integrators.pop(evolution, None)
            return None
        if not isinstance(integrator, GeneralizedAlpha):
            raise TypeError(
                "Linear dynamics currently requires fd.time.Newmark or "
                "fd.time.GeneralizedAlpha."
            )
        self.time_integrators[evolution] = integrator
        return integrator

    def initialize(self):
        """Initialize and cache the optional linear transient operators.

        The stiffness, consistent mass and optional Rayleigh damping matrices
        are built once. They are reused by subsequent constant-step solves;
        only the effective right-hand vector is rebuilt at each increment.
        """
        if not self.is_dynamic or self._dynamic_initialized:
            return
        if self.time_step is None:
            raise ValueError(
                "A positive time_step is required for a transient Linear problem."
            )

        self.set_X(self._dynamic_displacement.copy())
        self._dynamic_stiffness = as_global_csr(
            self.__assembly.get_global_matrix(), self.n_dof
        )
        force = self.__assembly.get_global_vector()
        if np.isscalar(force):
            force = self._new_vect_dof()
        else:
            force = np.asarray(force).ravel()
        self._dynamic_constant_force = force

        self._dynamic_storage_data = build_storage_assembly(
            self.__assembly, SECOND_ORDER
        )
        self._dynamic_mass_assembly = self._dynamic_storage_data.assembly
        if self._dynamic_mass_assembly is None:
            raise ValueError("No second-order storage was found for linear dynamics.")
        self._dynamic_mass_assembly.initialize(self)
        self._dynamic_mass_assembly.assemble_global_mat("matrix")
        self._dynamic_mass = as_global_csr(
            self._dynamic_mass_assembly.get_global_matrix(), self.n_dof
        )
        self._dynamic_damping = assemble_rayleigh_damping_matrix(
            self._dynamic_storage_data,
            self.n_dof,
        )
        self._dynamic_update_matrix()
        self._dynamic_update_rhs()
        self._dynamic_initialized = True

    def _dynamic_update_matrix(self):
        integrator = self.time_integrator
        dt = self.time_step
        a0 = 1.0 / (integrator.beta * dt**2)
        c0 = integrator.gamma / (integrator.beta * dt)
        self.set_A(
            (1.0 - integrator.alpha_f) * self._dynamic_stiffness
            + (1.0 - integrator.alpha_m) * a0 * self._dynamic_mass
            + (1.0 - integrator.alpha_f) * c0 * self._dynamic_damping
        )

    def _dynamic_update_rhs(self):
        integrator = self.time_integrator
        dt = self.time_step
        beta = integrator.beta
        gamma = integrator.gamma
        alpha_m = integrator.alpha_m
        alpha_f = integrator.alpha_f
        displacement = self._dynamic_displacement
        velocity = self._dynamic_velocity
        acceleration = self._dynamic_acceleration

        a0 = 1.0 / (beta * dt**2)
        a2 = 1.0 / (beta * dt)
        a3 = 0.5 / beta - 1.0
        c0 = gamma / (beta * dt)
        velocity_coef = (1.0 - alpha_f) * (1.0 - gamma / beta) + alpha_f
        acceleration_coef = (1.0 - alpha_f) * dt * (1.0 - gamma / (2.0 * beta))

        rhs = self._dynamic_constant_force.copy()
        rhs += np.asarray(
            self._dynamic_mass
            @ (
                (1.0 - alpha_m)
                * (a0 * displacement + a2 * velocity + a3 * acceleration)
                - alpha_m * acceleration
            )
        ).ravel()
        rhs += np.asarray(
            self._dynamic_damping
            @ (
                (1.0 - alpha_f) * c0 * displacement
                - velocity_coef * velocity
                - acceleration_coef * acceleration
            )
        ).ravel()
        if alpha_f != 0.0:
            rhs -= alpha_f * np.asarray(self._dynamic_stiffness @ displacement).ravel()
        self.set_D(rhs)

    def set_initial_displacement(self, name, value):
        """Set a transient initial displacement component."""
        self._set_vect_component(self._dynamic_displacement, name, value)
        self.set_X(self._dynamic_displacement.copy())

    def set_initial_velocity(self, name, value):
        """Set a transient initial velocity component."""
        self._set_vect_component(self._dynamic_velocity, name, value)

    def set_initial_acceleration(self, name, value):
        """Set a transient initial acceleration component."""
        self._set_vect_component(self._dynamic_acceleration, name, value)

    def get_velocity(self, name="all"):
        """Return the current transient velocity."""
        return self._get_vect_component(self._dynamic_velocity, name)

    def get_acceleration(self, name="all"):
        """Return the current transient acceleration."""
        return self._get_vect_component(self._dynamic_acceleration, name)

    def get_disp(self, name="all"):
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
        if name == "all":
            name = "Disp"
        return self.get_dof_solution(name)

    def get_rot(self, name="all"):
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
        if name == "all":
            name = "Rot"
        return self.get_dof_solution(name)

    def reset(self):
        self.__assembly.reset()

        if self.is_dynamic:
            # The transient effective (Newmark) matrix and the cached
            # stiffness/mass/damping operators are rebuilt by initialize() on
            # the next solve. Forcing the static stiffness here (and leaving
            # _dynamic_initialized set) would silently run that solve on the
            # static K with carried-over velocity/acceleration.
            self._dynamic_initialized = False
            return

        self.set_A(self.__assembly.get_global_matrix())  # tangent stiffness
        self.set_D(self.__assembly.get_global_vector())
        self.set_B(0)

    def update(self, dtime=1, compute="all", update_weakform=False):
        """
        Assemble the matrix including the following modification:
            - New initial Stress
            - New initial Displacement
            - Modification of the mesh
            - Change in constitutive law (internal variable)
        Update the problem with the new assembled global matrix and global vector
        """
        if self.is_dynamic:
            return self._dynamic_update(
                compute=compute, update_weakform=update_weakform
            )

        out_values = self.__assembly.update(self, compute)
        self._set_A_from_assembly(self.__assembly.get_global_matrix())
        self.set_D(self.__assembly.get_global_vector())
        return out_values

    def solve(self, **kargs):
        """Solve the linear problem and optionally update assembly fields.

        For a static problem, ``update_weakform=True`` (the default) updates
        quantities such as strain and stress after the solve without
        rebuilding the global matrix. Each static call solves the absolute
        equilibrium and replaces the previous solution. For a transient
        problem, ``solve`` only solves the prepared increment; call
        :meth:`update` to advance the kinematics and optionally update the
        weakform.
        """
        if self.is_dynamic:
            return self._dynamic_solve(**kargs)

        update_weakform = kargs.pop("update_weakform", True)
        if kargs:
            raise TypeError(f"Unexpected Linear.solve keyword: {next(iter(kargs))}")
        self._set_A_from_assembly(self.__assembly.get_global_matrix())
        self.set_D(self.__assembly.get_global_vector())
        self.apply_boundary_conditions()
        Problem.solve(self)

        if update_weakform:
            self.update(compute="none")

    def _dynamic_solve(self, **kargs):
        """Solve one prepared linear transient increment."""
        if kargs:
            raise TypeError(
                f"Unexpected transient Linear.solve keyword: {next(iter(kargs))}"
            )
        self.initialize()
        self._dynamic_update_rhs()
        Problem.solve(self)
        self._dynamic_step_solved = True
        return self.get_X()

    def _dynamic_update(self, compute="all", update_weakform=False):
        """Advance Newmark/generalized-alpha kinematics after ``solve``."""
        if not self._dynamic_step_solved:
            raise RuntimeError("solve() must be called before update().")
        new_displacement = np.asarray(self.get_X()).copy()
        delta_displacement = new_displacement - self._dynamic_displacement
        acceleration, velocity = newmark_acceleration_velocity(
            self.time_integrator.beta,
            self.time_integrator.gamma,
            self.time_step,
            delta_displacement,
            self._dynamic_velocity,
            self._dynamic_acceleration,
        )
        self._dynamic_displacement = new_displacement
        self._dynamic_velocity = np.asarray(velocity).ravel()
        self._dynamic_acceleration = np.asarray(acceleration).ravel()
        self.time += self.time_step
        self.dtime = self.time_step
        self.set_X(new_displacement.copy())
        if update_weakform:
            self.__assembly.update(self, compute)
        self._dynamic_step_solved = False
        self._dynamic_update_rhs()
        return self.get_X()

    def solve_time_increment(
        self,
        update_weakform=False,
        save_results=False,
        callback=None,
        apply_boundary_conditions=True,
        t_fact=1.0,
    ):
        """Solve and advance one complete linear transient increment.

        This convenience method applies boundary conditions, solves the
        effective system, advances the Newmark-family kinematics and
        optionally refreshes assembly-derived fields. For a user-managed loop,
        the same stages remain available separately as
        ``apply_boundary_conditions()``, ``solve()`` and ``update()``.
        """
        if not self.is_dynamic:
            raise RuntimeError("No time integrator is attached to this Linear problem.")
        self.initialize()
        self._dynamic_update_rhs()
        if apply_boundary_conditions:
            self.apply_boundary_conditions(t_fact=t_fact)
        Problem.solve(self)
        self._dynamic_step_solved = True
        self._dynamic_update(update_weakform=update_weakform)
        if save_results:
            if not update_weakform:
                self.__assembly.update(self, "vector")
            self.save_results(self._output_counter)
            self._output_counter += 1
        if callback is not None:
            callback(self)
        return self.get_X()

    def solve_history(
        self,
        tmax,
        dt=None,
        interval_output=None,
        save_results=False,
        update_weakform=False,
        callback=None,
    ):
        """Solve a complete cached linear transient history.

        This method is intended for genuinely linear problems: stiffness,
        mass and damping are cached by :meth:`initialize` and are not
        reassembled during the history. Set ``update_weakform=True`` when
        stresses, strains or other assembly state variables must be refreshed
        after every increment. Leaving it false is faster and is sufficient
        when only displacement, velocity, acceleration or energies are used;
        assembly fields are still refreshed at requested output times.

        A material, geometry, contact state or tangent operator that evolves
        with the solution is nonlinear and should be solved with
        :class:`NonLinear` instead. ``update_weakform=True`` here updates
        derived state but deliberately does not replace the cached linear
        operators.

        ``interval_output`` is a physical-time interval. Output times and
        ``tmax`` are reached exactly by temporarily shortening the nominal
        increment; the original time step is restored afterwards. Boundary
        conditions are reapplied at every increment using a load factor that
        progresses from zero to one over this call.
        """
        if not self.is_dynamic:
            raise RuntimeError("No time integrator is attached to this Linear problem.")
        if dt is not None:
            if dt <= 0:
                raise ValueError("dt must be strictly positive.")
            self.time_step = float(dt)
        if tmax < self.time:
            raise ValueError("tmax must not be lower than the current time.")
        self.initialize()

        nominal_dt = self.time_step
        t0 = self.time
        duration = tmax - t0
        tolerance = np.finfo(float).eps * max(1.0, abs(t0), abs(tmax))
        if interval_output is not None:
            if interval_output == -1:
                interval_output = nominal_dt
            elif interval_output <= 0:
                raise ValueError("interval_output must be positive or -1.")
            save_results = True
        elif save_results:
            interval_output = nominal_dt
        output_index = 1
        next_output = (
            min(t0 + float(interval_output), tmax)
            if save_results and duration > tolerance
            else None
        )

        try:
            while self.time < tmax - tolerance:
                step_end = min(self.time + nominal_dt, tmax)
                if next_output is not None and step_end >= next_output - tolerance:
                    step_end = next_output
                self.time_step = step_end - self.time
                self.dtime = self.time_step
                self._dynamic_update_matrix()
                save_increment = (
                    next_output is not None and abs(step_end - next_output) <= tolerance
                )
                self.solve_time_increment(
                    update_weakform=update_weakform,
                    save_results=save_increment,
                    callback=callback,
                    t_fact=(1.0 if duration == 0 else (step_end - t0) / duration),
                )
                if save_increment:
                    output_index += 1
                    candidate = t0 + output_index * float(interval_output)
                    next_output = tmax if candidate >= tmax - tolerance else candidate
        finally:
            self.time_step = nominal_dt
            self.dtime = nominal_dt
            self._dynamic_update_matrix()
            self._dynamic_update_rhs()
        return self

    def change_assembly(self, assembling, update=True):
        """
        Modify the assembly associated to the problem and update the problem (see Assembly.update for more information)
        """
        if isinstance(assembling, str):
            assembling = Assembly[assembling]

        self.__assembly = assembling
        if update:
            self.update()

    def get_elastic_energy(self):  # only work for classical FEM
        """
        returns : sum (0.5 * U.transposed * K * U)
        """

        if self.is_dynamic and self._dynamic_stiffness is not None:
            displacement = self.get_dof_solution("all")
            return 0.5 * np.dot(displacement, self._dynamic_stiffness @ displacement)

        return sum(
            0.5
            * self.get_dof_solution("all").transpose()
            * self.get_A()
            * self.get_dof_solution("all")
        )

    def get_nodal_elastic_energy(self):
        """
        returns : 0.5 * K * U . U
        """

        matrix = (
            self._dynamic_stiffness
            if self.is_dynamic and self._dynamic_stiffness is not None
            else self.get_A()
        )
        E = (
            0.5
            * self.get_dof_solution("all").transpose()
            * matrix
            * self.get_dof_solution("all")
        )

        E = np.reshape(E, (3, -1)).T

        return E

    GetElasticEnergy = deprecated_alias(get_elastic_energy, "GetElasticEnergy")
    GetNodalElasticEnergy = deprecated_alias(
        get_nodal_elastic_energy, "GetNodalElasticEnergy"
    )

    def get_kinetic_energy(self):
        """Return ``0.5 * velocity.T * M * velocity``."""
        if not self.is_dynamic or self._dynamic_mass is None:
            raise RuntimeError("Kinetic energy requires initialized dynamics.")
        return 0.5 * np.dot(
            self._dynamic_velocity, self._dynamic_mass @ self._dynamic_velocity
        )

    @property
    def assembly(self):
        return self.__assembly


# Convenience name for code that wants to make the integration scheme explicit.
# This is deliberately an alias, not a second implementation or a factory.
LinearNewmark = Linear
