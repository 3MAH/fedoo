import copy

import numpy as np
from scipy import sparse

from fedoo.core.time_evolution import SECOND_ORDER
from fedoo.core.weakform import WeakFormBase, WeakFormSum
from fedoo.time.base import TimeIntegratorBase
from fedoo.time.common import RayleighDamping, newmark_acceleration_velocity
from fedoo.weakform.inertia import Inertia


def _newmark_state(term, assembly, dt):
    """Newmark end-of-step (acceleration, velocity) from the assembly state.

    Every generalized-alpha term carries ``beta``/``gamma`` and reads the same
    three state variables, so the recurrence call lives in one place.
    """
    return newmark_acceleration_velocity(
        term.beta,
        term.gamma,
        dt,
        assembly.sv["_DeltaDisp"],
        assembly.sv["Velocity"],
        assembly.sv["Acceleration"],
    )


class GeneralizedAlphaAssemblyAdapter:
    """Apply generalized-alpha to an assembly-level matrix provider.

    Assembly providers keep ownership of their static tangent and force. The
    adapter owns the temporal state and augments those contributions on the
    provider's declared time DOF indices.
    """

    def __init__(self, integrator):
        self.alpha_m = integrator.alpha_m
        self.alpha_f = integrator.alpha_f
        self.beta = integrator.beta
        self.gamma = integrator.gamma
        self.force_start = None
        self.current_force = None

    @staticmethod
    def _copy_state(state):
        return {
            key: value.copy() if hasattr(value, "copy") else value
            for key, value in state.items()
        }

    def initialize(self, assembly, pb):
        n_dof = len(assembly.time_dof_indices)
        assembly.sv["Velocity"] = np.zeros(n_dof)
        assembly.sv["Acceleration"] = np.zeros(n_dof)
        assembly.sv["_DeltaDisp"] = np.zeros(n_dof)

        initial_force = getattr(assembly, "get_time_initial_force", None)
        if initial_force is not None:
            force = np.asarray(initial_force(pb), dtype=float)
            self.force_start = force.copy()
            mass = np.asarray(assembly.get_storage_matrix(pb), dtype=float)
            try:
                assembly.sv["Acceleration"] = np.linalg.solve(mass, force)
            except np.linalg.LinAlgError:
                pass
        else:
            self.force_start = np.zeros(n_dof)

        assembly.sv_start = self._copy_state(assembly.sv)

    def update(self, assembly, pb):
        if np.isscalar(pb._dU) and pb._dU == 0:
            delta_disp = np.zeros(len(assembly.time_dof_indices))
        else:
            delta_disp = np.asarray(pb._dU)[assembly.time_dof_indices]
        assembly.sv["_DeltaDisp"] = delta_disp.copy()

    def _local_static_matrix(self, assembly):
        provider = getattr(assembly, "get_time_stiffness_matrix", None)
        if provider is not None:
            return np.asarray(provider(), dtype=float)

        idx = assembly.time_dof_indices
        matrix = assembly.global_matrix
        if sparse.issparse(matrix):
            return matrix[idx][:, idx].toarray()
        return np.asarray(matrix)[np.ix_(idx, idx)]

    def _damping_matrix(self, assembly, mass, stiffness):
        damping = getattr(assembly, "dissipation", None)
        if damping is None:
            return np.zeros_like(mass)
        if not isinstance(damping, RayleighDamping):
            raise NotImplementedError(
                "Assembly-level generalized-alpha currently supports only "
                "RayleighDamping."
            )
        return damping.alpha * mass + damping.beta * stiffness

    @staticmethod
    def _inertia_force(assembly, pb, mass, acceleration, velocity):
        provider = getattr(assembly, "get_time_inertia_force", None)
        if provider is not None:
            return np.asarray(provider(pb, acceleration, velocity), dtype=float)
        return mass @ acceleration

    @staticmethod
    def _inertia_tangent(
        assembly,
        pb,
        mass,
        acceleration,
        velocity,
        acceleration_factor,
        velocity_factor,
    ):
        provider = getattr(assembly, "get_time_inertia_tangent", None)
        if provider is not None:
            return np.asarray(
                provider(
                    pb,
                    acceleration,
                    velocity,
                    acceleration_factor,
                    velocity_factor,
                ),
                dtype=float,
            )
        return acceleration_factor * mass

    def integrate(self, assembly, pb, compute="all"):
        dt = pb.dtime
        if dt == 0:
            return

        idx = np.asarray(assembly.time_dof_indices, dtype=int)
        mass = np.asarray(assembly.get_storage_matrix(pb), dtype=float)
        stiffness = self._local_static_matrix(assembly)
        damping = self._damping_matrix(assembly, mass, stiffness)
        acc_np1, vel_np1 = newmark_acceleration_velocity(
            self.beta,
            self.gamma,
            dt,
            assembly.sv["_DeltaDisp"],
            assembly.sv["Velocity"],
            assembly.sv["Acceleration"],
        )
        acc_alpha = (1.0 - self.alpha_m) * acc_np1 + self.alpha_m * assembly.sv[
            "Acceleration"
        ]
        vel_alpha = (1.0 - self.alpha_f) * vel_np1 + self.alpha_f * assembly.sv[
            "Velocity"
        ]

        a0 = 1.0 / (self.beta * dt**2)
        c0 = self.gamma / (self.beta * dt)

        if compute != "vector":
            if self.alpha_f != 0.0:
                assembly.global_matrix = (1.0 - self.alpha_f) * assembly.global_matrix
            tangent = self._inertia_tangent(
                assembly,
                pb,
                mass,
                acc_alpha,
                vel_alpha,
                (1.0 - self.alpha_m) * a0,
                (1.0 - self.alpha_f) * c0,
            )
            tangent += (1.0 - self.alpha_f) * c0 * damping
            rows = np.repeat(idx, len(idx))
            cols = np.tile(idx, len(idx))
            dynamic_matrix = sparse.csr_matrix(
                (tangent.ravel(), (rows, cols)),
                shape=assembly.global_matrix.shape,
            )
            assembly.global_matrix = assembly.global_matrix + dynamic_matrix

        if compute != "matrix":
            static_force = np.asarray(assembly.global_vector)[idx].copy()
            self.current_force = static_force
            force_start = (
                self.force_start if self.force_start is not None else static_force
            )
            force_alpha = (
                1.0 - self.alpha_f
            ) * static_force + self.alpha_f * force_start
            inertia_force = self._inertia_force(
                assembly, pb, mass, acc_alpha, vel_alpha
            )
            assembly.global_vector[idx] = (
                force_alpha - inertia_force - damping @ vel_alpha
            )

    def set_start(self, assembly, pb):
        dof_solution = pb.get_dof_solution()
        if not (np.isscalar(dof_solution) and dof_solution == 0):
            dt = getattr(pb, "_dtime_prev", None) or pb.dtime
            if dt:
                acc, vel = newmark_acceleration_velocity(
                    self.beta,
                    self.gamma,
                    dt,
                    assembly.sv["_DeltaDisp"],
                    assembly.sv["Velocity"],
                    assembly.sv["Acceleration"],
                )
                assembly.sv["Acceleration"] = acc
                assembly.sv["Velocity"] = vel
                assembly.sv["_DeltaDisp"] = np.zeros_like(assembly.sv["_DeltaDisp"])
                if self.current_force is not None:
                    self.force_start = self.current_force.copy()
        assembly.sv_start = self._copy_state(assembly.sv)

    def to_start(self, assembly, pb):
        assembly.sv = self._copy_state(assembly.sv_start)


class GeneralizedAlpha(TimeIntegratorBase):
    """Generalized-alpha time integrator for second-order evolutions.

    The internal force, stiffness, and damping are evaluated at the generalized
    mid-points ``t_{n+1-alpha_f}`` and the inertia at ``t_{n+1-alpha_m}``. The
    classical endpoint Newmark scheme is the ``alpha_m = alpha_f = 0``
    specialization (see :class:`fedoo.time.Newmark`).

    .. warning::
        External loads (Neumann/Dirichlet) are applied by the problem at the
        step endpoint ``t_{n+1}``, not at ``t_{n+1-alpha_f}``. For
        ``alpha_f != 0`` combined with **time-varying** loads this leaves an
        ``O(alpha_f*dt)`` inconsistency in the forced response and degrades the
        designed spectral damping. With ``alpha_f = 0`` (Newmark) or with loads
        that are constant over the step the scheme is consistent. Sampling the
        load factor at the alpha point is a known follow-up.
    """

    evolution = SECOND_ORDER

    def __init__(self, alpha_m=0.0, alpha_f=0.0, beta=None, gamma=None):
        self.alpha_m = float(alpha_m)
        self.alpha_f = float(alpha_f)
        if self.alpha_m >= 1.0:
            raise ValueError("alpha_m must be lower than 1.")
        if self.alpha_f >= 1.0:
            raise ValueError("alpha_f must be lower than 1.")

        if gamma is None:
            gamma = 0.5 - self.alpha_m + self.alpha_f
        if beta is None:
            beta = 0.25 * (1.0 - self.alpha_m + self.alpha_f) ** 2
        self.beta = float(beta)
        self.gamma = float(gamma)
        if self.beta <= 0.0:
            raise ValueError("beta must be strictly positive.")
        if self.gamma <= 0.0:
            raise ValueError("gamma must be strictly positive.")

    def _compile_assembly_level_provider(self, assembly):
        if getattr(assembly, "time_evolution", None) != self.evolution:
            return assembly
        if not hasattr(assembly, "get_storage_matrix") or not hasattr(
            assembly, "time_dof_indices"
        ):
            raise NotImplementedError(
                "Second-order assembly providers must expose "
                "get_storage_matrix(pb) and time_dof_indices."
            )
        assembly._time_integrator = GeneralizedAlphaAssemblyAdapter(self)
        assembly._fedoo_time_integrated = True
        return assembly

    def _integrate_leaf(self, weakform):
        storage = self._resolve_storage(weakform)
        if storage is None:
            raise ValueError(
                "A second-order (dynamic) time integrator is attached, but no "
                f"mass/inertia could be resolved for weakform "
                f"{getattr(weakform, 'name', weakform)!r}. Give the material a "
                "density (material.set_density(rho)) before building the weakform, "
                "or attach inertia explicitly with "
                "weakform.set_inertia(density_or_weakform)."
            )

        dissipation = self._resolve_dissipation(weakform)
        if dissipation is not None and not isinstance(
            dissipation, (RayleighDamping, WeakFormBase)
        ):
            raise NotImplementedError(
                "Only Rayleigh descriptors and dissipative weakforms are compiled "
                "by GeneralizedAlpha for now. Assembly-level dissipative providers "
                "can be stored with set_dissipation(), but need an assembly adapter."
            )

        return self._wrap_static_weakform(weakform, storage, dissipation)

    def _resolve_storage(self, weakform):
        storage = weakform.get_storage()
        if isinstance(storage, WeakFormBase):
            return storage
        return Inertia(storage, space=weakform.space)

    def _resolve_dissipation(self, weakform):
        if getattr(weakform, "dissipation", None) is not None:
            return weakform.dissipation
        return weakform.get_dissipation()

    def _wrap_static_weakform(self, weakform, storage, dissipation):
        # Decorate a *copy* of the user's weakform with the generalized-alpha
        # stiffness behavior so the original object is never mutated (it stays
        # usable in other, e.g. static, problems) and its WeakForm registry
        # entry keeps pointing at the static form. The transient sums are built
        # unnamed for the same reason (an empty name is not registered).
        stiffness = copy.copy(weakform)
        parent = type(weakform)

        class GeneralizedAlphaStiffness(GeneralizedAlphaStiffnessTerm, parent):
            pass

        stiffness.__class__ = GeneralizedAlphaStiffness
        GeneralizedAlphaStiffnessTerm.__init__(
            stiffness, self.beta, self.gamma, self.alpha_f
        )

        inertia = GeneralizedAlphaStorageTerm(
            storage,
            self.beta,
            self.gamma,
            space=weakform.space,
            alpha_m=self.alpha_m,
            alpha_f=self.alpha_f,
        )
        inertia.assembly_options["assume_sym"] = True

        # Flag the individual terms, not only the returned sum: WeakFormSum
        # flattens nested sums, so a flag carried by the sum alone would be
        # lost when this transient sum is absorbed into an outer sum, and a
        # later compile pass would wrap the terms a second time (double mass).
        stiffness._fedoo_time_integrated = True
        inertia._fedoo_time_integrated = True

        terms = [stiffness, inertia]
        if dissipation is not None:
            if isinstance(dissipation, WeakFormBase):
                dissipation_term = GeneralizedAlphaDissipationTerm(
                    dissipation, self.beta, self.gamma, self.alpha_f
                )
                dissipation_term._fedoo_time_integrated = True
                terms.append(dissipation_term)
            else:
                stiffness.damping_coef = dissipation.beta
                inertia.damping_coef = dissipation.alpha

        return GeneralizedAlphaWeakFormSum(terms)


class GeneralizedAlphaStorageTerm(WeakFormBase):
    """Generalized-alpha discretization of a pure storage weakform."""

    def __init__(
        self,
        wf,
        beta,
        gamma,
        name="",
        space=None,
        alpha_m=0.0,
        alpha_f=0.0,
    ):
        super().__init__(name, space)
        self.beta = beta
        self.gamma = gamma
        self.alpha_m = alpha_m
        self.alpha_f = alpha_f
        if not isinstance(wf, WeakFormBase):
            wf = Inertia(wf)
        self.mass_wf = wf
        self.damping_coef = None

    def initialize(self, assembly, pb):
        assembly.sv_type["Velocity"] = "Node"
        assembly.sv_type["Acceleration"] = "Node"
        assembly.sv_type["_DeltaDisp"] = "Node"
        shape = (assembly.space.nvar, assembly.mesh.n_nodes)
        assembly.sv["Velocity"] = np.zeros(shape)
        assembly.sv["Acceleration"] = np.zeros(shape)
        assembly.sv["_DeltaDisp"] = np.zeros(shape)

    def update(self, assembly, pb):
        n_node_dof = assembly.space.nvar * assembly.mesh.n_nodes
        if np.isscalar(pb._dU) and pb._dU == 0:
            assembly.sv["_DeltaDisp"] = np.zeros_like(assembly.sv["_DeltaDisp"])
        else:
            assembly.sv["_DeltaDisp"] = pb._dU[:n_node_dof].reshape(
                assembly.space.nvar, assembly.mesh.n_nodes
            )

    def set_start(self, assembly, pb):
        if not (np.isscalar(pb.get_dof_solution()) and pb.get_dof_solution() == 0):
            # _DeltaDisp was integrated over the increment that just completed,
            # so the recurrence must use that dt — pb.dtime already holds the
            # NEXT increment's step when set_start is called from nlsolve.
            dt = getattr(pb, "_dtime_prev", None) or pb.dtime
            acc, vel = _newmark_state(self, assembly, dt)
            assembly.sv["Velocity"] = vel
            assembly.sv["Acceleration"] = acc
            assembly.sv["_DeltaDisp"] = np.zeros_like(assembly.sv["_DeltaDisp"])

    def get_weak_equation(self, assembly, pb):
        dt = pb.dtime
        if dt == 0:
            return 0

        a_np1, v_np1 = _newmark_state(self, assembly, dt)
        a_alpha = (1.0 - self.alpha_m) * a_np1 + self.alpha_m * assembly.sv[
            "Acceleration"
        ]
        v_alpha = (1.0 - self.alpha_f) * v_np1 + self.alpha_f * assembly.sv["Velocity"]

        a0 = 1.0 / (self.beta * dt**2)
        c0 = self.gamma / (self.beta * dt)
        alpha = self.damping_coef if self.damping_coef is not None else 0.0

        tangent_coeff = (1.0 - self.alpha_m) * a0
        tangent_coeff += alpha * (1.0 - self.alpha_f) * c0
        residual_val = a_alpha + alpha * v_alpha

        wf = self.mass_wf.get_weak_equation(assembly, pb)
        diff_op = tangent_coeff * wf
        if np.any(residual_val):
            diff_op += assembly.operator_apply(wf, residual_val.ravel())
        return diff_op


class GeneralizedAlphaStiffnessTerm(WeakFormBase):
    """Static stiffness contribution weighted by generalized-alpha."""

    def __init__(self, beta, gamma, alpha_f=0.0):
        self.beta = beta
        self.gamma = gamma
        self.alpha_f = alpha_f
        self.damping_coef = None

    def get_weak_equation(self, assembly, pb):
        wf = super().get_weak_equation(assembly, pb)
        if wf == 0:
            return 0

        dt = pb.dtime
        if dt == 0:
            return wf

        damped = self.damping_coef is not None and self.damping_coef != 0.0
        # Newmark (alpha_f = 0) with no damping is just the plain static term.
        if self.alpha_f == 0.0 and not damped:
            return wf

        mat, vec = wf.split_mat_vec()
        delta_u = assembly.sv["_DeltaDisp"]

        # Internal force/stiffness evaluated at the generalized mid-point
        # t_{n+1-alpha_f}. For alpha_f = 0 this collapses to the endpoint scheme.
        static_wf = (1.0 - self.alpha_f) * mat + vec
        if self.alpha_f != 0.0 and np.any(delta_u):
            static_wf = static_wf - self.alpha_f * assembly.operator_apply(
                mat, delta_u.ravel()
            )

        if not damped:
            return static_wf

        # Stiffness-proportional Rayleigh damping (beta*K) contribution.
        _, v_np1 = _newmark_state(self, assembly, dt)
        v_alpha = (1.0 - self.alpha_f) * v_np1 + self.alpha_f * assembly.sv["Velocity"]
        c0 = self.gamma / (self.beta * dt)

        damping_op = self.damping_coef * (1.0 - self.alpha_f) * c0 * mat
        if np.any(v_alpha):
            damping_op = damping_op + self.damping_coef * assembly.operator_apply(
                mat, v_alpha.ravel()
            )
        return static_wf + damping_op


class GeneralizedAlphaWeakFormSum(WeakFormSum):
    """WeakFormSum with Rayleigh damping accessors for second-order terms."""

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            stiffness = self.list_weakform[0]
        except (AttributeError, IndexError):
            raise AttributeError(name) from None
        return getattr(stiffness, name)

    def _rayleigh_terms(self):
        """Return the [stiffness, inertia] terms if they carry damping_coef.

        The compiled sum is flat (WeakFormSum flattens nested sums), so a
        custom dissipative weakform shows up as a GeneralizedAlphaDissipationTerm
        member; Rayleigh accessors are not available in that case.
        """
        terms = self.list_weakform
        if any(isinstance(t, GeneralizedAlphaDissipationTerm) for t in terms):
            return None
        if len(terms) >= 2 and all(hasattr(t, "damping_coef") for t in terms[:2]):
            return terms
        return None

    @property
    def rayleigh_damping(self):
        """list: Coefficients [alpha, beta] for Rayleigh damping."""
        terms = self._rayleigh_terms()
        if terms is None or terms[0].damping_coef is None:
            return None
        return [terms[i].damping_coef for i in [1, 0]]

    @rayleigh_damping.setter
    def rayleigh_damping(self, value):
        terms = self._rayleigh_terms()
        if terms is None:
            raise TypeError(
                "rayleigh_damping is only available on a stiffness+inertia "
                "generalized-alpha sum, not when a custom dissipative weakform "
                "is attached."
            )
        if value is None:
            value = [None, None]

        terms[0].damping_coef = value[1]
        terms[1].damping_coef = value[0]


class GeneralizedAlphaDissipationTerm(WeakFormBase):
    """Generalized-alpha discretization of a pure dissipative weakform."""

    def __init__(self, weakform, beta, gamma, alpha_f=0.0):
        super().__init__("", weakform.space)
        self.weakform = weakform
        self.beta = beta
        self.gamma = gamma
        self.alpha_f = alpha_f

    def initialize(self, assembly, pb):
        self.weakform.initialize(assembly, pb)

    def update(self, assembly, pb):
        self.weakform.update(assembly, pb)

    def update_2(self, assembly, pb):
        self.weakform.update_2(assembly, pb)

    def set_start(self, assembly, pb):
        self.weakform.set_start(assembly, pb)

    def to_start(self, assembly, pb):
        self.weakform.to_start(assembly, pb)

    def get_weak_equation(self, assembly, pb):
        dt = pb.dtime
        if dt == 0:
            return 0

        damping_wf = self.weakform.get_weak_equation(assembly, pb)
        if damping_wf == 0:
            return 0

        c0 = self.gamma / (self.beta * dt)
        _, vel_np1 = _newmark_state(self, assembly, dt)
        vel_alpha = (1.0 - self.alpha_f) * vel_np1 + self.alpha_f * assembly.sv[
            "Velocity"
        ]

        diff_op = (1.0 - self.alpha_f) * c0 * damping_wf
        if np.any(vel_alpha):
            diff_op += assembly.operator_apply(damping_wf, vel_alpha.ravel())
        return diff_op
