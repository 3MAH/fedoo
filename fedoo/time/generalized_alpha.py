import numpy as np

from fedoo.core.assembly import Assembly
from fedoo.core.assembly_sum import AssemblySum
from fedoo.core.base import AssemblyBase
from fedoo.core.time_evolution import SECOND_ORDER, normalize_time_evolution
from fedoo.core.weakform import WeakFormBase, WeakFormSum
from fedoo.time.common import RayleighDamping
from fedoo.weakform.inertia import Inertia


class GeneralizedAlpha:
    """Generalized-alpha time integrator for second-order evolutions."""

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

    def compile_assembly(self, assembly, evolution=None):
        """Compile compatible weakforms in an assembly tree in place."""
        evolution = normalize_time_evolution(evolution or self.evolution)
        if isinstance(assembly, AssemblySum):
            for child in assembly.list_assembly:
                self.compile_assembly(child, evolution)
            return assembly

        if isinstance(assembly, Assembly) and assembly.weakform is not None:
            assembly.weakform = self.compile_weakform(assembly.weakform, evolution)
        elif isinstance(assembly, AssemblyBase):
            self._compile_assembly_level_provider(assembly)
        return assembly

    def compile_weakform(self, weakform, evolution=None):
        """Return a time-integrated version of ``weakform`` when applicable."""
        evolution = normalize_time_evolution(evolution or self.evolution)
        if getattr(weakform, "_fedoo_time_integrated", False):
            return weakform

        if isinstance(weakform, WeakFormSum):
            compiled = [
                self.compile_weakform(wf, evolution) for wf in weakform.list_weakform
            ]
            if all(old is new for old, new in zip(weakform.list_weakform, compiled)):
                return weakform
            return WeakFormSum(compiled, weakform.name)

        if getattr(weakform, "time_evolution", None) != evolution:
            return weakform

        storage = self._resolve_storage(weakform)
        if storage is None:
            return weakform

        dissipation = getattr(weakform, "dissipation", None)
        if dissipation is not None and not isinstance(
            dissipation, (RayleighDamping, WeakFormBase)
        ):
            raise NotImplementedError(
                "Only Rayleigh descriptors and dissipative weakforms are compiled "
                "by GeneralizedAlpha for now. Assembly-level dissipative providers "
                "can be stored with set_dissipation(), but need an assembly adapter."
            )

        integrated = self._wrap_static_weakform(weakform, storage, dissipation)
        integrated._fedoo_time_integrated = True
        return integrated

    def _resolve_storage(self, weakform):
        storage = getattr(weakform, "storage", None)
        if storage is not None:
            if isinstance(storage, WeakFormBase):
                return storage
            return Inertia(storage, space=weakform.space)

        constitutivelaw = getattr(weakform, "constitutivelaw", None)
        density = getattr(constitutivelaw, "density", None)
        if density is not None:
            return Inertia(density, space=weakform.space)
        return None

    def _wrap_static_weakform(self, weakform, storage, dissipation):
        parent = type(weakform)

        class GeneralizedAlphaStiffness(GeneralizedAlphaStiffnessTerm, parent):
            pass

        weakform.__class__ = GeneralizedAlphaStiffness
        GeneralizedAlphaStiffnessTerm.__init__(
            weakform, self.beta, self.gamma, self.alpha_f
        )

        inertia = GeneralizedAlphaStorageTerm(
            storage,
            self.beta,
            self.gamma,
            "",
            getattr(weakform, "nlgeom", None),
            weakform.space,
            self.alpha_m,
            self.alpha_f,
        )
        inertia.assembly_options["assume_sym"] = True

        if dissipation is not None:
            if isinstance(dissipation, WeakFormBase):
                return GeneralizedAlphaWeakFormSum(
                    [
                        GeneralizedAlphaWeakFormSum([weakform, inertia], weakform.name),
                        GeneralizedAlphaDissipationTerm(
                            dissipation, self.beta, self.gamma, self.alpha_f
                        ),
                    ],
                    weakform.name,
                )
            weakform.damping_coef = dissipation.beta
            inertia.damping_coef = dissipation.alpha

        return GeneralizedAlphaWeakFormSum([weakform, inertia], weakform.name)

    def _compile_assembly_level_provider(self, assembly):
        has_provider = any(
            hasattr(assembly, attr) for attr in ("storage", "dissipation")
        )
        if has_provider:
            raise NotImplementedError(
                "Assembly-level time providers are part of the architecture but "
                "do not have a concrete GeneralizedAlpha adapter yet."
            )


class GeneralizedAlphaStorageTerm(WeakFormBase):
    """Generalized-alpha discretization of a pure storage weakform."""

    def __init__(
        self,
        wf,
        beta,
        gamma,
        name="",
        nlgeom=None,
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
        self.nlgeom = nlgeom
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
        dt = pb.dtime
        if not (np.isscalar(pb.get_dof_solution()) and pb.get_dof_solution() == 0):
            new_acceleration = (1 / (self.beta * dt**2)) * (
                assembly.sv["_DeltaDisp"] - dt * assembly.sv["Velocity"]
            ) - 1 / self.beta * (0.5 - self.beta) * assembly.sv["Acceleration"]

            assembly.sv["Velocity"] += dt * (
                (1 - self.gamma) * assembly.sv["Acceleration"]
                + self.gamma * new_acceleration
            )
            assembly.sv["Acceleration"] = new_acceleration
            assembly.sv["_DeltaDisp"] = np.zeros_like(assembly.sv["_DeltaDisp"])

    def get_weak_equation(self, assembly, pb):
        dt = pb.dtime
        if dt == 0:
            return 0

        a_np1, v_np1 = self._current_acceleration_velocity(assembly, pb)
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
        if not np.array_equal(residual_val, 0):
            diff_op += assembly.operator_apply(wf, residual_val.ravel())
        return diff_op

    def _current_acceleration_velocity(self, assembly, pb):
        dt = pb.dtime
        a0 = 1.0 / (self.beta * dt**2)
        a_n = assembly.sv["Acceleration"]
        v_n = assembly.sv["Velocity"]
        delta_disp = assembly.sv["_DeltaDisp"]

        acc = a0 * (delta_disp - dt * v_n) + (1.0 - 0.5 / self.beta) * a_n
        vel = v_n + dt * ((1.0 - self.gamma) * a_n + self.gamma * acc)
        return acc, vel


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
        if self.alpha_f == 0.0 and (
            self.damping_coef is None or self.damping_coef == 0.0
        ):
            return wf
        if self.alpha_f == 0.0:
            return self._newmark_damped_weak_equation(wf, assembly, pb)

        mat, vec = wf.split_mat_vec()
        delta_u = assembly.sv["_DeltaDisp"]

        static_wf = (1.0 - self.alpha_f) * mat + vec
        if self.alpha_f != 0.0 and not np.array_equal(delta_u, 0):
            static_wf -= self.alpha_f * assembly.operator_apply(mat, delta_u.ravel())

        if self.damping_coef is None or self.damping_coef == 0.0:
            return static_wf

        a_np1, v_np1 = self._current_acceleration_velocity(assembly, pb)
        v_alpha = (1.0 - self.alpha_f) * v_np1 + self.alpha_f * assembly.sv["Velocity"]
        c0 = self.gamma / (self.beta * dt)

        damping_mat = self.damping_coef * (1.0 - self.alpha_f) * c0 * mat
        if not np.array_equal(v_alpha, 0):
            return (
                static_wf
                + damping_mat
                + self.damping_coef * assembly.operator_apply(mat, v_alpha.ravel())
            )
        return static_wf + damping_mat

    def _newmark_damped_weak_equation(self, wf, assembly, pb):
        dt = pb.dtime
        a_n_node = assembly.sv["Acceleration"]
        v_n_node = assembly.sv["Velocity"]
        delta_u = assembly.sv["_DeltaDisp"]

        c0 = self.gamma / (self.beta * dt)
        a0 = 1.0 / (self.beta * dt**2)
        mat, vec = wf.split_mat_vec()
        scaled_mat = mat * (1.0 + self.damping_coef * c0)

        a_curr = a0 * (delta_u - dt * v_n_node) - (0.5 / self.beta - 1.0) * a_n_node
        v_curr = v_n_node + dt * ((1.0 - self.gamma) * a_n_node + self.gamma * a_curr)

        if not np.array_equal(v_curr, 0):
            damping_force_wf = self.damping_coef * assembly.operator_apply(
                mat, v_curr.ravel()
            )
            return scaled_mat + vec + damping_force_wf

        return scaled_mat + vec

    def _current_acceleration_velocity(self, assembly, pb):
        dt = pb.dtime
        a0 = 1.0 / (self.beta * dt**2)
        a_n = assembly.sv["Acceleration"]
        v_n = assembly.sv["Velocity"]
        delta_u = assembly.sv["_DeltaDisp"]
        acc = a0 * (delta_u - dt * v_n) + (1.0 - 0.5 / self.beta) * a_n
        vel = v_n + dt * ((1.0 - self.gamma) * a_n + self.gamma * acc)
        return acc, vel


class GeneralizedAlphaWeakFormSum(WeakFormSum):
    """WeakFormSum with Rayleigh damping accessors for second-order terms."""

    @property
    def rayleigh_damping(self):
        """list: Coefficients [alpha, beta] for Rayleigh damping."""
        if self.list_weakform[0].damping_coef is None:
            return None
        return [self.list_weakform[i].damping_coef for i in [1, 0]]

    @rayleigh_damping.setter
    def rayleigh_damping(self, value):
        if value is None:
            value = [None, None]

        self.list_weakform[0].damping_coef = value[1]
        self.list_weakform[1].damping_coef = value[0]


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

        a0 = 1.0 / (self.beta * dt**2)
        c0 = self.gamma / (self.beta * dt)
        delta_disp = assembly.sv["_DeltaDisp"]
        v_n = assembly.sv["Velocity"]
        a_n = assembly.sv["Acceleration"]

        acc_np1 = a0 * (delta_disp - dt * v_n) + (1.0 - 0.5 / self.beta) * a_n
        vel_np1 = v_n + dt * ((1.0 - self.gamma) * a_n + self.gamma * acc_np1)
        vel_alpha = (1.0 - self.alpha_f) * vel_np1 + self.alpha_f * v_n

        diff_op = (1.0 - self.alpha_f) * c0 * damping_wf
        if not np.array_equal(vel_alpha, 0):
            diff_op += assembly.operator_apply(damping_wf, vel_alpha.ravel())
        return diff_op
