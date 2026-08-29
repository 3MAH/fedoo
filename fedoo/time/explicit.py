"""Explicit second-order time integrators."""

from dataclasses import dataclass

import numpy as np

from fedoo.core.time_evolution import SECOND_ORDER
from fedoo.time.base import TimeIntegratorBase


@dataclass
class ExplicitDynamicState:
    """Nodal state advanced by an explicit second-order integrator."""

    displacement: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray


class ExplicitAssemblyAdapter:
    """Mark an assembly-level provider for problem-driven explicit use.

    Assembly-level providers own their static force and tangent assembly. The
    :class:`~fedoo.problem.ExplicitDynamic` problem reads their storage matrix
    separately, so no temporal contribution is added here. Providers are
    considered configuration-dependent by default. They may declare
    ``storage_matrix_is_constant = True`` to let the problem cache their
    storage matrix after initialization.
    """

    def initialize(self, assembly, pb):
        pass

    def update(self, assembly, pb):
        pass

    def integrate(self, assembly, pb, compute="all"):
        pass

    def set_start(self, assembly, pb):
        pass

    def to_start(self, assembly, pb):
        pass


class ExplicitSecondOrderIntegrator(TimeIntegratorBase):
    """Base protocol for explicit structural-dynamics integrators.

    Explicit integrators do not compile inertia into an effective weakform.
    The problem keeps the static-force and lumped-mass operations separate,
    while the integrator supplies the predictor, alpha-point evaluation and
    corrector formulas.
    """

    evolution = SECOND_ORDER
    is_explicit = True

    @property
    def needs_end_step_acceleration(self):
        """True when the driver must re-evaluate acceleration at ``u_{n+1}``.

        Symplectic central difference (velocity Verlet) needs the acceleration
        at the end-of-step displacement to close the velocity update. Schemes
        that recover their end-step kinematics from the predictor alone return
        ``False``.
        """
        return False

    def compile_assembly(self, assembly, evolution=None):
        """Register assembly-level storage while keeping static weakforms."""
        return super().compile_assembly(assembly, evolution)

    def _integrate_leaf(self, weakform):
        return weakform

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
        assembly._time_integrator = ExplicitAssemblyAdapter()
        assembly._fedoo_time_integrated = True
        return assembly

    def predict(self, state, dt):
        """Return predicted displacement and velocity."""
        raise NotImplementedError

    def prepare_step(self, state, dt):
        """Return predictor and force-evaluation states for one increment."""
        predicted = self.predict(state, dt)
        evaluation = self.evaluation_state(state, *predicted)
        return predicted, evaluation

    def effective_mass_factor(self, dt):
        """Return the coefficient multiplying mass in the displacement solve."""
        raise NotImplementedError

    def mass_history(self, state, predicted, dt):
        """Return the known kinematic vector multiplied by mass on the RHS."""
        raise NotImplementedError

    def state_from_displacement(self, state, predicted, displacement, dt):
        """Return the end-step state from the solved displacement."""
        raise NotImplementedError

    def evaluation_state(self, state, displacement, velocity):
        """Return displacement and velocity used to evaluate forces."""
        return displacement, velocity

    def acceleration(self, unconstrained_acceleration, state):
        """Convert the mass-solve result into the end-step acceleration."""
        return unconstrained_acceleration

    def correct(self, state, displacement, velocity, acceleration, dt):
        """Return the corrected end-step state."""
        raise NotImplementedError

    def enforce_displacement(self, state, predicted, corrected, indices, values, dt):
        """Impose displacement values and update constrained kinematics."""
        displacement, velocity, acceleration = [value.copy() for value in corrected]
        if len(indices) == 0:
            return displacement, velocity, acceleration

        displacement[indices] = values
        velocity[indices] = (displacement[indices] - state.displacement[indices]) / dt
        acceleration[indices] = (velocity[indices] - state.velocity[indices]) / dt
        return displacement, velocity, acceleration


class ExplicitGeneralizedAlpha(ExplicitSecondOrderIntegrator):
    """One-corrector explicit generalized-alpha integrator.

    Internal forces are evaluated from the Newmark predictor at the
    generalized alpha point. The resulting acceleration is obtained with one
    diagonal mass solve and then used to correct displacement and velocity.

    Parameters
    ----------
    alpha_m, alpha_f : float, default=0
        Generalized-alpha weighting parameters.
    beta, gamma : float, optional
        Newmark corrector parameters. Their second-order defaults follow the
        generalized-alpha relations. ``beta=0`` is allowed for explicit
        central-difference integration.
    """

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
        if self.beta < 0.0:
            raise ValueError("beta must be non-negative for explicit integration.")
        if self.gamma <= 0.0:
            raise ValueError("gamma must be strictly positive.")

    @property
    def needs_end_step_acceleration(self):
        # beta == 0 is explicit central difference: close velocity Verlet with
        # the acceleration evaluated at the end-of-step displacement.
        return self.beta == 0.0

    def predict(self, state, dt):
        displacement = (
            state.displacement
            + dt * state.velocity
            + dt**2 * (0.5 - self.beta) * state.acceleration
        )
        velocity = state.velocity + dt * (1.0 - self.gamma) * state.acceleration
        return displacement, velocity

    def evaluation_state(self, state, displacement, velocity):
        displacement_alpha = (
            1.0 - self.alpha_f
        ) * displacement + self.alpha_f * state.displacement
        velocity_alpha = (1.0 - self.alpha_f) * velocity + self.alpha_f * state.velocity
        return displacement_alpha, velocity_alpha

    def acceleration(self, unconstrained_acceleration, state):
        return (unconstrained_acceleration - self.alpha_m * state.acceleration) / (
            1.0 - self.alpha_m
        )

    def correct(self, state, displacement, velocity, acceleration, dt):
        corrected_displacement = displacement + self.beta * dt**2 * acceleration
        corrected_velocity = velocity + self.gamma * dt * acceleration
        return corrected_displacement, corrected_velocity, acceleration

    def effective_mass_factor(self, dt):
        if self.beta == 0.0:
            return 1.0 / dt**2
        return (1.0 - self.alpha_m) / (self.beta * dt**2)

    def mass_history(self, state, predicted, dt):
        displacement, _ = predicted
        if self.beta == 0.0:
            base_displacement = (
                state.displacement
                + dt * state.velocity
                - 0.5 * dt**2 * state.acceleration
            )
            return base_displacement / dt**2
        factor = self.effective_mass_factor(dt)
        return factor * displacement - self.alpha_m * state.acceleration

    def state_from_displacement(self, state, predicted, displacement, dt):
        predicted_displacement, predicted_velocity = predicted
        if self.beta == 0.0:
            # Central difference (velocity Verlet). The end-step acceleration
            # a_{n+1} = M^-1 f(u_{n+1}) is evaluated at the new displacement by
            # the problem driver, which then closes the velocity update
            # v_{n+1} = v_n + 0.5*dt*(a_n + a_{n+1}). Only advance displacement
            # here; keep the current velocity/acceleration as provisional values
            # so the invariant ``state.acceleration == a_n = M^-1 f(u_n)`` holds
            # when the driver reads a_n for the trapezoidal velocity update.
            return ExplicitDynamicState(
                displacement.copy(),
                state.velocity.copy(),
                state.acceleration.copy(),
            )
        else:
            acceleration = (displacement - predicted_displacement) / (self.beta * dt**2)
            velocity = predicted_velocity + self.gamma * dt * acceleration
        return ExplicitDynamicState(displacement.copy(), velocity, acceleration)

    def enforce_displacement(self, state, predicted, corrected, indices, values, dt):
        if self.beta == 0.0:
            return super().enforce_displacement(
                state, predicted, corrected, indices, values, dt
            )

        displacement, velocity, acceleration = [value.copy() for value in corrected]
        if len(indices) == 0:
            return displacement, velocity, acceleration

        predicted_displacement, predicted_velocity = predicted
        displacement[indices] = values
        acceleration[indices] = (values - predicted_displacement[indices]) / (
            self.beta * dt**2
        )
        velocity[indices] = (
            predicted_velocity[indices] + self.gamma * dt * acceleration[indices]
        )
        return displacement, velocity, acceleration


class CentralDifference(ExplicitGeneralizedAlpha):
    """Explicit central-difference integrator.

    This is the explicit Newmark specialization ``beta=0`` and
    ``gamma=1/2`` without generalized-alpha filtering.
    """

    def __init__(self):
        super().__init__(alpha_m=0.0, alpha_f=0.0, beta=0.0, gamma=0.5)

    def prepare_step(self, state, dt):
        """Evaluate centered-difference forces at the current state."""
        predicted = self.predict(state, dt)
        evaluation = state.displacement.copy(), state.velocity.copy()
        return predicted, evaluation


# Alternative public name for the same explicit Newmark-family implementation.
ExplicitNewmark = ExplicitGeneralizedAlpha
