"""The vectorized radial return of ElastoPlasticity solves every point independently.

The residuals are only evaluated at the points still bracketed or iterated: the
result must be, bit for bit, the one of the algorithm that evaluates every point
at every iteration, and the one of solving each point alone.
"""

import numpy as np
import pytest

import fedoo as fd
from fedoo.constitutivelaw.elasto_plasticity import _is_identity


def _law(kind):
    law = fd.constitutivelaw.ElastoPlasticity(200e3, 0.3, 300.0)
    if kind == "power":
        law.set_hardening_function("power", h=1000.0, beta=0.3)
    else:  # saturating hardening: Newton candidates often leave the bracket
        law.set_hardening_function(
            "user",
            hardening_function=lambda p: 500.0 * (1 - np.exp(-50 * np.asarray(p))),
            hardening_function_derivative=lambda p: 25000.0
            * np.exp(-50 * np.asarray(p)),
        )
    law.return_mapping_tolerance = 1e-9
    return law


def _plastic_increment_all_points(law, equivalent_trial_stress, plasticity_old):
    """Reference radial return: every point is evaluated at every iteration."""

    def residual(plastic_increment):
        return (
            equivalent_trial_stress
            - 3.0 * law.shear_modulus * plastic_increment
            - law.yield_stress
            - law.hardening_function(plasticity_old + plastic_increment)
        )

    lower_bound = np.zeros_like(equivalent_trial_stress)
    upper_bound = residual(lower_bound) / (3.0 * law.shear_modulus)
    for _ in range(law.max_return_mapping_iterations):
        points_to_expand = residual(upper_bound) > 0
        if not np.any(points_to_expand):
            break
        upper_bound[points_to_expand] *= 2.0

    plastic_increment = 0.5 * upper_bound
    converged = np.zeros_like(plastic_increment, dtype=bool)
    for _ in range(law.max_return_mapping_iterations):
        value = residual(plastic_increment)
        converged |= np.abs(value) <= law.return_mapping_tolerance
        if np.all(converged):
            return plastic_increment
        active = ~converged
        positive_residual = active & (value > 0)
        negative_residual = active & ~positive_residual
        lower_bound[positive_residual] = plastic_increment[positive_residual]
        upper_bound[negative_residual] = plastic_increment[negative_residual]
        slope = law.hardening_function_derivative(plasticity_old + plastic_increment)
        denominator = 3.0 * law.shear_modulus + slope
        candidate = plastic_increment + value / denominator
        invalid_candidate = (
            ~np.isfinite(candidate)
            | (denominator <= 0)
            | (candidate <= lower_bound)
            | (candidate >= upper_bound)
        )
        candidate[invalid_candidate] = 0.5 * (
            lower_bound[invalid_candidate] + upper_bound[invalid_candidate]
        )
        plastic_increment[active] = candidate[active]
    raise RuntimeError("reference radial return did not converge")


@pytest.mark.parametrize("kind", ["power", "user"])
def test_points_are_solved_independently(kind):
    law = _law(kind)
    rng = np.random.default_rng(3)
    n = 200
    plasticity_old = rng.uniform(0.0, 0.05, n)
    plasticity_old[:20] = 0.0
    # from slightly to far beyond the yield surface: several bracketing passes
    equivalent = (
        300.0 + law.hardening_function(plasticity_old) + 10 ** rng.uniform(-2, 5, n)
    )
    together = law._plastic_increment(equivalent, plasticity_old)
    alone = np.array(
        [
            law._plastic_increment(equivalent[i : i + 1], plasticity_old[i : i + 1])[0]
            for i in range(n)
        ]
    )
    assert together.tobytes() == alone.tobytes()
    reference = _plastic_increment_all_points(law, equivalent, plasticity_old)
    assert together.tobytes() == reference.tobytes()


def test_identity_rotations():
    n = 5
    assert _is_identity(np.eye(3))
    assert _is_identity(np.repeat(np.eye(3)[:, :, None], n, axis=2))
    rotations = np.repeat(np.eye(3)[:, :, None], n, axis=2)
    rotations[0, 1, 3] = 1e-17
    assert not _is_identity(rotations)
    assert not _is_identity(np.zeros((3, 3)))


def test_return_mapping_failure_is_reported():
    law = _law("user")
    law.max_return_mapping_iterations = 2
    with pytest.raises(RuntimeError, match="did not converge after 2 iterations"):
        law._plastic_increment(np.array([400.0, 900.0]), np.zeros(2))
