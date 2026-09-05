"""Functions used for line search."""

import numpy as np

from fedoo.core.base import InvalidKinematicStateError

_ARMIJO_C1 = 1e-4  # Armijo sufficient-decrease constant (residual metric)
_NATURAL_C = (
    1e-3  # sufficient-decrease margin of the natural test (displacement metric)
)


def _line_search_manager(pb, dX):
    """Combine several reistered line search algorithms.

    Manager function that evaluates all registered line search algorithms
    and returns the most restrictive (minimum) step size.
    """
    if not pb._ls_callbacks:
        return 1.0  # Default full Newton step if nothing is registered

    alpha_min = 1.0

    # Evaluate every registered callback
    for name, callback in pb._ls_callbacks.items():
        alpha_trial = callback(pb, dX)

        if alpha_trial < alpha_min:
            alpha_min = alpha_trial
    return alpha_min


def _evaluate_residual_norm(pb, dX, alpha):
    """Calculate the norm of the residual at a trial step alpha.

    The problem is left exactly as found: the displacement increment, the
    assembly state variables and the out-of-balance vector D are restored.
    """
    # 1. Temporarily update the displacement increment
    pb.assembly._save_sv()  # save assembly stat_variables
    d_0 = pb.get_D()
    pb._dU += alpha * dX
    pb._line_search_update = True
    try:
        # 2. Re-assemble the internal force vector (D) at the new displacement
        # Note: 'vector' compute usually updates internal forces/stresses
        try:
            pb.update(compute="vector")
            pb._update_d()
        except InvalidKinematicStateError:
            # degenerated trial state (det F <= 0): infinitely bad candidate,
            # backtracking will shrink alpha
            return np.inf, None

        # 3. Calculate residual R = B + D
        # use MatBC to catch only the free_node values and eliminate mpc
        # slave nodes
        res = pb._get_free_dof_residual()
        return np.linalg.norm(res, pb.nr_parameters["norm_type"]), res
    finally:
        # 4. Cleanup: restore the original state so the search never
        # permanently alters it
        pb._line_search_update = False
        pb._dU -= alpha * dX
        pb.assembly._load_sv()
        pb.set_D(d_0)


def _natural_test(pb, res, dx_norm, alpha, norm_type):
    r"""Deuflhard's affine-invariant sufficient-decrease test.

    The merit is the SIMPLIFIED Newton correction
    :math:`\bar{dX}(\alpha) = K^{-1} R(u + \alpha\, dX)`, one
    back-substitution on the cached factorization of the current tangent.
    Unlike :math:`\|R\|` it lives in displacement space and is invariant to
    the scaling of the equations (Deuflhard, *Newton Methods for Nonlinear
    Problems*, NLEQ-ERR), so the stiff nodal equations no longer inflate the
    quadratic remainder of a legitimate large soft-mode step. It is the
    companion of the "Displacement" NR criterion, which measures the same
    quantity.

    Accepted on monotone decrease,
    :math:`\|\bar{dX}(\alpha)\| \le (1 - c\,\alpha) \|dX\|`. The
    contraction test of the theory (:math:`\theta \le 1 - \alpha/4`) assumes
    the EXACT Jacobian; with a modified Newton tangent it rejects legitimate
    steps down to alpha ~ 0, while monotonicity still catches an overshoot
    (:math:`\theta > 1`).
    """
    if dx_norm == 0:
        return True
    dx_bar = pb._solve_reduced(res)
    return np.linalg.norm(dx_bar, norm_type) <= (1.0 - _NATURAL_C * alpha) * dx_norm


def line_search(pb, dX):
    """Line search to find an appropriate step size alpha.

    To be assigned to self._step_size_callback. Three modes, selected by
    the nr parameter "ls_mode" (see Problem.add_line_search):

    - "natural" (default): validity filter, then the trial step is accepted
      when EITHER the Armijo residual test OR Deuflhard's affine-invariant
      test passes (see _natural_test). The first one is the proven throttle
      against the overshoot of penalty contact / elastic-plastic
      transitions, the second one lets the legitimate large steps of soft
      modes through. Backtracking as in "Quadratic".
    - "minimize": residual-descent methods (Residual, Armijo, Quadratic,
      Energy), selected by the nr parameter "ls_method";
    - "safeguard": pure validity filter.
    """
    if not pb._xbc_is_applied():
        # avoid using line_search if dirichlet increment values are not 0
        # to avoid problems related to BC scaling. NB: testing the actual
        # _Xbc content (and not only the _boundary_is_0 flag) keeps the
        # line search ACTIVE on the elastic prediction under pure force
        # control -- the most dangerous iterate for soft free modes.
        return 1

    ls_mode = pb.nr_parameters.get("ls_mode", "natural")
    norm_type = pb.nr_parameters["norm_type"]
    # Residual of the current iterate (alpha = 0). MUST be read before any
    # trial: _evaluate_residual_norm restores _dU, the state variables and D,
    # but a constraint with _update_during_inc rebuilds B, _MatCB and
    # _dof_free during the trial and those are NOT restored.
    res_0 = pb._get_free_dof_residual()
    norm_0 = np.linalg.norm(res_0, norm_type)

    # --- Validity filter -----------------------------------------------------
    # A trial state with det F <= 0 (norm inf) is rejected by geometric
    # backtracking to the first VALID alpha, whatever the mode.
    # TODO(perf): this full residual evaluation is, in safeguard mode, only
    # a det F > 0 test whose work is redone by the caller when the step is
    # valid (the common case). A kinematics-only validity probe would remove
    # one vector assembly + umat sweep per NR iteration.
    norm_1, res_1 = _evaluate_residual_norm(pb, dX, 1.0)
    alpha_valid = 1.0
    while not np.isfinite(norm_1) and alpha_valid > 1e-4:
        alpha_valid *= 0.5
        norm_1, res_1 = _evaluate_residual_norm(pb, dX, alpha_valid)
    if ls_mode == "safeguard":
        if alpha_valid == 1.0:
            return 1
        return max(0.8 * alpha_valid, 1e-4)

    # --- Descent tests ---------------------------------------------------------
    if ls_mode == "natural":
        method = "Natural"
    else:
        method = pb.nr_parameters.get("ls_method", "Quadratic")
    alpha = alpha_valid
    rho = 0.5  # Standard backtracking contraction factor
    max_iter = pb.nr_parameters.get("ls_max_iter", 5)

    f_0 = 0.5 * (norm_0**2)
    # Directional derivative (assumes exact Newton step: dX = -J^-1 * R)
    m = -(norm_0**2)

    if method in ("Energy", "Natural"):
        dX_free = dX[pb._dof_free] if hasattr(pb, "_dof_free") else dX
    if method == "Energy":
        work_0 = np.dot(res_0, dX_free)
    elif method == "Natural":
        dx_norm = np.linalg.norm(dX_free, norm_type)

    # Tracking the best step in case we exhaust max_iter. Start from the
    # last VALID alpha, never 1.0: if every trial below is invalid the
    # fallback must not return the full step the validity stage rejected.
    best_alpha = alpha_valid
    best_norm = float("inf")

    # the first trial (alpha = alpha_valid) reuses the evaluation of the
    # validity stage above instead of re-assembling the same point
    norm_alpha, res_alpha = norm_1, res_1
    for i in range(max_iter):
        if i > 0:
            # Evaluate trial alpha
            norm_alpha, res_alpha = _evaluate_residual_norm(pb, dX, alpha)
        if not np.isfinite(norm_alpha):
            # invalid trial state (det F <= 0): shrink and retry -- the
            # (inf, None) sentinel must not reach the acceptance tests
            alpha *= rho
            continue
        f_alpha = 0.5 * (norm_alpha**2)

        # Track the lowest residual seen so far
        if norm_alpha < best_norm:
            best_norm = norm_alpha
            best_alpha = alpha

        # --- 1. Acceptance Criterion Check ---
        if method == "Residual":
            if norm_alpha < norm_0:
                return alpha
        elif method in ["Armijo", "Quadratic", "Natural"]:
            # Armijo sufficient decrease rule (Quadratic shares it)
            if f_alpha <= f_0 + _ARMIJO_C1 * alpha * m:
                return alpha
            if method == "Natural" and _natural_test(
                pb, res_alpha, dx_norm, alpha, norm_type
            ):
                return alpha
        elif method == "Energy":
            work_alpha = np.dot(res_alpha, dX_free)
            if abs(work_alpha) < 0.5 * abs(work_0):
                return alpha

        # --- 2. Step Reduction Strategy (If rejected) ---
        if method in ["Quadratic", "Natural"]:
            # Calculate the vertex of the interpolating parabola
            denom = 2.0 * (f_alpha - f_0 - m * alpha)

            if abs(denom) > 1e-10:
                alpha_interp = -(m * (alpha**2)) / denom
            else:
                alpha_interp = (
                    alpha * rho
                )  # Fallback to linear backtracking if denom is near zero

            # Safeguard: Force alpha to shrink by at least 50%, but no more than 90%
            # This prevents interpolation from shooting off to effectively zero or barely moving.
            alpha = np.clip(alpha_interp, 0.1 * alpha, 0.5 * alpha)

        else:
            # Standard backtracking for Residual, Armijo, and Energy
            alpha *= rho

    if method == "Natural":
        # never fall back on the lowest-residual trial: that is precisely the
        # merit this mode rejects for soft modes (it would return the most
        # throttled alpha of the sweep). Keep the last kinematically valid one.
        return alpha_valid
    # Criteria were not met within max_iter. Fallback to best alpha found.
    return best_alpha
