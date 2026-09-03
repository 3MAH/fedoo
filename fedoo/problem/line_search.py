"""Functions used for line search."""

import numpy as np

from fedoo.core.base import InvalidKinematicStateError


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
    """Calculate the norm of the residual at a trial step alpha."""
    # 1. Temporarily update the displacement increment
    pb.assembly._save_sv()  # save assembly stat_variables
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


def line_search(pb, dX):
    """Line search to find an appropriate step size alpha.

    To be assigned to self._step_size_callback. Two modes, selected by the
    nr parameter "ls_mode" (see Problem.add_line_search):

    - "safeguard" (default): pure validity filter (see below);
    - "minimize": residual-descent methods (Residual, Armijo, Quadratic,
      Energy), selected by the nr parameter "ls_method".
    """
    if not pb._xbc_is_applied():
        # avoid using line_search if dirichlet increment values are not 0
        # to avoid problems related to BC scaling. NB: testing the actual
        # _Xbc content (and not only the _boundary_is_0 flag) keeps the
        # line search ACTIVE on the elastic prediction under pure force
        # control -- the most dangerous iterate for soft free modes.
        return 1

    # --- Safeguard-first acceptance of the full Newton step -----------------
    # The legitimate full Newton step of a soft mode can be LARGE (e.g. a
    # 13 deg cap rotation towards a distant equilibrium): its quadratic
    # remainder, amplified by the stiff nodal equations, grows the residual
    # norm as step^2 (x10-x1000) while remaining an excellent step that the
    # next iteration kills quadratically. Any residual-monotone rule then
    # strangles Newton. Default policy ("safeguard"): the line search is a
    # PURE VALIDITY FILTER (IPC-style) -- backtrack only on a degenerated
    # trial state (det F <= 0 -> norm inf), with a 0.8 safety factor, and
    # let the main loop's divergence detector handle true divergence.
    # TODO(perf): in safeguard mode this full residual evaluation only
    # tests det F > 0 -- when valid (the common case) its work is redone
    # by the caller. A kinematics-only validity probe would remove one
    # vector assembly + umat sweep per NR iteration.
    ls_mode = pb.nr_parameters.get("ls_mode", "safeguard")
    norm_1, res_1 = _evaluate_residual_norm(pb, dX, 1.0)
    # invalid full step: geometric backtracking to the first VALID alpha
    alpha_valid = 1.0
    while not np.isfinite(norm_1) and alpha_valid > 1e-4:
        alpha_valid *= 0.5
        norm_1, res_1 = _evaluate_residual_norm(pb, dX, alpha_valid)
    if ls_mode == "safeguard":
        if alpha_valid == 1.0:
            return 1
        return max(0.8 * alpha_valid, 1e-4)

    # --- minimize mode: residual-descent methods ---
    method = pb.nr_parameters.get("ls_method", "Quadratic")
    alpha = alpha_valid
    rho = 0.5  # Standard backtracking contraction factor
    c1 = 1e-4  # Armijo sufficient decrease constant
    max_iter = pb.nr_parameters.get("ls_max_iter", 5)

    # --- Initial State Evaluation (alpha = 0) ---
    norm_0, res_0 = _evaluate_residual_norm(pb, dX, 0)
    f_0 = 0.5 * (norm_0**2)

    # Directional derivative (assumes exact Newton step: dX = -J^-1 * R)
    m = -(norm_0**2)

    if method == "Energy":
        dX_free = dX[pb._dof_free] if hasattr(pb, "_dof_free") else dX
        work_0 = np.dot(res_0, dX_free)

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
        elif method in ["Armijo", "Quadratic"]:
            # Both Armijo and Quadratic use the Armijo sufficient decrease rule
            if f_alpha <= f_0 + c1 * alpha * m:
                return alpha
        elif method == "Energy":
            work_alpha = np.dot(res_alpha, dX_free)
            if abs(work_alpha) < 0.5 * abs(work_0):
                return alpha

        # --- 2. Step Reduction Strategy (If rejected) ---
        if method == "Quadratic":
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

    # Criteria were not met within max_iter. Fallback to best alpha found.
    return best_alpha
