"""Functions used for line search."""

import numpy as np


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

    # 2. Re-assemble the internal force vector (D) at the new displacement
    # Note: 'vector' compute usually updates internal forces/stresses
    pb._line_search_update = True
    pb.update(compute="vector")
    pb.updateD()
    pb._line_search_update = False

    # 3. Calculate residual R = B + D
    # use MatBC to catch only the free_node values and eliminate mpc
    # slave nodes
    res = pb._get_free_dof_residual()

    # 4. Cleanup: Restore original dU so we don't permanently alter state during search
    pb._dU -= alpha * dX
    pb.assembly._load_sv()

    return np.linalg.norm(res, pb.nr_parameters["norm_type"]), res


def line_search(pb, dX):
    """Line search to find an appropriate step size alpha.
    Methods: Residual, Armijo, Energy, Quadratic.

    To be assigned to self._step_size_callback.
    """
    # Configuration
    method = pb.nr_parameters.get("ls_method", "Residual")
    alpha = 1.0
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

    # Tracking the best step in case we exhaust max_iter
    best_alpha = 1.0
    best_norm = float("inf")

    for i in range(max_iter):
        # Evaluate trial alpha
        norm_alpha, res_alpha = _evaluate_residual_norm(pb, dX, alpha)
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

    # If the loop finishes without returning, criteria were not met.
    # logging.warning(f"Line search ({method}) reached max_iter={max_iter}. Fallback to best alpha={best_alpha:.4f}.")
    return best_alpha
