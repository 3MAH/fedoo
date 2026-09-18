"""Shared base class for eigenvalue problems."""

import numpy as np
from scipy import linalg
from scipy.sparse.linalg import eigsh

from fedoo.core.problem import Problem


class _EigenvalueProblem(Problem):
    """Problem base class overriding linear-system solver configuration."""

    @property
    def solver(self):
        """Configured eigensolver function, or ``None`` in automatic mode."""
        return {"auto": None, "eigsh": eigsh, "eigh": linalg.eigh}[self._eigensolver]

    @property
    def _solver_type(self):
        """Name of the configured eigensolver."""
        return self._eigensolver

    def set_solver(
        self,
        solver="auto",
        *,
        which=None,
        tol=0.0,
        maxiter=None,
        ncv=None,
    ):
        """Configure the symmetric generalized eigensolver.

        Parameters
        ----------
        solver : {"auto", "eigsh", "eigh"}, default="auto"
            ``"auto"`` retains the problem's automatic sparse/dense
            selection, ``"eigsh"`` forces SciPy's sparse ARPACK solver, and
            ``"eigh"`` forces SciPy's dense symmetric solver.
        which : str, optional
            ARPACK eigenvalue selection used by ``eigsh``. If omitted,
            ``solve`` selects a problem-specific default based on whether
            ``sigma`` is set.
        tol, maxiter, ncv : optional
            Parameters forwarded to :func:`scipy.sparse.linalg.eigsh` when the
            sparse solver is selected. They are retained but unused by a dense
            solve.
        """
        if not isinstance(solver, str):
            raise TypeError("The eigensolver name must be a string.")
        solver = solver.lower()
        if solver not in {"auto", "eigsh", "eigh"}:
            raise ValueError("solver must be 'auto', 'eigsh', or 'eigh'.")

        if which is not None:
            if not isinstance(which, str):
                raise TypeError("which must be a string or None.")
            which = which.upper()
            if which not in {"LM", "SM", "LA", "SA", "BE"}:
                raise ValueError("Invalid eigsh value for which.")

        try:
            tol = 0.0 if tol is None else float(tol)
        except (TypeError, ValueError) as error:
            raise TypeError("tol must be a real number.") from error
        if not np.isfinite(tol) or tol < 0:
            raise ValueError("tol must be non-negative.")

        for name, value in (("maxiter", maxiter), ("ncv", ncv)):
            if value is not None and (
                not isinstance(value, (int, np.integer)) or value <= 0
            ):
                raise ValueError(f"{name} must be a strictly positive integer.")

        self._eigensolver = solver
        self._eigensolver_options = {
            "which": which,
            "tol": tol,
            "maxiter": maxiter,
            "ncv": ncv,
        }
        return self
