"""Free-vibration modal analysis."""

import numpy as np
from scipy import linalg
from scipy.sparse.linalg import eigsh

from fedoo.core.assembly import Assembly
from fedoo.core.matrix import as_global_csr
from fedoo.core.problem import Problem
from fedoo.core.time_evolution import SECOND_ORDER
from fedoo.time.common import build_storage_assembly


class Modal(Problem):
    r"""Compute natural frequencies and mode shapes of a linear model.

    The undamped free-vibration problem is

    .. math:: \mathbf{K}\boldsymbol{\phi}
              = \omega^2\mathbf{M}\boldsymbol{\phi}.

    ``Modal`` obtains the stiffness and consistent mass matrices from the
    supplied assembly. Dirichlet and multi-point constraints are applied to
    both matrices before the generalized symmetric eigenproblem is solved.

    Parameters
    ----------
    assembly : Assembly-like object or str
        Linear mechanical assembly. Its weak forms must provide second-order
        storage, normally through a material density or ``set_inertia``.
    name : str, default="MainProblem"
        Name of the problem.

    Notes
    -----
    Mode shapes are mass-normalized and stored by row in :attr:`modes`.
    Frequencies are expressed in cycles per unit of time; angular frequencies
    are expressed in radians per unit of time.
    """

    def __init__(self, assembly: Assembly, name: str = "MainProblem"):
        if isinstance(assembly, str):
            assembly = Assembly.get_all()[assembly]

        super().__init__(mesh=assembly.mesh, name=name, space=assembly.space)
        self.nlgeom = False
        assembly.register_global_dofs(self)
        assembly.initialize(self)
        self.__assembly = assembly

        self._mass_assembly = None
        self._stiffness_matrix = None
        self._mass_matrix = None
        self._reduced_stiffness = None
        self._reduced_mass = None

        self.eigenvalues = np.empty(0)
        self.angular_frequencies = np.empty(0)
        self.frequencies = np.empty(0)
        self.modes = np.empty((0, self.n_dof))
        self._current_mode = None

    @property
    def assembly(self):
        """Mechanical assembly used by the modal problem."""
        return self.__assembly

    @property
    def mass_assembly(self):
        """Assembly used to construct the consistent mass matrix."""
        return self._mass_assembly

    @property
    def stiffness_matrix(self):
        """Full unconstrained stiffness matrix from the last solve."""
        return self._stiffness_matrix

    @property
    def mass_matrix(self):
        """Full unconstrained consistent mass matrix from the last solve."""
        return self._mass_matrix

    @property
    def current_mode(self):
        """Zero-based index of the mode installed as the current solution."""
        return self._current_mode

    def get_disp(self, name="all"):
        """Return displacement components of the currently selected mode."""
        if name == "all":
            name = "Disp"
        return self.get_dof_solution(name)

    def get_rot(self, name="all"):
        """Return rotation components of the currently selected mode."""
        if name == "all":
            name = "Rot"
        return self.get_dof_solution(name)

    def get_output_scalars(self):
        """Return metadata attached to the currently selected output mode."""
        if self._current_mode is None:
            return {}
        index = self._current_mode
        return {
            "Mode": index + 1,
            "Eigenvalue": self.eigenvalues[index],
            "AngularFrequency": self.angular_frequencies[index],
            "Frequency": self.frequencies[index],
        }

    def _assemble_operators(self):
        self._stiffness_matrix = as_global_csr(
            self.__assembly.get_global_matrix(), self.n_dof
        )

        storage_data = build_storage_assembly(self.__assembly, SECOND_ORDER)
        self._mass_assembly = storage_data.assembly
        if self._mass_assembly is None:
            raise ValueError(
                "No second-order storage was found for modal analysis. "
                "Set the material density or attach inertia with "
                "weakform.set_inertia(...)."
            )
        self._mass_assembly.initialize(self)
        self._mass_assembly.assemble_global_mat("matrix")
        self._mass_matrix = as_global_csr(
            self._mass_assembly.get_global_matrix(), self.n_dof
        )

    def _apply_constraints(self):
        self.apply_boundary_conditions()
        if not np.allclose(self._Xbc, 0.0):
            raise ValueError(
                "Modal analysis requires homogeneous Dirichlet and MPC "
                "constraints. Prescribed nonzero displacements are not valid "
                "in a free-vibration eigenproblem."
            )
        if self._MatCB.shape[1] == 0:
            raise ValueError("Modal analysis has no free degrees of freedom.")

        transform = self._MatCB
        self._reduced_stiffness = (
            transform.T @ self._stiffness_matrix @ transform
        ).tocsr()
        self._reduced_mass = (transform.T @ self._mass_matrix @ transform).tocsr()

        mass_row_norm = np.asarray(abs(self._reduced_mass).sum(axis=1)).ravel()
        if np.any(mass_row_norm == 0):
            count = int(np.count_nonzero(mass_row_norm == 0))
            raise ValueError(
                f"The constrained mass matrix contains {count} massless free "
                "DOF(s). Define inertia for those DOFs or constrain/condense them."
            )

    @staticmethod
    def _dense_eigenpairs(stiffness, mass, n_modes):
        try:
            values, vectors = linalg.eigh(
                stiffness.toarray(),
                mass.toarray(),
                subset_by_index=(0, n_modes - 1),
                check_finite=False,
            )
        except linalg.LinAlgError as error:
            raise ValueError(
                "The constrained mass matrix must be symmetric positive definite."
            ) from error
        return values, vectors

    def solve(
        self,
        n_modes=6,
        sigma=None,
        which=None,
        tol=0.0,
        maxiter=None,
        ncv=None,
    ):
        r"""Solve the generalized eigenvalue problem.

        Parameters
        ----------
        n_modes : int, default=6
            Number of eigenmodes to compute.
        sigma : float, optional
            Target eigenvalue :math:`\omega^2`. When supplied, shift-invert
            mode finds eigenvalues nearest ``sigma``. Leaving it unset avoids
            factorizing a potentially singular free-free stiffness matrix.
        which : str, optional
            ARPACK eigenvalue selection. Defaults to ``"SM"`` without a
            shift and ``"LM"`` with a shift.
        tol, maxiter, ncv : optional
            Parameters forwarded to :func:`scipy.sparse.linalg.eigsh`.
        Returns
        -------
        Modal
            This problem, with eigenpairs available through ``eigenvalues``,
            ``frequencies`` and ``modes``.
        """
        if not isinstance(n_modes, (int, np.integer)) or n_modes <= 0:
            raise ValueError("n_modes must be a strictly positive integer.")

        self._assemble_operators()
        self._apply_constraints()
        n_free = self._reduced_stiffness.shape[0]
        n_modes = min(int(n_modes), n_free)

        if n_modes == n_free:
            if n_free > 256:
                raise ValueError(
                    "Computing every mode requires a dense solve and is limited "
                    "to 256 free DOFs. Request fewer than the number of free DOFs."
                )
            values, reduced_vectors = self._dense_eigenpairs(
                self._reduced_stiffness, self._reduced_mass, n_modes
            )
        else:
            if which is None:
                which = "LM" if sigma is not None else "SM"
            values, reduced_vectors = eigsh(
                self._reduced_stiffness,
                k=n_modes,
                M=self._reduced_mass,
                sigma=sigma,
                which=which,
                tol=tol,
                maxiter=maxiter,
                ncv=ncv,
            )

        order = np.argsort(values)
        values = np.asarray(values[order], dtype=float)
        reduced_vectors = np.asarray(reduced_vectors[:, order], dtype=float)

        # A semidefinite free-free stiffness can return tiny signed roundoff
        # values for rigid-body modes. Preserve genuine negative eigenvalues,
        # but represent numerical zeros as zero so their frequency is 0.
        zero_tolerance = (
            100.0 * np.finfo(float).eps * max(1.0, float(np.max(np.abs(values))))
        )
        values[np.abs(values) <= zero_tolerance] = 0.0

        # eigsh normally returns M-normalized vectors. Normalize explicitly so
        # the same contract also holds for the dense path and custom tolerances.
        for index in range(n_modes):
            vector = reduced_vectors[:, index]
            norm_squared = float(vector @ (self._reduced_mass @ vector))
            if not np.isfinite(norm_squared) or norm_squared <= 0:
                raise ValueError(
                    "A computed mode has non-positive generalized mass; the "
                    "mass matrix must be symmetric positive definite."
                )
            vector /= np.sqrt(norm_squared)
            largest = int(np.argmax(np.abs(vector)))
            if vector[largest] < 0:
                vector *= -1

        full_vectors = np.asarray(self._MatCB @ reduced_vectors)
        self.eigenvalues = values
        self.angular_frequencies = np.full(values.shape, np.nan)
        nonnegative = values >= 0
        self.angular_frequencies[nonnegative] = np.sqrt(values[nonnegative])
        self.frequencies = self.angular_frequencies / (2.0 * np.pi)
        self.modes = full_vectors.T
        self.set_A(self._stiffness_matrix)

        # Make the first requested mode immediately available to get_disp(),
        # get_results() and viewers, without forcing a constitutive update.
        self.set_mode(0, update_weakform=False)
        if self._problem_output.has_outputs:
            self.save_modes()
            # Automatic output should not change the default current mode.
            self.set_mode(0, update_weakform=False)
        return self

    def set_mode(self, index, scale=1.0, update_weakform=True):
        """Install one mode shape as the current displacement solution.

        ``index`` is zero-based. ``scale`` affects only the displayed/current
        solution; the mass-normalized mode stored in :attr:`modes` is unchanged.
        """
        if not isinstance(index, (int, np.integer)):
            raise TypeError("Mode index must be an integer.")
        if index < 0 or index >= len(self.eigenvalues):
            raise IndexError(f"Mode index {index} is out of range.")
        self._current_mode = int(index)
        self.set_X(float(scale) * self.modes[index].copy())
        if update_weakform:
            self.__assembly.update(self, compute="none")
        return self.get_X()

    def save_modes(self, modes=None, scale=1.0, update_weakform=True):
        """Save selected modes through outputs registered with ``add_output``.

        Parameters
        ----------
        modes : iterable of int, optional
            Zero-based mode indices. All computed modes are saved by default.
        scale : float, default=1
            Visualization scale applied to each saved mode shape.
        update_weakform : bool, default=True
            Update derived assembly fields such as strain and stress before
            saving each mode. Every frame also receives ``Mode``,
            ``Eigenvalue``, ``AngularFrequency`` and ``Frequency`` scalar
            metadata.
        """
        if modes is None:
            modes = range(len(self.eigenvalues))
        for index in modes:
            self.set_mode(index, scale=scale, update_weakform=update_weakform)
            self.save_results(index)
        return self
