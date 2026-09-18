"""Linear eigenvalue buckling analysis from a prestressed state."""

import numpy as np
from scipy import linalg
from scipy.sparse.linalg import eigsh

from fedoo.core.assembly import Assembly
from fedoo.core.assembly_sum import AssemblySum
from fedoo.core.base import AssemblyBase
from fedoo.core.matrix import as_global_csr
from fedoo.problem._eigenvalue_problem import _EigenvalueProblem


def _weakform_leaves(assembly):
    for leaf_assembly in assembly.iter_leaf():
        weakform = getattr(leaf_assembly, "weakform", None)
        if weakform is None:
            raise NotImplementedError(
                f"Linear buckling does not support {type(leaf_assembly).__name__}."
            )
        for weakform_leaf in weakform.iter_leaf():
            yield leaf_assembly, weakform_leaf


def _geometric_stiffness_state(assembly):
    """Return the homogeneous geometric-stiffness state, or ``mixed``."""
    states = []
    unsupported = []
    for leaf_assembly, weakform in _weakform_leaves(assembly):
        if not hasattr(weakform, "geometric_stiffness"):
            unsupported.append(type(weakform).__name__)
            continue
        value = weakform.geometric_stiffness
        if value is None:
            value = bool(leaf_assembly._nlgeom)
        states.append(bool(value))

    if unsupported:
        names = ", ".join(sorted(set(unsupported)))
        raise NotImplementedError(
            "Linear buckling requires every weak form to expose a "
            f"'geometric_stiffness' attribute. Unsupported: {names}."
        )
    if not states:
        raise NotImplementedError(
            "Linear buckling requires at least one compatible weak form."
        )
    if all(states):
        return "enabled"
    if not any(states):
        return "disabled"
    return "mixed"


def _set_geometric_stiffness(weakform, value):
    for leaf in weakform.iter_leaf():
        leaf.geometric_stiffness = value


class LinearBuckling(_EigenvalueProblem):
    r"""Compute linear buckling modes about a prestressed equilibrium state.

    The problem solved is

    .. math:: \mathbf{K}_M\boldsymbol{\phi}
              = -\lambda\mathbf{K}_G\boldsymbol{\phi},

    where :math:`\mathbf{K}_M` is the material tangent stiffness and
    :math:`\mathbf{K}_G` is the initial-stress stiffness. The supplied
    assembly must already contain the converged preload state.

    Parameters
    ----------
    assembly : Assembly-like, Problem or str
        Prestressed assembly, a problem exposing it through ``assembly``, or
        the name of a registered assembly.
    reuse_assembled_matrix : bool, default=True
        Reuse the cached preload matrix when all contributing weak forms agree
        on whether geometric stiffness was enabled. If False, independently
        assemble both required matrices.
    name : str, default="MainProblem"
        Name of the problem.

    Notes
    -----
    Updated-Lagrangian preload states are not supported. Boundary conditions
    describe perturbations and must therefore be homogeneous.
    """

    def __init__(
        self,
        assembly,
        reuse_assembled_matrix=True,
        name="MainProblem",
    ):
        if not isinstance(reuse_assembled_matrix, (bool, np.bool_)):
            raise TypeError("reuse_assembled_matrix must be a boolean.")
        if not isinstance(assembly, (str, AssemblyBase)) and hasattr(
            assembly, "assembly"
        ):
            assembly = assembly.assembly
        if isinstance(assembly, str):
            assembly = AssemblyBase.get_all()[assembly]
        if not isinstance(assembly, AssemblyBase):
            raise TypeError("assembly must be an assembly or a problem with one.")

        super().__init__(mesh=assembly.mesh, name=name, space=assembly.space)
        self.nlgeom = False
        assembly.register_global_dofs(self)
        self.__assembly = assembly
        self.reuse_assembled_matrix = bool(reuse_assembled_matrix)

        self._material_assembly = None
        self._total_assembly = None
        self._material_stiffness_matrix = None
        self._geometric_stiffness_matrix = None
        self._total_stiffness_matrix = None
        self._reduced_material_stiffness = None
        self._reduced_geometric_stiffness = None
        self.eigenvalues = np.empty(0)
        self.load_factors = self.eigenvalues
        self.modes = np.empty((0, self.n_dof))
        self._current_mode = None

    @property
    def assembly(self):
        """Prestressed source assembly."""
        return self.__assembly

    @property
    def material_assembly(self):
        """Assembly built for material stiffness, or the reused source."""
        return self._material_assembly

    @property
    def total_assembly(self):
        """Assembly built for total tangent stiffness, or the reused source."""
        return self._total_assembly

    @property
    def material_stiffness_matrix(self):
        """Full unconstrained material tangent stiffness."""
        return self._material_stiffness_matrix

    @property
    def geometric_stiffness_matrix(self):
        """Full unconstrained initial-stress stiffness."""
        return self._geometric_stiffness_matrix

    @property
    def total_stiffness_matrix(self):
        """Full unconstrained material-plus-geometric tangent stiffness."""
        return self._total_stiffness_matrix

    @property
    def current_mode(self):
        """Zero-based index of the mode installed as the current solution."""
        return self._current_mode

    def get_disp(self, name="all"):
        """Return displacement components of the selected buckling mode."""
        if name == "all":
            name = "Disp"
        return self.get_dof_solution(name)

    def get_rot(self, name="all"):
        """Return rotation components of the selected buckling mode."""
        if name == "all":
            name = "Rot"
        return self.get_dof_solution(name)

    def get_output_scalars(self):
        """Return metadata attached to the currently selected mode."""
        if self._current_mode is None:
            return {}
        index = self._current_mode
        return {
            "Mode": index + 1,
            "Eigenvalue": self.eigenvalues[index],
            "LoadFactor": self.load_factors[index],
        }

    def _copy_leaf_assembly(self, source, geometric_stiffness):
        if source._pb is None:
            raise RuntimeError(
                "The preload assembly must be initialized before linear "
                "buckling analysis."
            )
        if source.current is not source or source._nlgeom == "UL":
            raise NotImplementedError(
                "Linear buckling does not support Updated-Lagrangian assemblies."
            )

        weakform = source.weakform.copy()
        _set_geometric_stiffness(weakform, geometric_stiffness)
        result = Assembly(
            weakform,
            source.mesh,
            source.elm_type,
            n_elm_gp=source.n_elm_gp,
        )
        result.assume_sym = source.assume_sym
        result.mat_lumping = source.mat_lumping
        result._pb = self
        result._nlgeom = source._nlgeom
        result.sv = source.sv.copy()
        result.sv_start = source.sv_start.copy()
        result.sv_type = source.sv_type.copy()
        result.sv_component = source.sv_component.copy()
        result._initial_element_local_frame = source._initial_element_local_frame
        result._element_local_frame = source._element_local_frame
        return result

    def _copy_assembly(self, geometric_stiffness):
        copied = [
            self._copy_leaf_assembly(source, geometric_stiffness)
            for source in self.__assembly.iter_leaf()
        ]
        result = copied[0] if len(copied) == 1 else AssemblySum(copied)
        result.register_global_dofs(self)
        return result

    def _assemble_matrix(self, geometric_stiffness):
        assembly = self._copy_assembly(geometric_stiffness)
        assembly.assemble_global_mat("matrix")
        matrix = as_global_csr(assembly.get_global_matrix(), self.n_dof).copy()
        return assembly, matrix

    def _assemble_operators(self):
        state = _geometric_stiffness_state(self.__assembly)
        source_matrix = self.__assembly.current.global_matrix
        can_reuse = self.reuse_assembled_matrix and source_matrix is not None

        if can_reuse and state == "disabled":
            self._material_assembly = self.__assembly
            self._material_stiffness_matrix = as_global_csr(
                source_matrix, self.n_dof
            ).copy()
            (
                self._total_assembly,
                self._total_stiffness_matrix,
            ) = self._assemble_matrix(True)
        elif can_reuse and state == "enabled":
            self._total_assembly = self.__assembly
            self._total_stiffness_matrix = as_global_csr(
                source_matrix, self.n_dof
            ).copy()
            (
                self._material_assembly,
                self._material_stiffness_matrix,
            ) = self._assemble_matrix(False)
        else:
            (
                self._material_assembly,
                self._material_stiffness_matrix,
            ) = self._assemble_matrix(False)
            (
                self._total_assembly,
                self._total_stiffness_matrix,
            ) = self._assemble_matrix(True)

        self._geometric_stiffness_matrix = (
            self._total_stiffness_matrix - self._material_stiffness_matrix
        ).tocsr()

    def _apply_constraints(self):
        self.apply_boundary_conditions()
        if not np.allclose(self._Xbc, 0.0):
            raise ValueError(
                "Linear buckling requires homogeneous Dirichlet and MPC " "constraints."
            )
        if self._MatCB.shape[1] == 0:
            raise ValueError("Linear buckling has no free degrees of freedom.")

        transform = self._MatCB
        self._reduced_material_stiffness = (
            transform.T @ self._material_stiffness_matrix @ transform
        ).tocsr()
        self._reduced_geometric_stiffness = (
            transform.T @ self._geometric_stiffness_matrix @ transform
        ).tocsr()

    def solve(self, n_modes=6, sigma=None):
        """Solve for positive critical load factors and buckling modes.

        ``sigma`` optionally targets modes near a positive load factor. The
        lowest positive factors are returned by default.
        """
        if not isinstance(n_modes, (int, np.integer)) or n_modes <= 0:
            raise ValueError("n_modes must be a strictly positive integer.")
        if sigma is not None and sigma <= 0:
            raise ValueError("sigma must be a positive load factor.")

        self._assemble_operators()
        self._apply_constraints()
        material = self._reduced_material_stiffness
        minus_geometric = -self._reduced_geometric_stiffness
        n_free = material.shape[0]

        try:
            use_dense = self._eigensolver == "eigh" or (
                self._eigensolver == "auto" and (n_free <= 256 or n_modes >= n_free)
            )
            if self._eigensolver == "eigsh" and n_modes >= n_free:
                raise ValueError(
                    "eigsh requires n_modes to be smaller than the number of "
                    "free degrees of freedom. Use solver='auto' or "
                    "solver='eigh' to compute every mode."
                )

            if use_dense:
                reciprocal_values, reduced_vectors = linalg.eigh(
                    minus_geometric.toarray(),
                    material.toarray(),
                    check_finite=False,
                )
            else:
                k = min(int(n_modes), n_free - 1)
                options = self._eigensolver_options
                eigsh_options = {
                    "k": k,
                    "M": material,
                    "tol": options["tol"],
                    "maxiter": options["maxiter"],
                    "ncv": options["ncv"],
                }
                if sigma is None:
                    eigsh_options["which"] = (
                        "LA" if options["which"] is None else options["which"]
                    )
                else:
                    eigsh_options["sigma"] = 1.0 / sigma
                    eigsh_options["which"] = (
                        "LM" if options["which"] is None else options["which"]
                    )
                reciprocal_values, reduced_vectors = eigsh(
                    minus_geometric, **eigsh_options
                )
        except linalg.LinAlgError as error:
            raise ValueError(
                "The constrained material stiffness must be symmetric positive "
                "definite."
            ) from error

        reciprocal_values = np.asarray(reciprocal_values, dtype=float)
        reduced_vectors = np.asarray(reduced_vectors, dtype=float)
        tolerance = (
            100.0
            * np.finfo(float).eps
            * max(1.0, float(np.max(np.abs(reciprocal_values))))
        )
        positive = np.isfinite(reciprocal_values) & (reciprocal_values > tolerance)
        if not np.any(positive):
            raise ValueError("No positive linear buckling load factor was found.")

        load_factors = 1.0 / reciprocal_values[positive]
        reduced_vectors = reduced_vectors[:, positive]
        if sigma is None:
            selected = np.argsort(load_factors)[: int(n_modes)]
        else:
            selected = np.argsort(np.abs(load_factors - sigma))[: int(n_modes)]
            selected = selected[np.argsort(load_factors[selected])]
        load_factors = load_factors[selected]
        reduced_vectors = reduced_vectors[:, selected]

        for index in range(len(load_factors)):
            vector = reduced_vectors[:, index]
            norm_squared = float(vector @ (material @ vector))
            if not np.isfinite(norm_squared) or norm_squared <= 0:
                raise ValueError("A buckling mode has non-positive strain energy.")
            vector /= np.sqrt(norm_squared)
            largest = int(np.argmax(np.abs(vector)))
            if vector[largest] < 0:
                vector *= -1

        self.load_factors = load_factors
        self.eigenvalues = self.load_factors
        self.modes = np.asarray(self._MatCB @ reduced_vectors).T
        self.set_A(self._material_stiffness_matrix)
        self.set_mode(0)
        if self._problem_output.has_outputs:
            self.save_modes()
            self.set_mode(0)
        return self

    def set_mode(self, index, scale=1.0):
        """Install one zero-based buckling mode as the current solution."""
        if not isinstance(index, (int, np.integer)):
            raise TypeError("Mode index must be an integer.")
        if index < 0 or index >= len(self.load_factors):
            raise IndexError(f"Mode index {index} is out of range.")
        self._current_mode = int(index)
        self.set_X(float(scale) * self.modes[index].copy())
        return self.get_X()

    def save_modes(self, modes=None, scale=1.0):
        """Save selected modes through outputs registered with ``add_output``."""
        if modes is None:
            modes = range(len(self.load_factors))
        for index in modes:
            self.set_mode(index, scale=scale)
            self.save_results(index)
        return self
