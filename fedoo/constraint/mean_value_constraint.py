"""Constraint on the mean value of a field using Lagrange multipliers."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from fedoo.core.base import AssemblyBase
from fedoo.core.modelingspace import ModelingSpace


class MeanValueConstraint(AssemblyBase):
    """Enforce the weighted mean of a field over a set of nodes.

    For each variable, the constraint reads:

    .. math::
        \\sum_i w_i \\; u(i) = value

    where the weights :math:`w_i` sum to 1 (mean value). The constraint is
    enforced with one Lagrange multiplier per variable (one additional global
    dof, see :py:meth:`fedoo.Problem.add_global_dof`), leading to a bordered
    (saddle-point) system.

    The main application is the removal of the rigid body translation in
    periodic homogenization problems: constraining the mean displacement of
    the RVE to zero avoids pinning an arbitrary node and makes the solution
    independent of the choice of that node.

    Parameters
    ----------
    mesh: fedoo.Mesh
        Mesh associated to the constraint. Should be the mesh of the
        assembly the constraint is summed with (same list of nodes).
    variable: str or list of str, default = "Disp"
        Vector name (e.g. "Disp"), variable name (e.g. "DispX") or list of
        variable names. One scalar constraint (and one Lagrange multiplier)
        is created per variable.
    value: float, default = 0.
        Imposed mean value. The value is not affected by the time factor of
        the problem (no ramp).
    node_set: str, array of int or None (default)
        Nodes over which the mean is computed. If None, all the mesh nodes
        are used.
    weights: None, "volume" or array, default = None
        Weights used to compute the mean value:

        * None: uniform weights 1/n_nodes (simple node average).
        * "volume": nodal integration weights so that the constraint is the
          true volume average of the interpolated field.
        * array of float with same len as the node set: custom weights.
          The weights are normalized so that their sum is 1.
    space: ModelingSpace, optional
        Modeling space. If None, the active ModelingSpace is used.
    name: str, default = "MeanValue"
        Name of the constraint. The Lagrange multiplier dofs are named
        "{name}_{variable}" and gathered in a global vector "{name}". Use
        distinct names to define several MeanValueConstraint on the same
        problem, or across several problems if they are looked up by name
        (like every named assembly, a default-named instance is registered
        in the global assembly registry and a later one overwrites it).

    Notes
    -----
    * The constraint is a linear relation enforced exactly at each Newton
      iteration: it can be used with both Linear and NonLinear problems.
    * The Lagrange multiplier adds a zero diagonal term to the system matrix
      (saddle-point structure): a direct solver is required (default solver).
      Iterative solvers like "cg" will fail.
    * The value of the Lagrange multipliers can be extracted with
      pb.get_dof_solution("{name}_{variable}").

    Example
    -------
    Remove the rigid body translation of a periodic problem:

    >>> import fedoo as fd
    >>> # ... mesh, material and wf definition ...
    >>> solid = fd.Assembly.create(wf, mesh)
    >>> pb = fd.problem.Linear(solid + fd.constraint.MeanValueConstraint(mesh))
    >>> pb.bc.add(fd.constraint.PeriodicBC())
    >>> pb.bc.add("Dirichlet", "MeanStrain", [0.01, 0, 0, 0, 0, 0])
    >>> pb.solve()
    """

    def __init__(
        self,
        mesh,
        variable: str | list[str] = "Disp",
        value: float = 0.0,
        node_set=None,
        weights=None,
        space: ModelingSpace | None = None,
        name: str = "MeanValue",
    ):
        if space is None:
            space = ModelingSpace.get_active()
        self.mesh = mesh
        AssemblyBase.__init__(self, name, space)

        self.variable = variable
        self.value = value
        self.node_set = node_set
        self.weights = weights

        self._pb = None
        self._registered_pbs = []  # problems whose lagrange dofs are defined

    def __repr__(self):
        return (
            f"fedoo.constraint.MeanValueConstraint(variable={self.variable!r}, "
            f"value={self.value!r}, name={self.name!r})"
        )

    def initialize(self, pb):
        self._pb = pb

        # resolve node set
        if self.node_set is None:
            nodes = np.arange(self.mesh.n_nodes)
        elif isinstance(self.node_set, str):
            nodes = np.asarray(self.mesh.node_sets[self.node_set], dtype=int)
        else:
            nodes = np.asarray(self.node_set, dtype=int)
        if len(nodes) == 0:
            raise ValueError("The node set of a MeanValueConstraint is empty.")
        self._nodes = nodes

        # resolve variables
        space = self.space
        if isinstance(self.variable, str):
            if self.variable in space.list_vectors():
                var_names = list(space.get_vector(self.variable))
            else:
                var_names = [self.variable]
        else:
            var_names = list(self.variable)
        self._var_names = var_names
        self._ranks = [space.variable_rank(var) for var in var_names]

        # the imposed value is broadcast over the constrained variables
        if not np.isscalar(self.value):
            value_arr = np.asarray(self.value, dtype=float).ravel()
            if len(value_arr) != len(var_names):
                raise ValueError(
                    f"value has {len(value_arr)} components but the constraint "
                    f"involves {len(var_names)} variable(s)."
                )

        # compute weights (normalized so that their sum is 1)
        if self.weights is None:
            w = np.full(len(nodes), 1.0 / len(nodes))
        elif isinstance(self.weights, str):
            if self.weights.lower() != "volume":
                raise ValueError(
                    f"weights={self.weights!r} unknown. "
                    'Use None, "volume" or an array of weights.'
                )
            # nodal integration weights: w_i = int_V N_i dV
            nodal_weights = np.asarray(
                (
                    self.mesh._get_gaussian_quadrature_mat()
                    @ self.mesh._get_node2gausspoint_mat()
                ).sum(axis=0)
            ).ravel()
            w = nodal_weights[nodes]
            sum_w = w.sum()
            if abs(sum_w) < 1e-15:
                raise ValueError(
                    "The nodal integration weights sum to 0 over the node "
                    "set; cannot build a volume-averaged constraint."
                )
            w = w / sum_w
        else:
            w = np.asarray(self.weights, dtype=float)
            if w.ndim != 1 or len(w) != len(nodes):
                raise ValueError(
                    "weights should be a 1D array with the same length as the "
                    f"node set (got shape {w.shape}, expected ({len(nodes)},))."
                )
            sum_w = w.sum()
            if abs(sum_w) < 1e-15:
                raise ValueError("The sum of the weights should not be 0.")
            w = w / sum_w
        self._weights = w

        # add the lagrange multiplier dofs (one global dof per variable)
        self._lm_names = [f"{self.name}_{var}" for var in var_names]
        if not any(pb is p for p in self._registered_pbs):
            if self.name in pb._global_dof._vector or any(
                lm_name in pb._global_dof for lm_name in self._lm_names
            ):
                raise NameError(
                    f"A global dof named '{self.name}' already exists in the "
                    "problem. Use a different name to define several "
                    "MeanValueConstraint on the same problem."
                )
            pb.add_global_dof(self._lm_names, 1, vector_name=self.name)
            self._registered_pbs.append(pb)

        self.delete_global_mat()

    def _lagrange_dofs(self, pb):
        return np.array(
            [
                pb.n_node_dof + pb._global_dof.indice_start(lm_name)
                for lm_name in self._lm_names
            ]
        )

    def assemble_global_mat(self, compute="all"):
        if compute == "none":
            return
        pb = self._pb
        if pb is None:
            raise RuntimeError(
                "MeanValueConstraint can't be assembled before being "
                "initialized by a problem."
            )
        n_dof = pb.n_dof

        # The bordered matrix is constant after initialization: (re)build it
        # only when a matrix is requested, missing, or resized. In the NR loop
        # only the vector (which depends on U) needs to be recomputed.
        if (
            compute != "vector"
            or self.global_matrix is None
            or self.global_matrix.shape[0] != n_dof
        ):
            self.global_matrix = self._assemble_matrix(pb, n_dof)

        if compute != "matrix":
            self.global_vector = self._assemble_vector(pb, n_dof)

    def _assemble_matrix(self, pb, n_dof):
        # bordered matrix: K_c[lm, u] = K_c[u, lm] = w
        n_nodes = pb.mesh.n_nodes
        lm_dofs = self._lagrange_dofs(pb)
        n_terms = len(self._nodes)
        row = np.empty(2 * n_terms * len(self._ranks), dtype=int)
        col = np.empty_like(row)
        data = np.empty(len(row), dtype=float)
        for k, rank in enumerate(self._ranks):
            dof_u = rank * n_nodes + self._nodes
            start = 2 * n_terms * k
            row[start : start + n_terms] = lm_dofs[k]
            col[start : start + n_terms] = dof_u
            row[start + n_terms : start + 2 * n_terms] = dof_u
            col[start + n_terms : start + 2 * n_terms] = lm_dofs[k]
            data[start : start + 2 * n_terms] = np.tile(self._weights, 2)
        return sparse.csr_matrix((data, (row, col)), shape=(n_dof, n_dof))

    def _assemble_vector(self, pb, n_dof):
        # global vector D = b - K_c @ U (fedoo convention: A@X = B + D)
        b = np.zeros(n_dof)
        b[self._lagrange_dofs(pb)] = self.value
        U = pb.get_dof_solution()
        if np.isscalar(U) and U == 0:
            return b
        if len(U) < n_dof:
            U = np.hstack((U, np.zeros(n_dof - len(U))))
        return b - self.global_matrix @ U

    def update(self, pb, compute="all"):
        self._pb = pb
        self.assemble_global_mat(compute)

    def set_start(self, pb):
        self._pb = pb
        if self.global_matrix is not None:
            self.assemble_global_mat("vector")

    def to_start(self, pb):
        self._pb = pb
        if self.global_matrix is not None:
            self.assemble_global_mat("vector")

    def reset(self):
        self.delete_global_mat()
