"""Constraint on the mean value of a field using Lagrange multipliers."""

from __future__ import annotations

import numpy as np

from fedoo.core.lagrange_multiplier import LagrangeMultiplierAssembly
from fedoo.core.boundary_conditions import ListBC, MPC
from fedoo.core.modelingspace import ModelingSpace


class MeanValueConstraint(LagrangeMultiplierAssembly):
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

    One MPC equation is created per variable and enforced by
    :class:`fedoo.LagrangeMultiplierAssembly`.

    Parameters
    ----------
    mesh: fedoo.Mesh
        Mesh associated with the constraint. It should have the same nodes as
        the assembly with which the constraint is summed.
    variable: str or list of str, default = "Disp"
        Vector name (for example, ``"Disp"``), variable name (for example,
        ``"DispX"``), or list of variable names. One scalar constraint and
        one Lagrange multiplier are created per variable.
    value: float or array, default = 0.
        Imposed mean value. A scalar is applied to every constrained variable;
        an array specifies one value per variable. Values are not affected by
        the problem time factor (no ramp).
    node_set: str, array of int or None, default = None
        Nodes over which the mean is computed. If None, all mesh nodes are
        used.
    weights: None, "volume" or array, default = None
        Weights used to compute the mean:

        * None: uniform weights ``1 / n_nodes`` (simple nodal average).
        * ``"volume"``: nodal integration weights, giving the true volume
          average of the interpolated field.
        * Array with the same length as ``node_set``: custom weights,
          normalized so that their sum is 1.

    space: ModelingSpace, optional
        Modeling space. If None, the active modeling space is used.
    name: str, default = "MeanValue"
        Constraint name. The Lagrange multiplier DOFs are named
        ``{name}_{variable}`` and gathered in the global vector ``{name}``.
        Use distinct names for multiple mean-value constraints on the same
        problem.

    Notes
    -----
    * The constraint is a linear relation enforced exactly at each Newton
      iteration, so it can be used with both linear and nonlinear problems.
    * Lagrange multipliers give the system a zero diagonal block and an
      indefinite saddle-point structure. A direct solver is required;
      iterative solvers such as conjugate gradient are not suitable.
    * A multiplier can be extracted with
      ``pb.get_dof_solution("{name}_{variable}")``.

    Example
    -------
    Remove rigid-body translation from a periodic problem:

    >>> import fedoo as fd
    >>> # ... mesh, material and weak-form definitions ...
    >>> solid = fd.Assembly.create(wf, mesh)
    >>> mean_value = fd.constraint.MeanValueConstraint(mesh)
    >>> pb = fd.problem.Linear(solid + mean_value)
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

        self.variable = variable
        self.value = value
        self.node_set = node_set
        self.weights = weights

        # MPCs depend on problem-resolved node sets and variables, so they are
        # built in initialize().
        super().__init__(mesh, ListBC(), name=name, space=space)

    def __repr__(self):
        return (
            f"fedoo.constraint.MeanValueConstraint(variable={self.variable!r}, "
            f"value={self.value!r}, name={self.name!r})"
        )

    def _resolve_nodes(self):
        if self.node_set is None:
            nodes = np.arange(self.mesh.n_nodes)
        elif isinstance(self.node_set, str):
            nodes = np.asarray(self.mesh.node_sets[self.node_set], dtype=int)
        else:
            nodes = np.asarray(self.node_set, dtype=int)

        if nodes.ndim != 1:
            nodes = nodes.reshape(-1)
        if len(nodes) == 0:
            raise ValueError("The node set of a MeanValueConstraint is empty.")
        return nodes

    def _resolve_variables(self):
        if isinstance(self.variable, str):
            if self.variable in self.space.list_vectors():
                return list(self.space.get_vector(self.variable))
            return [self.variable]
        return list(self.variable)

    def _resolve_values(self, n_variables):
        if np.isscalar(self.value):
            return np.full(n_variables, self.value, dtype=float)

        values = np.asarray(self.value, dtype=float).reshape(-1)
        if len(values) != n_variables:
            raise ValueError(
                f"value has {len(values)} components but the constraint "
                f"involves {n_variables} variable(s)."
            )
        return values

    def _resolve_weights(self, nodes):
        if self.weights is None:
            return np.full(len(nodes), 1.0 / len(nodes))

        if isinstance(self.weights, str):
            if self.weights.lower() != "volume":
                raise ValueError(
                    f"weights={self.weights!r} unknown. "
                    'Use None, "volume" or an array of weights.'
                )
            nodal_weights = np.asarray(
                (
                    self.mesh._get_gaussian_quadrature_mat()
                    @ self.mesh._get_node2gausspoint_mat()
                ).sum(axis=0)
            ).ravel()
            weights = nodal_weights[nodes]
        else:
            weights = np.asarray(self.weights, dtype=float)
            if weights.ndim != 1 or len(weights) != len(nodes):
                raise ValueError(
                    "weights should be a 1D array with the same length as the "
                    f"node set (got shape {weights.shape}, expected "
                    f"({len(nodes)},))."
                )

        weight_sum = weights.sum()
        if abs(weight_sum) < 1e-15:
            raise ValueError("The sum of the weights should not be 0.")
        return weights / weight_sum

    def _collect_equations(self, pb):
        dofs, coefficients, _ = super()._collect_equations(pb)
        return dofs, coefficients, self._mean_values.copy()

    def initialize(self, pb):
        nodes = self._resolve_nodes()
        variables = self._resolve_variables()
        values = self._resolve_values(len(variables))
        self._mean_values = values
        weights = self._resolve_weights(nodes)

        self._nodes = nodes
        self._var_names = variables
        self._ranks = [self.space.variable_rank(var) for var in variables]
        self._weights = weights
        self._lm_names = [f"{self.name}_{variable}" for variable in variables]

        node_terms = [[int(node)] for node in nodes]
        factor_terms = [float(weight) for weight in weights]
        self.constraints = ListBC(
            [
                MPC(
                    node_terms,
                    [variable] * len(nodes),
                    factor_terms,
                    constant=value,
                )
                for variable, value in zip(variables, values)
            ]
        )
        self.multiplier_names = self._lm_names
        super().initialize(pb)
