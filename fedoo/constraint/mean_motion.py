"""Mean motion constraint."""

from numbers import Number

import numpy as np
from scipy.spatial import cKDTree
from simcoon import Rotation as SimRotation
from simcoon import dR_drotvec

from fedoo.core.boundary_conditions import BCBase, ListBC, MPC


_DISP_VECTOR = "Disp"
_MEAN_DISP_VECTOR = "MeanDisp"
_MEAN_ROT_VECTOR = "MeanRot"


class MeanMotion(BCBase):
    """Constraint defining selected global DOFs equal to a mean motion.

    The constraint projects the displacement field of selected nodes onto the
    rigid-body modes using a weighted least-squares fit. Selected fitted
    translations are exposed through the global vector ``MeanDisp`` and
    selected fitted rotations are exposed through the global vector ``MeanRot``.
    Prescribing these global DOFs with standard Dirichlet boundary conditions
    drives the selected face in a mean sense while still allowing local
    deformation.

    Parameters
    ----------
    node_set : str, int, array_like, Mesh or None, optional
        Nodes used to compute the mean motion. If a mesh is given, its
        elements are used to compute area/length nodal weights automatically.
        If ``None`` all problem mesh nodes are used.
    components : str or list[str]
        Mean-motion components exposed as global DOFs. Accepted aliases include
        ``"RotX"``, ``"MeanRotX"``, ``"DispZ"``, ``"MeanDispZ"``, ``"Rot"``,
        ``"MeanRot"``, ``"Disp"`` and ``"MeanDisp"``. Vector aliases expand to
        all components available in the current modeling dimension.
    weights : array_like or None, optional
        Nodal weights. By default, weights are area/length based when
        ``node_set`` is a mesh, otherwise all selected nodes have equal weight.
    center : int, array_like or None, optional
        Rotation center. If ``None``, the weighted centroid of the selected
        nodes is used. If an int is given, the corresponding problem mesh node
        is used.
    surface_mesh : Mesh or None, optional
        Surface mesh used to define both selected nodes and area/length weights.
        This is equivalent to passing the mesh as ``node_set``.
    normalize_weights : bool, default=True
        If True, normalize weights so that their sum is one.
    finite_rotation : bool or None, default=None
        If ``None``, finite rotation is enabled automatically when geometrical
        nonlinearity is active on the problem and at least one rotation
        component is selected. If False, rotations are small-rotation vector
        components and the constraint is linear. If True, rotations are finite
        rotation-vector components and the constraint is linearized at each
        Newton iteration using ``simcoon.dR_drotvec``.
    name : str, optional
        Constraint name.
    """

    def __init__(
        self,
        node_set=None,
        components=None,
        weights=None,
        center=None,
        surface_mesh=None,
        normalize_weights=True,
        finite_rotation=None,
        name="Mean motion",
    ):
        super().__init__(name)
        self.bc_type = "MeanMotion"
        if surface_mesh is None and _looks_like_mesh(node_set):
            surface_mesh = node_set
            node_set = None
        self.node_set = node_set
        self.components = components
        self.weights = weights
        self.center = center
        self.surface_mesh = surface_mesh
        self.normalize_weights = normalize_weights
        self.finite_rotation = finite_rotation

        self.nodes = None
        self._weights = None
        self._disp_variables = None
        self._mean_disp_variables = None
        self._mean_rot_variables = None
        self._full_mean_disp_variables = None
        self._full_mean_rot_variables = None
        self._full_mean_variables = None
        self._mean_variables = None
        self._mode_indices = None
        self._node_by_variable = None
        self._projection = None
        self._mode_matrix = None
        self._node_coords = None
        self._phys_dof_index = None
        self.node_disp = None
        self.node_rot = None

    @property
    def node_by_variable(self):
        """Map global variable names to their local global-DOF index."""
        return self._node_by_variable

    def str_condensed(self):
        if self.name == "":
            return f"{self.bc_type} -> {', '.join(self._mean_variables or [])}"
        return (
            f"{self.bc_type} (name = '{self.name}') -> "
            f"{', '.join(self._mean_variables or [])}"
        )

    def initialize(self, problem):
        self._disp_variables = self._get_disp_variables(problem)
        n_dim = len(self._disp_variables)

        if n_dim == 2:
            self._full_mean_disp_variables = [
                f"{_MEAN_DISP_VECTOR}X",
                f"{_MEAN_DISP_VECTOR}Y",
            ]
            self._full_mean_rot_variables = [f"{_MEAN_ROT_VECTOR}Z"]
        elif n_dim == 3:
            self._full_mean_disp_variables = [
                f"{_MEAN_DISP_VECTOR}X",
                f"{_MEAN_DISP_VECTOR}Y",
                f"{_MEAN_DISP_VECTOR}Z",
            ]
            self._full_mean_rot_variables = [
                f"{_MEAN_ROT_VECTOR}X",
                f"{_MEAN_ROT_VECTOR}Y",
                f"{_MEAN_ROT_VECTOR}Z",
            ]
        else:
            raise NotImplementedError("MeanMotion is implemented in 2D and 3D.")

        self.nodes, self._weights = self._get_nodes_and_weights(problem)
        self.center = self._get_center(problem)
        self._node_coords = problem.mesh.nodes[self.nodes, :n_dim]
        self._projection, self._mode_matrix = self._build_projection(problem)

        self._full_mean_variables = (
            self._full_mean_disp_variables + self._full_mean_rot_variables
        )
        normalized = self._normalize_components(n_dim)
        self._mean_disp_variables = [
            var for var in normalized if var in self._full_mean_disp_variables
        ]
        self._mean_rot_variables = [
            var for var in normalized if var in self._full_mean_rot_variables
        ]
        self._mean_variables = self._mean_disp_variables + self._mean_rot_variables
        self._mode_indices = [
            self._full_mean_variables.index(var) for var in self._mean_variables
        ]

        n_nodes = problem.mesh.n_nodes
        self._phys_dof_index = np.array(
            [
                problem.space.variable_rank(var) * n_nodes + node
                for node in self.nodes
                for var in self._disp_variables
            ],
            dtype=int,
        )

        has_rotation = bool(self._mean_rot_variables)
        if self.finite_rotation is None:
            self.finite_rotation = bool(problem.nlgeom and has_rotation)
        if self.finite_rotation:
            self._keep_at_end = True
            # finite rotation is nonlinear: the MPCs must be relinearized every
            # Newton iteration, independently of problem.nlgeom (a user may force
            # finite_rotation=True on a geometrically linear problem).
            self._update_during_inc = True

        self._node_by_variable = {}
        if self._mean_disp_variables:
            self.node_disp = problem.add_global_dof(
                self._mean_disp_variables, 1, vector_name=_MEAN_DISP_VECTOR
            )
            self._node_by_variable.update(
                {var: self.node_disp for var in self._mean_disp_variables}
            )
        if self._mean_rot_variables:
            self.node_rot = problem.add_global_dof(
                self._mean_rot_variables, 1, vector_name=_MEAN_ROT_VECTOR
            )
            self._node_by_variable.update(
                {var: self.node_rot for var in self._mean_rot_variables}
            )

    def generate(self, problem, t_fact=1, t_fact_old=None):
        res = ListBC()
        if self.finite_rotation:
            self._project_finite_dirichlet_mean_motion(problem, t_fact, t_fact_old)
            u0 = self._get_total_physical_values(problem)
            q0 = self._get_full_total_mean_values(
                problem, u0, controlled=self._finite_dirichlet_mask(problem)
            )
            coeff_u, coeff_q, residual = self._build_finite_incremental_linearization(
                problem, q0, u0
            )
            # Track the constraint out-of-balance so the NR loop does not report
            # convergence while the mean-motion fit is still unsatisfied.
            problem._bc_residual_norm = max(
                getattr(problem, "_bc_residual_norm", 0.0),
                float(np.linalg.norm(residual)),
            )
            used_slave_cols = set()
            for mode_index, mean_var in enumerate(self._mean_variables):
                res.append(
                    self._make_linearized_mpc(
                        coeff_u[mode_index],
                        coeff_q[mode_index],
                        residual[mode_index],
                        mean_var,
                        used_slave_cols,
                    )
                )
        else:
            used_slave_cols = set()
            for mode_index, mean_var in enumerate(self._mean_variables):
                res.append(
                    self._make_mpc(
                        self._mode_indices[mode_index], mean_var, used_slave_cols
                    )
                )

        res.initialize(problem)
        return res.generate(problem, t_fact, t_fact_old)

    def _normalize_components(self, n_dim):
        if self.components is None:
            raise ValueError("MeanMotion requires an explicit components argument.")

        if isinstance(self.components, str):
            components = [self.components]
        else:
            components = list(self.components)

        if len(components) == 0:
            raise ValueError("MeanMotion requires at least one component.")

        disp_suffixes = ["X", "Y", "Z"][:n_dim]
        rot_suffixes = ["Z"] if n_dim == 2 else ["X", "Y", "Z"]
        allowed = set(self._full_mean_variables)

        normalized = []
        for component in components:
            for variable in self._expand_component_alias(
                component, disp_suffixes, rot_suffixes
            ):
                if variable not in allowed:
                    raise ValueError(
                        f"Unknown MeanMotion component '{component}' for "
                        f"{n_dim}D problem."
                    )
                if variable not in normalized:
                    normalized.append(variable)

        return normalized

    def _expand_component_alias(self, component, disp_suffixes, rot_suffixes):
        vector_aliases = {
            "Disp": self._full_mean_disp_variables,
            "MeanDisp": self._full_mean_disp_variables,
            "Rot": self._full_mean_rot_variables,
            "MeanRot": self._full_mean_rot_variables,
        }
        if component in vector_aliases:
            return list(vector_aliases[component])

        for prefix, vector_name, suffixes in [
            ("Disp", _MEAN_DISP_VECTOR, disp_suffixes),
            ("MeanDisp", _MEAN_DISP_VECTOR, disp_suffixes),
            ("Rot", _MEAN_ROT_VECTOR, rot_suffixes),
            ("MeanRot", _MEAN_ROT_VECTOR, rot_suffixes),
        ]:
            if component.startswith(prefix):
                suffix = component[len(prefix) :]
                if suffix in suffixes:
                    return [f"{vector_name}{suffix}"]

        return [component]

    def _get_nodes_and_weights(self, problem):
        if self.surface_mesh is not None:
            nodes, weights = self._get_surface_nodes_and_weights(problem)
        elif self.node_set is None:
            nodes = np.arange(problem.mesh.n_nodes)
            weights = self._get_explicit_or_uniform_weights(len(nodes))
        elif isinstance(self.node_set, str):
            nodes = np.asarray(problem.mesh.node_sets[self.node_set], dtype=int).ravel()
            weights = self._get_explicit_or_uniform_weights(len(nodes))
        elif isinstance(self.node_set, Number):
            nodes = np.array([self.node_set], dtype=int)
            weights = self._get_explicit_or_uniform_weights(len(nodes))
        else:
            nodes = np.asarray(self.node_set, dtype=int).ravel()
            weights = self._get_explicit_or_uniform_weights(len(nodes))

        if len(nodes) == 0:
            raise ValueError("MeanMotion requires at least one node.")
        if not np.any(np.abs(weights) > 0):
            raise ValueError("At least one mean-rigid-motion weight must be non-zero.")

        if self.normalize_weights:
            weights_sum = np.sum(weights)
            if weights_sum == 0:
                raise ValueError("Cannot normalize weights with a zero sum.")
            weights = weights / weights_sum

        return nodes, weights

    def _get_explicit_or_uniform_weights(self, n_nodes):
        if self.weights is None:
            return np.ones(n_nodes, dtype=float)

        weights = np.asarray(self.weights, dtype=float).ravel()
        if len(weights) != n_nodes:
            raise ValueError("weights must have the same length as the selected nodes.")
        return weights

    def _get_surface_nodes_and_weights(self, problem):
        if self.weights is not None:
            raise ValueError(
                "weights should not be given when surface_mesh is used. Pass a "
                "node_set instead to provide explicit nodal weights."
            )

        surface_nodes = self._map_surface_nodes_to_problem(problem)
        elements = surface_nodes[np.asarray(self.surface_mesh.elements, dtype=int)]
        measures = self.surface_mesh.get_element_volumes()

        nodes_per_elm = elements.shape[1]
        weights = np.zeros(problem.mesh.n_nodes)
        np.add.at(
            weights,
            elements.ravel(),
            np.repeat(measures / nodes_per_elm, nodes_per_elm),
        )

        nodes = np.flatnonzero(weights)
        return nodes.astype(int), weights[nodes]

    def _map_surface_nodes_to_problem(self, problem):
        surface_nodes = np.asarray(self.surface_mesh.nodes)
        if surface_nodes.shape == problem.mesh.nodes.shape and np.allclose(
            surface_nodes, problem.mesh.nodes
        ):
            return np.arange(problem.mesh.n_nodes)

        distances, mapped_nodes = cKDTree(problem.mesh.nodes).query(surface_nodes)
        if not np.allclose(distances, 0.0):
            raise ValueError(
                "surface_mesh nodes could not be mapped to the problem mesh. "
                "Use a surface mesh extracted from the problem mesh or pass "
                "node_set and weights explicitly."
            )
        return mapped_nodes.astype(int)

    def _get_center(self, problem):
        if self.center is None:
            return np.average(
                problem.mesh.nodes[self.nodes], axis=0, weights=self._weights
            )
        if np.isscalar(self.center):
            return np.asarray(problem.mesh.nodes[int(self.center)])
        return np.asarray(self.center, dtype=float)

    def _build_projection(self, problem):
        n_dim = len(self._disp_variables)
        n_modes = n_dim + (1 if n_dim == 2 else 3)
        n_phys = len(self.nodes) * n_dim
        weighted_modes = np.zeros((n_modes, n_modes))
        rhs = np.zeros((n_modes, n_phys))
        mode_matrix = np.zeros((n_phys, n_modes))

        coords = problem.mesh.nodes[self.nodes, :n_dim] - self.center[:n_dim]
        for i, (coord, weight) in enumerate(zip(coords, self._weights)):
            b = _rigid_motion_block(coord, n_dim)
            row = slice(i * n_dim, (i + 1) * n_dim)
            mode_matrix[row, :] = b
            weighted_modes += weight * b.T @ b
            rhs[:, row] = weight * b.T

        rank = np.linalg.matrix_rank(weighted_modes)
        if rank < n_modes:
            raise ValueError(
                "Selected nodes cannot define all requested mean-motion modes. "
                f"Rank is {rank}, expected {n_modes}."
            )

        return np.linalg.solve(weighted_modes, rhs), mode_matrix

    def _get_current_mean_values(self, problem, t_fact, t_fact_old):
        sol = problem.get_dof_solution()
        if np.isscalar(sol) and sol == 0:
            values = np.zeros(len(self._mean_variables))
        else:
            values = np.array(
                [
                    sol[self._global_dof_index(problem, var)]
                    for var in self._mean_variables
                ]
            )

        if not (np.isscalar(problem._Xbc) and problem._Xbc == 0):
            values += np.array(
                [
                    problem._Xbc[self._global_dof_index(problem, var)]
                    for var in self._mean_variables
                ]
            )

        return values

    def _get_full_total_mean_values(self, problem, u0, controlled=None):
        q = self._fit_finite_mean_motion(problem, u0)
        if not self._mean_variables:
            return q

        if controlled is None:
            controlled = np.ones(len(self._mean_variables), dtype=bool)
        elif not np.any(controlled):
            return q

        sol = problem.get_dof_solution()
        if np.isscalar(sol) and sol == 0:
            selected = np.zeros(len(self._mean_variables))
        else:
            selected = np.array(
                [
                    sol[self._global_dof_index(problem, var)]
                    for var in self._mean_variables
                ]
            )

        if not (np.isscalar(problem._Xbc) and problem._Xbc == 0):
            selected += np.array(
                [
                    problem._Xbc[self._global_dof_index(problem, var)]
                    for var in self._mean_variables
                ]
            )

        for selected_index, is_controlled in enumerate(controlled):
            if is_controlled:
                q[self._mode_indices[selected_index]] = selected[selected_index]
        return q

    def _get_total_physical_values(self, problem):
        sol = problem.get_dof_solution()
        if np.isscalar(sol) and sol == 0:
            return np.zeros(len(self._phys_dof_index))
        return np.asarray(sol)[self._phys_dof_index]

    def _global_dof_index(self, problem, variable):
        return (
            problem.n_node_dof
            + problem.global_dof.indice_start(variable)
            + self._node_by_variable[variable]
        )

    def _build_finite_incremental_linearization(self, problem, q0, u0):
        n_dim = len(self._disp_variables)
        j_mat, pred = self._finite_tangent_and_prediction(problem, q0)
        weights = np.repeat(self._weights, n_dim)
        residual_full = (j_mat.T * weights) @ (pred - u0)
        coeff_u_full = -(j_mat.T * weights)
        coeff_q_full = (j_mat.T * weights) @ j_mat

        return (
            coeff_u_full[self._mode_indices],
            coeff_q_full[np.ix_(self._mode_indices, self._mode_indices)],
            residual_full[self._mode_indices],
        )

    def _project_finite_dirichlet_mean_motion(self, problem, t_fact, t_fact_old):
        if (
            not hasattr(problem, "_dU")
            or np.isscalar(problem._dU)
            or np.array_equal(problem._dU, 0)
        ):
            return

        controlled = self._finite_dirichlet_mask(problem)
        if not np.any(controlled):
            return

        u0 = self._get_total_physical_values(problem)
        q_fit = self._fit_finite_mean_motion(problem, u0)
        q_target = q_fit.copy()
        prescribed = self._get_current_mean_values(problem, t_fact, t_fact_old)
        for selected_index, is_controlled in enumerate(controlled):
            if is_controlled:
                q_target[self._mode_indices[selected_index]] = prescribed[
                    selected_index
                ]

        correction = self._finite_rigid_displacement(problem, q_target)
        correction -= self._finite_rigid_displacement(problem, q_fit)
        problem._dU[self._phys_dof_index] += correction

    def _finite_dirichlet_mask(self, problem):
        controlled = np.zeros(len(self._mean_variables), dtype=bool)

        for bc in problem.bc.list_all():
            if getattr(bc, "bc_type", None) != "Dirichlet":
                continue
            variable = getattr(bc, "variable_name", None)
            if variable not in self._mean_variables:
                continue
            controlled[self._mean_variables.index(variable)] = True

        return controlled

    def _fit_finite_mean_motion(self, problem, u0):
        n_dim = len(self._disp_variables)
        initial = problem.mesh.nodes[self.nodes, :n_dim]
        current = initial + u0.reshape(len(self.nodes), n_dim)

        mean_initial = np.average(initial, axis=0, weights=self._weights)
        mean_current = np.average(current, axis=0, weights=self._weights)
        initial_centered = initial - mean_initial
        current_centered = current - mean_current

        if n_dim == 2:
            cov = (initial_centered * self._weights[:, None]).T @ current_centered
            rotation_angle = np.arctan2(cov[0, 1] - cov[1, 0], cov[0, 0] + cov[1, 1])
            cos_angle = np.cos(rotation_angle)
            sin_angle = np.sin(rotation_angle)
            rotation = np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])
            trans = (
                mean_current
                - self.center[:2]
                - rotation @ (mean_initial - self.center[:2])
            )
            return np.array([trans[0], trans[1], rotation_angle])

        cov = (initial_centered * self._weights[:, None]).T @ current_centered
        u_svd, _, vt = np.linalg.svd(cov)
        rotation = vt.T @ u_svd.T
        if np.linalg.det(rotation) < 0:
            vt[-1] *= -1
            rotation = vt.T @ u_svd.T

        trans = (
            mean_current - self.center[:3] - rotation @ (mean_initial - self.center[:3])
        )
        rotvec = SimRotation.from_matrix(rotation).as_rotvec()
        return np.r_[trans, rotvec]

    def _finite_rigid_displacement(self, problem, q):
        _, pred = self._finite_tangent_and_prediction(problem, q)
        return pred

    def _build_finite_linearization(self, problem, q0, u0):
        j_mat, pred = self._finite_tangent_and_prediction(problem, q0)
        return self._build_finite_linearization_from_tangent(j_mat, pred, q0, u0)

    def _build_finite_linearization_from_tangent(self, j_mat, pred, q0, u0):
        n_dim = len(self._disp_variables)
        weights = np.repeat(self._weights, n_dim)
        residual = pred - u0

        coeff_u = -(j_mat.T * weights)
        coeff_q = (j_mat.T * weights) @ j_mat
        h0 = (j_mat.T * weights) @ residual
        constants = h0 - coeff_u @ u0 - coeff_q @ q0
        return coeff_u, coeff_q, constants

    def _finite_tangent_and_prediction(self, problem, q):
        # rotation and its derivative are node-independent, so the prediction
        # and Jacobian use broadcasted ops (called every Newton iteration).
        n_dim = len(self._disp_variables)
        n_modes = len(self._full_mean_variables)
        n_phys = len(self.nodes) * n_dim
        j_mat = np.zeros((n_phys, n_modes))

        if n_dim == 2:
            trans = q[:2]
            rotvec = np.array([0.0, 0.0, q[2]])
            rotation = SimRotation.from_rotvec(rotvec).as_matrix()[:2, :2]
            drot = dR_drotvec(rotvec)[:2, :2, 2]
            coords = problem.mesh.nodes[self.nodes, :2] - self.center[:2]
            pred = (trans + coords @ rotation.T - coords).ravel()
            j_mat[:, :2] = np.tile(np.eye(2), (len(coords), 1))
            j_mat[:, 2] = (coords @ drot.T).ravel()
            return j_mat, pred

        trans = q[:3]
        rotvec = q[3:]
        rotation = SimRotation.from_rotvec(rotvec).as_matrix()
        drot = dR_drotvec(rotvec)
        coords = problem.mesh.nodes[self.nodes, :3] - self.center[:3]
        pred = (trans + coords @ rotation.T - coords).ravel()
        j_mat[:, :3] = np.tile(np.eye(3), (len(coords), 1))
        j_mat[:, 3:] = np.einsum("klr,ml->mkr", drot, coords).reshape(n_phys, 3)
        return j_mat, pred

    def _get_disp_variables(self, problem):
        if _DISP_VECTOR in problem.space.list_vectors():
            return [
                problem.space.variable_name(var_rank)
                for var_rank in problem.space.get_rank_vector(_DISP_VECTOR)
            ]

        if _DISP_VECTOR in problem.space.list_variables():
            return [_DISP_VECTOR]

        raise ValueError(f"Variable or vector '{_DISP_VECTOR}' doesn't exist.")

    def _make_mpc(self, row_index, mean_var, used_slave_cols=None):
        # ``row_index`` indexes ``_projection`` in the full rigid-body-mode
        # order; callers map a selected-subset index through ``_mode_indices``.
        if used_slave_cols is None:
            used_slave_cols = set()
        coeffs = self._projection[row_index]
        slave_col = self._select_physical_slave(coeffs, used_slave_cols, mean_var)
        used_slave_cols.add(slave_col)
        if coeffs[slave_col] == 0:
            raise ValueError(f"Mean-motion mode '{mean_var}' has no physical support.")

        n_dim = len(self._disp_variables)
        slave_node_index, slave_var_index = divmod(slave_col, n_dim)
        node_sets = [
            [int(self.nodes[slave_node_index])],
            [self._node_by_variable[mean_var]],
        ]
        variables = [self._disp_variables[slave_var_index], mean_var]
        factors = [[-float(coeffs[slave_col])], [1.0]]

        for col, coeff in enumerate(coeffs):
            if col == slave_col or coeff == 0:
                continue
            node_index, var_index = divmod(col, n_dim)
            node_sets.append([int(self.nodes[node_index])])
            variables.append(self._disp_variables[var_index])
            factors.append([-float(coeff)])

        return MPC(node_sets, variables, factors)

    def _make_linearized_mpc(
        self, coeff_u, coeff_q, constant, mean_var, used_slave_cols
    ):
        slave_col = self._select_physical_slave(coeff_u, used_slave_cols, mean_var)
        used_slave_cols.add(slave_col)

        n_dim = len(self._disp_variables)
        slave_node_index, slave_var_index = divmod(slave_col, n_dim)
        node_sets = [[int(self.nodes[slave_node_index])]]
        variables = [self._disp_variables[slave_var_index]]
        factors = [float(coeff_u[slave_col])]

        for i, coeff in enumerate(coeff_q):
            if coeff == 0:
                continue
            node_sets.append([self._node_by_variable[self._mean_variables[i]]])
            variables.append(self._mean_variables[i])
            factors.append(float(coeff))

        for col, coeff in enumerate(coeff_u):
            if col == slave_col or coeff == 0:
                continue
            node_index, var_index = divmod(col, n_dim)
            node_sets.append([int(self.nodes[node_index])])
            variables.append(self._disp_variables[var_index])
            factors.append(float(coeff))

        return MPC(node_sets, variables, factors, constant=float(constant))

    def _select_physical_slave(self, coeff_u, used_slave_cols, mean_var):
        abs_coeff = np.abs(coeff_u)
        available = np.ones(len(abs_coeff), dtype=bool)
        if used_slave_cols:
            available[list(used_slave_cols)] = False
        available &= abs_coeff > 0

        if np.any(available):
            max_coeff = np.max(abs_coeff[available])
            candidate_cols = np.flatnonzero(
                available & np.isclose(abs_coeff, max_coeff)
            )
            n_dim = len(self._disp_variables)
            candidate_node_indices = candidate_cols // n_dim
            distances = np.linalg.norm(
                self._node_coords[candidate_node_indices] - self.center[:n_dim],
                axis=1,
            )
            return int(candidate_cols[np.argmin(distances)])

        raise ValueError(f"Mean-motion mode '{mean_var}' has no physical support.")


def _looks_like_mesh(value):
    return (
        hasattr(value, "nodes")
        and hasattr(value, "elements")
        and hasattr(value, "elm_type")
    )


def _rigid_motion_block(coord, n_dim):
    if n_dim == 2:
        x, y = coord
        return np.array([[1.0, 0.0, -y], [0.0, 1.0, x]])

    x, y, z = coord
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0, z, -y],
            [0.0, 1.0, 0.0, -z, 0.0, x],
            [0.0, 0.0, 1.0, y, -x, 0.0],
        ]
    )
