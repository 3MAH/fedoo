"""Generic enforcement of MPC-generated constraints with Lagrange multipliers."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy import sparse

from fedoo.core.base import AssemblyBase
from fedoo.core.boundary_conditions import BCBase, ListBC, MPC
from fedoo.core.modelingspace import ModelingSpace


class LagrangeMultiplierAssembly(AssemblyBase):
    """Enforce MPC-generated linear constraints with Lagrange multipliers.

    The wrapped object may be an MPC, a ListBC, any BC object whose generate
    method produces only MPC leaves, or an iterable of such objects. Wrapped
    constraints must not also be added to the problem boundary conditions,
    since that would enforce them a second time through elimination.

    Parameters
    ----------
    mesh: fedoo.Mesh
        Mesh associated with the constraint. It should share the nodes of the
        assembly with which this constraint is summed.
    constraints: MPC, ListBC, BC object or iterable of BC objects
        Constraint(s) to enforce. Any wrapped object must generate only
        :py:class:`fedoo.MPC` leaves.
    name: str, default = "LagrangeMultiplier"
        Assembly name. It is also the name of the global vector that gathers
        the Lagrange-multiplier DOFs.
    multiplier_names: list of str or None, default = None
        Names of the multiplier DOFs, one per generated equation. If None, the
        multipliers are gathered as a single global vector named ``name``.
    space: ModelingSpace, optional
        Modeling space. If None, the active modeling space is used.

    Notes
    -----
    The multipliers give the system a zero diagonal block and an indefinite
    saddle-point structure, so a direct solver is required.
    """

    def __init__(
        self,
        mesh,
        constraints,
        name="LagrangeMultiplier",
        multiplier_names=None,
        space: ModelingSpace | None = None,
    ):
        if space is None:
            space = ModelingSpace.get_active()
        self.mesh = mesh
        super().__init__(name, space)

        if isinstance(constraints, BCBase):
            self.constraints = constraints
        elif isinstance(constraints, Iterable) and not isinstance(
            constraints, (str, bytes)
        ):
            self.constraints = ListBC(list(constraints))
        else:
            raise TypeError(
                "constraints should be a BC object or an iterable of BC objects."
            )

        self.multiplier_names = multiplier_names
        self._pb = None
        self._registered_pbs = []
        self._equation_dofs = None
        self._equation_coefficients = None

    def __repr__(self):
        return (
            "fedoo.LagrangeMultiplierAssembly("
            f"constraints={self.constraints!r}, name={self.name!r})"
        )

    @staticmethod
    def _time_factors(pb):
        try:
            return pb.t_fact, pb.t_fact_old
        except AttributeError:
            return 1, None

    def _collect_equations(self, pb):
        t_fact, t_fact_old = self._time_factors(pb)
        # Non-incremental constraints express the LM residual with the total
        # solution, not its increment, so they are generated with no previous
        # time factor.
        gen_t_fact_old = t_fact_old if self._incremental_equations else None
        generated = list(self.constraints.generate(pb, t_fact, gen_t_fact_old))
        if not generated:
            raise ValueError(
                "The wrapped boundary-condition object generated no constraints."
            )

        equation_dofs = []
        equation_coefficients = []
        values = []
        for constraint in generated:
            if not isinstance(constraint, MPC):
                raise TypeError(
                    "LagrangeMultiplierAssembly only accepts BC objects whose "
                    f"generated leaves are MPCs; got {type(constraint).__name__}."
                )
            dofs, coefficients, current_values = constraint.get_generated_equations()
            for dof_row, coefficient_row, value in zip(
                dofs, coefficients, current_values
            ):
                equation_dofs.append(np.asarray(dof_row, dtype=int))
                equation_coefficients.append(np.asarray(coefficient_row, dtype=float))
                values.append(value)

        return equation_dofs, equation_coefficients, np.asarray(values, dtype=float)

    def initialize(self, pb):
        self._pb = pb
        self.constraints.register_global_dofs(pb)
        self.constraints.initialize(pb)
        self._incremental_equations = bool(
            getattr(self.constraints, "_update_during_inc", False)
        )
        if np.isscalar(pb._Xbc):
            # Some MPC generators (for example RigidTie) inspect prescribed
            # driver values while generating their equations. Boundary
            # conditions have not been applied yet during assembly setup.
            pb._Xbc = np.zeros(pb.n_dof)
        dofs, coefficients, _ = self._collect_equations(pb)
        n_constraints = len(dofs)

        if self.multiplier_names is None:
            lm_names = None
            lm_indices = np.arange(n_constraints, dtype=int)
        else:
            lm_names = list(self.multiplier_names)
            if len(lm_names) != n_constraints:
                raise ValueError(
                    f"multiplier_names contains {len(lm_names)} names but the "
                    f"wrapped object generates {n_constraints} constraints."
                )
            lm_indices = None

        if not any(pb is registered for registered in self._registered_pbs):
            name_conflict = self.name in pb._global_dof._vector
            if lm_names is None:
                name_conflict |= self.name in pb._global_dof
            else:
                name_conflict |= any(lm_name in pb._global_dof for lm_name in lm_names)
            if name_conflict:
                raise NameError(
                    f"A global DOF associated with '{self.name}' already "
                    "exists in the problem. Use another assembly name."
                )
            if lm_names is None:
                pb.add_global_dof(self.name, n_constraints)
            else:
                pb.add_global_dof(lm_names, 1, vector_name=self.name)
            self._registered_pbs.append(pb)

        self._lm_names = lm_names
        self._lm_indices = lm_indices
        self._n_constraints = n_constraints
        self._equation_dofs = dofs
        self._equation_coefficients = coefficients
        self.delete_global_mat()

    def _register_global_dofs(self, pb):
        self.constraints.register_global_dofs(pb)

    def _lagrange_dofs(self, pb):
        if self._lm_names is None:
            return (
                pb.n_node_dof
                + pb._global_dof.indice_start(self.name)
                + self._lm_indices
            )
        return np.asarray(
            [
                pb.n_node_dof + pb._global_dof.indice_start(name)
                for name in self._lm_names
            ],
            dtype=int,
        )

    def _equations_changed(self, dofs, coefficients):
        if self._equation_dofs is None or len(dofs) != len(self._equation_dofs):
            return True
        return any(
            not np.array_equal(new_dofs, old_dofs)
            or not np.array_equal(new_coefficients, old_coefficients)
            for new_dofs, old_dofs, new_coefficients, old_coefficients in zip(
                dofs,
                self._equation_dofs,
                coefficients,
                self._equation_coefficients,
            )
        )

    def _assemble_matrix(self, pb, n_dof, dofs, coefficients):
        lm_dofs = self._lagrange_dofs(pb)
        n_entries = 2 * sum(len(row) for row in dofs)
        rows = np.empty(n_entries, dtype=int)
        cols = np.empty(n_entries, dtype=int)
        data = np.empty(n_entries, dtype=float)

        start = 0
        for lm_dof, dof_row, coefficient_row in zip(lm_dofs, dofs, coefficients):
            n_terms = len(dof_row)
            stop = start + n_terms
            rows[start:stop] = lm_dof
            cols[start:stop] = dof_row
            data[start:stop] = coefficient_row

            rows[stop : stop + n_terms] = dof_row
            cols[stop : stop + n_terms] = lm_dof
            data[stop : stop + n_terms] = coefficient_row
            start = stop + n_terms

        return sparse.csr_matrix((data, (rows, cols)), shape=(n_dof, n_dof))

    def _assemble_vector(self, pb, n_dof, values):
        vector = np.zeros(n_dof)
        lm_dofs = self._lagrange_dofs(pb)
        vector[lm_dofs] = values

        solution = pb.get_dof_solution()
        if np.isscalar(solution) and solution == 0:
            return vector
        if len(solution) < n_dof:
            solution = np.hstack((solution, np.zeros(n_dof - len(solution))))
        vector -= self.global_matrix @ solution
        if self._incremental_equations:
            vector[lm_dofs] = values
        return vector

    def assemble_global_mat(self, compute="all"):
        if compute == "none":
            return
        if self._pb is None:
            raise RuntimeError(
                "LagrangeMultiplierAssembly cannot be assembled before "
                "problem initialization."
            )

        pb = self._pb
        n_dof = pb.n_dof
        dofs, coefficients, values = self._collect_equations(pb)
        if len(dofs) != self._n_constraints:
            raise RuntimeError(
                "The number of generated MPC equations changed after "
                "initialization; multiplier DOFs cannot be resized safely."
            )

        if (
            self._equations_changed(dofs, coefficients)
            or self.global_matrix is None
            or self.global_matrix.shape != (n_dof, n_dof)
        ):
            self.global_matrix = self._assemble_matrix(pb, n_dof, dofs, coefficients)

        self._equation_dofs = dofs
        self._equation_coefficients = coefficients

        if compute != "matrix":
            self.global_vector = self._assemble_vector(pb, n_dof, values)

    def update(self, pb, compute="all"):
        self._pb = pb
        self.assemble_global_mat(compute)

    def set_start(self, pb):
        self._pb = pb
        if self.global_matrix is not None:
            self.assemble_global_mat("all")

    def to_start(self, pb):
        self._pb = pb
        if self.global_matrix is not None:
            self.assemble_global_mat("all")

    def reset(self):
        self.delete_global_mat()
        self._equation_dofs = None
        self._equation_coefficients = None
