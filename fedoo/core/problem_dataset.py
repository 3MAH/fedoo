"""A current-state DataSet view of a live Problem, with eigenmode navigation."""

import numpy as np

from fedoo.core.base import ProblemBase
from fedoo.core.dataset import DataSet


def _is_problem(value):
    """Recognize Problems created before a notebook reloaded the base module.

    Reloading replaces class identities while existing objects keep their
    original base classes. Restrict the fallback to Fedoo's ProblemBase in
    the actual inheritance tree, rather than accepting arbitrary objects
    with similarly named data methods.
    """
    return isinstance(value, ProblemBase) or any(
        base.__module__ == "fedoo.core.base" and base.__name__ == "ProblemBase"
        for base in type(value).__mro__
    )


class ProblemDataSet(DataSet):
    """Expose current Problem results through the DataSet interface.

    Parameters
    ----------
    problem : ProblemBase
        The live problem. No solve or assembly update is triggered.
    fields : str or sequence[str], optional
        Results requested through ``problem.get_results``. By default, nodal
        and global DOFs, external forces and available stress/strain fields
        are discovered. An explicit list is strict about missing results.
    assembly : AssemblyBase, optional
        Assembly used by ``get_results``; the problem assembly is the default.

    ``get_data`` refreshes from the current problem. Call ``refresh`` before
    directly inspecting the storage dictionaries. ``copy``/``deepcopy`` return
    detached DataSet snapshots, and FDH5 export writes one current state.
    Boundary conditions remain on ``problem`` and are not serialized as a
    solver restart. Viewer Reload refreshes this object without creating frames.
    Modal/buckling problems expose computed modes through ``n_iter`` and
    ``set_mode``; only the selected mode's fields are extracted at a time.
    """

    def __init__(self, problem, fields=None, assembly=None):
        if not _is_problem(problem):
            raise TypeError("ProblemDataSet requires a Problem object")
        self.problem = problem
        self.assembly = assembly
        self.fields = (
            None
            if fields is None
            else ([fields] if isinstance(fields, str) else list(fields))
        )
        self._reading_problem = False
        self._extra_fields = []
        self._mesh_signature = None
        super().__init__(problem.mesh)
        self.refresh()

    @property
    def n_modes(self):
        """Number of computed eigenmodes currently retained by the Problem."""
        modes = getattr(self.problem, "modes", None)
        if modes is None or not callable(getattr(self.problem, "set_mode", None)):
            return 0
        return len(modes)

    @property
    def n_iter(self):
        """Navigable modes, or one current state for other Problems."""
        return max(1, self.n_modes)

    @property
    def current_mode(self):
        """The Problem's selected zero-based mode index, if available."""
        return getattr(self.problem, "current_mode", None) if self.n_modes else None

    def set_mode(self, index):
        """Select a computed mode and refresh fields without an eigensolve.

        Uses the Problem's mode-selection API, including its normal derived
        field update. Selecting the already active mode preserves any custom
        scaling installed from Python. Export remains a current-mode snapshot.
        """
        if not isinstance(index, (int, np.integer)) or isinstance(
            index, (bool, np.bool_)
        ):
            raise TypeError("Mode index must be an integer")
        if not 0 <= index < self.n_modes:
            raise IndexError(f"Mode index {index} is out of range")
        if index != self.current_mode:
            self.problem.set_mode(int(index))
        return self.refresh()

    def _default_fields(self):
        problem = self.problem
        fields = list(problem.space.list_vectors()) + list(
            problem.space.list_variables()
        )
        fields += list(problem.global_dof._variable) + list(problem.global_dof._vector)
        if problem.get_A() is not None:
            fields += [f"Fext({name})" for name in problem.global_dof._variable]
            fields += [f"Fext({name})" for name in problem.global_dof._vector]
            fields.append("Fext")
        if self.assembly is not None or hasattr(problem, "assembly"):
            fields += ["Stress", "Strain"]
            assembly = self.assembly if self.assembly is not None else problem.assembly
            for block in getattr(assembly, "list_assembly", (assembly,)):
                for name, value in getattr(block, "sv", {}).items():
                    eligible = name in getattr(block, "sv_type", {}) or name in {
                        "Statev",
                        "Wm",
                        "BeamStress",
                        "BeamStrain",
                        "ShellStress",
                        "ShellStrain",
                        "DispGradient",
                        "Temp",
                    }
                    if (
                        eligible
                        and name != "TangentMatrix"
                        and not name.startswith("_")
                        and value is not None
                    ):
                        try:
                            array = np.asarray(value)
                        except ValueError:
                            continue
                        if 1 <= array.ndim <= 2 and array.dtype.kind in "biuf":
                            fields.append(name)
        return list(dict.fromkeys(fields + self._extra_fields))

    def refresh(self):
        """Refresh one state without solving or accumulating time frames."""
        previous = self._reading_problem
        self._reading_problem = True
        try:
            problem = self.problem
            fields = self._default_fields() if self.fields is None else self.fields
            if self.assembly is not None or hasattr(problem, "assembly"):
                options = dict(ignore_missing=self.fields is None)
                if self.assembly is not None:
                    options["assemb"] = self.assembly
                snapshot = problem.get_results(fields, **options)
            else:
                snapshot = DataSet(problem.mesh)
                for field in fields:
                    if field == "Fext":
                        value = problem.get_ext_forces(include_mpc=False)[
                            : problem.n_node_dof
                        ].reshape(problem.space.nvar, -1)
                        snapshot.node_data[field] = value
                    elif field.startswith("Fext(") and field.endswith(")"):
                        variable = field[5:-1]
                        snapshot.scalar_data[field] = problem.get_ext_forces(variable)
                    elif (
                        field in problem.global_dof._variable
                        or field in problem.global_dof._vector
                    ):
                        snapshot.scalar_data[field] = problem.get_dof_solution(field)
                    elif (
                        field in problem.space.list_variables()
                        or field in problem.space.list_vectors()
                    ):
                        snapshot.node_data[field] = problem.get_dof_solution(field)
                    else:
                        raise ValueError(f"Field '{field}' requires a problem assembly")
            blocks = getattr(snapshot.mesh, "submeshes", (snapshot.mesh,))
            signature = tuple(
                (id(block.nodes), id(block.elements), block.elm_type)
                for block in blocks
            )
            if signature == self._mesh_signature:
                snapshot.mesh.node_sets = self.mesh.node_sets
                old_blocks = getattr(self.mesh, "submeshes", (self.mesh,))
                for block, old_block in zip(blocks, old_blocks):
                    block.element_sets = old_block.element_sets
            self.mesh = snapshot.mesh
            self._mesh_signature = signature
            self.node_data = dict(snapshot.node_data)
            self.element_data = dict(snapshot.element_data)
            self.gausspoint_data = dict(snapshot.gausspoint_data)
            self.scalar_data = dict(snapshot.scalar_data)
            self.field_metadata = dict(snapshot.field_metadata)
            for name, value in self.node_data.items():
                if np.ndim(value) == 0:
                    components = (
                        len(problem.space.get_rank_vector(name))
                        if name in problem.space.list_vectors()
                        else 1
                    )
                    shape = (
                        (components, self.mesh.n_nodes)
                        if components > 1
                        else (self.mesh.n_nodes,)
                    )
                    self.node_data[name] = np.full(shape, value, dtype=float)
            if hasattr(problem, "time"):
                self.scalar_data["Time"] = float(problem.time)
            self._evaluate_scalar_expressions()
            return self
        finally:
            self._reading_problem = previous

    def get_data(self, field, *args, **kwargs):
        if self._reading_problem:
            return super().get_data(field, *args, **kwargs)
        self._reading_problem = True
        previous_fields = None if self.fields is None else list(self.fields)
        previous_extra = list(self._extra_fields)
        try:
            if (
                self.fields is None
                and field not in self.field_names()
                and field not in self.scalar_data
                and field not in self.scalar_expressions
                and field not in self.section_derived_fields()
            ):
                if field not in self._extra_fields:
                    self._extra_fields.append(field)
            if (
                self.fields is not None
                and field not in self.fields
                and field not in self.scalar_data
                and field not in self.scalar_expressions
                and field not in self.section_derived_fields()
                and field != "Time"
            ):
                self.fields.append(field)
            self.refresh()
            return super().get_data(field, *args, **kwargs)
        except Exception:
            self.fields = previous_fields
            self._extra_fields = previous_extra
            raise
        finally:
            self._reading_problem = False

    def copy(self):
        self.refresh()
        return DataSet.copy(self)

    def sync_problem_sets(self):
        """Publish viewer node sets to the live problem for boundary conditions."""
        self.problem.mesh.node_sets = self.mesh.node_sets
        assembly = (
            self.assembly
            if self.assembly is not None
            else getattr(self.problem, "assembly", None)
        )
        if assembly is not None:
            for block in getattr(assembly, "list_assembly", (assembly,)):
                block.mesh.node_sets = self.mesh.node_sets

    def deepcopy(self):
        self.refresh()
        return DataSet.deepcopy(self)

    def to_fdh5(self, *args, **kwargs):
        self.refresh()
        return super().to_fdh5(*args, **kwargs)
