"""Define a pressure constraint over a surface."""

from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np
from fedoo.core.assembly import Assembly
from fedoo.core.boundary_conditions import BCBase
from fedoo.weakform.distributed_load import ExternalPressure, DistributedLoad
from fedoo.core.mesh import Mesh
from fedoo.mesh.functions import extract_surface

if TYPE_CHECKING:
    from fedoo.core.base import ProblemBase


class _AssemblyNeumannBC(BCBase):
    """Neumann boundary condition generated from a load assembly.

    The wrapped assembly computes equivalent nodal load reference vectors.
    Fixed-geometry loads cache those vectors, then apply the time factor without
    reassembling. Follower loads refresh the reference vectors when generated.
    """

    def __init__(self, assembly: Assembly, name: str = "", time_func=None):
        BCBase.__init__(self, name)
        self.bc_type = "Neumann"
        self.assembly = assembly
        self._start_value_default = 0
        self.start_value = self._start_value_default
        self.value = 0
        if time_func is None:

            def time_func(t_fact):
                return t_fact

        self.time_func = time_func
        self._assembly_vector = None

    def str_condensed(self):
        """Return a condensed one line str describing the object."""
        if self.name == "":
            return "Neumann -> assembly load"
        return "Neumann (name = '{}') -> assembly load".format(self.name)

    def initialize(self, problem: ProblemBase):
        if self.assembly.mesh.n_nodes != problem.mesh.n_nodes:
            raise ValueError(
                "Assembly-based Neumann loads must share the problem mesh nodes."
            )

        self.assembly.initialize(problem)
        self._update_during_inc = bool(self.assembly._nlgeom)
        self._assembly_vector = None
        self.value = 0
        self.start_value = self._start_value_default

    def _get_factor(self, t_fact=1, t_fact_old=None):
        return self.time_func(t_fact)

    def get_value(self, t_fact=1, t_fact_old=None):
        factor = self._get_factor(t_fact, t_fact_old)
        if factor == 0:
            return self.start_value
        elif self.start_value is None:
            return factor * self.value
        else:
            return factor * (self.value - self.start_value) + self.start_value

    def get_true_value(self, t_fact=1, t_fact_old=None):
        return self.get_value(t_fact, t_fact_old)

    def _format_load_vector(self, problem: ProblemBase, value):
        if np.isscalar(value) and value == 0:
            return 0
        if problem.n_global_dof and len(value) < problem.n_dof:
            value = np.pad(value, (0, problem.n_dof - len(value)))
        return value.copy() if hasattr(value, "copy") else value

    def _assemble_load_vector(self, problem: ProblemBase, load_factor):
        self.assembly.set_start(problem, t_fact=load_factor)
        if self._update_during_inc:
            self.assembly.update(problem, compute="vector")
        else:
            self.assembly.assemble_global_mat(compute="vector")
        return self._format_load_vector(
            problem,
            self.assembly.current.get_global_vector(),
        )

    def _has_initial_load(self):
        return any(
            getattr(self.assembly, attr, None) is not None
            for attr in ("initial_pressure", "initial_force")
        )

    def _refresh_reference_values(self, problem: ProblemBase):
        if self._has_initial_load():
            self.start_value = self._assemble_load_vector(problem, 0)
        else:
            self.start_value = 0
        self.value = self._assemble_load_vector(problem, 1)
        self._assembly_vector = (
            self.value.copy() if hasattr(self.value, "copy") else self.value
        )

    def _format_current_value(self, problem: ProblemBase, value):
        if np.isscalar(value) and value == 0:
            self._dof_index = np.array([], dtype=int)
            self._current_value = np.array([])
            return

        self._dof_index = np.arange(problem.n_dof, dtype=int)
        self._current_value = value

    def generate(self, problem: ProblemBase, t_fact=1, t_fact_old=None):
        if self._update_during_inc or self._assembly_vector is None:
            self._refresh_reference_values(problem)

        self._format_current_value(problem, self.get_value(t_fact, t_fact_old))
        return [self]


class Pressure(Assembly):
    """Pressure load.

    Assembly object that define a surface pressure.
    Can be initialized with the following constructors:

        * the Pressure constructor describes bellow
        * :py:meth:`Pressure.from_nodes` to create the Assembly from a
          volume mesh and a set of nodes defining the surface.
        * :py:meth:`Pressure.from_elements` to create the Assembly from the
          volume mesh and a set of elements defining the surface.

    Parameters
    ----------
    surface_mesh: Mesh
        Mesh over which the pressure will be appplied. The mesh is
        assumed to be a surface mesh
        (ie 2d mesh for 3d problems or 1d mesh for 2d problems).
    pressure: float, array
        If float constant pressure value. If array, could be
        interpreted as gauss point, element or node
        values depending on the array dimension. In case the value type
        is confusing, gauss points will be choosen.
    initial_pressure: float, array, optional
        Initial value of the pressure.
        Used only to define the initial pressure condition
        for non linear problems.
    nlgeom: bool, str in {'UL', 'TL'}, optional
        If True, the geometrical non linearities are activate when used in the
        context of NonLinearProblems (default updated lagrangian method)
        such as :mod:`fedoo.problem.NonLinearStatic` or
        :mod:`fedoo.problem.NonLinearNewmark`
        If nlgeom == 'UL' the updated lagrangian method is used (same as True)
        If nlgeom == 'TL' the total lagrangian method is used
        If not defined, the problem.nlgeom attribute is used instead.
    name: str, optional
        Name of the created assembly.
    time_func: callable, optional
        Function that gives the temporal evolution of the pressure when the
        assembly is converted to a Neumann boundary condition. By default, a
        linear evolution is considered.

    Notes
    -----
    Pressure is an assembly and can still be combined directly with another
    assembly, for instance ``fd.problem.Linear(solid_assembly + pressure)``.
    It can also be used as an external Neumann boundary condition with
    ``pb.bc.add(pressure)`` or ``pb.bc.add(pressure.as_neumann())``. This form
    is useful for nonlinear problems because the equivalent nodal pressure is
    included in the external load vector and in residual normalization. When
    geometrical nonlinearities are active, the follower load is updated during
    Newton iterations.

    Example
    -------

    Apply a uniform pressure around a cube.

    .. code-block:: python

        import fedoo as fd

        fd.ModelingSpace('3D')
        material = fd.constitutivelaw.ElasticIsotrop(200e3, 0.3)

        mesh = fd.mesh.box_mesh()
        surface_mesh = fd.mesh.extract_surface(mesh)

        wf = fd.weakform.StressEquilibrium(material)
        solid_assembly = fd.Assembly.create(wf, mesh)
        pressure = fd.constraint.Pressure(surface_mesh, 1000, nlgeom=False)

        # define a problem from the solid and pressure assemblies
        pb = fd.problem.Linear(solid_assembly+pressure)
        pb.solve()

        # or add the same pressure as an external Neumann BC
        pb = fd.problem.NonLinear(solid_assembly, nlgeom=True)
        pb.bc.add(pressure)

        pb.get_results(solid_assembly,'Stress').plot('Stress', 'XX')
    """

    def __init__(
        self,
        surface_mesh: Mesh,
        pressure: float | np.ndarray,
        initial_pressure: float | np.ndarray | None = None,
        nlgeom: bool | None = None,
        name: str = "",
        time_func=None,
    ):
        self.pressure = pressure
        self.initial_pressure = initial_pressure
        self.nlgeom = nlgeom
        self.time_func = time_func
        wf = ExternalPressure(self.pressure, nlgeom=self.nlgeom)
        Assembly.__init__(self, wf, surface_mesh, name=name)
        if nlgeom == "TL":
            raise NotImplementedError("TL not implemented for distributed loads")

    def set_start(self, pb: ProblemBase, t_fact: float | None = None):
        """Start a new time increment."""
        if t_fact is None:
            t_fact = pb.t_fact

        if self.initial_pressure is None:
            self.weakform.pressure = t_fact * self.pressure
        else:
            self.weakform.pressure = (
                t_fact * (self.pressure - self.initial_pressure) + self.initial_pressure
            )

    def to_start(self, pb: ProblemBase):
        """Reset the assembly to the beginning of the time iteration."""
        self.set_start(pb)

    def as_neumann(self, name: str = "", time_func=None):
        """Return this pressure load as a Neumann boundary condition."""
        if time_func is None:
            time_func = self.time_func
        return _AssemblyNeumannBC(self, name, time_func=time_func)

    @staticmethod
    def from_nodes(
        mesh: Mesh,
        node_set: np.typing.ArrayLike[int] | str,
        pressure: float | np.ndarray,
        initial_pressure: float | np.ndarray | None = None,
        nlgeom: bool | None = None,
        name: str = "",
        time_func=None,
    ):
        """Create a pressure assembly from a node set.

        This constructor automatically extact the surface mesh
        from a node set.

        See :py:class:`Pressure` for more details on the parameters.
        """
        surface_mesh = extract_surface(mesh, node_set=node_set)
        return Pressure(
            surface_mesh, pressure, initial_pressure, nlgeom, name, time_func
        )

    @staticmethod
    def from_elements(
        mesh: Mesh,
        element_set: np.typing.ArrayLike[int] | str,
        pressure: float | np.ndarray,
        initial_pressure: float | np.ndarray | None = None,
        nlgeom: bool | None = None,
        name: str = "",
        time_func=None,
    ):
        """Create a pressure assembly from an element set.

        This constructor automatically extact the surface mesh
        from an element set.

        See :py:class:`Pressure` for more details on the parameters.
        """
        surface_mesh = extract_surface(mesh, element_set=element_set)
        return Pressure(
            surface_mesh, pressure, initial_pressure, nlgeom, name, time_func
        )


class DistributedForce(Assembly):
    """Distributed force (e.g gravity load).

    Assembly object that define a distributed force with fixed orientation.
    The physical nature of the force depend on the geometry dimension.

        * 2d problem with 1d mesh: line load for beams or
          surface force/stress for 2d plane element
        * 2d problem with 2d mesh: volume force
        * 3d problem with 1d mesh: line load
        * 3d problem with 2d mesh: surface force or stress
        * 3d porblem with 3d mesh: volume force

    Parameters
    ----------
    mesh: Mesh
        Mesh over which the force will be appplied.
    force: list or array with len(force)==3
        force[i] is the force along the ith dimension.
        if force[i] is an array, it is interpreted
        as gauss point, element or node values depending
        on the array dimension. In case the value type
        is confusing, gauss points will be choosen.
    initial_force: list or array with len(force)==3
        Initial value of the force.
        Used only to define the initial force condition
        for non linear problems.
    nlgeom: bool, str in {'UL', 'TL'}
        If True, the geometrical non linearities are activate when used in the
        context of NonLinearProblems (default updated lagrangian method)
        such as :mod:`fedoo.problem.NonLinearStatic` or
        :mod:`fedoo.problem.NonLinearNewmark`
        If nlgeom == 'UL' the updated lagrangian method is used (same as True)
        If nlgeom == 'TL' the total lagrangian method is used
    name: str, optional
        Name of the created assembly.
    time_func: callable, optional
        Function that gives the temporal evolution of the distributed load when
        the assembly is converted to a Neumann boundary condition. By default,
        a linear evolution is considered.

    Notes
    -----
    DistributedForce is an assembly and can still be combined directly with
    another assembly, for instance
    ``fd.problem.Linear(solid_assembly + volume_force)``. It can also be used
    as an external Neumann boundary condition with ``pb.bc.add(volume_force)``
    or ``pb.bc.add(volume_force.as_neumann())``. This form is useful for
    nonlinear problems because the equivalent nodal force is included in the
    external load vector and in residual normalization. When geometrical
    nonlinearities are active, the load is updated during Newton iterations.

    Example
    -------

    Apply a volume force on a cube.

    .. code-block:: python

        import fedoo as fd

        fd.ModelingSpace('3D')
        material = fd.constitutivelaw.ElasticIsotrop(200e3, 0.3)

        mesh = fd.mesh.box_mesh()

        wf = fd.weakform.StressEquilibrium(material)
        solid_assembly = fd.Assembly.create(wf, mesh)
        volume_force = fd.constraint.DistributedForce(
            mesh, [0,0,-1000], nlgeom=False)

        # define a problem from the solid and volume-force assemblies
        pb = fd.problem.Linear(solid_assembly+volume_force)
        pb.bc.add('Dirichlet', 'bottom', 'Disp', 0)
        pb.solve()

        # or add the same force as an external Neumann BC
        pb = fd.problem.NonLinear(solid_assembly, nlgeom=True)
        pb.bc.add(volume_force)

        pb.get_results(solid_assembly,'Stress').plot('Stress', 'XX', 'Node')
    """

    def __init__(
        self,
        mesh: Mesh,
        force: list | np.typing.ArrayLike[float],
        initial_force: np.typing.ArrayLike[float] | None = None,
        nlgeom: bool | None = None,
        name: str = "",
        time_func=None,
    ):
        self.force = force
        if initial_force is not None:
            self.initial_force = initial_force
        else:
            self.initial_force = None
        self.nlgeom = nlgeom
        self.time_func = time_func
        wf = DistributedLoad(self.force, nlgeom=self.nlgeom)
        Assembly.__init__(self, wf, mesh, name=name)

    def set_start(self, pb: ProblemBase, t_fact: float | None = None):
        """Start a new time increment."""
        if t_fact is None:
            t_fact = pb.t_fact

        if self.initial_force is None:
            if isinstance(self.force, np.ndarray):
                self.weakform.distributed_force = t_fact * self.force
            else:
                self.weakform.distributed_force = [t_fact * f for f in self.force]
        else:
            if isinstance(self.force, np.ndarray) and isinstance(
                self.initial_force, np.ndarray
            ):
                self.weakform.distributed_force = (
                    t_fact * (self.force - self.initial_force) + self.initial_force
                )
            else:
                self.weakform.distributed_force = [
                    t_fact * (f - self.initial_force[i]) + self.initial_force[i]
                    for i, f in enumerate(self.force)
                ]

    def to_start(self, pb: ProblemBase):
        """Reset the assembly to the beginning of the time iteration."""
        self.set_start(pb)

    def as_neumann(self, name: str = "", time_func=None):
        """Return this distributed load as a Neumann boundary condition."""
        if time_func is None:
            time_func = self.time_func
        return _AssemblyNeumannBC(self, name, time_func=time_func)


class SurfaceForce(DistributedForce):
    """Surface stress with a fixed orientation.

    Same as distributed load but the the mesh is supposed to be
    a surface mesh.
    The surface mesh can be extracted from a volume mesh using the
    from_nodes or from_elements constructors.
    """

    @staticmethod
    def from_nodes(
        mesh: Mesh,
        node_set: np.typing.ArrayLike[int] | str,
        force: np.typing.ArrayLike[float],
        initial_force: np.typing.ArrayLike[float] | None = None,
        nlgeom: bool | None = None,
        name: str = "",
        time_func=None,
    ):
        """Create a SurfaceForce assembly from an node set.

        This constructor automatically extact the surface mesh
        from an node set.

        See :py:class:`SurfaceForce` for more details on the parameters.
        """
        surface_mesh = extract_surface(mesh, node_set=node_set)
        return DistributedForce(
            surface_mesh, force, initial_force, nlgeom, name, time_func
        )

    @staticmethod
    def from_elements(
        mesh: Mesh,
        element_set: np.typing.ArrayLike[int] | str,
        force: np.typing.ArrayLike[float],
        initial_force: np.typing.ArrayLike[float] | None = None,
        nlgeom: bool | None = None,
        name: str = "",
        time_func=None,
    ):
        """Create a SurfaceForce assembly from an element set.

        This constructor automatically extact the surface mesh
        from an element set.

        See :py:class:`SurfaceForce` for more details on the parameters.
        """
        surface_mesh = extract_surface(mesh, element_set=element_set)
        return DistributedForce(
            surface_mesh, force, initial_force, nlgeom, name, time_func
        )
