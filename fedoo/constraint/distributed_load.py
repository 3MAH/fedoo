"""Define a pressure constraint over a surface."""

from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np
from fedoo.core.assembly import Assembly
from fedoo.weakform.distributed_load import ExternalPressure, DistributedLoad
from fedoo.core.mesh import Mesh
from fedoo.mesh.functions import extract_surface

if TYPE_CHECKING:
    from fedoo.core.base import ProblemBase


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

        pb.get_results(solid_assembly,'Stress').plot('Stress', 'XX')
    """

    def __init__(
        self,
        surface_mesh: Mesh,
        pressure: float | np.ndarray,
        initial_pressure: float | np.ndarray | None = None,
        nlgeom: bool | None = None,
        name: str = "",
    ):
        self.pressure = pressure
        self.initial_pressure = initial_pressure
        self.nlgeom = nlgeom
        wf = ExternalPressure(self.pressure, nlgeom=self.nlgeom)
        Assembly.__init__(self, wf, surface_mesh, name=name)
        if nlgeom == "TL":
            raise NotImplementedError("TL not implemented for distributed loads")

    def set_start(self, pb: ProblemBase):
        """Start a new time increment."""
        if self.initial_pressure is None:
            self.weakform.pressure = pb.t_fact * self.pressure
        else:
            self.weakform.pressure = (
                pb.t_fact * (self.pressure - self.initial_pressure)
                + self.initial_pressure
            )

    @staticmethod
    def from_nodes(
        mesh: Mesh,
        node_set: np.typing.ArrayLike[int] | str,
        pressure: float | np.ndarray,
        initial_pressure: float | np.ndarray | None = None,
        nlgeom: bool | None = None,
        name: str = "",
    ):
        """Create a pressure assembly from a node set.

        This constructor automatically extact the surface mesh
        from a node set.

        See :py:class:`Pressure` for more details on the parameters.
        """
        surface_mesh = extract_surface(mesh, node_set=node_set)
        return Pressure(surface_mesh, pressure, initial_pressure, nlgeom, name)

    @staticmethod
    def from_elements(
        mesh: Mesh,
        element_set: np.typing.ArrayLike[int] | str,
        pressure: float | np.ndarray,
        initial_pressure: float | np.ndarray | None = None,
        nlgeom: bool | None = None,
        name: str = "",
    ):
        """Create a pressure assembly from an element set.

        This constructor automatically extact the surface mesh
        from an element set.

        See :py:class:`Pressure` for more details on the parameters.
        """
        surface_mesh = extract_surface(mesh, element_set=element_set)
        return Pressure(surface_mesh, pressure, initial_pressure, nlgeom, name)


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

        # define a problem from the solid and pressure assemblies
        pb = fd.problem.Linear(solid_assembly+volume_force)
        pb.bc.add('Dirichlet', 'bottom', 'Disp', 0)
        pb.solve()

        pb.get_results(solid_assembly,'Stress').plot('Stress', 'XX', 'Node')
    """

    def __init__(
        self,
        mesh: Mesh,
        force: list | np.typing.ArrayLike[float],
        initial_force: np.typing.ArrayLike[float] | None = None,
        nlgeom: bool | None = None,
        name: str = "",
    ):
        self.force = force
        if initial_force is not None:
            self.initial_force = initial_force
        else:
            self.initial_force = None
        self.nlgeom = nlgeom
        wf = DistributedLoad(self.force, nlgeom=self.nlgeom)
        Assembly.__init__(self, wf, mesh, name=name)

    def set_start(self, pb: ProblemBase):
        """Start a new time increment."""
        if self.initial_force is None:
            if isinstance(self.force, np.ndarray):
                self.weakform.distributed_force = pb.t_fact * self.force
            else:
                self.weakform.distributed_force = [pb.t_fact * f for f in self.force]
        else:
            if isinstance(self.force, np.ndarray) and isinstance(
                self.initial_force, np.ndarray
            ):
                self.weakform.distributed_force = (
                    pb.t_fact * (self.force - self.initial_force) + self.initial_force
                )
            else:
                self.weakform.distributed_force = [
                    pb.t_fact * (f - self.initial_force[i]) + self.initial_force[i]
                    for i, f in enumerate(self.force)
                ]

    def to_start(self, pb: ProblemBase):
        """Reset the assembly to the begining of the time iteration."""
        self.set_start(pb)


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
    ):
        """Create a SurfaceForce assembly from an node set.

        This constructor automatically extact the surface mesh
        from an node set.

        See :py:class:`SurfaceForce` for more details on the parameters.
        """
        surface_mesh = extract_surface(mesh, node_set=node_set)
        return DistributedForce(surface_mesh, force, initial_force, nlgeom, name)

    @staticmethod
    def from_elements(
        mesh: Mesh,
        element_set: np.typing.ArrayLike[int] | str,
        force: np.typing.ArrayLike[float],
        initial_force: np.typing.ArrayLike[float] | None = None,
        nlgeom: bool | None = None,
        name: str = "",
    ):
        """Create a SurfaceForce assembly from an element set.

        This constructor automatically extact the surface mesh
        from an element set.

        See :py:class:`SurfaceForce` for more details on the parameters.
        """
        surface_mesh = extract_surface(mesh, element_set=element_set)
        return DistributedForce(surface_mesh, force, initial_force, nlgeom, name)
