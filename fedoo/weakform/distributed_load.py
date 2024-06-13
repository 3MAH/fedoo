"""Weak forms for distributed loads.

This file is part of the fedoo finite element code.
"""

from __future__ import annotations
from fedoo.core.weakform import WeakFormBase
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from fedoo.core.base import AssemblyBase, ProblemBase
    from fedoo.core.diffop import DiffOp
    from fedoo.core.modelingspace import ModelingSpace


class ExternalPressure(WeakFormBase):
    """Weak formulation for a pressure over a surface.

    This weak formulation permit to define pressure load and
    should be associated to another weak formulation (e.g. StressEquilibrium).

    The easiest way to define a pressure load is to directly
    build the corresponding assembly with :py:meth:`fd.constraint.Pressure`.
    """

    def __init__(
        self,
        pressure: float | np.ndarray,
        name: str = "",
        nlgeom: bool | None = False,
        space: ModelingSpace | None = None,
    ):
        WeakFormBase.__init__(self, name, space)

        self.space.new_variable("DispX")
        self.space.new_variable("DispY")
        if self.space.ndim == 3:
            self.space.new_variable("DispZ")
            self.space.new_vector("Disp", ("DispX", "DispY", "DispZ"))
        else:  # 2D assumed
            self.space.new_vector("Disp", ("DispX", "DispY"))

        self.pressure = pressure

        # default option for non linear
        self.nlgeom = nlgeom
        """Method used to treat the geometric non linearities.
            * Set to False if geometric non linarities are ignored (default).
            * Set to True or 'UL' to use the updated lagrangian method
              (update the mesh)
            * Set to 'TL' to use the total lagrangian method
              (base on the initial mesh with initial displacement effet)
        """

    def get_weak_equation(self, assembly: AssemblyBase, pb: ProblemBase) -> DiffOp:
        """Return the weak equation related to the current state."""
        vec_u = self.space.op_disp()
        normals = assembly.sv["Normals"]

        if self.space._dimension == "2Daxi":
            mesh = assembly.current.mesh
            rr = mesh.convert_data(
                mesh.nodes[:, 0],
                "Node",
                "GaussPoint",
                n_elm_gp=normals.shape[0] // mesh.n_elements,
            )
            return sum(
                [
                    u.virtual * ((normals[:, i] * rr) * (2 * np.pi * self.pressure))
                    for i, u in enumerate(vec_u)
                ]
            )
        else:
            return sum(
                [
                    u.virtual * (normals[:, i] * self.pressure)
                    for i, u in enumerate(vec_u)
                ]
            )

    def initialize(self, assembly: AssemblyBase, pb: ProblemBase):
        """Initialize the weakform for given assembly and pb."""

        # initialize the nlgeom value in assembly._nlgeom
        self._initialize_nlgeom(assembly, pb)

        assembly.sv["Normals"] = assembly.current.mesh.get_element_local_frame(
            n_elm_gp=assembly.n_elm_gp
        )[:, -1]  # normal should be the last axis

    def update(self, assembly: AssemblyBase, pb: ProblemBase):
        """Update the weakform for given assembly and pb."""
        if assembly._nlgeom == "UL":
            # updated lagragian method -> update the mesh
            assembly.set_disp(pb.get_disp())

        assembly.sv["Normals"] = assembly.current.mesh.get_element_local_frame(
            n_elm_gp=assembly.n_elm_gp
        )[:, -1]

    def to_start(self, assembly: AssemblyBase, pb: ProblemBase):
        """Reset the weakform to the begining of the time iteration."""
        if assembly._nlgeom == "UL":
            # updated lagragian method -> update the mesh
            assembly.set_disp(pb.get_disp())


class DistributedLoad(WeakFormBase):
    """Weak formulation for a distributed load.

    This weak formulation permit to define distributed loads and
    should be associated to another weak formulation (e.g. StressEquilibrium).

    The easiest way to define a distributed load is to directly
    build the corresponding assembly with
    :py:meth:`fd.constraint.DistributedForce` or
    :py:meth:`fd.constraint.SurfaceForce`.
    """

    def __init__(self, distributed_force: list, name="", nlgeom=False, space=None):
        WeakFormBase.__init__(self, name, space)

        self.space.new_variable("DispX")
        self.space.new_variable("DispY")
        if self.space.ndim == 3:
            self.space.new_variable("DispZ")
            self.space.new_vector("Disp", ("DispX", "DispY", "DispZ"))
        else:  # 2D assumed
            self.space.new_vector("Disp", ("DispX", "DispY"))

        self.distributed_force = distributed_force
        self.nlgeom = nlgeom

    def get_weak_equation(self, assembly: AssemblyBase, pb: ProblemBase) -> DiffOp:
        """Return the weak equation related to the current state."""
        vec_u = self.space.op_disp()

        if self.space._dimension == "2Daxi":
            mesh = assembly.current.mesh
            rr = mesh.convert_data(
                mesh.nodes[:, 0],
                "Node",
                "GaussPoint",
            )
            # perhaps distributed_force should be converted to gauss_point
            # to be compatible with rr
            return sum(
                [
                    u.virtual * ((self.distributed_force[i] * rr) * (-2 * np.pi))
                    for i, u in enumerate(vec_u)
                ]
            )
        else:
            return sum(
                [-u.virtual * self.distributed_force[i] for i, u in enumerate(vec_u)]
            )

    def initialize(self, assembly: AssemblyBase, pb: ProblemBase):
        """Initialize the weakform for given assembly and pb."""

        # initialize the nlgeom value in assembly._nlgeom
        self._initialize_nlgeom(assembly, pb)

    def update(self, assembly: AssemblyBase, pb: ProblemBase):
        """Update the weakform for given assembly and pb."""
        if assembly._nlgeom == "UL":
            # updated lagragian method -> update the mesh
            assembly.set_disp(pb.get_disp())

    def to_start(self, assembly: AssemblyBase, pb: ProblemBase):
        """Reset the weakform to the begining of the time iteration."""
        if assembly._nlgeom == "UL":
            # updated lagragian method -> update the mesh
            assembly.set_disp(pb.get_disp())
