"""FE² constitutive law."""

import numpy as np

from fedoo.constraint.periodic_bc import PeriodicBC
from fedoo.core.assembly import Assembly
from fedoo.core.mechanical3d import Mechanical3D
from fedoo.homogen.tangent_stiffness import (
    get_homogenized_stiffness,
    get_tangent_stiffness,
)
from fedoo.problem.non_linear import NonLinear
from fedoo.util.voigt_tensors import StressTensorList, StrainTensorList


class FE2(Mechanical3D):
    """FE² constitutive law based on periodic microscopic problems.

    A microscopic finite-element problem is solved at each macroscopic
    integration point. The macroscopic strain is prescribed through the
    problem-level ``MeanStrain`` degrees of freedom created by
    :class:`PeriodicBC`.

    Parameters
    ----------
    assemb : Assembly, str, or list of Assembly
        Microscopic assembly, its registered name, or one microscopic
        assembly per macroscopic integration point.
    name : str, optional
        Name of the constitutive law.
    """

    def __init__(self, assemb, name=""):
        if isinstance(assemb, str):
            assemb = Assembly.get_all()[assemb]
        super().__init__(name)

        if isinstance(assemb, list):
            self.__assembly = [
                Assembly.get_all()[item] if isinstance(item, str) else item
                for item in assemb
            ]
            self.__mesh = [item.mesh for item in self.__assembly]
        else:
            self.__assembly = assemb
            self.__mesh = assemb.mesh

        self.list_problem = None
        self.use_elastic_lt = True

    def initialize(self, assembly, pb):
        if self.list_problem is not None:
            return

        nb_points = assembly.n_gauss_points
        if isinstance(self.__mesh, list):
            if len(self.__assembly) != nb_points:
                raise ValueError(
                    "A list of microscopic assemblies must contain one "
                    "assembly per macroscopic integration point."
                )
            self.list_mesh = self.__mesh
            self.list_assembly = self.__assembly
        else:
            self.list_mesh = [self.__mesh for _ in range(nb_points)]
            self.list_assembly = [self.__assembly.copy() for _ in range(nb_points)]

        self.list_problem = []
        self._list_volume = np.empty(nb_points)
        assembly.sv["TangentMatrix"] = np.empty((6, 6, nb_points))

        print("-- Initialize micro problems --")
        for index in range(nb_points):
            print("\r", index + 1, "/", nb_points, end="")
            coordinates = self.list_mesh[index].nodes
            lower = np.min(coordinates, axis=0)
            upper = np.max(coordinates, axis=0)
            box_center = (lower + upper) / 2
            self._list_volume[index] = np.prod(upper - lower)
            center_node = np.linalg.norm(coordinates - box_center, axis=1).argmin()

            micro_problem = NonLinear(
                self.list_assembly[index],
                name=f"_fe2_cell_{index}",
            )
            self.list_problem.append(micro_problem)
            micro_problem.bc.add(
                PeriodicBC(
                    "small_strain",
                    dim=3,
                    name=f"_fe2_cell_{index}",
                )
            )
            micro_problem.bc.add("Dirichlet", [center_node], "Disp", 0)
            assembly.sv["TangentMatrix"][:, :, index] = get_homogenized_stiffness(
                self.list_assembly[index]
            )

        pb.make_active()
        if self.use_elastic_lt:
            assembly.sv["ElasticMatrix"] = assembly.sv["TangentMatrix"].copy()

        assembly.sv["Strain"] = StrainTensorList(np.zeros((6, nb_points)))
        assembly.sv["Stress"] = StressTensorList(np.zeros((6, nb_points)))
        assembly.sv["Wm"] = np.zeros((4, nb_points))
        print("")

    def set_start(self, assembly, pb):
        if self.use_elastic_lt:
            assembly.sv["TangentMatrix"] = assembly.sv["ElasticMatrix"].copy()

    def _update_pb(self, index, assembly_macro, pb_macro):
        strain = assembly_macro.sv["Strain"]
        strain_start = assembly_macro.sv_start["Strain"]
        micro_problem = self.list_problem[index]

        print("\r", index + 1, "/", len(self.list_problem), end="")
        micro_problem.bc.remove("Strain")
        micro_problem.bc.add(
            "Dirichlet",
            "MeanStrain",
            strain.asarray()[:, index],
            start_value=strain_start.asarray()[:, index],
            name="Strain",
        )
        micro_problem.nlsolve(
            dt=pb_macro.dtime,
            tmax=pb_macro.dtime,
            update_dt=True,
            tol_nr=0.05,
            print_info=0,
        )

        assembly_macro.sv["TangentMatrix"][:, :, index] = get_tangent_stiffness(
            micro_problem.name
        )

        micro_assembly = self.list_assembly[index]
        stress_field = micro_assembly.sv["Stress"]
        assembly_macro.sv["Stress"].asarray()[:, index] = np.array(
            [
                micro_assembly.integrate_field(component) / self._list_volume[index]
                for component in stress_field
            ]
        )

        energy_field = micro_assembly.sv.get("Wm")
        if energy_field is not None:
            assembly_macro.sv["Wm"][:, index] = (
                micro_assembly.integrate_field(energy_field) / self._list_volume[index]
            )

    def update(self, assembly, pb):
        print("-- Update micro cells --")
        for index in range(len(self.list_problem)):
            self._update_pb(index, assembly, pb)
        print("")
