from fedoo.core.base import ConstitutiveLaw
from fedoo.core.time_evolution import FIRST_ORDER
from fedoo.core.weakform import WeakFormBase
import numpy as np


class HeatEquation(WeakFormBase):
    """
    Weak formulation of the heat equation.

    This weakform defines the spatial conduction operator and declares
    :class:`HeatCapacity` as its storage term. The problem-level time
    integrator determines whether the analysis is steady or transient.

    Parameters
    ----------
    thermal_constitutivelaw: ConstitutiveLaw name (str) or ConstitutiveLaw object
        Thermal Constitutive Law (:mod:`fedoo.constitutivelaw`)
    name: str
        name of the WeakForm
    nlgeom: bool (default = False)
    """

    def __init__(self, thermal_constitutivelaw, name=None, nlgeom=False, space=None):
        if isinstance(thermal_constitutivelaw, str):
            thermal_constitutivelaw = ConstitutiveLaw.get_all()[thermal_constitutivelaw]

        if name is None:
            name = thermal_constitutivelaw.name

        WeakFormBase.__init__(self, name, space)
        if self.space.is_axisymmetric:
            raise NotImplementedError

        self.space.new_variable("Temp")  # temperature

        if self.space.ndim == 3:
            self.__op_grad_temp = [
                self.space.derivative("Temp", "X"),
                self.space.derivative("Temp", "Y"),
                self.space.derivative("Temp", "Z"),
            ]
        else:  # 2D
            self.__op_grad_temp = [
                self.space.derivative("Temp", "X"),
                self.space.derivative("Temp", "Y"),
                0,
            ]

        self.__op_grad_temp_vir = [
            0 if op == 0 else op.virtual for op in self.__op_grad_temp
        ]

        self.constitutivelaw = thermal_constitutivelaw
        self.time_evolution = FIRST_ORDER
        self.__nlgeom = nlgeom
        self.storage = HeatCapacity(thermal_constitutivelaw, "", nlgeom, self.space)
        self.storage.assembly_options["mat_lumping"] = True

        # self.__nlgeom = nlgeom #geometric non linearities

    def initialize(self, assembly, pb):
        if not (np.isscalar(pb.get_dof_solution())):
            assembly.sv["TempGradient"] = [
                0
                if operator == 0
                else assembly.get_gp_results(operator, pb.get_dof_solution())
                for operator in self.__op_grad_temp
            ]
        else:
            assembly.sv["TempGradient"] = [0 for operator in self.__op_grad_temp]

    def update(self, assembly, pb):
        assembly.sv["TempGradient"] = [
            0
            if operator == 0
            else assembly.get_gp_results(operator, pb.get_dof_solution())
            for operator in self.__op_grad_temp
        ]

    def get_weak_equation(self, assembly, pb):
        K = self.constitutivelaw.thermal_conductivity

        diff_op = sum(
            [
                0
                if self.__op_grad_temp_vir[i] == 0
                else self.__op_grad_temp_vir[i]
                * sum(
                    [
                        0
                        if self.__op_grad_temp[j] == 0
                        else self.__op_grad_temp[j] * K[i][j]
                        for j in range(3)
                    ]
                )
                for i in range(3)
            ]
        )

        temp_grad = assembly.sv["TempGradient"]
        # add initial state for incremental resolution
        diff_op += sum(
            [
                0
                if self.__op_grad_temp_vir[i] == 0
                else self.__op_grad_temp_vir[i]
                * sum(
                    [
                        temp_grad[j] * K[i][j]
                        for j in range(3)
                        if (
                            not (np.array_equal(K[i][j], 0))
                            and not (np.array_equal(temp_grad[j], 0))
                        )
                    ]
                )
                for i in range(3)
            ]
        )

        return diff_op

    @property
    def nlgeom(self):
        return self.__nlgeom


class HeatCapacity(WeakFormBase):
    """Heat capacity storage weakform.

    This weakform only defines the capacity operator ``rho*c``. Time
    discretization is handled by problem-level time integrators.
    """

    def __init__(self, thermal_constitutivelaw, name=None, nlgeom=False, space=None):
        if isinstance(thermal_constitutivelaw, str):
            thermal_constitutivelaw = ConstitutiveLaw.get_all()[thermal_constitutivelaw]

        if name is None:
            name = thermal_constitutivelaw.name

        WeakFormBase.__init__(self, name, space)

        self.space.new_variable("Temp")  # temperature

        self.constitutivelaw = thermal_constitutivelaw
        self.time_evolution = FIRST_ORDER
        self.__nlgeom = nlgeom

    def initialize(self, assembly, pb):
        if not (np.isscalar(pb.get_dof_solution())):
            assembly.sv["Temp"] = assembly.convert_data(
                pb.get_temp(), convert_from="Node", convert_to="GaussPoint"
            )
        else:
            assembly.sv["Temp"] = 0

    def update(self, assembly, pb):
        assembly.sv["Temp"] = assembly.convert_data(
            pb.get_temp(), convert_from="Node", convert_to="GaussPoint"
        )

    def get_weak_equation(self, assembly, pb):
        rho_c = self.constitutivelaw.density * self.constitutivelaw.specific_heat

        op_temp = self.space.variable(
            "Temp"
        )  # temperature increment (incremental weakform)
        return rho_c * op_temp.virtual * op_temp

    def get_storage_value(self, assembly, pb):
        return assembly.sv["Temp"]

    def get_weak_equation_for_value(self, assembly, pb, value):
        rho_c = self.constitutivelaw.density * self.constitutivelaw.specific_heat
        return rho_c * self.space.variable("Temp").virtual * value

    @property
    def nlgeom(self):
        return self.__nlgeom
