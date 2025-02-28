from fedoo.core.base import ConstitutiveLaw
from fedoo.core.weakform import WeakFormBase, WeakFormSum
import numpy as np


class SteadyHeatEquation(WeakFormBase):
    """
    Weak formulation of the steady heat equation (without time evolution).

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
        if self.space._dimension == "2Daxi":
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

    def reset(self):  # to update
        pass

    def to_start(self):  # to update
        pass

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


class TemperatureTimeDerivative(WeakFormBase):
    def __init__(self, thermal_constitutivelaw, name=None, nlgeom=False, space=None):
        if isinstance(thermal_constitutivelaw, str):
            thermal_constitutivelaw = ConstitutiveLaw.get_all()[thermal_constitutivelaw]

        if name is None:
            name = thermal_constitutivelaw.name

        WeakFormBase.__init__(self, name, space)

        self.space.new_variable("Temp")  # temperature

        self.constitutivelaw = thermal_constitutivelaw

    def initialize(self, assembly, pb):
        if not (np.isscalar(pb.get_dof_solution())):
            self.__temp_start = assembly.convert_data(
                pb.get_temp(), convert_from="Node", convert_to="GaussPoint"
            )
            assembly.sv["Temp"] = self.__temp_start
        else:
            self.__temp_start = assembly.sv["Temp"] = 0

    def update(self, assembly, pb):
        assembly.sv["Temp"] = assembly.convert_data(
            pb.get_temp(), convert_from="Node", convert_to="GaussPoint"
        )

    def reset(self):
        pass

    #     self.__ConstitutiveLaw.reset()
    #     self.__InitialStressTensor = 0
    #     self.__InitialGradDispTensor = None

    def to_start(self):
        pass

    def set_start(self, assembly, pb):
        self.__temp_start = assembly.sv["Temp"]
        # no need to update Initial Stress because the last computed stress remained unchanged

    def get_weak_equation(self, assembly, pb):
        rho_c = self.constitutivelaw.density * self.constitutivelaw.specific_heat

        op_temp = self.space.variable(
            "Temp"
        )  # temperature increment (incremental weakform)
        # steady state should not include the following term
        if pb.dtime != 0:
            return (
                1
                / pb.dtime
                * rho_c
                * (
                    op_temp.virtual * op_temp
                    + op_temp.virtual * (assembly.sv["Temp"] - self.__temp_start)
                )
            )
            # return 1/self.__dtime * rho_c * (op_temp.virtual * op_temp + op_temp.virtual *(assembly.sv['Temp']))
        else:
            return 0

    @property
    def nlgeom(self):
        return self.__nlgeom


def HeatEquation(thermal_constitutivelaw, name=None, nlgeom=False, space=None):
    """
    Weak formulation of the heat equation.

    Parameters
    ----------
    thermal_constitutivelaw: ConstitutiveLaw name (str) or ConstitutiveLaw object
        Thermal Constitutive Law (:mod:`fedoo.constitutivelaw`)
    name: str
        name of the WeakForm
    nlgeom: bool (default = False)

    This weakform use mat_lumping for the time derivative assembly.
    Without mat_lumping, the solution is generally wrong with notable temperature oscillations.

    However, it is possible to change this parameter using:

        .. code-block:: python

            wf = fedoo.weakform.HeatEquation(thermal_constitutivelaw)
            wf.list_weakform[1].assembly_options['mat_lumping'] = False
    """
    heat_eq_diffusion = SteadyHeatEquation(thermal_constitutivelaw, "", nlgeom, space)
    heat_eq_time = TemperatureTimeDerivative(thermal_constitutivelaw, "", nlgeom, space)
    heat_eq_time.assembly_options["mat_lumping"] = (
        True  # use mat_lumping for the temperature time derivative
    )
    if name is None:
        if isinstance(thermal_constitutivelaw, str):
            name = ConstitutiveLaw().get_all()[thermal_constitutivelaw].name
        else:
            name = thermal_constitutivelaw.name
    return WeakFormSum([heat_eq_diffusion, heat_eq_time], name)
