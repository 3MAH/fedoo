from fedoo.core.base import ConstitutiveLaw
from fedoo.core.weakform import WeakFormBase, WeakFormSum
from fedoo.weakform.stress_equilibrium import StressEquilibrium
from fedoo.weakform.inertia import Inertia

import numpy as np


class ImplicitDynamic(WeakFormBase):
    """
    Weak formulation.

    Parameters
    ----------
    thermal_constitutivelaw: ConstitutiveLaw name (str) or ConstitutiveLaw object
        Thermal Constitutive Law (:mod:`fedoo.constitutivelaw`)
    name: str
        name of the WeakForm
    nlgeom: bool (default = False)
    """

    def __init__(
        self, constitutivelaw, density, beta, gamma, name="", nlgeom=False, space=None
    ):
        super().__init__(name, space)

        if name != "":
            stiffness_name = name + "_stiffness"
        else:
            stiffness_name = inertia_name = ""

        self.stiffness_weakform = StressEquilibrium(
            constitutivelaw, stiffness_name, nlgeom, space
        )
        self.constitutivelaw = self.stiffness_weakform.constitutivelaw
        self.beta = beta
        self.gamma = gamma
        self.density = density
        self.__nlgeom = nlgeom

        self.rayleigh_damping = None

        self.assembly_options["assume_sym"] = True

    def initialize(self, assembly, pb):
        self.stiffness_weakform.initialize(assembly, pb)
        assembly.sv_type["Velocity"] = "Node"
        assembly.sv_type["Acceleration"] = "Node"
        assembly.sv_type["_DeltaDisp"] = "Node"
        assembly.sv["Velocity"] = 0
        assembly.sv["Acceleration"] = 0
        assembly.sv["Velocity_GP"] = np.zeros((self.space.ndim, 1))
        assembly.sv["Acceleration_GP"] = np.zeros((self.space.ndim, 1))
        assembly.sv["_DeltaDisp"] = 0

    def update(self, assembly, pb):
        self.stiffness_weakform.update(assembly, pb)
        assembly.sv["_DeltaDisp"] = pb._dU.reshape(-1, pb.mesh.n_nodes)
        assembly.sv["_DeltaDisp_GP"] = assembly.convert_data(
            assembly.sv["_DeltaDisp"], convert_from="Node", convert_to="GaussPoint"
        )

        # self.inertia_weakform.update(assembly, pb)
        # assembly.sv['TempGradient'] = [0 if operator == 0 else
        #             assembly.get_gp_results(operator, pb.get_dof_solution()) for operator in self.__op_grad_temp]

    def update_2(self, assembly, pb):
        self.stiffness_weakform.update_2(assembly, pb)
        # self.inertia_weakform.update_2(assembly, pb)
        # assembly.sv['TempGradient'] = [0 if operator == 0 else
        #             assembly.get_gp_results(operator, pb.get_dof_solution()) for operator in self.__op_grad_temp]

    # def reset(self): #to update
    #     pass

    def to_start(self, assembly, pb):  # to update
        self.stiffness_weakform.to_start(assembly, pb)

    def set_start(self, assembly, pb):  # to update
        dt = pb.dtime  ### dt is the time step of the previous increment
        if not (np.isscalar(pb.get_disp()) and pb.get_disp() == 0):
            # update velocity and acceleration
            new_acceleration = (1 / (self.beta * dt**2)) * (
                assembly.sv["_DeltaDisp"] - dt * assembly.sv["Velocity"]
            ) - 1 / self.beta * (0.5 - self.beta) * assembly.sv["Acceleration"]

            assembly.sv["Velocity"] += dt * (
                (1 - self.gamma) * assembly.sv["Acceleration"]
                + self.gamma * new_acceleration
            )

            assembly.sv["Acceleration"] = new_acceleration

            assembly.sv["Velocity_GP"] = assembly.convert_data(
                assembly.sv["Velocity"], convert_from="Node", convert_to="GaussPoint"
            )
            assembly.sv["Acceleration_GP"] = assembly.convert_data(
                assembly.sv["Acceleration"],
                convert_from="Node",
                convert_to="GaussPoint",
            )

        self.stiffness_weakform.set_start(assembly, pb)

    def get_weak_equation(self, assembly, pb):
        op_dU = self.space.op_disp()  # displacement increment (incremental formulation)
        op_dU_vir = [du.virtual if du != 0 else 0 for du in op_dU]
        dt = pb.dtime

        if self.rayleigh_damping is None:
            if np.isscalar(pb._dU) and pb._dU == 0:  # start of iteration
                delta_disp = np.zeros((self.space.ndim, 1))
            else:
                delta_disp = assembly.sv["_DeltaDisp_GP"]

            acceleration = assembly.sv["Acceleration_GP"]
            velocity = assembly.sv["Velocity_GP"]

            diff_op = self.stiffness_weakform.get_weak_equation(assembly, pb)

            # if delta_disp.shape[1] == 1: delta_disp = delta_disp.ravel()
            # if acceleration.shape[1] == 1: acceleration = acceleration.ravel()
            # if velocity.shape[1] == 1: velocity = velocity.ravel()
            # diff_op += sum([op_dU_vir[i]*( \
            #         (op_dU[i]+delta_disp[i])*(self.density/(self.beta*dt**2)) -
            #           velocity[i]*(self.density/(self.beta*dt)) -
            #           acceleration[i] * (self.density*(0.5/self.beta - 1)) )
            #           if op_dU_vir[i]!=0 else 0 for i in range(self.space.ndim)])

            new_acceleration = (1 / (self.beta * dt**2)) * (
                delta_disp - dt * velocity
            ) + (1 - 0.5 / self.beta) * acceleration  # gp values

            diff_op += sum(
                [
                    op_dU_vir[i]
                    * (
                        op_dU[i] * (self.density / (self.beta * dt**2))
                        + self.density * new_acceleration[i]
                    )
                    if op_dU_vir[i] != 0
                    else 0
                    for i in range(self.space.ndim)
                ]
            )
        else:
            # need nodes values in this case
            if np.isscalar(pb._dU) and pb._dU == 0:  # start of iteration
                delta_disp = np.zeros((self.space.ndim, 1))
            else:
                delta_disp = assembly.sv["_DeltaDisp"]

            acceleration = assembly.sv["Acceleration"]
            velocity = assembly.sv["Velocity"]

            # not working for now
            new_acceleration = (1 / (self.beta * dt**2)) * (
                delta_disp - dt * velocity
            ) + (1 - 0.5 / self.beta) * acceleration
            new_velocity = velocity + dt * (
                (1 - self.gamma) * acceleration + self.gamma * new_acceleration
            )

            # same as self.stiffness_weakform.get_weak_equation(assembly, pb)
            # but separate mat and vec weakforms to apply rayleigh damping
            if self.nlgeom == "TL":  # add initial displacement effect
                eps = self.space.op_strain(assembly.sv["DispGradient"])
                initial_stress = assembly.sv["PK2"]
            else:
                eps = self.space.op_strain()
                initial_stress = assembly.sv[
                    "Stress"
                ]  # Stress = Cauchy for updated lagrangian method

            H = assembly.sv["TangentMatrix"]

            sigma = [
                sum([0 if eps[j] == 0 else eps[j] * H[i][j] for j in range(6)])
                for i in range(6)
            ]

            stiffness_wf = sum(
                [0 if eps[i] == 0 else eps[i].virtual * sigma[i] for i in range(6)]
            )

            if not (np.isscalar(initial_stress) and initial_stress == 0):
                if self.nlgeom:
                    stiffness_mat_wf = stiffness_wf + sum(
                        [
                            0
                            if self.stiffness_weakform._nl_strain_op_vir[i] == 0
                            else self.stiffness_weakform._nl_strain_op_vir[i]
                            * initial_stress[i]
                            for i in range(6)
                        ]
                    )

                initial_stress_wf = sum(
                    [
                        0 if eps[i] == 0 else eps[i].virtual * initial_stress[i]
                        for i in range(6)
                    ]
                )

            else:
                initial_stress_wf = 0

            # includes rayleigh term for stiffness matrix
            diff_op = stiffness_wf * (
                1 + self.rayleigh_damping[1] * self.gamma / (self.beta * dt)
            )

            diff_op += initial_stress_wf

            # includes rayleigh term for the mass matrix
            diff_op += sum(
                [
                    op_dU_vir[i]
                    * op_dU[i]
                    * (
                        self.density / (self.beta * dt**2)
                        + (self.density * self.rayleigh_damping[0] * self.gamma)
                        / (self.beta * dt)
                    )
                    if op_dU_vir[i] != 0
                    else 0
                    for i in range(self.space.ndim)
                ]
            )

            diff_op += sum(
                [
                    op_dU_vir[i]
                    * (
                        self.density * new_acceleration[i]
                        + (self.density * self.rayleigh_damping[0]) * new_velocity[i]
                    )
                    if op_dU_vir[i] != 0
                    else 0
                    for i in range(self.space.ndim)
                ]
            )

            # compute the weaform corresponding to rayleigh_damping[0]x[K]*new_velocity
            # as we dont compute K explicitly, we need to update the weaform
            # to get [K]*new_velocity (gauss point values), we need to compute the
            # operator used to compute sigma and apply it to the nodal values of new_velocity
            if new_velocity.shape[1] == 1:
                new_velocity = new_velocity * np.ones(assembly.mesh.n_nodes)
            op_sigma = [
                sum([0 if eps[j] == 0 else eps[j] * H[i][j] for j in range(6)])
                for i in range(6)
            ]
            diff_op += sum(
                [
                    0
                    if (eps[i] == 0) or (op_sigma[i] == 0)
                    else eps[i].virtual
                    * (
                        self.rayleigh_damping[0]
                        * assembly.get_gp_results(op_sigma[i], new_velocity.ravel())
                    )
                    for i in range(6)
                ]
            )

            # mat_sigma_velocity = [0 if op_sigma[i] == 0 else assembly.get_gp_results(op_sigma[i], new_velocity.ravel()) for i in range(6)]
            # diff_op += sum([0 if eps[i] == 0 else \
            #                 eps[i].virtual * mat_sigma_velocity[i] for i in range(6)])

        return diff_op

    @property
    def nlgeom(self):
        return self.__nlgeom


class _NewmarkInertia(WeakFormBase):
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

    def __init__(self, density, beta, gamma, name="", nlgeom=False, space=None):
        super().__init__(name, space)

        if name != "":
            stiffness_name = name + "_stiffness"
        else:
            stiffness_name = inertia_name = ""

        self.beta = beta
        self.gamma = gamma
        self.density = density
        self.__nlgeom = nlgeom

    def initialize(self, assembly, pb):
        assembly.sv_type["Velocity"] = "Node"
        assembly.sv_type["Acceleration"] = "Node"
        assembly.sv_type["_DeltaDisp"] = "Node"
        assembly.sv["Velocity"] = 0
        assembly.sv["Acceleration"] = 0
        assembly.sv["Velocity_GP"] = [0, 0, 0]
        assembly.sv["Acceleration_GP"] = [0, 0, 0]
        assembly.sv["_DeltaDisp"] = 0

    def update(self, assembly, pb):
        assembly.sv["_DeltaDisp"] = pb._dU.reshape(-1, pb.mesh.n_nodes)
        assembly.sv["_DeltaDisp_GP"] = assembly.convert_data(
            assembly.sv["_DeltaDisp"], convert_from="Node", convert_to="GaussPoint"
        )

        # self.inertia_weakform.update(assembly, pb)
        # assembly.sv['TempGradient'] = [0 if operator == 0 else
        #             assembly.get_gp_results(operator, pb.get_dof_solution()) for operator in self.__op_grad_temp]

    def update_2(self, assembly, pb):
        pass
        # self.inertia_weakform.update_2(assembly, pb)
        # assembly.sv['TempGradient'] = [0 if operator == 0 else
        #             assembly.get_gp_results(operator, pb.get_dof_solution()) for operator in self.__op_grad_temp]

    # def reset(self): #to update
    #     pass

    def to_start(self, assembly, pb):  # to update
        pass

    def set_start(self, assembly, pb):  # to update
        dt = pb.dtime  ### dt is the time step of the previous increment
        if not (np.isscalar(pb.get_disp()) and pb.get_disp() == 0):
            # update velocity and acceleration
            new_acceleration = (1 / (self.beta * dt**2)) * (
                assembly.sv["_DeltaDisp"] - dt * assembly.sv["Velocity"]
            ) - 1 / self.beta * (0.5 - self.beta) * assembly.sv["Acceleration"]

            assembly.sv["Velocity"] += dt * (
                (1 - self.gamma) * assembly.sv["Acceleration"]
                + self.gamma * new_acceleration
            )

            assembly.sv["Acceleration"] = new_acceleration

            assembly.sv["Velocity_GP"] = assembly.convert_data(
                assembly.sv["Velocity"], convert_from="Node", convert_to="GaussPoint"
            )
            assembly.sv["Acceleration_GP"] = assembly.convert_data(
                assembly.sv["Acceleration"],
                convert_from="Node",
                convert_to="GaussPoint",
            )

    def get_weak_equation(self, assembly, pb):
        op_dU = self.space.op_disp()  # displacement increment (incremental formulation)
        op_dU_vir = [du.virtual if du != 0 else 0 for du in op_dU]
        dt = pb.dtime

        if np.isscalar(pb._dU) and pb._dU == 0:  # start of iteration
            delta_disp = [0, 0, 0]
        else:
            delta_disp = assembly.sv["_DeltaDisp_GP"]

        acceleration = assembly.sv["Acceleration_GP"]
        velocity = assembly.sv["Velocity_GP"]
        diff_op = sum(
            [
                op_dU_vir[i]
                * (
                    (op_dU[i] - delta_disp[i]) * (self.density / (self.beta * dt**2))
                    - velocity[i] * (self.density / (self.beta * dt))
                    - acceleration[i] * (self.density * (0.5 / self.beta - 1))
                )
                if op_dU_vir[i] != 0
                else 0
                for i in range(self.space.ndim)
            ]
        )

        return diff_op

    @property
    def nlgeom(self):
        return self.__nlgeom


def ImplicitDynamic2(
    constitutivelaw, density, beta, gamma, name="", nlgeom=False, space=None
):
    stiffness_weakform = StressEquilibrium(constitutivelaw, "", nlgeom, space)
    time_integration = _NewmarkInertia(density, beta, gamma, "", nlgeom, space)
    time_integration.assembly_options["assume_sym"] = True
    return WeakFormSum([stiffness_weakform, time_integration], name)
