from fedoo.core.base import ConstitutiveLaw
from fedoo.core.weakform import WeakFormBase, WeakFormSum
from fedoo.weakform.stress_equilibrium import StressEquilibrium

import numpy as np


class ImplicitDynamic2(WeakFormBase):
    r"""Weak formulation for implicit dynamic problems.

    Extends the :mod:`fedoo.weakform.StressEquilibrium` weak formulation
    by adding inertia effects and optional damping. Time integration is
    performed using the Newmark-beta scheme.

    By default, the **Constant Average Acceleration** method is used
    (:math:`\\gamma=0.5, \\beta=0.25`), which is unconditionally stable
    and energy-preserving.

    Notes
    -----
    * **Numerical Damping:** Setting :math:`\\gamma > 0.5` introduces
        algorithmic damping to suppress high-frequency noise.
    * **Physical Damping:** Rayleigh damping can be added via the
        `rayleigh_damping` attribute.

    Parameters
    ----------
    constitutivelaw : str or ConstitutiveLaw
        Material Constitutive Law (:mod:`fedoo.constitutivelaw`).
    density : float or ndarray
        Material density provided as a constant or an array of values
        at Gauss points.
    beta : float, default=0.25
        Newmark integration parameter :math:`\\beta`.
    gamma : float, default=0.5
        Newmark integration parameter :math:`\\gamma`.
    name : str, optional
        Name of the WeakForm instance.
    nlgeom : bool or {'UL', 'TL'}, optional
        Enable geometric nonlinearities:

        * ``True`` or ``'UL'``: Updated Lagrangian method.
        * ``'TL'``: Total Lagrangian method.
        * ``None``: Uses the global ``problem.nlgeom`` setting.
    space : ModelingSpace, optional
        Modeling space for the weakform. Defaults to the active
        ``ModelingSpace``.
    """

    def __init__(
        self,
        constitutivelaw,
        density,
        beta=0.25,
        gamma=0.5,
        name="",
        nlgeom=False,
        space=None,
    ):
        super().__init__(name, space)

        if name != "":
            stiffness_name = name + "_stiffness"
        else:
            stiffness_name = ""

        self.stiffness_weakform = StressEquilibrium(
            constitutivelaw, stiffness_name, nlgeom, space
        )
        self.constitutivelaw = self.stiffness_weakform.constitutivelaw
        self.beta = beta
        self.gamma = gamma
        self.density = density
        self.__nlgeom = nlgeom

        self.rayleigh_damping = None
        """list of float, optional
        Coefficients :math:`[a, b]` for Rayleigh damping. 
        
        - :math:`a`: Mass-proportional term.
        - :math:`b`: Stiffness-proportional term.
        
        Used to form :math:`C = aM + bK`. Defaults to ``None``.
        """

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
        assembly.sv["_DeltaDisp"] = np.zeros((self.space.ndim, 1))
        assembly.sv["_DeltaDisp_GP"] = np.zeros((self.space.ndim, 1))

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
            # reset acumulated displacement
            assembly.sv["_DeltaDisp"] = np.zeros((self.space.ndim, 1))
            assembly.sv["_DeltaDisp_GP"] = np.zeros((self.space.ndim, 1))

        self.stiffness_weakform.set_start(assembly, pb)

    def get_weak_equation(self, assembly, pb):
        op_dU = self.space.op_disp()  # displacement increment (incremental formulation)
        op_dU_vir = [du.virtual if du != 0 else 0 for du in op_dU]
        dt = pb.dtime
        if dt == 0:
            return self.stiffness_weakform.get_weak_equation(assembly, pb)
        elif self.rayleigh_damping is None:
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
                if self.space._dimension == "2Daxi":
                    raise NotImplementedError("2Daxi not implemented.")
                #     rr = assembly.sv["_R_gausspoints"]

                #     # nlgeom = False
                #     eps[2] = self.space.variable("DispX") * np.divide(
                #         1, rr, out=np.zeros_like(rr), where=rr != 0
                #     )  # put zero if X==0 (division by 0)
                #     # eps[2] = self.space.variable('DispX') * (1/rr)
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
                # if self.nlgeom:
                #     # geometrical stiffness not included
                #     stiffness_mat_wf = stiffness_wf + sum(
                #         [
                #             0
                #             if self.stiffness_weakform._nl_strain_op_vir[i] == 0
                #             else self.stiffness_weakform._nl_strain_op_vir[i]
                #             * initial_stress[i]
                #             for i in range(6)
                #         ]
                #     )

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

            # compute the weaform corresponding to rayleigh_damping[1]x[K]*new_velocity
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
                        self.rayleigh_damping[1]
                        * assembly.get_gp_results(op_sigma[i], new_velocity.ravel())
                    )
                    for i in range(6)
                ]
            )
            # if self.space._dimension == "2Daxi":
            #     diff_op = diff_op * ((2 * np.pi) * rr)

            # mat_sigma_velocity = [0 if op_sigma[i] == 0 else assembly.get_gp_results(op_sigma[i], new_velocity.ravel()) for i in range(6)]
            # diff_op += sum([0 if eps[i] == 0 else \
            #                 eps[i].virtual * mat_sigma_velocity[i] for i in range(6)])

        return diff_op

    @property
    def nlgeom(self):
        return self.__nlgeom


class _NewmarkInertia(WeakFormBase):
    """Newmark inertia Weakform. Not intended to be used alone."""

    def __init__(self, density, beta, gamma, name="", nlgeom=False, space=None):
        super().__init__(name, space)
        self.beta = beta
        self.gamma = gamma
        self.density = density
        self.__nlgeom = nlgeom
        self.damping_coef = None  # for Rayleigh damping

    def initialize(self, assembly, pb):
        assembly.sv_type["Velocity"] = "Node"
        assembly.sv_type["Acceleration"] = "Node"
        assembly.sv_type["_DeltaDisp"] = "Node"
        assembly.sv["Velocity"] = 0
        assembly.sv["Acceleration"] = 0
        assembly.sv["Velocity_GP"] = np.zeros((self.space.ndim, 1))
        assembly.sv["Acceleration_GP"] = np.zeros((self.space.ndim, 1))
        assembly.sv["_DeltaDisp"] = np.zeros((self.space.ndim, 1))
        assembly.sv["_DeltaDisp_GP"] = np.zeros((self.space.ndim, 1))

    def update(self, assembly, pb):
        assembly.sv["_DeltaDisp"] = pb._get_vect_component(pb._dU, "Disp")

        assembly.sv["_DeltaDisp_GP"] = assembly.convert_data(
            assembly.sv["_DeltaDisp"], convert_from="Node", convert_to="GaussPoint"
        )

    def update_2(self, assembly, pb):
        pass

    def to_start(self, assembly, pb):  # to update
        pass

    def set_start(self, assembly, pb):
        # This updates the historic variables to the newly converged step t
        dt = pb.dtime  # dt is the time step of the previous increment
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

            # save gauss point values
            assembly.sv["Velocity_GP"] = assembly.convert_data(
                assembly.sv["Velocity"], convert_from="Node", convert_to="GaussPoint"
            )
            assembly.sv["Acceleration_GP"] = assembly.convert_data(
                assembly.sv["Acceleration"],
                convert_from="Node",
                convert_to="GaussPoint",
            )
            # reset acumulated displacement
            assembly.sv["_DeltaDisp"] = np.zeros((self.space.ndim, 1))
            assembly.sv["_DeltaDisp_GP"] = np.zeros((self.space.ndim, 1))

    def get_weak_equation(self, assembly, pb):
        op_dU = self.space.op_disp()  # displacement increment (incremental formulation)
        op_dU_vir = [du.virtual if du != 0 else 0 for du in op_dU]
        dt = pb.dtime
        if dt == 0:
            return 0

        # Newmark integration constants
        a0 = 1 / (self.beta * dt**2)
        c0 = self.gamma / (self.beta * dt)
        alpha = self.damping_coef if self.damping_coef is not None else 0.0

        # Current NR Iteration state
        a_n = assembly.sv["Acceleration_GP"]  # at t_n
        v_n = assembly.sv["Velocity_GP"]  # at t_n
        delta_disp = assembly.sv["_DeltaDisp_GP"]  # accumulated since step start

        # Current estimate of acc and vel (includes accumulated delta_disp)
        acc_i = a0 * (delta_disp - dt * v_n) + (1 - 0.5 / self.beta) * a_n
        vel_i = v_n + dt * ((1 - self.gamma) * a_n + self.gamma * acc_i)

        # Integration: delta_u * rho * [ (a0 + alpha*c0)*du + (acc_i + alpha*vel_i) ]
        tangent_coeff = self.density * (a0 + alpha * c0)
        residual_val = self.density * (acc_i + alpha * vel_i)

        diff_op = sum(
            [
                op_dU_vir[i] * (tangent_coeff * op_dU[i] + residual_val[i])
                for i in range(self.space.ndim)
                if op_dU_vir[i] != 0
            ]
        )
        if self.space._dimension == "2Daxi":
            rr = assembly.sv["_R_gausspoints"]
            return diff_op * ((2 * np.pi) * rr)
        else:
            return diff_op

    @property
    def nlgeom(self):
        return self.__nlgeom


class _StressEquilibriumRayleighDamping(StressEquilibrium):
    """Stress Equilibirum with Rayleigh Damping used for Newmark weak form.

    Not intended to be used alone."""

    def __init__(self, constitutivelaw, beta, gamma, name="", nlgeom=None, space=None):
        super().__init__(constitutivelaw, name, nlgeom, space)
        self.beta = beta
        self.gamma = gamma
        self.damping_coef = None  # for Rayleigh damping

    def get_weak_equation(self, assembly, pb):
        # 1. Get standard static stiffness weakform (Tangent K and Internal Force residual)
        wf = super().get_weak_equation(assembly, pb)

        dt = pb.dtime
        # Return static WF if damping is 0 or it's the initial/static step
        if self.damping_coef is None or self.damping_coef == 0.0 or dt == 0:
            return wf

        # Get states from the start of the time step (t)
        a_n_node = assembly.sv["Acceleration"]
        v_n_node = assembly.sv["Velocity"]

        # Get the accumulated displacement increment for the current time step (delta_u)
        # This is updated at each Newton-Raphson iteration
        delta_u = assembly.sv["_DeltaDisp"]

        # if np.isscalar(a_n_node) or np.isscalar(v_n_node):
        #     return wf

        # Newmark Coefficients
        # c0 is the factor multiplying the tangent: d(vel)/d(delta_u) = gamma / (beta * dt)
        c0 = self.gamma / (self.beta * dt)
        a0 = 1 / (self.beta * dt**2)

        # 2. Tangent Matrix Scaling for Damping (beta_ray * c0 * K_tan)
        mat, vec = wf.split_mat_vec()
        scaled_mat = mat * (1 + self.damping_coef * c0)

        # 3. Current Velocity Calculation (v_curr)
        # Current acceleration: a_curr = a0*(delta_u - dt*v_n) - (1/(2*beta)-1)*a_n
        a_curr = a0 * (delta_u - dt * v_n_node) - (0.5 / self.beta - 1) * a_n_node

        # Current velocity: v_curr = v_n + dt*((1-gamma)*a_n + gamma*a_curr)
        v_curr = v_n_node + dt * ((1 - self.gamma) * a_n_node + self.gamma * a_curr)

        # Ensure correct shape for the operator application
        if v_curr.ndim > 1 and v_curr.shape[1] == 1:
            v_curr = v_curr * np.ones(assembly.mesh.n_nodes)

        # 4. Build the damping residual: delta_eps * (beta_ray * H * eps(v_curr))
        eps = self.space.op_strain()
        if self.space._dimension == "2Daxi":
            rr = assembly.sv["_R_gausspoints"]

            # nlgeom = False
            eps[2] = self.space.variable("DispX") * np.divide(
                1, rr, out=np.zeros_like(rr), where=rr != 0
            )  # put zero if X==0 (division by 0)

        H = assembly.sv["TangentMatrix"]

        # Define the operator for sigma_damping = H * eps(v_curr)
        op_sigma_v = [
            sum([0 if eps[j] == 0 else eps[j] * H[i][j] for j in range(6)])
            for i in range(6)
        ]

        # Apply the operator to the nodal v_curr to get the damping force residual
        damping_force_wf = self.damping_coef * assembly.operator_apply(
            mat, v_curr.ravel()
        )
        if self.space._dimension == "2Daxi":
            damping_force_wf = damping_force_wf * ((2 * np.pi) * rr)

        return scaled_mat + vec + damping_force_wf


class ImplicitDynamicSum(WeakFormSum):
    """WeakFormSum with the addition of rayleigh_damping property."""

    def __init__(self, weakforms, name=""):
        super().__init__(weakforms, name)

    @property
    def rayleigh_damping(self):
        """list: Coefficients [a, b] for Rayleigh damping."""
        if self.list_weakform[0].damping_coef is None:
            return None
        # list_weakform[0] is Stiffness (b), list_weakform[1] is Inertia (a)
        return [self.list_weakform[i].damping_coef for i in [1, 0]]

    @rayleigh_damping.setter
    def rayleigh_damping(self, value):
        if value is None:
            value = [None, None]

        self.list_weakform[0].damping_coef = value[1]  # stiffness matrix coef
        self.list_weakform[1].damping_coef = value[0]  # mass matrix coef


def ImplicitDynamic(
    constitutivelaw, density, beta=0.25, gamma=0.5, name="", nlgeom=False, space=None
):
    """Aleternative implementation of the ImplicitDynamic class.

    Should be 100% equivalent.
    """
    stiffness_weakform = _StressEquilibriumRayleighDamping(
        constitutivelaw, beta, gamma, "", nlgeom, space
    )
    time_integration = _NewmarkInertia(density, beta, gamma, "", nlgeom, space)
    time_integration.assembly_options["assume_sym"] = True
    return ImplicitDynamicSum([stiffness_weakform, time_integration], name)
