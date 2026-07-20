import numpy as np
from fedoo.core.assembly import Assembly
from fedoo.core.problem import Problem
from fedoo.util.deprecation import deprecated_alias


class _NewmarkBase:
    """Define a Newmark problem.

    The algorithm come from:
        Bathe KJ and Edward W, "Numerical methods in finite element analysis", Prentice Hall, 1976, pp 323-324
    """

    def __init__(
        self,
        stiffness_assembly,
        mass_assembly,
        beta,
        gamma,
        time_step,
        damping_assembly=None,
        name="MainProblem",
    ):
        if isinstance(stiffness_assembly, str):
            stiffness_assembly = Assembly.get_all()[stiffness_assembly]

        if isinstance(mass_assembly, str):
            mass_assembly = Assembly.get_all()[mass_assembly]

        if isinstance(damping_assembly, str):
            damping_assembly = Assembly.get_all()[damping_assembly]

        if damping_assembly is None:
            A = (
                stiffness_assembly.get_global_matrix()
                + 1 / (beta * (time_step**2)) * mass_assembly.get_global_matrix()
            )
        else:
            A = (
                stiffness_assembly.get_global_matrix()
                + 1 / (beta * (time_step**2)) * mass_assembly.get_global_matrix()
                + gamma / (beta * time_step) * damping_assembly.get_global_matrix()
            )

        B = 0
        D = 0

        self.__Beta = beta
        self.__Gamma = gamma
        self.__TimeStep = time_step

        self.__MassMatrix = mass_assembly.get_global_matrix()
        self.__StiffMatrix = stiffness_assembly.get_global_matrix()
        if damping_assembly is None:
            self.__DampMatrix = None
        else:
            self.__DampMatrix = damping_assembly.get_global_matrix()

        super().__init__(
            A, B, D, stiffness_assembly.mesh, name, stiffness_assembly.space
        )

        self.__Xold = self._new_vect_dof()  # displacement at the previous time step
        self.__Xdot = self._new_vect_dof()
        self.__Xdotdot = self._new_vect_dof()

    def __UpdateA(
        self,
    ):  # internal function to be used when modifying M, K or C
        if self.__DampMatrix is None:
            self.set_A(
                self.__StiffMatrix
                + 1 / (self.__Beta * (self.__TimeStep**2)) * self.__MassMatrix
            )
        else:
            self.set_A(
                self.__StiffMatrix
                + 1 / (self.__Beta * (self.__TimeStep**2)) * self.__MassMatrix
                + self.__Gamma / (self.__Beta * self.__TimeStep) * self.__DampMatrix
            )

    def get_X(self):
        return self.get_dof_solution("all")

    def get_velocity(self):
        return self.__Xdot

    def get_acceleration(self):
        return self.__Xdotdot

    def get_disp(self, name="Disp"):  # same as get_X
        return self.get_dof_solution(name)

    GetVelocity = deprecated_alias(get_velocity, "GetVelocity")
    get_Acceleration = deprecated_alias(get_acceleration, "get_Acceleration")
    GetAcceleration = deprecated_alias(get_acceleration, "GetAcceleration")
    get_Xdot = deprecated_alias(get_velocity, "get_Xdot")
    get_Xdotdot = deprecated_alias(get_acceleration, "get_Xdotdot")

    def set_initial_displacement(self, name, value):
        """
        name is the name of the associated variable (generaly 'DispX', 'DispY' or 'DispZ')
        value is an array containing the initial displacement of each nodes
        """
        self._set_vect_component(self.__Xold, name, value)

    def set_initial_velocity(self, name, value):
        """
        name is the name of the associated variable (generaly 'DispX', 'DispY' or 'DispZ')
        value is an array containing the initial velocity of each nodes
        """
        self._set_vect_component(self.__Xdot, name, value)

    def set_initial_acceleration(self, name, value):
        """
        name is the name of the associated variable (generaly 'DispX', 'DispY' or 'DispZ')
        value is an array containing the initial acceleration of each nodes
        """
        self._set_vect_component(self.__Xdotdot, name, value)

    SetInitialDisplacement = deprecated_alias(
        set_initial_displacement, "SetInitialDisplacement"
    )
    SetInitialVelocity = deprecated_alias(set_initial_velocity, "SetInitialVelocity")
    SetInitialAcceleration = deprecated_alias(
        set_initial_acceleration, "SetInitialAcceleration"
    )

    def set_rayleigh_damping(self, alpha, beta):
        """
        Compute the damping matrix from the Rayleigh's model:
        [C] = alpha*[M] + beta*[K]

        where [C] is the damping matrix, [M] is the mass matrix and [K] is the stiffness matrix
        Note: The rayleigh model with alpha = 0 and beta = Viscosity/YoungModulus is almost equivalent to the multi-axial Kelvin-Voigt model

        Warning: the damping matrix is not automatically updated when mass and stiffness matrix are modified.
        """

        self.__DampMatrix = alpha * self.__MassMatrix + beta * self.__StiffMatrix
        self.__UpdateA()

    SetRayleighDamping = deprecated_alias(set_rayleigh_damping, "SetRayleighDamping")

    def initialize(self, t0=0.0):
        D = self.__MassMatrix * (
            (1 / (self.__Beta * self.__TimeStep**2)) * self.__Xold
            + (1 / (self.__Beta * self.__TimeStep)) * self.__Xdot
            + (0.5 / self.__Beta - 1) * self.__Xdotdot
        )
        if self.__DampMatrix is not None:
            D += self.__DampMatrix * (
                (self.__Gamma / (self.__Beta * self.__TimeStep)) * self.__Xold
                + (self.__Gamma / self.__Beta - 1) * self.__Xdot
                + (0.5 * self.__TimeStep * (self.__Gamma / self.__Beta - 2))
                * self.__Xdotdot
            )
        self.set_D(D)

    def update(self):
        NewXdotdot = (1 / self.__Beta / (self.__TimeStep**2)) * (
            self.get_dof_solution("all") - self.__Xold - self.__TimeStep * self.__Xdot
        ) - 1 / self.__Beta * (0.5 - self.__Beta) * self.__Xdotdot
        self.__Xdot += self.__TimeStep * (
            (1 - self.__Gamma) * self.__Xdotdot + self.__Gamma * NewXdotdot
        )
        self.__Xdotdot = NewXdotdot
        self.__Xold[:] = self.get_dof_solution("all")
        self.initialize()

    #        self.set_D(self.__MassMatrix * ( (1/self.__Beta/(self.__TimeStep**2))*self.__Xold + (1/self.__Beta/self.__TimeStep)*self.__Xdot + (1/2/self.__Beta -1)*self.__Xdotdot) )

    def get_elastic_energy(self):
        """
        returns : sum(0.5 * U.transposed * K * U)
        """

        return 0.5 * np.dot(
            self.get_dof_solution("all"),
            self.__StiffMatrix * self.get_dof_solution("all"),
        )

    def get_nodal_elastic_energy(self):
        """
        returns : 0.5 * K * U . U
        """

        E = (
            0.5
            * self.get_dof_solution("all").transpose()
            * self.get_A()
            * self.get_dof_solution("all")
        )

        E = np.reshape(E, (3, -1)).T

        return E

    def get_kinetic_energy(self):
        """
        returns : 0.5 * Udot.transposed * M * Udot
        """

        return 0.5 * np.dot(self.__Xdot, self.__MassMatrix * self.__Xdot)

    def get_damping_power(self):
        """
        returns : Udot.transposed * C * Udot
        The damping disspated energy can be approximated by:
                Edis = DampingPower * TimeStep
        or
                Edis = scipy.integrate.cumtrapz(t,DampingPower)
        """
        return np.dot(self.__Xdot, self.__DampMatrix * self.__Xdot)

    def get_external_force_work(self):
        """
        with (KU + CU_dot + MU_dot_dot) = Fext
        this function returns sum(Fext.(U-Uold))
        """
        K = self.__StiffMatrix
        M = self.__MassMatrix
        C = self.__DampMatrix
        return np.sum(
            (K * self.get_X() + C * self.get_velocity() + M * self.get_acceleration())
            * (self.get_X() - self.__Xold)
        )

    def update_stiffness(self, stiffness_assembly):
        if isinstance(stiffness_assembly, str):
            stiffness_assembly = Assembly.get_all()[stiffness_assembly]
        self.__StiffMatrix = stiffness_assembly.get_global_matrix()
        self.__UpdateA()

    GetElasticEnergy = deprecated_alias(get_elastic_energy, "GetElasticEnergy")
    GetNodalElasticEnergy = deprecated_alias(
        get_nodal_elastic_energy, "GetNodalElasticEnergy"
    )
    GetKineticEnergy = deprecated_alias(get_kinetic_energy, "GetKineticEnergy")
    get_DampingPower = deprecated_alias(get_damping_power, "get_DampingPower")
    GetDampingPower = deprecated_alias(get_damping_power, "GetDampingPower")
    GetExternalForceWork = deprecated_alias(
        get_external_force_work, "GetExternalForceWork"
    )
    updateStiffness = deprecated_alias(update_stiffness, "updateStiffness")
    UpdateStiffness = deprecated_alias(update_stiffness, "UpdateStiffness")


class Newmark(_NewmarkBase, Problem):
    pass
