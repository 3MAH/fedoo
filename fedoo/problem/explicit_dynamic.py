import numpy as np
from fedoo.core.assembly import Assembly
from fedoo.core.problem import Problem
from fedoo.util.deprecation import deprecated_alias
import scipy.sparse as sparse


class _ExplicitDynamicBase:
    def __init__(
        self,
        StiffnessAssembly,
        MassAssembly,
        TimeStep,
        DampingAssembly=None,
        name="MainProblem",
    ):
        if isinstance(StiffnessAssembly, str):
            StiffnessAssembly = Assembly.get_all()[StiffnessAssembly]

        if isinstance(MassAssembly, str):
            MassAssembly = Assembly.get_all()[MassAssembly]

        if isinstance(DampingAssembly, str):
            DampingAssembly = Assembly.get_all()[DampingAssembly]

        A = 1 / (TimeStep**2) * MassAssembly.get_global_matrix()
        B = 0
        D = 0

        self.__Xold = self._new_vect_dof(A)  # displacement at the previous time step
        self.__Xdot = self._new_vect_dof(A)
        self.__Xdotdot = self._new_vect_dof(A)

        self.__TimeStep = TimeStep
        self.__MassLuming = False

        self.__MassMatrix = MassAssembly.get_global_matrix()
        self.__StiffMatrix = StiffnessAssembly.get_global_matrix()
        if DampingAssembly is None:
            self.__DampMatrix = None
        else:
            self.__DampMatrix = DampingAssembly.get_global_matrix()

        super().__init__(A, B, D, StiffnessAssembly.mesh, name)

    def __UpdateA(self):  # internal function to be used when modifying M
        # if MassLumping == True, A is a vector representing the diagonal value
        self.set_A(self.__MassMatrix / (self.__TimeStep**2))

    def update_stiffness(
        self, StiffnessAssembly
    ):  # internal function to be used when modifying the siffness matrix
        if isinstance(StiffnessAssembly, str):
            StiffnessAssembly = Assembly.get_all()[StiffnessAssembly]

        self.__StiffMatrix = StiffnessAssembly.get_global_matrix()

    def mass_lumping(self):  # internal function to be used when modifying M
        self.__MassLuming = True
        if len(self.__MassMatrix.shape) == 2:
            self.__MassMatrix = np.array(self.__MassMatrix.sum(1))[:, 0]
            self.__UpdateA()

    updateStiffness = deprecated_alias(update_stiffness, "updateStiffness")
    UpdateStiffness = deprecated_alias(update_stiffness, "UpdateStiffness")
    MassLumping = deprecated_alias(mass_lumping, "MassLumping")

    def get_X(self):
        return self.get_dof_solution("all")

    def get_velocity(self):
        return self.__Xdot

    get_Xdot = deprecated_alias(get_velocity, "get_Xdot")

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
        if len(self.__MassMatrix.shape) == 1:
            self.__DampMatrix = (
                alpha * sparse.diags(self.__MassMatrix, format="csr")
                + beta * self.__StiffMatrix
            )
        else:
            self.__DampMatrix = alpha * self.__MassMatrix + beta * self.__StiffMatrix
        self.__UpdateA()

    SetRayleighDamping = deprecated_alias(set_rayleigh_damping, "SetRayleighDamping")

    def initialize(self):
        D = (
            1
            / (self.__TimeStep**2)
            * self.__MassMatrix
            * (self.__Xold + self.__TimeStep * self.__Xdot)
            - self.__StiffMatrix * self.__Xold
        )
        if self.__DampMatrix is not None:
            D -= self.__DampMatrix * self.__Xdot

        self.set_D(D)

    def update(self):
        self.__Xdot = (Problem.get_dof_solution("all") - self.__Xold) / self.__TimeStep
        self.__Xold[:] = Problem.get_dof_solution("all")
        self.initialize()

    def get_elastic_energy(self):
        """
        returns : 0.5 * U.transposed * K * U
        """

        return 0.5 * np.dot(
            self.get_dof_solution("all"),
            self.__StiffMatrix * self.get_dof_solution("all"),
        )

    def get_kinetic_energy(self):
        """
        returns : 0.5 * Udot.transposed * M * Udot
        """

        return 0.5 * np.dot(self.__Xdot, self.__MassMatrix * self.__Xdot)

    def get_damping_power(self):
        """
        returns : Udot.transposed * C * Udot
        The damping disspated energy can be approximated by:
                Edis = cumtrapz(DampingPower * TimeStep)
        """
        return np.dot(self.__Xdot, self.__DampMatrix * self.__Xdot)

    def set_stiffness_matrix(self, e):
        self.__StiffMatrix = e

    def set_mass_matrix(self, e):
        self.__MassMatrix = e

    GetElasticEnergy = deprecated_alias(get_elastic_energy, "GetElasticEnergy")
    GetKineticEnergy = deprecated_alias(get_kinetic_energy, "GetKineticEnergy")
    get_DampingPower = deprecated_alias(get_damping_power, "get_DampingPower")
    GetDampingPower = deprecated_alias(get_damping_power, "GetDampingPower")
    SetStiffnessMatrix = deprecated_alias(set_stiffness_matrix, "SetStiffnessMatrix")
    SetMassMatrix = deprecated_alias(set_mass_matrix, "SetMassMatrix")


class ExplicitDynamic(_ExplicitDynamicBase, Problem):
    """
    Define a Centred Difference problem for structural dynamic
    For damping, the backward euler derivative is used to compute the velocity
    The algorithm come from:  Bathe KJ and Edward W, "Numerical methods in finite element analysis", Prentice Hall, 1976, pp 323-324
    """

    pass
