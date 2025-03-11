import numpy as np
from fedoo.core.assembly import Assembly
from fedoo.core.problem import Problem
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

    def updateStiffness(
        self, StiffnessAssembly
    ):  # internal function to be used when modifying the siffness matrix
        if isinstance(StiffnessAssembly, str):
            StiffnessAssembly = Assembly.get_all()[StiffnessAssembly]

        self.__StiffMatrix = StiffnessAssembly.get_global_matrix()

    def MassLumping(self):  # internal function to be used when modifying M
        self.__MassLuming = True
        if len(self.__MassMatrix.shape) == 2:
            self.__MassMatrix = np.array(self.__MassMatrix.sum(1))[:, 0]
            self.__UpdateA()

    def get_X(self):
        return self.get_dof_solution("all")

    def get_Xdot(self):
        return self.__Xdot

    def SetInitialDisplacement(self, name, value):
        """
        name is the name of the associated variable (generaly 'DispX', 'DispY' or 'DispZ')
        value is an array containing the initial displacement of each nodes
        """
        self._set_vect_component(self.__Xold, name, value)

    def SetInitialVelocity(self, name, value):
        """
        name is the name of the associated variable (generaly 'DispX', 'DispY' or 'DispZ')
        value is an array containing the initial velocity of each nodes
        """
        self._set_vect_component(self.__Xdot, name, value)

    def SetInitialAcceleration(self, name, value):
        """
        name is the name of the associated variable (generaly 'DispX', 'DispY' or 'DispZ')
        value is an array containing the initial acceleration of each nodes
        """
        self._set_vect_component(self.__Xdotdot, name, value)

    def SetRayleighDamping(self, alpha, beta):
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

    def GetElasticEnergy(self):
        """
        returns : 0.5 * U.transposed * K * U
        """

        return 0.5 * np.dot(
            self.get_dof_solution("all"),
            self.__StiffMatrix * self.get_dof_solution("all"),
        )

    def GetKineticEnergy(self):
        """
        returns : 0.5 * Udot.transposed * M * Udot
        """

        return 0.5 * np.dot(self.__Xdot, self.__MassMatrix * self.__Xdot)

    def get_DampingPower(self):
        """
        returns : Udot.transposed * C * Udot
        The damping disspated energy can be approximated by:
                Edis = cumtrapz(DampingPower * TimeStep)
        """
        return np.dot(self.__Xdot, self.__DampMatrix * self.__Xdot)

    def SetStiffnessMatrix(self, e):
        self.__StiffMatrix = e

    def SetMassMatrix(self, e):
        self.__MassMatrix = e


class ExplicitDynamic(_ExplicitDynamicBase, Problem):
    """
    Define a Centred Difference problem for structural dynamic
    For damping, the backward euler derivative is used to compute the velocity
    The algorithm come from:  Bathe KJ and Edward W, "Numerical methods in finite element analysis", Prentice Hall, 1976, pp 323-324
    """

    pass
