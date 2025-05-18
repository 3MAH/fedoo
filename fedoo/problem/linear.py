import numpy as np
from fedoo.core.assembly import Assembly
from fedoo.core.problem import Problem


class _LinearBase:
    def __init__(self, assembly: Assembly, name: str = "MainProblem"):
        if isinstance(assembly, str):
            assembly = Assembly.get_all()[assembly]

        self.nlgeom = False
        assembly.initialize(self)
        A = assembly.get_global_matrix()
        B = 0
        D = assembly.get_global_vector()
        self.__assembly = assembly

        super().__init__(A, B, D, assembly.mesh, name)

    def get_disp(self, name="all"):
        """Return the displacement components.


        Parameters
        ----------
        name : str, optional
            Name of the variable to return. For instance, if name == 'DispX'
            return only the X component of displacement.

        Returns
        -------
        numpy.ndarray
        """
        if name == "all":
            name = "Disp"
        return self.get_dof_solution(name)

    def get_rot(self, name="all"):
        """Return the rotation components.


        Parameters
        ----------
        name : str, optional
            Name of the variable to return. For instance, if name == 'RotX'
            return only the X component of rotation.

        Returns
        -------
        numpy.ndarray
        """
        if name == "all":
            name = "Rot"
        return self.get_dof_solution(name)

    def reset(self):
        self.__assembly.reset()

        self.set_A(self.__assembly.get_global_matrix())  # tangent stiffness
        self.set_D(self.__assembly.get_global_vector())
        B = 0

    def update(self, dtime=1, compute="all"):
        """
        Assemble the matrix including the following modification:
            - New initial Stress
            - New initial Displacement
            - Modification of the mesh
            - Change in constitutive law (internal variable)
        Update the problem with the new assembled global matrix and global vector
        """
        out_values = self.__assembly.update(self, compute)
        self.set_A(self.__assembly.get_global_matrix())
        self.set_D(self.__assembly.get_global_vector())
        return out_values

    def solve(self, **kargs):
        # Solve and update weakform (compute stress for instance) without updating global matrix
        # to avoid update weakform, use updateWF = True
        updateWF = kargs.pop("updateWF", True)
        self.init_bc_start_value()
        self.apply_boundary_conditions()
        if np.isscalar(self.get_X()) and self.get_X() == 0:
            Problem.solve(self)
        else:
            # solve increment
            X0 = self.get_X().copy()
            Problem.solve(self)
            self._Problem__X += X0

        if updateWF == True:
            self.update(compute="none")

    def change_assembly(self, assembling, update=True):
        """
        Modify the assembly associated to the problem and update the problem (see Assembly.update for more information)
        """
        if isinstance(assembling, str):
            assembling = Assembly[assembling]

        self.__assembly = assembling
        if update:
            self.update()

    def GetElasticEnergy(self):  # only work for classical FEM
        """
        returns : sum (0.5 * U.transposed * K * U)
        """

        return sum(
            0.5
            * self.get_dof_solution("all").transpose()
            * self.get_A()
            * self.get_dof_solution("all")
        )

    def GetNodalElasticEnergy(self):
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

    @property
    def assembly(self):
        return self.__assembly


class Linear(_LinearBase, Problem):
    """Class that defines linear problems.

    This simple class allow to build a linear problem from an assembly object.
    The discretized problem is written under the form:
    A*X = B+D
    where:
     * A is a square matrix build with the associated assembly object calling
         assembly.get_global_matrix()
     * X is the column vector containing the degrees of freedom (solution after solving)
     * B is a column vector used to set Neumann boundary conditions
     * D is a column vector build with the associated assembly object calling
         assembly.get_global_vector()

    Parameters
    ----------
    assembly: Assembly like object
        Assembly that construct the matrix A and column vector D.
    name: str
        name of the problem.
    """

    pass
