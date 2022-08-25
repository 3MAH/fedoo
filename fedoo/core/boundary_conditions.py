import numpy as np
from scipy import sparse

from fedoo.core.base import BCBase
# from fedoo.pgd.SeparatedArray import *


class UniqueBoundaryCondition():
    """
    Classe de condition limite

    Advice: For PGD problems, it is more efficient to define zeros values BC first  (especially for MPC)
    """

    def __init__(self, bc_type, variable, value, node_set, time_func=None, start_value=None, name=""):
        """
        Define some boundary conditions        

        Parameters
        ----------
        bc_type : str
            Type of boundary conditions : 'Dirichlet' or 'Neumann'.
        variable : str
            variable name (str) over which the bc is applied
        value : scalar or list of scalars
            Variable final value (Dirichlet) or force value (Neumann)
        node_set : list of int or str
            For FEM Problem: Nodes Index (list of int) or str (node set)
            For PGD Problem: node set (type str) defining a set of Nodes of the reference mesh
        time_func : function
            Function that gives the temporal evolution of the BC value (applyed as a factor to the specified BC). The function y=f(x) where x in [0,1] and y in [0,1]. For x, 0 denote the begining of the step and 1 the end.
        start_value : float, array or None
            if None, the start_value is keep to the current state.
            if scalar value: The start_value is the same for all dof defined in BC
            if array: the len of the array should be = to the number of dof defined in the BC

            Default: None
        name : str, optional
            Define an name for the Boundary Conditions. Default is "".

        Returns
        -------
        None.
        """
        assert bc_type in [
            'Dirichlet', 'Neumann'], "The type of Boundary conditions should be either 'Dirichlet' or 'Neumann'"

        BCBase.__init__(self, name)

        if time_func is None:
            def time_func(t_fact): return t_fact

        self.time_func = time_func

        # can be a float or an array or None ! if DefaultInitialvalue is None, initialvalue can be modified by the Problem
        self._start_value_default = self.start_value = start_value

        self.bc_type = bc_type
        if not(isinstance(variable, str)):
            assert 0, 'variable should be a str'
        
        self.variable = variable
        self.value = value  # can be a float or an array !
        self.node_set = node_set

    def initialize(self, problem):
        if isinstance(self.variable, str):
            self.variable = problem.space.variable_rank(self.variable)
        
        if isinstance(self.node_set, str):
            self.node_set = problem.mesh.node_sets[self.node_set]  # must be a string defining a set of nodes

        self.node_set = np.asarray(self.node_set, dtype = int)  # must be a np.array  #Not for PGD


    def generate(self, problem, t_fact, t_fact_old=None):
        self._dof_index = (
            self.variable*problem.mesh.n_nodes + self.node_set).astype(int)

        self._current_value = self.get_value(t_fact, t_fact_old)

        return [self]

    def generate_pgd(self, problem, t_fact, t_fact_old=None):
        pass

    def _get_factor(self, t_fact=1, t_fact_old=None):
        # return the time factor applied to the value of boundary conditions
        if t_fact_old is None or self.bc_type == 'Neumann':  # for Neumann, the force is applied in any cases
            return self.time_func(t_fact)
        else:
            return self.time_func(t_fact)-self.time_func(t_fact_old)

    def get_value(self, t_fact=1, t_fact_old=None):
        """
        Return the bc value to enforce. For incremental problems, this function return
        the increment for Dirichlet conditions and the full value for Neumann conditions.

        Parameters
        ----------
        t_fact : float between 0 and 1.
            The time factor. t_fact = 0 at the beginning of the increment (start value)
            t_fact = 1 at the end. The default is 1.
        t_fact_old : float between 0 and 1. 
            The time factor at the previous iteration (only used for incremental problems).
            The default is None.

        Returns
        -------
        The value to enforce for the specified iteration at the speficied time evolution.

        """
        factor = self._get_factor(t_fact, t_fact_old)
        if factor == 0:
            return 0
        elif self.start_value is None:
            return factor * self.value
        else:  # in case there is an initial value
            if self.bc_type == 'Neumann':  # for Neumann, the true value of force is applied
                return factor * (self.value - self.start_value) + self.start_value
            else:  # return the incremental value
                return factor * (self.value - self.start_value)

    # def change_index(self,newIndex):
    #     self.__Index = np.array(newIndex).astype(int) # must be a np.array

    def change_value(self, newvalue, start_value=None, time_func=None):
        # if start_value == 'Current', keep current value as initial values (change of step)
        # if start_value is None, don't change the initial value
        if start_value == 'Current':
            self.start_value = self.value
        elif start_value is not None:
            self.start_value = start_value
        if time_func is not None:
            self.time_func = time_func
        self.value = newvalue  # can be a float or an array !


class MPC(UniqueBoundaryCondition):
    """
    Class that define multi-point constraints
    """

    def __init__(self, list_variables, list_factors, list_node_sets, constant=None, time_func=None, start_constant=None, name=""):
        """
        Define some boundary conditions        

        Parameters
        ----------        
        list_variables : list of str, or list of int
            list of variable names (list of str) or list of variable ranks (list of int)
        list_factors : list of scalars
            list of factor (MPC)
        Index : list of int, str, list of list of int, list of str
            For FEM Problem: list Node Indexes (list of list of int) 
            For PGD Problem: list of SetOfname (str)
        constant : scalar, optional
            constant value on the MPC equation
        time_func : function
            Function that gives the temporal evolution of the constant value. The function y=f(x) where x in [0,1] and y in [0,1]. For x, 0 denote the begining of the step and 1 the end.
        start_constant : float, array or None
            if None, the start_constant is keep to the current state.
            if scalar value: The start_value is the same for all dof defined in BC
            if array: the len of the array should be = to the number of dof defined in the BC

            Default: None
        name : str, optional
            Define an name for the Boundary Conditions. Default is "". The same name may be used for several BC.

        Returns
        -------
        None.

        Remark  
        -------
        To define many MPC in one operation, use array where each line define a single MPC        
        """
        BCBase.__init__(self, name)
        self.bc_type = 'MPC'


        self._start_value_default  = None #not used for MPC
        
        if time_func is None:
            def time_func(t_fact): return t_fact

        self.time_func = time_func

        # can be a float or an array or None ! if DefaultInitialvalue is None, start_value can be modified by the Problem
        # self._start_constant_default = self.start_constant = start_constant

        self.list_variables = list_variables
        self.list_factors = list_factors
        self.list_node_sets = list_node_sets        
        self.constant = constant
        
        # self.__Var = list_variables[0]  # variable for slave DOF (eliminated DOF)
        # # Var for master DOF (not eliminated DOF in MPC)
        # self.__VarMaster = list_variables[1:]

        # if isinstance(Index[0], str):  # PGD problem
        #     # SetOfname decribing node indexes for slave DOF (eliminated DOF) #use SetOf for PGD
        #     self.__SetOfname = Index[0]
        #     self.__SetOfnameMaster = Index[1:]
        #     if constant is not 0:
        #         raise NameError(
        #             "MPC boundary condition with PGD problem isn't compatible with a non zero constant value")
        #     else:
        #         self.value = 0
        # else:  # FEM Problem
        

        
        # # Node index for master DOF (not eliminated DOF in MPC)
        # self.__IndexMaster = np.array(Index[1:], dtype=int)
        # # Node index for slave DOF (eliminated DOF) #use SetOf for PGD
        # self.__Index = np.array(Index[0], dtype=int)
        
        # if constant is not 0:
        #     # should be a numeric value or a 1D array for multiple MPC
        #     self.value = -constant/list_factors[0]
        # else:
        #     self.value = 0

        # # does not include the master node coef = 1
        # self.__Fact = -np.array(factor[1:])/factor[0]

    def initialize(self,problem):
        # list_variables should be a list or a numpy array
        if isinstance(self.list_variables[0], str):
            self.list_variables = [problem.space.variable_rank(v) for v in self.list_variables]
        if isinstance(self.list_node_sets[0], str):
            self.list_node_sets = [problem.mesh.node_sets[n_set] for n_set in self.list_node_sets]

    def generate(self, problem, t_fact, t_fact_old=None):
        # # Node index for master DOF (not eliminated DOF in MPC)
        # self.__IndexMaster = np.array(Index[1:], dtype=int)
        # # Node index for slave DOF (eliminated DOF) #use SetOf for PGD
        # self.__Index = np.array(Index[0], dtype=int)
        
        n_nodes = problem.mesh.n_nodes
        self._dof_index = np.asarray(self.list_node_sets, dtype=int) \
                          + np.c_[self.list_variables]*n_nodes
        # self._dof_index = [(
        #     self.list_variables[i]*n_nodes + self.list_node_sets[i]).astype(int)
        #     for i in range(len(self.list_variables))]
        
        if self.constant is not None:
            # should be a numeric value or a 1D array for multiple MPC
            
            value = -self.constant/self.list_factors[0]                                  
            
            factor = self._get_factor(t_fact, t_fact_old)
            if factor == 0:
                self._current_value = 0
            elif self.start_value is None:
                self._current_value = factor * value
            else:  # in case there is an initial value
                start_value = -self.start_constant/self.list_factors[0]
                self._current_value = factor * (value - start_value)
                        
        else:
            self._current_value = 0

        # does not include the master node coef = 1
        self._factors = -np.asarray(self.list_factors[1:])/self.list_factors[0]
                
        return [self]


    # @property
    # def variable(self):
    #     return self.__Var

    # @property
    # def SetOfname(self):
    #     return self.__SetOfname

    # @property
    # def Index(self):
    #     return self.__Index

    # @property
    # def IndexMaster(self):
    #     try:
    #         return self.__IndexMaster
    #     except:
    #         raise NameError('Master Index only defined for MPC boundary type')

    # @property
    # def SetOfnameMaster(self):
    #     try:
    #         return self.__SetOfnameMaster
    #     except:
    #         raise NameError(
    #             'SetOfnameMaster is only defined for MPC boundary type')

    # @property
    # def VariableMaster(self):
    #     try:
    #         return self.__VarMaster
    #     except:
    #         raise NameError(
    #             'Master Variable only defined for MPC boundary type')


if __name__ == "__main__":
    pass
