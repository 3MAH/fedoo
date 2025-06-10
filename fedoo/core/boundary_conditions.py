import numpy as np
from scipy import sparse
from numbers import Number
from itertools import chain

# from fedoo.core.base import BCBase
# from fedoo.pgd.SeparatedArray import *

# =============================================================
# Base class for boundary conditions (BC)
# =============================================================


class BCBase:
    """Base class for BC (boundary conditions) objects."""

    __dic = {}

    def __init__(self, name=""):
        assert isinstance(name, str), "name must be a string"
        self.__name = name

        if name != "":
            BCBase.__dic[self.__name] = self

        self._keep_at_end = (
            False  # set to True if bc need to be generated after standard bc
        )
        self._update_during_inc = False  # set to True if bc need to be generated during newton raphson increment

    def __class_getitem__(cls, item):
        return cls.__dic[item]

    def str_condensed(self):
        """Return a condensed one line str describing the object"""
        if self.name == "":
            return "{}".format(self.bc_type)
        else:
            return "{} (name = '{}')".format(self.bc_type, self.name)

    @property
    def name(self):
        return self.__name

    @staticmethod
    def get_all():
        return BCBase.__dic


class ListBC(BCBase):
    """Class that define a list of ordered elementary boundary conditions (the bc are applied in the list order).
    Derived from the python list object"""

    def __init__(self, l=None, name=""):
        BCBase.__init__(self, name)
        self._problem = None
        self.bc_type = "ListBC"
        if l is None:
            self.data = []
        else:
            self.data = l  # list containing the bc
        self.data_end = []  # bc that needs to be placed at the end of the list

    def __iter__(self):
        return chain(self.data, self.data_end)  # build a generator object

    def __repr__(self):
        list_str = ["List of boundary conditions:"]
        if self.name != "":
            list_str.append("name = '{}'".format(self.name))
        list_str.append("number of bc = " + str(len(self)) + "\n")

        list_str.extend(
            [str(i) + ": " + bc.str_condensed() for i, bc in enumerate(self)]
        )

        return "\n".join(list_str)

    def __len__(self):
        return len(self.data) + len(self.data_end)

    def __getitem__(self, items):
        if isinstance(items, int):
            if items < len(self.data):
                return self.data[items]
            else:
                return self.data_end[items - len(self.data)]

        elif isinstance(items, str):
            for bc in self.data:
                if bc.name == items:
                    return bc
            for bc in self.data_end:
                if bc.name == items:
                    return bc

        return (self.data + self.data_end)[items]

    def __delitem__(self, idx):
        self.remove(idx)

    def str_condensed(self):
        """Return a condensed one line str describing the object"""
        if self.name == "":
            return "ListBC -> list of {} bc".format(len(self))
        else:
            "ListBC (name = {}) -> list of {} bc".format(self.name, len(self))

    # def insert(self, index, bc):
    #     if self._problem is not None:
    #         bc.initialize(self._problem)
    #     self.data.insert(index, bc)

    def append(self, bc):
        if self._problem is not None:
            bc.initialize(self._problem)
        if bc._keep_at_end:
            self.data_end.append(bc)
        else:
            self.data.append(bc)

        if bc._update_during_inc:
            self._update_during_inc = True

    def extend(self, iterable):
        if self._problem is not None:
            for bc in iterable:
                bc.initialize(self._problem)
        if hasattr(iterable, "_keep_at_end") and iterable._keep_at_end:
            self.data_end.extend(iterable)
        else:
            self.data.extend(iterable)

        if hasattr(iterable, "_update_during_inc") and iterable._update_during_inc:
            self._update_during_inc = True

    def remove(self, bc):
        if isinstance(bc, int):
            if bc < len(self.data):
                del self.data[bc]
            else:
                del self.data_end[bc - len(self.data)]
        elif isinstance(bc, str):
            self.data = [item for item in self.data if item.name != bc]
            self.data_end = [item for item in self.data_end if item.name != bc]
        else:
            try:
                self.data.remove(bc)
            except:
                self.data_end.remove(bc)

        # recompute self._update_during_inc
        if self._update_during_inc:
            self._update_during_inc = False
            for bc in self:
                if bc._update_during_inc:
                    self._update_during_inc = True
                    break

    def add(self, *args, **kargs):
        """
        Add a boundary condition to the ListBC object, and then to the associated problem
        if there is one.

        Two possible use:

        * ListBC.add(bc), where bc is any BC object.

          Add the object bc at the end of the list. (equivalent to ListBC.append(bc))

        * ListBC.add(bc_type, node_set, variable, value, time_func=None, start_value=None, name=""):

          Define some standard boundary conditions (Dirichlet of Neumann BC), using
          the BoundaryCondition.create staticmethod, and add it at the end of the list.

          Same agruments as the :py:meth:`fedoo.BoundaryCondition.create` static method.
        """
        if len(args) == 1:  # assume arg[0] is a boundary condition object
            self.append(args[0])
            return args[0]
        else:  # define a boundary condition
            kargs["space"] = self._problem.space
            bc = BoundaryCondition.create(*args, **kargs)
            self.append(bc)
            return bc

    def mpc(self, *args, **kargs):
        """
        Add a linear multi-point constraint to the list of boundary conditions.

        This method is equivalent to use the method add with a MPC object:

          >>> boundary_conditions.add(fd.MPC(*args, **kargs))

        The arguments are the same as the :py:class:`fedoo.MPC` constructor.

        """
        self.append(MPC(*args, **kargs))

    def initialize(self, problem):
        for bc in self:
            bc.initialize(problem)

    def generate(self, problem, t_fact=1, t_fact_old=None):
        # return a generator function (the generate method will be called only when required)
        for listbc in self:
            for bc in listbc.generate(problem, t_fact, t_fact_old):
                yield bc

        # return chain(*(bc.generate(problem, t_fact, t_fact_old) for bc in self)) #use chain instead of sum would be better ?
        # return sum((bc.generate(problem, t_fact, t_fact_old) for bc in self), []) #use chain instead of sum would be better ?

    def list_all(self):
        # return a generator function (the generate method will be called only when required)
        for listbc in self:
            if isinstance(listbc, ListBC):
                for bc in listbc.list_all():
                    yield bc
            else:
                yield listbc


class BoundaryCondition(BCBase):
    """
    Classe de condition limite

    Advice: For PGD problems, it is more efficient to define zeros values BC first  (especially for MPC)
    """

    def __init__(
        self,
        bc_type,
        node_set,
        variable,
        value,
        time_func=None,
        start_value=None,
        name="",
    ):
        """
        Define some standard boundary conditions (Dirichlet of Neumann BC)

        Notes
        ----------
        * The created object is not automatically associated to a problem.
        In most cases, the Problem.bc.add method should be prefered to define
        standard boundary conditions.
        * To avoid errors, it is recommanded to avoid associating a BoundaryCondition
        object to several problems.

        Parameters
        ----------
        bc_type : str
            Type of boundary condition: 'Dirichlet' or 'Neumann'.
        variable : str
            Variable name (str) over which the bc is applied.
        value : scalar or scalar array
            Final value of the variable (Dirichlet) or the adjoint variable (Neumann).
            e.g. the adjoint variable associated to displacement is the force.
        node_set : list of int or str
            list of node index (list of int) or name of a node_set
            associated to the reference mesh (str)
        time_func : function
            Function that gives the temporal evolution of the BC value
            (applyed as a factor to the specified BC).
            The function should be y=f(x) where x in [0,1] and y in [0,1].
            For x, 0 denote the begining of the step and 1 the end.
            By default, a linear evolution of the value is considered.
        start_value : float, array or None (default)
            if None, the start_value is keep to the current state.
            if scalar value: The start_value is the same for all dof defined in BC
            if array: the len of the array should be = to the number of dof defined in the BC
        name : str (default = "")
            Define the name of the Boundary Conditions.
        """
        assert bc_type in [
            "Dirichlet",
            "Neumann",
        ], "The type of Boundary conditions should be either 'Dirichlet' or 'Neumann'"

        BCBase.__init__(self, name)

        if time_func is None:

            def time_func(t_fact):
                return t_fact

        self.time_func = time_func

        # can be a float or an array or None ! if DefaultInitialvalue is None, initialvalue can be modified by the Problem
        self._start_value_default = self.start_value = start_value

        self.bc_type = bc_type
        if not (isinstance(variable, str)):
            assert 0, "variable should be a str"

        self.variable = None  # need to be initialized
        self.variable_name = variable
        self.value = value  # can be a float or an array !

        if isinstance(node_set, Number):
            node_set = [node_set]
        self.node_set = self.node_set_name = node_set

    @staticmethod
    def create(
        bc_type,
        node_set,
        variable,
        value,
        time_func=None,
        start_value=None,
        name="",
        space=None,
    ):
        """
        Create one or several standard boundary conditions (Dirichlet or Neumann BC)

        Parameters
        ----------
        bc_type : str
            Type of boundary condition: 'Dirichlet' or 'Neumann'.
        variable : str
            Variable name (str) or Vector name (str) or list of variable/vector name (list of str)
            Variable over which the bc is applied.
            If a list of variable is given, apply the same BC for each variable.
        value : scalar or scalar array
            Final value of the variable (Dirichlet) or the adjoint variable (Neumann).
            e.g. the adjoint variable associated to displacement is the force.
        node_set : list of int or str
            list of node index (list of int) or name of a node_set
            associated to the reference mesh (str).
        time_func : function
            Function that gives the temporal evolution of the BC value
            (applyed as a factor to the specified BC).
            The function should be under the form y=f(x) where x in [0,1] and y in [0,1].
            For x, 0 denote the begining of the step and 1 the end.
            By default, a linear evolution of the value is considered.
        start_value : float, array or None (default)
            if None, the start_value is keep to the current state.
            if scalar value: The start_value is the same for all dof defined in BC
            if array: the len of the array should be = to the number of dof defined in the BC
        name : str (default = "")
            Define the name of the Boundary Conditions.

        Return
        ----------
        If only one variable is specified, return a BoundaryCondition object,
        if several variables are specified, return a ListBC object.
        """
        if space is not None:
            if isinstance(variable, str) and variable not in space.list_variables():
                # we assume that Var is a Vector
                try:
                    variable = [
                        space.variable_name(var_rank)
                        for var_rank in space.get_rank_vector(variable)
                    ]
                except:
                    raise NameError("Unknown variable name")

        if isinstance(variable, list):
            if np.isscalar(value):
                value = [value for var in variable]

            return ListBC(
                [
                    BoundaryCondition.create(
                        bc_type,
                        node_set,
                        var,
                        value[i],
                        time_func,
                        start_value,
                        name,
                        space,
                    )
                    for i, var in enumerate(variable)
                ]
            )
        else:
            return BoundaryCondition(
                bc_type, node_set, variable, value, time_func, start_value, name
            )

    def str_condensed(self):
        """Return a condensed one line str describing the object"""
        if self.name == "":
            res = ["{} ->".format(self.bc_type)]
        else:
            res = ["{} (name = '{}') -> ".format(self.bc_type, self.name)]

        res.append("'{}' for".format(self.variable_name))

        if isinstance(self.node_set_name, str):
            res.append("node_set '{}'".format(self.node_set_name))
        else:  # iterable
            res.append("{} nodes".format(len(self.node_set_name)))

        if np.isscalar(self.value):
            res.append("set to {}".format(self.value))
        else:
            res.append("set to array values")

        return " ".join(res)

    def __repr__(self):
        res = ["{} boundary condition:".format(self.bc_type)]
        if self.name != "":
            res.append("name = '{}'".format(self.name))
        res.append("var = '{}'".format(self.variable_name))

        if isinstance(self.node_set_name, str):
            res.append("node_set = '{}'".format(self.node_set_name))
        else:  # iterable
            res.append("n_nodes = {}".format(len(self.node_set_name)))

        if np.isscalar(self.value):
            res.append("value = {}".format(self.value))
        else:
            res.append("value = array")

        return "\n".join(res)

    def initialize(self, problem):
        self.variable = problem.space.variable_rank(self.variable_name)

        if isinstance(self.node_set_name, str):
            # must be a string defining a set of nodes
            self.node_set = problem.mesh.node_sets[self.node_set_name]

        if hasattr(problem.mesh, "GetListMesh"):  # associated to pgd problem
            self.pgd = True
        else:
            self.pgd = False
            # must be a np.array  #Not for PGD
            self.node_set = np.asarray(self.node_set, dtype=int)

    def generate(self, problem, t_fact, t_fact_old=None):
        self._current_value = self.get_value(t_fact, t_fact_old)

        if not (self.pgd):
            self._dof_index = (
                self.variable * problem.mesh.n_nodes + self.node_set
            ).astype(int)

        return [self]

    def _get_factor(self, t_fact=1, t_fact_old=None):
        # return the time factor applied to the value of boundary conditions
        if (
            t_fact_old is None or self.bc_type == "Neumann"
        ):  # for Neumann, the force is applied in any cases
            return self.time_func(t_fact)
        else:
            return self.time_func(t_fact) - self.time_func(t_fact_old)

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
            if (
                self.bc_type == "Neumann"
            ):  # for Neumann, the true value of force is applied
                return factor * (self.value - self.start_value) + self.start_value
            else:  # return the incremental value
                return factor * (self.value - self.start_value)

    def get_true_value(self, t_fact=1, t_fact_old=None):
        """
        Return the true bc value. For incremental problems, this function return
        the real bc values, which is in general different to the incremental values
        given by the get_value function.

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
        The value of the dirichlet of neumann boundary condition at the speficied time evolution.

        """
        factor = self._get_factor(t_fact, t_fact_old)
        if factor == 0:
            return 0
        elif self.start_value is None:
            return factor * self.value
        else:  # in case there is an initial value
            return factor * (self.value - self.start_value) + self.start_value

    # def change_index(self,newIndex):
    #     self.__Index = np.array(newIndex).astype(int) # must be a np.array

    def change_value(self, newvalue, start_value=None, time_func=None):
        # if start_value == 'Current', keep current value as initial values (change of step)
        # if start_value is None, don't change the initial value
        if start_value == "Current":
            self.start_value = self.value
        elif start_value is not None:
            self.start_value = start_value
        if time_func is not None:
            self.time_func = time_func
        self.value = newvalue  # can be a float or an array !


class MPC(BCBase):
    """
    Class that define multi-point constraints
    """

    def __init__(
        self,
        list_node_sets,
        list_variables,
        list_factors,
        constant=None,
        time_func=None,
        start_constant=None,
        name="",
    ):
        """
        Create a linear multi-point constraint object
        To create a MPC of the equation
        4*Ux_12 - 2*Uz_15 = 7 (Ux_12 is the displacement along x of the 12th node)
        use the following MPC:

          >>> MPC([12,15], ['Disp_X', 'Disp_Z'], [4, -2], 7)

        Parameters
        ----------
        list_node_sets : list of str or list of list of int (or array)
            List of node_set names (list of str) or list of node indexes (list of list of int)
        list_variables : list of str, or list of int
            list of variable names (list of str) or list of variable ranks (list of int)
        list_factors : list of scalars or numpy array
            list of factor (MPC).
            To define several mpc at once, it is possible to give an array of factors, where
            each line is associated to a signe mpc.
        constant : scalar, optional
            constant value on the MPC equation
            if not specified, no constant value.
        time_func : function
            Function that gives the temporal evolution of the constant value.
            The function should be on the form:
                y=f(x) where x in [0,1] and y in [0,1].
                For x, 0 denote the begining of the step and 1 the end.
        start_constant : float, array or None (default)
            if None, the start_constant is keep to the current state.
            if scalar value: The start_value is the same for all dof defined in BC
            if array: the len of the array should be = to the number of dof defined in the BC
        name : str, optional
            Define an name for the Boundary Conditions. Default is "". The same name may be used for several BC.

        Notes
        -------

          * To define many MPC in one operation, use array where each line define a single MPC
          * In fedoo, the mpc are applied with an elimination method.
            The first dof specified in list_node_sets[0] and list_variable[0]
            is the slave dof and will eliminated from the linear system. No other
            boundary conditions should be applied on this dof.
        """
        BCBase.__init__(self, name)
        self.bc_type = "MPC"

        self._start_value_default = None  # not used for MPC

        if time_func is None:

            def time_func(t_fact):
                return t_fact

        self.time_func = time_func

        # can be a float or an array or None ! if DefaultInitialvalue is None, start_value can be modified by the Problem
        # self._start_constant_default = self.start_constant = start_constant

        self.list_variables = list_variables
        self.list_factors = list_factors
        self.list_node_sets = list_node_sets
        self.constant = constant

    def __repr__(self):
        res = ["Multi Point Constraint"]
        # res.append(["list_variables = {}".format(self.list_variables)])
        return "\n".join(res)

    def initialize(self, problem):
        # list_variables should be a list or a numpy array
        if isinstance(self.list_variables[0], str):
            self.list_variables = [
                problem.space.variable_rank(v) for v in self.list_variables
            ]
        if isinstance(self.list_node_sets[0], str):
            self.list_node_sets = [
                problem.mesh.node_sets[n_set] for n_set in self.list_node_sets
            ]

        if hasattr(problem.mesh, "GetListMesh"):  # associated to pgd problem
            self.pgd = True
        else:
            self.pgd = False
            self.list_node_sets = np.asarray(self.list_node_sets, dtype=int)

    def generate(self, problem, t_fact, t_fact_old=None):
        # # Node index for master DOF (not eliminated DOF in MPC)
        # self.__IndexMaster = np.array(Index[1:], dtype=int)
        # # Node index for slave DOF (eliminated DOF) #use SetOf for PGD
        # self.__Index = np.array(Index[0], dtype=int)

        n_nodes = problem.mesh.n_nodes
        self._dof_index = self.list_node_sets + np.c_[self.list_variables] * n_nodes
        # self._dof_index = [(
        #     self.list_variables[i]*n_nodes + self.list_node_sets[i]).astype(int)
        #     for i in range(len(self.list_variables))]

        if (
            self.constant is not None
        ):  # need to be checked, especialy for varying constants
            # should be a numeric value or a 1D array for multiple MPC

            value = -self.constant / self.list_factors[0]

            factor = BoundaryCondition._get_factor(self, t_fact, t_fact_old)
            if factor == 0:
                self._current_value = 0
            elif self.start_value is None:
                self._current_value = factor * value
            else:  # in case there is an initial value
                start_value = -self.start_constant / self.list_factors[0]
                self._current_value = factor * (value - start_value)

        else:
            self._current_value = 0

        # does not include the master node coef = 1
        self._factors = -np.asarray(self.list_factors[1:]) / self.list_factors[0]

        return [self]


if __name__ == "__main__":
    pass
