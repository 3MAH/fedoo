# base class
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg

from fedoo.core.assembly import Assembly
from fedoo.core.base import ProblemBase, AssemblyBase
from fedoo.core.boundary_conditions import BoundaryCondition, MPC
from fedoo.core.output import _ProblemOutput, _get_results
from fedoo.core.dataset import DataSet

import time


class Problem(ProblemBase):
    """Base class to define a problem that generate a linear system and to solve
    the linear system with some defined boundary conditions.

    The linear problem is written under the form:
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
    A: scipy sparse matrix
        Matrix that define the discretized linear system to solve.
    B: np.ndarray or 0
        if 0, B is initialized to a zeros array with de adequat shape.
    D: np.ndarray or 0
        if 0, D is ignored.
    mesh: fedoo Mesh
        mesh associated to the problem.
    name: str (default = "MainProblem")
        name of the problem.
    space: ModelingSpace(Optional)
        ModelingSpace on which the problem is defined.
    name: str
        name of the problem.
    """

    def __init__(self, A=None, B=0, D=0, mesh=None, name="MainProblem", space=None):
        # the problem is AX = B + D

        ProblemBase.__init__(self, name, space)
        self.mesh = mesh

        self.__A = A

        if np.isscalar(B) and B == 0:
            self.__B = self._new_vect_dof()
        else:
            self.__B = B

        self.__D = D

        self.__X = 0  # np.ndarray( self.n_dof ) #empty array
        self._Xbc = 0

        self._dof_slave = np.array([])
        self._dof_free = np.array([])

        # prepering output demand to export results
        self._problem_output = _ProblemOutput()

        # Option to allow mpc of dof involved in other mpc. False by default
        # Set it to True may avoid unusefull computation time. Keep it to True
        # by default for safety.
        self.enable_mpc_coupling = True

    def _new_vect_dof(
        self,
    ):  # initialize a vector (force vector for instance) whose size is n_dof
        return np.zeros(self.n_dof)

    def _set_vect_component(
        self, vector, name, value
    ):  # initialize a vector (force vector for instance) being giving the stiffness matrix
        assert isinstance(name, str), "argument error"

        if name.lower() == "all":
            vector[:] = value
        else:
            i = self.space.variable_rank(name)
            n = self.mesh.n_nodes
            vector[i * n : (i + 1) * n] = value

    def _get_vect_component(
        self, vector, name
    ):  # Get component of a vector (force vector for instance) being given the name of a component (vector or single component)
        assert isinstance(name, str), "name should be a str"

        if name.lower() == "all" or np.isscalar(vector):
            return vector

        n = self.mesh.n_nodes

        if name in self.space.list_vectors():
            vec = self.space.get_rank_vector(name)
            # vector component are assumed defined as an increment sequence (i, i+1, i+2)
            i = vec[0]  # rank of the 1rst variable of the vector
            dim = len(vec)
            # return vector.reshape(-1, n)[i : i + dim]
            return vector[i * n : (i + dim) * n].reshape(-1, n)
        elif name in self.space.list_variables():
            i = self.space.variable_rank(name)
            return vector[i * n : (i + 1) * n]
        elif name in self._global_dof:
            global_var_rank = self._global_dof._variable[name]
            start, stop = (
                self.n_node_dof
                + self._global_dof._indptr[[global_var_rank, global_var_rank + 1]]
            )
            return vector[start:stop]
        elif name in self._global_dof._vector:
            global_vector_list = self._global_dof._vector[name]
            n_comp = len(global_vector_list)
            global_var_rank = self._global_dof._variable[global_vector_list[0]]
            start, stop = (
                self.n_node_dof
                + self._global_dof._indptr[[global_var_rank, global_var_rank + n_comp]]
            )
            return vector[start:stop].reshape(n_comp, -1)
        else:
            raise ValueError(f"Variable '{name}' doesn't exist.")

    def add_output(
        self,
        filename,
        assembly,
        output_list=None,
        output_type=None,
        file_format="fdz",
        compressed=False,
        position=1,
        element_set=None,
        save_mesh=True,
    ):
        """Add output requirement for automatic saving during nlsolve.

        optionnaly, the add_output can be used without specifying the assembly with the
        arguments:

        >>> problem.add_output(filename, output_list, **kargs)

        In this case, the Assembly defined in problem.assembly will be used.

        Parameters
        ----------
        filename: str
            name of the file used to save data to disk.

        assembly : Assembly, optional
            Assembly object used to extract the results.

        output_list : list[str]
            list of result labels.

        output_type : 'Node', 'Element' or 'GaussPoint', optional
            Type of results. If None, the type of output is not converted.
            Scalar results are not concerned by this parameter.

        file_format : "fdz", "vtk", "msh", "npz", "csv", "xlsx"
            file format used to save the results. The default file format
            and recommanding one is "fdz".

        compressed : bool, default = False
            if True, the fdz data are compressed.

        position : float in [-1, 1], optional
            Normalized position in the section for shell output.
            For instance, 1 is top face define by the local z direction,
            -1 is the bottom face and 0 is the midplane

        element_set : str or list[int], optional
            set of element indices or name of an element set associated to the mesh.
            If specified, only the results restriced to the set of elements
            are extracted.

        save_mesh : bool, default = True
            If True the mesh is saved.
        """
        if output_list is None and hasattr(self, "assembly"):
            output_list = assembly
            assembly = self.assembly

        return self._problem_output.add_output(
            filename,
            assembly,
            output_list,
            output_type,
            file_format,
            compressed,
            position,
            element_set,
            save_mesh,
        )

    def save_results(self, iterOutput=None):
        self._problem_output.save_results(self, iterOutput)

    def get_results(
        self,
        *args,
        **kargs,
        # self, assemb, output_list, output_type=None, position=1, element_set=None
    ):
        """Extract some results from the current problem.

        Two list of arguments are possible:

        * get_results(assemb, output_list, output_type, position, element_set)
        * get_results(output_list, output_type, position, element_set)

        Parameters
        ----------
        assemb : Assembly, optional
            Assembly object used to extract the results.
            If assemb is not specified, the assembly associated to the problem,
            (given by the assembly property) is used.

        output_list : list[str]
            list of result labels.

        output_type : 'Node', 'Element' or 'GaussPoint', optional
            Type of results. If None, the type of output is not converted.
            Scalar results are not concerned by this parameter.

        position : float in [-1, 1], optional
            Normalized position in the section for shell output.
            For instance, 1 is top face define by the local z direction,
            -1 is the bottom face and 0 is the midplane

        element_set : str or list[int], optional
            set of element indices or name of an element set associated to the mesh.
            If specified, only the results restriced to the set of elements
            are extracted.
        """
        if len(args) > 0 and (
            isinstance(args[0], AssemblyBase)
            or (isinstance(args[0], str) and args[0] in AssemblyBase.get_all())
        ):
            assemb = args[0]
            args = args[1:]
        elif "assemb" in kargs:
            assemb = kargs.pop("assemb")
        elif hasattr(self, "assembly"):
            assemb = self.assembly
        else:
            raise TypeError("Wrong argument type.")

        return _get_results(self, assemb, *args, **kargs)

    def set_A(self, A):
        self.__A = A

    def get_A(self):
        return self.__A

    def get_B(self):
        return self.__B

    def get_D(self):
        return self.__D

    def set_B(self, B):
        self.__B = B

    def set_D(self, D):
        self.__D = D

    def solve(self, **kargs):
        n_dof = self.n_dof
        if np.isscalar(self.__X) and self.__X == 0:
            self.__X = np.ndarray(self.n_dof)  # empty array

        if len(self.__A.shape) == 2:  # A is a matrix
            if self.__A.shape[0] != n_dof:
                self.__A.resize(n_dof, n_dof)

            if len(self._dof_free) != 0:
                if np.isscalar(self.__D) and self.__D == 0:
                    self.__X[self._dof_free] = self._solve(
                        self.__MatCB.T @ self.__A @ self.__MatCB,
                        self.__MatCB.T @ (self.__B - self.__A @ self._Xbc),
                    )
                else:
                    self.__X[self._dof_free] = self._solve(
                        self.__MatCB.T @ self.__A @ self.__MatCB,
                        self.__MatCB.T @ (self.__B + self.__D - self.__A @ self._Xbc),
                    )

                self.__X = self.__MatCB * self.__X[self._dof_free] + self._Xbc
            else:
                self.__X[:] = self._Xbc[:]

            # compute matrix conditionment. Uncomment for debug purpose
            # lambda_min = sparse.linalg.eigs(self.__MatCB.T @ self.__A @ self.__MatCB , 1, which="SM", return_eigenvectors=False)
            # lambda_max = sparse.linalg.eigs(self.__MatCB.T @ self.__A @ self.__MatCB , 1, which="LM", return_eigenvectors=False)
            # print('cond: ', lambda_max/lambda_min)

        elif (
            len(self.__A.shape) == 1
        ):  # A is a diagonal matrix stored as a vector containing diagonal values
            # No need to account for boundary condition here because the matrix is diagonal and the resolution is direct

            assert not (np.isscalar(self.__D)), "internal error, contact developper"

            if self.__A.shape[0] != n_dof:  # probably not required
                self.__A.resize((n_dof))

            self.__X[self._dof_free] = (
                self.__B[self._dof_free] + self.__D[self._dof_free]
            ) / self.__A[self._dof_free]

    def get_X(self):  # solution of the linear system
        return self.__X

    def set_X(self, value):  # solution of the linear system
        self.__X = value

    def get_dof_solution(
        self, name="all"
    ):  # solution of the problem (same as get_X for linear problems if name=='all')
        return self._get_vect_component(self.__X, name)

    def set_dof_solution(self, name, value):
        self._set_vect_component(self.__X, name, value)

    def apply_boundary_conditions(self, t_fact=1, t_fact_old=None):
        n = self.mesh.n_nodes
        nvar = self.space.nvar
        n_dof = self.n_dof
        self._Xbc = np.zeros(n_dof)
        F = np.zeros(n_dof)

        build_mpc = False
        dof_blocked = set()  # only dirichlet bc
        dof_slave = set()
        data = []
        row = []
        col = []

        for e in self.bc.generate(self, t_fact, t_fact_old):
            if e.bc_type == "Dirichlet":
                self._Xbc[e._dof_index] = e._current_value
                dof_blocked.update(e._dof_index)

            if e.bc_type == "Neumann":
                F[e._dof_index] = e._current_value

            if e.bc_type == "MPC":
                self._Xbc[e._dof_index[0]] = (
                    e._current_value
                )  # valid in this case ??? need to be checked
                dof_slave.update(e._dof_index[0])  # eliminated dof
                build_mpc = True

                n_fact = len(e._factors)  # only factor for non eliminated (master) dof
                # shape e.__Fact should be n_fact*nbMPC
                # shape self.__Index should be nbMPC
                # shape self.__IndexMaster should be n_fact*nbMPC
                data.append(np.array(e._factors.T).ravel())
                row.append(
                    (np.array(e._dof_index[0]).reshape(-1, 1) * np.ones(n_fact)).ravel()
                )
                col.append(e._dof_index[1:].T.ravel())
                # col.append((e.IndexMaster + np.c_[e.VariableMaster]*n).T.ravel())

        dof_slave.update(dof_blocked)
        dof_slave = np.fromiter(dof_slave, int, len(dof_slave))
        # dof_slave= np.unique(np.hstack(dof_slave)).astype(int)
        dof_free = np.setdiff1d(range(n_dof), dof_slave).astype(int)

        # build matrix MPC
        if build_mpc:
            # M is a matrix such as Xblocked = M@X + Xbc
            M = sparse.coo_matrix(
                (np.hstack(data), (np.hstack(row), np.hstack(col))),
                shape=(n_dof, n_dof),
            )

            # Treating the case where MPC includes some blocked nodes as master nodes
            # Compute M + M@M - require if slave dof in one mpc is master in another
            if self.enable_mpc_coupling:
                M = (M + M @ M).tocoo()
            # update Xbc with the eliminated dof that may be used as slave dof in mpc
            self._Xbc = self._Xbc + M @ self._Xbc

            data = M.data
            row = M.row
            col = M.col

            # modification col numbering from dof_free to np.arange(len(dof_free))
            changeInd = np.full(
                n_dof, np.nan
            )  # mettre des nan plutôt que des zeros pour générer une erreur si pb
            changeInd[dof_free] = np.arange(len(dof_free))
            col = changeInd[np.hstack(col)]

            mask = np.logical_not(np.isnan(col))  # mask to delete nan value
            # print(len(col) - len(col[mask]))
            col = col[mask]
            row = row[mask]
            data = data[mask]

            # self._MFext = M.tocsr().T
            self._MFext = M
            self._dof_blocked = dof_blocked
        else:
            self._MFext = None

        # #adding identity for free nodes
        col = np.hstack(
            (col, np.arange(len(dof_free)))
        )  # np.hstack((col,dof_free)) #col.append(dof_free)
        row = np.hstack((row, dof_free))  # row.append(dof_free)
        data = np.hstack(
            (data, np.ones(len(dof_free)))
        )  # data.append(np.ones(len(dof_free)))

        self.__MatCB = sparse.coo_matrix(
            (data, (row, col)), shape=(n_dof, len(dof_free))
        ).tocsc()  # so that self.__MatCB.T is csr

        self.__B = F
        self._dof_slave = dof_slave
        self._dof_free = dof_free

    def update_boundary_conditions(self):
        t_fact = self.t_fact
        self.apply_boundary_conditions(t_fact, t_fact)

    def init_bc_start_value(self):
        ### is used only for incremental problems
        U = self.get_dof_solution()
        if np.isscalar(U) and U == 0:
            return
        F = self.get_ext_forces()
        n_nodes = self.mesh.n_nodes
        # for e in self.bc.generate(self):
        for e in self.bc.list_all():
            if e.bc_type == "Dirichlet":
                if e._start_value_default is None:
                    if not (np.isscalar(U) and U == 0):
                        e.start_value = U[e.variable * n_nodes + e.node_set]
            if e.bc_type == "Neumann":
                if e._start_value_default is None:
                    if not (np.isscalar(F) and F == 0):
                        e.start_value = F[e.variable * n_nodes + e.node_set]

    def get_ext_forces(self, name="all", include_mpc=True):
        """Return the nodal Forces in global coordinates system.

        The resulting forces are the sum of :
        - External forces (associated to Neumann boundary conditions)
        - Node reaction (associated to Dirichelet boundary conditions)
        - Inertia forces

        Parameters
        ------------
        name : str
            Can be either the name of a variable associated to the requested force,
            the name of a vector or 'all'.
        include_mpc: bool (default = True)
            if True, the multi_point_constraint (mpc) are included in the returned ext forces. This make the external force
            accessible on mpc global dof (dof only used in mpc and not linked to a mesh). A direct consequence
            is that the external force can't be accessed on the mpc slave dof.

        Returns
        ------------
        np.ndarray
            The external forces. If a vector name is given, the function returns
            the external force of every vector component.
            if name == 'all', return a 1d numpy array containing the external forces
            associated to every dof. In this case, the returned array may be easily reshaped
            to separated components by using: pb.get_ext_forces().reshape(pb.space.nvar,-1)

        Notes
        ---------
        The "force" mentionned here must be seen as a generic term.
        The true physical meanings of the "external forces" depends
        on the problem and the nature of the dof (for instance moment for rotational dof or
        heat flux for temperature).
        """
        if self._MFext is None or not (include_mpc):
            if np.isscalar(self.get_D()) and self.get_D() == 0:
                return self._get_vect_component(self.get_A() @ self.get_X(), name)
            else:
                return self._get_vect_component(
                    self.get_A() @ self.get_X() - self.get_D(), name
                )
        else:
            M = self._MFext
            # adding identity for all dof execpted slave dof in mpc
            dof_idt = list(self._dof_free) + list(self._dof_blocked)
            col = np.hstack((M.col, dof_idt))
            row = np.hstack((M.row, dof_idt))
            data = np.hstack((M.data, np.ones(len(dof_idt))))
            M = M.tocsr().T
            # need to be checked
            # M = self._MFext + scipy.sparse.identity(self.n_dof, dtype='d')
            if np.isscalar(self.get_D()) and self.get_D() == 0:
                return self._get_vect_component(M @ self.get_A() @ self.get_X(), name)
            else:
                return self._get_vect_component(
                    M @ self.get_A() @ self.get_X() - M @ self.get_D(), name
                )

    @property
    def results(self):
        return self._problem_output.data_sets

    @property
    def n_dof(self):
        return self.mesh.n_nodes * self.space.nvar + self.n_global_dof

    @property
    def n_node_dof(self):
        return self.mesh.n_nodes * self.space.nvar
