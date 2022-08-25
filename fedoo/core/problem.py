# base class
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg

from fedoo.core.assembly  import Assembly
from fedoo.core.base import ProblemBase
from fedoo.core.boundary_conditions import UniqueBoundaryCondition, MPC
from fedoo.core.output import _ProblemOutput, _get_results
from fedoo.core.dataset import DataSet

import time 

class Problem(ProblemBase):
    
    def __init__(self, A, B, D, mesh, name = "MainProblem", space = None):

        # the problem is AX = B + D
        
        ProblemBase.__init__(self, name, space)
        self.mesh = mesh   

        self.__A = A

        if B is 0:
            self.__B = self._new_vect_dof()
        else:
            self.__B = B
        
        self.__D = D


        self.__X = np.ndarray( self.n_dof ) #empty array
        self.__Xbc = 0

        self.__DofBlocked = np.array([])
        self.__DofFree    = np.array([])
        
        #prepering output demand to export results
        self.__ProblemOutput = _ProblemOutput()        

    def _new_vect_dof(self): #initialize a vector (force vector for instance) whose size is n_dof
        return np.zeros(self.n_dof)     

    def _set_vect_component(self, vector, name, value): #initialize a vector (force vector for instance) being giving the stiffness matrix
        assert isinstance(name,str), 'argument error'
        
        if name.lower() == 'all': 
            vector[:] = value
        else:
            i = self.space.variable_rank(name)
            n = self.mesh.n_nodes
            vector[i*n : (i+1)*n] = value      

    def _get_vect_component(self, vector, name): #Get component of a vector (force vector for instance) being given the name of a component (vector or single component)    
        assert isinstance(name,str), 'argument error'
        
        if name.lower() == 'all':                             
            return vector

        n = self.mesh.n_nodes
        
        if name in self.space.list_vectors():
            vec = self.space.get_vector(name)
            i = vec[0] #rank of the 1rst variable of the vector
            dim = len(vec)
            return vector.reshape(-1,n)[i:i+dim]
            # return vector[i*n : (i+dim)*n].reshape(-1,n) 
        else:             
            #vector component are assumed defined as an increment sequence (i, i+1, i+2)
            i = self.space.variable_rank(name)
            return vector[i*n : (i+1)*n]   

    def add_output(self, filename, assemblyname, output_list, output_type=None, file_format ='npz', position = 1):
        return self.__ProblemOutput.add_output(filename, assemblyname, output_list, output_type, file_format, position)            

    def save_results(self, iterOutput=None):
        self.__ProblemOutput.save_results(self, iterOutput)

    def get_results(self, assemb, output_list, output_type=None, position = 1):        
        return _get_results(self, assemb, output_list, output_type, position)

    def SetA(self,A):
        self.__A = A     
        
    def GetA(self):
        return self.__A 

    def GetB(self):
        return self.__B 
    
    def GetD(self):
        return self.__D 
    
    def SetB(self,B):
        self.__B = B    
              
    def SetD(self,D):
        self.__D = D        

    def solve(self, **kargs):
        if len(self.__A.shape) == 2: #A is a matrix        
            if len(self.__DofBlocked) == 0: print('Warning: no dirichlet boundary conditions applied. "Problem.apply_boundary_conditions()" is probably missing')          
             # to delete after a careful validation of the other case
            # self.__X[self.__DofBlocked] = self.__Xbc[self.__DofBlocked]
            
            # Temp = self.__A[:,self.__DofBlocked].dot(self.__X[self.__DofBlocked])
            # if self.__D is 0:
            #     self.__X[self.__DofFree]  = self._solve(self.__A[self.__DofFree,:][:,self.__DofFree],self.__B[self.__DofFree] - Temp[self.__DofFree])
            # else:
            #     self.__X[self.__DofFree]  = self._solve(self.__A[self.__DofFree,:][:,self.__DofFree],self.__B[self.__DofFree] + self.__D[self.__DofFree] - Temp[self.__DofFree])

            if self.__D is 0:
                self.__X[self.__DofFree]  = self._solve(self.__MatCB.T @ self.__A @ self.__MatCB , self.__MatCB.T @ (self.__B - self.__A@ self.__Xbc)  )   
            else:
                self.__X[self.__DofFree]  = self._solve(self.__MatCB.T @ self.__A @ self.__MatCB , self.__MatCB.T @ (self.__B + self.__D - self.__A@ self.__Xbc)  )                   
                       
            self.__X = self.__MatCB * self.__X[self.__DofFree]  + self.__Xbc

                
        elif len(self.__A.shape) == 1: #A is a diagonal matrix stored as a vector containing diagonal values 
            #No need to account for boundary condition here because the matrix is diagonal and the resolution is direct
            
            assert self.__D is not 0, "internal error, contact developper"
            
            self.__X[self.__DofFree]  = (self.__B[self.__DofFree] + self.__D[self.__DofFree]) / self.__A[self.__DofFree]               

    def GetX(self): #solution of the linear system
        return self.__X
    
    def GetDoFSolution(self,name='all'): #solution of the problem (same as GetX for linear problems if name=='all')
        return self._get_vect_component(self.__X, name) 

    def SetDoFSolution(self,name,value):
        self._set_vect_component(self.__X, name, value)          
       

    def apply_boundary_conditions(self, t_fact=1, t_fact_old=None):
                
        n = self.mesh.n_nodes
        nvar = self.space.nvar
        Uimp = np.zeros(nvar*n)
        F = np.zeros(nvar*n)

        MPC = False
        DofB = []
        data = []
        row = []
        col = []
        for e in self._BoundaryConditions.generate(self, t_fact, t_fact_old):
            if e.bc_type == 'Dirichlet':
                Uimp[e._dof_index] = e._current_value
                DofB.append(e._dof_index)

            elif e.bc_type == 'Neumann':
                F[e._dof_index] = e._current_value
                # F = e._ApplyTo(F, n, t_fact)[0]
            
            elif e.bc_type == 'MPC':
                Uimp[e._dof_index[0]] = e._current_value  #valid in this case ??? need to be checked
                DofB.append(e._dof_index[0]) #eliminated dof
                MPC = True         

                n_fact = len(e._factors) #only factor for non eliminated (master) dof                
                #shape e.__Fact should be n_fact*nbMPC 
                #shape self.__Index should be nbMPC
                #shape self.__IndexMaster should be n_fact*nbMPC
                data.append(np.array(e._factors.T).ravel())
                row.append((np.array(e._dof_index[0]).reshape(-1,1)*np.ones(n_fact)).ravel())
                col.append(e._dof_index[1:].T.ravel())        
                # col.append((e.IndexMaster + np.c_[e.VariableMaster]*n).T.ravel())


        
        
        DofB = np.unique(np.hstack(DofB)).astype(int)
        DofL = np.setdiff1d(range(nvar*n),DofB).astype(int)
        
        #build matrix MPC
        if MPC:    
            #Treating the case where MPC includes some blocked nodes as master nodes
            #M is a matrix such as Ublocked = M@U + Uimp
            #Compute M + M@M
            M = sparse.coo_matrix( 
                (np.hstack(data), (np.hstack(row),np.hstack(col))), 
                shape=(nvar*n,nvar*n))
            
            # BoundaryCondition.M = M #test : used to compute the reaction - to delete later

                                   
            Uimp = Uimp	+ M@Uimp 
            
            M = (M+M@M).tocoo()
            data = M.data
            row = M.row
            col = M.col

            #modification col numbering from DofL to np.arange(len(DofL))
            changeInd = np.full(nvar*n,np.nan) #mettre des nan plutôt que des zeros pour générer une erreur si pb
            changeInd[DofL] = np.arange(len(DofL))
            col = changeInd[np.hstack(col)]
            mask = np.logical_not(np.isnan(col)) #mask to delete nan value 
            
            col = col[mask] ; row = row[mask] ; data = data[mask]

        # #adding identity for free nodes
        col = np.hstack((col,np.arange(len(DofL)))) #np.hstack((col,DofL)) #col.append(DofL)  
        row = np.hstack((row,DofL)) #row.append(DofL)            
        data = np.hstack((data, np.ones(len(DofL)))) #data.append(np.ones(len(DofL)))
        
        self.__MatCB = sparse.coo_matrix( 
                (data,(row,col)), 
                shape=(nvar*n,len(DofL))).tocsr()
        
        self.__Xbc = Uimp
        self.__B = F
        self.__DofBlocked = DofB
        self.__DofFree = DofL

    def SetInitialBCToCurrent(self):
        ### is used only for incremental problems
        U = self.GetDoFSolution() 
        F = self.get_ext_forces()
        n_nodes = self.mesh.n_nodes
        for e in self._BoundaryConditions:            
            if e._start_value_default is None:
                if e.bc_type == 'Dirichlet':
                    if U is not 0:
                        e.start_value = U[e.variable*n_nodes + e.node_set]
                elif e.bc_type == 'Neumann':
                    if F is not 0:
                        e.start_value = F[e.variable*n_nodes + e.node_set]
    
    
    ### Functions related to boundary contidions
    def BoundaryCondition(self,bc_type,Var,Value,Index,Constant = None, timeEvolution=None, initialValue = None, name = ""):
        """
        Define some boundary conditions        

        Parameters
        ----------
        bc_type : str
            Type of boundary conditions : 'Dirichlet', 'Neumann' or 'MPC' for multipoint constraints.
        Var : str, list of str, or list of int
            variable name (str) or list of variable name or for MPC only, list of variable rank 
        Value : scalar or array or list of scalars or list of array
            Variable final value (Dirichlet) or force Value (Neumann) or list of factor (MPC)
            For Neumann and Dirichlet, if Var is a list of str, Value may be :
                (i) scalar if the same Value is applied for all Variable
                (ii) list of scalars, if the scalar values are different for all Variable (in this case the len of Value should be equal to the lenght of Var)
                (iii) list of arrays, if the scalar Value is potentially different for all variables and for all indexes. In this case, Value[num_var][i] should give the value of the num_var variable related to the node i.
        Index : list of int, str, list of list of int, list of str
            For FEM Problem with Neumann/Dirichlet BC: Nodes Index (list of int) 
            For FEM Problem with MPC: list Node Indexes (list of list of int) 
            For PGD Problem with Neumann/Dirichlet BC: SetOfname (type str) defining a set of Nodes of the reference mesh
            For PGD Problem with MPC: list of SetOfname (str)
        Constant : scalar, optional
            For MPC only, constant value on the equation
        timeEvolution : function
            Function that gives the temporal evolution of the BC Value (applyed as a factor to the specified BC). The function y=f(x) where x in [0,1] and y in [0,1]. For x, 0 denote the begining of the step and 1 the end.
        initialValue : float, array or None
            if None, the initialValue is keep to the current state.
            if scalar value: The initialValue is the same for all dof defined in BC
            if array: the len of the array should be = to the number of dof defined in the BC

            Default: None
        name : str, optional
            Define an name for the Boundary Conditions. Default is "No name". The same name may be used for several BC.

        Returns
        -------
        None.

        Remark  
        -------
        To define many MPC in one operation, use array where each line define a single MPC        
        """
        if isinstance(Var, str) and Var not in self.space.list_variables():
            #we assume that Var is a Vector
            try: 
                Var = [self.space.variable_name(var_rank) for var_rank in self.space.get_vector(Var)]
            except:
                raise NameError('Unknown variable name')
                
        if isinstance(Var, list) and bc_type != 'MPC':          
            if np.isscalar(Value):
                Value = [Value for var in Var] 
            for i,var in enumerate(Var):
                self._BoundaryConditions.append(UniqueBoundaryCondition(bc_type,var,Value[i],Index, timeEvolution, initialValue, name))                
                self._BoundaryConditions[-1].initialize(self)
        else:
            if bc_type == 'MPC':
                self._BoundaryConditions.append(MPC(Var,Value,Index,Constant, timeEvolution, initialValue, name))
            else:
                self._BoundaryConditions.append(UniqueBoundaryCondition(bc_type,Var,Value,Index, timeEvolution, initialValue, name)) 
        
            self._BoundaryConditions[-1].initialize(self)


    @property
    def results(self):
        return self.__ProblemOutput.data_sets

    @property
    def n_dof(self):
        return self.mesh.n_nodes * self.space.nvar    