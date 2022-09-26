# base class
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg

from fedoo.core.assembly  import Assembly
from fedoo.core.base import ProblemBase
from fedoo.core.boundary_conditions import BoundaryCondition, MPC
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

        self.__dof_slave = np.array([])
        self.__dof_free    = np.array([])
        
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
            if len(self.__dof_slave) == 0: print('Warning: no dirichlet boundary conditions applied. "Problem.apply_boundary_conditions()" is probably missing')          
             # to delete after a careful validation of the other case
            # self.__X[self.__dof_slave] = self.__Xbc[self.__dof_slave]
            
            # Temp = self.__A[:,self.__dof_slave].dot(self.__X[self.__dof_slave])
            # if self.__D is 0:
            #     self.__X[self.__dof_free]  = self._solve(self.__A[self.__dof_free,:][:,self.__dof_free],self.__B[self.__dof_free] - Temp[self.__dof_free])
            # else:
            #     self.__X[self.__dof_free]  = self._solve(self.__A[self.__dof_free,:][:,self.__dof_free],self.__B[self.__dof_free] + self.__D[self.__dof_free] - Temp[self.__dof_free])

            if self.__D is 0:
                self.__X[self.__dof_free]  = self._solve(self.__MatCB.T @ self.__A @ self.__MatCB , self.__MatCB.T @ (self.__B - self.__A@ self.__Xbc)  )   
            else:
                self.__X[self.__dof_free]  = self._solve(self.__MatCB.T @ self.__A @ self.__MatCB , self.__MatCB.T @ (self.__B + self.__D - self.__A@ self.__Xbc)  )                   
                       
            self.__X = self.__MatCB * self.__X[self.__dof_free]  + self.__Xbc

                
        elif len(self.__A.shape) == 1: #A is a diagonal matrix stored as a vector containing diagonal values 
            #No need to account for boundary condition here because the matrix is diagonal and the resolution is direct
            
            assert self.__D is not 0, "internal error, contact developper"
            
            self.__X[self.__dof_free]  = (self.__B[self.__dof_free] + self.__D[self.__dof_free]) / self.__A[self.__dof_free]               

    def GetX(self): #solution of the linear system
        return self.__X
    
    def GetDoFSolution(self,name='all'): #solution of the problem (same as GetX for linear problems if name=='all')
        return self._get_vect_component(self.__X, name) 

    def SetDoFSolution(self,name,value):
        self._set_vect_component(self.__X, name, value)          
       

    def apply_boundary_conditions(self, t_fact=1, t_fact_old=None):
                
        n = self.mesh.n_nodes
        nvar = self.space.nvar
        Xbc = np.zeros(nvar*n)
        F = np.zeros(nvar*n)

        MPC = False
        dof_blocked = set() #only dirichlet bc
        dof_slave = set() 
        data = []
        row = []
        col = []
        for e in self.bc.generate(self, t_fact, t_fact_old):
            if e.bc_type == 'Dirichlet':
                Xbc[e._dof_index] = e._current_value
                dof_blocked.update(e._dof_index)

            if e.bc_type == 'Neumann':
                F[e._dof_index] = e._current_value
            
            if e.bc_type == 'MPC':
                Xbc[e._dof_index[0]] = e._current_value  #valid in this case ??? need to be checked
                dof_slave.update(e._dof_index[0]) #eliminated dof
                MPC = True         
    
                n_fact = len(e._factors) #only factor for non eliminated (master) dof                
                #shape e.__Fact should be n_fact*nbMPC 
                #shape self.__Index should be nbMPC
                #shape self.__IndexMaster should be n_fact*nbMPC
                data.append(np.array(e._factors.T).ravel())
                row.append((np.array(e._dof_index[0]).reshape(-1,1)*np.ones(n_fact)).ravel())
                col.append(e._dof_index[1:].T.ravel())        
                # col.append((e.IndexMaster + np.c_[e.VariableMaster]*n).T.ravel())

        
        dof_slave.update(dof_blocked)
        dof_slave = np.fromiter(dof_slave, int, len(dof_slave))
        # dof_slave= np.unique(np.hstack(dof_slave)).astype(int)
        dof_free = np.setdiff1d(range(nvar*n),dof_slave).astype(int)
        
        #build matrix MPC
        if MPC:    
            #Treating the case where MPC includes some blocked nodes as master nodes
            #M is a matrix such as Xblocked = M@U + Xbc
            #Compute M + M@M
            M = sparse.coo_matrix( 
                (np.hstack(data), (np.hstack(row),np.hstack(col))), 
                shape=(nvar*n,nvar*n))
            
            # BoundaryCondition.M = M #test : used to compute the reaction - to delete later
                                   
            Xbc = Xbc + M@Xbc 
            
            M = (M+M@M).tocoo()
            data = M.data
            row = M.row
            col = M.col

            #modification col numbering from dof_free to np.arange(len(dof_free))
            changeInd = np.full(nvar*n,np.nan) #mettre des nan plutôt que des zeros pour générer une erreur si pb
            changeInd[dof_free] = np.arange(len(dof_free))
            col = changeInd[np.hstack(col)]
            mask = np.logical_not(np.isnan(col)) #mask to delete nan value 
            
            col = col[mask] ; row = row[mask] ; data = data[mask]

        # #adding identity for free nodes
        col = np.hstack((col,np.arange(len(dof_free)))) #np.hstack((col,dof_free)) #col.append(dof_free)  
        row = np.hstack((row,dof_free)) #row.append(dof_free)            
        data = np.hstack((data, np.ones(len(dof_free)))) #data.append(np.ones(len(dof_free)))
        
        self.__MatCB = sparse.coo_matrix( 
                (data,(row,col)), 
                shape=(nvar*n,len(dof_free))).tocsr()
        
        self.__Xbc = Xbc
        self.__B = F
        self.__dof_slave = dof_slave
        self.__dof_free = dof_free

    def SetInitialBCToCurrent(self):
        ### is used only for incremental problems
        U = self.GetDoFSolution() 
        F = self.get_ext_forces()
        n_nodes = self.mesh.n_nodes
        for e in self.bc.generate(self):        
            if e.bc_type == 'Dirichlet':
                if e._start_value_default is None:
                    if U is not 0:
                        e.start_value = U[e.variable*n_nodes + e.node_set]
            if e.bc_type == 'Neumann':    
                if e._start_value_default is None:
                    if F is not 0:
                        e.start_value = F[e.variable*n_nodes + e.node_set]


    @property
    def results(self):
        return self.__ProblemOutput.data_sets

    @property
    def n_dof(self):
        return self.mesh.n_nodes * self.space.nvar    