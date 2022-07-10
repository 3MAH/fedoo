# base class
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg

from fedoo.problem.ProblemBase import ProblemBase
from fedoo.core.assembly  import Assembly
from fedoo.problem.Output import _ProblemOutput, _GetResults
from fedoo.utilities.dataset import DataSet

import time 

class Problem(ProblemBase):
    
    def __init__(self, A, B, D, mesh, name = "MainProblem", space = None):

        # the problem is AX = B + D
        
        ProblemBase.__init__(self, name, space)

        #self.__ModelingSpace = A.shape[0]
        self.__ModelingSpace = mesh.n_nodes * self.space.nvar

        self.__A = A

        if B is 0:
            self.__B = self._InitializeVector()
        else:
            self.__B = B
        
        self.__D = D

        self.mesh = mesh   

        self.__X = np.ndarray( self.__ModelingSpace )
        self.__Xbc = 0

        self.__DofBlocked = np.array([])
        self.__DofFree    = np.array([])
        
        #prepering output demand to export results
        self.__ProblemOutput = _ProblemOutput()        

    def _InitializeVector(self): #initialize a vector (force vector for instance) being giving the stiffness matrix
        return np.zeros(self.__ModelingSpace)     

    def _SetVectorComponent(self, vector, name, value): #initialize a vector (force vector for instance) being giving the stiffness matrix
        assert isinstance(name,str), 'argument error'
        
        if name.lower() == 'all': 
            vector[:] = value
        else:
            i = self.space.variable_rank(name)
            n = self.mesh.n_nodes
            vector[i*n : (i+1)*n] = value      

    def _get_global_vectorComponent(self, vector, name): #Get component of a vector (force vector for instance) being given the name of a component (vector or single component)    
        assert isinstance(name,str), 'argument error'
        
        if name.lower() == 'all':                             
            return vector

        n = self.mesh.n_nodes
        
        if name in self.space.list_vector():
            vec = self.space.get_vector(name)
            i = vec[0] #rank of the 1rst variable of the vector
            dim = len(vec)
            return vector.reshape(-1,n)[i:i+dim]
            # return vector[i*n : (i+dim)*n].reshape(-1,n) 
        else:             
            #vector component are assumed defined as an increment sequence (i, i+1, i+2)
            i = self.space.variable_rank(name)
        
            return vector[i*n : (i+1)*n]   

    def AddOutput(self, filename, assemblyname, output_list, output_type='Node', file_format ='vtk', position = 1):
        return self.__ProblemOutput.AddOutput(filename, assemblyname, output_list, output_type, file_format, position)            

    def SaveResults(self, iterOutput=None):
        self.__ProblemOutput.SaveResults(self, iterOutput)

    def GetResults(self, assemb, output_list, output_type='Node', position = 1, res_format = None):        
        return DataSet(self.mesh, _GetResults(self, assemb, output_list, output_type, position, res_format), output_type)

    def SetA(self,A):
        self.__A = A     
        
    def GetA(self):
        return self.__A 

    def GetB(self):
        return self.__B 
    
    def GetD(self):
        return self.__D 
 
    def GetMesh(self):
        return self.mesh
    
    def SetB(self,B):
        self.__B = B    
              
    def SetD(self,D):
        self.__D = D        

    def Solve(self, **kargs):
        if len(self.__A.shape) == 2: #A is a matrix        
            if len(self.__DofBlocked) == 0: print('Warning: no dirichlet boundary conditions applied. "Problem.ApplyBoundaryCondition()" is probably missing')          
             # to delete after a careful validation of the other case
            # self.__X[self.__DofBlocked] = self.__Xbc[self.__DofBlocked]
            
            # Temp = self.__A[:,self.__DofBlocked].dot(self.__X[self.__DofBlocked])
            # if self.__D is 0:
            #     self.__X[self.__DofFree]  = self._ProblemBase__Solve(self.__A[self.__DofFree,:][:,self.__DofFree],self.__B[self.__DofFree] - Temp[self.__DofFree])
            # else:
            #     self.__X[self.__DofFree]  = self._ProblemBase__Solve(self.__A[self.__DofFree,:][:,self.__DofFree],self.__B[self.__DofFree] + self.__D[self.__DofFree] - Temp[self.__DofFree])

            if self.__D is 0:
                self.__X[self.__DofFree]  = self._ProblemBase__Solve(self.__MatCB.T @ self.__A @ self.__MatCB , self.__MatCB.T @ (self.__B - self.__A@ self.__Xbc)  )   
            else:
                self.__X[self.__DofFree]  = self._ProblemBase__Solve(self.__MatCB.T @ self.__A @ self.__MatCB , self.__MatCB.T @ (self.__B + self.__D - self.__A@ self.__Xbc)  )                   
                       
            self.__X = self.__MatCB * self.__X[self.__DofFree]  + self.__Xbc

                
        elif len(self.__A.shape) == 1: #A is a diagonal matrix stored as a vector containing diagonal values 
            #No need to account for boundary condition here because the matrix is diagonal and the resolution is direct
            
            assert self.__D is not 0, "internal error, contact developper"
            
            self.__X[self.__DofFree]  = (self.__B[self.__DofFree] + self.__D[self.__DofFree]) / self.__A[self.__DofFree]               

    def GetX(self): #solution of the linear system
        return self.__X
    
    def GetDoFSolution(self,name='all'): #solution of the problem (same as GetX for linear problems if name=='all')
        return self._get_global_vectorComponent(self.__X, name) 

    def SetDoFSolution(self,name,value):
        self._SetVectorComponent(self.__X, name, value)          
       

    def ApplyBoundaryCondition(self, timeFactor=1, timeFactorOld=None):
                
        n = self.mesh.n_nodes
        nvar = self.space.nvar
        Uimp = np.zeros(nvar*n)
        F = np.zeros(nvar*n)

        MPC = False
        DofB = []
        data = []
        row = []
        col = []
        for e in self._BoundaryConditions:
            if e.BoundaryType == 'Dirichlet':
                Uimp, GlobalIndex = e._ApplyTo(Uimp, n, timeFactor, timeFactorOld)
                DofB.append(GlobalIndex)

            elif e.BoundaryType == 'Neumann':
                F = e._ApplyTo(F, n, timeFactor)[0]
            
            elif e.BoundaryType == 'MPC':
                Uimp, GlobalIndex = e._ApplyTo(Uimp, n, timeFactor, timeFactorOld) #valid in this case ??? need to be checked
                DofB.append(GlobalIndex)
                MPC = True         
#                if np.isscalar(self.__Index): nbMPC = 1
#                else: nbMPC = len(self.__Index)
                nbFact = len(e.Factor)
                
                #shape self.__Fact should be nbFact*nbMPC 
                #shape self.__Index should be nbMPC
                #shape self.__IndexMaster should be nbFact*nbMPC
                data.append(np.array(e.Factor.T).ravel())
                row.append((np.array(GlobalIndex).reshape(-1,1)*np.ones(nbFact)).ravel())
                col.append((e.IndexMaster + np.c_[e.VariableMaster]*n).T.ravel())
        
        
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
        Nnodes = self.mesh.n_nodes
        for e in self._BoundaryConditions:            
            if e.DefaultInitialValue is None:
                if e.BoundaryType == 'Dirichlet':
                    if U is not 0:
                        e.ChangeInitialValue(U[e.Variable*Nnodes + e.Index])
                elif e.BoundaryType == 'Neumann':
                    if F is not 0:
                        e.ChangeInitialValue(F[e.Variable*Nnodes + e.Index])
    
    @property
    def results(self):
        return self.__ProblemOutput.data_sets
      