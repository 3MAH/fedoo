# base class
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg

from fedoo.libProblem.BoundaryCondition import BoundaryCondition
from fedoo.libProblem.ProblemBase import ProblemBase
from fedoo.libUtil.Variable  import *
from fedoo.libAssembly.Assembly  import *
from fedoo.libUtil.ExportData import _ProblemOutput

import time 

class Problem(ProblemBase):
    
    def __init__(self, A, B, D, Mesh, ID = "MainProblem"):

        # the problem is AX = B + D
        
        #self.__ProblemDimension = A.shape[0]
        self.__ProblemDimension = Mesh.GetNumberOfNodes() * Variable.GetNumberOfVariable()

        self.__A = A

        if B is 0:
            self.__B = self._InitializeVector()
        else:
            self.__B = B
        
        self.__D = D

        self.__Mesh = Mesh   

        self.__X = np.ndarray( self.__ProblemDimension )
        self.__Xbc = 0

        self.__DofBlocked = np.array([])
        self.__DofFree    = np.array([])
        
        #prepering output demand to export results
        self.__ProblemOutput = _ProblemOutput()

        ProblemBase.__init__(self, ID)
        

    def _InitializeVector(self): #initialize a vector (force vector for instance) being giving the stiffness matrix
        return np.zeros(self.__ProblemDimension)     

    def _SetVectorComponent(self, vector, name, value): #initialize a vector (force vector for instance) being giving the stiffness matrix
        assert isinstance(name,str), 'argument error'
        
        if name.lower() == 'all': 
            vector[:] = value
        else:
            i = Variable.GetRank(name)
            n = self.GetMesh().GetNumberOfNodes()
            vector[i*n : (i+1)*n] = value      

    def _GetVectorComponent(self, vector, name): #Get component of a vector (force vector for instance) being given the name of a component (vector or single component)    
        assert isinstance(name,str), 'argument error'
        
        if name.lower() == 'all':                             
            return vector

        n = self.__Mesh.GetNumberOfNodes()
        
        if name in Variable.ListVector():
            vec = Variable.GetVector(name)
            i = vec[0] #rank of the 1rst variable of the vector
            dim = len(vec)
            return vector.reshape(-1,n)[i:i+dim]
            # return vector[i*n : (i+dim)*n].reshape(-1,n) 
        else:             
            #vector component are assumed defined as an increment sequence (i, i+1, i+2)
            i = Variable.GetRank(name)
        
            return vector[i*n : (i+1)*n]   

    def AddOutput(self, filename, assemblyID, output_list, output_type='Node', file_format ='vtk'):
        self.__ProblemOutput.AddOutput(filename, assemblyID, output_list, output_type, file_format)            

    def SaveResults(self, iterOutput=None):
        self.__ProblemOutput.SaveResults(self, iterOutput)                                

    def SetA(self,A):
        self.__A = A     
        
    def GetA(self):
        return self.__A 

    def GetB(self):
        return self.__B 
    
    def GetD(self):
        return self.__D 
 
    def GetMesh(self):
        return self.__Mesh
    
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

    def ApplyBoundaryCondition(self, timeFactor=1, timeFactorOld=None):
        self.__Xbc, self.__B, self.__DofBlocked, self.__DofFree, self.__MatCB = BoundaryCondition.Apply(self.__Mesh.GetNumberOfNodes(), timeFactor, timeFactorOld, self.GetID())

    def SetInitialBCToCurrent(self):
        BoundaryCondition.setInitialToCurrent(self)

    def GetX(self): #solution of the linear system
        return self.__X
    
    def GetDoFSolution(self,name='all'): #solution of the problem (same as GetX for linear problems if name=='all')
        return self._GetVectorComponent(self.__X, name) 

    def SetDoFSolution(self,name,value):
        self._SetVectorComponent(self.__X, name, value)          
       