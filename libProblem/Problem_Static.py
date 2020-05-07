import numpy as np
from fedoo.libAssembly.Assembly import *
from fedoo.libProblem.Problem   import *
from fedoo.libProblem.ProblemPGD   import ProblemPGD

#dynamical inheritance. The class is generated inside a function
def Static(Assembling, ID = "MainProblem"):
    if isinstance(Assembling,str):
        Assembling = Assembly.GetAll()[Assembling]
                
    if hasattr(Assembling.GetMesh(), 'GetListMesh'): libBase = ProblemPGD
    else: libBase = Problem
    
    class __Static(libBase):
                
        def __init__(self, Assembling, ID):                            
            A = Assembling.GetMatrix()
            B = 0             
            D = Assembling.GetVector()     
            self.__Assembly = Assembling
            
            libBase.__init__(self,A,B,D,Assembling.GetMesh(), ID)
    
        def GetElasticEnergy(self): #only work for classical FEM
            """
            returns : sum (0.5 * U.transposed * K * U)
            """
    
            return sum( 0.5*self.GetDoFSolution('all').transpose() * self.GetA() * self.GetDoFSolution('all') )

        def GetNodalElasticEnergy(self):
            """
            returns : 0.5 * K * U . U
            """
    
            E = 0.5*self.GetDoFSolution('all').transpose() * self.GetA() * self.GetDoFSolution('all')

            E = np.reshape(E,(3,-1)).T

            return E
        
        def Reset(self):
            self.__Assembly.Reset()
            
            self.SetA(self.__Assembly.GetMatrix()) #tangent stiffness 
            self.SetD(self.__Assembly.GetVector())            
            B = 0            
            self.ApplyBoundaryCondition()

        
        def Update(self, time=None, compute = 'all'):   
            """
            Assemble the matrix including the following modification:
                - New initial Stress
                - New initial Displacement
                - Modification of the mesh
                - Change in constitutive law (internal variable)
            Update the problem with the new assembled global matrix and global vector
            """
#            self.__Assembly.Update(self.GetD())
#            self.__Assembly.Update(self)
#            self.SetA(Assembling.GetMatrix())
#            self.SetD(Assembling.GetVector())
#            
            outValues = self.__Assembly.Update(self, time, compute)  
            self.SetA(Assembling.GetMatrix())
            self.SetD(Assembling.GetVector())
            return outValues 



        def ChangeAssembly(self,Assembling, update = True):
            """
            Modify the assembly associated to the problem and update the problem (see Assembly.Update for more information)
            """
            if isinstance(Assembling,str):
                Assembling = Assembly.GetAll()[Assembling]
                
            self.__Assembly = Assembling
            if update: self.Update()
        
        def GetDisp(self, name = 'all'):
            return self.GetDoFSolution(name)
        
    return __Static(Assembling, ID)


