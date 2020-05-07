import numpy as np
from fedoo.libAssembly.Assembly import *
from fedoo.libProblem.Problem   import *
import scipy.sparse as sparse

def ExplicitDynamic(StiffnessAssembling, MassAssembling , TimeStep, DampingAssembling = 0, ID = "MainProblem"):
    """
    Define a Centred Difference problem for structural dynamic
    For damping, the backward euler derivative is used to compute the velocity
    The algorithm come from:  Bathe KJ and Edward W, "Numerical methods in finite element analysis", Prentice Hall, 1976, pp 323-324    
    """
        
    if isinstance(StiffnessAssembling,str):
        StiffnessAssembling = Assembly.GetAll()[StiffnessAssembling]
                
    if isinstance(MassAssembling,str):
        MassAssembling = Assembly.GetAll()[MassAssembling]
        
    if isinstance(DampingAssembling,str):
        DampingAssembling = Assembly.GetAll()[DampingAssembling]

    if hasattr(StiffnessAssembling.GetMesh(), 'GetListMesh'): libBase = ProblemPGD
    else: libBase = Problem


    class __ExplicitDynamic(libBase):
        
        def __init__(self, StiffnessAssembling, MassAssembling , TimeStep, DampingAssembling, ID):  

            A = 1/(TimeStep**2)*MassAssembling.GetMatrix()   
            B = 0 ; D = 0
            
            self.__Xold    = self._InitializeVector(A) #displacement at the previous time step        
            self.__Xdot    = self._InitializeVector(A)
            self.__Xdotdot = self._InitializeVector(A)

            self.__TimeStep   = TimeStep
            self.__MassLuming = False
            
            self.__MassMatrix  = MassAssembling.GetMatrix()
            self.__StiffMatrix = StiffnessAssembling.GetMatrix()
            if DampingAssembling == 0: self.__DampMatrix = 0
            else: self.__DampMatrix = DampingAssembling.GetMatrix()
            
            libBase.__init__(self,A,B,D,StiffnessAssembling.GetMesh(),ID)        

        def __UpdateA(): #internal function to be used when modifying M
            # if MassLumping == True, A is a vector representing the diagonal value
            self.SetA(  self.__MassMatrix         / (self.__TimeStep**2))

        def UpdateStiffness(StiffnessAssembling): #internal function to be used when modifying the siffness matrix
            if isinstance(StiffnessAssembling,str):
                StiffnessAssembling = Assembly.GetAll()[StiffnessAssembling]
            
            self.__StiffMatrix = StiffnessAssembling.GetMatrix()

        def MassLumping(): #internal function to be used when modifying M
            self.__MassLuming = True
            if len(self.__MassMatrix.shape) == 2:
                self.__MassMatrix = np.array(self.__MassMatrix.sum(1))[:,0]
                self.__UpdateA()
               
        def GetX():
            return self.GetDoFSolution('all')
        
        def GetXdot():
            return self.__Xdot
    
        def SetInitialDisplacement(self, name,value):
            """
            name is the name of the associated variable (generaly 'DispX', 'DispY' or 'DispZ')    
            value is an array containing the initial displacement of each nodes
            """        
            self._SetVectorComponent(self.__Xold, name, value)          
        
        def SetInitialVelocity(self, name,value):
            """
            name is the name of the associated variable (generaly 'DispX', 'DispY' or 'DispZ')    
            value is an array containing the initial velocity of each nodes        
            """
            self._SetVectorComponent(self.__Xdot, name, value) 
    
    
        def SetInitialAcceleration(self, name,value):
            """
            name is the name of the associated variable (generaly 'DispX', 'DispY' or 'DispZ')    
            value is an array containing the initial acceleration of each nodes        
            """
            self._SetVectorComponent(self.__Xdotdot, name, value) 
                     
    
        def SetRayleighDamping(alpha, beta):        
            """
            Compute the damping matrix from the Rayleigh's model:
            [C] = alpha*[M] + beta*[K]         

            where [C] is the damping matrix, [M] is the mass matrix and [K] is the stiffness matrix        
            Note: The rayleigh model with alpha = 0 and beta = Viscosity/YoungModulus is almost equivalent to the multi-axial Kelvin-Voigt model

            Warning: the damping matrix is not automatically updated when mass and stiffness matrix are modified.        
            """
            if len(self.__MassMatrix.shape) == 1:
                self.__DampMatrix = alpha * sparse.diags(self.__MassMatrix,format='csr') + beta * self.__StiffMatrix  
            else:
                self.__DampMatrix = alpha * self.__MassMatrix + beta * self.__StiffMatrix    
            self.__UpdateA()

        def Initialize():        
            D = 1/(self.__TimeStep**2) * self.__MassMatrix * \
                  (self.__Xold + self.__TimeStep * self.__Xdot) \
                - self.__StiffMatrix * self.__Xold        
            if self.__DampMatrix is not 0:
                D -= self.__DampMatrix * self.__Xdot

            self.SetD(D)                        

        def Update():       
            self.__Xdot = (Problem.GetDoFSolution('all') - self.__Xold)/self.__TimeStep
            self.__Xold[:] = Problem.GetDoFSolution('all')
            self.Initialize()
            
        def GetElasticEnergy(self):
            """
            returns : 0.5 * U.transposed * K * U
            """
    
            return 0.5*np.dot(self.GetDoFSolution('all') , self.__StiffMatrix*self.GetDoFSolution('all') )
    
        def GetKineticEnergy(self):
            """
            returns : 0.5 * Udot.transposed * M * Udot
            """
    
            return 0.5*np.dot(self.__Xdot , self.__MassMatrix*self.__Xdot )
        
        def GetDampingPower(self):
            """
            returns : Udot.transposed * C * Udot
            The damping disspated energy can be approximated by:
                    Edis = cumtrapz(DampingPower * TimeStep)
            """        
            return np.dot(self.__Xdot , self.__DampMatrix*self.__Xdot)

        def SetStiffnessMatrix(e):
            self.__StiffMatrix = e

        def SetMassMatrix(e):
            self.__MassMatrix = e
