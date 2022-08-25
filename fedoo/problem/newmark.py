import numpy as np
from fedoo.core.assembly import Assembly
from fedoo.core.problem import Problem

#dynamical inheritance. The class is generated inside a function
def Newmark(StiffnessAssembling, MassAssembling , Beta, Gamma, TimeStep, DampingAssembling = 0, name = "MainProblem"):
    """
    Define a Newmark problem
    The algorithm come from:  Bathe KJ and Edward W, "Numerical methods in finite element analysis", Prentice Hall, 1976, pp 323-324    
    """
        
    if isinstance(StiffnessAssembling,str):
        StiffnessAssembling = Assembly.get_all()[StiffnessAssembling]
                
    if isinstance(MassAssembling,str):
        MassAssembling = Assembly.get_all()[MassAssembling]
        
    if isinstance(DampingAssembling,str):
        DampingAssembling = Assembly.get_all()[DampingAssembling]

    if hasattr(StiffnessAssembling.mesh, 'GetListMesh'): libBase = ProblemPGD
    else: libBase = Problem
    
    class __Newmark(libBase):    
            
        def __init__(self, StiffnessAssembling, MassAssembling , Beta, Gamma, TimeStep, DampingAssembling, name):
                    
            if DampingAssembling is 0:
                A = StiffnessAssembling.get_global_matrix() + 1/(Beta*(TimeStep**2))*MassAssembling.get_global_matrix()
            else:
                A = StiffnessAssembling.get_global_matrix() + 1/(Beta*(TimeStep**2))*MassAssembling.get_global_matrix() + Gamma/(Beta*TimeStep)*DampingAssembling.get_global_matrix()
                
            B = 0 ; D = 0
            
            self.__Xold    = self._new_vect_dof(A) #displacement at the previous time step        
            self.__Xdot    = self._new_vect_dof(A)
            self.__Xdotdot = self._new_vect_dof(A)
            
            self.__Beta       = Beta
            self.__Gamma      = Gamma
            self.__TimeStep   = TimeStep
            
            self.__MassMatrix  = MassAssembling.get_global_matrix()
            self.__StiffMatrix = StiffnessAssembling.get_global_matrix()
            if DampingAssembling == 0: self.__DampMatrix = 0
            else: self.__DampMatrix = DampingAssembling.get_global_matrix()
            
            libBase.__init__(self,A,B,D,StiffnessAssembling.mesh,name)        
    
        def __UpdateA(self): #internal function to be used when modifying M, K or C
            if self.__DampMatrix is 0:
                self.SetA(self.__StiffMatrix + 1/(self.__Beta*(self.__TimeStep**2))*self.__MassMatrix)
            else:
                self.SetA(self.__StiffMatrix + 1/(self.__Beta*(self.__TimeStep**2))*self.__MassMatrix + self.__Gamma/(self.__Beta*self.__TimeStep)*self.__DampMatrix)   
    
        def GetX(self):
            return self.GetDoFSolution('all')
        
        def GetXdot(self):
            return self.__Xdot
    
        def GetXdotdot(self):
            return self.__Xdotdot
    
        def get_disp(self, name = 'all'): #same as GetX
            return self.GetDoFSolution(name)        
               
        def GetVelocity(self): #same as GetXdot
            return self.__Xdot
    
        def GetAcceleration(self): #same as GetXdotdot
            return self.__Xdotdot
    
        def SetInitialDisplacement(self, name,value):
            """
            name is the name of the associated variable (generaly 'DispX', 'DispY' or 'DispZ')    
            value is an array containing the initial displacement of each nodes
            """        
            self._set_vect_component(self.__Xold, name, value)          
        
        def SetInitialVelocity(self, name,value):
            """
            name is the name of the associated variable (generaly 'DispX', 'DispY' or 'DispZ')    
            value is an array containing the initial velocity of each nodes        
            """
            self._set_vect_component(self.__Xdot, name, value) 
    
    
        def SetInitialAcceleration(self, name,value):
            """
            name is the name of the associated variable (generaly 'DispX', 'DispY' or 'DispZ')    
            value is an array containing the initial acceleration of each nodes        
            """
            self._set_vect_component(self.__Xdotdot, name, value) 
    
        
        def SetRayleighDamping(self, alpha, beta):        
            """
            Compute the damping matrix from the Rayleigh's model:
            [C] = alpha*[M] + beta*[K]         
    
            where [C] is the damping matrix, [M] is the mass matrix and [K] is the stiffness matrix        
            Note: The rayleigh model with alpha = 0 and beta = Viscosity/YoungModulus is almost equivalent to the multi-axial Kelvin-Voigt model
            
            Warning: the damping matrix is not automatically updated when mass and stiffness matrix are modified.        
            """
    
            self.__DampMatrix = alpha * self.__MassMatrix + beta * self.__StiffMatrix    
            self.__UpdateA()
    
        def initialize(self, t0=0.):
            D = self.__MassMatrix * ( \
                    (1/(self.__Beta*self.__TimeStep**2))*self.__Xold +   \
                    (1/(self.__Beta*self.__TimeStep))   *self.__Xdot +   \
                    (0.5/self.__Beta - 1)                  *self.__Xdotdot) 
            if self.__DampMatrix is not 0:
                D += self.__DampMatrix * ( \
                    (self.__Gamma/(self.__Beta*self.__TimeStep))*self.__Xold +   \
                    (self.__Gamma/self.__Beta - 1)                 *self.__Xdot +   \
                    (0.5*self.__TimeStep * (self.__Gamma/self.__Beta - 2)) *self.__Xdotdot)                                                                                      
            self.SetD(D)                        
    
        def update(self):
           
            NewXdotdot = (1/self.__Beta/(self.__TimeStep**2)) * (self.GetDoFSolution('all') - self.__Xold - self.__TimeStep*self.__Xdot) - 1/self.__Beta*(0.5 - self.__Beta)*self.__Xdotdot
            self.__Xdot += self.__TimeStep * ( (1-self.__Gamma)*self.__Xdotdot + self.__Gamma*NewXdotdot)
            self.__Xdotdot = NewXdotdot
            self.__Xold[:] = self.GetDoFSolution('all')
            self.initialize()
    #        self.SetD(self.__MassMatrix * ( (1/self.__Beta/(self.__TimeStep**2))*self.__Xold + (1/self.__Beta/self.__TimeStep)*self.__Xdot + (1/2/self.__Beta -1)*self.__Xdotdot) )
            
        def GetElasticEnergy(self):
            """
            returns : sum(0.5 * U.transposed * K * U)
            """
    
            return 0.5*np.dot(self.GetDoFSolution('all') , self.__StiffMatrix*self.GetDoFSolution('all') )
                            
        def GetNodalElasticEnergy(self):
            """
            returns : 0.5 * K * U . U
            """
    
            E = 0.5*self.GetDoFSolution('all').transpose() * self.GetA() * self.GetDoFSolution('all')

            E = np.reshape(E,(3,-1)).T
            
            return E
        
        def GetKineticEnergy(self):
            """
            returns : 0.5 * Udot.transposed * M * Udot
            """
    
            return 0.5*np.dot(self.__Xdot , self.__MassMatrix*self.__Xdot )
        
        def GetDampingPower(self):
            """
            returns : Udot.transposed * C * Udot
            The damping disspated energy can be approximated by:
                    Edis = DampingPower * TimeStep
            or
                    Edis = scipy.integrate.cumtrapz(t,DampingPower)
            """        
            return np.dot(self.__Xdot , self.__DampMatrix*self.__Xdot)
        
        def GetExternalForceWork(self):
            """
            with (KU + CU_dot + MU_dot_dot) = Fext
            this function returns sum(Fext.(U-Uold))
            """
            K = self.__StiffMatrix
            M = self.__MassMatrix
            C = self.__DampMatrix
            return np.sum((K*self.GetX() + C*self.GetXdot() + M*self.GetXdotdot())*(self.GetX()-self.__Xold))
        
        def updateStiffness(self, StiffnessAssembling):
            if isinstance(StiffnessAssembling,str):
                StiffnessAssembling = Assembly.get_all()[StiffnessAssembling]
            self.__StiffMatrix = StiffnessAssembling.get_global_matrix()
            self.__UpdateA()
    
    return __Newmark(StiffnessAssembling, MassAssembling , Beta, Gamma, TimeStep, DampingAssembling, name)




