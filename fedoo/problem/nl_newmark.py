import numpy as np
from fedoo.core.assembly import Assembly
from fedoo.core.problem import Problem
from fedoo.problem.nl_static import _GenerateClass_NonLinearStatic

#dynamical inheritance. The class is generated inside a function
def _GenerateClass_NonLinearNewmark(libBase):
    
    NLStaticClass = _GenerateClass_NonLinearStatic(libBase)
    class __Newmark(NLStaticClass):    
            
        def __init__(self, StiffnessAssembly, MassAssembly , Beta, Gamma, DampingAssembly, name):
            NLStaticClass.__init__(self, StiffnessAssembly, name)  
            
            # if DampingAssembly is 0:
            #     A = StiffnessAssembly.get_global_matrix() + 1/(Beta*(TimeStep**2))*MassAssembly.get_global_matrix() #tangent matrix
            # else:
            #     A = StiffnessAssembly.get_global_matrix() + 1/(Beta*(TimeStep**2))*MassAssembly.get_global_matrix() + Gamma/(Beta*TimeStep)*DampingAssembly.get_global_matrix()
                
            # B = 0 ; D = 0
                       
            self.__Beta       = Beta
            self.__Gamma      = Gamma

            self.__MassAssembly  = MassAssembly
            self.__StiffnessAssembly = StiffnessAssembly #alias of self._NLStaticClass__Assembly 
            self.__DampingAssembly = DampingAssembly
            self.__RayleighDamping = None

            # self.__Displacement = self._new_vect_dof(A)
            # self.__DisplacementStart = self._new_vect_dof(A) #displacement at the previous time iteration
            self.__Velocity = 0
            self.__Acceleration = 0        
                        
        def updateA(self, dt): #internal function to be used when modifying M, K or C
            if self.__DampingAssembly is 0:
                self.SetA(self.__StiffnessAssembly.get_global_matrix() + 1/(self.__Beta*(dt**2))*self.__MassAssembly.get_global_matrix())
            else:
                if self.__RayleighDamping is not None:
                    #In this case, self.__RayleighDamping = [alpha, beta]
                    DampMatrix = self.__RayleighDamping[0] * self.__MassAssembly.get_global_matrix() + self.__RayleighDamping[1] * self.__StiffnessAssembly.get_global_matrix() 
                else: DampMatrix = self.__DampingAssembly.get_global_matrix()

                self.SetA(self.__StiffnessAssembly.get_global_matrix() + 1/(self.__Beta*(dt**2))*self.__MassAssembly.get_global_matrix() + self.__Gamma/(self.__Beta*dt)*DampMatrix)   
        
        def updateD(self, dt, start=False):
            #start = True if begining of a new time increment (ie  DispOld-DispStart = 0)
            if start:
                DeltaDisp = 0 #DeltaDisp = Disp-DispStart = 0 for the 1st increment              
                if self.__Velocity is 0 and self.__Acceleration is 0: 
                    self.SetD(0)
                    return
            else: 
                # DeltaDisp = self._NonLinearStatic__TotalDisplacementOld - self._NonLinearStatic__TotalDisplacementStart
                DeltaDisp = self._NonLinearStatic__DU
            
            D = self.__MassAssembly.get_global_matrix() * ( \
                    (1/(self.__Beta*dt**2))*DeltaDisp +   \
                    (1/(self.__Beta*dt))   *self.__Velocity +   \
                    (0.5/self.__Beta - 1)               *self.__Acceleration) \
                    + self.__StiffnessAssembly.get_global_vector()
            if self.__DampingAssembly is not 0:
                assert 0, "Non linear Dynamic problem with damping needs to be checked"
                #need to be cheched
                if self.__RayleighDamping is not None:
                    #In this case, self.__RayleighDamping = [alpha, beta]
                    DampMatrix = self.__RayleighDamping[0] * self.__MassAssembly.get_global_matrix() + self.__RayleighDamping[1] * self.__StiffnessAssembly.get_global_matrix() 
                else: DampMatrix = self.__DampingAssembly.get_global_matrix()
                
                D += DampMatrix * ( \
                    (self.__Gamma/(self.__Beta*dt))*DisplacementStart +   \
                    (self.__Gamma/self.__Beta - 1)                 *self.__Velocity +   \
                    (0.5*dt * (self.__Gamma/self.__Beta - 2)) *self.__Acceleration) 
                
            self.SetD(D)       
        
        
        def initialize(self, t0):   
            """
            """
            self.__MassAssembly.initialize(self,t0)
            self.__StiffnessAssembly.initialize(self,t0)
            if self.__DampingAssembly is not 0 and self.__RayleighDamping is None:
                self.__DampingAssembly.initialize(self,t0)       
        
        def NewTimeIncrement(self, dt):         
            ### dt is the time step of the previous increment            
            self.__MassAssembly.NewTimeIncrement()
            self.__StiffnessAssembly.NewTimeIncrement()
            if self.__DampingAssembly is not 0:
                self.__DampingAssembly.NewTimeIncrement()
            
            #update velocity and acceleration
            NewAcceleration = (1/self.__Beta/(dt**2)) * (self._NonLinearStatic__DU - dt*self.__Velocity) - 1/self.__Beta*(0.5 - self.__Beta)*self.__Acceleration
            self.__Velocity += dt * ( (1-self.__Gamma)*self.__Acceleration + self.__Gamma*NewAcceleration)
            self.__Acceleration = NewAcceleration
            
            self._NonLinearStatic__Utot += self._NonLinearStatic__DU
            self._NonLinearStatic__DU = 0

            
        def to_start(self):     
            self._NonLinearStatic__DU = 0
            
            self.__MassAssembly.to_start()
            self.__StiffnessAssembly.to_start()
            if self.__DampingAssembly is not 0:
                self.__DampingAssembly.to_start()                                
        
        def update(self, dtime=None, compute = 'all', updateWeakForm = True):   
            """
            Assemble the matrix including the following modification:
                - New initial Stress
                - New initial Displacement
                - Modification of the mesh
                - Change in constitutive law (internal variable)
            Update the problem with the new assembled global matrix and global vector
            """
            if updateWeakForm == True:
                self.__StiffnessAssembly.update(self, dtime, compute)  
            else: 
                self.__StiffnessAssembly.assemble_global_mat(compute)

        def reset(self):
            self.__MassAssembly.reset()
            self.__StiffnessAssembly.reset()
            if self.__DampingAssembly is not 0:
                self.__DampingAssembly.reset()            
            self.SetA(0) #tangent stiffness 
            self.SetD(0)                 
            # self.SetA(self.__Assembly.get_global_matrix()) #tangent stiffness 
            # self.SetD(self.__Assembly.get_global_vector())            

            B = 0
            self._NonLinearStatic__Utot = 0
            self._NonLinearStatic__DU = 0
            self.__Velocity = 0
            self.__Acceleration = 0                    
            
            
            self.__Err0 = None #initial error for NR error estimation   
            self.t0 = 0 ; self.tmax = 1
            self.__iter = 0  
            self.apply_boundary_conditions() #perhaps not usefull here as the BC will be applied in the NewTimeIncrement method ?
        
        def GetAssembly(self):
            return self.__StiffnessAssembly
    
            
            
            
        # def NewTimeIncrement(self,time): #modifier la gestion du temps pour les CL
        #     LoadFactor = (time-self.t0)/(self.tmax-self.t0) #linear ramp
        #     # LoadFactor = 1

        #    # def update(self):
        #    #old update function to integrate in NewTimeIncrement
            
        #     self.__DisplacementStart = self.__Displacement.copy()            
        #     self.__DisplacementOld = self.__Displacement.copy()
        #     self.__UpdateD()

        #     self.apply_boundary_conditions()
        #     try:
        #         self._Problem__Xbc[self._Problem__DofBlocked] *= (LoadFactor-self.__LoadFactor)
        #         self._Problem__B *= LoadFactor
        #     except:
        #         self._ProblemPGD__Xbc = self._ProblemPGD__Xbc*(LoadFactor-self.__LoadFactor)
        #         self._ProblemPGD__B *= LoadFactor             
            
        #     self.__LoadFactorIni = self.__LoadFactor
        #     self.__LoadFactor = LoadFactor

        #     self.__StiffnessAssembly.NewTimeIncrement()            
            
        #     #udpate the problem
        #     self.__StiffnessAssembly.assemble_global_mat(compute = 'matrix')
        #     self.__UpdateA()

        #     self.solve()
                        
        #     #update total displacement            
        #     # self.__DisplacementOld = self.__Displacement
        #     self.__Displacement += self.GetX()   
        #     self.__Err0 = None             
            
        # def EndTimeIncrement(self): 
            
        #     NewAcceleration = (1/self.__Beta/(self.dt**2)) * (self.__Displacement - self.__DisplacementStart - self.dt*self.__Velocity) - 1/self.__Beta*(0.5 - self.__Beta)*self.__Acceleration
        #     self.__Velocity += self.dt * ( (1-self.__Gamma)*self.__Acceleration + self.__Gamma*NewAcceleration)
        #     self.__Acceleration = NewAcceleration
        
        # def to_start(self, update = True):                              
        #     self.__Displacement = self.__DisplacementStart
        #     self.__LoadFactor = self.__LoadFactorIni
        #     self.__StiffnessAssembly.to_start()
        #     if update: self.update()
      
        # def NewtonRaphsonIncr(self):          
        #     try:
        #         self._Problem__Xbc[self._Problem__DofBlocked] *= 0 
        #     except:
        #         self._ProblemPGD__Xbc = 0
                    
        #     #update total displacement
        #     self.solve()
        #     self.__DisplacementOld = self.__Displacement
        #     self.__Displacement += self.GetX()   

        # def get_disp(self,name='all'):
        #     # return self._get_vect_component(self.__Displacement, name)
               
        def GetVelocity(self):
            return self.__Velocity
    
        def GetAcceleration(self):
            return self.__Acceleration
    
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
            self._set_vect_component(self.__Velocity, name, value) 
    
    
        def SetInitialAcceleration(self, name,value):
            """
            name is the name of the associated variable (generaly 'DispX', 'DispY' or 'DispZ')    
            value is an array containing the initial acceleration of each nodes        
            """
            self._set_vect_component(self.__Acceleration, name, value) 
    
        
        def SetRayleighDamping(self, alpha, beta):        
            """
            Compute the damping matrix from the Rayleigh's model:
            [C] = alpha*[M] + beta*[K]         
    
            where [C] is the damping matrix, [M] is the mass matrix and [K] is the stiffness matrix        
            Note: The rayleigh model with alpha = 0 and beta = Viscosity/YoungModulus is almost equivalent to the multi-axial Kelvin-Voigt model
            
            Warning: the damping matrix is not automatically updated when mass and stiffness matrix are modified.        
            """
            
            self.__RayleighDamping = [alpha, beta]
            self.__DampingAssembly = 'Rayleigh'    

#         def SolveTimeIncrement(self,time, max_subiter = 5, ToleranceNR = 5e-3):            
            
#             self.NewTimeIncrement(time)
        
#             for subiter in range(max_subiter): #newton-raphson iterations                
#                 #update Stress and initial displacement and Update stiffness matrix
#                 self.update(time, compute = 'vector')   
# #                TotalStrain, TotalPKStress = self.update()   
                        
#                 #Check convergence     
#                 normRes = self.NewtonRaphsonError()       

#                 if normRes < ToleranceNR:                                                  
#                     return 1, subiter, normRes
                
#                 #--------------- Solve --------------------------------------------------------        
#                 self.__StiffnessAssembly.assemble_global_mat(compute = 'matrix')
#                 # self.SetA(self.__StiffnessAssembly.get_global_matrix())
#                 self.__UpdateA()
#                 self.NewtonRaphsonIncr()
            
#             return 0, subiter, normRes



    

                                                         
                    
                    
                    
                    




        # def GetElasticEnergy(self):
        #     """
        #     returns : sum(0.5 * U.transposed * K * U)
        #     """
    
        #     return 0.5*np.dot(self.GetX() , self.__StiffMatrix*self.GetX() )
                            
        # def GetNodalElasticEnergy(self):
        #     """
        #     returns : 0.5 * K * U . U
        #     """
    
        #     E = 0.5*self.GetX().transpose() * self.GetA() * self.GetX()

        #     E = np.reshape(E,(3,-1)).T
            
        #     return E
        
        # def GetKineticEnergy(self):
        #     """
        #     returns : 0.5 * Udot.transposed * M * Udot
        #     """
    
        #     return 0.5*np.dot(self.__Xdot , self.__MassMatrix*self.__Xdot )
        
        # def GetDampingPower(self):
        #     """
        #     returns : Udot.transposed * C * Udot
        #     The damping disspated energy can be approximated by:
        #             Edis = DampingPower * TimeStep
        #     or
        #             Edis = scipy.integrate.cumtrapz(t,DampingPower)
        #     """        
        #     return np.dot(self.__Xdot , self.__DampMatrix*self.__Xdot)
        
        # def GetExternalForceWork(self):
        #     """
        #     with (KU + CU_dot + MU_dot_dot) = Fext
        #     this function returns sum(Fext.(U-Uold))
        #     """
        #     K = self.__StiffMatrix
        #     M = self.__MassMatrix
        #     C = self.__DampMatrix
        #     return np.sum((K*self.GetX() + C*self.GetXdot() + M*self.GetXdotdot())*(self.GetX()-self.__Xold))
        
        # def updateStiffness(self, StiffnessAssembling):
        #     if isinstance(StiffnessAssembling,str):
        #         StiffnessAssembling = Assembly.get_all()[StiffnessAssembling]
        #     self.__StiffMatrix = StiffnessAssembling.get_global_matrix()
        #     self.__UpdateA()
    
    return __Newmark

def NonLinearNewmark(StiffnessAssembly, MassAssembly , Beta, Gamma, DampingAssembly = 0, name = "MainProblem"):
    """
    Define a Newmark problem
    The algorithm come from:  Bathe KJ and Edward W, "Numerical methods in finite element analysis", Prentice Hall, 1976, pp 323-324    
    """
        
    if isinstance(StiffnessAssembly,str):
        StiffnessAssembly = Assembly.get_all()[StiffnessAssembly]
                
    if isinstance(MassAssembly,str):
        MassAssembly = Assembly.get_all()[MassAssembly]
        
    if isinstance(DampingAssembly,str):
        DampingAssembly = Assembly.get_all()[DampingAssembly]

    if hasattr(StiffnessAssembly.mesh, 'GetListMesh'): libBase = ProblemPGD
    else: libBase = Problem

    __Newmark = _GenerateClass_NonLinearNewmark(libBase)
    return __Newmark(StiffnessAssembly, MassAssembly , Beta, Gamma, DampingAssembly, name)

