import numpy as np
from fedoo.libAssembly.Assembly import *
from fedoo.libProblem.Problem   import *

#dynamical inheritance. The class is generated inside a function
def NonLinearNewmark(StiffnessAssembly, MassAssembly , Beta, Gamma, TimeStep=0.1, DampingAssembly = 0, ID = "MainProblem"):
    """
    Define a Newmark problem
    The algorithm come from:  Bathe KJ and Edward W, "Numerical methods in finite element analysis", Prentice Hall, 1976, pp 323-324    
    """
        
    if isinstance(StiffnessAssembly,str):
        StiffnessAssembly = Assembly.GetAll()[StiffnessAssembly]
                
    if isinstance(MassAssembly,str):
        MassAssembly = Assembly.GetAll()[MassAssembly]
        
    if isinstance(DampingAssembly,str):
        DampingAssembly = Assembly.GetAll()[DampingAssembly]

    if hasattr(StiffnessAssembly.GetMesh(), 'GetListMesh'): libBase = ProblemPGD
    else: libBase = Problem
    
    class __Newmark(libBase):    
            
        def __init__(self, StiffnessAssembly, MassAssembly , Beta, Gamma, TimeStep, DampingAssembly, ID):
                    
            if DampingAssembly is 0:
                A = StiffnessAssembly.GetMatrix() + 1/(Beta*(TimeStep**2))*MassAssembly.GetMatrix() #tangent matrix
            else:
                A = StiffnessAssembly.GetMatrix() + 1/(Beta*(TimeStep**2))*MassAssembly.GetMatrix() + Gamma/(Beta*TimeStep)*DampingAssembly.GetMatrix()
                
            B = 0 ; D = 0
                       
            self.__Beta       = Beta
            self.__Gamma      = Gamma
            
            # self.__MassMatrix  = MassAssembling.GetMatrix()            
            # self.__StiffMatrix = StiffnessAssembling.GetMatrix()
            # if DampingAssembly == 0: self.__DampMatrix = 0
            # else: self.__DampMatrix = DampingAssembly.GetMatrix()


            self.__MassAssembly  = MassAssembly
            self.__StiffnessAssembly = StiffnessAssembly
            self.__DampingAssembly = DampingAssembly
            self.__RayleighDamping = None

           
            self.__Displacement = self._InitializeVector(A)
            self.__DisplacementIni = self._InitializeVector(A) #displacement at the previous time iteration
            self.__Velocity = self._InitializeVector(A)
            self.__Acceleration = self._InitializeVector(A)            
            
            libBase.__init__(self,A,B,D,StiffnessAssembly.GetMesh(),ID)        
            
            
#            D = Assembling.GetVector() #initial stress vector
            self.__LoadFactor = 0    
            self.__LoadFactorIni = 0
         
            #NLSolve parameters
            self.t0 = 0 ; self.tmax = 1
            self.__TimeStep = TimeStep #initial time step
            self.dt = TimeStep #current time step
            self.__Err0 = None #initial error for NR error estimation
            self.__ErrCriterion = 'Work' #Error criterion type   
                        
            self.__iter = 0
            
            
        def __UpdateA(self): #internal function to be used when modifying M, K or C
            if self.__DampingAssembly is 0:
                self.SetA(self.__StiffnessAssembly.GetMatrix() + 1/(self.__Beta*(self.dt**2))*self.__MassAssembly.GetMatrix())
            else:
                if self.__RayleighDamping is not None:
                    #In this case, self.__RayleighDamping = [alpha, beta]
                    DampMatrix = self.__RayleighDamping[0] * self.__MassAssembly.GetMatrix() + self.__RayleighDamping[1] * self.__StiffnessAssembly.GetMatrix() 
                else: DampMatrix = self.__DampingAssembly.GetMatrix()

                self.SetA(self.__StiffnessAssembly.GetMatrix() + 1/(self.__Beta*(self.dt**2))*self.__MassAssembly.GetMatrix() + self.__Gamma/(self.__Beta*self.dt)*DampMatrix)   
        
        def __UpdateD(self): 
            D = self.__MassAssembly.GetMatrix() * ( \
                    (1/(self.__Beta*self.dt**2))*(self.__DisplacementIni - self.__DisplacementOld) +   \
                    (1/(self.__Beta*self.dt))   *self.__Velocity +   \
                    (0.5/self.__Beta - 1)               *self.__Acceleration) \
                    + self.__StiffnessAssembly.GetVector()
            if self.__DampingAssembly is not 0:
                assert 0, "Non linear Dynamic problem with damping needs to be checked"
                #need to be cheched
                if self.__RayleighDamping is not None:
                    #In this case, self.__RayleighDamping = [alpha, beta]
                    DampMatrix = self.__RayleighDamping[0] * self.__MassAssembly.GetMatrix() + self.__RayleighDamping[1] * self.__StiffnessAssembly.GetMatrix() 
                else: DampMatrix = self.__DampingAssembly.GetMatrix()
                
                D += DampMatrix * ( \
                    (self.__Gamma/(self.__Beta*self.dt))*self.__DisplacementIni +   \
                    (self.__Gamma/self.__Beta - 1)                 *self.__Velocity +   \
                    (0.5*self.dt * (self.__Gamma/self.__Beta - 2)) *self.__Acceleration) 

            self.SetD(D)       
            
        
        def Update(self, time=None, compute = 'all'):   
            """
            Assemble the matrix including the following modification:
                - New initial Stress
                - New initial Displacement
                - Modification of the mesh
                - Change in constitutive law (internal variable)
            Update the problem with the new assembled global matrix and global vector
            """
            outValues = self.__StiffnessAssembly.Update(self, time, compute)  
            self.__UpdateA()
            self.__UpdateD()
            return outValues 
            
        def NewTimeIncrement(self,time): #modifier la gestion du temps pour les CL
            LoadFactor = (time-self.t0)/(self.tmax-self.t0) #linear ramp
            # LoadFactor = 1

           # def Update(self):
           #old update function to integrate in NewTimeIncrement
            
            self.__DisplacementIni = self.__Displacement.copy()            
            self.__DisplacementOld = self.__Displacement.copy()
            self.__UpdateD()

            self.ApplyBoundaryCondition()
            try:
                self._Problem__Xbc[self._Problem__DofBlocked] *= (LoadFactor-self.__LoadFactor)
                self._Problem__B *= LoadFactor
            except:
                self._ProblemPGD__Xbc = self._ProblemPGD__Xbc*(LoadFactor-self.__LoadFactor)
                self._ProblemPGD__B *= LoadFactor             
            
            self.__LoadFactorIni = self.__LoadFactor
            self.__LoadFactor = LoadFactor

            self.__StiffnessAssembly.NewTimeIncrement()            
            
            #udpate the problem
            self.__StiffnessAssembly.ComputeGlobalMatrix(compute = 'matrix')
            self.__UpdateA()

            self.Solve()
                        
            #update total displacement            
            # self.__DisplacementOld = self.__Displacement
            self.__Displacement += self.GetDoFSolution('all')   
            self.__Err0 = None             
            
        def EndTimeIncrement(self): 
            
            NewAcceleration = (1/self.__Beta/(self.dt**2)) * (self.__Displacement - self.__DisplacementIni - self.dt*self.__Velocity) - 1/self.__Beta*(0.5 - self.__Beta)*self.__Acceleration
            self.__Velocity += self.dt * ( (1-self.__Gamma)*self.__Acceleration + self.__Gamma*NewAcceleration)
            self.__Acceleration = NewAcceleration
        
        def ResetTimeIncrement(self, update = True):                              
            self.__Displacement = self.__DisplacementIni
            self.__LoadFactor = self.__LoadFactorIni
            self.__StiffnessAssembly.ResetTimeIncrement()
            if update: self.Update()
      
        def NewtonRaphsonIncr(self):          
            try:
                self._Problem__Xbc[self._Problem__DofBlocked] *= 0 
            except:
                self._ProblemPGD__Xbc = 0
                    
            #update total displacement
            self.Solve()
            self.__DisplacementOld = self.__Displacement
            self.__Displacement += self.GetDoFSolution('all')   

        def GetDisp(self,name='all'):
            return self._GetVectorComponent(self.__Displacement, name)
               
        def GetVelocity(self):
            return self.__Velocity
    
        def GetAcceleration(self):
            return self.__Acceleration
    
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
            self._SetVectorComponent(self.__Velocity, name, value) 
    
    
        def SetInitialAcceleration(self, name,value):
            """
            name is the name of the associated variable (generaly 'DispX', 'DispY' or 'DispZ')    
            value is an array containing the initial acceleration of each nodes        
            """
            self._SetVectorComponent(self.__Acceleration, name, value) 
    
        
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
            self.__UpdateA()
    

        def SolveTimeIncrement(self,time, max_subiter = 5, ToleranceNR = 5e-3):            
            
            self.NewTimeIncrement(time)
        
            for subiter in range(max_subiter): #newton-raphson iterations                
                #update Stress and initial displacement and Update stiffness matrix
                self.Update(time, compute = 'vector')   
#                TotalStrain, TotalPKStress = self.Update()   
                        
                #Check convergence     
                normRes = self.NewtonRaphsonError()       

                if normRes < ToleranceNR:                                                  
                    return 1, subiter, normRes
                
                #--------------- Solve --------------------------------------------------------        
                self.__StiffnessAssembly.ComputeGlobalMatrix(compute = 'matrix')
                # self.SetA(self.__StiffnessAssembly.GetMatrix())
                self.__UpdateA()
                self.NewtonRaphsonIncr()
            
            return 0, subiter, normRes



        def NewtonRaphsonError(self):
            """
            Compute the error of the Newton-Raphson algorithm
            For Force and Work error criterion, the problem must be updated
            (Update method).
            """
            DofFree = self._Problem__DofFree
            if self.__Err0 is None:
                if self.__ErrCriterion == 'Displacement': 
                    self.__Err0 = np.max(np.abs(self.GetDisp())) #Displacement criterion
                elif self.__ErrCriterion == 'Force': 
                    self.__Err0 = np.max(np.abs(self.GetB()[DofFree]+self.GetD()[DofFree])) #Force criterion
                else: #self.__ErrCriterion == 'Work':
                    self.__Err0 = np.max(np.abs(self.GetDoFSolution('all')[DofFree]) * np.abs(self.GetB()[DofFree]+self.GetD()[DofFree])) #work criterion
                return 1                
            else: 
                if self.__ErrCriterion == 'Displacement': 
                    return np.max(np.abs(self.GetDoFSolution('all')))/self.__Err0  #Displacement criterion
                elif self.__ErrCriterion == 'Force':                     
                    return np.max(np.abs(self.GetB()[DofFree]+self.GetD()[DofFree]))/self.__Err0 #Force criterion
                else: #self.__ErrCriterion == 'Work':
                    return np.max(np.abs(self.GetDoFSolution('all')[DofFree]) * np.abs(self.GetB()[DofFree]+self.GetD()[DofFree]))/self.__Err0 #work criterion

       
        def SetNewtonRaphsonErrorCriterion(self, ErrorCriterion):
            if ErrorCriterion in ['Displacement', 'Force','Work']:
                self.__ErrCriterion = ErrorCriterion            
            else: 
                raise NameError('ErrCriterion must be set to "Displacement", "Force" or "Work"')





        def NLSolve(self, **kargs):              
            #parameters
            max_subiter = kargs.get('max_subiter',6)
            ToleranceNR = kargs.get('ToleranceNR',5e-3)
            self.t0 = kargs.get('t0',self.t0)
            self.tmax = kargs.get('tmax',self.tmax)
            self.dt = kargs.get('dt',self.__TimeStep)

            update_dt = kargs.get('update_dt',True)
            output = kargs.get('output', None)
            
            err_num= 2e-16 #numerical error
            time = self.t0    

            while time < self.tmax - err_num:
                time = time+self.dt
                if time > self.tmax - err_num: 
                    self.dt = self.tmax - (time-self.dt)
                    time = self.tmax          
                    
                  
                convergence, nbNRiter, normRes = self.SolveTimeIncrement(time, max_subiter, ToleranceNR)

                if not(convergence):
                    if update_dt:
                        time = time - self.dt
                        self.dt *= 0.25
                        print('NR failed to converge (err: {:.5f}) - reduce the time increment to {:.5f}'.format(normRes, self.dt))
                        self.ResetTimeIncrement()   
                        #update Stress, initial displacement and assemble global matrix
                        self.Update(time)                                 
                        continue                    
                    else: 
                        raise NameError('Newton Raphson iteration has not converged (err: {:.5f})- Reduce the time step or use update_dt = True'.format(normRes))   
                
                self.EndTimeIncrement()
                print('Iter {} - Time: {:.5f} - NR iter: {} - Err: {:.5f}'.format(self.__iter, time, nbNRiter, normRes))
                if output is not None: output(self, self.__iter, time, nbNRiter, normRes)                  

                self.__iter += 1   

                if update_dt and nbNRiter < 2: 
                    self.dt *= 1.25
                    print('Increase the time increment to {:.5f}'.format(dt))               
                                                         
                    
                    
                    
                    




        # def GetElasticEnergy(self):
        #     """
        #     returns : sum(0.5 * U.transposed * K * U)
        #     """
    
        #     return 0.5*np.dot(self.GetDoFSolution('all') , self.__StiffMatrix*self.GetDoFSolution('all') )
                            
        # def GetNodalElasticEnergy(self):
        #     """
        #     returns : 0.5 * K * U . U
        #     """
    
        #     E = 0.5*self.GetDoFSolution('all').transpose() * self.GetA() * self.GetDoFSolution('all')

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
        
        # def UpdateStiffness(self, StiffnessAssembling):
        #     if isinstance(StiffnessAssembling,str):
        #         StiffnessAssembling = Assembly.GetAll()[StiffnessAssembling]
        #     self.__StiffMatrix = StiffnessAssembling.GetMatrix()
        #     self.__UpdateA()
    
    return __Newmark(StiffnessAssembly, MassAssembly , Beta, Gamma, TimeStep, DampingAssembly, ID)





