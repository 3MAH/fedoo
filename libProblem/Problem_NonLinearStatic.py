# import numpy as np
from fedoo.libAssembly.Assembly import Assembly
from fedoo.libProblem.Problem   import *
from fedoo.libProblem.ProblemPGD   import ProblemPGD

#dynamical inheritance. The class is generated inside a function
def NonLinearStatic(Assembling, ID = "MainProblem"):
    if isinstance(Assembling,str):
        Assembling = Assembly.GetAll()[Assembling]
               
    if hasattr(Assembling.GetMesh(), 'GetListMesh'): libBase = ProblemPGD
    else: libBase = Problem            
    
    class __NonLinearStatic(libBase):
        
        def __init__(self, Assembling, ID):                                    
            A = Assembling.GetMatrix() #tangent stiffness matrix
            B = 0             
            D = Assembling.GetVector() #initial stress vector
            self.__TotalDisplacement = 0
            self.__TotalDisplacementOld = 0
            self.__LoadFactor = 0    
            self.__LoadFactorIni = 0
            self.__Err0 = None #initial error for NR error estimation
            self.__ErrCriterion = 'Work' #Error criterion type   
            self.__Assembly = Assembling
            libBase.__init__(self,A,B,D,Assembling.GetMesh(), ID)        
            self.t0 = 0 ; self.tmax = 1
            self.__iter = 0
        
        def GetDisp(self,name='all'):
            return self._GetVectorComponent(self.__TotalDisplacement, name)
        
        # def ResetLoadFactor(self):
        #     self.__LoadFactor = 0    
        #     self.__LoadFactorIni = 0            
        
        def NewTimeIncrement(self,time,timeOld):
            self.__TotalDisplacementIni = self.__TotalDisplacement

            # LoadFactor = (time-self.t0)/(self.tmax-self.t0)
            timeFactor    = (time-self.t0)/(self.tmax-self.t0) #adimensional time            
            timeFactorOld = (timeOld-self.t0)/(self.tmax-self.t0)

            self.ApplyBoundaryCondition(timeFactor, timeFactorOld)
            # try:
            #     self._Problem__Xbc *= (LoadFactor-self.__LoadFactor)
            #     self._Problem__B *= LoadFactor
            # except:
            #     self._ProblemPGD__Xbc = self._ProblemPGD__Xbc*(LoadFactor-self.__LoadFactor)
            #     self._ProblemPGD__B *= LoadFactor             
                               
            # self.__LoadFactorIni = self.__LoadFactor
            # self.__LoadFactor = LoadFactor

            self.__Assembly.NewTimeIncrement()            
            #udpate the problem (no need to update the week form and vector because no change of state should have occur since the update prior to error estimation)
            self.Update(compute = 'matrix', updateWeakForm = False)
            # self.__Assembly.ComputeGlobalMatrix(compute = 'matrix')
            # self.SetA(self.__Assembly.GetMatrix())

            self.Solve()
                        
            #update total displacement            
            self.__TotalDisplacementOld = self.__TotalDisplacement
            self.__TotalDisplacement += self.GetDoFSolution('all')   
            self.__Err0 = None
            # self.NewtonRaphsonError() 
            # print(self.__Err0)

        def ResetTimeIncrement(self, time = None, update = True):                              
            self.__TotalDisplacement = self.__TotalDisplacementIni
            # self.__LoadFactor = self.__LoadFactorIni
            self.__Assembly.ResetTimeIncrement()
            if update: self.Update(time)
      
        def NewtonRaphsonIncr(self):                   
            try:
                self._Problem__Xbc *= 0 
            except:
                self._ProblemPGD__Xbc = 0
                    
            #update total displacement
            self.Solve()
            self.__TotalDisplacementOld = self.__TotalDisplacement
            self.__TotalDisplacement += self.GetDoFSolution('all')   
        
        def Update(self, time=None, compute = 'all', updateWeakForm = True):   
            """
            Assemble the matrix including the following modification:
                - New initial Stress
                - New initial Displacement
                - Modification of the mesh
                - Change in constitutive law (internal variable)
            Update the problem with the new assembled global matrix and global vector
            """
            if updateWeakForm == True:
                outValues = self.__Assembly.Update(self, time, compute)
            else: 
                outValues = None
                self.__Assembly.ComputeGlobalMatrix(compute)
            self.SetA(self.__Assembly.GetMatrix())
            self.SetD(self.__Assembly.GetVector())
            return outValues 

        def Reset(self):
            self.__Assembly.Reset()
            
            self.SetA(self.__Assembly.GetMatrix()) #tangent stiffness 
            self.SetD(self.__Assembly.GetVector())            

            B = 0
            self.__TotalDisplacement = 0
            self.__TotalDisplacementOld = 0
            # self.__LoadFactor = 0    
            # self.__LoadFactorIni = 0
            self.__Err0 = None #initial error for NR error estimation            
            self.__iter = 0  
            self.ApplyBoundaryCondition()
        
        def ChangeAssembly(self,Assembling, update = True):
            """
            Modify the assembly associated to the problem and update the problem (see Assembly.Update for more information)
            """
            if isinstance(Assembling,str):
                Assembling = Assembly.GetAll()[Assembling]
                
            self.__Assembly = Assembling
            if update: self.Update()
            
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
                else:
                    self.__Err0 = 1
                    self.__Err0 = self.NewtonRaphsonError() 
                return 1                
            else: 
                if self.__ErrCriterion == 'Displacement': 
                    return np.max(np.abs(self.GetDoFSolution('all')))/self.__Err0  #Displacement criterion
                elif self.__ErrCriterion == 'Force': #Force criterion              
                    if self.GetD() is 0: return np.max(np.abs(self.GetB()[DofFree]))/self.__Err0 
                    else: return np.max(np.abs(self.GetB()[DofFree]+self.GetD()[DofFree]))/self.__Err0                     
                else: #self.__ErrCriterion == 'Work': #work criterion
                    if self.GetD() is 0: return np.max(np.abs(self.GetDoFSolution('all')[DofFree]) * np.abs(self.GetB()[DofFree]))/self.__Err0 
                    else: return np.max(np.abs(self.GetDoFSolution('all')[DofFree]) * np.abs(self.GetB()[DofFree]+self.GetD()[DofFree]))/self.__Err0 
       
        def SetNewtonRaphsonErrorCriterion(self, ErrorCriterion):
            if ErrorCriterion in ['Displacement', 'Force','Work']:
                self.__ErrCriterion = ErrorCriterion            
            else: 
                raise NameError('ErrCriterion must be set to "Displacement", "Force" or "Work"')
        
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
        
        def SolveTimeIncrement(self, time, timeOld, max_subiter = 5, ToleranceNR = 5e-3):            
            
            self.NewTimeIncrement(time, timeOld)
            # self.__Err0 = 1
            
            for subiter in range(max_subiter): #newton-raphson iterations                
                #update Stress and initial displacement and Update stiffness matrix
                self.Update(time, compute = 'vector')   
#                TotalStrain, TotalPKStress = self.Update()   

                #Check convergence     
                normRes = self.NewtonRaphsonError()    

                if normRes < ToleranceNR:                                                  
                    return 1, subiter, normRes
                
                #--------------- Solve --------------------------------------------------------        
                # self.__Assembly.ComputeGlobalMatrix(compute = 'matrix')
                # self.SetA(self.__Assembly.GetMatrix())
                self.Update(compute = 'matrix', updateWeakForm = False)
                self.NewtonRaphsonIncr()
            
            return 0, subiter, normRes


        def NLSolve(self, **kargs):              
            #parameters
            max_subiter = kargs.get('max_subiter',6)
            ToleranceNR = kargs.get('ToleranceNR',5e-3)
            self.t0 = kargs.get('t0',self.t0)
            self.tmax = kargs.get('tmax',self.tmax)
            dt = kargs.get('dt',0.1)
            update_dt = kargs.get('update_dt',True)
            output = kargs.get('output', None)
            
            err_num= 2e-16 #numerical error
            time = self.t0    

            while time < self.tmax - err_num:
                time = time+dt
                if time > self.tmax - err_num: time = self.tmax          
                  
                convergence, nbNRiter, normRes = self.SolveTimeIncrement(time, time-dt, max_subiter, ToleranceNR)

                if not(convergence) :
                    if update_dt:
                        time = time - dt
                        dt *= 0.25
                        print('NR failed to converge (err: {:.5f}) - reduce the time increment to {:.5f}'.format(normRes, dt))
                        #reset internal variables, update Stress, initial displacement and assemble global matrix at previous time              
                        self.ResetTimeIncrement(time)  
                        continue                    
                    else: 
                        raise NameError('Newton Raphson iteration has not converged (err: {:.5f})- Reduce the time step or use update_dt = True'.format(normRes))   
                    
                print('Iter {} - Time: {:.5f} - NR iter: {} - Err: {:.5f}'.format(self.__iter, time, nbNRiter, normRes))
                if output is not None: output(self, self.__iter, time, nbNRiter, normRes)                  

                self.__iter += 1   

                if update_dt and nbNRiter < 2: 
                    dt *= 1.25
                    print('Increase the time increment to {:.5f}'.format(dt))               
                                                         
        

    return __NonLinearStatic(Assembling, ID)
