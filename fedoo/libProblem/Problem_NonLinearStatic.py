# import numpy as np
from fedoo.libAssembly.Assembly import Assembly
from fedoo.libProblem.Problem   import *
from fedoo.libProblem.ProblemPGD   import ProblemPGD
from fedoo.libUtil.ExportData import _ProblemOutput

#dynamical inheritance. The class is generated inside a function
def NonLinearStatic(Assembling, ID = "MainProblem"):
    if isinstance(Assembling,str):
        Assembling = Assembly.GetAll()[Assembling]
               
    if hasattr(Assembling.GetMesh(), 'GetListMesh'): libBase = ProblemPGD
    else: libBase = Problem            
    
    class __NonLinearStatic(libBase):
        
        def __init__(self, Assembling, ID):                                  
            #A = Assembling.GetMatrix() #tangent stiffness matrix
            A = 0 #tangent stiffness matrix - will be initialized only when required
            B = 0             
            #D = Assembling.GetVector() #initial stress vector
            D = 0 #initial stress vector #will be initialized later
            self.__TotalDisplacement = self.__TotalDisplacementStart = 0
            self.__TotalDisplacementOld = 0
            self.__Err0 = None #initial error for NR error estimation
            self.__ErrCriterion = 'Work' #Error criterion type   
            self.__Assembly = Assembling
            libBase.__init__(self,A,B,D,Assembling.GetMesh(), ID)        
            self.t0 = 0 ; self.tmax = 1
            self.__iter = 0
            self.__compteurOutput = 0
            self.__outputIter = 1 #save results every self.__outputIter iter or time step if self.__saveOutputAtExactTime = True
            self.__saveOutputAtExactTime = True
            self.err_num= 1e-8 #numerical error
            
            self.__ProblemOutput = _ProblemOutput()
        
        def GetDisp(self,name='all'):            
            return self._GetVectorComponent(self.__TotalDisplacement, name)
        
        #same as GetDisp but in future for thermomechanical problem, GetDisp will only return the displacement part
        def GetDoFSolution(self,name='all'):
            return self._GetVectorComponent(self.__TotalDisplacement, name)
        
        def GetExternalForce(self, name = 'all'):
            return self._GetVectorComponent(-self.GetD(), name)        
        
        def Initialize(self, initialTime=0.):   
            """
            """
            self.__Assembly.Initialize(self,initialTime)
            self.SetA(self.__Assembly.GetMatrix())
            self.SetD(self.__Assembly.GetVector())
        
        def ElasticPrediction(self, timeOld, dt):
            #update the boundary conditions with the time variation
            time = timeOld + dt
            timeFactor    = (time-self.t0)/(self.tmax-self.t0) #adimensional time            
            timeFactorOld = (timeOld-self.t0)/(self.tmax-self.t0)

            self.ApplyBoundaryCondition(timeFactor, timeFactorOld)
            
            #solve the linearized system with elastic rigidty matrix           
            self.Solve()        

            #the the increment Dirichlet boundray conditions to 0 (i.e. will not change during the NR interations)            
            try:
                self._Problem__Xbc *= 0 
            except:
                self._ProblemPGD__Xbc = 0

            #set the reference error to None to for a new estimation of err0            
            self.__Err0 = None

            #update total displacement            
            if self.__TotalDisplacement is not 0: self.__TotalDisplacementOld = self.__TotalDisplacement.copy()
            self.__TotalDisplacement += self.GetX()               
        
        def NewTimeIncrement(self):
            self.__TotalDisplacementStart = self.__TotalDisplacement.copy()

            # if timeOld == 0.:
            #     #First iteration ->initialize the A and D matrix. 
            #     self.Update(0.)
            # else: #check if this is usefull 
            #     #udpate the problem (no need to update the week form and vector because  no change of state should have occur since the update prior to error estimation)
            #     self.Update(compute = 'matrix', updateWeakForm = False) 
                       
            self.__Assembly.NewTimeIncrement()     
            
            self.SetA(self.__Assembly.GetMatrix()) #should be the elastic rigidity matrix
            self.SetD(self.__Assembly.GetVector()) #not modified in principle
            
            

        def ResetTimeIncrement(self):                              
            self.__TotalDisplacement = self.__TotalDisplacementStart.copy()
            self.__Assembly.ResetTimeIncrement()
            
            self.SetA(self.__Assembly.GetMatrix()) #should be the elastic rigidity matrix
            self.SetD(self.__Assembly.GetVector()) #not modified in principle
                              
        def NewtonRaphsonIncrement(self):                                      
            #update total displacement
            self.Solve()
            self.__TotalDisplacementOld = self.__TotalDisplacement
            self.__TotalDisplacement += self.GetX()           
        
        def Update(self, dtime=None, compute = 'all', updateWeakForm = True):   
            """
            Assemble the matrix including the following modification:
                - New initial Stress
                - New initial Displacement
                - Modification of the mesh
                - Change in constitutive law (internal variable)
            Update the problem with the new assembled global matrix and global vector
            """
            if updateWeakForm == True:
                self.__Assembly.Update(self, dtime, compute)
            else: 
                self.__Assembly.ComputeGlobalMatrix(compute)
            self.SetA(self.__Assembly.GetMatrix())
            self.SetD(self.__Assembly.GetVector())

        def Reset(self):
            self.__Assembly.Reset()
            
            self.SetA(0) #tangent stiffness 
            self.SetD(0)                 
            # self.SetA(self.__Assembly.GetMatrix()) #tangent stiffness 
            # self.SetD(self.__Assembly.GetVector())            

            B = 0
            self.__TotalDisplacement = 0
            self.__TotalDisplacementOld = 0
                        
            self.__Err0 = None #initial error for NR error estimation   
            self.t0 = 0 ; self.tmax = 1
            self.__iter = 0  
            self.ApplyBoundaryCondition() #perhaps not usefull here as the BC will be applied in the NewTimeIncrement method ?
        
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
                    print(self.__Err0)
                return 1                
            else: 
                if self.__ErrCriterion == 'Displacement': 
                    return np.max(np.abs(self.GetX()))/self.__Err0  #Displacement criterion
                elif self.__ErrCriterion == 'Force': #Force criterion              
                    if self.GetD() is 0: return np.max(np.abs(self.GetB()[DofFree]))/self.__Err0 
                    else: return np.max(np.abs(self.GetB()[DofFree]+self.GetD()[DofFree]))/self.__Err0                     
                else: #self.__ErrCriterion == 'Work': #work criterion
                    if self.GetD() is 0: return np.max(np.abs(self.GetX()[DofFree]) * np.abs(self.GetB()[DofFree]))/self.__Err0 
                    else: return np.max(np.abs(self.GetX()[DofFree]) * np.abs(self.GetB()[DofFree]+self.GetD()[DofFree]))/self.__Err0 
       
        def SetNewtonRaphsonErrorCriterion(self, ErrorCriterion):
            if ErrorCriterion in ['Displacement', 'Force','Work']:
                self.__ErrCriterion = ErrorCriterion            
            else: 
                raise NameError('ErrCriterion must be set to "Displacement", "Force" or "Work"')
        
        # def GetElasticEnergy(self): #only work for classical FEM
        #     """
        #     returns : sum (0.5 * U.transposed * K * U)
        #     """
    
        #     return sum( 0.5*self.GetX().transpose() * self.GetA() * self.GetX() )

        # def GetNodalElasticEnergy(self):
        #     """
        #     returns : 0.5 * K * U . U
        #     """
    
        #     E = 0.5*self.GetX().transpose() * self.GetA() * self.GetX()

        #     E = np.reshape(E,(3,-1)).T

        #     return E                   
        
        def SolveTimeIncrement(self, timeOld, dt, max_subiter = 5, ToleranceNR = 5e-3):            
            
            # self.NewTimeIncrement(timeOld, dt)
            # self.__Err0 = 1
            
            self.ElasticPrediction(timeOld, dt)
            
            for subiter in range(max_subiter): #newton-raphson iterations
                #update Stress and initial displacement and Update stiffness matrix
                self.Update(dt, compute = 'vector') #update the out of balance force vector

                #Check convergence     
                normRes = self.NewtonRaphsonError()    

                # print('     Subiter {} - Time: {:.5f} - Err: {:.5f}'.format(subiter, timeOld+dt, normRes))

                if normRes < ToleranceNR:                                                  
                    return 1, subiter, normRes
                
                #--------------- Solve --------------------------------------------------------        
                # self.__Assembly.ComputeGlobalMatrix(compute = 'matrix')
                # self.SetA(self.__Assembly.GetMatrix())
                self.Update(compute = 'matrix', updateWeakForm = False) #assemble the tangeant matrix
                self.NewtonRaphsonIncrement()
                
            return 0, subiter, normRes


        def NLSolve(self, **kargs):              
            #parameters
            max_subiter = kargs.get('max_subiter',6)
            ToleranceNR = kargs.get('ToleranceNR',5e-3)
            self.t0 = kargs.get('t0',self.t0)
            self.tmax = kargs.get('tmax',self.tmax)
            dt = kargs.get('dt',0.1)
            update_dt = kargs.get('update_dt',True)
            outputFile = kargs.get('outputFile', None)
            fraction_dt = 1 #used only when the time step is updated
            current_dt = fraction_dt * dt            
            
            self.SetInitialBCToCurrent()            
                        
            time = self.t0                            
            
            if self.__TotalDisplacement is 0:#Initialize only if 1st step
                self.Initialize(self.t0)
                            
            while time < self.tmax - self.err_num:
                timeOld = time
                time = time+current_dt
                if time > self.tmax - self.err_num: 
                    time = self.tmax
                    current_dt = time-timeOld
                
                # print(self.__Assembly.GetWeakForm().GetConstitutiveLaw().GetPKII()[0][0]) #for debug purpose
                
                #self.SolveTimeIncrement = Newton Raphson loop
                convergence, nbNRiter, normRes = self.SolveTimeIncrement(timeOld, current_dt, max_subiter, ToleranceNR)

                if (convergence) :
                    self.NewTimeIncrement()
                else:
                    if update_dt:
                        time = timeOld
                        fraction_dt *= 0.25 ; current_dt = fraction_dt * dt                                     
                        print('NR failed to converge (err: {:.5f}) - reduce the time increment to {:.5f}'.format(normRes, current_dt ))
                        #reset internal variables, update Stress, initial displacement and assemble global matrix at previous time              
                        
                        self.ResetTimeIncrement()  
                        continue                    
                    else: 
                        raise NameError('Newton Raphson iteration has not converged (err: {:.5f})- Reduce the time step or use update_dt = True'.format(normRes))   
                    
                print('Iter {} - Time: {:.5f} - NR iter: {} - Err: {:.5f}'.format(self.__iter, time, nbNRiter, normRes))
                if outputFile is not None: outputFile(self, self.__iter, time, nbNRiter, normRes)                            
                self.__postTreatment(self.__iter, time, dt)

                self.__iter += 1

                if update_dt and nbNRiter < 2: 
                    if self.__saveOutputAtExactTime == False:
                        fraction_dt *= 1.25 
                        print('Increase the time increment to {:.5f}'.format(current_dt))               
                    else:
                        if abs((time-self.t0)/dt - round((time-self.t0)/dt)) < self.err_num:
                            fraction_dt = round(1/(fraction_dt*1.25))
                            print('Increase the time increment to {:.5f}'.format(current_dt))               
                            
                    current_dt = fraction_dt * dt                        
                                                                     
        def __postTreatment(self, it, time, dt): 
            if self.__saveOutputAtExactTime == True:
                if abs((time-self.t0)/(self.__outputIter*dt) - round((time-self.t0)/(self.__outputIter*dt))) < self.err_num:
                    self.__ProblemOutput.SaveResults(self, self.__compteurOutput)                                
                    self.__compteurOutput += 1 
            else:        
                if it%self.__outputIter == 0: #save output ?  
                    self.__ProblemOutput.SaveResults(self, self.__compteurOutput)                                
                    self.__compteurOutput += 1

        def AddOutput(self, filename, assemblyID, output_list, output_type='Node', file_format ='vtk'):
            self.__ProblemOutput.AddOutput(filename, assemblyID, output_list, output_type, file_format)            
             
        #     #### SAVE results#####
        #     # TotalPKStress = Material.GetCurrentStress()
        #     # VM = Util.listStressTensor(TotalPKStress).vonMises()
        #     # VMEl = Assembly.GetAll()['Assembling'].ConvertData(VM, "GaussPoint", "Element")
        #     # VM1 = Assembly.GetAll()['Assembling'].ConvertData(VM, "GaussPoint", "Node")
        #     # VM2 = Assembly.GetAll()['Assembling'].ConvertData(VMEl, "Element", "Node")
            
        #     # TotalPKStressEl = Util.listStressTensor([Assembly.GetAll()['Assembling'].ConvertData(S, "GaussPoint", "Element") for S in TotalPKStress])
        #     # TotalPKStress = Util.listStressTensor([Assembly.GetAll()['Assembling'].ConvertData(S, "GaussPoint", "Node") for S in TotalPKStress])
        #     StressPG = Material.GetCurrentStress()
        #     Stress = Util.listStressTensor([Assembly.GetAll()['Assembling'].ConvertData(S, "GaussPoint", "Node") for S in StressPG])
        #     StrainPG = Material.GetCurrentStrain()
        #     Strain = Util.listStrainTensor([Assembly.GetAll()['Assembling'].ConvertData(S, "GaussPoint", "Node") for S in StrainPG])
            
        #     #Write the vtk file                            
        #     OUT = Util.ExportData(meshID)      
        #     OUT.addNodeData(Problem.GetDisp().reshape(3,-1).T,'Displacement')
        # #            OUT.addNodeData(Strain.vtkFormat(),'Strain')
             
        # #     outputPlasticity = 0
        # #     if outputPlasticity:
        # #         P = Assembly.GetAll()['Assembling'].ConvertData(ConstitutiveLaw.GetAll()['ElasticLaw'].GetPlasticity(),"GaussPoint", "Node")
        # #         P_El = Assembly.GetAll()['Assembling'].ConvertData(ConstitutiveLaw.GetAll()['ElasticLaw'].GetPlasticity(),"GaussPoint", "Element")
        # #         P2 = Assembly.GetAll()['Assembling'].ConvertData(P_El,"Element", "Node")
            
        # #         PlasticStrainTensor = ConstitutiveLaw.GetAll()['ElasticLaw'].GetPlasticStrainTensor()
        # #         PlasticStrainTensor = Util.listStrainTensor([Assembly.GetAll()['Assembling'].ConvertData(E, "GaussPoint", "Element") for E in PlasticStrainTensor])
        # # #            PlasticStrainTensor = Util.listStrainTensor([Assembly.GetAll()['Assembling'].ConvertData(E, "Element", "Node") for E in PlasticStrainTensor])
        
        # #         OUT.addNodeData(P,'Plasticity')
        # #         OUT.addNodeData(P2,'Plasticity2')
        # #         OUT.addElmData(P_El,'Plasticity')
        # #         OUT.addElmData(PlasticStrainTensor.vtkFormat(),'PlasticStrainTensor')
            
        #     OUT.addNodeData(Stress.vtkFormat(),'Stress')
        #     OUT.addNodeData(Strain.vtkFormat(),'Strain')
        #     OUT.addNodeData(Stress.vonMises(),'VonMises')
            
        #     freq = 1 #save results every freq
        #     if iter%freq == 0:
        #         if mat == 3: 
        #             OUT.toVTK("results/pliage2D/shear_simple_"+str(iter//freq)+".vtk")
        #         else:
        #             OUT.toVTK("results/pliage2D/shear_"+str(Material.corate)+"_"+str(iter//freq)+".vtk")
        #     # print(Material.GetStrain()[0])
        
                
        #     # print('[S: ', Material.GetCurrentStress()[0,0], ', Ep: ', Material.GetStatev()[0,1], ']')
        #     # print(Material.GetCurrentStress[0,0], np.array(ConstitutiveLaw.GetAll()['ElasticLaw'].GetPlasticStrainTensor())[0,0])
        #     # print('yo: ',Material.GetStatev()[0])
        #     # print('yo')
        #     # print(ConstitutiveLaw.GetAll()['ElasticLaw'].GetPlasticity()[0])
        #     # print(np.array(ConstitutiveLaw.GetAll()['ElasticLaw'].GetPlasticStrainTensor())[:,0])

    return __NonLinearStatic(Assembling, ID)
