#derive de ConstitutiveLaw
#This law should be used with an InternalForce WeakForm

try:
    from fedoo.libConstitutiveLaw.ConstitutiveLaw import ConstitutiveLaw
    from fedoo.libUtil.StrainOperator import *
    from fedoo.libUtil.Variable       import *
    from fedoo.libUtil.Dimension      import *
    from fedoo.libUtil.PostTreatement import listStressTensor, listStrainTensor
    
    import numpy as np
    from simcoon import simmit as sim
    
    class Simcoon(ConstitutiveLaw, sim.Umat_fedoo):
        def __init__(self,umat_name, props, statev, corate=0, ID=""):
            #props is a nparray containing all the material variables
            #nstatev is a nparray containing all the material variables
            ConstitutiveLaw.__init__(self, ID) # heritage
        
            
            self.__InitialStatev = statev #statev may be an int or an array        
            # self.__useElasticModulus = True ??
            
            self.__currentGradDisp = self.__initialGradDisp = 0        
    
            # Variable("DispX")
            # Variable("DispY")
        
            if ProblemDimension.Get() == "3D":
                # Variable("DispZ")
                ndi = 3 ; nshr = 3
            elif ProblemDimension.Get() in ['2Dstress']:
                # ndi = 2 ; nshr = 1
                ndi = 3 ; nshr = 3
            elif ProblemDimension.Get() in ['2Dplane']:
                 ndi = 3 ; nshr = 3 # the constitutive law is treated in a classical way
    
            #statev = ??? require to get te number of gauss point
            
            ### initialization of the simcoon UMAT
            sim.Umat_fedoo.__init__(self, umat_name, np.atleast_2d(props), corate, ndi, nshr)
            # sim.Umat_fedoo.__init__(self, umat_name, np.atleast_2d(props), statev, corate, ndi, nshr, 0.)
                
        def GetPKII(self):
            return listStressTensor(self.PKII.T)
        
        def GetKirchhoff(self):
            return listStressTensor(self.Kirchhoff.T)        
        
        def GetCauchy(self):
            return listStressTensor(self.Cauchy.T)        
        
        def GetStrain(self):
            return listStrainTensor(self.etot.T)
               
        def GetStatev(self):
            return self.statev.T

        # def GetHelas (self):
        #     # if self.__L is None:                
        #     #     self.RunUmat(np.eye(3).T.reshape(1,3,3), np.eye(3).T.reshape(1,3,3), time=0., dtime=1.)
    
        #     return np.squeeze(self.L.transpose(1,2,0)) 
        
        def GetCurrentGradDisp(self):
            if self.__currentGradDisp is 0: return 0
            else: return self.__currentGradDisp
            
        def GetH(self):
            # if self.__Lt is None:
            #     if self.__L is None:                
            #         self.RunUmat(np.eye(3).T.reshape(1,3,3), np.eye(3).T.reshape(1,3,3), time=0., dtime=1.)
            #         self.__Statev = None
            #         self.__currentStress = None
    
            #     return np.squeeze(self.__L.transpose(1,2,0))            
            # else: 
            #     return np.squeeze(self.__Lt.transpose(1,2,0))          
            return np.squeeze(self.Lt.transpose(1,2,0))
                        
        def __ChangeBasisH(self, H):       
            #Change of basis capability for laws on the form : StressTensor = H * StrainTensor
            #StressTensor and StrainTensor are column vectors based on the voigt notation 
            if self._ConstitutiveLaw__localFrame is not None:
                localFrame = self._ConstitutiveLaw__localFrame
                #building the matrix to change the basis of the stress and the strain
    #            theta = np.pi/8
    #            np.array([[np.cos(theta),np.sin(theta),0], [-np.sin(theta),np.cos(theta),0], [0,0,1]]) 
                R_epsilon = np.empty((len(localFrame), 6,6))
                R_epsilon[:,  :3,  :3] = localFrame**2 
                R_epsilon[:,  :3, 3:6] = localFrame[:,:,[0,2,1]]*localFrame[:,:,[1,0,2]]
                R_epsilon[:, 3:6,  :3] = 2*localFrame[:,[0,2,1]]*localFrame[:,[1,0,2]] 
                R_epsilon[:, 3:6, 3:6] = localFrame[:,[[0],[2],[1]], [0,2,1]]*localFrame[:,[[1],[0],[2]],[1,0,2]] + localFrame[:,[[1],[0],[2]],[0,2,1]]*localFrame[:,[[0],[2],[1]],[1,0,2]] 
                R_sigma_inv = R_epsilon.transpose(0,2,1)    # np.transpose(R_epsilon,[0,2,1])        
                
                if len(H.shape) == 3: H = np.rollaxis(H,2,0)
                H = np.matmul(R_sigma_inv, np.matmul(H,R_epsilon))
                if len(H.shape) == 3: H = np.rollaxis(H,0,3)  
                
            return H
        
        def GetStressOperator(self, localFrame=None):
            H = self.__ChangeBasisH(self.GetH())
                          
            eps, eps_vir = GetStrainOperator()
            sigma = [sum([0 if eps[j] is 0 else eps[j]*H[i][j] for j in range(6)]) for i in range(6)]
    
            return sigma # list de 6 objets de type OpDiff
        
        def NewTimeIncrement(self):
            self.set_start() #in set_start -> set tangeant matrix to elastic
            
            #save variable at the begining of the Time increment
            self.__initialGradDisp = self.__currentGradDisp
    
            
        def ResetTimeIncrement(self):
            self.to_start()         
            self.__currentGradDisp = self.__initialGradDisp    
        
        def Reset(self):
            """
            Reset the constitutive law (time history)
            """
            #a modifier
            # self.__Statev = None
            self.__currentStress = None #lissStressTensor object describing the last computed stress (GetStress method)
            # self.__currentGradDisp = 0
            # self.__F0 = None
    
        
        def Initialize(self, assembly, pb, initialTime = 0., nlgeom=True):
    
            #if the number of material points is not defined (=0) we need to initialize statev
            nb_points = assembly.GetNumberOfGaussPoints() * assembly.GetMesh().GetNumberOfElements()
            if np.isscalar(self.__InitialStatev): 
                statev = np.zeros((nb_points, int(self.__InitialStatev)))
            else: 
                statev = np.atleast_2d(self.__InitialStatev).astype(float)
                if len(statev) == 1: statev = np.tile(statev.copy(),[nb_points,1])
                else: statev = assembly.ConvertData(statev)    
    
            sim.Umat_fedoo.Initialize(self, initialTime, statev, nlgeom)
            self.Run(0.) #Launch the UMAT to compute the elastic matrix    
    
        def Update(self,assembly, pb, dtime, nlgeom=True):            
            displacement = pb.GetDoFSolution()
                
            #tranpose for comatibility with simcoon
            if displacement is 0: 
                self.__currentGradDisp = 0
                self.__currentStress = None
                F1 = np.multiply(np.eye(3).reshape(3,3,1) , np.ones((1,1,self.nb_points)), order='F').transpose(2,0,1)
                # self.__F1 = np.eye(3).T.reshape(1,3,3)
            else:                    
                self.__currentGradDisp = np.array(assembly.GetGradTensor(displacement, "GaussPoint"))            
                #F0.strides and F1.strides should be [n_cols*n_rows*8, 8, n_rows*8] for compatibiliy with the sim.RunUmat_fedoo function
                F1 = np.add( np.eye(3).reshape(3,3,1), self.__currentGradDisp, order='F').transpose(2,0,1)                        
    
            self.compute_Detot(dtime, F1)   
            self.Run(dtime)
            # (DRloc , listDR, Detot, statev) = self.Run(dtime)
            
            
        # def GetStress(self, GradDispTensor, time = None):
        #     self.__GradDispTenorOld = self.__GradDispTensor
        #     self.__GradDispTensor = GradDispTensor
            
            
        #     # time not used here because this law require no time effect
        #     # initilialize values plasticity variables if required
    
        #     self.__currentStress = listStressTensor(sigmaFull.T) # list of 6 objets
    
        
except:
    print('WARNING: Simcoon library not found. The simcoon constitutive law is disabled.')   

