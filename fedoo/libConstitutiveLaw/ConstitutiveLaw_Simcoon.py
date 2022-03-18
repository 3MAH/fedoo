#derive de ConstitutiveLaw
#This law should be used with an InternalForce WeakForm

USE_SIMCOON = True

if USE_SIMCOON: 
    try:
        from simcoon import simmit as sim
        USE_SIMCOON = True
    except:
        USE_SIMCOON = False
        print('WARNING: Simcoon library not found. The simcoon constitutive law is disabled.')       

if USE_SIMCOON:    
    from fedoo.libConstitutiveLaw.ConstitutiveLaw import Mechanical3D
    from fedoo.libUtil.StrainOperator import *
    from fedoo.libUtil.ModelingSpace  import Variable, GetDimension
    from fedoo.libUtil.PostTreatement import listStressTensor, listStrainTensor
    import numpy as np
    
    class Simcoon(Mechanical3D, sim.Umat_fedoo):
        def __init__(self,umat_name, props, statev, corate=0, ID=""):
            #props is a nparray containing all the material variables
            #nstatev is a nparray containing all the material variables
            Mechanical3D.__init__(self, ID) # heritage
        
            
            self.__InitialStatev = statev #statev may be an int or an array        
            # self.__useElasticModulus = True ??
            
            self.__currentGradDisp = self.__initialGradDisp = 0        
    
            # Variable("DispX")
            # Variable("DispY")
        
            if GetDimension() == "3D":
                # Variable("DispZ")
                ndi = 3 ; nshr = 3
            elif GetDimension() in ['2Dstress']:
                # ndi = 2 ; nshr = 1
                ndi = 3 ; nshr = 3###shoud be modified?
            elif GetDimension() in ['2Dplane']:
                 ndi = 3 ; nshr = 3 # the constitutive law is treated in a classical way
            
            self.umat_name = umat_name
            
            #self.__mask -> contains list of tangent matrix terms that are 0 (before potential change of Basis)
            #self.__mask[i] contains the column indice for the line i            
            self.__mask = None  #No mask defined
                        
            ### initialization of the simcoon UMAT
            sim.Umat_fedoo.__init__(self, umat_name, np.atleast_2d(props), corate, ndi, nshr)
            # sim.Umat_fedoo.__init__(self, umat_name, np.atleast_2d(props), statev, corate, ndi, nshr, 0.)
                
        def GetPKII(self):
            return listStressTensor(self.PKII.T)
        
        def GetKirchhoff(self):
            return listStressTensor(self.Kirchhoff.T)        
        
        def GetCauchy(self):
            return listStressTensor(self.Cauchy.T)        
        
        def GetStrain(self, **kargs):
            return listStrainTensor(self.etot.T)
               
        def GetStatev(self):
            return self.statev.T
        
        def GetCurrentStress(self): #same as GetPKII (used for small def)
            print('Warning : GetCurrentStress will be removed in future versions. Use GetStress instead')
            return listStressTensor(self.PKII.T)

        def GetStress(self, **kargs): #same as GetPKII (used for small def)
            return listStressTensor(self.PKII.T)
        
        # def GetHelas (self):
        #     # if self.__L is None:                
        #     #     self.RunUmat(np.eye(3).T.reshape(1,3,3), np.eye(3).T.reshape(1,3,3), time=0., dtime=1.)
    
        #     return np.squeeze(self.L.transpose(1,2,0)) 
        
        def GetCurrentGradDisp(self):
            if self.__currentGradDisp is 0: return 0
            else: return self.__currentGradDisp
            
        def GetTangentMatrix(self):
            #### TODO: try to implement the case pdim=="2Dstress"
            # pbdim = kargs.get(pbdim, GetDimension())
            

            # if self.__Lt is None:
            #     if self.__L is None:                
            #         self.RunUmat(np.eye(3).T.reshape(1,3,3), np.eye(3).T.reshape(1,3,3), time=0., dtime=1.)
            #         self.__Statev = None
            #         self.__currentStress = None
    
            #     return np.squeeze(self.__L.transpose(1,2,0))            
            # else: 
            #     return np.squeeze(self.__Lt.transpose(1,2,0)) 
            
            H = np.squeeze(self.Lt.transpose(1,2,0))
            # H = np.squeeze(self.L.transpose(1,2,0))
            if self.__mask is None:
                return H
            else:    
                return np.array([[0 if j in self.__mask[i] else H[i,j] for j in range(6)] for i in range(6)])
                    
        def GetStressOperator(self, **kargs):
            H = self.GetH(**kargs)
                          
            eps, eps_vir = GetStrainOperator(self.__currentGradDisp)
            sigma = [sum([0 if eps[j] is 0 else eps[j]*H[i][j] for j in range(6)]) for i in range(6)]
    
            return sigma # list de 6 objets de type OpDiff
        
        def NewTimeIncrement(self):
            self.set_start() #in set_start -> set tangeant matrix to elastic
            
            #save variable at the begining of the Time increment
            self.__initialGradDisp = self.__currentGradDisp
    
         
        def GetMaskH(self):
            """
            Return the actual mask applied to the tangent matrix
            mask -> contains list of tangent matrix terms that are 0 (before potential change of Basis)
            mask[i] contains the column indice for the line i            

            """
            return self.__mask
        
        def SetMaskH(self, mask):
            """
            Set the mask applied to the tangent matrix
            mask -> contains list of tangent matrix terms that are 0 (before potential change of Basis)
            mask[i] contains the column indice for the line i            

            """
            self.__mask = mask

        
        def ResetTimeIncrement(self):
            self.to_start()         
            self.__currentGradDisp = self.__initialGradDisp    
        
        def Reset(self):
            """
            Reset the constitutive law (time history)
            """
            #a modifier
            self.__currentGradDisp = self.__initialGradDisp = 0
            # self.__Statev = None
            self.__currentStress = None #lissStressTensor object describing the last computed stress (GetStress method)
            # self.__currentGradDisp = 0
            # self.__F0 = None
    
        
        def Initialize(self, assembly, pb, initialTime = 0., nlgeom=True):            
            #if the number of material points is not defined (=0) we need to initialize statev
            nb_points = assembly.GetNumberOfGaussPoints() * assembly.GetMesh().GetNumberOfElements()
            if np.isscalar(self.__InitialStatev): 
                statev = np.zeros((nb_points, int(self.__InitialStatev))).T
            else: 
                statev = np.atleast_2d(self.__InitialStatev).T.astype(float)
                if len(statev) == 1: statev = np.tile(statev.copy(),[nb_points,1]).T
                else: statev = assembly.ConvertData(statev).T
            
            sim.Umat_fedoo.Initialize(self, initialTime, statev, nlgeom)
            
            if not(nlgeom):
                if self.umat_name in ['ELISO'] and self.__mask is None:        
                    self.__mask = [[3,4,5] for i in range(3)]
                    self.__mask+= [[0,1,2,4,5], [0,1,2,3,5], [0,1,2,3,4]]
                
            self.Run(0.) #Launch the UMAT to compute the elastic matrix               
    
        def Update(self,assembly, pb, dtime, nlgeom=True):   
            displacement = pb.GetDoFSolution()

            #tranpose for compatibility with simcoon
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
            
            
            # test = np.array(assembly.GetStrainTensor(pb.GetDoFSolution(), "GaussPoint", False)).T #linearized strain tensor
            # print( (self.etot+self.Detot - test).max() )
            

            self.Run(dtime)

            # (DRloc , listDR, Detot, statev) = self.Run(dtime)
