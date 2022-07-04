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
    from fedoo.libUtil.PostTreatement import listStressTensor, listStrainTensor
    import numpy as np
    
    class Simcoon(Mechanical3D, sim.Umat_fedoo):
        def __init__(self,umat_name, props, statev, corate=0, name =""):
            #props is a nparray containing all the material variables
            #nstatev is a nparray containing all the material variables
            Mechanical3D.__init__(self, name) # heritage
        
            
            self.__statev_initial = statev #statev may be an int or an array        
            # self.__useElasticModulus = True ??
            
            self.__currentGradDisp = self.__initialGradDisp = 0        
                
            ndi = nshr = 3 #compute the 3D constitutive law even for 2D law 
            self.umat_name = umat_name
            
            #self.__mask -> contains list of tangent matrix terms that are 0 (before potential change of Basis)
            #self.__mask[i] contains the column indice for the line i            
            self.__mask = None  #No mask defined
            self.__props = props # keep this line until future simcoon release (where copy will be avoided) 
            
            ### initialization of the simcoon UMAT
            sim.Umat_fedoo.__init__(self, umat_name, np.atleast_2d(props), corate, ndi, nshr)
            # sim.Umat_fedoo.__init__(self, umat_name, np.atleast_2d(props), statev, corate, ndi, nshr, 0.)
            
            self.use_elastic_lt = False #mainly for debug purpose
                
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

        def GetWm(self):
            return self.Wm.T
        
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
           
            if len(self.Lt) == 0:
                #same as Initialize with only one points
                #if the number of material points is not defined (=0) we need to initialize statev
                
                if np.isscalar(self.__statev_initial): 
                    nb_points = 1
                    statev = np.zeros((nb_points, int(self.__statev_initial))).T
                else: 
                    statev = np.atleast_2d(self.__statev_initial).T.astype(float)
                    if len(statev) == 1: statev = statev.copy().T
                    else: assert 0, "Initialize simcoon constitutive law first"
                    # else: #statev = assembly.convert_data(statev).T
                
                sim.Umat_fedoo.Initialize(self, 0., statev, False)
                    
                self.Run(0.) #Launch the UMAT to compute the elastic matrix 
                                
            if  self.use_elastic_lt: return np.squeeze(self.elastic_Lt.transpose(1,2,0)) ### debut only ####
        
            H = np.squeeze(self.Lt.transpose(1,2,0))
            # H = np.squeeze(self.L.transpose(1,2,0))
            if self.__mask is None:
                return H
            else:    
                return np.array([[0 if j in self.__mask[i] else H[i,j] for j in range(6)] for i in range(6)])
                    
        # def GetStressOperator(self, **kargs):
        #     H = self.GetH(**kargs)
                          
        #     eps, eps_vir = GetStrainOperator(self.__currentGradDisp)
        #     sigma = [sum([0 if eps[j] is 0 else eps[j]*H[i][j] for j in range(6)]) for i in range(6)]
    
        #     return sigma # list de 6 objets de type OpDiff
        
        def set_start(self):
            
            # if self.__currentGradDisp is not 0:
            sim.Umat_fedoo.set_start(self) #in set_start -> set tangeant matrix to elastic
        
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

        
        def to_start(self):
            sim.Umat_fedoo.to_start(self)         
            self.__currentGradDisp = self.__initialGradDisp    
        
        def reset(self):
            """
            reset the constitutive law (time history)
            """
            #a modifier (créer une fonction reset dans l'umat simcoon)
            self.__currentGradDisp = self.__initialGradDisp = 0
            # self.__Statev = None
            # self.__currentStress = None #lissStressTensor object describing the last computed stress (GetStress method)
            # self.__F0 = None
    
        
        def initialize(self, assembly, pb, initialTime = 0., nlgeom=False):      
            
            if  self._dimension is None:
                self._dimension = assembly.space.GetDimension()
                
            #if the number of material points is not defined (=0) we need to initialize statev
            nb_points = assembly.nb_gp * assembly.mesh.n_elements
            if np.isscalar(self.__statev_initial): 
                statev = np.zeros((nb_points, int(self.__statev_initial))).T
            else: 
                statev = np.atleast_2d(self.__statev_initial).T.astype(float)
                if len(statev) == 1: statev = np.tile(statev.copy(),[nb_points,1]).T
                else: statev = assembly.convert_data(statev).T
            
            sim.Umat_fedoo.Initialize(self, initialTime, statev, nlgeom)
            self.Run(0.) #Launch the UMAT to compute the elastic matrix                 
            if self.use_elastic_lt: self.elastic_Lt = self.Lt.copy() ### debut only ####
    
        def update(self,assembly, pb, dtime):   
            displacement = pb.GetDoFSolution()

            #tranpose for compatibility with simcoon
            if displacement is 0: 
                self.__currentGradDisp = 0
                F1 = np.multiply(np.eye(3).reshape(3,3,1) , np.ones((1,1,self.nb_points)), order='F').transpose(2,0,1)
                # self.__F1 = np.eye(3).T.reshape(1,3,3)
            else:   
                self.__currentGradDisp = np.array(assembly.get_grad_disp(displacement, "GaussPoint"))            

                #F0.strides and F1.strides should be [n_cols*n_rows*8, 8, n_rows*8] for compatibiliy with the sim.RunUmat_fedoo function
                F1 = np.add(np.eye(3).reshape(3,3,1), self.__currentGradDisp, order='F').transpose(2,0,1)                        
                
            self.compute_Detot(dtime, F1)  
            
            
            # test = np.array(assembly.get_strain(pb.GetDoFSolution(), "GaussPoint", False)).T #linearized strain tensor
            # print( (self.etot+self.Detot - test).max() )
            

            self.Run(dtime)
        
        def copy(self, new_id=""):
            """
            Return a raw copy of the constitutive law without keeping current internal variables.

            Parameters
            ----------
            new_id : TYPE, optional
                The name of the created constitutive law. The default is "".

            Returns
            -------
            The copy of the constitutive law
            """
            return Simcoon(self.umat_name, self.props, self.__statev_initial, self.corate, new_id)
            
