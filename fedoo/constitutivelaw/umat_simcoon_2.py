# #derive de ConstitutiveLaw
# #This law should be used with an InternalForce WeakForm

# USE_SIMCOON = True

# if USE_SIMCOON: 
#     try:
#         from simcoon import simmit as sim
#         USE_SIMCOON = True
#     except:
#         USE_SIMCOON = False
#         print('WARNING: Simcoon library not found. The simcoon constitutive law is disabled.')       

# if USE_SIMCOON:    
#     from fedoo.core.mechanical3d import Mechanical3D
#     from fedoo.util.voigt_tensors import StressTensorList, StrainTensorList
#     import numpy as np
    
#     class Simcoon_new(Mechanical3D):
#         def __init__(self,umat_name, props, statev, corate=0, name =""):
#             #props is a nparray containing all the material variables
#             #nstatev is a nparray containing all the material variables
#             Mechanical3D.__init__(self, name) # heritage
        
            
#             self.__statev_initial = statev #statev may be an int or an array        
#             # self.__useElasticModulus = True ??
            
#             self.__currentGradDisp = self.__initialGradDisp = 0        
                
#             ndi = nshr = 3 #compute the 3D constitutive law even for 2D law 
#             self.umat_name = umat_name
#             self.props = np.atleast_2d(props)
#             self.corate = corate
#             self.statev = None
#             #self.umat = ... #need to define simcoon umat function from umat_name
            
#             #self.__mask -> contains list of tangent matrix terms that are 0 (before potential change of Basis)
#             #self.__mask[i] contains the column indice for the line i            
#             self.__mask = None  #No mask defined            
#             self.use_elastic_lt = False #mainly for debug purpose            
                
#         # def get_pk2(self):
#         #     return StressTensorList(self.PKII.T)
        
#         # def get_kirchhoff(self):
#         #     return StressTensorList(self.Kirchhoff.T)        
        
#         # def get_cauchy(self):
#         #     return StressTensorList(self.Cauchy.T)        
        
#         # def get_strain(self, **kargs):
#         #     return StrainTensorList(self.etot.T)
               
#         # def get_statev(self):
#         #     return self.statev.T

#         # def get_wm(self):
#         #     return self.Wm.T

#         # def get_stress(self, **kargs): #same as GetPKII (used for small def)
#         #     return StressTensorList(self.PKII.T)
        
#         # def GetHelas (self):
#         #     # if self.__L is None:                
#         #     #     self.RunUmat(np.eye(3).T.reshape(1,3,3), np.eye(3).T.reshape(1,3,3), time=0., dtime=1.)
    
#         #     return np.squeeze(self.L.transpose(1,2,0)) 
        
#         # def get_disp_grad(self):
#         #     if self.__currentGradDisp is 0: return 0
#         #     else: return self.__currentGradDisp
            
#         def get_tangent_matrix(self,assembly,pb):
           
#             if len(self.Lt) == 0:
#                 #same as Initialize with only one points
#                 #if the number of material points is not defined (=0) we need to initialize statev
                
#                 if np.isscalar(self.__statev_initial): 
#                     nb_points = 1
#                     statev = np.zeros((nb_points, int(self.__statev_initial))).T
#                 else: 
#                     statev = np.atleast_2d(self.__statev_initial).T.astype(float)
#                     if len(statev) == 1: statev = statev.copy().T
#                     else: assert 0, "Initialize simcoon constitutive law first"
#                     # else: #statev = assembly.convert_data(statev).T
                
#                 sim.Umat_fedoo.Initialize(self, 0., statev, False)
                    
#                 self.Run(0.) #Launch the UMAT to compute the elastic matrix 
                                
#             if  self.use_elastic_lt: return np.squeeze(self.elastic_Lt.transpose(1,2,0)) ### debut only ####
        
#             H = np.squeeze(self.Lt.transpose(1,2,0))
#             # H = np.squeeze(self.L.transpose(1,2,0))
#             if self.__mask is None:
#                 return H
#             else:    
#                 return np.array([[0 if j in self.__mask[i] else H[i,j] for j in range(6)] for i in range(6)])
                    
#         # def GetStressOperator(self, **kargs):
#         #     H = self.GetH(**kargs)
                          
#         #     eps, eps_vir = GetStrainOperator(self.__currentGradDisp)
#         #     sigma = [sum([0 if eps[j] is 0 else eps[j]*H[i][j] for j in range(6)]) for i in range(6)]
    
#         #     return sigma # list de 6 objets de type DiffOp
        
#         def set_start(self):
#             #save variables at the begining of the Time increment            
#             if self.__currentGradDisp is not 0:
#                 #not usefull at 1st iteration. Everything should have already been initialized at start value with the initialize func.
#                 sim.Umat_fedoo.set_start(self) #in set_start -> set tangeant matrix to elastic
                
#                 self.statev_start = self.statev;
#                 # m_start = list_Wm ###????
#                 F1 = self.__currentGradDisp
#                 if self.nlgeom == 1:
#                     #to use with total lagrangian method
#                     self.pkII_start = self.pkII
#                     if self.corate == 2: 
#                         self.etot = sim.t2v_strain(sim.Log_strain(F1))
#                     else: 
#                         self.etot = sim.rotate_strain(self.etot, self.DR) + self.Detot
#                     self.cauchy_start = sim.rotate_stress(self.cauchy, self.DR)
#                     self.Lt = self.L
            
#                 elif self.nlgeom == 2: 
#                     #to use with update lagrangian mathod. tangent matrix is expressed on the current configuration 
#                     if self.corate == 2: 
#                         self.etot = sim.t2v_strain(sim.Log_strain(F1))
#                     else: 
#                         self.etot = sim.rotate_strain(self.etot, self.DR) + self.Detot
#                     self.cauchy_start = sim.rotate_stress(self.cauchy, self.DR)
#                 else: 
#                     #nlgeom = false
#                     self.cauchy_start = self.cauchy
#                     self.etot = self.etot + self.Detot
#                     self.Lt = self.Lt
                
#                 self.Lt_start = self.Lt
                
#                 # Time = Time + DTime
#                 # listF0 = listF1 # F0 = initialGradDisp ???
#                 #T += DT; to add if required				            

#             self.__initialGradDisp = self.__currentGradDisp
    
         
#         def get_mask_H(self):
#             """
#             Return the actual mask applied to the tangent matrix
#             mask -> contains list of tangent matrix terms that are 0 (before potential change of Basis)
#             mask[i] contains the column indice for the line i            

#             """
#             return self.__mask
        
#         def set_mask_H(self, mask):
#             """
#             Set the mask applied to the tangent matrix
#             mask -> contains list of tangent matrix terms that are 0 (before potential change of Basis)
#             mask[i] contains the column indice for the line i            

#             """
#             self.__mask = mask

        
#         def to_start(self):
#         	# Replace the tangent matrix with the elastic matrix for a prediction
#             self.list_Lt = self.Lt_start
#             if self.nlgeom == 1:
#                 self.pkII = self.pkII_start
#             else:
#                 self.cauchy = self.cauchy_start
                
#             self.statev = self.statev_start
#             self.Wm = self.Wm_start
        	
#             self.__currentGradDisp = self.__initialGradDisp    
        
#         def reset(self):
#             """
#             reset the constitutive law (time history)
#             """
#             #a modifier (créer une fonction reset dans l'umat simcoon)
#             self.__currentGradDisp = self.__initialGradDisp = 0
#             # self.__Statev = None
#             # self.__currentStress = None #lissStressTensor object describing the last computed stress (GetStress method)
#             # self.__F0 = None
    
        
#         def initialize(self, assembly, pb, t0 = 0., nlgeom=False):      
            
#             if  self._dimension is None:
#                 self._dimension = assembly.space.get_dimension()
                
#             #if the number of material points is not defined (=0) we need to initialize statev            
#             if np.isscalar(self.__statev_initial): 
#                 self.statev = np.zeros((assembly.n_gauss_points, int(self.__statev_initial))).T
#             else: 
#                 self.statev = np.atleast_2d(self.__statev_initial).T.astype(float)
#                 if len(self.statev) == 1: self.statev = np.tile(self.statev.copy(),[assembly.n_gauss_points,1]).T
#                 else: self.statev = assembly.convert_data(self.statev).T
            
#             # sim.Umat_fedoo.Initialize(self, t0, self.statev, nlgeom)
            
#             self.Time = t0
#             self.nlgeom = nlgeom
#             nb_points = len(self.statev)
#             ncomp = 6 #6 strain components for the 3D general case
#             # initialization of Wm and etot to 0
#             self.Wm = np.zeros(4, nb_points);
#             self.etot = np.zeros(ncomp, nb_points)
#             self.Detot = np.zeros(ncomp, nb_points)

#             #initialization of F0 to identity matrix
#             #F0 = initialGradDisp ??
#             self.F0 = np.eye(3,3).reshape(1,3,3)
#             self.F1 = np.eye(3,3).reshape(1,3,3)*np.ones((nb_points, 1,1))

#             self.cauchy = np.zeros(ncomp, nb_points)
#             self.cauchy_start = np.zeros(ncomp, nb_points)
#             if nlgeom == 1: #only for total lagrangian method
#                 self.pkII = np.zeros(ncomp, nb_points)
#                 self.pkII_start = np.zeros(ncomp, nb_points)
            
#             #don't know if all these variables are required...
#             self.R = np.eye(3,3).reshape(1,3,3)*np.ones((nb_points, 1,1))
#             self.DR = np.eye(3,3).reshape(1,3,3)*np.ones((nb_points, 1,1))
#             self.Lt = np.empty((nb_points,6,6))
#             self.L = np.empty((nb_points,6,6))
            
#             self.statev_start = self.statev
#             self.Wm_start = self.Wm
     
#             self.Run(0.) #Launch the UMAT to compute the elastic matrix                 
#             if self.use_elastic_lt: self.elastic_Lt = self.Lt.copy() ### debut only ####
    
#         def update(self,assembly, pb, dtime):   
#             displacement = pb.get_dof_solution()

#             #tranpose for compatibility with simcoon
#             if displacement is 0: 
#                 self.__currentGradDisp = 0
#                 F1 = np.multiply(np.eye(3).reshape(3,3,1) , np.ones((1,1,assembly.n_gauss_points)), order='F').transpose(2,0,1)
#                 # self.__F1 = np.eye(3).T.reshape(1,3,3)
#             else:   
#                 self.__currentGradDisp = np.array(assembly.get_grad_disp(displacement, "GaussPoint"))            

#                 #F0.strides and F1.strides should be [n_cols*n_rows*8, 8, n_rows*8] for compatibiliy with the sim.RunUmat_fedoo function
#                 F1 = np.add(np.eye(3).reshape(3,3,1), self.__currentGradDisp, order='F').transpose(2,0,1)                        
                
#             self.compute_Detot(dtime, F1)  
            
            
#             # test = np.array(assembly.get_strain(pb.get_dof_solution(), "GaussPoint", False)).T #linearized strain tensor
#             # print( (self.etot+self.Detot - test).max() )
            

#             # self.Run(dtime)
#             pb.sv['stress'],pb.sv['tangent_matrix'] = sim.umat(stress, etot, Detot)
        
#         def copy(self, new_id=""):
#             """
#             Return a raw copy of the constitutive law without keeping current internal variables.

#             Parameters
#             ----------
#             new_id : TYPE, optional
#                 The name of the created constitutive law. The default is "".

#             Returns
#             -------
#             The copy of the constitutive law
#             """
#             return Simcoon(self.umat_name, self.props, self.__statev_initial, self.corate, new_id)
            
