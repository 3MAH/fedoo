#derive de ConstitutiveLaw
#This law should be used with an InternalForce WeakForm


from fedoo.core.mechanical3d import Mechanical3D
from fedoo.weakform.stress_equilibrium import StressEquilibrium
from fedoo.core.assembly import Assembly
from fedoo.problem.non_linear import NonLinear
from fedoo.util.voigt_tensors import StressTensorList, StrainTensorList
from fedoo.constraint.periodic_bc import PeriodicBC #, DefinePeriodicBoundaryConditionNonPerioMesh
from fedoo.homogen.tangent_stiffness import get_tangent_stiffness, get_homogenized_stiffness
import numpy as np
import multiprocessing 

class FE2(Mechanical3D):
    """
    ConstitutiveLaw that solve a Finite Element Problem at each point of gauss
    in the contexte of the so called "FE²" method. 
    
    Parameters
    ----------
    assemb: Assembly or Assembly name (str), or list of Assembly (with len(list) = number of integration points).
        Assembly that correspond to the microscopic problem
    name: str, optional
        The name of the constitutive law
    """
    def __init__(self, assemb, name =""):
        #props is a nparray containing all the material variables
        #nstatev is a nparray containing all the material variables
        if isinstance(assemb, str): assemb = Assembly.get_all()[assemb]
        Mechanical3D.__init__(self, name) # heritage      
        
        if isinstance(assemb, list):
            self.__assembly = [Assembly.get_all()[a] if isinstance(a,str) else a for a in assemb]
            self.__mesh = [a.mesh for a in self.__assembly]
        else:
            self.__mesh = assemb.mesh
            self.__assembly = assemb
            
        self.list_problem = None
                
        # self.__currentGradDisp = self.__initialGradDisp = 0        
            
    def get_pk2(self):
        return StressTensorList(self.__stress)
    
    # def get_kirchhoff(self):
    #     return StressTensorList(self.Kirchhoff.T)        
    
    # def get_cauchy(self):
    #     return StressTensorList(self.Cauchy.T)        
    
    def get_strain(self, **kargs):
        return StrainTensorList(self.__strain)
           
    # def get_statev(self):
    #     return self.statev.T

    def get_stress(self, **kargs): #same as GetPKII (used for small def)
        return StressTensorList(self.__stress)
    
    # def GetHelas (self):
    #     # if self.__L is None:                
    #     #     self.RunUmat(np.eye(3).T.reshape(1,3,3), np.eye(3).T.reshape(1,3,3), time=0., dtime=1.)

    #     return np.squeeze(self.L.transpose(1,2,0)) 
    
    def get_wm(self):
        return self.__Wm
    
    def get_disp_grad(self):
        if self.__currentGradDisp is 0: return 0
        else: return self.__currentGradDisp
        
    def get_tangent_matrix(self):
        
        H = np.squeeze(self.Lt.transpose(1,2,0))
        return H
        
    def NewTimeIncrement(self):
        # self.set_start() #in set_start -> set tangeant matrix to elastic
        
        #save variable at the begining of the Time increment
        self.__initialGradDisp = self.__currentGradDisp
        self.Lt = self.L.copy()

    
    def to_start(self):
        # self.to_start()         
        self.__currentGradDisp = self.__initialGradDisp  
        self.Lt = self.L.copy()
    
    def reset(self):
        """
        reset the constitutive law (time history)
        """
        #a modifier
        self.__currentGradDisp = self.__initialGradDisp = 0
        # self.__Statev = None
        self.__currentStress = None #lissStressTensor object describing the last computed stress (GetStress method)
        # self.__currentGradDisp = 0
        # self.__F0 = None

    
    def initialize(self, assembly, pb, t0 = 0., nlgeom=False):  
        self.nlgeom = nlgeom            
        if self.list_problem is None:  #only initialize once
            nb_points = assembly.n_elm_gp * assembly.mesh.n_elements
            
            #Definition of the set of nodes for boundary conditions
            if not(isinstance(self.__mesh, list)):            
                self.list_mesh = [self.__mesh for i in range(nb_points)]
                self.list_assembly = [self.__assembly.copy() for i in range(nb_points)]
            else: 
                self.list_mesh = self.__mesh
                self.list_assembly = self.__assembly
        
            self.list_problem = []
            self._list_volume = np.empty(nb_points)
            self._list_center = np.empty(nb_points, dtype=int)
            self.L = np.empty((nb_points,6,6))
                    
            print('-- Initialize micro problems --')
            for i in range(nb_points):
                print("\r", str(i+1),'/',str(nb_points), end="")
                
                crd = self.list_mesh[i].nodes
                type_el = self.list_mesh[i].elm_type
                xmax = np.max(crd[:,0]) ; xmin = np.min(crd[:,0])
                ymax = np.max(crd[:,1]) ; ymin = np.min(crd[:,1])
                zmax = np.max(crd[:,2]) ; zmin = np.min(crd[:,2])
                        
                crd_center = (np.array([xmin, ymin, zmin]) + np.array([xmax, ymax, zmax]))/2           
                self._list_volume[i] = (xmax-xmin)*(ymax-ymin)*(zmax-zmin) #total volume of the domain
        
                if '_StrainNodes' in self.list_mesh[i].ListSetOfNodes():
                    strain_nodes = self.list_mesh[i].node_sets['_StrainNodes']            
                else:
                    strain_nodes = self.list_mesh[i].add_nodes(crd_center,2) #add virtual nodes for macro strain
                    self.list_mesh[i].add_node_set(strain_nodes,'_StrainNodes')
               
                self._list_center[i] = np.linalg.norm(crd[:-2]-crd_center,axis=1).argmin()
                            # list_material.append(self.__constitutivelaw.copy())
                      
                #Type of problem
                self.list_problem.append(NonLinear(self.list_assembly[i], name = '_fe2_cell_'+str(i)))            
                pb_micro = self.list_problem[-1]
                meshperio = True
                
                #Shall add other conditions later on
                if meshperio:
                    DefinePeriodicBoundaryCondition(self.list_mesh[i],
                    [strain_nodes[0], strain_nodes[0], strain_nodes[0], strain_nodes[1], strain_nodes[1], strain_nodes[1]],
                    ['DispX',        'DispY',        'DispZ',       'DispX',         'DispY',        'DispZ'], dim='3D', Problemname = '_fe2_cell_'+str(i))
                else:
                    DefinePeriodicBoundaryConditionNonPerioMesh(self.list_mesh[i],
                    [strain_nodes[0], strain_nodes[0], strain_nodes[0], strain_nodes[1], strain_nodes[1], strain_nodes[1]],
                    ['DispX',        'DispY',        'DispZ',       'DispX',         'DispY',        'DispZ'], dim='3D', Problemname = '_fe2_cell_'+str(i))
                    
                pb_micro.BoundaryCondition('Dirichlet','DispX', 0, [self._list_center[i]])
                pb_micro.BoundaryCondition('Dirichlet','DispY', 0, [self._list_center[i]])
                pb_micro.BoundaryCondition('Dirichlet','DispZ', 0, [self._list_center[i]])
                
                self.L[i] = get_homogenized_stiffness(self.list_assembly[i])
            
            pb.MakeActive()
            self.Lt = self.L.copy()
            
            self.__strain = np.zeros((6, nb_points))
            self.__stress = np.zeros((6, nb_points))
            self.__Wm = np.zeros((4, nb_points))
    
            print('')

    def _update_pb(self, id_pb):
        dtime = self.__dtime
        strain = self.__new_strain
        nb_points = len(self.list_problem)
        pb = self.list_problem[id_pb]

        print("\r", str(id_pb+1),'/',str(nb_points), end="")
        strain_nodes = self.list_mesh[id_pb].node_sets['_StrainNodes']  

        pb.RemoveBC("Strain")
        pb.BoundaryCondition('Dirichlet','DispX', strain[0][id_pb], [strain_nodes[0]], initialValue = self.__strain[0][id_pb], name = 'Strain') #EpsXX
        pb.BoundaryCondition('Dirichlet','DispY', strain[1][id_pb], [strain_nodes[0]], initialValue = self.__strain[1][id_pb], name = 'Strain') #EpsYY
        pb.BoundaryCondition('Dirichlet','DispZ', strain[2][id_pb], [strain_nodes[0]], initialValue = self.__strain[2][id_pb], name = 'Strain') #EpsZZ
        pb.BoundaryCondition('Dirichlet','DispX', strain[3][id_pb], [strain_nodes[1]], initialValue = self.__strain[3][id_pb], name = 'Strain') #EpsXY
        pb.BoundaryCondition('Dirichlet','DispY', strain[4][id_pb], [strain_nodes[1]], initialValue = self.__strain[4][id_pb], name = 'Strain') #EpsXZ
        pb.BoundaryCondition('Dirichlet','DispZ', strain[5][id_pb], [strain_nodes[1]], initialValue = self.__strain[5][id_pb], name = 'Strain') #EpsYZ
        
        
        pb.nlsolve(dt = dtime, tmax = dtime, update_dt = True, ToleranceNR = 0.05, print_info = 0)        
        
        self.Lt[id_pb]= get_tangent_stiffness(pb.name)
        
        material = self.list_assembly[id_pb].weakform.GetConstitutiveLaw()
        stress_field = material.get_stress()
        self.__stress[:,id_pb] = np.array([1/self._list_volume[id_pb]*self.list_assembly[id_pb].integrate_field(stress_field[i]) for i in range(6)])
    
        Wm_field = material.Wm
        self.__Wm[:,id_pb] = (1/self._list_volume[id_pb]) * self.list_assembly[id_pb].integrate_field(Wm_field)


    def update(self,assembly, pb, dtime):   
        displacement = pb.get_dof_solution()

        if displacement is 0: 
            self.__currentGradDisp = 0
            self.__currentSigma = 0                        
        else:
            self.__currentGradDisp = assembly.get_grad_disp(displacement, "GaussPoint")

            grad_values = self.__currentGradDisp
            if self.nlgeom == False:
                strain  = [grad_values[i][i] for i in range(3)] 
                strain += [grad_values[0][1] + grad_values[1][0], grad_values[0][2] + grad_values[2][0], grad_values[1][2] + grad_values[2][1]]
            else:            
                strain  = [grad_values[i][i] + 0.5*sum([grad_values[k][i]**2 for k in range(3)]) for i in range(3)] 
                strain += [grad_values[0][1] + grad_values[1][0] + sum([grad_values[k][0]*grad_values[k][1] for k in range(3)])] 
                strain += [grad_values[0][2] + grad_values[2][0] + sum([grad_values[k][0]*grad_values[k][2] for k in range(3)])]
                strain += [grad_values[1][2] + grad_values[2][1] + sum([grad_values[k][1]*grad_values[k][2] for k in range(3)])]

        #resolution of the micro problem at each gauss points
        self.__new_strain = strain
        self.__dtime = dtime
        nb_points = len(self.list_problem)
        self.__stress = np.empty((6,nb_points))
        self.__Wm = np.empty((4,nb_points))
        
        print('-- Update micro cells --')
        
        # with multiprocessing.Pool(4) as pool:
        #     pool.map(self._update_pb, range(nb_points))
            
        for id_pb in range(nb_points):
            self._update_pb(id_pb)

        self.__strain = strain
        # self.__strain = StrainTensorList(strain)
        # self.__stress = StressTensorList([stress[i] for i in range(6)])        
        # self.__Wm = Wm
        
        print('')

       
            # H = self.GetH()
        
            # self.__currentSigma = StressTensorList([sum([TotalStrain[j]*assembly.convert_data(H[i][j]) for j in range(6)]) for i in range(6)]) #H[i][j] are converted to gauss point excepted if scalar

        

        # self.Run(dtime)

        # (DRloc , listDR, Detot, statev) = self.Run(dtime)
