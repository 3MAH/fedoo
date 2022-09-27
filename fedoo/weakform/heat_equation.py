from fedoo.core.base import ConstitutiveLaw, WeakForm
from fedoo.core.assembly_sum import WeakFormSum
import numpy as np


class SteadyHeatEquation(WeakForm):
    """
    Weak formulation of the mechanical equilibrium equation for solid models (without volume force).
    
    * This weak form can be used for solid in 3D or using a 2D plane assumption (plane strain or plane stress).
    * May include initial stress depending on the ConstitutiveLaw.
    * This weak form accepts geometrical non linearities (with nlgeom = True). In this case the initial displacement is also considered. 
    * For Non-Linear Problem (material or geometrical non linearities), it is strongly recomanded to use the :mod:`fedoo.constitutivelaw.Simcoon` Constitutive Law
    
    Parameters
    ----------
    CurrentConstitutiveLaw: ConstitutiveLaw name (str) or ConstitutiveLaw object
        Material Constitutive Law (:mod:`fedoo.constitutivelaw`)
    name: str
        name of the WeakForm     
    nlgeom: bool (default = False)
        If True, the geometrical non linearities are activate when used in the context of NonLinearProblems 
        such as :mod:`fedoo.problem.NonLinearStatic` or :mod:`fedoo.problem.NonLinearNewmark`
    """
    def __init__(self, thermal_constitutivelaw, name = None, nlgeom = False, space = None):
        if isinstance(thermal_constitutivelaw, str):
            thermal_constitutivelaw = ConstitutiveLaw.get_all()[thermal_constitutivelaw]

        if name is None:
            name = thermal_constitutivelaw.name
            
        WeakForm.__init__(self,name,space)
        
        self.space.new_variable("Temp") #temperature
        
        if self.space.ndim == 3:    
            self.__op_grad_temp = [self.space.derivative('Temp', 'X'), self.space.derivative('Temp', 'Y'), self.space.derivative('Temp', 'Z')]
        else: #2D
            self.__op_grad_temp = [self.space.derivative('Temp', 'X'), self.space.derivative('Temp', 'Y'), 0]
        
        self.__op_grad_temp_vir = [0 if op is 0 else op.virtual for op in self.__op_grad_temp]

            
        self.__ConstitutiveLaw = thermal_constitutivelaw
                        
        # self.__nlgeom = nlgeom #geometric non linearities
        
    def initialize(self, assembly, pb, t0 = 0.):
        if not(np.isscalar(pb.GetDoFSolution())):
            self.__grad_temp = [0 if operator is 0 else 
                                assembly.get_gp_results(operator, pb.GetDoFSolution()) for operator in self.__op_grad_temp]
        else: 
            self.__grad_temp = [0 for operator in self.__op_grad_temp]

    def update(self, assembly, pb, dtime):
        self.__grad_temp = [0 if operator is 0 else 
                    assembly.get_gp_results(operator, pb.GetDoFSolution()) for operator in self.__op_grad_temp]   

    def reset(self): #to update
        pass

    def to_start(self): #to update
        pass       
        
    def get_DifferentialOperator(self, mesh=None, localFrame = None):      
             
        K = self.__ConstitutiveLaw.thermal_conductivity

        DiffOp = sum([0 if self.__op_grad_temp_vir[i] is 0 else self.__op_grad_temp_vir[i] * 
                      sum([0 if self.__op_grad_temp[j] is 0 else self.__op_grad_temp[j] * K[i][j] for j in range(3)]) 
                      for i in range(3)])
        
        #add initial state for incremental resolution
        DiffOp += sum([0 if self.__op_grad_temp_vir[i] is 0 else self.__op_grad_temp_vir[i] * 
                      sum([self.__grad_temp[j] * K[i][j] for j in range(3) if  K[i][j] is not 0 and self.__grad_temp[j] is not 0]) 
                      for i in range(3)])
                
        return DiffOp

    def GetConstitutiveLaw(self):
        return self.__ConstitutiveLaw
    
    @property
    def nlgeom(self):
        return self.__nlgeom



class TemperatureTimeDerivative(WeakForm):
    """
    Weak formulation of the mechanical equilibrium equation for solid models (without volume force).
    
    * This weak form can be used for solid in 3D or using a 2D plane assumption (plane strain or plane stress).
    * May include initial stress depending on the ConstitutiveLaw.
    * This weak form accepts geometrical non linearities (with nlgeom = True). In this case the initial displacement is also considered. 
    * For Non-Linear Problem (material or geometrical non linearities), it is strongly recomanded to use the :mod:`fedoo.constitutivelaw.Simcoon` Constitutive Law
    
    Parameters
    ----------
    CurrentConstitutiveLaw: ConstitutiveLaw name (str) or ConstitutiveLaw object
        Material Constitutive Law (:mod:`fedoo.constitutivelaw`)
    name: str
        name of the WeakForm     
    nlgeom: bool (default = False)
        If True, the geometrical non linearities are activate when used in the context of NonLinearProblems 
        such as :mod:`fedoo.problem.NonLinearStatic` or :mod:`fedoo.problem.NonLinearNewmark`
    """
    def __init__(self, thermal_constitutivelaw, name = None, nlgeom = False, space = None):
        if isinstance(thermal_constitutivelaw, str):
            thermal_constitutivelaw = ConstitutiveLaw.get_all()[thermal_constitutivelaw]

        if name is None:
            name = thermal_constitutivelaw.name
            
        WeakForm.__init__(self,name,space)
        
        self.space.new_variable("Temp") #temperature
            
        self.__ConstitutiveLaw = thermal_constitutivelaw
        self.__dtime = 0
        
        # self.__InitialStressTensor = 0
        # self.__InitialGradDispTensor = None
        
        # self.__nlgeom = nlgeom #geometric non linearities
        
    # def updateInitialStress(self,InitialStressTensor):                                                
    #     self.__InitialStressTensor = InitialStressTensor       

    # def GetInitialStress(self):                                                
    #     return self.__InitialStressTensor 
        
    def initialize(self, assembly, pb, t0 = 0.):
        if not(np.isscalar(pb.GetDoFSolution())):
            self.__temp_start = assembly.convert_data(pb.get_temp(), convert_from='Node', convert_to='GaussPoint')
            self.__temp = self.__temp_start
        else: 
            self.__temp_start = self.__temp = 0

    def update(self, assembly, pb, dtime):
        self.__temp = assembly.convert_data(pb.get_temp(), convert_from='Node', convert_to='GaussPoint')
        
    def reset(self):
        pass
    #     self.__ConstitutiveLaw.reset()
    #     self.__InitialStressTensor = 0
    #     self.__InitialGradDispTensor = None

    def to_start(self):
        pass
           
    def set_start(self, assembly, pb, dtime):
        self.__dtime = dtime        
        self.__temp_start = self.__temp                                  
        #no need to update Initial Stress because the last computed stress remained unchanged

    def get_DifferentialOperator(self, mesh=None, localFrame = None):      
        
        rho_c = self.__ConstitutiveLaw.density * self.__ConstitutiveLaw.specific_heat        
        
        op_temp = self.space.variable('Temp') #temperature increment (incremental weakform)
        #steady state should not include the following term
        if self.__dtime != 0:
            return 1/self.__dtime * rho_c * (op_temp.virtual * op_temp + op_temp.virtual *(self.__temp - self.__temp_start)) 
            # return 1/self.__dtime * rho_c * (op_temp.virtual * op_temp + op_temp.virtual *(self.__temp))            
        else:            
            return 0
    
    def GetConstitutiveLaw(self):
        return self.__ConstitutiveLaw
    
    @property
    def nlgeom(self):
        return self.__nlgeom
    
    
    
def HeatEquation(thermal_constitutivelaw, name = None, nlgeom = False, space = None):
    heat_eq_diffusion = SteadyHeatEquation(thermal_constitutivelaw, "", nlgeom, space)
    heat_eq_time = TemperatureTimeDerivative(thermal_constitutivelaw, "", nlgeom, space)
    heat_eq_time.assembly_options['mat_lumping'] = True #use mat_lumping for the temperature time derivative 
    if name is None: 
        if isinstance(thermal_constitutivelaw,str): name = ConstitutiveLaw().get_all()[thermal_constitutivelaw].name
        else: name = thermal_constitutivelaw.name
    return WeakFormSum([heat_eq_diffusion, heat_eq_time], name)

