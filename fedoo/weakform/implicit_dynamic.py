from fedoo.core.base import ConstitutiveLaw
from fedoo.core.weakform import WeakFormBase, WeakFormSum
from fedoo.weakform.stress_equilibrium import StressEquilibrium
from fedoo.weakform.inertia import Inertia

import numpy as np


class ImplicitDynamic(WeakFormBase):
    """
    Weak formulation of the steady heat equation (without time evolution).
        
    Parameters
    ----------
    thermal_constitutivelaw: ConstitutiveLaw name (str) or ConstitutiveLaw object
        Thermal Constitutive Law (:mod:`fedoo.constitutivelaw`)
    name: str
        name of the WeakForm     
    nlgeom: bool (default = False)
    """
    def __init__(self, constitutivelaw, density, beta, gamma, name = "", nlgeom = False, space = None):
        super().__init__(name, space)

        if name != "": 
            stiffness_name = name+'_stiffness'
        else: stiffness_name = inertia_name = ""
        
        self.stiffness_weakform = StressEquilibrium(constitutivelaw, stiffness_name, nlgeom, space)
        self.constitutivelaw = self.stiffness_weakform.constitutivelaw
        self.beta = beta
        self.gamma = gamma
        self.density = density
        self.__nlgeom = nlgeom
        
        self.assembly_options['assume_sym'] = True        
        
    def initialize(self, assembly, pb):
        self.stiffness_weakform.initialize(assembly, pb)
        assembly.sv_type['Velocity'] = 'Node'
        assembly.sv_type['Acceleration'] = 'Node'
        assembly.sv_type['_DeltaDisp'] = 'Node'
        assembly.sv['Velocity'] = 0
        assembly.sv['Acceleration'] = 0
        assembly.sv['Velocity_GP'] = [0,0,0]
        assembly.sv['Acceleration_GP'] = [0,0,0]
        assembly.sv['_DeltaDisp'] = 0

    def update(self, assembly, pb):
        self.stiffness_weakform.update(assembly, pb)
        assembly.sv['_DeltaDisp'] = pb._dU.reshape(-1,pb.mesh.n_nodes)
        assembly.sv['_DeltaDisp_GP'] = assembly.convert_data(assembly.sv['_DeltaDisp'], convert_from='Node', convert_to='GaussPoint')
        
        # self.inertia_weakform.update(assembly, pb)
        # assembly.sv['TempGradient'] = [0 if operator is 0 else 
        #             assembly.get_gp_results(operator, pb.get_dof_solution()) for operator in self.__op_grad_temp]   

    def update_2(self, assembly, pb):
        self.stiffness_weakform.update_2(assembly, pb)
        # self.inertia_weakform.update_2(assembly, pb)
        # assembly.sv['TempGradient'] = [0 if operator is 0 else 
        #             assembly.get_gp_results(operator, pb.get_dof_solution()) for operator in self.__op_grad_temp]   

    # def reset(self): #to update
    #     pass

    def to_start(self, assembly, pb): #to update
        self.stiffness_weakform.to_start(assembly, pb)


    def set_start(self, assembly, pb): #to update
        dt = pb.dtime ### dt is the time step of the previous increment       
        if pb.get_disp() is not 0: 
            #update velocity and acceleration
            new_acceleration = (1/(self.beta*dt**2)) * (assembly.sv['_DeltaDisp'] - dt*assembly.sv['Velocity']) - 1/self.beta*(0.5 - self.beta)*assembly.sv['Acceleration']
            
            assembly.sv['Velocity'] += dt * ( (1-self.gamma)*assembly.sv['Acceleration'] + self.gamma*new_acceleration)

            assembly.sv['Acceleration'] = new_acceleration            
            
            assembly.sv['Velocity_GP'] = assembly.convert_data(assembly.sv['Velocity'], convert_from='Node', convert_to='GaussPoint')
            assembly.sv['Acceleration_GP'] = assembly.convert_data(assembly.sv['Acceleration'], convert_from='Node', convert_to='GaussPoint')
        
        self.stiffness_weakform.set_start(assembly, pb)
        
        
    def get_weak_equation(self, assembly, pb):              
        op_dU = self.space.op_disp() #displacement increment (incremental formulation)
        op_dU_vir = [du.virtual if du != 0 else 0 for du in op_dU]
        dt = pb.dtime
                
        if pb._dU is 0: #start of iteration
            delta_disp = [0,0,0]
        else:
            delta_disp = assembly.sv['_DeltaDisp_GP']
                    
        acceleration = assembly.sv['Acceleration_GP'] 
        velocity = assembly.sv['Velocity_GP'] 

        diff_op = self.stiffness_weakform.get_weak_equation(assembly, pb)
        
        diff_op += sum([op_dU_vir[i]*( \
                    (op_dU[i]-delta_disp[i])*(self.density/(self.beta*dt**2)) -
                      velocity[i]*(self.density/(self.beta*dt)) -
                      acceleration[i] * (self.density*(0.5/self.beta - 1)) ) 
                      if op_dU_vir[i]!=0 else 0 for i in range(self.space.ndim)])

        return diff_op
    
    
    @property
    def nlgeom(self):
        return self.__nlgeom


class _NewmarkInteria(WeakFormBase):
    """
    Weak formulation of the steady heat equation (without time evolution).
        
    Parameters
    ----------
    thermal_constitutivelaw: ConstitutiveLaw name (str) or ConstitutiveLaw object
        Thermal Constitutive Law (:mod:`fedoo.constitutivelaw`)
    name: str
        name of the WeakForm     
    nlgeom: bool (default = False)
    """
    def __init__(self, density, beta, gamma, name = "", nlgeom = False, space = None):
        super().__init__(name, space)

        if name != "": 
            stiffness_name = name+'_stiffness'
        else: stiffness_name = inertia_name = ""
        
        self.beta = beta
        self.gamma = gamma
        self.density = density
        self.__nlgeom = nlgeom
        
        
    def initialize(self, assembly, pb):
        assembly.sv_type['Velocity'] = 'Node'
        assembly.sv_type['Acceleration'] = 'Node'
        assembly.sv_type['_DeltaDisp'] = 'Node'
        assembly.sv['Velocity'] = 0
        assembly.sv['Acceleration'] = 0
        assembly.sv['Velocity_GP'] = [0,0,0]
        assembly.sv['Acceleration_GP'] = [0,0,0]
        assembly.sv['_DeltaDisp'] = 0

    def update(self, assembly, pb):
        assembly.sv['_DeltaDisp'] = pb._dU.reshape(-1,pb.mesh.n_nodes)
        assembly.sv['_DeltaDisp_GP'] = assembly.convert_data(assembly.sv['_DeltaDisp'], convert_from='Node', convert_to='GaussPoint')
        
        # self.inertia_weakform.update(assembly, pb)
        # assembly.sv['TempGradient'] = [0 if operator is 0 else 
        #             assembly.get_gp_results(operator, pb.get_dof_solution()) for operator in self.__op_grad_temp]   

    def update_2(self, assembly, pb):
        pass
        # self.inertia_weakform.update_2(assembly, pb)
        # assembly.sv['TempGradient'] = [0 if operator is 0 else 
        #             assembly.get_gp_results(operator, pb.get_dof_solution()) for operator in self.__op_grad_temp]   

    # def reset(self): #to update
    #     pass

    def to_start(self, assembly, pb): #to update
        pass


    def set_start(self, assembly, pb): #to update
        dt = pb.dtime ### dt is the time step of the previous increment       
        if pb.get_disp() is not 0: 
            #update velocity and acceleration
            new_acceleration = (1/(self.beta*dt**2)) * (assembly.sv['_DeltaDisp'] - dt*assembly.sv['Velocity']) - 1/self.beta*(0.5 - self.beta)*assembly.sv['Acceleration']
            
            assembly.sv['Velocity'] += dt * ( (1-self.gamma)*assembly.sv['Acceleration'] + self.gamma*new_acceleration)

            assembly.sv['Acceleration'] = new_acceleration            
            
            assembly.sv['Velocity_GP'] = assembly.convert_data(assembly.sv['Velocity'], convert_from='Node', convert_to='GaussPoint')
            assembly.sv['Acceleration_GP'] = assembly.convert_data(assembly.sv['Acceleration'], convert_from='Node', convert_to='GaussPoint')
                
        
    def get_weak_equation(self, assembly, pb):              
        op_dU = self.space.op_disp() #displacement increment (incremental formulation)
        op_dU_vir = [du.virtual if du != 0 else 0 for du in op_dU]
        dt = pb.dtime
                
        if pb._dU is 0: #start of iteration
            delta_disp = [0,0,0]
        else:
            delta_disp = assembly.sv['_DeltaDisp_GP']
                    
        acceleration = assembly.sv['Acceleration_GP'] 
        velocity = assembly.sv['Velocity_GP'] 
        diff_op = sum([op_dU_vir[i]*( \
                    (op_dU[i]-delta_disp[i])*(self.density/(self.beta*dt**2)) -
                      velocity[i]*(self.density/(self.beta*dt)) -
                      acceleration[i] * (self.density*(0.5/self.beta - 1)) ) 
                      if op_dU_vir[i]!=0 else 0 for i in range(self.space.ndim)])

        return diff_op    
    
    @property
    def nlgeom(self):
        return self.__nlgeom


def ImplicitDynamic2(constitutivelaw, density, beta, gamma, name = "", nlgeom = False, space = None):
    
    stiffness_weakform = StressEquilibrium(constitutivelaw, "", nlgeom, space)  
    time_integration = _NewmarkInteria(density, beta, gamma, "", nlgeom, space)
    time_integration.assembly_options['assume_sym'] = True
    return WeakFormSum([stiffness_weakform, time_integration], name)

