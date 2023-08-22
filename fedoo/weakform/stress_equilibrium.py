from fedoo.core.weakform import WeakFormBase
from fedoo.core.base import ConstitutiveLaw
from fedoo.util.voigt_tensors import StressTensorList, StrainTensorList
try:
    from simcoon import simmit as sim
    USE_SIMCOON = True
except ImportError: 
    USE_SIMCOON = False
    
import numpy as np

class StressEquilibrium(WeakFormBase):
    """
    Weak formulation of the mechanical equilibrium equation for solid models (without volume force).
    
    * This weak form can be used for solid in 3D or using a 2D plane assumption (plane strain or plane stress).
    * May include initial stress depending on the ConstitutiveLaw.
    * This weak form accepts geometrical non linearities (with nlgeom = True, 'UL' or 'TL'). In this case the initial displacement is also considered. 
    * For Non-Linear Problem (material or geometrical non linearities), it is strongly recomanded to use the :mod:`fedoo.constitutivelaw.Simcoon` Constitutive Law
    
    Parameters
    ----------
    constitutivelaw: ConstitutiveLaw name (str) or ConstitutiveLaw object
        Material Constitutive Law (:mod:`fedoo.constitutivelaw`)
    name: str
        name of the WeakForm     
    nlgeom: bool (default = False), 'UL' or 'TL'
        If True, the geometrical non linearities are activate when used in the context of NonLinearProblems (default updated lagrangian method)        
        such as :mod:`fedoo.problem.NonLinearStatic` or :mod:`fedoo.problem.NonLinearNewmark`
        If nlgeom == 'UL' the updated lagrangian method is used (same as True)
        If nlgeom == 'TL' the total lagrangian method is used
    space: ModelingSpace
        Modeling space associated to the weakform. If None is specified, the active ModelingSpace is considered.
    """
    def __init__(self, constitutivelaw, name = "", nlgeom = False, space = None):
        if isinstance(constitutivelaw, str):
            constitutivelaw = ConstitutiveLaw[constitutivelaw]

        if name == "":
            name = constitutivelaw.name
            
        WeakFormBase.__init__(self,name, space)
        
        self.space.new_variable("DispX") 
        self.space.new_variable("DispY")                
        if self.space.ndim == 3: 
            self.space.new_variable("DispZ")
            self.space.new_vector('Disp' , ('DispX', 'DispY', 'DispZ'))
        else: #2D assumed
            self.space.new_vector('Disp' , ('DispX', 'DispY'))
        
        self.constitutivelaw = constitutivelaw
        
        #default option for non linear 
        self.nlgeom = nlgeom #geometric non linearities -> False, True, 'UL' or 'TL' (True or 'UL': updated lagrangian - 'TL': total lagrangian)                
        """Method used to treat the geometric non linearities. 
            * Set to False if geometric non linarities are ignored (default). 
            * Set to True or 'UL' to use the updated lagrangian method (update the mesh)
            * Set to 'TL' to use the total lagrangian method (base on the initial mesh with initial displacement effet)
        """
        
        self.corate = 'log' #'log': logarithmic strain, 'jaumann': jaumann strain, 'green_naghdi', 'gn'...        

        self.assembly_options['assume_sym'] = True     #internalForce weak form should be symmetric (if TangentMatrix is symmetric) -> need to be checked for general case
            
    def get_weak_equation(self, assembly, pb):
        
        if self.nlgeom == 'TL': #add initial displacement effect 
            # assert 'DispGradient' in assembly.sv and 'PK2' in assembly.sv, ""
            # if not(hasattr(self.constitutivelaw, 'get_disp_grad')):
            #     raise NameError("The actual constitutive law is not compatible with NonLinear Internal Force weak form")                        
            eps = self.space.op_strain(assembly.sv['DispGradient'])
            initial_stress = assembly.sv['PK2']
            # initial_stress = self.constitutivelaw.get_pk2()
        else: 
            eps = self.space.op_strain()
            initial_stress = assembly.sv['Stress'] #Stress = Cauchy for updated lagrangian method
            # self.constitutivelaw.get_cauchy() #required for updated lagrangian method
        
        # H = self.constitutivelaw.get_H(self._dimension)
        H = assembly.sv['TangentMatrix']
        
        sigma = [sum([0 if eps[j] is 0 else eps[j]*H[i][j] for j in range(6)]) for i in range(6)]
                
        DiffOp = sum([0 if eps[i] is 0 else eps[i].virtual * sigma[i] for i in range(6)])
        
        if initial_stress is not 0:   
            if self.nlgeom:  #this term doesnt seem to improve convergence !
                DiffOp = DiffOp + sum([0 if self.__nl_strain_op_vir[i] is 0 else \
                                    self.__nl_strain_op_vir[i] * initial_stress[i] for i in range(6)])

            DiffOp = DiffOp + sum([0 if eps[i] is 0 else \
                                    eps[i].virtual * initial_stress[i] for i in range(6)])

        return DiffOp


    def initialize(self, assembly, pb):
        #### Put the require field to zeros if they don't exist in the assembly
        if 'Stress' not in assembly.sv: assembly.sv['Stress'] = 0
        if 'Strain' not in assembly.sv: assembly.sv['Strain'] = 0
        assembly.sv['DispGradient'] = 0
        
        if self.nlgeom:
            if not(USE_SIMCOON):
                raise NameError('Simcoon library need to be installed to deal with geometric non linearities (nlgeom = True)')
            
            if isinstance(self.nlgeom, str): self.nlgeom =self.nlgeom.upper()
            if self.nlgeom is True: self.nlgeom = 'UL'             
        
            if self.nlgeom == 'TL': 
                assembly.sv['PK2'] = 0
                                    
            #initialize non linear operator for strain
            op_grad_du = self.space.op_grad_u() #grad of displacement increment in the context of incremental problems
            if self.space.ndim == "3D":        
                #nl_strain_op_vir = 0.5*(vir(duk/dxi) * duk/dxj + duk/dxi * vir(duk/dxj)) using voigt notation and with a 2 factor on non diagonal terms
                nl_strain_op_vir = [sum([op_grad_du[k][i].virtual*op_grad_du[k][i] for k in range(3)]) for i in range(3)] 
                nl_strain_op_vir += [sum([op_grad_du[k][0].virtual*op_grad_du[k][1] + op_grad_du[k][1].virtual*op_grad_du[k][0] for k in range(3)])]  
                nl_strain_op_vir += [sum([op_grad_du[k][0].virtual*op_grad_du[k][2] + op_grad_du[k][2].virtual*op_grad_du[k][0] for k in range(3)])]
                nl_strain_op_vir += [sum([op_grad_du[k][1].virtual*op_grad_du[k][2] + op_grad_du[k][2].virtual*op_grad_du[k][1] for k in range(3)])]
            else:
                nl_strain_op_vir = [sum([op_grad_du[k][i].virtual*op_grad_du[k][i] for k in range(2)]) for i in range(2)] + [0]            
                nl_strain_op_vir += [sum([op_grad_du[k][0].virtual*op_grad_du[k][1] + op_grad_du[k][1].virtual*op_grad_du[k][0] for k in range(2)])] + [0,0]
            
            self.__nl_strain_op_vir = nl_strain_op_vir
            

    def update(self, assembly, pb):
        if self.nlgeom == 'UL':
            # if updated lagragian method -> update the mesh and recompute elementary op
            assembly.set_disp(pb.get_disp())               
            if assembly.current.mesh in assembly._saved_change_of_basis_mat:
                del assembly._saved_change_of_basis_mat[assembly.current.mesh]
                        
            assembly.current.compute_elementary_operators()        

        displacement = pb.get_dof_solution()
        
        if displacement is 0: 
            assembly.sv['DispGradient'] = 0
            assembly.sv['Stress'] = 0                        
            assembly.sv['Strain'] = 0
        else:
            grad_values = assembly.get_grad_disp(displacement, "GaussPoint")
            assembly.sv['DispGradient'] = grad_values
        
            
            #Compute the strain required for the constitutive law.             
            if self.nlgeom:
                self._corate_func(self, assembly, pb)
            else:
                _comp_linear_strain(self, assembly, pb)
                
   
    def update_2(self, assembly, pb):
        if self.nlgeom == 'TL':
            assembly.sv['PK2'] = assembly.sv['Stress'].cauchy_to_pk2(assembly.sv['F'])
            assembly.sv['TangentMatrix'] = sim.Lt_convert(assembly.sv['TangentMatrix'], assembly.sv['F'], assembly.sv['Stress'].asarray(), self._convert_Lt_tag)            
            # assembly.sv['TangentMatrix'] = sim.Lt_convert(assembly.sv['TangentMatrix'], assembly.sv['F'], assembly.sv['Stress'].asarray(), "DsigmaDe_JaumannDD_2_DSDE")

        
    def to_start(self, assembly, pb):    
        if self.nlgeom == 'UL':
            # if updated lagragian method -> reset the mesh to the begining of the increment
            assembly.set_disp(pb.get_disp())               
            if assembly.current.mesh in assembly._saved_change_of_basis_mat:
                del assembly._saved_change_of_basis_mat[assembly.current.mesh] 
            
            assembly.current.compute_elementary_operators()            
    
    def set_start(self, assembly, pb):
        if self.nlgeom:
            if 'DStrain' in assembly.sv:
                #rotate strain and stress -> need to be checked
                assembly.sv['Strain'] = StrainTensorList(sim.rotate_strain_R(assembly.sv_start['Strain'].asarray(),assembly.sv['DR']) + assembly.sv['DStrain'])
                
                #update cauchy stress 
                if assembly.sv['Stress'] is not 0:
                    stress = assembly.sv['Stress'].asarray() 
                    assembly.sv['Stress'] = StressTensorList(sim.rotate_stress_R(stress, assembly.sv['DR']))
                    if self.nlgeom == 'TL':
                        assembly.sv['PK2'] = assembly.sv['Stress'].cauchy_to_pk2(assembly.sv['F'])
                        
            
            #### debug test - should do nothing -> to delete  ####
            if self.nlgeom == 'UL':
                # if updated lagragian method -> reset the mesh to the begining of the increment
                assembly.set_disp(pb.get_disp())               
                if assembly.current.mesh in assembly._saved_change_of_basis_mat:
                    del assembly._saved_change_of_basis_mat[assembly.current.mesh] 
                
                assembly.current.compute_elementary_operators()            
            ### end ###

    @property
    def corate(self):
        """
        Properties defining the way strain is treated in finite strain problem (using a weakform with nlgeom = True)        
        corate can take the following str values:
            * "log" (default): exact logarithmic strain (strain is recomputed at each iteration)
            * "jaumann": Strain using the Jaumann derivative (strain is incremented)
            * "green_nagdhi" or "gn": Strain using the Green_Nagdhi derivative (strain is incremented)
        if nlgeom is False, this property has no effect.

        """
        return self._corate
    
    @corate.setter
    def corate(self, value):
        value = value.lower()
        if value == "log":
            self._corate_func = _comp_log_strain
            self._convert_Lt_tag = "DsigmaDe_2_DSDE"
        elif value in ["gn", "green_naghdi"]:
            self._corate_func = _comp_gn_strain
            self._convert_Lt_tag = "DsigmaDe_2_DSDE"
        elif value == "jaumann":
            self._corate_func = _comp_jaumann_strain
            self._convert_Lt_tag = "DsigmaDe_JaumannDD_2_DSDE"
        else: 
            raise NameError('corate value not understood. Choose between "log", "green_naghdi" or "jaumann"')
        self._corate = value
    
    
    
    # def copy(self, new_id = ""):
    #     """
    #     Return a raw deep copy of the weak form without keeping current state (internal variable).

    #     Parameters
    #     ----------
    #     new_id : TYPE, optional
    #         The name of the created constitutive law. The default is "".

    #     Returns
    #     -------
    #     The copy of the weakform
    #     """
    #     new_cl = self.constitutivelaw.copy()
        
    #     return StressEquilibrium(new_cl, name = "", nlgeom = self.nlgeom, space = self.space)
    



        
        
#funtions to compute strain
def _comp_linear_strain(wf, assembly, pb):    
    #only compatible with standard FE assembly. compatible with simcoon umat
    assert not(wf.nlgeom), "the current strain measure isn't adapted for finite strain"
    grad_values = assembly.sv['DispGradient']
    
    strain = np.empty((6,len(grad_values[0][0])),order='F') #order = F for compatibility with simcoon without performance loss in other cases
    strain[0:3] = [grad_values[i][i] for i in range(3)] 
    strain[3] = grad_values[0][1] + grad_values[1][0]
    strain[4] = grad_values[0][2] + grad_values[2][0]
    strain[5] = grad_values[1][2] + grad_values[2][1]
    assembly.sv['Strain'] = StrainTensorList(strain)

def _comp_log_strain(wf, assembly, pb):    
    grad_values = assembly.sv['DispGradient']
    eye_3 = np.empty((3,3,1), order='F')
    eye_3[:,:,0] = np.eye(3)
    F1 = np.add(eye_3, grad_values)
    assembly.sv['F'] = F1
    if 'F' not in assembly.sv_start:
        F0 = np.empty_like(F1)
        F0[...] = eye_3
        assembly.sv_start['F'] = F0
        
    (D,DR, Omega) = sim.objective_rate("log", assembly.sv_start['F'], F1, pb.dtime, False)
    assembly.sv['DR'] = DR
    assembly.sv['Strain'] = StrainTensorList(sim.Log_strain(F1, True, False))  

def _comp_jaumann_strain(wf, assembly, pb):    
    grad_values = assembly.sv['DispGradient']
    eye_3 = np.empty((3,3,1), order='F')
    eye_3[:,:,0] = np.eye(3)
    F1 = np.add(eye_3, grad_values)
    assembly.sv['F'] = F1
    if 'F' not in assembly.sv_start:
        F0 = np.empty_like(F1)
        F0[...] = eye_3
        assembly.sv_start['F'] = F0
        
    (DStrain, D,DR, Omega) = sim.objective_rate("jaumann", assembly.sv_start['F'], F1, pb.dtime, True)
    assembly.sv['DR'] = DR
    assembly.sv['DStrain'] = StrainTensorList(DStrain)  

def _comp_gn_strain(wf, assembly, pb):    
    #green_naghdi corate
    grad_values = assembly.sv['DispGradient']
    eye_3 = np.empty((3,3,1), order='F')
    eye_3[:,:,0] = np.eye(3)
    F1 = np.add(eye_3, grad_values)
    assembly.sv['F'] = F1
    if 'F' not in assembly.sv_start:
        F0 = np.empty_like(F1)
        F0[...] = eye_3
        assembly.sv_start['F'] = F0
        
    (DStrain, D,DR, Omega) = sim.objective_rate("green_naghdi", assembly.sv_start['F'], F1, pb.dtime, True)
    assembly.sv['DR'] = DR
    assembly.sv['DStrain'] = StrainTensorList(DStrain)  

def _comp_linear_strain_pgd(wf, assembly, pb): 
    #may be compatible with other methods like PGD but not compatible with simcoon
    assert not(wf.nlgeom), "the current strain measure isn't adapted for finite strain"
    grad_values = assembly.sv['DispGradient']
    
    strain  = [grad_values[i][i] for i in range(3)] 
    strain += [grad_values[0][1] + grad_values[1][0], grad_values[0][2] + grad_values[2][0], grad_values[1][2] + grad_values[2][1]]
    assembly.sv['Strain'] = StrainTensorList(strain)
    
def _comp_gl_strain(wf, assembly, pb): 
    #not compatible with simcoon
    if not(wf.nlgeom):
        return _comp_linear_strain_pgd(wf, assembly, pb)
    else:
        grad_values = assembly.sv['DispGradient']    
        #GL strain tensor #need to be improve from simcoon functions to get the logarithmic strain tensor...
        strain  = [grad_values[i][i] + 0.5*sum([grad_values[k][i]**2 for k in range(3)]) for i in range(3)] 
        strain += [grad_values[0][1] + grad_values[1][0] + sum([grad_values[k][0]*grad_values[k][1] for k in range(3)])] 
        strain += [grad_values[0][2] + grad_values[2][0] + sum([grad_values[k][0]*grad_values[k][2] for k in range(3)])]
        strain += [grad_values[1][2] + grad_values[2][1] + sum([grad_values[k][1]*grad_values[k][2] for k in range(3)])]
        return StrainTensorList(strain)