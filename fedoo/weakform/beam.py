from fedoo.core.base import ConstitutiveLaw
from fedoo.constitutivelaw.beam import BeamProperties
from fedoo.core.weakform import WeakFormBase
import numpy as np

class BeamEquilibrium(WeakFormBase):
    """
    Weak formulation of the mechanical equilibrium equation for beam models.
    
    Geometrical non linearities not implemented for now.
    
    Parameters
    ----------
    material: ConstitutiveLaw name (str) or ConstitutiveLaw object
        Material Constitutive Law used to get the young modulus and poisson ratio
        The ConstitutiveLaw object should have a GetYoungModulus and GetPoissonRatio methods
        (as :mod:`fedoo.constitutivelaw.ElasticIsotrop`)        
    A: scalar or arrays of gauss point values
        Beam section area
    Jx: scalar or arrays of gauss point values
        Torsion constant
    Iyy: scalar or arrays of gauss point values
        Second moment of area with respect to y (beam local coordinate system)
    Izz:
        Second moment of area with respect to z (beam local coordinate system)
    k=0: scalar or arrays of gauss point values
        Shear coefficient. If k=0 (*default*) the beam use the bernoulli hypothesis
    name: str
        name of the WeakForm     
    """
    def __init__(self, material, A=None, Jx=None, Iyy=None, Izz=None, k=0, name = "",  nlgeom = False, space = None):
        #k: shear shape factor
        
        WeakFormBase.__init__(self,name, space)

        self.space.new_variable("DispX") 
        self.space.new_variable("DispY")            
        if self.space.ndim == 3:
            self.space.new_variable("DispZ")   
            self.space.new_variable("RotX") #torsion rotation 
            self.space.new_variable("RotY")   
            self.space.new_variable("RotZ")
            self.space.new_vector('Disp' , ('DispX', 'DispY', 'DispZ'))
            self.space.new_vector('Rot' , ('RotX', 'RotY', 'RotZ'))            
        elif self.space.ndim == 2:
            self.space.new_variable("RotZ")
            self.space.new_vector('Disp' , ['DispX', 'DispY'])            
            self.space.new_vector('Rot' , ['RotZ'] ) 
        # elif get_Dimension() == '2Dstress':
        #     assert 0, "No 2Dstress model for a beam kinematic. Choose '2Dplane' instead."
        
        if isinstance(material, BeamProperties):
            self.properties = material
        else:
            self.properties = BeamProperties(material, A, Jx, Iyy, Izz, k, name+"_properties")
            
        self.nlgeom = nlgeom #geometric non linearities -> False, True, 'UL' or 'TL' (True or 'UL': updated lagrangian - 'TL': total lagrangian)                
        """Method used to treat the geometric non linearities. 
            * Set to False if geometric non linarities are ignored (default). 
            * Set to True or 'UL' to use the updated lagrangian method (update the mesh)
            * Set to 'TL' to use the total lagrangian method (base on the initial mesh with initial displacement effet)
        """

    
    def initialize(self, assembly, pb):
        assembly.sv['BeamStrain'] = 0
        assembly.sv['BeamStress'] = 0
        if self.nlgeom: 
            if self.nlgeom is True: 
                self.nlgeom = 'UL'                
            elif isinstance(self.nlgeom, str): 
                self.nlgeom =self.nlgeom.upper()
                if self.nlgeom != 'UL':
                    raise NotImplementedError(f'{self.nlgeom} nlgeom not implemented for Interface force.')
            else:
                raise TypeError("nlgeom should be in {'TL', 'UL', True, False}")


    def update(self, assembly, pb):
        #function called when the problem is updated (NR loop or time increment)
        #Nlgeom implemented only for updated lagragian formulation        
        if self.nlgeom == 'UL':
            # if updated lagragian method -> update the mesh and recompute elementary op
            assembly.set_disp(pb.get_disp())               
            if assembly.current.mesh in assembly._saved_change_of_basis_mat:
                del assembly._saved_change_of_basis_mat[assembly.current.mesh]
                        
            assembly.current.compute_elementary_operators()           
        
        dof = pb.get_dof_solution() #displacement and rotation node values
        if dof is 0: 
            assembly.sv['BeamStrain'] = assembly.sv['BeamStress'] = 0
        else:
            op_beam_strain = assembly.space.op_beam_strain()    
            Ke = [0 if (np.isscalar(k) and k == 0)
                  else k for k in self.properties.get_beam_rigidity()] #make sure 0 values of rigidity are int
            
            assembly.sv['BeamStrain'] = [0 if Ke[i] == 0 else assembly.get_gp_results(op, dof) 
                                         for i,op in enumerate(op_beam_strain)]
            assembly.sv['BeamStress'] =  [Ke[i] * assembly.sv['BeamStrain'][i] for i in range(6)]
    
    
    def to_start(self, assembly, pb):    
        if self.nlgeom == 'UL':
            # if updated lagragian method -> reset the mesh to the begining of the increment
            assembly.set_disp(pb.get_disp())               
            if assembly.current.mesh in assembly._saved_change_of_basis_mat:
                del assembly._saved_change_of_basis_mat[assembly.current.mesh] 
            
            assembly.current.compute_elementary_operators()            
            
    
    def get_weak_equation(self, assembly, pb):
        eps = self.space.op_beam_strain()           
        Ke = self.properties.get_beam_rigidity()

        diff_op = sum([eps[i].virtual * eps[i] * Ke[i] if eps[i] != 0 else 0 for i in range(6)])

        initial_stress = assembly.sv['BeamStress']
        
        if initial_stress is not 0:    
            diff_op = diff_op + sum([eps[i].virtual * initial_stress[i] if eps[i] != 0 else 0 for i in range(6)])

        return diff_op


    def _get_generalized_stress_op(self):
        #only for post treatment
        eps = self.space.op_beam_strain()         
        Ke = self.properties.get_beam_rigidity()
        return [eps[i] * Ke[i] for i in range(6)]


    


    
