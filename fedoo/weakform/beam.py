from fedoo.core.base import ConstitutiveLaw
from fedoo.constitutivelaw.beam import BeamProperties
from fedoo.core.weakform import WeakFormBase

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
    def __init__(self, material, A=None, Jx=None, Iyy=None, Izz=None, k=0, name = "", space = None):
        #k: shear shape factor
        
        # if isinstance(material, str):
        #     material = ConstitutiveLaw[material]

        # if name == "":
        #     name = material.name
            
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
        
        
    
    def get_weak_equation(self, assembly, pb):
        eps = self.space.op_beam_strain()           
        Ke = self.properties.get_beam_rigidity()

        return sum([eps[i].virtual * eps[i] * Ke[i] if eps[i] != 0 else 0 for i in range(6)])                

    
    def GetGeneralizedStress(self):
        #only for post treatment
        eps = self.space.op_beam_strain()         
        Ke = self.properties.get_beam_rigidity()
        return [eps[i] * Ke[i] for i in range(6)]


    


    
