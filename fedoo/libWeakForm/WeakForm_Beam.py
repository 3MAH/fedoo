from fedoo.libWeakForm.WeakForm import WeakForm
from fedoo.libConstitutiveLaw.ConstitutiveLaw import ConstitutiveLaw

class Beam(WeakForm):
    """
    Weak formulation of the mechanical equilibrium equation for beam models.
    Geometrical non linearities not implemented for now
    
    Parameters
    ----------
    CurrentConstitutiveLaw: ConstitutiveLaw name (str) or ConstitutiveLaw object
        Material Constitutive Law used to get the young modulus and poisson ratio
        The ConstitutiveLaw object should have a GetYoungModulus and GetPoissonRatio methods
        (as :mod:`fedoo.libConstitutiveLaw.ElasticIsotrop`)        
    Section: scalar or arrays of gauss point values
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
    def __init__(self, CurrentConstitutiveLaw, Section, Jx, Iyy, Izz, k=0, name = "", space = None):
        #k: shear shape factor
        
        if isinstance(CurrentConstitutiveLaw, str):
            CurrentConstitutiveLaw = ConstitutiveLaw.get_all()[CurrentConstitutiveLaw]

        if name == "":
            name = CurrentConstitutiveLaw.name
            
        WeakForm.__init__(self,name, space)

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
        # elif GetDimension() == '2Dstress':
        #     assert 0, "No 2Dstress model for a beam kinematic. Choose '2Dplane' instead."
                          
        self.__ConstitutiveLaw = CurrentConstitutiveLaw
        self.__parameters = {'Section': Section, 'Jx': Jx, 'Iyy':Iyy, 'Izz':Izz, 'k':k}        
    
    def GetDifferentialOperator(self, localFrame):
        E  = self.__ConstitutiveLaw.GetYoungModulus()
        nu = self.__ConstitutiveLaw.GetPoissonRatio()       
        G = E/(1+nu)/2
        
        eps = self.space.op_beam_strain()           
        beamShearStifness = self.__parameters['k'] * G * self.__parameters['Section']

        Ke = [E*self.__parameters['Section'], beamShearStifness, beamShearStifness, G*self.__parameters['Jx'], E*self.__parameters['Iyy'], E*self.__parameters['Izz']]
        return sum([eps[i].virtual * eps[i] * Ke[i] if eps[i] != 0 else 0 for i in range(6)])                

    
    def GetGeneralizedStress(self):
        #only for post treatment

        eps, eps_vir = GetBeamStrainOperator()
        E  = self.__ConstitutiveLaw.GetYoungModulus()
        nu = self.__ConstitutiveLaw.GetPoissonRatio()       
        G = E/(1+nu)/2
        beamShearStifness = self.__parameters['k'] * G * self.__parameters['Section']

        Ke = [E*self.__parameters['Section'], beamShearStifness, beamShearStifness, G*self.__parameters['Jx'], E*self.__parameters['Iyy'], E*self.__parameters['Izz']]
        return [eps[i] * Ke[i] for i in range(6)]


def BernoulliBeam(CurrentConstitutiveLaw, Section, Jx, Iyy, Izz, name = ""):
    """
    Weak formulation of the mechanical equilibrium equation for beam model base on the Bernoulli hypothesis (no shear strain)   
    This weak formulation is an alias for :mod:`fedoo.libWeakForm.Beam` with k=0
    
    Parameters
    ----------
    CurrentConstitutiveLaw: ConstitutiveLaw name (str) or ConstitutiveLaw object
        Material Constitutive Law used to get the young modulus and poisson ratio
        The ConstitutiveLaw object should have a GetYoungModulus and GetPoissonRatio methods
        (as :mod:`fedoo.libConstitutiveLaw.ElasticIsotrop`)        
    Section: scalar or arrays of gauss point values
        Beam section area
    Jx: scalar or arrays of gauss point values
        Torsion constant
    Iyy: scalar or arrays of gauss point values
        Second moment of area with respect to y (beam local coordinate system)
    Izz:
        Second moment of area with respect to z (beam local coordinate system)
    name: str
        name of the WeakForm     
    """
    #same as beam with k=0 (no shear effect)
    return Beam(CurrentConstitutiveLaw, Section, Jx, Iyy, Izz, k=0, name = name)

# class BernoulliBeam(WeakForm):
#     def __init__(self, CurrentConstitutiveLaw, Section, Jx, Iyy, Izz, name = ""):
#         if isinstance(CurrentConstitutiveLaw, str):
#             CurrentConstitutiveLaw = ConstitutiveLaw.get_all()[CurrentConstitutiveLaw]

#         if name == "":
#             name = CurrentConstitutiveLaw.name
            
#         WeakForm.__init__(self,name)

#         Variable("DispX") 
#         Variable("DispY")     
        
#         if GetDimension() == '3D':
#             Variable("DispZ")   
#             Variable("RotX") #torsion rotation   
#             Variable("RotY") #flexion   
#             Variable("RotZ") #flexion   
#             Variable.SetVector('Disp' , ('DispX', 'DispY', 'DispZ') , 'global')
#             Variable.SetVector('Rot' , ('RotX', 'RotY', 'RotZ') , 'global')
#         elif GetDimension() == '2Dplane':
#             Variable("RotZ")
#             # Variable.SetDerivative('DispY', 'RotZ') #only valid with Bernoulli model       
#             Variable.SetVector('Disp' , ('DispX', 'DispY') )            
#         elif GetDimension() == '2Dstress':
#             assert 0, "No 2Dstress model for a beam kinematic. Choose '2Dplane' instead."
                  
#         self.__ConstitutiveLaw = CurrentConstitutiveLaw
#         self.__parameters = {'Section': Section, 'Jx': Jx, 'Iyy':Iyy, 'Izz':Izz}
    
#     def GetDifferentialOperator(self, localFrame):
#         E  = self.__ConstitutiveLaw.GetYoungModulus()
#         nu = self.__ConstitutiveLaw.GetPoissonRatio()       
#         G = E/(1+nu)/2
        
#         eps, eps_vir = GetBernoulliBeamStrainOperator()           

#         Ke = [E*self.__parameters['Section'], 0, 0, G*self.__parameters['Jx'], E*self.__parameters['Iyy'], E*self.__parameters['Izz']]
#         return sum([eps_vir[i] * eps[i] * Ke[i] for i in range(6)])                

    
#     def GetGeneralizedStress(self):
#         #only for post treatment

#         eps, eps_vir = GetBernoulliBeamStrainOperator()
#         E  = self.__ConstitutiveLaw.GetYoungModulus()
#         nu = self.__ConstitutiveLaw.GetPoissonRatio()       
#         G = E/(1+nu)/2
#         Ke = [E*self.__parameters['Section'], 0, 0, G*self.__parameters['Jx'], E*self.__parameters['Iyy'], E*self.__parameters['Izz']]
        
#         return [eps[i] * Ke[i] for i in range(6)]



    


    
