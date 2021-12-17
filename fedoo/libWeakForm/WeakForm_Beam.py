from fedoo.libWeakForm.WeakForm   import *
from fedoo.libConstitutiveLaw.ConstitutiveLaw import ConstitutiveLaw
from fedoo.libUtil.BeamStrainOperator import GetBeamStrainOperator
from fedoo.libUtil.ModelingSpace import Variable, Vector, GetDimension

class Beam(WeakForm):
    """
    Weak formulation of the mechanical equilibrium equation for beam models.
    Geometrical non linearities not implemented for now
    
    Parameters
    ----------
    CurrentConstitutiveLaw: ConstitutiveLaw ID (str) or ConstitutiveLaw object
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
    ID: str
        ID of the WeakForm     
    """
    def __init__(self, CurrentConstitutiveLaw, Section, Jx, Iyy, Izz, k=0, ID = ""):
        #k: shear shape factor
        
        if isinstance(CurrentConstitutiveLaw, str):
            CurrentConstitutiveLaw = ConstitutiveLaw.GetAll()[CurrentConstitutiveLaw]

        if ID == "":
            ID = CurrentConstitutiveLaw.GetID()
            
        WeakForm.__init__(self,ID)

        Variable("DispX") 
        Variable("DispY")            
        if GetDimension() == '3D':
            Variable("DispZ")   
            Variable("RotX") #torsion rotation 
            Variable("RotY")   
            Variable("RotZ")
            Vector('Disp' , ('DispX', 'DispY', 'DispZ'))
            Vector('Rot' , ('RotX', 'RotY', 'RotZ'))            
        elif GetDimension() == '2Dplane':
            Variable("RotZ")
            Vector('Disp' , ['DispX', 'DispY'])            
            Vector('Rot' , ['RotZ'] ) 
        elif GetDimension() == '2Dstress':
            assert 0, "No 2Dstress model for a beam kinematic. Choose '2Dplane' instead."
                          
        self.__ConstitutiveLaw = CurrentConstitutiveLaw
        self.__parameters = {'Section': Section, 'Jx': Jx, 'Iyy':Iyy, 'Izz':Izz, 'k':k}        
    
    def GetDifferentialOperator(self, localFrame):
        E  = self.__ConstitutiveLaw.GetYoungModulus()
        nu = self.__ConstitutiveLaw.GetPoissonRatio()       
        G = E/(1+nu)/2
        
        eps, eps_vir = GetBeamStrainOperator()           
        beamShearStifness = self.__parameters['k'] * G * self.__parameters['Section']

        Ke = [E*self.__parameters['Section'], beamShearStifness, beamShearStifness, G*self.__parameters['Jx'], E*self.__parameters['Iyy'], E*self.__parameters['Izz']]
        return sum([eps_vir[i] * eps[i] * Ke[i] for i in range(6)])                

    
    def GetGeneralizedStress(self):
        #only for post treatment

        eps, eps_vir = GetBeamStrainOperator()
        E  = self.__ConstitutiveLaw.GetYoungModulus()
        nu = self.__ConstitutiveLaw.GetPoissonRatio()       
        G = E/(1+nu)/2
        beamShearStifness = self.__parameters['k'] * G * self.__parameters['Section']

        Ke = [E*self.__parameters['Section'], beamShearStifness, beamShearStifness, G*self.__parameters['Jx'], E*self.__parameters['Iyy'], E*self.__parameters['Izz']]
        return [eps[i] * Ke[i] for i in range(6)]


def BernoulliBeam(CurrentConstitutiveLaw, Section, Jx, Iyy, Izz, ID = ""):
    """
    Weak formulation of the mechanical equilibrium equation for beam model base on the Bernoulli hypothesis (no shear strain)   
    This weak formulation is an alias for :mod:`fedoo.libWeakForm.Beam` with k=0
    
    Parameters
    ----------
    CurrentConstitutiveLaw: ConstitutiveLaw ID (str) or ConstitutiveLaw object
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
    ID: str
        ID of the WeakForm     
    """
    #same as beam with k=0 (no shear effect)
    return Beam(CurrentConstitutiveLaw, Section, Jx, Iyy, Izz, k=0, ID = ID)

# class BernoulliBeam(WeakForm):
#     def __init__(self, CurrentConstitutiveLaw, Section, Jx, Iyy, Izz, ID = ""):
#         if isinstance(CurrentConstitutiveLaw, str):
#             CurrentConstitutiveLaw = ConstitutiveLaw.GetAll()[CurrentConstitutiveLaw]

#         if ID == "":
#             ID = CurrentConstitutiveLaw.GetID()
            
#         WeakForm.__init__(self,ID)

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



    


    
