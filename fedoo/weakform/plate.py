from fedoo.core.weakform import WeakFormBase, WeakFormSum
from fedoo.core.base import ConstitutiveLaw
# from fedoo.util.StrainOperator import GetStrainOperator
# from fedoo.core.modelingspace import Variable, Vector, get_Dimension

class PlateFI(WeakFormBase):
    """
    Weak formulation of the mechanical equilibrium equation for plate models.
    This weak form has to be used in combination with a Shell Constitutive Law
    like :mod:`fedoo.constitutivelaw.ShellHomogeneous` or `fedoo.constitutivelaw.ShellLaminate`.
    Geometrical non linearities not implemented for now.
    
    Parameters
    ----------
    PlateConstitutiveLaw: ConstitutiveLaw name (str) or ConstitutiveLaw object
        Shell Constitutive Law (:mod:`fedoo.constitutivelaw`)
    name: str
        name of the WeakForm     
    """
    def __init__(self, PlateConstitutiveLaw, name = "", space=None):
        #k: shear shape factor
        
        if isinstance(PlateConstitutiveLaw, str):
            PlateConstitutiveLaw = ConstitutiveLaw.get_all()[PlateConstitutiveLaw]

        if name == "":
            name = PlateConstitutiveLaw.name
            
        WeakFormBase.__init__(self,name, space)
        
        assert self.space.ndim == 3, "No 2D model for a plate kinematic. Choose '3D' problem dimension."

        self.space.new_variable("DispX") 
        self.space.new_variable("DispY")            
        self.space.new_variable("DispZ")   
        self.space.new_variable("RotX") #torsion rotation 
        self.space.new_variable("RotY")   
        self.space.new_variable("RotZ")
        self.space.new_vector('Disp' , ('DispX', 'DispY', 'DispZ'))
        self.space.new_vector('Rot' , ('RotX', 'RotY', 'RotZ'))     
        
        self.__ShellConstitutiveLaw = PlateConstitutiveLaw
        
    def GetGeneralizedStrainOperator(self):
        #membrane strain
        EpsX = self.space.derivative('DispX', 'X')
        EpsY = self.space.derivative('DispY', 'Y')
        GammaXY = self.space.derivative('DispX', 'Y')+self.space.derivative('DispY', 'X')
        
        #bending curvature
        XsiX = -self.space.derivative('RotY', 'X') # flexion autour de Y -> courbure suivant x
        XsiY =  self.space.derivative('RotX',  'Y') # flexion autour de X -> courbure suivant y #ok
        XsiXY = self.space.derivative('RotX',  'X') - self.space.derivative('RotY',  'Y')
        
        #shear
        GammaXZ = self.space.derivative('DispZ', 'X') + self.space.variable('RotY')
        GammaYZ = self.space.derivative('DispZ', 'Y') - self.space.variable('RotX') 
        
        return [EpsX, EpsY, GammaXY, XsiX, XsiY, XsiXY, GammaXZ, GammaYZ]                
        
    def get_weak_equation(self, localFrame):        
        H = self.__ShellConstitutiveLaw.GetShellRigidityMatrix()

        GeneralizedStrain = self.GetGeneralizedStrainOperator()                
        GeneralizedStress = [sum([0 if GeneralizedStrain[j] is 0 else GeneralizedStrain[j]*H[i][j] for j in range(8)]) for i in range(8)]
        
        diffop = sum([0 if GeneralizedStrain[i] is 0 else GeneralizedStrain[i].virtual*GeneralizedStress[i] for i in range(8)])
        
        #penalty for RotZ
        penalty = 1e-6
        diffop += self.space.variable('RotZ').virtual*self.space.variable('RotZ')*penalty
        
        return diffop
      
    def GetConstitutiveLaw(self):
        return self.__ShellConstitutiveLaw

    def update(self, assembly, pb, dtime):
        pass
        # self.__ShellConstitutiveLaw.update(assembly, pb, dtime)    


class PlateShear(PlateFI): #weak form of plate shear energy containing only the shear strain energy

    def get_weak_equation(self, localFrame):   
        #shear
        H = self._PlateFI__ShellConstitutiveLaw.GetShellRigidityMatrix_RI()
                
        GammaXZ = self.space.derivative('DispZ', 'X') + self.space.variable('RotY')
        GammaYZ = self.space.derivative('DispZ', 'Y') - self.space.variable('RotX') 
        
        GeneralizedStrain = [GammaXZ, GammaYZ]
        GeneralizedStress = [sum([0 if GeneralizedStrain[j] is 0 else GeneralizedStrain[j]*H[i][j] for j in range(2)]) for i in range(2)]
        
        return sum([0 if GeneralizedStrain[i] is 0 else GeneralizedStrain[i].virtual*GeneralizedStress[i] for i in range(2)])        
        
    
class PlateKirchhoffLove(PlateFI): #plate without shear strain

    def get_weak_equation(self, localFrame):  
        #all component but shear, for full integration
        H = self._PlateFI__ShellConstitutiveLaw.GetShellRigidityMatrix_FI()
        
        GeneralizedStrain = self.GetGeneralizedStrainOperator()                       
        GeneralizedStress = [sum([0 if GeneralizedStrain[j] is 0 else GeneralizedStrain[j]*H[i][j] for j in range(6)]) for i in range(6)]
        
        diffop = sum([GeneralizedStrain[i].virtual*GeneralizedStress[i] for i in range(6)])
        
        #penalty for RotZ
        penalty = 1e-6
        diffop += self.space.variable('RotZ').virtual*self.space.variable('RotZ')*penalty
        
        return diffop

    
def PlateRI(PlateConstitutiveLaw, name = None, space=None): #plate weakform which force reduced integration for shear terms
    plate_shear = PlateShear(PlateConstitutiveLaw, "", space)
    plate_kl = PlateKirchhoffLove(PlateConstitutiveLaw, "", space)

    plate_shear.assembly_options['n_elm_gp'] = 1 #use reduced integration for shear components
    if name is None: 
        if isinstance(PlateConstitutiveLaw,str): name = ConstitutiveLaw().get_all()[PlateConstitutiveLaw].name
        else: name = PlateConstitutiveLaw.name
    return WeakFormSum([plate_kl, plate_shear], name)


def Plate(PlateConstitutiveLaw, name = None, space=None):
    plate_shear = PlateShear(PlateConstitutiveLaw, "", space)
    plate_kl = PlateKirchhoffLove(PlateConstitutiveLaw, "", space)
    
    #if linear element 'ptri3' and 'pquad4': use reduced integration for shear terms
    plate_shear.assembly_options['n_elm_gp', 'ptri3'] = 1 
    plate_shear.assembly_options['n_elm_gp', 'pquad4'] = 1 
    if name is None: 
        if isinstance(PlateConstitutiveLaw,str): name = ConstitutiveLaw().get_all()[PlateConstitutiveLaw].name
        else: name = PlateConstitutiveLaw.name
    return WeakFormSum([plate_kl, plate_shear], name)
    


    
