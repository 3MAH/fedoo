from fedoo.libWeakForm.WeakForm   import *
from fedoo.libConstitutiveLaw.ConstitutiveLaw import ConstitutiveLaw
# from fedoo.utilities.StrainOperator import GetStrainOperator
# from fedoo.utilities.modelingspace import Variable, Vector, GetDimension
from fedoo.utilities.operator  import OpDiff

class Plate(WeakForm):
    """
    Weak formulation of the mechanical equilibrium equation for plate models.
    This weak form has to be used in combination with a Shell Constitutive Law
    like :mod:`fedoo.libConstitutiveLaw.ShellHomogeneous` or `fedoo.libConstitutiveLaw.ShellLaminate`.
    Geometrical non linearities not implemented for now.
    
    Parameters
    ----------
    PlateConstitutiveLaw: ConstitutiveLaw name (str) or ConstitutiveLaw object
        Shell Constitutive Law (:mod:`fedoo.libConstitutiveLaw`)
    name: str
        name of the WeakForm     
    """
    def __init__(self, PlateConstitutiveLaw, name = "", space=None):
        #k: shear shape factor
        
        if isinstance(PlateConstitutiveLaw, str):
            PlateConstitutiveLaw = ConstitutiveLaw.get_all()[PlateConstitutiveLaw]

        if name == "":
            name = PlateConstitutiveLaw.name
            
        WeakForm.__init__(self,name, space)
        
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
        EpsX = self.space.opdiff('DispX', 'X', 1)
        EpsY = self.space.opdiff('DispY', 'Y', 1)
        GammaXY = self.space.opdiff('DispX', 'Y', 1)+self.space.opdiff('DispY', 'X', 1)
        
        #bending curvature
        XsiX = -self.space.opdiff('RotY', 'X', 1) # flexion autour de Y -> courbure suivant x
        XsiY =  self.space.opdiff('RotX',  'Y', 1) # flexion autour de X -> courbure suivant y #ok
        XsiXY = self.space.opdiff('RotX',  'X', 1) - self.space.opdiff('RotY',  'Y', 1)
        
        #shear
        GammaXZ = self.space.opdiff('DispZ', 'X', 1) + self.space.opdiff('RotY')
        GammaYZ = self.space.opdiff('DispZ', 'Y', 1) - self.space.opdiff('RotX') 
        
        return [EpsX, EpsY, GammaXY, XsiX, XsiY, XsiXY, GammaXZ, GammaYZ]                
        
    def GetDifferentialOperator(self, localFrame):        
        H = self.__ShellConstitutiveLaw.GetShellRigidityMatrix()

        GeneralizedStrain = self.GetGeneralizedStrainOperator()                
        GeneralizedStress = [sum([0 if GeneralizedStrain[j] is 0 else GeneralizedStrain[j]*H[i][j] for j in range(8)]) for i in range(8)]
        
        DiffOp = sum([0 if GeneralizedStrain[i] is 0 else GeneralizedStrain[i].virtual*GeneralizedStress[i] for i in range(8)])
        
        #penalty for RotZ
        penalty = 1e-6
        DiffOp += self.space.opdiff('RotZ').virtual*self.space.opdiff('RotZ')*penalty
        
        return DiffOp        
      
    def GetConstitutiveLaw(self):
        return self.__ShellConstitutiveLaw

    def update(self, assembly, pb, dtime):
        pass
        # self.__ShellConstitutiveLaw.update(assembly, pb, dtime)    

class Plate_RI(Plate):

    def GetDifferentialOperator(self, localFrame):   
        #shear
        H = self._Plate__ShellConstitutiveLaw.GetShellRigidityMatrix_RI()
                
        GammaXZ = self.space.opdiff('DispZ', 'X', 1) + self.space.opdiff('RotY')
        GammaYZ = self.space.opdiff('DispZ', 'Y', 1) - self.space.opdiff('RotX') 
        
        GeneralizedStrain = [GammaXZ, GammaYZ]
        GeneralizedStress = [sum([0 if GeneralizedStrain[j] is 0 else GeneralizedStrain[j]*H[i][j] for j in range(2)]) for i in range(2)]
        
        return sum([0 if GeneralizedStrain[i] is 0 else GeneralizedStrain[i].virtual*GeneralizedStress[i] for i in range(2)])        
        
class Plate_FI(Plate):

    def GetDifferentialOperator(self, localFrame):  
        #all component but shear, for full integration
        H = self._Plate__ShellConstitutiveLaw.GetShellRigidityMatrix_FI()
        
        GeneralizedStrain = self.GetGeneralizedStrainOperator()                       
        GeneralizedStress = [sum([0 if GeneralizedStrain[j] is 0 else GeneralizedStrain[j]*H[i][j] for j in range(6)]) for i in range(6)]
        
        DiffOp = sum([GeneralizedStrain[i].virtual*GeneralizedStress[i] for i in range(6)])
        
        #penalty for RotZ
        penalty = 1e-6
        DiffOp += self.space.opdiff('RotZ').virtual*self.space.opdiff('RotZ')*penalty
        
        return DiffOp

    


    
