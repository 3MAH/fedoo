from fedoo.libWeakForm.WeakForm   import *
from fedoo.libConstitutiveLaw.ConstitutiveLaw import ConstitutiveLaw
from fedoo.libUtil.StrainOperator import GetStrainOperator
from fedoo.libUtil.ModelingSpace import Variable, Vector, GetDimension
from fedoo.libUtil.Operator  import OpDiff

class Plate(WeakForm):
    def __init__(self, PlateConstitutiveLaw, ID = ""):
        #k: shear shape factor
        
        assert GetDimension() == '3D', "No 2D model for a plate kinematic. Choose '3D' problem dimension."

        if isinstance(PlateConstitutiveLaw, str):
            PlateConstitutiveLaw = ConstitutiveLaw.GetAll()[PlateConstitutiveLaw]

        if ID == "":
            ID = PlateConstitutiveLaw.GetID()
            
        WeakForm.__init__(self,ID)

        Variable("DispX") 
        Variable("DispY")            
        Variable("DispZ")   
        Variable("RotX") #torsion rotation 
        Variable("RotY")   
        Variable("RotZ")
        Vector('Disp' , ('DispX', 'DispY', 'DispZ'))
        Vector('Rot' , ('RotX', 'RotY', 'RotZ'))     
        
        self.__ShellConstitutiveLaw = PlateConstitutiveLaw
        
    def GetGeneralizedStrainOperator(self):
        #membrane strain
        EpsX = OpDiff('DispX', 'X', 1)
        EpsY = OpDiff('DispY', 'Y', 1)
        GammaXY = OpDiff('DispX', 'Y', 1)+OpDiff('DispY', 'X', 1)
        
        #bending curvature
        XsiX = -OpDiff('RotY', 'X', 1) # flexion autour de Y -> courbure suivant x
        XsiY =  OpDiff('RotX',  'Y', 1) # flexion autour de X -> courbure suivant y #ok
        XsiXY = OpDiff('RotX',  'X', 1) - OpDiff('RotY',  'Y', 1)
        
        #shear
        GammaXZ = OpDiff('DispZ', 'X', 1) + OpDiff('RotY')
        GammaYZ = OpDiff('DispZ', 'Y', 1) - OpDiff('RotX') 
        
        return [EpsX, EpsY, GammaXY, XsiX, XsiY, XsiXY, GammaXZ, GammaYZ]                
        
    def GetDifferentialOperator(self, localFrame):        
        H = self.__ShellConstitutiveLaw.GetShellRigidityMatrix()

        GeneralizedStrain = self.GetGeneralizedStrainOperator()                
        GeneralizedStress = [sum([0 if GeneralizedStrain[j] is 0 else GeneralizedStrain[j]*H[i][j] for j in range(8)]) for i in range(8)]
        
        DiffOp = sum([0 if GeneralizedStrain[i] is 0 else GeneralizedStrain[i].virtual()*GeneralizedStress[i] for i in range(8)])
        
        #penalty for RotZ
        penalty = 1e-6
        DiffOp += OpDiff('RotZ').virtual()*OpDiff('RotZ')*penalty
        
        return DiffOp        
      
    def GetConstitutiveLaw(self):
        return self.__ShellConstitutiveLaw

    def Update(self, assembly, pb, dtime):
        self.__ShellConstitutiveLaw.Update(assembly, pb, dtime, nlgeom = False)    

class Plate_RI(Plate):

    def GetDifferentialOperator(self, localFrame):   
        #shear
        H = self._Plate__ShellConstitutiveLaw.GetShellRigidityMatrix_RI()
                
        GammaXZ = OpDiff('DispZ', 'X', 1) + OpDiff('RotY')
        GammaYZ = OpDiff('DispZ', 'Y', 1) - OpDiff('RotX') 
        
        GeneralizedStrain = [GammaXZ, GammaYZ]
        GeneralizedStress = [sum([0 if GeneralizedStrain[j] is 0 else GeneralizedStrain[j]*H[i][j] for j in range(2)]) for i in range(2)]
        
        return sum([0 if GeneralizedStrain[i] is 0 else GeneralizedStrain[i].virtual()*GeneralizedStress[i] for i in range(2)])        
        
class Plate_FI(Plate):

    def GetDifferentialOperator(self, localFrame):  
        #all component but shear, for full integration
        H = self._Plate__ShellConstitutiveLaw.GetShellRigidityMatrix_FI()
        
        GeneralizedStrain = self.GetGeneralizedStrainOperator()                       
        GeneralizedStress = [sum([0 if GeneralizedStrain[j] is 0 else GeneralizedStrain[j]*H[i][j] for j in range(6)]) for i in range(6)]
        
        DiffOp = sum([GeneralizedStrain[i].virtual()*GeneralizedStress[i] for i in range(6)])
        
        #penalty for RotZ
        penalty = 1e-6
        DiffOp += OpDiff('RotZ').virtual()*OpDiff('RotZ')*penalty
        
        return DiffOp

    


    
