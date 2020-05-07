from fedoo.libWeakForm.WeakForm   import *
from fedoo.libConstitutiveLaw.ConstitutiveLaw import ConstitutiveLaw
from fedoo.libUtil.StrainOperator import GetStrainOperator, StrainOperator
from fedoo.libUtil.Variable import Variable
from fedoo.libUtil.Dimension import ProblemDimension
from fedoo.libUtil.Operator  import OpDiff

class InternalForce(WeakForm):
    def __init__(self, CurrentConstitutiveLaw, ID = "", nlgeom = False):
        if isinstance(CurrentConstitutiveLaw, str):
            CurrentConstitutiveLaw = ConstitutiveLaw.GetAll()[CurrentConstitutiveLaw]

        if ID == "":
            ID = CurrentConstitutiveLaw.GetID()
            
        WeakForm.__init__(self,ID)
        
        Variable("DispX") 
        Variable("DispY")                
        if ProblemDimension.Get() == "3D": Variable("DispZ")
        
        self.__ConstitutiveLaw = CurrentConstitutiveLaw
        self.__InitialStressTensor = 0
        self.__InitialGradDispTensor = None
        
        self.__nlgeom = nlgeom #geometric non linearities
                
        if nlgeom:
            if ProblemDimension.Get() == "3D":        
                GradOperator = [[OpDiff(IDvar, IDcoord,1) for IDcoord in ['X','Y','Z']] for IDvar in ['DispX','DispY','DispZ']]
                #NonLinearStrainOperatorVirtual = 0.5*(vir(duk/dxi) * duk/dxj + duk/dxi * vir(duk/dxj)) using voigt notation and with a 2 factor on non diagonal terms
                NonLinearStrainOperatorVirtual = [sum([GradOperator[k][i].virtual()*GradOperator[k][i] for k in range(3)]) for i in range(3)] 
                NonLinearStrainOperatorVirtual += [sum([GradOperator[k][1].virtual()*GradOperator[k][2] + GradOperator[k][2].virtual()*GradOperator[k][1] for k in range(3)])]
                NonLinearStrainOperatorVirtual += [sum([GradOperator[k][0].virtual()*GradOperator[k][2] + GradOperator[k][2].virtual()*GradOperator[k][0] for k in range(3)])]
                NonLinearStrainOperatorVirtual += [sum([GradOperator[k][0].virtual()*GradOperator[k][1] + GradOperator[k][1].virtual()*GradOperator[k][0] for k in range(3)])]  
            else:
                GradOperator = [[OpDiff(IDvar, IDcoord,1) for IDcoord in ['X','Y']] for IDvar in ['DispX','DispY']]
                NonLinearStrainOperatorVirtual = [sum([GradOperator[k][i].virtual()*GradOperator[k][i] for k in range(2)]) for i in range(2)] + [0,0,0]            
                NonLinearStrainOperatorVirtual += [sum([GradOperator[k][0].virtual()*GradOperator[k][1] + GradOperator[k][1].virtual()*GradOperator[k][0] for k in range(2)])]  
            
            self.__NonLinearStrainOperatorVirtual = NonLinearStrainOperatorVirtual
        
        else: self.__NonLinearStrainOperatorVirtual = 0
                     
        
    def UpdateInitialStress(self,InitialStressTensor):                                                
        self.__InitialStressTensor = InitialStressTensor
        

    def Update(self, assembly, pb, time):
        displacement = pb.GetDisp()
        if displacement is 0: TotalStress = TotalStrain = 0
        else:
            TotalStrain = assembly.GetStrainTensor(displacement, "GaussPoint", nlgeom=self.__nlgeom)
            TotalStress = self.__ConstitutiveLaw.GetStress(TotalStrain, time)     

        self.UpdateInitialStress(TotalStress)
        
        if self.__nlgeom:
            if displacement is 0: self.__InitialGradDispTensor = 0
            else: self.__InitialGradDispTensor = assembly.GetGradTensor(displacement, "GaussPoint")
        
        return TotalStrain, TotalStress

    def Reset(self):
        self.__ConstitutiveLaw.Reset()
        self.__InitialStressTensor = 0
        self.__InitialGradDispTensor = None

    def ResetTimeIncrement(self):
        self.__ConstitutiveLaw.ResetTimeIncrement()

    def NewTimeIncrement(self):
        self.__ConstitutiveLaw.NewTimeIncrement()

    def GetDifferentialOperator(self, mesh=None, localFrame = None):
        
        sigma = self.__ConstitutiveLaw.GetStressOperator(localFrame=localFrame)   
#        try:
#            sigma = self.__ConstitutiveLaw.GetStressOperator(localFrame)            
#        except:
#            assert 0, "Warning, you put a non mechanical consitutive law in the InternalForce"      
        
        eps, eps_vir = GetStrainOperator(self.__InitialGradDispTensor)
        
        DiffOp = sum([eps_vir[i] * sigma[i] for i in range(6)])
        
        if self.__InitialStressTensor is not 0:    
            if self.__NonLinearStrainOperatorVirtual is not 0 :     
                DiffOp = DiffOp + sum([0 if self.__NonLinearStrainOperatorVirtual[i] is 0 else \
                                   self.__NonLinearStrainOperatorVirtual[i] * self.__InitialStressTensor[i] for i in range(6)])
            
            DiffOp = DiffOp + sum([0 if eps_vir[i] is 0 else \
                                   eps_vir[i] * self.__InitialStressTensor[i] for i in range(6)])

        return DiffOp
    
    @property
    def  nlgeom(self):
        return self.__nlgeom        