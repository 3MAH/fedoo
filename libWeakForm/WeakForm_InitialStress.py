from fedoo.libWeakForm.WeakForm   import *
from fedoo.libConstitutiveLaw.ConstitutiveLaw import ConstitutiveLaw
from fedoo.libUtil.StrainOperator import GetStrainOperator, OpDiff
from fedoo.libUtil.Variable import Variable
from fedoo.libUtil.Dimension import ProblemDimension

class InitialStress(WeakForm):
    def __init__(self, InitialStressTensor = 0, ID = ""):
        if ID == "": ID = "InitialStress"
            
        WeakForm.__init__(self,ID)
        
        if InitialStressTensor == 0:
            InitialStressTensor = [0,0,0,0,0,0] #list of the six stress component (sig_xx, sig_yy, sig_zz, sig_yz, sig_xz, sigxy)
        
        Variable("DispX") 
        Variable("DispY")                
        if ProblemDimension.Get() == "3D": Variable("DispZ")
        
        if ProblemDimension.Get() == "3D":        
            GradOperator = [[OpDiff(IDvar, IDcoord,1) for IDcoord in ['X','Y','Z']] for IDvar in ['DispX','DispY','DispZ']]
            #NonLinearStrainOperatorVirtual = 0.5*(vir(duk/dxi) * duk/dxj + duk/dxi * vir(duk/dxj)) using voigt notation and with a 2 factor on non diagonal terms
            NonLinearStrainOperatorVirtual = [sum([GradOperator[k][i].virtual()*GradOperator[k][i] for k in range(3)]) for i in range(3)] 
            NonLinearStrainOperatorVirtual += [sum([GradOperator[k][1].virtual()*GradOperator[k][2] + GradOperator[k][2].virtual()*GradOperator[k][1] for k in range(3)])]
            NonLinearStrainOperatorVirtual += [sum([GradOperator[k][0].virtual()*GradOperator[k][2] + GradOperator[k][2].virtual()*GradOperator[k][0] for k in range(3)])]
            NonLinearStrainOperatorVirtual += [sum([GradOperator[k][0].virtual()*GradOperator[k][1] + GradOperator[k][1].virtual()*GradOperator[k][0] for k in range(3)])]  
        else:
            GradOperator = [[Util.OpDiff(IDvar, IDcoord,1) for IDcoord in ['X','Y']] for IDvar in ['DispX','DispY']]
            NonLinearStrainOperatorVirtual = [sum([GradOperator[k][i].virtual()*GradOperator[k][i] for k in range(2)]) for i in range(2)] + [0,0,0]            
            NonLinearStrainOperatorVirtual += [sum([GradOperator[k][0].virtual()*GradOperator[k][1] + GradOperator[k][1].virtual()*GradOperator[k][0] for k in range(2)])]  
                
        self.__NonLinearStrainOperatorVirtual = NonLinearStrainOperatorVirtual
        self.__InitialStressTensor = InitialStressTensor
        self.__typeOperator = 'all'

    def GetDifferentialOperator(self, mesh=None, localFrame = None):               
        eps, eps_vir = GetStrainOperator()        
        if self.__typeOperator == 'Matrix':
            return sum([self.__NonLinearStrainOperatorVirtual[i] * self.__InitialStressTensor[i] for i in range(6)])
        elif self.__typeOperator == 'Vector':
            return sum([eps_vir[i] * self.__InitialStressTensor[i] for i in range(6)])
        elif self.__typeOperator == 'all':
            return sum([self.__NonLinearStrainOperatorVirtual[i] * self.__InitialStressTensor[i] for i in range(6)] + [eps_vir[i] * self.__InitialStressTensor[i] for i in range(6)])
    
    def SetTypeOperator(TypeOperator):
        self.__typeOperator = TypeOperator
        

    def UpdateInitialStress(self,InitialStressTensor):
        self.__InitialStressTensor = InitialStressTensor
        
        

 