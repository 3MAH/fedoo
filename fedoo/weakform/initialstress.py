from fedoo.core.weakform import WeakFormBase
from fedoo.core.base import ConstitutiveLaw
# from fedoo.util.StrainOperator import GetStrainOperator, DiffOp
# from fedoo.core.modelingspace import Variable, Vector, get_Dimension

class InitialStress(WeakFormBase):
    def __init__(self, InitialStressTensor = 0, name = "", space = None):
        if name == "": name = "InitialStress"
            
        WeakFormBase.__init__(self,name, space)
        
        if InitialStressTensor == 0:
            InitialStressTensor = [0,0,0,0,0,0] #list of the six stress component (sig_xx, sig_yy, sig_zz, sig_yz, sig_xz, sigxy)
        
        self.space.variable("DispX") 
        self.space.variable("DispY")                
        if self.space.ndim == 3: 
            self.space.variable("DispZ")
            self.space.vector('Disp' , ('DispX', 'DispY', 'DispZ'))
        else: #2D assumed
            self.space.vector('Disp' , ('DispX', 'DispY'))
        
        if self.space.ndim:        
            GradOperator = [[DiffOp(namevar, namecoord,1) for namecoord in ['X','Y','Z']] for namevar in ['DispX','DispY','DispZ']]
            #NonLinearStrainOperatorVirtual = 0.5*(vir(duk/dxi) * duk/dxj + duk/dxi * vir(duk/dxj)) using voigt notation and with a 2 factor on non diagonal terms
            NonLinearStrainOperatorVirtual = [sum([GradOperator[k][i].virtual*GradOperator[k][i] for k in range(3)]) for i in range(3)] 
            NonLinearStrainOperatorVirtual += [sum([GradOperator[k][1].virtual*GradOperator[k][2] + GradOperator[k][2].virtual*GradOperator[k][1] for k in range(3)])]
            NonLinearStrainOperatorVirtual += [sum([GradOperator[k][0].virtual*GradOperator[k][2] + GradOperator[k][2].virtual*GradOperator[k][0] for k in range(3)])]
            NonLinearStrainOperatorVirtual += [sum([GradOperator[k][0].virtual*GradOperator[k][1] + GradOperator[k][1].virtual*GradOperator[k][0] for k in range(3)])]  
        else:
            GradOperator = [[Util.DiffOp(namevar, namecoord,1) for namecoord in ['X','Y']] for namevar in ['DispX','DispY']]
            NonLinearStrainOperatorVirtual = [sum([GradOperator[k][i].virtual*GradOperator[k][i] for k in range(2)]) for i in range(2)] + [0,0,0]            
            NonLinearStrainOperatorVirtual += [sum([GradOperator[k][0].virtual*GradOperator[k][1] + GradOperator[k][1].virtual*GradOperator[k][0] for k in range(2)])]  
                
        self.__NonLinearStrainOperatorVirtual = NonLinearStrainOperatorVirtual
        self.__InitialStressTensor = InitialStressTensor
        self.__typeOperator = 'all'

    def get_weak_equation(self, assembly, pb):               
        eps = self.space.op_strain()
        if self.__typeOperator == 'Matrix':
            return sum([self.__NonLinearStrainOperatorVirtual[i] * self.__InitialStressTensor[i] for i in range(6)])
        elif self.__typeOperator == 'Vector':
            return sum([eps[i].virtual * self.__InitialStressTensor[i] for i in range(6)])
        elif self.__typeOperator == 'all':
            return sum([self.__NonLinearStrainOperatorVirtual[i] * self.__InitialStressTensor[i] for i in range(6)] + [eps[i].virtual * self.__InitialStressTensor[i] for i in range(6)])
    
    def SetTypeOperator(self, TypeOperator):
        self.__typeOperator = TypeOperator
        

    def updateInitialStress(self,InitialStressTensor):
        self.__InitialStressTensor = InitialStressTensor
        
        

 