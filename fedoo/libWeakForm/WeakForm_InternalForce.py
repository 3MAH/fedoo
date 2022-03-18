from fedoo.libWeakForm.WeakForm   import *
from fedoo.libConstitutiveLaw.ConstitutiveLaw import ConstitutiveLaw
from fedoo.libUtil.StrainOperator import GetStrainOperator, StrainOperator
from fedoo.libUtil.ModelingSpace import Variable, Vector, GetDimension
from fedoo.libUtil.Operator  import OpDiff

class InternalForce(WeakForm):
    """
    Weak formulation of the mechanical equilibrium equation for solid models (without volume force).
    
    * This weak form can be used for solid in 3D or using a 2D plane assumption (plane strain or plane stress).
    * May include initial stress depending on the ConstitutiveLaw.
    * This weak form accepts geometrical non linearities (with nlgeom = True). In this case the initial displacement is also considered. 
    * For Non-Linear Problem (material or geometrical non linearities), it is strongly recomanded to use the :mod:`fedoo.libConstitutiveLaw.Simcoon` Constitutive Law
    
    Parameters
    ----------
    CurrentConstitutiveLaw: ConstitutiveLaw ID (str) or ConstitutiveLaw object
        Material Constitutive Law (:mod:`fedoo.libConstitutiveLaw`)
    ID: str
        ID of the WeakForm     
    nlgeom: bool (default = False)
        If True, the geometrical non linearities are activate when used in the context of NonLinearProblems 
        such as :mod:`fedoo.libProblem.NonLinearStatic` or :mod:`fedoo.libProblem.NonLinearNewmark`
    """
    def __init__(self, CurrentConstitutiveLaw, ID = "", nlgeom = False):
        if isinstance(CurrentConstitutiveLaw, str):
            CurrentConstitutiveLaw = ConstitutiveLaw.GetAll()[CurrentConstitutiveLaw]

        if ID == "":
            ID = CurrentConstitutiveLaw.GetID()
            
        WeakForm.__init__(self,ID)
        
        Variable("DispX") 
        Variable("DispY")                
        if GetDimension() == "3D": 
            Variable("DispZ")
            Vector('Disp' , ('DispX', 'DispY', 'DispZ'))
        else: #2D assumed
            Vector('Disp' , ('DispX', 'DispY'))
        
        self.__ConstitutiveLaw = CurrentConstitutiveLaw
        self.__InitialStressTensor = 0
        self.__InitialGradDispTensor = None
        
        self.__nlgeom = nlgeom #geometric non linearities
        self.assumeSymmetric = True     #internalForce weak form should be symmetric (if TangentMatrix is symmetric) -> need to be checked for general case
        
        if nlgeom:
            if GetDimension() == "3D":        
                GradOperator = [[OpDiff(IDvar, IDcoord,1) for IDcoord in ['X','Y','Z']] for IDvar in ['DispX','DispY','DispZ']]
                #NonLinearStrainOperatorVirtual = 0.5*(vir(duk/dxi) * duk/dxj + duk/dxi * vir(duk/dxj)) using voigt notation and with a 2 factor on non diagonal terms
                NonLinearStrainOperatorVirtual = [sum([GradOperator[k][i].virtual()*GradOperator[k][i] for k in range(3)]) for i in range(3)] 
                NonLinearStrainOperatorVirtual += [sum([GradOperator[k][0].virtual()*GradOperator[k][1] + GradOperator[k][1].virtual()*GradOperator[k][0] for k in range(3)])]  
                NonLinearStrainOperatorVirtual += [sum([GradOperator[k][0].virtual()*GradOperator[k][2] + GradOperator[k][2].virtual()*GradOperator[k][0] for k in range(3)])]
                NonLinearStrainOperatorVirtual += [sum([GradOperator[k][1].virtual()*GradOperator[k][2] + GradOperator[k][2].virtual()*GradOperator[k][1] for k in range(3)])]
            else:
                GradOperator = [[OpDiff(IDvar, IDcoord,1) for IDcoord in ['X','Y']] for IDvar in ['DispX','DispY']]
                NonLinearStrainOperatorVirtual = [sum([GradOperator[k][i].virtual()*GradOperator[k][i] for k in range(2)]) for i in range(2)] + [0]            
                NonLinearStrainOperatorVirtual += [sum([GradOperator[k][0].virtual()*GradOperator[k][1] + GradOperator[k][1].virtual()*GradOperator[k][0] for k in range(2)])] + [0,0]
            
            self.__NonLinearStrainOperatorVirtual = NonLinearStrainOperatorVirtual
            
        else: self.__NonLinearStrainOperatorVirtual = 0                     
        
    def UpdateInitialStress(self,InitialStressTensor):                                                
        self.__InitialStressTensor = InitialStressTensor       

    def GetInitialStress(self):                                                
        return self.__InitialStressTensor 
        
    def Initialize(self, assembly, pb, initialTime = 0.):
        self.__ConstitutiveLaw.Initialize(assembly, pb, initialTime, self.__nlgeom)
        

    def Update(self, assembly, pb, dtime):
        self.__ConstitutiveLaw.Update(assembly, pb, dtime, self.__nlgeom)                           
        self.UpdateInitialStress(self.__ConstitutiveLaw.GetPKII())
        # self.UpdateInitialStress(self.__ConstitutiveLaw.GetKirchhoff())
        
        
        # #### DEBUG ONLY
        # # # print(self.__ConstitutiveLaw.GetPKII())
        # # # print(self.__ConstitutiveLaw.GetKirchhoff())
        # # # print(self.__ConstitutiveLaw.etot)
        # # # print(self.__ConstitutiveLaw.Detot)
        # # # print('Lt: ', self.__ConstitutiveLaw.Lt[0][1][1])
        # print('PKII: ', self.__ConstitutiveLaw.GetPKII()[1][0])
        # print('Kirch: ', self.__ConstitutiveLaw.GetKirchhoff()[1][0])
        # print('Cauchy ', self.__ConstitutiveLaw.GetCauchy()[1][0])
        
        # # print('Eps: ', self.__ConstitutiveLaw.etot[0][1]+self.__ConstitutiveLaw.Detot[0][1], 'PKII: ', self.__ConstitutiveLaw.GetPKII()[1][0])
        # crd = assembly.GetMesh().GetNodeCoordinates() + pb.GetDisp().T
        # import numpy as np
        # S = np.linalg.norm(crd[1]-crd[0]) * np.linalg.norm(crd[4]-crd[0])
        # F = assembly.GetExternalForces(pb.GetX())
        # print(np.sum(F[[0,1,4,5]], axis = 0)[1]/S)
        # # print('Surface: ', S)
        
        # # # print('GradU: ', self.__ConstitutiveLaw.GetCurrentGradDisp())
        ##### FIN DEBUG ONLY
        # self.__InitialGradDispTensor = self.__ConstitutiveLaw.GetCurrentGradDisp()
        if self.__nlgeom:
            if not(hasattr(self.__ConstitutiveLaw, 'GetCurrentGradDisp')):
                raise NameError("The actual constitutive law is not compatible with NonLinear Internal Force weak form")            
            self.__InitialGradDispTensor = self.__ConstitutiveLaw.GetCurrentGradDisp()
            


    def Reset(self):
        self.__ConstitutiveLaw.Reset()
        self.__InitialStressTensor = 0
        self.__InitialGradDispTensor = None

    def ResetTimeIncrement(self):
        self.__ConstitutiveLaw.ResetTimeIncrement()
        
        self.UpdateInitialStress(self.__ConstitutiveLaw.GetPKII())
        if self.__nlgeom:
            if not(hasattr(self.__ConstitutiveLaw, 'GetCurrentGradDisp')):
                raise NameError("The actual constitutive law is not compatible with NonLinear Internal Force weak form")            
            self.__InitialGradDispTensor = self.__ConstitutiveLaw.GetCurrentGradDisp()
        
    def NewTimeIncrement(self):
        self.__ConstitutiveLaw.NewTimeIncrement()
        #no need to update Initial Stress because the last computed stress remained unchanged

    def GetDifferentialOperator(self, mesh=None, localFrame = None):
        eps, eps_vir = GetStrainOperator(self.__InitialGradDispTensor)
        # sigma = self.__ConstitutiveLaw.GetStressOperator(localFrame=localFrame)   
        
        H = self.__ConstitutiveLaw.GetH()
        sigma = [sum([0 if eps[j] is 0 else eps[j]*H[i][j] for j in range(6)]) for i in range(6)]
                
        DiffOp = sum([eps_vir[i] * sigma[i] for i in range(6)])
        
        if self.__InitialStressTensor is not 0:    
            if self.__NonLinearStrainOperatorVirtual is not 0 :  
                DiffOp = DiffOp + sum([0 if self.__NonLinearStrainOperatorVirtual[i] is 0 else \
                                    self.__NonLinearStrainOperatorVirtual[i] * self.__InitialStressTensor[i] for i in range(6)])

            DiffOp = DiffOp + sum([0 if eps_vir[i] is 0 else \
                                    eps_vir[i] * self.__InitialStressTensor[i] for i in range(6)])

        return DiffOp

    def GetConstitutiveLaw(self):
        return self.__ConstitutiveLaw
    
    @property
    def nlgeom(self):
        return self.__nlgeom
    
    