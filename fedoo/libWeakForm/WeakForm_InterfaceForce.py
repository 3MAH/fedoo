from fedoo.libWeakForm.WeakForm   import *
from fedoo.libConstitutiveLaw.ConstitutiveLaw import ConstitutiveLaw
from fedoo.libUtil.DispOperator import GetDispOperator
from fedoo.libUtil.Variable import Variable
from fedoo.libUtil.Dimension import ProblemDimension

class InterfaceForce(WeakForm):
    def __init__(self, CurrentConstitutiveLaw, ID = "", nlgeom = False):
        if isinstance(CurrentConstitutiveLaw, str):
            CurrentConstitutiveLaw = ConstitutiveLaw.GetAll()[CurrentConstitutiveLaw]

        if ID == "":
            ID = CurrentConstitutiveLaw.GetID()
            
        WeakForm.__init__(self,ID)
        
        Variable("DispX") 
        Variable("DispY")                
        if ProblemDimension.Get() == "3D": 
            Variable("DispZ")
            Variable.SetVector('Disp' , ('DispX', 'DispY', 'DispZ'))
        else: #2D assumed
            Variable.SetVector('Disp' , ('DispX', 'DispY'))
               
        self.__ConstitutiveLaw = CurrentConstitutiveLaw
        self.__InitialStressVector = 0
        
        if nlgeom == True:
            raise NameError('nlgeom non implemented for Interface force')
        self.__nlgeom = nlgeom

    def UpdateInitialStress(self,InitialStressVector):                                                
        self.__InitialStressVector = InitialStressVector

    def Update(self, assembly, pb, dtime):
        #function called when the problem is updated (NR loop or time increment)
        #- No nlgeom effect for now
        #- Change in constitutive law (internal variable)
        
        self.__ConstitutiveLaw.Update(assembly, pb, dtime)                           
        
        self.UpdateInitialStress(self.__ConstitutiveLaw.GetInterfaceStress())
        
        if self.__nlgeom: #need to be modifed for nlgeom
            if not(hasattr(self.__ConstitutiveLaw, 'GetCurrentGradDisp')):
                raise NameError("The actual constitutive law is not compatible with NonLinear Internal Force weak form")            
            self.__InitialGradDispTensor = self.__ConstitutiveLaw.GetCurrentGradDisp()
        

    def ResetTimeIncrement(self):
        self.__ConstitutiveLaw.ResetTimeIncrement()

    def NewTimeIncrement(self):
        self.__ConstitutiveLaw.NewTimeIncrement()

    def Reset(self):
        self.__ConstitutiveLaw.Reset()
        self.__InitialStressVector = 0

    def GetDifferentialOperator(self, mesh=None, localFrame = None):
        
        F = self.__ConstitutiveLaw.GetInterfaceStressOperator(localFrame=localFrame)            
        
        U, U_vir = GetDispOperator()
        
        dim = ProblemDimension.GetDoF()

        DiffOp = sum([U_vir[i] * F[i] for i in range(dim)])    
        
        if self.__InitialStressVector is not 0:    
            DiffOp = DiffOp + sum([0 if U_vir[i] is 0 else \
                                   U_vir[i] * self.__InitialStressVector[i] for i in range(dim)])

        return DiffOp



            
