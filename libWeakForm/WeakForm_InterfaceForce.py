from fedoo.libWeakForm.WeakForm   import *
from fedoo.libConstitutiveLaw.ConstitutiveLaw import ConstitutiveLaw
from fedoo.libUtil.DispOperator import GetDispOperator
from fedoo.libUtil.Variable import Variable
from fedoo.libUtil.Dimension import ProblemDimension

class InterfaceForce(WeakForm):
    def __init__(self, CurrentConstitutiveLaw, ID = ""):
        if isinstance(CurrentConstitutiveLaw, str):
            CurrentConstitutiveLaw = ConstitutiveLaw.GetAll()[CurrentConstitutiveLaw]

        if ID == "":
            ID = CurrentConstitutiveLaw.GetID()
            
        WeakForm.__init__(self,ID)
        
        Variable("DispX") 
        Variable("DispY")                
        if ProblemDimension.Get() == "3D": Variable("DispZ")
               
        self.__ConstitutiveLaw = CurrentConstitutiveLaw
        self.__InitialStressVector = 0

    def UpdateInitialStress(self,InitialStressVector):                                                
        self.__InitialStressVector = InitialStressVector

    def Update(self, assembly, pb, time):
        #function called when the problem is updated (NR loop or time increment)
        #- No nlgeom effect for now
        #- Change in constitutive law (internal variable)
        
        displacement = pb.GetDisp()
        if displacement is 0: InterfaceStress = Delta = 0
        else:
            OpDelta = self.__ConstitutiveLaw.GetOperartorDelta() #Delta is the relative displacement
            Delta = [assembly.GetGaussPointResult(op, displacement) for op in OpDelta]
            InterfaceStress = self.__ConstitutiveLaw.GetInterfaceStress(Delta, time)     
        
        self.UpdateInitialStress(InterfaceStress)
        
        return Delta, InterfaceStress

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

        DiffOp = sum([U_vir[i] * F[i] for i in range(3)])    
        
        if self.__InitialStressVector is not 0:    
            DiffOp = DiffOp + sum([0 if U_vir[i] is 0 else \
                                   U_vir[i] * self.__InitialStressVector[i] for i in range(3)])

        return DiffOp



            
