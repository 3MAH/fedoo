from fedoo.libWeakForm.WeakForm   import *
from fedoo.libConstitutiveLaw.ConstitutiveLaw import ConstitutiveLaw
from fedoo.libUtil.DispOperator import GetDispOperator
from fedoo.libUtil.Variable import Variable
from fedoo.libUtil.Dimension import ProblemDimension

class Inertia(WeakForm):
    def __init__(self, Density, ID = ""):
           
        if ID == "":
            ID = "Inertia"
            
        WeakForm.__init__(self,ID)

        Variable("DispX") 
        Variable("DispY")                
        if ProblemDimension.Get() == "3D": Variable("DispZ")        
        
        self.__Density = Density        

    def GetDifferentialOperator(self, mesh=None, localFrame = None):
        # localFrame is not used for Inertia weak form 
        U, U_vir = GetDispOperator()
        return sum([a*b*self.__Density for (a,b) in zip(U_vir,U)])
