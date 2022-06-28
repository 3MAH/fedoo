from fedoo.libWeakForm.WeakForm   import *
from fedoo.libConstitutiveLaw.ConstitutiveLaw import ConstitutiveLaw

class InterfaceForce(WeakForm):
    """
    Weak formulation of the interface equilibrium equation.
    
    * Require an interface constitutive law such as :mod:`fedoo.libConstitutiveLaw.CohesiveLaw` or :mod:`fedoo.libConstitutiveLaw.Spring`
    * Geometrical non linearities not implemented
    
    Parameters
    ----------
    CurrentConstitutiveLaw: ConstitutiveLaw name (str) or ConstitutiveLaw object
        Constitutive Law (:mod:`fedoo.libConstitutiveLaw`)
    name: str
        name of the WeakForm     
    nlgeom: bool (default = False)
        For future development
        If True, return a NotImplemented Error
    """
    def __init__(self, CurrentConstitutiveLaw, name = "", nlgeom = False):
        if isinstance(CurrentConstitutiveLaw, str):
            CurrentConstitutiveLaw = ConstitutiveLaw.get_all()[CurrentConstitutiveLaw]

        if name == "":
            name = CurrentConstitutiveLaw.name
            
        WeakForm.__init__(self,name)
        
        self.space.new_variable("DispX") 
        self.space.new_variable("DispY")                
        if self.space.ndim == 3: 
            self.space.new_variable("DispZ")
            self.space.new_vector('Disp' , ('DispX', 'DispY', 'DispZ'))
        else: #2D assumed
            self.space.new_vector('Disp' , ('DispX', 'DispY'))
               
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
        
        ### Operator for Interface Stress Operator ###
        dim = self.space.ndim
        K = self.__ConstitutiveLaw.GetK()
        
        U = self.space.op_disp() #relative displacement if used with cohesive element
        U_vir = [u.virtual for u in U]
        F = [sum([U[j]*K[i][j] for j in range(dim)]) for i in range(dim)] #Interface stress operator
        
        DiffOp = sum([0 if U[i]==0 else U[i].virtual * F[i] for i in range(dim)])    
        
        if self.__InitialStressVector is not 0:    
            DiffOp = DiffOp + sum([0 if U_vir[i] is 0 else \
                                   U_vir[i] * self.__InitialStressVector[i] for i in range(dim)])

        return DiffOp



            
