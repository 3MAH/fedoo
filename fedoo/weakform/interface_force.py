from fedoo.core.weakform import WeakFormBase
from fedoo.core.base import ConstitutiveLaw

class InterfaceForce(WeakFormBase):
    """
    Weak formulation of the interface equilibrium equation.
    
    * Require an interface constitutive law such as :mod:`fedoo.constitutivelaw.CohesiveLaw` or :mod:`fedoo.constitutivelaw.Spring`
    * Geometrical non linearities not implemented
    
    Parameters
    ----------
    CurrentConstitutiveLaw: ConstitutiveLaw name (str) or ConstitutiveLaw object
        Constitutive Law (:mod:`fedoo.constitutivelaw`)
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
            
        WeakFormBase.__init__(self,name)
        
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

    def updateInitialStress(self,InitialStressVector):                                                
        self.__InitialStressVector = InitialStressVector

    def update(self, assembly, pb, dtime):
        #function called when the problem is updated (NR loop or time increment)
        #- No nlgeom effect for now
        #- Change in constitutive law (internal variable)
        self.updateInitialStress(self.__ConstitutiveLaw.GetInterfaceStress())
        
        if self.__nlgeom: #need to be modifed for nlgeom
            if not(hasattr(self.__ConstitutiveLaw, 'GetCurrentGradDisp')):
                raise NameError("The actual constitutive law is not compatible with NonLinear Internal Force weak form")            
            self.__InitialGradDispTensor = self.__ConstitutiveLaw.get_disp_grad()
        

    def to_start(self):
        self.__ConstitutiveLaw.to_start()

    def NewTimeIncrement(self):
        self.__ConstitutiveLaw.NewTimeIncrement()

    def reset(self):
        self.__ConstitutiveLaw.reset()
        self.__InitialStressVector = 0

    def get_weak_equation(self, assembly, pb):
        
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



            
