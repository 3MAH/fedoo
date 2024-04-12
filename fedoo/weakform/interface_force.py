from fedoo.core.weakform import WeakFormBase
from fedoo.core.base import ConstitutiveLaw

class InterfaceForce(WeakFormBase):
    """
    Weak formulation of the interface equilibrium equation.
    
    * Require an interface constitutive law such as :mod:`fedoo.constitutivelaw.CohesiveLaw` or :mod:`fedoo.constitutivelaw.Spring`
    * Geometrical non linearities not implemented
    
    Parameters
    ----------
    constitutivelaw: str or ConstitutiveLaw
        Interface constitutive law (ConstitutiveLaw object or name)
        (:mod:`fedoo.constitutivelaw`)
    name: str, optional
        name of the WeakForm     
    nlgeom: bool (default = False)
        For future development
        If True, return a NotImplemented Error
    """
    def __init__(self, constitutivelaw, name = "", nlgeom = False,  space = None):
        if isinstance(constitutivelaw, str):
            constitutivelaw = ConstitutiveLaw[constitutivelaw]
            
        WeakFormBase.__init__(self,name)
        
        self.space.new_variable("DispX") 
        self.space.new_variable("DispY")                
        if self.space.ndim == 3: 
            self.space.new_variable("DispZ")
            self.space.new_vector('Disp' , ('DispX', 'DispY', 'DispZ'))
        else: #2D assumed
            self.space.new_vector('Disp' , ('DispX', 'DispY'))
               
        self.constitutivelaw = constitutivelaw
        self.__InitialStressVector = 0
        
        if nlgeom == True:
            raise NameError('nlgeom non implemented for Interface force')
        self.__nlgeom = nlgeom
        
        self.assembly_options['assume_sym'] = False #symetric ?

    def update(self, assembly, pb):
        #function called when the problem is updated (NR loop or time increment)
        #- No nlgeom effect for now
        #- Change in constitutive law (internal variable)
        
        if self.__nlgeom: #need to be modifed for nlgeom
            if not(hasattr(self.constitutivelaw, 'GetCurrentGradDisp')):
                raise NameError("The actual constitutive law is not compatible with NonLinear Internal Force weak form")            
            self.__InitialGradDispTensor = self.constitutivelaw.get_disp_grad()
        

    # def to_start(self, assembly, pb):
    #     pass

    # def set_start(self, assembly, pb):
    #     pass

    # def reset(self):
    #     pass

    def get_weak_equation(self, assembly, pb):
        
        ### Operator for Interface Stress Operator ###
        dim = self.space.ndim
        K = assembly.sv['TangentMatrix']
        
        U = self.space.op_disp() #relative displacement if used with cohesive element
        U_vir = [u.virtual for u in U]
        F = [sum([U[j]*K[i][j] for j in range(dim)]) for i in range(dim)] #Interface stress operator
        
        DiffOp = sum([0 if U[i]==0 else U[i].virtual * F[i] for i in range(dim)])    
        
        initial_stress = assembly.sv['InterfaceStress']
        
        if initial_stress is not 0:    
            DiffOp = DiffOp + sum([0 if U_vir[i] is 0 else \
                                   U_vir[i] * initial_stress[i] for i in range(dim)])

        return DiffOp



            
