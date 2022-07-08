from fedoo.weakform.weakform   import WeakForm
from fedoo.constitutivelaw.constitutivelaw import ConstitutiveLaw

class Inertia(WeakForm):
    """
    Weak formulation related to the inertia effect into dynamical simulation.
    Should be used in :mod:`fedoo.problem.Newmark`,  :mod:`fedoo.problem.NonLinearNewmark` or :mod:`fedoo.problem.ExplicitDynamic`
            
    Parameters
    ----------
    Density: scalar or arrays of gauss point values.
        Material density
    name: str
        name of the WeakForm 
    """
    def __init__(self, Density, name = "", space = None):
           
        if name == "":
            name = "Inertia"
            
        WeakForm.__init__(self,name,space)

        self.space.new_variable("DispX") 
        self.space.new_variable("DispY")                
        if self.space.ndim == 3: 
            self.space.new_variable("DispZ")  
            self.space.new_vector('Disp' , ('DispX', 'DispY', 'DispZ'))
        else: self.space.new_vector('Disp' , ('DispX', 'DispY'))
        
        self.__Density = Density        

    def GetDifferentialOperator(self, mesh=None, localFrame = None):
        # localFrame is not used for Inertia weak form 
        U = self.space.op_disp()
        U_vir = [u.virtual if u != 0 else 0 for u in U]
        return sum([a*b*self.__Density for (a,b) in zip(U_vir,U)])
