from fedoo.core.weakform import WeakFormBase
from fedoo.core.base import ConstitutiveLaw

class StressEquilibrium(WeakFormBase):
    """
    Weak formulation of the mechanical equilibrium equation for solid models (without volume force).
    
    * This weak form can be used for solid in 3D or using a 2D plane assumption (plane strain or plane stress).
    * May include initial stress depending on the ConstitutiveLaw.
    * This weak form accepts geometrical non linearities (with nlgeom = True). In this case the initial displacement is also considered. 
    * For Non-Linear Problem (material or geometrical non linearities), it is strongly recomanded to use the :mod:`fedoo.constitutivelaw.Simcoon` Constitutive Law
    
    Parameters
    ----------
    constitutivelaw: ConstitutiveLaw name (str) or ConstitutiveLaw object
        Material Constitutive Law (:mod:`fedoo.constitutivelaw`)
    name: str
        name of the WeakForm     
    nlgeom: bool (default = False)
        If True, the geometrical non linearities are activate when used in the context of NonLinearProblems 
        such as :mod:`fedoo.problem.NonLinearStatic` or :mod:`fedoo.problem.NonLinearNewmark`
    """
    def __init__(self, constitutivelaw, name = "", nlgeom = 0, space = None):
        if isinstance(constitutivelaw, str):
            constitutivelaw = ConstitutiveLaw[constitutivelaw]

        if name == "":
            name = constitutivelaw.name
            
        WeakFormBase.__init__(self,name, space)
        
        self.space.new_variable("DispX") 
        self.space.new_variable("DispY")                
        if self.space.ndim == 3: 
            self.space.new_variable("DispZ")
            self.space.new_vector('Disp' , ('DispX', 'DispY', 'DispZ'))
        else: #2D assumed
            self.space.new_vector('Disp' , ('DispX', 'DispY'))
        
        self.constitutivelaw = constitutivelaw
        self._nlgeom = nlgeom #geometric non linearities
        self.assembly_options['assume_sym'] = True     #internalForce weak form should be symmetric (if TangentMatrix is symmetric) -> need to be checked for general case
        
        if nlgeom:
            GradOperator = self.space.op_grad_u()
            if self.space.ndim == "3D":        
                #NonLinearStrainOperatorVirtual = 0.5*(vir(duk/dxi) * duk/dxj + duk/dxi * vir(duk/dxj)) using voigt notation and with a 2 factor on non diagonal terms
                NonLinearStrainOperatorVirtual = [sum([GradOperator[k][i].virtual*GradOperator[k][i] for k in range(3)]) for i in range(3)] 
                NonLinearStrainOperatorVirtual += [sum([GradOperator[k][0].virtual*GradOperator[k][1] + GradOperator[k][1].virtual*GradOperator[k][0] for k in range(3)])]  
                NonLinearStrainOperatorVirtual += [sum([GradOperator[k][0].virtual*GradOperator[k][2] + GradOperator[k][2].virtual*GradOperator[k][0] for k in range(3)])]
                NonLinearStrainOperatorVirtual += [sum([GradOperator[k][1].virtual*GradOperator[k][2] + GradOperator[k][2].virtual*GradOperator[k][1] for k in range(3)])]
            else:
                NonLinearStrainOperatorVirtual = [sum([GradOperator[k][i].virtual*GradOperator[k][i] for k in range(2)]) for i in range(2)] + [0]            
                NonLinearStrainOperatorVirtual += [sum([GradOperator[k][0].virtual*GradOperator[k][1] + GradOperator[k][1].virtual*GradOperator[k][0] for k in range(2)])] + [0,0]
            
            self.__NonLinearStrainOperatorVirtual = NonLinearStrainOperatorVirtual
            
    def get_weak_equation(self, mesh=None):
        
        if self._nlgeom == 1: #add initial displacement effect 
            if not(hasattr(self.constitutivelaw, 'get_disp_grad')):
                raise NameError("The actual constitutive law is not compatible with NonLinear Internal Force weak form")                        
            eps = self.space.op_strain(self.constitutivelaw.get_disp_grad())
            initial_stress = self.constitutivelaw.get_pk2()
        else: 
            eps = self.space.op_strain()
            initial_stress = self.constitutivelaw.get_cauchy() #required for updated lagrangian method
        
        H = self.constitutivelaw.GetH()
        sigma = [sum([0 if eps[j] is 0 else eps[j]*H[i][j] for j in range(6)]) for i in range(6)]
                
        DiffOp = sum([0 if eps[i] is 0 else eps[i].virtual * sigma[i] for i in range(6)])
        
        if initial_stress is not 0:    
            # if self._nlgeom:  #this term doesnt seem to improve convergence !
            #     DiffOp = DiffOp + sum([0 if self.__NonLinearStrainOperatorVirtual[i] is 0 else \
            #                         self.__NonLinearStrainOperatorVirtual[i] * initial_stress[i] for i in range(6)])

            DiffOp = DiffOp + sum([0 if eps[i] is 0 else \
                                    eps[i].virtual * initial_stress[i] for i in range(6)])

        return DiffOp

    def initialize(self, assembly, pb, t0 = 0.):
        self._assembly = assembly
        self._pb = pb

    def update(self, assembly, pb, dtime):
        if self._nlgeom == 2:
            # if updated lagragian method -> update the mesh and recompute elementary op
            assembly.set_disp(pb.get_disp())               
            # assembly.mesh.SetNodeCoordinates(self._crd_ini + pb.get_disp().T)
            if assembly.current.mesh in assembly._saved_change_of_basis_mat:
                del assembly._saved_change_of_basis_mat[assembly.current.mesh]
            assembly.current.compute_elementary_operators()        
                        
    def to_start(self):    
        if self._nlgeom == 2:
            # if updated lagragian method -> reset the mesh to the begining of the increment
            self._assembly.set_disp(self._pb.get_disp())               
            if self._assembly.current.mesh in self._assembly._saved_change_of_basis_mat:
                del self._assembly._saved_change_of_basis_mat[self._assembly.current.mesh]
            self._assembly.current.compute_elementary_operators()            
    
    def copy(self, new_id = ""):
        """
        Return a raw deep copy of the weak form without keeping current state (internal variable).

        Parameters
        ----------
        new_id : TYPE, optional
            The name of the created constitutive law. The default is "".

        Returns
        -------
        The copy of the weakform
        """
        new_cl = self.constitutivelaw.copy()
        
        return StressEquilibrium(new_cl, name = "", nlgeom = self.nlgeom, space = self.space)
    
    @property
    def nlgeom(self):
        return self._nlgeom
    
