from fedoo.libWeakForm.WeakForm import WeakForm
from fedoo.libConstitutiveLaw.ConstitutiveLaw import ConstitutiveLaw

class InternalForce(WeakForm):
    """
    Weak formulation of the mechanical equilibrium equation for solid models (without volume force).
    
    * This weak form can be used for solid in 3D or using a 2D plane assumption (plane strain or plane stress).
    * May include initial stress depending on the ConstitutiveLaw.
    * This weak form accepts geometrical non linearities (with nlgeom = True). In this case the initial displacement is also considered. 
    * For Non-Linear Problem (material or geometrical non linearities), it is strongly recomanded to use the :mod:`fedoo.libConstitutiveLaw.Simcoon` Constitutive Law
    
    Parameters
    ----------
    CurrentConstitutiveLaw: ConstitutiveLaw name (str) or ConstitutiveLaw object
        Material Constitutive Law (:mod:`fedoo.libConstitutiveLaw`)
    name: str
        name of the WeakForm     
    nlgeom: bool (default = False)
        If True, the geometrical non linearities are activate when used in the context of NonLinearProblems 
        such as :mod:`fedoo.libProblem.NonLinearStatic` or :mod:`fedoo.libProblem.NonLinearNewmark`
    """
    def __init__(self, CurrentConstitutiveLaw, name = "", nlgeom = 0, space = None):
        if isinstance(CurrentConstitutiveLaw, str):
            CurrentConstitutiveLaw = ConstitutiveLaw.get_all()[CurrentConstitutiveLaw]

        if name == "":
            name = CurrentConstitutiveLaw.name
            
        WeakForm.__init__(self,name, space)
        
        self.space.new_variable("DispX") 
        self.space.new_variable("DispY")                
        if self.space.ndim == 3: 
            self.space.new_variable("DispZ")
            self.space.new_vector('Disp' , ('DispX', 'DispY', 'DispZ'))
        else: #2D assumed
            self.space.new_vector('Disp' , ('DispX', 'DispY'))
        
        self.__ConstitutiveLaw = CurrentConstitutiveLaw
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
            
    def GetDifferentialOperator(self, mesh=None, localFrame = None):
        
        if self._nlgeom == 1: #add initial displacement effect 
            if not(hasattr(self.__ConstitutiveLaw, 'GetCurrentGradDisp')):
                raise NameError("The actual constitutive law is not compatible with NonLinear Internal Force weak form")                        
            eps = self.space.op_strain(self.__ConstitutiveLaw.GetCurrentGradDisp())
            initial_stress = self.__ConstitutiveLaw.GetPKII()
        else: 
            eps = self.space.op_strain()
            initial_stress = self.__ConstitutiveLaw.GetCauchy() #required for updated lagrangian method
        
        H = self.__ConstitutiveLaw.GetH()
        sigma = [sum([0 if eps[j] is 0 else eps[j]*H[i][j] for j in range(6)]) for i in range(6)]
                
        DiffOp = sum([0 if eps[i] is 0 else eps[i].virtual * sigma[i] for i in range(6)])
        
        if initial_stress is not 0:    
            # if self._nlgeom:  #this term doesnt seem to improve convergence !
            #     DiffOp = DiffOp + sum([0 if self.__NonLinearStrainOperatorVirtual[i] is 0 else \
            #                         self.__NonLinearStrainOperatorVirtual[i] * initial_stress[i] for i in range(6)])

            DiffOp = DiffOp + sum([0 if eps[i] is 0 else \
                                    eps[i].virtual * initial_stress[i] for i in range(6)])

        return DiffOp

    def initialize(self, assembly, pb, initialTime = 0.):
        self._assembly = assembly
        self._pb = pb

    def update(self, assembly, pb, dtime):
        if self._nlgeom == 2:
            # if updated lagragian method -> update the mesh and recompute elementary op
            assembly.set_disp(pb.GetDisp())               
            # assembly.mesh.SetNodeCoordinates(self._crd_ini + pb.GetDisp().T)
            if assembly.current.mesh in assembly._saved_change_of_basis_mat:
                del assembly._saved_change_of_basis_mat[assembly.current.mesh]
            assembly.current.compute_elementary_operators()        
                        
    def to_start(self):    
        if self._nlgeom == 2:
            # if updated lagragian method -> reset the mesh to the begining of the increment
            self._assembly.set_disp(self._pb.GetDisp())               
            if self._assembly.current.mesh in self._assembly._saved_change_of_basis_mat:
                del self._assembly._saved_change_of_basis_mat[self._assembly.current.mesh]
            self._assembly.current.compute_elementary_operators()            

    def GetConstitutiveLaw(self):
        return self.__ConstitutiveLaw
    
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
        new_cl = self.__ConstitutiveLaw.copy()
        
        return InternalForce(new_cl, name = "", nlgeom = self.nlgeom, space = self.space)
    
    @property
    def nlgeom(self):
        return self._nlgeom
    





# class InternalForce2(WeakForm):
#     """
#     Weak formulation of the mechanical equilibrium equation for solid models (without volume force).
    
#     * This weak form can be used for solid in 3D or using a 2D plane assumption (plane strain or plane stress).
#     * May include initial stress depending on the ConstitutiveLaw.
#     * This weak form accepts geometrical non linearities (with nlgeom = True). In this case the initial displacement is also considered. 
#     * For Non-Linear Problem (material or geometrical non linearities), it is strongly recomanded to use the :mod:`fedoo.libConstitutiveLaw.Simcoon` Constitutive Law
    
#     Parameters
#     ----------
#     CurrentConstitutiveLaw: ConstitutiveLaw name (str) or ConstitutiveLaw object
#         Material Constitutive Law (:mod:`fedoo.libConstitutiveLaw`)
#     name: str
#         name of the WeakForm     
#     nlgeom: bool (default = False)
#         If True, the geometrical non linearities are activate when used in the context of NonLinearProblems 
#         such as :mod:`fedoo.libProblem.NonLinearStatic` or :mod:`fedoo.libProblem.NonLinearNewmark`
#     """
#     def __init__(self, CurrentConstitutiveLaw, name = "", nlgeom = False, space = None):
#         if isinstance(CurrentConstitutiveLaw, str):
#             CurrentConstitutiveLaw = ConstitutiveLaw.get_all()[CurrentConstitutiveLaw]

#         if name == "":
#             name = CurrentConstitutiveLaw.name
            
#         WeakForm.__init__(self,name, space)
        
#         self.space.new_variable("DispX") 
#         self.space.new_variable("DispY")                
#         if self.space.ndim == 3: 
#             self.space.new_variable("DispZ")
#             self.space.new_vector('Disp' , ('DispX', 'DispY', 'DispZ'))
#         else: #2D assumed
#             self.space.new_vector('Disp' , ('DispX', 'DispY'))
        
#         self.__ConstitutiveLaw = CurrentConstitutiveLaw
#         self.__InitialStressTensor = 0
#         self.__InitialGradDispTensor = None
#         self.__nlgeom = nlgeom #geometric non linearities
#         self.assembly_options['assume_sym'] = True     #internalForce weak form should be symmetric (if TangentMatrix is symmetric) -> need to be checked for general case
        
#         if nlgeom:
#             GradOperator = self.space.op_grad_u()
#             if self.space.ndim == "3D":        
#                 #NonLinearStrainOperatorVirtual = 0.5*(vir(duk/dxi) * duk/dxj + duk/dxi * vir(duk/dxj)) using voigt notation and with a 2 factor on non diagonal terms
#                 NonLinearStrainOperatorVirtual = [sum([GradOperator[k][i].virtual*GradOperator[k][i] for k in range(3)]) for i in range(3)] 
#                 NonLinearStrainOperatorVirtual += [sum([GradOperator[k][0].virtual*GradOperator[k][1] + GradOperator[k][1].virtual*GradOperator[k][0] for k in range(3)])]  
#                 NonLinearStrainOperatorVirtual += [sum([GradOperator[k][0].virtual*GradOperator[k][2] + GradOperator[k][2].virtual*GradOperator[k][0] for k in range(3)])]
#                 NonLinearStrainOperatorVirtual += [sum([GradOperator[k][1].virtual*GradOperator[k][2] + GradOperator[k][2].virtual*GradOperator[k][1] for k in range(3)])]
#             else:
#                 NonLinearStrainOperatorVirtual = [sum([GradOperator[k][i].virtual*GradOperator[k][i] for k in range(2)]) for i in range(2)] + [0]            
#                 NonLinearStrainOperatorVirtual += [sum([GradOperator[k][0].virtual*GradOperator[k][1] + GradOperator[k][1].virtual*GradOperator[k][0] for k in range(2)])] + [0,0]
            
#             self.__NonLinearStrainOperatorVirtual = NonLinearStrainOperatorVirtual
            
#         else: self.__NonLinearStrainOperatorVirtual = 0                     
        
#     def updateInitialStress(self,InitialStressTensor):                                                
#         self.__InitialStressTensor = InitialStressTensor

#     def GetInitialStress(self):                                                
#         return self.__InitialStressTensor 

#     def update(self, assembly, pb, dtime):
#         self.updateInitialStress(self.__ConstitutiveLaw.GetPKII())
                        
#         if self.__nlgeom:
#             if not(hasattr(self.__ConstitutiveLaw, 'GetCurrentGradDisp')):
#                 raise NameError("The actual constitutive law is not compatible with NonLinear Internal Force weak form")            
#             self.__InitialGradDispTensor = self.__ConstitutiveLaw.GetCurrentGradDisp()
            


#     def reset(self):
#         self.__InitialStressTensor = 0
#         self.__InitialGradDispTensor = None

#     def to_start(self):       
#         self.updateInitialStress(self.__ConstitutiveLaw.GetPKII())
#         if self.__nlgeom:
#             if not(hasattr(self.__ConstitutiveLaw, 'GetCurrentGradDisp')):
#                 raise NameError("The actual constitutive law is not compatible with NonLinear Internal Force weak form")            
#             self.__InitialGradDispTensor = self.__ConstitutiveLaw.GetCurrentGradDisp()
        
#     def NewTimeIncrement(self):
#         pass
#         # no need to update Initial Stress because the last computed stress remained unchanged

#     def GetDifferentialOperator(self, mesh=None, localFrame = None):
        
#         eps = self.space.op_strain(self.__InitialGradDispTensor)
   
#         H = self.__ConstitutiveLaw.GetH()
#         sigma = [sum([0 if eps[j] is 0 else eps[j]*H[i][j] for j in range(6)]) for i in range(6)]
                
#         DiffOp = sum([0 if eps[i] is 0 else eps[i].virtual * sigma[i] for i in range(6)])
                
#         if self.__InitialStressTensor is not 0:    
#             if self.__nlgeom:  
#                 DiffOp = DiffOp + sum([0 if self.__NonLinearStrainOperatorVirtual[i] is 0 else \
#                                     self.__NonLinearStrainOperatorVirtual[i] * self.__InitialStressTensor[i] for i in range(6)])

#             DiffOp = DiffOp + sum([0 if eps[i] is 0 else \
#                                     eps[i].virtual * self.__InitialStressTensor[i] for i in range(6)])

#         return DiffOp

#     def GetConstitutiveLaw(self):
#         return self.__ConstitutiveLaw
    
#     def copy(self, new_id = ""):
#         """
#         Return a raw deep copy of the weak form without keeping current state (internal variable).

#         Parameters
#         ----------
#         new_id : TYPE, optional
#             The name of the created constitutive law. The default is "".

#         Returns
#         -------
#         The copy of the weakform
#         """
#         new_cl = self.__ConstitutiveLaw.copy()
        
#         return InternalForce(new_cl, name = "", nlgeom = self.nlgeom, space = self.space)
    
#     @property
#     def nlgeom(self):
#         return self.__nlgeom    