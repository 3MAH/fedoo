"""Base classes for principles objects.
Should not be used, excepted to create inherited classes.
"""
from copy import deepcopy
from fedoo.core.modelingspace import ModelingSpace

import numpy as np
import scipy.sparse.linalg
import scipy.sparse as sparse
try: 
    from pypardiso import spsolve
    USE_PYPARDISO = True
except:     
    USE_PYPARDISO = False
    try: 
        from scikits import umfpack
    except:
        print('WARNING: no fast direct sparse solver has been found. Consider installing pypardiso or scikit-umfpack to improve computation performance')


#=============================================================
# Base class for Mesh object
#=============================================================
class MeshBase:
    """Base class for Mesh object."""

    __dic = {}

    def __init__(self, name = ""):
        assert isinstance(name, str) , "name must be a string" 
        self.__name = name
        
        if name != "":
            MeshBase.__dic[self.__name] = self

    def __class_getitem__(cls, item):
        return cls.__dic[item]

    @property
    def name(self):
        return self.__name

    @staticmethod
    def get_all():
        return MeshBase.__dic
    

#=============================================================
# Base class for Assembly object
#=============================================================
class AssemblyBase:
    """Base class for Assembly object."""

    __dic = {}

    def __init__(self, name = "", space=None):
        assert isinstance(name, str) , "An name must be a string" 
        self.__name = name

        self.global_matrix = None
        self.global_vector = None
                
        if not hasattr(self, 'mesh'): #in case mesh is a property
            self.mesh = None 
            
        self.current = self
        
        if name != "": AssemblyBase.__dic[self.__name] = self
        self.__space = space

    def __class_getitem__(cls, item):
        return cls.__dic[item]

    def get_global_matrix(self):
        if self.global_matrix is None: self.assemble_global_mat()        
        return self.global_matrix

    def get_global_vector(self):
        if self.global_vector is None: self.assemble_global_mat()        
        return self.global_vector
        
    def assemble_global_mat(self, compute = 'all'):
        #needs to be defined in inherited classes
        pass

    def delete_global_mat(self):
        """
        Delete Global Matrix and Global Vector related to the assembly. 
        This method allow to force a new assembly
        """
        self.global_matrix = None
        self.global_vector = None
    
    def set_start(self, pb):
        pass
    
    def to_start(self, pb):
        pass
    
    def initialize(self, pb):
        pass
    
    def update(self, pb, compute = 'all'):
        pass
    
    def reset(self):
        pass    
    
    @staticmethod
    def get_all():
        return AssemblyBase.__dic
    
    # @staticmethod
    # def Launch(name):
    #     """
    #     Assemble the global matrix and global vector of the assembly name
    #     name is a str
    #     """
    #     AssemblyBase.get_all()[name].assemble_global_mat()    
    
    @property
    def space(self):
        return self.__space
   
    @property
    def name(self):
        return self.__name


#=============================================================
# Base class for constitutive laws (cf constitutive law lib)
#=============================================================
class ConstitutiveLaw:
    """Base class for constitutive laws (cf constitutive law lib)."""

    __dic = {}

    def __init__(self, name = ""):
        assert isinstance(name, str) , "An name must be a string" 
        self.__name = name
        self.local_frame = None
        self._dimension = None #str or None to specify a space and associated model (for instance "2Dstress" for plane stress)

        ConstitutiveLaw.__dic[self.__name] = self        


    def __class_getitem__(cls, item):
        return cls.__dic[item]
    
    
    def reset(self): 
        #function called to restart a problem (reset all internal variables)
        pass

    
    def set_start(self, assembly, pb):  
        #function called when the time is increased. Not used for elastic laws
        pass

    
    def to_start(self, assembly, pb):
        #function called if the time step is reinitialized. Not used for elastic laws
        pass


    def initialize(self, assembly, pb):
        #function called to initialize the constutive law 
        pass

    
    def update(self,assembly, pb):
        #function called to update the state of constitutive law 
        pass

    
    def copy(self, new_id = ""):
        """
        Return a raw copy of the constitutive law without keeping current internal variables.

        Parameters
        ----------
        new_id : TYPE, optional
            The name of the created constitutive law. The default is "".

        Returns
        -------
        The copy of the constitutive law
        """
        new_cl = deepcopy(self)        
        new_cl._ConstitutiveLaw__name = new_id
        self.__dic[new_id] = new_cl
        new_cl.reset()
        return new_cl

    
    @staticmethod
    def get_all():
        return ConstitutiveLaw.__dic
    
    
    @property
    def name(self):
        return self.__name


#=============================================================
# Base class for problems (cf problems lib)
#=============================================================   
from fedoo.core.boundary_conditions import ListBC

class ProblemBase:
    """
    Base class for defining Problems.
    
    All problem objects are derived from the ProblemBase class.
    """

    __dic = {}
    active = None #name of the current active problem


    def __init__(self, name = "", space = None):
        assert isinstance(name, str) , "A name must be a string" 
        self.__name = name
        self.__solver = ['direct']
        self.bc = ListBC(name=self.name+"_bc") #list containing boundary contidions associated to the problem        
        """Boundary conditions defined on the problem."""

        self.bc._problem = self
        
        ProblemBase.__dic[self.__name] = self
        
        if space is None: 
            space = ModelingSpace.get_active()
        self.__space = space
        
        self.set_solver() #initialize solver properties
        
        self.make_active()


    def __class_getitem__(cls, item):
        return cls.__dic[item]


    @property
    def name(self):
        """Return the name of the Problem."""
        return self.__name
    
    
    @property
    def space(self):
        """Return the ModelingSpace associated to the Problem if defined."""
        return self.__space
    
    
    def make_active(self):
        """Define the problem instance as the active Problem."""
        ProblemBase.active = self
        
    
    @staticmethod
    def set_active(name):
        """
        Static method.
        Define the active Problem from its name.
        """
        if isinstance(name, ProblemBase): ProblemBase.active = name
        elif name in ProblemBase.__dic: ProblemBase.active = ProblemBase.__dic[name]
        else: raise NameError("{} is not a valid Problem.".format(name))
    
    
    @staticmethod
    def get_active():
        """
        Static method.
        Return the active Problem.
        """
        return ProblemBase.active
        
    
    def set_solver(self, solver: str='direct', precond: bool=True, **kargs): #tol: float = 1e-5, precond: bool = True):
        """Define the solver for the linear system resolution.
        
        Parameters
        ---------------
        solver: str
            Type of solver.        
            The possible choice are : 
            * 'direct': direct solver. If pypardiso is installed, the pypardiso solver is used.
                If not, the function scipy.sparse.linalg.spsolve is used instead. 
                If sckikit-umfpack is installed, scipy will use the umfpack solver which is significantly
                more efficient than the base scipy solver.
            * 'cg', 'bicg', 'bicgstab','minres','gmres', 'lgmres' or 'gcrotmk' using the corresponding
                iterative method from scipy.sparse.linalg. For instance, 'cg' is the conjugate gradient based on 
                the function scipy.sparse.linalg.cg 
            * 'pardiso': force the use of the pypardiso solver
            * 'direct_scipy': force the use of the direct scipy solver (umfpack if installed)
            * function: 
                A user spsolve function that should have the signature res = solver(A,B,**kargs)
                where A is a scipy sparse matrix and B a 1d numpy array. kargs may contains
                optional parameters.
        precond: bool, default = True
            use precond = False to desactivate the diagonal matrix preconditionning. 
            Only used for iterative method. 
        """
        return_info = False
        if isinstance(solver, str):
            solver = solver.lower()
            if solver == 'direct':
                if USE_PYPARDISO:
                    solver_func = spsolve
                else:
                    solver_func = sparse.linalg.spsolve                          
            elif solver in ['cg', 'bicg', 'bicgstab','minres','gmres', 'lgmres', 'gcrotmk']: #use scipy solver
                solver_func = eval('sparse.linalg.'+solver)
                return_info = True
            elif solver == 'pardiso':
                if USE_PYPARDISO:
                    solver_func = spsolve
                else:
                    raise NameError('pypardiso not installed. Use "pip install pypardiso".')
            elif solver == 'direct_scipy':
                solver_func = sparse.linalg.spsolve
            else: 
                raise NameError('Choosen solver not available')
        else: #assume solver is a function
            solver_func = solver
                        
        self.__solver = [solver, solver_func, kargs, return_info, precond]
        
        
    def _solve(self, A, B):
        kargs = self.__solver[2]
        if self.__solver[3]: #return_info = True
            if self.__solver[4] and 'M' not in kargs: #precond
                Mprecond = sparse.diags(1/A.diagonal(), 0)
                res, info = self.__solver[1](A,B, M=Mprecond, **kargs) 
            else:
                res, info = self.__solver[1](A,B, **kargs) 
            if info > 0: print(f'Warning: {self.__solver[0]} solver convergence to tolerance not achieved') 
            return res
        else:            
            return self.__solver[1](A,B, **kargs)
        
        
    @staticmethod
    def get_all():
        """Return the list of all problems."""
        return ProblemBase.__dic
           
    @property
    def solver(self):
        """Return the current solver used for the problem."""
        return self.__solver[1]
    
    
    
    
    
    
    
    
    
    
    # ### Functions that may be defined depending on the type of problem
    # def get_disp(self,name='all'):
    #      raise NameError("The method 'get_Disp' is not defined for this kind of problem")

    # def get_rot(self,name='all'):
    #      raise NameError("The method 'GetRot' is not defined for this kind of problem")

    # def get_temp(self):
    #      raise NameError("The method 'GetTemp' is not defined for this kind of problem")
    
    # def update(self,):
    #     raise NameError("The method 'Update' is not defined for this kind of problem")    
        
    # def ChangeAssembly(self,Assembling):
    #     raise NameError("The method 'ChangeAssembly' is not defined for this kind of problem")    
        
    # def SetNewtonRaphsonErrorCriterion(self,ErrorCriterion, tol=5e-3, max_subiter = 5, err0 = None):
    #     """
    #     Set the error criterion used for the newton raphson algorithm

    #     Parameters
    #     ----------
    #     ErrorCriterion : str in ['Displacement', 'Force', 'Work']             
    #         Set the type of error criterion.             
    #     tol : float
    #         Tolerance of the NewtonRaphson algorithm (default = 5e-3)
    #     max_subiter: int
    #         Number of newton raphson iteration before returning an error
    #     err0 : scalar
    #         Reference value of error used for normalization
    #     """
    #     raise NameError("The method 'SetNewtonRaphsonErrorCriterion' is not defined for this kind of problem")    
        
    # def NewtonRaphsonError(self):
    #     raise NameError("The method 'NewtonRaphsonError' is not defined for this kind of problem")        
        
    # def NewTimeIncrement(self,LoadFactor):
    #     raise NameError("The method 'NewTimeIncrement' is not defined for this kind of problem")    
            
    # def NewtonRaphsonIncr(self):                   
    #     raise NameError("The method 'NewtonRaphsonIncr' is not defined for this kind of problem")            
        
    # def to_start(self):
    #     raise NameError("The method 'to_start' is not defined for this kind of problem")    
    
    # def resetLoadFactor(self):             
    #     raise NameError("The method 'resetLoadFactor' is not defined for this kind of problem")    
    
    # def reset(self):
    #     raise NameError("The method 'reset' is not defined for this kind of problem")    
        
    # def GetElasticEnergy(self):
    #     raise NameError("The method 'GetElasticEnergy' is not defined for this kind of problem")    
    
    # def GetNodalElasticEnergy(self):
    #     raise NameError("The method 'GetNodalElasticEnergy' is not defined for this kind of problem")    
        
    # def get_ext_forces(self, name = 'all'):
    #     raise NameError("The method 'get_ext_forces' is not defined for this kind of problem")    

    # def add_output(self, filename, assemblyname, output_list, output_type='Node', file_format ='vtk', position = 'top'):
    #     raise NameError("The method 'add_output' is not defined for this kind of problem")    
        
    # def save_results(self, iterOutput=None):        
    #     raise NameError("The method 'save_results' is not defined for this kind of problem")
    
    # def get_results(self, assemb, output_list, output_type='Node', position = 1, res_format = None):
    #     raise NameError("The method 'get_results' is not defined for this kind of problem")        

    # #defined in the ProblemPGD classes
    # def get_X(self): raise NameError("Method only defined for PGD Problems") 
    # def get_Xbc(self): raise NameError("Method only defined for PGD Problems") 
    # def ComputeResidualNorm(self,err_0=None): raise NameError("Method only defined for PGD Problems") 
    # def GetResidual(self): raise NameError("Method only defined for PGD Problems") 
    # def updatePGD(self,termToChange, ddcalc='all'): raise NameError("Method only defined for PGD Problems") 
    # def updateAlpha(self): raise NameError("Method only defined for PGD Problems") 
    # def AddNewTerm(self,numberOfTerm = 1, value = None, variable = 'all'): raise NameError("Method only defined for PGD Problems") 
    

    




    
    
    
    
    
    
    
    
# =============================================================================
# Functions that call methods of ProblemBase for the current active problem
# =============================================================================

# def get_all():
#     return ProblemBase.get_all()
# def get_Active():
#     return ProblemBase.get_active()
# def set_Active(Problemname):
#     ProblemBase.set_Active(Problemname)



# def SetSolver(solver, tol=1e-5, precond=True):
#     ProblemBase.get_active().SetSolver(solver,tol,precond)


# # ## Functions related to boundary contidions
# def BoundaryCondition(bc_type,Var,Value,Index,Constant = None, timeEvolution=None, initialValue = None, name = "No name", Problemname = None):
#     if Problemname is None: problem = ProblemBase.get_active()
#     else: problem = ProblemBase.get_all()[Problemname]
#     problem.BoundaryCondition(bc_type,Var,Value,Index,Constant, timeEvolution, initialValue, name)

# def get_BC(): return ProblemBase.get_active().bc    
# def RemoveBC(name =None): ProblemBase.get_active().RemoveBC(name)    
# def PrintBC(): ProblemBase.get_active().PrintBC()    
 



# ### Functions that may be defined depending on the type of problem
# def get_disp(name='Disp'): return ProblemBase.get_active().get_disp(name)
# def get_rot(name='all'): return ProblemBase.get_active().get_rot(name)
# def get_temp(): return ProblemBase.get_active().get_temp()
# def update(**kargs): return ProblemBase.get_active().update(**kargs) 
# def ChangeAssembly(Assembling): ProblemBase.get_active().ChangeAssembly(Assembling)
# def SetNewtonRaphsonErrorCriterion(ErrorCriterion, tol=5e-3, max_subiter = 5, err0 = None): ProblemBase.get_active().SetNewtonRaphsonErrorCriterion(ErrorCriterion, tol, max_subiter, err0)
# def NewtonRaphsonError(): return ProblemBase.get_active().NewtonRaphsonError()
# def NewTimeIncrement(LoadFactor): ProblemBase.get_active().NewTimeIncrement(LoadFactor)
# def NewtonRaphsonIncr(): ProblemBase.get_active().NewtonRaphsonIncr()
# def to_start(): ProblemBase.get_active().to_start()
# def reset(): ProblemBase.get_active().reset()
# def resetLoadFactor(): ProblemBase.get_active().resetLoadFactor()
# def NLSolve(**kargs): return ProblemBase.get_active().nlsolve(**kargs)  
# def add_output(filename, assemblyname, output_list, output_type='Node', file_format ='vtk', position = 'top'):
#     return ProblemBase.get_active().add_output(filename, assemblyname, output_list, output_type, file_format, position)
# def save_results(iterOutput=None):        
#     return ProblemBase.get_active().save_results(iterOutput)
# def get_results(assemb, output_list, output_type='Node', position = 1, res_format = None):
#     return ProblemBase.get_active().get_results(assemb, output_list, output_type, position, res_format)



# #functions that should be define in the Problem and in the ProblemPGD classes
# def set_A(A): ProblemBase.get_active().set_A(A)
# def get_A(): return ProblemBase.get_active().get_A()
# def get_B(): return ProblemBase.get_active().get_B()
# def get_D(): return ProblemBase.get_active().get_D()
# def GetMesh(): return ProblemBase.get_active().mesh
# def set_D(D): ProblemBase.get_active().set_D(D)
# def set_B(B): ProblemBase.get_active().set_B(B)
# def Solve(**kargs): ProblemBase.get_active().solve(**kargs)
# def get_X(): return ProblemBase.get_active().get_X()
# def apply_boundary_conditions(): ProblemBase.get_active().apply_boundary_conditions()
# def get_dof_solution(name='all'): return ProblemBase.get_active().get_dof_solution(name)
# def set_DoFSolution(name,value): ProblemBase.get_active().set_DoFSolution(name,value)
# def SetInitialBCToCurrent(): ProblemBase.get_active().SetInitialBCToCurrent()
# def get_global_vectorComponent(vector, name='all'): return ProblemBase.get_active()._get_vect_component(vector, name)

# #functions only defined for Newmark problem 
# def get_Xdot():
#     return ProblemBase.get_active().get_Xdot()

# def get_Xdotdot():
#     return ProblemBase.get_active().get_Xdotdot()

  
# def GetVelocity():
#     return ProblemBase.get_active().GetVelocity()

# def get_Acceleration():
#     return ProblemBase.get_active().get_Acceleration()


# def SetInitialDisplacement(name,value):
#     """
#     name is the name of the associated variable (generaly 'DispX', 'DispY' or 'DispZ')    
#     value is an array containing the initial displacement of each nodes
#     """
#     ProblemBase.get_active().SetInitialDisplacement(name,value)          

# def SetInitialVelocity(name,value):
#     """
#     name is the name of the associated variable (generaly 'DispX', 'DispY' or 'DispZ')    
#     value is an array containing the initial velocity of each nodes        
#     """
#     ProblemBase.get_active().SetInitialVelocity(name,value)          
      

# def SetInitialAcceleration(name,value):
#     """
#     name is the name of the associated variable (generaly 'DispX', 'DispY' or 'DispZ')    
#     value is an array containing the initial acceleration of each nodes        
#     """
#     ProblemBase.get_active().SetInitialAcceleration(name,value)           
    

# def SetRayleighDamping(alpha, beta):        
#     """
#     Compute the damping matrix from the Rayleigh's model:
#     [C] = alpha*[M] + beta*[K]         

#     where [C] is the damping matrix, [M] is the mass matrix and [K] is the stiffness matrix        
#     Note: The rayleigh model with alpha = 0 and beta = Viscosity/YoungModulus is almost equivalent to the multi-axial Kelvin-Voigt model
    
#     Warning: the damping matrix is not automatically updated when mass and stiffness matrix are modified.        
#     """
#     ProblemBase.get_active().SetRayleighDamping(alpha, beta)

# def initialize(t0 = 0.):
#     ProblemBase.get_active().initialize(t0)           

# def GetElasticEnergy():
#     """
#     returns : sum(0.5 * U.transposed * K * U)
#     """
#     return ProblemBase.get_active().GetElasticEnergy()
    
# def GetNodalElasticEnergy():
#     """
#     returns : 0.5 * U.transposed * K * U
#     """
#     return ProblemBase.get_active().GetNodalElasticEnergy()

# def get_ext_forces(name='all'):
#     return ProblemBase.get_active().get_ext_forces(name)


# def GetKineticEnergy():
#     """
#     returns : 0.5 * Udot.transposed * M * Udot
#     """
#     return ProblemBase.get_active().GetKineticEnergy()

# def get_DampingPower():
#     """
#     returns : Udot.transposed * C * Udot
#     The damping disspated energy can be approximated by:
#             Edis = cumtrapz(DampingPower * TimeStep)
#     """        
#     return ProblemBase.get_active().get_DampingPower()

# def updateStiffness(StiffnessAssembling):
#     ProblemBase.get_active().updateStiffness(StiffnessAssembling)




# #functions only used define in the ProblemPGD subclasses
# def get_Xbc(): return ProblemBase.get_active().get_Xbc() 
# def ComputeResidualNorm(err_0=None): return ProblemBase.get_active().ComputeResidualNorm(err_0)
# def GetResidual(): return ProblemBase.get_active().GetResidual()
# def updatePGD(termToChange, ddcalc='all'): return ProblemBase.get_active().updatePGD(termToChange, ddcalc)
# def updateAlpha(): return ProblemBase.get_active().updateAlpha()
# def AddNewTerm(numberOfTerm = 1, value = None, variable = 'all'): return ProblemBase.get_active().AddNewTerm(numberOfTerm, value, variable)
