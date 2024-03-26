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
    USE_UMFPACK = False
except:     
    USE_PYPARDISO = False
    try: 
        from scikits.umfpack import spsolve
        scipy.sparse.linalg.use_solver(assumeSortedIndicesbool = True)        
        USE_UMFPACK = True
    except:
        print('WARNING: no fast direct sparse solver has been found. Consider installing pypardiso or scikit-umfpack to improve computation performance')
        USE_UMFPACK = False

def _reload_external_solvers(config_dict):    
    if config_dict['USE_PYPARDISO']:
        from pypardiso import spsolve
    if config_dict['USE_UMFPACK']:
        from scikits.umfpack import spsolve
        scipy.sparse.linalg.use_solver(assumeSortedIndicesbool = True)
        

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
