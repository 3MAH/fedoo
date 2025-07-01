"""Base classes for principles objects.

Should not be used, excepted to create inherited classes.
"""

from fedoo.core.modelingspace import ModelingSpace

import scipy.sparse.linalg
import scipy.sparse as sparse
import warnings

# try to find the best available direct solver
try:
    from pypardiso import spsolve

    USE_PYPARDISO = True
    USE_UMFPACK = False
    USE_PETSC = False  # only for the direct mumps solver
except ModuleNotFoundError:
    USE_PYPARDISO = False

if not USE_PYPARDISO:
    try:
        import petsc4py
        import sys

        petsc4py.init(sys.argv)
        from petsc4py import PETSc

        USE_PYPARDISO = False
        USE_UMFPACK = False
        USE_PETSC = True
    except ModuleNotFoundError:
        USE_PETSC = False

if not USE_PYPARDISO and not USE_PETSC:
    try:
        from scikits.umfpack import spsolve

        scipy.sparse.linalg.use_solver(assumeSortedIndicesbool=True)
        USE_PYPARDISO = False
        USE_UMFPACK = True
        USE_PETSC = False
    except ModuleNotFoundError:
        USE_UMFPACK = False

if not USE_PYPARDISO and not USE_PETSC and not USE_UMFPACK:
    warnings.warn(
        "WARNING: no fast direct sparse solver has been found. "
        "Consider installing pypardiso, petsc, or scikit-umfpack to improve "
        "computation performance"
    )


def _reload_external_solvers(config_dict):
    if config_dict["USE_PYPARDISO"]:
        from pypardiso import spsolve
    if config_dict["USE_PETSC"]:
        global PETSc

        import petsc4py
        import sys

        petsc4py.init(sys.argv)
        from petsc4py import PETSc
    if config_dict["USE_UMFPACK"]:
        from scikits.umfpack import spsolve

        scipy.sparse.linalg.use_solver(assumeSortedIndicesbool=True)


# =============================================================
# Base class for Mesh object
# =============================================================
class MeshBase:
    """Base class for Mesh object."""

    __dic: dict[str, "MeshBase"] = {}

    def __init__(self, name=""):
        assert isinstance(name, str), "name must be a string"
        self.__name = name

        if name != "":
            MeshBase.__dic[self.__name] = self

    def __class_getitem__(cls, item):
        """Get the mesh whose name is item."""
        return cls.__dic[item]

    @property
    def name(self):
        """Name of the mesh."""
        return self.__name

    @staticmethod
    def get_all():
        """Return a dict with all the known Mesh (with a name)."""
        return MeshBase.__dic


# =============================================================
# Base class for Assembly object
# =============================================================
class AssemblyBase:
    """Base class for Assembly object."""

    __dic: dict[str, "AssemblyBase"] = {}

    def __init__(self, name="", space=None):
        assert isinstance(name, str), "An name must be a string"
        self.__name = name

        self.global_matrix = None
        self.global_vector = None

        if not hasattr(self, "mesh"):  # in case mesh is a property
            self.mesh = None

        self.current = self
        """Assembly associated to the mesh of the deformed geometry.

        Used only for problem that required to modify the geometry (Updated
        Lagrangian formalism)
        """
        self.associated_assembly_sum = None
        """AssemblySum object that contains the assembly."""

        if name != "":
            AssemblyBase.__dic[self.__name] = self
        self.__space = space

    def __class_getitem__(cls, item):
        """Get the assembly whose name is item."""
        return cls.__dic[item]

    def get_global_matrix(self):
        """Get the last computed global matrix."""
        if self.global_matrix is None:
            self.assemble_global_mat()
        return self.global_matrix

    def get_global_vector(self):
        """Get the last computed global vector."""
        if self.global_vector is None:
            self.assemble_global_mat()
        return self.global_vector

    def assemble_global_mat(self, compute="all"):
        """Launch the assembly of global matrix."""
        # needs to be defined in inherited classes
        pass

    def delete_global_mat(self):
        """Delete Global Matrix and Global Vector related to the assembly.

        This method allow to force a new assembly
        """
        self.global_matrix = None
        self.global_vector = None

    def set_start(self, pb):
        """Begin a new time iteration."""
        pass

    def to_start(self, pb):
        """Restart the current time iteration."""
        pass

    def initialize(self, pb):
        """Initialize the assembly for the current problem."""
        pass

    def update(self, pb, compute="all"):
        """Update the assembly for the current problem state."""
        pass

    def reset(self):
        """Reset the assembly."""
        pass

    @staticmethod
    def get_all():
        """Return a dict with all the known Assembly (with a name)."""
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
        """Modeling space associated to the assembly."""
        return self.__space

    @property
    def name(self):
        """Name of the assembly if defined."""
        return self.__name


# =============================================================
# Base class for constitutive laws (cf constitutive law lib)
# =============================================================
class ConstitutiveLaw:
    """Base class for constitutive laws (cf constitutive law lib)."""

    __dic: dict[str, "ConstitutiveLaw"] = {}

    def __init__(self, name=""):
        assert isinstance(name, str), "An name must be a string"
        self.__name = name
        self.local_frame = None
        self._dimension = None
        # str or None to specify a space and associated model
        # (for instance "2Dstress" for plane stress)

        ConstitutiveLaw.__dic[self.__name] = self

    def __class_getitem__(cls, item):
        """Get the constitutive law whose name is item."""
        return cls.__dic[item]

    def reset(self):
        """Reset the constitutive law.

        This function is called when a problem is reset.
        """
        pass

    def set_start(self, assembly, pb):
        """Begin a new time iteration."""
        pass

    def to_start(self, assembly, pb):
        """Restart the current time iteration."""
        pass

    def initialize(self, assembly, pb):
        """Initialize the constitutive law for the current problem."""
        pass

    def update(self, assembly, pb):
        """Update the constitutive law for the current problem state."""
        pass

    @staticmethod
    def get_all():
        """Return a dict with all the known ConstitutiveLaw (with a name)."""
        return ConstitutiveLaw.__dic

    @property
    def name(self):
        """Name of the constitutive law."""
        return self.__name


# =============================================================
# Base class for problems (cf problems lib)
# =============================================================
from fedoo.core.boundary_conditions import ListBC


class ProblemBase:
    """
    Base class for defining Problems.

    All problem objects are derived from the ProblemBase class.
    """

    __dic = {}
    active = None  # name of the current active problem

    def __init__(self, name="", space=None):
        assert isinstance(name, str), "A name must be a string"
        self.__name = name
        self.__solver = ["direct"]
        self.bc = ListBC(
            name=self.name + "_bc"
        )  # list containing boundary contidions associated to the problem
        """Boundary conditions defined on the problem."""

        self.bc._problem = self

        if name != "":
            ProblemBase.__dic[self.__name] = self

        if space is None:
            space = ModelingSpace.get_active()
        self.__space = space

        self.set_solver()  # initialize solver properties

        self.make_active()
        self.nlgeom = False

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
        """Define the active Problem from its name."""
        if isinstance(name, ProblemBase):
            ProblemBase.active = name
        elif name in ProblemBase.__dic:
            ProblemBase.active = ProblemBase.__dic[name]
        else:
            raise NameError("{} is not a valid Problem.".format(name))

    @staticmethod
    def get_active():
        """Return the active Problem."""
        return ProblemBase.active

    def set_solver(
        self, solver: str = "direct", **kargs
    ):  # tol: float = 1e-5, precond: bool = True):
        """Define the solver for the linear system resolution.

        Parameters
        ----------
        solver: str, ufunc
            Type of solver.
            The possible choice are :
            * 'direct': direct solver. If pypardiso is installed, the
              pypardiso solver is used. Else, if petsc is installed, the mumps
              solver is used. If not, the function scipy.sparse.linalg.spsolve
              is used. If sckikit-umfpack is installed, scipy will use the
              umfpack solver which is significantly more efficient than the
              base scipy solver.
            * 'cg', 'bicg', 'bicgstab','minres','gmres', 'lgmres' or 'gcrotmk'
              using the corresponding iterative method from
              scipy.sparse.linalg. For instance, 'cg' is the conjugate
              gradient based on the function scipy.sparse.linalg.cg.
            * 'pardiso': force the use of the pypardiso solver
            * 'direct_scipy': force the use of the direct scipy solver
              (umfpack if installed)
            * 'petsc': use the petsc methods (iterative or direct). petsc4py
              should be installed.
            * function: A user spsolve function that should have the signature
              res = solver(A,B,**kargs).
              where A is a scipy sparse matrix and B a 1d numpy array.
              kargs may contains optional parameters.
        kargs: optional parameters depending on the type of solver
            precond: bool
              Use precond = False to desactivate the diagonal matrix
              preconditionning for scipy iterative methods.
              If this parametre is not given, the precoditionning is activated
              if M is not given.
            M: {sparse matrix, ndarray, LinearOperator}
              Preconditioner for A used for scipy iterative methods.
            solver_type: str
              Petsc solver. The default is 'bcgs'.
              See the petsc documentation for available solvers.
            pc_type: str
              Petsc type of preconditioner. The default is 'eisenstat'.
              See the petsc documentation for available preconditioners.
            pc_factor_mat_solver_type: str
              Petsc solver for matrix factorization when applicable.
              See the petsc documentation for details about this parameter.

        Notes
        -----
        * To change the many available petsc options, the
          petsc4py.PETSc.Options dict should be imported and modified
          (see example below).
        * To use the petsc with MPI parallelization (PCMPI), the script
          mpi4py needs to be installed, and the script should be launch in
          command line using mpiexec. For instance:

          >>> mpiexec -n 4 python petsc_examples.py -mpi_linear_solver_server -ksp_type cg -pc_type bjacobi

          A performance gain may be observed for very large problems, but
          this method should be avoid for problem with moderate size.

        Examples
        --------
          >>> # Use the scipy cg solver without preconditioner
          >>> pb.set_solver('cg', precond=False, rtol=1e-4)
          >>>
          >>> # Use the petsc cg solver with jacobi preconditioner and modify
          >>> # the rtol default parameter.
          >>> pb.set_solver('petsc', solver_type='cg', pc_type='jacobi')
          >>> from petsc4py.PETSc import Options
          >>> petsc_opt = Options()
          >>> petsc_opt['ksp_rtol'] = 1e-4
          >>>
          >>> # Use the MUMPS direct solver with petsc
          >>> pb.set_solver('petsc', solver_type='preonly', pc_type='lu', pc_factor_mat_solver_type='mumps')
        """
        return_info = False
        precond = False
        if isinstance(solver, str):
            solver = solver.lower()
            if solver == "direct":
                if USE_PYPARDISO:
                    solver_func = spsolve
                    # print(
                    #     f"Problem {self.name} : direct solver : PYPARDISO solver has been utilized"
                    # )
                elif USE_PETSC:
                    global PETSc
                    solver_func = _solver_petsc
                    kargs["solver_type"] = "preonly"
                    kargs["pc_type"] = "lu"
                    kargs["pc_factor_mat_solver_type"] = "mumps"
                    # print(
                    #     f"Problem {self.name} : direct solver : MUMPS solver from the PETSC lib "
                    # )
                else:
                    solver_func = sparse.linalg.spsolve
                    # print(
                    #     f"Problem {self.name} : direct solver : Scipy direct solver has been utilized : if SCIPY-UMFPACK is installed, it will be used"
                    # )
            elif solver in [
                "cg",
                "bicg",
                "bicgstab",
                "minres",
                "gmres",
                "lgmres",
                "gcrotmk",
            ]:  # use scipy solver
                solver_func = eval("sparse.linalg." + solver)
                return_info = True
                precond = kargs.pop("precond", True)
            elif solver == "petsc":
                global PETSc

                if "PETSc" not in dir():
                    try:
                        import sys

                        import petsc4py
                        from petsc4py import PETSc

                        petsc4py.init(sys.argv)
                    except ImportError:
                        raise ImportError(
                            'PETSc is not installed. Use "pip install mpi4py petsc petsc4py".'
                        )

                solver_func = _solver_petsc
            elif solver == "pardiso":
                if USE_PYPARDISO:
                    solver_func = spsolve
                else:
                    raise NameError(
                        'pypardiso not installed. Use "pip install pypardiso".'
                    )
            elif solver == "direct_scipy":
                solver_func = sparse.linalg.spsolve
            else:
                raise NameError("Choosen solver not available")
        else:  # assume solver is a function
            solver_func = solver

        self.__solver = [solver, solver_func, kargs, return_info, precond]

    def _solve(self, A, B):
        kargs = self.__solver[2]
        if self.__solver[3]:  # return_info = True
            precond = self.__solver[4]
            if precond and "M" not in kargs:  # precond
                Mprecond = sparse.diags(1 / A.diagonal(), 0)
                res, info = self.__solver[1](A, B, M=Mprecond, **kargs)
            else:
                res, info = self.__solver[1](A, B, **kargs)
            if info > 0:
                print(
                    f"Warning: {self.__solver[0]} solver convergence to \
                    tolerance not achieved"
                )
            return res
        else:
            return self.__solver[1](A, B, **kargs)

    @staticmethod
    def get_all():
        """Return the list of all problems."""
        return ProblemBase.__dic

    @property
    def solver(self):
        """Return the current solver used for the problem."""
        return self.__solver[1]


def _solver_petsc(
    A,
    B,
    solver_type="bcgs",
    pc_type="eisenstat",
    pc_factor_mat_solver_type=None,
    **kargs,
):
    global PETSc

    A_petsc = PETSc.Mat().createAIJWithArrays(A.shape, (A.indptr, A.indices, A.data))
    B_petsc = PETSc.Vec().createWithArray(B)
    ksp = PETSc.KSP()
    ksp.create(comm=A_petsc.getComm())
    ksp.setType(solver_type)
    pc = ksp.getPC()
    pc.setType(pc_type)
    if pc_factor_mat_solver_type is not None:
        pc.setFactorSolverType(pc_factor_mat_solver_type)

    ksp.setOperators(A_petsc)
    ksp.setFromOptions()

    res = A_petsc.createVecLeft()
    ksp.setUp()

    if isinstance(A, (sparse.csr_array, sparse.csr_matrix)):
        ksp.solve(B_petsc, res)
    elif isinstance(A, (sparse.csc_array, sparse.csc_matrix)):
        ksp.solveTranspose(B_petsc, res)
    return res
