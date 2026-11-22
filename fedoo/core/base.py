"""Base classes for principles objects.

Should not be used, excepted to create inherited classes.
"""

import importlib.util
import os
import sys


from fedoo.core.modelingspace import ModelingSpace
from fedoo.util.localframe import (
    _normalize_location,
    as_local_frame,
    local_frame_to_gausspoints,
)

import scipy.sparse.linalg
import scipy.sparse as sparse
import numpy as np
import warnings
from weakref import WeakSet


class InvalidKinematicStateError(RuntimeError):
    """A trial displacement state degenerates the kinematics.

    Raised when a Newton iterate (or a line-search trial state) produces
    det(F) <= 0 at some Gauss points: the constitutive evaluation is
    meaningless there and assembling would fill the system with NaN.
    RECOVERABLE in the NonLinear solver: its NR loop and line search treat
    it as a failed iterate/increment (step backtracking or time-step
    reduction). Other drivers (explicit dynamics, Linear solve_history)
    have no rollback mechanism and let it propagate -- there it is a clear,
    early abort instead of the silent NaN degradation it replaces.
    """

    pass


# Number of threads of the simcoon batch calls (umat, objective_rate, Lt_convert,
# rotate_voigt): 0 uses all the cores. Set it with fedoo.get_config()["N_THREADS"] = n.
N_THREADS = 0

# try to find the best available direct solver
try:
    from pypardiso import spsolve

    USE_PYPARDISO = True
    USE_MUMPS = False
    USE_UMFPACK = False
    USE_PETSC = False  # only for the direct mumps solver
except ModuleNotFoundError:
    USE_PYPARDISO = False

if not USE_PYPARDISO:
    try:
        import mumps

        USE_MUMPS = True
        USE_UMFPACK = False
        USE_PETSC = False
    except ModuleNotFoundError:
        USE_MUMPS = False

# petsc4py is optional, installed by the user: only found here, PETSc itself
# is initialized at its first use, by _petsc()
HAS_PETSC = importlib.util.find_spec("petsc4py") is not None
PETSc = None

if not USE_PYPARDISO and not USE_MUMPS:
    USE_PETSC = HAS_PETSC
    if USE_PETSC:
        USE_UMFPACK = False

if not USE_PYPARDISO and not USE_MUMPS and not USE_PETSC:
    try:
        from scikits.umfpack import spsolve

        scipy.sparse.linalg.use_solver(assumeSortedIndicesbool=True)
        USE_UMFPACK = True
    except ModuleNotFoundError:
        USE_UMFPACK = False

if not USE_PYPARDISO and not USE_MUMPS and not USE_PETSC and not USE_UMFPACK:
    warnings.warn(
        "WARNING: no fast direct sparse solver has been found. "
        "Consider installing pypardiso (Intel) or python-mumps (any platform) "
        "to improve computation performance"
    )


def _petsc():
    """Return the petsc4py.PETSc module, initializing PETSc at the first call.

    PETSc is initialized with the command line, so that the PETSc options
    given to the script are used (e.g. ``-ksp_type cg -pc_type gamg``,
    ``-ksp_view``, ``-mpi_linear_solver_server``). The arguments that are not
    PETSc options (those of the script itself) are not reported at exit.
    If petsc4py.PETSc has already been imported by the user, its
    initialization is kept.
    """
    global PETSc

    if PETSc is None:
        try:
            import petsc4py
        except ImportError:
            raise ImportError(
                "petsc4py is not installed. Install it with conda "
                '("conda install -c conda-forge petsc4py") or pip ("pip '
                'install petsc petsc4py", built from sources).'
            ) from None
        if "petsc4py.PETSc" not in sys.modules:
            argv = sys.argv if sys.argv and sys.argv[0] else ["fedoo"]
            defaults = ["-options_left", "no"]
            if sys.platform == "darwin":
                # the System V shared memory of macOS (4 MB by default) is
                # too small for the linear solver server
                defaults += ["-mpi_linear_solver_server_use_shared_memory", "false"]
            # options given on the command line come after, and win
            petsc4py.init([argv[0], *defaults, *argv[1:]])
        from petsc4py import PETSc as _PETSc

        PETSc = _PETSc
    return PETSc


# With -mpi_linear_solver_server, the ranks other than 0 serve the solves of
# rank 0 from the PETSc initialization on: initialize before anything else.
if HAS_PETSC and "-mpi_linear_solver_server" in sys.argv:
    _petsc()


def _reload_external_solvers(config_dict):
    if config_dict["USE_PYPARDISO"]:
        from pypardiso import spsolve
    if config_dict["USE_MUMPS"]:
        import mumps  # noqa: F401
    if config_dict["USE_PETSC"]:
        _petsc()
    if config_dict["USE_UMFPACK"]:
        from scikits.umfpack import spsolve

        scipy.sparse.linalg.use_solver(assumeSortedIndicesbool=True)


# =============================================================
# Base class for Mesh object
# =============================================================
class MeshBase:
    """Base class for Mesh object."""

    __dic: dict[str, "MeshBase"] = {}

    def __init__(self, name="", register=True):
        assert isinstance(name, str), "name must be a string"
        self.__name = name

        if register and name != "":
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
        self._global_dof_registered_problems = None

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

    def iter_leaf(self):
        """Iterate over the leaf assemblies contained in this object."""
        yield self

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

    def register_global_dofs(self, pb):
        """Register global DOFs required by the assembly on a problem.

        Subclasses should implement :meth:`_register_global_dofs` instead of
        overriding this method. Registration is performed once per problem.
        """
        if type(self)._register_global_dofs is AssemblyBase._register_global_dofs:
            return
        if self._global_dof_registered_problems is None:
            self._global_dof_registered_problems = WeakSet()
        elif pb in self._global_dof_registered_problems:
            return
        self._register_global_dofs(pb)
        self._global_dof_registered_problems.add(pb)

    def _register_global_dofs(self, pb):
        """Implement assembly-specific global-DOF registration if required."""
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
def _corate_on_law_message(law):
    return (
        f"{type(law).__name__}.corate is ignored: the objective rate belongs "
        "to the weak form. Set it with wf.corate = ... on the "
        "StressEquilibrium weak form (used with nlgeom='UL'). Setting corate "
        "on a constitutive law is deprecated and will raise an AttributeError "
        "in a future release."
    )


class ConstitutiveLaw:
    """Base class for constitutive laws (cf constitutive law lib)."""

    __dic: dict[str, "ConstitutiveLaw"] = {}

    def __init__(self, name=""):
        assert isinstance(name, str), "An name must be a string"
        self.__name = name
        self.local_frame = None
        self._local_frame_location = None
        self.is_initialized = False
        """Tag set to True once the law is intialized.
        
        Only used when relevant.
        """
        self._dimension = None
        # str or None to specify a space and associated model
        # (for instance "2Dstress" for plane stress)

        ConstitutiveLaw.__dic[self.__name] = self

    @property
    def corate(self):
        """Deprecated, ignored: the objective rate is ``weakform.corate``.

        The objective rate of the updated-Lagrangian formulation is chosen on
        the weak form (:attr:`fedoo.weakform.StressEquilibrium.corate`) and
        passed to the law at each call. A value set on a constitutive law has
        never been read: setting it emits a :class:`DeprecationWarning`, and
        will raise an :class:`AttributeError` in a future release.

        :meta private:
        """
        if "_ignored_corate" not in self.__dict__:
            raise AttributeError(
                f"{type(self).__name__} has no 'corate' attribute: the "
                "objective rate belongs to the weak form (wf.corate)."
            )
        warnings.warn(_corate_on_law_message(self), DeprecationWarning, stacklevel=2)
        return self._ignored_corate

    @corate.setter
    def corate(self, value):
        warnings.warn(_corate_on_law_message(self), DeprecationWarning, stacklevel=2)
        self._ignored_corate = value

    def __class_getitem__(cls, item):
        """Get the constitutive law whose name is item."""
        return cls.__dic[item]

    def reset(self):
        """Reset the constitutive law.

        This function is called when a problem is reset.
        """
        self.is_initialized = False  # tag set to True once the law is intialized

    def set_start(self, assembly, pb):
        """Begin a new time iteration."""
        pass

    def to_start(self, assembly, pb):
        """Restart the current time iteration."""
        pass

    def initialize(self, assembly, pb):
        """Initialize the constitutive law for the current problem."""
        self.is_initialized = True  # tag set to True once the law is intialized

    def update(self, assembly, pb):
        """Update the constitutive law for the current problem state."""
        pass

    def set_local_frame(self, local_frame, location=None):
        """Define the material coordinate frame.

        Parameters
        ----------
        local_frame : array_like or scipy/simcoon Rotation
            One rotation, or rotations stored at nodes, elements, or Gauss
            points. Matrix arrays must use their last two axes for 3x3
            rotation matrices.
        location : {'Node', 'Element', 'GaussPoint'}, optional
            Storage location. By default it is inferred from the number of
            frames when an assembly is available.

        Returns
        -------
        self
            The constitutive law, for fluent material construction.
        """
        if local_frame is None:
            self.local_frame = None
            self._local_frame_location = None
            return self

        # Validate immediately while retaining multidimensional storage, which
        # can carry explicit (element, Gauss point, 3, 3) ordering information.
        self.local_frame = as_local_frame(local_frame, dimension=3)
        self._local_frame_location = _normalize_location(location)
        return self

    def get_local_frame(self, assembly=None):
        """Return material frames, resolved at ``assembly`` Gauss points."""
        if self.local_frame is None:
            return None

        if assembly is None:
            return as_local_frame(self.local_frame, dimension=3)
        return local_frame_to_gausspoints(
            self.local_frame, assembly, self._local_frame_location
        )

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
        self._factor_context = None
        self._factor_valid = False
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

        self._global_dof = _GlobalDof()

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

    def add_global_dof(
        self,
        variable_names: str | list[str],
        nb_new_dof: int = 1,
        vector_name: str | None = None,
    ):
        """Add global degrees of freedom to the problem.

        The global dof are global in the sense that they are not associated
        to a node of a mesh as usual finite element dof. They are usefull to
        define some constraints, like periodic boundary conditions.

        Parameters
        ----------
        variable_names: str, list[str]
            Name of the global variable (ie not associated to nodes).
            Name is required to catch the results using the get_results or
            get_ext_forces method.
        nb_new_dof: int, default=1
            Number of new global dof for each new variable
        vector_name: str, optional
            If a vector name is given, add a vector that allow to catch
            all the associated variables at the same time.

        Note
        -----
        If the variables already exists, some dof are added to the variables
        In this case, the vector_name argument is ignored. A vector should be defined
        when adding new variables only.
        """
        return self._global_dof.add(variable_names, nb_new_dof, vector_name)

    @property
    def global_dof(self):
        return self._global_dof

    @property
    def n_global_dof(self):
        return self._global_dof.n_dof

    def set_solver(
        self, solver: str = "direct", symmetric: bool | None = None, **kargs
    ):  # tol: float = 1e-5, precond: bool = True):
        """Define the solver for the linear system resolution.

        Parameters
        ----------
        solver: str, ufunc
            Type of solver.
            The possible choice are :
            * 'direct': direct solver. If pypardiso is installed, the
              pypardiso solver is used. Else, if python-mumps is installed,
              the standalone MUMPS solver is used. Else, if petsc is installed,
              the mumps solver from petsc is used. If not, the function
              scipy.sparse.linalg.spsolve is used. If scikit-umfpack is
              installed, scipy will use the umfpack solver which is
              significantly more efficient than the base scipy solver.
            * 'cg', 'bicg', 'bicgstab','minres','gmres', 'lgmres' or 'gcrotmk'
              using the corresponding iterative method from
              scipy.sparse.linalg. For instance, 'cg' is the conjugate
              gradient based on the function scipy.sparse.linalg.cg.
            * 'pardiso': force the use of the pypardiso solver
            * 'mumps': force the use of the standalone MUMPS solver
              (python-mumps must be installed). This is the recommended direct
              solver on arm64 platforms where pypardiso is not available.
              Unless ``OMP_NUM_THREADS`` is set, fedoo gives MUMPS at most
              8 threads, and at most the performance cores on Apple chips.
              Its analysis (ordering, symbolic factorization) is reused
              across the solves of a problem while the sparsity pattern of
              the matrix does not change.
            * 'direct_scipy': force the use of the direct scipy solver
              (umfpack if installed)
            * 'petsc': use the petsc methods (iterative or direct). petsc4py
              is not a dependency of fedoo and should be installed by the
              user (e.g. "conda install -c conda-forge petsc4py"). The PETSc
              solver is kept across the solves of the problem: the symbolic
              factorization (direct solvers) is reused while the sparsity
              pattern of the matrix does not change.
            * function: A user spsolve function that should have the signature
              res = solver(A,B,**kargs).
              where A is a scipy sparse matrix and B a 1d numpy array.
              kargs may contains optional parameters.
        symmetric: bool, optional
            Use a symmetric-indefinite direct factorization (LDL^T) when
            supported. It handles the indefinite matrices of Lagrange
            multipliers and mixed formulations, at about half the work of LU.

            * None (default): automatic with the 'direct' solver of
              python-mumps. LDL^T is used when every assembly of the problem
              is assembled with ``assume_sym=True`` (the default of the weak
              forms whose tangent matrix is symmetric), LU otherwise, e.g.
              as soon as a weak form declares a nonsymmetric tangent or a
              constraint adds its own assembly. The choice is made again at
              each solve. The other solvers keep the general path.
            * True: always. Only one triangle of the matrix is read: with a
              nonsymmetric tangent forced to ``assume_sym=True``, the
              diagonal blocks are not symmetrized by the assembly and the
              factorization then solves with one triangle of them.
            * False: never (LU).

            Pardiso uses ``mtype=-2`` and standalone MUMPS uses ``sym=2``
            (python-mumps 0.0.6: the symmetry is given with the matrix).
            The 'direct' solver through PETSc uses a MUMPS Cholesky
            (symmetric-indefinite) factorization. SciPy/UMFPACK, 'petsc' and
            user-supplied solvers keep their usual general path. The caller
            is responsible for ensuring that the reduced system matrix is
            symmetric.
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
            reuse_preconditioner: bool
              Petsc only. Keep the preconditioner built for the first matrix
              (e.g. a GAMG hierarchy) for the next solves of the problem,
              the Krylov iterations making up for the difference. The
              default is False.
            rtol, atol, max_it: float, float, int
              Petsc only. Tolerances of the Krylov solver.
            near_null_space: bool
              Petsc only. With the GAMG preconditioner, give it the rigid
              body modes of the problem (near null space), needed for an
              efficient multigrid on mechanical problems. The default is
              True.

        Notes
        -----
        * To change the many available petsc options, the
          petsc4py.PETSc.Options dict should be imported and modified
          before set_solver (see example below), or the options given on the
          command line of the script (e.g. ``python script.py -ksp_type cg
          -pc_type gamg -ksp_converged_reason``).
        * To use the petsc with MPI parallelization (PCMPI), the script
          mpi4py needs to be installed, and the script should be launch in
          command line using mpiexec. For instance:

          >>> mpiexec -n 4 python petsc_examples.py -mpi_linear_solver_server -ksp_type cg -pc_type bjacobi

          fedoo initializes PETSc when it is imported in this case: the
          processes other than the first one only serve the solves.

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
          >>>
          >>> # Use the symmetric-indefinite Pardiso or standalone MUMPS path
          >>> pb.set_solver('direct', symmetric=True)
        """
        if symmetric is not None and not isinstance(symmetric, bool):
            raise TypeError(
                "bool expected for symmetric, or None for the automatic choice"
            )
        # None: decided at each solve from the assemblies (see _solve)
        auto_symmetric = symmetric is None
        symmetric = bool(symmetric)

        # These options apply only to PETSc. Treat None as "not specified" so
        # callers can forward optional solver settings to any backend.
        for option in ("solver_type", "pc_type", "pc_factor_mat_solver_type"):
            if kargs.get(option) is None:
                kargs.pop(option, None)

        return_info = False
        precond = False
        if isinstance(solver, str):
            solver = solver.lower()
            if solver == "direct":
                if USE_PYPARDISO:
                    solver_func = _solver_pardiso if symmetric else spsolve
                elif USE_MUMPS:
                    solver_func = _solver_mumps
                elif USE_PETSC:
                    _petsc()
                    solver_func = _solver_petsc
                    kargs["solver_type"] = "preonly"
                    kargs["pc_type"] = "cholesky" if symmetric else "lu"
                    kargs["pc_factor_mat_solver_type"] = "mumps"
                else:
                    solver_func = sparse.linalg.spsolve
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
                _petsc()
                solver_func = _solver_petsc
            elif solver == "pardiso":
                if USE_PYPARDISO:
                    solver_func = _solver_pardiso if symmetric else spsolve
                else:
                    raise NameError(
                        'pypardiso not installed. Use "pip install pypardiso".'
                    )
            elif solver == "mumps":
                if USE_MUMPS:
                    solver_func = _solver_mumps
                else:
                    raise NameError(
                        'python-mumps not installed. Use "pip install python-mumps".'
                    )
            elif solver == "direct_scipy":
                solver_func = sparse.linalg.spsolve
            else:
                raise NameError("Choosen solver not available")
        else:  # assume solver is a function
            solver_func = solver

        if solver_func in (_solver_pardiso, _solver_mumps):
            kargs["symmetric"] = symmetric

        rebuild_factor_context = self._factor_context is not None
        self._solver_symmetric = symmetric
        self._solver_auto_symmetric = auto_symmetric and solver_func is _solver_mumps
        self.__solver = [solver, solver_func, kargs, return_info, precond]
        # MUMPS context or PETSc solver kept across the solves of the problem
        # to reuse the analysis of the matrix (and the PETSc preconditioner)
        if solver_func is _solver_mumps:
            self._kept_solver = _MumpsFactor(symmetric)
        elif solver_func is _solver_petsc:
            self._kept_solver = _PetscSolver(**kargs)
            self._kept_solver.near_null_space_provider = self._rigid_body_modes
        else:
            self._kept_solver = None
        if rebuild_factor_context:
            self.set_reuse_factorization(True)

    def _rigid_body_modes(self):
        # near null space of the reduced system (see Problem); None: unknown
        return None

    def _new_solver_context(self):
        """Return a new solver context of the kind used by :meth:`_solve`.

        For another matrix solved alongside the system matrix (each keeps its
        own analysis and factorization). None if the solver keeps no context.
        """
        if self._factor_context is not None:
            return type(self._factor_context)(self._solver_symmetric)
        if isinstance(self._kept_solver, _MumpsFactor):
            return _MumpsFactor(self._solver_symmetric)
        if isinstance(self._kept_solver, _PetscSolver):
            return _PetscSolver(**self.__solver[2])
        return None

    def _assume_sym(self):
        # True if the matrix of the problem is assembled as symmetric
        return False

    def _solve(self, A, B):
        if self._solver_auto_symmetric:
            symmetric = self._assume_sym()
            if symmetric != self._solver_symmetric:
                self._solver_symmetric = symmetric
                self.__solver[2]["symmetric"] = symmetric
                self._kept_solver = _MumpsFactor(symmetric)
                if self._factor_context is not None:
                    self.set_reuse_factorization(True)
        if self._factor_context is not None:
            if not self._factor_valid:
                self._factor_context.factor(A)
                self._factor_valid = True
            return self._factor_context.solve(B)

        if self._kept_solver is not None:
            self._kept_solver.factor(A)
            return self._kept_solver.solve(B)

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

    def set_reuse_factorization(self, reuse: bool = True):
        """Enable or disable factorization reuse for repeated solves.

        When enabled, the system matrix is factorized on the first call to
        :meth:`solve`, then subsequent calls reuse the factorization and only
        perform back-substitution. This is useful when the matrix does not
        change between solves but the right-hand side does (e.g. tangent
        stiffness computation by perturbation, multiple load cases on a
        linear problem).

        Requires one of `pypardiso`, `python-mumps` or `petsc4py` to be
        installed (same backend priority as ``solver="direct"``).

        Parameters
        ----------
        reuse: bool, default=True
            If True, enable factorization reuse. If False, disable it and
            release the cached factorization.

        Notes
        -----
        The factorization is automatically invalidated when :meth:`set_A` is
        called. The user is responsible for calling
        :meth:`invalidate_factorization` if anything else modifies the
        reduced system matrix (e.g. Dirichlet boundary conditions affecting
        the constraint reduction matrix).

        Backend is selected automatically (pypardiso > python-mumps > petsc),
        regardless of the solver previously set with :meth:`set_solver`. The
        reuse path is direct LU only; if an iterative solver was set with
        :meth:`set_solver`, it is silently bypassed while reuse is enabled.
        For pypardiso and python-mumps, the ``symmetric`` choice passed to
        :meth:`set_solver` is also applied to the reused factorization.
        Disable reuse with ``set_reuse_factorization(False)`` to restore the
        normal solver dispatch.

        Examples
        --------
        >>> pb.set_reuse_factorization(True)
        >>> for B in load_cases:
        ...     pb.set_B(B)
        ...     pb.apply_boundary_conditions()
        ...     pb.solve()
        ...     # ... read result
        >>> pb.set_reuse_factorization(False)
        """
        context = None
        if reuse:
            if USE_PYPARDISO:
                context = _PypardisoFactor(self._solver_symmetric)
            elif USE_MUMPS:
                context = _MumpsFactor(self._solver_symmetric)
            elif USE_PETSC:
                context = _PetscFactor(self._solver_symmetric)
            else:
                raise RuntimeError(
                    "Factorization reuse requires pypardiso, python-mumps, "
                    "or petsc4py. Install one of them, e.g. "
                    '"pip install python-mumps".'
                )
        if isinstance(self._factor_context, _PypardisoFactor):
            self._factor_context.close()
        self._factor_context = context
        self._factor_valid = False

    def invalidate_factorization(self):
        """Invalidate any cached factorization.

        Call this after modifying the system matrix when factorization reuse
        is enabled. :meth:`set_A` calls it automatically.
        """
        self._factor_valid = False

    @staticmethod
    def get_all():
        """Return the list of all problems."""
        return ProblemBase.__dic

    @property
    def solver(self):
        """Return the current solver used for the problem."""
        return self.__solver[1]

    @property
    def _solver_type(self):
        # name of the solver selected with set_solver ('direct', 'cg', ...)
        return self.__solver[0]


# Class to define global dof that are not associated with a mesh
class _GlobalDof:
    def __init__(self):
        self._variable = {}  # dict containing global variable ranks
        self._nvar = 0  # number of global variables
        self._vector = {}  # dic of vectors containing the variable names
        self._rank_vector = {}  # dict of vectors containing the var ranks
        self._indptr = np.array([0])
        # pointer to the indices in the global dof array for each variables
        # ie the indices for the variable rank "var" are:
        # global_dof[self._indptr[var]: self._indptr[var+1]]

        self.n_dof = 0
        """Number of non-local DoF."""

    def __repr__(self):
        return (
            f"Global dof labels:\n{list(self._variable)}\n\n"
            "Global vectors:\n"
            f"{self._vector}"
        )

    def __getitem__(self, item: str):
        # only for user interface
        # return an array of indices for the variable or vector named "item"
        # item should be a str
        if item in self._variable:
            rank = self._variable[item]
            return np.arange(self._indptr[rank], self._indptr[rank + 1])
        elif item in self._vector:
            # asume variables inside the vector are contigus
            rank = self._rank_vector[item][0]
            n_var_in_vec = len(self._rank_vector[item])
            return np.arange(
                self._indptr[rank], self._indptr[rank + n_var_in_vec]
            ).reshape(n_var_in_vec, -1)
        else:
            raise ValueError("Global variable or vector unknown.")

    def __contains__(self, key):
        return key in self._variable

    def add(
        self,
        variable_names: str | list[str],
        nb_new_dof: int = 1,
        vector_name: str | None = None,
    ):
        """Add global degrees of freedom to the problem.

        The global dof are global in the sense that they are not associated
        to a node of a mesh as usual finite element dof. They are usefull to
        define some constraints, like periodic boundary conditions.

        Parameters
        ----------
        variable_names: str, list[str]
            Name of the global variable. Name is
            required to catch the results using the get_results or
            get_ext_forces method.
        nb_new_dof: int, default=1
            Number of new dof for each new global variable
        vector_name: str, optional
            If a vector name is given, add vector that allow to catch
            all the associated variables at the same time.

        Note
        -----
        If the variables already exists, some dof are added to the variables
        In this case, the vector_name argument is ignored. A vector should be defined
        when adding new variables only.
        """
        if not isinstance(variable_names, list):
            variable_names = [variable_names]

        n_new_variables = len(variable_names)
        if variable_names[0] in self._variable:
            # assume all the variable in variable_names should already exist
            # in this case, only add some dof to each variables
            variable_ranks = [self._variable[var_name] for var_name in variable_names]
            variable_ranks.sort()  # should aleady be the case but kept for safety
            dof_indice = (
                self._indptr[variable_ranks[0] + 1] - self._indptr[variable_ranks[0]]
            )
            # dof_indice = indice of the first new dof in each variable
            for r in variable_ranks:
                self._indptr[r + 1 :] += nb_new_dof
        else:
            # add new variable
            for i, var_name in enumerate(variable_names):
                self._variable[var_name] = self._nvar + i
            indptr = list(self._indptr)
            indptr.extend(
                [self.n_dof + i * nb_new_dof for i in range(1, n_new_variables + 1)]
            )
            self._indptr = np.array(indptr)

            if vector_name is not None:
                self._vector[vector_name] = variable_names
                self._rank_vector[vector_name] = [
                    self._nvar + i for i in range(n_new_variables)
                ]

            self._nvar += n_new_variables
            dof_indice = 0

        self.n_dof = self._indptr[-1]

        return dof_indice

    def indice_start(self, variable_name):
        """Return the first indice of the given variable name in the global dofs."""
        return self._indptr[self._variable[variable_name]]


def _solver_petsc(A, B, **kargs):
    """Solve A x = B with PETSc (a one-shot :class:`_PetscSolver`)."""
    solver = _PetscSolver(**kargs)
    solver.factor(A)
    return solver.solve(B)


class _PetscSolver:
    """PETSc linear solver (KSP) kept across the solves of a problem.

    The PETSc matrix is updated in place while the sparsity pattern of the
    system does not change, so that PETSc reuses what only depends on the
    pattern: the ordering and symbolic factorization of a direct solver
    (MUMPS, ...) or the structures of the preconditioner. With
    ``reuse_preconditioner=True``, the preconditioner itself (e.g. a GAMG
    hierarchy) built for the first matrix is kept for the next ones, the
    Krylov iterations making up for the difference: a Newton iteration then
    only costs the iterations. Call :meth:`reset_preconditioner` to rebuild
    it.

    The objects are created on PETSC_COMM_SELF: the system is held by one
    process. Use the PETSc linear solver server (``mpiexec -n <np> python
    script.py -mpi_linear_solver_server -ksp_type cg -pc_type gamg``) to
    solve it on several MPI processes; the near null space is not passed to
    the server.

    Parameters
    ----------
    solver_type: str, default="bcgs"
        PETSc KSP type.
    pc_type: str, default="eisenstat"
        PETSc preconditioner type.
    pc_factor_mat_solver_type: str, optional
        Package of the factorization for the "lu" and "cholesky"
        preconditioners (e.g. "mumps").
    reuse_preconditioner: bool, default=False
        Keep the preconditioner of the first matrix for the next ones.
    near_null_space: bool, default=True
        With the GAMG preconditioner, give it the near null space of the
        problem (rigid body modes of a mechanical problem), which it needs
        to be efficient for elasticity.
    rtol, atol, max_it: optional
        Tolerances of the Krylov solver (PETSc defaults if not given).

    Notes
    -----
    The PETSc options of the command line or of petsc4py.PETSc.Options()
    (e.g. ``ksp_rtol``, ``pc_gamg_threshold``) are applied on top. With the
    PETSc linear solver server, the solver of the server is only set by these
    options: the parameters given here are then ignored, with a warning.
    """

    def __init__(
        self,
        solver_type=None,
        pc_type=None,
        pc_factor_mat_solver_type=None,
        reuse_preconditioner=False,
        rtol=None,
        atol=None,
        max_it=None,
        near_null_space=True,
        **kargs,
    ):
        PETSc = _petsc()
        given = [
            name
            for name, value in (
                ("solver_type", solver_type),
                ("pc_type", pc_type),
                ("pc_factor_mat_solver_type", pc_factor_mat_solver_type),
                ("rtol", rtol),
                ("atol", atol),
                ("max_it", max_it),
            )
            if value is not None
        ]
        if given and PETSc.Options().hasName("mpi_linear_solver_server"):
            # PCMPI: the solver of the server only reads the options database
            warnings.warn(
                "With -mpi_linear_solver_server, the PETSc solver is set by "
                "the command line options only (e.g. -ksp_type cg -pc_type "
                f"gamg -ksp_rtol 1e-8): {', '.join(given)} given to "
                "set_solver are ignored, and so is the near null space."
            )
        solver_type = "bcgs" if solver_type is None else solver_type
        pc_type = "eisenstat" if pc_type is None else pc_type
        self._comm = PETSc.COMM_SELF
        self._ksp = PETSc.KSP().create(comm=self._comm)
        self._ksp.setType(solver_type)
        pc = self._ksp.getPC()
        pc.setType(pc_type)
        if pc_factor_mat_solver_type is not None:
            pc.setFactorSolverType(pc_factor_mat_solver_type)
        if rtol is not None or atol is not None or max_it is not None:
            self._ksp.setTolerances(rtol=rtol, atol=atol, max_it=max_it)
        self._ksp.setFromOptions()
        self._reuse_preconditioner = reuse_preconditioner
        self._near_null_space = near_null_space
        self.near_null_space_provider = None  # set by the problem
        self._mat = None
        self._pattern = None
        self._values = None  # values of the matrix set up last

    def factor(self, A):
        """Set the matrix of the next solves (factorized if direct)."""
        PETSc = _petsc()
        A = A.tocsr()
        if not A.has_sorted_indices:
            A = A.sorted_indices()
        same = (
            self._pattern is not None
            and A.shape == self._pattern[0]
            and np.array_equal(A.indptr, self._pattern[1])
            and np.array_equal(A.indices, self._pattern[2])
        )
        if same and self._values is not None and np.array_equal(A.data, self._values):
            # values of the matrix set up: factorization and preconditioner
            # are still valid
            return
        self._values = None
        pattern = self._pattern
        if same:
            # same Mat object, new values: PETSc keeps the symbolic data
            self._mat.zeroEntries()
            self._mat.setValuesCSR(A.indptr, A.indices, A.data)
            self._mat.assemble()
        else:
            pattern = (A.shape, A.indptr.copy(), A.indices.copy())
            self._pattern = None  # until the setup has succeeded
            self._mat = PETSc.Mat().createAIJ(
                A.shape, csr=(A.indptr, A.indices, A.data), comm=self._comm
            )
            self._mat.assemble()
            self._set_near_null_space()
            # new pattern, possibly new size (boundary conditions, contact):
            # the data of the previous one are freed, the options kept
            self._ksp.reset()
            self._ksp.setOperators(self._mat)
            self._ksp.getPC().setReusePreconditioner(False)
        self._ksp.setUp()  # factorization (direct) or preconditioner setup
        if self._reuse_preconditioner:
            self._ksp.getPC().setReusePreconditioner(True)
        self._pattern = pattern
        self._values = A.data.copy()

    def _set_near_null_space(self):
        if not (
            self._near_null_space
            and self.near_null_space_provider is not None
            and self._ksp.getPC().getType() == "gamg"
        ):
            return
        modes = self.near_null_space_provider()
        if modes is None or modes.shape[0] != self._mat.getSize()[1]:
            return
        q = np.linalg.qr(modes)[0]  # PETSc wants orthonormal vectors
        vectors = []
        for k in range(q.shape[1]):
            v = self._mat.createVecRight()
            v.setArray(np.ascontiguousarray(q[:, k]))
            vectors.append(v)
        PETSc = _petsc()
        self._mat.setNearNullSpace(
            PETSc.NullSpace().create(vectors=vectors, comm=self._comm)
        )

    def reset_preconditioner(self):
        """Rebuild the preconditioner at the next :meth:`factor`."""
        self._ksp.getPC().setReusePreconditioner(False)
        self._values = None  # even for a matrix with the same values

    def solve(self, B):
        b = self._mat.createVecLeft()
        b.setArray(np.ascontiguousarray(B, dtype=np.float64).reshape(-1))
        x = self._mat.createVecRight()
        self._ksp.solve(b, x)
        if self._ksp.getConvergedReason() < 0:
            warnings.warn(
                "PETSc solver did not converge (reason "
                f"{self._ksp.getConvergedReason()}, "
                f"{self._ksp.getIterationNumber()} iterations)"
            )
        return x.getArray().copy()

    @property
    def ksp(self):
        """The petsc4py KSP object, to change or inspect the solver."""
        return self._ksp


def _solver_pardiso(A, B, symmetric=False, **kargs):
    """Solve with Pardiso using a general or symmetric-indefinite matrix."""
    from pypardiso import PyPardisoSolver

    solver = PyPardisoSolver(mtype=-2 if symmetric else 11)
    if symmetric:
        A = sparse.triu(A, format="csr")
    return solver.solve(A, B)


def _mumps_comm():
    """MPI communicator of the python-mumps contexts: fedoo solves on one
    process. Under mpiexec, an MPI build of python-mumps (conda-forge, next to
    PETSc) would use MPI_COMM_WORLD and wait for the other processes, e.g.
    those of the PETSc linear solver server, which never come."""
    try:
        from mpi4py import MPI
    except ImportError:
        return None
    return MPI.COMM_SELF if MPI.COMM_WORLD.Get_size() > 1 else None


def _mumps_threads():
    """Number of OpenMP threads given to MUMPS (ICNTL(16)), or None to keep
    the OpenMP default. MUMPS gains little beyond about 8 threads and loses
    a lot on the efficiency cores of a hybrid chip, while the OpenMP default
    is all the logical cores: at most 8, and at most the performance cores.
    A user setting of OMP_NUM_THREADS is kept."""
    if "OMP_NUM_THREADS" in os.environ:
        return None
    from fedoo.core._sparsematrix import _performance_cores

    return max(1, min(8, _performance_cores() or os.cpu_count() or 1))


def _mumps_context(symmetric):
    """Build a python-mumps ``Context``, symmetric-indefinite if requested.

    Return the context and whether the symmetry has to be given with the
    matrix instead. The ``sym`` keyword only exists in recent python-mumps
    releases, so it is passed exclusively when a symmetric factorization is
    asked for. Older releases (0.0.6) take it in ``set_matrix(a,
    symmetric=True)``, which keeps the upper triangle.
    """
    import mumps

    comm = _mumps_comm()
    kargs = {} if comm is None else {"comm": comm}
    if not symmetric:
        return mumps.Context(**kargs), False
    try:
        return mumps.Context(sym=2, **kargs), False
    except TypeError:
        return mumps.Context(**kargs), True


def _solver_mumps(A, B, symmetric=False, **kargs):
    # python-mumps exposes a Context-based API only; there is no
    # top-level spsolve. Problem keeps a _MumpsFactor across its solves
    # instead (see ProblemBase._solve), to reuse the analysis of the matrix.
    factor = _MumpsFactor(symmetric)
    factor.factor(A)
    return factor.solve(B)


# =============================================================
# Factor-reuse adapters used by ProblemBase.set_reuse_factorization
# All expose a uniform .factor(A) / .solve(B) interface.
# =============================================================
class _MumpsFactor:
    """MUMPS factorization that keeps its analysis (ordering and symbolic
    factorization) while the sparsity pattern of the matrix does not change:
    only the numerical factorization is redone for a new matrix with the
    same pattern, explicit zeros included."""

    def __init__(self, symmetric=False):
        self._symmetric = symmetric
        self._ctx, self._sym_with_matrix = _mumps_context(symmetric)
        self._pattern = None
        # positions in A.data of the values held by the context (same pattern)
        self._entries = None
        self._all_entries = False  # _entries is arange(A.nnz)
        # values of the factorized matrix, kept only when the context does not
        # expose its own (_entries is None); None: no valid factorization
        self._values = None
        self._factored = False  # the context holds the factors of its values

    def factor(self, A):
        A = A.tocsr()
        same = (
            self._pattern is not None
            and A.shape == self._pattern[0]
            and np.array_equal(A.indptr, self._pattern[1])
            and np.array_equal(A.indices, self._pattern[2])
        )
        if same and self._entries is not None:
            # MUMPS points to the arrays given by set_matrix: new values are
            # written in place, without its coo conversion and triangle extraction
            data = self._ctx.data
            values = A.data if self._all_entries else A.data[self._entries]
            if self._factored and np.array_equal(values, data):
                return  # values of the factorized matrix: factors still valid
            data[...] = values
            self._factored = False
            self._ctx.factor(reuse_analysis=True)
            self._factored = True
            return
        if same and self._factored and np.array_equal(A.data, self._values):
            return  # same values, see above
        self._factored = False
        self._values = None
        self._entries = None
        pattern = self._pattern
        if not same:
            pattern = (A.shape, A.indptr.copy(), A.indices.copy())
            self._pattern = None  # until the factorization has succeeded
        if self._sym_with_matrix:
            self._ctx.set_matrix(A, symmetric=True)
        else:
            A_given = sparse.tril(A, format="csr") if self._symmetric else A
            self._ctx.set_matrix(A_given)
        if not same:
            # thread count before the analysis: MUMPS keeps it for the
            # factorizations (its tree parallelism requires the same one)
            n_threads = _mumps_threads()
            instance = getattr(self._ctx, "mumps_instance", None)
            if n_threads is not None and instance is not None:
                instance.icntl[16] = n_threads
        self._ctx.factor(reuse_analysis=same)
        self._pattern = pattern
        self._factored = True
        self._entries = self._given_entries(A)
        self._all_entries = self._entries is not None and len(self._entries) == A.nnz
        if self._entries is None:
            self._values = A.data.copy()

    def _given_entries(self, A):
        """Positions in A.data of the values that set_matrix gave to MUMPS
        (coo order of the full matrix or of the triangle kept), or None if
        they cannot be identified (the values are then always set by
        set_matrix)."""
        data = getattr(self._ctx, "data", None)
        if not isinstance(data, np.ndarray) or data.dtype != A.data.dtype:
            return None
        if not data.flags.writeable or not data.flags.c_contiguous:
            return None
        rows = np.repeat(np.arange(A.shape[0]), np.diff(A.indptr))
        if self._sym_with_matrix:
            entries = np.flatnonzero(A.indices >= rows)  # upper triangle
        elif self._symmetric:
            entries = np.flatnonzero(A.indices <= rows)  # lower triangle
        else:
            entries = np.arange(A.nnz)
        if len(entries) != len(data) or not np.array_equal(A.data[entries], data):
            return None
        return entries

    def solve(self, B):
        return self._ctx.solve(B)


class _PypardisoFactor:
    def __init__(self, symmetric=False):
        from pypardiso import PyPardisoSolver

        self._symmetric = symmetric
        self._solver = PyPardisoSolver(mtype=-2 if symmetric else 11)
        self._A = None

    def factor(self, A):
        if self._symmetric:
            A = sparse.triu(A, format="csr")
        self._solver.factorize(A)
        self._A = A

    def solve(self, B):
        return self._solver.solve(self._A, B)

    def close(self):
        if self._A is not None:
            # Dropping PyPardisoSolver leaves its native factorization allocated.
            self._solver.free_memory(everything=True)
            self._A = None


class _PetscFactor(_PetscSolver):
    """Direct factorization by PETSc (MUMPS, Cholesky if symmetric)."""

    def __init__(self, symmetric=False):
        super().__init__(
            solver_type="preonly",
            pc_type="cholesky" if symmetric else "lu",
            pc_factor_mat_solver_type="mumps",
        )
