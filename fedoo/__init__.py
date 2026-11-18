import importlib.util as _importlib_util
import os as _os
import sys as _sys

if _sys.platform == "darwin" and _importlib_util.find_spec("pypardiso") is None:
    # libomp sleeps its threads after each parallel region on macOS: a 1 ms spin
    # spares MUMPS the wake-ups. Not with pypardiso, whose MKL reads it too.
    _os.environ.setdefault("KMP_BLOCKTIME", "1")

from . import (
    constitutivelaw,
    constraint,
    core,
    homogen,
    mesh,
    problem,
    time,
    weakform,
    post_processing,
)
from ._version import __version__
from .core import (
    MPC,
    Assembly,
    AssemblySum,
    ConstitutiveLaw,
    DataSet,
    ListBC,
    LagrangeMultiplierAssembly,
    Mesh,
    MultiMesh,
    ModelingSpace,
    MultiFrameDataSet,
    MultiMeshData,
    Problem,
    ProblemDataSet,
    WeakForm,
    WeakFormSum,
    read_data,
)
from .util.recovery import (
    recover_gradient,
    recover_hessian,
    to_upper_diagonal,
    to_voigt,
)

try:
    from .util.viewer import viewer
except ImportError:
    # module not installed
    def viewer(res=None):
        raise ImportError(
            "pyvistaqt is required to launch the viewer. "
            "Install it with: pip install pyvistaqt pyqt5"
        )


class get_config:
    """Extract the current fedoo configuration.

    The current state of optional modules is extracted from submodules
    and returend into a dict form.

    The returned information are:
        * 'fedoo version': The current version of fedoo (non modifiable)
        * 'USE_PYPARDISO': bool that define if the pardiso direct solver may be used
        * 'USE_PETSC': bool that define if the mumps direct solver should be used.
        * 'USE_UMFPACK': bool that define if the scikit-umfpack solver may be used
          Only one between pardiso, petsc and umfpack direct solver may be used.
        * 'USE_PYVISTA': bool that define if the pyvista library may be used
        * 'USE_MPL': bool that define if the matplotlib library may be used
        * 'USE_PYVISTA_QT': bool that define if the pyvista_qt library
           may be used for background plotter. If True, it is the default plotter
           for DataSet objects.
        * 'USE_PANDAS': bool that define if the pandas library may be used
        * 'N_THREADS': int, number of threads of the simcoon batch calls (constitutive
          laws, objective rates, tangent conversions). 0 (default) uses all the cores,
          1 runs them serially.

    To change a configuration option use fedoo.get_config()[option] = value
    To print the current configuration use: fedoo.show_config()

    Example
    --------

        >>> import fedoo as fd
        >>> fd.get_config()['USE_PYPARDISO'] = False
    """

    def __init__(self):
        self._dict = {
            "fedoo version": __version__,
            "USE_PYPARDISO": core.base.USE_PYPARDISO,
            "USE_PETSC": core.base.USE_PETSC,
            "USE_UMFPACK": core.base.USE_UMFPACK,
            "USE_PYVISTA": core.mesh.USE_PYVISTA,
            "USE_MPL": core.dataset.USE_MPL,
            "USE_PYVISTA_QT": core.dataset.USE_PYVISTA_QT,
            "USE_PANDAS": core.dataset.USE_PANDAS,
            "N_THREADS": core.base.N_THREADS,
        }

    def __repr__(self):
        return self._dict.__repr__()

    def __str__(self):
        res = ""
        for key in self._dict:
            res += key + f": {self._dict[key]}\n"
        return res

    def __getitem__(self, item):
        return self._dict[item]

    def __setitem__(self, item, value):
        if item == "fedoo version":
            raise ValueError("fedoo version is not modifiable")
        if item == "USE_PYPARDISO":
            core.base.USE_PYPARDISO = value
            if value:
                core.base.USE_UMFPACK = False
                core.base.USE_PETSC = False
                core.base._reload_external_solvers(get_config())
        elif item == "USE_PETSC":
            core.base.USE_PETSC = value
            if value:
                core.base.USE_PYPARDISO = False
                core.base.USE_UMFPACK = False
                core.base._reload_external_solvers(get_config())
        elif item == "USE_UMFPACK":
            core.base.USE_UMFPACK = value
            if value:
                core.base.USE_PYPARDISO = False
                core.base.USE_PETSC = False
                core.base._reload_external_solvers(get_config())
        elif item == "USE_PYVISTA":
            core.mesh.USE_PYVISTA = value
            core.dataset.USE_PYVISTA = value
        elif item == "USE_MPL":
            core.dataset.USE_MPL = value
        elif item == "USE_PYVISTA_QT":
            core.dataset.USE_PYVISTA_QT = value
            core.mesh.USE_PYVISTA_QT = value
        elif item == "USE_PANDAS":
            core.dataset.USE_PANDAS = value
        elif item == "N_THREADS":
            if isinstance(value, bool) or int(value) != value or value < 0:
                raise ValueError(
                    "N_THREADS must be a non-negative integer (0 = all cores)"
                )
            core.base.N_THREADS = int(value)
        else:
            raise KeyError


def show_config():
    """Print the state of the optional modules and the version of fedoo"""
    print(get_config())


from .util.fdh5 import get_fdh5_options, set_fdh5_options, reset_fdh5_options
