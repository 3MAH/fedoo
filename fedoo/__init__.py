from . import constitutivelaw, constraint, core, homogen, mesh, problem, weakform
from ._version import __version__
from .core import (
    MPC,
    Assembly,
    AssemblySum,
    ConstitutiveLaw,
    DataSet,
    ListBC,
    Mesh,
    ModelingSpace,
    MultiFrameDataSet,
    Problem,
    WeakForm,
    WeakFormSum,
    read_data,
)


class get_config:
    """Extract the current fedoo configuration (state of optional modules)
    from submodules into a dict form.

    The returned information are:
        * 'fedoo version': The current version of fedoo (non modifiable)
        * 'USE_SIMCOON': bool that define if the simcoon librairie may be used
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
            "USE_SIMCOON": constitutivelaw.simcoon_umat.USE_SIMCOON,
            "USE_PYPARDISO": core.base.USE_PYPARDISO,
            "USE_PETSC": core.base.USE_PETSC,
            "USE_UMFPACK": core.base.USE_UMFPACK,
            "USE_PYVISTA": core.mesh.USE_PYVISTA,
            "USE_MPL": core.dataset.USE_MPL,
            "USE_PYVISTA_QT": core.dataset.USE_PYVISTA_QT,
            "USE_PANDAS": core.dataset.USE_PANDAS,
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
            raise NameError("fedoo version is not modifiable")
        if item == "USE_SIMCOON":
            constitutivelaw.simcoon_umat.USE_SIMCOON = value
            weakform.stress_equilibrium.USE_SIMCOON = value
        elif item == "USE_PYPARDISO":
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
        elif item == "USE_PANDAS":
            core.dataset.USE_PANDAS = value
        else:
            raise KeyError


def show_config():
    """Print the state of the optional modules and the version of fedoo"""
    print(get_config())
