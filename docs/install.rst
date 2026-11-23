Installation
=================================

Installation using conda with recommanded dependencies: 

.. code-block:: none

    $ conda install -c conda-forge -c set3mah fedoo

Minimal installation with pip:

.. code-block:: none

    $ pip install fedoo

The required dependencies that are automatically installed with fedoo are:

    * `Numpy <https://numpy.org/>`_

    * `Scipy <https://scipy.org/>`_ mainly for sparse matrices.

In addition, the conda package also includes some recommended dependencies:

    * `Simcoon <https://simcoon.readthedocs.io/en/latest/>`_
      brings a lot of features (finite strain, non-linear constitutive
      laws, ...). fedoo requires ``simcoon >= 2.6``. Simcoon can be
      installed using conda or pip.

    * `PyVista <https://docs.pyvista.org/version/stable/>`_
      for results visualization and mesh utils.

    * An efficient sparse matrix solver (pypardiso, python-mumps or
      petsc4py) depending on the processor as described below.


Full pip install
----------------

It is also possible to install fedoo with all recommended dependencies
(sparse solver, plotting, IPC contact) in one line:

.. code-block:: none

    $ pip install fedoo[all]

This installs the following optional groups: ``solver``, ``plot``,
``test``, ``ipc``, ``io`` and ``zstd``. Zstandard compression for FDH5 can
also be installed separately with ``pip install "fedoo[zstd]"``; both writers
and readers need the optional ``hdf5plugin`` package.

``pyvistaqt``, which is required for the viewer, is not included in the all
group. This allows you to choose your preferred Qt binding (``pyqt5``,
``pyqt6`` or ``pyside6``). We recommend installing only one of these to avoid
potential library conflicts.

To enable the viewer, you can install the dependencies explicitly:

.. code-block:: none

    $ pip install fedoo[all] pyvistaqt pyqt5

Alternatively, use the ``gui`` install group that includes ``pyvistaqt``
and ``pyside6``:

.. code-block:: none

    $ pip install "fedoo[all,gui]"


Individual optional groups
--------------------------

You can also install optional groups individually:

.. code-block:: none

    $ pip install fedoo[solver]      # fast sparse solver (pypardiso, or python-mumps on macOS/arm64)
    $ pip install fedoo[plot]        # matplotlib + pyvista
    $ pip install fedoo[simcoon]     # simcoon
    $ pip install fedoo[ipc]         # IPC contact (ipctk)
    $ pip install fedoo[gui]         # pyvistaqt + pyside6

`numba <https://numba.pydata.org/>`_ is optional and not installed by fedoo.
When it is present (``pip install numba`` or ``conda install -c conda-forge
numba``), the assembly uses compiled kernels instead of numpy, on at most the
number of threads given by ``fedoo.get_config()["N_THREADS"]`` (fewer for a
small model).


Sparse solvers
--------------

It is highly recommended to install a fast direct sparse matrix solver
to improve performances. fedoo dispatches to the first one available in
this priority order: pypardiso → python-mumps → petsc4py:

    * `Pypardiso <https://pypi.org/project/pypardiso/>`_
      for intel processors (binding to the pardiso solver).

    * `python-mumps <https://pypi.org/project/python-mumps/>`_
      standalone Python bindings for the MUMPS direct solver. Recommended
      on macOS and arm64 (ARM Linux), where pypardiso is not available
      (``fedoo[solver]`` installs it there), and as a lighter alternative to
      PETSc when only direct solving is needed.

    * `Petsc4Py <https://pypi.org/project/petsc4py/>`_
      mainly compatible with linux or macos including the MUMPS solver.
      Use this if you need PETSc's iterative solvers (e.g. conjugate
      gradient with the GAMG algebraic multigrid for large 3D models) or
      MPI parallelism. petsc4py is not a dependency of fedoo: install it
      yourself (see :ref:`petsc-install`).

    * `Scikit-umfpack <https://scikit-umfpack.github.io/scikit-umfpack/>`_
      optional fallback to ``python-mumps``. Detected automatically if
      installed, useful in very specific cases (e.g. very small problems,
      or as a serial backup). Not included in ``[solver]`` extras because
      its install can be tricky on some platforms; install it manually if
      you need it.

To be able to launch the fedoo viewer, the module
`pyvistaqt <https://qtdocs.pyvista.org/>`_ is also required.

.. note::

    On macOS, prefer ``conda install -c
    conda-forge python-mumps`` over ``pip install python-mumps``: the
    PyPI sdist requires a system MUMPS lib via pkg-config, while the
    conda-forge package bundles ``mumps-seq`` directly. Pin the BLAS
    variant to Accelerate at the same time to route the dense block
    kernels through Apple's vecLib::

        $ conda install -c conda-forge "libblas=*=*accelerate" python-mumps

    fedoo runs MUMPS on at most 8 threads (performance cores only on Apple
    chips) unless ``OMP_NUM_THREADS`` is set, and sets ``KMP_BLOCKTIME=1``
    at import on macOS when pypardiso is absent. Define either variable before
    importing fedoo to override.

    On Linux and Windows x86_64 (Intel or AMD), prefer the MKL variant,
    with pypardiso, which then shares the same MKL::

        $ conda install -c conda-forge "libblas=*=*mkl" pypardiso

    On arm64 Linux, use the OpenBLAS variant::

        $ conda install -c conda-forge "libblas=*=*openblas" python-mumps

    You can verify which BLAS got linked with::

        $ python -c "import numpy; numpy.show_config()"
        $ otool -L $(python -c 'import numpy.linalg._umath_linalg as m; print(m.__file__)')   # macOS
        $ ldd  $(python -c 'import numpy.linalg._umath_linalg as m; print(m.__file__)')      # Linux


.. _petsc-install:

PETSc
-----

fedoo uses PETSc through petsc4py when it is installed, with
``pb.set_solver('petsc', ...)``, or as direct solver if neither pypardiso
nor python-mumps is installed (or with
``fd.get_config()['USE_PETSC'] = True``). fedoo does not depend on it: it is
detected at import and initialized at its first use.

The simplest installation is with conda, in the environment of fedoo::

    $ conda install -c conda-forge petsc4py

conda-forge PETSc is built with MPI and MUMPS: the packages that depend on
MPI-sensitive libraries are then installed in their MPI variant (``h5py``,
``python-mumps``, which then shares the MUMPS library of PETSc). With pip,
PETSc is built from sources (long, needs compilers and MPI)::

    $ pip install mpi4py petsc petsc4py

Large 3D mechanical models: conjugate gradient and GAMG, which fedoo gives
the rigid body modes of the problem (near null space). With
``reuse_preconditioner=True``, the multigrid hierarchy of the first matrix is
kept for the next Newton iterations and increments::

    pb.set_solver('petsc', solver_type='cg', pc_type='gamg', rtol=1e-8,
                  reuse_preconditioner=True)

The PETSc options can also be given on the command line of the script
(``python script.py -ksp_type cg -pc_type gamg -ksp_converged_reason``) or in
``petsc4py.PETSc.Options()`` before ``set_solver``. To solve on several MPI
processes, use the PETSc linear solver server: the script runs on the first
process, the others only take part in the solves::

    $ mpiexec -n 4 python script.py -mpi_linear_solver_server \
          -ksp_type cg -pc_type gamg -ksp_rtol 1e-8

The solver of the server is then set by the command line options only (the
parameters of ``set_solver`` are ignored, with a warning), and it does not
get the rigid body modes: GAMG needs about ten times more iterations. For a
3D elastic box of 105,000 dof, the serial GAMG with the rigid body modes
(16 iterations) was faster than the server on 4 processes (192 iterations).

On macOS, fedoo disables the shared memory of the server
(``-mpi_linear_solver_server_use_shared_memory false``): the System V shared
memory is limited to 4 MB by default.


Simcoon
-------

Many features (such as finite strain and non-linear constitutive laws) require
Simcoon to be installed. Simcoon is available via both pip and conda.
To install Simcoon individually, use either:

.. code-block:: none

    $ conda install -c conda-forge -c set3mah simcoon

Or:

.. code-block:: none

    $ pip install simcoon
