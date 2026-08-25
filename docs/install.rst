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
      laws, ...). Simcoon can be installed using conda or pip.

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
``simcoon``, ``test`` and ``ipc``.

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

    $ pip install fedoo[all, gui]


Individual optional groups
--------------------------

You can also install optional groups individually:

.. code-block:: none

    $ pip install fedoo[solver]      # fast sparse solver (pypardiso or python-mumps)
    $ pip install fedoo[plot]        # matplotlib + pyvista
    $ pip install fedoo[simcoon]     # simcoon
    $ pip install fedoo[ipc]         # IPC contact (ipctk)
    $ pip install fedoo[gui]         # pyvistaqt + pyside6


Sparse solvers
--------------

It is highly recommended to install a fast direct sparse matrix solver
to improve performances. fedoo dispatches to the first one available in
this priority order: pypardiso → python-mumps → petsc4py:

    * `Pypardiso <https://pypi.org/project/pypardiso/>`_
      for intel processors (binding to the pardiso solver).

    * `python-mumps <https://pypi.org/project/python-mumps/>`_
      standalone Python bindings for the MUMPS direct solver. Recommended
      on arm64 (Apple Silicon, ARM Linux) where pypardiso is not available
      and as a lighter alternative to PETSc when only direct solving is
      needed.

    * `Petsc4Py <https://pypi.org/project/petsc4py/>`_
      mainly compatible with linux or macos including the MUMPS solver.
      Use this if you need PETSc's iterative solvers or MPI parallelism.

    * `Scikit-umfpack <https://scikit-umfpack.github.io/scikit-umfpack/>`_
      optional fallback to ``python-mumps``. Detected automatically if
      installed, useful in very specific cases (e.g. very small problems,
      or as a serial backup). Not included in ``[solver]`` extras because
      its install can be tricky on some platforms; install it manually if
      you need it.

To be able to launch the fedoo viewer, the module
`pyvistaqt <https://qtdocs.pyvista.org/>`_ is also required.

.. note::

    On macOS (Apple Silicon especially), prefer ``conda install -c
    conda-forge python-mumps`` over ``pip install python-mumps``: the
    PyPI sdist requires a system MUMPS lib via pkg-config, while the
    conda-forge package bundles ``mumps-seq`` directly. Pin the BLAS
    variant to Accelerate at the same time to route the dense block
    kernels through Apple's vecLib::

        $ conda install -c conda-forge "libblas=*=*accelerate" python-mumps

    On Linux / AMD, the OpenBLAS variant is the equivalent::

        $ conda install -c conda-forge "libblas=*=*openblas" python-mumps

    You can verify which BLAS got linked with::

        $ python -c "import numpy; numpy.show_config()"
        $ otool -L $(python -c 'import numpy.linalg._umath_linalg as m; print(m.__file__)')   # macOS
        $ ldd  $(python -c 'import numpy.linalg._umath_linalg as m; print(m.__file__)')      # Linux


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