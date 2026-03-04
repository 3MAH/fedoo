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

    * An efficient sparse matrix solver (pypardiso or petsc4py) depending
      on the processor as described below.


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

    $ pip install fedoo[solver]      # fast sparse solver (pypardiso or umfpack)
    $ pip install fedoo[plot]        # matplotlib + pyvista
    $ pip install fedoo[simcoon]     # simcoon
    $ pip install fedoo[ipc]         # IPC contact (ipctk)
    $ pip install fedoo[gui]         # pyvistaqt + pyside6


Sparse solvers
--------------

It is highly recommended to install a fast direct sparse matrix solver
to improve performances:

    * `Pypardiso <https://pypi.org/project/pypardiso/>`_
      for intel processors (binding to the pardiso solver)

    * `Petsc4Py <https://pypi.org/project/petsc4py/>`_
      mainly compatible with linux or macos including the MUMPS solver.

    * `Scikit-umfpack <https://scikit-umfpack.github.io/scikit-umfpack/>`_

To be able to launch the fedoo viewer, the module 
`pyvistaqt <https://qtdocs.pyvista.org/>`_ is also required.


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