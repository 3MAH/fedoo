Installation
=================================

Installation using conda (recommended):

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
      laws, ...).

    * `PyVista <https://docs.pyvista.org/version/stable/>`_
      for results visualization and mesh utils.

    * An efficient sparse matrix solver (pypardiso or petsc4py) depending
      on the processor as described below.


Full pip install
----------------

It is also possible to install fedoo with all recommended pip dependencies
(sparse solver, plotting, IPC contact) in one line:

.. code-block:: none

    $ pip install fedoo[all]

This installs the following optional groups: ``solver``, ``plot``,
``dataio``, ``test`` and ``ipc``.

.. note::
   Simcoon is only available through conda and is **not** included in
   ``fedoo[all]``.  See the Simcoon section below for its installation.


Individual optional groups
--------------------------

You can also install optional groups individually:

.. code-block:: none

    $ pip install fedoo[solver]      # fast sparse solver (pypardiso or umfpack)
    $ pip install fedoo[plot]        # matplotlib + pyvista
    $ pip install fedoo[ipc]         # IPC contact (ipctk)


Sparse solvers
--------------

It is highly recommended to install a fast direct sparse matrix solver
to improve performances:

    * `Pypardiso <https://pypi.org/project/pypardiso/>`_
      for intel processors (binding to the pardiso solver)

    * `Petsc4Py <https://pypi.org/project/petsc4py/>`_
      mainly compatible with linux or macos including the MUMPS solver.

    * `Scikit-umfpack <https://scikit-umfpack.github.io/scikit-umfpack/>`_


Simcoon
-------

A lot of features (finite strain, non-linear constitutive laws, ...) require
the installation of simcoon. Simcoon is available on conda only and can be
installed with:

.. code-block:: none

    $ conda install -c conda-forge -c set3mah simcoon
