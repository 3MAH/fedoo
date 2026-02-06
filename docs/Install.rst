Installation
=================================

Installation using conda with recommanded dependencies: 

.. code-block:: none

    $ conda install -c conda-forge -c set3mah fedoo

Minimal installation with pip:

.. code-block:: none

    $ pip install fedoo

Full installation with pip with all dependencies:

.. code-block:: none

    $ pip install fedoo[all]
    

The required dependencies that are automatically installed with fedoo are:

    * `Numpy <https://numpy.org/>`_
    
    * `Scipy <https://scipy.org/>`_ mainly for sparse matrices. 

In addition, the conda package also includes some recommanded dependencies:

    * `Simcoon <https://simcoon.readthedocs.io/en/latest/>`_ 
      brings a lot of features (finite strain, non-linear constitutive
      laws, ...). Simcoon can be installed using conda or pip.

    * `PyVista <https://docs.pyvista.org/version/stable/>`_ 
      for results visulisation and mesh utils.
      
    * An efficient sparse matrix solver (pypardiso or petsc4py) depending
      on the processor as described below.


It is highly recommanded to also install a fast direct sparse matrix solver
to improve performances:

    * `Pypardiso <https://pypi.org/project/pypardiso/>`_ 
      for intel processors (binding to the pardiso solver)
    
    * `Petsc4Py <https://pypi.org/project/petsc4py/>`_
      mainly compatible with linux or macos including the MUMPS solver.

    * `Scikit-umfpack <https://scikit-umfpack.github.io/scikit-umfpack/>`_

To be able to launch the fedoo viewer, the module 
`pyvistaqt <https://qtdocs.pyvista.org/>`_ is also required.

As mentioned earlier, a lot of features (finite strain, non-linear
constitutive laws, ...) requires the installation of simcoon. Simcoon can
be installed from pip or conda. To install simcoon using conda:

.. code-block:: none

    $ conda install -c conda-forge -c set3mah simcoon
    