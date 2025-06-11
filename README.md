# fedoo
Finite element library
[![FEDOO Logo](https://github.com/3MAH/fedoo/blob/master/fedOOLogos.png)](https://github.com/3MAH/fedoo)

About
-----

Fedoo is an open source Finite Element library developed in Python.
It is mainly dedicated for mechanical problems but is easily developpable for other kind of problems (thermal laws already included).
One of the main application of fedoo is to simulate the mechanical response of heterogeneous materials. 
For that purpose, fedoo is part of the 3mah set that also include microgen for the CAD and meshing of heterogeneous materials 
and simcoon for the non linear material constitutive models in finite strain. 

Here are the main features:

- Entirely written in Python 3
- Implicit finite element Solver for Static and Dynamics poblems
- Finite strain constitutive laws based on the simcoon library (simcoon is developped in C++ allowing a fast execution)
- Geometrical non linearities based on the simcoon library 
- Wide constitutive equation library including composites law, elasto-plastic law, ...
- Include many types of elements like 2D, 3D, beam, plate, cohesive elements, ...
- Homogeneisation: Easy application of periodic boundary conditions and fast automatic extraction of mean tangent matrices
- Embedded results visualization using the powerfull pyvista library
- Multi-point constraints
- Easy scripting
- Good compromise between a reasonable execution time and an open and lisible code. fedoo is not the fastest finite element software and doesn't intend to be, but a particular attention is paid 
  to the computational cost.
- Contact in 2D and 3D, Self contact
- And many other....

|               |                                                                                                                                        |
|---------------|----------------------------------------------------------------------------------------------------------------------------------------|
| PyPI package  | [![PyPI version](https://badge.fury.io/py/fedoo.svg)](https://badge.fury.io/py/fedoo)                                                  |
| Conda package | [![Conda](https://anaconda.org/set3mah/fedoo/badges/version.svg)](https://anaconda.org/set3mah/fedoo)                                  |
| Documentation | [![Documentation](https://readthedocs.org/projects/fedoo/badge/?version=latest)](https://fedoo.readthedocs.io/en/latest/?badge=latest) |
| License       | [![License](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)                                |
| Website       | [![Website](https://img.shields.io/badge/website-3MAH-blue)](https://3mah.github.io/)                                                  |


Documentation
--------------
The [documentation](https://fedoo.readthedocs.io/en/latest/?badge=latest) is
provided by readthedocs at
[https://fedoo.readthedocs.io](https://fedoo.readthedocs.io).


Installation
--------------
Installation with conda with recommanded dependencices (including simcoon): 
```
conda install -c conda-forge -c set3mah fedoo
```

Minimal installation with pip:
```
pip install fedoo
```

Installation with pip including recommanded dependencies excepted simcoon:
```
pip install fedoo[all]
```

In mac OS make sure to use petsc4py (included in the conda package) or
scikit-umfpack. To install scikit-umfpack, it is recommanded to use pip
(the conda package is outdated):
```
pip install scikit-umpack>=0.41 
```

For performance considerations, it is in general strongly recommended to make
sure that numpy is correctly installed using a fast implementation of BLAS
(*MKL* for windows or linux or *Accelerate* for mac OS).

In mac OS, the best way to take the most of MacOs accelerate framework
is to install numpy from sources first:
```
pip install cython pybind11
pip install --no-binary :all: numpy
```

Also, make sure that the default number of threads is not leading to
performance degradations: using explicit number of threads might help,
see below for extensive control of threads number.
```
#Set the number of threads
import os

n_threads = 4
os.environ["OMP_NUM_THREADS"] = f"{n_threads}"
os.environ["OPENBLAS_NUM_THREADS"] = f"{n_threads}"
os.environ["MKL_NUM_THREADS"] = f"{n_threads}"
os.environ["VECLIB_MAXIMUM_THREADS"] = f"{n_threads}"
os.environ["NUMEXPR_NUM_THREADS"] = f"{n_threads}"
```
