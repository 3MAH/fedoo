# fedoo
Finite element library
[![FEDOO Logo](https://github.com/3MAH/fedoo/blob/master/fedOOLogos.png)](https://github.com/3MAH/fedoo)

About
-----

FEDOO is an open source Finite Element library developed in Python. It has originally been developed to allow easy creation and resolution of problems with the Proper Orthogonal Decomposition algorithm (based on a separated decomposition) and is now also actively developed for standard finite element features.

Here are the main features:

- FEDOO is entirely written in Python 3
- Resolution of problems based on a separated decomposition (PGD, POD, Reduced bases)
- Static and Dynamics poblems
- Mesh import/export from msh (GMSH) and vtk format
- Export results in vtk file for easy visualisation with Paraview (https://www.paraview.org/)
- Constitutive equation library including elasto-plastic law, composites law, ...
- Include many type of elements like cohesive elements, 2D, 3D, beam, ...
- Geometrical non linearities
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
The [documentation](https://fedoo.readthedocs.io/en/latest/?badge=latest) is provided by readthedocs at
[https://fedoo.readthedocs.io](https://fedoo.readthedocs.io).


Installation
--------------
Installation with pip including recommanded dependencies excepted simcoon:
```
pip install fedoo[all]
```

Minimal installation with pip:
```
pip install fedoo
```

Installation with conda with recommanded dependencices (including simcoon): 
```
conda install -c conda-forge -c set3mah fedoo
```
