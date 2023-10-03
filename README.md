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

Documentation
--------------
Provider      | Status
--------      | ------
Read the Docs | [![Documentation Status](https://readthedocs.org/projects/fedoo/badge/?version=latest)](https://fedoo.readthedocs.io/en/latest/?badge=latest)


Installation
--------------
Installation with pip including recommanded dependencices:
```
pip install fedoo[all]
```

Minimal installation with pip:
```
pip install fedoo
```

Installation with conda: 
```
conda install -c conda-forge -c set3mah fedoo
```
