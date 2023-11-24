Overview
=================================

About
______________

.. image:: _static/fedOOLogos.png

Fedoo is a free open source Finite Element library developed in Python.
It is mainly dedicated for mechanical problems but is easily developpable for other kind of problems (thermal laws already includes).
One of the main application of FEDOO is to simulate the mechanical response of heterogeneous materials. 
For that purpose, fedoo is part of the 3mah set that also include microgen for the CAD and meshing of heterogeneous materials 
and simcoon for the non linear material constitutive models in finite strain. 


Main features
______________

* Entirely written in Python 3
* Implicit finite element Solver for Static and Dynamics poblems
* Finite strain constitutive laws based on the simcoon library (simcoon is developped in C++ allowing a fast execution)
* Geometrical non linearities based on the simcoon library 
* Wide constitutive equation library including composites law, elasto-plastic law, ...
* Include many type of elements like 2D, 3D, beam, plate, cohesive elements, ...
* Homogeneisation: Easy application of periodic boundary conditions and fast automatic extraction of mean tangent matrices
* Embedded results visualization using the powerfull pyvista library (pyvista should be installed for that feature)
* Multi-point constraints
* Easy Mesh import/export from VTK and GMSH formats (and others) 
* Easy scripting
* Good compromise between a reasonable execution time and an open and lisible code. fedoo is not the fastest finite element software and doesn't intend to be, but a particular attention is paid 
to the computational cost.
* Contact in 2D (contact algorithm will be improved and extended to 3D in future versions)
* And many other....


Road map
______________

* Thermo-Mechanical problems (almost done)
* Contact in 3D, Contat with friction

