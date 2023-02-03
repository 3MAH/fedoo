Overview
=================================

About
______________

.. image:: _static/fedOOLogos.png

FEDOO is an open source Finite Element library developed in Python.
It has originally been developed to allow easy creation and resolution of 
problems with the Proper Orthogonal Decomposition algorithm (based on a 
separated decomposition) and is now also mainly developed for standard 
finite element features. 

It is mainly dedicated for mechanical problems but is easily developpable for other kind of problems (thermal laws already includes).

One of the main application of FEDOO is to simulate the mechanical behavior of microstructures VER behavior, in the context of Architectured materials. 


Main features
______________


* FEDOO is written in Python 3
* Implicit finite element Solver for Static and Dynamics poblems
* Geometrical non linearities
* Based on the SIMCOON library for finite strain constitutive laws (simcoon is developped in C++ allowing a fast execution)
* Constitutive equation library including elasto-plastic law, composites law, ...
* Include many type of elements like cohesive elements, 2D, 3D, beam, plate, ...
* Homogeneisation: Easy application of periodic boundary conditions and fast automatic extraction of mean tangeant matrices
* Embedded results visualization using the powerfull pyvista library
* Multi-point constraints
* Mesh import/export from VTK and GMSH formats (and others) 
* Easy scripting
* Good compromise between a reasonable execution time (not the fastest) and an open and lisible code. 
* And many other....


Road map
______________

* Thermo-Mechanical problems (almost done)
* Contact algorithm (and especially self contact)

