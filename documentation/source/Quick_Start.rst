Quick Start
=================================

Steps to define and solve a Problem
___________________________________

The main steps to define a problem are:

1. Import the fedoo library
2. Define the space dimension 
3. Create and/or import the geometry (mesh)
4. Define one or many weak formulations (including constitutive equations)
5. Create the Assembly associated to the weak formulations and the mesh
6. Define the type of Problem (Static, Dynamic, ...) and the solver
7. Solve the Problem
8. Analyse and export results (vizualisation with Paraview recommanded)


Import library
______________

The first step is to import the fedoo library. 

.. code-block:: none

   from fedoo import * #Import all the fedoo library
   import numpy as np #Numpy is often usefull
   

Define the space dimension
___________________________

The dimension of the problem should be '3D' or '2Dplane' (for planar problem)
The dimension of the problem may also be set to '2Dstress' to set
plane stress assumption by default 

.. code-block:: none

    Util.ProblemDimension("3D")


Create and/or import the geometry (mesh)
_________________________________________

The module :doc:`Mesh <Mesh>` contains several functions to build up simple meshes. 
For instance to build the mesh of a box, we can use: 

.. code-block:: none

    Mesh.BoxMesh(Nx=31, Ny=21, Nz=21, x_min=0, x_max=1000, y_min=0, y_max=100, z_min=0, z_max=100, ElementShape = 'hex8', ID = 'Domain')

You can also import a Mesh from the VTK or GMSH format with the function: 
Mesh.ImportFromFile (see :mod:`fedoo.libMesh.MeshImport.ImportFromFile`)


Create weak formulations 
___________________________


Create global matrix assembies
__________________________________



Set the Problem and the solver
________________________________


Boundary conditions
_____________________



Solve the Problem
__________________________________



Analyse and export results
________________________________