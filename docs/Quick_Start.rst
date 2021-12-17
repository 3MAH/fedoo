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
   

Create the modeling space
___________________________

In fedoo, most of objects are associated to a modeling space. The modeling space is a space in which the variables and coordinates are created. 
The most convenient way to create a modeling space is to define the dimension of the problem using the Util.ProblemDimension(dimension) method. 
The dimension of the problem should be '3D' or '2Dplane' (for planar problem). The dimension of the problem may also be set to '2Dstress' for planar problem based on the plane stress assumption.

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

We need to define one or several equations to solve. 
In a finite element analyses, the equations are written using a weak formulation. 
The type of weak formulation may be related to a specific type of elements. 

For instance, a beam weak formulation should be associated to beam elements.
The module :doc:`WeakForm <WeakForm>` list the available type of weak formulation. 
Most of weak formulations needs the definition of a material constitutive law.
The module :doc:`ConstitutiveLaw <ConstitutiveLaw>` list the available ConstitutiveLaw.

For instance a simple weak formulation may be defined by:

.. code-block:: none
    
    ConstitutiveLaw.ElasticIsotrop(200e3, 0.3, ID = 'ElasticLaw')
    WeakForm.InternalForce("ElasticLaw", ID="MyWeakForm")


Create global matrix assembies
__________________________________

Once mesh and weak formulations have been created, we can now proceed to the assembly of the global matrices.
For each weak formulation associated to a mesh, an Assembly object needs to be created. To combine several assembly, there is a dedicated function Assembly.Sum(ListAssembly).

For instance, a simple assembly for the previously defined weak formulation and mesh is:

.. code-block:: none
    
    Assembly.Create("MyWeakForm", 'Domain', ID="MyAssembly") 


Set the Problem and the solver
________________________________

.. code-block:: none
    
    Problem.Static("MyAssembly") 



.. code-block:: none

    Problem.SetSolver('cg') #for conjugate gradient solver


Boundary conditions
_____________________

To apply the boundary conditions, we need to define some list of nodes indices for boundaries.
The boundaries are automatically stored as set of nodes ('left', 'right', 'top', 'bottom', ...) when the function Mesh.BoxMesh is used.

.. code-block:: none

    nodes_left = Mesh.GetAll()['Domain'].GetSetOfNodes("left")
    nodes_right = Mesh.GetAll()['Domain'].GetSetOfNodes("right")

An easy way to get some set of nodes at a given position is to use the numpy function where altogether to a condition on the node coordiantes.
For instance, to get the left and right list of nodes with a 1e-10 position tolerance: 

.. code-block:: none
    
    crd = Mesh.GetAll()['Domain'].GetNodeCoordinates() #Get the coordinates of nodes
    X = crd[:,0] #Get the x position of nodes
    x_min = X.min() 
    x_max = X.max()
    
    #extract a list of nodes using the numpy np.where function
    nodes_left = np.where(np.abs(X - xmin) < 1e-10)[0]
    nodes_right = np.where(np.abs(X - xmax) < 1e-10)[0]
    
Once the list of nodes have been defined, the boundary conditions can be applied with the
Problem.BoundaryCondition function. 

.. code-block:: none

    Problem.BoundaryCondition('Dirichlet','DispX',0,nodes_left)
    Problem.BoundaryCondition('Dirichlet','DispY', 0,nodes_left)
    Problem.BoundaryCondition('Dirichlet','DispZ', 0,nodes_left)
    
    Problem.BoundaryCondition('Dirichlet','DispY', -10, nodes_right)

To apply the boundary conditions to the active problem use the command: 

.. code-block:: none

    Problem.ApplyBoundaryCondition()



Solve the Problem
__________________________________



Analyse and export results
________________________________