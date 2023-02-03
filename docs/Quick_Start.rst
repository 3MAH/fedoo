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

The first step is to import the fedoo library. Obvisously, fedoo needs to be :doc:`installed <Install>`.

.. code-block:: python

   import fedoo as fd #Import all the fedoo library
   import numpy as np #Numpy is often usefull
   

Create the modeling space
___________________________

In fedoo, most of objects are associated to a modeling space. The modeling space is a space in which the variables and coordinates are created. 
The dimension of the modeling space car be '3D' or '2D' (for planar problem). The dimension of the space may also be set to '2Dstress' for planar problem based on the plane stress assumption.

.. code-block:: python

    fd.ModelingSpace("3D") 


Create and/or import the geometry (mesh)
_________________________________________

The module :ref:`mesh <build_simple_mesh>` contains several functions to build up simple meshes. 
For instance to build the mesh of a box (3d rectangle), we can use: 

.. code-block:: python

    fd.mesh.box_mesh(nx=31, ny=21, nz=21, x_min=0, x_max=1000, y_min=0, y_max=100, z_min=0, z_max=100, elm_type = 'hex8', name = "Domain") 

You can also import a Mesh by using one of the following Mesh constructor :py:meth:`fedoo.Mesh.read`, :py:meth:`fedoo.Mesh.from_pyvista` or with the function fd.mesh.import_file (see :ref:`importmesh`) that allow to import meshes from the VTK or GMSH format.


Create weak formulations 
___________________________

We need to define one or several equations to solve. 
In a finite element analyses, the equations are written using a weak formulations. 
In fedoo, the equation is defined by creating a WeakForm object. 

The type of WeakForm may be related to a specific type of elements. 

For instance, a beam weak formulation should be associated to beam elements.
The module :doc:`weakform <WeakForm>` list the available type of weak formulation. 
Most of weak formulations needs the definition of a material constitutive law.
The module :doc:`constitutivelaw <ConstitutiveLaw>` list the available ConstitutiveLaw.

For instance a simple weak formulation may be defined by:

.. code-block:: python
    
    fd.constitutivelaw.ElasticIsotrop(200e3, 0.3, name = 'ElasticLaw')
    fd.weakform.InternalForce("ElasticLaw", name = "MyWeakForm")


Create global matrix assembies
__________________________________

Once mesh and weak formulations have been created, we can now proceed to the assembly of the global matrices.
For each weak formulation associated to a mesh, an Assembly object needs to be created. To combine several assembly, there is a dedicated constructor :py:meth:`fedoo.Assembly.sum`.

For instance, a simple assembly for the previously defined weak formulation and mesh is:

.. code-block:: python
    
    fd.Assembly.create("MyWeakForm", 'Domain', name = "MyAssembly") 


Set the Problem and the solver
________________________________

.. code-block:: python
    
    pb = fd.problem.Static("MyAssembly") 



.. code-block:: python

    Problem.SetSolver('cg') #for conjugate gradient solver


Boundary conditions
_____________________

To apply the boundary conditions, we need to define some list of nodes indices for boundaries.
The boundaries are automatically stored as set of nodes ('left', 'right', 'top', 'bottom', ...) when the function Mesh.BoxMesh is used.

.. code-block:: python

    nodes_left = fd.Mesh['Domain'].GetSetOfNodes("left")
    nodes_right = fd.Mesh['Domain'].GetSetOfNodes("right")

An easy way to get some set of nodes at a given position is to use the numpy function where altogether to a condition on the node coordiantes.
For instance, to get the left and right list of nodes with a 1e-10 position tolerance: 

.. code-block:: python
    
    crd = Mesh['Domain'].nodes #Get the coordinates of nodes
    X = crd[:,0] #Get the x position of nodes
    x_min = X.min() 
    x_max = X.max()
    
    #extract a list of nodes using the numpy np.where function
    nodes_left = np.where(np.abs(X - xmin) < 1e-10)[0]
    nodes_right = np.where(np.abs(X - xmax) < 1e-10)[0]
    
Once the list of nodes have been defined, the boundary conditions can be applied with the
Problem.BoundaryCondition function. 

.. code-block:: python

    Problem.BoundaryCondition('Dirichlet','DispX',0,nodes_left)
    Problem.BoundaryCondition('Dirichlet','DispY', 0,nodes_left)
    Problem.BoundaryCondition('Dirichlet','DispZ', 0,nodes_left)
    
    Problem.BoundaryCondition('Dirichlet','DispY', -10, nodes_right)

To apply the boundary conditions to the active problem use the command: 

.. code-block:: python

    Problem.ApplyBoundaryCondition()



Solve the Problem
__________________________________



Analyse and export results
________________________________