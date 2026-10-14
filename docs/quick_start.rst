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
8. Analyse results with the Fedoo viewer and export them if needed

Import library
______________

The first step is to import the fedoo library. Fedoo needs to be :doc:`installed <install>`.

.. code-block:: python

   import fedoo as fd #Import all the fedoo library
   import numpy as np #NumPy is useful for array operations
   

Create the modeling space
___________________________

In fedoo, most of objects are associated to a modeling space. The modeling space is a space in which the variables and coordinates are created. 
Use ``'3D'`` for three-dimensional problems, ``'2Dplane'`` (or ``'2D'``)
for plane strain, ``'2Dstress'`` for plane stress, and ``'2Daxi'`` for
axisymmetric problems.

.. code-block:: python

    fd.ModelingSpace("3D")


Create and/or import the geometry (mesh)
_________________________________________

The module :ref:`mesh <build_simple_mesh>` contains several functions to build up simple meshes. 
For instance to build the mesh of a box (3d rectangle), we can use: 

.. code-block:: python

    my_mesh = fd.mesh.box_mesh(nx=11, ny=3, nz=3, x_min=0, x_max=1000, y_min=0, y_max=100, z_min=0, z_max=100, elm_type='hex8')

You can also import a mesh with :py:meth:`fedoo.Mesh.read` or
:py:meth:`fedoo.Mesh.from_pyvista`. See :ref:`importmesh` for supported formats.


Create weak formulations 
___________________________

We need to define one or several equations to solve. 
In a finite element analyses, the equations are written using a weak formulations. 
In fedoo, the equation is defined by creating a WeakForm object. 

The type of WeakForm may be related to a specific type of elements. 

For instance, a beam weak formulation should be associated to beam elements.
The module :doc:`weakform <weak_form>` list the available types of weak formulation.
Most of weak formulations needs the definition of a material constitutive law.
The module :doc:`constitutivelaw <constitutive_law>` list the available ConstitutiveLaw.

For instance a simple weak formulation may be defined by:

.. code-block:: python
    
    material = fd.constitutivelaw.ElasticIsotrop(200e3, 0.3)
    my_weakform = fd.weakform.StressEquilibrium(material)


Create global matrix assemblies
__________________________________

Once mesh and weak formulations have been created, we can now proceed to the assembly of the global matrices.
For each weak formulation associated to a mesh, an Assembly object needs to be created. To combine several assembly, there is a dedicated constructor :py:meth:`fedoo.Assembly.sum`.

For instance, a simple assembly for the previously defined weak formulation and mesh is:

.. code-block:: python
    
    my_assembly = fd.Assembly.create(my_weakform, my_mesh) 

The problem initializes the assembly and builds its global stiffness matrix
in SciPy sparse format when solving. Retrieve the matrix after solving with
``my_assembly.get_global_matrix()``.


Set the Problem and the solver
________________________________

The problem should now be defined. Here a Linear problem is created based on the previously defined Assembly.

.. code-block:: python
    
    pb = fd.problem.Linear(my_assembly)

The solver can be modified as follow:

.. code-block:: python

    pb.set_solver('cg') #for conjugate gradient solver

For direct solver, it is strongly recommended to install one of
``pypardiso`` (Intel CPUs), ``python-mumps`` (cross-platform, also
required for :meth:`Problem.set_reuse_factorization`) or ``petsc4py``
for drastically improved computation performance. See
:doc:`install` for the recommended commands per platform.

Boundary conditions
_____________________

To apply the boundary conditions, we need to define some list of nodes indices for boundaries.
The boundaries are automatically stored as sets of nodes ('left', 'right',
'top', 'bottom', ...) when ``fd.mesh.box_mesh`` is used.

.. code-block:: python

    nodes_left = my_mesh.node_sets["left"]
    nodes_right = my_mesh.node_sets["right"]

An easy way to get a set of nodes from the position of nodes is to use the :py:meth:`fedoo.Mesh.find_nodes` method.
For instance, to get the left and right list of nodes with a 1e-10 position tolerance: 

.. code-block:: python

    nodes_left = my_mesh.find_nodes('X', my_mesh.bounding_box.xmin, tol=1e-10)
    nodes_right = my_mesh.find_nodes('X', my_mesh.bounding_box.xmax, tol=1e-10)

    
Once the list of nodes have been defined, the boundary conditions can be added to the problem. 
The boundary conditions associated to a problem are stored in the attribute pb.bc.
We can add two type of boundary conditions: 'Dirichlet' to prescribed dof values at given nodes
or 'Neumann' (nodal force for mechanical problems). 

.. code-block:: python

    pb.bc.add('Dirichlet',nodes_left, 'Disp', 0) #displacement vector set to 0 on the left    
    
    pb.bc.add('Dirichlet',nodes_right, 'DispY', -10) #displacement along y set to -10 on right


Solve the Problem
__________________________________

.. code-block:: python

    pb.solve()


Analyse and export results
________________________________

Retrieve the fields from the solved problem and open them in the Fedoo viewer:

.. code-block:: python

    results = pb.get_results(my_assembly, ["Disp", "Stress", "Strain"])
    fd.viewer(results)

Install the viewer dependencies with ``pip install "fedoo[plot,gui]"``.
The viewer provides field/component selection, deformation scaling and
image export. It can also open saved results:

.. code-block:: python

    results.save("cantilever.fdh5")
    # In a later session:
    fd.viewer("cantilever.fdh5")

For external tools such as ParaView, export a VTK file with
``results.save("cantilever.vtk")``. See
:doc:`Post-Processing <post_processing>` for nonlinear time histories,
section output and more viewer options.
