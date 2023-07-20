
=====================================
Boundary conditions and constraints
=====================================

.. currentmodule:: fedoo


Basic boundary conditions
==============================

There is 3 basic types of boundary conditions in fedoo:

    #. Dirichlet boundary conditions: Value specified for a degree of freedom 
       (for instance displacement).
    #. Neumann boundary conditions: Value specified for the dual value of 
       a degree of freedom (for instance force)
    #. Multi-point constraints (MPC): linear coupling between degrees of freedom. 
       MPC are not properly speaking boundary conditions. 
       In fedoo they are considered as part of boundary conditions, because they often
       used to specify complexe boundary conditions, but they are more general
       that simple boundary conditions. 
       

Each time a problem is created, a list of boundary conditions is associated to
the problem in the "*bc*" attribute of the problem instance.

problem.bc is then a :py:class:`fedoo.ListBC` Object.

.. autoclass:: fedoo.ListBC
    :members:



Dirichlet and Neumann Boundary conditions
____________________________________________

Here are some examples to add some Dirichlet or Neumann Boundary conditions for
a given problem  *pb*:

Displacement along the X axis blocked (=0) for the nodes 0, 1 and 8

>>> pb.bc.add('Dirichlet',[0,1,8], 'DispX', 0) 

Displacement along the all axis blocked (=0) for the nodes defined by the node sets
'node_sets' (the given node nets is supposed to be present in the associated mesh. 
If note, it will raise an error)

>>> pb.bc.add('Dirichlet','node_sets', 'Disp', 0)

External force on the node 112 with Fx = 5, Fy = 10 and Fz = 0.

>>> pb.bc.add('Neumann', [112], 'Disp', [5, 10, 0])

Enforce a nul displacement on nodes 1, 2 and 3, and a rotation around X = 0 for node 1, 0.1 for node 2 
and 0.2 for node 3.

>>> pb.bc.add('Dirichlet', [1,2,3], ['Disp','RotX'], [0, np.array([0,0.1,0.2])])

Instead of adding directly the boundary conditions to the problem, it
may be convenient to create boundary conditions and add them afterwards

>>> my_bc = fd.BoundaryCondition.create('Dirichlet', [1,2,3], 'Disp', 0)])
>>> pb.bc.add(my_bc)

For non linear problem, we can also defined the time_func to sepcify how the 
boundary condition evolve during time. By default, a linear evolution is 
enforced. 
The time function depend on the time_factor. The time_factor is 0 at the begining and 
1 at the end of the iteration. The time function is also a factor to the prescribed value 
and should be between 0 and 1.

>>> def step_function(t_fact): 
>>>     if t_fact == 0: return 0
>>>     else: return 1
>>> pb.bc.add('Dirichlet', 'nodeset', 'Disp', 1, time_func = step_function)])



Multi Point Constraints
____________________________________________

.. autosummary::   
   :toctree: generated/
   :template: custom-class-template.rst

   fedoo.MPC


Avdanced BC and constraints
==============================


Some advanced constraints base on multiple linearized MPC are available in fedoo. 
They can be created and add to the problem with the pb.bc.add method.

.. autosummary::   
   :toctree: generated/
   :template: custom-class-template.rst

   fedoo.constraint.PeriodicBC
   fedoo.constraint.RigidTie 
   

A first implementation of the contact is also proposed. Contact is still 
in development and subject to change in future versions.
   
.. autosummary::   
   :toctree: generated/
   :template: custom-class-template.rst
   
   fedoo.constraint.Contact   
   
The contact class is derived from assembly. To add a contact contraint to a problem,
we need first to create the contact assembly (using the class :py:class:`fedoo.constraint.Contact`)
and then to add it to the global assembly with :py:meth:`fedoo.Assembly.sum`.





