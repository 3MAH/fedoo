=====================================
Boundary conditions and constraints
=====================================

.. currentmodule:: fedoo


Basic boundary conditions
==============================

There is 3 basic types of boundary conditions in fedoo:

    #. Dirichlet boundary conditions: Value specified for a degree of freedom
       (for instance displacement).
    #. Neumann boundary conditions: Value specified for the dual value of a
       degree of freedom (for instance force)
    #. Multi-point constraints (MPC): linear coupling between degrees of
       freedom. MPC are not properly speaking boundary conditions.
       In fedoo they are considered as part of boundary conditions, because
       they often used to specify complexe boundary conditions, but they are
       more general that simple boundary conditions.


Each time a problem is created, a list of boundary conditions is associated to
the problem in the "*bc*" attribute of the problem instance.

problem.bc is then a :py:class:`fedoo.ListBC` Object.

.. autosummary::
   :toctree: generated/
   :template: custom-class-template.rst
   
   fedoo.ListBC



Dirichlet and Neumann Boundary conditions
____________________________________________

The recommanded way to apply some boundary conditions is to use the method
:py:meth:`fedoo.ListBC.add`, directly from problem attribute
:py:attr:`fedoo.Problem.bc`.
This method build a :py:class:`fedoo.core.BoundaryCondition` object and add it to
the problem boundary conditions.

.. autosummary::
   :toctree: generated/
   :template: custom-class-template.rst

   fedoo.core.BoundaryCondition

Here are some examples to add some Dirichlet or Neumann Boundary conditions
for a given problem  *pb*:

Displacement along the X axis blocked (=0) for the nodes 0, 1 and 8

>>> pb.bc.add('Dirichlet',[0,1,8], 'DispX', 0)

Displacement along the all axis blocked (=0) for the nodes defined by the
node sets 'node_sets' (the given node nets is supposed to be present in the
associated mesh. If note, it will raise an error).

>>> pb.bc.add('Dirichlet','node_sets', 'Disp', 0)

External force on the node 112 with Fx = 5, Fy = 10 and Fz = 0.

>>> pb.bc.add('Neumann', [112], 'Disp', [5, 10, 0])

Enforce a nul displacement on nodes 1, 2 and 3, and a rotation around X = 0
for node 1, 0.1 for node 2 and 0.2 for node 3.

>>> pb.bc.add('Dirichlet',
>>>           [1,2,3],
>>>           ['Disp','RotX'],
>>>           [0, np.array([0,0.1,0.2])]
>>>          )

Instead of adding directly the boundary conditions to the problem, it
may be convenient to create boundary conditions and add them afterwards

>>> my_bc = fd.BoundaryCondition.create('Dirichlet', [1,2,3], 'Disp', 0)])
>>> pb.bc.add(my_bc)

For non linear problem, we can also defined the time_func to sepcify how the
boundary condition evolve during time. By default, a linear evolution is
enforced.
The time function depend on the time_factor. The time_factor is 0 at the
begining and 1 at the end of the iteration. The time function is also a
factor to the prescribed value and should be between 0 and 1.

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

Distributed loads are defined by dedicated assembly objects. The recommended
way to apply them is to add the load assembly to the problem boundary
conditions. If the assembly provides an ``as_neumann()`` method,
:py:meth:`fedoo.ListBC.add` automatically converts it to an equivalent
Neumann boundary condition. This is especially useful for nonlinear problems:
the load is included in the external force vector, follows the usual load
coefficient, and is taken into account by the residual normalization.

.. code-block:: python

    solid_assembly = fd.Assembly.create(wf, mesh)
    pressure = fd.constraint.Pressure(surface_mesh, value, nlgeom=True)

    pb = fd.problem.NonLinear(solid_assembly, nlgeom=True)
    pb.bc.add(pressure)

The explicit form is also available:

.. code-block:: python

    pb.bc.add(pressure.as_neumann())

The load assembly can still be combined directly with the mechanical assembly.
This form is kept as an alternative, but is not recommended for nonlinear
analyses because the load is part of the residual assembly rather than a
boundary-condition force.

.. code-block:: python

    pb = fd.problem.Linear(solid_assembly + pressure)

.. autosummary::
    :toctree: generated/
    :template: custom-class-template.rst

    fedoo.constraint.DistributedForce
    fedoo.constraint.Pressure
    fedoo.constraint.SurfaceForce

Some advanced constraints based on multiple linearized MPC are available in
fedoo. They can be created and added to the problem with the pb.bc.add method.

Mean motion control
___________________

The :py:class:`fedoo.constraint.MeanMotion` constraint is a less
restrictive alternative to :py:class:`fedoo.constraint.RigidTie`. It extracts
selected components of the best-fit motion of a node set or surface mesh in a
mean sense, without forcing every selected node to follow a rigid motion. The
selected surface can therefore deform locally while selected mean translation
and rotation components are controlled.

The ``components`` argument is required and defines which global DOFs are
created. Accepted component names include ``"RotX"``, ``"MeanRotX"``,
``"DispZ"`` and ``"MeanDispZ"``. The vector aliases ``"Rot"``/``"MeanRot"``
and ``"Disp"``/``"MeanDisp"`` select all rotation or displacement components
available in the current modeling dimension. In 2D, the rotation vector
contains only ``MeanRotZ``.

When a surface mesh is passed, nodal weights are computed from the element
area, or from the element length in 2D. These weights define an area-weighted
least-squares projection of the displacement field onto rigid-body modes.

For displacement-only mean control, select only displacement components:

.. code-block:: python

    mean_disp = fd.constraint.MeanMotion(surface_mesh, components="DispZ")
    pb.bc.add(mean_disp)
    pb.bc.add("Dirichlet", "MeanDispZ", -delta)

When ``finite_rotation`` is left to ``None`` (the default), rotational
components use finite rotations automatically if geometrical nonlinearity is
enabled on the problem. Otherwise, the rotation components are small-rotation
vector components. The finite-rotation mode uses a rotation-vector convention
and linearizes the constraint at each Newton iteration using the derivative of
the rotation matrix with respect to the rotation vector.

.. code-block:: python

    face_motion = fd.constraint.MeanMotion(
        surface_mesh,
        components="RotZ",
    )
    pb.bc.add(face_motion)
    pb.bc.add("Dirichlet", face_motion.node_by_variable["MeanRotZ"], "MeanRotZ", angle)

.. code-block:: python

    face_motion = fd.constraint.MeanMotion(
        surface_mesh,
        components=["RotZ", "DispZ"],
    )
    pb.bc.add(face_motion)
    pb.bc.add("Dirichlet", "MeanRotZ", angle)
    pb.bc.add("Dirichlet", "MeanDispZ", -delta)

The reactions on the created global DOFs are generalized forces. For instance,
the reaction associated with ``MeanRotZ`` is the conjugate moment around the
mean-motion center.

.. autosummary::
   :toctree: generated/
   :template: custom-class-template.rst

   fedoo.constraint.MeanMotion
   fedoo.constraint.PeriodicBC
   fedoo.constraint.RigidTie
   fedoo.constraint.RigidTie2D


Contact
==============================

Fedoo provides two contact approaches: a **penalty-based** method and an
**IPC (Incremental Potential Contact)** method. Both are implemented as
assembly objects and can be combined with other assemblies using
:py:meth:`fedoo.Assembly.sum`.

Penalty-based contact
______________________

The penalty method uses a node-to-surface formulation. It is available for
2D problems and supports frictionless contact.

.. autosummary::
   :toctree: generated/
   :template: custom-class-template.rst

   fedoo.constraint.Contact
   fedoo.constraint.SelfContact

The contact class is derived from assembly. To add a contact contraint
to a problem, we need first to create the contact assembly (using the class
:py:class:`fedoo.constraint.Contact`) and then to add it to the global
assembly with :py:meth:`fedoo.Assembly.sum`.


IPC contact
______________________

The IPC (Incremental Potential Contact) method uses barrier potentials from
the `ipctk <https://ipctk.xyz/>`_ library to guarantee intersection-free
configurations.  It supports both 2D and 3D problems, friction, and optional
CCD (Continuous Collision Detection) line search.

Unlike the penalty method, IPC does **not** require tuning a penalty
parameter.  The barrier stiffness :math:`\kappa` is automatically computed
and adaptively updated to balance the elastic and contact forces.  The only
physical parameter is ``dhat`` — the barrier activation distance that
controls the minimum gap between surfaces (default: 0.1% of the bounding
box diagonal).

**Choosing dhat** --- The default ``dhat=1e-3`` (relative) means the
barrier activates when surfaces are within 0.1 % of the bounding-box
diagonal.  For problems with a very small initial gap, increase ``dhat``
(e.g. ``1e-2``) so the barrier catches contact early.  For tight-fitting
assemblies where a visible gap is unacceptable, decrease it (e.g.
``1e-4``), but expect more Newton–Raphson iterations.

**CCD line search** --- Enabling ``use_ccd=True`` is recommended for
problems where first contact occurs suddenly (e.g. a punch hitting a
plate) or where self-contact can cause rapid topology changes.

**Energy-based backtracking** --- When ``use_ccd=True``, an
energy-based backtracking phase is automatically enabled after CCD
filtering: the step is halved until total energy (exact barrier +
quadratic elastic approximation) decreases.  This matches the
reference IPC algorithm and improves convergence robustness.  Set
``line_search_energy=False`` to disable (faster but may degrade
convergence for difficult contact scenarios).

**Convergence criterion** --- The ``'Force'`` convergence criterion
is recommended for IPC contact problems.  It measures the relative
decrease of the force residual, matching the gradient-norm convergence
used by reference IPC implementations::

    pb.set_nr_criterion('Force', tol=5e-3, max_subiter=15)

The ``'Displacement'`` criterion may become unreliable as contact
stiffness grows.

The ``ipctk`` package is required and can be installed with:

.. code-block:: bash

   pip install ipctk
   # or
   pip install fedoo[ipc]

.. autosummary::
   :toctree: generated/
   :template: custom-class-template.rst

   fedoo.constraint.IPCContact
   fedoo.constraint.IPCSelfContact


Penalty contact example
__________________________

Here an example of a contact between a square and a disk using the penalty method.

.. code-block:: python


    import fedoo as fd
    import numpy as np

    fd.ModelingSpace("2D")

    filename = 'disk_rectangle_contact' #file to save results

    #---- Create geometries --------------
    mesh_rect = fd.mesh.rectangle_mesh(nx=11, ny=21,
                                       x_min=0, x_max=1, y_min=0, y_max=1,
                                       elm_type = 'quad4', name = 'Domain'
                                      )
    mesh_rect.element_sets['rect'] = np.arange(0, mesh_rect.n_elements)
    mesh_disk = fd.mesh.disk_mesh(radius=0.5, nr=6, nt=6, elm_type = 'quad4')
    mesh_disk.nodes+=np.array([1.5,0.48]) # translate disk on the right
    mesh_disk.element_sets['disk'] = np.arange(0,mesh_disk.n_elements)

    mesh = fd.Mesh.stack(mesh_rect,mesh_disk)

    #node sets for boundary conditions
    nodes_left = mesh.find_nodes('X',0)
    nodes_right = mesh.find_nodes('X',1)

    nodes_bc = mesh.find_nodes('X>1.5')
    nodes_bc = list(set(nodes_bc).intersection(mesh.node_sets['boundary']))

    #---- Define contact --------------
    #slave surface = right face of rectangle mesh
    nodes_contact = nodes_right
    surf = fd.mesh.extract_surface(mesh.extract_elements('disk'))
    contact = fd.constraint.Contact(nodes_contact, surf)
    contact.contact_search_once = True
    contact.eps_n = 5e5

    #---- Material properties --------------
    props = np.array([200e3, 0.3, 1e-5, 300, 1000, 0.3])
    # E, nu, alpha (non used), Re, k, m
    material_rect = fd.constitutivelaw.Simcoon("EPICP", props)
    material_disk = fd.constitutivelaw.ElasticIsotrop(50e3, 0.3) #E, nu
    material = fd.constitutivelaw.Heterogeneous(
        (material_rect, material_disk),
        ('rect', 'disk')
        )

    #---- Build problem --------------
    wf = fd.weakform.StressEquilibrium(material, nlgeom = True)
    solid_assembly = fd.Assembly.create(wf, mesh)

    assembly = fd.Assembly.sum(solid_assembly, contact)

    pb = fd.problem.NonLinear(assembly)
    results = pb.add_output(filename,
                            solid_assembly,
                            ['Disp', 'Stress', 'Strain', 'Fext']
                            )

    pb.bc.add('Dirichlet',nodes_left, 'Disp',0)
    pb.bc.add('Dirichlet',nodes_bc, 'Disp', [-0.4,0.2])

    pb.set_nr_criterion("Displacement", tol = 5e-3, max_subiter = 5)

    #---- Solve problem in two steps: load, unload --------------
    pb.nlsolve(dt = 0.005, tmax = 1, update_dt = True, interval_output = 0.01)

    pb.bc.remove(-1) #remove last boundary contidion
    pb.bc.add('Dirichlet',nodes_bc, 'Disp', [0,0])

    pb.nlsolve(dt = 0.005, tmax = 1, update_dt = True, interval_output = 0.01)


    # =============================================================
    # Example of plots with pyvista - uncomment the desired plot
    # =============================================================

    # ------------------------------------
    # Simple plot with default options
    # ------------------------------------
    results.plot('Stress', component='vm', data_type='Node')
    results.plot('Disp', component = 0, data_type='Node')

    # ------------------------------------
    # Write movie with default options
    # ------------------------------------
    results.write_movie(filename,
                        'Stress',
                        component = 'XX',
                        data_type = 'Node',
                        framerate = 24,
                        quality = 5,
                        clim = [-3e3, 3e3]
                       )

Video of results:
:download:`contact video <./_static/examples/disk_rectangle_contact.mp4>`


IPC contact example
__________________________

The same disk-rectangle contact problem can be solved with the IPC method.
The IPC method does not require choosing slave/master nodes or tuning a
penalty parameter. The barrier stiffness is automatically computed and
adapted.

.. code-block:: python

    import fedoo as fd
    import numpy as np

    fd.ModelingSpace("2D")

    #---- Create geometries (same as penalty example) --------------
    mesh_rect = fd.mesh.rectangle_mesh(nx=11, ny=21,
                                       x_min=0, x_max=1, y_min=0, y_max=1,
                                       elm_type='quad4', name='Domain')
    mesh_disk = fd.mesh.disk_mesh(radius=0.5, nr=6, nt=6, elm_type='quad4')
    mesh_disk.nodes += np.array([1.5, 0.48])
    mesh = fd.Mesh.stack(mesh_rect, mesh_disk)

    #---- Define IPC contact --------------
    surf = fd.mesh.extract_surface(mesh)
    ipc_contact = fd.constraint.IPCContact(
        mesh, surface_mesh=surf,
        friction_coefficient=0.3,     # Coulomb friction coefficient
        use_ccd=True,                 # enable CCD line search for robustness
    )
    # barrier_stiffness is auto-computed; dhat defaults to 1e-3 * bbox_diag

    #---- Material and problem setup --------------
    material = fd.constitutivelaw.ElasticIsotrop(200e3, 0.3)
    wf = fd.weakform.StressEquilibrium(material, nlgeom=True)
    solid_assembly = fd.Assembly.create(wf, mesh)
    assembly = fd.Assembly.sum(solid_assembly, ipc_contact)

    pb = fd.problem.NonLinear(assembly)
    res = pb.add_output('results', solid_assembly, ['Disp', 'Stress'])
    # ... add BCs ...
    pb.nlsolve(dt=0.005, tmax=1)

.. note::

   When using ``add_output``, pass the **solid assembly** (not the sum).
   ``AssemblySum`` objects cannot be used directly for output extraction.


IPC self-contact example
__________________________

For self-contact problems, use :py:class:`~fedoo.constraint.IPCSelfContact`
which automatically extracts the surface from the volumetric mesh.

.. code-block:: python

    import fedoo as fd
    import numpy as np

    fd.ModelingSpace("3D")

    mesh = fd.Mesh.read("gyroid.vtk")
    material = fd.constitutivelaw.ElasticIsotrop(1e5, 0.3)

    # Self-contact: auto surface extraction, auto barrier stiffness
    # Add line_search_energy=True for extra robustness (slower)
    contact = fd.constraint.IPCSelfContact(mesh, use_ccd=True)

    wf = fd.weakform.StressEquilibrium(material, nlgeom="UL")
    solid = fd.Assembly.create(wf, mesh)
    assembly = fd.Assembly.sum(solid, contact)

    pb = fd.problem.NonLinear(assembly)
    res = pb.add_output("results", solid, ["Disp", "Stress"])
    # ... add BCs ...
    pb.nlsolve(dt=0.05, tmax=1, update_dt=True)
