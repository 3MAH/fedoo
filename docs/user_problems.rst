Define user problems
====================

Fedoo can be extended at two complementary levels:

* a weak form defines the equation assembled on the mesh;
* a constitutive law defines the material response used by a mechanical weak
  form.

For a linear equation, a :class:`~fedoo.core.diffop.DiffOp` expression and
:class:`fedoo.WeakForm` are generally sufficient. For a nonlinear or
history-dependent equation, derive from
:class:`~fedoo.core.weakform.WeakFormBase`. For a mechanical material,
:class:`~fedoo.core.mechanical3d.MechanicalUMAT` is the recommended extension
point; derive directly from :class:`~fedoo.core.mechanical3d.Mechanical3D`
only when full control over the Fedoo constitutive lifecycle is required.


Define a weak form from differential operators
------------------------------------------------

A modeling space owns the unknown fields and creates their differential
operators. The ``virtual`` property selects the test-function side of a weak
term. Ordinary Python arithmetic combines these objects into a
:class:`~fedoo.core.diffop.DiffOp` expression.

The following example defines the weak form of the Poisson equation

.. math::

   \int_\Omega \nabla \delta u \mathbin{\cdot} \nabla u\,d\Omega = 0.

.. code-block:: python

   import fedoo as fd

   space = fd.ModelingSpace("2D")
   space.new_variable("U")

   dU_dX = space.derivative("U", "X")
   dU_dY = space.derivative("U", "Y")

   equation = dU_dX.virtual * dU_dX + dU_dY.virtual * dU_dY
   weakform = fd.WeakForm(equation)

   mesh = fd.mesh.rectangle_mesh()
   assembly = fd.Assembly.create(weakform, mesh)
   problem = fd.problem.Linear(assembly)

   problem.bc.add("Dirichlet", "left", "U", 0)
   problem.bc.add("Dirichlet", "right", "U", 1)
   problem.solve()

The same construction applies to systems of equations. Declare every scalar
unknown with :meth:`fedoo.ModelingSpace.new_variable`, group components with
:meth:`fedoo.ModelingSpace.new_vector` when useful, and add the weak terms.
Mixed formulations may assign different element interpolations to different
variables.

Complete examples are available in
:download:`poison_eq.py <../examples/01-simple/poison_eq.py>` and
:download:`fluid_mechanics.py <../examples/user_equation/fluid_mechanics.py>`.


State-dependent weak forms
~~~~~~~~~~~~~~~~~~~~~~~~~~

When coefficients or residual terms depend on the current solution, derive a
class from :class:`~fedoo.core.weakform.WeakFormBase`. Its principal methods
are:

``initialize(assembly, problem)``
   Allocate values required by the formulation in ``assembly.sv``.

``update(assembly, problem)``
   Evaluate the current fields and update the residual state before assembly.

``get_weak_equation(assembly, problem)``
   Return the linearized :class:`~fedoo.core.diffop.DiffOp`, including the
   tangent and current residual terms.

``set_start(assembly, problem)``
   Commit the state at the beginning of a new time increment.

``to_start(assembly, problem)``
   Restore the beginning-of-increment state after a rejected increment.

``reset()``
   Clear the complete problem history when necessary.

The problem may have ``problem.dtime == 0`` during initialization. A custom
weak form must not divide by ``dtime`` in that state. Time-dependent terms can
be omitted until a positive increment is supplied. See
:download:`navier_stokes.py <../examples/user_equation/navier_stokes.py>` for
a complete nonlinear example.


Define a mechanical constitutive law
------------------------------------

Use ``MechanicalUMAT`` for a conventional material-point update. It handles
the Fedoo lifecycle, state allocation, temperature lookup, material local
frames, finite-rotation increments, and the transformations of strains,
stresses, and tangents. The callback only integrates the constitutive equation
in the supplied corotational material coordinates.


Recommended: ``MechanicalUMAT``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The callback has the following signature:

.. code-block:: python

   def umat(
       strain,
       dstrain,
       F0,
       F1,
       stress,
       DR,
       props,
       statev_start,
       time,
       dtime,
       wm_start,
       temperature,
       *,
       ndi,
       tangent_mode,
   ):
       ...
       return stress_new, statev_new, wm_new, tangent

Its arguments follow these conventions:

===================  =========================================================
Argument             Meaning
===================  =========================================================
``strain``           Total strain at the beginning of the increment, shape
                     ``(6, n_points)``.
``dstrain``          Current strain increment, shape ``(6, n_points)``.
``F0``, ``F1``       Beginning and current deformation gradients, shape
                     ``(3, 3, n_points)`` in a finite-strain analysis. They
                     are empty in a small-strain analysis.
``stress``           Stress at the beginning of the increment, shape
                     ``(6, n_points)``.
``DR``               Objective incremental rotation, shape
                     ``(3, 3, n_points)``.
``props``            Material properties stored as a Fortran-contiguous
                     column.
``statev_start``     Private state at the beginning of the increment, shape
                     ``(n_statev, n_points)``.
``time``, ``dtime``  Current problem time and time increment.
``wm_start``         Beginning-of-increment work variables, shape
                     ``(4, n_points)``.
``temperature``      Gauss-point temperatures, or ``None`` when no
                     temperature field is available.
``ndi``              Number of direct components: 3 normally and 2 for the
                     plane-stress interface.
``tangent_mode``     User-defined tangent selector forwarded unchanged by
                     Fedoo.
===================  =========================================================

Stress and strain use engineering Voigt notation. The component ordering,
including the axisymmetric convention, is documented by
:class:`~fedoo.core.mechanical3d.Mechanical3D`.

The callback returns the updated stress, state variables, work variables, and
the material tangent. A tangent can have shape ``(6, 6)`` when it is uniform,
or ``(6, 6, n_points)`` when it varies between material points. Do not modify
``statev_start`` or ``wm_start`` in place: they represent the committed state
and must remain available if the global increment is rejected.


Initialization call
^^^^^^^^^^^^^^^^^^^

Fedoo calls the UMAT once during initialization with:

* ``time == 0`` and ``dtime == 0``;
* zero strain, strain increment, stress, and work arrays;
* the assembly's initial state-variable values, or zeros when none were
  supplied;
* identity ``DR``;
* identity ``F0`` and ``F1`` for finite strain, or empty deformation-gradient
  arrays for small strain.

The ``dtime == 0`` call is not a physical material increment. It must return
the initial elastic tangent without advancing irreversible state, accumulating
work, or evaluating expressions that divide by ``dtime``. This tangent is
stored as ``ElasticMatrix`` when ``use_elastic_tangent=True`` and is restored
at the beginning of each increment for the elastic predictor.

Initial state variables
^^^^^^^^^^^^^^^^^^^^^^^

State-variable initial conditions belong to an assembly rather than to the
material, allowing one material instance to be reused by several problems
with different histories. Set labeled values after creating the assembly and
before initializing the problem:

.. code-block:: python

   import fedoo as fd

   material = MechanicalUMAT(
       umat,
       props,
       n_statev=7,
       statev_label={"P": 0, "EP": slice(1, 7)},
   )
   weakform = fd.weakform.StressEquilibrium(material)
   assembly = fd.Assembly.create(weakform, mesh)

   material.set_initial_statev(assembly, "P", 0.02)
   material.set_initial_statev(assembly, "EP", initial_plastic_strain)

A scalar is broadcast over the selected components and all Gauss points. A
component vector is broadcast over the Gauss points, while a complete field
has shape ``(n_components, assembly.n_gauss_points)``. For a scalar label, a
one-dimensional array of length ``assembly.n_gauss_points`` supplies spatially
varying values. Node and element fields are not converted implicitly.

Advanced users may instead assign the complete
``assembly.sv["Statev"]`` array manually before initialization. Its shape must
be exactly ``(material.n_statev, assembly.n_gauss_points)``. In either case,
the initial state is passed to the ``dtime == 0`` UMAT call, so it may affect
the initial tangent.

Calling ``assembly.reset()`` clears runtime state, including these initial
values. Call ``set_initial_statev`` again before reinitializing a reset
assembly.


Minimal elastic callback
^^^^^^^^^^^^^^^^^^^^^^^^

This small 3D example illustrates the callback contract. Production laws
should also validate their parameters and supported dimensional assumptions.

.. code-block:: python

   import numpy as np
   from fedoo.core import MechanicalUMAT


   def elastic_umat(
       strain, dstrain, F0, F1, stress, DR, props,
       statev_start, time, dtime, wm_start, temperature,
       *, ndi, tangent_mode,
   ):
       E, nu = props[:, 0]
       shear = E / (2 * (1 + nu))
       lame = E * nu / ((1 + nu) * (1 - 2 * nu))

       tangent = np.zeros((6, 6))
       tangent[:3, :3] = lame
       tangent[0, 0] += 2 * shear
       tangent[1, 1] += 2 * shear
       tangent[2, 2] += 2 * shear
       tangent[3, 3] = shear
       tangent[4, 4] = shear
       tangent[5, 5] = shear

       stress_new = stress + tangent @ dstrain
       return (
           stress_new,
           statev_start.copy(),
           wm_start.copy(),
           tangent,
       )


   material = MechanicalUMAT(
       elastic_umat,
       props=[210_000.0, 0.3],
       props_label={"E": 0, "nu": 1},
       is_isotropic=True,
   )

The constructor also accepts:

``n_statev`` and ``statev_label``
   Allocate private history and expose selected components to post-processing.

``density``
   Define the mass density for dynamic mechanical problems.

``tangent_from_F``
   Mark a finite-strain law whose tangent is derived from deformation-gradient
   kinematics instead of the default corotational strain measure. Such a law
   requires a geometrically nonlinear formulation; ``F0`` and ``F1`` remain
   empty in small strain.

``use_elastic_tangent``
   Control whether the initialization tangent is restored at the beginning of
   each increment.

``tangent_mode``
   Select the tangent requested from the callback.

``is_isotropic``
   Avoid unnecessary frame transformations for an invariant response.


Local frames and finite rotations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Assigning a frame with ``material.set_local_frame(...)`` defines the initial
material coordinates. ``MechanicalUMAT`` transforms the strain, deformation
gradient, starting stress, and incremental rotation into those coordinates,
then transforms the returned stress and tangent back to the global frame.
Uniform, nodal, elemental, and Gauss-point material frames are supported as
described in :doc:`constitutive_law`.

Private state variables are opaque to Fedoo because their tensorial meaning is
law-dependent. The callback must apply ``DR`` to any tensor-valued history that
requires objective transport. Scalar history variables require no rotation.

The objective-rate formulation is selected on
:class:`fedoo.weakform.StressEquilibrium` through ``weakform.corate``. A UMAT
subclass with a strict requirement can set, for example:

.. code-block:: python

   class MyUMAT(MechanicalUMAT):
       required_corate = ("log_r", "log_r_inc")

Fedoo then validates the selected formulation when the material is initialized.


Advanced: derive directly from ``Mechanical3D``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Deriving from ``Mechanical3D`` provides complete control but also transfers
the constitutive lifecycle to the implementation. This route is appropriate
when the response cannot be expressed by the UMAT contract, needs custom
assembly state, or uses a specialized tangent representation.

A derived law is generally responsible for:

* calling ``super().__init__(name=name, density=density)``;
* declaring ``is_isotropic`` correctly;
* allocating its state and initial ``assembly.sv["TangentMatrix"]`` in
  ``initialize``;
* reading ``assembly.sv["Strain"]`` and ``assembly.sv["DStrain"]`` and writing
  ``assembly.sv["Stress"]`` and ``assembly.sv["TangentMatrix"]`` in ``update``;
* implementing commit, rollback, and reset behavior through ``set_start``,
  ``to_start``, and ``reset`` when it owns history variables;
* handling plane stress explicitly when that assumption is supported;
* using the frame-transformation helpers supplied by ``Mechanical3D`` when
  the law is anisotropic;
* setting ``_Lt_from_F`` only when the tangent follows deformation-gradient
  rather than strain-increment kinematics.

``assembly.sv_start`` is a shallow beginning-of-increment snapshot. Do not
modify arrays shared with it in place during a trial iteration. Replace trial
arrays in ``assembly.sv`` or make explicit copies so that ``to_start`` can
restore the committed state.

Direct ``Mechanical3D`` laws must also tolerate initialization before a
positive time increment exists. They should always provide the initial
tangent required to assemble the first global system. The implementations of
:class:`fedoo.constitutivelaw.ElasticAnisotropic` and the pedagogical
:class:`fedoo.constitutivelaw.ElastoPlasticity` provide useful references; the
latter uses ``MechanicalUMAT`` for its lifecycle.
