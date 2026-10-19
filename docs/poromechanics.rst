Poromechanics
=============

fedoo provides a Biot-type poromechanics module for saturated porous media:
a monolithic (u, PorePressure) coupling between the solid skeleton momentum
balance and the fluid mass balance (Darcy flow + storage). The skeleton
constitutive law is reused unchanged — it never sees the pore pressure. Any
:py:class:`fedoo.constitutivelaw` law works, including the simcoon
hyperelastic laws (``NEOHC``, ``MOORI``, ``YEOHH``, ``ISHAH``, ``GETHH``,
``SWANH``).

Governing equations
-------------------

The total Cauchy stress follows the Biot/Terzaghi effective stress split
(:math:`p > 0` in compression):

.. math::

   \sigma = \sigma_{\mathrm{eff}}(\varepsilon) - \alpha \, p \, \mathbf{I}

and the fluid mass balance combines Darcy flow, storage and the volumetric
coupling (backward-Euler in time):

.. math::

   \frac{1}{M}\dot p + \alpha\, \dot\varepsilon_v
   - \nabla \cdot \left( \frac{k}{\mu_f} \nabla p \right) = 0

where :math:`\alpha` is the Biot coefficient, :math:`M` the Biot modulus,
:math:`k` the intrinsic permeability and :math:`\mu_f` the fluid viscosity.
In finite strain the volumetric strain measure is the logarithmic volume
change :math:`\varepsilon_v = \ln J`.

Weak forms and choosing a variant
---------------------------------

Each variant is a :py:class:`~fedoo.core.weakform.WeakFormSum` of the
momentum balance, the Darcy term and the storage/coupling term. They differ
by the treatment of the volume change of the skeleton.

* :py:class:`fedoo.weakform.PoroMechanicsSimple` (**default**). Unknowns
  ``u`` and ``PorePressure``; the skeleton is handled by the standard
  :py:class:`~fedoo.weakform.StressEquilibrium`. Use it unless you have a
  specific reason not to.

* :py:class:`fedoo.weakform.PoroMechanics`. Adds a third unknown,
  ``Pressure``, which enforces the volume change of the skeleton
  (:py:class:`~fedoo.weakform.StressEquilibriumMixed`). It is only useful
  when the *skeleton itself* is nearly incompressible (drained Poisson ratio
  close to 0.5) and the sample is confined. Its pressure oscillates near
  free-traction faces (Mandel consolidation, unconfined compression,
  indentation). ``bulk_modulus`` is required with ``nlgeom``.

**Soft tissues and gels.** A saturated tissue behaves as incompressible
when it is loaded quickly, but this comes from the fluid, which has no time
to escape (undrained response), not from the skeleton. The fluid part is
already carried by ``PorePressure``, and the drained skeleton is
compressible (:math:`\nu \approx 0.1` to :math:`0.3`). Use
``PoroMechanicsSimple`` with a compressible skeleton law, and a Taylor-Hood
interpolation if the loading is fast (see below).

Element interpolation: equal order or Taylor-Hood
-------------------------------------------------

**Why the choice matters.** When the fluid has no time to flow (low
permeability, short time step) and is nearly incompressible (large
``biot_modulus``), the mass balance reduces to
:math:`\alpha\,\dot\varepsilon_v \approx 0`: the skeleton cannot change volume,
and the pore pressure acts as the Lagrange multiplier that enforces this
constraint. This is the *undrained* limit. It has the same mathematical
structure as incompressible elasticity or Stokes flow, a saddle-point problem,
and its discretization is stable only if the pressure field is not too rich
compared with the displacement field. This requirement is the *inf-sup*
condition, also called the LBB condition after Ladyzhenskaya, Babuska and
Brezzi.

When it is violated, some pressure patterns produce no nodal force on the
displacement field. Nothing in the equations controls them, so they appear as
spurious oscillations of the pressure, typically alternating in sign from one
node to the next (*checkerboard* modes). They are strongest near drained
boundaries and loaded faces during the first increments. In the strictly
undrained limit, the system matrix is singular.

**Equal-order interpolation** uses the same element for ``u`` and
``PorePressure``, e.g. ``hex8`` for both. It does not satisfy the inf-sup
condition, but the storage term (:math:`1/M`) and the Darcy term stabilise it
as long as the fluid can flow within a time step. A classical rule of thumb
for linear elements is that oscillations stay small when

.. math::

   \Delta t \gtrsim \frac{h^2}{6\,c_v}, \qquad
   c_v = \frac{k}{\mu_f}\Big(K + \tfrac43 G\Big),

with :math:`h` the element size near the drained boundary or the load,
:math:`c_v` the consolidation coefficient and :math:`K`, :math:`G` the drained
bulk and shear moduli of the skeleton (:math:`K + \tfrac43 G` is the
oedometric modulus). This is the criterion of Vermeer and Verruijt (1981),
derived for one-dimensional consolidation with linear elements and
incompressible constituents; with a hydraulic conductivity
:math:`k_h = k\,\gamma_w/\mu_f` it reads
:math:`\Delta t \gtrsim \gamma_w h^2/(6\,E_{oed}\,k_h)`, the form quoted in
the Abaqus documentation. Fine meshes,
low permeabilities and short first steps of a fast loading are therefore the
risky cases. Equal-order elements are fine for most consolidation analyses
that do not need the very first instants.

**Taylor-Hood interpolation** interpolates the displacement one order higher
than the pressure: quadratic ``u`` and linear ``PorePressure``. This pair
satisfies the inf-sup condition and gives a smooth pressure even in the
undrained limit, at the cost of a quadratic mesh.

.. list-table::
   :header-rows: 1

   * - Quadratic mesh (``u``)
     - Linear pressure element
   * - ``hex20``
     - ``hex8``
   * - ``tet10``
     - ``tet4``
   * - ``quad8``
     - ``quad4``
   * - ``tri6``
     - ``tri3``

**How to build it in fedoo.** A
:py:class:`~fedoo.lib_elements.CombinedElement` defines a new
element type in which each variable may use its own interpolation. Its base
element (here the quadratic one) is used for the geometry and for every
variable by default; ``set_variable_interpolation`` assigns a linear
interpolation to one variable, built on the vertex nodes of the quadratic
element. The mesh itself stays quadratic. The combined element is registered
under the name given as first argument, which must therefore be created
*before* the assembly and passed as ``elm_type``::

    mesh = fd.mesh.box_mesh(..., elm_type="hex20")  # quadratic mesh

    # New element type "hex20lbb": hex20 for u and the geometry,
    # hex8 (vertex nodes only) for PorePressure.
    elm = fd.lib_elements.CombinedElement("hex20lbb", "hex20")
    elm.set_variable_interpolation("PorePressure", "hex8")

    assembly = fd.Assembly.create(wf, mesh, elm_type="hex20lbb")

With the three-field :py:class:`~fedoo.weakform.PoroMechanics`, interpolate
the skeleton ``Pressure`` linearly as well
(``elm.set_variable_interpolation("Pressure", "hex8")``).

.. note::

   With a Taylor-Hood element, the ``PorePressure`` DOFs of the mid-side
   nodes carry no physical value: the pressure is defined by the vertex
   nodes only. A nodal plot of ``PorePressure`` that includes the mid-side
   nodes is therefore misleading. Read the pressure at the vertex nodes, or
   at the integration points with
   ``assembly.get_gp_results(pb.space.variable("PorePressure"),
   pb.get_dof_solution())``.

The example :ref:`sphx_glr_examples_poromechanics_taylor_hood_lbb.py`
compares equal-order and Taylor-Hood interpolations in the undrained limit.

Fluid properties and permeability models
----------------------------------------

Fluid-phase parameters are grouped in
:py:class:`fedoo.constitutivelaw.PoroFluidProperties`:
``biot_coefficient`` (:math:`\alpha`, default 1.0 — the Terzaghi limit),
``biot_modulus`` (:math:`M`; ``None`` means an incompressible fluid,
:math:`1/M = 0`), ``permeability``, ``fluid_viscosity`` and
``initial_porosity``. ``fluid_density`` is stored but not used yet (reserved
for gravity loading).

``permeability`` may be a constant or a callable ``k(J, sv)`` for
deformation-dependent permeability. Two standard models are provided:

* :py:class:`fedoo.constitutivelaw.HolmesMowPermeability` — articular
  cartilage model, permeability drops in compression;
* :py:class:`fedoo.constitutivelaw.KozenyCarmanPermeability` — porosity
  driven, with :math:`\phi(J)` from solid mass conservation.

In finite strain (``nlgeom``), both weak-form variants compute ``lnJ`` at
gauss points and feed :math:`J = e^{\ln J}` to the callable. If a callable
permeability is used in a context where ``lnJ`` is unavailable, a warning is
emitted and the reference value :math:`k(J{=}1)` is used.

Finite strain formulation
-------------------------

The formulation follows the simcoon kinematic conventions: UL works in the
:math:`(\ln \mathbf{V}, \boldsymbol{\tau})` box and TL in the
:math:`(\ln \mathbf{U}, \mathbf{T})` box. Both use the same volumetric
measure, :math:`\mathrm{tr}\,\ln\mathbf{U} = \mathrm{tr}\,\ln\mathbf{V} = \ln J`.
The TL weak form is the exact pull-back of the UL one, so the two modes
assemble the same residual and give the same solution.

Written per unit reference volume, the model is:

.. math::

   \boldsymbol{\tau} = \boldsymbol{\tau}_{\mathrm{eff}} - \alpha p J \mathbf{I}
   \quad \left(\mathbf{S} = \mathbf{S}_{\mathrm{eff}}
   - \alpha p J \mathbf{C}^{-1}\right)

.. math::

   \alpha \dot J + \frac{J}{M}\dot p
   - \mathrm{Div}\left( J \mathbf{F}^{-1} \frac{k(J)}{\mu_f}
   \mathbf{F}^{-T} \nabla_X p \right) = 0

which is, per unit current volume,
:math:`\alpha\, \mathrm{div}\, \mathbf{v} + \dot p / M
- \mathrm{div}(k/\mu_f\, \nabla p) = 0`. The coefficients
:math:`\alpha` and :math:`M` are constants.

* **UL (updated Lagrangian, log_R corotational)**: the recommended mode
  with simcoon laws. The coupling
  :math:`-\alpha p\, \mathrm{tr}(\delta\varepsilon)` and the mass balance
  are assembled on the current mesh, with the geometric tangent of the
  pore stress. The volumetric rate is the backward-Euler rate of
  :math:`\ln J`.

* **TL (total Lagrangian, Miehe logarithmic strain)**: the same equations
  on the reference mesh. The momentum coupling is
  :math:`-\alpha p J (\mathbf{C}^{-1} : \delta\mathbf{E})`. The storage and
  coupling terms of the mass balance carry the factor :math:`J` of
  :math:`dv = J\, dV`, with
  :math:`\delta(\ln J) = \mathbf{C}^{-1} : \delta\mathbf{E}` and
  :math:`\delta J = J \mathbf{C}^{-1} : \delta\mathbf{E}`. The mobility is
  pulled back to :math:`(k/\mu_f) J \mathbf{C}^{-1}`.

In both modes the dependence of the Darcy term on the displacement (moving
mesh in UL, :math:`J \mathbf{C}^{-1}` in TL), the variation of the volume
element :math:`dv` in the UL storage and coupling terms of the mass balance,
and the derivative :math:`d(J\mathbf{C}^{-1})/du` of the TL momentum
coupling are not linearized: the residual is exact, these tangent blocks
use frozen kinematics.

**Relation to Coussy's poromechanics.** In the Lagrangian finite-strain
poroelasticity of Coussy (*Poromechanics*, 2004), the Lagrangian porosity
:math:`\phi = J n` obeys :math:`\phi - \phi_0 = \mathbf{b} : \mathbf{E}
+ p/N` and the stress :math:`\mathbf{S} = \mathbf{S}_{\mathrm{eff}}
- p \mathbf{b}`, with the same Biot tensor :math:`\mathbf{b}` in both
(Maxwell symmetry). fedoo corresponds to
:math:`\mathbf{b} = \alpha J \mathbf{C}^{-1}` (exact for incompressible
grains with :math:`\alpha = 1`), so the coupling terms coincide with Coussy.
The only difference is the storage term: fedoo uses a modulus :math:`M`
constant per unit current volume (:math:`J \dot p / M` per reference
volume), Coussy a modulus :math:`N` constant per unit reference volume
(:math:`\dot p / N`). Both coincide for incompressible constituents
(:math:`1/M = 0`), which is the classical biphasic model of soft tissues.

Sign conventions
----------------

The Biot tangent blocks are intentionally non-symmetric
(:math:`K_{up} = -\alpha`, :math:`K_{pu} = +\alpha/\Delta t`), so the
momentum weak forms set ``assembly_options["assume_sym"] = False``. Do not
re-enable the symmetric assumption: it would mirror the one-sided coupling
into a phantom block and split the assembly, breaking the Newton tangent.
Pore pressure is positive in compression.

Validation
----------

* Terzaghi 1D consolidation vs the analytical series (< 1.5 %) —
  ``tests/test_poromech_terzaghi_analytical.py``;
* Mandel 2D vs the Abousleiman analytical solution, including the
  Mandel-Cryer non-monotonic pressure peak (~1 %) —
  ``tests/test_poromech_mandel_analytical.py``;
* Taylor-Hood LBB stability — ``tests/test_poromech_taylor_hood.py``;
* Finite strain with a simcoon Neo-Hookean skeleton (``NEOHC``, UL) and
  Holmes-Mow permeability: exact confined kinematics
  :math:`\ln J = \ln(1 + \delta/L)` at the drained state —
  ``tests/test_poromech_finite_strain.py``;
* TL against UL on the same consolidation, and the TL mass-balance tangent
  against finite differences — ``tests/test_poromech_finite_strain_tl.py``;
* ``2Daxi``: homogeneous undrained unconfined compression of a cylinder vs
  the closed-form solution (both formulations), drained relaxation, and
  steady radial flow through a hollow cylinder vs the logarithmic pressure
  profile — ``tests/test_poromech_axi.py``. The draining case was also
  compared once with a 3D model of the full cylinder, with close agreement
  (this comparison is not part of the test suite).

Limitations
-----------

* **Large compression with simcoon hyperelastic laws (UL)**: beyond roughly
  12 % compression the Newton loop can develop a slowly-amplifying u-p
  oscillation. The poro coupling blocks were verified consistent against a
  finite-difference Jacobian; the amplification originates in the fedoo core
  UL displacement tangent of hyperelastic (``_Lt_from_F``) laws, which is
  self-consistent only in the modes excited by pure mechanical loading —
  the Biot coupling seeds the other modes at every iteration. Workarounds:
  smaller total strain, an elasto-plastic/corotational law (``ELISO`` etc.,
  which are unaffected), or finer load steps do *not* help (the threshold is
  on the strain level, not the step size).
* ``2Daxi`` is supported in small strain and in UL, like
  :py:class:`~fedoo.weakform.StressEquilibrium`. It is not available in TL.
* The permeability tensor is isotropic (:math:`k \mathbf{I}`); anisotropic
  permeability is not implemented.
* No gravity/body-force term for the fluid yet (``fluid_density`` is
  reserved).

Examples
--------

* :ref:`sphx_glr_examples_poromechanics_terzaghi_1d.py` — small-strain 1D
  consolidation compared with the analytical Terzaghi series;
* :ref:`sphx_glr_examples_poromechanics_finite_strain_consolidation.py` —
  finite-strain (UL) confined consolidation of a Neo-Hookean skeleton with
  Holmes-Mow permeability;
* :ref:`sphx_glr_examples_poromechanics_indentation_soft_tissue.py` —
  axisymmetric (``2Daxi``) indentation of a hyperelastic soft tissue by a
  rigid spherical indenter with IPC contact, an impermeable contact zone
  that follows the contact, and force relaxation as the fluid is squeezed
  out;
* :ref:`sphx_glr_examples_poromechanics_taylor_hood_lbb.py` — equal-order vs
  Taylor-Hood interpolation in the undrained limit.

API
---

Weak forms:

* :py:class:`fedoo.weakform.PoroMechanicsSimple`
* :py:class:`fedoo.weakform.PoroMechanics`

Fluid properties and permeability models:

* :py:class:`fedoo.constitutivelaw.PoroFluidProperties`
* :py:class:`fedoo.constitutivelaw.HolmesMowPermeability`
* :py:class:`fedoo.constitutivelaw.KozenyCarmanPermeability`
