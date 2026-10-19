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

Two ready-made :py:class:`~fedoo.core.weakform.WeakFormSum` classes group
the momentum balance, the Darcy term and the storage/coupling term:

* :py:class:`fedoo.weakform.PoroMechanicsSimple` — two-field
  ``(u, PorePressure)`` formulation built on the standard
  :py:class:`~fedoo.weakform.StressEquilibrium`. **This is the default
  choice.** Use it whenever the boundary conditions include a free-traction
  face (Mandel consolidation, unconfined compression of cartilage, soft
  tissue indentation): the three-field variant oscillates there because its
  skeleton Lagrange multiplier is under-constrained.

* :py:class:`fedoo.weakform.PoroMechanics` — three-field
  ``(u, PorePressure, Pressure)`` formulation built on
  :py:class:`~fedoo.weakform.StressEquilibriumMixed`, where ``Pressure`` is
  the volumetric Lagrange multiplier of the skeleton. Reserve it for a
  genuinely quasi-incompressible **skeleton** (drained Poisson ratio close
  to 0.5) with well-constrained (confined) boundaries. Requires
  ``bulk_modulus`` in ``nlgeom`` modes.

Note that for most biological tissues the *drained* skeleton is quite
compressible (:math:`\nu \sim 0.1..0.3`); the observed near-incompressibility
of the tissue is the *undrained* response, and it is already carried by the
``PorePressure`` field. The two-field ``PoroMechanicsSimple`` variant with a
compressible skeleton law is therefore the recommended setting for
free-traction soft-tissue problems, combined with a Taylor-Hood
interpolation for stability (below).

Element interpolation (LBB stability)
-------------------------------------

Equal-order linear interpolation of ``u`` and ``PorePressure`` works for
moderate consolidation problems thanks to the storage term, but exhibits
checkerboard pressure modes in the undrained limit (large ``biot_modulus``,
fast loading). fedoo supports Taylor-Hood interpolation natively through
combined elements: define the combined element *before* creating the
assembly, e.g. quadratic ``u`` with linear pressure::

    mesh = fd.mesh.box_mesh(..., elm_type="hex20")
    # Quadratic displacement, PorePressure interpolated linearly on the
    # hex8 sub-element:
    elm = fd.lib_elements.element_list.CombinedElement("hex20lbb", "hex20")
    elm.set_variable_interpolation("PorePressure", "hex8")
    assembly = fd.Assembly.create(wf, mesh, elm_type="hex20lbb")

See the example :ref:`sphx_glr_examples_poromechanics_taylor_hood_lbb.py`.

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
mesh in UL, :math:`J \mathbf{C}^{-1}` in TL) and the derivative
:math:`d(J\mathbf{C}^{-1})/du` of the TL momentum coupling are not
linearized: the residual is exact, these tangent blocks use frozen
kinematics.

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
  against finite differences — ``tests/test_poromech_finite_strain_tl.py``.

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
* ``2Daxi`` modeling spaces are not supported: all poromechanics weak forms
  raise an explicit ``NotImplementedError`` (the Biot coupling lacks the
  ``2*pi*r`` measure and the hoop strain term).
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
