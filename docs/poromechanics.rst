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

Two ready-made factories return a :py:class:`~fedoo.core.weakform.WeakFormSum`
of the momentum balance, the Darcy term and the storage/coupling term:

* :py:func:`fedoo.weakform.PoroMechanicsSimple` — two-field
  ``(u, PorePressure)`` formulation built on the standard
  :py:class:`~fedoo.weakform.StressEquilibrium`. **This is the default
  choice.** Use it whenever the boundary conditions include a free-traction
  face (Mandel consolidation, unconfined compression of cartilage, soft
  tissue indentation): the three-field variant oscillates there because its
  skeleton Lagrange multiplier is under-constrained.

* :py:func:`fedoo.weakform.PoroMechanics` — three-field
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

See ``examples/poromechanics/taylor_hood_lbb.py``.

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

The formulation follows the simcoon kinematic conventions:

* **UL (updated Lagrangian, log_R corotational)** — the recommended mode
  with simcoon laws. The Cauchy coupling
  :math:`-\alpha p\, \mathrm{tr}(\delta\varepsilon)` is assembled on the
  deformed mesh (exact), complemented by the geometric tangent of the pore
  stress. The mass-balance volumetric term uses the rate of :math:`\ln J`.

* **TL (total Lagrangian, Miehe logarithmic strain)** — the coupling is
  written against the volumetric log-strain variation
  :math:`\delta(\mathrm{tr}\,\ln \mathbf{U}) = \delta(\ln J)
  = \mathbf{C}^{-1} : \delta\mathbf{E}`, i.e. the momentum contribution is
  :math:`-\alpha p J (\mathbf{C}^{-1} : \delta\mathbf{E})` — the exact
  pull-back of the Cauchy Terzaghi split to the PK2/Green-Lagrange pair.
  The kinematic weights are evaluated at the current state (the geometric
  derivative :math:`d(J\mathbf{C}^{-1})/du` is not linearized): the residual
  is exact, the tangent is frozen-kinematics.

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
  ``tests/test_poromech_finite_strain.py``.

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

API
---

.. autofunction:: fedoo.weakform.PoroMechanicsSimple

.. autofunction:: fedoo.weakform.PoroMechanics

.. autoclass:: fedoo.constitutivelaw.PoroFluidProperties
   :members:

.. autoclass:: fedoo.constitutivelaw.HolmesMowPermeability

.. autoclass:: fedoo.constitutivelaw.KozenyCarmanPermeability
