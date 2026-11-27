Thermo-mechanical analysis
=========================

Coupled analysis supports 3D and plane-strain solids, with native elastic laws
in small or finite strain. A monolithic problem sums the existing mechanical
and thermal weakforms with the **same material instance**::

    thermal = fd.constitutivelaw.ThermalProperties(
        thermal_conductivity=180, specific_heat=900, density=2700,
        alpha=23e-6, reference_temperature=293.15,
    )
    material = fd.constitutivelaw.ElasticIsotrop(E=70e9, nu=0.3, thermal=thermal)
    wf = fd.weakform.StressEquilibrium(material) + fd.weakform.HeatEquation(material)
    assembly = fd.Assembly.create(wf, mesh)
    pb = fd.problem.NonLinear(assembly)
    pb.set_dof("Temp", 293.15)
    pb.set_time_integrator(fd.time.FIRST_ORDER, fd.time.BackwardEuler())
    pb.set_nr_criterion("Fields", fields={
        "Disp": {"atol": 1e-5, "rtol": 1e-8},
        "Temp": {"atol": 1e-6, "rtol": 1e-8},
    })

Mechanics and heat must be assembled together: call ``fd.Assembly.create`` on
the summed weak form. The same coupled material in two assemblies created
separately and then added raises ``ValueError``, because the two assemblies do
not share the Gauss-point state and the coupling terms would be lost. Heat and
mechanics with two different materials (a Simcoon mechanical law and plain
``ThermalProperties`` for instance) remain possible: the problem is then
solved without coupling terms and a ``UserWarning`` says so.

With thermo-mechanical coupling, the temperature field and
``reference_temperature`` are necessarily absolute temperatures in kelvin,
never degrees Celsius: the thermoelastic heat source is proportional to the
absolute temperature. They must stay strictly positive with
``SimcoonThermoMechanical`` and in finite strain: set the initial field with
``pb.set_dof("Temp", T0)``. A nonpositive initial temperature raises
``ValueError``; a nonpositive Newton iterate is a failed increment and the time
step is reduced.

Add mechanical and thermal boundary conditions, then call ``pb.nlsolve``.
See ``examples/thermal/thermomechanical_monolithic.py`` for a complete example,
including an optional FDH5 output and a Simcoon variant.

Material response and time integration
-------------------------------------

The standard ``ElasticIsotrop``, ``ElasticOrthotropic``, ``ElasticAnisotropic``
and ``CompositeUD`` laws accept optional ``thermal=ThermalProperties(...)``.
There is no separate thermoelastic material class. Properties can also be
attached after defining the mechanical material, before constructing its
weakforms::

    material = fd.constitutivelaw.ElasticIsotrop(E, nu)
    material.set_thermal_properties(thermal)

Or pass the ``ThermalProperties`` constructor keywords directly::

    material.set_thermal_properties(
        thermal_conductivity=k, specific_heat=cp, density=rho,
        alpha=alpha, reference_temperature=T_ref, coupling=False,
    )

Attaching thermal properties also supplies the density used for mechanical
inertia: a coupled material has a single density, and a different value given
through ``set_density`` raises ``ValueError``. Unattached elastic laws retain
their mechanical behavior, including their existing finite-strain support.

Thermal properties enable thermal expansion and reversible mechanical feedback
into heat, evaluated at the current temperature. Specific heat is the
capacity per unit mass at fixed strain. Conductivity may be a constant scalar
or a symmetric positive-semidefinite 3x3 tensor in current global axes. ``alpha`` may be
a scalar for isotropic expansion, three normal coefficients, or six engineering
strain coefficients; these use the initial global axes. Capacity and density are
positive constant scalars. Spatially varying native elastic moduli remain
supported. Temperatures must be in kelvin for the thermal feedback.

With ``thermal.coupling=False``, only thermal expansion remains active. The material can
also be used with a mechanical weakform alone and a prescribed temperature in
``assembly.sv["Temp"]``. A separate thermal-only problem can provide this field
or its nodal history. The history transfer and synchronization of independent
problems remain user-managed; there is no staggered driver yet.

``SimcoonThermoMechanical`` calls ``simcoon.umat_T``. For example::

    material = fd.constitutivelaw.SimcoonThermoMechanical(
        "ELISO", [rho, cp, E, nu, alpha], thermal_conductivity=k,
    )

These are the *thermal* UMAT properties, not the mechanical UMAT layout: the
first two are always the reference density and the specific heat. Every law of
``umat_T`` is accepted (ELISO, ELIST, ELORT, EPICP, EPKCP, EPJCK, ZENER, ZENNK,
PRONK and the shape memory alloy laws). ELISO uses one state variable; the
other laws require an explicit ``n_statev`` (at least 8 for EPICP and 14 for
EPKCP) and the corresponding kernel properties.
Initialization uses ``start=True`` once; trial evaluations use ``start=False``,
including the first increment at time zero.

Simcoon takes the temperature of each point at initialization as the
stress-free reference. Give ``reference_temperature`` (in kelvin) to set
another one: a body initialized at a different temperature is then stressed if
it cannot expand, as with the native laws.

In finite strain (``StressEquilibrium(material, nlgeom="TL")`` or ``"UL"``
with ``HeatEquation(material, nlgeom="TL")``) the law runs under Simcoon's
total form: it works on the material logarithmic strain ``ln U`` and its dual
stress, so that ``alpha`` is a logarithmic expansion coefficient and ELISO is
the Hencky model of the native laws. Simcoon returns ``dS/dE``, ``dS/dT`` and
``dr/dE`` at fixed deformation gradient, converted for UL mechanics; the heat
stays per unit reference volume. Anisotropy axes are those of the global
frame and are not convected.

The Simcoon local thermal balance already includes capacity. Its HeatEquation
therefore has no additional HeatCapacity term. Attach FIRST_ORDER/BackwardEuler
to declare transient thermal evolution; the local discretization remains owned
by the UMAT. Kernel thermal/cross tangents are used as supplied and can be
continuum rather than exact algorithmic derivatives.

The capacity of a Simcoon law (``props[1]``) is the one at constant stress,
:math:`c_\sigma`, while ``specific_heat`` of the native laws is the one at
constant strain, :math:`c_\varepsilon`. For an isotropic solid of bulk modulus
:math:`K` and linear expansion :math:`\alpha`,
:math:`c_\sigma = c_\varepsilon + 9 K \alpha^2 T / \rho`: convert the value
when comparing the two families on the same material.

Native elastic materials instead use BackwardEuler and consistent HeatCapacity.
Both displacement-temperature blocks are assembled, and the global matrix is
nonsymmetric. Weakforms must share quadrature so that one assembly owns and
commits the complete material state. Incompatible integration rules or different
material instances in the same coupled sum are rejected. Constitutive trial
updates always use the committed state; rollback rebuilds the local response
for the retry's time increment.

Finite strain with a TL heat equation
------------------------------------

For native elastic laws, select either mechanical formulation and use reference
integration for heat::

    mechanical = fd.weakform.StressEquilibrium(material, nlgeom="UL")  # or "TL"
    heat = fd.weakform.HeatEquation(material, nlgeom="TL")
    assembly = fd.Assembly.create(mechanical + heat, mesh)

With a thermo-mechanical material, ``HeatEquation(nlgeom=None)`` (the default)
follows the mechanical geometry and uses the TL thermal balance when finite
strain is active. ``True`` is an alias for ``"TL"``. A fixed-geometry heat
equation (``nlgeom=False``) is rejected in a coupled finite-strain solid. An
uncoupled thermal material keeps the heat equation of the small-strain case
unless ``nlgeom="TL"`` is given: its conductivity and capacity are then
properties of the configuration on which the assembly integrates, the
reference one next to TL mechanics and the current one next to UL mechanics
(same assembly). The two models differ by the volume change.

The native constitutive response uses the same reference Hencky energy for
both mechanical assemblies. It evaluates ``h = ln U`` and its dual stress
``T = C:(h-alpha*(temperature-T_ref))``, then uses Simcoon 2.2 to convert the
stress and tangent to the mechanical formulation. The total logarithmic
mapping also handles anisotropic elasticity and noncoaxial stresses. It is
independent of corotational history; UL mechanics accepts ``log_r`` or ``log``.

Conductivity is specified as a constant spatial tensor ``k`` in current global
axes. The thermal equation uses its pullback
``K0 = J F^-1 k F^-T``, reference density and reference-volume heat sources.
Heat capacity is ``rho0*c``; material density need not be updated as the body
deforms. The deformation derivative of ``K0`` contributes to the thermal versus
displacement block. The reference thermal gradients and quadrature measure
are preserved when sharing an assembly with UL mechanics, without duplicating
the constitutive state. Consistent mechanical inertia also uses reference mass.

Finite-strain reversible feedback uses actual absolute temperature::

    r0 = -temperature * beta:(h-h_start) / dt
    beta = C:alpha

Its deformation and temperature derivatives are assembled. This corresponds
to constant specific heat at fixed logarithmic strain. The small-strain model
uses the same source with the small strain, also at the current temperature:
a coupled small-strain problem is therefore nonlinear. ``coupling=False`` disables this
feedback in either formulation. Positive absolute temperature is required.

The native example accepts ``--nlgeom TL`` or ``--nlgeom UL``. Both use TL heat
integration. Gauss-point ``HeatFlux`` and ``ReferenceHeatFlux`` fields contain
the spatial heat flux and its reference counterpart. A heat-only problem can
also use a prescribed total deformation gradient in ``assembly.sv["F"]``.

Field convergence
-----------------

``Fields`` checks each field independently using::

    norm(residual_f) < atol_f + rtol_f * reference_f

All fields must pass. The reference defaults to the norm of the reactions and
applied loads of that field; a fixed nonnegative ``reference`` can be supplied
per field. Choose an absolute tolerance appropriate to the units of each
field. No arbitrary force scale of one is substituted by this criterion: a
field without reaction nor load (an adiabatic temperature field for instance)
has a zero reference, and a relative tolerance alone then raises
``ValueError``.

The maximum normalized field error must be below one; the scalar ``tol`` or
``tol_nr`` does not set the Fields residual threshold. It still bounds the
residual of nonlinear constraints such as ``MeanMotion``. ``pb.nr_errors``
reports the latest per-field errors.

The default ``Force`` criterion is also evaluated field by field when the
problem has displacement and temperature unknowns: the error is the largest of
the two residual norms, each divided by the reactions and loads of its own
field, and it is compared to ``tol``. A field without reaction nor load takes a
unit reference, as ``Force`` always did: ``tol`` is then an absolute tolerance
for that field, and ``Fields`` with an explicit ``atol`` is the better choice.
A field whose reactions vanish at the solution (free thermal expansion on
isostatic supports) converges once its residual reaches the round-off level of
the assembled fluxes.
With other free unknowns (rotations, global DOFs) or an MPC mixing the two
fields, ``Force`` keeps its single norm.
Fields must cover every free DOF, including global DOFs, without overlaps.
MPC forces are reduced before checking the master DOFs of each field. Constraints
that mix fields with different units are not a supported convergence partition.

Current limits
--------------

Static mechanics and undamped endpoint Newmark mechanics can be combined with
the thermal solve. Plane stress,
axisymmetry, incompressibility projections, material-frame transformations, damped coupled mechanics and
generalized-alpha away from the endpoint are not yet supported. These modes
raise explicit errors rather than silently dropping coupling terms. A staggered
coupling driver remains a subsequent step. Finite thermoelasticity and the
finite-strain ``umat_T`` call require Simcoon 2.6.
