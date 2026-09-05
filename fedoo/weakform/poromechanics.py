"""Poromechanics weak formulations.

Three-field mixed formulation ``(u, PorePressure, Pressure)`` for saturated
porous media, where ``Pressure`` is the volumetric Lagrange multiplier of the
quasi-incompressible solid skeleton (inherited from
:py:class:`StressEquilibriumMixed`) and ``PorePressure`` is the Biot pore
fluid pressure.

The Terzaghi/Biot coupling adds ``-alpha * p_pore * I`` to the total Cauchy
stress at the weak form level: the skeleton constitutive law (e.g. simcoon
``NEOHC``, ``YEOHH``, ``PRONK``) is reused unchanged and never sees the pore
pressure. The fluid mass balance is split into a steady Darcy diffusion term
and a transient storage / volumetric coupling term, in the spirit of
:py:class:`HeatEquation`.

Backward-Euler time integration is used for the mass balance. The
formulation is written in log_R corotational strain (the simcoon convention),
which means the weak form expressions remain identical in form to the
small-strain case; only the underlying constitutive law and the
``StressEquilibriumMixed`` parent take care of the kinematic mapping.
"""

import numpy as np

from fedoo.core.weakform import WeakFormBase, WeakFormSum
from fedoo.weakform.stress_equilibrium import StressEquilibrium
from fedoo.weakform.stress_equilibrium_mixed import StressEquilibriumMixed


# ----------------------------------------------------------------------
# 1. Momentum balance: skeleton equilibrium + Terzaghi pore-pressure coupling
# ----------------------------------------------------------------------
class PoroMomentum(StressEquilibriumMixed):
    """Momentum balance for a saturated porous skeleton (Biot/Terzaghi).

    Inherits from :py:class:`StressEquilibriumMixed`, which already handles
    the mixed (u, Pressure) decomposition of the skeleton — ``Pressure`` is
    the volumetric Lagrange multiplier enforcing ``Pressure = K * ln J``.
    This class adds a second scalar variable ``PorePressure`` and contributes
    the Terzaghi coupling ``-alpha * p_pore * tr(delta_eps)`` to the momentum
    weak form.

    The skeleton constitutive law is reused unchanged: it receives ``F_bar``
    (isochoric part of ``F``) and returns the effective Cauchy stress and
    its log-strain tangent. The pore pressure is added afterwards at the
    weak-form level.

    Parameters
    ----------
    constitutivelaw : ConstitutiveLaw
        Skeleton constitutive law. In large-strain mode it must be a
        hyperelastic law whose tangent is computed from F
        (``_Lt_from_F = True`` — the case for all simcoon hyperelastic and
        visco-elastic laws). NB: the UL box -> Lie tangent conversion is
        driven by the separate ``_corotational_box_tangent`` attribute
        (True for every simcoon umat, see
        :class:`fedoo.core.mechanical3d.Mechanical3D`).
    fluid_props : PoroFluidProperties
        Fluid-phase parameters. Only the Biot coefficient ``alpha`` is used
        here; ``PoroDarcy`` and ``PoroMassStorage`` use the other fields.
    bulk_modulus : float, optional
        Skeleton bulk modulus used to scale the ``Pressure`` constraint.
        Required for ``nlgeom`` modes.
    name : str, default ""
    nlgeom : bool or {'TL', 'UL'}, optional
        Geometric non-linearity flag.
    space : ModelingSpace, optional
    """

    def __init__(
        self,
        constitutivelaw,
        fluid_props,
        bulk_modulus=None,
        name="",
        nlgeom=None,
        space=None,
    ):
        super().__init__(
            constitutivelaw,
            bulk_modulus=bulk_modulus,
            name=name,
            nlgeom=nlgeom,
            space=space,
        )
        self.fluid_props = fluid_props
        self.space.new_variable("PorePressure")
        # The Biot u-PorePressure coupling is genuinely one-sided (it only
        # writes the upper [Disp][PorePressure] block). StressEquilibrium
        # defaults assume_sym=True, which would mirror that block into a
        # *phantom* [PorePressure][Disp] = K_up^T entry with no matching
        # residual, AND split the WeakFormSum into two assemblies with
        # disjoint state-variable dicts (so _tr_eps_gp never reaches
        # PoroMassStorage). Both make the Newton tangent inconsistent. The
        # assembled Biot tangent is intentionally non-symmetric
        # (K_up = -alpha vs K_pu = +alpha/dt), so disable the assumption.
        self.assembly_options["assume_sym"] = False

    def get_weak_equation(self, assembly, pb):
        """Build the momentum weak form augmented with the Biot coupling."""
        diff_op = super().get_weak_equation(assembly, pb)

        alpha = self.fluid_props.biot_coefficient

        # Linearized strain operator (same in small / UL / TL within this
        # context: we use the volumetric part of op_strain for the coupling).
        if assembly._nlgeom == "TL":
            eps = self.space.op_strain(assembly.sv["DispGradient"])
        else:
            eps = self.space.op_strain()

        p_pore_inc = self.space.variable("PorePressure")
        p_pore_curr = assembly.sv.get("_PorePressure_gp", 0)
        p_pore_total = p_pore_inc + p_pore_curr

        # Biot/Terzaghi coupling: sigma_total = sigma_eff - alpha * p_pore * I.
        # The momentum residual contribution is -alpha * int(p_pore * tr(delta_eps)).
        # In fedoo's weak-form convention the assembled D vector carries
        # -R(U_curr) while the matrix is +d R/dU. With assume_sym=False (the
        # genuine, non-mirrored assembly), writing -alpha here gives the
        # physically correct tangent K_up = -alpha and a pore pressure that is
        # positive in compression (validated on Terzaghi consolidation and the
        # Mandel benchmark). This sign is paired with +alpha/dt in
        # PoroMassStorage to form the consistent (non-symmetric) Biot Jacobian.
        diff_op -= alpha * sum(
            [0 if eps[i] == 0 else eps[i].virtual * p_pore_total for i in range(3)]
        )

        if self.space._dimension == "2Daxi":
            rr = assembly.sv["_R_gausspoints"]
            diff_op = diff_op  # 2Daxi factor already applied by parent

        return diff_op

    def initialize(self, assembly, pb):
        super().initialize(assembly, pb)
        assembly.sv["_PorePressure_gp"] = 0
        assembly.sv["_tr_eps_gp"] = np.zeros(assembly.n_gauss_points)

    def update(self, assembly, pb):
        super().update(assembly, pb)
        disp = pb.get_dof_solution()
        if np.isscalar(disp) and disp == 0:
            assembly.sv["_PorePressure_gp"] = 0
            assembly.sv["_tr_eps_gp"] = np.zeros(assembly.n_gauss_points)
            return

        assembly.sv["_PorePressure_gp"] = assembly.get_gp_results(
            self.space.variable("PorePressure"), disp
        )
        if assembly._nlgeom and "lnJ" in assembly.sv:
            assembly.sv["_tr_eps_gp"] = assembly.sv["lnJ"]
        else:
            eps_op = self.space.op_strain()
            tr_eps = 0
            for i in range(3):
                if eps_op[i] != 0:
                    tr_eps = tr_eps + assembly.get_gp_results(eps_op[i], disp)
            assembly.sv["_tr_eps_gp"] = tr_eps


# ----------------------------------------------------------------------
# 2. Mass balance: steady Darcy diffusion of pore pressure
# ----------------------------------------------------------------------
class PoroDarcy(WeakFormBase):
    """Steady Darcy diffusion of pore pressure.

    Contributes ``int( grad(delta p) . (k / mu_f) . grad(p) ) dV`` to the
    fluid mass balance. Pairs with :py:class:`PoroMassStorage` for the
    transient and volumetric-coupling terms.

    Parameters
    ----------
    fluid_props : PoroFluidProperties
        Fluid parameters. Permeability may be constant or a callable
        depending on ``J``.
    name : str, default ""
    space : ModelingSpace, optional
    """

    def __init__(self, fluid_props, name="", space=None):
        WeakFormBase.__init__(self, name, space)
        if self.space._dimension == "2Daxi":
            raise NotImplementedError(
                "PoroDarcy is not implemented for the '2Daxi' modeling space."
            )

        self.fluid_props = fluid_props
        self.space.new_variable("PorePressure")

        if self.space.ndim == 3:
            self._op_grad_p = [
                self.space.derivative("PorePressure", "X"),
                self.space.derivative("PorePressure", "Y"),
                self.space.derivative("PorePressure", "Z"),
            ]
        else:
            self._op_grad_p = [
                self.space.derivative("PorePressure", "X"),
                self.space.derivative("PorePressure", "Y"),
                0,
            ]
        self._op_grad_p_vir = [0 if op == 0 else op.virtual for op in self._op_grad_p]

    def initialize(self, assembly, pb):
        assembly.sv["_PorePressureGradient_gp"] = [0, 0, 0]

    def update(self, assembly, pb):
        disp = pb.get_dof_solution()
        if np.isscalar(disp) and disp == 0:
            assembly.sv["_PorePressureGradient_gp"] = [0, 0, 0]
            return
        assembly.sv["_PorePressureGradient_gp"] = [
            0 if op == 0 else assembly.get_gp_results(op, disp)
            for op in self._op_grad_p
        ]

    def get_weak_equation(self, assembly, pb):
        # Mobility tensor k(J) / mu_f at gauss points
        if "lnJ" in assembly.sv:
            J = np.exp(assembly.sv["lnJ"])
        else:
            J = None
        K_mob = self.fluid_props.get_mobility(J=J, sv=assembly.sv)

        # Tangent: grad(delta p) . K . grad(p_inc)
        diff_op = sum(
            [
                0
                if self._op_grad_p_vir[i] == 0
                else self._op_grad_p_vir[i]
                * sum(
                    [
                        0
                        if self._op_grad_p[j] == 0
                        else self._op_grad_p[j] * K_mob[i][j]
                        for j in range(3)
                    ]
                )
                for i in range(3)
            ]
        )

        # Residual: grad(delta p) . K . grad(p_curr)
        grad_p_curr = assembly.sv.get("_PorePressureGradient_gp", [0, 0, 0])
        diff_op += sum(
            [
                0
                if self._op_grad_p_vir[i] == 0
                else self._op_grad_p_vir[i]
                * sum(
                    [
                        grad_p_curr[j] * K_mob[i][j]
                        for j in range(3)
                        if not np.array_equal(K_mob[i][j], 0)
                        and not np.array_equal(grad_p_curr[j], 0)
                    ]
                )
                for i in range(3)
            ]
        )
        return diff_op


# ----------------------------------------------------------------------
# 3. Mass balance: storage + volumetric coupling (transient terms)
# ----------------------------------------------------------------------
class PoroMassStorage(WeakFormBase):
    """Transient storage and volumetric coupling of the fluid mass balance.

    Backward Euler:

      ``(1/M)/dt * delta_p * (p^{n+1} - p^n) + (alpha/dt) * delta_p
        * (tr eps^{n+1} - tr eps^n)``

    The tangent contributions ``(1/M)/dt * delta_p * p_inc`` and
    ``(alpha/dt) * delta_p * tr(delta_eps)`` build the diagonal storage
    block ``K_pp`` and the off-diagonal coupling block ``K_pu``.

    Parameters
    ----------
    fluid_props : PoroFluidProperties
    name : str, default ""
    space : ModelingSpace, optional

    Notes
    -----
    Unlike :py:class:`HeatCapacity`, the storage term is
    **not** lumped: mass lumping on ``K_pp`` destroys the inf-sup
    stabilization brought by the storage coefficient and amplifies
    checkerboard pressure oscillations.
    """

    def __init__(self, fluid_props, name="", space=None):
        WeakFormBase.__init__(self, name, space)
        if self.space._dimension == "2Daxi":
            raise NotImplementedError(
                "PoroMassStorage is not implemented for the '2Daxi' modeling space."
            )

        self.fluid_props = fluid_props
        self.space.new_variable("PorePressure")

    def initialize(self, assembly, pb):
        n_gp = assembly.n_gauss_points
        assembly.sv["_PorePressure_gp"] = 0
        # Scalar placeholder (not a zeros array) so the update() guard below can
        # tell "no live value yet" from "PoroMomentum wrote a live array", and
        # recompute tr(eps) itself in the pure Darcy + storage composition.
        assembly.sv.setdefault("_tr_eps_gp", 0)
        assembly.sv["_PorePressure_gp_start"] = np.zeros(n_gp)
        assembly.sv["_tr_eps_gp_start"] = np.zeros(n_gp)

    def set_start(self, assembly, pb):
        # Freeze the converged state of the previous time step.
        p_curr = assembly.sv.get("_PorePressure_gp", 0)
        if np.isscalar(p_curr):
            assembly.sv["_PorePressure_gp_start"] = np.full(
                assembly.n_gauss_points, p_curr, dtype=float
            )
        else:
            assembly.sv["_PorePressure_gp_start"] = np.asarray(p_curr).copy()

        tr_curr = assembly.sv.get("_tr_eps_gp", 0)
        if np.isscalar(tr_curr):
            assembly.sv["_tr_eps_gp_start"] = np.full(
                assembly.n_gauss_points, tr_curr, dtype=float
            )
        else:
            assembly.sv["_tr_eps_gp_start"] = np.asarray(tr_curr).copy()

    def update(self, assembly, pb):
        disp = pb.get_dof_solution()
        if np.isscalar(disp) and disp == 0:
            assembly.sv["_PorePressure_gp"] = 0
            assembly.sv["_tr_eps_gp"] = 0
            return

        # If PoroMomentum sits in the same assembly it has already populated
        # these arrays; otherwise we compute them locally so that PoroDarcy +
        # PoroMassStorage can be used without PoroMomentum (pure Darcy case).
        # Both guards recompute whenever the value is still the scalar
        # placeholder (i.e. no PoroMomentum wrote a live gauss-point array).
        if "_PorePressure_gp" not in assembly.sv or np.isscalar(
            assembly.sv["_PorePressure_gp"]
        ):
            assembly.sv["_PorePressure_gp"] = assembly.get_gp_results(
                self.space.variable("PorePressure"), disp
            )
        if "_tr_eps_gp" not in assembly.sv or np.isscalar(assembly.sv["_tr_eps_gp"]):
            eps_op = self.space.op_strain()
            tr_eps = 0
            for i in range(3):
                if eps_op[i] != 0:
                    tr_eps = tr_eps + assembly.get_gp_results(eps_op[i], disp)
            assembly.sv["_tr_eps_gp"] = tr_eps

    def get_weak_equation(self, assembly, pb):
        if pb.dtime == 0:
            return 0

        inv_M = self.fluid_props.get_storage()
        alpha = self.fluid_props.biot_coefficient

        p_inc = self.space.variable("PorePressure")
        p_curr = assembly.sv.get("_PorePressure_gp", 0)
        p_start = assembly.sv.get(
            "_PorePressure_gp_start", np.zeros(assembly.n_gauss_points)
        )

        eps_op = self.space.op_strain()
        eps_vol_inc = sum([eps_op[i] for i in range(3) if eps_op[i] != 0])

        tr_eps_curr = assembly.sv.get("_tr_eps_gp", np.zeros(assembly.n_gauss_points))
        tr_eps_start = assembly.sv.get(
            "_tr_eps_gp_start", np.zeros(assembly.n_gauss_points)
        )

        inv_dt = 1.0 / pb.dtime

        diff_op = 0

        # Storage: (1/M)/dt * delta_p * (p_inc + (p_curr - p_start))
        if inv_M != 0.0:
            diff_op = diff_op + inv_dt * inv_M * (
                p_inc.virtual * p_inc + p_inc.virtual * (p_curr - p_start)
            )

        # Volumetric coupling: +alpha/dt * delta_p * (tr(eps_inc) + (tr_curr - tr_start)).
        # The mass-balance residual is +alpha/dt * (tr_curr - tr_start) (the
        # backward-Euler rate of the Biot volumetric storage). With fedoo's
        # D = -R convention this term assembles K_pu = +alpha/dt and a residual
        # that matches it, forming the consistent Biot Jacobian together with
        # K_up = -alpha from PoroMomentum. (Validated: Terzaghi p>0 then
        # consolidates; Mandel p>0 with a non-monotonic Mandel-Cryer peak.)
        if alpha != 0.0 and eps_vol_inc != 0:
            diff_op = diff_op + inv_dt * alpha * (
                p_inc.virtual * eps_vol_inc
                + p_inc.virtual * (tr_eps_curr - tr_eps_start)
            )

        return diff_op


# ----------------------------------------------------------------------
# 1b. Momentum balance — non-mixed variant (no skeleton Lagrange multiplier)
# ----------------------------------------------------------------------
class PoroMomentumSimple(StressEquilibrium):
    """Momentum balance for a saturated porous skeleton — non-mixed variant.

    Inherits directly from :py:class:`StressEquilibrium` (no skeleton
    volumetric Lagrange multiplier ``Pressure``). Use this variant when:

      * the skeleton is compressible enough that quasi-incompressibility is
        not a concern (Poisson ratio not too close to 0.5);
      * OR the boundary conditions include a **free-traction face** (no
        Dirichlet on ``u`` on every boundary), which causes the mixed
        formulation to oscillate due to the under-constrained Lagrange
        multiplier — typical Mandel, unconfined cartilage compression,
        skin indentation.

    The Biot/Terzaghi coupling ``sigma_total = sigma_eff - alpha * p * I``
    is added at the weak-form level. The skeleton constitutive law sees
    only the strain and returns the effective stress, unchanged.

    Parameters
    ----------
    constitutivelaw : ConstitutiveLaw
        Skeleton constitutive law. Any standard ``fedoo.constitutivelaw``
        law works (``ElasticIsotrop``, ``Simcoon`` with ``NEOHC``, etc.).
    fluid_props : PoroFluidProperties
    name : str, default ""
    nlgeom : bool or {'TL', 'UL'}, optional
    space : ModelingSpace, optional
    """

    def __init__(self, constitutivelaw, fluid_props, name="", nlgeom=None, space=None):
        super().__init__(constitutivelaw, name=name, nlgeom=nlgeom, space=space)
        self.fluid_props = fluid_props
        self.space.new_variable("PorePressure")
        # See PoroMomentum.__init__: disable the symmetric-assembly assumption
        # so the one-sided Biot coupling is not mirrored into a phantom block
        # and the three sub-forms co-assemble in a single shared state dict.
        self.assembly_options["assume_sym"] = False

    def get_weak_equation(self, assembly, pb):
        """Build the momentum weak form augmented with the Biot coupling."""
        diff_op = super().get_weak_equation(assembly, pb)

        alpha = self.fluid_props.biot_coefficient
        if assembly._nlgeom == "TL":
            eps = self.space.op_strain(assembly.sv["DispGradient"])
        else:
            eps = self.space.op_strain()

        p_pore_inc = self.space.variable("PorePressure")
        p_pore_curr = assembly.sv.get("_PorePressure_gp", 0)
        p_pore_total = p_pore_inc + p_pore_curr

        # Same sign convention as PoroMomentum (assume_sym=False): -alpha here
        # gives the physically correct K_up = -alpha and PorePressure > 0 in
        # compression, paired with +alpha/dt in PoroMassStorage.
        diff_op -= alpha * sum(
            [0 if eps[i] == 0 else eps[i].virtual * p_pore_total for i in range(3)]
        )
        return diff_op

    def initialize(self, assembly, pb):
        super().initialize(assembly, pb)
        assembly.sv["_PorePressure_gp"] = 0
        assembly.sv["_tr_eps_gp"] = np.zeros(assembly.n_gauss_points)

    def update(self, assembly, pb):
        super().update(assembly, pb)
        disp = pb.get_dof_solution()
        if np.isscalar(disp) and disp == 0:
            assembly.sv["_PorePressure_gp"] = 0
            assembly.sv["_tr_eps_gp"] = np.zeros(assembly.n_gauss_points)
            return

        assembly.sv["_PorePressure_gp"] = assembly.get_gp_results(
            self.space.variable("PorePressure"), disp
        )
        if assembly._nlgeom and "lnJ" in assembly.sv:
            assembly.sv["_tr_eps_gp"] = assembly.sv["lnJ"]
        else:
            eps_op = self.space.op_strain()
            tr_eps = 0
            for i in range(3):
                if eps_op[i] != 0:
                    tr_eps = tr_eps + assembly.get_gp_results(eps_op[i], disp)
            assembly.sv["_tr_eps_gp"] = tr_eps


# ----------------------------------------------------------------------
# 4. Composite weak forms
# ----------------------------------------------------------------------
class PoroMechanics(WeakFormSum):
    """Full mixed ``(u, PorePressure, Pressure)`` poromechanics weak form.

    This composite weak form contains:

    * :class:`PoroMomentum` — momentum balance with Terzaghi coupling.
    * :class:`PoroDarcy` — steady Darcy diffusion of pore pressure.
    * :class:`PoroMassStorage` — storage and volumetric coupling.

    Parameters
    ----------
    skeleton_law : ConstitutiveLaw
        Skeleton constitutive law. This can be
        :class:`fedoo.constitutivelaw.ElasticIsotrop` for small-strain
        analysis or a suitable finite-strain constitutive law.
    fluid_props : PoroFluidProperties
        Fluid-phase constitutive properties.
    bulk_modulus : float, optional
        Skeleton bulk modulus. Required when ``nlgeom`` is not ``False``.
    name : str, optional
        Name of the weak form. Defaults to the skeleton-law name or
        ``"poromechanics"``.
    nlgeom : bool or {'TL', 'UL'}, optional
        Geometric-nonlinearity formulation.
    space : ModelingSpace, optional
        Modeling space. Defaults to the active modeling space.
    """

    def __init__(
        self,
        skeleton_law,
        fluid_props,
        bulk_modulus=None,
        name="",
        nlgeom=None,
        space=None,
    ):
        momentum = PoroMomentum(
            skeleton_law,
            fluid_props,
            bulk_modulus=bulk_modulus,
            name="",
            nlgeom=nlgeom,
            space=space,
        )
        darcy = PoroDarcy(fluid_props, name="", space=space)
        storage = PoroMassStorage(fluid_props, name="", space=space)
        if name == "":
            name = getattr(skeleton_law, "name", "") or "poromechanics"
        super().__init__([momentum, darcy, storage], name)


class PoroMechanicsSimple(WeakFormSum):
    """Non-mixed ``(u, PorePressure)`` poromechanics weak form.

    This composite weak form contains:

    * :class:`PoroMomentumSimple` — momentum balance and Biot coupling
      without a skeleton-pressure Lagrange multiplier.
    * :class:`PoroDarcy` — Darcy diffusion.
    * :class:`PoroMassStorage` — storage and volumetric coupling.

    Use this variant for problems with free-traction boundaries, such as
    Mandel consolidation, unconfined compression, and soft-tissue indentation,
    where the mixed :class:`PoroMechanics` formulation can oscillate. No
    ``bulk_modulus`` argument is required.

    Parameters
    ----------
    skeleton_law : ConstitutiveLaw
        Skeleton constitutive law.
    fluid_props : PoroFluidProperties
        Fluid-phase constitutive properties.
    name : str, optional
        Name of the weak form. Defaults to the skeleton-law name or
        ``"poromechanics_simple"``.
    nlgeom : bool or {'TL', 'UL'}, optional
        Geometric-nonlinearity formulation.
    space : ModelingSpace, optional
        Modeling space. Defaults to the active modeling space.
    """

    def __init__(
        self,
        skeleton_law,
        fluid_props,
        name="",
        nlgeom=None,
        space=None,
    ):
        momentum = PoroMomentumSimple(
            skeleton_law,
            fluid_props,
            name="",
            nlgeom=nlgeom,
            space=space,
        )
        darcy = PoroDarcy(fluid_props, name="", space=space)
        storage = PoroMassStorage(fluid_props, name="", space=space)
        if name == "":
            name = getattr(skeleton_law, "name", "") or "poromechanics_simple"
        super().__init__([momentum, darcy, storage], name)
