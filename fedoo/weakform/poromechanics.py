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
finite-strain formulation follows the simcoon conventions: log_R
corotational strain in updated Lagrangian (UL), where the Cauchy coupling
``-alpha * p * tr(delta_eps)`` on the deformed mesh is exact, and the Miehe
logarithmic-strain formalism in total Lagrangian (TL), where the coupling is
expressed against the volumetric log-strain variation
``delta(tr ln U) = delta(ln J) = C^{-1} : delta(E)`` — i.e. the momentum
contribution is ``-alpha * p * J * (C^{-1} : delta E)``, the exact pull-back
of the Cauchy Terzaghi split to the PK2/Green-Lagrange pair.
"""

import warnings

import numpy as np

from fedoo.core.weakform import WeakFormBase, WeakFormSum
from fedoo.weakform.stress_equilibrium import StressEquilibrium
from fedoo.weakform.stress_equilibrium_mixed import StressEquilibriumMixed


def _comp_lnJ(assembly):
    """Store lnJ = ln(det F) at gauss points from the deformation gradient.

    Only needed for weakforms built on the plain :py:class:`StressEquilibrium`
    (the mixed parent computes lnJ itself in ``_comp_F``).
    """
    F = assembly.sv["F"]
    J = np.linalg.det(np.ascontiguousarray(F.transpose((2, 0, 1))))
    assembly.sv["lnJ"] = np.log(J)


def _comp_dlnJ_weights(assembly, f_is_isochoric):
    """Compute and cache the Voigt weights of ``delta(lnJ) = C^{-1}:delta E``.

    Total Lagrangian only. In TL the weak form is assembled against the
    Green-Lagrange variation ``delta E``, while the natural volumetric
    coupling of the Miehe log-strain formalism is written against
    ``delta(tr ln U) = delta(ln J) = C^{-1} : delta E``. This stores in
    ``assembly.sv["_dlnJ_weights_gp"]`` the six coefficient arrays ``m_k``
    such that ``delta(lnJ) = sum_k m_k deltaE_k`` in fedoo's Voigt order
    ``[xx, yy, zz, xy, xz, yz]`` (engineering shear: each off-diagonal
    ``C^{-1}`` entry appears once against ``gamma = 2E``).

    Called once per iteration from the momentum ``update`` and shared by the
    two consumers: :py:class:`PoroMassStorage` uses ``m_k`` directly (exact
    tangent of the ``lnJ`` storage residual, ``K_pu`` block) and the momentum
    coupling rescales by ``J`` (exact pull-back of the Cauchy coupling
    ``-alpha p I`` to PK2).

    ``f_is_isochoric`` states whether ``sv["F"]`` holds the isochoric
    ``F_bar`` of the mixed parent — then the full inverse metric is
    ``C^{-1} = J^{-2/3} C_bar^{-1}`` with ``J`` from the total ``lnJ`` — or
    the full ``F``. The cache is set to None (consumers fall back to the
    plain trace, identity ``C``) when the kinematic state is unavailable or
    ``C`` is numerically singular on a distorted trial Newton iterate, so
    the assembly stays usable and the NR failure path (dt reduction) can
    handle the iterate.
    """
    F = assembly.sv.get("F")
    if not isinstance(F, np.ndarray) or F.ndim != 3:
        assembly.sv["_dlnJ_weights_gp"] = None
        return
    Ft = np.ascontiguousarray(F.transpose((2, 0, 1)))
    C = np.matmul(Ft.transpose((0, 2, 1)), Ft)
    try:
        Cinv = np.linalg.inv(C)
    except np.linalg.LinAlgError:
        assembly.sv["_dlnJ_weights_gp"] = None
        return
    if f_is_isochoric:
        scale = np.exp(assembly.sv["lnJ"]) ** (-2.0 / 3.0)
    else:
        scale = 1.0
    assembly.sv["_dlnJ_weights_gp"] = [
        scale * Cinv[:, 0, 0],
        scale * Cinv[:, 1, 1],
        scale * Cinv[:, 2, 2],
        scale * Cinv[:, 0, 1],
        scale * Cinv[:, 0, 2],
        scale * Cinv[:, 1, 2],
    ]


def _ul_pore_geometric_tangent(space, p_tilde):
    """Geometric (initial-stress-like) tangent of the pore stress in UL.

    In updated Lagrangian the coupling residual is assembled on the
    deformed mesh: ``G_p(u) = -alpha int(p tr(delta_eps) dv)``. Both the
    spatial gradient of the test functions and the volume element depend on
    the displacement, so the exact linearization at fixed ``p`` carries a
    geometric term::

        DG_p . du = alpha p int( tr(grad(delta_u) grad(du))
                                 - tr(delta_eps) tr(eps(du)) ) dv

    (from ``d(dN/dx) = -dN/dx grad(du)`` and ``d(dv) = tr(grad du) dv``).
    fedoo's UL machinery embeds the analogous terms of the *constitutive*
    stress in the converted (logarithmic-rate) tangent; the pore stress is
    added at the weak-form level and bypasses that path, so the term must
    be added explicitly. It needs the full displacement-gradient operators
    (the spin part does not cancel), hence ``op_grad_u`` and not
    ``op_strain``. This is a tangent-only (matrix) contribution: it cannot
    change the converged solution, only restore Newton convergence — without
    it the tangent error grows with p and Newton stalls at finite strain.
    """
    gu = space.op_grad_u()
    diff_op = 0
    for k in range(3):
        for l in range(3):
            if gu[k][l] != 0 and gu[l][k] != 0:
                diff_op += gu[k][l].virtual * (gu[l][k] * p_tilde)
    tr_op = space.op_div_u()
    diff_op -= tr_op.virtual * (tr_op * p_tilde)
    return diff_op


def _biot_coupling_diff_op(space, assembly, alpha):
    """Biot/Terzaghi coupling contribution to the momentum weak form.

    ``sigma_total = sigma_eff - alpha * p_pore * I``. In fedoo's weak-form
    convention the assembled D vector carries ``-R(U_curr)`` while the matrix
    is ``+dR/dU``. With ``assume_sym=False`` (the genuine, non-mirrored
    assembly) the ``-alpha`` sign gives the physically correct tangent
    ``K_up = -alpha`` and a pore pressure positive in compression (validated
    on Terzaghi consolidation and the Mandel benchmark); it pairs with
    ``+alpha/dt`` in :py:class:`PoroMassStorage` to form the consistent
    (non-symmetric) Biot Jacobian.

    Small strain and UL use the Cauchy form ``-alpha p tr(delta_eps)``
    (exact on the deformed mesh in UL), complemented in UL by the geometric
    tangent of the pore stress. TL uses the lnU-consistent form
    ``-alpha p J (C^{-1}:delta E)`` with ``J C^{-1}`` evaluated at the
    current state (frozen-kinematics tangent: the geometric derivative
    ``d(J C^{-1})/du * p`` is neglected in the matrix; the residual is
    exact, so Newton converges to the right solution).
    """
    p_pore_inc = space.variable("PorePressure")
    p_pore_curr = assembly.sv.get("_PorePressure_gp", 0)
    p_pore_total = p_pore_inc + p_pore_curr

    m_vol = None
    if assembly._nlgeom == "TL":
        eps = space.op_strain(assembly.sv["DispGradient"])
        m_vol = assembly.sv.get("_dlnJ_weights_gp")
        if m_vol is not None:
            J = np.exp(assembly.sv["lnJ"])
            m_vol = [J * m for m in m_vol]
    else:
        eps = space.op_strain()

    if m_vol is None:
        diff_op = -alpha * sum(
            [0 if eps[i] == 0 else eps[i].virtual * p_pore_total for i in range(3)]
        )
        if assembly._nlgeom == "UL" and not (
            np.isscalar(p_pore_curr) and p_pore_curr == 0
        ):
            diff_op += _ul_pore_geometric_tangent(space, alpha * p_pore_curr)
    else:
        diff_op = -alpha * sum(
            [
                0 if eps[i] == 0 else eps[i].virtual * (p_pore_total * m_vol[i])
                for i in range(6)
            ]
        )
    return diff_op


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
        Skeleton constitutive law. Must satisfy ``_Lt_from_F = True`` in
        large-strain mode (this is the case for all simcoon hyperelastic
        and visco-elastic laws).
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
        if self.space._dimension == "2Daxi":
            # The parent supports 2Daxi, but the Biot coupling terms added
            # here carry neither the 2*pi*r integration weight nor the hoop
            # strain contribution: refuse rather than assemble silently wrong.
            raise NotImplementedError(
                "PoroMomentum is not implemented for the '2Daxi' modeling space."
            )
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
        return diff_op + _biot_coupling_diff_op(
            self.space, assembly, self.fluid_props.biot_coefficient
        )

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
            if assembly._nlgeom == "TL":
                assembly.sv["_dlnJ_weights_gp"] = None
            return

        assembly.sv["_PorePressure_gp"] = assembly.get_gp_results(
            self.space.variable("PorePressure"), disp
        )
        if assembly._nlgeom:
            # lnJ is stored by the mixed parent's _comp_F.
            assembly.sv["_tr_eps_gp"] = assembly.sv["lnJ"]
            if assembly._nlgeom == "TL":
                # sv["F"] holds the isochoric F_bar in the mixed formulation.
                _comp_dlnJ_weights(assembly, f_is_isochoric=True)
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
        self._warned_no_J = False

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
        # Mobility tensor k(J) / mu_f at gauss points. A scalar lnJ (e.g. a
        # uniform value provided by the user) is honored like an array.
        lnJ = assembly.sv.get("lnJ")
        if lnJ is not None:
            J = np.exp(lnJ)
        else:
            J = None
            if (
                getattr(assembly, "_nlgeom", False)
                and callable(self.fluid_props.permeability)
                and not self._warned_no_J
            ):
                self._warned_no_J = True
                warnings.warn(
                    "PoroDarcy: the assembly runs in finite strain (nlgeom) "
                    "but no 'lnJ' state variable is available, so the "
                    "deformation-dependent permeability falls back to its "
                    "reference value k(J=1). Use PoroMechanics or "
                    "PoroMechanicsSimple (which compute lnJ), or provide lnJ "
                    "in assembly.sv."
                )
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

        # Volumetric strain-increment operator of the K_pu block. In small
        # strain and UL this is tr(delta_eps) (exact on the deformed mesh in
        # UL). In TL the residual is written in lnJ, whose exact variation is
        # delta(lnJ) = C^{-1}:delta(E) — no J factor here, unlike the J C^{-1}
        # of K_up: the Biot Jacobian is inherently non-symmetric.
        if getattr(assembly, "_nlgeom", None) == "TL":
            eps_op = self.space.op_strain(assembly.sv.get("DispGradient", 0))
            m_vol = assembly.sv.get("_dlnJ_weights_gp")
        else:
            eps_op = self.space.op_strain()
            m_vol = None

        if m_vol is None:
            eps_vol_inc = sum([eps_op[i] for i in range(3) if eps_op[i] != 0])
        else:
            eps_vol_inc = sum(
                [eps_op[i] * m_vol[i] for i in range(6) if eps_op[i] != 0]
            )

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
        if self.space._dimension == "2Daxi":
            # See PoroMomentum.__init__: the Biot coupling terms lack the
            # 2*pi*r measure and the hoop strain contribution.
            raise NotImplementedError(
                "PoroMomentumSimple is not implemented for the '2Daxi' "
                "modeling space."
            )
        # See PoroMomentum.__init__: disable the symmetric-assembly assumption
        # so the one-sided Biot coupling is not mirrored into a phantom block
        # and the three sub-forms co-assemble in a single shared state dict.
        self.assembly_options["assume_sym"] = False

    def get_weak_equation(self, assembly, pb):
        """Build the momentum weak form augmented with the Biot coupling."""
        diff_op = super().get_weak_equation(assembly, pb)
        return diff_op + _biot_coupling_diff_op(
            self.space, assembly, self.fluid_props.biot_coefficient
        )

    def initialize(self, assembly, pb):
        super().initialize(assembly, pb)
        assembly.sv["_PorePressure_gp"] = 0
        assembly.sv["_tr_eps_gp"] = np.zeros(assembly.n_gauss_points)
        if assembly._nlgeom:
            assembly.sv["lnJ"] = np.zeros(assembly.n_gauss_points)

    def update(self, assembly, pb):
        super().update(assembly, pb)
        disp = pb.get_dof_solution()
        if np.isscalar(disp) and disp == 0:
            assembly.sv["_PorePressure_gp"] = 0
            assembly.sv["_tr_eps_gp"] = np.zeros(assembly.n_gauss_points)
            if assembly._nlgeom:
                assembly.sv["lnJ"] = np.zeros(assembly.n_gauss_points)
                if assembly._nlgeom == "TL":
                    assembly.sv["_dlnJ_weights_gp"] = None
            return

        assembly.sv["_PorePressure_gp"] = assembly.get_gp_results(
            self.space.variable("PorePressure"), disp
        )
        if assembly._nlgeom:
            # Plain StressEquilibrium stores the full deformation gradient in
            # sv["F"] but never lnJ (only the mixed parent computes it).
            # Compute it here so the storage coupling uses the true log-volume
            # change and PoroDarcy can feed J to deformation-dependent
            # permeability models (Holmes-Mow, Kozeny-Carman).
            _comp_lnJ(assembly)
            assembly.sv["_tr_eps_gp"] = assembly.sv["lnJ"]
            if assembly._nlgeom == "TL":
                _comp_dlnJ_weights(assembly, f_is_isochoric=False)
        else:
            eps_op = self.space.op_strain()
            tr_eps = 0
            for i in range(3):
                if eps_op[i] != 0:
                    tr_eps = tr_eps + assembly.get_gp_results(eps_op[i], disp)
            assembly.sv["_tr_eps_gp"] = tr_eps


# ----------------------------------------------------------------------
# 4. Factory
# ----------------------------------------------------------------------
def PoroMechanics(
    skeleton_law,
    fluid_props,
    bulk_modulus=None,
    name="",
    nlgeom=None,
    space=None,
):
    """Build the full (u, PorePressure, Pressure) poromechanics weak form.

    Returns a :py:class:`WeakFormSum` of:

      * :py:class:`PoroMomentum` — momentum balance with Terzaghi coupling
      * :py:class:`PoroDarcy` — steady Darcy diffusion of pore pressure
      * :py:class:`PoroMassStorage` — storage and volumetric coupling

    Reserve this three-field variant for a genuinely quasi-incompressible
    **skeleton** (drained Poisson ratio close to 0.5) with well-constrained
    (confined) boundaries: with a free-traction face the skeleton Lagrange
    multiplier is under-constrained and the solution oscillates — use
    :py:func:`PoroMechanicsSimple` instead (the recommended default).

    Parameters
    ----------
    skeleton_law : ConstitutiveLaw
        Skeleton constitutive law. Use ``fedoo.constitutivelaw.Simcoon`` with
        any of the generic hyperelastic laws (``NEOHC``, ``MOORI``,
        ``YEOHH``, ``ISHAH``, ``GETHH``, ``SWANH``), or
        ``fedoo.constitutivelaw.ElasticIsotrop`` for small-strain validation.
    fluid_props : PoroFluidProperties
    bulk_modulus : float, optional
        Skeleton bulk modulus (required when ``nlgeom`` is not ``False``).
    name : str, default ""
    nlgeom : bool or {'TL', 'UL'}, optional
    space : ModelingSpace, optional

    Returns
    -------
    WeakFormSum
    """
    wf_mom = PoroMomentum(
        skeleton_law,
        fluid_props,
        bulk_modulus=bulk_modulus,
        name="",
        nlgeom=nlgeom,
        space=space,
    )
    wf_darcy = PoroDarcy(fluid_props, name="", space=space)
    wf_storage = PoroMassStorage(fluid_props, name="", space=space)

    if name == "":
        name = getattr(skeleton_law, "name", "") or "poromechanics"

    return WeakFormSum([wf_mom, wf_darcy, wf_storage], name)


def PoroMechanicsSimple(skeleton_law, fluid_props, name="", nlgeom=None, space=None):
    """Build the non-mixed (u, PorePressure) poromechanics weak form.

    Returns a :py:class:`WeakFormSum` of:

      * :py:class:`PoroMomentumSimple` — momentum + Biot coupling (no
        skeleton Lagrange multiplier)
      * :py:class:`PoroDarcy` — Darcy diffusion
      * :py:class:`PoroMassStorage` — storage + volumetric coupling

    **This is the recommended default variant.** Use it in particular for
    problems with free-traction boundaries (Mandel consolidation, unconfined
    compression of cartilage, soft tissue indentation) where the mixed
    :py:class:`PoroMechanics` oscillates. No ``bulk_modulus`` parameter is
    required. For soft tissues, remember that the near-incompressibility of
    the tissue is the *undrained* response already carried by the
    ``PorePressure`` field: a compressible skeleton law (drained
    ``nu ~ 0.1..0.3``) with this variant is usually the right model. For
    stability in the undrained limit prefer a Taylor-Hood interpolation
    (quadratic ``u``, linear ``PorePressure`` via a ``CombinedElement`` —
    see the poromechanics documentation).

    With simcoon hyperelastic laws use ``nlgeom="UL"`` (log_R corotational,
    the simcoon-native mode) and keep the compression below roughly 12% —
    see the Limitations section of the poromechanics documentation.

    Parameters
    ----------
    skeleton_law : ConstitutiveLaw
        Skeleton constitutive law (``ElasticIsotrop``, simcoon ``NEOHC``,
        ``YEOHH``, ``PRONK``, ...).
    fluid_props : PoroFluidProperties
    name : str, default ""
    nlgeom : bool or {'TL', 'UL'}, optional
    space : ModelingSpace, optional

    Returns
    -------
    WeakFormSum
    """
    wf_mom = PoroMomentumSimple(
        skeleton_law, fluid_props, name="", nlgeom=nlgeom, space=space
    )
    wf_darcy = PoroDarcy(fluid_props, name="", space=space)
    wf_storage = PoroMassStorage(fluid_props, name="", space=space)

    if name == "":
        name = getattr(skeleton_law, "name", "") or "poromechanics_simple"

    return WeakFormSum([wf_mom, wf_darcy, wf_storage], name)
