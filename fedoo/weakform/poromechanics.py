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

        # -alpha * p_pore_total * (delta_eps_xx + delta_eps_yy + delta_eps_zz)
        diff_op += -alpha * sum(
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
    Unlike :py:class:`TemperatureTimeDerivative`, the storage term is
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
        assembly.sv.setdefault("_tr_eps_gp", np.zeros(n_gp))
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
            assembly.sv["_tr_eps_gp"] = np.zeros(assembly.n_gauss_points)
            return

        # If PoroMomentum sits in the same assembly it has already populated
        # these arrays; otherwise we compute them locally so that PoroDarcy +
        # PoroMassStorage can be used without PoroMomentum (pure Darcy case).
        if "_PorePressure_gp" not in assembly.sv or np.isscalar(
            assembly.sv["_PorePressure_gp"]
        ):
            assembly.sv["_PorePressure_gp"] = assembly.get_gp_results(
                self.space.variable("PorePressure"), disp
            )
        if "_tr_eps_gp" not in assembly.sv:
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

        # Volumetric coupling: alpha/dt * delta_p * (tr(eps_inc) + (tr_curr - tr_start))
        if alpha != 0.0 and eps_vol_inc != 0:
            diff_op = diff_op + inv_dt * alpha * (
                p_inc.virtual * eps_vol_inc
                + p_inc.virtual * (tr_eps_curr - tr_eps_start)
            )

        return diff_op


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
