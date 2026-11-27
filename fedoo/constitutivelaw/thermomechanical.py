"""Simcoon thermo-mechanical adapter using the standard lifecycle."""

from copy import copy

import numpy as np

from fedoo.core.mechanical3d import Mechanical3D
from fedoo.core.base import InvalidKinematicStateError
from fedoo.util.voigt_tensors import StressTensorList, StrainTensorList
from fedoo.weakform._reference import deformation_gradient
from ._thermal import (
    check_absolute_temperature,
    conductivity_tensor,
    thermal_fields,
    initialize_thermal_state,
    commit_thermal_fields,
)


class SimcoonThermoMechanical(Mechanical3D):
    """Simcoon ``umat_T`` material, at small or finite strain.

    The returned heat includes capacity: HeatEquation therefore does not add a
    second HeatCapacity term. Attach FIRST_ORDER/BackwardEuler to declare the
    transient thermal problem; umat_T owns this local discretization.
    Conductivity is supplied separately: umat_T returns a local thermal
    balance, not a heat flux.

    In finite strain the law runs under Simcoon's total form, on the material
    logarithmic strain: ``alpha`` is then a logarithmic expansion coefficient
    and ``props[0]`` the reference density.

    Temperatures are necessarily absolute temperatures in kelvin (never
    degrees Celsius) and must stay strictly positive: set the initial field
    with ``pb.set_dof("Temp", T0)`` before solving.

    Parameters
    ----------
    umat_name : str
        Name of a thermomechanical Simcoon law (ELISO, ELIST, ELORT, EPICP,
        EPKCP, EPJCK, ZENER, ZENNK, PRONK, shape memory alloy laws).
    props : array_like
        Thermal UMAT properties, which differ from the mechanical UMAT layout:
        ``[rho, c_p, ...]``, e.g. ``[rho, c_p, E, nu, alpha]`` for ELISO.
        ``c_p`` is the capacity per mass at constant stress, whereas
        ``ThermalProperties.specific_heat`` is the one at constant strain.
    thermal_conductivity : float or array_like
        Scalar or symmetric positive-semidefinite 3x3 conductivity tensor.
    n_statev : int, optional
        Number of state variables. Required except for ELISO (1).
    reference_temperature : float, optional
        Stress-free temperature in kelvin. If omitted, the temperature of each
        point at initialization, as Simcoon does.
    tangent_mode : int, default=2
        Simcoon tangent: 0 elastic, 1 continuum, 2 algorithmic.
    name : str, optional
        Name of the constitutive law.
    """

    is_isotropic = True
    is_thermomechanical = True
    thermal_storage = "included"

    def initialize(self, assembly, pb):
        initialize_thermal_state(self, assembly, pb)
        self._evaluate(assembly, pb, start=True)
        self.is_initialized = True

    def update(self, assembly, pb):
        self._evaluate(assembly, pb, start=False)

    def set_start(self, assembly, pb):
        commit_thermal_fields(assembly)
        self._evaluate(
            assembly, pb, start=False, committed=dict(assembly.sv), frozen=True
        )

    def to_start(self, assembly, pb):
        restored = copy(assembly)
        restored.sv = dict(assembly.sv_start)
        self._evaluate(restored, pb, start=False, committed=restored.sv, frozen=True)
        assembly.sv_start = dict(restored.sv)

    def __init__(
        self,
        umat_name,
        props,
        thermal_conductivity,
        n_statev=None,
        reference_temperature=None,
        tangent_mode=2,
        name="",
    ):
        props = np.asarray(props, dtype=float)
        if props.ndim != 1 or props.size < 5 or not np.all(np.isfinite(props)):
            raise ValueError(
                "props must be a finite one-dimensional thermal UMAT property vector."
            )
        minimum_props, minimum_statev = {
            "ELISO": (5, 1),
            "EPICP": (8, 8),
            "EPKCP": (9, 14),
        }.get(umat_name, (2, 1))
        if props.size < minimum_props:
            raise ValueError(
                f"{umat_name} needs at least {minimum_props} thermal properties."
            )
        if props[0] <= 0 or props[1] <= 0:
            raise ValueError("Thermal UMAT requires rho>0 and c_p>0.")
        if umat_name in ("ELISO", "EPICP", "EPKCP") and (
            props[2] <= 0 or not -1 < props[3] < 0.5
        ):
            raise ValueError("Thermal UMAT requires E>0 and -1<nu<0.5.")
        if n_statev is None:
            if umat_name != "ELISO":
                raise ValueError("Specify n_statev for this thermal UMAT.")
            n_statev = 1
        if int(n_statev) != n_statev or n_statev < 1:
            raise ValueError("n_statev must be a positive integer.")
        if n_statev < minimum_statev:
            raise ValueError(
                f"{umat_name} needs at least {minimum_statev} state variables."
            )
        super().__init__(density=props[0], name=name)
        self.umat_name = umat_name
        self.props = np.asfortranarray(props[:, None])
        self.n_statev = int(n_statev)
        # None: Simcoon takes the initial temperature of each point as reference
        self._reference_given = reference_temperature is not None
        self.reference_temperature = (
            float(reference_temperature) if self._reference_given else 293.15
        )
        self.tangent_mode = tangent_mode
        self.thermal_conductivity = conductivity_tensor(thermal_conductivity)
        self.specific_heat = props[1]

    def set_density(self, density):
        """Refuse a density that differs from ``props[0]``, read by ``umat_T``."""
        if not np.array_equal(density, self.props[0, 0]):
            raise ValueError(
                "The density of a Simcoon thermo-mechanical material is its "
                "first property (props[0]), also used by umat_T."
            )
        return super().set_density(density)

    def get_tangent_matrix(self, assembly, dimension=None):
        return assembly.sv["TangentMatrix"]

    def _evaluate(self, assembly, pb, start, committed=None, frozen=False):
        import simcoon as sim

        finite = "_TM_LogStrain" in assembly.sv
        strain, temperature = thermal_fields(self, assembly, pb)
        options = {}
        if finite:
            # total form: ln U and its dual stress, exact maps to dS/dE
            F = np.array(deformation_gradient(assembly)[0], order="F")
            strain = sim.Log_strain(
                np.asfortranarray(F.transpose(1, 0, 2)), True, False
            )
            options = dict(
                F0=F if start else np.asfortranarray(assembly.sv["_TM_FStart"]),
                F1=F,
                kinematic_form="total",
                tangent_output="material",
            )
            assembly.sv["_TM_LogStrain"] = strain
            if getattr(assembly, "_kinematic_form", None) != "total":
                assembly.sv["Strain"] = StrainTensorList(sim.Log_strain(F, True, False))
            else:
                assembly.sv["Strain"] = StrainTensorList(strain)
        else:
            assembly.sv["Strain"] = StrainTensorList(strain)
        assembly.sv["Temp"] = temperature
        check_absolute_temperature(temperature, initial=start)
        n = assembly.n_gauss_points
        if committed is None:
            committed = assembly.sv if start else assembly.sv_start
        statev = np.array(
            committed.get("Statev", np.zeros((self.n_statev, n))), order="F", copy=True
        )
        stress = committed.get("Stress", np.zeros((6, n)))
        stress = stress.asarray() if hasattr(stress, "asarray") else stress
        stress = np.broadcast_to(np.asarray(stress), (6, n)).copy(order="F")
        wm = np.array(committed.get("Wm", np.zeros((4, n))), order="F", copy=True)
        wt = np.array(committed.get("Wt", np.zeros((3, n))), order="F", copy=True)
        if frozen:
            state = stress.copy(order="F"), statev.copy(order="F"), wm.copy(), wt.copy()
        temperature_start = temperature if start else assembly.sv["_TM_TempStart"]
        strain_start = strain if start else assembly.sv["_TM_StrainStart"]
        rotation = np.repeat(np.eye(3)[:, :, None], n, axis=2)
        try:
            response = sim.umat_T(
                self.umat_name,
                np.asfortranarray(strain_start),
                np.asfortranarray(strain - strain_start),
                stress,
                np.asfortranarray(rotation),
                self.props,
                statev,
                pb.time,
                0.0 if start else pb.dtime,
                wm,
                wt,
                temperature_start,
                temperature - temperature_start,
                ndi=3,
                n_threads=1,
                tangent_mode=self.tangent_mode,
                start=start,
                **options,
            )
        except sim.StepCut as exc:
            raise InvalidKinematicStateError(str(exc)) from exc
        stress, statev, wm, wt, source, C, dsdT, drde, drdT = response
        if frozen:
            # tangents of the next step only: a rate-dependent law must not age
            stress, statev, wm, wt = state
        if start and self._reference_given:
            # Simcoon resets statev[0] to the temperature of the start call
            statev = np.array(statev, order="F")
            statev[0] = self.reference_temperature
        source = np.asarray(source).reshape(n)
        drde = np.asarray(drde).reshape(6, n)
        drdT = np.asarray(drdT).reshape(n)
        if not start and pb.dtime > 0:
            # umat_T returns the heat of the increment (dtime times the
            # source, Simcoon >= 2.3); HeatEquation integrates a rate.
            source = source / pb.dtime
            drde = drde / pb.dtime
            drdT = drdT / pb.dtime
        stress = StressTensorList(stress)
        assembly.sv.update(
            Stress=stress,
            Statev=statev,
            Wm=wm,
            Wt=wt,
            HeatSource=source,
            HeatSourceTemperatureTangent=drdT,
        )
        if not finite:
            assembly.sv.update(
                TangentMatrix=C,
                StressTemperatureTangent=dsdT,
                HeatSourceStrainTangent=drde,
            )
            return
        # dS/dE, dS/dT and dr/dE at fixed F; heat per unit reference volume
        F = options["F1"]
        dsdT = StressTensorList(np.asarray(dsdT).reshape(6, n))
        if assembly._nlgeom == "UL":
            C = sim.Lt_convert(C, F, stress.asarray(), "DSDE_2_Dsigma_LieDD")
            dsdT = dsdT.pk2_to_cauchy(F)
        assembly.sv["TangentMatrix"] = C
        assembly._tangent_converted = True
        assembly.sv["PK2"] = stress.cauchy_to_pk2(F)
        assembly.sv["StressTemperatureTangent"] = dsdT.asarray()
        # dr = drdE:dE with dE = sym(F^T dF): dr = (F drdE):dF
        assembly.sv["HeatSourceDeformationTangent"] = np.einsum(
            "iAn,ABn->iBn", F, StressTensorList(drde).to_tensor()
        )
