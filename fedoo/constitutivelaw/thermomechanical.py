"""Small-strain Simcoon thermo-mechanical adapter using the standard lifecycle."""

from copy import copy

import numpy as np

from fedoo.core.mechanical3d import Mechanical3D
from fedoo.core.base import InvalidKinematicStateError
from fedoo.util.voigt_tensors import StressTensorList, StrainTensorList
from ._thermal import (
    conductivity_tensor,
    thermal_fields,
    initialize_thermal_state,
    commit_thermal_fields,
)


class SimcoonThermoMechanical(Mechanical3D):
    """Small-strain isotropic Simcoon ``umat_T`` material.

    ``props`` uses the thermal UMAT layout, which differs from the mechanical
    UMAT layout. For ELISO it is [rho, c_p, E, nu, alpha], with one state variable.
    Other kernels require an explicit ``n_statev``. This initial adapter accepts
    the isotropic ELISO, EPICP and EPKCP kernels. Conductivity is supplied
    separately: umat_T returns a local thermal balance, not a heat flux.

    The returned r includes capacity. HeatEquation therefore does not add a
    second HeatCapacity term. Attach FIRST_ORDER/BackwardEuler to declare the
    transient thermal problem; umat_T owns this local discretization. The
    kernel's derivatives are used as supplied (some are continuum tangents).
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
        self._evaluate(assembly, pb, start=False, committed=dict(assembly.sv))

    def to_start(self, assembly, pb):
        restored = copy(assembly)
        restored.sv = dict(assembly.sv_start)
        self._evaluate(restored, pb, start=False, committed=restored.sv)
        assembly.sv_start = dict(restored.sv)

    def __init__(
        self,
        umat_name,
        props,
        thermal_conductivity,
        n_statev=None,
        reference_temperature=293.15,
        tangent_mode=2,
        name="",
    ):
        props = np.asarray(props, dtype=float)
        if props.ndim != 1 or props.size < 5 or not np.all(np.isfinite(props)):
            raise ValueError(
                "props must be a finite one-dimensional thermal UMAT property vector."
            )
        if umat_name not in ("ELISO", "EPICP", "EPKCP"):
            raise NotImplementedError(
                "Initial umat_T support is limited to ELISO, EPICP and EPKCP."
            )
        minimum_props, minimum_statev = {
            "ELISO": (5, 1),
            "EPICP": (8, 8),
            "EPKCP": (9, 14),
        }[umat_name]
        if props.size < minimum_props:
            raise ValueError(
                f"{umat_name} needs at least {minimum_props} thermal properties."
            )
        if props[0] <= 0 or props[1] <= 0 or props[2] <= 0 or not -1 < props[3] < 0.5:
            raise ValueError("Thermal UMAT requires rho>0, c_p>0, E>0 and -1<nu<0.5.")
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
        self.reference_temperature = float(reference_temperature)
        self.tangent_mode = tangent_mode
        self.thermal_conductivity = conductivity_tensor(thermal_conductivity)
        self.specific_heat = props[1]

    def get_tangent_matrix(self, assembly, dimension=None):
        return assembly.sv["TangentMatrix"]

    def _evaluate(self, assembly, pb, start, committed=None):
        import simcoon as sim

        strain, temperature = thermal_fields(self, assembly, pb)
        assembly.sv["Strain"] = StrainTensorList(strain)
        assembly.sv["Temp"] = temperature
        if np.any(temperature <= 0):
            raise ValueError(
                "umat_T requires positive absolute temperatures; set the initial Temp field."
            )
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
        temperature_start = assembly.sv["_TM_TempStart"]
        strain_start = assembly.sv["_TM_StrainStart"]
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
            )
        except sim.StepCut as exc:
            raise InvalidKinematicStateError(str(exc)) from exc
        stress, statev, wm, wt, source, C, dsdT, drde, drdT = response
        assembly.sv.update(
            Stress=StressTensorList(stress),
            Statev=statev,
            Wm=wm,
            Wt=wt,
            TangentMatrix=C,
            StressTemperatureTangent=dsdT,
            HeatSource=np.asarray(source).reshape(n),
            HeatSourceStrainTangent=np.asarray(drde).reshape(6, n),
            HeatSourceTemperatureTangent=np.asarray(drdT).reshape(n),
        )
