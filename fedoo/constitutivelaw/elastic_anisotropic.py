# derive de ConstitutiveLaw
# compatible with the simcoon strain and stress notation

from fedoo.core.mechanical3d import Mechanical3D
from fedoo.util.voigt_tensors import StressTensorList, StrainTensorList
from fedoo.core.time_evolution import FIRST_ORDER
from fedoo.core.base import InvalidKinematicStateError
from .thermal_prop import ThermalProperties
from ._thermal import (
    conductivity_tensor,
    thermal_fields,
    initialize_thermal_state,
    commit_thermal_fields,
)

from copy import copy
import numpy as np
from simcoon import Rotation


class ElasticAnisotropic(Mechanical3D):
    """
    Linear full Anistropic constitutive law defined from the rigidity matrix H.

    The constitutive Law should be associated with
    :mod:`fedoo.weakform.StressEquilibrium`

    Parameters
    ----------
    H : list of list or an array (shape=(6,6)) of scalars or arrays of gauss point values.
        The rigidity matrix.
        If H is a list of gauss point values, the shape shoud be H.shape = (6,6,NumberOfGaussPoints)
    name : str, optional
        The name of the constitutive law
    thermal : ThermalProperties, optional
        Thermal expansion, transport and capacity. May also be attached with
        ``set_thermal_properties`` before constructing the weakforms.

    Notes
    -----
    H is interpreted in the slot ordering documented on
    :class:`fedoo.core.mechanical3d.Mechanical3D`. In ``2Daxi``, slot 2 is
    the **hoop** direction (not Cartesian z): rows/columns labelled "3" of
    H apply to the hoop response. Define the rigidity matrix accordingly,
    or use a local frame.
    """

    thermal = None
    thermal_storage = "separate"

    def __init__(self, H, name="", *, thermal=None):
        Mechanical3D.__init__(self, name=name)  # heritage

        self._H = H
        if thermal is not None:
            self.set_thermal_properties(thermal)
        # self._stress = 0
        # self._grad_disp = 0

    def initialize(self, assembly, pb):
        self.is_initialized = True
        assembly.sv["TangentMatrix"] = self.get_tangent_matrix(assembly)
        if self.is_thermomechanical:
            initialize_thermal_state(self, assembly, pb)
            self.update(assembly, pb)

    @property
    def is_thermomechanical(self):
        return self.thermal is not None

    @property
    def supports_total_formulation(self):
        return self.is_thermomechanical

    @property
    def thermal_conductivity(self):
        return self.thermal.thermal_conductivity

    @property
    def specific_heat(self):
        return self.thermal.specific_heat

    @property
    def alpha(self):
        return self.thermal.alpha

    @property
    def reference_temperature(self):
        return self.thermal.reference_temperature

    def set_thermal_properties(self, thermal=None, **properties):
        """Attach ThermalProperties, or construct them from keyword arguments.

        Call before constructing the weakforms. Returns this material, so it
        can be used fluently. Coupled solids support small and finite strain
        in 3D and plane strain. Finite thermoelasticity uses a reference Hencky
        energy with TL heat integration and requires Simcoon 2.2 conversions.
        """
        if thermal is None:
            thermal = ThermalProperties(**properties)
        elif properties:
            raise TypeError(
                "Provide either ThermalProperties or its constructor keywords."
            )
        if not isinstance(thermal, ThermalProperties):
            raise TypeError("thermal must be a ThermalProperties instance.")
        conductivity_tensor(thermal.thermal_conductivity)
        values = np.asarray([thermal.specific_heat, thermal.density], dtype=float)
        if (
            values.shape != (2,)
            or not np.all(np.isfinite(values))
            or np.any(values <= 0)
        ):
            raise ValueError(
                "Coupled thermal properties require scalar specific_heat>0 and density>0."
            )
        self.thermal = thermal
        self.density = thermal.density
        return self

    def set_start(self, assembly, pb):
        if self.is_thermomechanical:
            commit_thermal_fields(assembly)
            self.update(assembly, pb)

    def to_start(self, assembly, pb):
        if self.is_thermomechanical:
            restored = copy(assembly)
            restored.sv = dict(assembly.sv_start)
            self.update(restored, pb)
            assembly.sv_start = dict(restored.sv)

    def update(self, assembly, pb):
        if self.is_thermomechanical and (
            assembly._nlgeom or "_TM_LogStrain" in assembly.sv
        ):
            self._update_finite_thermal(assembly, pb)
            return
        # linear problem = no need to recompute tangent matrix if it has already been computed
        if not (assembly._nlgeom) and "TangentMatrix" in assembly.sv:
            H = assembly.sv["TangentMatrix"]
        else:
            H = self.get_tangent_matrix(assembly)
            assembly.sv["TangentMatrix"] = H

        if "DStrain" in assembly.sv:
            strain_start = assembly.sv["Strain"]
            if assembly._nlgeom:
                # Evaluate stress in the trial configuration without changing the
                # committed strain, which the weak form transports at set_start.
                rotation = Rotation.from_matrix(assembly.sv["DR"].transpose(2, 0, 1))
                strain_start = StrainTensorList(
                    rotation.apply_strain(
                        self._as_mechanical_array(assembly.sv_start["Strain"])
                    )
                )
            total_strain = strain_start + assembly.sv["DStrain"]
        else:
            total_strain = assembly.sv["Strain"]

        # Handle uninitialized strain (e.g. first Newmark step with zero
        # displacement). The tangent ``H`` has already been stored in
        # ``assembly.sv["TangentMatrix"]`` above, so the early return only
        # skips the stress evaluation (correctly zero for zero strain) — no
        # stale tangent is left behind.
        if (
            np.isscalar(total_strain)
            and total_strain == 0
            and not self.is_thermomechanical
        ):
            assembly.sv["Stress"] = 0
            return

        if self.is_thermomechanical:
            total_strain, temperature = thermal_fields(self, assembly, pb)
            assembly.sv["Temp"] = temperature
            expansion = self.thermal._expansion_vector()
            beta = np.array(
                [
                    sum(assembly.convert_data(H[i][j]) * expansion[j] for j in range(6))
                    + np.zeros(assembly.n_gauss_points)
                    for i in range(6)
                ]
            )
            elastic_strain = total_strain - expansion[:, None] * (
                temperature - self.reference_temperature
            )
            self._update_thermal_response(assembly, pb, total_strain, beta)
        else:
            elastic_strain = total_strain

        stress = StressTensorList(
            [
                sum(
                    [
                        elastic_strain[j] * assembly.convert_data(H[i][j])
                        for j in range(6)
                    ]
                )
                for i in range(6)
            ]
        )  # H[i][j] are converted to gauss point excepted if scalar

        if assembly._nlgeom:
            # In finite strain H acts on the corotational strain to produce
            # Kirchhoff stress. StressEquilibrium integrates true Cauchy
            # stress over the current configuration, hence sigma = tau / J.
            F = np.asarray(assembly.sv["F"])
            J = np.linalg.det(F.transpose(2, 0, 1))
            stress = StressTensorList(stress.asarray() / J)

        assembly.sv["Stress"] = stress

    def _update_finite_thermal(self, assembly, pb):
        """Reference Hencky energy; exact stress and tangent maps for TL/UL."""
        import simcoon as sim

        from fedoo.weakform._reference import deformation_gradient

        F, _, _ = deformation_gradient(assembly)
        h = sim.Log_strain(np.asfortranarray(F.transpose(1, 0, 2)), True, False)
        _, temperature = thermal_fields(self, assembly, pb)
        if np.any(~np.isfinite(temperature)) or np.any(temperature <= 0):
            raise InvalidKinematicStateError(
                "Finite thermoelasticity requires positive absolute temperatures."
            )
        # Material coefficients refer to the initial axes, also with UL
        # mechanics. Do not convect the reference elasticity tensor.
        reference = copy(assembly)
        reference._nlgeom = False
        H = self.get_tangent_matrix(reference)
        C = np.array(
            [
                [
                    assembly.convert_data(H[i][j]) + np.zeros(assembly.n_gauss_points)
                    for j in range(6)
                ]
                for i in range(6)
            ],
            order="F",
        )
        expansion = self.thermal._expansion_vector()
        beta = np.einsum("ijn,j->in", C, expansion)
        elastic_strain = h - expansion[:, None] * (
            temperature - self.reference_temperature
        )
        dual_stress = np.einsum("ijn,jn->in", C, elastic_strain)
        stress = StressTensorList(dual_stress).log_stress_u_to_cauchy(F)
        beta_spatial = StressTensorList(beta).log_stress_u_to_cauchy(F)
        beta_material = beta_spatial.cauchy_to_pk2(F)
        PK2 = stress.cauchy_to_pk2(F)
        tangent = sim.Lt_convert(C, F, stress.asarray(), "DTDH_2_DSDE")
        if assembly._nlgeom == "UL":
            tangent = sim.Lt_convert(
                tangent, F, stress.asarray(), "DSDE_2_Dsigma_LieDD"
            )
        assembly.sv["TangentMatrix"] = tangent
        assembly._tangent_converted = True
        assembly.sv["Stress"] = stress
        assembly.sv["PK2"] = PK2
        assembly.sv["StressTemperatureTangent"] = -(
            beta_material.asarray()
            if assembly._nlgeom == "TL"
            else beta_spatial.asarray()
        )
        assembly.sv["Temp"] = temperature
        assembly.sv["_TM_LogStrain"] = h
        assembly.sv["Strain"] = StrainTensorList(
            h
            if getattr(assembly, "_kinematic_form", None) == "total"
            else sim.Log_strain(F, True, False)
        )
        source = np.broadcast_to(
            np.asarray(self.thermal.heat_source, dtype=float), temperature.shape
        ).copy()
        source_temperature = np.zeros_like(temperature)
        source_deformation = np.zeros_like(F)
        if (
            self.thermal.coupling
            and pb.dtime > 0
            and FIRST_ORDER in pb.time_integrators
        ):
            rate = (
                np.sum(beta * (h - assembly.sv["_TM_StrainStart"]), axis=0) / pb.dtime
            )
            source -= temperature * rate
            source_temperature = -rate
            # beta:delta(ln U) = (F S_beta):delta F. This includes the
            # logarithm derivative, including repeated stretch eigenvalues.
            source_deformation = -np.einsum(
                "iAn,ABn->iBn", F, beta_material.to_tensor()
            ) * (temperature / pb.dtime)
        assembly.sv["HeatSource"] = source
        assembly.sv["HeatSourceTemperatureTangent"] = source_temperature
        assembly.sv["HeatSourceDeformationTangent"] = source_deformation

    def _update_thermal_response(self, assembly, pb, strain, beta):
        assembly.sv["StressTemperatureTangent"] = -beta
        source = np.broadcast_to(
            np.asarray(self.thermal.heat_source, dtype=float),
            (assembly.n_gauss_points,),
        ).copy()
        derivative = np.zeros_like(beta)
        if (
            self.thermal.coupling
            and pb.dtime > 0
            and FIRST_ORDER in pb.time_integrators
        ):
            derivative = -self.reference_temperature * beta / pb.dtime
            source += np.sum(
                derivative * (strain - assembly.sv["_TM_StrainStart"]), axis=0
            )
        assembly.sv["HeatSource"] = source
        assembly.sv["HeatSourceStrainTangent"] = derivative
        assembly.sv["HeatSourceTemperatureTangent"] = 0.0

    def get_stress_from_strain(self, assembly, strain_tensor):
        finite = self.is_thermomechanical and "_TM_LogStrain" in assembly.sv
        if finite:
            reference = copy(assembly)
            reference._nlgeom = False
            H = self.get_tangent_matrix(reference)
        else:
            H = self.get_tangent_matrix(assembly)

        if self.is_thermomechanical:
            strain_tensor = self._as_mechanical_array(strain_tensor)
            if strain_tensor.ndim == 1:
                strain_tensor = strain_tensor[:, None]
            if finite and getattr(assembly, "_kinematic_form", None) != "total":
                # Native UL strain output is ln V; the energy uses ln U.
                F = assembly.sv["F"]
                left, _, right = np.linalg.svd(F.transpose(2, 0, 1))
                R = (left @ right).transpose(1, 2, 0)
                spatial = StrainTensorList(strain_tensor).to_tensor()
                material = np.einsum("iAn,ijn,jBn->ABn", R, spatial, R)
                strain_tensor = material[[0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]].copy()
                strain_tensor[3:] *= 2
            temperature = assembly.sv.get("Temp", self.reference_temperature)
            strain_tensor = strain_tensor - self.thermal._expansion_vector()[
                :, None
            ] * (np.asarray(temperature) - self.reference_temperature)
            H = [[assembly.convert_data(H[i][j]) for j in range(6)] for i in range(6)]

        sigma = StressTensorList(
            [sum([strain_tensor[j] * H[i][j] for j in range(6)]) for i in range(6)]
        )

        if finite:
            return sigma.log_stress_u_to_cauchy(assembly.sv["F"])
        return sigma  # list of 6 objets

    def get_tangent_matrix(
        self, assembly, dimension=None
    ):  # Tangent Matrix in lobal coordinate system (no change of basis)
        if dimension is None:
            dimension = assembly.space.get_dimension()

        H = self.local2global_H(self._H, assembly)
        if dimension == "2Dstress":
            return self.get_H_plane_stress(H)
        else:
            return H

    def get_elastic_matrix(self, dimension="3D"):
        return self.get_tangent_matrix(None, dimension)

    # def ComputeStrain(self, assembly, pb, nlgeom, type_output='GaussPoint'):
    #     displacement = pb.get_dof_solution()
    #     if np.isscalar(displacement) and displacement == 0:
    #         return 0 #if displacement = 0, Strain = 0
    #     else:
    #         return assembly.get_strain(displacement, type_output)
