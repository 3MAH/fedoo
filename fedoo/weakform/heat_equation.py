from fedoo.core.base import ConstitutiveLaw
from fedoo.core.time_evolution import FIRST_ORDER
from fedoo.core.weakform import WeakFormBase
import numpy as np
from ._reference import (
    deformation_gradient,
    reference_derivatives,
    reference_volume_weight,
)
from fedoo.constitutivelaw._thermal import conductivity_tensor


def _finite_heat(assembly, pb, nlgeom):
    if nlgeom is not None:
        return bool(nlgeom)
    return bool(getattr(assembly, "_nlgeom", False) or pb.nlgeom or "F" in assembly.sv)


class HeatEquation(WeakFormBase):
    """
    Weak formulation of the heat equation.

    This weakform defines the spatial conduction operator and declares
    :class:`HeatCapacity` as its storage term. The problem-level time
    integrator determines whether the analysis is steady or transient.

    Parameters
    ----------
    thermal_constitutivelaw: ConstitutiveLaw name (str) or ConstitutiveLaw object
        Thermal Constitutive Law (:mod:`fedoo.constitutivelaw`)
    name: str, optional
        Name of the weak form. An empty name leaves it unregistered.
    nlgeom: None, bool or 'TL', optional
        None follows the mechanical geometry (or a supplied F field).
        True/'TL' uses reference integration with J F^-1 k F^-T; this also
        works with UL mechanics. False keeps the undeformed heat equation.
    """

    def __init__(self, thermal_constitutivelaw, name="", nlgeom=None, space=None):
        if isinstance(thermal_constitutivelaw, str):
            thermal_constitutivelaw = ConstitutiveLaw.get_all()[thermal_constitutivelaw]

        WeakFormBase.__init__(self, name, space)
        if self.space.is_axisymmetric:
            raise NotImplementedError

        self.space.new_variable("Temp")  # temperature

        if self.space.ndim == 3:
            self.__op_grad_temp = [
                self.space.derivative("Temp", "X"),
                self.space.derivative("Temp", "Y"),
                self.space.derivative("Temp", "Z"),
            ]
        else:  # 2D
            self.__op_grad_temp = [
                self.space.derivative("Temp", "X"),
                self.space.derivative("Temp", "Y"),
                0,
            ]

        self.__op_grad_temp_vir = [
            0 if op == 0 else op.virtual for op in self.__op_grad_temp
        ]

        self.constitutivelaw = thermal_constitutivelaw
        if nlgeom not in (False, None, True, "TL"):
            raise ValueError(
                "HeatEquation uses nlgeom='TL' (or True) for finite strain."
            )
        if getattr(thermal_constitutivelaw, "is_thermomechanical", False):
            self.assembly_options["assume_sym"] = False
        self.time_evolution = FIRST_ORDER
        self.__nlgeom = nlgeom
        self.storage = HeatCapacity(
            thermal_constitutivelaw,
            name="",
            nlgeom=nlgeom,
            space=self.space,
        )
        # Coupled Newton blocks use a consistent capacity: the GP residual
        # must have the same derivative as the assembled storage operator.
        self.storage.assembly_options["mat_lumping"] = not getattr(
            thermal_constitutivelaw, "is_thermomechanical", False
        )
        if getattr(thermal_constitutivelaw, "thermal_storage", None) == "included":
            self.storage = None

        # self.__nlgeom = nlgeom #geometric non linearities

    def initialize(self, assembly, pb):
        assembly._thermal_nlgeom = self.nlgeom
        if not hasattr(assembly, "_nlgeom"):
            assembly._nlgeom = False
        if self.nlgeom in (True, "TL") and "F" not in assembly.sv:
            assembly.sv["F"] = np.repeat(
                np.eye(3)[:, :, None], assembly.n_gauss_points, axis=2
            )
        if self.storage is not None and _finite_heat(assembly, pb, self.nlgeom):
            self.storage.assembly_options["mat_lumping"] = False
        if getattr(self.constitutivelaw, "thermal_storage", None) == "included":
            if FIRST_ORDER not in pb.time_integrators:
                raise ValueError(
                    "umat_T heat balance requires FIRST_ORDER/BackwardEuler."
                )
        if not (np.isscalar(pb.get_dof_solution())):
            assembly.sv["TempGradient"] = [
                0
                if operator == 0
                else assembly.get_gp_results(operator, pb.get_dof_solution())
                for operator in self.__op_grad_temp
            ]
        else:
            assembly.sv["TempGradient"] = [0 for operator in self.__op_grad_temp]

    def update(self, assembly, pb):
        assembly.sv["TempGradient"] = [
            0
            if operator == 0
            else assembly.get_gp_results(operator, pb.get_dof_solution())
            for operator in self.__op_grad_temp
        ]

    def get_weak_equation(self, assembly, pb):
        if _finite_heat(assembly, pb, self.nlgeom):
            return self._get_reference_equation(assembly, pb)
        K = self.constitutivelaw.thermal_conductivity

        diff_op = sum(
            [
                0
                if self.__op_grad_temp_vir[i] == 0
                else self.__op_grad_temp_vir[i]
                * sum(
                    [
                        0
                        if self.__op_grad_temp[j] == 0
                        else self.__op_grad_temp[j] * K[i][j]
                        for j in range(3)
                    ]
                )
                for i in range(3)
            ]
        )

        temp_grad = assembly.sv["TempGradient"]
        # add initial state for incremental resolution
        diff_op += sum(
            [
                0
                if self.__op_grad_temp_vir[i] == 0
                else self.__op_grad_temp_vir[i]
                * sum(
                    [
                        temp_grad[j] * K[i][j]
                        for j in range(3)
                        if (
                            not (np.array_equal(K[i][j], 0))
                            and not (np.array_equal(temp_grad[j], 0))
                        )
                    ]
                )
                for i in range(3)
            ]
        )

        if "HeatSource" in assembly.sv:
            temp = self.space.variable("Temp")
            diff_op -= temp.virtual * assembly.sv["HeatSource"]
            diff_op -= (
                temp.virtual * temp * assembly.sv.get("HeatSourceTemperatureTangent", 0)
            )
            if "DispX" in self.space.list_variables():
                strain = self.space.op_strain()
                derivative = assembly.sv.get("HeatSourceStrainTangent", np.zeros(6))
                diff_op -= sum(
                    temp.virtual * strain[i] * derivative[i]
                    for i in range(6)
                    if strain[i] != 0
                )
        return diff_op

    def _get_reference_equation(self, assembly, pb):
        F, inverse, J = deformation_gradient(assembly)
        k = conductivity_tensor(self.constitutivelaw.thermal_conductivity)
        K = np.einsum("Ain,ij,Bjn,n->ABn", inverse, k, inverse, J)
        gradient = np.array(
            [
                value + np.zeros(assembly.n_gauss_points)
                for value in assembly.sv["TempGradient"]
            ]
        )
        flux = np.einsum("ABn,Bn->An", K, gradient)
        op = reference_derivatives(assembly, "Temp", F)
        diff_op = sum(
            op[A].virtual
            * (sum(op[B] * K[A, B] for B in range(3) if op[B] != 0) + flux[A])
            for A in range(3)
            if op[A] != 0
        )
        if "DispX" in self.space.list_variables():
            displacement = [
                reference_derivatives(assembly, "Disp" + axis, F)
                for axis in "XYZ"[: self.space.ndim]
            ]
            spatial_gradient = np.einsum("Bin,Bn->in", inverse, gradient)
            # Exact variation of K0 = J F^-1 k F^-T with respect to F_iC.
            for A in range(3):
                if op[A] == 0:
                    continue
                for i in range(self.space.ndim):
                    for C in range(self.space.ndim):
                        coefficient = (
                            inverse[C, i] * flux[A]
                            - inverse[A, i] * flux[C]
                            - K[A, C] * spatial_gradient[i]
                        )
                        diff_op += op[A].virtual * displacement[i][C] * coefficient
        if "HeatSource" in assembly.sv:
            temp = self.space.variable("Temp")
            diff_op -= temp.virtual * assembly.sv["HeatSource"]
            diff_op -= (
                temp.virtual * temp * assembly.sv.get("HeatSourceTemperatureTangent", 0)
            )
            derivative = assembly.sv.get("HeatSourceDeformationTangent")
            if derivative is not None and "DispX" in self.space.list_variables():
                diff_op -= sum(
                    temp.virtual * displacement[i][A] * derivative[i, A]
                    for i in range(self.space.ndim)
                    for A in range(self.space.ndim)
                )
        assembly.sv["ReferenceHeatFlux"] = -flux
        assembly.sv["HeatFlux"] = -np.einsum("ij,Ajn,An->in", k, inverse, gradient)
        return diff_op * reference_volume_weight(assembly)

    @property
    def nlgeom(self):
        return self.__nlgeom


class HeatCapacity(WeakFormBase):
    """Heat capacity storage weakform.

    This weakform only defines the capacity operator ``rho*c``. Time
    discretization is handled by problem-level time integrators.
    """

    def __init__(self, thermal_constitutivelaw, name="", nlgeom=None, space=None):
        if isinstance(thermal_constitutivelaw, str):
            thermal_constitutivelaw = ConstitutiveLaw.get_all()[thermal_constitutivelaw]

        WeakFormBase.__init__(self, name, space)

        self.space.new_variable("Temp")  # temperature

        self.constitutivelaw = thermal_constitutivelaw
        self.time_evolution = FIRST_ORDER
        self.__nlgeom = nlgeom

    def initialize(self, assembly, pb):
        if not (np.isscalar(pb.get_dof_solution())):
            assembly.sv["Temp"] = assembly.convert_data(
                pb.get_temp(), convert_from="Node", convert_to="GaussPoint"
            )
        else:
            assembly.sv["Temp"] = 0

    def update(self, assembly, pb):
        assembly.sv["Temp"] = assembly.convert_data(
            pb.get_temp(), convert_from="Node", convert_to="GaussPoint"
        )

    def get_weak_equation(self, assembly, pb):
        rho_c = self.constitutivelaw.density * self.constitutivelaw.specific_heat
        if _finite_heat(assembly, pb, self.nlgeom):
            rho_c *= reference_volume_weight(assembly)

        op_temp = self.space.variable(
            "Temp"
        )  # temperature increment (incremental weakform)
        return rho_c * op_temp.virtual * op_temp

    def get_storage_value(self, assembly, pb):
        return assembly.sv["Temp"]

    def get_weak_equation_for_value(self, assembly, pb, value):
        rho_c = self.constitutivelaw.density * self.constitutivelaw.specific_heat
        if _finite_heat(assembly, pb, self.nlgeom):
            rho_c *= reference_volume_weight(assembly)
        return rho_c * self.space.variable("Temp").virtual * value

    @property
    def nlgeom(self):
        return self.__nlgeom
