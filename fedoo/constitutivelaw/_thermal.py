"""Shared field/history helpers for the standard constitutive-law lifecycle."""

import numpy as np

from fedoo.core.base import InvalidKinematicStateError
from fedoo.core.time_evolution import FIRST_ORDER, SECOND_ORDER
from fedoo.util.voigt_tensors import StrainTensorList


def conductivity_tensor(conductivity):
    tensor = np.asarray(conductivity, dtype=float)
    if tensor.ndim == 0:
        tensor = np.eye(3) * tensor
    if (
        tensor.shape != (3, 3)
        or not np.all(np.isfinite(tensor))
        or not np.allclose(tensor, tensor.T)
        or np.min(np.linalg.eigvalsh(tensor)) < 0
    ):
        raise ValueError(
            "Conductivity must be a nonnegative scalar or a symmetric positive-semidefinite 3x3 tensor."
        )
    return tensor


def check_absolute_temperature(temperature, initial=False):
    """Refuse a nonpositive or nonfinite absolute temperature (kelvin).

    At initialization this is a model error (the initial ``Temp`` field was
    not set) and raises ``ValueError``. During the Newton iterations it is an
    overshoot of the iterate: ``InvalidKinematicStateError`` lets the solver
    cut the time increment.
    """
    temperature = np.asarray(temperature)
    if np.all(np.isfinite(temperature)) and np.all(temperature > 0):
        return
    if initial:
        raise ValueError(
            "Thermo-mechanical coupling requires absolute temperatures in "
            "kelvin (T > 0); set the initial Temp field, e.g. "
            "pb.set_dof('Temp', 293.15)."
        )
    raise InvalidKinematicStateError(
        "Thermo-mechanical coupling requires absolute temperatures in kelvin "
        "(T > 0)."
    )


def thermal_fields(material, assembly, pb, initial=False):
    """Use weakform strains; only compute them when initializing or heat-only."""
    n = assembly.n_gauss_points
    solution = pb.get_dof_solution()
    if initial or "Strain" not in assembly.sv:
        strain = np.zeros((6, n), order="F")
        if "DispX" in assembly.space.list_variables() and not np.isscalar(solution):
            for i, op in enumerate(assembly.space.op_strain()):
                if op != 0:
                    strain[i] = assembly.get_gp_results(op, solution)
    else:
        value = assembly.sv["Strain"]
        value = value.asarray() if hasattr(value, "asarray") else value
        strain = np.broadcast_to(np.asarray(value), (6, n)).copy(order="F")
    if "Temp" in assembly.space.list_variables() and not np.isscalar(solution):
        temperature = assembly.convert_data(pb.get_temp(), "Node", "GaussPoint")
    else:
        temperature = assembly.sv.get("Temp", material.reference_temperature)
    temperature = np.broadcast_to(np.asarray(temperature), (n,)).copy()
    return strain, temperature


def initialize_thermal_state(material, assembly, pb):
    finite = bool(getattr(assembly, "_nlgeom", False) or pb.nlgeom)
    if (
        not finite
        and "DispX" in assembly.space.list_variables()
        and getattr(assembly, "_thermal_nlgeom", None) in (True, "TL")
    ):
        raise ValueError("A coupled TL heat equation requires finite-strain mechanics.")
    finite = finite or (
        "F" in assembly.sv
        and "DispX" not in assembly.space.list_variables()
        and getattr(assembly, "_thermal_nlgeom", None) is not False
    )
    if finite and getattr(assembly, "_corate", None) not in (None, "log", "log_r"):
        raise NotImplementedError(
            "Finite thermoelasticity uses the total logarithmic energy; use log_r."
        )
    if assembly.space.get_dimension() not in ("3D", "2Dplane"):
        raise NotImplementedError(
            "Coupled thermal materials support 3D and plane strain."
        )
    if material.local_frame is not None:
        raise NotImplementedError(
            "Coupled thermal materials currently use global axes."
        )
    integrator = pb.time_integrators.get(SECOND_ORDER)
    if getattr(integrator, "is_explicit", False):
        raise NotImplementedError(
            "Coupled dynamics currently requires implicit endpoint Newmark."
        )
    if integrator is not None and (
        getattr(integrator, "alpha_m", 0) != 0 or getattr(integrator, "alpha_f", 0) != 0
    ):
        raise NotImplementedError(
            "Coupled dynamics currently requires endpoint Newmark."
        )
    if not hasattr(assembly, "_nlgeom"):
        assembly._nlgeom = False
    if finite and getattr(assembly, "_thermal_nlgeom", None) is False:
        raise ValueError("Finite-strain coupling requires HeatEquation(nlgeom='TL').")
    if assembly.assume_sym is True:
        raise ValueError("Thermo-mechanical coupling requires nonsymmetric assembly.")
    assembly.assume_sym = False
    strain, temperature = thermal_fields(material, assembly, pb, initial=not finite)
    coupling = getattr(getattr(material, "thermal", None), "coupling", False)
    if not finite and coupling and FIRST_ORDER in pb.time_integrators:
        # the thermoelastic source is proportional to the absolute temperature
        check_absolute_temperature(temperature, initial=True)
    if finite:
        import simcoon as sim

        check_absolute_temperature(temperature, initial=True)

        if "DispX" in assembly.space.list_variables():
            solution = pb.get_dof_solution()
            if not np.isscalar(solution):
                gradient = np.asarray(assembly.get_grad_disp(solution, "GaussPoint"))
                assembly.sv["F"] = gradient + np.eye(3)[:, :, None]
                assembly.sv["DispGradient"] = gradient
        from fedoo.weakform._reference import deformation_gradient

        F = deformation_gradient(assembly)[0]
        strain = sim.Log_strain(np.asfortranarray(F.transpose(1, 0, 2)), True, False)
        assembly.sv["_TM_LogStrain"] = strain
        assembly.sv["_TM_FStart"] = np.array(F, order="F")
    else:
        assembly.sv["Strain"] = StrainTensorList(strain)
    assembly.sv["Temp"] = temperature
    assembly.sv["_TM_StrainStart"] = strain.copy()
    assembly.sv["_TM_TempStart"] = temperature.copy()


def commit_thermal_fields(assembly):
    assembly.sv["_TM_StrainStart"] = np.asarray(
        assembly.sv.get("_TM_LogStrain", assembly.sv["Strain"])
    ).copy()
    assembly.sv["_TM_TempStart"] = np.asarray(assembly.sv["Temp"]).copy()
    if "_TM_FStart" in assembly.sv:
        assembly.sv["_TM_FStart"] = np.array(assembly.sv["F"], order="F")
