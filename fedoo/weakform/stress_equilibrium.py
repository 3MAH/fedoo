"""The Strain equilibrium weak form from the fedoo finite element code."""

import warnings

from fedoo.core.weakform import WeakFormBase
from fedoo.core.base import ConstitutiveLaw, InvalidKinematicStateError
from fedoo.core.time_evolution import SECOND_ORDER
from fedoo.lib_elements.element_base import _evaluate_monomials, _monomial_exponents
from fedoo.lib_elements.element_list import get_element
from fedoo.weakform.inertia import Inertia
from fedoo.util.voigt_tensors import StressTensorList, StrainTensorList
import numpy as np
import simcoon as sim
from simcoon import Rotation as SimRotation


class StressEquilibrium(WeakFormBase):
    """Mechanical equilibrium equation for solids.

    The main point to consider are:
      * This weak form can be used for solid in 3D or using a 2D plane
        assumption (plane strain or plane stress).
      * Include initial stress for non linear problems or if defined in
        the associated assembly.
      * This weak form accepts geometrical non linearities if simcoon is
        installed. (nlgeom should be in {True, 'UL', 'TL'}. In this case
        the initial displacement is also considered and several different
        corotational formulation may be used by setting the corate attribute.
      * For nearly incompressible material, a method to avoid the volumetric
        locking should be selected with the incompressibility attribute.
      * The consistent geometric stiffness is enabled automatically for
        finite-strain tangent conversion. Set ``geometric_stiffness``
        explicitly to override that default.

    Parameters
    ----------
    constitutivelaw: ConstitutiveLaw name (str) or ConstitutiveLaw object
        Material Constitutive Law (:mod:`fedoo.constitutivelaw`)
    incompressibility: str, optional
        Method used to avoid the volumetric locking of nearly incompressible
        materials. See the :py:attr:`incompressibility` attribute.
        By default (None), the standard displacement formulation is used.
    convert_tangent: bool, default=True
        Convert the constitutive corotational Kirchhoff tangent to the
        formulation tangent (``dS/dE`` for TL, spatial Lie for UL). Set to
        False only when the constitutive law already returns that final
        formulation-dependent tangent.
    name: str
        name of the WeakForm
    nlgeom: bool, 'UL' or 'TL', optional
        If True, the geometrical non linearities are activate based on the
        updated lagrangian method. This parameters is used only in the
        context of NonLinearProblems such as
        :mod:`fedoo.problem.NonLinearStatic` or
        :mod:`fedoo.problem.NonLinearNewmark`.
        If nlgeom == 'UL' the updated lagrangian method is used (same as True).
        If nlgeom == 'TL' the total lagrangian method is used.
        If not defined, the problem.nlgeom parameter is used instead.
    space: ModelingSpace
        Modeling space associated to the weakform. If None is specified,
        the active ModelingSpace is considered.
    """

    def __init__(
        self,
        constitutivelaw,
        incompressibility=None,
        convert_tangent=True,
        name="",
        nlgeom=None,
        space=None,
    ):
        if isinstance(constitutivelaw, str):
            constitutivelaw = ConstitutiveLaw[constitutivelaw]

        WeakFormBase.__init__(self, name, space)

        self.space.new_variable("DispX")
        self.space.new_variable("DispY")
        if self.space.ndim == 3:
            self.space.new_variable("DispZ")
            self.space.new_vector("Disp", ("DispX", "DispY", "DispZ"))
        else:  # 2D assumed
            self.space.new_vector("Disp", ("DispX", "DispY"))

        self.constitutivelaw = constitutivelaw
        # Tag a second-order (dynamic) evolution. The mass term is resolved from
        # the material density (or an explicit set_inertia) at integrator
        # compile time, so a set_density() call made after the weakform is built
        # is still honored, and the inertia stays absent for static analyses.
        self.time_evolution = SECOND_ORDER

        self.nlgeom = nlgeom
        """Method used to treat the geometric non linearities.
            * Set to False if geometric non linarities are ignored.
            * Set to True or 'UL' to use the updated lagrangian method
              (update the mesh)
            * Set to 'TL' to use the total lagrangian method (base on the
              initial mesh with initial displacement effet)
        """

        self.corate = "log_r"
        # 'log_r' (default): logarithmic strain with the exact polar frame
        # increment DR = R1 R0^T — the corate whose tangent transport is
        # exact in simcoon, including with rotated internal-variable history.
        # Other options: 'log' (XBM), 'jaumann', 'green_naghdi'/'gn',
        # 'log_inc', 'log_r_inc'...

        # by default, no treatment of the volumetric locking
        self._incompressibility_elm_types = []
        self.incompressibility = incompressibility
        self.convert_tangent = convert_tangent
        self.geometric_stiffness = None

        # assume_sym is decided at initialize time: True for linear / TL
        # (symmetric tangent), False for UL where the Lie spatial tangent is
        # not major-symmetric. A user-set value takes precedence.

    def get_storage(self):
        if self.storage is not None:
            return self.storage
        density = getattr(self.constitutivelaw, "density", None)
        if density is None:
            material_name = getattr(
                self.constitutivelaw, "name", type(self.constitutivelaw).__name__
            )
            raise ValueError(
                "StressEquilibrium requires a material density for dynamic "
                f"analysis, but material {material_name!r} has no density. "
                "Set it with material.set_density(rho), or attach inertia "
                "explicitly with weakform.set_inertia(density_or_weakform)."
            )
        return Inertia(density, space=self.space)

    def get_weak_equation(self, assembly, pb):
        """Get the weak equation related to the current problem state."""
        if assembly._nlgeom == "TL":  # add initial displacement effect
            eps = self.space.op_strain(assembly.sv["DispGradient"])
            initial_stress = assembly.sv["PK2"]
        else:
            eps = self.space.op_strain()
            initial_stress = assembly.sv[
                "Stress"
            ]  # Stress = Cauchy for updated lagrangian method

            if self.space.is_axisymmetric:
                rr = assembly.sv["_R_gausspoints"]

                # nlgeom = False
                eps[2] = self.space.variable("DispX") * np.divide(
                    1, rr, out=np.zeros_like(rr), where=rr != 0
                )  # put zero if X==0 (division by 0)
                # eps[2] = self.space.variable('DispX') * (1/rr)

        H = assembly.sv["TangentMatrix"]

        dilatation_correction = self._get_dilatation_correction_op(assembly)
        if dilatation_correction is not None:
            return self._get_weak_equation_bbar(
                assembly, eps, H, initial_stress, dilatation_correction
            )

        sigma = [
            sum([0 if eps[j] == 0 else eps[j] * H[i][j] for j in range(6)])
            for i in range(6)
        ]

        DiffOp = sum(
            [0 if eps[i] == 0 else eps[i].virtual * sigma[i] for i in range(6)]
        )

        if not (np.isscalar(initial_stress) and initial_stress == 0):
            # this term doesnt seem to improve convergence in general !
            if self.geometric_stiffness:
                DiffOp = DiffOp + sum(
                    [
                        0
                        if self._nl_strain_op_vir[i] == 0
                        else self._nl_strain_op_vir[i] * initial_stress[i]
                        for i in range(6)
                    ]
                )

            DiffOp = DiffOp + sum(
                [
                    0 if eps[i] == 0 else eps[i].virtual * initial_stress[i]
                    for i in range(6)
                ]
            )

        if self.incompressibility == "fbar":
            DiffOp = DiffOp + self._get_fbar_tangent_op(
                assembly, eps, H, initial_stress
            )

        if self.space.is_axisymmetric:
            DiffOp = DiffOp * ((2 * np.pi) * rr)

        return DiffOp

    def _get_fbar_tangent_op(self, assembly, eps, H, stress):
        # Non symmetric term of the F-bar tangent matrix, related to the
        # variation of the volume change evaluated at the element center.
        # ref: Design of simple low order finite elements for large strain
        # analysis of nearly incompressible solids, de Souza Neto et al, IJSS.
        # Return 0 if the operators at the element center are not available.
        if (
            assembly._nlgeom == "TL"
            or self.space.is_axisymmetric
            or "_DispX"
            not in getattr(get_element(assembly.elm_type), "dict_elm_type", {})
            or _get_dilatation_degree(assembly) is not None
        ):
            return 0

        q = [(1 / 3.0) * (H[i][0] + H[i][1] + H[i][2]) for i in range(6)]
        if assembly._nlgeom and not (np.isscalar(stress) and stress == 0):
            # H is the Lie (Truesdell) tangent: the initial stress stiffness is
            # treated separately (geometric_stiffness), hence the 1/3 factor
            # instead of the 2/3 associated to the spatial modulus of the ref.
            q = [q[i] - (1 / 3.0) * stress[i] for i in range(6)]

        crd = ["X", "Y", "Z"][: self.space.ndim]
        # dilatation at the element center - dilatation
        correction = sum(self.space.derivative("_Disp" + x, x) for x in crd) - sum(
            self.space.derivative("Disp" + x, x) for x in crd
        )
        return sum(
            0 if eps[i] == 0 else eps[i].virtual * q[i] * correction for i in range(6)
        )

    def _get_dilatation_correction_op(self, assembly):
        # Return the operator theta_bar - div(u), where theta_bar is the
        # dilatation modified to avoid the volumetric locking.
        # Return None if the assembly uses a standard element.
        if self.incompressibility not in ["auto", "mean_dilatation", "sri"]:
            return None
        elm = get_element(assembly.elm_type)
        if "_DispX" not in getattr(elm, "dict_elm_type", {}):
            return None
        crd = ["X", "Y", "Z"][: self.space.ndim]
        correction = sum(self.space.derivative("_Disp" + x, x) for x in crd)
        if _get_dilatation_degree(assembly) is None:
            # reduced interpolation (sri): the operator is theta_bar
            correction = correction - sum(
                self.space.derivative("Disp" + x, x) for x in crd
            )
        return correction

    def _get_weak_equation_bbar(self, assembly, eps, H, stress, correction):
        # B-bar form: the normal strains are eps_bar[i] = eps[i] + correction/n
        # for the real and virtual fields. Instead of expanding the products of
        # the modified strains, the tangent is pre-contracted so that every
        # couple of operators appears only once:
        # eps_bar^T H eps_bar = eps^T H eps + c (a.eps) + (b.eps) c + k c c
        # Solid constitutive laws use a three-dimensional volumetric split,
        # including with the plane-strain assumption. In 2D plane strain the
        # physical eps_zz is zero, but the assumed B-bar strain receives one
        # third of the volumetric correction in all three normal components.
        n = 3
        # scale = theta/J: the weak form is integrated over the current volume
        # (dv = J dV0) whereas the mean dilatation form is theta dV0
        scale = assembly.sv.get("_MeanDilatation_scale", 1)

        H = [[H[i][j] * scale for j in range(6)] for i in range(6)]
        h_col = [sum(H[i][j] for i in range(n)) * (1 / n) for j in range(6)]
        h_row = [sum(H[i][j] for j in range(n)) * (1 / n) for i in range(6)]
        h_vol = sum(h_col[j] for j in range(n)) * (1 / n)

        list_eps = [i for i in range(6) if not (np.isscalar(eps[i]) and eps[i] == 0)]
        diff_op = sum(
            eps[i].virtual * sum(eps[j] * H[i][j] for j in list_eps) for i in list_eps
        )
        diff_op = diff_op + correction.virtual * sum(
            eps[j] * h_col[j] for j in list_eps
        )
        diff_op = (
            diff_op + sum(eps[i] * h_row[i] for i in list_eps).virtual * correction
        )
        diff_op = diff_op + correction.virtual * (correction * h_vol)

        if np.isscalar(stress) and stress == 0:
            return diff_op

        stress = [stress[i] * scale for i in range(6)]
        pressure = sum(stress[i] for i in range(n)) * (1 / n)

        if self.geometric_stiffness:
            diff_op = diff_op + sum(
                0
                if self._nl_strain_op_vir[i] == 0
                else self._nl_strain_op_vir[i] * stress[i]
                for i in range(6)
            )
            # contribution of the modified dilatation to the initial stress
            # stiffness
            stress_work = sum(eps[i] * stress[i] for i in list_eps)
            diff_op = diff_op + (
                correction.virtual * stress_work + stress_work.virtual * correction
            ) * (2 / n)
            if "_MeanDilatation_scale" in assembly.sv:
                grad = self.space.op_grad_u()
                div = sum(grad[i][i] for i in range(n))
                grad_grad = sum(
                    0
                    if grad[i][j] == 0 or grad[j][i] == 0
                    else grad[i][j].virtual * grad[j][i]
                    for i in range(n)
                    for j in range(n)
                )
                pressure_proj = (
                    _element_projection(
                        (pressure / scale).reshape(assembly.n_elm_gp, -1),
                        assembly.sv["_MeanDilatation_dV0"],
                        _get_dilatation_basis(assembly),
                    ).ravel()
                    - pressure
                )
                diff_op = diff_op + (div.virtual * div - grad_grad) * pressure_proj
                diff_op = (
                    diff_op
                    + (correction.virtual * correction) * ((2 / n - 1) * pressure)
                    - (correction.virtual * div + div.virtual * correction) * pressure
                )
            else:
                diff_op = diff_op + (correction.virtual * correction) * (
                    (2 / n) * pressure
                )

        # internal force
        diff_op = diff_op + sum(eps[i].virtual * stress[i] for i in list_eps)
        return diff_op + correction.virtual * pressure

    def initialize(self, assembly, pb):
        """Initialize the weakform at the begining of a problem."""
        # TO DO: change stress initialization to remove initial stress
        # term in the global vector assembly

        # initialize nlgeom value in assembly._nlgeom
        self._initialize_nlgeom(assembly, pb)
        self.nlgeom = assembly._nlgeom
        self.corate = self._corate  # to force the setter function
        self._initialize_incompressibility(assembly, pb)

        # ``_initialize_incompressibility`` may already have selected a value
        # (the consistent F-bar tangent is non-symmetric). Otherwise resolve
        # the deferred assembly value here while preserving an explicit user
        # choice made on either the weak form or the assembly.
        if assembly.assume_sym is None:
            assume_sym = self.assembly_options.get("assume_sym", assembly.elm_type)
            if assume_sym is None:
                assume_sym = not (assembly._nlgeom == "UL" and self.convert_tangent)
            assembly.assume_sym = assume_sym

        # The initial-stress term completes the consistent TL/UL tangent.
        if self.geometric_stiffness is None:
            self.geometric_stiffness = bool(assembly._nlgeom and self.convert_tangent)

        # Put the require field to zeros if they don't exist in the assembly
        if "Stress" not in assembly.sv:
            assembly.sv["Stress"] = StressTensorList(
                np.zeros((6, assembly.n_gauss_points), order="F")
            )
        if "Strain" not in assembly.sv:
            assembly.sv["Strain"] = StrainTensorList(
                np.zeros((6, assembly.n_gauss_points), order="F")
            )
        assembly.sv["DispGradient"] = 0

        if self.space.is_axisymmetric:
            # ``assembly.mesh`` is treated as the *reference* configuration:
            # any subsequent ``set_disp`` only mutates ``assembly.current.mesh``
            # (see fedoo/core/assembly.py: set_disp). Capture R0 from the
            # reference mesh here, once. If the mesh has already been deformed
            # before initialize is reached (unusual, e.g. chained problems
            # sharing an assembly), the captured R0 will be the deformed
            # radius, breaking F_theta-theta = r/R at finite strain. Callers
            # should rebuild the assembly before initializing a 2Daxi problem
            # in that case.
            r_nodes = assembly.mesh.nodes[:, 0]
            if r_nodes.min() < 0:
                raise ValueError(
                    "2Daxi requires non-negative radial coordinates "
                    "(mesh.nodes[:, 0] >= 0). Found "
                    f"min(r) = {r_nodes.min():.6g}. "
                    "The convention is r = X (column 0), z = Y (column 1)."
                )
            # Reference radial coordinate at gauss points (initial mesh).
            # Captured once and never overwritten: used to form the canonical
            # hoop deformation gradient F_theta-theta = r_current / R_reference
            # in finite-strain UL+axi (Bonet & Wood, Box 8.3).
            assembly.sv["_R0_gausspoints"] = assembly.mesh.convert_data(
                r_nodes,
                "Node",
                "GaussPoint",
                n_elm_gp=assembly.n_elm_gp,
            )
            # Current radial coordinate at gauss points. Equal to the reference
            # at initialize; refreshed each iteration in _comp_grad_disp to the
            # deformed mesh (used for the 2*pi*r weak-form weight and the
            # symbolic operator eps[2] = DispX / r at the current config).
            assembly.sv["_R_gausspoints"] = assembly.sv["_R0_gausspoints"].copy()

        if assembly._nlgeom:
            if assembly._nlgeom == "TL":
                assembly.sv["PK2"] = 0
                if self.space.is_axisymmetric:
                    raise NotImplementedError(
                        "'2Daxi' ModelingSpace is not implemented with \
                         total lagrangian formulation. Use update \
                         lagrangian instead."
                    )

    def _initialize_incompressibility(self, assembly, pb):
        method = self.incompressibility
        if method is None:
            return
        if self.space.get_dimension() == "2Dstress":
            # Plane stress eliminates the out-of-plane stress through the
            # constitutive reduction and does not use the 3D incompressibility
            # treatments implemented here.
            assembly.elm_type = assembly.mesh.elm_type
            self._comp_F = _comp_F
            if method != "auto":
                warnings.warn(
                    f"incompressibility={method!r} is ignored with the "
                    "'2Dstress' modeling space; the standard plane-stress "
                    "formulation is used.",
                    stacklevel=2,
                )
            return
        if method == "fbar":
            has_center_interpolation = "_DispX" in getattr(
                get_element(assembly.elm_type), "dict_elm_type", {}
            )
            consistent_tangent = (
                has_center_interpolation
                and assembly._nlgeom != "TL"
                and not self.space.is_axisymmetric
            )
            if consistent_tangent:
                # consistent (non symmetric) tangent matrix
                for x in ["X", "Y", "Z"][: self.space.ndim]:
                    self.space.variable_alias("_Disp" + x, "Disp" + x)
            else:
                reasons = []
                if not has_center_interpolation:
                    reasons.append(
                        f"'{assembly.mesh.elm_type}' has no element-center "
                        "displacement interpolation"
                    )
                if assembly._nlgeom == "TL":
                    reasons.append("the total-Lagrangian formulation is used")
                if self.space.is_axisymmetric:
                    reasons.append("the modeling space is axisymmetric")
                warnings.warn(
                    "incompressibility='fbar': the consistent F-bar tangent "
                    "is unavailable because "
                    + ", and ".join(reasons)
                    + ". The F-bar kinematic correction is kept, but the "
                    "standard tangent is used; convergence may be degraded.",
                    stacklevel=2,
                )
            return

        has_alias = "_DispX" in getattr(
            get_element(assembly.elm_type), "dict_elm_type", {}
        )
        degree = _get_dilatation_degree(assembly)
        if method == "sri":
            available = has_alias and degree is None
        else:
            available = degree is not None

        if not available:
            if method == "auto":
                warnings.warn(
                    "incompressibility='auto': no method available for "
                    f"'{assembly.mesh.elm_type}' elements that are prone to "
                    "volumetric locking. The standard displacement "
                    "formulation is used. Consider quadratic elements or "
                    "StressEquilibriumMixed.",
                    stacklevel=2,
                )
                return
            raise ValueError(
                f"incompressibility='{method}' is not available for "
                f"'{assembly.mesh.elm_type}' elements."
            )

        if method == "sri" and assembly._nlgeom == "UL":
            warnings.warn(
                "incompressibility='sri' is a legacy small-strain method. "
                "With updated-Lagrangian finite strains, it corrects the "
                "weak-form strain operator but not the deformation gradient "
                "passed to the constitutive law; convergence and results may "
                "therefore be unreliable. Prefer 'mean_dilatation' or 'fbar'.",
                stacklevel=2,
            )

        if degree is not None:
            if self.space.is_axisymmetric or assembly._nlgeom == "TL":
                raise NotImplementedError(
                    "The mean dilatation method is not implemented for '2Daxi' "
                    "ModelingSpace or with the total lagrangian formulation. Use "
                    "updated lagrangian instead."
                )
            basis = _get_dilatation_basis(assembly)
            if assembly.n_elm_gp <= basis.shape[1]:
                raise ValueError(
                    f"The mean dilatation method with {basis.shape[1]} pressure "
                    f"value(s) per element requires more than {basis.shape[1]} "
                    f"gauss points ({assembly.n_elm_gp} used)."
                )
            if assembly._nlgeom == "UL":
                # reference gauss point volumes (n_elm_gp, n_elements)
                assembly.sv["_MeanDilatation_dV0"] = (
                    assembly._get_gaussian_quadrature_mat()
                    .data.reshape(assembly.n_elm_gp, -1)
                    .copy()
                )
        elif assembly._nlgeom == "TL":
            raise NotImplementedError(
                "incompressibility='sri' is not implemented with the total "
                "lagrangian formulation. Use updated lagrangian instead."
            )

        for x in ["X", "Y", "Z"][: self.space.ndim]:
            self.space.variable_alias("_Disp" + x, "Disp" + x)

    def update(self, assembly, pb):
        """Update the weakform to the current state.

        This method is applyed before the update of constutive law (stress and
        stiffness matrix).
        """
        if assembly._nlgeom == "UL":
            # if updated lagragian method
            # -> update the mesh and recompute elementary op
            assembly.set_disp(pb.get_disp())

        displacement = pb.get_dof_solution()
        if np.isscalar(displacement) and displacement == 0:
            assembly.sv["DispGradient"] = 0
            if "Stress" not in assembly.sv:
                assembly.sv["Stress"] = 0
                assembly.sv["Strain"] = 0
        else:
            # Compute the strain required for the constitutive law.
            if assembly._nlgeom:
                self._comp_F(assembly, displacement)
                self._corate_func(self, assembly, pb)
            else:
                if self.incompressibility == "fbar":
                    _comp_grad_disp_fbar(assembly, displacement)
                elif (
                    self.incompressibility in ["auto", "mean_dilatation"]
                    and _get_dilatation_degree(assembly) is not None
                ):
                    _comp_grad_disp_mean_dilatation(assembly, displacement)
                else:
                    _comp_grad_disp(assembly, displacement)
                _comp_linear_strain(self, assembly, pb)

    def update_2(self, assembly, pb):
        """Update the weakform to the current state.

        This method is applyed after the constutive law update (stress and
        stiffness matrix).
        """
        if not assembly._nlgeom:
            return

        if assembly._nlgeom == "TL":
            # PK2 feeds the TL residual and must be refreshed even when the
            # tangent conversion is disabled or during a line-search trial.
            assembly.sv["PK2"] = assembly.sv["Stress"].cauchy_to_pk2(assembly.sv["F"])

        if not self.convert_tangent or getattr(pb, "_line_search_update", False):
            return

        if assembly._nlgeom in ("TL", "UL"):
            # check if TangentMatrix has the consistent array shape
            if not isinstance(assembly.sv["TangentMatrix"], np.ndarray):
                Lt = np.empty(
                    (6, 6, assembly.n_gauss_points),
                    order="F",
                )
                for i in range(6):
                    for j in range(6):
                        Lt[i, j, :] = assembly.sv["TangentMatrix"][i][j]
                assembly.sv["TangentMatrix"] = Lt

            elif (
                len(assembly.sv["TangentMatrix"].shape) == 2
                and len(assembly.sv["F"].shape) == 3
            ):
                # assembly.sv["TangentMatrix"] = assembly.sv["TangentMatrix"].reshape(6, 6, -1
                #     ) * np.ones((1, 1, assembly.sv["F"].shape[2]))

                assembly.sv["TangentMatrix"] = np.multiply(
                    assembly.sv["TangentMatrix"].reshape(6, 6, -1),
                    np.ones((1, 1, assembly.sv["F"].shape[2])),
                    order="F",
                )

            # Solid constitutive laws return the "box" tangent
            # d(tau_hat)/dD (Kirchhoff, corotational rate). Convert it to what
            # this configuration integrates.
            if assembly._nlgeom == "TL":
                # material tangent dS/dE (DsigmaDe_2_DSDE)
                assembly.sv["TangentMatrix"] = sim.Lt_convert(
                    assembly.sv["TangentMatrix"],
                    assembly.sv["F"],
                    assembly.sv["Stress"].asarray(),
                    self._convert_Lt_tag,
                )

            else:
                # current config (UL), all corates: the box
                # tangent d(tau_hat)/dD must be converted to the Lie
                # (Truesdell) spatial tangent, which is the one consistent
                # with the Cauchy-stress residual plus the standard
                # initial-stress geometric term assembled here. Using the box
                # (or box/J) directly leaves an O(sigma) inconsistency that
                # destroys Newton convergence at large rotation (soft bending
                # modes). _convert_Lt_tag holds the corate-specific
                # box -> dS/dE first stage (see the corate setter).
                stress = assembly.sv["Stress"].asarray()
                dsde = sim.Lt_convert(
                    assembly.sv["TangentMatrix"],
                    assembly.sv["F"],
                    stress,
                    self._convert_Lt_tag,
                )
                assembly.sv["TangentMatrix"] = sim.Lt_convert(
                    dsde,
                    assembly.sv["F"],
                    stress,
                    "DSDE_2_Dsigma_LieDD",
                )

    def to_start(self, assembly, pb):
        """Reset the current time increment."""
        if assembly._nlgeom == "UL":
            # if updated lagragian method -> reset the mesh to the begining
            # of the increment
            assembly.set_disp(pb.get_disp())

    def set_start(self, assembly, pb):
        """Start a new time increment."""
        if assembly._nlgeom:
            if not (np.array_equal(assembly.sv["DispGradient"], 0)):
                # True when the problem have been updated once
                rot = SimRotation.from_matrix(assembly.sv["DR"].transpose(2, 0, 1))
                if "DStrain" in assembly.sv:
                    # rotate strain
                    assembly.sv["Strain"] = StrainTensorList(
                        rot.apply_strain(
                            assembly.sv_start["Strain"].asarray(),
                        )
                        + assembly.sv["DStrain"]
                    )
                    assembly.sv["DStrain"] = StrainTensorList(
                        np.zeros((6, assembly.n_gauss_points), order="F")
                    )

                # update cauchy stress
                stress = assembly.sv["Stress"].asarray()
                assembly.sv["Stress"] = StressTensorList(rot.apply_stress(stress))
                if assembly._nlgeom == "TL":
                    assembly.sv["PK2"] = assembly.sv["Stress"].cauchy_to_pk2(
                        assembly.sv["F"]
                    )

    def _init_nl_strain_op_vir(self):
        # initialize non linear operator for strain
        # don't improve the convergence, but kept in case it may be usefull
        # later.

        op_grad_du = self.space.op_grad_u()
        # grad of displacement increment in incremental problems

        if self.space.ndim == 3:
            # using voigt notation and with a 2 factor on non diagonal terms:
            # nl_strain_op_vir =
            #      0.5*(vir(duk/dxi) * duk/dxj + duk/dxi * vir(duk/dxj))
            nl_strain_op_vir = [
                sum([op_grad_du[k][i].virtual * op_grad_du[k][i] for k in range(3)])
                for i in range(3)
            ]
            nl_strain_op_vir += [
                sum(
                    [
                        op_grad_du[k][0].virtual * op_grad_du[k][1]
                        + op_grad_du[k][1].virtual * op_grad_du[k][0]
                        for k in range(3)
                    ]
                )
            ]
            nl_strain_op_vir += [
                sum(
                    [
                        op_grad_du[k][0].virtual * op_grad_du[k][2]
                        + op_grad_du[k][2].virtual * op_grad_du[k][0]
                        for k in range(3)
                    ]
                )
            ]
            nl_strain_op_vir += [
                sum(
                    [
                        op_grad_du[k][1].virtual * op_grad_du[k][2]
                        + op_grad_du[k][2].virtual * op_grad_du[k][1]
                        for k in range(3)
                    ]
                )
            ]
        else:
            nl_strain_op_vir = [
                sum([op_grad_du[k][i].virtual * op_grad_du[k][i] for k in range(2)])
                for i in range(2)
            ] + [0]
            nl_strain_op_vir += [
                sum(
                    [
                        op_grad_du[k][0].virtual * op_grad_du[k][1]
                        + op_grad_du[k][1].virtual * op_grad_du[k][0]
                        for k in range(2)
                    ]
                )
            ] + [0, 0]

        self._nl_strain_op_vir = nl_strain_op_vir

    @property
    def incompressibility(self):
        """Method used to avoid volumetric locking.

        Nearly incompressible materials lead to volumetric locking with the
        standard displacement formulation. The available methods are:
            * None (default): standard displacement formulation.
            * "auto": use the most suitable method for each type of element
              (currently "mean_dilatation" where available). The standard
              formulation is kept with a warning for elements without any
              available method ("tri3", "tet4", ...).
            * "mean_dilatation": mean dilatation method (B-bar method based on
              a volume weighted projection of the dilatation). Equivalent to
              a mixed formulation with element-wise discontinuous pressure and
              dilatation that are eliminated at the element level: the problem
              only contains the displacement and the tangent matrix remains
              symmetric. One pressure value per element is used for "quad4",
              "hex8", "wed6", "tri6" and "tet10" elements, and a linear pressure
              for "quad8", "quad9", "hex20", "wed15" and "wed18" elements.
              The number of gauss points should be higher than the number of
              pressure values (no effect with reduced integration).
              Available in small strain and with the updated lagrangian
              method. The geometric_stiffness attribute should be set to True
              to get the consistent tangent matrix in finite strain.
            * "sri": legacy small-strain B-bar method where the dilatation is
              evaluated at the element center, for "quad4" and "hex8"
              elements only. Its use with updated-Lagrangian finite strains
              emits a warning because the formulation is not consistent.
            * "fbar": F-bar method where the volume change is evaluated at the
              element center. The consistent non symmetric tangent matrix is
              used for "quad4" and "hex8" elements (not available for '2Daxi'
              and with the total lagrangian method). In small strain, it
              reduces to a center-based B-bar volumetric correction.

        This attribute should be defined before the creation of the associated
        assembly. Elements with constant strain ("tri3", "tet4") or
        reduced integration elements are not improved by these methods: see
        :py:class:`fedoo.weakform.StressEquilibriumMixed` or
        :py:class:`fedoo.weakform.StressEquilibriumRI`. Incompressibility
        treatments are ignored with the plane-stress assumption.
        """
        return self._incompressibility

    @incompressibility.setter
    def incompressibility(self, value):
        if isinstance(value, str):
            value = value.lower()
        if value not in _INCOMPRESSIBILITY_ELEMENTS:
            raise ValueError(
                "incompressibility should be None, 'auto', 'mean_dilatation', "
                f"'sri' or 'fbar'. Got {value!r}."
            )
        previous = getattr(self, "_incompressibility", None)
        # remove options installed by a previous incompressibility value
        for elm_type in self._incompressibility_elm_types:
            self.assembly_options.elm_options[elm_type].pop("elm_type", None)
        if previous == "fbar" and self.assembly_options.get("assume_sym") is False:
            self.assembly_options.elm_options[None].pop("assume_sym", None)
        self._incompressibility_elm_types = list(_INCOMPRESSIBILITY_ELEMENTS[value])
        for elm_type, new_elm_type in _INCOMPRESSIBILITY_ELEMENTS[value].items():
            self.assembly_options["elm_type", elm_type] = new_elm_type

        self._incompressibility = value
        if value == "fbar":
            # Set the default before assembly creation. A user can still
            # override it on the weak form before creation, or directly on the
            # assembly before initialize; initialize preserves either choice.
            self.assembly_options["assume_sym"] = False
            self._comp_F = _comp_Fbar
        elif value in ["auto", "mean_dilatation"]:
            self._comp_F = _comp_F_mean_dilatation
        else:
            self._comp_F = _comp_F

    @property
    def convert_tangent(self):
        """Whether Fedoo converts the finite-strain constitutive tangent."""
        return self._convert_tangent

    @convert_tangent.setter
    def convert_tangent(self, value):
        if not isinstance(value, bool):
            raise TypeError("bool expected for convert_tangent")
        self._convert_tangent = value

    @property
    def geometric_stiffness(self):
        """Whether to add geometric effects to the stiffness matrix.

        ``None`` selects the consistent default: enabled for finite-strain
        tangent conversion and disabled otherwise. Set a boolean explicitly
        to override this behavior.
        """
        return self._geometric_stiffness

    @geometric_stiffness.setter
    def geometric_stiffness(self, value):
        if value is not None and not isinstance(value, bool):
            raise TypeError("bool or None expected for geometric_stiffness")
        self._geometric_stiffness = value
        if value is True:
            self._init_nl_strain_op_vir()

    @property
    def corate(self):
        """Corotational strain mesure for strain.

        Properties defining the way strain is treated in finite strain problem
        (using a weakform with nlgeom = True)
        corate can take the following str values:
            * "log_r" (default): exact logarithmic strain transported by the
              exact polar rotation increment (strain is recomputed at each
              iteration). This is the corate with an EXACT simcoon tangent
              transport, including across committed increments with plastic
              (or other rotated) history.
            * "log": exact logarithmic strain, XBM logarithmic spin
              transport (small tangent residual with rotated history)
            * "jaumann": Strain using the Jaumann derivative (strain is
              incremented)
            * "green_nagdhi" or "gn": Strain using the Green_Nagdhi derivative
              (strain is incremented)
        if nlgeom is False, this property has no effect.
        """
        return self._corate

    @corate.setter
    def corate(self, value):
        self._corate = value
        if self.nlgeom == "UL":
            # In UL, the assembled tangent is always the Lie (Truesdell)
            # spatial tangent (see update_2); _convert_Lt_tag holds the
            # corate-specific first-stage conversion of the umat box tangent
            # d(tau_hat)/dD to the material tangent dS/dE.
            value = value.lower()
            if value == "log":
                self._corate_func = _comp_log_strain
                self._convert_Lt_tag = "DsigmaDe_2_DSDE"
            elif value == "log_inc":
                self._corate_func = _comp_log_strain_inc
                self._convert_Lt_tag = "DsigmaDe_2_DSDE"
            elif value in ["gn", "green_naghdi"]:
                self._corate_func = _comp_gn_strain
                self._convert_Lt_tag = "DsigmaDe_GreenNaghdiDD_2_DSDE"
            elif value == "jaumann":
                self._corate_func = _comp_jaumann_strain
                self._convert_Lt_tag = "DsigmaDe_JaumannDD_2_DSDE"
            elif value == "log_r":
                self._corate_func = _comp_log_strain_R
                self._convert_Lt_tag = "DsigmaDe_2_DSDE"
            elif value == "log_r_inc":
                self._corate_func = _comp_log_strain_R_inc
                self._convert_Lt_tag = "DsigmaDe_2_DSDE"
            else:
                raise ValueError(
                    'corate value not understood. Choose between "log", "log_R", \
                    "green_naghdi" or "jaumann"'
                )

        if self.nlgeom == "TL":
            value = value.lower()
            if value == "log":
                self._corate_func = _comp_log_strain
                self._convert_Lt_tag = "DsigmaDe_2_DSDE"
            elif value == "log_inc":
                self._corate_func = _comp_log_strain_inc
                self._convert_Lt_tag = "DsigmaDe_2_DSDE"
            elif value in ["gn", "green_naghdi"]:
                self._corate_func = _comp_gn_strain
                self._convert_Lt_tag = "DsigmaDe_2_DSDE"
            elif value == "jaumann":
                self._corate_func = _comp_jaumann_strain
                self._convert_Lt_tag = "DsigmaDe_2_DSDE"
            elif value == "log_r":
                self._corate_func = _comp_log_strain_R
                self._convert_Lt_tag = "DsigmaDe_2_DSDE"
            elif value == "log_r_inc":
                self._corate_func = _comp_log_strain_R_inc
                self._convert_Lt_tag = "DsigmaDe_2_DSDE"
            else:
                raise ValueError(
                    'corate value not understood. Choose between "log", "log_R", \
                    "green_naghdi" or "jaumann"'
                )


# elements used to avoid volumetric locking
_MEAN_DILATATION_ELEMENTS = {
    elm: elm + "md"
    for elm in [
        "quad4",
        "quad8",
        "quad9",
        "tri6",
        "hex8",
        "hex20",
        "wed6",
        "wed15",
        "wed18",
        "tet10",
    ]
}
_INCOMPRESSIBILITY_ELEMENTS = {
    None: {},
    "fbar": {"quad4": "quad4sri", "hex8": "hex8sri"},
    "sri": {"quad4": "quad4sri", "hex8": "hex8sri"},
    "mean_dilatation": _MEAN_DILATATION_ELEMENTS,
    "auto": _MEAN_DILATATION_ELEMENTS,
}


def _get_dilatation_degree(assembly):
    # polynomial degree of the discontinuous dilatation of the mean dilatation
    # elements. Return None for the other elements.
    elm = get_element(assembly.elm_type)
    if not hasattr(elm, "get_elm_type"):
        return None
    return getattr(elm.get_elm_type("_DispX"), "pressure_degree", None)


def _get_dilatation_basis(assembly):
    # basis functions of the dilatation at gauss points: (n_elm_gp, n_pvals)
    mesh = assembly.mesh
    if assembly.n_elm_gp not in mesh._elm_interpolation:
        mesh.init_interpolation(assembly.n_elm_gp)
    xi_pg = mesh._elm_interpolation[assembly.n_elm_gp].xi_pg
    return _evaluate_monomials(
        xi_pg, _monomial_exponents(xi_pg.shape[1], _get_dilatation_degree(assembly))
    )


def _element_projection(gp_field, weights, basis):
    # Volume weighted L2 projection over each element of a gauss point field
    # onto the given basis. gp_field and weights: (n_elm_gp, n_elements),
    # basis: (n_elm_gp, n_basis). Return an array of shape (n_elm_gp, n_elements)
    if basis.shape[1] == 1:  # mean value
        return np.broadcast_to(
            (weights * gp_field).sum(axis=0) / weights.sum(axis=0), gp_field.shape
        )
    mass = np.einsum("gi,ge,gj->eij", basis, weights, basis)
    rhs = np.einsum("gi,ge->ei", basis, weights * gp_field)
    return np.einsum("gi,ei->ge", basis, np.linalg.solve(mass, rhs[..., None])[..., 0])


# function to compute the displacement gradient
def _comp_grad_disp(assembly, displacement):
    grad_values = assembly.get_grad_disp(displacement, "GaussPoint")
    if assembly.space.is_axisymmetric:
        mesh = assembly.current.mesh
        # Refresh r_current at gauss points: used by the symbolic
        # operator eps[2] = DispX / r in get_weak_equation and by the
        # 2*pi*r weak-form integration weight, both at the current config.
        assembly.sv["_R_gausspoints"] = mesh.convert_data(
            mesh.nodes[:, 0],
            "Node",
            "GaussPoint",
            n_elm_gp=assembly.n_elm_gp,
        )

        # F_theta-theta = r_current / R_reference, hence
        # grad_values[2][2] = u_r / R_reference. See the
        # "Theory of axisymmetric kinematics" section of
        # :class:`fedoo.core.mechanical3d.Mechanical3D` for the derivation.
        R0 = assembly.sv["_R0_gausspoints"]
        rank_dispx = assembly.space.variable_rank("DispX")
        n = mesh.n_nodes
        grad_values[2][2] = np.divide(
            mesh.convert_data(
                displacement[rank_dispx * n : (rank_dispx + 1) * n],
                "Node",
                "GaussPoint",
                n_elm_gp=assembly.n_elm_gp,
            ),
            R0,
            out=np.zeros_like(R0),
            where=R0 != 0,
        )  # zero at the symmetry axis (R0 == 0)
    assembly.sv["DispGradient"] = grad_values
    return grad_values


def _get_grad_disp_center(assembly, displacement):
    return [
        [
            assembly.get_gp_results(op, displacement, n_elm_gp=1)
            if op != 0
            else np.zeros(assembly.mesh.n_elements)
            for op in line_op
        ]
        for line_op in assembly.space.op_grad_u()
    ]


def _comp_grad_disp_fbar(assembly, displacement):
    # Small-strain limit of F-bar: replace the volumetric strain by its value
    # at the element centroid, consistently with the finite-strain F-bar
    # definition and with _get_fbar_tangent_op.
    grad_values = np.array(_comp_grad_disp(assembly, displacement))
    dvol = np.trace(grad_values)

    grad_values_center = np.array(_get_grad_disp_center(assembly, displacement))
    dvol_center = np.trace(grad_values_center)
    grad_values[[0, 1, 2], [0, 1, 2]] -= (
        1 / 3 * (dvol.reshape(assembly.n_elm_gp, -1) - dvol_center).ravel()
    )
    assembly.sv["DispGradient"] = grad_values
    return grad_values


def _comp_grad_disp_mean_dilatation(assembly, displacement):
    # small strain mean dilatation: the dilatation is replaced by its
    # projection over each element
    grad_values = np.array(_comp_grad_disp(assembly, displacement))
    n = 3
    dvol = sum(grad_values[i, i] for i in range(n)).reshape(assembly.n_elm_gp, -1)
    weights = assembly._get_gaussian_quadrature_mat().data.reshape(
        assembly.n_elm_gp, -1
    )
    correction = (
        _element_projection(dvol, weights, _get_dilatation_basis(assembly)) - dvol
    ).ravel() / n
    for i in range(n):
        grad_values[i, i] += correction
    assembly.sv["DispGradient"] = grad_values
    return grad_values


# function to compute F tensor (required nl corate function used with simcoon)
def _check_F_validity(F1):
    """Reject degenerated trial kinematics (det F <= 0) as a recoverable error.

    A Newton iterate that inverts elements must be treated as a FAILED step
    (backtrack / time-step cut), never assembled: downstream, the polar
    decomposition aborts and the assembled matrix fills with NaN (reported
    by direct solvers as a spurious 'numerically singular').

    Returns det F (per Gauss point) so callers can reuse it as J.
    """
    # explicit 3x3 cofactor expansion: ~10x faster than np.linalg.det on
    # (3, 3, n_gp) arrays, and the result is reused as J by _comp_Fbar
    det_f = (
        F1[0, 0] * (F1[1, 1] * F1[2, 2] - F1[1, 2] * F1[2, 1])
        - F1[0, 1] * (F1[1, 0] * F1[2, 2] - F1[1, 2] * F1[2, 0])
        + F1[0, 2] * (F1[1, 0] * F1[2, 1] - F1[1, 1] * F1[2, 0])
    )
    min_det = det_f.min()
    if not np.isfinite(min_det) or min_det <= 1e-12:
        n_bad = int(np.count_nonzero(~(det_f > 1e-12)))
        raise InvalidKinematicStateError(
            f"trial state degenerates the kinematics: {n_bad} Gauss "
            f"point(s) with det F <= 0 (min det F = {min_det:.3e}). "
            "Treated as a failed Newton iterate."
        )
    return det_f


def _comp_F(assembly, displacement):
    grad_values = _comp_grad_disp(assembly, displacement)

    eye_3 = np.empty((3, 3, 1), order="F")
    eye_3[:, :, 0] = np.eye(3)
    F1 = np.add(eye_3, grad_values, order="F")
    _check_F_validity(F1)
    assembly.sv["F"] = F1
    if "F" not in assembly.sv_start:
        F0 = np.empty_like(F1)
        F0[...] = eye_3
        assembly.sv_start["F"] = F0


def _comp_F_mean_dilatation(assembly, displacement):
    # F tensor for the mean dilatation method: the volume change J is replaced
    # by its projection theta over each element (reference configuration)
    if _get_dilatation_degree(assembly) is None:
        return _comp_F(assembly, displacement)

    grad_values = _comp_grad_disp(assembly, displacement)

    eye_3 = np.empty((3, 3, 1), order="F")
    eye_3[:, :, 0] = np.eye(3)
    F1 = np.add(eye_3, grad_values, order="F")
    J = _check_F_validity(F1)

    theta = _element_projection(
        J.reshape(assembly.n_elm_gp, -1),
        assembly.sv["_MeanDilatation_dV0"],
        _get_dilatation_basis(assembly),
    ).ravel()
    if theta.min() <= 1e-12:
        raise InvalidKinematicStateError(
            "trial state degenerates the kinematics: projected volume change "
            f"<= 0 (min = {theta.min():.3e}). Treated as a failed Newton iterate."
        )
    scale = theta / J
    # The assumed constitutive deformation gradient uses the 3D volumetric
    # split also in plane strain. The physical kinematics still have F_zz=1;
    # only the assumed F-bar/mean-dilatation tensor has a modified F_zz.
    F1 *= scale ** (1 / 3)

    assembly.sv["_MeanDilatation_scale"] = scale
    assembly.sv["Jbar"] = theta
    assembly.sv["F"] = F1
    if "F" not in assembly.sv_start:
        F0 = np.empty_like(F1)
        F0[...] = eye_3
        assembly.sv_start["F"] = F0


def _comp_Fbar(assembly, displacement):
    # funciton to compute F tensor using the Fbar stabilization for
    # nearly incompressible materials
    grad_values = _comp_grad_disp(assembly, displacement)

    eye_3 = np.empty((3, 3, 1), order="F")
    eye_3[:, :, 0] = np.eye(3)
    F1 = np.add(eye_3, grad_values, order="F")
    J = _check_F_validity(F1)

    grad_values_center = _get_grad_disp_center(assembly, displacement)
    # the element-center Jacobian must be validated too: a negative Jcenter
    # would silently turn the fractional power below into NaN
    Jcenter = _check_F_validity(np.add(eye_3, grad_values_center))
    # Jcenter = np.mean(J.reshape(assembly.n_elm_gp, -1), axis=0)
    F1 = F1 * ((Jcenter / J.reshape(assembly.n_elm_gp, -1)).ravel() ** (1 / 3))

    assembly.sv["F"] = F1
    if "F" not in assembly.sv_start:
        F0 = np.empty_like(F1)
        F0[...] = eye_3
        assembly.sv_start["F"] = F0


# funtions to compute strain
def _comp_linear_strain(wf, assembly, pb):
    # not compatible with PGD assembly.
    assert not (wf.nlgeom), "the current strain measure isn't adapted for finite strain"
    grad_values = assembly.sv["DispGradient"]

    strain = np.empty((6, len(grad_values[0][0])), order="F")
    # order = F for compatibility with simcoon without performance loss
    # in other cases
    strain[0:3] = [grad_values[i][i] for i in range(3)]
    strain[3] = grad_values[0][1] + grad_values[1][0]
    strain[4] = grad_values[0][2] + grad_values[2][0]
    strain[5] = grad_values[1][2] + grad_values[2][1]
    assembly.sv["Strain"] = StrainTensorList(strain)


def _comp_log_strain(wf, assembly, pb):
    F1 = assembly.sv["F"]
    (D, DR, Omega) = sim.objective_rate(
        "log", assembly.sv_start["F"], F1, pb.dtime, False
    )
    assembly.sv["DR"] = DR
    assembly.sv["Strain"] = StrainTensorList(sim.Log_strain(F1, True, False))


def _comp_log_strain_inc(wf, assembly, pb):
    F1 = assembly.sv["F"]
    (DStrain, D, DR, Omega) = sim.objective_rate(
        "log", assembly.sv_start["F"], F1, pb.dtime, True
    )
    assembly.sv["DR"] = DR
    assembly.sv["DStrain"] = StrainTensorList(DStrain)


def _comp_log_strain_R(wf, assembly, pb):
    F1 = assembly.sv["F"]
    (D, DR, Omega) = sim.objective_rate(
        "log_R", assembly.sv_start["F"], F1, pb.dtime, False
    )
    assembly.sv["DR"] = DR
    assembly.sv["Strain"] = StrainTensorList(sim.Log_strain(F1, True, False))


def _comp_log_strain_R_inc(wf, assembly, pb):
    F1 = assembly.sv["F"]
    (DStrain, D, DR, Omega) = sim.objective_rate(
        "log_R", assembly.sv_start["F"], F1, pb.dtime, True
    )
    assembly.sv["DR"] = DR
    assembly.sv["DStrain"] = StrainTensorList(DStrain)


def _comp_jaumann_strain(wf, assembly, pb):
    F1 = assembly.sv["F"]
    (DStrain, D, DR, Omega) = sim.objective_rate(
        "jaumann", assembly.sv_start["F"], F1, pb.dtime, True
    )
    assembly.sv["DR"] = DR
    assembly.sv["DStrain"] = StrainTensorList(DStrain)


def _comp_gn_strain(wf, assembly, pb):
    # green_naghdi corate
    F1 = assembly.sv["F"]
    (DStrain, D, DR, Omega) = sim.objective_rate(
        "green_naghdi", assembly.sv_start["F"], F1, pb.dtime, True
    )
    assembly.sv["DR"] = DR
    assembly.sv["DStrain"] = StrainTensorList(DStrain)
