"""Mixed Stress Equilibrium weak form."""

from fedoo.weakform.stress_equilibrium import StressEquilibrium, _comp_grad_disp
import numpy as np


class StressEquilibriumMixed(StressEquilibrium):
    """
    Mixed displacement/pressure weak formulation for solid mechanics.

    This weak formulation is suitable for nearly incompressible materials.
    It introduces a pressure variable to treat the volumetric behavior
    separately from the deviatoric behavior, avoiding volumetric locking.

    The element formulation should satisfy the LBB condition (e.g. Q2-Q1 or
    P1+/P1).

    Parameters
    ----------
    constitutivelaw: ConstitutiveLaw
        Material constitutive law.
    name: str, optional
        Name of the weak form.
    nlgeom: bool, optional
        Non-linear geometry flag.
    bulk_modulus: float, optional
        Bulk modulus used for scaling the pressure equation. If None, it
        is estimated from the tangent matrix trace.
        Note: This is used for scaling the pressure constraint equation.
    space: ModelingSpace, optional
        Modeling space.

    Attributes
    ----------
    symmetrized_matrix : bool
        Only used when nlgeom is False.
        If True, the deviatoric tangent is computed using the symmetric
        projection (P:H:P) to ensure a symmetric displacement stiffness block.
        If False, a single projection (P:H) is used. Default is True.
    """

    def __init__(
        self,
        constitutivelaw,
        bulk_modulus=None,
        name="",
        nlgeom=None,
        space=None,
    ):
        super().__init__(constitutivelaw, name, nlgeom, space)

        self.space.new_variable("Pressure")
        self.bulk_modulus = bulk_modulus
        self.symmetrized_matrix = True

    def get_weak_equation(self, assembly, pb):
        """Get the weak equation related to the current problem state."""
        # -----------------------------------------------------------
        # 1. Kinematics (Strain / Deformation Gradient)
        # -----------------------------------------------------------
        if assembly._nlgeom == "TL":  # add initial displacement effect
            if self.space.is_axisymmetric:
                raise NotImplementedError(
                    "'2Daxi' ModelingSpace is not implemented with the total "
                    "lagrangian formulation for the mixed (u, p) weak form. "
                    "Use updated lagrangian instead."
                )
            eps = self.space.op_strain(assembly.sv["DispGradient"])
            initial_stress = assembly.sv["PK2"]
        else:
            eps = self.space.op_strain()
            # Stress = Cauchy for updated lagrangian method
            initial_stress = assembly.sv["Stress"]

            if self.space.is_axisymmetric:
                rr = assembly.sv["_R_gausspoints"]
                eps[2] = self.space.variable("DispX") * np.divide(
                    1, rr, out=np.zeros_like(rr), where=rr != 0
                )

        # -----------------------------------------------------------
        # 2. Decompose Stress and Tangent Matrix
        # -----------------------------------------------------------
        H = assembly.sv["TangentMatrix"]

        if assembly._nlgeom:
            # For finite strain, Constitutive Law evaluated with F_bar.
            # StressDev is directly the output of the constitutive law in update_2.
            # H is the isochoric part of the tangent as it is computed from F_bar.
            sigma_dev = assembly.sv.get("StressDev", initial_stress)
            H_dev = [list(row) for row in H]  # Copy to avoid modifying SV directly

            # # Correction term: -2/3 * (sigma_dev_i + sigma_dev_j)
            # # This accounts for the derivative of J^(-1/3)
            for i in range(6):
                for j in range(6):
                    term_i = sigma_dev[i] if j < 3 else 0.0
                    term_j = sigma_dev[j] if i < 3 else 0.0
                    H_dev[i][j] -= (2.0 / 3.0) * (term_i + term_j)
        else:
            # Small strain:
            # p_mat is > 0 for compressive
            if not (np.isscalar(initial_stress) and initial_stress == 0):
                p_mat = (-1 / 3.0) * (
                    initial_stress[0] + initial_stress[1] + initial_stress[2]
                )
                sigma_dev = [
                    initial_stress[i] + (p_mat if i < 3 else 0) for i in range(6)
                ]
            else:
                p_mat = 0
                sigma_dev = [0 for i in range(6)]

            # Decompose Tangent Matrix H into Deviatoric and Volumetric parts
            # row_p[j] = d(p_mat)/d(eps_j) = 1/3 * sum_i=1..3 H[i][j]
            # Compute exact single-projected tangent: H_dev = P:H

            if self.symmetrized_matrix:
                # Compute the symmetric dev part of the tangent matrix: Hdv = P:H:P
                row_p = [(H[0][j] + H[1][j] + H[2][j]) / 3.0 for j in range(6)]
                col_p = [(H[i][0] + H[i][1] + H[i][2]) / 3.0 for i in range(6)]
                vol_vol = (row_p[0] + row_p[1] + row_p[2]) / 3.0

                H_dev = [
                    [
                        H[i][j]
                        - (row_p[j] if i < 3 else 0.0)
                        - (col_p[i] if j < 3 else 0.0)
                        + (vol_vol if (i < 3 and j < 3) else 0.0)
                        for j in range(6)
                    ]
                    for i in range(6)
                ]
            else:
                # H_dev[i][j] = H[i][j] - I[i] * row_p[j]
                # This removes the stiff volumetric part from the displacement stiffness
                row_p = [(H[0][j] + H[1][j] + H[2][j]) / 3.0 for j in range(6)]
                H_dev = [
                    [H[i][j] - (row_p[j] if i < 3 else 0.0) for j in range(6)]
                    for i in range(6)
                ]

        # -----------------------------------------------------------
        # 3. Pressure Variables
        # -----------------------------------------------------------
        # P_inc: Incremental pressure (operator used to build the stiffness matrix)
        P_inc = self.space.variable("Pressure")

        # P_curr: Current total pressure at gauss points
        P_curr = assembly.sv["_Pressure_gp"]

        # Total Pressure used for equilibrium
        P_total = P_inc + P_curr

        # -----------------------------------------------------------
        # 4. Momentum Equation (Displacement)
        # -----------------------------------------------------------
        # The weak form for the momentum equation is based on a deviatoric/volumetric
        # split of the stress tensor: sigma = sigma_dev - p * I (p>0 in compression))
        # The virtual work is integral(delta_eps : sigma) dV
        # delta_eps : sigma = delta_eps_dev : sigma_dev + delta_eps_vol * p
        # The implementation below is equivalent to:
        # delta_eps : (H_dev:eps + sigma_dev) + delta_eps_vol * P_total

        sigma_wf = []
        for i in range(6):
            # Constant part (sigma_dev) + Linear part from Displacement (H_dev * eps)
            val = (
                sum([0 if eps[j] == 0 else eps[j] * H_dev[i][j] for j in range(6)])
                + sigma_dev[i]
            )
            # Add Pressure part (Volumetric)
            if i < 3:
                val = val - P_total
            sigma_wf.append(val)

        DiffOp = sum(
            [0 if eps[i] == 0 else eps[i].virtual * sigma_wf[i] for i in range(6)]
        )

        # Add geometric stiffness if requested (same as standard StressEquilibrium)
        if self.geometric_stiffness and not (
            np.isscalar(initial_stress) and initial_stress == 0
        ):
            DiffOp = DiffOp + sum(
                [
                    0
                    if self._nl_strain_op_vir[i] == 0
                    else self._nl_strain_op_vir[i] * initial_stress[i]
                    for i in range(6)
                ]
            )

        # -----------------------------------------------------------
        # 5. Pressure Constraint Equation
        # -----------------------------------------------------------
        if assembly._nlgeom:
            if self.bulk_modulus is None:
                raise ValueError(
                    "bulk_modulus must be explicitly defined for non-linear geometry in Mixed formulations."
                )

            # Retrieve the TOTAL J saved from _comp_F
            lnJ_curr = assembly.sv.get("lnJ", np.zeros(assembly.n_gauss_points))
            eps_vol = eps[0] + eps[1] + eps[2]

            # Linearized constraint: delta_p * (lnJ_curr - P_curr/K + eps_vol - P_inc/K) = 0
            pressure_residual = -lnJ_curr - (P_curr / self.bulk_modulus)
            pressure_stiffness = -eps_vol - (P_inc / self.bulk_modulus)

            DiffOp += P_inc.virtual * (pressure_residual + pressure_stiffness)
        else:
            # Small strain: constraint on p = p_mat
            if self.bulk_modulus is None:
                # Bulk modulus read from the tangent: K = (1/9) sum_{i,j<3} H[i][j]
                # (this equals vol_vol of the P:H:P split).
                K_scale = sum(H[i][j] for i in range(3) for j in range(3)) / 9.0
                K_scale = np.where(K_scale == 0, 1.0, K_scale)
            else:
                K_scale = self.bulk_modulus

            # d(p_mat) term
            dp_mat = -sum([0 if eps[j] == 0 else eps[j] * row_p[j] for j in range(6)])

            # Constraint Weak Form: integral( delta_P * 1/K * (P_total - p_mat_total) )
            pressure_residual = (1.0 / K_scale) * (-P_curr + p_mat)
            pressure_stiffness_term = (1.0 / K_scale) * (-P_inc + dp_mat)

            DiffOp += P_inc.virtual * (pressure_residual + pressure_stiffness_term)

        # Axisymmetric volume integration factor
        if self.space.is_axisymmetric:
            DiffOp = DiffOp * ((2 * np.pi) * rr)

        return DiffOp

    def initialize(self, assembly, pb):
        super().initialize(assembly, pb)
        assembly.sv["_Pressure_gp"] = 0
        assembly.sv["StressDev"] = assembly.sv["Stress"]
        assembly.sv["StrainIsochoric"] = assembly.sv["Strain"]
        if assembly._nlgeom:
            assembly.sv["lnJ"] = np.zeros(assembly.n_gauss_points)

    def set_start(self, assembly, pb):
        """Start a new time increment."""
        super().set_start(assembly, pb)
        if assembly._nlgeom and not np.array_equal(pb.get_dof_solution(), 0):
            # Stress and Strain has been rotated. Update StressDev and StrainIsochoric.
            assembly.sv["StressDev"] = assembly.sv["Stress"].copy(asarray=True)
            assembly.sv["StressDev"].array[:3] += assembly.sv["_Pressure_gp"]
            assembly.sv["StrainIsochoric"] = assembly.sv["Strain"].copy(asarray=True)
            assembly.sv["StrainIsochoric"].array[:3] -= (1 / 3.0) * assembly.sv["lnJ"]

    def update(self, assembly, pb):
        """Update the weakform to the current state.

        This method is applyed before the update of constutive law (stress and
        stiffness matrix).
        """
        super().update(assembly, pb)
        # get and store the current pressure at GaussPoint from the pressure field
        pressure_gp = assembly.get_gp_results(
            self.space.variable("Pressure"), pb.get_dof_solution()
        )
        assembly.sv["_Pressure_gp"] = pressure_gp
        if assembly._nlgeom:
            # put the isochoric stress in entry of the constitutive law
            # strain is already the isochoric part here, but not the start value
            assembly.sv["Stress"] = assembly.sv["StressDev"]
            assembly.sv_start["Stress"] = assembly.sv_start["StressDev"]
            assembly.sv_start["Strain"] = assembly.sv_start["StrainIsochoric"]

    def update_2(self, assembly, pb):
        """Update the weakform to the current state.

        This method is applyed after the constutive law update (stress and
        stiffness matrix).
        """
        super().update_2(assembly, pb)

        if assembly._nlgeom:
            pressure_gp = assembly.sv["_Pressure_gp"]

            # the stress computed by the constitutive law is the isochor stress
            assembly.sv["StressDev"] = assembly.sv["Stress"]
            assembly.sv["Stress"] = assembly.sv["Stress"].copy(asarray=True)
            assembly.sv["Stress"].array[:3] -= pressure_gp
            # strain from _comp_F is the isochoric part as it comes from Fbar
            assembly.sv["StrainIsochoric"] = assembly.sv["Strain"]
            assembly.sv["Strain"] = assembly.sv["Strain"].copy(asarray=True)
            assembly.sv["Strain"].array[:3] += (1 / 3.0) * assembly.sv["lnJ"]

    @property
    def fbar(self):
        """Set to True to use the F-bar method.

        The F-bar method should be used to stabilized constitutive laws with
        nearly incompressible behavior.
        """
        return False

    @fbar.setter
    def fbar(self, value):
        if not isinstance(value, bool):
            raise TypeError("bool expeted for fbar")
        if value:
            raise ValueError("fbar can't be activated for Mixed formulation")

    def _comp_F(self, assembly, displacement):
        # compute only the isochoric part of F and volume change J
        grad_values = _comp_grad_disp(assembly, displacement)

        eye_3 = np.empty((3, 3, 1), order="F")
        eye_3[:, :, 0] = np.eye(3)
        F1 = np.add(eye_3, grad_values, order="F")

        # Calculate J and save its log for the pressure constraint
        J = np.linalg.det(F1.transpose((2, 0, 1)))
        assembly.sv["lnJ"] = np.log(J)

        F1 = F1 * (J.reshape(assembly.n_elm_gp, -1).ravel() ** (-1 / 3))

        assembly.sv["F"] = F1
        if "F" not in assembly.sv_start:
            F0 = np.empty_like(F1)
            F0[...] = eye_3
            assembly.sv_start["F"] = F0
