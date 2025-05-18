# doesn't seem to work. Not imported by default.
# use with hex8sri elements


from fedoo.core.weakform import WeakFormBase, WeakFormSum
from fedoo.weakform.stress_equilibrium import StressEquilibrium
import numpy as np


class StressEquilibriumBbar(StressEquilibrium):
    """Weak formulation of the mechanical equilibrium equation for solids.

    This method is still experimental ! Use with caution. The fbar method
    from the standard StressEquilibrium should be prefered.

    The main point to consider are:
      * This weak form is the same as the standard StressEquilibrium but
        allow to use the Bbar method.
      * This weak form can be used for solid in 3D or using a 2D plane
        assumption (plane strain or plane stress).
      * Include initial stress for non linear problems or if defined in
        the associated assembly.
      * This weak form accepts geometrical non linearities if simcoon is
        installed. (nlgeom should be in {True, 'UL', 'TL'}. In this case
        the initial displacement is also
        considered.

    Parameters
    ----------
    constitutivelaw: ConstitutiveLaw name (str) or ConstitutiveLaw object
        Material Constitutive Law (:mod:`fedoo.constitutivelaw`)
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

    def get_weak_equation(self, assembly, pb):
        """Get the weak equation related to the current problem state."""
        if assembly._nlgeom == "TL":  # add initial displacement effect
            eps = self.space.op_strain(assembly.sv["DispGradient"])
            initial_stress = assembly.sv["PK2"]
        else:
            eps = self.space.op_strain()
            if assembly.elm_type in ["hex8sri", "quad4sri"]:
                eps[0] = (
                    eps[0]
                    - self.space.derivative("DispX", "X")
                    + self.space.derivative("_DispX", "X")
                )
                eps[1] = (
                    eps[1]
                    - self.space.derivative("DispY", "Y")
                    + self.space.derivative("_DispY", "Y")
                )
                if assembly.space.ndim == 3:
                    eps[2] = (
                        eps[2]
                        - self.space.derivative("DispZ", "Z")
                        + self.space.derivative("_DispZ", "Z")
                    )

            initial_stress = assembly.sv[
                "Stress"
            ]  # Stress = Cauchy for updated lagrangian method

            if self.space._dimension == "2Daxi":
                rr = assembly.sv["_R_gausspoints"]

                # nlgeom = False
                eps[2] = self.space.variable("DispX") * np.divide(
                    1, rr, out=np.zeros_like(rr), where=rr != 0
                )  # put zero if X==0 (division by 0)
                # eps[2] = self.space.variable('DispX') * (1/rr)

        H = assembly.sv["TangentMatrix"]

        sigma = [
            sum([0 if eps[j] == 0 else eps[j] * H[i][j] for j in range(6)])
            for i in range(6)
        ]

        DiffOp = sum(
            [0 if eps[i] == 0 else eps[i].virtual * sigma[i] for i in range(6)]
        )

        if not np.array_equal(initial_stress, 0):
            # this term doesnt seem to improve convergence !
            # if assembly._nlgeom:
            #     DiffOp = DiffOp + \
            #         sum([0 if self._nl_strain_op_vir[i] == 0 else
            #              self._nl_strain_op_vir[i] * initial_stress[i]
            #              for i in range(6)])

            DiffOp = DiffOp + sum(
                [
                    0 if eps[i] == 0 else eps[i].virtual * initial_stress[i]
                    for i in range(6)
                ]
            )

        if self.space._dimension == "2Daxi":
            DiffOp = DiffOp * ((2 * np.pi) * rr)

        return DiffOp

    def initialize(self, assembly, pb):
        """Initialize the weakform at the begining of a problem."""
        super().initialize(assembly, pb)
        if assembly.elm_type in ["hex8sri", "quad4sri"]:
            if assembly._nlgeom == "TL":
                raise NotImplementedError(
                    f"{assembly.elm_type} not \
                                          implemented with total lagrangian \
                                          formulation"
                )
            self.space.variable_alias("_DispX", "DispX")
            self.space.variable_alias("_DispY", "DispY")
            if self.space.ndim == 3:
                self.space.variable_alias("_DispZ", "DispZ")
                self.space.new_vector("_Disp", ("_DispX", "_DispY", "_DispZ"))
            else:
                self.space.new_vector("_Disp", ("_DispX", "_DispY"))


class StressEquilibriumFbar(StressEquilibrium):
    """Weak formulation of the mechanical equilibrium equation for solids.

    This method is still experimental ! Use with caution. The fbar method
    can be used with the the standard StressEquilibrium weak form.

    The main point to consider are:
      * This weak form is the same as the standard StressEquilibrium but
        allow to use the consistant tangent matrix with the Fbar method.
      * This weak form can be used for solid in 3D or using a 2D plane
        assumption (plane strain or plane stress).
      * Include initial stress for non linear problems or if defined in
        the associated assembly.
      * This weak form accepts geometrical non linearities if simcoon is
        installed. (nlgeom should be in {True, 'UL', 'TL'}. In this case
        the initial displacement is also
        considered.

    Parameters
    ----------
    constitutivelaw: ConstitutiveLaw name (str) or ConstitutiveLaw object
        Material Constitutive Law (:mod:`fedoo.constitutivelaw`)
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

    def __init__(self, constitutivelaw, name="", nlgeom=None, space=None):
        super().__init__(constitutivelaw, name, nlgeom, space)
        self.fbar = True
        self.assembly_options["elm_type", "quad4"] = "quad4sri"
        self.assembly_options["elm_type", "hex8"] = "hex8sri"

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

            if self.space._dimension == "2Daxi":
                rr = assembly.sv["_R_gausspoints"]

                # nlgeom = False
                eps[2] = self.space.variable("DispX") * np.divide(
                    1, rr, out=np.zeros_like(rr), where=rr != 0
                )  # put zero if X==0 (division by 0)
                # eps[2] = self.space.variable('DispX') * (1/rr)

        H = assembly.sv["TangentMatrix"]

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

        #
        # Fbar additional term fro
        #
        if self.fbar:
            # ref: DESIGN OF SIMPLE LOW ORDER FINITE ELEMENTS FOR LARGE STRAIN
            # ANALYSIS OF NEARLY INCOMPRESSIBLE SOLIDS, Neto et al,
            # International Journal of Solids and Structures

            q = [
                (1 / 3.0) * (H[i][0] + H[i][1] + H[i][2])
                - (2 / 3.0) * initial_stress[i]
                for i in range(6)
            ]

            # q = np.column_stack([q, q, q, 0, 0, 0])
            eps0 = self._op_strain_center()
            # eps0 = [0,0,0] -> eps operator at the element centroid

            DiffOp = DiffOp + sum(
                [
                    (
                        0
                        if eps[i] == 0
                        else eps[i].virtual
                        * q[i]
                        * (eps0[0] - eps[0] + eps0[1] - eps[1] + eps0[2] - eps[2])
                    )
                    for i in range(6)
                ]
            )

        if self.space._dimension == "2Daxi":
            DiffOp = DiffOp * ((2 * np.pi) * rr)

        return DiffOp

    def _op_strain_center(self):
        du_dx = self.space.derivative("_DispX", "X")
        dv_dy = self.space.derivative("_DispY", "Y")
        du_dy = self.space.derivative("_DispX", "Y")
        dv_dx = self.space.derivative("_DispY", "X")

        if self.space.ndim == 2:
            eps = [du_dx, dv_dy, 0, du_dy + dv_dx, 0, 0]

        else:  # assume ndim == 3
            dw_dz = self.space.derivative("_DispZ", "Z")
            du_dz = self.space.derivative("_DispX", "Z")
            dv_dz = self.space.derivative("_DispY", "Z")
            dw_dx = self.space.derivative("_DispZ", "X")
            dw_dy = self.space.derivative("_DispZ", "Y")
            eps = [
                du_dx,
                dv_dy,
                dw_dz,
                du_dy + dv_dx,
                du_dz + dw_dx,
                dv_dz + dw_dy,
            ]
        return eps

    def initialize(self, assembly, pb):
        """Initialize the weakform at the begining of a problem."""
        super().initialize(assembly, pb)
        self.assembly_options["assume_sym"] = False

        if assembly.elm_type in ["hex8sri", "quad4sri"]:
            if assembly._nlgeom == "TL":
                raise NotImplementedError(
                    f"{assembly.elm_type} not \
                                          implemented with total lagrangian \
                                          formulation"
                )
            self.space.variable_alias("_DispX", "DispX")
            self.space.variable_alias("_DispY", "DispY")
            if self.space.ndim == 3:
                self.space.variable_alias("_DispZ", "DispZ")
                self.space.new_vector("_Disp", ("_DispX", "_DispY", "_DispZ"))
            else:
                self.space.new_vector("_Disp", ("_DispX", "_DispY"))

class HourglassStiffness(WeakFormBase):
    # """Weak formulation of the mechanical equilibrium equation for solids.

    # This method is still experimental ! Use with caution. The fbar method
    # can be used with the the standard StressEquilibrium weak form.

    # The main point to consider are:
    #   * This weak form is the same as the standard StressEquilibrium but
    #     allow to use the consistant tangent matrix with the Fbar method.
    #   * This weak form can be used for solid in 3D or using a 2D plane
    #     assumption (plane strain or plane stress).
    #   * Include initial stress for non linear problems or if defined in
    #     the associated assembly.
    #   * This weak form accepts geometrical non linearities if simcoon is
    #     installed. (nlgeom should be in {True, 'UL', 'TL'}. In this case
    #     the initial displacement is also
    #     considered.

    # Parameters
    # ----------
    # constitutivelaw: ConstitutiveLaw name (str) or ConstitutiveLaw object
    #     Material Constitutive Law (:mod:`fedoo.constitutivelaw`)
    # name: str
    #     name of the WeakForm
    # nlgeom: bool, 'UL' or 'TL', optional
    #     If True, the geometrical non linearities are activate based on the
    #     updated lagrangian method. This parameters is used only in the
    #     context of NonLinearProblems such as
    #     :mod:`fedoo.problem.NonLinearStatic` or
    #     :mod:`fedoo.problem.NonLinearNewmark`.
    #     If nlgeom == 'UL' the updated lagrangian method is used (same as True).
    #     If nlgeom == 'TL' the total lagrangian method is used.
    #     If not defined, the problem.nlgeom parameter is used instead.
    # space: ModelingSpace
    #     Modeling space associated to the weakform. If None is specified,
    #     the active ModelingSpace is considered.
    # """

    def __init__(self, stiffness_coef, constitutivelaw = None, name="", nlgeom=False, space=None):
        WeakFormBase.__init__(self, name, space)
        self.assembly_options["n_elm_gp"] = 1
        self.assembly_options["elm_type", "quad4"] = "quad4hourglass"
        self.assembly_options["elm_type", "hex8"] = "hex8hourglass"
        self.stiffness_coef = stiffness_coef
        self.nlgeom = nlgeom
        """Method used to treat the geometric non linearities.
            * Set to False if geometric non linarities are ignored.
            * Set to True or 'UL' to use the updated lagrangian method
              (update the mesh)
            * Set to 'TL' to use the total lagrangian method (base on the
              initial mesh with initial displacement effet)
        """

    def get_weak_equation(self, assembly, pb):
        """Get the weak equation related to the current problem state."""
        op_dq = assembly.space.op_disp()
        ndim = len(op_dq)

        if np.array_equal(pb.get_dof_solution(), 0):
            q0 = [0,0,0]
        else:
            q0 = [assembly.get_gp_results(op_dq[i], pb.get_dof_solution()) for i in range(ndim)]

        DiffOp = sum([op_dq[i].virtual * (op_dq[i]+q0[i]) for i in range(len(op_dq))])
        # DiffOp = sum([op_du[i].virtual * op_du[i] for i in range(len(op_du))])
        try:
            b = assembly._b_matrix.transpose(0,2,1)
        except AttributeError:
            assembly.compute_elementary_operators()
            b = assembly._b_matrix.transpose(0,2,1)

        A = assembly.mesh.get_element_volumes()
        coef = A * sum([(b[i] ** 2).sum(axis = 1) for i in range(ndim)]) 
        # M = E*(1-nu)/((1+nu)*(1-2*nu))  # = K+4/3 * mu = p wave modulus
        DiffOp = DiffOp * ((0.5 * self.stiffness_coef) * coef)

        # if self.space._dimension == "2Daxi":
        #     # not sur it works
        #     DiffOp = DiffOp * ((2 * np.pi) * rr)

        return DiffOp

    def initialize(self, assembly, pb):
        """Initialize the weakform at the begining of a problem."""
        super().initialize(assembly, pb)
        self._initialize_nlgeom(assembly, pb)
        # if assembly.elm_type in ["hex8sri", "quad4sri"]:

    def update(self, assembly, pb):
        """Update the weakform to the current state.

        This method is applyed before the update of constutive law (stress and
        stiffness matrix).
        """
        if assembly._nlgeom == "UL":
            # if updated lagragian method
            # -> update the mesh and recompute elementary op
            assembly.set_disp(pb.get_disp())


def StressEquilibriumRI(
    constitutivelaw,
    hourglass_stiffness=1e4,
    name="",
    nlgeom=None,
    nlgeom_hourglass=False,
    space=None,
):
    wf = StressEquilibrium(constitutivelaw, nlgeom=nlgeom, space=space)
    # use reduced intagration with quad4 or hex8
    wf.assembly_options["n_elm_gp", "quad4"] = 1
    wf.assembly_options["n_elm_gp", "hex8"] = 1

    wf = WeakFormSum(
        [
            wf,
            HourglassStiffness(
                hourglass_stiffness, nlgeom=nlgeom_hourglass, space=space
            ),
        ],
        name=name,
    )
    return wf

























# # TEST TEST TEST 


# class HourglassStiffness2(WeakFormBase):
#     # """Weak formulation of the mechanical equilibrium equation for solids.

#     # This method is still experimental ! Use with caution. The fbar method
#     # can be used with the the standard StressEquilibrium weak form.

#     # The main point to consider are:
#     #   * This weak form is the same as the standard StressEquilibrium but
#     #     allow to use the consistant tangent matrix with the Fbar method.
#     #   * This weak form can be used for solid in 3D or using a 2D plane
#     #     assumption (plane strain or plane stress).
#     #   * Include initial stress for non linear problems or if defined in
#     #     the associated assembly.
#     #   * This weak form accepts geometrical non linearities if simcoon is
#     #     installed. (nlgeom should be in {True, 'UL', 'TL'}. In this case
#     #     the initial displacement is also
#     #     considered.

#     # Parameters
#     # ----------
#     # constitutivelaw: ConstitutiveLaw name (str) or ConstitutiveLaw object
#     #     Material Constitutive Law (:mod:`fedoo.constitutivelaw`)
#     # name: str
#     #     name of the WeakForm
#     # nlgeom: bool, 'UL' or 'TL', optional
#     #     If True, the geometrical non linearities are activate based on the
#     #     updated lagrangian method. This parameters is used only in the
#     #     context of NonLinearProblems such as
#     #     :mod:`fedoo.problem.NonLinearStatic` or
#     #     :mod:`fedoo.problem.NonLinearNewmark`.
#     #     If nlgeom == 'UL' the updated lagrangian method is used (same as True).
#     #     If nlgeom == 'TL' the total lagrangian method is used.
#     #     If not defined, the problem.nlgeom parameter is used instead.
#     # space: ModelingSpace
#     #     Modeling space associated to the weakform. If None is specified,
#     #     the active ModelingSpace is considered.
#     # """

#     def __init__(self, stiffness_coef, constitutivelaw = None, name="", nlgeom=False, space=None):
#         WeakFormBase.__init__(self, name, space)
#         self.assembly_options["n_elm_gp"] = 1
#         self.assembly_options["elm_type", "quad4"] = "quad4hourglass"
#         self.assembly_options["elm_type", "hex8"] = "hex8hourglass"
#         self.stiffness_coef = stiffness_coef
#         self.nlgeom = nlgeom
#         """Method used to treat the geometric non linearities.
#             * Set to False if geometric non linarities are ignored.
#             * Set to True or 'UL' to use the updated lagrangian method
#               (update the mesh)
#             * Set to 'TL' to use the total lagrangian method (base on the
#               initial mesh with initial displacement effet)
#         """

#     def get_weak_equation(self, assembly, pb):
#         """Get the weak equation related to the current problem state."""
#         op_dq = assembly.space.op_disp()
#         ndim = len(op_dq)

#         if np.array_equal(pb.get_dof_solution(), 0):
#             q0 = [0,0,0]
#         else:
#             q0 = [assembly.get_gp_results(op_dq[i], pb.get_dof_solution()) for i in range(ndim)]

#         DiffOp = sum([op_dq[i].virtual * (op_dq[i]+q0[i]) for i in range(len(op_dq))])
#         # DiffOp = sum([op_du[i].virtual * op_du[i] for i in range(len(op_du))])

#         # du_dx = assembly._get_assembled_operator(assembly.space.op_grad_u()[0][0], 1)
#         # du_dx = assembly._get_elementary_operator(assembly.space.op_grad_u()[0][0].op[0], 1)
#         # du_dy = assembly._get_elementary_operator(assembly.space.op_grad_u()[1][1].op[0], 1)
#         # sum([np.array((du_dx[0].multiply(du_dx[0])).sum(axis=1))[:,0] 
#         try:
#             b = assembly._b_matrix.transpose(0,2,1)
#         except AttributeError:
#             assembly.compute_elementary_operators()
#             b = assembly._b_matrix.transpose(0,2,1)

#         A = assembly.mesh.get_element_volumes()
#         coef = A * sum([(b[i] ** 2).sum(axis = 1) for i in range(ndim)]) 
#         # M = E*(1-nu)/((1+nu)*(1-2*nu))  # = K+4/3 * mu = p wave modulus
#         DiffOp = DiffOp * ((0.5 * self.stiffness_coef) * coef)

#         # if self.space._dimension == "2Daxi":
#         #     # not sur it works
#         #     DiffOp = DiffOp * ((2 * np.pi) * rr)

#         return DiffOp

#     def initialize(self, assembly, pb):
#         """Initialize the weakform at the begining of a problem."""
#         super().initialize(assembly, pb)
#         self._initialize_nlgeom(assembly, pb)
#         # if assembly.elm_type in ["hex8sri", "quad4sri"]:

#     def update(self, assembly, pb):
#         """Update the weakform to the current state.

#         This method is applyed before the update of constutive law (stress and
#         stiffness matrix).
#         """
#         if assembly._nlgeom == "UL":
#             # if updated lagragian method
#             # -> update the mesh and recompute elementary op
#             assembly.set_disp(pb.get_disp())