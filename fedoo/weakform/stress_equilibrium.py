"""The Strain equilibrium weak form from the fedoo finite element code."""

from fedoo.core.weakform import WeakFormBase
from fedoo.core.base import ConstitutiveLaw
from fedoo.util.voigt_tensors import StressTensorList, StrainTensorList

try:
    from simcoon import simmit as sim

    USE_SIMCOON = True
except ModuleNotFoundError:
    USE_SIMCOON = False

import numpy as np


class StressEquilibrium(WeakFormBase):
    """Weak formulation of the mechanical equilibrium equation for solids.

    The main point to consider are:
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

        self.nlgeom = nlgeom
        """Method used to treat the geometric non linearities.
            * Set to False if geometric non linarities are ignored.
            * Set to True or 'UL' to use the updated lagrangian method
              (update the mesh)
            * Set to 'TL' to use the total lagrangian method (base on the
              initial mesh with initial displacement effet)
        """

        self.corate = "log"
        # 'log': logarithmic strain, 'jaumann': jaumann strain,
        # or 'green_naghdi', 'gn', 'log_inc'...

        self.assembly_options["assume_sym"] = True
        # internalForce weak form should be symmetric
        # (if TangentMatrix is symmetric)
        # -> need to be checked for general case

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

        if initial_stress is not 0:
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
        # TO DO: change stress initialization to remove initial stress
        # term in the global vector assembly

        # initialize nlgeom value in assembly._nlgeom
        self._initialize_nlgeom(assembly, pb)

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

        if self.space._dimension == "2Daxi":
            assembly.sv["_R_gausspoints"] = assembly.mesh.convert_data(
                assembly.mesh.nodes[:, 0],
                "Node",
                "GaussPoint",
                n_elm_gp=assembly.n_elm_gp,
            )

        if assembly._nlgeom:
            if not (USE_SIMCOON):
                raise ModuleNotFoundError(
                    "Simcoon library need to be installed to deal with \
                     geometric non linearities (nlgeom = True)"
                )
            if assembly._nlgeom == "TL":
                assembly.sv["PK2"] = 0
                if self.space._dimension == "2Daxi":
                    raise NotImplementedError(
                        "'2Daxi' ModelingSpace is not implemented with \
                         total lagrangian formulation. Use update \
                         lagrangian instead."
                    )

            # initialize non linear operator for strain
            # don't improve the convergence, but kept in case it may be usefull
            # later.
            # self._init_nl_strain_op_vir()

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

        if displacement is 0:
            assembly.sv["DispGradient"] = 0
            assembly.sv["Stress"] = 0
            assembly.sv["Strain"] = 0
        else:
            grad_values = assembly.get_grad_disp(displacement, "GaussPoint")
            if self.space._dimension == "2Daxi":
                # mesh = assembly.current.mesh
                # eps_tt = np.divide(
                #     pb.get_disp()[0],
                #     mesh.nodes[:, 0],
                #     out=np.zeros(mesh.n_nodes),
                #     where=mesh.nodes[:, 0] != 0,
                # )  # put zero if X==0 (division by 0)
                # grad_values[2][2] = mesh.convert_data(
                #     eps_tt, "Node", "GaussPoint"
                # )
                mesh = assembly.current.mesh
                rr = mesh.convert_data(
                    mesh.nodes[:, 0],
                    "Node",
                    "GaussPoint",
                    n_elm_gp=assembly.n_elm_gp,
                )

                assembly.sv["_R_gausspoints"] = rr
                # grad_values[2][2] = mesh.convert_data(
                #     pb.get_disp()[0]/mesh.nodes[:, 0], 'Node')
                grad_values[2][2] = np.divide(
                    mesh.convert_data(
                        pb.get_disp()[0],
                        "Node",
                        "GaussPoint",
                        n_elm_gp=assembly.n_elm_gp,
                    ),
                    rr,
                    out=np.zeros_like(rr),
                    where=rr != 0,
                )  # put zero if X==0 (division by 0)

            assembly.sv["DispGradient"] = grad_values

            # Compute the strain required for the constitutive law.
            if assembly._nlgeom:
                self._corate_func(self, assembly, pb)
            else:
                _comp_linear_strain(self, assembly, pb)

    def update_2(self, assembly, pb):
        """Update the weakform to the current state.

        This method is applyed after the constutive law update (stress and
        stiffness matrix).
        """
        if assembly._nlgeom == "TL":
            assembly.sv["PK2"] = assembly.sv["Stress"].cauchy_to_pk2(assembly.sv["F"])
            if len(assembly.sv["TangentMatrix"].shape) == 2:
                if len(assembly.sv["F"].shape) == 3:
                    assembly.sv["TangentMatrix"] = assembly.sv["TangentMatrix"].reshape(
                        6, 6, -1
                    ) * np.ones((1, 1, assembly.sv["F"].shape[2]))

            assembly.sv["TangentMatrix"] = sim.Lt_convert(
                assembly.sv["TangentMatrix"],
                assembly.sv["F"],
                assembly.sv["Stress"].asarray(),
                self._convert_Lt_tag,
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
            if "DStrain" in assembly.sv:
                # rotate strain and stress -> need to be checked
                assembly.sv["Strain"] = StrainTensorList(
                    sim.rotate_strain_R(
                        assembly.sv_start["Strain"].asarray(),
                        assembly.sv["DR"],
                    )
                    + assembly.sv["DStrain"]
                )
                assembly.sv["DStrain"] = StrainTensorList(
                    np.zeros((6, assembly.n_gauss_points), order="F")
                )
            # or assembly.sv['DStrain'] = 0 perhaps more efficient to avoid a
            # nul sum

            # update cauchy stress
            if (
                assembly.sv["DispGradient"] is not 0
            ):  # True when the problem have been updated once
                stress = assembly.sv["Stress"].asarray()
                assembly.sv["Stress"] = StressTensorList(
                    sim.rotate_stress_R(stress, assembly.sv["DR"])
                )
                if assembly._nlgeom == "TL":
                    assembly.sv["PK2"] = assembly.sv["Stress"].cauchy_to_pk2(
                        assembly.sv["F"]
                    )

    # def _init_nl_strain_op_vir(self):
    #     # initialize non linear operator for strain
    #     # don't improve the convergence, but kept in case it may be usefull
    #     # later.

    #     op_grad_du = self.space.op_grad_u()
    #     # grad of displacement increment in incremental problems

    #     if self.space.ndim == "3D":
    #         # using voigt notation and with a 2 factor on non diagonal terms:
    #         # nl_strain_op_vir =
    #         #      0.5*(vir(duk/dxi) * duk/dxj + duk/dxi * vir(duk/dxj))
    #         nl_strain_op_vir = [
    #             sum([op_grad_du[k][i].virtual * op_grad_du[k][i]
    #                  for k in range(3)])
    #             for i in range(3)
    #         ]
    #         nl_strain_op_vir += [
    #             sum([op_grad_du[k][0].virtual * op_grad_du[k][1]
    #                  + op_grad_du[k][1].virtual * op_grad_du[k][0]
    #                  for k in range(3)])
    #         ]
    #         nl_strain_op_vir += [
    #             sum([op_grad_du[k][0].virtual * op_grad_du[k][2]
    #                  + op_grad_du[k][2].virtual * op_grad_du[k][0]
    #                  for k in range(3)])
    #         ]
    #         nl_strain_op_vir += [
    #             sum([op_grad_du[k][1].virtual * op_grad_du[k][2]
    #                  + op_grad_du[k][2].virtual * op_grad_du[k][1]
    #                  for k in range(3)])
    #         ]
    #     else:
    #         nl_strain_op_vir = [
    #             sum([op_grad_du[k][i].virtual * op_grad_du[k][i]
    #                  for k in range(2)])
    #             for i in range(2)
    #         ] + [0]
    #         nl_strain_op_vir += [
    #             sum([op_grad_du[k][0].virtual * op_grad_du[k][1]
    #                  + op_grad_du[k][1].virtual * op_grad_du[k][0]
    #                  for k in range(2)])
    #         ] + [0, 0]

    #     self._nl_strain_op_vir = nl_strain_op_vir

    @property
    def corate(self):
        """Corotational strain mesure for strain.

        Properties defining the way strain is treated in finite strain problem
        (using a weakform with nlgeom = True)
        corate can take the following str values:
            * "log" (default): exact logarithmic strain (strain is recomputed
              at each iteration)
            * "jaumann": Strain using the Jaumann derivative (strain is
              incremented)
            * "green_nagdhi" or "gn": Strain using the Green_Nagdhi derivative
              (strain is incremented)
        if nlgeom is False, this property has no effect.
        """
        return self._corate

    @corate.setter
    def corate(self, value):
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
            self._convert_Lt_tag = "DsigmaDe_JaumannDD_2_DSDE"
        elif value == "log_r":
            self._corate_func = _comp_log_strain_R
            self._convert_Lt_tag = "DsigmaDe_2_DSDE"
        elif value == "log_r_inc":
            self._corate_func = _comp_log_strain_R_inc
            self._convert_Lt_tag = "DsigmaDe_2_DSDE"
        else:
            raise NameError(
                'corate value not understood. Choose between "log", "log_R", \
                "green_naghdi" or "jaumann"'
            )
        self._corate = value


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
    grad_values = assembly.sv["DispGradient"]
    eye_3 = np.empty((3, 3, 1), order="F")
    eye_3[:, :, 0] = np.eye(3)
    F1 = np.add(eye_3, grad_values)
    assembly.sv["F"] = F1
    if "F" not in assembly.sv_start:
        F0 = np.empty_like(F1)
        F0[...] = eye_3
        assembly.sv_start["F"] = F0

    (D, DR, Omega) = sim.objective_rate(
        "log", assembly.sv_start["F"], F1, pb.dtime, False
    )
    assembly.sv["DR"] = DR
    assembly.sv["Strain"] = StrainTensorList(sim.Log_strain(F1, True, False))


def _comp_log_strain_inc(wf, assembly, pb):
    grad_values = assembly.sv["DispGradient"]
    eye_3 = np.empty((3, 3, 1), order="F")
    eye_3[:, :, 0] = np.eye(3)
    F1 = np.add(eye_3, grad_values)
    assembly.sv["F"] = F1
    if "F" not in assembly.sv_start:
        F0 = np.empty_like(F1)
        F0[...] = eye_3
        assembly.sv_start["F"] = F0

    (DStrain, D, DR, Omega) = sim.objective_rate(
        "log", assembly.sv_start["F"], F1, pb.dtime, True
    )
    assembly.sv["DR"] = DR
    assembly.sv["DStrain"] = StrainTensorList(DStrain)


def _comp_log_strain_R(wf, assembly, pb):
    grad_values = assembly.sv["DispGradient"]
    eye_3 = np.empty((3, 3, 1), order="F")
    eye_3[:, :, 0] = np.eye(3)
    F1 = np.add(eye_3, grad_values)
    assembly.sv["F"] = F1
    if "F" not in assembly.sv_start:
        F0 = np.empty_like(F1)
        F0[...] = eye_3
        assembly.sv_start["F"] = F0

    (D, DR, Omega) = sim.objective_rate(
        "log_R", assembly.sv_start["F"], F1, pb.dtime, False
    )
    assembly.sv["DR"] = DR
    assembly.sv["Strain"] = StrainTensorList(sim.Log_strain(F1, True, False))


def _comp_log_strain_R_inc(wf, assembly, pb):
    grad_values = assembly.sv["DispGradient"]
    eye_3 = np.empty((3, 3, 1), order="F")
    eye_3[:, :, 0] = np.eye(3)
    F1 = np.add(eye_3, grad_values)
    assembly.sv["F"] = F1
    if "F" not in assembly.sv_start:
        F0 = np.empty_like(F1)
        F0[...] = eye_3
        assembly.sv_start["F"] = F0

    (DStrain, D, DR, Omega) = sim.objective_rate(
        "log_R", assembly.sv_start["F"], F1, pb.dtime, True
    )
    assembly.sv["DR"] = DR
    assembly.sv["DStrain"] = StrainTensorList(DStrain)


def _comp_jaumann_strain(wf, assembly, pb):
    grad_values = assembly.sv["DispGradient"]
    eye_3 = np.empty((3, 3, 1), order="F")
    eye_3[:, :, 0] = np.eye(3)
    F1 = np.add(eye_3, grad_values)
    assembly.sv["F"] = F1
    if "F" not in assembly.sv_start:
        F0 = np.empty_like(F1)
        F0[...] = eye_3
        assembly.sv_start["F"] = F0

    (DStrain, D, DR, Omega) = sim.objective_rate(
        "jaumann", assembly.sv_start["F"], F1, pb.dtime, True
    )
    assembly.sv["DR"] = DR
    assembly.sv["DStrain"] = StrainTensorList(DStrain)


def _comp_gn_strain(wf, assembly, pb):
    # green_naghdi corate
    grad_values = assembly.sv["DispGradient"]
    eye_3 = np.empty((3, 3, 1), order="F")
    eye_3[:, :, 0] = np.eye(3)
    F1 = np.add(eye_3, grad_values)
    assembly.sv["F"] = F1
    if "F" not in assembly.sv_start:
        F0 = np.empty_like(F1)
        F0[...] = eye_3
        assembly.sv_start["F"] = F0

    (DStrain, D, DR, Omega) = sim.objective_rate(
        "green_naghdi", assembly.sv_start["F"], F1, pb.dtime, True
    )
    assembly.sv["DR"] = DR
    assembly.sv["DStrain"] = StrainTensorList(DStrain)


def _comp_linear_strain_pgd(wf, assembly, pb):
    # may be compatible with other methods like PGD
    # but not compatible with simcoon
    assert not (wf.nlgeom), "the current strain measure isn't adapted for finite strain"
    grad_values = assembly.sv["DispGradient"]

    strain = [grad_values[i][i] for i in range(3)]
    strain += [
        grad_values[0][1] + grad_values[1][0],
        grad_values[0][2] + grad_values[2][0],
        grad_values[1][2] + grad_values[2][1],
    ]
    assembly.sv["Strain"] = StrainTensorList(strain)


def _comp_gl_strain(wf, assembly, pb):
    # not compatible with simcoon
    if not (wf.nlgeom):
        return _comp_linear_strain_pgd(wf, assembly, pb)
    else:
        grad_values = assembly.sv["DispGradient"]
        # GL strain tensor
        # possibility to be improve from simcoon functions
        # to get the logarithmic strain tensor...
        strain = [
            grad_values[i][i] + 0.5 * sum([grad_values[k][i] ** 2 for k in range(3)])
            for i in range(3)
        ]
        strain += [
            grad_values[0][1]
            + grad_values[1][0]
            + sum([grad_values[k][0] * grad_values[k][1] for k in range(3)])
        ]
        strain += [
            grad_values[0][2]
            + grad_values[2][0]
            + sum([grad_values[k][0] * grad_values[k][2] for k in range(3)])
        ]
        strain += [
            grad_values[1][2]
            + grad_values[2][1]
            + sum([grad_values[k][1] * grad_values[k][2] for k in range(3)])
        ]
        return StrainTensorList(strain)
