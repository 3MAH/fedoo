"""The Strain equilibrium weak form from the fedoo finite element code."""

from fedoo.core.weakform import WeakFormBase
from fedoo.core.base import ConstitutiveLaw
from fedoo.core.time_evolution import SECOND_ORDER
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
      * For nearly incompressible material, the F-bar method should be used
        by setting the fbar attribute to True.
      * For problems involving geometrical instabilities, the geometrical
        stiffness should be used by setting the geometric_stiffness attribute
        to True.

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

        self.corate = "log"
        # 'log': logarithmic strain, 'jaumann': jaumann strain,
        # or 'green_naghdi', 'gn', 'log_inc'...

        self.fbar = False  # by default, the fbar stabilization is not used
        self.geometric_stiffness = False

        self.assembly_options["assume_sym"] = True
        # internalForce weak form should be symmetric
        # (if TangentMatrix is symmetric)
        # -> need to be checked for general case

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

        if self.space.is_axisymmetric:
            DiffOp = DiffOp * ((2 * np.pi) * rr)

        return DiffOp

    def initialize(self, assembly, pb):
        """Initialize the weakform at the begining of a problem."""
        # TO DO: change stress initialization to remove initial stress
        # term in the global vector assembly

        # initialize nlgeom value in assembly._nlgeom
        self._initialize_nlgeom(assembly, pb)
        self.nlgeom = assembly._nlgeom
        self.corate = self._corate  # to force the setter function

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
                if self.fbar:
                    _comp_grad_disp_fbar(assembly, displacement)
                else:
                    _comp_grad_disp(assembly, displacement)
                _comp_linear_strain(self, assembly, pb)

    def update_2(self, assembly, pb):
        """Update the weakform to the current state.

        This method is applyed after the constutive law update (stress and
        stiffness matrix).
        """
        if assembly._nlgeom == "TL" or (
            assembly._nlgeom == "UL" and self.constitutivelaw._Lt_from_F
        ):
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

            if assembly._nlgeom == "TL":
                assembly.sv["PK2"] = assembly.sv["Stress"].cauchy_to_pk2(
                    assembly.sv["F"]
                )

                assembly.sv["TangentMatrix"] = sim.Lt_convert(
                    assembly.sv["TangentMatrix"],
                    assembly.sv["F"],
                    assembly.sv["Stress"].asarray(),
                    self._convert_Lt_tag,
                )

            else:  # _Lt_from_F = True and assembly._nlgeom == "UL"
                # need to convert Lt if Lt is defined related to F instead
                # of log strain
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

        if self.space.ndim == "3D":
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
    def fbar(self):
        """Set to True to use the F-bar method.

        The F-bar method should be used to stabilized constitutive laws with
        nearly incompressible behavior.
        """
        return self._fbar

    @fbar.setter
    def fbar(self, value):
        if not isinstance(value, bool):
            raise TypeError("bool expeted for fbar")
        self._fbar = value
        if value:
            self._comp_F = _comp_Fbar
        else:
            self._comp_F = _comp_F

    @property
    def geometric_stiffness(self):
        """Set to True to add the geometric effects to the stiffness matrix.

        The use of a geometric stiffness matrix usually don't improve the
        convergence and may even require smaller time step.
        However, geometric_stiffness should be included to reach convergence
        when the problem involve geometrical instabilities like buckling or
        when using very large strain (with hyperelastic materials for
        instance).
        """
        return self._geometric_stiffness

    @geometric_stiffness.setter
    def geometric_stiffness(self, value):
        if not isinstance(value, bool):
            raise TypeError("bool expeted for geometric_stiffness")
        self._geometric_stiffness = value
        if value:
            self._init_nl_strain_op_vir()

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
        self._corate = value
        if self.nlgeom == "UL":
            value = value.lower()
            if value == "log":
                self._corate_func = _comp_log_strain
                self._convert_Lt_tag = "Dsigma_LieDD_Dsigma_logarithmicDD"
            elif value == "log_inc":
                self._corate_func = _comp_log_strain_inc
                self._convert_Lt_tag = "Dsigma_LieDD_Dsigma_logarithmicDD"
            elif value in ["gn", "green_naghdi"]:
                self._corate_func = _comp_gn_strain
                self._convert_Lt_tag = "Dsigma_LieDD_Dsigma_GreenNaghdiDD"
            elif value == "jaumann":
                self._corate_func = _comp_jaumann_strain
                self._convert_Lt_tag = "Dsigma_LieDD_Dsigma_JaumannDD"
            elif value == "log_r":
                self._corate_func = _comp_log_strain_R
                self._convert_Lt_tag = "Dsigma_LieDD_Dsigma_logarithmicDD"
            elif value == "log_r_inc":
                self._corate_func = _comp_log_strain_R_inc
                self._convert_Lt_tag = "Dsigma_LieDD_Dsigma_logarithmicDD"
            else:
                raise ValueError(
                    'corate value not understood. Choose between "log", "log_R", \
                    "green_naghdi" or "jaumann"'
                )

        if self.nlgeom == "TL":
            value = value.lower()
            if value == "log":
                self._corate_func = _comp_log_strain
                self._convert_Lt_tag = "Dsigma_LieDD_2_DSDE"
            elif value == "log_inc":
                self._corate_func = _comp_log_strain_inc
                self._convert_Lt_tag = "Dsigma_LieDD_2_DSDE"
            elif value in ["gn", "green_naghdi"]:
                self._corate_func = _comp_gn_strain
                self._convert_Lt_tag = "Dsigma_LieDD_2_DSDE"
            elif value == "jaumann":
                self._corate_func = _comp_jaumann_strain
                self._convert_Lt_tag = "Dsigma_LieDD_2_DSDE"
            elif value == "log_r":
                self._corate_func = _comp_log_strain_R
                self._convert_Lt_tag = "Dsigma_LieDD_2_DSDE"
            elif value == "log_r_inc":
                self._corate_func = _comp_log_strain_R_inc
                self._convert_Lt_tag = "Dsigma_LieDD_2_DSDE"
            else:
                raise ValueError(
                    'corate value not understood. Choose between "log", "log_R", \
                    "green_naghdi" or "jaumann"'
                )


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


def _comp_grad_disp_fbar(assembly, displacement):
    # #small strain fbar. Only valid in small strain
    grad_values = np.array(_comp_grad_disp(assembly, displacement))
    # return grad_values
    dvol = np.trace(grad_values)

    dvol_center = np.mean(dvol.reshape(assembly.n_elm_gp, -1), axis=0)
    # grad_values = grad_values - (dvol.reshape(assembly.n_elm_gp,-1) - dvol_center).ravel()
    grad_values[[0, 1, 2], [0, 1, 2]] -= (
        1 / 3 * (dvol.reshape(assembly.n_elm_gp, -1) - dvol_center).ravel()
    )
    assembly.sv["DispGradient"] = grad_values
    return grad_values


# function to compute F tensor (required nl corate function used with simcoon)
def _comp_F(assembly, displacement):
    grad_values = _comp_grad_disp(assembly, displacement)

    eye_3 = np.empty((3, 3, 1), order="F")
    eye_3[:, :, 0] = np.eye(3)
    F1 = np.add(eye_3, grad_values, order="F")
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

    J = np.linalg.det(F1.transpose((2, 0, 1)))

    # grad_values_center = [
    #     [
    #         assembly.get_gp_results(op, displacement, n_elm_gp=1)
    #         if op != 0
    #         else np.zeros(assembly.mesh.n_elements)
    #         for op in line_op
    #     ] for line_op in assembly.space.op_grad_u()
    # ]
    # Jcenter = np.linalg.det(
    #     np.add(eye_3, grad_values_center).transpose((2,0,1))
    # )
    Jcenter = np.mean(J.reshape(assembly.n_elm_gp, -1), axis=0)
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
