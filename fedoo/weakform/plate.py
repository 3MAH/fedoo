from fedoo.core.weakform import WeakFormBase
from fedoo.core.base import ConstitutiveLaw
from scipy.spatial.transform import Rotation
import numpy as np


class PlateEquilibriumFI(WeakFormBase):  # plate weakform whith full integration.
    """Mechanical equilibrium equation for plate models, with full integration.

    This weakform implements the mechanical equilibrium for shell/plate elements.
    It uses full integration (FI) by default, which may lead to shear or
    membrane locking for linear elements (e.g., Tri3, Quad4).

    *:mod:`fedoo.weakform.PlateEquilibrium` should be prefered unless you know
    what you are doing.*

    Geometrical nonlinearities are implemented via a corotational approach,
    decomposing the motion into a rigid-body rotation and a local
    deformational part.

    Parameters
    ----------
    plate_properties : ConstitutiveLaw name (str) or ConstitutiveLaw object
        Shell Constitutive Law defining the membrane, bending, and shear
        stiffness (e.g., :mod:`fedoo.constitutivelaw.ShellHomogeneous` or
        :mod:`fedoo.constitutivelaw.ShellLaminate`).
    true_drilling_rotation : bool, default=True
        Only active if nlgeom is enabled.
        - If True: Enforces a kinematic link between the nodal RotZ and
          the in-plane displacement gradients (0.5 * [dv/dx - du/dy]).
          This allows for physically consistent in-plane material
          rotations, essential for large-deformation anisotropic analysis.
        - If False: Applies a simple numerical penalty on the nodal RotZ
          to prevent matrix singularity without coupling it to the
          membrane deformation.
    drill_stiffness_coefficient : float, default=1e-2
        The penalty coefficient used for the drilling rotation constraint scaled by the
        membrane shear stiffness.
        Increasing this value enforces the constraint more strictly but
        may introduce drilling locking in fully integrated elements.
    name : str, optional
        Name of the WeakForm.
    nlgeom : bool or str, optional
        Property used to treat geometric nonlinearities.
        If True, a step-by-step frame update is used.
    """

    def __init__(
        self,
        plate_properties,
        true_drilling_rotation=True,
        drill_stiffness_coefficient=1e-2,
        name="",
        nlgeom=None,
        space=None,
    ):
        if isinstance(plate_properties, str):
            plate_properties = ConstitutiveLaw.get_all()[plate_properties]

        if name == "":
            name = plate_properties.name

        WeakFormBase.__init__(self, name, space)

        assert (
            self.space.ndim == 3
        ), "No 2D model for a plate kinematic. Choose '3D' problem dimension."

        self.space.new_variable("DispX")
        self.space.new_variable("DispY")
        self.space.new_variable("DispZ")
        self.space.new_variable("RotX")  # torsion rotation
        self.space.new_variable("RotY")
        self.space.new_variable("RotZ")
        self.space.new_vector("Disp", ("DispX", "DispY", "DispZ"))
        self.space.new_vector("Rot", ("RotX", "RotY", "RotZ"))

        self.constitutivelaw = plate_properties

        # automatically set the right assembly element formulation
        self.assembly_options["elm_type", "tri3"] = "ptri3"
        self.assembly_options["elm_type", "quad4"] = "pquad4"
        self.assembly_options["elm_type", "tri6"] = "ptri6"
        self.assembly_options["elm_type", "quad8"] = "pquad8"
        self.assembly_options["elm_type", "quad9"] = "pquad9"

        self.nlgeom = nlgeom
        self.true_drilling_rotation = true_drilling_rotation
        self.drill_stiffness_coefficient = drill_stiffness_coefficient

    def initialize(self, assembly, pb):
        assembly.sv["ShellStrain"] = 0
        assembly.sv["ShellStress"] = 0
        assembly.sv["_DrillConstraint"] = 0

        self._initialize_nlgeom(assembly, pb)
        self.nlgeom = assembly._nlgeom

        if self.nlgeom:
            # =========================================================
            # Reference state initialization
            # =========================================================
            if "_InitialRigidRotationMat" not in assembly.sv:
                if assembly._element_local_frame is None:
                    init_frame = assembly.mesh.get_element_local_frame()
                else:
                    init_frame = assembly._element_local_frame
                assembly.sv["_InitialRigidRotationMat"] = init_frame

                assembly.sv["RigidRotationMat"] = init_frame
                assembly.current._element_local_frame = init_frame.reshape(
                    assembly.current.mesh.n_elements, -1, 3, 3
                )

                # dof rotation matrix at node (without elm initial rotation)
                assembly.sv["_NodesRotationMatrix"] = np.tile(
                    np.eye(3), (assembly.mesh.n_nodes, 1, 1)
                )

                nodes_pos_init = assembly.mesh.nodes[assembly.mesh.elements]
                init_center = nodes_pos_init.mean(axis=1)
                assembly.sv["_InitialNodeLocalPos"] = np.matmul(
                    init_frame,
                    (nodes_pos_init - init_center[:, np.newaxis, :]).transpose(0, 2, 1),
                ).transpose(0, 2, 1)

    def _compute_local_dof(self, assembly, pb):
        mesh = assembly.current.mesh
        nodes_pos = mesh.nodes[mesh.elements]
        current_center = nodes_pos.mean(axis=1)

        initial_frame = assembly.sv["_InitialRigidRotationMat"]
        initial_node_local_pos = assembly.sv["_InitialNodeLocalPos"]

        # =========================================================
        # 2. ELEMENT FRAME EXTRACTION
        # =========================================================
        # Incremental frame update
        delta_rotvec_nodes = pb._get_vect_component(pb.get_X(), "Rot")
        delta_rotvec = delta_rotvec_nodes.T[mesh.elements].mean(axis=1)

        rigid_rotmat_trial = assembly.sv["RigidRotationMat"] @ (
            Rotation.from_rotvec(delta_rotvec).as_matrix()
        )

        current_local_frame = mesh.get_element_local_frame()
        e3 = current_local_frame[:, 2]
        e1_trial = rigid_rotmat_trial[:, 0]

        if self.true_drilling_rotation:
            # Drill rotation (RotZ) consistant with inplane displacement
            # Project trial onto exact normal (Prevents 90-deg flip singularity)
            e1_prov = e1_trial - np.sum(e1_trial * e3, axis=1, keepdims=True) * e3
            e1_norm = np.linalg.norm(e1_prov, axis=1, keepdims=True)
            e1_prov = np.where(e1_norm > 1e-12, e1_prov / e1_norm, e1_trial)
            e2_prov = np.cross(e3, e1_prov)

            # Map initial and current relative positions to 2D
            X_loc = initial_node_local_pos[:, :, :2]  # Initial 2D shape

            rel_pos = nodes_pos - current_center[:, np.newaxis, :]
            x_loc_1 = np.sum(rel_pos * e1_prov[:, np.newaxis, :], axis=2)
            x_loc_2 = np.sum(rel_pos * e2_prov[:, np.newaxis, :], axis=2)
            x_loc = np.stack([x_loc_1, x_loc_2], axis=-1)  # Current 2D shape

            # SVD of Cross-Covariance
            C = np.einsum("eni,enj->eij", x_loc, X_loc)
            U, S, Vh = np.linalg.svd(C)
            R_2D = np.einsum("eij,ejk->eik", U, Vh)

            # Prevent reflections
            detR = np.linalg.det(R_2D)
            reflection_mask = detR < 0
            if np.any(reflection_mask):
                U_corr = U.copy()
                U_corr[reflection_mask, :, 1] *= -1
                R_2D = np.einsum("eij,ejk->eik", U_corr, Vh)

            # Apply exact in-plane correction to provisional axes
            e1 = R_2D[:, 0, 0:1] * e1_prov + R_2D[:, 1, 0:1] * e2_prov
            e2 = R_2D[:, 0, 1:2] * e1_prov + R_2D[:, 1, 1:2] * e2_prov
        else:
            # Inplane local frame not consistant with rotZ
            # Less computational cost but less accurate
            e1 = e1_trial - (e1_trial * e3).sum(axis=1).reshape(-1, 1) * e3
            e1_norm = np.linalg.norm(e1, axis=1).reshape(-1, 1)
            e1 = np.where(e1_norm > 1e-12, e1 / e1_norm, e1_trial)
            e2 = np.cross(e3, e1)

        rigid_rotmat = np.empty_like(rigid_rotmat_trial)
        rigid_rotmat[:, 0, :] = e1
        rigid_rotmat[:, 1, :] = e2
        rigid_rotmat[:, 2, :] = e3

        nodes_rotmat = (
            Rotation.from_rotvec(delta_rotvec_nodes.T).as_matrix()  # transpose(0,2,1)
            @ assembly.sv["_NodesRotationMatrix"]
        )

        R_global_nodes = nodes_rotmat[mesh.elements]

        # State Updates
        if not pb._line_search_update:
            assembly.sv["RigidRotationMat"] = rigid_rotmat
            assembly.sv["_NodesRotationMatrix"] = nodes_rotmat
            assembly.current._element_local_frame = rigid_rotmat.reshape(
                mesh.n_elements, -1, 3, 3
            )

            rot_var = self.space.get_rank_vector("Rot")
            n_slice = slice(rot_var[0] * mesh.n_nodes, (rot_var[0] + 3) * mesh.n_nodes)

            if np.isscalar(pb._U) and pb._U == 0:
                pb._dU[n_slice] = (
                    Rotation.from_matrix(nodes_rotmat).as_rotvec().T.ravel()
                )
            else:
                nodes_rotmat_from_U = Rotation.from_rotvec(
                    pb._U[n_slice].reshape(3, -1).T
                ).as_matrix()

                pb._dU[n_slice] = (
                    Rotation.from_matrix(
                        nodes_rotmat @ nodes_rotmat_from_U.transpose(0, 2, 1)
                    )
                    .as_rotvec()
                    .T.ravel()
                )

            for bc in pb.bc.list_all():
                if bc.bc_type == "Dirichlet" and bc.variable in rot_var:
                    pb._Xbc[bc._dof_index] = (
                        bc.get_true_value(pb.t_fact)
                        - pb.get_dof_solution()[bc._dof_index]
                    )

        # =========================================================
        # 3. DEFORMATIONAL KINEMATICS & LOCAL DOF ASSEMBLY
        # =========================================================
        rel_pos = nodes_pos - current_center[:, np.newaxis, :]
        u_loc = (
            np.matmul(rigid_rotmat, rel_pos.transpose(0, 2, 1)).transpose(0, 2, 1)
            - initial_node_local_pos
        )

        T_loc = np.matmul(
            rigid_rotmat[:, np.newaxis, :, :],
            np.matmul(
                R_global_nodes,
                initial_frame[:, np.newaxis, :, :].transpose(0, 1, 3, 2),
            ),
        )

        rot_loc = (
            Rotation.from_matrix(T_loc.reshape(-1, 3, 3))
            .as_rotvec()
            .reshape(mesh.n_elements, mesh.n_elm_nodes, 3)
        )

        n_dof = mesh.n_elm_nodes * mesh.n_elements
        dof_local = np.zeros(self.space.nvar * n_dof)

        v_disp = self.space.variable_rank("DispX")
        dof_local[v_disp * n_dof : (v_disp + 3) * n_dof] = u_loc.transpose(
            2, 1, 0
        ).ravel()

        v_rot = self.space.variable_rank("RotX")
        dof_local[v_rot * n_dof : (v_rot + 3) * n_dof] = rot_loc.transpose(
            2, 1, 0
        ).ravel()
        return dof_local

    def update(self, assembly, pb):
        if self.nlgeom == "UL":
            assembly.set_disp(pb.get_disp())

        if np.array_equal(pb.get_dof_solution(), 0):
            assembly.sv["MembraneStrain"] = assembly.sv["MembraneForce"] = 0
            assembly.sv["BendingStrain"] = assembly["BendingMoment"] = 0
            assembly.sv["TransverseShearStrain"] = assembly.sv["ShearForce"] = 0
            assembly.sv["_DrillStrain"] = assembly.sv["_DrillMoment"] = 0
            # assembly.sv["ShellStrain"] = assembly.sv["ShellStress"] = 0
            return

        op_plate_strain = self.generalized_strain_operator()
        op_drill_constraint = self.drill_constraint_operator()

        H = self.constitutivelaw.get_shell_stiffness_matrix()
        assembly.sv["_ShellStiffnessMatrix"] = H

        if self.nlgeom:
            use_local_dof = True
            dof = self._compute_local_dof(assembly, pb)
        else:
            # Small displacement (Linear) update
            use_local_dof = False
            dof = pb.get_dof_solution()

        # =========================================================
        # 4. EVALUATE STRAINS
        # =========================================================
        assembly.sv["ShellStrain"] = _ShellComponentList(
            [
                (
                    assembly.current.get_gp_results(
                        op, dof, use_local_dof=use_local_dof
                    )
                    if op != 0
                    else 0
                )
                for op in op_plate_strain
            ]
        )

        assembly.sv["_DrillConstraint"] = assembly.current.get_gp_results(
            op_drill_constraint, dof, use_local_dof=use_local_dof
        )

        # Evaluate Stresses (Shared linear/nonlinear)
        assembly.sv["ShellStress"] = _ShellComponentList(
            [
                sum(
                    [
                        (
                            assembly.sv["ShellStrain"][j] * H[i][j]
                            if not np.array_equal(assembly.sv["ShellStrain"][j], 0)
                            else 0
                        )
                        for j in range(8)
                    ]
                )
                for i in range(8)
            ]
        )

    def to_start(self, assembly, pb):
        if self.nlgeom == "UL":
            assembly.set_disp(pb.get_disp())
            if "RigidRotationMat" in assembly.sv_start:
                assembly.current._element_local_frame = assembly.sv_start[
                    "RigidRotationMat"
                ].reshape(
                    assembly.current.mesh.n_elements,
                    -1,
                    3,
                    3,
                )
            else:
                assembly.current._element_local_frame = None

    def generalized_strain_operator(self):
        # membrane strain
        EpsX = self.space.derivative("DispX", "X")
        EpsY = self.space.derivative("DispY", "Y")
        GammaXY = self.space.derivative("DispX", "Y") + self.space.derivative(
            "DispY", "X"
        )

        # bending curvature
        XsiX = -self.space.derivative(
            "RotY", "X"
        )  # flexion autour de Y -> courbure suivant x
        XsiY = self.space.derivative(
            "RotX", "Y"
        )  # flexion autour de X -> courbure suivant y
        XsiXY = self.space.derivative("RotX", "X") - self.space.derivative("RotY", "Y")

        # shear
        GammaXZ = self.space.derivative("DispZ", "X") + self.space.variable("RotY")
        GammaYZ = self.space.derivative("DispZ", "Y") - self.space.variable("RotX")

        return [EpsX, EpsY, GammaXY, XsiX, XsiY, XsiXY, GammaXZ, GammaYZ]

    def drill_constraint_operator(self):
        # Drilling strain (measure of the difference between nodal RotZ
        # and the actual rotation of the element plane)
        if self.true_drilling_rotation:
            return self.space.variable("RotZ") - 0.5 * (
                self.space.derivative("DispY", "X")
                - self.space.derivative("DispX", "Y")
            )
        else:
            return self.space.variable("RotZ")

    def get_weak_equation(self, assembly, pb):
        if "_ShellStiffnessMatrix" not in assembly.sv:
            assembly.sv["_ShellStiffnessMatrix"] = (
                self.constitutivelaw.get_shell_stiffness_matrix()
            )
        H = assembly.sv["_ShellStiffnessMatrix"]
        op_plate_strain = self.generalized_strain_operator()
        op_drill_constraint = self.drill_constraint_operator()
        initial_stress = assembly.sv["ShellStress"]

        diffop = 0
        if not (np.array_equal(initial_stress, 0)):
            diffop = sum(
                [
                    (
                        op_plate_strain[i].virtual * initial_stress[i]
                        if op_plate_strain[i] != 0
                        else 0
                    )
                    for i in range(8)
                ]
            )

            # Geometrical stiffness (Local String Effect)
            if assembly._nlgeom:
                Nx = initial_stress[0]
                Ny = initial_stress[1]
                Nxy = initial_stress[2]
                dw_dx = self.space.derivative("DispZ", "X")
                dw_dy = self.space.derivative("DispZ", "Y")
                diffop += (
                    dw_dx.virtual * dw_dx * Nx
                    + dw_dy.virtual * dw_dy * Ny
                    + dw_dx.virtual * dw_dy * Nxy
                    + dw_dy.virtual * dw_dx * Nxy
                )

        # Tangent stiffness
        diffop += sum(
            [
                sum(
                    [
                        (
                            0
                            if (op_plate_strain[j] == 0 or op_plate_strain[i] == 0)
                            else op_plate_strain[i].virtual
                            * op_plate_strain[j]
                            * H[i][j]
                        )
                        for j in range(8)
                    ]
                )
                for i in range(8)
            ]
        )

        if self.drill_stiffness_coefficient != 0:
            # penalty for RotZ (drilling DOF stabilization)
            representative_stiffness = H[2][2]
            penalty = representative_stiffness * self.drill_stiffness_coefficient

            diffop += (
                op_drill_constraint.virtual
                * (op_drill_constraint + assembly.sv["_DrillConstraint"])
                * penalty
            )

        return diffop


class PlateEquilibrium(
    PlateEquilibriumFI
):  # weak form of plate shear energy containing only the shear strain energy
    """Mechanical equilibrium equation for plate models.

    The shear terms are treated with a full or reduced integration depending on
    the order of the element interpolation (reduced integration for linear
    element or full integration for quadratic element).
    This weak form has to be used in combination with a Shell Constitutive Law
    like :mod:`fedoo.constitutivelaw.ShellHomogeneous` or
    :mod:`fedoo.constitutivelaw.ShellLaminate`.
    Geometrical non linearities are implemented with a corotational approach.

    Parameters
    ----------
    plate_properties : ConstitutiveLaw name (str) or ConstitutiveLaw object
        Shell Constitutive Law defining the membrane, bending, and shear
        stiffness (e.g., :mod:`fedoo.constitutivelaw.ShellHomogeneous` or
        :mod:`fedoo.constitutivelaw.ShellLaminate`).
    true_drilling_rotation : bool, default=True
        Only active if nlgeom is enabled.
        - If True: Enforces a kinematic link between the nodal RotZ and
          the in-plane displacement gradients (0.5 * [dv/dx - du/dy]).
          This allows for physically consistent in-plane material
          rotations, essential for large-deformation anisotropic analysis.
        - If False: Applies a simple numerical penalty on the nodal RotZ
          to prevent matrix singularity without coupling it to the
          membrane deformation.
    drill_stiffness_coefficient : float, default=1e-2
        The penalty coefficient used for the drilling rotation constraint.
        Typically scaled by the membrane shear stiffness (e.g., 1e-2).
        Increasing this value enforces the constraint more strictly but
        may introduce drilling locking in fully integrated elements.
    name : str, optional
        Name of the WeakForm.
    nlgeom : bool or str, optional
        Property used to treat geometric nonlinearities.
        If True, a step-by-step frame update is used.
    """

    def __init__(
        self,
        plate_properties,
        true_drilling_rotation=True,
        drill_stiffness_coefficient=1e-2,
        name="",
        nlgeom=None,
        space=None,
    ):
        super().__init__(
            plate_properties,
            true_drilling_rotation,
            drill_stiffness_coefficient,
            name,
            nlgeom,
            space,
        )
        # alias with "_" prefix may be used for reduced integration.
        self.space.variable_alias("_DispX", "DispX")
        self.space.variable_alias("_DispY", "DispY")
        self.space.variable_alias("_DispZ", "DispZ")
        self.space.variable_alias("_RotX", "RotX")
        self.space.variable_alias("_RotY", "RotY")
        self.space.variable_alias("_RotZ", "RotZ")

        self.assembly_options["elm_type", "tri3"] = "ptri3sri"
        self.assembly_options["elm_type", "quad4"] = "pquad4sri"
        # self.assembly_options["elm_type", "tri6"] = "ptri6sri"
        # self.assembly_options["elm_type", "quad8"] = "pquad8ri"
        # self.assembly_options["elm_type", "quad9"] = "pquad9sri"

    def generalized_strain_operator(self):
        # membrane strain
        EpsX = self.space.derivative("DispX", "X")
        EpsY = self.space.derivative("DispY", "Y")
        GammaXY = self.space.derivative("DispX", "Y") + self.space.derivative(
            "DispY", "X"
        )

        # bending curvature
        XsiX = -self.space.derivative(
            "RotY", "X"
        )  # flexion autour de Y -> courbure suivant x
        XsiY = self.space.derivative(
            "RotX", "Y"
        )  # flexion autour de X -> courbure suivant y
        XsiXY = self.space.derivative("RotX", "X") - self.space.derivative("RotY", "Y")

        # shear
        GammaXZ = self.space.derivative("_DispZ", "X") + self.space.variable("_RotY")
        GammaYZ = self.space.derivative("_DispZ", "Y") - self.space.variable("_RotX")

        return [EpsX, EpsY, GammaXY, XsiX, XsiY, XsiXY, GammaXZ, GammaYZ]

    def drill_constraint_operator(self):
        if self.true_drilling_rotation:
            # Drilling strain (measure of the difference between nodal RotZ
            # and the actual rotation of the element plane)
            # Risk of locking -> use reduced integration
            return self.space.variable("_RotZ") - 0.5 * (
                self.space.derivative("_DispY", "X")
                - self.space.derivative("_DispX", "Y")
            )
        else:
            # Risk of singularity -> use full integration
            return self.space.variable("RotZ")


class PlateShearEquilibrium(PlateEquilibriumFI):
    """Mechanical weak form for the transverse shear contribution in plate models.

    This weak form is derived from PlateEquilibriumFI but isolates and returns only
    the transverse shear terms (stiffness/force contributions).

    It is intended solely to be paired with a PlateKirchhoffLoveEquilibrium weak
    form (which carries the membrane and bending energy) to manually construct
    selective reduced integration (SRI) elements.

    Note:
        Drilling Rotation (RotZ) Handling:
        The simple drilling penalty stiffness is excluded from this reduced
        integration weak form when `true_drilling_rotation` is False, ensuring
        it is instead evaluated using full integration to avoid rank deficiency.

    Note:
        This function is dedicated to pedagogical purposes (e.g., demonstrating
        manual SRI assembly). For production or performance-critical applications,
        prefer using PlateEquilibrium, which automatically handles selective
        reduced integration internally.
    """

    def get_weak_equation(self, assembly, pb):
        # shear
        if "_ShellStiffnessMatrix" not in assembly.sv:
            assembly.sv["_ShellStiffnessMatrix"] = (
                self.constitutivelaw.get_shell_stiffness_matrix()
            )
        H = assembly.sv["_ShellStiffnessMatrix"]

        GammaXZ = self.space.derivative("DispZ", "X") + self.space.variable("RotY")
        GammaYZ = self.space.derivative("DispZ", "Y") - self.space.variable("RotX")

        generalized_strain = [GammaXZ, GammaYZ]
        initial_stress = assembly.sv["ShellStress"]

        diffop = 0
        if not (np.array_equal(initial_stress, 0)):
            diffop = sum(
                [
                    (
                        generalized_strain[i].virtual * initial_stress[i + 6]
                        if generalized_strain[i] != 0
                        else 0
                    )
                    for i in range(2)
                ]
            )

        diffop += sum(
            [
                sum(
                    [
                        (
                            0
                            if (
                                generalized_strain[j] == 0 or generalized_strain[i] == 0
                            )
                            else generalized_strain[i].virtual
                            * generalized_strain[j]
                            * H[i + 6][j + 6]
                        )
                        for j in range(2)
                    ]
                )
                for i in range(2)
            ]
        )

        if self.true_drilling_rotation and self.drill_stiffness_coefficient != 0:
            # penalty for RotZ
            # use full integration if simple stiffness to avoid singularity
            op_drill_constraint = self.drill_constraint_operator()
            representative_stiffness = H[2][2]
            penalty = representative_stiffness * self.drill_stiffness_coefficient

            diffop += (
                op_drill_constraint.virtual
                * (op_drill_constraint + assembly.sv["_DrillConstraint"])
                * penalty
            )

        return diffop


class PlateKirchhoffLoveEquilibrium(PlateEquilibriumFI):
    """Mechanical weak form for the membrane and bending contributions in plate models.

    This weak form is derived from PlateEquilibriumFI but isolates and returns only
    the membrane and bending terms (stiffness/force contributions), omitting
    transverse shear.

    It is intended solely to be paired with a PlateShearEquilibrium weak form
    (which carries the transverse shear energy) to manually construct
    selective reduced integration (SRI) elements.

    Note:
        Drilling Rotation (RotZ) Handling:
        If `true_drilling_rotation` is set to False, a simple drilling penalty
        stiffness is added within this fully integrated weak form to prevent
        numerical singularity at the nodes.

    Note:
        This class is dedicated to pedagogical purposes (e.g., demonstrating
        manual SRI assembly). For production or performance-critical applications,
        prefer using PlateEquilibrium, which automatically handles selective
        reduced integration internally.
    """

    def get_weak_equation(self, assembly, pb):
        if "_ShellStiffnessMatrix" not in assembly.sv:
            assembly.sv["_ShellStiffnessMatrix"] = (
                self.constitutivelaw.get_shell_stiffness_matrix()
            )
        H = assembly.sv["_ShellStiffnessMatrix"]

        generalized_strain = self.generalized_strain_operator()[:6]
        initial_stress = assembly.sv["ShellStress"]

        diffop = 0
        if not (np.array_equal(initial_stress, 0)):
            diffop = sum(
                [
                    (
                        generalized_strain[i].virtual * initial_stress[i]
                        if generalized_strain[i] != 0
                        else 0
                    )
                    for i in range(6)
                ]
            )

            # Geometrical stiffness
            if assembly._nlgeom:
                Nx = initial_stress[0]
                Ny = initial_stress[1]
                Nxy = initial_stress[2]
                dw_dx = self.space.derivative("DispZ", "X")
                dw_dy = self.space.derivative("DispZ", "Y")
                diffop += (
                    dw_dx.virtual * dw_dx * Nx
                    + dw_dy.virtual * dw_dy * Ny
                    + dw_dx.virtual * dw_dy * Nxy
                    + dw_dy.virtual * dw_dx * Nxy
                )

        diffop += sum(
            [
                sum(
                    [
                        (
                            0
                            if (
                                generalized_strain[j] == 0 or generalized_strain[i] == 0
                            )
                            else generalized_strain[i].virtual
                            * generalized_strain[j]
                            * H[i][j]
                        )
                        for j in range(6)
                    ]
                )
                for i in range(6)
            ]
        )

        if not (self.true_drilling_rotation) and self.drill_stiffness_coefficient != 0:
            # penalty for RotZ
            # use full integration if simple stiffness to avoid singularity
            op_drill_constraint = self.drill_constraint_operator()
            representative_stiffness = H[2][2]
            penalty = representative_stiffness * self.drill_stiffness_coefficient

            diffop += (
                op_drill_constraint.virtual
                * (op_drill_constraint + assembly.sv["_DrillConstraint"])
                * penalty
            )

        return diffop


class PlateDrillingPenalty(PlateEquilibriumFI):
    """Mechanical weak form for the drilling penalty in plate models.

    This weak form is derived from PlateEquilibriumFI but isolates and returns only
    the drilling penalty.

    It is intended solely to be paired with other weak form to manually construct
    a new selective reduced integration (SRI) element.
    """

    def get_weak_equation(self, assembly, pb):
        if "_ShellStiffnessMatrix" not in assembly.sv:
            assembly.sv["_ShellStiffnessMatrix"] = (
                self.constitutivelaw.get_shell_stiffness_matrix()
            )
        H = assembly.sv["_ShellStiffnessMatrix"]

        # penalty for RotZ
        # use full integration if simple stiffness to avoid singularity
        op_drill_constraint = self.drill_constraint_operator()
        representative_stiffness = H[2][2]
        penalty = representative_stiffness * self.drill_stiffness_coefficient

        diffop = (
            op_drill_constraint.virtual
            * (op_drill_constraint + assembly.sv["_DrillConstraint"])
            * penalty
        )
        return diffop


class _ShellComponentList(list):
    def asarray(self):
        try:
            return np.array(self)
        except ValueError:  # fill zeros first
            for i in range(8):
                if not (np.isscalar(self[i])):
                    N = len(self[i])  # number of stress values
                    break

            res = np.empty((8, N))
            for i in range(8):
                res[i] = self[i]
            return res
