from fedoo.core.weakform import WeakFormBase, WeakFormSum
from fedoo.core.base import ConstitutiveLaw
from scipy.spatial.transform import Rotation
import numpy as np


class PlateEquilibriumFI(WeakFormBase):  # plate weakform whith full integration.
    """Mechanical equilibrium equation for plate models, with full integration.

    This weakform uses a full integration of the equation that leads to locking
    for elements with linear interpolation. Should be used in combination with
    a Shell Constitutive Law like :mod:`fedoo.constitutivelaw.ShellHomogeneous`.

    *:mod:`fedoo.weakform.PlateEquilibrium` should be prefered unless you know
    what you are doing.*
    This weakform use a full integration of the equation that leads to locking
    for elements with linear interpolation.
    Should be used in combination with a Shell Constitutive Law
    like :mod:`fedoo.constitutivelaw.ShellHomogeneous` or
    :mod:`fedoo.constitutivelaw.ShellLaminate`.
    Geometrical non linearities are implemented with a corotational approach.

    Parameters
    ----------
    PlateConstitutiveLaw: ConstitutiveLaw name (str) or ConstitutiveLaw object
        Shell Constitutive Law (:mod:`fedoo.constitutivelaw`)
    name: str
        name of the WeakForm
    nlgeom: bool or str
        Property used to treat the geometric non linearities.
    strategy: str
        Ignored if nlgeom is False
        'total' for vectorized total corotational extraction (default).
        'incremental' for step-by-step incremental local frame update.
    """

    def __init__(
        self,
        PlateConstitutiveLaw,
        name="",
        nlgeom=None,
        space=None,
        strategy="incremental",
    ):
        if isinstance(PlateConstitutiveLaw, str):
            PlateConstitutiveLaw = ConstitutiveLaw.get_all()[PlateConstitutiveLaw]

        if name == "":
            name = PlateConstitutiveLaw.name

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

        self.constitutivelaw = PlateConstitutiveLaw

        # automatically set the right assembly element formulation
        self.assembly_options["elm_type", "tri3"] = "ptri3"
        self.assembly_options["elm_type", "quad4"] = "pquad4"
        self.assembly_options["elm_type", "tri6"] = "ptri6"
        self.assembly_options["elm_type", "quad8"] = "pquad8"
        self.assembly_options["elm_type", "quad9"] = "pquad9"

        self.nlgeom = nlgeom
        self.strategy = strategy

    def initialize(self, assembly, pb):
        assembly.sv["ShellStrain"] = 0
        assembly.sv["ShellStress"] = 0

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

                if self.strategy == "incremental":
                    assembly.sv["RigidRotationMat"] = init_frame
                    # assembly.current._element_local_frame = init_frame.reshape(
                    #     mesh.n_elements, -1, 3, 3
                    # )  # usefull?

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

    def update(self, assembly, pb):
        if self.nlgeom == "UL":
            assembly.set_disp(pb.get_disp())

        dof = pb.get_dof_solution()
        if np.isscalar(dof) and dof == 0:
            assembly.sv["ShellStrain"] = assembly.sv["ShellStress"] = 0
            return

        op_plate_strain = self.GetGeneralizedStrainOperator()
        H = self.constitutivelaw.GetShellRigidityMatrix()
        mesh = assembly.current.mesh

        if self.nlgeom:
            nodes_pos = mesh.nodes[mesh.elements]
            current_center = nodes_pos.mean(axis=1)

            initial_frame = assembly.sv["_InitialRigidRotationMat"]
            initial_node_local_pos = assembly.sv["_InitialNodeLocalPos"]

            # =========================================================
            # 2. ELEMENT FRAME EXTRACTION
            # =========================================================
            if self.strategy == "total":
                # Dynamic Normal (e3) computation
                if mesh.elm_type[:3] == "tri":
                    d1 = nodes_pos[:, 1, :] - nodes_pos[:, 0, :]
                    d2 = nodes_pos[:, 2, :] - nodes_pos[:, 0, :]
                else:  # assume quad type elements -> use diagonal
                    d1 = nodes_pos[:, 2, :] - nodes_pos[:, 0, :]
                    d2 = nodes_pos[:, 3, :] - nodes_pos[:, 1, :]

                e3 = np.cross(d1, d2)
                e3 /= np.linalg.norm(e3, axis=1)[:, np.newaxis]

                # Local X (e1) and Y (e2) computation
                e1_approx = nodes_pos[:, 1, :] - nodes_pos[:, 0, :]
                e1 = e1_approx - (
                    np.einsum("ei,ei->e", e1_approx, e3)[:, np.newaxis] * e3
                )
                e1 /= np.linalg.norm(e1, axis=1)[:, np.newaxis]
                e2 = np.cross(e3, e1)

                rigid_rotmat = np.stack([e1, e2, e3], axis=1)

                # Global Rotations
                rot_var = self.space.get_rank_vector("Rot")
                dof_slice = slice(
                    rot_var[0] * mesh.n_nodes, (rot_var[0] + 3) * mesh.n_nodes
                )
                rot_vecs_global = dof[dof_slice].reshape(3, -1).T

                R_global_nodes = (
                    Rotation.from_rotvec(rot_vecs_global[mesh.elements].reshape(-1, 3))
                    .as_matrix()
                    .reshape(mesh.n_elements, mesh.n_elm_nodes, 3, 3)
                )
                # if not pb._line_search_update:
                assembly.sv["RigidRotationMat"] = rigid_rotmat
                assembly.current._element_local_frame = rigid_rotmat.reshape(
                    mesh.n_elements, -1, 3, 3
                )

            elif self.strategy == "incremental":
                # Incremental frame update
                delta_rotvec_nodes = pb._get_vect_component(pb.get_X(), "Rot")
                delta_rotvec = delta_rotvec_nodes.T[mesh.elements].mean(axis=1)

                rigid_rotmat_trial = assembly.sv["RigidRotationMat"] @ (
                    Rotation.from_rotvec(delta_rotvec).as_matrix()
                )

                current_local_frame = mesh.get_element_local_frame()
                e3 = current_local_frame[:, 2]
                e1_trial = rigid_rotmat_trial[:, 0]
                e1 = e1_trial - (e1_trial * e3).sum(axis=1).reshape(-1, 1) * e3
                e1_norm = np.linalg.norm(e1, axis=1).reshape(-1, 1)
                e1 = np.where(e1_norm > 1e-12, e1 / e1_norm, e1_trial)
                e2 = np.cross(e3, e1)

                rigid_rotmat = np.empty_like(rigid_rotmat_trial)
                rigid_rotmat[:, 0, :] = e1
                rigid_rotmat[:, 1, :] = e2
                rigid_rotmat[:, 2, :] = e3

                nodes_rotmat = (
                    Rotation.from_rotvec(
                        delta_rotvec_nodes.T
                    ).as_matrix()  # transpose(0,2,1)
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
                    n_slice = slice(
                        rot_var[0] * mesh.n_nodes, (rot_var[0] + 3) * mesh.n_nodes
                    )

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

            else:
                raise ValueError(
                    f"Unknown corotational strategy: '{self.strategy}'. "
                    "Choose 'total' or 'incremental'."
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

            # =========================================================
            # 4. EVALUATE STRAINS
            # =========================================================
            assembly.sv["ShellStrain"] = _ShellComponentList(
                [
                    (
                        assembly.current.get_gp_results(
                            op, dof_local, use_local_dof=True
                        )
                        if op != 0
                        else 0
                    )
                    for op in op_plate_strain
                ]
            )
        else:
            # Small displacement (Linear) update
            assembly.sv["ShellStrain"] = [
                assembly.get_gp_results(op, dof) if op != 0 else 0
                for op in op_plate_strain
            ]

        # Evaluate Stresses (Shared linear/nonlinear)
        try:
            assembly.sv["ShellStress"] = _ShellComponentList(
                [
                    sum(
                        [
                            (
                                assembly.sv["ShellStrain"][j] * H[i][j]
                                if not np.isscalar(assembly.sv["ShellStrain"][j])
                                or assembly.sv["ShellStrain"][j] != 0
                                else 0
                            )
                            for j in range(8)
                        ]
                    )
                    for i in range(8)
                ]
            )
            # print(assembly.sv["ShellStress"][2])
        except NameError:
            assembly.sv["ShellStress"] = [
                sum(
                    [
                        (
                            assembly.sv["ShellStrain"][j] * H[i][j]
                            if not np.isscalar(assembly.sv["ShellStrain"][j])
                            or assembly.sv["ShellStrain"][j] != 0
                            else 0
                        )
                        for j in range(8)
                    ]
                )
                for i in range(8)
            ]

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

    def GetGeneralizedStrainOperator(self):
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

    def get_weak_equation(self, assembly, pb):
        H = self.constitutivelaw.GetShellRigidityMatrix()
        GeneralizedStrain = self.GetGeneralizedStrainOperator()
        initial_stress = assembly.sv["ShellStress"]

        diffop = 0
        if not (np.array_equal(initial_stress, 0)):
            diffop = sum(
                [
                    (
                        GeneralizedStrain[i].virtual * initial_stress[i]
                        if GeneralizedStrain[i] != 0
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
                            if (GeneralizedStrain[j] == 0 or GeneralizedStrain[i] == 0)
                            else GeneralizedStrain[i].virtual
                            * GeneralizedStrain[j]
                            * H[i][j]
                        )
                        for j in range(8)
                    ]
                )
                for i in range(8)
            ]
        )

        # penalty for RotZ (drilling DOF stabilization)
        representative_stiffness = H[0][0]
        penalty = representative_stiffness * 1e-2
        diffop += (
            self.space.variable("RotZ").virtual * self.space.variable("RotZ") * penalty
        )

        return diffop


class PlateShearEquilibrium(
    PlateEquilibriumFI
):  # weak form of plate shear energy containing only the shear strain energy
    def get_weak_equation(self, assembly, pb):
        # shear
        H = self.constitutivelaw.GetShellRigidityMatrix_RI()

        GammaXZ = self.space.derivative("DispZ", "X") + self.space.variable("RotY")
        GammaYZ = self.space.derivative("DispZ", "Y") - self.space.variable("RotX")

        GeneralizedStrain = [GammaXZ, GammaYZ]
        initial_stress = assembly.sv["ShellStress"]

        diffop = 0
        if not (np.array_equal(initial_stress, 0)):
            diffop = sum(
                [
                    (
                        GeneralizedStrain[i].virtual * initial_stress[i + 6]
                        if GeneralizedStrain[i] != 0
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
                            if (GeneralizedStrain[j] == 0 or GeneralizedStrain[i] == 0)
                            else GeneralizedStrain[i].virtual
                            * GeneralizedStrain[j]
                            * H[i][j]
                        )
                        for j in range(2)
                    ]
                )
                for i in range(2)
            ]
        )

        return diffop


class PlateKirchhoffLoveEquilibrium(PlateEquilibriumFI):  # plate without shear strain
    def get_weak_equation(self, assembly, pb):
        # all component but shear, for full integration
        H = self.constitutivelaw.GetShellRigidityMatrix_FI()

        GeneralizedStrain = self.GetGeneralizedStrainOperator()[:6]
        initial_stress = assembly.sv["ShellStress"]

        diffop = 0
        if not (np.array_equal(initial_stress, 0)):
            diffop = sum(
                [
                    (
                        GeneralizedStrain[i].virtual * initial_stress[i]
                        if GeneralizedStrain[i] != 0
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
                            if (GeneralizedStrain[j] == 0 or GeneralizedStrain[i] == 0)
                            else GeneralizedStrain[i].virtual
                            * GeneralizedStrain[j]
                            * H[i][j]
                        )
                        for j in range(6)
                    ]
                )
                for i in range(6)
            ]
        )

        # penalty for RotZ
        H = self.constitutivelaw.GetShellRigidityMatrix()
        representative_stiffness = H[0][0]
        penalty = representative_stiffness * 1e-2
        diffop += (
            self.space.variable("RotZ").virtual * self.space.variable("RotZ") * penalty
        )

        return diffop


def PlateEquilibriumSI(
    PlateConstitutiveLaw, name=None, nlgeom=None, space=None
):  # plate weakform which force reduced integration for shear terms
    """Mechanical equilibrium equation for plate models.

    *:mod:`fedoo.weakform.PlateEquilibrium` should be prefered unless you know
    what you are doing.*
    This weakform use a reduced integration to treat the shear terms. That
    avoid locking problems for elements with linear interpolation but may lead
    to instability when used with quadratic interpolations.

    Should be used in combination with a Shell Constitutive Law
    like :mod:`fedoo.constitutivelaw.ShellHomogeneous` or
    :mod:`fedoo.constitutivelaw.ShellLaminate`.
    Geometrical non linearities are implemented with a corotational approach.

    Parameters
    ----------
    PlateConstitutiveLaw: ConstitutiveLaw name (str) or ConstitutiveLaw object
        Shell Constitutive Law (:mod:`fedoo.constitutivelaw`)
    name: str
        name of the WeakForm
    """
    plate_shear = PlateShearEquilibrium(PlateConstitutiveLaw, "", nlgeom, space)
    plate_kl = PlateKirchhoffLoveEquilibrium(PlateConstitutiveLaw, "", nlgeom, space)

    plate_shear.assembly_options["n_elm_gp"] = (
        1  # use reduced integration for shear components
    )
    if name is None:
        if isinstance(PlateConstitutiveLaw, str):
            name = ConstitutiveLaw().get_all()[PlateConstitutiveLaw].name
        else:
            name = PlateConstitutiveLaw.name
    return WeakFormSum([plate_kl, plate_shear], name)


def PlateEquilibrium(PlateConstitutiveLaw, name=None, nlgeom=None, space=None):
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
    PlateConstitutiveLaw: ConstitutiveLaw name (str) or ConstitutiveLaw object
        Shell Constitutive Law (:mod:`fedoo.constitutivelaw`)
    name: str
        name of the WeakForm
    """
    plate_shear = PlateShearEquilibrium(PlateConstitutiveLaw, "", nlgeom, space)
    plate_kl = PlateKirchhoffLoveEquilibrium(PlateConstitutiveLaw, "", nlgeom, space)

    # if linear element 'ptri3' and 'pquad4': use reduced integration for shear terms
    plate_shear.assembly_options["n_elm_gp", "ptri3"] = 1
    plate_shear.assembly_options["n_elm_gp", "pquad4"] = 1
    if name is None:
        if isinstance(PlateConstitutiveLaw, str):
            name = ConstitutiveLaw().get_all()[PlateConstitutiveLaw].name
        else:
            name = PlateConstitutiveLaw.name
    return WeakFormSum([plate_kl, plate_shear], name)


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
