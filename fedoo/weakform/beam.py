from fedoo.core.base import ConstitutiveLaw
from fedoo.constitutivelaw.beam import BeamProperties
from fedoo.core.weakform import WeakFormBase
from scipy.spatial.transform import Rotation
import numpy as np


class BeamEquilibrium(WeakFormBase):
    """
    Weak formulation of the mechanical equilibrium equation for beam models.

    Geometrical are implemented with the updated lagrangian approach.

    Parameters
    ----------
    material: ConstitutiveLaw name (str) or ConstitutiveLaw object
        material can be either a :ref:`beam_constitutive_laws` or a material
        constitutive law with attributes E and G for elastic and shear modulus
        (for instance :mod:`fedoo.constitutivelaw.ElasticIsotrop`). If material
        is a beam constitutive law, the following parameters A, Jx, Iyy, Izz
        and k are ignored.
    A: scalar or arrays of gauss point values, optional
        Beam section area.
    Jx: scalar or arrays of gauss point values, optional
        Torsion constant.
    Iyy: scalar or arrays of gauss point values, optional
        Second moment of area with respect to y (beam local coordinate system).
    Izz: scalar or arrays of gauss point values, optional
        Second moment of area with respect to z (beam local coordinate system).
    k: scalar or arrays of gauss point values, optional
        Shear coefficient. If k=0 (*default*) the beam use the bernoulli
        hypothesis.
    name: str
        name of the WeakForm.
    """

    def __init__(
        self,
        material,
        A=None,
        Jx=None,
        Iyy=None,
        Izz=None,
        k=0,
        name="",
        nlgeom=None,
        space=None,
    ):
        # k: shear shape factor

        WeakFormBase.__init__(self, name, space)

        self.space.new_variable("DispX")
        self.space.new_variable("DispY")
        if self.space.ndim == 3:
            self.space.new_variable("DispZ")
            self.space.new_variable("RotX")  # torsion rotation
            self.space.new_variable("RotY")
            self.space.new_variable("RotZ")
            self.space.new_vector("Disp", ("DispX", "DispY", "DispZ"))
            self.space.new_vector("Rot", ("RotX", "RotY", "RotZ"))
        elif self.space.ndim == 2:
            self.space.new_variable("RotZ")
            self.space.variable_alias("Rot", "RotZ")
            self.space.new_vector("Disp", ["DispX", "DispY"])
            # self.space.new_vector('Rot' , ['RotZ'] )
        # elif get_Dimension() == '2Dstress':
        #     assert 0, "No 2Dstress model for a beam kinematic. Choose '2Dplane' instead."

        if isinstance(material, BeamProperties):
            self.properties = material
        else:
            self.properties = BeamProperties(
                material, A, Jx, Iyy, Izz, k, name + "_properties"
            )
        self.assembly_options.set("elm_type", "beam", elm_type="lin2")

        self.nlgeom = nlgeom  # geometric non linearities -> False, True, 'UL' or 'TL' (True or 'UL': updated lagrangian - 'TL': total lagrangian)
        """Method used to treat the geometric non linearities.
            * Set to False if geometric non linarities are ignored (default).
            * Set to True or 'UL' to use the updated lagrangian method
              (update the mesh)
            * Set to 'TL' to use the total lagrangian method (base on the
              initial mesh with initial displacement effet)
        """

    def initialize(self, assembly, pb):
        assembly.sv["BeamStrain"] = 0
        assembly.sv["BeamStress"] = 0

        self._initialize_nlgeom(assembly, pb)
        self.nlgeom = assembly._nlgeom

    def update(self, assembly, pb):
        # function called when the problem is updated (NR loop or time increment)
        # Nlgeom implemented only for updated lagragian formulation
        if self.nlgeom == "UL":
            # if updated lagragian method -> update the mesh and recompute elementary op
            assembly.set_disp(pb.get_disp())
            # if assembly.current.mesh in assembly._saved_change_of_basis_mat:
            #     del assembly._saved_change_of_basis_mat[assembly.current.mesh]

            # assembly.current.compute_elementary_operators()

        dof = pb.get_dof_solution()  # displacement and rotation node values
        if np.isscalar(dof) and dof == 0:
            assembly.sv["BeamStrain"] = assembly.sv["BeamStress"] = 0
        else:
            op_beam_strain = assembly.space.op_beam_strain()
            Ke = [
                0 if (np.isscalar(k) and k == 0) else k
                for k in self.properties.get_beam_rigidity()
            ]  # make sure 0 values of rigidity are int

            # evaluate Strain
            if self.nlgeom:
                # Compute Beam Strain
                mesh = assembly.current.mesh  # deformed mesh
                if self.space.ndim == 2:
                    # values from initial and last iterations
                    if "_ElementVectors" in assembly.sv:
                        element_vectors_old = assembly.sv["_ElementVectors"]
                        initial_element_vectors = assembly.sv["_InitialElementVectors"]
                    else:
                        initial_element_vectors = (
                            assembly.mesh.nodes[mesh.elements[:, 1]]
                            - assembly.mesh.nodes[mesh.elements[:, 0]]
                        )
                        element_vectors_old = initial_element_vectors
                        assembly.sv["RigidRotation"] = 0
                        assembly.sv["_InitialElementVectors"] = initial_element_vectors

                    # coordinates of vector between node 1 and 2 for each element
                    element_vectors = (
                        mesh.nodes[mesh.elements[:, 1]]
                        - mesh.nodes[mesh.elements[:, 0]]
                    )

                    # compute new angle for rigid body motion
                    delta_angle = np.arctan2(
                        element_vectors_old[:, 0] * element_vectors[:, 1]
                        - element_vectors_old[:, 1] * element_vectors[:, 0],
                        element_vectors_old[:, 0] * element_vectors[:, 0]
                        + element_vectors_old[:, 1] * element_vectors[:, 1],
                    )
                    rigid_rot = assembly.sv["RigidRotation"] + delta_angle

                    # compute u2, rot1 and rot2, ie the rot dof at node 1 and 2
                    # u1 = 0 as the first node is the center of the local bases
                    rot1 = pb.get_rot().T[mesh.elements[:, 0]].ravel() - rigid_rot
                    rot2 = pb.get_rot().T[mesh.elements[:, 1]].ravel() - rigid_rot

                    # longitunal displacement in local coordinates (u2y=0)
                    u2x = np.linalg.norm(element_vectors, axis=1) - np.linalg.norm(
                        initial_element_vectors, axis=1
                    )

                    # build the local dof vector
                    dof_local = np.zeros(
                        self.space.nvar * mesh.n_elm_nodes * mesh.n_elements
                    )
                    n_dof_per_var = mesh.n_elm_nodes * mesh.n_elements
                    var = self.space.variable_rank("RotZ")
                    dof_local[var * n_dof_per_var : (var + 1) * n_dof_per_var] = (
                        np.hstack([rot1.T, rot2.T]).ravel()
                    )

                    var = self.space.variable_rank("DispX")
                    dof_local[
                        var * n_dof_per_var + mesh.n_elements : (var + 1)
                        * n_dof_per_var
                    ] = u2x

                    # update saved values for next iteration
                    assembly.sv["_ElementVectors"] = element_vectors
                    assembly.sv["RigidRotation"] = rigid_rot
                    # print(pb.get_rot()[[1,7]])
                    # pass

                else:  # ndim === 3
                    # values from initial and last iterations
                    if "_InitialElementVectors" in assembly.sv:
                        initial_element_vectors = assembly.sv["_InitialElementVectors"]
                        initial_element_rotmat = assembly.sv["_InitialRigidRotationMat"]
                    else:
                        # vector along the element axis (local X direction)
                        initial_element_vectors = (
                            assembly.mesh.nodes[mesh.elements[:, 1]]
                            - assembly.mesh.nodes[mesh.elements[:, 0]]
                        )
                        assembly.sv["_InitialElementVectors"] = initial_element_vectors

                        # element rigid rotation mat to get the local frame from global
                        assembly.sv["RigidRotationMat"] = (
                            assembly.mesh.get_element_local_frame()
                        )
                        # initial rotation mat to get the initial element local frame
                        initial_element_rotmat = assembly.sv[
                            "_InitialRigidRotationMat"
                        ] = assembly.sv["RigidRotationMat"]

                        # dof rotation matrix at node (without elm initial rotation)
                        assembly.sv["_NodesRotationMatrix"] = np.tile(
                            np.eye(3), (assembly.mesh.n_nodes, 1, 1)
                        )

                    # coordinates of vector between node 1 and 2 for each element
                    element_vectors = (
                        mesh.nodes[mesh.elements[:, 1]]
                        - mesh.nodes[mesh.elements[:, 0]]
                    )

                    # get rotvec increment at nodes
                    delta_rotvec_nodes = pb._get_vect_component(pb.get_X(), "Rot")

                    # get element mean rotvec increment
                    delta_rotvec = 0.5 * delta_rotvec_nodes.T[mesh.elements].sum(axis=1)

                    # update rigid rotation local_frame (rotmatrix from global frame)
                    rigid_rotmat_trial = (
                        assembly.sv["RigidRotationMat"]
                        @ Rotation.from_rotvec(delta_rotvec).as_matrix()
                    )

                    # get rigid rotation by projection (X axis is the poutre normal)
                    rigid_rotmat = np.empty_like(rigid_rotmat_trial)
                    rigid_rotmat[:, 0] = element_vectors / np.linalg.norm(
                        element_vectors, axis=1
                    ).reshape(-1, 1)
                    rigid_rotmat[:, 1] = np.cross(
                        rigid_rotmat_trial[:, 2], rigid_rotmat[:, 0]
                    )
                    rigid_rotmat[:, 1] /= np.linalg.norm(
                        rigid_rotmat[:, 1], axis=1
                    ).reshape(-1, 1)
                    rigid_rotmat[:, 2] = np.cross(
                        rigid_rotmat[:, 0], rigid_rotmat[:, 1]
                    )

                    # compute local dof vector
                    # compute u2, rot1 and rot2, ie the rot dof at node 1 and 2
                    # u1 = 0 as the first node is the center of the local bases
                    nodes_rotmat = (
                        Rotation.from_rotvec(delta_rotvec_nodes.T).as_matrix()
                        @ assembly.sv["_NodesRotationMatrix"]
                    )

                    rot1 = Rotation.from_matrix(
                        rigid_rotmat
                        @ nodes_rotmat[mesh.elements[:, 0]]
                        @ initial_element_rotmat.transpose(0, 2, 1)
                    ).as_rotvec()
                    rot2 = Rotation.from_matrix(
                        rigid_rotmat
                        @ nodes_rotmat[mesh.elements[:, 1]]
                        @ initial_element_rotmat.transpose(0, 2, 1)
                    ).as_rotvec()

                    # longitunal displacement in local coordinates (u2y=0)
                    u2x = np.linalg.norm(element_vectors, axis=1) - np.linalg.norm(
                        initial_element_vectors, axis=1
                    )

                    # build the local dof vector
                    dof_local = np.zeros(
                        self.space.nvar * mesh.n_elm_nodes * mesh.n_elements
                    )
                    n_dof_per_var = mesh.n_elm_nodes * mesh.n_elements
                    # ### WARNING only work if vectors are contigous in the variable order, perhaps we should enforce this for more security
                    var = self.space.variable_rank("RotX")
                    dof_local[var * n_dof_per_var : (var + 3) * n_dof_per_var] = (
                        np.hstack([rot1.T, rot2.T]).ravel()
                    )

                    var = self.space.variable_rank("DispX")
                    dof_local[
                        var * n_dof_per_var + mesh.n_elements : (var + 1)
                        * n_dof_per_var
                    ] = u2x

                    # update saved values for next iteration
                    # assembly.sv["_ElementVectors"] = element_vectors
                    assembly.sv["RigidRotationMat"] = rigid_rotmat
                    assembly.sv["_NodesRotationMatrix"] = nodes_rotmat
                    assembly.current._element_local_frame = rigid_rotmat.reshape(
                        mesh.n_elements, -1, self.space.ndim, self.space.ndim
                    )

                    # update rot values and dirichlet boundary conditions

                    rot_var = self.space.get_rank_vector("Rot")
                    ### WARNING only work if vectors are contigous in the variable order
                    if np.isscalar(pb._U) and pb._U == 0:
                        pb._dU[
                            rot_var[0] * assembly.mesh.n_nodes : (rot_var[0] + 3)
                            * assembly.mesh.n_nodes
                        ] = (Rotation.from_matrix(nodes_rotmat).as_rotvec().T).ravel()
                    else:
                        nodes_rotmat_from_U = Rotation.from_rotvec(
                            pb._U[
                                rot_var[0] * assembly.mesh.n_nodes : (rot_var[0] + 3)
                                * assembly.mesh.n_nodes
                            ]
                            .reshape(3, -1)
                            .T
                        ).as_matrix()

                        # d_angle = final_angle - angle_start (computed with rotmat from angular point of view)
                        pb._dU[
                            rot_var[0] * assembly.mesh.n_nodes : (rot_var[0] + 3)
                            * assembly.mesh.n_nodes
                        ] = (
                            Rotation.from_matrix(
                                nodes_rotmat @ nodes_rotmat_from_U.transpose(0, 2, 1)
                            )
                            .as_rotvec()
                            .T.ravel()
                        )

                        # Or equivalent bug sigularité if angle > pi
                        # pb._dU[
                        #     rot_var[0] * assembly.mesh.n_nodes : (rot_var[0] + 3)
                        #     * assembly.mesh.n_nodes
                        # ] = (
                        #     Rotation.from_matrix(nodes_rotmat).as_rotvec().T
                        # ).ravel() - pb._U[
                        #     rot_var[0] * assembly.mesh.n_nodes : (rot_var[0] + 3)
                        #     * assembly.mesh.n_nodes
                        # ]

                    for bc in pb.bc.list_all():
                        if bc.bc_type == "Dirichlet":
                            if bc.variable in rot_var:
                                # if bc._dof_index[0] == 605:
                                #     print(bc.get_true_value(pb._t_fact))
                                pb._Xbc[bc._dof_index] = (
                                    bc.get_true_value(pb.t_fact)
                                    - pb.get_dof_solution()[bc._dof_index]
                                )

                # compute the beam strain at gausspoint
                assembly.sv["BeamStrain"] = _BeamComponentList(
                    [
                        0
                        if ((np.isscalar(Ke[i]) and Ke[i] == 0) or (op == 0))
                        else assembly.current.get_gp_results(
                            op, dof_local, use_local_dof=True
                        )
                        for i, op in enumerate(op_beam_strain)
                    ]
                )

            else:
                assembly.sv["BeamStrain"] = [
                    (
                        0
                        if ((np.isscalar(Ke[i]) and Ke[i] == 0) or (op == 0))
                        else assembly.get_gp_results(op, dof)
                    )
                    for i, op in enumerate(op_beam_strain)
                ]

            assembly.sv["BeamStress"] = _BeamComponentList(
                [Ke[i] * assembly.sv["BeamStrain"][i] for i in range(6)]
            )

    def to_start(self, assembly, pb):
        if self.nlgeom == "UL":
            # if updated lagragian method -> reset the mesh to the begining of the increment
            assembly.set_disp(pb.get_disp())
            if self.space.ndim == 3:
                assembly.current._element_local_frame = assembly.sv_start[
                    "RigidRotationMat"
                ].reshape(
                    assembly.current.mesh.n_elements,
                    -1,
                    self.space.ndim,
                    self.space.ndim,
                )

    def set_start(self, assembly, pb):
        if self.nlgeom and self.space.ndim == 3:  # only UL for now
            # update rot dof because
            pass
            # if not(np.isscalar(pb.get_dof_solution()) and pb.get_dof_solution() == 0):
            #     #update rotation vector values
            #     ### WARNING only work if vectors are contigous in the variable order
            #     var = self.space.variable_rank('RotX')
            #     pb._U[var*assembly.mesh.n_nodes:(var+3)*assembly.mesh.n_nodes] = \
            #         (Rotation.from_matrix(assembly.sv['_NodesRotationMatrix']).as_rotvec().T).ravel()
            # print(Rotation.from_matrix(assembly.sv['_NodesRotationMatrix']).as_rotvec().T - pb.get_rot())
            # rot_dof = Rotation.from_matrix(assembly.sv['_NodesRotationMatrix']).as_rotvec()

    def get_weak_equation(self, assembly, pb):
        eps = self.space.op_beam_strain()
        Ke = self.properties.get_beam_rigidity()

        diff_op = sum(
            [eps[i].virtual * eps[i] * Ke[i] if eps[i] != 0 else 0 for i in range(6)]
        )

        initial_stress = assembly.sv["BeamStress"]

        if not (np.array_equal(initial_stress, 0)):
            diff_op = diff_op + sum(
                [
                    eps[i].virtual * initial_stress[i] if eps[i] != 0 else 0
                    for i in range(6)
                ]
            )

            # Geometrical stiffness (or initial stress stiffness)
            if assembly._nlgeom:
                N = initial_stress[0]  # normal force
                dv_dx = self.space.derivative("DispY", "X")
                diff_op = diff_op + dv_dx.virtual * dv_dx * N
                if self.space.ndim == 3:
                    dw_dx = self.space.derivative(
                        "DispZ", "X"
                    )  # self.space.derivative("DispY", "Y")
                    diff_op = diff_op + dw_dx.virtual * dw_dx * N
                # axial geometrical generally not significant
                # uncomment the following line to activate
                # diff_op = diff_op + eps[0].virtual * eps[0] * N

        return diff_op

    def _get_generalized_stress_op(self):
        # only for post treatment
        eps = self.space.op_beam_strain()
        Ke = self.properties.get_beam_rigidity()
        return [eps[i] * Ke[i] for i in range(6)]


class _BeamComponentList(list):
    def asarray(self):
        try:
            return np.array(self)
        except ValueError:  # fill zeros first
            for i in range(6):
                if not (np.isscalar(self[i])):
                    N = len(self[i])  # number of stress values
                    break

            res = np.empty((6, N))
            for i in range(6):
                res[i] = self[i]
            return res
