from fedoo.weakform._beam_tangent import beam_tangent_operators
from fedoo.weakform._nodal_rotations import (
    commit_nodal_rotations,
    compose_nodal_rotations,
)
from fedoo.weakform._beam_exact import (
    element_operators,
    exact_beam_kinematics,
    to_field_basis,
)

from fedoo.constitutivelaw.beam import BeamProperties
from fedoo.core.time_evolution import SECOND_ORDER
from fedoo.core.weakform import WeakFormBase, WeakFormSum
from fedoo.weakform.inertia import Inertia, RotaryInertia
from scipy.spatial.transform import Rotation
import numpy as np
from fedoo.util.beam_tensors import BeamStressList, BeamStrainList


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
    consistent_tangent: bool, default=False
        For standard two-node ``beam`` elements with ``nlgeom=True`` or
        ``'UL'``, differentiate the complete existing internal-force residual.
        Includes the changing length, frame and local finite rotations. The
        resulting tangent may be nonsymmetric. Set False for the historical
        material plus axial-force geometric tangent. Alternative beam
        interpolations retain that historical tangent.
        This option is read-only after construction. Create a new weak form
        to select a different tangent, including when sharing forms across assemblies.
    name: str
        name of the WeakForm.

    Notes
    -----
    The current beam formulation supports isotropic linear elastic materials
    only. Material orientations are ignored; the beam cross-section
    orientation is defined by the assembly element frame.
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
        *,
        consistent_tangent=False,
        kinematics=None,
    ):
        # k: shear shape factor

        if kinematics not in (None, "corotational", "exact"):
            raise ValueError(
                "kinematics should be None, 'corotational' or 'exact', "
                f"got {kinematics!r}"
            )
        if kinematics == "exact" and consistent_tangent:
            raise ValueError(
                "consistent_tangent applies to the corotational kinematics; "
                "kinematics='exact' always assembles its exact tangent"
            )
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
        if getattr(self.properties, "is_section_law", False):
            # A nonlinear section (e.g. BeamFiberSection) owns BeamStress and
            # the section tangent: the assembly drives its lifecycle.
            self.constitutivelaw = self.properties
        self.time_evolution = SECOND_ORDER
        self.assembly_options.set("elm_type", "beam", elm_type="lin2")

        self.nlgeom = nlgeom  # geometric non linearities -> False, True, 'UL' or 'TL' (True or 'UL': updated lagrangian - 'TL': total lagrangian)
        """Property used to treat the geometric non linearities.
            * Set to False if geometric non linarities are ignored (default).
            * Set to True or 'UL' to use the updated lagrangian method
              (update the mesh)
            * Set to 'TL' to use the total lagrangian method (base on the
              initial mesh with initial displacement effet)
        """
        self.geometric_stiffness = None
        """Whether to include initial-stress stiffness.

        ``None`` preserves the historical behavior, in which the term follows
        ``nlgeom``. A boolean value explicitly enables or disables it.
        """
        self._consistent_tangent = consistent_tangent
        self._kinematics = kinematics
        if consistent_tangent or kinematics == "exact":
            for variable in ("DispX", "DispY", "DispZ", "RotX", "RotY", "RotZ"):
                if variable in self.space.list_variables():
                    self.space.variable_alias("_BeamMean" + variable, variable)
        if kinematics == "exact":
            # Reduced one-point integration of the two-node element (no
            # shear locking); all element quantities are midpoint values.
            self.assembly_options.set("n_elm_gp", 1, elm_type="beam")

    @property
    def kinematics(self):
        """Read-only beam kinematics, set through the constructor.

        * ``None`` (default): ``"corotational"`` when geometric nonlinearities
          are active, unused otherwise.
        * ``"corotational"``: updated Lagrangian corotational beam, small
          strain in each element frame (``nlgeom="UL"`` only).
        * ``"exact"``: geometrically exact Simo-Reissner beam (Crisfield-Jelenic
          rotation interpolation, one Gauss point per two-node element),
          finite rotations and finite strains of the beam axis. Requires
          ``nlgeom`` (``"UL"`` or ``"TL"``: the formulation is total
          Lagrangian in both cases), the ``beam`` element and ``k > 0``.
        """
        return self._kinematics

    def _is_exact(self):
        return self._kinematics == "exact"

    @property
    def consistent_tangent(self):
        """Read-only tangent selection, set through the constructor.

        False (default) retains the historical tangent. True differentiates the
        UL corotational residual for standard two-node beams, including length,
        frame and finite-rotation derivatives; the tangent may be nonsymmetric.
        geometric_stiffness=False retains the material-only tangent. Other beam
        interpolations use the historical tangent. Create a new weak form to
        change this option; a form may be shared by multiple assemblies.
        """
        return self._consistent_tangent

    def get_storage(self):
        if self.storage is not None:
            return self.storage
        translational_inertia = Inertia(
            self.properties.linear_density, space=self.space
        )
        rotary_density = self.properties.rotary_density
        if self.space.ndim == 2:
            rotary_density = rotary_density[2]
        rotary_inertia = RotaryInertia(rotary_density, space=self.space)
        return WeakFormSum([translational_inertia, rotary_inertia])

    def initialize(self, assembly, pb):
        assembly.sv["BeamStrain"] = 0
        assembly.sv["BeamStress"] = 0

        self._initialize_nlgeom(assembly, pb)
        self.nlgeom = assembly._nlgeom
        if self._is_exact():
            self._initialize_exact(assembly)
        elif assembly._nlgeom == "TL":
            raise NotImplementedError(
                "The corotational beam is updated Lagrangian only: use "
                "nlgeom='UL', or kinematics='exact' for a total Lagrangian "
                "geometrically exact beam"
            )
        # Resolve the formulation default here, as in StressEquilibrium,
        # while preserving explicit weak-form or assembly settings.
        if assembly.assume_sym is None:
            assume_sym = self.assembly_options.get("assume_sym", assembly.elm_type)
            if assume_sym is None:
                # Coupled interpolations expand a weak operator onto several
                # variables. The symmetric assembly shortcut filters terms
                # before that expansion and can omit required contributions.
                assume_sym = not (
                    self._use_consistent_tangent(assembly)
                    or assembly._get_associated_variables()
                )
            assembly.assume_sym = assume_sym

    def update(self, assembly, pb):
        if self._is_exact():
            self._update_exact(assembly, pb)
            return
        assembly.sv.pop("_BeamTangentOperators", None)
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
            assembly.sv.pop("_BeamTangentData", None)
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
                    if not pb._line_search_update:
                        assembly.sv["_ElementVectors"] = element_vectors
                        assembly.sv["RigidRotation"] = rigid_rot

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
                            assembly.get_element_local_frame()
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

                    delta_rotvec_nodes, nodes_rotmat = compose_nodal_rotations(
                        assembly, pb
                    )

                    # get element mean rotvec increment
                    delta_rotvec = 0.5 * delta_rotvec_nodes.T[mesh.elements].sum(axis=1)

                    # Local axes are rows: a spatial rotation Q updates R to R Q.T.
                    rigid_rotmat_trial = assembly.sv[
                        "RigidRotationMat"
                    ] @ Rotation.from_rotvec(delta_rotvec).as_matrix().transpose(
                        0, 2, 1
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

                    if not pb._line_search_update:
                        # update saved values for next iteration
                        # don't do for line_search iteration that is only a trial
                        assembly.sv["RigidRotationMat"] = rigid_rotmat
                        assembly.current._element_local_frame = rigid_rotmat.reshape(
                            mesh.n_elements, -1, self.space.ndim, self.space.ndim
                        )
                        commit_nodal_rotations(self.space, assembly, pb, nodes_rotmat)

                if self._use_consistent_tangent(assembly):
                    frame = (
                        mesh.get_element_local_frame()
                        if self.space.ndim == 2
                        else rigid_rotmat
                    )
                    # Trial frames must also be used when evaluating a line search.
                    assembly.current._element_local_frame = frame[:, None]
                    assembly.sv["_BeamTangentData"] = (
                        dof_local.reshape(self.space.nvar, 2, mesh.n_elements)
                        .transpose(2, 0, 1)
                        .reshape(mesh.n_elements, -1),
                        frame,
                        None if self.space.ndim == 2 else nodes_rotmat[mesh.elements],
                        None if self.space.ndim == 2 else initial_element_rotmat,
                    )

                # compute the beam strain at gausspoint
                assembly.sv["BeamStrain"] = BeamStrainList(
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
                assembly.sv["BeamStrain"] = BeamStrainList(
                    [
                        (
                            0
                            if ((np.isscalar(Ke[i]) and Ke[i] == 0) or (op == 0))
                            else assembly.get_gp_results(op, dof)
                        )
                        for i, op in enumerate(op_beam_strain)
                    ]
                )

            if self.constitutivelaw is None:
                # A section law computes BeamStress from BeamStrain itself.
                assembly.sv["BeamStress"] = BeamStressList(
                    [Ke[i] * assembly.sv["BeamStrain"][i] for i in range(6)]
                )

    def to_start(self, assembly, pb):
        if self.nlgeom == "UL" and not self._is_exact():
            # if updated lagragian method -> reset the mesh to the begining of the increment
            assembly.set_disp(pb.get_disp())
            if self.space.ndim == 2 and self._use_consistent_tangent(assembly):
                data = assembly.sv_start.get("_BeamTangentData")
                assembly.current._element_local_frame = (
                    None if data is None else data[1][:, None]
                )
            if self.space.ndim == 3:
                if "RigidRotationMat" in assembly.sv_start:
                    assembly.current._element_local_frame = assembly.sv_start[
                        "RigidRotationMat"
                    ].reshape(
                        assembly.current.mesh.n_elements,
                        -1,
                        self.space.ndim,
                        self.space.ndim,
                    )
                else:
                    assembly.current._element_local_frame = None

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
        if self._is_exact():
            return self._get_exact_weak_equation(assembly, pb)
        eps = self.space.op_beam_strain()
        tangent = self._section_tangent(assembly)
        if tangent is None:
            Ke = self.properties.get_beam_rigidity()
            diff_op = sum(
                [
                    eps[i].virtual * eps[i] * Ke[i] if eps[i] != 0 else 0
                    for i in range(6)
                ]
            )
        else:
            # Full section tangent of a section law, one value per Gauss point
            diff_op = sum(
                eps[i].virtual * eps[j] * tangent[i, j]
                for i in range(6)
                for j in range(6)
                if eps[i] != 0 and eps[j] != 0 and np.any(tangent[i, j])
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
            geometric_stiffness = self.geometric_stiffness
            if geometric_stiffness is None:
                geometric_stiffness = bool(assembly._nlgeom)
            if geometric_stiffness and not self._use_consistent_tangent(assembly):
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

        # A line-search trial only evaluates the residual: skip the tangent.
        if self._use_consistent_tangent(assembly) and not getattr(
            pb, "_line_search_update", False
        ):
            data = assembly.sv.get("_BeamTangentData")
            if data is not None:
                diff_op += self._get_consistent_tangent(assembly, data)
        return diff_op

    def _get_consistent_tangent(self, assembly, data):
        """Correction to the material tangent, written as four weak terms.

        r(v) = integral eps(v).T s ds, eps(v) = B T v.
        The ordinary eps.virtual * Ke * eps term is already assembled.
        Only its correction and the three stress-dependent terms follow.
        """

        # Built once per update and reused by the residual and matrix
        # assemblies. The entry is replaced, never mutated, so the sv_start
        # snapshot keeps the operators of the state it was taken from.
        tangent = assembly.sv.get("_BeamTangentOperators")
        if tangent is None:
            tangent = beam_tangent_operators(self, assembly, data)
            assembly.sv["_BeamTangentOperators"] = tangent
        eps = self.space.op_beam_strain()
        stress = assembly.sv["BeamStress"]
        correction = tangent.strain_correction
        section_tangent = self._section_tangent(assembly)

        # 1. Strain change: eps(v).T D [B(delta_q-delta_u_local)+B_L q delta_L].
        if section_tangent is None:
            Ke = self.properties.get_beam_rigidity()
            strain_change = sum(
                eps[i].virtual * correction[i] * Ke[i]
                for i in range(6)
                if eps[i] != 0 and correction[i] != 0
            )
        else:
            strain_change = sum(
                eps[i].virtual * correction[j] * section_tangent[i, j]
                for i in range(6)
                for j in range(6)
                if eps[i] != 0 and correction[j] != 0 and np.any(section_tangent[i, j])
            )

        # 2. Virtual interpolation change: (B_L v_local).T s delta_L.
        interpolation_change = 0
        if tangent.interpolation_work != 0:
            interpolation_change = (
                tangent.interpolation_work.virtual * tangent.delta_length
            )

        # 3. Integration measure change: eps(v).T s delta_L/L.
        measure_change = 0
        if not np.array_equal(stress, 0):
            measure_change = sum(
                eps[i].virtual * tangent.relative_length * stress[i]
                for i in range(6)
                if eps[i] != 0
            )

        # 4. Frame rotation: (B delta_T v).T s = sum_j work_j(v) spin_j.
        frame_rotation = sum(
            work.virtual * spin
            for work, spin in zip(tangent.frame_work, tangent.frame_spin)
            if work != 0 and spin != 0
        )
        return strain_change + interpolation_change + measure_change + frame_rotation

    # Geometrically exact (Simo-Reissner) kinematics
    def _initialize_exact(self, assembly):
        if not assembly._nlgeom:
            raise ValueError(
                "kinematics='exact' is a nonlinear formulation: set nlgeom "
                "to 'UL' or 'TL'"
            )
        if assembly.elm_type != "beam":
            raise NotImplementedError(
                "kinematics='exact' requires the two-node 'beam' element, "
                f"got {assembly.elm_type!r}"
            )
        rigidity = self.properties.get_beam_rigidity()
        if any(np.isscalar(r) and r == 0 for r in rigidity[1:3]):
            raise ValueError(
                "kinematics='exact' is a shear deformable beam: set the shear "
                "coefficient k > 0 (for instance 5/6)"
            )
        if self.space.ndim == 3:
            assembly.sv["_NodesRotationMatrix"] = np.tile(
                np.eye(3), (assembly.mesh.n_nodes, 1, 1)
            )
        # The mesh is never updated: the exact strains are measured from the
        # reference configuration.
        self._update_exact(assembly, None)

    def _update_exact(self, assembly, pb):
        mesh = assembly.mesh
        ne, ng, dim = mesh.n_elements, assembly.n_elm_gp, self.space.ndim
        frame = np.asarray(assembly.get_element_local_frame())
        length = np.linalg.norm(
            mesh.nodes[mesh.elements[:, 1]] - mesh.nodes[mesh.elements[:, 0]], axis=1
        )
        disp = 0 if pb is None else pb.get_disp()
        if np.isscalar(disp) and disp == 0:
            u_local = np.zeros((ne, 2, dim))
        else:
            u_local = np.einsum("eij,enj->eni", frame, disp.T[mesh.elements])

        if dim == 2:
            rot = 0 if pb is None else pb.get_rot()
            rotations = (
                np.zeros((ne, 2))
                if np.isscalar(rot) and rot == 0
                else np.reshape(rot, -1)[mesh.elements]
            )
        else:
            nodes_rotmat = assembly.sv["_NodesRotationMatrix"]
            if pb is not None and not (np.isscalar(pb.get_X()) and pb.get_X() == 0):
                _, nodes_rotmat = compose_nodal_rotations(assembly, pb)
                if not pb._line_search_update:
                    commit_nodal_rotations(self.space, assembly, pb, nodes_rotmat)
            # Nodal triads in the reference element frame, where the reference
            # triads are the identity.
            rotations = (
                frame[:, None]
                @ nodes_rotmat[mesh.elements]
                @ frame[:, None].swapaxes(-1, -2)
            )

        # A line-search trial only evaluates the residual: skip d B.
        need_tangent = pb is None or not pb._line_search_update
        kinematics = exact_beam_kinematics(
            self.space, length, u_local, rotations, need_tangent
        )
        # Replaced, never mutated: sv_start keeps the kinematics of its state.
        assembly.sv["_BeamExactKinematics"] = (length, kinematics)
        assembly.sv["BeamStrain"] = BeamStrainList(
            [np.tile(kinematics.strain[:, i], ng) for i in range(6)]
        )
        if self.constitutivelaw is None:
            rigidity = self.properties.get_beam_rigidity()
            assembly.sv["BeamStress"] = BeamStressList(
                [rigidity[i] * assembly.sv["BeamStrain"][i] for i in range(6)]
            )

    def _get_exact_weak_equation(self, assembly, pb):
        length, kinematics = assembly.sv["_BeamExactKinematics"]
        ne, ng = assembly.mesh.n_elements, assembly.n_elm_gp
        stress = np.zeros((ng, ne, 6))
        values = assembly.sv["BeamStress"]
        if not (np.isscalar(values) and values == 0):
            for i, value in enumerate(values):
                stress[..., i] = np.broadcast_to(value, (ng * ne,)).reshape(ng, ne)
        # A line-search trial only evaluates the residual: skip the tangent.
        need_tangent = pb is None or not getattr(pb, "_line_search_update", False)
        tangent = self._exact_section_tangent(assembly) if need_tangent else None
        force, stiffness = element_operators(kinematics, stress, tangent)
        force, stiffness = to_field_basis(length, force, stiffness, self.space.nvar)

        basis = {}
        for variable in ("DispX", "DispY", "DispZ", "RotX", "RotY", "RotZ"):
            if variable in self.space.list_variables():
                rank = self.space.variable_rank(variable)
                basis[2 * rank] = self.space.variable("_BeamMean" + variable)
                basis[2 * rank + 1] = self.space.derivative("_BeamMean" + variable, "X")
        diff_op = 0
        for p, op in basis.items():
            if np.any(force[..., p]):
                diff_op += op.virtual * force[..., p].ravel()
        if stiffness is not None:
            for p, op_p in basis.items():
                for q, op_q in basis.items():
                    if np.any(stiffness[..., p, q]):
                        diff_op += op_p.virtual * op_q * stiffness[..., p, q].ravel()
        return diff_op

    def _exact_section_tangent(self, assembly):
        """Section tangent (ng, ne, 6, 6) at the Gauss points of the elements."""
        ne, ng = assembly.mesh.n_elements, assembly.n_elm_gp
        section_tangent = self._section_tangent(assembly)
        tangent = np.zeros((ng, ne, 6, 6))
        if section_tangent is None:
            rigidity = self.properties.get_beam_rigidity()
            for i in range(6):
                tangent[..., i, i] = np.broadcast_to(rigidity[i], (ng * ne,)).reshape(
                    ng, ne
                )
        else:
            for i in range(6):
                for j in range(6):
                    tangent[..., i, j] = np.broadcast_to(
                        section_tangent[i][j], (ng * ne,)
                    ).reshape(ng, ne)
        return tangent

    def _section_tangent(self, assembly):
        """Section tangent (6, 6, n_gp) of a section law, or None."""
        if self.constitutivelaw is None:
            return None
        return assembly.sv.get("_BeamStiffnessMatrix")

    def _use_consistent_tangent(self, assembly):
        return (
            self.consistent_tangent
            and self.geometric_stiffness is not False
            and assembly._nlgeom == "UL"
            and assembly.elm_type == "beam"
        )

    def _get_generalized_stress_op(self):
        # only for post treatment
        eps = self.space.op_beam_strain()
        Ke = self.properties.get_beam_rigidity()
        return [eps[i] * Ke[i] for i in range(6)]
