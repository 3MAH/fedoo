"""Rigid Tie constraint."""

import warnings

import numpy as np
from fedoo.core.boundary_conditions import BCBase, MPC, ListBC
from simcoon import Rotation
from simcoon import dR_drotvec


class RigidTie(BCBase):
    """Constraint that eliminate dof assuming a rigid body tie between nodes.

    Create an object that defines a rigid tie coupling between some nodes using
    several multi-points constraints. Some constraint drivers (cd) dof are
    used to define rigid body displacement and rotation. By default, the center of
    rotation is located at the center of the bounding box defined by all nodes
    tied together. The rotation center move with the rigid displacement.
    RigidTie constraint add 6 global_dof to the problem:
    "RigidDispX", "RigidDispY", "RigidDispZ", "RigidRotX", "RigidRotY", "RigidRotZ"
    and two global_dot vectors:

    * "RigidDisp" = ["RigidDispX", "RigidDispY", "RigidDispZ"] for rigid displacement
    * "RigidRot" = ["RigidRotX", "RigidRotY", "RigidRotZ"] for rigid rotation

    If several RigidTime constraints are used with the same problem, all the previous
    variables will contains several dof, the indice of the dof being associated to
    order in which the constraint has been added.
    For instance, pb.global_dof['RigidDispX'][0] will be associated to the first
    added RigidTie and dof['RigidDispX'][1] to the second one.


    Parameters
    ----------
    list_nodes: list (int) or 1d np.array
        List of nodes that will be eliminated considering a rigid body tie.
    center: int of np.array[float] with shape = (3), optional
        If center is an int, the rotation center will be initialized at the coordinates
        of the corresponding node in the mesh. If center is an array (or list, ...)
        it contains the initial coordinates of the rotation center. By default, the
        center is set at the midle of the rigid sets.
    use_quaternion : bool, optional
        If ``True`` (default), the rotational DOFs describe an incremental
        rotation vector relative to the last converged orientation. The
        increment is composed multiplicatively and the converged orientation
        is stored as a quaternion. If ``False``, the DOFs are interpreted as
        one total rotation vector.
    name : str, optional
        Name of the created boundary condition. The default is "Rigid Tie".


    Definition of rotations
    -----------------------
    ``RigidRotX``, ``RigidRotY`` and ``RigidRotZ`` use a rotation-vector
    convention. The vector direction defines the rotation axis and its norm is
    the rotation angle. With ``use_quaternion=True``, this rotation vector is
    the current increment and is composed with the last converged orientation;
    the quaternion is an orientation storage format, not a different DOF
    convention.


    Notes
    -----
    * The node given in list_nodes are eliminated from the system (slave nodes)
      and can't be used in other boundary conditions. The boundary conditions should
      be enforce using the added global dof.
    * The rigid coupling is highly non-linear and the multi-point constraints
      are modified at each iteration.
    * Once created the RigidTie object needs to be associated to the problem
      using the Problem.add method.


    Example
    -------

    .. code-block:: python

        import fedoo as fd

        mesh = fd.mesh.box_mesh()

        left_face = mesh.find_nodes('X', mesh.bounding_box.xmin)
        right_face = mesh.find_nodes('X', mesh.bounding_box.xmax)

        rigid_tie = fd.constraint.RigidTie(right_face)
    """

    def __init__(self, list_nodes, center=None, use_quaternion=True, name="Rigid Tie"):
        self.list_nodes = list_nodes
        self.center = center
        self.use_quaternion = bool(use_quaternion)
        self.bc_type = "RigidTie"
        BCBase.__init__(self, name)
        self._keep_at_end = True

        self._update_during_inc = 1

    def __repr__(self):
        list_str = ["Rigid Tie:"]
        if self.name != "":
            list_str.append("name = '{}'".format(self.name))

        return "\n".join(list_str)

    def initialize(self, problem):
        if self.center is None:
            # initialize the rotation center at center of rigid nodes bounding box
            nodes_crd = problem.mesh.nodes[self.list_nodes]
            self.center = 0.5 * (nodes_crd.min(axis=0) + nodes_crd.max(axis=0))
        elif np.isscalar(self.center):
            # initialize the center at a position of a node
            self.center = problem.mesh.nodes[self.center]
        else:
            self.center = np.asarray(self.center)

        dof_indice_disp = problem.add_global_dof(
            ["RigidDispX", "RigidDispY", "RigidDispZ"], 1, "RigidDisp"
        )
        dof_indice_rot = problem.add_global_dof(
            ["RigidRotX", "RigidRotY", "RigidRotZ"], 1, "RigidRot"
        )
        self.var_cd = [
            "RigidDispX",
            "RigidDispY",
            "RigidDispZ",
            "RigidRotX",
            "RigidRotY",
            "RigidRotZ",
        ]
        self.node_cd = [
            dof_indice_disp,
            dof_indice_disp,
            dof_indice_disp,
            dof_indice_rot,
            dof_indice_rot,
            dof_indice_rot,
        ]

        # extract indices array that gives the disp from the full dof solution
        n_nodes = problem.mesh.n_nodes
        rank = problem.space.variable_rank("DispX")
        # rank = rank of variable "DispX". rank of "DispY" and "DispZ" should follow
        self._disp_indices = (
            np.c_[rank * n_nodes, (rank + 1) * n_nodes, (rank + 2) * n_nodes]
            + self.list_nodes[:, None]
        )

        if self.use_quaternion:
            self._Q_base = Rotation.identity()
            self._angles_at_base = np.zeros(3)

    def _get_dof_ref(self, problem):
        """Return the current six rigid-body DOFs and their global indices."""
        dof_cd = [
            problem.n_node_dof
            + problem._global_dof.indice_start(self.var_cd[i])
            + self.node_cd[i]
            for i in range(6)
        ]
        dof_sol = problem.get_dof_solution()
        xbc = problem._Xbc

        if np.isscalar(dof_sol) and dof_sol == 0:
            if np.isscalar(xbc) and xbc == 0:
                return np.zeros(6), dof_cd
            return np.asarray(xbc)[dof_cd], dof_cd
        if np.isscalar(xbc) and xbc == 0:
            return np.asarray(dof_sol)[dof_cd], dof_cd
        return np.asarray(dof_sol)[dof_cd] + np.asarray(xbc)[dof_cd], dof_cd

    def _compute_rotation(self, rotvec):
        """Return the orientation and exact derivatives for the active mode."""
        rotvec = np.asarray(rotvec, dtype=float)
        if self.use_quaternion and hasattr(self, "_Q_base"):
            increment = rotvec - self._angles_at_base
            if not np.all(np.isfinite(increment)):
                increment = np.zeros(3)
            R_base = self._Q_base.as_matrix()
        else:
            increment = rotvec
            R_base = np.eye(3)

        R_increment = Rotation.from_rotvec(increment).as_matrix()
        derivatives = dR_drotvec(increment)
        R = R_increment @ R_base
        return (
            R,
            derivatives[:, :, 0] @ R_base,
            derivatives[:, :, 1] @ R_base,
            derivatives[:, :, 2] @ R_base,
        )

    def _compute_slave_disp(self, problem, disp_ref, R):
        """Update slave-node displacements for a given rigid motion."""
        list_nodes = self.list_nodes
        mesh = problem.mesh
        new_disp = (
            (mesh.nodes[list_nodes] - self.center) @ R.T
            + self.center
            + disp_ref
            - mesh.nodes[list_nodes]
        )

        if not np.array_equal(problem._dU, 0):
            if np.array_equal(problem._U, 0):
                problem._dU[self._disp_indices] = new_disp
            else:
                problem._dU[self._disp_indices] = (
                    new_disp - problem._U[self._disp_indices]
                )
        return new_disp

    def pre_update(self, problem):
        """Refresh rigid slave positions before contact assemblies update."""
        dof_ref, _ = self._get_dof_ref(problem)
        R, _, _, _ = self._compute_rotation(dof_ref[3:])
        self._compute_slave_disp(problem, dof_ref[:3], R)

    def set_start(self, problem):
        """Commit a converged incremental rotation to the quaternion state."""
        if not self.use_quaternion or not hasattr(self, "_Q_base"):
            return
        dof_ref, _ = self._get_dof_ref(problem)
        angles = np.asarray(dof_ref[3:], dtype=float)
        if not np.all(np.isfinite(angles)):
            return
        increment = angles - self._angles_at_base
        if not np.allclose(increment, 0.0, atol=1e-15):
            self._Q_base = Rotation.from_rotvec(increment) * self._Q_base
            self._angles_at_base = angles.copy()

    def to_start(self, problem):
        """Leave the last converged quaternion unchanged after a failed step."""
        return

    @property
    def Q_total(self):
        """Last converged quaternion orientation, or ``None`` in total mode."""
        if self.use_quaternion and hasattr(self, "_Q_base"):
            return self._Q_base
        return None

    def generate(self, problem, t_fact=1, t_fact_old=None):
        mesh = problem.mesh
        var_cd = self.var_cd
        node_cd = self.node_cd
        list_nodes = self.list_nodes

        dof_ref, dof_cd = self._get_dof_ref(problem)

        disp_ref = dof_ref[:3]  # reference displacement
        rotvec = dof_ref[3:]

        R, dR_drx, dR_dry, dR_drz = self._compute_rotation(rotvec)

        self._compute_slave_disp(problem, disp_ref, R)

        # approche incrémentale:
        crd = mesh.nodes[list_nodes] - self.center
        du_drx = crd @ dR_drx.T
        du_dry = crd @ dR_dry.T
        du_drz = crd @ dR_drz.T

        #### MPC ####

        # dU - dU_ref - du_drx*drx_ref - du_dry*dry_ref - du_drz*drz_ref = 0
        # with shapes: dU, du_drx, ... -> (nnodes, nvar) - dU_ref -> (nvar), drx_ref, ... -> scalar
        # dU are associated to eliminated dof and should be different than ref dof
        # or
        # dUx - dUx_ref - du_drx[:,0]*drx_ref - du_dry[:,0]*dry_ref - du_drz[:,0]*drz_ref = 0
        # dUy - dUy_ref - du_drx[1]*drx_ref - du_dry[1]*dry_ref - du_drz[1]*drz_ref = 0
        # dUz - dUz_ref - du_drx[2]*drx_ref - du_dry[2]*dry_ref - du_drz[2]*drz_ref = 0
        res = ListBC()
        res.append(
            MPC(
                [
                    list_nodes,
                    np.full_like(list_nodes, node_cd[0]),
                    np.full_like(list_nodes, node_cd[3]),
                    np.full_like(list_nodes, node_cd[4]),
                    np.full_like(list_nodes, node_cd[5]),
                ],
                ["DispX", var_cd[0], var_cd[3], var_cd[4], var_cd[5]],
                [
                    np.full_like(list_nodes, 1.0),
                    np.full_like(list_nodes, -1.0),
                    -du_drx[:, 0],
                    -du_dry[:, 0],
                    -du_drz[:, 0],
                ],
            )
        )
        res.append(
            MPC(
                [
                    list_nodes,
                    np.full_like(list_nodes, node_cd[1]),
                    np.full_like(list_nodes, node_cd[3]),
                    np.full_like(list_nodes, node_cd[4]),
                    np.full_like(list_nodes, node_cd[5]),
                ],
                ["DispY", var_cd[1], var_cd[3], var_cd[4], var_cd[5]],
                [
                    np.full_like(list_nodes, 1.0),
                    np.full_like(list_nodes, -1.0),
                    -du_drx[:, 1],
                    -du_dry[:, 1],
                    -du_drz[:, 1],
                ],
            )
        )
        res.append(
            MPC(
                [
                    list_nodes,
                    np.full_like(list_nodes, node_cd[2]),
                    np.full_like(list_nodes, node_cd[3]),
                    np.full_like(list_nodes, node_cd[4]),
                    np.full_like(list_nodes, node_cd[5]),
                ],
                ["DispZ", var_cd[2], var_cd[3], var_cd[4], var_cd[5]],
                [
                    np.full_like(list_nodes, 1.0),
                    np.full_like(list_nodes, -1.0),
                    -du_drx[:, 2],
                    -du_dry[:, 2],
                    -du_drz[:, 2],
                ],
            )
        )

        res.initialize(problem)
        return res.generate(problem, t_fact, t_fact_old)


class RigidTie2D(BCBase):
    """Constraint that eliminate dof assuming a rigid body tie between nodes in 2D.

    Same constraint as RigidTie, but for 2D problems.
    See RigidTie documentation for more details.
    """

    def __init__(self, list_nodes, center=None, name="Rigid Tie 2D"):
        self.list_nodes = list_nodes
        self.center = center
        self.bc_type = "RigidTie2D"
        BCBase.__init__(self, name)
        self._keep_at_end = True

        self._update_during_inc = 1

    def __repr__(self):
        list_str = ["Rigid Tie 2D:"]
        if self.name != "":
            list_str.append("name = '{}'".format(self.name))

        return "\n".join(list_str)

    def initialize(self, problem):
        if problem.space.is_axisymmetric:
            warnings.warn(
                "RigidTie2D under the '2Daxi' ModelingSpace: the constraint "
                "is applied as pure-kinematics MPCs and will function, but "
                "only axial (z) translation preserves axisymmetry. Radial "
                "translation and the in-plane RigidRotZ rotation break it.",
                stacklevel=2,
            )
        if self.center is None:
            # initialize the rotation center at center of rigid nodes bounding box
            nodes_crd = problem.mesh.nodes[self.list_nodes]
            self.center = 0.5 * (nodes_crd.min(axis=0) + nodes_crd.max(axis=0))
        elif np.isscalar(self.center):
            # initialize the center at a position of a node
            self.center = problem.mesh.nodes[self.center]
        else:
            self.center = np.asarray(self.center)

        dof_indice_disp = problem.add_global_dof(
            ["RigidDispX", "RigidDispY"], 1, "RidigDisp"
        )
        dof_indice_rot = problem.add_global_dof(["RigidRotZ"], 1, "RidigRot")
        self.var_cd = [
            "RigidDispX",
            "RigidDispY",
            "RigidRotZ",
        ]
        self.node_cd = [dof_indice_disp, dof_indice_disp, dof_indice_rot]

        # extract indices array that gives the disp from the full dof solution
        n_nodes = problem.mesh.n_nodes
        rank = problem.space.variable_rank("DispX")
        # rank = rank of variable "DispX". rank of "DispY" should be rank+1
        self._disp_indices = (
            np.c_[rank * n_nodes, (rank + 1) * n_nodes] + self.list_nodes[:, None]
        )

    def _get_dof_ref(self, problem):
        """Return [dx, dy, rotZ] and their global DOF indices."""
        dof_cd = [
            problem.n_node_dof
            + problem._global_dof.indice_start(self.var_cd[i])
            + self.node_cd[i]
            for i in range(3)
        ]
        dof_sol = problem.get_dof_solution()
        xbc = problem._Xbc

        if np.isscalar(dof_sol) and dof_sol == 0:
            if np.isscalar(xbc) and xbc == 0:
                return np.zeros(3), dof_cd
            return np.asarray(xbc)[dof_cd], dof_cd
        if np.isscalar(xbc) and xbc == 0:
            return np.asarray(dof_sol)[dof_cd], dof_cd
        return np.asarray(dof_sol)[dof_cd] + np.asarray(xbc)[dof_cd], dof_cd

    @staticmethod
    def _compute_rotation(angle):
        """Return the 2D rotation matrix and its angle derivative."""
        sin = np.sin(angle)
        cos = np.cos(angle)
        rotation = np.array([[cos, -sin], [sin, cos]])
        derivative = np.array([[-sin, -cos], [cos, -sin]])
        return rotation, derivative

    def _compute_slave_disp(self, problem, disp_ref, rotation):
        """Update 2D slave-node displacements for a rigid motion."""
        nodes = problem.mesh.nodes[self.list_nodes]
        new_disp = (nodes - self.center) @ rotation.T + self.center + disp_ref - nodes
        if not np.array_equal(problem._dU, 0):
            if np.array_equal(problem._U, 0):
                problem._dU[self._disp_indices] = new_disp
            else:
                problem._dU[self._disp_indices] = (
                    new_disp - problem._U[self._disp_indices]
                )
        return new_disp

    def pre_update(self, problem):
        """Refresh rigid slave positions before contact assemblies update."""
        dof_ref, _ = self._get_dof_ref(problem)
        rotation, _ = self._compute_rotation(dof_ref[2])
        self._compute_slave_disp(problem, dof_ref[:2], rotation)

    def generate(self, problem, t_fact=1, t_fact_old=None):
        var_cd = self.var_cd
        node_cd = self.node_cd
        list_nodes = self.list_nodes

        dof_ref, _ = self._get_dof_ref(problem)

        disp_ref = dof_ref[:2]  # reference displacement
        rotation, dR_drz = self._compute_rotation(dof_ref[2])
        # Correct displacement of slave nodes to be consistent with the masters
        self._compute_slave_disp(problem, disp_ref, rotation)
        # approche incrémentale:

        # MPC linearization
        crd = problem.mesh.nodes[list_nodes, :2] - self.center

        du_drz = crd @ dR_drz.T  # shape = (nnodes, 2)

        #### MPC ####

        # dU - dU_ref - du_drx*drx_ref - du_dry*dry_ref - du_drz*drz_ref = 0
        # with shapes: dU, du_drx, ... -> (nnodes, nvar) - dU_ref -> (nvar), drx_ref, ... -> scalar
        # dU are associated to eliminated dof and should be different than ref dof
        # or
        # dUx - dUx_ref - du_drx[:,0]*drx_ref - du_dry[:,0]*dry_ref - du_drz[:,0]*drz_ref = 0
        # dUy - dUy_ref - du_drx[1]*drx_ref - du_dry[1]*dry_ref - du_drz[1]*drz_ref = 0
        # dUz - dUz_ref - du_drx[2]*drx_ref - du_dry[2]*dry_ref - du_drz[2]*drz_ref = 0
        res = ListBC()
        res.append(
            MPC(
                [
                    list_nodes,
                    np.full_like(list_nodes, node_cd[0]),
                    np.full_like(list_nodes, node_cd[2]),
                ],
                ["DispX", var_cd[0], var_cd[2]],
                [
                    np.full_like(list_nodes, 1.0),
                    np.full_like(list_nodes, -1.0),
                    -du_drz[:, 0],
                ],
            )
        )
        res.append(
            MPC(
                [
                    list_nodes,
                    np.full_like(list_nodes, node_cd[1]),
                    np.full_like(list_nodes, node_cd[2]),
                ],
                ["DispY", var_cd[1], var_cd[2]],
                [
                    np.full_like(list_nodes, 1.0),
                    np.full_like(list_nodes, -1.0),
                    -du_drz[:, 1],
                ],
            )
        )

        res.initialize(problem)
        return res.generate(problem, t_fact, t_fact_old)
