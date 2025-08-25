"""Rigid Tie constraint."""

import numpy as np
from fedoo.core.boundary_conditions import BCBase, MPC, ListBC


from scipy.spatial.transform import Rotation


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
    name : str, optional
        Name of the created boundary condition. The default is "Rigid Tie".


    Definition of rotations
    -----------------------
    A convention needs to be defined the orders of rotations.
    The convention used in this class is: First rotation around X, then
    rotation around Y' (Y' being the new Y afte r rotation around X)
    and finaly the rotation arould Z" (Z" beging the new Z after the 2 first
    rotations).

    We can note that this convention can also be interpreted using global axis
    not attached to the solid by applying first the rotation around Z, then the
    rotation around Y and finally, the rotation around X.


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

    def __init__(self, list_nodes, center=None, name="Rigid Tie"):
        self.list_nodes = list_nodes
        self.center = center
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
            ["RigidDispX", "RigidDispY", "RigidDispZ"], 1, "RidigDisp"
        )
        dof_indice_rot = problem.add_global_dof(
            ["RigidRotX", "RigidRotY", "RigidRotZ"], 1, "RidigRot"
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

    def generate(self, problem, t_fact=1, t_fact_old=None):
        mesh = problem.mesh
        var_cd = self.var_cd
        node_cd = self.node_cd
        list_nodes = self.list_nodes

        dof_cd = [
            problem.n_node_dof
            + problem._global_dof.indice_start(var_cd[i])
            + node_cd[i]
            for i in range(len(var_cd))
        ]

        if np.isscalar(problem.get_dof_solution()) and problem.get_dof_solution() == 0:
            dof_ref = np.array([problem._Xbc[dof] for dof in dof_cd])
        else:
            dof_ref = np.array(
                [problem.get_dof_solution()[dof] + problem._Xbc[dof] for dof in dof_cd]
            )

        disp_ref = dof_ref[:3]  # reference displacement
        angles = dof_ref[3:]  # rotation angle

        sin = np.sin(angles)
        cos = np.cos(angles)

        R = Rotation.from_euler("XYZ", angles).as_matrix()
        # #or
        # R = np.array([[cos[1]*cos[2], -cos[1]*sin[2], sin[1]],
        #           [cos[0]*sin[2] + cos[2]*sin[0]*sin[1], cos[0]*cos[2]-sin[0]*sin[1]*sin[2], -cos[1]*sin[0]],
        #           [sin[0]*sin[2] - cos[0]*cos[2]*sin[1], cos[2]*sin[0]+cos[0]*sin[1]*sin[2], cos[0]*cos[1]]] )

        # Correct displacement of slave nodes to be consistent with the master nodes
        new_disp = (
            (mesh.nodes[list_nodes] - self.center) @ R.T
            + self.center
            + disp_ref
            - mesh.nodes[list_nodes]
        )

        if not (np.array_equal(problem._dU, 0)):
            if np.array_equal(problem._U, 0):
                problem._dU[self._disp_indices] = new_disp
            else:
                problem._dU[self._disp_indices] = (
                    new_disp - problem._U[self._disp_indices]
                )

        # approche incrémentale:
        dR_drx = np.array(
            [
                [0, 0, 0],
                [
                    -sin[0] * sin[2] + cos[2] * cos[0] * sin[1],
                    -sin[0] * cos[2] - cos[0] * sin[1] * sin[2],
                    -cos[1] * cos[0],
                ],
                [
                    cos[0] * sin[2] + sin[0] * cos[2] * sin[1],
                    cos[2] * cos[0] - sin[0] * sin[1] * sin[2],
                    -sin[0] * cos[1],
                ],
            ]
        )

        dR_dry = np.array(
            [
                [-sin[1] * cos[2], +sin[1] * sin[2], cos[1]],
                [
                    cos[2] * sin[0] * cos[1],
                    -sin[0] * cos[1] * sin[2],
                    sin[1] * sin[0],
                ],
                [
                    -cos[0] * cos[2] * cos[1],
                    cos[0] * cos[1] * sin[2],
                    -cos[0] * sin[1],
                ],
            ]
        )

        dR_drz = np.array(
            [
                [-cos[1] * sin[2], -cos[1] * cos[2], 0],
                [
                    cos[0] * cos[2] - sin[2] * sin[0] * sin[1],
                    -cos[0] * sin[2] - sin[0] * sin[1] * cos[2],
                    0,
                ],
                [
                    sin[0] * cos[2] + cos[0] * sin[2] * sin[1],
                    -sin[2] * sin[0] + cos[0] * sin[1] * cos[2],
                    0,
                ],
            ]
        )

        crd = mesh.nodes[list_nodes] - self.center
        du_drx = crd @ dR_drx.T
        du_dry = crd @ dR_dry.T
        du_drz = (
            crd @ dR_drz.T
        )  # shape = (nnodes, nvar) with nvar = 3 in 3d (ux, uy, uz)

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
        if self.center is None:
            # initialize the rotation center at center of rigid nodes bounding box
            nodes_crd = problem.mesh.nodes[self.list_nodes]
            self.center = 0.5 * (nodes_crd.min(axis=0) + nodes_crd.max(axis=0))
        elif np.isscalar(self.center):
            # initialize the center at a position of a node
            self.center = problem.mesh.nodes[self.center]
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

    def generate(self, problem, t_fact=1, t_fact_old=None):
        mesh = problem.mesh
        var_cd = self.var_cd
        node_cd = self.node_cd
        list_nodes = self.list_nodes

        dof_cd = [
            problem.n_node_dof
            + problem._global_dof.indice_start(var_cd[i])
            + node_cd[i]
            for i in range(len(var_cd))
        ]

        if np.isscalar(problem.get_dof_solution()) and problem.get_dof_solution() == 0:
            dof_ref = np.array([problem._Xbc[dof] for dof in dof_cd])
        else:
            dof_ref = np.array(
                [problem.get_dof_solution()[dof] + problem._Xbc[dof] for dof in dof_cd]
            )

        disp_ref = dof_ref[:2]  # reference displacement
        angles = dof_ref[2]  # rotation Z angle

        sin = np.sin(angles)
        cos = np.cos(angles)

        # Correct displacement of slave nodes to be consistent with the master nodes
        R = np.array([[cos, -sin], [sin, cos]])

        new_disp = (
            (mesh.nodes[list_nodes] - self.center) @ R.T
            + self.center
            + disp_ref
            - mesh.nodes[list_nodes]
        )

        if not (np.array_equal(problem._dU, 0)):
            if np.array_equal(problem._U, 0):
                problem._dU[self._disp_indices] = new_disp
            else:
                problem._dU[self._disp_indices] = (
                    new_disp - problem._U[self._disp_indices]
                )

        # approche incrémentale:
        dR_drz = np.array([[-sin, -cos], [cos, -sin]])

        crd = mesh.nodes[list_nodes, :2] - self.center

        du_drz = (
            crd @ dR_drz.T
        )  # shape = (nnodes, nvar) with nvar = 3 in 3d (ux, uy, uz)

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
