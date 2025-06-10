# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 15:27:43 2020

@author: Etienne
"""

# from fedoo.core.base   import ProblemBase
import numpy as np
from fedoo.core.boundary_conditions import BCBase, MPC, ListBC
from fedoo.core.base import ProblemBase
from fedoo.core.mesh import MeshBase


from scipy.spatial.transform import Rotation


class RigidTie(BCBase):
    """
    Boundary conditions class that eliminate dof assuming a rigid body tie between nodes

    Create an object that defines a rigid tie coupling between some nodes using several multi-points constraints.
    Some constraint drivers (cd) dof  are used to define rigid body displacement and rotation.
    The center of rotation is assumed to be at the position of the first node given in the constraint driver nodes (node_cd[0])
    The dof associated to a contraint driver is difined by the node indice (defined in node_cd)
    and the associated variable (defined in var_vd) with the following order:
    [DispX, DispY, DispZ, RotX, RotY, RotZ] where:

    * DispX, DispY and DispZ are the displacement of the reference node of the
      rigid group (reference node = node_cd[0])
    * RotX, RotY and RotZ are the rigid rotation around the 3 axes with the reference node
      being the center of rotation. The rotation axis are attached to the solid.


    Parameters
    ----------
    list_nodes : list (int) or 1d np.array
        list of nodes that will be eliminated considering a rigid body tie.
    node_cd : list of int
        Nodes used as constraint drivers for each rigid displacement and rotation.
        The associated dof used as contraint drivers are defined in var_cd.
        len(node_cd) should be 6 because 6 constraint driver dof are required.
    var_cd : list of str.
        Variables used as constraint drivers.
        The len of the list should be the same as node_cd (ie 6).
    name : str, optional
        Name of the created boundary condition. The default is "Rigid Tie".


    Definition of rotations
    ----------------------------
    A convention needs to be defined the orders of rotations.
    The convention used in this class is: First rotation around X, then
    rotation around Y' (Y' being the new Y afte r rotation around X)
    and finaly the rotation arould Z" (Z" beging the new Z after the 2 first rotations).

    We can note that this convention can also be interpreted using global axis not attached to the solid
    by applying first the rotation around Z, then the rotation around Y and finally, the rotation around X.


    Notes
    ---------------

    * The node given in list_nodes are eliminated from the system (slave nodes)
      and can't be used in another mpc.
    * The rigid coupling is highly non-linear and the multi-point constraints are
      modified at each iteration.
    * Once created the RigidTie object needs to be associated to the problem using the Problem.add method.


    Example
    ---------

    .. code-block:: python

        import fedoo as fd

        mesh = fd.mesh.box_mesh()

        #add nodes not associated to any element for constraint driver
        node_cd = mesh.add_virtual_nodes(2)
        var_cd = ['DispX', 'DispY', 'DispZ', 'DispX', 'DispY', 'DispZ']

        left_face = mesh.find_nodes('X', mesh.bounding_box.xmin)
        right_face = mesh.find_nodes('X', mesh.bounding_box.xmax)

        rigid_tie = fd.constraint.RigidTie(right_face, node_cd, var_cd)
    """

    def __init__(self, list_nodes, node_cd, var_cd, name="Rigid Tie"):
        self.list_nodes = list_nodes
        self.node_cd = node_cd
        self.var_cd = var_cd
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
        pass
        # for i,var in enumerate(self.var_cd):
        #     if isinstance(var, str):
        #         self.var_cd[i] = problem.space.variable_rank(var)

    def generate(self, problem, t_fact=1, t_fact_old=None):
        mesh = problem.mesh
        var_cd = self.var_cd
        node_cd = self.node_cd  # node_cd[0] -> node defining center of rotation
        list_nodes = self.list_nodes

        # rot_center = node_cd[0]
        res = ListBC()

        dof_cd = [
            problem.space.variable_rank(var_cd[i]) * mesh.n_nodes + node_cd[i]
            for i in range(len(var_cd))
        ]

        # dof_ref  = [problem._Xbc[dof] if dof in problem.dof_blocked else problem._X[dof] for dof in dof_cd]
        # dof_ref  = [problem.get_dof_solution()[dof] + problem._Xbc[dof] if dof in problem.dof_blocked else problem.get_dof_solution()[dof] for dof in dof_cd]
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
        # R2 = np.array([[cos[1]*cos[2], -cos[1]*sin[2], sin[1]],
        #           [cos[0]*sin[2] + cos[2]*sin[0]*sin[1], cos[0]*cos[2]-sin[0]*sin[1]*sin[2], -cos[1]*sin[0]],
        #           [sin[0]*sin[2] - cos[0]*cos[2]*sin[1], cos[2]*sin[0]+cos[0]*sin[1]*sin[2], cos[0]*cos[1]]] )

        # approche globale :
        # crd = mesh.nodes + problem.get_disp()
        # Uini = (crd - crd[0]) @ R.T + disp_ref #node disp at the begining of the iteration

        # Correct displacement of slave nodes to be consistent with the master nodes
        new_disp = (
            (mesh.nodes[list_nodes] - mesh.nodes[node_cd[0]]) @ R.T
            + mesh.nodes[node_cd[0]]
            + disp_ref
            - mesh.nodes[list_nodes]
        )

        if not (np.array_equal(problem._dU, 0)):
            if np.array_equal(problem._U, 0):
                problem._dU.reshape(3, -1)[:, list_nodes] = new_disp.T
            else:
                problem._dU.reshape(3, -1)[:, list_nodes] = (
                    new_disp.T - problem._U.reshape(3, -1)[:, list_nodes]
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

        crd = mesh.nodes[list_nodes] - mesh.nodes[node_cd[0]]
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


# not tested class
class RigidTie2D(BCBase):
    """Boundary conditions class that eliminate dof assuming a rigid body tie between nodes in 2d problem"""

    def __init__(self, list_nodes, node_cd, var_cd, name="Rigid Tie 2D"):
        """
        Same constraint as RigidTie, but for 2D problems.
        In this case, only 3 constraint driver needs to be defined:
            ['DispX','DispY', 'RotZ']

        See RigidTie documentation for more details.
        """

        self.list_nodes = list_nodes
        self.node_cd = node_cd
        self.var_cd = var_cd
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
        pass
        # for i,var in enumerate(self.var_cd):
        #     if isinstance(var, str):
        #         self.var_cd[i] = problem.space.variable_rank(var)

    def generate(self, problem, t_fact=1, t_fact_old=None):
        mesh = problem.mesh
        var_cd = self.var_cd
        node_cd = self.node_cd  # node_cd[0] -> node defining center of rotation
        list_nodes = self.list_nodes

        # rot_center = node_cd[0]
        res = ListBC()

        dof_cd = [
            problem.space.variable_rank(var_cd[i]) * mesh.n_nodes + node_cd[i]
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
            (mesh.nodes[list_nodes] - mesh.nodes[node_cd[0]]) @ R.T
            + mesh.nodes[node_cd[0]]
            + disp_ref
            - mesh.nodes[list_nodes]
        )

        if not (np.array_equal(problem._dU, 0)):
            if np.array_equal(problem._U, 0):
                problem._dU.reshape(2, -1)[:, list_nodes] = new_disp.T
            else:
                problem._dU.reshape(2, -1)[:, list_nodes] = (
                    new_disp.T - problem._U.reshape(2, -1)[:, list_nodes]
                )

        # approche incrémentale:

        dR_drz = np.array([[-sin, -cos], [cos, -sin]])

        crd = mesh.nodes[list_nodes, :2] - mesh.nodes[node_cd[0], :2]

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
