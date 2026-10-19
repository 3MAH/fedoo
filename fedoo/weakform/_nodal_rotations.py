"""Multiplicative nodal rotation state shared by the beam and plate weak forms.

The Rot DOFs solved by Newton are spatial spin increments: a nodal rotation
is updated as R <- exp([dtheta]) R and stored in
``assembly.sv["_NodesRotationMatrix"]``.
"""

import numpy as np
from scipy.spatial.transform import Rotation


def compose_nodal_rotations(assembly, pb):
    """Trial nodal rotations: the last spatial spin increment composed on the left.

    Returns the nodal spin increment (3, n_nodes) and the trial rotation
    matrices (n_nodes, 3, 3). Nodes whose rotation is imposed by a boundary
    condition (:py:meth:`BCBase.get_nodal_rotations`) take it exactly.
    Nothing is stored: see commit_nodal_rotations.
    """
    delta_rotvec_nodes = pb._get_vect_component(pb.get_X(), "Rot")
    nodes_rotmat = (
        Rotation.from_rotvec(delta_rotvec_nodes.T).as_matrix()
        @ assembly.sv["_NodesRotationMatrix"]
    )
    for bc in pb.bc.list_all():
        imposed = bc.get_nodal_rotations(pb)
        if imposed is not None:
            nodes, rotmat = imposed
            nodes_rotmat[nodes] = rotmat
    return delta_rotvec_nodes, nodes_rotmat


def commit_nodal_rotations(space, assembly, pb, nodes_rotmat):
    """Store accepted nodal rotations and rewrite the Rot DOFs as total rotvecs.

    The Rot block of ``pb._dU`` is replaced by the rotation vector of the
    accepted nodal rotation relative to the start of the increment, and the
    Dirichlet corrections on Rot are recomputed accordingly. Must not be
    called for line-search trials.
    """
    assembly.sv["_NodesRotationMatrix"] = nodes_rotmat
    n_nodes = assembly.mesh.n_nodes
    rot_var = space.get_rank_vector("Rot")
    # Only valid if the Rot variables are contiguous in the variable order.
    rot_slice = slice(rot_var[0] * n_nodes, (rot_var[0] + 3) * n_nodes)
    if np.isscalar(pb._U) and pb._U == 0:
        pb._dU[rot_slice] = Rotation.from_matrix(nodes_rotmat).as_rotvec().T.ravel()
    else:
        nodes_rotmat_from_U = Rotation.from_rotvec(
            pb._U[rot_slice].reshape(3, -1).T
        ).as_matrix()
        # d_angle = final_angle - angle_start, computed on rotation matrices
        # (a difference of rotation vectors is singular beyond pi).
        pb._dU[rot_slice] = (
            Rotation.from_matrix(nodes_rotmat @ nodes_rotmat_from_U.transpose(0, 2, 1))
            .as_rotvec()
            .T.ravel()
        )

    for bc in pb.bc.list_all():
        if bc.bc_type == "Dirichlet" and bc.variable in rot_var:
            pb._Xbc[bc._dof_index] = (
                bc.get_true_value(pb.t_fact) - pb.get_dof_solution()[bc._dof_index]
            )
