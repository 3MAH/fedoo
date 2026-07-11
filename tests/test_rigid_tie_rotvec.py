import numpy as np
from scipy import sparse
from simcoon import Rotation

import fedoo as fd


def test_rigid_tie_rotvec_matches_single_component_x_torsion():
    space = fd.ModelingSpace("3D")
    space.new_variable("DispX")
    space.new_variable("DispY")
    space.new_variable("DispZ")
    space.new_vector("Disp", ("DispX", "DispY", "DispZ"))

    nodes = np.array(
        [
            [0.0, -1.0, -1.0],
            [0.0, 1.0, -1.0],
            [0.0, 1.0, 1.0],
            [0.0, -1.0, 1.0],
        ]
    )
    mesh = fd.Mesh(nodes, np.array([[0, 1, 2, 3]]), "quad4")
    pb = fd.Problem(A=sparse.eye(mesh.n_nodes * space.nvar), mesh=mesh, space=space)

    tied_nodes = np.arange(mesh.n_nodes)
    control = fd.constraint.RigidTie(tied_nodes)
    pb.bc.add(control)
    angle = 0.8
    pb.bc.add("Dirichlet", "RigidRotX", angle)
    pb._U = np.zeros(pb.n_dof)
    pb._dU = np.zeros(pb.n_dof)

    pb.apply_boundary_conditions()

    center = 0.5 * (nodes.min(axis=0) + nodes.max(axis=0))
    rotation = Rotation.from_rotvec([angle, 0.0, 0.0]).as_matrix()
    expected = (nodes - center) @ rotation.T + center - nodes

    np.testing.assert_allclose(pb._dU[control._disp_indices], expected, atol=1e-12)
