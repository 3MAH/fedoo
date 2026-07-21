import numpy as np
from scipy import sparse

import fedoo as fd


def test_rigid_tie_2d_helpers_and_array_like_center():
    space = fd.ModelingSpace("2D")
    space.new_variable("DispX")
    space.new_variable("DispY")
    space.new_vector("Disp", ("DispX", "DispY"))

    nodes = np.array(
        [
            [-1.0, -0.5],
            [1.0, -0.5],
            [1.0, 0.5],
            [-1.0, 0.5],
        ]
    )
    mesh = fd.Mesh(nodes, np.array([[0, 1, 2, 3]]), "quad4")
    problem = fd.Problem(
        A=sparse.eye(mesh.n_nodes * space.nvar), mesh=mesh, space=space
    )

    center = [0.2, -0.1]
    tied_nodes = np.arange(mesh.n_nodes)
    control = fd.constraint.RigidTie2D(tied_nodes, center=center)
    problem.bc.add(control)
    problem.bc.add("Dirichlet", "RigidDispX", 0.3)
    problem.bc.add("Dirichlet", "RigidDispY", -0.2)
    problem.bc.add("Dirichlet", "RigidRotZ", 0.4)
    problem._U = np.zeros(problem.n_dof)
    problem._dU = np.zeros(problem.n_dof)

    problem.apply_boundary_conditions()

    assert isinstance(control.center, np.ndarray)
    rotation, _ = control._compute_rotation(0.4)
    expected = (
        (nodes - control.center) @ rotation.T
        + control.center
        + np.array([0.3, -0.2])
        - nodes
    )
    np.testing.assert_allclose(problem._dU[control._disp_indices], expected)

    problem._dU[:] = 0.0
    control.pre_update(problem)
