import numpy as np

import fedoo as fd


def test_lin2interface_with_cohesive_law():
    fd.ModelingSpace("2D")

    nodes = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
        ]
    )
    mesh = fd.Mesh(
        nodes,
        np.array([[0, 1, 2, 3]]),
        "lin2interface",
    )
    law = fd.constitutivelaw.CohesiveLaw(
        axis=1,
        tangent_mode="consistent",
    )
    assembly = fd.Assembly.create(
        fd.weakform.InterfaceForce(law),
        mesh,
    )
    problem = fd.problem.NonLinear(assembly)
    problem.bc.add("Dirichlet", [0, 1], "Disp", 0.0)
    problem.bc.add("Dirichlet", [2, 3], "DispX", 0.0)
    problem.bc.add("Dirichlet", [2, 3], "DispY", 0.008)

    problem.nlsolve(
        dt=1.0,
        tmax=1.0,
        tol_nr=1.0e-8,
        print_info=0,
    )

    assert np.allclose(problem.get_disp()[1, 2:], 0.008)
    assert np.allclose(assembly.sv["DamageVariable"], 0.625)
    tangent = assembly.sv["TangentMatrix"]
    assert np.asarray(tangent[0][0]).shape == (2,)
    assert np.asarray(tangent[1][1]).shape == (2,)
