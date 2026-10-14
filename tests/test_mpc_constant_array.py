"""Several mpc defined at once with a different constant for each equation.

The constraint equation is sum(factor * dof) + constant = 0.
"""

import numpy as np
import pytest

import fedoo as fd


def test_mpc_with_one_constant_per_equation():
    fd.Assembly.delete_memory()
    fd.ModelingSpace("3D")

    mesh = fd.mesh.box_mesh(2, 2, 2, name="mpc_box")
    material = fd.constitutivelaw.ElasticIsotrop(1000.0, 0.3, name="MpcMat")
    wf = fd.weakform.StressEquilibrium(material)
    solid = fd.Assembly.create(wf, mesh, name="mpc_solid")
    pb = fd.problem.Linear(solid)

    box = mesh.bounding_box
    top = mesh.find_nodes("Z", box.zmax)
    slaves, masters = top[:2], top[2:4]
    constant = np.array([0.02, -0.05])

    pb.bc.add("Dirichlet", mesh.find_nodes("Z", box.zmin), "Disp", 0)
    pb.bc.mpc(
        [slaves, masters],
        ["DispX", "DispX"],
        [np.ones(2), -np.ones(2)],
        constant=constant,
    )
    pb.solve()

    ux = pb.get_dof_solution("DispX")
    np.testing.assert_allclose(ux[slaves] - ux[masters], -constant, atol=1e-12)


if __name__ == "__main__":
    pytest.main([__file__])
