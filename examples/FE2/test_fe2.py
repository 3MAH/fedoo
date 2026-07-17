# FE2 light test file

import numpy as np

import fedoo as fd
from fedoo.util.voigt_tensors import StrainTensorList


def test_fe2_uses_problem_level_mean_strain_dofs():
    fd.ModelingSpace("3D")

    micro_mesh = fd.mesh.box_mesh(nx=2, ny=2, nz=2)
    micro_material = fd.constitutivelaw.ElasticIsotrop(200_000.0, 0.3)
    micro_weakform = fd.weakform.StressEquilibrium(micro_material)
    micro_assembly = fd.Assembly.create(micro_weakform, micro_mesh, n_elm_gp=1)

    fe2 = fd.constitutivelaw.FE2(micro_assembly)
    macro_mesh = fd.mesh.box_mesh(nx=2, ny=2, nz=2)
    macro_weakform = fd.weakform.StressEquilibrium(fe2)
    macro_assembly = fd.Assembly.create(macro_weakform, macro_mesh, n_elm_gp=1)
    problem = fd.problem.NonLinear(macro_assembly)

    problem.initialize()

    assert "_StrainNodes" not in micro_mesh.node_sets
    assert "MeanStrain" in fe2.list_problem[0].global_dof._vector
    expected = micro_material.get_elastic_matrix()
    np.testing.assert_allclose(
        macro_assembly.sv["TangentMatrix"][:, :, 0],
        expected,
        rtol=1e-10,
        atol=1e-8,
    )

    imposed_strain = np.array([1e-4, 0, 0, 0, 0, 0])
    macro_assembly.sv["Strain"] = StrainTensorList(imposed_strain[:, None])
    problem.dtime = 1.0
    fe2._update_pb(0, macro_assembly, problem)

    np.testing.assert_allclose(
        macro_assembly.sv["Stress"].asarray()[:, 0],
        expected @ imposed_strain,
        rtol=1e-8,
        atol=1e-8,
    )


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
