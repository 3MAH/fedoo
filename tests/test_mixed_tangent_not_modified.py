"""The finite-strain mixed weak form must not modify the stored tangent."""

import numpy as np
import pytest

import fedoo as fd
from fedoo.util.voigt_tensors import StressTensorList


@pytest.mark.parametrize("nlgeom", ["TL", "UL"])
def test_mixed_weak_equation_keeps_the_stored_tangent(nlgeom):
    fd.ModelingSpace("3D")
    mesh = fd.mesh.box_mesh(nx=3, ny=3, nz=3, elm_type="hex8")
    material = fd.constitutivelaw.ElasticIsotrop(1.0, 0.3)
    wf = fd.weakform.StressEquilibriumMixed(material, bulk_modulus=1.0, nlgeom=nlgeom)
    assembly = fd.Assembly.create(wf, mesh)
    pb = fd.problem.NonLinear(assembly)
    pb.initialize()
    rng = np.random.default_rng(0)
    n = assembly.n_gauss_points
    assembly.sv["DispGradient"] = [
        [0.05 * rng.standard_normal(n) for _ in range(3)] for _ in range(3)
    ]
    tangent = rng.standard_normal((6, 6, n))
    assembly.sv["TangentMatrix"] = tangent.copy()
    for field in ("PK2", "Stress", "StressDev"):
        assembly.sv[field] = StressTensorList(rng.standard_normal((6, n)))

    def global_matrix():
        assembly._saved_bloc_structure = None
        assembly.assemble_global_mat()
        return assembly.get_global_matrix().toarray()

    first = global_matrix()
    assert np.array_equal(assembly.sv["TangentMatrix"], tangent)
    # a second assembly from the same state gives the same matrix
    assert np.array_equal(global_matrix(), first)
