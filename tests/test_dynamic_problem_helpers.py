import numpy as np
from scipy import sparse

import fedoo as fd
from fedoo.core.matrix import as_global_csr
from fedoo.time.common import build_storage_assembly


def _assembly_with_damping(mesh, young_modulus, density, alpha, beta):
    material = fd.constitutivelaw.ElasticIsotrop(young_modulus, 0.3)
    material.set_density(density)
    weakform = fd.weakform.StressEquilibrium(material)
    weakform.set_damping(alpha=alpha, beta=beta)
    return fd.Assembly.create(weakform, mesh, "quad4")


def _heterogeneous_assembly():
    fd.ModelingSpace("2Dplane")
    mesh = fd.mesh.rectangle_mesh(nx=3, ny=2, elm_type="quad4")
    assembly_1 = _assembly_with_damping(mesh, 100.0, 2.0, 0.1, 0.01)
    assembly_2 = _assembly_with_damping(mesh, 40.0, 3.0, 0.3, 0.04)
    return mesh, assembly_1, assembly_2


def _expected_damping(storage_data, size, mass_lumping=False):
    expected = sparse.csr_matrix((size, size))
    for group in storage_data.groups:
        damping = group.rayleigh_damping
        mass = as_global_csr(group.storage_assembly.get_global_matrix(), size)
        if mass_lumping:
            mass = sparse.diags(np.asarray(mass.sum(axis=1)).ravel())
        stiffness = as_global_csr(
            group.source_assembly.current.get_global_matrix(), size
        )
        expected += damping.alpha * mass + damping.beta * stiffness
    return expected


def test_iter_leaf_preserves_assembly_sum_hierarchy():
    _, assembly_1, assembly_2 = _heterogeneous_assembly()
    inner = fd.AssemblySum([assembly_1, assembly_2])
    outer = fd.AssemblySum([inner, assembly_1])

    assert outer.list_assembly[0] is inner
    assert list(outer.iter_leaf()) == [assembly_1, assembly_2, assembly_1]

    weakform_sum = assembly_1.weakform + assembly_2.weakform
    assert list(weakform_sum.iter_leaf()) == weakform_sum.list_weakform


def test_as_global_csr_normalizes_placeholder_and_shape():
    assert as_global_csr(0, 3).shape == (3, 3)
    assert as_global_csr(0, 3).nnz == 0

    matrix = as_global_csr(np.eye(2), 3)
    assert sparse.isspmatrix_csr(matrix)
    assert matrix.shape == (3, 3)
    np.testing.assert_array_equal(matrix.diagonal(), [1.0, 1.0, 0.0])


def test_linear_supports_rayleigh_coefficients_per_assembly_part():
    _, assembly_1, assembly_2 = _heterogeneous_assembly()
    problem = fd.problem.Linear(
        fd.AssemblySum([assembly_1, assembly_2]),
        time_step=0.01,
        integrator=fd.time.Newmark(),
    )
    problem.initialize()

    expected = _expected_damping(problem._dynamic_storage_data, problem.n_dof)
    np.testing.assert_allclose(problem._dynamic_damping.toarray(), expected.toarray())


def test_explicit_supports_rayleigh_coefficients_per_assembly_part():
    _, assembly_1, assembly_2 = _heterogeneous_assembly()
    problem = fd.problem.ExplicitDynamic(
        fd.AssemblySum([assembly_1, assembly_2]),
        time_step=0.01,
        mass_lumping=True,
    )
    problem.initialize()

    expected = _expected_damping(
        problem._storage_data, problem.n_dof, mass_lumping=True
    )
    np.testing.assert_allclose(
        problem._rayleigh_damping_matrix.toarray(), expected.toarray()
    )


def test_conflicting_damping_in_one_leaf_requires_separate_assemblies():
    _, assembly_1, assembly_2 = _heterogeneous_assembly()
    combined_weakform = assembly_1.weakform + assembly_2.weakform
    combined_assembly = fd.Assembly.create(combined_weakform, assembly_1.mesh, "quad4")

    with np.testing.assert_raises_regex(
        NotImplementedError, "separate leaf assemblies"
    ):
        build_storage_assembly(combined_assembly, fd.time.SECOND_ORDER)
