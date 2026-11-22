"""Total-Lagrangian tangent term written with one term per couple of derivatives.

``_tangent_op(eps, H)`` must equal the term-by-term expansion
sum_i eps_i.virtual * sum_j eps_j * H_ij once the duplicate couples of the
expansion are summed (only the summation order differs).
"""

import numpy as np
import pytest

import fedoo as fd
import fedoo.weakform.stress_equilibrium_mixed as mixed_module
import fedoo.weakform.stress_equilibrium as plain_module
from fedoo.weakform.stress_equilibrium import _tangent_op
from fedoo.util.voigt_tensors import StressTensorList


def _key(op):
    return op if op == 1 else (op.u_name, op.x, op.ordre)


def _merged(diff_op):
    terms = {}
    for op, op_vir, coef in zip(diff_op.op, diff_op.op_vir, diff_op.coef):
        key = (_key(op), _key(op_vir))
        terms[key] = terms[key] + coef if key in terms else coef
    return terms


def _expanded(eps, H):
    sigma = [
        sum([0 if eps[j] == 0 else eps[j] * H[i][j] for j in range(6)])
        for i in range(6)
    ]
    return sum([0 if eps[i] == 0 else eps[i].virtual * sigma[i] for i in range(6)])


@pytest.mark.parametrize("dimension", ["3D", "2Dplane"])
@pytest.mark.parametrize("pointwise_tangent", [True, False])
def test_tangent_op_matches_expansion(dimension, pointwise_tangent):
    space = fd.ModelingSpace(dimension)
    space.new_variable("DispX")
    space.new_variable("DispY")
    if space.ndim == 3:
        space.new_variable("DispZ")
    rng = np.random.default_rng(1)
    n = 7
    grad = [[0.1 * rng.standard_normal(n) for _ in range(3)] for _ in range(3)]
    eps = space.op_strain(grad)
    if pointwise_tangent:
        H = rng.standard_normal((6, 6, n))
    else:  # scalar components, some of them zero
        H = np.diag(rng.uniform(1, 2, 6))
        H[0, 1] = H[1, 0] = 0.3

    expected = _merged(_expanded(eps, H))
    actual = _tangent_op(eps, H)
    assert len(actual.op) == len(_merged(actual))  # one term per couple
    actual = _merged(actual)
    assert actual.keys() == expected.keys()
    for key in expected:
        np.testing.assert_allclose(actual[key], expected[key], rtol=1e-13, atol=1e-15)


def test_tangent_op_declines_operators_with_constant_terms():
    space = fd.ModelingSpace("3D")
    for name in ("DispX", "DispY", "DispZ"):
        space.new_variable(name)
    eps = space.op_strain()
    eps[0] = eps[0] + 1.0  # not a plain sum of derivatives
    assert _tangent_op(eps, np.eye(6)) is None


def test_mixed_tl_weak_equation_matches_expansion(monkeypatch):
    fd.ModelingSpace("3D")
    mesh = fd.mesh.box_mesh(nx=3, ny=3, nz=3, elm_type="hex8")
    material = fd.constitutivelaw.ElasticIsotrop(1.0, 0.3)
    wf = fd.weakform.StressEquilibriumMixed(material, bulk_modulus=1.0, nlgeom="TL")
    assembly = fd.Assembly.create(wf, mesh)
    pb = fd.problem.NonLinear(assembly)
    pb.initialize()
    rng = np.random.default_rng(2)
    n = assembly.n_gauss_points
    assembly.sv["DispGradient"] = [
        [0.05 * rng.standard_normal(n) for _ in range(3)] for _ in range(3)
    ]
    tangent = rng.standard_normal((6, 6, n))
    assembly.sv["PK2"] = StressTensorList(rng.standard_normal((6, n)))
    assembly.sv["StressDev"] = StressTensorList(rng.standard_normal((6, n)))

    def global_matrix():
        # a fresh tangent for every assembly: the finite-strain mixed weak form
        # modifies the components of sv["TangentMatrix"] in place
        assembly.sv["TangentMatrix"] = tangent.copy()
        assembly._saved_bloc_structure = None
        assembly.assemble_global_mat()
        return assembly.get_global_matrix().toarray(), assembly.get_global_vector()

    matrix, vector = global_matrix()
    monkeypatch.setattr(mixed_module, "_tangent_op", lambda eps, H: None)
    expected_matrix, expected_vector = global_matrix()
    np.testing.assert_allclose(
        matrix, expected_matrix, rtol=1e-12, atol=1e-12 * abs(expected_matrix).max()
    )
    np.testing.assert_array_equal(vector, expected_vector)


@pytest.mark.parametrize("zero_gradient", [True, False])
def test_tangent_op_leaves_its_inputs_unchanged(zero_gradient):
    space = fd.ModelingSpace("3D")
    for name in ("DispX", "DispY", "DispZ"):
        space.new_variable(name)
    rng = np.random.default_rng(3)
    n = 5
    scale = 0.0 if zero_gradient else 0.1  # zero: coefficients equal to 1
    grad = [[scale * rng.standard_normal(n) for _ in range(3)] for _ in range(3)]
    grad_copy = [[g.copy() for g in row] for row in grad]
    H = rng.standard_normal((6, 6, n))
    H_copy = H.copy()

    _tangent_op(space.op_strain(grad), H)

    assert np.array_equal(H, H_copy)
    for row, row_copy in zip(grad, grad_copy):
        for g, g_copy in zip(row, row_copy):
            assert np.array_equal(g, g_copy)


def test_tl_weak_equation_matches_expansion(monkeypatch):
    fd.ModelingSpace("3D")
    mesh = fd.mesh.box_mesh(nx=3, ny=3, nz=3, elm_type="hex8")
    material = fd.constitutivelaw.ElasticIsotrop(1.0, 0.3)
    wf = fd.weakform.StressEquilibrium(material, nlgeom="TL")
    wf.assume_sym = False
    assembly = fd.Assembly.create(wf, mesh)
    pb = fd.problem.NonLinear(assembly)
    pb.initialize()
    rng = np.random.default_rng(4)
    n = assembly.n_gauss_points
    assembly.sv["DispGradient"] = [
        [0.05 * rng.standard_normal(n) for _ in range(3)] for _ in range(3)
    ]
    assembly.sv["TangentMatrix"] = rng.standard_normal((6, 6, n))  # not symmetric
    assembly.sv["PK2"] = StressTensorList(rng.standard_normal((6, n)))

    def global_matrix():
        assembly._saved_bloc_structure = None
        assembly.assemble_global_mat()
        return assembly.get_global_matrix().toarray(), assembly.get_global_vector()

    matrix, vector = global_matrix()
    monkeypatch.setattr(plain_module, "_tangent_op", lambda eps, H: None)
    expected_matrix, expected_vector = global_matrix()
    assert abs(matrix - matrix.T).max() > 1e-3 * abs(matrix).max()
    np.testing.assert_allclose(
        matrix, expected_matrix, rtol=1e-12, atol=1e-12 * abs(expected_matrix).max()
    )
    np.testing.assert_array_equal(vector, expected_vector)
