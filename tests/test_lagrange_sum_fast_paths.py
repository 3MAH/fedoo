"""Lagrange multiplier equations compared as flat arrays and sums of sparse
matrices without the leading ``0 + A``: same results as the simpler forms."""

import numpy as np
from scipy import sparse

import fedoo as fd
from fedoo.core.assembly_sum import _sum_matrices


def test_sum_matrices():
    rng = np.random.default_rng(6)
    A = sparse.random(20, 20, density=0.2, format="csr", random_state=rng)
    B = sparse.random(20, 20, density=0.2, format="csr", random_state=rng)
    C = sparse.random(20, 20, density=0.2, format="csr", random_state=rng)
    for items in ([A, B], [A, B, C], [A], [0, A], [A, 0], [0, 0]):
        res, ref = _sum_matrices(items), sum(items)
        if np.isscalar(ref):
            assert res == ref
        else:
            assert (res != ref).nnz == 0 and np.array_equal(res.data, ref.data)
            assert res.format == ref.format and res is not items[0]


def test_lagrange_equations_changed():
    fd.ModelingSpace("2D")
    mesh = fd.mesh.rectangle_mesh(nx=4, ny=4)
    lm = fd.LagrangeMultiplierAssembly(mesh, fd.constraint.PeriodicBC(), name="lm_test")
    dofs = [np.array([1, 2]), np.array([3, 4, 5])]
    coefs = [np.array([1.0, -1.0]), np.array([1.0, 2.0, 3.0])]
    lm._equation_dofs, lm._equation_coefficients = dofs, coefs
    copy = lambda rows: [r.copy() for r in rows]  # noqa: E731
    assert not lm._equations_changed(copy(dofs), copy(coefs))
    lm._equation_dofs, lm._equation_coefficients = copy(dofs), copy(coefs)
    changed = copy(coefs)
    changed[1][2] = 3.5
    assert lm._equations_changed(copy(dofs), changed)
    # same concatenation, other rows
    assert lm._equations_changed(
        [np.array([1, 2, 3]), np.array([4, 5])],
        [np.array([1.0, -1.0, 1.0]), np.array([2.0, 3.0])],
    )
    assert lm._equations_changed(copy(dofs)[:1], copy(coefs)[:1])
    # the flat form kept at the previous call is the one compared
    kept_dofs, kept_coefs = copy(dofs), copy(coefs)
    lm._equations_changed(kept_dofs, kept_coefs)
    lm._equation_dofs, lm._equation_coefficients = kept_dofs, kept_coefs
    assert lm._flat_equation_cache[0] is kept_dofs
    assert not lm._equations_changed(copy(dofs), copy(coefs))
    lm._equation_dofs, lm._equation_coefficients = kept_dofs, kept_coefs
    lm._flat_equation_cache = (kept_dofs, lm._flat_equations(kept_dofs, kept_coefs))
    assert lm._equations_changed(copy(dofs), changed)
