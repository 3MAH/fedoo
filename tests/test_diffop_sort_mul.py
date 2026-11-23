"""DiffOp.sort and DiffOp * DiffOp against the operator-based reference."""

import itertools

import numpy as np
import pytest

import fedoo as fd
from fedoo.core.diffop import DiffOp, _Derivative


def _reference_sort(wf):
    """Order and duplicate flags from the comparison operators of _Derivative."""
    pairs = list(zip(wf.op, wf.op_vir))
    order = sorted(range(len(pairs)), key=pairs.__getitem__)
    same = [pairs[order[i]] == pairs[order[i + 1]] for i in range(len(order) - 1)]
    return same, order


def _reference_mul(a, b):
    """a * b built as a sum of single term DiffOp."""
    res = DiffOp([], [], [])
    for ii in range(len(b.op)):
        for jj in range(len(a.op)):
            assert b.op_vir[ii] == 1 and a.op[jj] == 1
            res += DiffOp([b.op[ii]], [a.op_vir[jj]], [b.coef[ii] * a.coef[jj]])
    return res


def _operators():
    names = ["DispX", "DispY", "Temp"]
    ops = [1]
    for u, name in enumerate(names):
        ops.append(_Derivative(u, u_name=name))
        ops += [_Derivative(u, x, 1, u_name=name) for x in range(3)]
    return ops


@pytest.mark.parametrize("seed", range(20))
def test_sort_matches_operator_ordering(seed):
    rng = np.random.default_rng(seed)
    ops = _operators()
    n = int(rng.integers(1, 40))
    op = [ops[i] for i in rng.integers(0, len(ops), n)]
    op_vir = [ops[i] for i in rng.integers(0, len(ops), n)]
    coef = list(rng.random(n))
    wf = DiffOp(list(op), list(op_vir), list(coef))

    same_ref, order_ref = _reference_sort(wf)
    same, order = wf.sort()

    assert order == order_ref
    assert same == same_ref
    assert wf.coef == [coef[i] for i in order_ref]
    assert all(a is op[i] for a, i in zip(wf.op, order_ref))
    assert all(a is op_vir[i] for a, i in zip(wf.op_vir, order_ref))


def test_sort_of_every_operator_pair():
    ops = _operators()
    pairs = list(itertools.product(ops, repeat=2))[::-1]
    wf = DiffOp([p[0] for p in pairs], [p[1] for p in pairs], [1.0] * len(pairs))
    same_ref, order_ref = _reference_sort(wf)
    assert wf.sort() == (same_ref, order_ref)
    assert not any(same_ref)


def test_sort_groups_terms_whatever_rank_and_decentrement():
    # as before, u and decentrement do not take part in the comparison
    a = _Derivative(0, 1, 1, u_name="DispX")
    b = _Derivative(3, 1, 1, decentrement=1, u_name="DispX")
    c = _Derivative(0, 2, 1, u_name="DispX")
    wf = DiffOp([c, a, 1, b], [1, c, 1, c], [1.0, 2.0, 3.0, 4.0])
    same_ref, order_ref = _reference_sort(wf)
    assert wf.sort() == (same_ref, order_ref)
    assert order_ref == [2, 1, 3, 0]
    assert same_ref == [False, True, False]


def test_mul_matches_sum_of_single_terms():
    space = fd.ModelingSpace("3D")
    space.new_variable("DispX")
    space.new_variable("DispY")
    arr = np.arange(4.0)
    real = (
        space.derivative("DispX", "X") * 2.0
        + space.derivative("DispY", "Z") * arr
        + space.variable("DispX")
    )
    virtual = (
        space.derivative("DispX", "Y").virtual * 3.0
        + space.variable("DispY").virtual
        + space.derivative("DispY", "X").virtual * arr
    )

    res = virtual * real
    ref = _reference_mul(virtual, real)
    assert len(res) == len(ref) == 9
    assert all(a is b for a, b in zip(res.op, ref.op))
    assert all(a is b for a, b in zip(res.op_vir, ref.op_vir))
    for c, c_ref in zip(res.coef, ref.coef):
        np.testing.assert_array_equal(c, c_ref)
    # a coefficient multiplied by 1 is shared, not copied
    assert sum(c is arr for c in res.coef) == 2


def _terms(wf):
    """Coefficient of each (trial operator, test operator) pair."""
    return {
        (op.u_name, op.x, vir.u_name, vir.x): coef
        for op, vir, coef in zip(wf.op, wf.op_vir, wf.coef)
    }


def test_real_times_virtual_is_the_same_form():
    space = fd.ModelingSpace("3D")
    space.new_variable("DispX")
    space.new_variable("DispY")
    real = space.derivative("DispX", "X") * 2.0 + space.derivative("DispY", "Y") * 3.0
    virtual = (
        space.variable("DispX").virtual * 5.0 + space.variable("DispY").virtual * 7.0
    )
    assert _terms(real * virtual) == _terms(virtual * real)
    assert _terms(real * virtual)[("DispX", 0, "DispY", 0)] == 14.0

    # operators of different lengths
    real = real + space.variable("DispX") * 11.0
    assert len(real * virtual) == 6
    assert _terms(real * virtual) == _terms(virtual * real)


def test_mul_refuses_two_real_operators():
    space = fd.ModelingSpace("2D")
    space.new_variable("DispX")
    with pytest.raises(NameError):
        space.variable("DispX") * space.variable("DispX")
