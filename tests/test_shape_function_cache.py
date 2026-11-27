"""Cached shape function derivatives of the elements: same values as the
uncached methods, bit for bit."""

import numpy as np
import pytest

from fedoo.lib_elements.element_base import _cached_by_points
from fedoo.lib_elements.element_list import get_element


@pytest.mark.parametrize("name", ["hex8", "hex20", "hex20r", "tet10", "quad8", "quad9"])
def test_cached_shape_function_derivative(name):
    elm = get_element(name)()
    method = type(elm).shape_function_derivative
    uncached = method.__wrapped__
    xi = elm.xi_pg
    expected = uncached(elm, xi)
    first = elm.shape_function_derivative(xi)
    second = elm.shape_function_derivative(xi.copy())
    assert isinstance(second, list) and len(second) == len(expected)
    for e, a, b in zip(expected, first, second):
        assert e.tobytes() == a.tobytes() == b.tobytes()
    # the returned arrays are new arrays: modifying them leaves the cache intact
    second[0][...] = np.nan
    third = elm.shape_function_derivative(xi)
    assert np.array_equal(third[0], expected[0])
    # other points
    pts = np.random.default_rng(1).uniform(-1, 1, (5, xi.shape[1]))
    for e, a in zip(uncached(elm, pts), elm.shape_function_derivative(pts)):
        assert np.array_equal(e, a)


def test_cached_by_points_is_bounded():
    calls = []

    class _Elm:
        @_cached_by_points
        def values(self, vec_xi):
            calls.append(1)
            return [np.asarray(vec_xi, dtype=float) * 2.0]

    elm = _Elm()
    for i in range(40):
        elm.values(np.array([[float(i)]]))
    assert len(calls) == 40
    elm.values(np.array([[39.0]]))
    assert len(calls) == 40  # cached
    assert len(_Elm.values.cache) <= 16


def test_inputs_that_are_not_numeric_arrays_are_not_cached():
    elm = get_element("hex8")()
    cache = type(elm).shape_function_derivative.cache
    for value in (0.0, 0.5):  # same bytes for an object array: must not be reused
        n_cached = len(cache)
        points = np.array([[value, value, value]])
        expected = elm.shape_function_derivative(points)
        from_list = elm.shape_function_derivative(points.tolist())
        from_objects = elm.shape_function_derivative(points.astype(object))
        assert len(cache) <= n_cached + 1
        assert np.array_equal(from_list[0], expected[0])
        assert np.array_equal(from_objects[0].astype(float), expected[0])
