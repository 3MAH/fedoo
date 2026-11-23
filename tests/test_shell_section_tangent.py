"""Section tangent of the nonlinear shells: the 36 non-zero products of
_add_section_tangent equal the dense einsum they replace, bit for bit."""

import numpy as np
import pytest

from fedoo.constitutivelaw.shell import _add_section_tangent, _strain_matrix


@pytest.mark.parametrize("bad_value", [np.nan, np.inf])
def test_section_tangent_matches_einsum(bad_value):
    """Nonlinear shells: the section tangent equals the dense einsum bit for bit,
    including the propagation of non-finite material tangents."""
    rng = np.random.default_rng(5)
    n = 50
    for finite in (True, False):
        tangent = rng.standard_normal((6, 6, n))
        tangent[0, 1, :3] = -0.0
        if not finite:
            tangent[2, 4, 7] = bad_value  # out-of-plane component, zero in S
        for z, weight in ((0.0, 0.5), (-0.02, 0.3)):
            S = _strain_matrix(z)
            expected = np.zeros((8, 8, n))
            expected += weight * np.einsum("ia,ijp,jb->abp", S, tangent, S)
            actual = np.zeros((8, 8, n))
            _add_section_tangent(actual, tangent, z, weight)
            assert actual.tobytes() == expected.tobytes()
