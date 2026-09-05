"""Stability guards of the implicit second-order time integrators.

A Newmark/generalized-alpha pair violating the unconditional (A-) stability
conditions (gamma >= 1/2 - alpha_m + alpha_f, beta >= gamma/2,
alpha_m <= alpha_f <= 1/2) is only conditionally stable: with FE meshes
(omega_max*dt >> 1) the high-frequency modes grow geometrically and the
solve collapses after a few tens of increments, regardless of dt or load —
a failure that masquerades as Newton divergence / element inversion (this
cost a full day of root-causing on the force-controlled RigidTie benchmark:
beta=0.25 with gamma=0.6 died at ~20-26 committed increments at any dt).
Both entry points must warn on violating pairs and stay silent on valid
ones.
"""

import warnings

import numpy as np
import pytest

import fedoo as fd


def _material():
    fd.ModelingSpace("3D")
    mat = fd.constitutivelaw.ElasticIsotrop(1e6, 0.3)
    mat.set_density(1000.0)
    return mat


def _n_stability_warnings(record):
    return sum("CONDITIONALLY stable" in str(w.message) for w in record)


def test_implicit_dynamic_warns_on_conditionally_stable_pair():
    mat = _material()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        fd.weakform.ImplicitDynamic(mat, 1000.0, beta=0.25, gamma=0.6)
    assert _n_stability_warnings(rec) == 1


def test_implicit_dynamic_silent_on_stable_pairs():
    mat = _material()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        fd.weakform.ImplicitDynamic(mat, 1000.0)  # default beta=0.25 gamma=0.5
        fd.weakform.ImplicitDynamic(mat, 1000.0, beta=0.3025, gamma=0.6)
        fd.weakform.ImplicitDynamic(mat, 1000.0, beta=0.30, gamma=0.6)
    assert _n_stability_warnings(rec) == 0


def test_generalized_alpha_warns_on_conditionally_stable_pairs():
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        fd.time.Newmark(beta=0.25, gamma=0.6)  # beta < gamma/2
    assert _n_stability_warnings(rec) == 1

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        fd.time.Newmark(beta=0.25, gamma=0.4)  # gamma < 1/2
    assert _n_stability_warnings(rec) == 1

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        fd.time.GeneralizedAlpha(alpha_m=0.3, alpha_f=0.1)  # alpha_m > alpha_f
    assert _n_stability_warnings(rec) == 1


def test_generalized_alpha_silent_on_stable_sets():
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        fd.time.Newmark()  # defaults
        fd.time.Newmark(beta=0.3025, gamma=0.6)
        fd.time.GeneralizedAlpha()  # Newmark trapezoidal
        # Chung-Hulbert defaults from the alphas (gamma and beta computed)
        fd.time.GeneralizedAlpha(alpha_m=0.2, alpha_f=0.4)
    assert _n_stability_warnings(rec) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
