"""Guard the line-search policy defaults of NonLinear.add_line_search.

The default must both throttle penalty-contact / plastic overshoot (a
validity-only default broke examples/03-advanced/tube_compression.py) and
let soft-mode force-control steps through (a residual-only default breaks
neohookean_cantilever_force.py): that is the "natural" mode.
"""

import importlib

import numpy as np
import pytest

import fedoo as fd
from fedoo.core.base import InvalidKinematicStateError
from fedoo.problem.line_search import line_search

line_search_module = importlib.import_module("fedoo.problem.line_search")


def _make_problem():
    fd.Assembly.delete_memory()
    fd.ModelingSpace("2D")
    mesh = fd.mesh.rectangle_mesh(2, 2, 0, 1, 0, 1, elm_type="quad4", name="sq")
    mat = fd.constitutivelaw.ElasticIsotrop(1e3, 0.3, name="law")
    wf = fd.weakform.StressEquilibrium(mat, name="wf")
    asm = fd.Assembly.create(wf, mesh, name="asm")
    return fd.problem.NonLinear(asm)


@pytest.mark.parametrize(
    "kwargs, mode, method, apply_to_bc",
    [
        ({}, "natural", "Quadratic", True),  # defaults
        (
            {"mode": "minimize", "method": "Residual"},
            "minimize",
            "Residual",
            True,
        ),
        (
            {"mode": "safeguard", "apply_to_bc": False},
            "safeguard",
            "Quadratic",
            False,
        ),
    ],
)
def test_add_line_search_records_configuration(kwargs, mode, method, apply_to_bc):
    pb = _make_problem()
    pb.add_line_search(**kwargs)
    assert pb.nr_parameters["ls_mode"] == mode
    assert pb.nr_parameters["ls_method"] == method
    assert pb.nr_parameters["ls_apply_to_bc"] is apply_to_bc
    assert pb._step_size_callback is line_search


def test_mode_can_be_set_through_nr_criterion():
    pb = _make_problem()
    pb.add_line_search()
    pb.set_nr_criterion(
        "Displacement", ls_mode="safeguard", ls_max_iter=8, ls_apply_to_bc=False
    )
    assert pb.nr_parameters["ls_mode"] == "safeguard"
    assert pb.nr_parameters["ls_max_iter"] == 8
    assert pb.nr_parameters["ls_apply_to_bc"] is False


def test_invalid_mode_raises():
    pb = _make_problem()
    with pytest.raises(ValueError):
        pb.add_line_search(mode="foo")


def test_natural_fallback_returns_last_valid_trial(monkeypatch):
    pb = _make_problem()
    pb.add_line_search(mode="natural")
    pb.nr_parameters["ls_max_iter"] = 3
    pb.apply_boundary_conditions(1.0, 0.0)
    trials = []

    def evaluate(problem, dX, alpha):
        trials.append(alpha)
        return 1.0, np.ones(len(problem._dof_free))

    monkeypatch.setattr(line_search_module, "_evaluate_residual_norm", evaluate)
    monkeypatch.setattr(line_search_module, "_natural_test", lambda *args: False)

    alpha = line_search(pb, np.ones(pb.n_dof))
    assert alpha == trials[-1]
    assert alpha < 1.0


def test_invalid_line_search_trial_reports_failed_time_increment(monkeypatch):
    pb = _make_problem()

    def reject_increment(*args, **kwargs):
        raise InvalidKinematicStateError

    monkeypatch.setattr(pb, "elastic_prediction", reject_increment)
    convergence, subiter, error = pb.solve_time_increment()

    assert convergence == 0
    assert subiter == 1
    assert np.isinf(error)
    assert pb._t_fact_inc is None


def test_invalid_newton_correction_reports_failed_time_increment(monkeypatch):
    pb = _make_problem()

    monkeypatch.setattr(pb, "elastic_prediction", lambda: None)
    monkeypatch.setattr(pb, "update", lambda *args, **kwargs: None)
    monkeypatch.setattr(pb, "_update_d", lambda *args, **kwargs: None)
    monkeypatch.setattr(pb, "compute_nr_error", lambda: 1.0)
    monkeypatch.setattr(pb.assembly.current, "get_global_matrix", lambda: 0)

    def reject_increment():
        raise InvalidKinematicStateError

    monkeypatch.setattr(pb, "solve_nr_increment", reject_increment)
    convergence, subiter, error = pb.solve_time_increment(max_subiter=2, tol_nr=0.0)

    assert convergence == 0
    assert subiter == 2
    assert error == 1.0
    assert pb._t_fact_inc is None


def test_removing_natural_line_search_releases_its_factorization(monkeypatch):
    pb = _make_problem()
    pb.add_line_search()

    def set_reuse(reuse=True):
        pb._factor_context = object() if reuse else None
        pb._factor_valid = False

    monkeypatch.setattr(pb, "set_reuse_factorization", set_reuse)
    pb._enable_factorization_reuse()
    assert pb._factor_context is pb._line_search_factor_context

    pb.remove_line_search()
    assert pb._factor_context is None
    assert pb._line_search_factor_context is None


def test_removing_line_search_preserves_user_replacement_factorization(monkeypatch):
    pb = _make_problem()
    pb.add_line_search()

    def set_reuse(reuse=True):
        pb._factor_context = object() if reuse else None
        pb._factor_valid = False

    monkeypatch.setattr(pb, "set_reuse_factorization", set_reuse)
    pb._enable_factorization_reuse()
    user_context = object()
    pb._factor_context = user_context

    pb.remove_line_search()
    assert pb._factor_context is user_context
    assert pb._line_search_factor_context is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
