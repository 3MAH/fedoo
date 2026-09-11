"""Line-search policies for a pending Dirichlet increment.

Scaling the elastic prediction by alpha defers the remaining `1 - alpha` of
the Dirichlet increment to the following iterations, and convergence is only
declared once nothing is left to apply (`_xbc_is_applied`). This progressive
application is the default. `apply_to_bc=False` instead applies the prescribed
increment in one full step before line search scales equilibrium corrections.
"""

import importlib

import numpy as np
import pytest

import fedoo as fd
from fedoo.core.base import InvalidKinematicStateError
from fedoo.problem.line_search import line_search

line_search_module = importlib.import_module("fedoo.problem.line_search")


def _problem_with_pending_dirichlet(apply_to_bc=True):
    fd.Assembly.delete_memory()
    fd.ModelingSpace("2Dplane")
    mesh = fd.mesh.rectangle_mesh(3, 3, 0, 1, 0, 1, elm_type="quad4", name="sq")
    material = fd.constitutivelaw.ElasticIsotrop(1e3, 0.3, name="law")
    weakform = fd.weakform.StressEquilibrium(material, name="wf")
    assembly = fd.Assembly.create(weakform, mesh, name="asm")
    problem = fd.problem.NonLinear(assembly)
    problem.add_line_search(mode="safeguard", apply_to_bc=apply_to_bc)
    problem.bc.add("Dirichlet", mesh.node_sets["bottom"], "Disp", 0)
    problem.bc.add("Dirichlet", mesh.node_sets["top"], "Disp", [0.0, 0.1])
    problem.dtime = 1.0
    problem.initialize()
    problem.apply_boundary_conditions(1.0, 0.0)
    return problem


def test_line_search_evaluates_pending_dirichlet_by_default(monkeypatch):
    problem = _problem_with_pending_dirichlet(apply_to_bc=True)
    assert np.any(problem._Xbc)  # a prescribed increment is waiting
    assert not problem._xbc_is_applied()

    trials = []

    def evaluate(pb, dX, alpha):
        trials.append(alpha)
        return 0.0, np.zeros(len(pb._dof_free))

    monkeypatch.setattr(line_search_module, "_evaluate_residual_norm", evaluate)
    assert line_search(problem, np.zeros(problem.n_dof)) == 1
    assert trials == [1.0]


def test_pending_dirichlet_uses_only_validity_filter_in_minimize_mode(monkeypatch):
    problem = _problem_with_pending_dirichlet(apply_to_bc=True)
    problem.nr_parameters["ls_mode"] = "minimize"
    trials = []

    def evaluate(pb, dX, alpha):
        trials.append(alpha)
        # The previous equilibrium has zero residual, so a positive residual
        # would make residual minimization throttle an otherwise valid BC step.
        return 1.0, np.ones(len(pb._dof_free))

    monkeypatch.setattr(line_search_module, "_evaluate_residual_norm", evaluate)
    assert line_search(problem, np.zeros(problem.n_dof)) == 1
    assert trials == [1.0]


def test_line_search_can_leave_pending_dirichlet_unscaled(monkeypatch):
    problem = _problem_with_pending_dirichlet(apply_to_bc=False)

    def should_not_run(*args):
        raise AssertionError("strict Dirichlet policy must bypass trial evaluation")

    monkeypatch.setattr(line_search_module, "_evaluate_residual_norm", should_not_run)
    assert line_search(problem, np.zeros(problem.n_dof)) == 1


def test_reduced_dirichlet_step_keeps_the_unapplied_remainder(monkeypatch):
    problem = _problem_with_pending_dirichlet(apply_to_bc=True)

    def reject_full_step(pb, dX, alpha):
        residual = np.zeros(len(pb._dof_free))
        return (np.inf, None) if alpha == 1.0 else (0.0, residual)

    monkeypatch.setattr(line_search_module, "_evaluate_residual_norm", reject_full_step)
    problem.elastic_prediction()

    top = np.asarray(fd.Mesh["sq"].node_sets["top"])
    disp_y = problem.space.variable_rank("DispY") * problem.mesh.n_nodes + top
    # safeguard finds alpha=0.5 and applies its 0.8 safety factor -> 0.4
    assert np.allclose(problem._dU[disp_y], 0.04)
    assert np.allclose(problem._Xbc[disp_y], 0.06)
    assert not problem._xbc_is_applied()


def test_line_search_rejects_increment_if_every_trial_is_invalid(monkeypatch):
    problem = _problem_with_pending_dirichlet(apply_to_bc=True)
    trials = []

    def reject_trial(pb, dX, alpha):
        trials.append(alpha)
        return np.inf, None

    monkeypatch.setattr(line_search_module, "_evaluate_residual_norm", reject_trial)
    with pytest.raises(InvalidKinematicStateError):
        line_search(problem, np.zeros(problem.n_dof))
    assert trials[-1] == line_search_module._ALPHA_MIN


def test_displacement_control_reaches_the_prescribed_value():
    """End to end: the built-in line search never strands the increment."""
    problem = _problem_with_pending_dirichlet()
    problem.nlsolve(dt=1.0, tmax=1.0, update_dt=False, print_info=0)
    top = fd.Mesh["sq"].node_sets["top"]
    assert abs(problem.get_disp()[1, top].max() - 0.1) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
