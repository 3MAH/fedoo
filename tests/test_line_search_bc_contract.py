"""A step-size callback must not damp a pending Dirichlet increment.

Scaling the elastic prediction by alpha defers the remaining `1 - alpha` of
the Dirichlet increment to the following iterations, and convergence is only
declared once nothing is left to apply (`_xbc_is_applied`). A callback that
never returns exactly 1 therefore leaves the increment unfinished forever and
the solver keeps cutting the time step down to `dt_min`. The built-in line
search guarantees this by returning 1 while `_Xbc` is non-zero; a custom
callback has to do the same (see `NonLinear.add_line_search`).
"""

import numpy as np
import pytest

import fedoo as fd
from fedoo.problem.line_search import line_search


def _problem_with_pending_dirichlet():
    fd.Assembly.delete_memory()
    fd.ModelingSpace("2Dplane")
    mesh = fd.mesh.rectangle_mesh(3, 3, 0, 1, 0, 1, elm_type="quad4", name="sq")
    material = fd.constitutivelaw.ElasticIsotrop(1e3, 0.3, name="law")
    weakform = fd.weakform.StressEquilibrium(material, name="wf")
    assembly = fd.Assembly.create(weakform, mesh, name="asm")
    problem = fd.problem.NonLinear(assembly)
    problem.add_line_search()
    problem.bc.add("Dirichlet", mesh.node_sets["bottom"], "Disp", 0)
    problem.bc.add("Dirichlet", mesh.node_sets["top"], "Disp", [0.0, 0.1])
    problem.dtime = 1.0
    problem.initialize()
    problem.apply_boundary_conditions(1.0, 0.0)
    return problem


def test_built_in_line_search_yields_while_a_dirichlet_increment_is_pending():
    problem = _problem_with_pending_dirichlet()
    assert np.any(problem._Xbc)  # a prescribed increment is waiting
    assert not problem._xbc_is_applied()

    assert line_search(problem, np.zeros(problem.n_dof)) == 1


def test_displacement_control_reaches_the_prescribed_value():
    """End to end: the built-in line search never strands the increment."""
    problem = _problem_with_pending_dirichlet()
    problem.nlsolve(dt=1.0, tmax=1.0, update_dt=False, print_info=0)
    top = fd.Mesh["sq"].node_sets["top"]
    assert abs(problem.get_disp()[1, top].max() - 0.1) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
