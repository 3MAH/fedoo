"""Guard the line-search policy defaults of NonLinear.add_line_search.

The default must both throttle penalty-contact / plastic overshoot (a
validity-only default broke examples/03-advanced/tube_compression.py) and
let soft-mode force-control steps through (a residual-only default breaks
neohookean_cantilever_force.py): that is the "natural" mode.
"""

import pytest

import fedoo as fd
from fedoo.problem.line_search import line_search


def _make_problem():
    fd.Assembly.delete_memory()
    fd.ModelingSpace("2D")
    mesh = fd.mesh.rectangle_mesh(2, 2, 0, 1, 0, 1, elm_type="quad4", name="sq")
    mat = fd.constitutivelaw.ElasticIsotrop(1e3, 0.3, name="law")
    wf = fd.weakform.StressEquilibrium(mat, name="wf")
    asm = fd.Assembly.create(wf, mesh, name="asm")
    return fd.problem.NonLinear(asm)


@pytest.mark.parametrize(
    "kwargs, mode, method",
    [
        ({}, "natural", "Quadratic"),  # default
        ({"mode": "minimize", "method": "Residual"}, "minimize", "Residual"),
        ({"mode": "safeguard"}, "safeguard", "Quadratic"),
    ],
)
def test_add_line_search_records_mode(kwargs, mode, method):
    pb = _make_problem()
    pb.add_line_search(**kwargs)
    assert pb.nr_parameters["ls_mode"] == mode
    assert pb.nr_parameters["ls_method"] == method
    assert pb._step_size_callback is line_search


def test_mode_can_be_set_through_nr_criterion():
    pb = _make_problem()
    pb.add_line_search()
    pb.set_nr_criterion("Displacement", ls_mode="safeguard", ls_max_iter=8)
    assert pb.nr_parameters["ls_mode"] == "safeguard"
    assert pb.nr_parameters["ls_max_iter"] == 8


def test_invalid_mode_raises():
    pb = _make_problem()
    with pytest.raises(ValueError):
        pb.add_line_search(mode="foo")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
