"""The transient elastic prediction must not reuse a tangent of another dt.

A `fd.time` tangent carries the Newmark inertia term `1/(beta dt^2)`, so
reusing the previous increment's matrix after a time-step change leaves it
wrong by `(dt_prev/dt)^2` -- a factor 16 after the standard x0.25 cut, i.e.
exactly when the solver is already struggling. Static problems must keep
reusing the matrix: the "time integrators compiled" flag is True even with
none attached, so the refresh is gated on an actually attached integrator.
"""

import pytest

import fedoo as fd


def _problem(with_integrator, density=100.0):
    fd.Assembly.delete_memory()
    fd.ModelingSpace("2Dplane")
    mesh = fd.mesh.rectangle_mesh(4, 2, 0, 4, 0, 1, elm_type="quad4", name="beam")
    material = fd.constitutivelaw.ElasticIsotrop(2e5, 0.3, name="law")
    material.set_density(density)
    assembly = fd.Assembly.create(fd.weakform.StressEquilibrium(material), mesh)
    problem = fd.problem.NonLinear(assembly)
    if with_integrator:
        problem.set_time_integrator(fd.time.SECOND_ORDER, fd.time.Newmark())
    problem.bc.add("Dirichlet", mesh.find_nodes("X", 0), "Disp", 0)
    problem.bc.add("Neumann", mesh.find_nodes("X", 4), "DispY", -1.0)
    problem.nlsolve(dt=0.01, tmax=0.02, update_dt=False, print_info=0)
    return problem, assembly


def _predict_after_step_change(problem, new_dt):
    """Run one elastic prediction with a changed dt; return the tangent used."""
    problem._dtime_prev = problem.dtime
    problem.dtime = new_dt
    problem.set_start()
    stale = problem.get_A()
    problem.elastic_prediction()
    return stale, problem.get_A()


def test_transient_tangent_is_refreshed_after_a_step_change():
    problem, assembly = _problem(with_integrator=True)
    stale, used = _predict_after_step_change(problem, problem.dtime * 0.25)
    # the tangent actually used is the one re-assembled at the new time step
    assert used is assembly.current.get_global_matrix()
    assert used is not stale
    assert abs(used - stale).max() > 0
    # the inertia term scales as 1/dt^2, so a x0.25 cut stiffens the matrix
    assert abs(used).max() > 4 * abs(stale).max()


def test_static_tangent_is_still_reused_after_a_step_change():
    problem, _ = _problem(with_integrator=False)
    assert not problem.time_integrators
    stale, used = _predict_after_step_change(problem, problem.dtime * 0.25)
    assert used is stale


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
