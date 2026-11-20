"""Linear transient problems keep their effective matrix while the time step
and the cached operators do not change."""

import numpy as np

import fedoo as fd


def _problem(dt=1e-3):
    fd.ModelingSpace("2Dplane")
    mesh = fd.mesh.rectangle_mesh(nx=9, ny=3, x_max=4.0, elm_type="quad4")
    material = fd.constitutivelaw.ElasticIsotrop(1000.0, 0.3)
    material.set_density(1.0)
    pb = fd.problem.Linear(
        fd.Assembly.create(fd.weakform.StressEquilibrium(material), mesh),
        time_step=dt,
        integrator=fd.time.Newmark(),
    )
    pb.bc.add("Dirichlet", mesh.find_nodes("X", 0.0), "Disp", 0)
    pb.bc.add("Neumann", mesh.find_nodes("X", 4.0), "DispY", -1.0)
    pb.initialize()
    return pb


def _effective_matrix(pb):
    integrator = pb.time_integrator
    dt = pb.time_step
    a0 = 1.0 / (integrator.beta * dt**2)
    c0 = integrator.gamma / (integrator.beta * dt)
    return (
        (1.0 - integrator.alpha_f) * pb._dynamic_stiffness
        + (1.0 - integrator.alpha_m) * a0 * pb._dynamic_mass
        + (1.0 - integrator.alpha_f) * c0 * pb._dynamic_damping
    )


def test_effective_matrix_kept_for_the_same_time_step():
    pb = _problem()
    A = pb.get_A()
    pb._dynamic_update_matrix()
    assert pb.get_A() is A
    pb.time_step = 2e-3
    pb._dynamic_update_matrix()
    assert pb.get_A() is not A
    assert abs(pb.get_A() - _effective_matrix(pb)).max() == 0
    pb.set_A(A)  # replaced by the user: rebuilt
    pb._dynamic_update_matrix()
    assert pb.get_A() is not A


def test_history_with_shortened_steps_matches_step_by_step():
    pb = _problem()
    pb.solve_history(0.0105)  # last increment shortened to reach tmax
    ref = _problem()
    while ref.time < 0.0105 - 1e-12:
        step_end = min(ref.time + 1e-3, 0.0105)
        ref.time_step = step_end - ref.time
        ref._dynamic_update_matrix()
        assert abs(ref.get_A() - _effective_matrix(ref)).max() == 0
        ref.solve_time_increment(t_fact=step_end / 0.0105)
    assert np.allclose(pb.get_disp(), ref.get_disp(), rtol=1e-12, atol=0)


def test_constant_step_history_rebuilds_the_matrix_rarely():
    """The rounded step ``step_end - time`` changes at a few increments only
    (when the time crosses a power of two)."""
    pb = _problem()
    calls = []
    set_A = pb.set_A
    pb.set_A = lambda A: (calls.append(1), set_A(A))[1]
    pb.solve_history(0.1)  # 100 increments
    assert len(calls) <= 10


def test_reset_drops_the_kept_matrix():
    pb = _problem()
    pb.solve_history(0.005)
    assert pb._dynamic_matrix_cache is not None
    pb.reset()
    assert pb._dynamic_matrix_cache is None
