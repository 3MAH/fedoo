"""A set_start without a solved increment must not consume a time step.

`nlsolve` resets the clock (`time = t0`), so a second stage -- a restart
from the current state with a new loading -- enters its loop with an empty
increment (`_dU == 0`) and calls `set_start` again. The Newmark recurrence
is NOT the identity for a zero displacement increment:

    v <- v (1 - gamma/beta) + dt (1 - gamma/2beta) a

which is exactly ``-v`` for the standard parameters (beta=1/4, gamma=1/2).
Velocity and acceleration were therefore corrupted at the start of every
stage after the first, for the `fd.time` integrators and for the legacy
`ImplicitDynamic` weak form alike.
"""

import numpy as np
import pytest

import fedoo as fd


def _dynamic_problem(legacy=False):
    fd.Assembly.delete_memory()
    fd.ModelingSpace("2Dplane")
    mesh = fd.mesh.rectangle_mesh(6, 3, 0, 3, 0, 1, elm_type="quad4")
    material = fd.constitutivelaw.ElasticIsotrop(2e5, 0.3)
    material.set_density(7800e-9)
    if legacy:
        weakform = fd.weakform.ImplicitDynamic(material, 0.01)
        assembly = fd.Assembly.create(weakform, mesh)
        problem = fd.problem.NonLinear(assembly)
    else:
        assembly = fd.Assembly.create(fd.weakform.StressEquilibrium(material), mesh)
        problem = fd.problem.NonLinear(assembly)
        problem.set_time_integrator(fd.time.SECOND_ORDER, fd.time.Newmark())
    problem.bc.add("Dirichlet", mesh.find_nodes("X", 0), "Disp", 0)
    problem.bc.add(
        "Neumann",
        mesh.find_nodes("X", 3),
        "DispY",
        -0.5,
        time_func=lambda t_fact: 1.0,
    )
    problem.nlsolve(dt=0.01, tmax=0.1, update_dt=False, print_info=0)
    return problem, assembly


@pytest.mark.parametrize("legacy", [False, True])
def test_empty_set_start_keeps_the_dynamic_state(legacy):
    problem, assembly = _dynamic_problem(legacy)
    velocity = np.array(assembly.sv["Velocity"], copy=True)
    acceleration = np.array(assembly.sv["Acceleration"], copy=True)
    assert np.abs(velocity).max() > 0  # the state is not trivially preserved

    problem.set_start()  # what the first loop turn of a new stage does

    assert np.allclose(assembly.sv["Velocity"], velocity, rtol=0, atol=0)
    assert np.allclose(assembly.sv["Acceleration"], acceleration, rtol=0, atol=0)


def test_increment_solved_flag_tracks_the_increment():
    problem, _ = _dynamic_problem()
    problem.set_start()
    assert problem._increment_solved is False  # nothing left to commit


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
