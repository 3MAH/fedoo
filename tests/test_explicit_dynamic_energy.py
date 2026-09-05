"""Energy conservation of the explicit central-difference integrator.

An undamped, free-vibrating elastic bar must conserve total mechanical energy
(kinetic + elastic) to :math:`O(\\Delta t^2)` under symplectic central
difference. This pins the velocity-Verlet corrector: a stale end-of-step
acceleration injects energy and the total diverges over many increments,
whereas the correct scheme keeps it bounded.
"""

import numpy as np
import pytest

import fedoo as fd


def _build_free_vibration_bar(dt, mass_lumping=True):
    """Axial bar (nu=0), fixed at x=0, no external load, explicit dynamics."""
    fd.Assembly.delete_memory()
    fd.ModelingSpace("2Dplane")
    E, nu, rho = 1.0, 0.0, 1.0
    fd.mesh.rectangle_mesh(
        nx=6,
        ny=2,
        x_min=0,
        x_max=1.0,
        y_min=0,
        y_max=0.2,
        elm_type="quad4",
        name="Domain",
    )
    mesh = fd.Mesh["Domain"]
    material = fd.constitutivelaw.ElasticIsotrop(E, nu, name="law")
    material.set_density(rho)
    fd.weakform.StressEquilibrium("law", name="wf")
    fd.Assembly.create("wf", "Domain", "quad4", name="asm")
    pb = fd.problem.ExplicitDynamic("asm", time_step=dt, mass_lumping=mass_lumping)
    pb.bc.add("Dirichlet", mesh.find_nodes("X", 0.0), "Disp", 0)
    return pb, mesh


def _run_energy_history(dt, tmax, mass_lumping=True):
    pb, mesh = _build_free_vibration_bar(dt, mass_lumping=mass_lumping)
    # Velocity kick from the u=0 equilibrium: only the free nodes are set so the
    # initial energy matches the constrained motion the driver actually runs.
    fixed = mesh.find_nodes("X", 0.0)
    velocity = 0.05 * np.ones(mesh.n_nodes)
    velocity[fixed] = 0.0
    pb.set_initial_velocity("DispX", velocity)
    pb.initialize()

    reference = pb.get_kinetic_energy() + pb.get_elastic_energy()
    history = []

    def record(problem):
        history.append(problem.get_kinetic_energy() + problem.get_elastic_energy())

    pb.solve_history(tmax=tmax, callback=record)
    return reference, np.array(history)


def test_central_difference_conserves_energy():
    """Total energy must stay bounded over a long free-vibration history."""
    reference, history = _run_energy_history(dt=0.02, tmax=20.0)

    assert reference > 0.0
    assert np.all(np.isfinite(history)), "energy diverged to a non-finite value"
    # A stale end-step acceleration injects energy and the total blows up by
    # many orders of magnitude; the symplectic scheme keeps it near reference.
    assert history.max() < 1.1 * reference, (
        f"total energy grew to {history.max():.3e} (reference {reference:.3e}): "
        "the central-difference corrector is injecting energy"
    )
    drift = (history.max() - history.min()) / reference
    assert drift < 0.1, f"energy drift {drift:.3e} too large for a symplectic scheme"


def test_central_difference_conserves_energy_consistent_mass():
    """Same guarantee with a consistent (non-lumped) mass matrix.

    Exercises the mass-solve branch of the end-of-step acceleration, which the
    lumped default (a diagonal divide) does not reach.
    """
    reference, history = _run_energy_history(dt=0.02, tmax=4.0, mass_lumping=False)

    assert reference > 0.0
    assert np.all(np.isfinite(history)), "energy diverged to a non-finite value"
    assert (
        history.max() < 1.1 * reference
    ), f"total energy grew to {history.max():.3e} (reference {reference:.3e})"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
