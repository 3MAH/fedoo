"""adaptive_stiffness keeps the last IMPROVING iterate for its rollback.

When the error rises twice in a row the solver switches to the elastic
matrix and redoes the last iteration from `_dU_old`. That backup used to be
taken once the error had already risen, i.e. it stored the very iterate the
rollback is meant to undo -- and on a cleanly converging increment it was
never taken at all. It is now refreshed on every iteration that improves the
error.
"""

import numpy as np
import pytest

import fedoo as fd


def _problem():
    fd.Assembly.delete_memory()
    fd.ModelingSpace("2Dplane")
    mesh = fd.mesh.rectangle_mesh(3, 3, 0, 1, 0, 1, elm_type="quad4", name="sq")
    material = fd.constitutivelaw.ElasticIsotrop(1e3, 0.3, name="law")
    weakform = fd.weakform.StressEquilibrium(material, name="wf")
    assembly = fd.Assembly.create(weakform, mesh, name="asm")
    problem = fd.problem.NonLinear(assembly, nlgeom="UL")
    problem.set_nr_criterion(
        "Displacement", tol=1e-9, max_subiter=15, adaptive_stiffness=True
    )
    problem.bc.add("Dirichlet", mesh.node_sets["bottom"], "Disp", 0)
    problem.bc.add("Dirichlet", mesh.node_sets["top"], "Disp", [0.15, 0.05])
    return problem


def _run_recording_backups(problem):
    """Return the sequence of distinct `_dU_old` backups seen during a solve."""
    backups = []
    original = problem.compute_nr_error

    def spy():
        backup = problem._dU_old
        if not np.isscalar(backup):
            if not backups or not np.array_equal(backups[-1], backup):
                backups.append(np.asarray(backup).copy())
        return original()

    problem.compute_nr_error = spy
    problem.nlsolve(dt=1.0, tmax=1.0, update_dt=False, print_info=0)
    return backups


def test_backup_is_refreshed_on_every_improving_iteration():
    problem = _problem()
    backups = _run_recording_backups(problem)
    # A cleanly converging increment only has improving iterations: the old
    # rule (save when the error has risen) recorded none at all.
    assert len(backups) >= 2
    assert np.linalg.norm(backups[-1]) > 0


def test_backup_is_unset_before_any_increment():
    assert _problem()._dU_old == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
