import numpy as np
import pytest

import fedoo as fd
from fedoo.homogen import get_homogenized_stiffness


def _build_periodic_problem(rigid_body="pin", weights=None):
    """Build a small 2D periodic problem on a plate with a hole.

    rigid_body: "pin" -> block the node nearest to the center.
                "mean" -> MeanValueConstraint on the mean displacement.
    """
    fd.Assembly.delete_memory()
    fd.ModelingSpace("2Dstress")

    mesh = fd.mesh.hole_plate_mesh(nr=5, nt=5, length=10, height=10, radius=2)

    material = fd.constitutivelaw.ElasticIsotrop(1e5, 0.3)
    wf = fd.weakform.StressEquilibrium(material)
    assemb = fd.Assembly.create(wf, mesh)

    if rigid_body == "mean":
        constraint = fd.constraint.MeanValueConstraint(mesh, "Disp", weights=weights)
        pb = fd.problem.Linear(assemb + constraint)
    else:
        pb = fd.problem.Linear(assemb)

    pb.bc.add(fd.constraint.PeriodicBC(periodicity_type="small_strain"))

    if rigid_body == "pin":
        center = mesh.nearest_node(mesh.bounding_box.center)
        pb.bc.add("Dirichlet", center, "Disp", 0)

    E = [0.1, 0.05, 0.02]  # [EXX, EYY, EXY]
    pb.bc.add("Dirichlet", "MeanStrain", E)

    return pb, mesh, assemb


def test_mean_disp_constraint_vs_center_pin():
    """The mean-displacement constraint should give the same solution as the
    center pin, up to a rigid translation."""
    pb_pin, mesh, _ = _build_periodic_problem("pin")
    pb_pin.solve()
    volume = mesh.bounding_box.volume
    stress_pin = pb_pin.get_ext_forces("MeanStrain").ravel() / volume
    disp_pin = pb_pin.get_disp()

    pb_mean, mesh, _ = _build_periodic_problem("mean")
    pb_mean.solve()
    stress_mean = pb_mean.get_ext_forces("MeanStrain").ravel() / volume
    disp_mean = pb_mean.get_disp()

    # same mean stress
    assert np.allclose(stress_pin, stress_mean, rtol=1e-6)

    # the mean displacement is zero
    assert np.abs(disp_mean.mean(axis=1)).max() < 1e-10

    # the two displacement fields only differ by a rigid translation
    diff = disp_mean - disp_pin
    assert np.abs(diff - diff.mean(axis=1)[:, None]).max() < 1e-8

    # the lagrange multipliers are ~ 0 (self equilibrated periodic solution)
    force_scale = np.abs(pb_mean.get_ext_forces("MeanStrain")).max()
    for lm_name in ("MeanValue_DispX", "MeanValue_DispY"):
        assert abs(pb_mean.get_dof_solution(lm_name)[0]) < 1e-8 * force_scale


def test_homogenized_stiffness_mean_constraint():
    """get_homogenized_stiffness should give the same result with the mean
    displacement constraint and with the default center pin."""
    fd.Assembly.delete_memory()
    fd.ModelingSpace("3D")

    mesh = fd.mesh.box_mesh(nx=4, ny=4, nz=4, elm_type="hex8")
    material = fd.constitutivelaw.ElasticIsotrop(1e5, 0.3)
    wf = fd.weakform.StressEquilibrium(material)
    assemb = fd.Assembly.create(wf, mesh)

    C_pin = get_homogenized_stiffness(assemb, rigid_body_constraint="pin")
    C_mean = get_homogenized_stiffness(assemb, rigid_body_constraint="mean")

    assert np.allclose(C_pin, C_mean, rtol=1e-6)

    # homogeneous box -> the homogenized stiffness is the material stiffness
    assert np.allclose(C_mean, material.get_elastic_matrix(), rtol=1e-6)


def test_mean_disp_constraint_nonlinear():
    """The constraint should be enforced at each increment of a NonLinear
    problem."""
    fd.Assembly.delete_memory()
    fd.ModelingSpace("2Dstress")

    mesh = fd.mesh.hole_plate_mesh(nr=5, nt=5, length=10, height=10, radius=2)
    material = fd.constitutivelaw.ElasticIsotrop(1e5, 0.3)
    wf = fd.weakform.StressEquilibrium(material)
    assemb = fd.Assembly.create(wf, mesh)

    constraint = fd.constraint.MeanValueConstraint(mesh, "Disp")
    pb = fd.problem.NonLinear(assemb + constraint)

    pb.bc.add(fd.constraint.PeriodicBC(periodicity_type="small_strain"))
    pb.bc.add("Dirichlet", "MeanStrain", [0.05, 0, 0.01])

    pb.nlsolve(dt=0.5, tmax=1, update_dt=False)

    disp = pb.get_disp()
    assert np.abs(disp.mean(axis=1)).max() < 1e-8


def test_mean_disp_constraint_volume_weights():
    """With weights="volume", the volume average of the displacement field
    should be zero."""
    pb, mesh, _ = _build_periodic_problem("mean", weights="volume")
    pb.solve()

    disp = pb.get_disp()

    # recompute the nodal integration weights
    w = np.asarray(
        (mesh._get_gaussian_quadrature_mat() @ mesh._get_node2gausspoint_mat()).sum(
            axis=0
        )
    ).ravel()
    w = w / w.sum()

    assert np.abs(disp @ w).max() < 1e-10


def test_mean_value_constraint_duplicate_name():
    """Two constraints with the same name on the same problem should raise."""
    fd.Assembly.delete_memory()
    fd.ModelingSpace("2Dstress")

    mesh = fd.mesh.rectangle_mesh(nx=3, ny=3)
    material = fd.constitutivelaw.ElasticIsotrop(1e5, 0.3)
    wf = fd.weakform.StressEquilibrium(material)
    assemb = fd.Assembly.create(wf, mesh)

    c1 = fd.constraint.MeanValueConstraint(mesh, "DispX")
    c2 = fd.constraint.MeanValueConstraint(mesh, "DispY")  # same default name

    with pytest.raises(NameError):
        fd.problem.Linear(assemb + c1 + c2)


def _rectangle_assembly():
    fd.Assembly.delete_memory()
    fd.ModelingSpace("2Dstress")
    mesh = fd.mesh.rectangle_mesh(nx=4, ny=4)
    material = fd.constitutivelaw.ElasticIsotrop(1e5, 0.3)
    wf = fd.weakform.StressEquilibrium(material)
    return fd.Assembly.create(wf, mesh), mesh


def test_empty_node_set_raises():
    assemb, mesh = _rectangle_assembly()
    with pytest.raises(ValueError, match="empty"):
        fd.problem.Linear(
            assemb + fd.constraint.MeanValueConstraint(mesh, "Disp", node_set=[])
        )


def test_value_length_mismatch_raises():
    assemb, mesh = _rectangle_assembly()
    # 3 values but only 2 displacement components in 2D
    with pytest.raises(ValueError, match="components"):
        fd.problem.Linear(
            assemb
            + fd.constraint.MeanValueConstraint(mesh, "Disp", value=[0.1, 0.2, 0.3])
        )


def test_scalar_weights_raises():
    assemb, mesh = _rectangle_assembly()
    # a scalar weight is a shape error, must raise clearly (not TypeError)
    with pytest.raises(ValueError, match="1D array"):
        fd.problem.Linear(
            assemb + fd.constraint.MeanValueConstraint(mesh, "Disp", weights=0.5)
        )


def test_homogenized_stiffness_mean_rejects_iterative_solver():
    assemb, _ = _rectangle_assembly()
    with pytest.raises(ValueError, match="direct solver"):
        get_homogenized_stiffness(assemb, solver="cg", rigid_body_constraint="mean")


def test_matrix_cached_across_vector_updates():
    """The constant bordered matrix must be reused (not rebuilt) when only the
    vector is requested, as happens each Newton-Raphson iteration."""
    assemb, mesh = _rectangle_assembly()
    c = fd.constraint.MeanValueConstraint(mesh, "Disp")
    pb = fd.problem.Linear(assemb + c)
    pb.bc.add(fd.constraint.PeriodicBC(periodicity_type="small_strain"))
    pb.bc.add("Dirichlet", "MeanStrain", [0.1, 0, 0])
    pb.solve()

    mat_before = c.global_matrix
    c.assemble_global_mat("vector")  # NR-style vector-only refresh
    assert c.global_matrix is mat_before  # same object -> not rebuilt


if __name__ == "__main__":
    pytest.main([__file__])
