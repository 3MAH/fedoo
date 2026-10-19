"""RigidTie(include_rotation=True): beam nodes clamped to a rigid body."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import fedoo as fd


def _beams(offsets, length=10.0, n=10):
    """Parallel exact beams along X, one per offset (y, z), clamped at X = 0."""
    fd.ModelingSpace("3D")
    meshes = []
    for y, z in offsets:
        line = fd.mesh.line_mesh(n + 1, 0, length, "lin2", ndim=3)
        line.nodes[:, 1] += y
        line.nodes[:, 2] += z
        meshes.append(line)
    mesh = fd.Mesh.stack(*meshes) if len(meshes) > 1 else meshes[0]
    section = fd.constitutivelaw.BeamRectangular(
        fd.constitutivelaw.ElasticIsotrop(1.0e3, 0.3), 0.5, 0.5, k=5 / 6
    )
    wf = fd.weakform.BeamEquilibrium(section, nlgeom="TL", kinematics="exact")
    pb = fd.problem.NonLinear(fd.Assembly.create(wf, mesh))
    base = mesh.find_nodes("X", 0)
    tip = mesh.find_nodes("X", length)
    pb.bc.add("Dirichlet", base, "Disp", 0)
    pb.bc.add("Dirichlet", base, "Rot", 0)
    pb.set_nr_criterion("Displacement", tol=1e-10, max_subiter=25)
    return pb, mesh, tip


def test_end_moment_on_the_rigid_body_rolls_the_beam_into_a_circle():
    pb, mesh, tip = _beams([(0, 0)])
    tie = fd.constraint.RigidTie(tip, center=tip[0], include_rotation=True)
    pb.bc.add(tie)
    pb.bc.add("Neumann", "RigidRotZ", 2 * np.pi * 1.0e3 * 0.5**4 / 12 / 10.0)
    pb.nlsolve(dt=0.1, update_dt=True, print_info=0)
    end = mesh.nodes[tip[0]] + pb.get_disp()[:, tip[0]]
    np.testing.assert_allclose(end, 0, atol=1e-10)
    assert pb.get_rot()[2, tip[0]] == pytest.approx(2 * np.pi, abs=1e-10)


def test_slave_nodes_follow_the_exact_rigid_rotation():
    pb, mesh, tip = _beams([(-1.0, 0.0), (1.0, 0.5)])
    tie = fd.constraint.RigidTie(tip, include_rotation=True)
    pb.bc.add(tie)
    rotvec = np.array([0.5, -0.4, 0.8])
    for name, value in zip(("RigidRotX", "RigidRotY", "RigidRotZ"), rotvec):
        pb.bc.add("Dirichlet", name, value)
    pb.bc.add("Dirichlet", "RigidDispX", -1.0)
    pb.nlsolve(dt=0.25, update_dt=True, print_info=0)

    rigid = tie.current_rotation(pb)
    # the converged quaternion state composes the increments exactly
    np.testing.assert_allclose(
        rigid, Rotation.from_rotvec(rotvec).as_matrix(), atol=1e-10
    )
    assembly = pb.assembly
    np.testing.assert_allclose(
        assembly.sv["_NodesRotationMatrix"][tip],
        np.broadcast_to(rigid, (2, 3, 3)),
        atol=1e-12,
    )
    # the tied ends keep their relative position on the rigid body
    ends = mesh.nodes[tip] + pb.get_disp()[:, tip].T
    initial = mesh.nodes[tip]
    np.testing.assert_allclose(
        ends[1] - ends[0], rigid @ (initial[1] - initial[0]), atol=1e-10
    )
    residual = assembly.current.global_vector
    np.testing.assert_allclose(residual[pb._dof_free], 0, atol=1e-7)


def test_rotation_tie_requires_rotation_dofs():
    fd.ModelingSpace("3D")
    mesh = fd.mesh.box_mesh(2, 2, 2)
    wf = fd.weakform.StressEquilibrium(fd.constitutivelaw.ElasticIsotrop(1.0, 0.3))
    pb = fd.problem.NonLinear(fd.Assembly.create(wf, mesh))
    with pytest.raises(ValueError, match="include_rotation"):
        pb.bc.add(
            fd.constraint.RigidTie(mesh.find_nodes("X", 1), include_rotation=True)
        )
        pb.initialize()


def test_shell_nodes_follow_the_exact_rigid_rotation():
    fd.ModelingSpace("3D")
    mesh = fd.mesh.rectangle_mesh(5, 3, 0, 2, 0, 1, "quad4", ndim=3)
    section = fd.constitutivelaw.ShellHomogeneous(
        fd.constitutivelaw.ElasticIsotrop(1.0e3, 0.3), 0.1
    )
    wf = fd.weakform.PlateEquilibrium(section, nlgeom=True)
    pb = fd.problem.NonLinear(fd.Assembly.create(wf, mesh, "pquad4mitc"))
    base = mesh.find_nodes("X", 0)
    tip = mesh.find_nodes("X", 2)
    pb.bc.add("Dirichlet", base, "Disp", 0)
    pb.bc.add("Dirichlet", base, "Rot", 0)
    tie = fd.constraint.RigidTie(tip, include_rotation=True)
    pb.bc.add(tie)
    rotvec = np.array([0.4, -0.3, 0.2])
    for name, value in zip(("RigidRotX", "RigidRotY", "RigidRotZ"), rotvec):
        pb.bc.add("Dirichlet", name, value)
    pb.set_nr_criterion("Displacement", tol=1e-10, max_subiter=25)
    pb.nlsolve(dt=0.25, update_dt=True, print_info=0)

    rigid = tie.current_rotation(pb)
    np.testing.assert_allclose(
        rigid, Rotation.from_rotvec(rotvec).as_matrix(), atol=1e-10
    )
    np.testing.assert_allclose(
        pb.assembly.sv["_NodesRotationMatrix"][tip],
        np.broadcast_to(rigid, (len(tip), 3, 3)),
        atol=1e-12,
    )
