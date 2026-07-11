"""Regression guards for MeanMotion bugs.

These complement the main ``test_mean_motion.py`` suite. They lock in the
linear (small-rotation) path's projection-row bug: a component selection that
is not the leading rigid-body modes in natural order (e.g. ``components="DispZ"``
or a reordered list) must still drive the *selected* mean components — not the
wrong projection row. Both cases run a full ``nlsolve`` and check the enforced
mean of the physical field, which the wrong-row bug silently violated.
"""

import numpy as np

import fedoo as fd


def _box(label):
    """Small clamped-left cube with a MeanMotion-ready right surface."""
    fd.ModelingSpace("3D")
    mesh = fd.mesh.box_mesh(
        nx=3,
        ny=3,
        nz=3,
        x_min=0.0,
        x_max=2.0,
        y_min=-1.0,
        y_max=1.0,
        z_min=-1.0,
        z_max=1.0,
        elm_type="hex8",
        name=f"mmreg_{label}_mesh",
    )
    mat = fd.constitutivelaw.ElasticIsotrop(200e3, 0.3, name=f"mmreg_{label}_mat")
    wf = fd.weakform.StressEquilibrium(mat, name=f"mmreg_{label}_wf")
    assemb = fd.Assembly.create(wf, mesh, "hex8", name=f"mmreg_{label}_asm")
    pb = fd.problem.NonLinear(assemb, name=f"mmreg_{label}_pb")

    left = mesh.find_nodes("X", mesh.bounding_box.xmin)
    right = mesh.find_nodes("X", mesh.bounding_box.xmax)
    surf = fd.mesh.extract_surface(mesh, node_set=right, reduce_order=False)
    return mesh, pb, left, surf


def _face_disp(pb, nodes):
    """(n_face, 3) displacement of the selected nodes."""
    return pb.get_dof_solution("Disp").T[nodes]


def test_mean_disp_single_non_leading_component_enforced():
    # components="DispZ": selected subset index 0 maps to full-mode index 2.
    # The wrong-row bug tied MeanDispZ to the DispX fit instead.
    mesh, pb, left, surf = _box("dispz")
    mm = fd.constraint.MeanMotion(surf, components="DispZ")
    pb.bc.add(mm)
    pb.bc.add("Dirichlet", left, "Disp", 0.0)
    target = -0.30
    pb.bc.add("Dirichlet", "MeanDispZ", target)
    pb.nlsolve(dt=1.0, tmax=1.0, print_info=0)

    u = _face_disp(pb, mm.nodes)
    assert abs(np.average(u[:, 2], weights=mm._weights) - target) < 1e-8
    # the unselected x/y means stay free (not accidentally pinned)
    assert abs(np.average(u[:, 0], weights=mm._weights)) < 1e-6


def test_mean_disp_reordered_multi_component_enforced():
    # Reordered selection -> _mode_indices = [2, 0]; each selected mean must
    # reach its own target, confirming the subset->full-mode row mapping.
    mesh, pb, left, surf = _box("dispzx")
    mm = fd.constraint.MeanMotion(surf, components=["DispZ", "DispX"])
    pb.bc.add(mm)
    pb.bc.add("Dirichlet", left, "Disp", 0.0)
    pb.bc.add("Dirichlet", "MeanDispZ", -0.20)
    pb.bc.add("Dirichlet", "MeanDispX", 0.10)
    pb.nlsolve(dt=1.0, tmax=1.0, print_info=0)

    u = _face_disp(pb, mm.nodes)
    assert abs(np.average(u[:, 2], weights=mm._weights) - (-0.20)) < 1e-8
    assert abs(np.average(u[:, 0], weights=mm._weights) - 0.10) < 1e-8
