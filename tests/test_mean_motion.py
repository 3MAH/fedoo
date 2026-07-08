import numpy as np
from scipy import sparse

import fedoo as fd


def _make_problem_2d():
    space = fd.ModelingSpace("2Dplane")
    space.new_variable("DispX")
    space.new_variable("DispY")
    space.new_vector("Disp", ("DispX", "DispY"))

    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    elements = np.array([[0, 1], [1, 2]])
    mesh = fd.Mesh(nodes, elements, "lin2")
    return fd.Problem(A=sparse.eye(mesh.n_nodes * space.nvar), mesh=mesh, space=space)


def _make_problem_3d():
    space = fd.ModelingSpace("3D")
    space.new_variable("DispX")
    space.new_variable("DispY")
    space.new_variable("DispZ")
    space.new_vector("Disp", ("DispX", "DispY", "DispZ"))

    nodes = np.array(
        [
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
        ]
    )
    elements = np.array([[0, 1, 2, 3]])
    mesh = fd.Mesh(nodes, elements, "quad4")
    return fd.Problem(A=sparse.eye(mesh.n_nodes * space.nvar), mesh=mesh, space=space)


def _make_square_grid_problem_3d():
    space = fd.ModelingSpace("3D")
    space.new_variable("DispX")
    space.new_variable("DispY")
    space.new_variable("DispZ")
    space.new_vector("Disp", ("DispX", "DispY", "DispZ"))

    nodes = np.array([[0.0, y, z] for z in [-1.0, 0.0, 1.0] for y in [-1.0, 0.0, 1.0]])
    elements = np.array([[0, 1, 4, 3], [1, 2, 5, 4], [3, 4, 7, 6], [4, 5, 8, 7]])
    mesh = fd.Mesh(nodes, elements, "quad4")
    return fd.Problem(A=sparse.eye(mesh.n_nodes * space.nvar), mesh=mesh, space=space)


def test_mean_motion_disp_creates_global_vector_and_supports_dirichlet():
    pb = _make_problem_2d()

    control = fd.constraint.MeanMotion([0, 1, 2], components="Disp")
    pb.bc.add(control)
    pb.bc.add("Dirichlet", control.node_disp, "MeanDispX", 3.0)
    pb.apply_boundary_conditions()

    assert pb.global_dof._vector["MeanDisp"] == ["MeanDispX", "MeanDispY"]

    mean_x = pb.n_node_dof + pb.global_dof.indice_start("MeanDispX") + control.node_disp
    slave_x = 1

    assert pb._Xbc[mean_x] == 3.0
    assert pb._Xbc[slave_x] == 9.0
    assert mean_x not in pb._dof_free
    assert slave_x not in pb._dof_free

    pb.solve()
    assert pb.get_dof_solution("MeanDispX")[control.node_disp] == 3.0
    np.testing.assert_allclose(pb.get_dof_solution("DispX"), [3.0, 3.0, 3.0])
    assert pb.get_ext_forces("MeanDispX")[control.node_disp] == 9.0


def test_mean_motion_disp_default_slave_prefers_central_node():
    pb = _make_square_grid_problem_3d()

    control = fd.constraint.MeanMotion(np.arange(pb.mesh.n_nodes), components="DispX")
    pb.bc.add(control)
    mpc = control._make_mpc(control._mode_indices[0], "MeanDispX")

    assert mpc.list_node_sets[0] == [4]


def test_mean_motion_disp_can_be_prescribed_with_standard_dirichlet_bc():
    pb = _make_problem_2d()

    control = fd.constraint.MeanMotion([0, 1, 2], components="DispY")
    pb.bc.add(control)
    pb.bc.add("Dirichlet", control.node_disp, "MeanDispY", -2.0)
    pb.apply_boundary_conditions()

    mean_y = pb.n_node_dof + pb.global_dof.indice_start("MeanDispY") + control.node_disp

    assert pb._Xbc[mean_y] == -2.0
    assert np.any(np.isclose(pb._Xbc[: pb.n_node_dof], -6.0))


def test_mean_motion_disp_accepts_custom_weights():
    pb = _make_problem_2d()

    control = fd.constraint.MeanMotion(
        [0, 1, 2],
        components="DispX",
        weights=[1.0, 2.0, 1.0],
    )
    pb.bc.add(control)
    pb.bc.add("Dirichlet", control.node_disp, "MeanDispX", 4.0)
    pb.apply_boundary_conditions()

    slave_x = 1
    assert pb._Xbc[slave_x] == 8.0


def test_mean_rigid_motion_uses_surface_area_weights():
    space = fd.ModelingSpace("2Dplane")
    space.new_variable("DispX")
    space.new_variable("DispY")
    space.new_vector("Disp", ("DispX", "DispY"))

    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [4.0, 0.0]])
    elements = np.array([[0, 1], [1, 2]])
    mesh = fd.Mesh(nodes, elements, "lin2")
    pb = fd.Problem(A=sparse.eye(mesh.n_nodes * space.nvar), mesh=mesh, space=space)

    control = fd.constraint.MeanMotion(mesh, components="RotZ")
    pb.bc.add(control)

    np.testing.assert_allclose(control._weights, [0.125, 0.5, 0.375])
    np.testing.assert_allclose(control.center, [2.0, 0.0])


def test_mean_motion_requires_explicit_components():
    pb = _make_problem_3d()

    control = fd.constraint.MeanMotion(pb.mesh)
    try:
        pb.bc.add(control)
    except ValueError as exc:
        assert "components" in str(exc)
    else:
        raise AssertionError("MeanMotion should require explicit components.")


def test_mean_motion_component_aliases_and_duplicates():
    pb = _make_problem_3d()

    control = fd.constraint.MeanMotion(
        pb.mesh,
        components=["RotX", "MeanRotX", "DispZ", "MeanDispZ"],
    )
    pb.bc.add(control)

    assert control._mean_variables == ["MeanDispZ", "MeanRotX"]
    assert pb.global_dof._vector["MeanDisp"] == ["MeanDispZ"]
    assert pb.global_dof._vector["MeanRot"] == ["MeanRotX"]


def test_global_dof_dirichlet_shorthand_accepts_variable_list():
    pb = _make_problem_3d()

    control = fd.constraint.MeanMotion(pb.mesh, components=["DispY", "DispZ"])
    pb.bc.add(control)
    pb.bc.add("Dirichlet", ["MeanDispY", "MeanDispZ"], [-1.0, 0.0])
    pb.apply_boundary_conditions()

    mean_y = control._global_dof_index(pb, "MeanDispY")
    mean_z = control._global_dof_index(pb, "MeanDispZ")
    assert pb._Xbc[mean_y] == -1.0
    assert pb._Xbc[mean_z] == 0.0


def test_mean_motion_vector_aliases_expand_for_dimension():
    pb = _make_problem_3d()

    control = fd.constraint.MeanMotion(pb.mesh, components=["Rot", "Disp"])
    pb.bc.add(control)

    assert control._mean_variables == [
        "MeanDispX",
        "MeanDispY",
        "MeanDispZ",
        "MeanRotX",
        "MeanRotY",
        "MeanRotZ",
    ]


def test_mean_motion_auto_finite_rotation_depends_on_nlgeom_and_rot_components():
    pb = _make_problem_3d()

    linear_rot = fd.constraint.MeanMotion(pb.mesh, components="RotX")
    pb.bc.add(linear_rot)
    assert linear_rot.finite_rotation is False

    pb_nlgeom = _make_problem_3d()
    pb_nlgeom.nlgeom = "UL"
    finite_rot = fd.constraint.MeanMotion(pb_nlgeom.mesh, components="RotX")
    pb_nlgeom.bc.add(finite_rot)
    assert finite_rot.finite_rotation is True
    assert finite_rot._update_during_inc

    pb_disp = _make_problem_3d()
    pb_disp.nlgeom = "UL"
    disp_only = fd.constraint.MeanMotion(pb_disp.mesh, components="Disp")
    pb_disp.bc.add(disp_only)
    assert disp_only.finite_rotation is False


def test_mean_motion_default_translation_slave_prefers_central_node():
    pb = _make_square_grid_problem_3d()

    control = fd.constraint.MeanMotion(np.arange(pb.mesh.n_nodes), components="DispX")
    pb.bc.add(control)
    mpc = control._make_mpc(control._mode_indices[0], "MeanDispX")

    assert mpc.list_node_sets[0] == [4]


def test_mean_motion_quad_surface_weights_keep_corner_symmetry():
    pb = _make_square_grid_problem_3d()

    control = fd.constraint.MeanMotion(pb.mesh, components="DispX")
    pb.bc.add(control)

    corner_nodes = [0, 2, 6, 8]
    corner_weights = [
        control._weights[np.flatnonzero(control.nodes == node)[0]]
        for node in corner_nodes
    ]

    np.testing.assert_allclose(corner_weights, np.full(4, corner_weights[0]))


def test_mean_rigid_motion_projects_2d_rigid_rotation():
    space = fd.ModelingSpace("2Dplane")
    space.new_variable("DispX")
    space.new_variable("DispY")
    space.new_vector("Disp", ("DispX", "DispY"))

    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [4.0, 0.0]])
    elements = np.array([[0, 1], [1, 2]])
    mesh = fd.Mesh(nodes, elements, "lin2")
    pb = fd.Problem(A=sparse.eye(mesh.n_nodes * space.nvar), mesh=mesh, space=space)

    control = fd.constraint.MeanMotion(mesh, components=["Disp", "Rot"])
    pb.bc.add(control)
    pb.bc.add("Dirichlet", control.node_rot, "MeanRotZ", 1.0)
    pb.apply_boundary_conditions()

    q = np.array([0.2, -0.4, 1.5])
    physical_disp = control._mode_matrix @ q

    np.testing.assert_allclose(control._projection @ physical_disp, q)

    rot_z = pb.n_node_dof + pb.global_dof.indice_start("MeanRotZ") + control.node_rot
    assert pb._Xbc[rot_z] == 1.0


def test_mean_rigid_motion_projects_3d_rigid_motion():
    pb = _make_problem_3d()

    control = fd.constraint.MeanMotion(pb.mesh, components=["Disp", "Rot"])
    pb.bc.add(control)

    q = np.array([0.2, -0.4, 0.7, 0.1, -0.2, 0.3])
    physical_disp = control._mode_matrix @ q

    np.testing.assert_allclose(control._projection @ physical_disp, q)
    assert pb.global_dof._vector["MeanRot"] == ["MeanRotX", "MeanRotY", "MeanRotZ"]


def test_mean_motion_rotation_reaction_can_be_reused_as_neumann_moment():
    angle = 0.2
    pb = _make_problem_3d()

    control = fd.constraint.MeanMotion(pb.mesh, components="RotX")
    pb.bc.add(control)
    pb.bc.add("Dirichlet", "MeanRotX", angle)
    pb.apply_boundary_conditions()
    pb.solve()

    moment = pb.get_ext_forces("MeanRotX")[control.node_by_variable["MeanRotX"]]
    assert moment != 0

    pb_neumann = _make_problem_3d()
    control_neumann = fd.constraint.MeanMotion(pb_neumann.mesh, components="RotX")
    pb_neumann.bc.add(control_neumann)
    pb_neumann.bc.add("Neumann", "MeanRotX", moment)
    pb_neumann.apply_boundary_conditions()
    pb_neumann.solve()

    rotation = pb_neumann.get_dof_solution("MeanRotX")[
        control_neumann.node_by_variable["MeanRotX"]
    ]
    np.testing.assert_allclose(rotation, angle)


def test_mean_rigid_motion_finite_rotation_linearization_is_consistent():
    pb = _make_problem_3d()

    control = fd.constraint.MeanMotion(
        pb.mesh, components=["Disp", "Rot"], finite_rotation=True
    )
    pb.bc.add(control)

    q = np.array([0.2, -0.4, 0.7, 0.15, -0.2, 0.35])
    jac, pred = control._finite_tangent_and_prediction(pb, q)
    coeff_u, coeff_q, constants = control._build_finite_linearization(pb, q, pred)

    np.testing.assert_allclose(
        coeff_u @ pred + coeff_q @ q + constants,
        np.zeros(len(q)),
        atol=1e-12,
    )

    eps = 1e-7
    for i in range(len(q)):
        q_eps = q.copy()
        q_eps[i] += eps
        _, pred_eps = control._finite_tangent_and_prediction(pb, q_eps)
        np.testing.assert_allclose((pred_eps - pred) / eps, jac[:, i], atol=1e-6)


def test_mean_rigid_motion_finite_rotation_incremental_linearization():
    pb = _make_problem_3d()

    control = fd.constraint.MeanMotion(
        pb.mesh, components=["Disp", "Rot"], finite_rotation=True
    )
    pb.bc.add(control)

    q = np.array([0.2, -0.4, 0.7, 0.6, -0.2, 0.35])
    u0 = np.zeros(len(control.nodes) * 3)
    coeff_u, coeff_q, residual = control._build_finite_incremental_linearization(
        pb, q, u0
    )

    correction = np.linalg.solve(coeff_q, -residual)
    np.testing.assert_allclose(
        coeff_u @ np.zeros_like(u0) + coeff_q @ correction + residual,
        np.zeros(len(q)),
        atol=1e-12,
    )


def test_mean_rigid_motion_finite_rotation_keeps_free_components_fitted():
    pb = _make_problem_3d()

    control = fd.constraint.MeanMotion(
        pb.mesh, components=["Disp", "RotX"], finite_rotation=True
    )
    pb.bc.add(control)
    pb.bc.add("Dirichlet", control.node_rot, "MeanRotX", 0.45)
    pb._U = np.zeros(pb.n_dof)
    pb._dU = np.zeros(pb.n_dof)
    pb._Xbc = np.zeros(pb.n_dof)
    pb._Xbc[control._global_dof_index(pb, "MeanRotX")] = 0.45
    pb.get_dof_solution = lambda name="all": pb._U + pb._dU

    q_fit = np.array([0.2, -0.4, 0.7, 0.15, -0.2, 0.35])
    _, physical_disp = control._finite_tangent_and_prediction(pb, q_fit)
    for col, value in enumerate(physical_disp):
        node_index, var_index = divmod(col, 3)
        dof = (
            pb.space.variable_rank(control._disp_variables[var_index]) * pb.mesh.n_nodes
            + control.nodes[node_index]
        )
        pb._U[dof] = value

    for var in ["MeanDispX", "MeanDispY", "MeanDispZ"]:
        pb._U[control._global_dof_index(pb, var)] = 10.0

    q0 = control._get_full_total_mean_values(
        pb,
        control._get_total_physical_values(pb),
        controlled=control._finite_dirichlet_mask(pb),
    )

    np.testing.assert_allclose(q0[:3], q_fit[:3], atol=1e-12)
    assert q0[3] == 0.45


def test_mean_rigid_motion_finite_rotation_projects_dirichlet_components():
    pb = _make_problem_3d()

    control = fd.constraint.MeanMotion(pb.mesh, components="RotX", finite_rotation=True)
    pb.bc.add(control)
    pb.bc.add("Dirichlet", control.node_rot, "MeanRotX", 0.6)
    pb._U = np.zeros(pb.n_dof)
    pb._dU = np.zeros(pb.n_dof)
    pb._Xbc = np.zeros(pb.n_dof)
    pb._Xbc[control._global_dof_index(pb, "MeanRotX")] = 0.6
    pb.get_dof_solution = lambda name="all": pb._U + pb._dU

    q_initial = np.array([0.0, 0.0, 0.0, 0.2, 0.1, -0.05])
    _, under_rotated = control._finite_tangent_and_prediction(pb, q_initial)
    for col, value in enumerate(under_rotated):
        node_index, var_index = divmod(col, 3)
        dof = (
            pb.space.variable_rank(control._disp_variables[var_index]) * pb.mesh.n_nodes
            + control.nodes[node_index]
        )
        pb._dU[dof] = value

    control._project_finite_dirichlet_mean_motion(pb, 1.0, None)

    q_fit = control._fit_finite_mean_motion(pb, control._get_total_physical_values(pb))
    np.testing.assert_allclose(q_fit[3], 0.6, atol=1e-12)
    np.testing.assert_allclose(q_fit[4:], q_initial[4:], atol=1e-12)


def test_mean_rigid_motion_finite_rotation_external_dirichlet():
    pb = _make_problem_3d()

    control = fd.constraint.MeanMotion(pb.mesh, components="RotX", finite_rotation=True)
    pb.bc.add(control)
    pb.bc.add("Dirichlet", control.node_rot, "MeanRotX", 0.25)
    pb.apply_boundary_conditions()

    rot_x = pb.n_node_dof + pb.global_dof.indice_start("MeanRotX") + control.node_rot
    assert not control._update_during_inc
    assert pb._Xbc[rot_x] == 0.25
