"""Boundary conditions sharing some dof.

The generalized forces of several Neumann bc are summed (they used to
overwrite each other, so that a distributed load erased every load defined
before it) and the bc that can't be applied together on a dof are reported
by ListBC.check.
"""

import warnings

import numpy as np
import pytest
from scipy import sparse

import fedoo as fd
from fedoo.core.boundary_conditions import (
    BoundaryConditionConflict,
    BoundaryConditionWarning,
)


def _build_problem():
    fd.Assembly.delete_memory()
    fd.ModelingSpace("3D")

    mesh = fd.mesh.box_mesh(2, 2, 2, name="bc_box")
    material = fd.constitutivelaw.ElasticIsotrop(1000.0, 0.3, name="BcMat")
    wf = fd.weakform.StressEquilibrium(material)
    solid = fd.Assembly.create(wf, mesh, name="bc_solid")
    pb = fd.problem.NonLinear(solid, nlgeom=False)
    return pb, mesh


def _face(mesh, name):
    box = mesh.bounding_box
    crd = {"top": ("Z", box.zmax), "bottom": ("Z", box.zmin), "right": ("X", box.xmax)}
    return mesh.find_nodes(*crd[name])


def _pressure(mesh, face, value):
    return fd.constraint.Pressure.from_nodes(
        mesh, _face(mesh, face), value, nlgeom=False
    )


def _load_vector(loads):
    pb, mesh = _build_problem()
    for face, value in loads:
        pb.bc.add(_pressure(mesh, face, value))
    pb.apply_boundary_conditions(t_fact=1.0, t_fact_old=0.0)
    return pb.get_B().copy()


def _apply_without_warning(pb):
    with warnings.catch_warnings():
        warnings.simplefilter("error", BoundaryConditionWarning)
        pb.apply_boundary_conditions(t_fact=1.0, t_fact_old=0.0)


def test_distributed_loads_are_summed():
    top = _load_vector([("top", 7.0)])
    right = _load_vector([("right", 3.0)])
    both = _load_vector([("top", 7.0), ("right", 3.0)])

    assert np.linalg.norm(top) > 0.0
    assert np.linalg.norm(right) > 0.0
    np.testing.assert_allclose(both, top + right, atol=1e-12)


def test_nodal_forces_are_summed():
    pb, mesh = _build_problem()
    node = int(_face(mesh, "top")[0])
    pb.bc.add("Neumann", [node], "DispX", 2.0)
    pb.bc.add("Neumann", [node], "DispX", 3.0)

    pb.apply_boundary_conditions(t_fact=1.0, t_fact_old=0.0)

    assert pb.get_B()[node] == 5.0
    assert pb.check_boundary_conditions() == []


def test_node_repeated_in_a_single_bc_is_counted_once():
    pb, mesh = _build_problem()
    node = int(_face(mesh, "top")[0])
    pb.bc.add("Neumann", [node, node], "DispX", 2.0)

    pb.apply_boundary_conditions(t_fact=1.0, t_fact_old=0.0)

    assert pb.get_B()[node] == 2.0


@pytest.mark.parametrize("nodal_first", [True, False])
def test_distributed_load_does_not_erase_nodal_force(nodal_first):
    pressure_only = _load_vector([("top", 7.0)])

    pb, mesh = _build_problem()
    node = int(_face(mesh, "bottom")[0])
    if nodal_first:
        pb.bc.add("Neumann", [node], "DispX", 2.0)
    pb.bc.add(_pressure(mesh, "top", 7.0))
    if not nodal_first:
        pb.bc.add("Neumann", [node], "DispX", 2.0)
    pb.apply_boundary_conditions(t_fact=1.0, t_fact_old=0.0)

    expected = pressure_only.copy()
    expected[node] += 2.0
    np.testing.assert_allclose(pb.get_B(), expected, atol=1e-12)


def _set_converged_state(pb):
    # external forces = A @ X - D = load vector
    load = pb.get_B().copy()
    pb._U = np.ones(pb.n_dof)
    pb.set_A(sparse.identity(pb.n_dof, format="csr"))
    pb.set_X(load)
    pb.set_D(0)


def test_start_value_of_summed_nodal_forces_is_not_doubled():
    pb, mesh = _build_problem()
    node = int(_face(mesh, "top")[0])
    bc1 = pb.bc.add("Neumann", [node], "DispX", 2.0)
    bc2 = pb.bc.add("Neumann", [node], "DispX", 3.0)
    pb.initialize()
    pb.apply_boundary_conditions(t_fact=1.0, t_fact_old=0.0)
    _set_converged_state(pb)

    pb.init_bc_start_value()

    np.testing.assert_allclose(bc1.start_value, [2.0])
    np.testing.assert_allclose(bc2.start_value, [3.0])


def test_start_value_of_nodal_force_excludes_distributed_load():
    pb, mesh = _build_problem()
    node = int(_face(mesh, "top")[0])
    n_nodes = mesh.n_nodes
    bc = pb.bc.add("Neumann", [node], "DispZ", 2.0)
    pb.bc.add(_pressure(mesh, "top", 7.0))
    pb.initialize()
    pb.apply_boundary_conditions(t_fact=1.0, t_fact_old=0.0)
    assert pb.get_B()[2 * n_nodes + node] != 2.0  # pressure on the same dof
    _set_converged_state(pb)

    pb.init_bc_start_value()

    np.testing.assert_allclose(bc.start_value, [2.0])


def test_new_nodal_force_starts_from_the_current_force():
    # usual way to unload: the bc is replaced by a new one on the same dof
    pb, mesh = _build_problem()
    node = int(_face(mesh, "top")[0])
    bc = pb.bc.add("Neumann", [node], "DispX", 2.0)
    pb.initialize()
    pb.apply_boundary_conditions(t_fact=1.0, t_fact_old=0.0)
    _set_converged_state(pb)

    pb.bc.remove(bc)
    new_bc = pb.bc.add("Neumann", [node], "DispX", 0.0)
    pb.init_bc_start_value()

    np.testing.assert_allclose(new_bc.start_value, [2.0])


def test_same_dirichlet_value_on_shared_nodes_is_accepted():
    pb, mesh = _build_problem()
    pb.bc.add("Dirichlet", _face(mesh, "bottom"), "Disp", 0)
    pb.bc.add("Dirichlet", _face(mesh, "right"), "DispX", 0)

    _apply_without_warning(pb)

    assert pb.check_boundary_conditions() == []


def test_different_dirichlet_values_are_reported():
    pb, mesh = _build_problem()
    bottom = _face(mesh, "bottom")
    right = _face(mesh, "right")
    bc1 = pb.bc.add("Dirichlet", bottom, "Disp", 0)
    bc2 = pb.bc.add("Dirichlet", right, "DispX", 0.1)

    with pytest.warns(BoundaryConditionWarning, match="different values"):
        pb.apply_boundary_conditions(t_fact=1.0, t_fact_old=0.0)

    (record,) = pb.check_boundary_conditions()
    assert record["kind"] == "dirichlet_conflict"
    assert record["severity"] == "conflict"
    assert np.array_equal(record["node"], np.intersect1d(bottom, right))
    assert set(record["variable"]) == {"DispX"}
    assert record["bc"] == [bc1, bc2]


def test_check_mode_raise_and_ignore():
    pb, mesh = _build_problem()
    pb.bc.add("Dirichlet", _face(mesh, "bottom"), "Disp", 0)
    pb.bc.add("Dirichlet", _face(mesh, "right"), "DispX", 0.1)

    pb.bc.check_mode = "raise"
    with pytest.raises(BoundaryConditionConflict, match="different values"):
        pb.apply_boundary_conditions(t_fact=1.0, t_fact_old=0.0)

    pb.bc.check_mode = "ignore"
    _apply_without_warning(pb)

    with pytest.raises(ValueError):
        pb.bc.check_mode = "unknown"


def test_check_is_done_once_until_the_list_is_modified():
    pb, mesh = _build_problem()
    pb.bc.add("Dirichlet", _face(mesh, "bottom"), "Disp", 0)
    pb.bc.add("Dirichlet", _face(mesh, "right"), "DispX", 0.1)

    with pytest.warns(BoundaryConditionWarning):
        pb.apply_boundary_conditions(t_fact=1.0, t_fact_old=0.0)
    _apply_without_warning(pb)

    pb.bc.add("Dirichlet", _face(mesh, "top"), "DispZ", 0.2)
    with pytest.warns(BoundaryConditionWarning):
        pb.apply_boundary_conditions(t_fact=1.0, t_fact_old=0.0)


def test_displacement_and_force_on_different_dof_of_a_node():
    pb, mesh = _build_problem()
    top = _face(mesh, "top")
    pb.bc.add("Dirichlet", top, "DispZ", 0.1)
    pb.bc.add("Neumann", top, "DispX", 2.0)

    _apply_without_warning(pb)

    assert pb.check_boundary_conditions() == []


def test_nodal_force_on_a_blocked_dof_is_a_warning():
    pb, mesh = _build_problem()
    top = _face(mesh, "top")
    pb.bc.add("Dirichlet", top, "DispX", 0)
    pb.bc.add("Neumann", top, "DispX", 2.0)
    pb.bc.check_mode = "raise"  # only the conflicts are raised

    with pytest.warns(BoundaryConditionWarning, match="without effect"):
        pb.apply_boundary_conditions(t_fact=1.0, t_fact_old=0.0)

    (record,) = pb.check_boundary_conditions()
    assert record["kind"] == "neumann_on_dirichlet"
    assert record["severity"] == "warning"
    assert np.array_equal(record["node"], np.sort(top))


def test_zero_nodal_force_on_a_blocked_dof_is_accepted():
    pb, mesh = _build_problem()
    top = _face(mesh, "top")
    pb.bc.add("Dirichlet", top, "DispX", 0)
    pb.bc.add("Neumann", top, "DispX", 0)

    _apply_without_warning(pb)


def test_distributed_load_next_to_a_blocked_face_is_accepted():
    pb, mesh = _build_problem()
    pb.bc.add("Dirichlet", _face(mesh, "right"), "Disp", 0)
    pb.bc.add(_pressure(mesh, "top", 7.0))

    _apply_without_warning(pb)

    assert pb.check_boundary_conditions() == []


def test_dirichlet_on_mpc_slave_is_reported():
    pb, mesh = _build_problem()
    slave, master = (int(node) for node in _face(mesh, "top")[:2])
    pb.bc.mpc([[slave], [master]], ["DispX", "DispX"], [1.0, -1.0])
    pb.bc.add("Dirichlet", [slave], "DispX", 0.1)

    with pytest.warns(BoundaryConditionWarning, match="eliminated by a mpc"):
        pb.apply_boundary_conditions(t_fact=1.0, t_fact_old=0.0)

    (record,) = pb.check_boundary_conditions()
    assert record["kind"] == "mpc_slave_dirichlet"
    assert np.array_equal(record["node"], [slave])


def test_dirichlet_on_mpc_master_is_accepted():
    pb, mesh = _build_problem()
    slave, master = (int(node) for node in _face(mesh, "top")[:2])
    pb.bc.mpc([[slave], [master]], ["DispX", "DispX"], [1.0, -1.0])
    pb.bc.add("Dirichlet", [master], "DispX", 0.1)

    _apply_without_warning(pb)


def test_dof_eliminated_by_two_mpc_is_reported():
    pb, mesh = _build_problem()
    slave, master1, master2 = (int(node) for node in _face(mesh, "top")[:3])
    pb.bc.mpc([[slave], [master1]], ["DispX", "DispX"], [1.0, -1.0])
    pb.bc.mpc([[slave], [master2]], ["DispX", "DispX"], [1.0, -1.0])

    with pytest.warns(BoundaryConditionWarning, match="several mpc"):
        pb.apply_boundary_conditions(t_fact=1.0, t_fact_old=0.0)

    (record,) = pb.check_boundary_conditions()
    assert record["kind"] == "mpc_slave_duplicate"
    assert np.array_equal(record["node"], [slave])


if __name__ == "__main__":
    pytest.main([__file__])
