"""Shared mathematical expressions and scalar histories across frame loads."""

import numpy as np
import pytest

import fedoo as fd
from fedoo.util.expressions import (
    evaluate_expression,
    elements_to_nodes,
    nodes_to_elements,
    selection_mask,
)


@pytest.fixture(params=[False, True], ids=["mesh", "multimesh"])
def data(request):
    mesh = fd.mesh.rectangle_mesh(3, 2, elm_type="quad4", ndim=3)
    if request.param:
        mesh = fd.MultiMesh(
            mesh.nodes, {"quad4": mesh.elements[:1], "tri3": mesh.elements[1:, :3]}
        )
    mesh.node_sets["top edge"] = np.flatnonzero(mesh.nodes[:, 1] == 1)
    if request.param:
        mesh[0].element_sets["wall"] = np.array([0])
    else:
        mesh.element_sets["wall"] = np.array([0])
    result = fd.DataSet(mesh)
    result.node_data["Disp"] = np.vstack(
        (np.arange(mesh.n_nodes), np.zeros((2, mesh.n_nodes)))
    )
    stress = np.zeros((6, mesh.n_elements))
    stress[0] = [10, 20]
    result.element_data["Stress"] = (
        {0: stress[:, :1], 1: stress[:, 1:]} if request.param else stress
    )
    result.scalar_data.update(
        Time=0.0, Force=5.0, **{"Fext(driver_dof)": np.array([1.0, 2.0])}
    )
    return result


def test_shared_mesh_and_field_selection(data):
    mesh = data.mesh
    np.testing.assert_array_equal(
        mesh.find_nodes("X>0 and Y<1"), mesh.find_nodes("X>0 & Y<1")
    )
    np.testing.assert_array_equal(
        mesh.find_nodes("0 < X <= 1"),
        np.flatnonzero((mesh.nodes[:, 0] > 0) & (mesh.nodes[:, 0] <= 1)),
    )
    np.testing.assert_array_equal(mesh.find_nodes("Disp_X >= 3", data=data), [3, 4, 5])
    np.testing.assert_array_equal(mesh.find_elements("Stress_XX > 15", data=data), [1])
    np.testing.assert_array_equal(mesh.find_elements("X", 0.25, data=data), [0])
    np.testing.assert_allclose(
        evaluate_expression('np.sum(Disp_X[nodes("top edge")]) + Force', data=data),
        17.0,
    )
    assert evaluate_expression('sum(scalar("Fext(driver_dof)"))', data=data) == 3.0
    assert (
        evaluate_expression(
            'mean(field("Stress_XX", "Element")[elements("wall")])', data=data
        )
        == 10.0
    )


def test_set_conversion_is_global_and_unique(data):
    mesh = data.mesh
    block = mesh[0] if isinstance(mesh, fd.MultiMesh) else mesh
    nodes = block.elements[0]
    np.testing.assert_array_equal(elements_to_nodes(mesh, [0]), np.unique(nodes))
    np.testing.assert_array_equal(nodes_to_elements(mesh, nodes), [0])
    assert set(nodes_to_elements(mesh, nodes, all_nodes=False)) == {0, 1}
    assert elements_to_nodes(mesh, []).size == 0
    assert nodes_to_elements(mesh, []).size == 0
    with pytest.raises(ValueError):
        elements_to_nodes(mesh, [mesh.n_elements])


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os')",
        "np.load('file')",
        "X.__class__",
        "sum(X, out=X)",
        "[x for x in X]",
    ],
)
def test_expression_disallows_python_execution(data, expression):
    with pytest.raises(ValueError):
        evaluate_expression(expression, data=data)


def test_selection_requires_boolean_per_location(data):
    with pytest.raises(ValueError, match="boolean"):
        selection_mask("Disp_X + 1", data=data)
    with pytest.raises(ValueError):
        selection_mask("Stress > 0", data=data, data_type="Element")


def test_scalar_dependency_validation_is_atomic(data):
    assert data.add_scalar_expression("Total", 'sum(Disp_X[nodes("top edge")])') == 12.0
    assert data.add_scalar_expression("Work", "Total * Force") == 60.0
    with pytest.raises(ValueError, match="Cyclic"):
        data.add_scalar_expression("Total", "Work")
    assert (
        data.scalar_expressions["Total"]["expression"]
        == 'sum(Disp_X[nodes("top edge")])'
    )
    with pytest.raises(ValueError, match="finite"):
        data.add_scalar_expression("Bad", "Disp_X")
    assert "Bad" not in data.scalar_data
    with pytest.raises(ValueError, match="already exists"):
        data.add_scalar_expression("Force", "2")


def test_derived_scalar_history_and_saved_definition(data, tmp_path):
    source = str(tmp_path / "source.fdh5")
    data.to_fdh5(source, iteration=0, overwrite=True)
    data.node_data["Disp"][0] *= 2
    data.scalar_data["Time"] = 1.0
    data.to_fdh5(source, iteration=1)
    sequence = fd.read_data(source)
    sequence.load(0)
    sequence.add_scalar_expression("TopSum", 'sum(Disp_X[nodes("top edge")])')
    sequence.add_scalar_expression("Work", "TopSum * Force")
    np.testing.assert_allclose(sequence.get_history("Work"), [60.0, 120.0])
    sequence_copy = sequence.deepcopy()
    np.testing.assert_allclose(sequence_copy.get_history("Work"), [60.0, 120.0])
    saved = str(tmp_path / "saved.fdh5")
    sequence_copy.save_all(saved)
    reopened = fd.read_data(saved)
    np.testing.assert_allclose(reopened.get_history("Work"), [60.0, 120.0])
    assert (
        reopened.scalar_expressions["TopSum"]["expression"]
        == 'sum(Disp_X[nodes("top edge")])'
    )
    reopened.add_scalar_expression("Work", "TopSum * 2")
    np.testing.assert_allclose(reopened.get_history("Work"), [24.0, 48.0])
