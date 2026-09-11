import numpy as np
import pytest

import fedoo as fd


def _beam_mesh():
    return fd.Mesh(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 1.0, 0.0]]),
        np.array([[0, 1], [1, 2]]),
        "lin2",
    )


def _assembly(mesh):
    space = fd.ModelingSpace("3D")
    weakform = fd.WeakForm(None, space=space)
    return fd.Assembly(weakform, mesh)


def test_mesh_accepts_one_beam_guide_per_element():
    mesh = _beam_mesh()
    guides = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    frames = mesh.get_element_local_frame(guide=guides, location="Element")

    tangents = mesh.nodes[mesh.elements[:, 1]] - mesh.nodes[mesh.elements[:, 0]]
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
    np.testing.assert_allclose(frames[:, 0], tangents)
    np.testing.assert_allclose(frames[0, 1], guides[0])
    np.testing.assert_allclose(frames[1, 1], guides[1])


def test_mesh_interpolates_nodal_beam_guides():
    mesh = _beam_mesh()
    nodal_guides = np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]])

    frames = mesh.get_element_local_frame(guide=nodal_guides, location="Node")

    expected_midpoint_guides = np.array([[0.0, 1.0, 0.5], [0.0, 0.5, 1.0]])
    # The projected guide defines the local y direction.
    projected = (
        expected_midpoint_guides
        - np.sum(expected_midpoint_guides * frames[:, 0], axis=1, keepdims=True)
        * frames[:, 0]
    )
    projected /= np.linalg.norm(projected, axis=1, keepdims=True)
    np.testing.assert_allclose(frames[:, 1], projected)


def test_assembly_projects_full_shell_frame_onto_geometry():
    mesh = fd.mesh.rectangle_mesh(2, 2, elm_type="quad4", ndim=3)
    assembly = _assembly(mesh)
    supplied = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    assert assembly.set_element_local_frame(supplied) is assembly
    frames = assembly.get_element_local_frame()

    np.testing.assert_allclose(frames[:, 0], [[0.0, 1.0, 0.0]])
    np.testing.assert_allclose(frames[:, 2], [[0.0, 0.0, 1.0]])
    assert assembly._element_local_frame.shape == (mesh.n_elements, 1, 3, 3)


def test_invalid_element_frame_definitions_are_rejected():
    mesh = _beam_mesh()
    with pytest.raises(ValueError, match="parallel to the element tangent"):
        mesh.get_element_local_frame(guide=[1.0, 0.0, 0.0])

    assembly = _assembly(_beam_mesh())
    with pytest.raises(ValueError, match="Node.*Element"):
        assembly.set_element_local_frame(np.eye(3), location="GaussPoint")

    with pytest.raises(ValueError, match="guide location"):
        mesh.get_element_local_frame(
            guide=np.tile([0.0, 1.0, 0.0], (mesh.n_elements, 1)),
            location="element",
        )
