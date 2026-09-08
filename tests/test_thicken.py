import numpy as np
import pytest

import fedoo as fd


@pytest.mark.parametrize(
    ("surface_type", "solid_type"), (("tri3", "wed6"), ("quad4", "hex8"))
)
def test_thicken_planar_surface_about_midsurface(surface_type, solid_type):
    surface = fd.mesh.rectangle_mesh(2, 2, elm_type=surface_type, ndim=3)

    solid = fd.mesh.thicken(surface, 0.2)

    assert solid.elm_type == solid_type
    assert solid.n_elements == surface.n_elements
    np.testing.assert_allclose(solid.nodes[0::2, 2], -0.1)
    np.testing.assert_allclose(solid.nodes[1::2, 2], 0.1)
    np.testing.assert_allclose(solid.get_volume(), 0.2)


def test_thicken_supports_multiple_layers_and_explicit_bounds():
    surface = fd.mesh.rectangle_mesh(2, 2, elm_type="quad4", ndim=3)

    solid = fd.mesh.thicken(surface, (-0.1, 0.3), n_nodes=3)

    assert solid.n_elements == 2 * surface.n_elements
    np.testing.assert_allclose(solid.nodes[:3, 2], [-0.1, 0.1, 0.3])
    np.testing.assert_array_equal(
        solid.node_sets["thickness_bottom"], np.arange(surface.n_nodes) * 3
    )
    np.testing.assert_array_equal(
        solid.node_sets["thickness_top"], np.arange(surface.n_nodes) * 3 + 2
    )


def test_thicken_accepts_prescribed_nodal_normals():
    surface = fd.mesh.rectangle_mesh(2, 2, elm_type="quad4", ndim=3)
    normals = np.tile([-2.0, 0.0, 0.0], (surface.n_nodes, 1))

    solid = fd.mesh.thicken(surface, 0.2, normal=normals)

    displacement = solid.nodes[1::2] - solid.nodes[0::2]
    np.testing.assert_allclose(displacement, [[-0.2, 0.0, 0.0]] * 4, atol=1e-12)
