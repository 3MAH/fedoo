import numpy as np

from fedoo.constitutivelaw import CompositeUD


def test_composite_ud_rotates_scalar_angle():
    unrotated = CompositeUD(angle=0).get_tangent_matrix(None, "3D")
    rotated = CompositeUD(angle=45).get_tangent_matrix(None, "3D")

    assert rotated.shape == (6, 6)
    assert not np.allclose(rotated, unrotated)


def test_composite_ud_rotates_gauss_point_angles():
    angles = np.array([0.0, 45.0])
    rotated = CompositeUD(angle=angles).get_tangent_matrix(None, "3D")

    assert rotated.shape == (6, 6, 2)
    for index, angle in enumerate(angles):
        expected = CompositeUD(angle=angle).get_tangent_matrix(None, "3D")
        np.testing.assert_allclose(rotated[:, :, index], expected)


def test_composite_ud_accepts_radian_angles():
    degrees = CompositeUD(angle=45).get_tangent_matrix(None, "3D")
    radians = CompositeUD(angle=np.pi / 4, degrees=False).get_tangent_matrix(None, "3D")
    np.testing.assert_allclose(radians, degrees)

    degree_field = CompositeUD(angle=np.array([0.0, 45.0])).get_tangent_matrix(
        None, "3D"
    )
    radian_field = CompositeUD(
        angle=np.array([0.0, np.pi / 4]), degrees=False
    ).get_tangent_matrix(None, "3D")
    np.testing.assert_allclose(radian_field, degree_field)
