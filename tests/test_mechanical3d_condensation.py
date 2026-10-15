"""Static condensation of 3D tangents onto reduced stress states."""

import numpy as np
import pytest

import fedoo as fd
from fedoo.constitutivelaw._material_points import _MaterialPointAssembly
from fedoo.core.mechanical3d import Mechanical3D
from fedoo.util.voigt_tensors import StrainTensorList

E, NU = 2.0e5, 0.3


def _orthotropic_H():
    law = fd.constitutivelaw.ElasticOrthotropic(
        1.5e5, 9e3, 8e3, 3e3, 4e3, 5e3, 0.4, 0.3, 0.35
    )
    return np.asarray(law.get_tangent_matrix(None, "3D"), dtype=float)


def test_condensation_is_the_schur_complement():
    H = _orthotropic_H()
    reduced = Mechanical3D.get_H_condensed(H, (0,))
    assert reduced[0, 0] == pytest.approx(1 / np.linalg.inv(H)[0, 0], rel=1e-12)
    mask = np.ones((6, 6), dtype=bool)
    mask[0, 0] = False
    assert np.all(reduced[mask] == 0)
    # Keeping several slots inverts the kept block of the compliance.
    kept = [0, 3, 4]
    reduced = Mechanical3D.get_H_condensed(H, kept)
    np.testing.assert_allclose(
        reduced[np.ix_(kept, kept)],
        np.linalg.inv(np.linalg.inv(H)[np.ix_(kept, kept)]),
        rtol=1e-12,
    )


def test_condensation_of_pointwise_tangents():
    H = _orthotropic_H()
    scales = np.array([1.0, 2.0, 0.5])
    batch = H[:, :, None] * scales
    reduced = Mechanical3D.get_H_condensed(batch, (0,))
    assert reduced.shape == (6, 6, 3)
    np.testing.assert_allclose(
        reduced[0, 0], scales / np.linalg.inv(H)[0, 0], rtol=1e-12
    )
    # Entries mixing scalars and point arrays are broadcast.
    mixed = [
        [batch[i, j] if i == j == 0 else H[i, j] for j in range(6)] for i in range(6)
    ]
    assert Mechanical3D.get_H_condensed(mixed, (0,)).shape == (6, 6, 3)


def test_plane_stress_condensation_is_unchanged_for_isotropic_laws():
    law = fd.constitutivelaw.ElasticIsotrop(E, NU)
    H = np.asarray(law.get_tangent_matrix(None, "3D"), dtype=float)
    np.testing.assert_allclose(
        Mechanical3D.get_H_condensed(H, (0, 1, 3)),
        np.asarray(law.get_H_plane_stress(H), dtype=float),
        rtol=1e-12,
        atol=1e-9,
    )


@pytest.mark.parametrize(
    "law",
    [
        fd.constitutivelaw.ElasticIsotrop(E, NU),
        fd.constitutivelaw.ElasticAnisotropic(
            np.asarray(
                fd.constitutivelaw.ElasticIsotrop(E, NU).get_tangent_matrix(None, "3D"),
                dtype=float,
            )
        ),
    ],
    ids=["isotropic", "anisotropic"],
)
def test_native_elastic_laws_support_uniaxial_stress(law):
    H = np.asarray(law.get_tangent_matrix(None, "1Dstress"), dtype=float)
    expected = np.zeros((6, 6))
    expected[0, 0] = E
    np.testing.assert_allclose(H, expected, rtol=1e-12, atol=1e-6)
    # The stress of a material point is sigma_11 = E eps_11, without
    # Poisson locking.
    points = _MaterialPointAssembly(2, "1Dstress")
    law.initialize(points, None)
    strain = np.zeros((6, 2))
    strain[0] = [1e-3, -2e-3]
    points.sv["Strain"] = StrainTensorList(strain)
    law.update(points, None)
    np.testing.assert_allclose(points.sv["Stress"][0], E * strain[0], rtol=1e-12)


def test_elastoplasticity_rejects_uniaxial_stress():
    law = fd.constitutivelaw.ElastoPlasticity(E, NU, 300.0)
    with pytest.raises(NotImplementedError, match="1Dstress"):
        law.get_elastic_matrix("1Dstress")


def test_simcoon_tangent_is_condensed_for_uniaxial_stress():
    law = fd.constitutivelaw.Simcoon("ELISO", [E, NU, 0.0])
    points = _MaterialPointAssembly(1, "1Dstress")
    assert law._get_ndi(points) == 1
    law.initialize(points, None)
    tangent = np.asarray(points.sv["TangentMatrix"], dtype=float)
    assert tangent[0, 0, 0] == pytest.approx(E, rel=1e-10)
    tangent[0, 0] = 0
    np.testing.assert_allclose(tangent, 0, atol=1e-6)
