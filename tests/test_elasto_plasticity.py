import numpy as np

from fedoo.constitutivelaw import ElastoPlasticity
from fedoo.util.voigt_tensors import StrainTensorList


E = 200e3
NU = 0.3
SIGMA_Y = 300.0
H = 1000.0
BETA = 0.3


def make_material():
    material = ElastoPlasticity(E, NU, SIGMA_Y)
    material.set_hardening_function("power", h=H, beta=BETA)
    return material


def test_radial_return_matches_simcoon_simple_shear_reference():
    material = make_material()
    plasticity = np.zeros(1)
    plastic_strain = np.zeros((6, 1))

    for gamma in np.linspace(0.0, 0.02, 21)[1:]:
        strain = np.zeros((6, 1))
        strain[3, 0] = gamma
        stress = material.compute_stress(
            StrainTensorList(strain),
            plasticity,
            plastic_strain,
        )
        plasticity = material.get_plasticity().copy()
        plastic_strain = material.get_plastic_strain().asarray(copy=True)

    # Reference values produced by Simcoon's EPICP law for the same 20
    # proportional shear increments.
    np.testing.assert_allclose(stress[3], [314.581143696], rtol=1e-9)
    np.testing.assert_allclose(material.get_plasticity(), [0.00918589977986], rtol=1e-9)
    np.testing.assert_allclose(
        material.yield_function(stress, material.get_plasticity()),
        [0.0],
        atol=1e-6,
    )


def test_plastic_tangent_uses_positive_hardening_denominator():
    material = make_material()
    strain = np.zeros((6, 1))
    strain[3, 0] = 0.02
    material.compute_stress(StrainTensorList(strain))

    plasticity = material.get_plasticity()[0]
    hardening_slope = BETA * H * plasticity ** (BETA - 1)
    shear_modulus = E / (2 * (1 + NU))
    expected_shear_tangent = (
        shear_modulus * hardening_slope / (3 * shear_modulus + hardening_slope)
    )

    tangent = material.get_tangent_matrix()
    np.testing.assert_allclose(tangent[3, 3, 0], expected_shear_tangent, rtol=1e-12)


def test_vectorized_integration_matches_independent_material_points():
    gammas = np.array([0.0, 0.001, 0.003, 0.01, 0.02])
    strains = np.zeros((6, len(gammas)))
    strains[3] = gammas

    vectorized_material = make_material()
    vectorized_stress = vectorized_material.compute_stress(StrainTensorList(strains))

    independent_stress = np.empty_like(strains)
    for point, gamma in enumerate(gammas):
        material = make_material()
        strain = np.zeros((6, 1))
        strain[3, 0] = gamma
        independent_stress[:, point] = material.compute_stress(
            StrainTensorList(strain)
        ).asarray()[:, 0]

    np.testing.assert_allclose(
        vectorized_stress.asarray(),
        independent_stress,
        rtol=1e-12,
        atol=1e-12,
    )


def test_legacy_commit_api_was_removed():
    material = make_material()
    assert not hasattr(material, "NewTimeIncrement")
    assert not hasattr(material, "ComputeStress")


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
