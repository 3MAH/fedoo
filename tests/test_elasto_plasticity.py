from copy import deepcopy
from types import SimpleNamespace

import numpy as np
from simcoon import Rotation as SimRotation

import fedoo as fd
from fedoo.constitutivelaw import ElastoPlasticity
from fedoo.core.mechanical3d import MechanicalUMAT
from fedoo.util.voigt_tensors import StrainTensorList, StressTensorList


E = 200e3
NU = 0.3
SIGMA_Y = 300.0
H = 1000.0
BETA = 0.3


def make_material():
    material = ElastoPlasticity(E, NU, SIGMA_Y)
    material.set_hardening_function("power", h=H, beta=BETA)
    return material


def make_assembly(n_points=1, nlgeom=False):
    sv = {
        "Strain": StrainTensorList(np.zeros((6, n_points))),
        "Stress": StressTensorList(np.zeros((6, n_points))),
    }
    if nlgeom:
        # Finite-strain kinematics, including the initial F field, are owned
        # by the weak form. Reproduce that contract in this material-point
        # fixture, which calls the constitutive law directly.
        sv["F"] = np.repeat(np.eye(3)[:, :, None], n_points, axis=2)

    return SimpleNamespace(
        mesh=fd.mesh.rectangle_mesh(2, 2, elm_type="quad4", ndim=3),
        n_elm_gp=n_points,
        n_gauss_points=n_points,
        _nlgeom=nlgeom,
        space=SimpleNamespace(get_dimension=lambda: "3D", list_variables=lambda: []),
        sv=sv,
        sv_component={},
    )


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


def test_finite_element_state_uses_mechanical_umat_labels_only():
    material = ElastoPlasticity(E, NU, SIGMA_Y)
    assembly = make_assembly(n_points=2)
    problem = SimpleNamespace(time=0.0, dtime=0.0)

    material.initialize(assembly, problem)

    assert isinstance(material, MechanicalUMAT)
    assert material.props_label == {
        "young_modulus": 0,
        "poisson_ratio": 1,
        "yield_stress": 2,
    }
    assert material.statev_label == {"P": 0, "EP": slice(1, 7)}
    assert assembly.sv_component["P"] == ("Statev", 0)
    assert assembly.sv_component["EP"] == ("Statev", slice(1, 7))
    assert assembly.sv["Statev"].shape == (7, 2)
    assert "P" not in assembly.sv
    assert "EP" not in assembly.sv


def test_mechanical_umat_update_restarts_from_committed_state():
    material = make_material()
    assembly = make_assembly()
    problem = SimpleNamespace(time=1.0, dtime=0.1)
    material.initialize(assembly, problem)
    assembly.sv_start = deepcopy(assembly.sv)

    first_trial = np.zeros((6, 1))
    first_trial[3, 0] = 0.02
    assembly.sv["DStrain"] = StrainTensorList(first_trial)
    material.update(assembly, problem)
    first_state = assembly.sv["Statev"].copy()

    second_trial = np.zeros((6, 1))
    second_trial[3, 0] = 0.01
    assembly.sv["DStrain"] = StrainTensorList(second_trial)
    material.update(assembly, problem)

    reference = make_material()
    expected_stress = reference.compute_stress(StrainTensorList(second_trial))
    np.testing.assert_allclose(
        assembly.sv["Stress"].asarray(), expected_stress.asarray()
    )
    np.testing.assert_allclose(assembly.sv["Statev"][0], reference.get_plasticity())
    assert not np.array_equal(first_state, assembly.sv["Statev"])


def test_corotational_update_transports_plastic_strain_with_dr():
    material = make_material()
    assembly = make_assembly(nlgeom="UL")
    problem = SimpleNamespace(time=1.0, dtime=0.1)
    material.initialize(assembly, problem)

    plastic_strain = np.array([[0.01], [-0.005], [-0.005], [0.004], [0.0], [0.0]])
    angle = np.pi / 3.0
    rotation_matrix = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    expected = SimRotation.from_matrix(rotation_matrix).apply_strain(plastic_strain)
    expected = np.asarray(expected).reshape(6, 1)

    assembly.sv["Statev"][0, 0] = 0.02
    assembly.sv["Statev"][1:7] = plastic_strain
    assembly.sv["Strain"] = StrainTensorList(expected)
    assembly.sv_start = deepcopy(assembly.sv)
    assembly.sv["DR"] = rotation_matrix[:, :, None]
    assembly.sv["DStrain"] = StrainTensorList(np.zeros((6, 1)))
    material.update(assembly, problem)

    np.testing.assert_allclose(assembly.sv["Statev"][1:7], expected, atol=1e-14)
    np.testing.assert_allclose(assembly.sv["Statev"][0], [0.02])
    np.testing.assert_allclose(assembly.sv["Stress"].asarray(), 0.0, atol=1e-10)


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
