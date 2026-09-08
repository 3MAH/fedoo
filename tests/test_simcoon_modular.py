from types import SimpleNamespace

import numpy as np
import pytest
from simcoon import modular

import fedoo as fd
from fedoo.util.voigt_tensors import StrainTensorList, StressTensorList


def _initialized_assembly(material):
    assembly = SimpleNamespace(
        mesh=fd.mesh.rectangle_mesh(2, 2, elm_type="quad4", ndim=3),
        n_elm_gp=1,
        n_gauss_points=1,
        _nlgeom=False,
        space=SimpleNamespace(get_dimension=lambda: "3D", list_variables=lambda: []),
        sv={},
        sv_component={},
    )
    problem = SimpleNamespace(time=0.0, dtime=1.0)
    material.initialize(assembly, problem)
    assembly.sv["Strain"] = StrainTensorList(np.zeros((6, 1)))
    assembly.sv["Stress"] = StressTensorList(np.zeros((6, 1)))
    assembly.sv_start = dict(assembly.sv)
    return assembly, problem


def test_from_modular_builds_a_modul_law_with_documented_metadata():
    configuration = modular.elastic_model(E=210000.0, nu=0.3, alpha=1e-5)
    material = fd.constitutivelaw.Simcoon.from_modular(
        configuration, name="modular_elastic"
    )

    assert material.umat_name == "MODUL"
    assert material.modular_material is configuration
    assert material.n_statev == configuration.nstatev == 1
    assert material.is_isotropic
    assert material.props.flags.f_contiguous
    np.testing.assert_allclose(material.props[:, 0], configuration.props)
    assert material.props_label == {
        "elasticity.type": 0,
        "elasticity.convention": 1,
        "elasticity.C1": 2,
        "elasticity.C2": 3,
        "elasticity.alpha": 4,
        "mechanisms.count": 5,
    }
    assert material.statev_label == {"T_init": 0}
    assert "from_modular" in fd.constitutivelaw.Simcoon.__doc__


def test_simcoon_constructors_forward_tangent_mode():
    legacy = fd.constitutivelaw.Simcoon(
        "ELISO", np.array([210000.0, 0.3, 0.0]), tangent_mode=2
    )
    modular_material = fd.constitutivelaw.Simcoon.from_modular(
        modular.elastic_model(E=210000.0, nu=0.3), tangent_mode=2
    )

    assert legacy.tangent_mode == 2
    assert modular_material.tangent_mode == 2


def test_simcoon_constructors_keep_positional_name_compatibility():
    legacy = fd.constitutivelaw.Simcoon(
        "ELISO", np.array([210000.0, 0.3, 0.0]), "legacy_name"
    )
    modular_material = fd.constitutivelaw.Simcoon.from_modular(
        modular.elastic_model(E=210000.0, nu=0.3), "modular_name"
    )

    assert legacy.name == "legacy_name"
    assert legacy.tangent_mode == 1
    assert modular_material.name == "modular_name"
    assert modular_material.tangent_mode == 1


def test_modular_labels_follow_multiple_mechanism_state_layout():
    plasticity = modular.Plasticity(
        sigma_Y=300.0,
        isotropic_hardening=modular.VoceHardening(Q=100.0, b=10.0),
        kinematic_hardening=modular.ChabocheHardening(
            terms=((1000.0, 10.0), (500.0, 5.0))
        ),
    )
    viscoelasticity = modular.Viscoelasticity(terms=((30000.0, 0.3, 1e5, 5e4),))
    damage = modular.Damage(
        Y_0=1.0,
        Y_c=10.0,
        damage_type=modular.DamageType.WEIBULL,
        A=2.0,
        n=3.0,
    )
    configuration = modular.ModularMaterial(
        elasticity=modular.IsotropicElasticity(210000.0, 0.3),
        mechanisms=[plasticity, viscoelasticity, damage],
    )

    material = fd.constitutivelaw.Simcoon.from_modular(configuration)

    assert len(material.props_label) == configuration.nprops
    assert material.statev_label["mechanism_0.p"] == 1
    assert material.statev_label["mechanism_0.EP"] == slice(2, 8)
    assert material.statev_label["mechanism_0.a_0"] == slice(8, 14)
    assert material.statev_label["mechanism_0.a_1"] == slice(14, 20)
    assert material.statev_label["mechanism_1.v_0"] == 20
    assert material.statev_label["mechanism_1.EV_0"] == slice(21, 27)
    assert material.statev_label["mechanism_2.D"] == 27
    assert material.statev_label["mechanism_2.Y_max"] == 28
    assert material.n_statev == configuration.nstatev == 29
    assert material.props_label["mechanism_2.A"] == configuration.nprops - 2
    assert material.props_label["mechanism_2.n"] == configuration.nprops - 1


def test_modular_isotropy_accounts_for_elasticity_and_yield_criterion():
    isotropic = modular.ModularMaterial(
        modular.IsotropicElasticity(210000.0, 0.3),
        [modular.Plasticity(300.0)],
    )
    anisotropic_elasticity = modular.ModularMaterial(
        modular.CubicElasticity(210000.0, 0.3, 80000.0)
    )
    anisotropic_yield = modular.ModularMaterial(
        modular.IsotropicElasticity(210000.0, 0.3),
        [
            modular.Plasticity(
                300.0,
                yield_criterion=modular.HillYield(1, 1, 1, 1, 1, 1),
            )
        ],
    )

    assert fd.constitutivelaw.Simcoon.from_modular(isotropic).is_isotropic
    assert not fd.constitutivelaw.Simcoon.from_modular(
        anisotropic_elasticity
    ).is_isotropic
    assert not fd.constitutivelaw.Simcoon.from_modular(anisotropic_yield).is_isotropic


def test_modular_elastic_response_matches_the_legacy_eliso_umat():
    E, nu, alpha = 210000.0, 0.3, 0.0
    modular_material = fd.constitutivelaw.Simcoon.from_modular(
        modular.elastic_model(E, nu, alpha)
    )
    legacy_material = fd.constitutivelaw.Simcoon("ELISO", np.array([E, nu, alpha]))
    modular_assembly, problem = _initialized_assembly(modular_material)
    legacy_assembly, _ = _initialized_assembly(legacy_material)
    dstrain = np.array([[0.01], [-0.003], [0.002], [0.004], [0.001], [-0.002]])
    modular_assembly.sv["DStrain"] = StrainTensorList(dstrain)
    legacy_assembly.sv["DStrain"] = StrainTensorList(dstrain)

    modular_material.update(modular_assembly, problem)
    legacy_material.update(legacy_assembly, problem)

    np.testing.assert_allclose(
        modular_assembly.sv["Stress"].asarray(),
        legacy_assembly.sv["Stress"].asarray(),
        rtol=1e-13,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        modular_assembly.sv["TangentMatrix"],
        legacy_assembly.sv["TangentMatrix"],
        rtol=1e-13,
        atol=1e-13,
    )


def test_modular_finite_strain_requires_log_r_corotation():
    material = fd.constitutivelaw.Simcoon.from_modular(
        modular.elastic_model(210000.0, 0.3)
    )
    assembly = SimpleNamespace(_nlgeom="UL", weakform=SimpleNamespace(corate="log"))

    with pytest.raises(ValueError, match="requires corate=.*log_r"):
        material._validate_kinematics(assembly)

    assembly.weakform.corate = "log_R"
    material._validate_kinematics(assembly)
    assembly.weakform.corate = "log_R_inc"
    material._validate_kinematics(assembly)
