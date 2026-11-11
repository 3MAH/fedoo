"""Plane-stress Simcoon laws: condensed state and return-mapping options."""

import numpy as np
import pytest

import fedoo as fd

pytest.importorskip("simcoon")

ELASTIC = ("ELISO", [1200.0, 0.25, 0.0])
PLASTIC = ("EPICP", [1200.0, 0.25, 0.0, 5.0, 100.0, 0.3])


def _uniaxial_tension(umat, props, nlgeom=False, **options):
    """Uniaxial tension of a plane-stress square, free to contract laterally."""
    fd.ModelingSpace("2Dstress")
    mesh = fd.mesh.rectangle_mesh(3, 3, 0, 1, 0, 1, "quad4")
    material = fd.constitutivelaw.Simcoon(umat, props, **options)
    weakform = fd.weakform.StressEquilibrium(material, nlgeom=nlgeom)
    assembly = fd.Assembly.create(weakform, mesh)
    pb = fd.problem.NonLinear(assembly)
    pb.set_nr_criterion("Displacement", tol=1e-8, max_subiter=10, err0=1)
    pb.bc.add("Dirichlet", mesh.find_nodes("X", 0), "DispX", 0)
    pb.bc.add("Dirichlet", mesh.find_nodes("Y", 0), "DispY", 0)
    pb.bc.add("Dirichlet", mesh.find_nodes("X", 1), "DispX", 0.02)
    pb.nlsolve(dt=0.2, tmax=1, update_dt=False, print_info=0)
    return (
        np.asarray(assembly.sv["Stress"]),
        np.asarray(assembly.sv["Strain"]),
        material,
    )


@pytest.mark.parametrize("nlgeom", [False, True])
@pytest.mark.parametrize("umat,props", [ELASTIC, PLASTIC])
def test_plane_stress_carries_the_out_of_plane_strain(umat, props, nlgeom):
    stress, strain, _ = _uniaxial_tension(umat, props, nlgeom)

    np.testing.assert_allclose(stress[2], 0, atol=1e-12)
    np.testing.assert_allclose(stress[1], 0, atol=1e-6)
    # uniaxial stress of an isotropic law: both lateral strains are equal
    assert np.all(strain[2] < -1e-3)
    np.testing.assert_allclose(strain[2], strain[1], rtol=1e-6)
    np.testing.assert_allclose(strain[4:], 0, atol=1e-14)


def test_plane_stress_elastic_strain_and_stress_match_hooke():
    stress, strain, _ = _uniaxial_tension(*ELASTIC)

    np.testing.assert_allclose(stress[0], 1200.0 * 0.02, rtol=1e-10)
    np.testing.assert_allclose(strain[2], -0.25 * 0.02, rtol=1e-10)


@pytest.mark.parametrize(
    "options",
    [{"ncp": "min"}, {"integrator": "closest_point"}],
)
def test_return_mapping_options_reach_simcoon(options):
    reference, _, _ = _uniaxial_tension(*PLASTIC)
    stress, _, material = _uniaxial_tension(*PLASTIC, **options)

    # different local algorithms, same converged elastoplastic response
    np.testing.assert_allclose(stress[0], reference[0], rtol=1e-6)
    assert not np.allclose(stress[0], 1200.0 * 0.02, rtol=1e-2)
    for key, value in options.items():
        assert getattr(material, key) == value


def test_default_tangent_mode_follows_the_integrator():
    law = fd.constitutivelaw.Simcoon
    assert law(*PLASTIC).tangent_mode == 1
    assert law(*PLASTIC, integrator="closest_point").tangent_mode == 2
    explicit = law(*PLASTIC, tangent_mode=0, integrator="closest_point")
    assert explicit.tangent_mode == 0
    assert law(*PLASTIC).ncp == "fischer_burmeister"
    assert law(*PLASTIC).integrator == "cutting_plane"


def test_simcoon_rejects_unsupported_return_mapping_options():
    # the complementarity function is only read by the cutting-plane laws
    with pytest.raises(ValueError, match="ncp"):
        _uniaxial_tension(*ELASTIC, ncp="min")
    # the closest-point integrator only returns its algorithmic operator
    with pytest.raises(ValueError, match="closest_point"):
        _uniaxial_tension(*PLASTIC, integrator="closest_point", tangent_mode=1)
