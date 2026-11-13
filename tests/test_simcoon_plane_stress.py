"""Plane-stress Simcoon laws: condensed state and return-mapping options."""

from types import SimpleNamespace

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


@pytest.mark.parametrize("nlgeom", [False, True, "TL"])
@pytest.mark.parametrize("umat,props", [ELASTIC, PLASTIC])
def test_plane_stress_carries_the_out_of_plane_strain(umat, props, nlgeom):
    stress, strain, _ = _uniaxial_tension(umat, props, nlgeom)

    np.testing.assert_allclose(stress[2], 0, atol=1e-12)
    np.testing.assert_allclose(stress[1], 0, atol=1e-6)
    # uniaxial stress of an isotropic law: both lateral strains are equal
    assert np.all(strain[2] < -1e-3)
    np.testing.assert_allclose(strain[2], strain[1], rtol=1e-6)
    np.testing.assert_allclose(strain[4:], 0, atol=1e-14)


@pytest.mark.parametrize("nlgeom", [False, True])
@pytest.mark.parametrize("umat,props", [ELASTIC, PLASTIC])
def test_shell_thickness_points_carry_the_thickness_strain(umat, props, nlgeom):
    space = fd.ModelingSpace("3D")
    mesh = fd.mesh.rectangle_mesh(3, 2, 0, 2, 0, 1, "quad4", ndim=3)
    section = fd.constitutivelaw.ShellHomogeneousNonLinear(
        fd.constitutivelaw.Simcoon(umat, props), 0.15, n_thickness_points=2
    )
    weakform = fd.weakform.PlateEquilibrium(section, nlgeom=nlgeom, space=space)
    assembly = fd.Assembly.create(weakform, mesh)
    pb = fd.problem.NonLinear(assembly)
    pb.set_solver("direct_scipy")
    pb.set_nr_criterion("Displacement", tol=1e-8, max_subiter=20)
    left, right = mesh.find_nodes("X", 0), mesh.find_nodes("X", 2)
    pb.bc.add("Dirichlet", left, ["DispX", "DispZ", "Rot"], 0)
    pb.bc.add("Dirichlet", right, ["DispZ", "Rot"], 0)
    pb.bc.add("Dirichlet", mesh.find_nodes("Y", 0), "DispY", 0)
    pb.bc.add("Dirichlet", right, "DispX", 0.04)
    pb.nlsolve(dt=0.5, update_dt=False, print_info=0)

    # membrane tension: every thickness point is in uniaxial stress
    points = section._material_assembly
    stress = np.asarray(points.sv["Stress"])
    strain = np.asarray(points.sv["Strain"])
    np.testing.assert_allclose(stress[1:], 0, atol=1e-6)
    assert np.all(strain[2] < -1e-3)
    np.testing.assert_allclose(strain[2], strain[1], rtol=1e-6)
    # committed with the increment, hence handed back as the start state
    np.testing.assert_allclose(
        np.asarray(points.sv_start["Strain"])[2], strain[2], rtol=1e-12
    )


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
    # simcoon validates the name of the complementarity function. Which laws
    # read ncp="min" depends on the simcoon version (2.4 serves the legacy
    # names, elastic ones included, with its modular engine), so only the
    # name is checked here.
    with pytest.raises(ValueError, match="ncp"):
        _uniaxial_tension(*PLASTIC, ncp="unknown")
    # the closest-point integrator only returns its algorithmic operator
    with pytest.raises(ValueError, match="closest_point"):
        _uniaxial_tension(*PLASTIC, integrator="closest_point", tangent_mode=1)


@pytest.mark.parametrize("laminate", [False, True])
def test_shell_tangent_follows_the_material_tangent_at_increment_start(laminate):
    # use_elastic_lt: the material starts each increment from its elastic
    # tangent, and so does the shell section
    material = fd.constitutivelaw.Simcoon(*PLASTIC)
    assert material.use_elastic_lt
    if laminate:
        section = fd.constitutivelaw.ShellLaminateNonLinear(
            [material, fd.constitutivelaw.Simcoon(*PLASTIC)], [0.05, 0.1]
        )
    else:
        section = fd.constitutivelaw.ShellHomogeneousNonLinear(material, 0.15)

    class ShellPoint:
        n_gauss_points = 1
        sv = {"ShellStrain": 0}

    point, pb = ShellPoint(), SimpleNamespace(time=0.0, dtime=1.0)
    section.initialize(point, pb)
    elastic = point.sv["_ShellStiffnessMatrix"].copy()

    point.sv["ShellStrain"] = [np.array([value]) for value in [0.02] + 7 * [0.0]]
    section.update(point, pb)
    assert point.sv["_ShellStiffnessMatrix"][0, 0, 0] < 0.9 * elastic[0, 0, 0]
    section.set_start(point, pb)
    np.testing.assert_allclose(point.sv["_ShellStiffnessMatrix"], elastic, rtol=1e-12)
