"""Nonlinear multifiber beam sections (BeamFiberSection)."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

import fedoo as fd
from fedoo.constitutivelaw._material_points import _MaterialPointAssembly
from fedoo.core.mechanical3d import MechanicalUMAT
from fedoo.util.beam_recovery import SectionRecoveryUnavailable
from fedoo.util.voigt_tensors import StrainTensorList

E, NU = 2.0e5, 0.3
SIGMA_Y = 300.0


def _simcoon_epicp():
    """Perfectly plastic J2 Simcoon law (power-law hardening with k = 0)."""
    return fd.constitutivelaw.Simcoon("EPICP", [E, NU, 0.0, SIGMA_Y, 0.0, 1.0])


def _simcoon_supports_uniaxial_stress():
    """Whether Simcoon condenses the uniaxial stress state (ndi=1)."""
    law = _simcoon_epicp()
    points = _MaterialPointAssembly(1, "1Dstress")
    law.initialize(points, None)
    points.sv_start = dict(points.sv)
    strain = np.zeros((6, 1))
    strain[0] = 0.1 * SIGMA_Y / E
    points.sv["Strain"] = StrainTensorList(strain)
    law.update(points, SimpleNamespace(time=0.0, dtime=1.0))
    return np.isclose(points.sv["Stress"][0][0], 0.1 * SIGMA_Y, rtol=1e-10)


def _isotropic_compliance(n_points):
    lam = E * NU / ((1 + NU) * (1 - 2 * NU))
    mu = E / (2 * (1 + NU))
    L = np.zeros((6, 6))
    L[:3, :3] = lam
    L[np.arange(3), np.arange(3)] += 2 * mu
    L[np.arange(3, 6), np.arange(3, 6)] = mu
    return np.repeat(np.linalg.inv(L)[None], n_points, axis=0)


def _uniaxial_plasticity(
    strain,
    dstrain,
    F0,
    F1,
    stress,
    DR,
    props,
    statev,
    time,
    dtime,
    wm,
    temperature,
    *,
    ndi,
    tangent_mode,
    start,
    corate,
):
    """Uniaxial J2 plasticity with linear isotropic hardening (test UMAT).

    The 3D tangent is built so that its uniaxial condensation is the
    elastoplastic modulus E H / (E + H).
    """
    # The section probes the elastic tangent at a 3D point at construction.
    assert ndi == 1 or start
    young, sigma_y, hardening = np.ravel(props)[:3]
    strain = np.asarray(strain) + np.asarray(dstrain)
    n = strain.shape[1]
    plastic, cumulated = np.asarray(statev)
    trial = young * (strain[0] - plastic)
    yield_function = np.abs(trial) - (sigma_y + hardening * cumulated)
    loading = yield_function > 0
    dp = np.where(loading, yield_function / (young + hardening), 0.0)
    sigma = trial - young * dp * np.sign(trial)
    modulus = np.where(loading, young * hardening / (young + hardening), young)
    # Keep the compliance finite under perfect plasticity.
    modulus = np.maximum(modulus, 1e-9 * young)
    compliance = _isotropic_compliance(n)
    compliance[:, 0, 0] = 1 / modulus
    tangent = np.linalg.inv(compliance).transpose(1, 2, 0)
    stress_out = np.zeros((6, n))
    stress_out[0] = sigma
    statev_out = np.asfortranarray(
        [plastic + dp * np.sign(trial), cumulated + dp], dtype=float
    )
    return stress_out, statev_out, wm, tangent


def _plastic_material(hardening=0.0):
    return MechanicalUMAT(
        _uniaxial_plasticity,
        [E, SIGMA_Y, hardening],
        n_statev=2,
        is_isotropic=True,
    )


SECTIONS = {
    "rectangle": lambda m, k=5 / 6: fd.constitutivelaw.BeamRectangular(
        m, 0.2, 0.1, k=k
    ),
    "circle": lambda m, k=0.9: fd.constitutivelaw.BeamCircular(m, 0.05, k=k),
    "pipe": lambda m, k=0.5: fd.constitutivelaw.BeamPipe(m, 0.03, 0.05, k=k),
}


class _SectionPoint:
    """A single beam Gauss point driving a section law."""

    def __init__(self, section):
        self.n_gauss_points = 1
        self.sv = {"BeamStrain": 0, "BeamStress": 0}
        self.pb = SimpleNamespace(time=0.0, dtime=1.0)
        self.section = section
        section.initialize(self, self.pb)

    def stress(self, strain):
        self.sv["BeamStrain"] = [np.atleast_1d(float(value)) for value in strain]
        self.section.update(self, self.pb)
        return np.array([np.ravel(c)[0] for c in self.sv["BeamStress"]])

    def tangent(self):
        return self.sv["_BeamStiffnessMatrix"][:, :, 0]

    def commit(self):
        self.section.set_start(self, self.pb)


@pytest.mark.parametrize("shape", SECTIONS)
def test_elastic_section_tangent_matches_the_plain_section(shape):
    material = fd.constitutivelaw.ElasticIsotrop(E, NU)
    plain = SECTIONS[shape](material)
    section = fd.constitutivelaw.BeamFiberSection(plain)
    point = _SectionPoint(section)
    expected = np.diag(np.asarray(plain.get_beam_rigidity(), dtype=float))
    scale = np.abs(expected).max()
    np.testing.assert_allclose(
        point.tangent(), expected, rtol=1e-12, atol=1e-13 * scale
    )
    strain = np.array([1e-4, 2e-4, -1e-4, 3e-3, -2e-3, 4e-3])
    np.testing.assert_allclose(
        point.stress(strain), expected @ strain, rtol=1e-12, atol=1e-13 * scale
    )
    # Section properties and elastic constants come from the plain section.
    assert section.material.E == pytest.approx(E)
    assert section.material.nu == pytest.approx(NU)
    np.testing.assert_allclose(section.get_beam_rigidity(), plain.get_beam_rigidity())


def test_elastic_constants_are_probed_from_a_umat():
    section = fd.constitutivelaw.BeamFiberSection(
        SECTIONS["rectangle"](_plastic_material())
    )
    assert section.material.E == pytest.approx(E, rel=1e-12)
    assert section.material.G == pytest.approx(E / (2 * (1 + NU)), rel=1e-12)
    assert section.material.nu == pytest.approx(NU, rel=1e-12)


def _moment_curvature(a, b, kappa_ratio):
    """Elastic-perfectly plastic rectangle bent about z (depth a along y)."""
    kappa_y = 2 * SIGMA_Y / (E * a)
    plastic_moment = SIGMA_Y * b * a**2 / 4
    return kappa_y * kappa_ratio, plastic_moment * (1 - 1 / (3 * kappa_ratio**2))


def test_plastic_moment_curvature_of_a_rectangle():
    a, b = 0.2, 0.1
    section = fd.constitutivelaw.BeamFiberSection(
        fd.constitutivelaw.BeamRectangular(_plastic_material(), a, b),
        n_fibers=(8, 2),
    )
    point = _SectionPoint(section)
    # With 8 layers the elastic core edge falls on a layer boundary for
    # these curvatures, where the composite Gauss rule is exact.
    for ratio in (0.5, 1.0, 2.0, 4.0):
        kappa, moment = _moment_curvature(a, b, ratio)
        if ratio <= 1:
            moment = E * b * a**3 / 12 * kappa
        stress = point.stress([0, 0, 0, 0, 0, kappa])
        assert stress[5] == pytest.approx(moment, rel=1e-10)
        assert stress[0] == pytest.approx(0, abs=1e-8 * SIGMA_Y * a * b)
        point.commit()
    kappa, moment = _moment_curvature(a, b, 50.0)
    assert point.stress([0, 0, 0, 0, 0, kappa])[5] == pytest.approx(moment, rel=1e-3)
    # Bending about y uses the z coordinate: M_Y = +int z sigma.
    other = _SectionPoint(
        fd.constitutivelaw.BeamFiberSection(
            fd.constitutivelaw.BeamRectangular(_plastic_material(), b, a),
            n_fibers=(2, 8),
        )
    )
    kappa, moment = _moment_curvature(a, b, 2.0)
    assert other.stress([0, 0, 0, 0, kappa, 0])[4] == pytest.approx(moment, rel=1e-10)


@pytest.mark.parametrize("shape", SECTIONS)
def test_plastic_section_tangent_matches_finite_differences(shape):
    section = fd.constitutivelaw.BeamFiberSection(
        SECTIONS[shape](_plastic_material(hardening=E / 20))
    )
    point = _SectionPoint(section)
    depth = 0.2 if shape == "rectangle" else 0.1
    kappa_y = 2 * SIGMA_Y / (E * depth)
    strain = np.array(
        [0.4 * SIGMA_Y / E, 1e-4, -2e-4, 1e-3, 1.5 * kappa_y, 2.5 * kappa_y]
    )
    point.stress(strain)
    tangent = point.tangent().copy()
    # Yielding couples the axial force to both bending moments.
    assert abs(tangent[0, 5]) > 1e-3 * abs(tangent[0, 0]) * depth
    np.testing.assert_allclose(tangent, tangent.T, rtol=1e-10, atol=1e-10)
    h = 1e-7
    numerical = np.empty((6, 6))
    for j in range(6):
        step = np.zeros(6)
        step[j] = h * max(abs(strain[j]), 1e-4)
        numerical[:, j] = (
            point.stress(strain + step) - point.stress(strain - step)
        ) / (2 * step[j])
    scale = np.abs(tangent).max()
    np.testing.assert_allclose(numerical, tangent, rtol=1e-5, atol=1e-6 * scale)


def test_rollback_restores_the_fiber_state():
    section = fd.constitutivelaw.BeamFiberSection(
        SECTIONS["rectangle"](_plastic_material(hardening=E / 20))
    )
    point = _SectionPoint(section)
    kappa = 3 * 2 * SIGMA_Y / (E * 0.2)
    first = point.stress([0, 0, 0, 0, 0, kappa])
    point.commit()
    committed = deepcopy(section._material_assembly.sv)
    point.stress([0, 0, 0, 0, 0, 3 * kappa])
    section.to_start(point, point.pb)
    for key in ("Statev", "Stress", "Strain"):
        np.testing.assert_array_equal(
            np.asarray(section._material_assembly.sv[key]), np.asarray(committed[key])
        )
    # Unloading back to the committed curvature is elastic from the start state.
    np.testing.assert_allclose(
        point.stress([0, 0, 0, 0, 0, kappa]),
        first,
        rtol=1e-12,
        atol=1e-12 * np.abs(first).max(),
    )


def _cantilever(shape, dim, nlgeom, elm_type, fiber, k=None, consistent=False):
    space = fd.ModelingSpace("2D" if dim == 2 else "3D")
    mesh = fd.mesh.line_mesh(n_nodes=9, x_min=0, x_max=2.0, ndim=dim)
    material = fd.constitutivelaw.ElasticIsotrop(E, NU)
    section = SECTIONS[shape](material) if k is None else SECTIONS[shape](material, k=k)
    if fiber:
        section = fd.constitutivelaw.BeamFiberSection(section)
    wf = fd.weakform.BeamEquilibrium(
        section, nlgeom=nlgeom, space=space, consistent_tangent=consistent
    )
    assembly = fd.Assembly.create(wf, mesh, elm_type)
    pb = fd.problem.NonLinear(assembly)
    end = [mesh.n_nodes - 1]
    # Loads giving a 1e-3 axial strain, tip deflections of 3% and 2% of the
    # length and a 0.01 rad twist.
    EA, _, _, GJ, EIyy, EIzz = section.get_beam_rigidity()
    length = 2.0
    pb.bc.add("Dirichlet", [0], "Disp", 0)
    pb.bc.add("Dirichlet", [0], "Rot", 0)
    pb.bc.add("Neumann", end, "DispX", 1e-3 * EA)
    pb.bc.add("Neumann", end, "DispY", -0.09 * EIzz / length**2)
    if dim == 3:
        pb.bc.add("Neumann", end, "DispZ", 0.06 * EIyy / length**2)
        pb.bc.add("Neumann", end, "RotX", 0.01 * GJ / length)
    pb.set_nr_criterion("Displacement", tol=1e-10, max_subiter=20)
    pb.nlsolve(dt=0.5, tmax=1.0, update_dt=False, print_info=0)
    return pb, assembly


@pytest.mark.parametrize(
    "shape,dim,nlgeom,elm_type,k",
    [
        ("rectangle", 2, False, "beam", None),
        ("rectangle", 3, False, "bernoullibeam", 0),
        ("rectangle", 2, "UL", "beam", 0),
        ("circle", 3, False, "beam", None),
        ("pipe", 3, "UL", "beam", None),
        ("rectangle", 3, "UL", "beam", 0),
    ],
)
def test_elastic_fiber_cantilever_matches_the_plain_section(
    shape, dim, nlgeom, elm_type, k
):
    reference, _ = _cantilever(shape, dim, nlgeom, elm_type, fiber=False, k=k)
    pb, assembly = _cantilever(shape, dim, nlgeom, elm_type, fiber=True, k=k)
    expected = reference.get_dof_solution()
    np.testing.assert_allclose(
        pb.get_dof_solution(), expected, rtol=1e-9, atol=1e-12 * np.abs(expected).max()
    )
    assert assembly.weakform.constitutivelaw is assembly.weakform.properties


def test_elastic_sections_keep_no_constitutive_law():
    fd.ModelingSpace("3D")
    section = SECTIONS["circle"](fd.constitutivelaw.ElasticIsotrop(E, NU))
    assert fd.weakform.BeamEquilibrium(section).constitutivelaw is None


def test_saved_stress_is_the_fiber_stress():
    pb, assembly = _cantilever("rectangle", 3, False, "beam", fiber=True)
    section = assembly.weakform.properties
    data = pb.get_results(assembly, ["Stress", "Strain"])
    stress = np.asarray(data.gausspoint_data["Stress"])
    assert stress.shape == (6, section.n_fibers, assembly.n_gauss_points)
    np.testing.assert_allclose(stress, section.integration_stresses(assembly))
    metadata = data.field_metadata["Stress"]
    assert metadata["recovery"] == "stored_only"
    np.testing.assert_allclose(metadata["stored_points"], section.output_points)
    # The fiber axial strain is the kinematic section strain at the fibers.
    strain = np.asarray(data.gausspoint_data["Strain"])
    np.testing.assert_allclose(
        stress[0], E * strain[0], rtol=1e-9, atol=1e-9 * np.abs(stress[0]).max()
    )
    with pytest.raises(SectionRecoveryUnavailable):
        section.get_stress(assembly, position=(0.0, 0.0))


@pytest.mark.skipif(
    not _simcoon_supports_uniaxial_stress(),
    reason="this Simcoon version does not condense the ndi=1 stress state",
)
def test_simcoon_plastic_moment_of_a_rectangle():
    a, b = 0.2, 0.1
    section = fd.constitutivelaw.BeamFiberSection(
        fd.constitutivelaw.BeamRectangular(_simcoon_epicp(), a, b),
        n_fibers=(8, 2),
    )
    point = _SectionPoint(section)
    for ratio in (2.0, 4.0):
        kappa, moment = _moment_curvature(a, b, ratio)
        assert point.stress([0, 0, 0, 0, 0, kappa])[5] == pytest.approx(
            moment, rel=1e-8
        )
        point.commit()


def test_one_section_per_assembly():
    material = fd.constitutivelaw.ElasticIsotrop(E, NU)
    section = fd.constitutivelaw.BeamFiberSection(SECTIONS["circle"](material))
    _SectionPoint(section)
    with pytest.raises(RuntimeError, match="one assembly"):
        _SectionPoint(section)


def _plastic_beam(dim, shear):
    """Deformed UL beam with yielded fibers, as in the corotational tests."""
    space = fd.ModelingSpace("2D" if dim == 2 else "3D")
    nodes = np.array([[0.0, 0.0, 0.0], [2.0, 0.3, 0.2], [3.0, 1.0, 0.5]])[:, :dim]
    mesh = fd.Mesh(nodes, np.array([[0, 1], [1, 2]]), "lin2")
    section = fd.constitutivelaw.BeamFiberSection(
        fd.constitutivelaw.BeamRectangular(
            _plastic_material(hardening=E / 20), 0.3, 0.2, k=shear
        ),
        n_fibers=(4, 2),
    )
    wf = fd.weakform.BeamEquilibrium(
        section, nlgeom=True, space=space, consistent_tangent=True
    )
    assembly = fd.Assembly.create(wf, mesh)
    pb = fd.problem.NonLinear(assembly)
    assembly.initialize(pb)
    u = np.zeros((space.nvar, mesh.n_nodes))
    u[space.variable_rank("DispX"), 1:] = 0.012
    u[space.variable_rank("DispY"), 1:] = 0.4
    if dim == 2:
        u[space.variable_rank("RotZ")] = [0.0, 0.05, 0.08]
    else:
        u[space.variable_rank("DispZ"), 1:] = -0.25
        for name, values in zip(
            ["RotX", "RotY", "RotZ"],
            [[0.0, 0.01, 0.02], [0, 0.03, -0.04], [0, 0.05, 0.08]],
        ):
            u[space.variable_rank(name)] = values
    pb._dU = u.ravel()
    pb.set_X(pb._dU.copy())
    assembly.update(pb)
    pb.set_X(np.zeros_like(pb._dU))
    assembly.update(pb)
    return wf, assembly, pb


@pytest.mark.parametrize("dim,shear", [(2, 0), (2, 5 / 6), (3, 0), (3, 5 / 6)])
def test_consistent_tangent_with_yielded_fibers(dim, shear):
    wf, assembly, pb = _plastic_beam(dim, shear)
    section = wf.properties
    cumulated = np.asarray(section._material_assembly.sv["Statev"])[1]
    assert np.any(cumulated > 0), "the fibers must have yielded"
    tangent = assembly.current.global_matrix.toarray()
    state = deepcopy(assembly.sv)
    base = pb._dU.copy()
    frame = assembly.current._element_local_frame.copy()
    pb._line_search_update = True

    def residual(delta):
        assembly.sv = assembly.current.sv = deepcopy(state)
        assembly.current._element_local_frame = frame.copy()
        pb._dU = base + delta
        pb.set_X(delta)
        assembly.update(pb, compute="vector")
        return -assembly.current.global_vector.copy()

    h = 1e-7
    numerical = np.empty_like(tangent)
    for col in range(len(base)):
        delta = np.zeros_like(base)
        delta[col] = h
        numerical[:, col] = (residual(delta) - residual(-delta)) / (2 * h)
    scale = np.abs(tangent).max()
    np.testing.assert_allclose(tangent, numerical, rtol=1e-5, atol=1e-6 * scale)
