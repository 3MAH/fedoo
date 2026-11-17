"""Finite-strain thermoelasticity, with a reference thermal balance for TL/UL."""

import numpy as np
import pytest

import fedoo as fd


def _model(
    formulation="TL",
    kind="isotropic",
    heat_first=False,
    dynamic=False,
    dimension="3D",
    coupling=True,
):
    fd.ModelingSpace(dimension)
    if dimension == "3D":
        mesh = fd.mesh.box_mesh(nx=2, ny=2, nz=2, elm_type="hex8")
    else:
        mesh = fd.mesh.rectangle_mesh(nx=3, ny=2, elm_type="quad4")
    thermal = fd.constitutivelaw.ThermalProperties(
        [[1.2, 0.15, 0], [0.15, 0.8, 0], [0, 0, 0.6]],
        2,
        1,
        alpha=[0.001, 0.002, 0.0015, 0.0003, 0, 0],
        reference_temperature=300,
        coupling=coupling,
    )
    if kind == "isotropic":
        material = fd.constitutivelaw.ElasticIsotrop(100, 0.2, thermal=thermal)
    elif kind == "orthotropic":
        material = fd.constitutivelaw.ElasticOrthotropic(
            150, 100, 90, 30, 35, 40, 0.2, 0.15, 0.25, thermal=thermal
        )
    else:
        rng = np.random.default_rng(27)
        A = rng.normal(size=(6, 6))
        material = fd.constitutivelaw.ElasticAnisotropic(
            20 * (A @ A.T + np.eye(6)), thermal=thermal
        )
    mechanical = fd.weakform.StressEquilibrium(material, nlgeom=formulation)
    heat = fd.weakform.HeatEquation(material, nlgeom="TL")
    wf = heat + mechanical if heat_first else mechanical + heat
    assembly = fd.Assembly.create(wf, mesh)
    pb = fd.problem.NonLinear(assembly)
    pb.set_solver("direct_scipy")
    pb.set_dof("Temp", 300)
    pb.set_time_integrator(fd.time.FIRST_ORDER, fd.time.BackwardEuler())
    if dynamic:
        pb.set_time_integrator(fd.time.SECOND_ORDER, fd.time.Newmark())
    return pb, assembly, material


def _trial(pb):
    rng = np.random.default_rng(17)
    trial = np.zeros(pb.n_dof)
    F = np.array([[1.25, 0.22, 0.04], [0.06, 0.92, 0.03], [0.02, 0.07, 1.1]])
    for i, axis in enumerate("XYZ"[: pb.mesh.ndim]):
        values = (
            (F[: pb.mesh.ndim, : pb.mesh.ndim] - np.eye(pb.mesh.ndim)) @ pb.mesh.nodes.T
        )[i]
        values += rng.normal(size=pb.mesh.n_nodes) * 0.005
        rank = pb.space.variable_rank("Disp" + axis)
        trial[rank * pb.mesh.n_nodes : (rank + 1) * pb.mesh.n_nodes] = values
    rank = pb.space.variable_rank("Temp")
    trial[rank * pb.mesh.n_nodes : (rank + 1) * pb.mesh.n_nodes] = (
        4 + 3 * pb.mesh.nodes[:, 0]
    )
    return trial


@pytest.mark.parametrize("formulation", ["TL", "UL"])
@pytest.mark.parametrize("kind", ["isotropic", "orthotropic", "anisotropic"])
@pytest.mark.parametrize("heat_first", [False, True])
@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("dimension", ["3D", "2Dplane"])
def test_four_finite_strain_blocks_match_residual_derivatives(
    formulation, kind, heat_first, dynamic, dimension
):
    pb, assembly, _ = _model(formulation, kind, heat_first, dynamic, dimension)
    pb.initialize()
    pb.dtime = 0.2
    pb.set_start()
    trial = _trial(pb)
    pb._dU = trial.copy()
    pb.update()
    tangent = assembly.current.get_global_matrix().copy()
    rank = assembly.space.variable_rank("Temp")
    thermal = np.arange(rank * pb.mesh.n_nodes, (rank + 1) * pb.mesh.n_nodes)
    mechanical = np.setdiff1d(np.arange(pb.n_dof), thermal)
    rng = np.random.default_rng(13)
    for indices in (mechanical, thermal):
        direction = np.zeros(pb.n_dof)
        direction[indices] = rng.normal(size=len(indices))
        h = 1e-6
        pb._dU = trial + h * direction
        pb.update(compute="vector")
        plus = -pb.get_D().copy()
        pb._dU = trial - h * direction
        pb.update(compute="vector")
        minus = -pb.get_D().copy()
        actual, expected = (plus - minus) / (2 * h), tangent @ direction
        for block in (mechanical, thermal):
            np.testing.assert_allclose(
                actual[block], expected[block], rtol=2e-6, atol=2e-6
            )


def _criterion(pb):
    fields = {"Disp": {"atol": 1e-9, "rtol": 1e-9}}
    if "Temp" in pb.space.list_variables():
        fields["Temp"] = {"atol": 1e-9, "rtol": 1e-9}
    pb.set_nr_criterion("Fields", fields=fields)


@pytest.mark.parametrize("formulation", ["TL", "UL"])
@pytest.mark.parametrize("kind", ["isotropic", "orthotropic", "anisotropic"])
def test_large_stress_free_thermal_expansion(formulation, kind):
    pb, assembly, material = _model(formulation, kind)
    material.thermal.alpha = np.array([0.008, 0.010, 0.012])
    material.thermal.coupling = False
    for axis in "XYZ":
        pb.bc.add("Dirichlet", pb.mesh.find_nodes(axis, 0), "Disp" + axis, 0)
    pb.bc.add("Dirichlet", np.arange(pb.mesh.n_nodes), "Temp", 320, start_value=300)
    _criterion(pb)
    pb.nlsolve(dt=0.1, tmax=0.2, update_dt=False, print_info=0)
    expected = np.expm1(material.thermal.alpha * 20)[:, None] * pb.mesh.nodes.T
    np.testing.assert_allclose(pb.get_disp(), expected, atol=1e-9)
    np.testing.assert_allclose(assembly.sv["Stress"].asarray(), 0, atol=1e-8)


@pytest.mark.parametrize("formulation", ["TL", "UL"])
@pytest.mark.parametrize("dimension", ["3D", "2Dplane"])
def test_adiabatic_finite_extension_and_retry(formulation, dimension):
    pb, assembly, material = _model(formulation, dimension=dimension)
    material.thermal.alpha = 0.01
    diagonal = np.array([1.2, 0.9, 1.05])[: pb.mesh.ndim]
    for i, axis in enumerate("XYZ"[: pb.mesh.ndim]):
        pb.bc.add(
            "Dirichlet",
            np.arange(pb.mesh.n_nodes),
            "Disp" + axis,
            (diagonal[i] - 1) * pb.mesh.nodes[:, i],
        )
    _criterion(pb)
    pb.nlsolve(dt=0.2, tmax=0.2, update_dt=False, print_info=0)
    beta = material.E * 0.01 / (1 - 2 * material.nu)
    expected = 300 / (
        1
        + beta * np.log(np.prod(diagonal)) / (material.density * material.specific_heat)
    )
    np.testing.assert_allclose(pb.get_temp(), expected, atol=1e-9)
    saved_F = assembly.sv["F"].copy()
    saved_stress = assembly.sv["Stress"].asarray().copy()
    pb._dU = np.full(pb.n_dof, 0.02)
    pb.update()
    pb.dtime = 0.05
    pb.to_start()
    np.testing.assert_allclose(assembly.sv["F"], saved_F, atol=1e-12)
    np.testing.assert_allclose(
        assembly.sv["Stress"].asarray(), saved_stress, atol=1e-10
    )
    np.testing.assert_allclose(assembly.sv["HeatSource"], 0, atol=1e-9)


@pytest.mark.parametrize("kind", ["isotropic", "orthotropic", "anisotropic"])
@pytest.mark.parametrize("dynamic", [False, True])
def test_tl_and_ul_produce_the_same_coupled_system(kind, dynamic):
    systems = []
    for formulation in ("TL", "UL"):
        pb, assembly, _ = _model(formulation, kind, dynamic=dynamic)
        pb.initialize()
        pb.dtime = 0.2
        pb.set_start()
        pb._dU = _trial(pb)
        pb.update()
        systems.append(
            (assembly.current.get_global_matrix().toarray(), pb.get_D().copy())
        )
    for tl, ul in zip(*systems):
        np.testing.assert_allclose(tl, ul, atol=1e-9, rtol=1e-9)


@pytest.mark.parametrize("formulation", ["TL", "UL"])
def test_reference_mass_is_preserved_at_finite_volume(formulation):
    systems = []
    for dynamic in (False, True):
        pb, assembly, _ = _model(formulation, dynamic=dynamic, coupling=False)
        pb.initialize()
        pb.dtime = 0.2
        pb.set_start()
        pb._dU = _trial(pb)
        pb.update()
        systems.append(assembly.current.get_global_matrix().toarray())
    a0 = 1 / (0.25 * 0.2**2)
    n = pb.mesh.n_nodes
    mass_x = (systems[1][:n, :n] - systems[0][:n, :n]) / a0
    np.testing.assert_allclose(mass_x.sum(), 1, atol=1e-10)


@pytest.mark.parametrize("formulation", ["TL", "UL"])
def test_rigid_rotation_preserves_thermal_history(formulation):
    pb, assembly, _ = _model(formulation, "anisotropic")
    F0 = np.array([[1.2, 0.25, 0], [0.05, 0.9, 0], [0, 0, 1.1]])
    for i, axis in enumerate("XYZ"):
        values = ((F0 - np.eye(3)) @ pb.mesh.nodes.T)[i]
        pb.bc.add("Dirichlet", np.arange(pb.mesh.n_nodes), "Disp" + axis, values)
    pb.bc.add("Dirichlet", np.arange(pb.mesh.n_nodes), "Temp", 300)
    _criterion(pb)
    pb.nlsolve(dt=0.2, tmax=0.2, update_dt=False, print_info=0)
    stress = assembly.sv["Stress"].to_tensor().copy()
    angle = 1.1
    R = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    increment = (R @ F0 - F0) @ pb.mesh.nodes.T
    pb._dU = np.zeros(pb.n_dof)
    for i, axis in enumerate("XYZ"):
        rank = pb.space.variable_rank("Disp" + axis)
        pb._dU[rank * pb.mesh.n_nodes : (rank + 1) * pb.mesh.n_nodes] = increment[i]
    pb.update()
    expected = np.einsum("ij,jkn,lk->iln", R, stress, R)
    np.testing.assert_allclose(assembly.sv["Stress"].to_tensor(), expected, atol=1e-9)
    np.testing.assert_allclose(assembly.sv["HeatSource"], 0, atol=1e-9)
    material = assembly.weakform.constitutivelaw[0]
    np.testing.assert_allclose(
        material.get_stress_from_strain(assembly, assembly.sv["Strain"]).to_tensor(),
        expected,
        atol=1e-9,
    )


def test_heat_only_problem_uses_prescribed_deformation_and_reference_capacity():
    F = np.array([[1.4, 0.2, 0], [0, 0.8, 0], [0, 0, 1.1]])
    k = np.diag([1.2, 0.8, 0.6])
    inverse = np.linalg.inv(F)
    K0 = np.linalg.det(F) * inverse @ k @ inverse.T
    systems = []
    for conductivity, finite in ((k, True), (K0, False)):
        fd.ModelingSpace("3D")
        mesh = fd.mesh.box_mesh(nx=2, ny=2, nz=2, elm_type="hex8")
        material = fd.constitutivelaw.ThermalProperties(conductivity, 2, 1)
        heat = fd.weakform.HeatEquation(material, nlgeom="TL" if finite else False)
        heat.storage.assembly_options["mat_lumping"] = False
        assembly = fd.Assembly.create(heat, mesh)
        if finite:
            assembly.sv["F"] = np.repeat(F[:, :, None], assembly.n_gauss_points, axis=2)
        pb = fd.problem.NonLinear(assembly)
        pb.set_solver("direct_scipy")
        pb.set_dof("Temp", 300)
        pb.set_time_integrator(fd.time.FIRST_ORDER, fd.time.BackwardEuler())
        pb.initialize()
        pb.dtime = 0.2
        pb.set_start()
        pb._dU = 4 + 3 * mesh.nodes[:, 0]
        pb.update()
        systems.append(
            (assembly.current.get_global_matrix().toarray(), pb.get_D().copy())
        )
    for transformed, reference in zip(*systems):
        np.testing.assert_allclose(transformed, reference, atol=1e-9, rtol=1e-9)


def test_simcoon_umat_t_still_rejects_finite_strain():
    fd.ModelingSpace("3D")
    mesh = fd.mesh.box_mesh(nx=2, ny=2, nz=2, elm_type="hex8")
    material = fd.constitutivelaw.SimcoonThermoMechanical(
        "ELISO", [1, 2, 100, 0.2, 0.001], 1
    )
    assembly = fd.Assembly.create(
        fd.weakform.StressEquilibrium(material, nlgeom="TL")
        + fd.weakform.HeatEquation(material, nlgeom="TL"),
        mesh,
    )
    pb = fd.problem.NonLinear(assembly)
    pb.set_dof("Temp", 300)
    pb.set_time_integrator(fd.time.FIRST_ORDER, fd.time.BackwardEuler())
    with pytest.raises(NotImplementedError, match="Simcoon.*small strain"):
        pb.initialize()
