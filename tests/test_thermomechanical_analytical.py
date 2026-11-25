"""Thermo-mechanical coupling against closed-form solutions."""

import numpy as np
import pytest

import fedoo as fd

E, NU, RHO, CAPACITY = 100.0, 0.2, 1.0, 2.0


def _material(simcoon, alpha=0.0, conductivity=1.0, coupling=True):
    if simcoon:
        return fd.constitutivelaw.SimcoonThermoMechanical(
            "ELISO", [RHO, CAPACITY, E, NU, alpha], conductivity
        )
    return fd.constitutivelaw.ElasticIsotrop(
        E,
        NU,
        thermal=fd.constitutivelaw.ThermalProperties(
            conductivity,
            CAPACITY,
            RHO,
            alpha=alpha,
            reference_temperature=300,
            coupling=coupling,
        ),
    )


def _problem(material, mesh, nlgeom=False, transient=True):
    heat = fd.weakform.HeatEquation(material, nlgeom="TL" if nlgeom else None)
    wf = fd.weakform.StressEquilibrium(material, nlgeom=nlgeom) + heat
    assembly = fd.Assembly.create(wf, mesh)
    pb = fd.problem.NonLinear(assembly)
    pb.set_solver("direct_scipy")
    if transient:
        pb.set_time_integrator(fd.time.FIRST_ORDER, fd.time.BackwardEuler())
    return pb, assembly


def _tight(pb):
    pb.set_nr_criterion(
        "Fields",
        fields={
            "Disp": {"atol": 1e-12, "rtol": 1e-10},
            "Temp": {"atol": 1e-12, "rtol": 1e-10},
        },
    )


def _conduction_error(simcoon, dt, n_elements=80):
    """Decay of a sine temperature profile between two ends kept at 300 K."""
    fd.ModelingSpace("2Dplane")
    mesh = fd.mesh.rectangle_mesh(
        nx=n_elements + 1, ny=2, x_max=1.0, y_max=1.0 / n_elements, elm_type="quad4"
    )
    pb, _ = _problem(_material(simcoon), mesh)
    amplitude, t_end = 10.0, 0.05
    x = mesh.nodes[:, 0]
    pb.set_dof("Temp", 300 + amplitude * np.sin(np.pi * x))
    pb.bc.add("Dirichlet", np.arange(mesh.n_nodes), "Disp", 0)
    ends = np.r_[mesh.find_nodes("X", 0.0), mesh.find_nodes("X", 1.0)]
    pb.bc.add("Dirichlet", ends, "Temp", 300, start_value=300)
    _tight(pb)
    pb.nlsolve(dt=dt, tmax=t_end, update_dt=False, print_info=0)
    diffusivity = 1.0 / (RHO * CAPACITY)
    exact = 300 + amplitude * np.sin(np.pi * x) * np.exp(
        -diffusivity * np.pi**2 * t_end
    )
    return np.abs(pb.get_temp() - exact).max() / amplitude


@pytest.mark.parametrize("simcoon", [False, True])
def test_transient_conduction_converges_at_first_order(simcoon):
    coarse = _conduction_error(simcoon, 0.005)
    fine = _conduction_error(simcoon, 0.0025)
    assert fine < 2e-3
    assert 1.8 < coarse / fine < 2.2  # backward Euler


@pytest.mark.parametrize("simcoon", [False, True])
def test_adiabatic_cooling_under_uniaxial_stress(simcoon):
    """Kelvin's formula dT = -T alpha dsigma / (rho c_sigma), to first order."""
    alpha, stress, temperature = 1e-3, 0.01, 300.0
    fd.ModelingSpace("3D")
    mesh = fd.mesh.box_mesh(nx=2, ny=2, nz=2, elm_type="hex8")
    pb, _ = _problem(_material(simcoon, alpha), mesh)
    pb.set_dof("Temp", temperature)
    for axis in "XYZ":
        pb.bc.add("Dirichlet", mesh.find_nodes(axis, 0), "Disp" + axis, 0)
    face = mesh.find_nodes("X", 1.0)
    pb.bc.add("Neumann", face, "DispX", stress / len(face))
    _tight(pb)
    pb.nlsolve(dt=1.0, tmax=1.0, update_dt=False, print_info=0)
    if simcoon:  # props[1] is the capacity at constant stress
        capacity_stress = CAPACITY
    else:  # the native law takes the capacity at constant strain
        bulk = E / (3 * (1 - 2 * NU))
        capacity_stress = CAPACITY + 9 * bulk * alpha**2 * temperature / RHO
    expected = -temperature * alpha * stress / (RHO * capacity_stress)
    np.testing.assert_allclose(pb.get_temp() - temperature, expected, rtol=1e-3)


@pytest.mark.parametrize("nlgeom", ["TL", "UL"])
def test_steady_conduction_through_a_stretched_bar(nlgeom):
    """Isotropic conduction is linear in the current coordinate."""
    fd.ModelingSpace("2Dplane")
    mesh = fd.mesh.rectangle_mesh(nx=11, ny=2, x_max=1.0, y_max=0.1, elm_type="quad4")
    pb, assembly = _problem(_material(False, coupling=False), mesh, nlgeom)
    pb.set_dof("Temp", 300)
    x = mesh.nodes[:, 0]
    nodes = np.arange(mesh.n_nodes)
    pb.bc.add("Dirichlet", nodes, "DispX", 0.4 * x**2)  # stretch 1 + 0.8 x
    pb.bc.add("Dirichlet", nodes, "DispY", 0)
    pb.bc.add("Dirichlet", mesh.find_nodes("X", 0.0), "Temp", 300, start_value=300)
    pb.bc.add("Dirichlet", mesh.find_nodes("X", 1.0), "Temp", 340, start_value=300)
    _tight(pb)
    # one long step: the capacity term is negligible
    pb.nlsolve(dt=1e8, tmax=1e8, update_dt=False, print_info=0)
    current = x + 0.4 * x**2
    np.testing.assert_allclose(pb.get_temp(), 300 + 40 * current / 1.4, atol=1e-5)
    flux = np.asarray(assembly.sv["HeatFlux"])
    np.testing.assert_allclose(flux[0], -40 / 1.4, rtol=1e-6)
    np.testing.assert_allclose(flux[1], 0, atol=1e-6)
