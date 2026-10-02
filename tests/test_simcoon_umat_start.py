"""The Simcoon law initialises its points once, at MechanicalUMAT.initialize.

simcoon re-initialises a point (reference temperature statev[0], stress, internal
variables) when sim.umat is called with start=True, and by default infers it from
time 0. The Newton corrections of the first increment are also at time 0 and, in a
thermomechanical problem, see a temperature that changes between corrections: the
reference temperature must stay the one of the initialization.
"""

import numpy as np
import fedoo as fd

EPICP = np.array([200e3, 0.3, 1e-5, 300.0, 1000.0, 0.5])


def _setup(tag, temp=300.0, initial_T=None):
    space = fd.ModelingSpace("3D")
    mesh = fd.mesh.box_mesh(nx=2, ny=2, nz=2, name="m_start_" + tag)
    law = fd.constitutivelaw.Simcoon("EPICP", EPICP, name="law_start_" + tag)
    wf = fd.weakform.StressEquilibrium(law, space=space, name="wf_start_" + tag)
    assembly = fd.Assembly.create(wf, mesh, name="asm_start_" + tag)
    pb = fd.problem.NonLinear(assembly, name="pb_start_" + tag)
    assembly.sv["Temp"] = np.full(assembly.n_gauss_points, temp)
    if initial_T is not None:
        law.set_initial_statev(assembly, "T", initial_T)
    assembly.initialize(pb)
    return law, assembly, pb


def test_reference_temperature_is_set_at_initialization():
    _, assembly, _ = _setup("init")
    np.testing.assert_array_equal(assembly.sv["Statev"][0], 300.0)


def test_time_zero_corrections_keep_the_reference_temperature():
    law, assembly, pb = _setup("corr")
    assembly.set_start(pb)
    assert pb.time == 0
    for temp in (350.0, 420.0):  # two corrections of the first increment
        assembly.sv["Temp"] = np.full(assembly.n_gauss_points, temp)
        law.update(assembly, pb)
        np.testing.assert_array_equal(assembly.sv["Statev"][0], 300.0)


def test_set_initial_statev_wins_over_the_law_initialization():
    _, assembly, _ = _setup("user", initial_T=280.0)
    np.testing.assert_array_equal(assembly.sv["Statev"][0], 280.0)
