"""Regression tests for how the Simcoon law obtains its temperature field.

See ``Simcoon.get_temp_gp``. The key guarantee is that an explicitly set
``assembly.sv['Temp']`` is honoured even when ``'Temp'`` is *also* a registered
ModelingSpace variable whose dof is not driven. Before the fix, the undriven dof
(read as 0) silently shadowed ``sv['Temp']`` and the constitutive law received
T = 0 instead of the imposed value (e.g. freezing a SMA simulation).
"""

import numpy as np

import fedoo as fd

ELISO = np.array([200e3, 0.3, 1e-5])  # E, nu, alpha


def _build(register_temp, tag):
    space = fd.ModelingSpace("3D")
    if register_temp:
        space.new_variable("Temp")
    mesh = fd.mesh.box_mesh(nx=2, ny=2, nz=2, name="m_temp_" + tag)
    law = fd.constitutivelaw.Simcoon("ELISO", ELISO, name="law_temp_" + tag)
    wf = fd.weakform.StressEquilibrium(law, space=space, name="wf_temp_" + tag)
    assembly = fd.Assembly.create(wf, mesh, name="asm_temp_" + tag)
    pb = fd.problem.NonLinear(assembly, name="pb_temp_" + tag)
    return space, mesh, law, assembly, pb


def test_sv_temp_takes_precedence_over_undriven_dof():
    """sv['Temp'] must win when 'Temp' is also a registered (undriven) variable."""
    space, mesh, law, assembly, pb = _build(register_temp=True, tag="svprec")
    assert "Temp" in space.list_variables()
    assembly.sv["Temp"] = np.full(assembly.n_gauss_points, 400.0)
    # mid-solve-like state: a real zero dof vector, Temp dof not driven
    pb._U = np.zeros(space.nvar * mesh.n_nodes)
    temp = law.get_temp_gp(assembly, pb)
    assert temp is not None
    assert np.allclose(temp, 400.0)


def test_temp_read_from_dof_when_no_sv():
    """When sv['Temp'] is absent, the temperature is read from the dof field."""
    space, mesh, law, assembly, pb = _build(register_temp=True, tag="dof")
    assert "Temp" not in assembly.sv
    rank = space.variable_rank("Temp")
    u = np.zeros(space.nvar * mesh.n_nodes)
    u[rank * mesh.n_nodes : (rank + 1) * mesh.n_nodes] = 400.0
    pb._U = u
    temp = law.get_temp_gp(assembly, pb)
    assert np.allclose(temp, 400.0)


def test_temp_none_when_undefined():
    """No temperature defined anywhere -> None (isothermal)."""
    space, mesh, law, assembly, pb = _build(register_temp=False, tag="none")
    assert "Temp" not in assembly.sv
    assert law.get_temp_gp(assembly, pb) is None


def test_scalar_zero_temp_is_none():
    """A scalar sv['Temp'] == 0 is treated as 'no temperature' (isothermal)."""
    space, mesh, law, assembly, pb = _build(register_temp=False, tag="zero")
    assembly.sv["Temp"] = 0
    assert law.get_temp_gp(assembly, pb) is None
