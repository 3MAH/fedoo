"""Taylor-Hood pairs built with CombinedElement (linear pressure, quadratic u).

The pressure sub-element uses the vertex nodes only. Its operators must be
exactly zero on the other nodes, and the pressure DOFs of those nodes, which
no element interpolates, must be eliminated automatically.
"""

import numpy as np
import pytest

import fedoo as fd

PAIRS = [("quad8", "quad4", 4), ("tri6", "tri3", 3), ("hex20", "hex8", 8)]


def _poro(quad, lin, name, n=3, biot_modulus=1e8, permeability=1e-12):
    fd.Assembly.delete_memory()
    if quad in ("hex20",):
        fd.ModelingSpace("3D")
        mesh = fd.mesh.box_mesh(n, n, n, 0, 1, 0, 1, 0, 1, quad)
    else:
        fd.ModelingSpace("2D")
        mesh = fd.mesh.rectangle_mesh(n, n, 0, 1, 0, 1, quad)
    fluid = fd.constitutivelaw.PoroFluidProperties(
        permeability=permeability, fluid_viscosity=1.0, biot_modulus=biot_modulus
    )
    wf = fd.weakform.PoroMechanicsSimple(
        fd.constitutivelaw.ElasticIsotrop(1e6, 0.3), fluid, nlgeom=False
    )
    elm = fd.lib_elements.CombinedElement(name, quad)
    elm.set_variable_interpolation("PorePressure", lin)
    return mesh, fd.Assembly.create(wf, mesh, elm_type=name)


def _mid_nodes(mesh, n_vertex):
    vertices = np.unique(mesh.elements[:, :n_vertex])
    return np.setdiff1d(np.arange(mesh.n_nodes), vertices)


@pytest.mark.parametrize("quad, lin, n_vertex", PAIRS)
def test_pressure_operators_vanish_on_unused_nodes(quad, lin, n_vertex, monkeypatch):
    # Uninitialised buffers used to leak into the unused columns: poison them.
    empty = np.empty

    def poisoned(*args, **kwargs):
        array = empty(*args, **kwargs)
        if np.issubdtype(array.dtype, np.floating):
            array.fill(np.nan)
        return array

    monkeypatch.setattr(np, "empty", poisoned)
    mesh, asm = _poro(quad, lin, f"op_{quad}")
    mids = _mid_nodes(mesh, n_vertex)
    space = asm.space
    ops = [space.variable("PorePressure")] + [
        space.derivative("PorePressure", x) for x in space.list_coordinates()
    ]
    for op in ops:
        matrix = asm._get_elementary_operator(op.op[0])[0].tocsc()
        assert np.all(np.isfinite(matrix.data))
        assert np.count_nonzero(matrix[:, mids].toarray()) == 0


@pytest.mark.parametrize("quad, lin, n_vertex", PAIRS)
def test_unused_pressure_dofs_are_eliminated(quad, lin, n_vertex):
    mesh, asm = _poro(quad, lin, f"elim_{quad}")
    pb = fd.problem.NonLinear(asm)
    pb.initialize()
    pb.dtime = 1e-3
    pb.update()
    pb.apply_boundary_conditions()
    n = mesh.n_nodes
    rank = asm.space.variable_rank("PorePressure")
    mids = rank * n + _mid_nodes(mesh, n_vertex)
    matrix = asm.current.get_global_matrix().tocsr()
    assert abs(matrix[mids]).max() == 0
    assert abs(matrix.tocsc()[:, mids]).max() == 0
    assert not np.isin(mids, pb._dof_free).any()
    vertices = rank * n + np.unique(mesh.elements[:, :n_vertex])
    assert np.isin(vertices, pb._dof_free).all()


@pytest.mark.parametrize("quad, lin, n_vertex", PAIRS)
def test_sealed_undrained_oedometer_is_exact(quad, lin, n_vertex):
    # Undrained (tiny k, small dt): p = -alpha M tr(eps), whatever the pair.
    mesh, asm = _poro(quad, lin, f"oedo_{quad}")
    pb = fd.problem.NonLinear(asm)
    dim = asm.space.ndim
    top = "Z" if dim == 3 else "Y"
    for axis in ("X", "Y", "Z")[:dim]:
        if axis != top:
            for value in (0, 1):
                pb.bc.add("Dirichlet", mesh.find_nodes(axis, value), "Disp" + axis, 0)
    pb.bc.add("Dirichlet", mesh.find_nodes(top, 0), "Disp" + top, 0)
    pb.bc.add("Dirichlet", mesh.find_nodes(top, 1), "Disp" + top, -1e-4)
    pb.set_nr_criterion("Displacement", tol=1e-8, max_subiter=10)
    pb.nlsolve(dt=1e-3, tmax=1e-3, update_dt=False, print_info=0)
    pressure = asm.get_gp_results(
        pb.space.variable("PorePressure"), pb.get_dof_solution()
    )
    np.testing.assert_allclose(pressure, 1e8 * 1e-4, rtol=1e-8)


@pytest.mark.parametrize("other_assembly", ["boundary", "volume"])
def test_orphan_dofs_in_nested_assembly_sum(other_assembly):
    mesh, assembly = _poro("quad8", "quad4", f"nested_{other_assembly}")
    if other_assembly == "boundary":
        other = fd.constraint.Pressure.from_nodes(mesh, mesh.find_nodes("Y", 1), 1)
    else:
        other = fd.Assembly.create(assembly.weakform, mesh, elm_type="quad8")
    inner = fd.AssemblySum([assembly, other])
    outer = fd.AssemblySum([inner, assembly])
    problem = fd.problem.NonLinear(outer)
    problem.apply_boundary_conditions()

    rank = assembly.space.variable_rank("PorePressure")
    mids = rank * mesh.n_nodes + _mid_nodes(mesh, 4)
    # A boundary load cannot activate unused pressure DOFs, whereas another
    # volume assembly with quadratic pressure interpolation does activate them.
    assert np.all(np.isin(mids, problem._dof_free) == (other_assembly == "volume"))
    assert outer.list_assembly[0] is inner


def _strip_footing(taylor_hood, n=32, half_width=1.0, size=8.0, q=1e3):
    """Undrained strip footing, incompressible constituents (B = 1, nu_u = 1/2).

    Elastic half-space solution on the axis: p = (2 q / pi) atan(b / z).
    """
    fd.Assembly.delete_memory()
    fd.ModelingSpace("2D")
    if taylor_hood:
        mesh = fd.mesh.rectangle_mesh(
            n // 2 + 1, n // 2 + 1, 0, size, -size, 0, "quad8"
        )
        elm = fd.lib_elements.CombinedElement("footing_th", "quad8")
        elm.set_variable_interpolation("PorePressure", "quad4")
        elm_type = "footing_th"
    else:
        mesh = fd.mesh.rectangle_mesh(n + 1, n + 1, 0, size, -size, 0, "quad4")
        elm_type = "quad4"
    fluid = fd.constitutivelaw.PoroFluidProperties(permeability=1e-12)
    wf = fd.weakform.PoroMechanicsSimple(
        fd.constitutivelaw.ElasticIsotrop(1e6, 0.3), fluid, nlgeom=False
    )
    asm = fd.Assembly.create(wf, mesh, elm_type=elm_type)
    top = mesh.find_nodes("Y", 0.0)
    loaded = top[mesh.nodes[top, 0] <= half_width + 1e-9]
    load = fd.constraint.Pressure.from_nodes(mesh, loaded, q, initial_pressure=q)
    pb = fd.problem.NonLinear(asm + load)
    pb.bc.add("Dirichlet", "left", "DispX", 0)
    pb.bc.add("Dirichlet", "right", "DispX", 0)
    pb.bc.add("Dirichlet", "bottom", "Disp", 0)
    pb.nlsolve(dt=1e-6, tmax=1e-6, update_dt=False, print_info=0)

    pressure = pb.get_dof_solution("PorePressure")
    vertices = np.unique(mesh.elements[:, :4])
    axis = vertices[np.isclose(mesh.nodes[vertices, 0], 0)]
    depth = -mesh.nodes[axis, 1]
    selected = (depth >= 0.5 * half_width) & (depth <= 2 * half_width)
    reference = 2 * q / np.pi * np.arctan(half_width / depth[selected])
    return np.abs(pressure[axis][selected] - reference).max() / q


def test_taylor_hood_strip_footing_matches_the_undrained_solution():
    assert _strip_footing(taylor_hood=True) < 0.03
    # Equal order on the same pressure resolution: LBB checkerboard.
    assert _strip_footing(taylor_hood=False) > 0.3
