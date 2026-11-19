"""Reduced tetrahedra and the reduced-mass warning.

``tet4`` and ``tet10`` keep their full default rules (4 and 15 Gauss points);
``tet4r`` (1 point, interpolation at the centroid, as ``quad4r``) and
``tet10r`` (4 points) are the reduced variants. The mass does not follow a
reduced stiffness: it is assembled on the element of the mesh with its default
rule, in a separate assembly. A reduced integration requested on the storage
weak form itself is kept, with a warning.
"""

import warnings

import numpy as np
import pytest

import fedoo as fd
from fedoo.lib_elements import (
    get_default_n_gp,
    get_element,
    get_full_integration,
    is_reduced_integration,
)


def test_default_gauss_points():
    assert get_default_n_gp("tet4") == 4
    assert get_default_n_gp("tet10") == 15
    assert get_default_n_gp("tet4r") == 1
    assert get_default_n_gp("tet10r") == 4


def test_tet4r_interpolation_is_evaluated_at_the_centroid():
    # same convention as quad4r: centroid values whatever the quadrature
    elm = get_element("tet4r")(4)
    assert np.allclose(elm.shape_function(elm.xi_pg), 0.25)
    ref = get_element("tet4")(4)
    for a, b in zip(
        elm.shape_function_derivative(elm.xi_pg),
        ref.shape_function_derivative(ref.xi_pg),
    ):
        assert np.array_equal(a, b)


@pytest.mark.parametrize(
    "elm_type, n_elm_gp, expected",
    [
        ("tet4", 4, False),
        ("tet4", 1, True),
        ("tet4r", 1, True),
        ("tet4r", 4, True),
        ("tet10", 15, False),
        ("tet10", 4, True),
        ("tet10r", 4, True),
        ("tet10r", 15, False),
        ("hex20r", 14, True),
        ("hex20r", 27, False),
        ("hex8", 8, False),
        ("hex8r", 1, True),
        ("quad4r", 4, True),
    ],
)
def test_is_reduced_integration(elm_type, n_elm_gp, expected):
    assert is_reduced_integration(elm_type, n_elm_gp) is expected


def test_full_integration_of_reduced_elements():
    fd.ModelingSpace("3D")
    mesh = _tet4_mesh()
    assert get_full_integration("tet4r", mesh) == ("tet4", 4)
    assert get_full_integration("tet4", mesh) == ("tet4", 4)
    assert get_full_integration("tet10r") == ("tet10r", 15)
    assert get_full_integration("hex20r") == ("hex20r", 27)
    mesh10 = fd.mesh.change_elm_type(mesh, "tet10")
    assert get_full_integration("tet10r", mesh10) == ("tet10", 15)


def _tet4_mesh():
    # two tetrahedra sharing a face
    nodes = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    elements = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
    return fd.Mesh(nodes, elements, "tet4")


def _stiffness(mesh, elm_type):
    wf = fd.weakform.StressEquilibrium(fd.constitutivelaw.ElasticIsotrop(1e5, 0.3))
    assembly = fd.Assembly.create(wf, mesh, elm_type)
    fd.problem.Linear(assembly)
    return assembly.get_global_matrix()


def test_tet4r_stiffness_equals_tet4():
    # constant strain: the 1-point rule is exact for the linear tetrahedron
    fd.ModelingSpace("3D")
    mesh = _tet4_mesh()
    full = _stiffness(mesh, "tet4")
    reduced = _stiffness(mesh, "tet4r")
    assert abs(full - reduced).max() < 1e-9 * abs(full).max()


def test_changing_the_rule_of_a_reduced_element_warns():
    fd.ModelingSpace("3D")
    mesh = fd.mesh.change_elm_type(_tet4_mesh(), "tet10")
    wf = fd.weakform.StressEquilibrium(fd.constitutivelaw.ElasticIsotrop(1e5, 0.3))
    with pytest.warns(UserWarning, match='is "tet10" with that rule'):
        fd.Assembly.create(wf, mesh, "tet10r", n_elm_gp=15)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fd.Assembly.create(wf, mesh, "tet10r")
        fd.Assembly.create(wf, mesh, "tet10", n_elm_gp=4)
    mesh = _tet4_mesh()
    with pytest.warns(UserWarning, match="still taken at the element centre"):
        fd.Assembly.create(wf, mesh, "tet4r", n_elm_gp=4)


@pytest.mark.parametrize("elm_type", ["tri3r", "quad4r", "hex8r", "tet4r"])
def test_centroid_variants_default_to_one_gauss_point(elm_type):
    assert get_default_n_gp(elm_type) == 1


def _dynamic_assembly(
    elm_type, storage=None, damping=False, n_elm_gp=None, new_space=True
):
    if new_space:
        fd.ModelingSpace("3D")
    mesh = _tet4_mesh()
    material = fd.constitutivelaw.ElasticIsotrop(1e5, 0.3)
    material.set_density(1000.0)
    weakform = fd.weakform.StressEquilibrium(material)
    if storage is not None:
        weakform.set_inertia(storage)
    if damping:
        weakform.set_damping(alpha=0.5, beta=1e-4)
    return fd.Assembly.create(weakform, mesh, elm_type, n_elm_gp=n_elm_gp)


def _count_reduced_warnings(function, *args):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        result = function(*args)
    return result, sum("reduced integration" in str(w.message) for w in rec)


def _compile(assembly):
    return _count_reduced_warnings(fd.time.Newmark().compile_assembly, assembly)


def test_mass_does_not_follow_a_reduced_stiffness():
    # stiffness on tet4r (1 point), mass on the mesh element (tet4, 4 points)
    assembly = _dynamic_assembly("tet4r")
    compiled, n_warn = _compile(assembly)
    mass, stiffness = compiled.list_assembly
    assert stiffness is assembly
    assert (stiffness.elm_type, stiffness.n_elm_gp) == ("tet4r", 1)
    assert (mass.elm_type, mass.n_elm_gp) == ("tet4", 4)
    assert n_warn == 0


def test_full_element_keeps_a_single_assembly():
    assembly = _dynamic_assembly("tet4")
    compiled, n_warn = _compile(assembly)
    assert compiled is assembly
    assert n_warn == 0


def test_reduced_mass_requested_on_the_storage_warns():
    fd.ModelingSpace("3D")
    inertia = fd.weakform.Inertia(1000.0)
    inertia.assembly_options["n_elm_gp"] = 1
    compiled, n_warn = _compile(_dynamic_assembly("tet4", inertia))
    mass = compiled.list_assembly[0]
    assert (mass.elm_type, mass.n_elm_gp) == ("tet4", 1)
    assert n_warn == 1


def test_gauss_points_requested_on_the_storage_use_the_mesh_element():
    # n_elm_gp alone must not leave the mass on the centroid interpolation
    fd.ModelingSpace("3D")
    inertia = fd.weakform.Inertia(1000.0)
    inertia.assembly_options["n_elm_gp"] = 4
    compiled, n_warn = _compile(_dynamic_assembly("tet4r", inertia))
    mass = compiled.list_assembly[0]
    assert (mass.elm_type, mass.n_elm_gp) == ("tet4", 4)
    assert n_warn == 0


def test_fewer_gauss_points_on_a_full_element_split_the_mass():
    assembly = _dynamic_assembly("tet4", n_elm_gp=1)
    compiled, n_warn = _compile(assembly)
    mass, stiffness = compiled.list_assembly
    assert (stiffness.elm_type, stiffness.n_elm_gp) == ("tet4", 1)
    assert (mass.elm_type, mass.n_elm_gp) == ("tet4", 4)
    assert n_warn == 0


def test_storage_is_inserted_flat_in_a_user_sum():
    # code walking ``list_assembly`` one level deep must still find the leaves
    assembly = _dynamic_assembly("tet4r")
    other = _dynamic_assembly("tet4", new_space=False)
    total = fd.Assembly.sum(assembly, other)
    compiled, _ = _compile(total)
    assert compiled is total
    mass, stiffness, unsplit = total.list_assembly
    assert stiffness is assembly and unsplit is other
    assert (mass.elm_type, mass.n_elm_gp) == ("tet4", 4)


def test_reduced_integration_weak_form_runs_with_an_implicit_integrator():
    # StressEquilibriumRI: 1-point stiffness + hourglass assembly, whose
    # lookup of the stiffness assembly walks the sum one level deep
    fd.ModelingSpace("3D")
    mesh = fd.mesh.box_mesh(nx=3, ny=3, nz=3, elm_type="hex8")
    material = fd.constitutivelaw.ElasticIsotrop(1e5, 0.3)
    material.set_density(1000.0)
    assembly = fd.Assembly.create(fd.weakform.StressEquilibriumRI(material), mesh)
    problem = fd.problem.NonLinear(assembly)
    problem.set_time_integrator(fd.time.SECOND_ORDER, fd.time.Newmark())
    problem.bc.add("Dirichlet", mesh.find_nodes("X", mesh.bounding_box.xmin), "Disp", 0)
    problem.bc.add("Neumann", mesh.find_nodes("X", mesh.bounding_box.xmax), "DispZ", 1)
    problem.nlsolve(dt=1e-3, tmax=2e-3, update_dt=False, print_info=0)
    masses = [
        a
        for a in problem.assembly.iter_leaf()
        if (a.elm_type, a.n_elm_gp) == ("hex8", 8)
    ]
    assert len(masses) == 1
    assert np.abs(problem.get_disp()).max() > 0


def test_second_compile_keeps_the_mass():
    # the assembly is reused by another problem: its storage must come along
    assembly = _dynamic_assembly("tet4r")
    first, _ = _compile(assembly)
    second, _ = _compile(assembly)
    assert second.list_assembly[1] is assembly
    assert second.list_assembly[0] is first.list_assembly[0]


def test_rayleigh_accessor_survives_the_split():
    assembly = _dynamic_assembly("tet4r", damping=True)
    compiled, _ = _compile(assembly)
    assert assembly.weakform.rayleigh_damping == [0.5, 1e-4]
    assembly.weakform.rayleigh_damping = [0.1, 2e-4]
    assert compiled.list_assembly[0].weakform.damping_coef == 0.1


def test_first_order_integrator_does_not_warn_about_mass():
    fd.ModelingSpace("3D")
    mesh = _tet4_mesh()
    material = fd.constitutivelaw.ThermalProperties(1.0, 1.0, 1.0)
    assembly = fd.Assembly.create(fd.weakform.HeatEquation(material), mesh, "tet4r")
    _, n_warn = _count_reduced_warnings(
        fd.time.BackwardEuler().compile_assembly, assembly
    )
    assert n_warn == 0


def test_cached_storage_assembly_uses_the_mesh_element():
    # path of the explicit, linear dynamic and modal problems
    from fedoo.time.common import build_storage_assembly

    data, n_warn = _count_reduced_warnings(
        build_storage_assembly, _dynamic_assembly("tet4r"), fd.time.SECOND_ORDER
    )
    assert (data.assembly.elm_type, data.assembly.n_elm_gp) == ("tet4", 4)
    assert n_warn == 0


def _tip_history(elm_type, integrator, damping):
    assembly = _dynamic_assembly(elm_type, damping=damping)
    problem = fd.problem.NonLinear(assembly)
    problem.set_time_integrator(fd.time.SECOND_ORDER, integrator)
    problem.bc.add("Dirichlet", [0, 1, 2], "Disp", 0)
    problem.bc.add("Neumann", [4], "DispZ", 50.0)
    problem.nlsolve(dt=2e-3, tmax=2e-2, update_dt=False, print_info=0)
    return problem.get_disp().copy(), assembly.sv["Velocity"].copy()


@pytest.mark.parametrize(
    "integrator, damping",
    [
        (fd.time.Newmark, False),
        (fd.time.Newmark, True),
        (lambda: fd.time.GeneralizedAlpha(alpha_m=0.2, alpha_f=0.3), True),
    ],
)
def test_split_mass_gives_the_full_element_response(integrator, damping):
    # small strain, linear elastic: tet4r stiffness == tet4 stiffness and both
    # masses use tet4, so the response is the same (state on the user assembly)
    disp_ref, vel_ref = _tip_history("tet4", integrator(), damping)
    disp, vel = _tip_history("tet4r", integrator(), damping)
    assert np.abs(disp_ref).max() > 0
    assert np.allclose(disp, disp_ref, rtol=1e-8, atol=1e-14)
    assert np.allclose(vel, vel_ref, rtol=1e-8, atol=1e-12)
