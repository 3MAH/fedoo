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
from fedoo.lib_elements import get_default_n_gp, get_element, is_reduced_integration


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
        ("hex8", 8, False),
        ("hex8r", 1, True),
        ("quad4r", 4, True),
    ],
)
def test_is_reduced_integration(elm_type, n_elm_gp, expected):
    assert is_reduced_integration(elm_type, n_elm_gp) is expected


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


def _dynamic_assembly(elm_type, storage=None):
    fd.ModelingSpace("3D")
    mesh = _tet4_mesh()
    material = fd.constitutivelaw.ElasticIsotrop(1e5, 0.3)
    material.set_density(1000.0)
    weakform = fd.weakform.StressEquilibrium(material)
    if storage is not None:
        weakform.set_inertia(storage)
    return fd.Assembly.create(weakform, mesh, elm_type)


def _compile(assembly):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        compiled = fd.time.Newmark().compile_assembly(assembly)
    n_warn = sum("reduced integration" in str(w.message) for w in rec)
    return compiled, n_warn


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


def test_cached_storage_assembly_uses_the_mesh_element():
    # path of the explicit, linear dynamic and modal problems
    from fedoo.time.common import build_storage_assembly

    assembly = _dynamic_assembly("tet4r")
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        data = build_storage_assembly(assembly, fd.time.SECOND_ORDER)
    assert (data.assembly.elm_type, data.assembly.n_elm_gp) == ("tet4", 4)
    assert not any("reduced integration" in str(w.message) for w in rec)


def _tip_history(elm_type, integrator, damping):
    fd.ModelingSpace("3D")
    mesh = _tet4_mesh()
    material = fd.constitutivelaw.ElasticIsotrop(1e5, 0.3)
    material.set_density(1000.0)
    weakform = fd.weakform.StressEquilibrium(material)
    if damping:
        weakform.set_damping(alpha=0.5, beta=1e-4)
    assembly = fd.Assembly.create(weakform, mesh, elm_type)
    problem = fd.problem.NonLinear(assembly)
    problem.set_time_integrator(fd.time.SECOND_ORDER, integrator)
    problem.bc.add("Dirichlet", [0, 1, 2], "Disp", 0)
    problem.bc.add("Neumann", [4], "DispZ", 50.0)
    problem.nlsolve(dt=2e-3, tmax=2e-2, update_dt=False, print_info=0)
    return problem.get_disp().copy(), assembly.sv["Velocity"].copy()


@pytest.mark.parametrize(
    "integrator, damping",
    [
        (lambda: fd.time.Newmark(), False),
        (lambda: fd.time.Newmark(), True),
        (lambda: fd.time.GeneralizedAlpha(alpha_m=0.2, alpha_f=0.3), True),
    ],
)
def test_split_mass_gives_the_full_element_response(integrator, damping):
    # tet4r stiffness == tet4 stiffness and both masses use tet4: the
    # transient response must be the same, with the state on the user assembly
    disp_ref, vel_ref = _tip_history("tet4", integrator(), damping)
    disp, vel = _tip_history("tet4r", integrator(), damping)
    assert np.abs(disp_ref).max() > 0
    assert np.allclose(disp, disp_ref, rtol=1e-8, atol=1e-14)
    assert np.allclose(vel, vel_ref, rtol=1e-8, atol=1e-12)
