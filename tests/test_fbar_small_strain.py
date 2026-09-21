import numpy as np
import pytest

import fedoo as fd
from fedoo.weakform.stress_equilibrium import _comp_grad_disp_fbar

DISTORTED_ELEMENTS = [
    (
        "2Dplane",
        "quad4",
        np.array([[0.0, 0.0], [2.0, 0.0], [2.6, 1.2], [-0.2, 1.0]]),
    ),
    (
        "3D",
        "hex8",
        np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [2.2, 1.1, 0.0],
                [-0.1, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [2.1, -0.1, 1.1],
                [2.5, 1.2, 1.4],
                [-0.2, 1.1, 0.9],
            ]
        ),
    ),
]


@pytest.mark.parametrize("dimension,mesh_elm,nodes", DISTORTED_ELEMENTS)
def test_small_strain_fbar_uses_element_centroid(dimension, mesh_elm, nodes):
    fd.Assembly.delete_memory()
    space = fd.ModelingSpace(dimension)
    mesh = fd.Mesh(
        nodes,
        np.array([np.arange(len(nodes))]),
        mesh_elm,
        register_name=False,
    )
    material = fd.constitutivelaw.ElasticIsotrop(1000.0, 0.3)
    weakform = fd.weakform.StressEquilibrium(
        material, incompressibility="fbar", space=space
    )
    assembly = fd.Assembly.create(weakform, mesh)

    # displacement field with a non uniform divergence
    displacement = (0.01 * nodes**2).T.ravel()

    grad_center = np.array(
        [
            [
                assembly.get_gp_results(op, displacement, n_elm_gp=1)
                if op != 0
                else np.zeros(mesh.n_elements)
                for op in line_op
            ]
            for line_op in space.op_grad_u()
        ]
    )
    expected = np.trace(grad_center)

    div_u_bar = np.trace(_comp_grad_disp_fbar(assembly, displacement))

    assert np.allclose(div_u_bar, expected)


@pytest.mark.parametrize("dimension,mesh_elm,nodes", DISTORTED_ELEMENTS)
def test_small_strain_fbar_tangent_matches_internal_force(dimension, mesh_elm, nodes):
    fd.Assembly.delete_memory()
    fd.ModelingSpace(dimension)
    mesh = fd.Mesh(
        nodes,
        np.array([np.arange(len(nodes))]),
        mesh_elm,
        register_name=False,
    )
    material = fd.constitutivelaw.ElasticIsotrop(1000.0, 0.3)
    weakform = fd.weakform.StressEquilibrium(material, incompressibility="fbar")
    assembly = fd.Assembly.create(weakform, mesh)
    pb = fd.problem.NonLinear(assembly)
    pb.print_info = 0
    pb.initialize()

    rng = np.random.default_rng(2)
    displacement = rng.normal(scale=0.01, size=pb.n_dof)
    pb._U = displacement
    assembly.update(pb, "all")

    matrix = assembly.current.get_global_matrix()
    internal_force = np.asarray(assembly.current.get_global_vector()).ravel()
    matrix_force = np.asarray(matrix @ displacement).ravel()
    assert np.allclose(internal_force, -matrix_force)


@pytest.mark.parametrize("print_info", [0, 1])
def test_unsupported_fbar_warning_is_independent_of_verbosity(print_info):
    def make_problem(print_info):
        fd.Assembly.delete_memory()
        fd.ModelingSpace("2Dplane")
        mesh = fd.mesh.rectangle_mesh(3, 3, elm_type="quad8")
        material = fd.constitutivelaw.ElasticIsotrop(1000.0, 0.3)
        weakform = fd.weakform.StressEquilibrium(material, incompressibility="fbar")
        assembly = fd.Assembly.create(weakform, mesh)
        pb = fd.problem.NonLinear(assembly)
        pb.print_info = print_info
        return pb

    with pytest.warns(UserWarning, match="consistent F-bar tangent"):
        make_problem(print_info).initialize()
