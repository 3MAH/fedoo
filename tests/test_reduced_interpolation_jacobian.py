import numpy as np
import pytest

import fedoo as fd
from fedoo.lib_elements.element_list import get_element


DISTORTED_ELEMENTS = [
    (
        "2Dplane",
        "quad4",
        "quad4r",
        "quad4sri",
        np.array(
            [
                [0.0, 0.0],
                [2.0, 0.0],
                [2.6, 1.2],
                [-0.2, 1.0],
            ]
        ),
    ),
    (
        "3D",
        "hex8",
        "hex8r",
        "hex8sri",
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


@pytest.mark.parametrize(
    "dimension,mesh_elm,reduced_elm,assembly_elm,nodes", DISTORTED_ELEMENTS
)
def test_reduced_solid_operator_uses_center_jacobian(
    dimension, mesh_elm, reduced_elm, assembly_elm, nodes
):
    fd.Assembly.delete_memory()
    space = fd.ModelingSpace(dimension)
    mesh = fd.Mesh(
        nodes,
        np.array([np.arange(len(nodes))]),
        mesh_elm,
        register_name=False,
    )
    material = fd.constitutivelaw.ElasticIsotrop(1000.0, 0.3)
    weakform = fd.weakform.StressEquilibriumBbar(material, space=space)
    assembly = fd.Assembly.create(weakform, mesh)
    fd.problem.Linear(assembly)
    assembly.compute_elementary_operators()

    assert assembly.elm_type == assembly_elm
    mesh._compute_gaussian_quadrature_mat(1)
    center_inverse_jacobian = mesh._elm_interpolation[1].inv_jacobian_matrix[0, 0]
    center_natural_derivative = get_element(reduced_elm)(
        assembly.n_elm_gp
    ).shape_function_derivative_gp[0]
    expected_derivative = center_inverse_jacobian @ center_natural_derivative

    coordinates = ("X", "Y", "Z")[: space.ndim]
    reduced_variable = "_DispX"
    full_variable = "DispX"
    for direction, coordinate in enumerate(coordinates):
        reduced_derivative = space.derivative(reduced_variable, coordinate).op[0]
        reduced_operator = assembly._get_elementary_operator(
            reduced_derivative
        )[0].toarray()
        expected_operator = np.tile(
            expected_derivative[direction], (assembly.n_elm_gp, 1)
        )
        assert np.allclose(reduced_operator, expected_operator)

        full_derivative = space.derivative(full_variable, coordinate).op[0]
        full_operator = assembly._get_elementary_operator(full_derivative)[0].toarray()
        assert not np.allclose(full_operator, expected_operator)


def test_reduced_shell_operator_uses_center_jacobian():
    fd.Assembly.delete_memory()
    space = fd.ModelingSpace("3D")
    nodes = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.6, 1.2, 0.0],
            [-0.2, 1.0, 0.0],
        ]
    )
    mesh = fd.Mesh(
        nodes,
        np.array([[0, 1, 2, 3]]),
        "quad4",
        ndim=3,
        register_name=False,
    )
    material = fd.constitutivelaw.ElasticIsotrop(1000.0, 0.3)
    shell = fd.constitutivelaw.ShellHomogeneous(material, 0.1)
    weakform = fd.weakform.PlateEquilibrium(shell, space=space)
    assembly = fd.Assembly.create(weakform, mesh, elm_type="pquad4sri")
    assembly.compute_elementary_operators()

    mesh._compute_gaussian_quadrature_mat(1, assembly._element_local_frame)
    center_inverse_jacobian = mesh._elm_interpolation[1].inv_jacobian_matrix[0, 0]
    center_natural_derivative = get_element("quad4r")(
        assembly.n_elm_gp
    ).shape_function_derivative_gp[0]
    expected_x_derivative = (
        center_inverse_jacobian @ center_natural_derivative
    )[0]

    reduced_derivative = space.derivative("_DispZ", "X").op[0]
    reduced_operator = assembly._get_elementary_operator(
        reduced_derivative
    )[0].toarray()
    expected_operator = np.tile(expected_x_derivative, (assembly.n_elm_gp, 1))
    assert np.allclose(reduced_operator, expected_operator)

    full_derivative = space.derivative("DispZ", "X").op[0]
    full_operator = assembly._get_elementary_operator(full_derivative)[0].toarray()
    assert not np.allclose(full_operator, expected_operator)
