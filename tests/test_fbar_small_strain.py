import numpy as np
import pytest

import fedoo as fd
from fedoo.weakform.stress_equilibrium import (
    _comp_grad_disp,
    _comp_grad_disp_fbar,
)

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
def test_small_strain_fbar_uses_volume_weighted_mean(dimension, mesh_elm, nodes):
    fd.Assembly.delete_memory()
    space = fd.ModelingSpace(dimension)
    mesh = fd.Mesh(
        nodes,
        np.array([np.arange(len(nodes))]),
        mesh_elm,
        register_name=False,
    )
    material = fd.constitutivelaw.ElasticIsotrop(1000.0, 0.3)
    weakform = fd.weakform.StressEquilibrium(material, space=space)
    weakform.fbar = True
    assembly = fd.Assembly.create(weakform, mesh)

    # displacement field with a non uniform divergence
    displacement = (0.01 * nodes**2).T.ravel()

    div_u = np.trace(np.array(_comp_grad_disp(assembly, displacement)))
    weights = assembly._get_gaussian_quadrature_mat().data
    expected = (weights @ div_u) / weights.sum()

    div_u_bar = np.trace(_comp_grad_disp_fbar(assembly, displacement))

    assert np.allclose(div_u_bar, expected)
    # a plain arithmetic mean over the gauss points is not the element mean
    # volumetric strain for a distorted element
    assert not np.isclose(div_u.mean(), expected)
