import warnings

import numpy as np
import pytest

import fedoo as fd
from fedoo.weakform.stress_equilibrium import _element_projection

DISTORTED_HEX8 = np.array(
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
)


class _ReferenceBbar(fd.weakform.StressEquilibrium):
    """Explicit B-bar form: eps_bar^T H eps_bar with the modified strains."""

    def get_weak_equation(self, assembly, pb):
        eps = self.space.op_strain()
        correction = self._get_dilatation_correction_op(assembly)
        n = self.space.ndim
        for i in range(n):
            eps[i] = eps[i] + correction * (1 / n)
        H = assembly.sv["TangentMatrix"]
        sigma = [
            sum(0 if eps[j] == 0 else eps[j] * H[i][j] for j in range(6))
            for i in range(6)
        ]
        return sum(0 if eps[i] == 0 else eps[i].virtual * sigma[i] for i in range(6))


def _distorted_mesh(dimension, elm_type):
    if dimension == "3D":
        mesh = fd.mesh.box_mesh(3, 3, 3, elm_type=elm_type)
    else:
        mesh = fd.mesh.rectangle_mesh(3, 3, elm_type=elm_type)
    rng = np.random.default_rng(1)
    mesh.nodes = mesh.nodes + 0.08 * rng.random(mesh.nodes.shape)
    return mesh


def _assemble_matrix(dimension, elm_type, weakform_type, **kargs):
    fd.Assembly.delete_memory()
    fd.ModelingSpace(dimension)
    material = fd.constitutivelaw.ElasticIsotrop(1000.0, 0.499)
    weakform = weakform_type(material, **kargs)
    assembly = fd.Assembly.create(weakform, _distorted_mesh(dimension, elm_type))
    fd.problem.Linear(assembly)
    assembly.assemble_global_mat("matrix")
    return assembly.get_global_matrix().toarray(), assembly


def test_mean_dilatation_operator_is_the_deviation_from_the_volume_mean():
    fd.Assembly.delete_memory()
    space = fd.ModelingSpace("3D")
    mesh = fd.Mesh(
        DISTORTED_HEX8, np.array([np.arange(8)]), "hex8", register_name=False
    )
    material = fd.constitutivelaw.ElasticIsotrop(1000.0, 0.3)
    weakform = fd.weakform.StressEquilibrium(
        material, space=space, incompressibility="mean_dilatation"
    )
    assembly = fd.Assembly.create(weakform, mesh)
    fd.problem.Linear(assembly)

    assert assembly.elm_type == "hex8md"
    weights = assembly._get_gaussian_quadrature_mat().data
    for direction, coordinate in enumerate(("X", "Y", "Z")):
        standard = assembly._get_elementary_operator(
            space.derivative("DispX", coordinate).op[0]
        )[0].toarray()
        deviation = assembly._get_elementary_operator(
            space.derivative("_DispX", coordinate).op[0]
        )[0].toarray()
        volume_mean = (weights @ standard) / weights.sum()
        assert np.allclose(deviation, volume_mean - standard)
        # the volume mean is not the arithmetic mean for a distorted element
        assert not np.allclose(deviation, standard.mean(axis=0) - standard)


@pytest.mark.parametrize("n_basis", [1, 4])
def test_element_projection_preserves_the_basis(n_basis):
    # a projection with non uniform weights should preserve constant fields
    # (and linear fields with a linear basis)
    rng = np.random.default_rng(0)
    n_elm_gp, n_elements = 14, 5
    xi = rng.random((n_elm_gp, 3))
    basis = np.c_[np.ones(n_elm_gp), xi][:, :n_basis]
    weights = 0.1 + rng.random((n_elm_gp, n_elements))
    field = (basis @ rng.random((n_basis, n_elements))).reshape(n_elm_gp, n_elements)
    assert np.allclose(_element_projection(field, weights, basis), field)


@pytest.mark.parametrize(
    "dimension,elm_type",
    [
        ("3D", "hex8"),
        ("3D", "hex20"),
        ("2Dplane", "quad4"),
        ("2Dplane", "quad8"),
        ("2Dplane", "tri6"),
    ],
)
def test_mean_dilatation_matrix_is_the_symmetric_bbar_form(dimension, elm_type):
    matrix, assembly = _assemble_matrix(
        dimension,
        elm_type,
        fd.weakform.StressEquilibrium,
        incompressibility="mean_dilatation",
    )
    reference, _ = _assemble_matrix(
        dimension, elm_type, _ReferenceBbar, incompressibility="mean_dilatation"
    )
    standard, _ = _assemble_matrix(dimension, elm_type, fd.weakform.StressEquilibrium)

    assert assembly.elm_type == elm_type + "md"
    scale = np.abs(matrix).max()
    assert np.allclose(matrix, reference, rtol=1e-10, atol=1e-10 * scale)
    assert np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-10 * scale)
    assert not np.allclose(matrix, standard, rtol=1e-3, atol=1e-3 * scale)


@pytest.mark.parametrize("dimension,elm_type", [("3D", "hex8"), ("2Dplane", "quad4")])
def test_sri_attribute_is_the_bbar_weakform(dimension, elm_type):
    matrix, assembly = _assemble_matrix(
        dimension, elm_type, fd.weakform.StressEquilibrium, incompressibility="sri"
    )
    reference, _ = _assemble_matrix(
        dimension, elm_type, fd.weakform.StressEquilibriumBbar
    )
    assert assembly.elm_type == elm_type + "sri"
    scale = np.abs(matrix).max()
    assert np.allclose(matrix, reference, rtol=1e-10, atol=1e-10 * scale)


def test_incompressibility_attribute():
    fd.ModelingSpace("3D")
    material = fd.constitutivelaw.ElasticIsotrop(1000.0, 0.3)
    weakform = fd.weakform.StressEquilibrium(material)
    assert weakform.incompressibility is None
    assert weakform.fbar is False
    assert weakform.assembly_options.get("elm_type", "hex8") is None

    weakform.incompressibility = "auto"
    assert weakform.assembly_options.get("elm_type", "hex8") == "hex8md"
    assert weakform.assembly_options.get("elm_type", "hex20") == "hex20md"
    assert weakform.assembly_options.get("elm_type", "tet4") is None

    weakform.incompressibility = "sri"
    assert weakform.assembly_options.get("elm_type", "hex8") == "hex8sri"
    assert weakform.assembly_options.get("elm_type", "hex20") is None

    weakform.fbar = True
    assert weakform.incompressibility == "fbar"
    assert weakform.assembly_options.get("elm_type", "hex8") == "hex8sri"
    weakform.fbar = False
    assert weakform.incompressibility is None

    with pytest.raises(ValueError):
        weakform.incompressibility = "unknown"
    with pytest.raises(ValueError):
        fd.weakform.StressEquilibriumMixed(material).incompressibility = "auto"


def test_incompressibility_unavailable_elements():
    fd.Assembly.delete_memory()
    fd.ModelingSpace("2Dplane")
    material = fd.constitutivelaw.ElasticIsotrop(1000.0, 0.499)
    mesh = fd.mesh.rectangle_mesh(3, 3, elm_type="tri3")

    weakform = fd.weakform.StressEquilibrium(material, incompressibility="auto")
    assembly = fd.Assembly.create(weakform, mesh)
    with pytest.warns(UserWarning, match="volumetric locking"):
        fd.problem.Linear(assembly)
    assert assembly.elm_type == "tri3"

    weakform = fd.weakform.StressEquilibrium(
        material, incompressibility="mean_dilatation"
    )
    assembly = fd.Assembly.create(weakform, mesh)
    with pytest.raises(ValueError, match="not available"):
        fd.problem.Linear(assembly)

    # not enough gauss points for the projection
    mesh = fd.mesh.rectangle_mesh(3, 3, elm_type="quad4")
    weakform = fd.weakform.StressEquilibrium(
        material, incompressibility="mean_dilatation"
    )
    assembly = fd.Assembly.create(weakform, mesh, n_elm_gp=1)
    with pytest.raises(ValueError, match="gauss points"):
        fd.problem.Linear(assembly)


def test_default_formulation_is_unchanged():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        matrix, assembly = _assemble_matrix("3D", "hex8", fd.weakform.StressEquilibrium)
    assert assembly.elm_type == "hex8"


def test_mean_dilatation_removes_volumetric_locking():
    # bending of a nearly incompressible beam in plane strain
    def tip_deflection(incompressibility, nu):
        fd.Assembly.delete_memory()
        fd.ModelingSpace("2Dplane")
        mesh = fd.mesh.rectangle_mesh(
            21, 5, x_min=0, x_max=10, y_min=0, y_max=1, elm_type="quad4"
        )
        material = fd.constitutivelaw.ElasticIsotrop(1000.0, nu)
        weakform = fd.weakform.StressEquilibrium(
            material, incompressibility=incompressibility
        )
        assembly = fd.Assembly.create(weakform, mesh)
        pb = fd.problem.Linear(assembly)
        left = mesh.find_nodes("X", 0)
        right = mesh.find_nodes("X", 10)
        pb.bc.add("Dirichlet", left, "Disp", 0)
        pb.bc.add("Neumann", right, "DispY", -0.01)
        pb.solve()
        return -pb.get_disp("DispY")[right].mean()

    # without locking, the deflection barely depends on the compressibility
    ratio_standard = tip_deflection(None, 0.4999) / tip_deflection(None, 0.3)
    ratio_mean_dilatation = tip_deflection("mean_dilatation", 0.4999) / tip_deflection(
        "mean_dilatation", 0.3
    )
    assert ratio_standard < 0.2
    assert 0.7 < ratio_mean_dilatation < 1.1


@pytest.mark.parametrize(
    "dimension,elm_type,method",
    [
        ("3D", "hex8", "mean_dilatation"),
        ("3D", "hex20", "mean_dilatation"),
        ("2Dplane", "quad4", "mean_dilatation"),
        ("2Dplane", "quad8", "mean_dilatation"),
        ("3D", "hex8", "fbar"),
        ("2Dplane", "quad4", "fbar"),
    ],
)
def test_incompressible_finite_strain_tangent_is_fd_consistent(
    dimension, elm_type, method
):
    from .test_finite_strain_tangent_consistency import (
        KAPPA,
        MU,
        _directional_fd_error,
    )

    fd.Assembly.delete_memory()
    fd.ModelingSpace(dimension)
    mesh = _distorted_mesh(dimension, elm_type)
    material = fd.constitutivelaw.Simcoon("NEOHC", [MU, KAPPA], name="law")
    weakform = fd.weakform.StressEquilibrium(
        material, nlgeom="UL", incompressibility=method
    )
    weakform.geometric_stiffness = True
    weakform.corate = "log"
    assembly = fd.Assembly.create(weakform, mesh, name="asm")

    pb = fd.problem.NonLinear("asm")
    pb.set_nr_criterion("Displacement", err0=1.0, tol=1e-6, max_subiter=12)
    ndim = mesh.nodes.shape[1]
    height = mesh.nodes[:, ndim - 1]
    bottom = np.where(height < height.min() + 0.2)[0]
    top = np.where(height > height.max() - 0.2)[0]
    # non homogeneous state (stretch + shear) so that J is not uniform
    pb.bc.add("Dirichlet", bottom, "Disp", 0)
    pb.bc.add("Dirichlet", top, "Disp", 0)
    pb.bc.add("Dirichlet", top, "DispX", 0.04)
    pb.bc.add("Dirichlet", top, ["DispY", "DispZ"][ndim - 2], 0.06)
    pb.nlsolve(dt=0.25, tmax=1.0, update_dt=True, print_info=0)

    blocked = np.concatenate([bottom, top])
    blocked = np.concatenate([blocked + i * mesh.n_nodes for i in range(ndim)])
    free = np.setdiff1d(np.arange(ndim * mesh.n_nodes), blocked)
    err = _directional_fd_error(pb, assembly, free)

    assert err < 1e-6, f"{method} tangent inconsistent with FD ({err:.3e})"
    if method == "mean_dilatation":
        volume_change = assembly.sv["Jbar"]
        assert volume_change.max() - volume_change.min() > 1e-4
        matrix = assembly.current.get_global_matrix()
        assert abs(matrix - matrix.T).max() < 1e-10 * abs(matrix).max()
