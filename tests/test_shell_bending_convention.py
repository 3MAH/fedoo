"""Physical face and layer orientation checks for the shell curvature convention."""

import numpy as np
import pytest

import fedoo as fd
from fedoo.weakform.plate import _ShellComponentList


def test_downward_cantilever_has_tension_on_positive_normal_face():
    fd.ModelingSpace("3D")
    material = fd.constitutivelaw.ElasticIsotrop(1e5, 0.25)
    section = fd.constitutivelaw.ShellHomogeneous(material, 0.05)
    mesh = fd.mesh.rectangle_mesh(9, 3, 0, 1, -0.1, 0.1, "quad4", ndim=3)
    assembly = fd.Assembly.create(
        fd.weakform.PlateEquilibrium(section), mesh, "pquad4mitc"
    )
    pb = fd.problem.Linear(assembly)
    pb.bc.add("Dirichlet", mesh.node_sets["left"], ["Disp", "Rot"], 0)
    right = mesh.node_sets["right"]
    pb.bc.add("Neumann", right, "DispZ", -1 / len(right))
    pb.apply_boundary_conditions()
    pb.solve()
    pb.update()
    assert np.mean(pb.get_disp()[2, right]) < 0
    assert np.mean(assembly.get_element_local_frame()[:, 2, 2]) > 0
    assert np.mean(assembly.sv["ShellStrain"][3]) < 0
    top = section.get_stress(assembly, position=1)["XX"]
    bottom = section.get_stress(assembly, position=-1)["XX"]
    assert np.mean(top) > 0
    np.testing.assert_allclose(bottom, -top, atol=1e-10)
    # Generalized moments and analytical tensor recovery use the same sign.
    result = pb.get_results(assembly, ["ShellStress"])
    result.shell_options.update(reduction=None, position=1)
    np.testing.assert_allclose(result["Stress", "XX"], top)


@pytest.mark.parametrize("nonlinear", [False, True])
def test_asymmetric_laminate_extension_bends_toward_stiffer_top_layer(nonlinear):
    fd.ModelingSpace("3D")
    materials = [fd.constitutivelaw.ElasticIsotrop(e, 0) for e in (1000, 4000)]
    cls = (
        fd.constitutivelaw.ShellLaminateNonLinear
        if nonlinear
        else fd.constitutivelaw.ShellLaminate
    )
    section = cls(materials, [1.0, 1.0])
    mesh = fd.mesh.rectangle_mesh(2, 2, 0, 1, 0, 1, "quad4", ndim=3)
    assembly = fd.Assembly.create(fd.weakform.PlateEquilibrium(section), mesh)
    pb = fd.problem.Linear(assembly)
    stiffness = np.asarray(section.get_shell_stiffness_matrix(), dtype=float)
    if stiffness.ndim == 3:
        stiffness = stiffness[:, :, 0]
    expected = np.array([[5000, -1500], [-1500, 5000 / 3]])
    np.testing.assert_allclose(stiffness[np.ix_([0, 3], [0, 3])], expected)
    if not nonlinear:
        np.testing.assert_allclose(
            np.asarray(section.get_shell_stiffness_matrix_FI(), dtype=float),
            stiffness[:6, :6],
        )
    loading = np.zeros(8)
    loading[0] = 1
    strains = np.linalg.solve(stiffness, loading)
    assert strains[3] > 0  # theta_y,x < 0; strain is smaller at the stiff top.
    n = assembly.n_gauss_points
    assembly.sv["ShellStrain"] = _ShellComponentList(
        [np.full(n, value) for value in strains]
    )
    section.update(assembly, pb)
    np.testing.assert_allclose(
        np.asarray(assembly.sv["ShellStress"]),
        np.broadcast_to(loading[:, None], (8, n)),
        atol=1e-12,
    )
    # Independently integrate physical sigma = E*(epsilon0-z*kappa).
    points, weights = np.polynomial.legendre.leggauss(3)
    force, moment = 0.0, 0.0
    for lower, upper, modulus in ((-1, 0, 1000), (0, 1, 4000)):
        z = (lower + upper) / 2 + points / 2
        sigma = modulus * (strains[0] - z * strains[3])
        force += np.sum(weights / 2 * sigma)
        moment += np.sum(weights / 2 * z * sigma)
    assert force == pytest.approx(1)
    assert abs(moment) < 1e-12
    # Strain at material points must follow physical thickness coordinates.
    np.testing.assert_allclose(
        section.get_strain(assembly, position=1)["XX"], strains[0] - strains[3]
    )
