"""Consistent F-bar method in finite strain, updated and total Lagrangian.

The total-Lagrangian residual is the pull-back of the updated-Lagrangian one
(de Souza Neto et al., IJSS 1996): S = J F^-1 sigma(F_bar) F^-T, with the
actual F and J. Both formulations reach the same solution, and both use the
consistent (non symmetric) tangent.
"""

import warnings

import numpy as np
import pytest

import fedoo as fd

NODES = np.array(
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
MU, KAPPA = 1.0, 1.0e3  # nearly incompressible Neo-Hookean (NEOHC)


def _single_element_problem(nlgeom):
    fd.Assembly.delete_memory()
    fd.ModelingSpace("3D")
    mesh = fd.Mesh(NODES, np.array([np.arange(8)]), "hex8", register_name=False)
    material = fd.constitutivelaw.Simcoon("NEOHC", np.array([MU, KAPPA]))
    weakform = fd.weakform.StressEquilibrium(
        material, incompressibility="fbar", nlgeom=nlgeom
    )
    assembly = fd.Assembly.create(weakform, mesh)
    pb = fd.problem.NonLinear(assembly)
    pb.print_info = 0
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)  # no "tangent unavailable"
        pb.initialize()
    return pb, assembly


def _residual(pb, assembly, displacement):
    pb._U = displacement
    assembly.update(pb, "all")
    return np.asarray(assembly.current.get_global_vector()).ravel()


@pytest.mark.parametrize("nlgeom", ["UL", "TL"])
def test_finite_strain_fbar_tangent_matches_residual(nlgeom):
    pb, assembly = _single_element_problem(nlgeom)
    rng = np.random.default_rng(3)
    displacement = rng.normal(scale=0.08, size=pb.n_dof)

    _residual(pb, assembly, displacement)
    matrix = assembly.current.get_global_matrix().toarray()

    h = 1.0e-6
    fd_matrix = np.empty_like(matrix)
    for k in range(pb.n_dof):
        step = np.zeros(pb.n_dof)
        step[k] = h
        fd_matrix[:, k] = -(
            _residual(pb, assembly, displacement + step)
            - _residual(pb, assembly, displacement - step)
        ) / (2 * h)

    assert np.abs(matrix - fd_matrix).max() <= 1.0e-6 * np.abs(fd_matrix).max()


def _clamped_cube_reaction(nlgeom):
    # cube clamped at both ends and stretched: heterogeneous, nearly
    # incompressible field. A fixed time step and few Newton iterations make
    # the solve fail unless the tangent is consistent.
    fd.Assembly.delete_memory()
    fd.ModelingSpace("3D")
    mesh = fd.mesh.box_mesh(3, 3, 3, 0, 1, 0, 1, 0, 1, "hex8", name="")
    material = fd.constitutivelaw.Simcoon("NEOHC", np.array([MU, KAPPA]))
    weakform = fd.weakform.StressEquilibrium(
        material, incompressibility="fbar", nlgeom=nlgeom
    )
    assembly = fd.Assembly.create(weakform, mesh)
    pb = fd.problem.NonLinear(assembly)
    pb.set_nr_criterion("Force", tol=1.0e-10, max_subiter=8)
    left, right = mesh.find_nodes("X", 0.0), mesh.find_nodes("X", 1.0)
    pb.bc.add("Dirichlet", left, "Disp", 0.0)
    pb.bc.add("Dirichlet", right, "DispY", 0.0)
    pb.bc.add("Dirichlet", right, "DispZ", 0.0)
    pb.bc.add("Dirichlet", right, "DispX", 0.3)
    pb.nlsolve(dt=0.25, tmax=1.0, update_dt=False, print_info=0)
    return pb.get_ext_forces("Disp")[0, right].sum()


def test_ul_and_tl_fbar_reach_the_same_solution():
    reaction_ul = _clamped_cube_reaction("UL")
    reaction_tl = _clamped_cube_reaction("TL")
    assert reaction_tl == pytest.approx(reaction_ul, rel=1.0e-9)
