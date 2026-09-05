"""The adaptive-stiffness fallback matrix must follow the current configuration.

`solve_time_increment` captures a "safe" elastic matrix and installs it when
the divergence guard restarts an increment. Reading it from the *reference*
assembly froze it on the undeformed configuration under `nlgeom="UL"` -- and,
for an assembly sum, on a state without the contact block. Since that matrix
stays installed across the following time-step cuts, every retry failed at its
first iteration with an infinite error, down to `dt_min`.
"""

import pytest

import fedoo as fd


def _stretched_ul_problem():
    fd.Assembly.delete_memory()
    fd.ModelingSpace("2Dplane")
    mesh = fd.mesh.rectangle_mesh(4, 4, 0, 1, 0, 1, elm_type="quad4", name="sq")
    material = fd.constitutivelaw.ElasticIsotrop(1e3, 0.3, name="law")
    weakform = fd.weakform.StressEquilibrium(material, name="wf")
    assembly = fd.Assembly.create(weakform, mesh, name="asm")
    problem = fd.problem.NonLinear(assembly, nlgeom="UL")
    problem.set_nr_criterion("Displacement", tol=1e-4, adaptive_stiffness=True)
    problem.bc.add("Dirichlet", mesh.node_sets["bottom"], "Disp", 0)
    problem.bc.add("Dirichlet", mesh.node_sets["top"], "Disp", [0.0, 0.3])
    problem.nlsolve(dt=0.5, tmax=1.0, update_dt=False, print_info=0)
    return problem, assembly


def test_fallback_matrix_comes_from_the_current_assembly():
    problem, assembly = _stretched_ul_problem()
    assert assembly.current is not assembly  # UL works on a deformed copy

    fallback = problem._elastic_reference_matrix()

    assert fallback is assembly.current.get_global_matrix()
    # the reference assembly still holds the undeformed matrix
    assert abs(fallback - assembly.get_global_matrix()).max() > 0


def test_fallback_matrix_can_skip_the_assembly_pass():
    problem, assembly = _stretched_ul_problem()
    problem.assembly.current.assemble_global_mat("matrix")
    expected = assembly.current.get_global_matrix()
    assert problem._elastic_reference_matrix(assemble=False) is expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
