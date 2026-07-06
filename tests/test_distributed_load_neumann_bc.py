import numpy as np
import pytest
from scipy import sparse

import fedoo as fd


def _build_problem(load_nlgeom=None, problem_nlgeom=True, load_kind="pressure"):
    fd.Assembly.delete_memory()
    fd.ModelingSpace("3D")

    mesh = fd.mesh.box_mesh(2, 2, 2, name="load_box")
    material = fd.constitutivelaw.ElasticIsotrop(1000.0, 0.3, name="LoadMat")
    wf = fd.weakform.StressEquilibrium(material)
    solid = fd.Assembly.create(wf, mesh, name="solid")
    pb = fd.problem.NonLinear(solid, nlgeom=problem_nlgeom)

    if load_kind == "pressure":
        top = mesh.find_nodes("Z", mesh.bounding_box.zmax)
        load = fd.constraint.Pressure.from_nodes(
            mesh, top, 7.0, nlgeom=load_nlgeom
        )
    elif load_kind == "distributed":
        load = fd.constraint.DistributedForce(
            mesh, [0.0, 0.0, -7.0], nlgeom=load_nlgeom
        )
    else:
        raise ValueError(load_kind)

    bc = pb.bc.add(load)
    assert isinstance(bc, fd.constraint.AssemblyNeumannBC)
    return pb, bc



def test_assembly_neumann_load_can_still_be_added_explicitly():
    pb, bc = _build_problem(load_nlgeom=False, problem_nlgeom=False)

    pb.bc.remove(bc)
    explicit_bc = bc.assembly.as_neumann()

    assert pb.bc.add(explicit_bc) is explicit_bc


def test_assembly_neumann_load_accepts_time_function_and_caches_fixed_vector():
    pb, bc = _build_problem(load_nlgeom=False, problem_nlgeom=False)
    pb.bc.remove(bc)

    load = bc.assembly
    bc = pb.bc.add(load.as_neumann(time_func=lambda t: t**2))
    assemble_global_mat = load.assemble_global_mat
    call_count = 0

    def counted_assemble_global_mat(compute="all"):
        nonlocal call_count
        call_count += 1
        return assemble_global_mat(compute)

    load.assemble_global_mat = counted_assemble_global_mat

    pb.apply_boundary_conditions(t_fact=0.5, t_fact_old=0.0)
    quarter_load = pb.get_B().copy()

    pb.apply_boundary_conditions(t_fact=0.5, t_fact_old=0.0)
    quarter_load_again = pb.get_B().copy()

    pb.apply_boundary_conditions(t_fact=1.0, t_fact_old=0.5)
    full_load = pb.get_B().copy()

    assert call_count == 2
    np.testing.assert_allclose(quarter_load_again, quarter_load)
    np.testing.assert_allclose(quarter_load, 0.25 * full_load)


def test_raw_assembly_without_neumann_conversion_is_rejected():
    fd.Assembly.delete_memory()
    fd.ModelingSpace("3D")

    mesh = fd.mesh.box_mesh(2, 2, 2, name="raw_assembly_box")
    material = fd.constitutivelaw.ElasticIsotrop(1000.0, 0.3, name="RawMat")
    wf = fd.weakform.StressEquilibrium(material)
    solid = fd.Assembly.create(wf, mesh, name="raw_solid")
    pb = fd.problem.NonLinear(solid, nlgeom=False)

    with pytest.raises(TypeError, match="as_neumann"):
        pb.bc.add(solid)


@pytest.mark.parametrize("load_kind", ["pressure", "distributed"])
def test_assembly_neumann_load_is_ramped_in_external_force_vector(load_kind):
    pb, _ = _build_problem(
        load_nlgeom=False, problem_nlgeom=True, load_kind=load_kind
    )

    pb.apply_boundary_conditions(t_fact=0.5, t_fact_old=0.0)
    half_load = pb.get_B().copy()

    pb.apply_boundary_conditions(t_fact=1.0, t_fact_old=0.5)
    full_load = pb.get_B().copy()

    assert np.linalg.norm(full_load) > 0.0
    np.testing.assert_allclose(half_load, 0.5 * full_load)


def test_assembly_neumann_load_updates_during_increment_only_for_nlgeom():
    _, inherited = _build_problem(load_nlgeom=None, problem_nlgeom=True)
    _, fixed = _build_problem(load_nlgeom=False, problem_nlgeom=True)
    _, small_strain = _build_problem(load_nlgeom=None, problem_nlgeom=False)

    assert inherited._update_during_inc is True
    assert fixed._update_during_inc is False
    assert small_strain._update_during_inc is False


def test_assembly_neumann_load_is_ignored_by_standard_start_value_update():
    pb, _ = _build_problem(load_nlgeom=False, problem_nlgeom=False)

    pb.initialize()
    pb.apply_boundary_conditions(t_fact=1.0, t_fact_old=0.0)
    pb._U = np.ones(pb.n_dof)
    pb.set_X(np.zeros(pb.n_dof))
    pb.set_A(sparse.identity(pb.n_dof, format="csr"))
    pb.set_D(0)

    pb.init_bc_start_value()
