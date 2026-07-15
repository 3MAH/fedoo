import numpy as np
import pytest

import fedoo as fd
from fedoo.core.base import AssemblyBase
from fedoo.core.boundary_conditions import BCBase, BoundaryCondition, ListBC


def _rectangle_assembly():
    fd.Assembly.delete_memory()
    fd.ModelingSpace("2Dstress")
    mesh = fd.mesh.rectangle_mesh(nx=3, ny=3)
    material = fd.constitutivelaw.ElasticIsotrop(1e5, 0.3)
    weakform = fd.weakform.StressEquilibrium(material)
    return fd.Assembly.create(weakform, mesh), mesh


class _GlobalDofBC(BCBase):
    def __init__(self):
        super().__init__()
        self.bc_type = "GlobalDofBC"
        self.registration_count = 0
        self.initialization_count = 0

    def _register_global_dofs(self, problem):
        self.registration_count += 1
        problem.add_global_dof("BCDriver", 1)

    def initialize(self, problem):
        assert "BCDriver" in problem.global_dof
        self.initialization_count += 1


class _GlobalDofAssembly(AssemblyBase):
    def __init__(self, mesh, space):
        super().__init__(space=space)
        self.mesh = mesh
        self.registration_count = 0

    def _register_global_dofs(self, problem):
        self.registration_count += 1
        problem.add_global_dof("AssemblyDriver", 1)

    def initialize(self, problem):
        assert "AssemblyDriver" in problem.global_dof


def test_mpc_canonical_equations_preserve_coefficients_and_constant():
    assembly, mesh = _rectangle_assembly()
    pb = fd.problem.Linear(assembly)
    mpc = fd.MPC(
        [[0], [1]],
        ["DispX", "DispX"],
        [2.0, -3.0],
        constant=7.0,
    )
    mpc.initialize(pb)

    list(mpc.generate(pb, t_fact=0.5))
    dofs, coefficients, values = mpc.get_generated_equations()

    assert np.array_equal(dofs, [[0, 1]])
    assert np.array_equal(coefficients, [[2.0, -3.0]])
    assert np.array_equal(values, [-3.5])
    assert np.array_equal(mpc._dof_index, [[0], [1]])
    assert np.array_equal(mpc._factors, [1.5])
    assert mpc._current_value == -1.75


def test_bc_base_registers_global_dofs_once_before_initialization():
    assembly, _ = _rectangle_assembly()
    pb = fd.problem.Linear(assembly)
    bc = _GlobalDofBC()

    pb.bc.add(bc)
    bc.register_global_dofs(pb)

    assert bc.registration_count == 1
    assert bc.initialization_count == 1


def test_assembly_base_registers_global_dofs_once_before_initialization():
    assembly, mesh = _rectangle_assembly()
    global_dof_assembly = _GlobalDofAssembly(mesh, assembly.space)

    pb = fd.problem.Linear(assembly + global_dof_assembly)
    global_dof_assembly.register_global_dofs(pb)

    assert assembly._global_dof_registered_problems is None
    assert global_dof_assembly.registration_count == 1
    assert pb in global_dof_assembly._global_dof_registered_problems


def test_mpc_constant_uses_incremental_time_function():
    assembly, _ = _rectangle_assembly()
    pb = fd.problem.Linear(assembly)
    mpc = fd.MPC(
        [[0], [1]],
        ["DispX", "DispX"],
        [1.0, -1.0],
        constant=4.0,
        start_constant=1.0,
        time_func=lambda time: time**2,
    )
    mpc.initialize(pb)

    list(mpc.generate(pb, t_fact=0.75, t_fact_old=0.5))
    _, _, values = mpc.get_generated_equations()
    assert np.allclose(values, [-(0.75**2 - 0.5**2) * (4.0 - 1.0)])

    list(mpc.generate(pb, t_fact=0.75))
    _, _, absolute_values = mpc.get_generated_equations()
    assert np.allclose(absolute_values, [-(0.75**2) * (4.0 - 1.0)])


def test_lagrange_assembly_preserves_non_unit_first_coefficient():
    assembly, mesh = _rectangle_assembly()
    mpc = fd.MPC(
        [[0], [1]],
        ["DispX", "DispX"],
        [2.0, -3.0],
        constant=7.0,
    )
    constraint = fd.LagrangeMultiplierAssembly(mesh, mpc, name="Equation")
    pb = fd.problem.Linear(assembly + constraint)

    matrix = constraint.get_global_matrix()
    assert np.array_equal(pb.global_dof["Equation"], [0])
    lm_dof = pb.n_node_dof + pb.global_dof.indice_start("Equation")

    assert matrix[lm_dof, 0] == 2.0
    assert matrix[lm_dof, 1] == -3.0
    assert constraint.get_global_vector()[lm_dof] == -7.0


def test_lagrange_assembly_wraps_periodic_bc_generator():
    assembly, mesh = _rectangle_assembly()
    constraint = fd.LagrangeMultiplierAssembly(
        mesh,
        fd.constraint.PeriodicBC(periodicity_type="small_strain"),
        name="PeriodicLM",
    )
    pb = fd.problem.Linear(assembly + constraint)

    matrix = constraint.get_global_matrix()

    assert constraint._n_constraints > 0
    assert matrix.shape == (pb.n_dof, pb.n_dof)
    assert np.count_nonzero(matrix.diagonal()) == 0


def test_lagrange_assembly_wraps_nonlinear_rigid_tie_generator():
    assembly, mesh = _rectangle_assembly()
    nodes = mesh.find_nodes("X", mesh.bounding_box.xmax)
    constraint = fd.LagrangeMultiplierAssembly(
        mesh, fd.constraint.RigidTie2D(nodes), name="RigidLM"
    )
    pb = fd.problem.NonLinear(assembly + constraint)
    pb.t0, pb.tmax, pb.time, pb.dtime = 0, 1, 0, 0.1

    pb.initialize()

    assert constraint._n_constraints == 2 * len(nodes)
    assert constraint.get_global_matrix().shape == (pb.n_dof, pb.n_dof)


def test_lagrange_assembly_rejects_non_mpc_generated_leaf():
    assembly, mesh = _rectangle_assembly()
    mixed = ListBC(
        [
            fd.MPC([[0], [1]], ["DispX", "DispX"], [1.0, -1.0]),
            BoundaryCondition("Dirichlet", 0, "DispY", 0),
        ]
    )
    constraint = fd.LagrangeMultiplierAssembly(mesh, mixed)

    with pytest.raises(TypeError, match="generated leaves are MPCs"):
        fd.problem.Linear(assembly + constraint)


def test_lagrange_assembly_tracks_nonlinear_absolute_target():
    assembly, mesh = _rectangle_assembly()
    nodes = np.arange(mesh.n_nodes)
    weights = np.full(mesh.n_nodes, 1.0 / mesh.n_nodes)
    targets = np.array([0.02, -0.01])
    mpcs = [
        fd.MPC(
            [[int(node)] for node in nodes],
            [variable] * len(nodes),
            [float(weight) for weight in weights],
            constant=-target,
        )
        for variable, target in zip(("DispX", "DispY"), targets)
    ]
    constraint = fd.LagrangeMultiplierAssembly(mesh, mpcs, name="NonlinearMeanLM")
    pb = fd.problem.NonLinear(assembly + constraint)
    pb.set_solver("direct_scipy", solver_type=None, pc_type=None)
    pb.bc.add(fd.constraint.PeriodicBC(periodicity_type="small_strain"))
    pb.bc.add("Dirichlet", "MeanStrain", [0.01, 0.0, 0.0])

    pb.nlsolve(dt=0.5, tmax=1, update_dt=False, print_info=0)

    assert np.allclose(pb.get_disp().mean(axis=1), targets, atol=1e-10)


def test_mean_value_constraint_uses_generic_lagrange_assembly():
    assembly, mesh = _rectangle_assembly()
    constraint = fd.constraint.MeanValueConstraint(mesh, "Disp")
    pb = fd.problem.Linear(assembly + constraint)
    pb.set_solver("direct_scipy")
    pb.bc.add(fd.constraint.PeriodicBC(periodicity_type="small_strain"))
    pb.bc.add("Dirichlet", "MeanStrain", [0.01, 0.0, 0.0])

    pb.solve()

    assert isinstance(constraint, fd.LagrangeMultiplierAssembly)
    assert np.max(np.abs(pb.get_disp().mean(axis=1))) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__])
