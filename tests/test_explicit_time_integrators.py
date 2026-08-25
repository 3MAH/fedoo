import numpy as np
from scipy import sparse

import fedoo as fd
from fedoo.core.base import AssemblyBase
from fedoo.core.time_evolution import SECOND_ORDER
from fedoo.time.explicit import ExplicitDynamicState


class _MatrixStorageAssembly(AssemblyBase):
    storage_matrix_is_constant = True

    def __init__(self, mesh, space):
        space.new_variable("DispX")
        space.new_variable("DispY")
        space.new_vector("Disp", ("DispX", "DispY"))
        super().__init__(space=space)
        self.mesh = mesh
        self.time_evolution = SECOND_ORDER
        self.storage = self
        self.dissipation = None
        self._time_integrator = None
        self._fedoo_time_integrated = False
        self._indices = None
        self._pb = None
        self.storage_calls = 0
        self.update_calls = 0
        self.set_start_calls = 0

    @property
    def time_dof_indices(self):
        return self._indices

    def get_storage_matrix(self, pb=None):
        self.storage_calls += 1
        return self._storage_matrix()

    def _storage_matrix(self):
        mass = np.eye(len(self._indices))
        mass[:2, :2] = [[2.0, 1.0], [1.0, 2.0]]
        return mass

    def get_time_inertia_force(self, pb, acceleration, velocity):
        force = self._storage_matrix() @ acceleration
        force[0] += velocity[0]
        return force

    def initialize(self, pb):
        self._pb = pb
        self._indices = np.arange(pb.n_dof)
        if self._time_integrator is not None:
            self._time_integrator.initialize(self, pb)

    def update(self, pb, compute="all"):
        self.update_calls += 1
        self.global_matrix = sparse.csr_matrix((pb.n_dof, pb.n_dof))
        self.global_vector = np.zeros(pb.n_dof)
        self.global_vector[0] = 1.0
        if self._time_integrator is not None:
            self._time_integrator.update(self, pb)
            self._time_integrator.integrate(self, pb, compute)

    def set_start(self, pb):
        self.set_start_calls += 1
        if self._time_integrator is not None:
            self._time_integrator.set_start(self, pb)

    def reset(self):
        self.global_matrix = None
        self.global_vector = None


def test_explicit_newmark_is_generalized_alpha_alias():
    assert fd.time.ExplicitNewmark is fd.time.ExplicitGeneralizedAlpha


def test_central_difference_predictor_is_explicit_newmark():
    integrator = fd.time.CentralDifference()
    state = ExplicitDynamicState(
        displacement=np.array([1.0]),
        velocity=np.array([2.0]),
        acceleration=np.array([3.0]),
    )

    displacement, velocity = integrator.predict(state, dt=0.1)

    np.testing.assert_allclose(displacement, [1.215])
    np.testing.assert_allclose(velocity, [2.15])


def test_explicit_generalized_alpha_uses_alpha_evaluation_state():
    integrator = fd.time.ExplicitGeneralizedAlpha(
        alpha_m=0.2,
        alpha_f=0.25,
        beta=0.1,
        gamma=0.6,
    )
    state = ExplicitDynamicState(
        displacement=np.array([1.0]),
        velocity=np.array([2.0]),
        acceleration=np.array([3.0]),
    )
    predicted = integrator.predict(state, dt=0.1)

    displacement, velocity = integrator.evaluation_state(state, *predicted)

    np.testing.assert_allclose(displacement, 0.75 * predicted[0] + 0.25)
    np.testing.assert_allclose(velocity, 0.75 * predicted[1] + 0.5)


def test_explicit_dynamic_preserves_constant_velocity_without_forces():
    fd.ModelingSpace("2Dplane")
    mesh = fd.mesh.rectangle_mesh(nx=2, ny=2, elm_type="tri3")
    material = fd.constitutivelaw.ElasticIsotrop(0.0, 0.3)
    material.set_density(2.0)
    weakform = fd.weakform.StressEquilibrium(material)
    assembly = fd.Assembly.create(weakform, mesh)
    problem = fd.problem.ExplicitDynamic(assembly, time_step=0.1)
    problem.set_initial_velocity("DispX", 1.0)
    problem.initialize()

    problem.apply_boundary_conditions()
    problem.solve()
    problem.update()

    np.testing.assert_allclose(problem.get_disp("DispX"), 0.1)
    np.testing.assert_allclose(problem.get_disp("DispY"), 0.0)
    np.testing.assert_allclose(problem.get_velocity("DispX"), 1.0)
    np.testing.assert_allclose(problem.get_acceleration(), 0.0, atol=1e-14)
    assert problem.time == 0.1


def test_central_difference_acceleration_opposes_elastic_displacement():
    fd.ModelingSpace("2Dplane")
    mesh = fd.mesh.rectangle_mesh(nx=2, ny=2, elm_type="tri3")
    material = fd.constitutivelaw.ElasticIsotrop(100.0, 0.3)
    material.set_density(1.0)
    weakform = fd.weakform.StressEquilibrium(material)
    assembly = fd.Assembly.create(weakform, mesh)
    problem = fd.problem.ExplicitDynamic(assembly, time_step=1.0e-3)

    left = mesh.find_nodes("X", mesh.bounding_box.xmin)
    right = mesh.find_nodes("X", mesh.bounding_box.xmax)
    initial_x = np.zeros(mesh.n_nodes)
    initial_x[right] = 0.01
    problem.set_initial_displacement("DispX", initial_x)
    problem.initialize()
    problem.bc.add("Dirichlet", left, "Disp", 0.0)

    problem.apply_boundary_conditions()
    problem.solve()
    problem.update()

    assert np.all(problem.get_acceleration("DispX")[right] < 0.0)
    assert np.all(problem.get_velocity("DispX")[right] < 0.0)


def test_explicit_dynamic_uses_assembly_level_storage_matrix_directly():
    space = fd.ModelingSpace("2Dplane")
    mesh = fd.mesh.rectangle_mesh(nx=2, ny=2, elm_type="tri3")
    assembly = _MatrixStorageAssembly(mesh, space)
    problem = fd.problem.ExplicitDynamic(assembly, time_step=0.1)

    problem.initialize()
    problem.apply_boundary_conditions()
    problem.solve()
    problem.update()

    np.testing.assert_allclose(problem.get_acceleration()[:2], [2.0 / 3.0, -1.0 / 3.0])
    assert assembly._fedoo_time_integrated
    assert assembly.storage_calls == 1


def test_explicit_dynamic_uses_assembly_inertia_force_callback():
    space = fd.ModelingSpace("2Dplane")
    mesh = fd.mesh.rectangle_mesh(nx=2, ny=2, elm_type="tri3")
    assembly = _MatrixStorageAssembly(mesh, space)
    problem = fd.problem.ExplicitDynamic(assembly, time_step=0.1)
    initial_velocity = np.zeros(mesh.n_nodes)
    initial_velocity[0] = 1.0
    problem.set_initial_velocity("DispX", initial_velocity)

    problem.initialize()
    problem.apply_boundary_conditions()
    problem.solve()
    problem.update()

    np.testing.assert_allclose(problem.get_acceleration(), 0.0, atol=1e-14)


def test_explicit_dynamic_refreshes_configuration_dependent_storage():
    space = fd.ModelingSpace("2Dplane")
    mesh = fd.mesh.rectangle_mesh(nx=2, ny=2, elm_type="tri3")
    assembly = _MatrixStorageAssembly(mesh, space)
    assembly.storage_matrix_is_constant = False
    problem = fd.problem.ExplicitDynamic(assembly, time_step=0.1)

    problem.initialize()
    assert assembly.storage_calls == 1
    problem.apply_boundary_conditions()
    problem.solve()

    assert assembly.storage_calls == 2


def test_manual_linear_step_does_not_update_or_commit_assembly():
    space = fd.ModelingSpace("2Dplane")
    mesh = fd.mesh.rectangle_mesh(nx=2, ny=2, elm_type="tri3")
    assembly = _MatrixStorageAssembly(mesh, space)
    problem = fd.problem.ExplicitDynamic(assembly, time_step=0.1)

    problem.initialize()
    problem.apply_boundary_conditions()
    problem.solve()
    problem.update()

    assert assembly.update_calls == 1
    assert assembly.set_start_calls == 0

    problem.set_start()
    assert assembly.set_start_calls == 1


def test_update_can_refresh_weakform_at_committed_state():
    space = fd.ModelingSpace("2Dplane")
    mesh = fd.mesh.rectangle_mesh(nx=2, ny=2, elm_type="tri3")
    assembly = _MatrixStorageAssembly(mesh, space)
    problem = fd.problem.ExplicitDynamic(assembly, time_step=0.1)

    problem.initialize()
    problem.apply_boundary_conditions()
    problem.solve()
    problem.update(update_weakform=True)

    assert assembly.update_calls == 2
    assert assembly.set_start_calls == 0


def test_matrix_only_update_discards_cached_internal_force():
    space = fd.ModelingSpace("2Dplane")
    mesh = fd.mesh.rectangle_mesh(nx=2, ny=2, elm_type="tri3")
    assembly = _MatrixStorageAssembly(mesh, space)
    problem = fd.problem.ExplicitDynamic(assembly, time_step=0.1)

    problem.initialize()
    problem.apply_boundary_conditions()
    problem.solve()
    problem.update(update_weakform=True, compute="matrix")

    assert problem._assembled_internal_force is None


def test_solve_time_increment_manages_nonlinear_update_and_commit():
    space = fd.ModelingSpace("2Dplane")
    mesh = fd.mesh.rectangle_mesh(nx=2, ny=2, elm_type="tri3")
    assembly = _MatrixStorageAssembly(mesh, space)
    problem = fd.problem.ExplicitDynamic(assembly, time_step=0.1)

    problem.initialize()
    problem.solve_time_increment(update_weakform=True, set_start=True)

    assert assembly.update_calls == 3
    assert assembly.set_start_calls == 1
    assert problem.time == 0.1


def test_solve_history_selects_fixed_or_updated_weakform():
    space = fd.ModelingSpace("2Dplane")
    mesh = fd.mesh.rectangle_mesh(nx=2, ny=2, elm_type="tri3")
    linear_assembly = _MatrixStorageAssembly(mesh, space)
    linear = fd.problem.ExplicitDynamic(linear_assembly, time_step=0.1)
    linear.solve_history(tmax=0.2, update_weakform=False)

    assert linear_assembly.update_calls == 1
    assert linear_assembly.set_start_calls == 0
    assert linear.time == 0.2

    nonlinear_assembly = _MatrixStorageAssembly(mesh, space)
    nonlinear = fd.problem.ExplicitDynamic(nonlinear_assembly, time_step=0.1)
    nonlinear.solve_history(tmax=0.2, update_weakform=True)

    assert nonlinear_assembly.update_calls == 5
    assert nonlinear_assembly.set_start_calls == 2
    assert nonlinear.time == 0.2


def test_solve_history_saves_at_exact_time_intervals():
    for update_weakform in (False, True):
        space = fd.ModelingSpace("2Dplane")
        mesh = fd.mesh.rectangle_mesh(nx=2, ny=2, elm_type="tri3")
        assembly = _MatrixStorageAssembly(mesh, space)
        problem = fd.problem.ExplicitDynamic(assembly, time_step=0.03)
        saved = []
        problem.save_results = lambda iteration: saved.append((iteration, problem.time))

        problem.solve_history(
            tmax=0.25,
            update_weakform=update_weakform,
            interval_output=0.1,
        )

        np.testing.assert_allclose(
            [time for _, time in saved],
            [0.1, 0.2, 0.25],
        )
        assert [iteration for iteration, _ in saved] == [0, 1, 2]
        assert problem.time_step == 0.03


def test_fixed_solve_history_refreshes_assembly_only_at_output_times():
    space = fd.ModelingSpace("2Dplane")
    mesh = fd.mesh.rectangle_mesh(nx=2, ny=2, elm_type="tri3")
    assembly = _MatrixStorageAssembly(mesh, space)
    problem = fd.problem.ExplicitDynamic(assembly, time_step=0.03)
    problem.save_results = lambda iteration: None

    problem.solve_history(
        tmax=0.25,
        update_weakform=False,
        interval_output=0.1,
    )

    assert assembly.update_calls == 4


def test_explicit_dynamic_has_one_complete_history_api():
    assert hasattr(fd.problem.ExplicitDynamic, "solve_history")
    assert not hasattr(fd.problem.ExplicitDynamic, "lsolve")
    assert not hasattr(fd.problem.ExplicitDynamic, "nlsolve")


def test_consistent_fe_mass_can_be_kept_without_lumping():
    fd.ModelingSpace("2Dplane")
    mesh = fd.mesh.rectangle_mesh(nx=2, ny=2, elm_type="tri3")
    material = fd.constitutivelaw.ElasticIsotrop(1.0, 0.3)
    material.set_density(1.0)
    weakform = fd.weakform.StressEquilibrium(material)
    assembly = fd.Assembly.create(weakform, mesh)
    problem = fd.problem.ExplicitDynamic(
        assembly,
        time_step=0.1,
        mass_lumping=False,
    )

    problem.initialize()

    assert problem.get_A().ndim == 2
