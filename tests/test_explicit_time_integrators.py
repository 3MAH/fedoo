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
        self.velocity_bias = True

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
        if self.velocity_bias:
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


def _make_fe_problem(
    *, young=0.0, density=1.0, dt=0.1, elm_type="tri3", mass_lumping=True
):
    fd.Assembly.delete_memory()
    fd.ModelingSpace("2Dplane")
    mesh = fd.mesh.rectangle_mesh(nx=2, ny=2, elm_type=elm_type)
    material = fd.constitutivelaw.ElasticIsotrop(young, 0.3)
    material.set_density(density)
    assembly = fd.Assembly.create(fd.weakform.StressEquilibrium(material), mesh)
    problem = fd.problem.ExplicitDynamic(
        assembly,
        time_step=dt,
        mass_lumping=mass_lumping,
    )
    return problem, mesh


def _make_storage_problem(dt=0.1):
    space = fd.ModelingSpace("2Dplane")
    mesh = fd.mesh.rectangle_mesh(nx=2, ny=2, elm_type="tri3")
    assembly = _MatrixStorageAssembly(mesh, space)
    return fd.problem.ExplicitDynamic(assembly, time_step=dt), assembly


def test_explicit_dynamic_state_copy_is_independent():
    state = ExplicitDynamicState(
        displacement=np.array([1.0]),
        velocity=np.array([2.0]),
        acceleration=np.array([3.0]),
    )

    copied = state.copy()
    copied.displacement[0] = 4.0
    copied.velocity[0] = 5.0
    copied.acceleration[0] = 6.0

    np.testing.assert_allclose(state.displacement, [1.0])
    np.testing.assert_allclose(state.velocity, [2.0])
    np.testing.assert_allclose(state.acceleration, [3.0])


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
    problem, _ = _make_fe_problem(density=2.0)
    problem.set_initial_velocity("DispX", 1.0)
    problem.solve_time_increment()

    np.testing.assert_allclose(problem.get_disp("DispX"), 0.1)
    np.testing.assert_allclose(problem.get_disp("DispY"), 0.0)
    np.testing.assert_allclose(problem.get_velocity("DispX"), 1.0)
    np.testing.assert_allclose(problem.get_acceleration(), 0.0, atol=1e-14)
    assert problem.time == 0.1


def test_central_difference_acceleration_opposes_elastic_displacement():
    problem, mesh = _make_fe_problem(young=100.0, dt=1.0e-3)

    left = mesh.find_nodes("X", mesh.bounding_box.xmin)
    right = mesh.find_nodes("X", mesh.bounding_box.xmax)
    initial_x = np.zeros(mesh.n_nodes)
    initial_x[right] = 0.01
    problem.set_initial_displacement("DispX", initial_x)
    problem.bc.add("Dirichlet", left, "Disp", 0.0)
    problem.solve_time_increment()

    assert np.all(problem.get_acceleration("DispX")[right] < 0.0)
    assert np.all(problem.get_velocity("DispX")[right] < 0.0)


def test_central_difference_end_acceleration_includes_neumann_loads():
    """The accepted acceleration must satisfy equilibrium with applied loads."""
    problem, mesh = _make_fe_problem(elm_type="quad4")
    right = mesh.find_nodes("X", mesh.bounding_box.xmax)
    problem.bc.add("Neumann", right, "DispX", 1.0)

    problem.solve_time_increment()

    acceleration = problem.get_acceleration()
    velocity = problem.get_velocity()
    external_force = problem.get_B()
    assert np.linalg.norm(acceleration) > 0.0
    np.testing.assert_allclose(problem._mass_matrix @ acceleration, external_force)
    np.testing.assert_allclose(velocity, problem.time_step * acceleration)


def test_central_difference_end_acceleration_includes_rayleigh_damping():
    """Damping must reduce both the accepted velocity and kinetic energy."""
    problem, mesh = _make_fe_problem()
    problem.set_initial_velocity("DispX", 1.0)
    problem.set_rayleigh_damping(alpha=1.0, beta=0.0)

    problem.solve_time_increment()

    np.testing.assert_allclose(problem.get_velocity("DispX"), 0.905)
    np.testing.assert_allclose(problem.get_acceleration("DispX"), -0.9)
    initial_energy = 0.5 * np.sum(problem._mass[: mesh.n_nodes])
    assert problem.get_kinetic_energy() < initial_energy


def test_central_difference_updates_moving_dirichlet_kinematics_without_mpc():
    """A prescribed displacement ramp must update constrained v and a."""
    problem, mesh = _make_fe_problem(young=1.0, elm_type="quad4")
    right = mesh.find_nodes("X", mesh.bounding_box.xmax)
    problem.bc.add("Dirichlet", right, "DispX", 0.1)

    problem.solve_time_increment(t_fact=1.0)

    np.testing.assert_allclose(problem.get_disp("DispX")[right], 0.1)
    np.testing.assert_allclose(problem.get_velocity("DispX")[right], 1.0)
    np.testing.assert_allclose(problem.get_acceleration("DispX")[right], 10.0)


def test_explicit_dynamic_uses_assembly_level_storage_matrix_directly():
    problem, assembly = _make_storage_problem()
    assembly.velocity_bias = False

    problem.initialize()
    problem.apply_boundary_conditions()
    problem.solve()
    problem.update()

    np.testing.assert_allclose(problem.get_acceleration()[:2], [2.0 / 3.0, -1.0 / 3.0])
    assert assembly._fedoo_time_integrated
    assert assembly.storage_calls == 1


def test_explicit_dynamic_uses_assembly_inertia_force_callback():
    problem, _ = _make_storage_problem()
    initial_velocity = np.zeros(problem.mesh.n_nodes)
    initial_velocity[0] = 1.0
    problem.set_initial_velocity("DispX", initial_velocity)

    problem.initialize()
    problem.apply_boundary_conditions()
    problem.solve()
    problem.update()

    np.testing.assert_allclose(problem.get_acceleration(), 0.0, atol=1e-14)


def test_explicit_dynamic_refreshes_configuration_dependent_storage():
    problem, assembly = _make_storage_problem()
    assembly.storage_matrix_is_constant = False

    problem.initialize()
    assert assembly.storage_calls == 1
    problem.apply_boundary_conditions()
    problem.solve()

    assert assembly.storage_calls == 2


def test_manual_steps_control_assembly_update_and_commit():
    problem, assembly = _make_storage_problem()

    problem.initialize()
    problem.apply_boundary_conditions()
    problem.solve()
    problem.update()

    assert assembly.update_calls == 1
    assert assembly.set_start_calls == 0

    problem.apply_boundary_conditions()
    problem.solve()
    problem.update(update_weakform=True)

    assert assembly.update_calls == 2
    assert assembly.set_start_calls == 0

    problem.set_start()
    assert assembly.set_start_calls == 1


def test_matrix_only_update_discards_cached_internal_force():
    problem, _ = _make_storage_problem()

    problem.initialize()
    problem.apply_boundary_conditions()
    problem.solve()
    problem.update(update_weakform=True, compute="matrix")

    assert problem._assembled_internal_force is None


def test_solve_time_increment_manages_nonlinear_update_and_commit():
    problem, assembly = _make_storage_problem()

    problem.initialize()
    problem.solve_time_increment(update_weakform=True, set_start=True)

    # Initialization and the accepted end state are each assembled once. The
    # initial force is reused for evaluation instead of being assembled twice.
    assert assembly.update_calls == 2
    assert assembly.set_start_calls == 1
    assert problem.time == 0.1


def test_solve_history_selects_fixed_or_updated_weakform():
    linear, linear_assembly = _make_storage_problem()
    linear.solve_history(tmax=0.2, update_weakform=False)

    assert linear_assembly.update_calls == 1
    assert linear_assembly.set_start_calls == 0
    assert linear.time == 0.2

    nonlinear, nonlinear_assembly = _make_storage_problem()
    nonlinear.solve_history(tmax=0.2, update_weakform=True)

    assert nonlinear_assembly.update_calls == 3
    assert nonlinear_assembly.set_start_calls == 2
    assert nonlinear.time == 0.2


def test_solve_history_saves_at_exact_time_intervals():
    for update_weakform in (False, True):
        problem, assembly = _make_storage_problem(dt=0.03)
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
        if not update_weakform:
            # Initialization plus one vector refresh at each saved output.
            assert assembly.update_calls == 4
