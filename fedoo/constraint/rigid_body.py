"""Rigid body support for large-displacement dynamic simulation.

Provides:
- ``RigidBodyAssembly``: AssemblyBase that contributes a 6x6 mass matrix and
  a generalized force vector on the rigid body's global DOFs.
- ``RigidBody``: Facade that coordinates a RigidTie constraint (kinematics)
  with a RigidBodyAssembly (dynamics) into a single user-facing object.
  Optionally handles IPC contact via direct Jacobian projection (J^T @ F).

Large rotations
---------------
By default, each step uses an incremental rotation vector composed
multiplicatively with the last converged quaternion orientation. ``RigidTie``
evaluates the exact rotation matrix and its derivatives, so the kinematics do
not rely on a small-angle approximation.

The contact Jacobian ``J = [I_3 | dR/d(angle) @ r_ref]`` uses the exact
rotation derivatives from ``RigidTie.rotation_jacobian()``, not the
infinitesimal skew-symmetric approximation.
"""

import numpy as np

from fedoo.core.base import AssemblyBase
from fedoo.core.mesh import Mesh
from fedoo.core._sparsematrix import scatter_dense_block
from fedoo.core.time_evolution import SECOND_ORDER
from fedoo.constraint.rigid_tie import RigidTie
from fedoo.time.common import RayleighDamping

# Rigid-body dynamics is formulated in 3D: 3 translational + 3 rotational DOFs.
_N_RIGID_DOF = 6


class RigidBodyAssembly(AssemblyBase):
    """Assembly contributing rigid body mass and generalized forces.

    Injects a 6x6 mass matrix and a 6-component force vector at the global
    DOF positions created by a :class:`RigidTie` constraint.

    The mass matrix has the structure::

        M = [[m*I_3,   0   ],
             [  0,   J_global]]

    where ``J_global = R @ J_body @ R.T`` is the inertia tensor rotated
    to the current trial configuration.

    Parameters
    ----------
    mass : float
        Total mass of the rigid body.
    inertia_tensor : array_like, shape (3, 3)
        Inertia tensor about the center of mass in the body-fixed frame.
    rigid_tie : RigidTie
        The associated rigid tie constraint.
    mesh : fedoo.Mesh, optional
        Mesh (needed by the Fedoo Problem constructor).
    space : ModelingSpace, optional
        Modeling space.
    name : str, optional
        Name of the assembly.
    """

    def __init__(
        self,
        mass,
        inertia_tensor,
        rigid_tie,
        mesh=None,
        space=None,
        dynamic=True,
        name="RigidBodyAssembly",
    ):
        if space is None:
            from fedoo.core.modelingspace import ModelingSpace

            space = ModelingSpace.get_active()
        AssemblyBase.__init__(self, name, space)
        if self.space.ndim != 3:
            raise NotImplementedError(
                "RigidBodyAssembly supports 3D modeling spaces only "
                f"(got ndim={self.space.ndim}). RigidTie2D provides 2D rigid "
                "kinematics, but 2D rigid-body dynamics is not yet implemented."
            )
        self.mass = float(mass)
        self.inertia_body = np.asarray(inertia_tensor, dtype=float)
        self.rigid_tie = rigid_tie
        self.force = np.zeros(_N_RIGID_DOF)
        self.mesh = mesh
        self.time_evolution = SECOND_ORDER if dynamic else None
        self.storage = self if dynamic else None
        self.dissipation = None
        self._time_integrator = None
        self._fedoo_time_integrated = False
        self.dynamic = bool(dynamic)
        # A compiled time integrator supplies a mass tangent. Without one, the
        # assembly is static and this tiny diagonal keeps the free rigid DOFs
        # well posed before contact.
        self.static_regularisation = 1e-9

        self._dof_indices = None
        self._pb_ref = None

        self.sv = {}
        self.sv_start = {}

        self._ipc_collision_mesh = None
        self._ipc_collisions = None
        self._ipc_barrier = None
        self._ipc_kappa = None
        self._ipc_dhat = None
        self._ipc_broad_phase = None
        self._ipc_rest_positions = None
        self._ipc_n_body = 0
        self._ipc_obstacle_nodes = None
        self._ipc_obstacle_mesh = None
        self._ipc_obstacle_source_id = None
        self._contact_force = np.zeros(_N_RIGID_DOF)
        self._contact_stiffness = np.zeros((_N_RIGID_DOF, _N_RIGID_DOF))

    def _register_global_dofs(self, pb):
        """Register the rigid kinematic constraint and its global DOFs."""
        if self.rigid_tie not in pb.bc:
            pb.bc.add(self.rigid_tie)

    def initialize(self, pb):
        """Extract global DOF indices from the problem."""
        rt = self.rigid_tie
        # Same global-DOF layout the RigidTie constraint resolves; reuse it
        # so the two never drift apart.
        self._dof_indices = np.array(rt._get_dof_ref(pb)[1])
        self._pb_ref = pb
        if self.mesh is None:
            self.mesh = pb.mesh

        if SECOND_ORDER not in getattr(pb, "time_integrators", {}):
            self._time_integrator = None
            self._fedoo_time_integrated = False
        elif self._time_integrator is not None:
            self._time_integrator.initialize(self, pb)

    @property
    def dof_indices(self):
        """Global DOF indices [Fx,Fy,Fz,Mx,My,Mz] in the problem."""
        return self._dof_indices

    def _build_ipc_jacobian(self, rt, angles):
        """Build contact Jacobian J mapping 6 rigid DOFs to all vertex DOFs.

        Uses exact rotation derivatives from RigidTie for consistency.

        Returns
        -------
        J : ndarray, shape (n_all*3, 6)
            Body vertices use exact derivatives; obstacle rows are zero.
        """
        _, du_drx, du_dry, du_drz = rt.rotation_jacobian(
            angles, self._ipc_rest_positions
        )

        n_body = self._ipc_n_body
        n_all = n_body + len(self._ipc_obstacle_nodes)
        J = np.zeros((n_all * 3, _N_RIGID_DOF))
        for d in range(3):
            J[d : n_body * 3 : 3, d] = 1.0
            J[d : n_body * 3 : 3, 3] = du_drx[:, d]
            J[d : n_body * 3 : 3, 4] = du_dry[:, d]
            J[d : n_body * 3 : 3, 5] = du_drz[:, d]
        return J

    def _ipc_vertices(self, q, rt):
        """Compute all vertex positions (body + obstacle) from q."""
        R, _, _, _ = rt._compute_rotation(q[3:])
        r_ref = self._ipc_rest_positions - rt.center
        body_verts = r_ref @ R.T + rt.center + q[:3]
        return np.vstack([body_verts, self._ipc_obstacle_nodes])

    def compute_contact(self, q, rt, compute="all"):
        """Compute IPC contact force and/or stiffness on 6 rigid DOFs.

        Parameters
        ----------
        q : ndarray (6,)
            Rigid body DOFs [dx, dy, dz, rx, ry, rz].
        rt : RigidTie
            Associated rigid tie constraint.
        compute : str
            ``"all"`` (force + stiffness), ``"force"`` (skip hessian),
            or ``"stiffness"``.

        Returns
        -------
        force : ndarray (6,)
        stiffness : ndarray (6, 6)
        """
        if self._ipc_collision_mesh is None:
            self._contact_force[:] = 0
            self._contact_stiffness[:] = 0
            return self._contact_force, self._contact_stiffness

        vertices = self._ipc_vertices(q, rt)
        self._ipc_collisions.build(
            self._ipc_collision_mesh,
            vertices,
            self._ipc_dhat,
            broad_phase=self._ipc_broad_phase,
        )

        if len(self._ipc_collisions) == 0:
            self._contact_force[:] = 0
            self._contact_stiffness[:] = 0
            return self._contact_force, self._contact_stiffness

        from fedoo.constraint.ipc_contact import _barrier_hessian_psd

        J = self._build_ipc_jacobian(rt, q[3:])

        grad = self._ipc_barrier.gradient(
            self._ipc_collisions, self._ipc_collision_mesh, vertices
        )
        self._contact_force = -self._ipc_kappa * (J.T @ grad)

        if compute in ("all", "stiffness"):
            hess = _barrier_hessian_psd(
                self._ipc_barrier,
                self._ipc_collisions,
                self._ipc_collision_mesh,
                vertices,
            )
            self._contact_stiffness = self._ipc_kappa * (J.T @ (hess @ J))
        else:
            self._contact_stiffness[:] = 0

        return self._contact_force, self._contact_stiffness

    @property
    def time_dof_indices(self):
        return self._dof_indices

    def _get_rotation_inertia(self, pb=None):
        """Return trial orientation, world inertia, and its DOF derivatives."""
        pb = pb or self._pb_ref
        rotvec = np.zeros(3)
        if pb is not None and self._dof_indices is not None:
            dof_ref, _ = self.rigid_tie._get_dof_ref(pb)
            rotvec = np.asarray(dof_ref[3:], dtype=float)

        R, *dR = self.rigid_tie._compute_rotation(rotvec)
        inertia = R @ self.inertia_body @ R.T
        derivatives = tuple(
            derivative @ self.inertia_body @ R.T + R @ self.inertia_body @ derivative.T
            for derivative in dR
        )
        return R, inertia, derivatives

    def _get_mass_matrix(self, pb=None):
        """Return the rigid-body mass matrix in the current global frame."""
        M = np.zeros((_N_RIGID_DOF, _N_RIGID_DOF))
        M[:3, :3] = self.mass * np.eye(3)
        _, M[3:, 3:], _ = self._get_rotation_inertia(pb)
        return M

    def get_time_inertia_force(self, pb, acceleration, velocity):
        """Return translational inertia and Euler's rotational inertia force."""
        acceleration = np.asarray(acceleration, dtype=float)
        velocity = np.asarray(velocity, dtype=float)
        force = np.zeros(_N_RIGID_DOF)
        force[:3] = self.mass * acceleration[:3]

        _, inertia, _ = self._get_rotation_inertia(pb)
        angular_acceleration = acceleration[3:]
        angular_velocity = velocity[3:]
        force[3:] = inertia @ angular_acceleration
        if self.rigid_tie.use_quaternion:
            force[3:] += np.cross(angular_velocity, inertia @ angular_velocity)
        return force

    def get_time_inertia_tangent(
        self,
        pb,
        acceleration,
        velocity,
        acceleration_factor,
        velocity_factor,
    ):
        """Return the effective tangent of the rigid-body inertia force."""
        acceleration = np.asarray(acceleration, dtype=float)
        velocity = np.asarray(velocity, dtype=float)
        tangent = np.zeros((_N_RIGID_DOF, _N_RIGID_DOF))
        tangent[:3, :3] = acceleration_factor * self.mass * np.eye(3)

        _, inertia, inertia_derivatives = self._get_rotation_inertia(pb)
        if not self.rigid_tie.use_quaternion:
            tangent[3:, 3:] = acceleration_factor * inertia
            return tangent

        angular_acceleration = acceleration[3:]
        angular_velocity = velocity[3:]
        angular_momentum = inertia @ angular_velocity
        basis = np.eye(3)
        for axis, inertia_derivative in enumerate(inertia_derivatives):
            d_acceleration = acceleration_factor * basis[axis]
            d_velocity = velocity_factor * basis[axis]
            tangent[3:, 3 + axis] = (
                inertia_derivative @ angular_acceleration
                + inertia @ d_acceleration
                + np.cross(d_velocity, angular_momentum)
                + np.cross(
                    angular_velocity,
                    inertia_derivative @ angular_velocity + inertia @ d_velocity,
                )
            )
        return tangent

    def get_storage_matrix(self, pb=None):
        """Assembly-level storage provider used by fedoo.time."""
        return self._get_mass_matrix(pb)

    def get_time_initial_force(self, pb=None):
        return self.force + self._contact_force

    def get_time_stiffness_matrix(self, pb=None):
        """Return the static/contact tangent before time integration."""
        return self._contact_stiffness

    def _get_n_dof(self):
        if self._pb_ref is not None:
            return self._pb_ref.n_dof
        return max(
            self.mesh.n_nodes * self.space.nvar,
            int(self._dof_indices.max()) + 1,
        )

    def assemble_global_mat(self, compute="all"):
        """Assemble static/contact terms, then apply the attached integrator."""
        if self._dof_indices is None:
            return

        n = self._get_n_dof()
        idx = self._dof_indices
        regularisation = (
            self.static_regularisation if self._time_integrator is None else 0.0
        )

        if compute in ("all", "matrix"):
            stiffness = self._contact_stiffness + regularisation * np.eye(_N_RIGID_DOF)
            self.global_matrix = scatter_dense_block(stiffness, idx, (n, n))
        if compute in ("all", "vector"):
            self.global_vector = np.zeros(n)
            self.global_vector[idx] = self.force + self._contact_force

        if self._time_integrator is not None:
            self._time_integrator.integrate(self, self._pb_ref, compute)

    def update(self, pb, compute="all"):
        self._pb_ref = pb
        if self._time_integrator is not None:
            self._time_integrator.update(self, pb)

        dof_solution = pb.get_dof_solution()
        if self._ipc_collision_mesh is not None:
            q = (
                np.asarray(dof_solution)[self._dof_indices]
                if not np.isscalar(dof_solution)
                else np.zeros(_N_RIGID_DOF)
            )
            self.compute_contact(q, self.rigid_tie, compute="all")

        self.assemble_global_mat(compute)

    def set_start(self, pb):
        self._pb_ref = pb
        if self._time_integrator is not None:
            self._time_integrator.set_start(self, pb)

    def to_start(self, pb):
        if self._time_integrator is not None:
            self._time_integrator.to_start(self, pb)


class RigidBody:
    """Rigid body for large-displacement dynamic simulation in Fedoo.

    Coordinates a :class:`RigidTie` constraint (kinematics: MPC coupling
    of surface nodes to 6 rigid DOFs) with a :class:`RigidBodyAssembly`
    (dynamics: 6x6 mass matrix and generalized force vector).

    Supports large translations and multiplicative incremental rotations.
    IPC barrier contact is handled via direct Jacobian
    projection ``J^T @ F`` in the 6-DOF space.

    Solve via ``body.solve(dt, tmax)`` (wraps ``NonLinear``) or include
    ``body.assembly`` in an ``Assembly.sum`` — the kinematic tie is then
    registered automatically when the problem is constructed. Attach a
    second-order integrator to manually constructed dynamic problems with
    ``pb.set_time_integrator(fd.time.SECOND_ORDER, fd.time.Newmark())``.

    Parameters
    ----------
    mesh : fedoo.Mesh
        Mesh of the rigid body. For a standalone body, this is its own
        mesh (tri3 surface for IPC contact, or volume for inertia
        integration). For a body that is one part of a larger stacked
        problem mesh (rigid-vs-deformable IPC), pass the output of
        :meth:`Mesh.extract_elements` — the carried ``parent_node_indices``
        attribute tells RigidTie which DOFs to slave in the parent.
    mass : float, optional
        Total mass. Required if ``density`` is not given.
    density : float, optional
        Material density. Computes mass and inertia from mesh.
    inertia_tensor : array_like, shape (3, 3), optional
        Inertia tensor about center of mass in body frame.
    center_of_mass : array_like, shape (3,), optional
        Center of mass. Default: mean of mesh nodes.
    use_quaternion : bool, optional
        If ``True`` (default), compose incremental rotation vectors into a
        quaternion orientation and include Euler's gyroscopic inertia term.
        If ``False``, use one total rotation vector; this mode is primarily
        retained for kinematic comparisons.
    name : str, optional
        Name of the rigid body.

    Example
    -------
    .. code-block:: python

        import fedoo as fd
        import numpy as np

        fd.ModelingSpace("3D")
        sphere_mesh = fd.Mesh.from_pyvista(pv.Sphere(radius=0.1))

        body = fd.constraint.RigidBody(sphere_mesh, mass=1.0,
                                        inertia_tensor=0.004*np.eye(3))
        body.set_force([0, 0, -9.81])
        body.set_rayleigh_damping(1.0)
        body.set_static_obstacle(plane_mesh, dhat=0.01, kappa=1e8)

        # solve() returns the underlying NonLinear problem; read the rigid
        # DOFs from it via the assembly's _dof_indices.
        pb = body.solve(dt=5e-4, tmax=2.0)
        q = pb.get_dof_solution()[body.assembly._dof_indices]
    """

    def __init__(
        self,
        mesh,
        mass=None,
        density=None,
        inertia_tensor=None,
        center_of_mass=None,
        use_quaternion=True,
        dynamic=True,
        name="RigidBody",
    ):
        self.mesh = mesh
        self.name = name
        self.dynamic = bool(dynamic)

        if center_of_mass is None:
            # Use only nodes actually referenced by elements — protects
            # the default against ``extract_elements`` submeshes that
            # carry the parent's full node array.
            active = np.unique(mesh.elements.ravel())
            center_of_mass = mesh.nodes[active].mean(axis=0)
        center_of_mass = np.asarray(center_of_mass, dtype=float)

        if self.dynamic:
            # Mass and inertia are consumed by the problem's second-order
            # time integrator. Static mode allows callers to omit them.
            if density is not None:
                vol = mesh.get_volume()
                if mass is None:
                    mass = density * vol
                if inertia_tensor is None:
                    inertia_tensor = self._compute_inertia(
                        mesh, density, center_of_mass
                    )
            else:
                if mass is None:
                    raise ValueError("Either mass or density must be provided.")
                if inertia_tensor is None:
                    raise ValueError(
                        "inertia_tensor required when density is not given."
                    )
        else:
            mass = 0.0 if mass is None else mass
            inertia_tensor = (
                np.zeros((3, 3)) if inertia_tensor is None else inertia_tensor
            )

        self.mass = float(mass)
        self.center_of_mass = center_of_mass
        self.inertia_tensor = np.asarray(inertia_tensor, dtype=float)
        if self.inertia_tensor.shape != (3, 3):
            raise ValueError("inertia_tensor must have shape (3, 3).")
        if not np.allclose(self.inertia_tensor, self.inertia_tensor.T):
            raise ValueError("inertia_tensor must be symmetric.")
        self.use_quaternion = bool(use_quaternion)

        # When ``mesh`` came from ``Mesh.extract_elements`` it carries
        # the parent-mesh active-node list used to slave the right DOFs
        # in a stacked rigid-plus-deformable problem. A standalone
        # body's mesh has no such attribute — every node is tied.
        tie_node_indices = getattr(mesh, "parent_node_indices", None)
        if tie_node_indices is None:
            tie_node_indices = np.arange(mesh.n_nodes)

        self.constraint = RigidTie(
            tie_node_indices,
            center=center_of_mass,
            use_quaternion=self.use_quaternion,
            name=f"{name}_tie",
        )
        self.assembly = RigidBodyAssembly(
            self.mass,
            self.inertia_tensor,
            self.constraint,
            mesh=mesh,
            dynamic=self.dynamic,
            name=f"{name}_asm",
        )

    def set_force(self, force):
        """Set external force [Fx, Fy, Fz] on translational DOFs."""
        self.assembly.force[:3] = force

    def set_torque(self, torque):
        """Set external torque [Mx, My, Mz] on rotational DOFs."""
        self.assembly.force[3:] = torque

    def set_generalized_force(self, f):
        """Set full generalized force [Fx, Fy, Fz, Mx, My, Mz]."""
        self.assembly.force[:] = f

    def set_rayleigh_damping(self, alpha=0.0, beta=0.0):
        """Set Rayleigh damping: ``C = alpha*M + beta*K``.

        A rigid body has no elastic stiffness. With the private contact made
        by :meth:`set_static_obstacle`, ``K`` is that contact tangent. With a
        shared :class:`IPCContact`, the contact tangent belongs to the IPC
        assembly instead and the rigid-body ``beta`` contribution is zero.
        """
        self.assembly.dissipation = RayleighDamping(
            alpha=float(alpha), beta=float(beta)
        )

    def set_static_obstacle(self, obstacle_mesh, dhat=0.01, kappa=None):
        """Enable IPC barrier contact with a STATIC obstacle surface.

        Builds a private collision mesh and barrier on this rigid body's
        :class:`RigidBodyAssembly`. The obstacle is snapshotted once at
        setup and treated as frozen geometry — use this for rigid-vs-static
        scenarios (e.g., a rigid body falling onto a fixed floor).

        For rigid-vs-deformable contact (e.g., a punch crushing an elastic
        disc), build a shared :class:`IPCContact` over the union of all
        surfaces instead. Calling ``ipc.add_rigid_body(body)`` optionally
        projects contact directly onto the rigid DOFs; otherwise the same
        projection is performed by ``RigidTie`` MPC condensation. In both
        cases the deformable obstacle's vertex positions are tracked at every
        NR iteration and its reaction is assembled on the mesh DOFs.

        Parameters
        ----------
        obstacle_mesh : fedoo.Mesh
            Surface mesh of the obstacle (tri3), treated as static.
        dhat : float
            Barrier activation distance (meters).
        kappa : float or None
            Barrier stiffness. If None (default), automatically tuned at
            first contact to balance external forces and barrier gradient.
        """
        from fedoo.constraint.ipc_contact import _import_ipctk

        ipctk = _import_ipctk()

        body_nodes = self.mesh.nodes
        obst_nodes = obstacle_mesh.nodes
        n_body = len(body_nodes)

        if self.mesh.elements.shape[1] != 3:
            raise ValueError("Body mesh must be tri3 for IPC contact.")
        if obstacle_mesh.elements.shape[1] != 3:
            raise ValueError("Obstacle mesh must be tri3 for IPC contact.")

        all_nodes = np.vstack([body_nodes, obst_nodes])
        all_elms = np.vstack([self.mesh.elements, obstacle_mesh.elements + n_body])

        edges = ipctk.edges(all_elms)
        cm = ipctk.CollisionMesh(all_nodes, edges, all_elms)

        # Only body-obstacle collisions (no self-contact within either mesh):
        # the filter blocks same-patch pairs, so body (0) and obstacle (1) only.
        patches = np.zeros(len(all_nodes), dtype=np.int32)
        patches[n_body:] = 1
        cm.can_collide = ipctk.make_vertex_patches_filter(patches)

        asm = self.assembly
        asm._ipc_collision_mesh = cm
        asm._ipc_collisions = ipctk.NormalCollisions()
        # Fedoo applies the contact stiffness separately through _ipc_kappa.
        asm._ipc_barrier = ipctk.BarrierPotential(dhat, 1.0)
        if kappa is None:
            kappa = 1e9
        asm._ipc_kappa = kappa
        asm._ipc_dhat = dhat
        asm._ipc_broad_phase = ipctk.LBVH()
        asm._ipc_rest_positions = body_nodes.copy()
        asm._ipc_n_body = n_body
        asm._ipc_obstacle_nodes = obst_nodes.copy()
        # Keep the same frozen geometry for optional result export. The source
        # identity lets shared obstacles be deduplicated across rigid bodies.
        asm._ipc_obstacle_mesh = Mesh(
            obst_nodes.copy(),
            obstacle_mesh.elements.copy(),
            obstacle_mesh.elm_type,
            node_sets=obstacle_mesh.node_sets,
            element_sets=obstacle_mesh.element_sets,
            ndim=obstacle_mesh.ndim,
            name=obstacle_mesh.name,
            register_name=False,
        )
        asm._ipc_obstacle_source_id = id(obstacle_mesh)

    def add_to_problem(self, pb):
        """Register the rigid body's kinematic tie with a Fedoo problem.

        Usually unnecessary — :class:`NonLinear` auto-registers ties from
        any :class:`RigidBodyAssembly` it discovers in its assembly sum.
        Kept as an explicit hook for advanced flows (custom problem
        classes, late attachment) and made idempotent so calling it
        after auto-registration is a no-op.
        """
        if self.constraint not in pb.bc:
            pb.bc.add(self.constraint)

    def solve(self, dt, tmax, t0=0, print_info=1, solver=None):
        """Solve rigid body dynamics using Fedoo's NonLinear solver.

        Creates an internal ``NonLinear`` problem, attaches
        :class:`fedoo.time.Newmark` for a dynamic body, and calls ``nlsolve``.
        IPC contact and Rayleigh damping remain assembly-level providers and
        are integrated by the problem's time integrator.

        Parameters
        ----------
        dt : float
            Time step.
        tmax : float
            End time.
        t0 : float
            Start time.
        print_info : int
            Verbosity (0=silent, 1=iterations).
        solver : str or callable, optional
            Linear solver forwarded to :meth:`Problem.set_solver`.

        Returns
        -------
        pb : NonLinear
            The solved problem (access DOFs via ``pb.get_dof_solution()``).
        """
        from fedoo.problem.non_linear import NonLinear

        pb = NonLinear(self.assembly)
        if self.dynamic:
            from fedoo.time import Newmark

            pb.set_time_integrator(SECOND_ORDER, Newmark())
        if solver is not None:
            pb.set_solver(solver)
        pb.nlsolve(dt=dt, tmax=tmax, t0=t0, print_info=print_info, update_dt=False)
        return pb

    @staticmethod
    def _compute_inertia(mesh, density, center_of_mass):
        """Compute inertia tensor from a closed triangulated surface mesh.

        Uses the divergence theorem (Mirtich 1996).
        """
        nodes = mesh.nodes - center_of_mass
        elements = mesh.elements

        if elements.shape[1] != 3:
            raise ValueError(
                "Automatic inertia computation requires tri3 elements. "
                f"Got elements with {elements.shape[1]} nodes."
            )

        v0 = nodes[elements[:, 0]]
        v1 = nodes[elements[:, 1]]
        v2 = nodes[elements[:, 2]]

        normals = np.cross(v1 - v0, v2 - v0)

        x2 = (
            v0[:, 0] ** 2
            + v1[:, 0] ** 2
            + v2[:, 0] ** 2
            + v0[:, 0] * v1[:, 0]
            + v0[:, 0] * v2[:, 0]
            + v1[:, 0] * v2[:, 0]
        )
        y2 = (
            v0[:, 1] ** 2
            + v1[:, 1] ** 2
            + v2[:, 1] ** 2
            + v0[:, 1] * v1[:, 1]
            + v0[:, 1] * v2[:, 1]
            + v1[:, 1] * v2[:, 1]
        )
        z2 = (
            v0[:, 2] ** 2
            + v1[:, 2] ** 2
            + v2[:, 2] ** 2
            + v0[:, 2] * v1[:, 2]
            + v0[:, 2] * v2[:, 2]
            + v1[:, 2] * v2[:, 2]
        )

        xy = (
            2 * (v0[:, 0] * v0[:, 1] + v1[:, 0] * v1[:, 1] + v2[:, 0] * v2[:, 1])
            + v0[:, 0] * v1[:, 1]
            + v0[:, 1] * v1[:, 0]
            + v0[:, 0] * v2[:, 1]
            + v0[:, 1] * v2[:, 0]
            + v1[:, 0] * v2[:, 1]
            + v1[:, 1] * v2[:, 0]
        )
        xz = (
            2 * (v0[:, 0] * v0[:, 2] + v1[:, 0] * v1[:, 2] + v2[:, 0] * v2[:, 2])
            + v0[:, 0] * v1[:, 2]
            + v0[:, 2] * v1[:, 0]
            + v0[:, 0] * v2[:, 2]
            + v0[:, 2] * v2[:, 0]
            + v1[:, 0] * v2[:, 2]
            + v1[:, 2] * v2[:, 0]
        )
        yz = (
            2 * (v0[:, 1] * v0[:, 2] + v1[:, 1] * v1[:, 2] + v2[:, 1] * v2[:, 2])
            + v0[:, 1] * v1[:, 2]
            + v0[:, 2] * v1[:, 1]
            + v0[:, 1] * v2[:, 2]
            + v0[:, 2] * v2[:, 1]
            + v1[:, 1] * v2[:, 2]
            + v1[:, 2] * v2[:, 1]
        )

        Ixx = np.sum(normals[:, 0] * x2) * density / 60.0
        Iyy = np.sum(normals[:, 1] * y2) * density / 60.0
        Izz = np.sum(normals[:, 2] * z2) * density / 60.0
        Ixy = np.sum(normals[:, 0] * xy) * density / 120.0
        Ixz = np.sum(normals[:, 0] * xz) * density / 120.0
        Iyz = np.sum(normals[:, 1] * yz) * density / 120.0

        return np.array(
            [
                [Iyy + Izz, -Ixy, -Ixz],
                [-Ixy, Ixx + Izz, -Iyz],
                [-Ixz, -Iyz, Ixx + Iyy],
            ]
        )
