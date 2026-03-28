"""Rigid body support for large-displacement dynamic simulation.

Provides:
- ``RigidBodyAssembly``: AssemblyBase that contributes a 6x6 mass matrix and
  a generalized force vector on the rigid body's global DOFs.
- ``RigidBody``: Facade that coordinates a RigidTie constraint (kinematics)
  with a RigidBodyAssembly (dynamics) into a single user-facing object.
  Optionally handles IPC contact via direct Jacobian projection (J^T @ F).

Large rotations
---------------
Rotations are handled exactly (no small-angle approximation) via a
multiplicative quaternion update (using ``simcoon.Rotation``):

- The total rotation is stored as a quaternion ``Q_base`` in the RigidTie.
- The rotation DOFs ``[rx, ry, rz]`` represent a small Euler increment
  from the current base state.
- At each converged time step, the increment is composed into the
  quaternion: ``Q_base = R_inc * Q_base`` (quaternion multiplication).
- This avoids gimbal lock and supports arbitrarily large rotations.

The contact Jacobian ``J = [I_3 | dR/d(angle) @ r_ref]`` uses the exact
rotation derivatives from ``RigidTie._compute_rotation()``, not the
infinitesimal skew-symmetric approximation.
"""

import numpy as np
from scipy import sparse
from fedoo.core.base import AssemblyBase
from fedoo.constraint.rigid_tie import RigidTie

try:
    from simcoon import Rotation
except ImportError:
    from scipy.spatial.transform import Rotation


class RigidBodyAssembly(AssemblyBase):
    """Assembly contributing rigid body mass and generalized forces.

    Injects a 6x6 mass matrix and a 6-component force vector at the global
    DOF positions created by a :class:`RigidTie` constraint.

    The mass matrix has the structure::

        M = [[m*I_3,   0   ],
             [  0,   J_global]]

    where ``J_global = Q.apply_tensor(J_body)`` is the inertia tensor rotated
    to the current configuration via the RigidTie's quaternion.

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
        name="RigidBodyAssembly",
    ):
        if space is None:
            from fedoo.core.modelingspace import ModelingSpace

            space = ModelingSpace.get_active()
        AssemblyBase.__init__(self, name, space)
        self.mass = float(mass)
        self.inertia_body = np.asarray(inertia_tensor, dtype=float)
        self.rigid_tie = rigid_tie
        self.force = np.zeros(6)
        self.mesh = mesh

        self._dof_indices = None
        self._pb_ref = None

        # IPC contact state (set by RigidBody.enable_ipc_contact)
        self._ipc_collision_mesh = None
        self._ipc_collisions = None
        self._ipc_barrier = None
        self._ipc_kappa = None
        self._ipc_kappa_auto = False
        self._ipc_dhat = None
        self._ipc_broad_phase = None
        self._ipc_rest_positions = None
        self._ipc_n_body = 0
        self._ipc_obstacle_nodes = None
        self._contact_force = np.zeros(6)
        self._contact_stiffness = np.zeros((6, 6))

    def initialize(self, pb):
        """Extract global DOF indices from the problem."""
        rt = self.rigid_tie
        self._dof_indices = np.array(
            [
                pb.n_node_dof
                + pb._global_dof.indice_start(rt.var_cd[i])
                + rt.node_cd[i]
                for i in range(6)
            ]
        )
        self._pb_ref = pb
        if self.mesh is None:
            self.mesh = pb.mesh

    @property
    def dof_indices(self):
        """Global DOF indices [Fx,Fy,Fz,Mx,My,Mz] in the problem."""
        return self._dof_indices

    # ------------------------------------------------------------------
    # IPC contact: Jacobian and force/stiffness computation
    # ------------------------------------------------------------------

    def _build_ipc_jacobian(self, rt, angles):
        """Build contact Jacobian J mapping 6 rigid DOFs to all vertex DOFs.

        Uses exact rotation derivatives from RigidTie for consistency.

        Returns
        -------
        J : ndarray, shape (n_all*3, 6)
            Body vertices use exact derivatives; obstacle rows are zero.
        """
        R, dR_drx, dR_dry, dR_drz = rt._compute_rotation(angles)
        r_ref = self._ipc_rest_positions - rt.center
        du_drx = r_ref @ dR_drx.T
        du_dry = r_ref @ dR_dry.T
        du_drz = r_ref @ dR_drz.T

        n_body = self._ipc_n_body
        n_all = n_body + len(self._ipc_obstacle_nodes)
        J = np.zeros((n_all * 3, 6))
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

        from fedoo.constraint.ipc_contact import _import_ipctk

        ipctk = _import_ipctk()

        J = self._build_ipc_jacobian(rt, q[3:])

        # Gradient → force
        grad = self._ipc_barrier.gradient(
            self._ipc_collisions, self._ipc_collision_mesh, vertices
        )
        self._contact_force = -self._ipc_kappa * (J.T @ grad)

        # Hessian → stiffness (skipped in line-search for efficiency)
        if compute in ("all", "stiffness"):
            try:
                hess = self._ipc_barrier.hessian(
                    self._ipc_collisions,
                    self._ipc_collision_mesh,
                    vertices,
                    project_hessian_to_psd=ipctk.PSDProjectionMethod.CLAMP,
                )
            except RuntimeError:
                hess = self._ipc_barrier.hessian(
                    self._ipc_collisions,
                    self._ipc_collision_mesh,
                    vertices,
                    project_hessian_to_psd=ipctk.PSDProjectionMethod.NONE,
                )
            self._contact_stiffness = self._ipc_kappa * (J.T @ (hess @ J))
        else:
            self._contact_stiffness[:] = 0

        return self._contact_force, self._contact_stiffness

    # ------------------------------------------------------------------
    # Fedoo assembly interface (for use with NonLinearNewmark)
    # ------------------------------------------------------------------

    def assemble_global_mat(self, compute="all"):
        if self._dof_indices is None:
            return

        n = (
            self._pb_ref.n_dof
            if self._pb_ref is not None
            else (
                max(
                    self.mesh.n_nodes * self.space.nvar,
                    int(self._dof_indices.max()) + 1,
                )
            )
        )
        idx = self._dof_indices

        if compute in ("all", "matrix"):
            M = np.zeros((6, 6))
            M[0, 0] = M[1, 1] = M[2, 2] = self.mass
            Q = self.rigid_tie.Q_total
            M[3:, 3:] = (
                Q.apply_tensor(self.inertia_body)
                if Q is not None
                else self.inertia_body
            )

            self.global_matrix = sparse.csr_matrix(
                (M.ravel(), (np.repeat(idx, 6), np.tile(idx, 6))),
                shape=(n, n),
            )

        if compute in ("all", "vector"):
            vec = np.zeros(n)
            vec[idx] = self.force + self._contact_force
            self.global_vector = vec

    def update(self, pb, compute="all"):
        self._pb_ref = pb
        self.assemble_global_mat(compute)

    def set_start(self, pb):
        self._pb_ref = pb

    def to_start(self, pb):
        pass


class RigidBody:
    """Rigid body for large-displacement dynamic simulation in Fedoo.

    Coordinates a :class:`RigidTie` constraint (kinematics: MPC coupling
    of surface nodes to 6 rigid DOFs) with a :class:`RigidBodyAssembly`
    (dynamics: 6x6 mass matrix and generalized force vector).

    Supports arbitrarily large translations and rotations via quaternion-
    based multiplicative update (no gimbal lock, no small-angle
    approximation). IPC barrier contact is handled via direct Jacobian
    projection ``J^T @ F`` in the 6-DOF space.

    Two solver modes:
    - ``body.solve(dt, tmax)`` — direct 6x6 Newmark integrator (fast,
      standalone, no Fedoo Problem needed).
    - ``body.add_to_problem(pb)`` — integrate with Fedoo's
      NonLinearNewmark for mixed rigid-deformable problems.

    Parameters
    ----------
    mesh : fedoo.Mesh
        Surface mesh of the rigid body (tri3 for IPC contact).
    mass : float, optional
        Total mass. Required if ``density`` is not given.
    density : float, optional
        Material density. Computes mass and inertia from mesh.
    inertia_tensor : array_like, shape (3, 3), optional
        Inertia tensor about center of mass in body frame.
    center_of_mass : array_like, shape (3,), optional
        Center of mass. Default: mean of mesh nodes.
    use_quaternion : bool, optional
        Quaternion-based rotation in RigidTie (default True).
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
        body.enable_ipc_contact(plane_mesh, dhat=0.01, kappa=1e8)

        q, v, a = body.solve(dt=5e-4, tmax=2.0)
    """

    def __init__(
        self,
        mesh,
        mass=None,
        density=None,
        inertia_tensor=None,
        center_of_mass=None,
        use_quaternion=True,
        name="RigidBody",
    ):
        self.mesh = mesh
        self.name = name

        if center_of_mass is None:
            center_of_mass = mesh.nodes.mean(axis=0)
        center_of_mass = np.asarray(center_of_mass, dtype=float)

        if density is not None:
            vol = mesh.get_volume()
            if mass is None:
                mass = density * vol
            if inertia_tensor is None:
                inertia_tensor = self._compute_inertia(mesh, density, center_of_mass)
        else:
            if mass is None:
                raise ValueError("Either mass or density must be provided.")
            if inertia_tensor is None:
                raise ValueError("inertia_tensor required when density is not given.")

        self.mass = float(mass)
        self.center_of_mass = center_of_mass
        self.inertia_tensor = np.asarray(inertia_tensor, dtype=float)

        self.constraint = RigidTie(
            np.arange(mesh.n_nodes),
            center=center_of_mass,
            use_quaternion=use_quaternion,
            name=f"{name}_tie",
        )
        self.assembly = RigidBodyAssembly(
            self.mass,
            self.inertia_tensor,
            self.constraint,
            mesh=mesh,
            name=f"{name}_asm",
        )
        self._rayleigh_alpha = 0.0

    # ------------------------------------------------------------------
    # Forces and damping
    # ------------------------------------------------------------------

    def set_force(self, force):
        """Set external force [Fx, Fy, Fz] on translational DOFs."""
        self.assembly.force[:3] = force

    def set_torque(self, torque):
        """Set external torque [Mx, My, Mz] on rotational DOFs."""
        self.assembly.force[3:] = torque

    def set_generalized_force(self, f):
        """Set full generalized force [Fx, Fy, Fz, Mx, My, Mz]."""
        self.assembly.force[:] = f

    def set_rayleigh_damping(self, alpha):
        """Set mass-proportional damping: C = alpha * M."""
        self._rayleigh_alpha = float(alpha)

    # ------------------------------------------------------------------
    # IPC contact
    # ------------------------------------------------------------------

    def enable_ipc_contact(self, obstacle_mesh, dhat=0.01, kappa=None):
        """Enable IPC barrier contact with an obstacle surface.

        Parameters
        ----------
        obstacle_mesh : fedoo.Mesh
            Surface mesh of the obstacle (tri3).
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

        # Only body↔obstacle collisions (no self-contact within rigid body)
        patches = np.zeros(len(all_nodes), dtype=np.int32)
        patches[n_body:] = 1
        cm.can_collide = ipctk.VertexPatchesCanCollide(patches)

        asm = self.assembly
        asm._ipc_collision_mesh = cm
        asm._ipc_collisions = ipctk.NormalCollisions()
        asm._ipc_barrier = ipctk.BarrierPotential(dhat)
        if kappa is None:
            kappa = 1e9
        asm._ipc_kappa = kappa
        asm._ipc_kappa_auto = False
        asm._ipc_dhat = dhat
        asm._ipc_broad_phase = ipctk.HashGrid()
        asm._ipc_rest_positions = body_nodes.copy()
        asm._ipc_n_body = n_body
        asm._ipc_obstacle_nodes = obst_nodes.copy()

    # ------------------------------------------------------------------
    # Fedoo integration
    # ------------------------------------------------------------------

    def add_to_problem(self, pb):
        """Register the rigid body constraint with a Fedoo problem."""
        pb.bc.add(self.constraint)

    @property
    def Q_total(self):
        """Current total rotation as a Rotation object."""
        return self.constraint.Q_total

    # ------------------------------------------------------------------
    # Direct 6x6 Newmark solver
    # ------------------------------------------------------------------

    @staticmethod
    def _ccd_filter(asm, rt, q_current, dq_step):
        """Limit dq_step so the trajectory is collision-free (Tight-Inclusion CCD).

        Computes the maximal alpha in [0, 1] such that the linear motion
        from q_current to q_current + alpha*dq_step produces no collision,
        then returns alpha * dq_step.
        """
        from fedoo.constraint.ipc_contact import _import_ipctk

        ipctk = _import_ipctk()

        v0 = asm._ipc_vertices(q_current, rt)
        v1 = asm._ipc_vertices(q_current + dq_step, rt)

        alpha = ipctk.compute_collision_free_stepsize(
            asm._ipc_collision_mesh,
            v0,
            v1,
            broad_phase=asm._ipc_broad_phase,
        )

        if alpha < 1.0:
            # Leave a small margin (0.9 factor) to stay inside barrier zone
            return 0.9 * alpha * dq_step
        return dq_step

    def solve(self, dt, tmax, t0=0, callback=None, print_info=True):
        """Solve rigid body dynamics with a direct 6x6 Newmark integrator.

        Bypasses the Fedoo FEM solver — solves directly in the 6-DOF
        rigid body space with IPC contact and Rayleigh damping.

        Parameters
        ----------
        dt : float
            Time step.
        tmax : float
            End time.
        t0 : float
            Start time.
        callback : callable, optional
            Called at each step: ``callback(t, q, v)``.
        print_info : bool
            Print progress.

        Returns
        -------
        q : ndarray (6,)
            Final DOFs [dx, dy, dz, rx, ry, rz].
        v : ndarray (6,)
            Final velocity.
        a : ndarray (6,)
            Final acceleration.
        """
        n_steps = int(round((tmax - t0) / dt))
        q = np.zeros(6)
        v = np.zeros(6)
        a = np.zeros(6)
        beta, gamma = 0.25, 0.5
        rt = self.constraint
        asm = self.assembly

        # Ensure quaternion state is initialized
        if not hasattr(rt, "_Q_base") or rt._Q_base is None:
            rt._Q_base = Rotation.identity()
            rt._Q_base_backup = Rotation.identity()
            rt._angles_at_base = np.zeros(3)

        nr_tol = 1e-10 * max(np.linalg.norm(asm.force), 1.0)

        for step in range(n_steps):
            t = t0 + (step + 1) * dt

            # Mass matrix
            M = np.zeros((6, 6))
            M[0, 0] = M[1, 1] = M[2, 2] = self.mass
            Q = rt._Q_base
            M[3:, 3:] = (
                Q.apply_tensor(self.inertia_tensor)
                if Q is not None
                else self.inertia_tensor
            )

            # Damping
            C = (
                self._rayleigh_alpha * M
                if self._rayleigh_alpha > 0
                else np.zeros((6, 6))
            )

            # Initial prediction
            F_c, K_c = asm.compute_contact(q, rt, compute="all")
            A_eff = K_c + (1 / (beta * dt**2)) * M + (gamma / (beta * dt)) * C
            v_pred = (1 - gamma / beta) * v + dt * (1 - gamma / (2 * beta)) * a
            rhs = (
                asm.force
                + F_c
                + M @ (-(1 / (beta * dt)) * v - (0.5 / beta - 1) * a)
                + C @ (-v_pred)
            )

            try:
                dq = np.linalg.solve(A_eff, rhs)
            except np.linalg.LinAlgError:
                if print_info:
                    print(f"  Singular at t={t:.4f}s")
                break

            # CCD: limit step to collision-free trajectory
            if asm._ipc_collision_mesh is not None:
                dq = self._ccd_filter(asm, rt, q, dq)

            # Newton-Raphson with line search
            for nr_iter in range(20):
                F_c, K_c = asm.compute_contact(q + dq, rt, compute="all")
                new_a = (
                    (1 / (beta * dt**2)) * dq
                    - (1 / (beta * dt)) * v
                    - (0.5 / beta - 1) * a
                )
                new_v = v + dt * ((1 - gamma) * a + gamma * new_a)
                res = asm.force + F_c - M @ new_a - C @ new_v
                res_norm = np.linalg.norm(res)

                if res_norm < nr_tol:
                    break

                A_eff = K_c + (1 / (beta * dt**2)) * M + (gamma / (beta * dt)) * C
                try:
                    ddq = np.linalg.solve(A_eff, res)
                except np.linalg.LinAlgError:
                    break

                # CCD-filtered line search
                ddq_safe = self._ccd_filter(asm, rt, q + dq, ddq)
                alpha_ls = 1.0
                for _ in range(8):
                    dq_test = dq + alpha_ls * ddq_safe
                    F_test, _ = asm.compute_contact(q + dq_test, rt, compute="force")
                    a_test = (
                        (1 / (beta * dt**2)) * dq_test
                        - (1 / (beta * dt)) * v
                        - (0.5 / beta - 1) * a
                    )
                    v_test = v + dt * ((1 - gamma) * a + gamma * a_test)
                    if (
                        np.linalg.norm(asm.force + F_test - M @ a_test - C @ v_test)
                        < res_norm
                    ):
                        break
                    alpha_ls *= 0.5
                dq = dq + alpha_ls * ddq_safe

            # Update state
            new_a = (
                (1 / (beta * dt**2)) * dq - (1 / (beta * dt)) * v - (0.5 / beta - 1) * a
            )
            v = v + dt * ((1 - gamma) * a + gamma * new_a)
            a = new_a
            q = q + dq

            # Absorb rotation into quaternion base
            delta = q[3:] - rt._angles_at_base
            if np.any(np.abs(delta) > 1e-14):
                rt._Q_base = Rotation.from_euler("XYZ", delta) * rt._Q_base
                rt._angles_at_base = q[3:].copy()

            # Adaptive barrier stiffness via ipctk heuristics
            if asm._ipc_collision_mesh is not None and len(asm._ipc_collisions) > 0:
                min_dist = asm._ipc_collisions.compute_minimum_distance(
                    asm._ipc_collision_mesh,
                    asm._ipc_vertices(q, rt),
                )
                if not hasattr(asm, "_prev_min_dist"):
                    # First contact: initialize kappa
                    from fedoo.constraint.ipc_contact import _import_ipctk

                    _ipctk = _import_ipctk()
                    bbox_diag = np.linalg.norm(
                        asm._ipc_collision_mesh.rest_positions.max(axis=0)
                        - asm._ipc_collision_mesh.rest_positions.min(axis=0)
                    )
                    grad_barrier = asm._ipc_barrier.gradient(
                        asm._ipc_collisions,
                        asm._ipc_collision_mesh,
                        asm._ipc_vertices(q, rt),
                    )
                    J = asm._build_ipc_jacobian(rt, q[3:])
                    grad_b_proj = J.T @ grad_barrier
                    kappa_init, kappa_max = _ipctk.initial_barrier_stiffness(
                        bbox_diag,
                        asm._ipc_barrier.barrier,
                        asm._ipc_dhat,
                        self.mass,
                        asm.force,
                        grad_b_proj,
                    )
                    asm._ipc_kappa = max(asm._ipc_kappa, kappa_init)
                    asm._ipc_kappa_max = kappa_max
                    asm._bbox_diag = bbox_diag
                    asm._prev_min_dist = min_dist
                else:
                    from fedoo.constraint.ipc_contact import _import_ipctk

                    _ipctk = _import_ipctk()
                    asm._ipc_kappa = _ipctk.update_barrier_stiffness(
                        asm._prev_min_dist,
                        min_dist,
                        asm._ipc_kappa_max,
                        asm._ipc_kappa,
                        asm._bbox_diag,
                    )
                    asm._prev_min_dist = min_dist

            if callback is not None:
                callback(t, q, v)

            if print_info and step % max(1, n_steps // 20) == 0:
                n_c = len(asm._ipc_collisions) if asm._ipc_collisions is not None else 0
                print(
                    f"    t={t:.3f}s  z={self.center_of_mass[2]+q[2]:.4f}m  contacts={n_c}"
                )

        return q, v, a

    # ------------------------------------------------------------------
    # Inertia computation
    # ------------------------------------------------------------------

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
