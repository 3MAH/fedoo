from __future__ import annotations

import numpy as np
from scipy import sparse
from fedoo.core.base import AssemblyBase
from fedoo.core.modelingspace import ModelingSpace


def _import_ipctk():
    """Lazy import of ipctk with a clear error message."""
    try:
        import ipctk
    except ImportError:
        raise ImportError(
            "The 'ipctk' package is required for IPC contact. "
            "Install it with: pip install ipctk\n"
            "Or install fedoo with IPC support: pip install fedoo[ipc]"
        )

    version = getattr(ipctk, "__version__", "0")
    try:
        major_minor = tuple(int(part) for part in version.split(".")[:2])
    except (AttributeError, ValueError):
        major_minor = (0, 0)
    if major_minor < (1, 6):
        raise ImportError(
            f"Fedoo IPC requires ipctk >= 1.6, but the loaded runtime is "
            f"ipctk {version}. Restart Python after upgrading ipctk; mixing "
            "Fedoo's 1.6 API calls with an already-loaded ipctk 1.5 changes "
            "BarrierPotential argument semantics and produces invalid contact "
            "forces."
        )
    return ipctk


class IPCContact(AssemblyBase):
    r"""Contact Assembly based on the IPC (Incremental Potential Contact) method.

    Uses the `ipctk <https://ipctk.xyz/>`_ library to guarantee
    intersection-free configurations through barrier potentials and optional
    CCD (Continuous Collision Detection) line search.

    The IPC method adds a barrier energy term :math:`\kappa\,B(\mathbf{x})`
    to the total potential energy.  The barrier :math:`B` grows to infinity
    as the distance between surfaces approaches zero and vanishes when the
    distance exceeds ``dhat``.  This guarantees that the deformed
    configuration remains intersection-free at every Newton–Raphson
    iteration.

    **Barrier stiffness auto-tuning** — When ``barrier_stiffness`` is left
    to ``None`` (recommended), the stiffness :math:`\kappa` is automatically
    initialised by ``ipctk.initial_barrier_stiffness`` so that it balances
    the elastic and barrier energy gradients.  If no contact exists at the
    initial configuration (common for self-contact problems), :math:`\kappa`
    is re-initialised the first time collisions appear, using the actual
    elastic forces at that stage.  Afterwards, :math:`\kappa` is updated
    adaptively at each converged time increment via
    ``ipctk.update_barrier_stiffness``.

    **Convergence safeguards (SDI)** — When the set of active collisions
    changes during a Newton–Raphson iteration (new contacts appear or
    existing ones separate), or when the minimum gap drops below
    0.1 × d_hat, the solver is forced to perform at least one additional
    NR iteration even if the displacement criterion is already satisfied.
    When surfaces are dangerously close (gap < 0.01 × d_hat), at least
    three extra iterations are forced.  This prevents premature
    convergence when the contact state is still evolving.

    **CCD line search** — When ``use_ccd=True``, the Newton–Raphson step
    size is clamped to the largest value that keeps all surfaces
    intersection-free.  For the elastic prediction of an increment where
    no contacts existed at the start, a conservative minimum distance of
    0.1 × d_hat is enforced to prevent surfaces from jumping deep into
    the barrier zone.  If this conservative bound yields a zero step
    (e.g. due to shared mesh edges at zero distance in self-contact),
    the CCD falls back to the standard ``min_distance=0`` computation.

    **OGC trust-region** — When ``use_ogc=True``, the Offset Geometric
    Contact (OGC) trust-region method from ``ipctk.ogc`` replaces CCD.
    Instead of uniformly scaling the displacement step by a single
    scalar, OGC filters the displacement per vertex, clamping each
    vertex's move to lie within a trust region that guarantees no
    intersection.  This can provide better convergence by not
    penalising distant vertices for a tight contact elsewhere.
    ``use_ogc`` and ``use_ccd`` are mutually exclusive.

    .. warning::
       OGC is only supported for **shell / surface meshes** where every
       node lies on the contact surface.  Solid meshes with interior
       nodes must use ``use_ccd=True`` instead.  An error is raised at
       initialization if ``use_ogc=True`` is used with a solid mesh.

    **Axisymmetric (2Daxi)** — Under a ``"2Daxi"`` ModelingSpace the
    barrier gradient and hessian returned by ipctk are scaled by the
    circumferential integration weight ``2*pi*r`` per surface vertex
    before being scattered into the global system.  This is the IPC
    analogue of fedoo's per-Gauss-point ``axi_volume_weight`` for weak
    forms.  The scaling is applied per vertex (using the reference
    radius ``R₀``); it is exact when both endpoints of a contact pair
    sit at similar radii, and slightly asymmetric otherwise.  Vertices
    on the symmetry axis (r = 0) get weight 0, which is geometrically
    correct (a "ring" of zero radius collapses to a point).  In the
    current implementation, the axisymmetric residual uses a single
    integration weight ``W = diag(2*pi*r)`` while the tangent uses the
    two-sided product ``W @ H @ W``.  This intentionally increases the
    tangent stiffness for 2Daxi IPC as a numerical stabilization; a
    strictly single-weighted tangent should be reconsidered together
    with a consistently weighted IPC energy and line search.  In 2Daxi
    the planar (r, z) distance equals the true 3D minimum distance
    between the corresponding circles (closest points share the
    azimuth), so the barrier distance and PSD projection are unchanged.

    Parameters
    ----------
    mesh : fedoo.Mesh
        Full volumetric mesh (needed for DOF compatibility).
    surface_mesh : fedoo.Mesh, optional
        Pre-extracted surface mesh (``lin2`` in 2D, ``tri3`` in 3D).
    extract_surface : bool, default=False
        If ``True``, automatically extract the surface from the volumetric
        mesh.
    dhat : float, default=1e-3
        Barrier activation distance.  Pairs of primitives closer than
        ``dhat`` receive a repulsive barrier force.  A smaller value gives
        a tighter contact (less visible gap) but requires finer time steps.
    dhat_is_relative : bool, default=True
        If ``True``, ``dhat`` is interpreted as a fraction of the bounding
        box diagonal of the surface mesh.
    barrier_stiffness : float, optional
        Barrier stiffness :math:`\kappa`.  If ``None`` (default), it is
        automatically computed and adaptively updated — **this is the
        recommended setting**.  When specified explicitly, the value is
        kept fixed unless ``adaptive_barrier_stiffness`` is ``True``, in
        which case it is used as the initial value and may be updated.
    friction_coefficient : float, default=0.0
        Coulomb friction coefficient :math:`\mu`.  Set to ``0`` for
        frictionless contact.
    eps_v : float, default=1e-3
        Friction smoothing velocity (only relevant when
        ``friction_coefficient > 0``).
    broad_phase : str, default="lbvh"
        Broad-phase collision detection method.  One of ``"lbvh"``,
        ``"hash_grid"``, ``"brute_force"`` or ``"spatial_hash"``.
    adaptive_barrier_stiffness : bool, default=True
        If ``True``, :math:`\kappa` is updated adaptively at each converged
        time increment using ``ipctk.update_barrier_stiffness``.
    use_ccd : bool, optional
        Enable Continuous Collision Detection (CCD) line search. When active,
        the Newton–Raphson step size is limited to prevent interpenetration
        between iterations. Default is True unless ``use_ogc`` is enabled.
        Note: ``use_ccd`` and ``use_ogc`` are mutually exclusive.
    line_search_energy : bool or None, default=None
        Controls energy-based backtracking after CCD line search.  The
        step is halved until total energy (exact barrier + quadratic
        elastic approximation) decreases.  When ``None`` (default),
        automatically enabled when ``use_ccd=True`` — this matches the
        reference IPC algorithm.  Set to ``False`` to disable (faster
        but may degrade convergence).  Has no effect when ``use_ogc``
        is enabled.
    use_ogc : bool, default=False
        Enable OGC (Offset Geometric Contact) trust-region step
        filtering.  Per-vertex displacement clamping replaces the
        uniform scalar step of CCD.  Mutually exclusive with
        ``use_ccd``.  **Only supported for shell / surface meshes**
        where all nodes are on the surface; raises ``ValueError``
        for solid meshes with interior nodes.
    space : ModelingSpace, optional
        Modeling space.  If ``None``, the active ``ModelingSpace`` is used.
    name : str, default="IPC Contact"
        Name of the contact assembly.

    Notes
    -----
    The IPC assembly is used by adding it to the structural assembly with
    :py:meth:`fedoo.Assembly.sum`:

    .. code-block:: python

        solid = fd.Assembly.create(wf, mesh)
        ipc   = fd.constraint.IPCContact(mesh, surface_mesh=surf)
        assembly = fd.Assembly.sum(solid, ipc)

    When using ``add_output``, pass the *solid* assembly (not the sum)
    to avoid errors.

    See Also
    --------
    IPCSelfContact : Convenience wrapper for self-contact problems.
    fedoo.constraint.Contact : Penalty-based contact method.
    """

    def __init__(
        self,
        mesh,
        surface_mesh=None,
        extract_surface=False,
        dhat=1e-3,
        dhat_is_relative=True,
        barrier_stiffness=None,
        friction_coefficient=0.0,
        eps_v=1e-3,
        broad_phase="lbvh",
        adaptive_barrier_stiffness=True,
        use_ccd=None,
        line_search_energy=None,
        use_ogc=False,
        space=None,
        name="IPC Contact",
    ):
        if use_ccd is None:
            use_ccd = not use_ogc
        if use_ccd and use_ogc:
            raise ValueError(
                "use_ccd and use_ogc are mutually exclusive. "
                "Choose one line-search / step-filtering strategy."
            )

        if space is None:
            space = ModelingSpace.get_active()
        AssemblyBase.__init__(self, name, space)

        self.mesh = mesh
        self._surface_mesh = surface_mesh
        self._extract_surface = extract_surface
        self._dhat = dhat
        self._dhat_is_relative = dhat_is_relative
        self._barrier_stiffness_init = barrier_stiffness
        self.friction_coefficient = friction_coefficient
        self._eps_v = eps_v
        self._broad_phase_str = broad_phase
        self._adaptive_barrier_stiffness = adaptive_barrier_stiffness
        self._use_ccd = use_ccd
        # Auto-enable energy backtracking when CCD is active (reference IPC
        # algorithm always uses energy-based line search after CCD filtering)
        if line_search_energy is None:
            self._line_search_energy = use_ccd
        else:
            self._line_search_energy = line_search_energy
        self._use_ogc = use_ogc

        self.current = self

        # Will be initialized in initialize()
        self._collision_mesh = None
        self._barrier_potential = None
        self._friction_potential = None
        self._collisions = None
        self._friction_collisions = None
        self._kappa = barrier_stiffness
        self._max_kappa = None
        self._max_kappa_set = False
        self._actual_dhat = None
        self._rest_positions = None
        self._prev_min_distance = None
        self._surface_node_indices = None
        self._scatter_matrix = None  # maps ipctk surface DOFs to fedoo full DOFs
        self._broad_phase = None

        self._n_global_dof = 0  # extra DOFs from e.g. PeriodicBC
        self._last_vertices = None  # cached for assemble_global_mat
        self._pb = None  # reference to problem (for accessing elastic gradient)
        self._n_collisions_at_start = 0  # collision count at start of increment
        self._kappa_boosted_this_increment = False  # prevent repeated within-NR boosts
        self._bbox_diag = None  # bounding box diagonal (cached)

        # OGC trust-region state
        self._ogc_trust_region = None
        self._ogc_vertices_at_start = None

        # Axisymmetric (2Daxi) integration weights — built in initialize()
        self._axi_weight = None  # 1D array (n_surf*ndim,) interleaved 2*pi*r
        self._axi_diag = None  # sparse diag(_axi_weight) for hessian scaling

        self.sv = {}
        self.sv_start = {}

        # Rigid body support: list of (node_indices, rigid_body) tuples
        self._rigid_bodies = []
        self._rigid_node_set = None  # set of rigid node indices (built in initialize)

    def add_rigid_body(self, rigid_body, surface_node_indices=None):
        """Register a :class:`RigidBody` with this IPCContact.

        This optional convenience enables direct reduced-coordinate projection
        for rigid-vs-deformable (or rigid-vs-rigid) contact where a single
        ``IPCContact`` carries the full collision mesh. The body's surface
        nodes are routed onto the 6 rigid-body global DOFs by the scatter
        matrix via the exact rigid Jacobian
        ``J = [I_3 | dR/d(omega) @ r_ref]``, instead of placing forces at
        the per-node mesh DOFs.

        Registration is not required for correctness when the rigid surface
        nodes already belong to the contact mesh and a :class:`RigidTie` is
        attached to the problem. In that generic path, IPC assembles nodal
        contact contributions and the problem's MPC condensation projects
        them onto the rigid DOFs. Registration additionally associates the
        body with the shared mesh and lets automatic barrier-stiffness
        estimation account for generalized rigid-body forces directly.

        For rigid-vs-static contact (rigid body bouncing on a fixed floor),
        use :meth:`RigidBody.set_static_obstacle` instead — that path uses
        a private 6x6 fast route with no shared mesh.

        Parameters
        ----------
        rigid_body : RigidBody
            The rigid body. When the body's nodes live inside this
            IPCContact's mesh (stacked rigid-plus-deformable case),
            construct it from a :meth:`Mesh.extract_elements` of the
            shared mesh so the carried ``parent_node_indices`` wires
            RigidTie to the correct parent DOFs.
            ``rigid_body.constraint.list_nodes`` is read as the default
            surface index set.
        surface_node_indices : array_like, optional
            Override for ``rigid_body.constraint.list_nodes``. Rarely
            needed; useful only when the rigid body's surface is a strict
            subset of its tie node set.
        """
        if rigid_body.assembly._ipc_collision_mesh is not None:
            raise RuntimeError(
                f"{rigid_body.name}: cannot use set_static_obstacle() and "
                "IPCContact.add_rigid_body() on the same body — choose "
                "either the private rigid-vs-static path or the shared "
                "IPCContact."
            )
        if surface_node_indices is None:
            surface_node_indices = np.asarray(
                rigid_body.constraint.list_nodes, dtype=int
            )
        else:
            surface_node_indices = np.asarray(surface_node_indices, dtype=int)
        # Point the body's RigidBodyAssembly at the shared problem mesh so
        # ``Assembly.sum`` accepts it (mass / inertia / Newmark stay 6x6 and
        # are scattered at the 6 global DOFs regardless of mesh size).
        # This rebinds ``rigid_body.assembly.mesh`` as a side effect; guard
        # against silently re-pointing a body already registered on a
        # *different* IPCContact mesh (re-registering on the same mesh is OK).
        prev_mesh = getattr(rigid_body, "_ipc_registered_mesh", None)
        if prev_mesh is not None and prev_mesh is not self.mesh:
            raise RuntimeError(
                f"{rigid_body.name}: already registered on another IPCContact "
                "mesh; create a fresh RigidBody per shared-IPC problem."
            )
        rigid_body.assembly.mesh = self.mesh
        rigid_body._ipc_registered_mesh = self.mesh
        self._rigid_bodies.append((surface_node_indices, rigid_body))

    def _create_broad_phase(self):
        """Create an ipctk BroadPhase instance from the string name."""
        ipctk = _import_ipctk()
        mapping = {
            "lbvh": ipctk.LBVH,
            "hash_grid": ipctk.HashGrid,
            "brute_force": ipctk.BruteForce,
            "spatial_hash": ipctk.SpatialHash,
        }
        cls = mapping.get(self._broad_phase_str.lower())
        if cls is None:
            raise ValueError(
                f"Unknown broad phase method '{self._broad_phase_str}'. "
                f"Choose from: {list(mapping.keys())}"
            )
        return cls()

    def _build_scatter_matrix(self, surface_node_indices, n_nodes, ndim, pb=None):
        """Build sparse matrix mapping ipctk surface DOFs to fedoo full DOFs.

        ipctk uses interleaved layout for surface nodes:
            [x0, y0, z0, x1, y1, z1, ...]
        fedoo uses blocked layout for all nodes:
            [ux_0..ux_N, uy_0..uy_N, uz_0..uz_N, global_DOFs...]

        For deformable nodes, the mapping is a permutation (identity values).
        For rigid body nodes, the mapping uses the exact contact Jacobian
        ``J = [I_3 | dR/d(omega) @ r_ref]`` derived from the RigidTie
        rotation derivatives, projecting surface DOFs directly onto the 6
        rigid body global DOFs. Only ``r_ref`` (reference positions)
        enters the Jacobian — current positions are not needed.

        Parameters
        ----------
        surface_node_indices : array
            Global indices of surface nodes in the full mesh.
        n_nodes : int
            Total number of nodes in the full mesh.
        ndim : int
            Number of spatial dimensions (2 or 3).
        pb : Problem, optional
            Problem instance (needed for rigid body global DOF indices).

        Returns
        -------
        P : sparse csc_matrix
            Shape (n_dof, n_surf * ndim) where n_dof includes global DOFs.
        """
        disp_ranks = np.array(self.space.get_rank_vector("Disp"))
        n_surf = len(surface_node_indices)
        nvar = self.space.nvar
        n_global_dof = pb.n_global_dof if pb is not None else self._n_global_dof
        n_dof = nvar * n_nodes + n_global_dof

        # Collect sparse entries
        all_rows = []
        all_cols = []
        all_data = []

        # Identify which surface nodes are rigid
        if self._rigid_node_set is not None and len(self._rigid_node_set) > 0:
            is_rigid = np.isin(surface_node_indices, list(self._rigid_node_set))
        else:
            is_rigid = np.zeros(n_surf, dtype=bool)
        deformable_mask = ~is_rigid

        # --- Deformable nodes: identity mapping ---
        deform_local = np.where(deformable_mask)[0]
        if len(deform_local) > 0:
            deform_global = surface_node_indices[deform_local]
            for d in range(ndim):
                all_rows.append(disp_ranks[d] * n_nodes + deform_global)
                all_cols.append(deform_local * ndim + d)
                all_data.append(np.ones(len(deform_local)))

        # --- Rigid body nodes: contact Jacobian ---
        for node_set, rb in self._rigid_bodies:
            rt = rb.constraint
            dof_indices = rb.assembly._dof_indices  # (6,) global DOF positions

            # Find which surface indices belong to this rigid body
            mask = np.isin(surface_node_indices, node_set)
            local_indices = np.where(mask)[0]
            if len(local_indices) == 0:
                continue

            n_rb = len(local_indices)

            # Get exact rotation derivatives from RigidTie. The Jacobian
            # J = [I_3 | dR/d(omega) @ r_ref] uses only r_ref (reference
            # positions) — current vertex positions and translation are
            # not needed.
            if pb is not None:
                dof_ref, _ = rt._get_dof_ref(pb)
                angles = dof_ref[3:]
            else:
                angles = np.zeros(3)

            _, dR_drx, dR_dry, dR_drz = rt._compute_rotation(angles)
            center = rt.center  # initial center position

            # Translation part: J_i[:3, :3] = I_3
            for d in range(ndim):
                all_rows.append(np.full(n_rb, dof_indices[d]))
                all_cols.append(local_indices * ndim + d)
                all_data.append(np.ones(n_rb))

            # Rotation part: exact derivatives from RigidTie
            # J_i[d, 3+k] = (dR_drk @ r_ref_i)[d]
            # where r_ref = vertex_ref - center (in reference config)
            r_ref = self._rest_positions[local_indices] - center  # (n_rb, 3)
            du_drx = r_ref @ dR_drx.T  # (n_rb, 3)
            du_dry = r_ref @ dR_dry.T
            du_drz = r_ref @ dR_drz.T

            for d in range(ndim):
                # RotX contribution
                nonzero = np.abs(du_drx[:, d]) > 1e-15
                if np.any(nonzero):
                    all_rows.append(np.full(nonzero.sum(), dof_indices[3]))
                    all_cols.append(local_indices[nonzero] * ndim + d)
                    all_data.append(du_drx[nonzero, d])
                # RotY contribution
                nonzero = np.abs(du_dry[:, d]) > 1e-15
                if np.any(nonzero):
                    all_rows.append(np.full(nonzero.sum(), dof_indices[4]))
                    all_cols.append(local_indices[nonzero] * ndim + d)
                    all_data.append(du_dry[nonzero, d])
                # RotZ contribution
                nonzero = np.abs(du_drz[:, d]) > 1e-15
                if np.any(nonzero):
                    all_rows.append(np.full(nonzero.sum(), dof_indices[5]))
                    all_cols.append(local_indices[nonzero] * ndim + d)
                    all_data.append(du_drz[nonzero, d])

        rows = np.concatenate(all_rows) if all_rows else np.array([], dtype=int)
        cols = np.concatenate(all_cols) if all_cols else np.array([], dtype=int)
        data = np.concatenate(all_data) if all_data else np.array([])

        P = sparse.csc_matrix(
            (data, (rows, cols)),
            shape=(n_dof, n_surf * ndim),
        )
        return P

    def _get_current_vertices(self, pb):
        """Extract current surface vertex positions from problem displacement.

        Returns
        -------
        vertices : ndarray, shape (n_surf_verts, ndim)
            Current positions of surface vertices.
        """
        disp = pb.get_disp()  # shape (ndim, n_nodes)

        if np.isscalar(disp) and disp == 0:
            return self._rest_positions.copy()

        # disp has shape (ndim, n_nodes)
        surf_disp = disp[:, self._surface_node_indices].T  # (n_surf, ndim)
        return self._rest_positions + surf_disp

    def _get_elastic_gradient_on_surface(self):
        """Extract elastic energy gradient projected onto surface DOFs.

        Uses the global vector from the parent assembly (excluding IPC)
        to get the elastic force contribution, then projects it to
        ipctk's interleaved surface DOF layout via P^T.
        """
        if self._pb is None:
            return None

        # Get the global vector from the parent assembly
        # The parent assembly is the AssemblySum containing this IPC assembly
        # We need the elastic part only (without IPC contribution)
        assembly = self._pb.assembly
        if not hasattr(assembly, "_list_assembly"):
            return None

        # Sum global vectors from all assemblies except this IPC one.
        # RigidBodyAssembly safely returns zeros at init time (its
        # assemble_global_mat guards against dt=0 internally), so we
        # include it like any other assembly — its contribution at the
        # 6 rigid DOFs is exactly the external/contact force balance,
        # which is what kappa auto-tune wants to match against the
        # barrier gradient.
        grad_energy_full = None
        for a in assembly._list_assembly:
            if a is self:
                continue
            gv = a.current.get_global_vector()
            if gv is not None and not (np.isscalar(gv) and gv == 0):
                if grad_energy_full is None:
                    grad_energy_full = np.array(gv, dtype=float)
                else:
                    grad_energy_full += gv

        if grad_energy_full is None:
            return None

        # When rigid bodies are present, P includes global DOF rows,
        # so keep the full vector. Otherwise truncate to mesh DOFs.
        if not self._rigid_bodies:
            n_mesh_dof = self.space.nvar * self.mesh.n_nodes
            grad_energy_full = grad_energy_full[:n_mesh_dof]

        # Ensure dimension matches P rows
        n_P_rows = self._scatter_matrix.shape[0]
        if len(grad_energy_full) < n_P_rows:
            grad_energy_full = np.pad(
                grad_energy_full, (0, n_P_rows - len(grad_energy_full))
            )

        # Project to surface DOFs: P^T @ full_grad -> surface_grad
        return self._scatter_matrix.T @ grad_energy_full

    def _initialize_kappa(self, vertices):
        """Compute barrier stiffness from gradient balance.

        Called each time step from ``set_start`` (when auto-kappa is
        enabled and contacts exist) and from ``update`` (when contacts
        first appear during an increment that started contactless).

        The MAX guard ensures kappa never decreases: only a larger
        value from gradient balance is accepted.
        """
        ipctk = _import_ipctk()

        bbox_diag = self._bbox_diag

        # Barrier gradient on surface DOFs
        n_surf_dof = len(self._surface_node_indices) * self.space.ndim
        if len(self._collisions) > 0:
            grad_barrier = self._barrier_potential.gradient(
                self._collisions, self._collision_mesh, vertices
            )
        else:
            grad_barrier = np.zeros(n_surf_dof)

        # Elastic gradient projected to surface DOFs
        grad_energy = self._get_elastic_gradient_on_surface()
        if grad_energy is None:
            grad_energy = np.zeros(n_surf_dof)

        new_kappa, new_max_kappa = ipctk.initial_barrier_stiffness(
            bbox_diag,
            self._barrier_potential.barrier,
            self._actual_dhat,
            1.0,  # average_mass (1.0 for quasi-static)
            grad_energy,
            grad_barrier,
        )
        # MAX guard: kappa can only increase to prevent oscillations
        if self._kappa is None or new_kappa > self._kappa:
            self._kappa = new_kappa
        # max_kappa is only updated until _max_kappa_set is True
        # (standard IPC: set once, then frozen as the adaptive cap)
        if not self._max_kappa_set:
            if self._max_kappa is None or new_max_kappa > self._max_kappa:
                self._max_kappa = new_max_kappa
        self.sv["kappa"] = self._kappa
        self.sv["max_kappa"] = self._max_kappa
        self.sv["prev_min_distance"] = self._prev_min_distance

    def _compute_ipc_contributions(self, vertices, compute="all"):
        """Compute barrier (and friction) gradient and hessian.

        ipctk returns gradient/hessian in surface interleaved DOF space.
        The scatter matrix P maps them to fedoo's full blocked DOF space:
            global_vector = P @ ipctk_gradient
            global_matrix = P @ ipctk_hessian @ P.T

        Under a 2Daxi ModelingSpace, the per-surface-DOF gradient/hessian
        are first scaled by ``2*pi*r`` (per surface vertex, broadcast over
        coordinates) so that the planar barrier energy is integrated
        around the symmetry axis — the IPC analogue of fedoo's
        ``axi_volume_weight`` for weak forms.

        Sets self.global_matrix and self.global_vector.
        """
        n_mesh_dof = self.space.nvar * self.mesh.n_nodes
        n_dof = n_mesh_dof + self._n_global_dof
        if len(self._collisions) == 0:
            self.global_matrix = sparse.csr_array((n_dof, n_dof))
            self.global_vector = np.zeros(n_dof)
            return

        ipctk = _import_ipctk()
        P = self._scatter_matrix
        axi_w = self._axi_weight  # None outside 2Daxi
        axi_D = self._axi_diag  # None outside 2Daxi

        # Barrier contributions
        if compute != "matrix":
            grad_surf = self._barrier_potential.gradient(
                self._collisions, self._collision_mesh, vertices
            )
            if axi_w is not None:
                grad_surf = axi_w * grad_surf
            # ipctk gradient points toward increasing barrier (toward contact).
            # The repulsive force is -gradient. In fedoo, global_vector is
            # added to RHS: K*dX = B + D, so D = -kappa * gradient.
            self.global_vector = -self._kappa * (P @ grad_surf)

        if compute != "vector":
            try:
                hess_surf = self._barrier_potential.hessian(
                    self._collisions,
                    self._collision_mesh,
                    vertices,
                    project_hessian_to_psd=ipctk.PSDProjectionMethod.CLAMP,
                )
            except RuntimeError:
                # CLAMP/ABS projection can fail on degenerate contact
                # pairs (zero-distance); skip projection entirely.
                hess_surf = self._barrier_potential.hessian(
                    self._collisions,
                    self._collision_mesh,
                    vertices,
                    project_hessian_to_psd=ipctk.PSDProjectionMethod.NONE,
                )
            if axi_D is not None:
                # Axisymmetric IPC uses a single 2*pi*r weight on the residual.
                # The physically closer single-weight tangent would be:
                # hess_surf = 0.5 * (axi_D @ hess_surf + hess_surf @ axi_D)
                # The two-sided product below preserves symmetry/PSD and
                # intentionally increases the contact tangent stiffness for
                # numerical stabilization of the current post-weighted
                # axisymmetric IPC formulation.
                hess_surf = axi_D @ hess_surf @ axi_D
            self.global_matrix = self._kappa * (P @ hess_surf @ P.T)

        # Friction contributions
        if self.friction_coefficient > 0 and self._friction_collisions is not None:
            if len(self._friction_collisions) > 0:
                # Friction gradient/hessian from ipctk already include
                # the effect of kappa and mu through the TangentialCollisions
                # that were built with normal_stiffness and mu.
                if compute != "matrix":
                    fric_grad_surf = self._friction_potential.gradient(
                        self._friction_collisions,
                        self._collision_mesh,
                        vertices,
                    )
                    if axi_w is not None:
                        fric_grad_surf = axi_w * fric_grad_surf
                    self.global_vector += -(P @ fric_grad_surf)

                if compute != "vector":
                    fric_hess_surf = self._friction_potential.hessian(
                        self._friction_collisions,
                        self._collision_mesh,
                        vertices,
                        project_hessian_to_psd=ipctk.PSDProjectionMethod.CLAMP,
                    )
                    if axi_D is not None:
                        # Single-weight tangent to revisit with a fully
                        # weighted axisymmetric IPC energy:
                        # fric_hess_surf = 0.5 * (
                        #     axi_D @ fric_hess_surf + fric_hess_surf @ axi_D
                        # )
                        fric_hess_surf = axi_D @ fric_hess_surf @ axi_D
                    self.global_matrix += P @ fric_hess_surf @ P.T

        # P already has n_dof rows (mesh DOFs + global DOFs) — see
        # _build_scatter_matrix — so P @ hess @ P.T and P @ grad are already
        # the right size. No padding is needed for PeriodicBC, RigidTie, or
        # registered RigidBody global DOFs.

    def _ccd_line_search(self, pb, dX):
        """Compute step size using CCD + energy-based backtracking.

        **Phase 1 — CCD**: limits alpha to the largest collision-free
        step.

        **Phase 2 — Energy backtracking**: starting from the CCD alpha,
        halves the step until the total energy decreases.  Barrier
        energy is evaluated exactly from ipctk; elastic energy change
        is approximated via a quadratic model (Lt * deps at the global
        level) using the already-computed elastic residual and tangent
        stiffness — no ``umat()`` call needed.

        Parameters
        ----------
        pb : Problem
            The current problem.
        dX : ndarray
            The displacement increment (full DOF vector).

        Returns
        -------
        alpha : float
            Step size in (0, 1] that is collision-free and decreases
            total energy.
        """
        ipctk = _import_ipctk()
        ndim = self.space.ndim

        # Current vertices
        vertices_current = self._get_current_vertices(pb)

        # Surface displacement from dX (fedoo blocked layout). For
        # rigid-tied slave nodes, ``_MatCB`` has already written the
        # kinematic motion back into the mesh-DOF rows (see
        # ``problem.Problem.solve``: ``X = MatCB @ X_free + Xbc``), so we
        # can read the surface displacement directly regardless of
        # whether a node is deformable or rigid-slaved.
        disp_ranks = np.array(self.space.get_rank_vector("Disp"))
        n_nodes = self.mesh.n_nodes
        n_surf = len(self._surface_node_indices)
        surf_disp = np.zeros((n_surf, ndim))
        for d in range(ndim):
            surf_disp[:, d] = dX[disp_ranks[d] * n_nodes + self._surface_node_indices]

        vertices_next = vertices_current + surf_disp

        # --- Phase 1: CCD (collision-free step size) ---
        # When no contacts at step start, use conservative min_distance
        # to prevent jumping deep into the barrier zone.
        # When contacts already exist, use min_distance=0 to avoid
        # ipctk "initial distance <= d_min" warnings and double CCD.
        if self._n_collisions_at_start == 0:
            min_distance = 0.1 * self._actual_dhat
        else:
            min_distance = 0.0

        alpha = ipctk.compute_collision_free_stepsize(
            self._collision_mesh,
            vertices_current,
            vertices_next,
            min_distance=min_distance,
            broad_phase=self._broad_phase,
        )

        # Fallback: if min_distance caused alpha=0 (due to pre-existing
        # zero-distance pairs from shared mesh edges), recompute with
        # standard CCD (min_distance=0).
        if alpha <= 0 and min_distance > 0:
            alpha = ipctk.compute_collision_free_stepsize(
                self._collision_mesh,
                vertices_current,
                vertices_next,
                min_distance=0.0,
                broad_phase=self._broad_phase,
            )

        if alpha < 1.0:
            alpha *= 0.9

        if alpha <= 0:
            return alpha

        # --- Phase 2: Energy-based backtracking (opt-in) ---
        if not self._line_search_energy:
            return alpha

        # Skip during elastic prediction: the elastic residual D_e is
        # near zero (previous step converged) and doesn't account for
        # external work from the BC increment, so the quadratic model
        # would incorrectly predict energy increase.  CCD alone
        # suffices for elastic prediction safety.
        if not (pb._boundary_is_0):
            return alpha

        # Skip if no contacts exist and none expected at trial position
        if len(self._collisions) == 0 and self._n_collisions_at_start == 0:
            return alpha

        # Barrier energy at current position
        E_barrier_current = self._kappa * self._barrier_potential(
            self._collisions, self._collision_mesh, vertices_current
        )

        # Quadratic elastic energy approximation:
        #   ΔΠ_elastic ≈ α*c1 + 0.5*α²*c2
        # where c1 = -D_elastic·dX (gradient·step),
        #       c2 = dX^T · K_elastic · dX (curvature)
        # Uses already-computed elastic residual and tangent from the
        # current NR iteration — no umat() needed at trial points.
        c1, c2 = 0.0, 0.0
        parent = self.associated_assembly_sum
        if parent is not None:
            for asm in parent._list_assembly:
                if asm is not self:
                    D_e = asm.current.get_global_vector()
                    K_e = asm.current.get_global_matrix()
                    if D_e is not None and K_e is not None:
                        c1 = -np.dot(D_e, dX)
                        c2 = np.dot(dX, K_e @ dX)
                    break

        for _ls_iter in range(12):
            vertices_trial = vertices_current + alpha * surf_disp

            # Rebuild collisions at trial position
            trial_collisions = ipctk.NormalCollisions()
            trial_collisions.build(
                self._collision_mesh,
                vertices_trial,
                self._actual_dhat,
                broad_phase=self._broad_phase,
            )
            E_barrier_trial = self._kappa * self._barrier_potential(
                trial_collisions, self._collision_mesh, vertices_trial
            )

            dE_elastic = alpha * c1 + 0.5 * alpha**2 * c2
            dE_total = dE_elastic + (E_barrier_trial - E_barrier_current)

            if dE_total <= 0:
                break
            alpha *= 0.5

        return alpha

    # ------------------------------------------------------------------
    # OGC trust-region methods
    # ------------------------------------------------------------------

    def _ogc_warm_start(self, pb, dX):
        """Filter the elastic prediction step using OGC warm-start.

        Converts the solver's displacement increment *dX* to surface
        vertex positions, calls ``warm_start_time_step`` which moves
        vertices toward the predicted position while respecting trust
        regions, then writes the filtered displacement back into *dX*.
        """
        ipctk = _import_ipctk()
        ndim = self.space.ndim
        disp_ranks = np.array(self.space.get_rank_vector("Disp"))
        n_nodes = self.mesh.n_nodes
        n_surf = len(self._surface_node_indices)

        # Surface displacement extracted from dX (fedoo blocked layout)
        surf_disp = np.zeros((n_surf, ndim))
        for d in range(ndim):
            surf_disp[:, d] = dX[disp_ranks[d] * n_nodes + self._surface_node_indices]

        # Predicted position = start-of-increment position + displacement
        # ipctk requires F-contiguous arrays; x_start must also be writeable
        x_start = np.asfortranarray(self._ogc_vertices_at_start.copy())
        pred_x = np.asfortranarray(self._ogc_vertices_at_start + surf_disp)

        # warm_start_time_step modifies x_start in-place toward pred_x
        self._ogc_trust_region.warm_start_time_step(
            self._collision_mesh,
            x_start,
            pred_x,
            self._collisions,
            self._actual_dhat,
            broad_phase=self._broad_phase,
        )

        # Per-vertex OGC filtering (all nodes are on the surface)
        filtered_disp = x_start - self._ogc_vertices_at_start
        for d in range(ndim):
            dX[disp_ranks[d] * n_nodes + self._surface_node_indices] = filtered_disp[
                :, d
            ]

    def _ogc_filter_step(self, pb, dX):
        """Filter a Newton–Raphson displacement step using OGC.

        Extracts the surface displacement from *dX*, calls
        ``filter_step`` which clamps per-vertex moves in-place,
        then writes the filtered values back into *dX*.
        """
        ipctk = _import_ipctk()
        ndim = self.space.ndim
        disp_ranks = np.array(self.space.get_rank_vector("Disp"))
        n_nodes = self.mesh.n_nodes
        n_surf = len(self._surface_node_indices)

        # Current surface vertex positions (F-contiguous for ipctk)
        vertices_current = np.asfortranarray(self._get_current_vertices(pb))

        # Surface displacement from dX (F-contiguous + writeable for ipctk)
        surf_dx = np.zeros((n_surf, ndim), order="F")
        for d in range(ndim):
            surf_dx[:, d] = dX[disp_ranks[d] * n_nodes + self._surface_node_indices]

        # Per-vertex OGC filtering (all nodes are on the surface)
        self._ogc_trust_region.filter_step(
            self._collision_mesh, vertices_current, surf_dx
        )
        for d in range(ndim):
            dX[disp_ranks[d] * n_nodes + self._surface_node_indices] = surf_dx[:, d]

    def _ogc_step_filter_callback(self, pb, dX, is_elastic_prediction=False):
        """Dispatch to warm-start or NR filter depending on context."""
        if is_elastic_prediction:
            self._ogc_warm_start(pb, dX)
        else:
            self._ogc_filter_step(pb, dX)

    def assemble_global_mat(self, compute="all"):
        """Recompute barrier contributions from cached vertex positions.

        Called by the solver when the tangent matrix needs reassembly
        without a full ``update()`` cycle.  Uses vertex positions
        cached by the most recent ``update()`` or ``set_start()`` call.

        When rigid bodies are registered the scatter matrix depends on the
        current rigid rotation; the MPC coefficients have just been
        refreshed by ``update_boundary_conditions`` for this NR step, so
        refresh the scatter here too to keep the contact tangent
        consistent with the kinematic constraint.
        """
        if self._last_vertices is not None:
            if self._rigid_bodies and self._pb is not None:
                self._scatter_matrix = self._build_scatter_matrix(
                    self._surface_node_indices,
                    self.mesh.n_nodes,
                    self.space.ndim,
                    self._pb,
                )
            self._compute_ipc_contributions(self._last_vertices, compute)

    def initialize(self, pb):
        """Initialize the IPC contact assembly.

        Called once by the solver at the start of the problem.  Builds
        the collision mesh, scatter matrix, and broad-phase instance.
        Computes the initial barrier stiffness (or uses the user-provided
        value).  When ``use_ccd`` is ``True``, registers a CCD
        line-search callback on the solver.
        """
        ipctk = _import_ipctk()
        ndim = self.space.ndim

        self._pb = pb

        # Track extra global DOFs (e.g. from PeriodicBC)
        self._n_global_dof = pb.n_global_dof

        # Extract surface mesh if needed
        if self._extract_surface and self._surface_mesh is None:
            from fedoo.mesh import extract_surface as extract_surface_mesh

            self._surface_mesh = extract_surface_mesh(self.mesh, reduce_order=True)

        if self._surface_mesh is None:
            raise ValueError(
                "A surface mesh must be provided or extract_surface must be True."
            )

        surf_mesh = self._surface_mesh

        # Get surface node indices (global indices in the full mesh)
        self._surface_node_indices = np.unique(surf_mesh.elements)

        # OGC per-vertex filtering is only valid for shell/surface meshes
        # where every node is on the surface.  Solid meshes with interior
        # DOFs cannot be handled correctly by per-vertex clamping.
        if self._use_ogc and len(self._surface_node_indices) != self.mesh.n_nodes:
            raise ValueError(
                "use_ogc=True requires a shell/surface mesh where all nodes "
                "are on the surface. This mesh has interior nodes "
                f"({self.mesh.n_nodes - len(self._surface_node_indices)} "
                f"interior, {len(self._surface_node_indices)} surface).  "
                "Use use_ccd=True instead for solid meshes."
            )

        # Build mapping from global node indices to local surface ordering
        global_to_local = np.full(self.mesh.n_nodes, -1, dtype=int)
        global_to_local[self._surface_node_indices] = np.arange(
            len(self._surface_node_indices)
        )

        # Remap surface elements to local ordering
        local_elements = global_to_local[surf_mesh.elements]

        # Rest positions of surface vertices
        self._rest_positions = self.mesh.nodes[self._surface_node_indices]

        # Build rigid body node lookup set
        if self._rigid_bodies:
            self._rigid_node_set = set()
            for node_set, rb in self._rigid_bodies:
                self._rigid_node_set.update(node_set.tolist())

            # OGC is incompatible with rigid bodies: per-vertex trust
            # region cannot be reconciled with the 6-DOF rigid Jacobian
            # reduction.  CCD is supported — _ccd_line_search reconstructs
            # rigid surface motion from dX[rigid_global_dofs] via J.
            if self._use_ogc:
                raise ValueError("use_ogc=True is not supported with rigid bodies.")

        # Build scatter matrix for DOF mapping (ipctk surface -> fedoo full)
        self._scatter_matrix = self._build_scatter_matrix(
            self._surface_node_indices, self.mesh.n_nodes, ndim, pb
        )

        # Create broad phase instance
        self._broad_phase = self._create_broad_phase()

        # Build edges and faces for the collision mesh
        if ndim == 2:
            edges = local_elements  # lin2 elements are edges
            faces = np.empty((0, 3), dtype=int)
        else:  # 3D
            edges = ipctk.edges(local_elements)
            faces = local_elements  # tri3 elements are faces

        # Create CollisionMesh without displacement_map
        # (displacement_map causes segfault in ipctk 1.5.0;
        # DOF reordering is handled manually via scatter matrix)
        self._collision_mesh = ipctk.CollisionMesh(
            self._rest_positions,
            edges,
            faces,
        )

        # Cache bounding-box diagonal (used for dhat, kappa, energy line search)
        self._bbox_diag = np.linalg.norm(
            self._rest_positions.max(axis=0) - self._rest_positions.min(axis=0)
        )

        # Build axisymmetric (2π·r) integration weight on surface DOFs.
        # Applied per surface vertex, broadcast over the ipctk interleaved
        # layout [x0, y0, x1, y1, ...]. Reference radius R₀ is used,
        # consistent with the rest of the 2Daxi pipeline.
        if self.space.is_axisymmetric:
            r_surf = self._rest_positions[:, 0]
            self._axi_weight = np.repeat(2.0 * np.pi * r_surf, ndim)
            self._axi_diag = sparse.diags(self._axi_weight)

        # Compute actual dhat
        if self._dhat_is_relative:
            self._actual_dhat = self._dhat * self._bbox_diag
        else:
            self._actual_dhat = self._dhat

        # Create barrier potential
        # Fedoo applies the adaptive barrier stiffness separately.
        self._barrier_potential = ipctk.BarrierPotential(self._actual_dhat, 1.0)

        # Create friction potential if needed
        if self.friction_coefficient > 0:
            self._friction_potential = ipctk.FrictionPotential(self._eps_v)

        # Build initial collisions
        self._collisions = ipctk.NormalCollisions()
        vertices = self._get_current_vertices(pb)
        self._collisions.build(
            self._collision_mesh,
            vertices,
            self._actual_dhat,
            broad_phase=self._broad_phase,
        )

        # Auto-compute or set barrier stiffness
        if self._kappa is None:
            self._initialize_kappa(vertices)
        else:
            self._max_kappa = 100.0 * self._kappa

        # Build friction collisions if needed
        if self.friction_coefficient > 0:
            self._friction_collisions = ipctk.TangentialCollisions()
            self._friction_collisions.build(
                self._collision_mesh,
                vertices,
                self._collisions,
                self._barrier_potential,
                self._kappa,
                self.friction_coefficient,
            )

        # Store minimum distance
        if len(self._collisions) > 0:
            self._prev_min_distance = self._collisions.compute_minimum_distance(
                self._collision_mesh, vertices
            )
        else:
            self._prev_min_distance = np.inf

        # Compute initial contributions
        self._compute_ipc_contributions(vertices)

        # Store initial state
        self.sv["kappa"] = self._kappa
        self.sv["max_kappa"] = self._max_kappa
        self.sv["prev_min_distance"] = self._prev_min_distance
        self.sv["max_kappa_set"] = self._max_kappa_set
        self.sv_start = dict(self.sv)

        # Register line-search / step-filter callback
        if self._use_ccd:
            pb.add_line_search(self._ccd_line_search, name="ccd")
        elif self._use_ogc:
            self._ogc_trust_region = ipctk.ogc.TrustRegion(self._actual_dhat)
            pb._step_filter_callback = self._ogc_step_filter_callback

    def _update_kappa_adaptive(self, vertices):
        """Double kappa when the gap is small and decreasing.

        Uses ``ipctk.update_barrier_stiffness`` with
        ``dhat_epsilon_scale = dhat / bbox_diag`` so the doubling
        triggers within the barrier zone.  Only called between time
        steps (in ``set_start``), never inside the NR loop.
        """
        ipctk = _import_ipctk()
        bbox_diag = self._bbox_diag

        if len(self._collisions) > 0:
            min_dist = self._collisions.compute_minimum_distance(
                self._collision_mesh, vertices
            )
        else:
            min_dist = np.inf

        eps_scale = self._actual_dhat / bbox_diag
        new_kappa = ipctk.update_barrier_stiffness(
            self._prev_min_distance,
            min_dist,
            self._max_kappa,
            self._kappa,
            bbox_diag,
            dhat_epsilon_scale=eps_scale,
        )
        if new_kappa > self._kappa:
            self._kappa = new_kappa
            self.sv["kappa"] = self._kappa

        self._prev_min_distance = min_dist
        self.sv["max_kappa"] = self._max_kappa
        self.sv["prev_min_distance"] = self._prev_min_distance

    def set_start(self, pb):
        """Begin a new time increment.

        Kappa management follows the reference IPC algorithm:

        1. **Re-initialization** from gradient balance at each time
           step when contacts exist.  The MAX guard in
           ``_initialize_kappa`` prevents kappa from decreasing.

        2. **Adaptive doubling** — ``ipctk.update_barrier_stiffness``
           doubles kappa when the gap is small and decreasing.  This
           runs between time steps only (never within the NR loop,
           as changing kappa mid-solve prevents NR convergence).
        """
        self.sv_start = dict(self.sv)

        vertices = self._get_current_vertices(pb)

        # Rebuild collisions
        self._collisions.build(
            self._collision_mesh,
            vertices,
            self._actual_dhat,
            broad_phase=self._broad_phase,
        )

        # Re-initialize kappa from gradient balance each time step
        # when contacts exist.  The MAX guard in _initialize_kappa
        # prevents kappa from decreasing.  max_kappa is set once
        # (the first time contacts appear) and kept fixed so that it
        # actually caps the adaptive doubling.
        if self._barrier_stiffness_init is None and len(self._collisions) > 0:
            self._initialize_kappa(vertices)
            if not self._max_kappa_set:
                self._max_kappa_set = True

        # Adaptive doubling when gap is small and decreasing
        if self._adaptive_barrier_stiffness and self._max_kappa is not None:
            self._update_kappa_adaptive(vertices)

        # Track collision count for detecting new contacts during NR
        self._n_collisions_at_start = len(self._collisions)
        self._kappa_boosted_this_increment = False

        # Store last vertices for next increment
        self._last_vertices = vertices

        # Store start-of-increment vertices for OGC warm-start
        if self._use_ogc:
            self._ogc_vertices_at_start = vertices.copy()

        # Compute IPC contributions for the elastic prediction
        self._compute_ipc_contributions(vertices)

    def update(self, pb, compute="all"):
        """Update the IPC assembly for the current Newton–Raphson iteration.

        Rebuilds the collision set from the current vertex positions,
        re-initialises kappa if contacts appeared for the first time in
        an increment that started contact-free, enforces SDI (minimum
        NR sub-iterations) when the contact state changed or surfaces
        are dangerously close, and recomputes the barrier contributions.

        Parameters
        ----------
        pb : NonLinear
            The current problem instance.
        compute : str, default "all"
            What to compute: ``"all"``, ``"matrix"``, or ``"vector"``.
        """
        vertices = self._get_current_vertices(pb)
        self._last_vertices = vertices

        # Rebuild scatter matrix for rigid bodies (Jacobian uses r_ref but
        # depends on the current rotation, so refresh after every update)
        if self._rigid_bodies:
            self._scatter_matrix = self._build_scatter_matrix(
                self._surface_node_indices,
                self.mesh.n_nodes,
                self.space.ndim,
                pb,
            )

        # Rebuild collision set
        self._collisions.build(
            self._collision_mesh,
            vertices,
            self._actual_dhat,
            broad_phase=self._broad_phase,
        )

        # Initialize kappa when contacts first appear during an
        # increment that started with zero contacts.  MAX guard in
        # _initialize_kappa ensures kappa never decreases.
        if (
            not self._kappa_boosted_this_increment
            and self._n_collisions_at_start == 0
            and self._barrier_stiffness_init is None
            and len(self._collisions) > 0
        ):
            self._initialize_kappa(vertices)
            self._kappa_boosted_this_increment = True
            # Set generous max_kappa ceiling
            self._max_kappa = max(self._max_kappa, 1e4 * max(self._kappa, 1.0))

        # SDI: force at least one NR correction when contact state
        # changed (collision count differs from start) or when surfaces
        # are dangerously close (min_d/dhat < 0.1).
        n_collisions_now = len(self._collisions)
        need_sdi = n_collisions_now != self._n_collisions_at_start
        min_d_val = None
        if not need_sdi and n_collisions_now > 0:
            min_d_val = self._collisions.compute_minimum_distance(
                self._collision_mesh, vertices
            )
            if min_d_val < 0.1 * self._actual_dhat:
                need_sdi = True
        elif n_collisions_now > 0:
            min_d_val = self._collisions.compute_minimum_distance(
                self._collision_mesh, vertices
            )
        if need_sdi:
            # Scale min_subiter by proximity: more iterations when closer
            if min_d_val is not None and min_d_val < 0.01 * self._actual_dhat:
                self._pb._nr_min_subiter = max(self._pb._nr_min_subiter, 3)

        # Always force at least 1 NR iteration when IPC is active.
        # The Displacement criterion's reference error grows with
        # accumulated displacement, making the tolerance progressively
        # looser.  Without this floor, many steps are accepted at
        # iter 0 (elastic prediction only), accumulating equilibrium
        # errors that cause stress oscillations.
        self._pb._nr_min_subiter = max(self._pb._nr_min_subiter, 1)

        # Build friction collisions if enabled
        if self.friction_coefficient > 0:
            self._friction_collisions.build(
                self._collision_mesh,
                vertices,
                self._collisions,
                self._barrier_potential,
                self._kappa,
                self.friction_coefficient,
            )

        # Update OGC trust regions after rebuilding collisions
        if self._use_ogc and self._ogc_trust_region is not None:
            self._ogc_trust_region.update_if_needed(
                self._collision_mesh,
                np.asfortranarray(vertices),
                self._collisions,
                broad_phase=self._broad_phase,
            )

        # Compute contributions
        self._compute_ipc_contributions(vertices, compute)

    def to_start(self, pb):
        """Restart the current time increment after NR failure.

        Restores kappa, max_kappa and prev_min_distance from saved
        state, rebuilds collisions from the restored displacement,
        and recomputes barrier contributions.
        """
        self.sv = dict(self.sv_start)
        self._kappa = self.sv_start.get("kappa", self._kappa)
        self._max_kappa = self.sv_start.get("max_kappa", self._max_kappa)
        self._prev_min_distance = self.sv_start.get(
            "prev_min_distance", self._prev_min_distance
        )
        self._max_kappa_set = self.sv_start.get("max_kappa_set", self._max_kappa_set)
        self._kappa_boosted_this_increment = False

        # Recompute from current displacement state
        vertices = self._get_current_vertices(pb)
        self._last_vertices = vertices

        self._collisions.build(
            self._collision_mesh,
            vertices,
            self._actual_dhat,
            broad_phase=self._broad_phase,
        )

        if self.friction_coefficient > 0:
            self._friction_collisions.build(
                self._collision_mesh,
                vertices,
                self._collisions,
                self._barrier_potential,
                self._kappa,
                self.friction_coefficient,
            )

        self._compute_ipc_contributions(vertices)

        # Reset OGC trust region on NR failure
        if self._use_ogc:
            self._ogc_vertices_at_start = vertices.copy()
            self._ogc_trust_region = _import_ipctk().ogc.TrustRegion(self._actual_dhat)


class IPCSelfContact(IPCContact):
    r"""Self-contact Assembly using the IPC method.

    Convenience wrapper around :py:class:`IPCContact` that automatically
    extracts the surface mesh from the volumetric mesh. Useful for
    problems where a single body may come into contact with itself (e.g.
    buckling, folding, compression of porous structures).

    All barrier stiffness tuning is automatic (see :py:class:`IPCContact`
    for details).

    Parameters
    ----------
    mesh : fedoo.Mesh
        Full volumetric mesh.
    dhat : float, default=1e-3
        Barrier activation distance (relative to bounding box diagonal by
        default, see ``dhat_is_relative``).
    dhat_is_relative : bool, default=True
        If ``True``, ``dhat`` is a fraction of the bounding box diagonal.
    barrier_stiffness : float, optional
        Barrier stiffness :math:`\kappa`.  ``None`` (default) for automatic
        computation and adaptive update — **recommended**.
    friction_coefficient : float, default=0.0
        Coulomb friction coefficient :math:`\mu`.
    eps_v : float, default=1e-3
        Friction smoothing velocity.
    broad_phase : str, default="lbvh"
        Broad-phase collision detection method.
    adaptive_barrier_stiffness : bool, default=True
        Adaptively update :math:`\kappa` at each converged time increment.
    use_ccd : bool, default=False
        Enable CCD line search for robustness.  Mutually exclusive
        with ``use_ogc``.
    line_search_energy : bool or None, default=None
        Energy-based backtracking after CCD.  ``None`` auto-enables
        when ``use_ccd=True`` (see :py:class:`IPCContact`).
    use_ogc : bool, default=False
        Enable OGC trust-region step filtering.  Mutually exclusive
        with ``use_ccd``.  **Only supported for shell / surface
        meshes**; raises ``ValueError`` for solid meshes.
    space : ModelingSpace, optional
        Modeling space.
    name : str, default="IPC Self Contact"
        Name of the contact assembly.

    Examples
    --------
    .. code-block:: python

        import fedoo as fd

        fd.ModelingSpace("3D")
        mesh = fd.Mesh.read("gyroid.vtk")
        material = fd.constitutivelaw.ElasticIsotrop(1e5, 0.3)

        contact = fd.constraint.IPCSelfContact(mesh, use_ccd=True)

        wf = fd.weakform.StressEquilibrium(material, nlgeom="UL")
        solid = fd.Assembly.create(wf, mesh)
        assembly = fd.Assembly.sum(solid, contact)

        pb = fd.problem.NonLinear(assembly)
        res = pb.add_output("results", solid, ["Disp", "Stress"])
        # ... add BCs and solve ...

    See Also
    --------
    IPCContact : Base class with full parameter documentation.
    """

    def __init__(
        self,
        mesh,
        dhat=1e-3,
        dhat_is_relative=True,
        barrier_stiffness=None,
        friction_coefficient=0.0,
        eps_v=1e-3,
        broad_phase="lbvh",
        adaptive_barrier_stiffness=True,
        use_ccd=None,
        line_search_energy=None,
        use_ogc=False,
        space=None,
        name="IPC Self Contact",
    ):
        super().__init__(
            mesh=mesh,
            surface_mesh=None,
            extract_surface=True,
            dhat=dhat,
            dhat_is_relative=dhat_is_relative,
            barrier_stiffness=barrier_stiffness,
            friction_coefficient=friction_coefficient,
            eps_v=eps_v,
            broad_phase=broad_phase,
            adaptive_barrier_stiffness=adaptive_barrier_stiffness,
            use_ccd=use_ccd,
            line_search_energy=line_search_energy,
            use_ogc=use_ogc,
            space=space,
            name=name,
        )
