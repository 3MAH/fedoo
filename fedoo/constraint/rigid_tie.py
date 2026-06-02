"""Rigid Tie constraint."""

import numpy as np
from fedoo.core.boundary_conditions import BCBase, MPC, ListBC

from simcoon import Rotation, dR_drotvec as _simcoon_dR_drotvec


class RigidTie(BCBase):
    """Constraint that eliminate dof assuming a rigid body tie between nodes.

    Create an object that defines a rigid tie coupling between some nodes using
    several multi-points constraints. Some constraint drivers (cd) dof are
    used to define rigid body displacement and rotation. By default, the center of
    rotation is located at the center of the bounding box defined by all nodes
    tied together. The rotation center move with the rigid displacement.
    RigidTie constraint add 6 global_dof to the problem:
    "RigidDispX", "RigidDispY", "RigidDispZ", "RigidRotX", "RigidRotY", "RigidRotZ"
    and two global_dot vectors:

    * "RigidDisp" = ["RigidDispX", "RigidDispY", "RigidDispZ"] for rigid displacement
    * "RigidRot" = ["RigidRotX", "RigidRotY", "RigidRotZ"] for rigid rotation

    If several RigidTie constraints are used with the same problem, all the previous
    variables will contains several dof, the indice of the dof being associated to
    order in which the constraint has been added.
    For instance, pb.global_dof['RigidDispX'][0] will be associated to the first
    added RigidTie and dof['RigidDispX'][1] to the second one.


    Parameters
    ----------
    list_nodes: list (int) or 1d np.array
        List of nodes that will be eliminated considering a rigid body tie.
    center: int or np.array[float] with shape = (3), optional
        If center is an int, the rotation center will be initialized at the coordinates
        of the corresponding node in the mesh. If center is an array (or list, ...)
        it contains the initial coordinates of the rotation center. By default, the
        center is set at the middle of the rigid sets.
    use_quaternion: bool, optional
        If True (default), use a multiplicative quaternion update to avoid
        gimbal lock for large rotations. The rotation DOFs (RigidRotX/Y/Z)
        are interpreted as the **rotation-vector** components of a small
        increment relative to a quaternion base state that is updated at each
        converged increment.
        If False, there is no base state and the DOFs are the components of
        the total rotation vector (a single exponential map).
    name : str, optional
        Name of the created boundary condition. The default is "Rigid Tie".


    Definition of rotations
    -----------------------
    The rotation DOFs ``RigidRotX/Y/Z`` are the components of a **rotation
    vector** (exponential map): ``rotvec = theta * axis``, with ``theta`` the
    rotation angle and ``axis`` the unit rotation axis, so that
    ``R = expm(skew(rotvec))`` (Rodrigues' formula). This is convention-free
    — there is no XYZ ordering — and is consistent with the rotation-vector
    DOFs used by Fedoo's beam elements.

    .. note::
        This differs from earlier versions of ``RigidTie``, which interpreted
        ``RigidRotX/Y/Z`` as intrinsic Euler **XYZ** angles
        (``Rotation.from_euler("XYZ", ...)``). For a single-axis rotation the
        two parameterizations are identical; they differ only for finite,
        simultaneous multi-axis prescribed rotations. Small-strain /
        linearized results (rotations near zero) are unchanged.

    When ``use_quaternion=True`` (default), the total rotation is stored as a
    quaternion (via ``simcoon.Rotation``) using a multiplicative update:

    - The rotation-vector DOFs represent a **small incremental** rotation
      ``delta`` from the current quaternion base state ``Q_base``.
    - At each converged increment, the increment is composed:
      ``Q_base = Rotation.from_rotvec(delta) * Q_base``
    - The total rotation is exact for arbitrarily large angles — no gimbal
      lock, no small-angle approximation.
    - The rotation derivatives ``dR/d(rotvec)`` used in the MPC linearization
      are evaluated at the small increment (well-conditioned), then composed
      with ``R_base`` via the chain rule for exact consistency.

    When ``use_quaternion=False``, there is no base state: the DOFs are the
    components of the total rotation vector and ``Rotation.from_rotvec`` is
    used directly.


    Notes
    -----
    * The node given in list_nodes are eliminated from the system (slave nodes)
      and can't be used in other boundary conditions. The boundary conditions should
      be enforce using the added global dof.
    * The rigid coupling is highly non-linear and the multi-point constraints
      are modified at each iteration.
    * Once created the RigidTie object needs to be associated to the problem
      using the Problem.add method.


    Example
    -------

    .. code-block:: python

        import fedoo as fd

        mesh = fd.mesh.box_mesh()

        left_face = mesh.find_nodes('X', mesh.bounding_box.xmin)
        right_face = mesh.find_nodes('X', mesh.bounding_box.xmax)

        rigid_tie = fd.constraint.RigidTie(right_face)
    """

    def __init__(self, list_nodes, center=None, use_quaternion=True, name="Rigid Tie"):
        self.list_nodes = list_nodes
        self.center = center
        self.use_quaternion = use_quaternion
        self.bc_type = "RigidTie"
        BCBase.__init__(self, name)
        self._keep_at_end = True

        self._update_during_inc = 1

    def __repr__(self):
        list_str = ["Rigid Tie:"]
        if self.name != "":
            list_str.append("name = '{}'".format(self.name))

        return "\n".join(list_str)

    def initialize(self, problem):
        if self.center is None:
            # initialize the rotation center at center of rigid nodes bounding box
            nodes_crd = problem.mesh.nodes[self.list_nodes]
            self.center = 0.5 * (nodes_crd.min(axis=0) + nodes_crd.max(axis=0))
        elif np.isscalar(self.center):
            # initialize the center at a position of a node
            self.center = problem.mesh.nodes[self.center]
        else:
            self.center = np.asarray(self.center)

        dof_indice_disp = problem.add_global_dof(
            ["RigidDispX", "RigidDispY", "RigidDispZ"], 1, "RigidDisp"
        )
        dof_indice_rot = problem.add_global_dof(
            ["RigidRotX", "RigidRotY", "RigidRotZ"], 1, "RigidRot"
        )
        self.var_cd = [
            "RigidDispX",
            "RigidDispY",
            "RigidDispZ",
            "RigidRotX",
            "RigidRotY",
            "RigidRotZ",
        ]
        self.node_cd = [
            dof_indice_disp,
            dof_indice_disp,
            dof_indice_disp,
            dof_indice_rot,
            dof_indice_rot,
            dof_indice_rot,
        ]

        # extract indices array that gives the disp from the full dof solution
        n_nodes = problem.mesh.n_nodes
        rank = problem.space.variable_rank("DispX")
        # rank = rank of variable "DispX". rank of "DispY" and "DispZ" should follow
        self._disp_indices = (
            np.c_[rank * n_nodes, (rank + 1) * n_nodes, (rank + 2) * n_nodes]
            + self.list_nodes[:, None]
        )

        # Quaternion base state for multiplicative rotation update.
        # ``_Q_base`` advances only on a converged increment (``set_start``)
        # and is never reverted: a failed increment never advances it, so the
        # rollback hook ``to_start_bc`` is a no-op. See ``to_start_bc``.
        if self.use_quaternion:
            self._Q_base = Rotation.identity()
            self._angles_at_base = np.zeros(3)

    def _get_dof_ref(self, problem):
        """Read current values of the 6 rigid DOFs from the problem."""
        dof_cd = [
            problem.n_node_dof
            + problem._global_dof.indice_start(self.var_cd[i])
            + self.node_cd[i]
            for i in range(6)
        ]
        dof_sol = problem.get_dof_solution()
        xbc = problem._Xbc
        if np.isscalar(dof_sol) and dof_sol == 0:
            if np.isscalar(xbc) and xbc == 0:
                return np.zeros(6), dof_cd
            return np.array([xbc[dof] for dof in dof_cd]), dof_cd
        else:
            if np.isscalar(xbc) and xbc == 0:
                return np.array([dof_sol[dof] for dof in dof_cd]), dof_cd
            return np.array([dof_sol[dof] + xbc[dof] for dof in dof_cd]), dof_cd

    def _compute_rotation(self, angles):
        """Compute total rotation matrix and derivatives w.r.t. rotation DOFs.

        When ``use_quaternion=True``, the rotation is composed multiplicatively:
        ``R_total = R_inc(delta) * Q_base`` where ``delta`` is always small.

        The rotation DOFs are **rotation vector** components (exponential map):
        ``rotvec = theta * axis``. This is convention-free (no XYZ ordering)
        and consistent with Fedoo's beam elements.

        Derivatives ``dR/dω_k`` use the exact Rodrigues differentiation
        (Gallego & Yezzi 2015).

        Returns R (3x3), dR_dw0, dR_dw1, dR_dw2 (3x3 each).
        """
        if self.use_quaternion and hasattr(self, "_Q_base"):
            delta = angles - self._angles_at_base
            if np.any(np.isnan(delta)) or np.any(np.isinf(delta)):
                delta = np.zeros(3)
            R_inc = Rotation.from_rotvec(delta)
            R_total = R_inc * self._Q_base
            R = R_total.as_matrix()
            R_base_mat = self._Q_base.as_matrix()
            omega = delta
        else:
            R = Rotation.from_rotvec(angles).as_matrix()
            R_base_mat = np.eye(3)
            omega = angles

        # Exact derivatives dR_inc/dω_k via Rodrigues differentiation
        dRinc = self._dR_drotvec(omega)

        # Chain rule: dR_total/dω_k = dR_inc/dω_k @ R_base
        dR_dw0 = dRinc[0] @ R_base_mat
        dR_dw1 = dRinc[1] @ R_base_mat
        dR_dw2 = dRinc[2] @ R_base_mat

        return R, dR_dw0, dR_dw1, dR_dw2

    @staticmethod
    def _dR_drotvec(omega):
        """Exact derivatives of R(ω) w.r.t. rotation vector components.

        Delegates to simcoon's ``dR_drotvec`` (Gallego & Yezzi, 2015).

        Returns
        -------
        dR : tuple of 3 ndarrays (3x3)
            (dR/dω₀, dR/dω₁, dR/dω₂).
        """
        cube = _simcoon_dR_drotvec(np.asarray(omega, dtype=float))
        return (cube[:, :, 0], cube[:, :, 1], cube[:, :, 2])

    def _compute_slave_disp(self, problem, disp_ref, R):
        """Compute and write slave node displacements from rigid body state."""
        mesh = problem.mesh
        list_nodes = self.list_nodes
        new_disp = (
            (mesh.nodes[list_nodes] - self.center) @ R.T
            + self.center
            + disp_ref
            - mesh.nodes[list_nodes]
        )
        if not np.array_equal(problem._dU, 0):
            if np.array_equal(problem._U, 0):
                problem._dU[self._disp_indices] = new_disp
            else:
                problem._dU[self._disp_indices] = (
                    new_disp - problem._U[self._disp_indices]
                )
        return new_disp

    def pre_update(self, problem):
        """Refresh slave node positions in _dU before assembly update.

        Called by the solver BEFORE assembly.update() so that other assemblies
        (e.g. IPCContact) see the correct geometry of the rigid body surface.
        """
        dof_ref, _ = self._get_dof_ref(problem)
        disp_ref = dof_ref[:3]
        angles = dof_ref[3:]
        R, _, _, _ = self._compute_rotation(angles)
        self._compute_slave_disp(problem, disp_ref, R)

    def set_start(self, problem):
        """Absorb the converged incremental rotation into the quaternion base.

        Called by the solver after a converged increment, before _dU is reset.
        This is the only place ``_Q_base`` advances; it is never reverted
        (see ``to_start_bc``).
        """
        if not self.use_quaternion or not hasattr(self, "_Q_base"):
            return
        dof_ref, _ = self._get_dof_ref(problem)
        angles = dof_ref[3:]
        if np.any(np.isnan(angles)) or np.any(np.isinf(angles)):
            return
        delta = angles - self._angles_at_base
        if not np.allclose(delta, 0, atol=1e-15):
            R_inc = Rotation.from_rotvec(delta)
            self._Q_base = R_inc * self._Q_base
            self._angles_at_base = angles.copy()

    def to_start_bc(self, problem):
        """No-op rollback hook for a failed increment.

        The solver advances ``_Q_base`` only via ``set_start``, and only for
        *converged* increments (``set_start`` runs at the top of the following
        increment). A failed increment therefore never touches ``_Q_base`` or
        ``_angles_at_base`` — they already hold the last-converged state — so
        rolling back requires no action here.

        The previous implementation reverted to a ``_Q_base_backup`` captured
        *before* the last converged advance, which silently discarded the last
        converged rotation on any ``dt`` reduction during a rotating solve.
        """
        return

    @property
    def Q_total(self):
        """Current total rotation as a Rotation object (quaternion-backed)."""
        if self.use_quaternion:
            return self._Q_base
        return None

    def generate(self, problem, t_fact=1, t_fact_old=None):
        mesh = problem.mesh
        var_cd = self.var_cd
        node_cd = self.node_cd
        list_nodes = self.list_nodes

        dof_ref, dof_cd = self._get_dof_ref(problem)
        disp_ref = dof_ref[:3]
        angles = dof_ref[3:]

        # Compute rotation and derivatives
        R, dR_drx, dR_dry, dR_drz = self._compute_rotation(angles)

        # Set slave node displacements
        self._compute_slave_disp(problem, disp_ref, R)

        # MPC linearization
        crd = mesh.nodes[list_nodes] - self.center
        du_drx = crd @ dR_drx.T
        du_dry = crd @ dR_dry.T
        du_drz = crd @ dR_drz.T

        #### MPC ####
        res = ListBC()
        res.append(
            MPC(
                [
                    list_nodes,
                    np.full_like(list_nodes, node_cd[0]),
                    np.full_like(list_nodes, node_cd[3]),
                    np.full_like(list_nodes, node_cd[4]),
                    np.full_like(list_nodes, node_cd[5]),
                ],
                ["DispX", var_cd[0], var_cd[3], var_cd[4], var_cd[5]],
                [
                    np.full_like(list_nodes, 1.0),
                    np.full_like(list_nodes, -1.0),
                    -du_drx[:, 0],
                    -du_dry[:, 0],
                    -du_drz[:, 0],
                ],
            )
        )
        res.append(
            MPC(
                [
                    list_nodes,
                    np.full_like(list_nodes, node_cd[1]),
                    np.full_like(list_nodes, node_cd[3]),
                    np.full_like(list_nodes, node_cd[4]),
                    np.full_like(list_nodes, node_cd[5]),
                ],
                ["DispY", var_cd[1], var_cd[3], var_cd[4], var_cd[5]],
                [
                    np.full_like(list_nodes, 1.0),
                    np.full_like(list_nodes, -1.0),
                    -du_drx[:, 1],
                    -du_dry[:, 1],
                    -du_drz[:, 1],
                ],
            )
        )
        res.append(
            MPC(
                [
                    list_nodes,
                    np.full_like(list_nodes, node_cd[2]),
                    np.full_like(list_nodes, node_cd[3]),
                    np.full_like(list_nodes, node_cd[4]),
                    np.full_like(list_nodes, node_cd[5]),
                ],
                ["DispZ", var_cd[2], var_cd[3], var_cd[4], var_cd[5]],
                [
                    np.full_like(list_nodes, 1.0),
                    np.full_like(list_nodes, -1.0),
                    -du_drx[:, 2],
                    -du_dry[:, 2],
                    -du_drz[:, 2],
                ],
            )
        )

        res.initialize(problem)
        return res.generate(problem, t_fact, t_fact_old)


class RigidTie2D(BCBase):
    """Constraint that eliminate dof assuming a rigid body tie between nodes in 2D.

    Same constraint as RigidTie, but for 2D problems.
    See RigidTie documentation for more details.
    """

    def __init__(self, list_nodes, center=None, name="Rigid Tie 2D"):
        self.list_nodes = list_nodes
        self.center = center
        self.bc_type = "RigidTie2D"
        BCBase.__init__(self, name)
        self._keep_at_end = True

        self._update_during_inc = 1

    def __repr__(self):
        list_str = ["Rigid Tie 2D:"]
        if self.name != "":
            list_str.append("name = '{}'".format(self.name))

        return "\n".join(list_str)

    def initialize(self, problem):
        if self.center is None:
            # initialize the rotation center at center of rigid nodes bounding box
            nodes_crd = problem.mesh.nodes[self.list_nodes]
            self.center = 0.5 * (nodes_crd.min(axis=0) + nodes_crd.max(axis=0))
        elif np.isscalar(self.center):
            # initialize the center at a position of a node
            self.center = problem.mesh.nodes[self.center]
        else:
            self.center = np.asarray(self.center)
        dof_indice_disp = problem.add_global_dof(
            ["RigidDispX", "RigidDispY"], 1, "RidigDisp"
        )
        dof_indice_rot = problem.add_global_dof(["RigidRotZ"], 1, "RidigRot")
        self.var_cd = [
            "RigidDispX",
            "RigidDispY",
            "RigidRotZ",
        ]
        self.node_cd = [dof_indice_disp, dof_indice_disp, dof_indice_rot]

        # extract indices array that gives the disp from the full dof solution
        n_nodes = problem.mesh.n_nodes
        rank = problem.space.variable_rank("DispX")
        # rank = rank of variable "DispX". rank of "DispY" should be rank+1
        self._disp_indices = (
            np.c_[rank * n_nodes, (rank + 1) * n_nodes] + self.list_nodes[:, None]
        )

    def _get_dof_ref(self, problem):
        """Read current values of the 3 rigid DOFs [dx, dy, rotZ].

        Mirrors :meth:`RigidTie._get_dof_ref`, including the guards for an
        uninitialized state (``get_dof_solution()`` or ``_Xbc`` still the
        scalar ``0``) — needed because ``pre_update`` may read the DOFs
        before boundary conditions populate ``_Xbc``.
        """
        dof_cd = [
            problem.n_node_dof
            + problem._global_dof.indice_start(self.var_cd[i])
            + self.node_cd[i]
            for i in range(len(self.var_cd))
        ]
        dof_sol = problem.get_dof_solution()
        xbc = problem._Xbc
        if np.isscalar(dof_sol) and dof_sol == 0:
            if np.isscalar(xbc) and xbc == 0:
                return np.zeros(3), dof_cd
            return np.array([xbc[dof] for dof in dof_cd]), dof_cd
        if np.isscalar(xbc) and xbc == 0:
            return np.array([dof_sol[dof] for dof in dof_cd]), dof_cd
        return np.array([dof_sol[dof] + xbc[dof] for dof in dof_cd]), dof_cd

    def _compute_rotation(self, angle):
        """2D rotation matrix and its derivative w.r.t. the Z angle.

        A single in-plane angle has no gimbal lock, so the closed-form
        ``[[cos, -sin], [sin, cos]]`` is exact for arbitrarily large rotation
        — no quaternion base state is needed (unlike 3D ``RigidTie``).
        """
        sin = np.sin(angle)
        cos = np.cos(angle)
        R = np.array([[cos, -sin], [sin, cos]])
        dR_drz = np.array([[-sin, -cos], [cos, -sin]])
        return R, dR_drz

    def _compute_slave_disp(self, problem, disp_ref, R):
        """Compute and write 2D slave node displacements into _dU."""
        mesh = problem.mesh
        list_nodes = self.list_nodes
        new_disp = (
            (mesh.nodes[list_nodes] - self.center) @ R.T
            + self.center
            + disp_ref
            - mesh.nodes[list_nodes]
        )
        if not np.array_equal(problem._dU, 0):
            if np.array_equal(problem._U, 0):
                problem._dU[self._disp_indices] = new_disp
            else:
                problem._dU[self._disp_indices] = (
                    new_disp - problem._U[self._disp_indices]
                )
        return new_disp

    def pre_update(self, problem):
        """Refresh slave node positions in _dU before assembly update.

        Mirrors :meth:`RigidTie.pre_update` for the 2D case so other
        assemblies (e.g. IPCContact) see the correct geometry of the rigid
        surface. Without this hook a ``RigidTie2D`` combined with IPC contact
        would expose stale slave positions for one iteration.
        """
        dof_ref, _ = self._get_dof_ref(problem)
        disp_ref = dof_ref[:2]
        R, _ = self._compute_rotation(dof_ref[2])
        self._compute_slave_disp(problem, disp_ref, R)

    def generate(self, problem, t_fact=1, t_fact_old=None):
        var_cd = self.var_cd
        node_cd = self.node_cd
        list_nodes = self.list_nodes

        dof_ref, _ = self._get_dof_ref(problem)
        disp_ref = dof_ref[:2]  # reference displacement
        R, dR_drz = self._compute_rotation(dof_ref[2])  # rotation Z angle

        # Correct displacement of slave nodes to be consistent with the masters
        self._compute_slave_disp(problem, disp_ref, R)

        # MPC linearization
        crd = problem.mesh.nodes[list_nodes, :2] - self.center
        du_drz = crd @ dR_drz.T  # shape = (nnodes, 2)

        #### MPC ####

        # dUx - dUx_ref - du_drz[:,0]*drz_ref = 0
        # dUy - dUy_ref - du_drz[:,1]*drz_ref = 0
        # dU are associated to eliminated dof and should be different than ref dof
        res = ListBC()
        res.append(
            MPC(
                [
                    list_nodes,
                    np.full_like(list_nodes, node_cd[0]),
                    np.full_like(list_nodes, node_cd[2]),
                ],
                ["DispX", var_cd[0], var_cd[2]],
                [
                    np.full_like(list_nodes, 1.0),
                    np.full_like(list_nodes, -1.0),
                    -du_drz[:, 0],
                ],
            )
        )
        res.append(
            MPC(
                [
                    list_nodes,
                    np.full_like(list_nodes, node_cd[1]),
                    np.full_like(list_nodes, node_cd[2]),
                ],
                ["DispY", var_cd[1], var_cd[2]],
                [
                    np.full_like(list_nodes, 1.0),
                    np.full_like(list_nodes, -1.0),
                    -du_drz[:, 1],
                ],
            )
        )

        res.initialize(problem)
        return res.generate(problem, t_fact, t_fact_old)
