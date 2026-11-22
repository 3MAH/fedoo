from __future__ import annotations

import numpy as np
from fedoo.core.base import AssemblyBase

from fedoo.core.mesh import Mesh
from fedoo.lib_elements.element_list import get_element
from scipy import sparse, spatial
from fedoo.core.modelingspace import ModelingSpace
from fedoo.mesh import extract_surface as extract_surface_mesh
from copy import copy


def _ranges(starts, counts):
    """Concatenate the integer ranges [starts[i], starts[i] + counts[i])."""
    counts = np.asarray(counts, dtype=np.intp)
    total = int(counts.sum())
    if total == 0:
        return np.zeros(0, dtype=np.intp)
    offsets = np.cumsum(counts) - counts
    return np.repeat(np.asarray(starts, dtype=np.intp) - offsets, counts) + np.arange(
        total, dtype=np.intp
    )


def _elements_of_nodes(elements, nodes):
    """Return, for each node, the sorted indices of the elements containing it.

    Same result as ``(elements == nd).sum(axis=1).nonzero()[0]`` for each
    node ``nd`` of ``nodes``, with a single sort of the connectivity instead
    of a full scan per node.
    """
    n_per_elm = elements.shape[1]
    sorted_elm = np.sort(elements, axis=1)
    if (sorted_elm[:, 1:] == sorted_elm[:, :-1]).any():
        # degenerated elements (repeated node): keep the scanning definition
        return [(elements == nd).sum(axis=1).nonzero()[0] for nd in nodes]
    flat = elements.ravel()
    order = np.argsort(flat, kind="stable")
    elm_ids = order // n_per_elm
    ptr = np.searchsorted(flat[order], nodes, side="left")
    end = np.searchsorted(flat[order], nodes, side="right")
    return [elm_ids[ptr[i] : end[i]] for i in range(len(nodes))]


def _bucket_nearest_master(
    nodes,
    bbox_min,
    n_buckets,
    w,
    ndim,
    master_nodes,
    slave_nodes,
    max_dist,
    offsets,
    self_contact,
):
    """Nearest master node of each slave node using a bucket sort.

    Vectorized version of the historical per-bucket loop, with the same
    results: same bucket definition, same neighbor-bucket order (that fixes
    which node is selected between equidistant ones), same distance formula
    as ``scipy.spatial.distance_matrix`` and the same exclusions (the last
    bucket of the sorted list is not used and, for self contact, the slave
    node itself is skipped).

    Returns a float array: index of the nearest master node or -1.
    """
    nearest_neighbors = np.full(len(slave_nodes), -1.0)

    def bucket_ids(crd):
        bucket_id = crd[:, 0] + crd[:, 1] * n_buckets[0]
        if ndim == 3:
            bucket_id = bucket_id + crd[:, 2] * n_buckets[0] * n_buckets[1]
        return bucket_id

    def groups(bucket_id):
        sorted_indices = bucket_id.argsort()
        sorted_ids = bucket_id[sorted_indices]
        starts = np.r_[0, np.flatnonzero(sorted_ids[1:] != sorted_ids[:-1]) + 1]
        # the last group of the sorted list is not used (as historically)
        return sorted_indices, sorted_ids[starts[:-1]], starts

    # bucket sort of master nodes
    m_sorted, m_keys, m_starts = groups(
        bucket_ids(((nodes[master_nodes] - bbox_min) / w).astype(int))
    )
    # bucket sort of slave nodes
    s_crd = ((nodes[slave_nodes] - bbox_min) / w).astype(int)
    s_sorted, s_keys, s_starts = groups(bucket_ids(s_crd))
    n_groups = len(s_keys)
    if n_groups == 0 or len(m_keys) == 0:
        return nearest_neighbors

    # neighbor buckets of each slave bucket, in the historical order
    # (last dimension outermost, first dimension innermost)
    g_crd = s_crd[s_sorted[s_starts[:-1]]]  # crd of the first node of a bucket
    off_side_m, off_side_p, off_inside = offsets
    slot_off = np.empty((n_groups, ndim, 3), dtype=int)
    slot_ok = np.ones((n_groups, ndim, 3), dtype=bool)
    for dd in range(ndim):
        is_min = g_crd[:, dd] == 0
        is_max = (~is_min) & (g_crd[:, dd] == n_buckets[dd] - 1)
        slot_off[:, dd, :] = off_inside
        slot_off[is_min, dd, :2] = off_side_m
        slot_off[is_max, dd, :2] = off_side_p
        slot_ok[is_min | is_max, dd, 2] = False
    stride = [1, n_buckets[0], n_buckets[0] * n_buckets[1]]
    nb_id = []
    nb_ok = []
    for combo in np.ndindex(*(3,) * ndim):
        # combo = (slot of last dim, ..., slot of first dim)
        nid = s_keys.copy()
        ok = np.ones(n_groups, dtype=bool)
        for dd in range(ndim):
            slot = combo[ndim - 1 - dd]
            nid = nid + stride[dd] * slot_off[:, dd, slot]
            ok &= slot_ok[:, dd, slot]
        nb_id.append(nid)
        nb_ok.append(ok)
    nb_id = np.stack(nb_id, axis=1)  # (n_groups, 3**ndim)
    nb_ok = np.stack(nb_ok, axis=1)

    # locate the neighbor buckets among the master buckets
    pos = np.searchsorted(m_keys, nb_id)
    pos_c = np.minimum(pos, len(m_keys) - 1)
    found = nb_ok & (m_keys[pos_c] == nb_id)
    nb_start = m_starts[pos_c]
    nb_count = np.where(found, m_starts[pos_c + 1] - nb_start, 0)

    # neighbor master list of each bucket (concatenated, row-major order)
    g_count = nb_count.sum(axis=1)  # number of neighbor masters per bucket
    g_ptr = np.cumsum(g_count) - g_count
    nb_master_pos = _ranges(nb_start.ravel(), nb_count.ravel())

    # (slave, neighbor master) pairs, processed by blocks of slave nodes
    s_count = np.diff(s_starts)  # number of slaves per bucket
    slave_group = np.repeat(np.arange(n_groups), s_count)
    slave_row = np.arange(len(slave_group)) - np.repeat(s_starts[:-1], s_count)
    slave_idx = s_sorted[: len(slave_group)]
    slave_count = g_count[slave_group]  # number of pairs of each slave
    slave_ptr = np.cumsum(slave_count)
    block_size = 2**20  # max number of pairs in a block
    start = 0
    while start < len(slave_group):
        budget = slave_ptr[start] - slave_count[start] + block_size
        stop = max(start + 1, int(np.searchsorted(slave_ptr, budget, "right")))
        sl = slice(start, stop)
        start = stop
        pair_count = slave_count[sl]
        if pair_count.sum() == 0:
            continue
        pair_col = _ranges(np.zeros(len(pair_count), dtype=np.intp), pair_count)
        pair_slave = np.repeat(np.arange(len(pair_count)), pair_count)
        pair_master = master_nodes[
            m_sorted[
                nb_master_pos[np.repeat(g_ptr[slave_group[sl]], pair_count) + pair_col]
            ]
        ]
        x = nodes[slave_nodes[slave_idx[sl]]][pair_slave]
        y = nodes[pair_master]
        # same expression as scipy.spatial.distance_matrix (p=2)
        dist = np.sum(np.abs(y.astype(float) - x.astype(float)) ** 2, axis=-1) ** (
            1.0 / 2
        )
        if self_contact:
            # the diagonal of each bucket block is the slave node itself
            dist[pair_col == slave_row[sl][pair_slave]] = np.inf

        # first nearest master of each slave (argmin semantics: first
        # minimum, or first nan)
        has_pair = pair_count > 0
        seg_start = (np.cumsum(pair_count) - pair_count)[has_pair]
        seg_min = np.minimum.reduceat(dist, seg_start)
        rep_min = np.repeat(seg_min, pair_count[has_pair])
        is_min = (dist == rep_min) | (np.isnan(rep_min) & np.isnan(dist))
        pos_min = np.flatnonzero(is_min)
        best = pos_min[np.searchsorted(pos_min, seg_start)]
        min_dist = dist[best]
        nearest_neighbors[slave_idx[sl][has_pair]] = -1 + (min_dist <= max_dist) * (
            pair_master[best] + 1
        )
    return nearest_neighbors


class Contact(AssemblyBase):
    """Contact Assembly based on a node 2 surface formulation"""

    def __init__(
        self,
        slave_nodes: list[int] | Mesh,
        surface_mesh: Mesh,
        normal_law: str = "linear",
        search_algorithm: str = "bucket",  # 'bucket', 'search_nearest', 'ipctk'
        name: str = "Contact nodes 2 surface",
        space: ModelingSpace | None = None,
    ):
        """Contact Assembly based on a node 2 surface penality formulation.

        Parameters
        ----------
        slave_nodes: numpy array of int or Mesh
            List of nodes indices: nodes that belong to the slave surface
            or Mesh of the slave surface (in this case, the slave node indices
            are extracted).
        surface_mesh: fedoo.Mesh
            Mesh of the master surface
        normal_law: str in {'linear', 'bilinear'}, default = 'linear'
            Type of contact law for the normal contact.
        name: str
            The name of contact assembly
        space: ModelingSpace
            Modeling space associated to the weakform. If None is specified,
            the active ModelingSpace is considered.

        Notes
        -----
        Several attributes can be modified to change the contact behavior:
          * eps_n: contact penalty parameter. Need to be adjusted depending of
            the rigidity of the material in contact.
          * clearance: Contact clearance to adjust the two surfaces.
          * contact_search_once: If True (default), the elmement in contact
            are only searched at the begining of iteration.
            It avoid oscilations between contact and non contact state during
            the NR iterations.
          * tol: Tolerance for possible slide of a node outside an element.
          * max_dist: Max distance from nodes at which contact is considered.
          * eps_a: Penalty parameter for soft contact in bilinear contact law
            (only used if bilinear law is choosen).
          * limit_soft_contact: For bilinear contact law, define the
            penetration limit between soft contact (stiffness eps_a),
            and hard contact (stiffness eps_n) - only used if bilinear law is
            choosen.
        """
        if space is None:
            space = ModelingSpace.get_active()
        AssemblyBase.__init__(self, name, space)

        if isinstance(slave_nodes, Mesh):
            slave_nodes = np.unique(slave_nodes.elements)
        self.slave_nodes = slave_nodes
        self.master_nodes = np.unique(surface_mesh.elements)
        self.mesh = surface_mesh

        self.current = self

        self.search_algorithm = search_algorithm.lower()
        if self.search_algorithm == "bucket":
            self.bucket_size = np.sqrt(2) * self._compute_max_edge_size()
            # self.bucket_size = self.mesh.bounding_box.size.max()/10 #bucket size #bounding box includes all nodes
        elif self.search_algorithm == "ipctk":
            self._setup_ipctk_broad_phase(space.ndim)
        elif search_algorithm.lower() != "search_nearest":
            raise (NameError, f"Search alogithm {search_algorithm} not unknown.")

        self.eps_a = 1e5
        """ Penalty parameter for soft contact in bilinear contact law."""

        self.eps_n = 1e7
        # self.eps_nn = 1e5
        """ Contact penalty parameter. """

        self.max_dist = 0.5
        """ Max distance from nodes at which contact is considered."""

        self.clearance = 0.0
        """ Contact clearance to adjust the two surfaces."""

        self.tol = 0.1
        """ Tolerance for possible slide of a node outside an element."""

        self.max_penetration = None

        self.contact_search_once = True
        """It True, only search contact at the begining of iteration. If False, 
        contact is search at each newton raphson iteration (non stable)"""

        self.limit_soft_contact = 0.01  # for bilinear_law
        """ For bilinear contact law, define the penetration limit between soft contact 
        (stiffness eps_a), and hard contact (stiffness eps_n)."""

        self.sv = {}
        """ Dictionary of state variables associated to the associated for the current problem."""
        self.sv_start = {}

        self.normal_law = normal_law.lower()
        """Name of the contact law. The normal law can be run using the "run_normal_law" method."""

    # def __repr__(self):
    #     list_str = ['Contact:']
    #     if self.name != "": list_str.append("name = '{}'".format(self.name))

    #     return "\n".join(list_str)

    def global_search(self):
        if self.search_algorithm == "bucket":
            return self._nearest_node_bucket_sort()
        elif self.search_algorithm == "ipctk":
            return self._nearest_element_ipctk()
        else:  #'search_nearest'
            return self._nearest_node()

    def _compute_max_edge_size(self):
        """Maximum edge length across all master surface elements."""
        elm_crd = self.mesh.nodes[self.mesh.elements]
        n_per_elm = elm_crd.shape[1]
        if n_per_elm == 2:  # lin2
            return np.linalg.norm(elm_crd[:, 1] - elm_crd[:, 0], axis=1).max()
        elif n_per_elm == 3:  # tri3
            return max(
                np.linalg.norm(elm_crd[:, j] - elm_crd[:, i], axis=1).max()
                for i, j in [(0, 1), (1, 2), (2, 0)]
            )
        else:
            raise NotImplementedError(
                f"'{self.mesh.elm_type}' elements are not supported in "
                "contact. Use either 'tri3' or 'lin2' elements."
            )

    def _setup_ipctk_broad_phase(self, ndim):
        """Build ipctk CollisionMesh for broad-phase contact search.

        Slave nodes already in master elements get codimensional
        duplicates so ipctk still generates EV/FV candidates for them.
        """
        try:
            import ipctk

            self._ipctk = ipctk
        except ImportError:
            raise ImportError(
                "The 'ipctk' package is required for search_algorithm='ipctk'. "
                "Install it with: pip install ipctk"
            )

        # Compact vertex set: master ∪ slave
        all_relevant = np.union1d(self.master_nodes, self.slave_nodes)
        n_base = len(all_relevant)

        # Shared slaves need codimensional duplicates for EV/FV candidates
        shared_mask = np.isin(self.slave_nodes, self.master_nodes)
        shared_slaves = self.slave_nodes[shared_mask]

        # Global → compact index mapping
        g2c = np.full(self.mesh.n_nodes, -1, dtype=int)
        g2c[all_relevant] = np.arange(n_base)

        # Compact vertex ID for each slave (shared ones get duplicate IDs)
        slave_cvids = g2c[self.slave_nodes].copy()
        slave_cvids[shared_mask] = n_base + np.arange(shared_mask.sum())

        # Reverse lookup: compact vertex ID → slave index (-1 = not slave)
        n_verts = n_base + len(shared_slaves)
        cvid_to_slave = np.full(n_verts, -1, dtype=int)
        cvid_to_slave[slave_cvids] = np.arange(len(self.slave_nodes))

        # For self-contact: map duplicate IDs back to element-space IDs
        vid_to_orig = np.arange(n_verts, dtype=int)
        if len(shared_slaves):
            vid_to_orig[n_base:] = g2c[shared_slaves]

        self._ipctk_cvid_to_slave = cvid_to_slave
        self._ipctk_vid_to_orig = vid_to_orig
        self._ipctk_compact_elements = g2c[self.mesh.elements]
        self._ipctk_all_relevant = all_relevant
        self._ipctk_shared_slaves = shared_slaves
        self._ipctk_ndim = ndim

        # Build CollisionMesh
        pos = self._ipctk_build_vertices()
        elems = self._ipctk_compact_elements
        if ndim == 2:
            edges, faces = elems, np.empty((0, 3), dtype=int)
        else:
            edges, faces = ipctk.edges(elems), elems
        self._ipctk_cmesh = ipctk.CollisionMesh(pos, edges, faces)
        self._ipctk_inflation = np.sqrt(2) * self._compute_max_edge_size()

    def _ipctk_build_vertices(self):
        """Current vertex positions for ipctk (compact + shared dups, 3D, F-contiguous)."""
        pos = self.mesh.nodes[self._ipctk_all_relevant]
        if len(self._ipctk_shared_slaves):
            pos = np.vstack([pos, self.mesh.nodes[self._ipctk_shared_slaves]])
        if pos.shape[1] == 2:
            pos = np.hstack([pos, np.zeros((len(pos), 1))])
        return np.asfortranarray(pos, dtype=float)

    def _nearest_element_ipctk(self):
        """Broad-phase search using ipctk HashGrid.

        Returns list of candidate master element indices per slave node.
        """
        cands = self._ipctk.Candidates()
        cands.build(
            self._ipctk_cmesh,
            self._ipctk_build_vertices(),
            min(self.max_dist, self._ipctk_inflation),
            broad_phase=self._ipctk.HashGrid(),
        )

        if self._ipctk_ndim == 2:
            pairs = [(c.vertex_id, c.edge_id) for c in cands.ev_candidates]
        else:
            pairs = [(c.vertex_id, c.face_id) for c in cands.fv_candidates]

        result = [[] for _ in range(len(self.slave_nodes))]
        if len(pairs) == 0:
            return result
        vertex_id, elm_id = np.array(pairs, dtype=np.intp).T
        slave_id = self._ipctk_cvid_to_slave[vertex_id]
        keep = slave_id >= 0
        if isinstance(self, SelfContact):
            # skip the elements that contain the slave node itself
            keep &= ~(
                self._ipctk_compact_elements[elm_id]
                == self._ipctk_vid_to_orig[vertex_id][:, np.newaxis]
            ).any(axis=1)
        slave_id = slave_id[keep]
        elm_id = elm_id[keep]
        if len(slave_id) == 0:
            return result

        # group the candidates by slave node, keeping their order
        order = np.argsort(slave_id, kind="stable")
        slave_id = slave_id[order]
        elm_id = elm_id[order].tolist()
        bounds = np.flatnonzero(np.diff(slave_id)) + 1
        starts = np.r_[0, bounds].tolist()
        stops = np.r_[bounds, len(slave_id)].tolist()
        for si, start, stop in zip(slave_id[starts].tolist(), starts, stops):
            result[si] = elm_id[start:stop]
        return result

    def assemble_global_mat(self, compute="all"):
        pass

    def contact_search(self, contact_list={}, update_contact=True):
        nodes = np.asarray(self.slave_nodes)
        surf = self.mesh  # mesh of the surface

        # All the slave nodes are processed at once, on stacks of small
        # arrays: same kernel per item as a loop over the nodes.

        # get the normal surface on the center of the elements for each element on
        # the master sufrace
        elm_ref = get_element(surf.elm_type)(
            1
        )  # 1 gauss point to compute the local base (center of the element)
        # plain ndarray views (avoid the overhead of array subclasses, such as
        # pyvista arrays, in all the following operations)
        crd = np.asarray(surf.nodes)
        surf_elements = np.asarray(surf.elements)
        elm_nodes_crd = crd[surf_elements]

        if surf.elm_type == "lin2":
            local_frame = elm_ref.get_local_frame(
                elm_nodes_crd, elm_ref.get_gp_elm_coordinates(1)
            )

            tangents = local_frame[:, 0, 0]
            normals = local_frame[:, 0, 1]
            dim = 1
            length = np.linalg.norm(
                elm_nodes_crd[:, 1, :] - elm_nodes_crd[:, 0, :], axis=1
            )

        else:  # surf.elm_type == 'tri3':
            tangents = (
                elm_nodes_crd[:, 1:, :] - elm_nodes_crd[:, [0], :]
            )  # tangents[:,i,:] gives the ith tangent axis i in [0,1]. tangents axes are not orthogonal

            normals = np.cross(tangents[:, 0, :], tangents[:, 1, :])
            normals = normals / np.linalg.norm(normals, axis=1).reshape(-1, 1)
            dim = 2
            length = None

        is_axi = self.space.is_axisymmetric

        first_indices_disp = (
            np.array(self.space.get_rank_vector("Disp")).reshape(-1, 1) * surf.n_nodes
        )  # column vector with the first indice for each disp component
        n_dof_contact = (surf_elements.shape[1] + 1) * len(
            first_indices_disp
        )  # number of dof involved in a contact point

        if update_contact:
            # slave nodes (index in nodes) and element in contact
            id_slave, contact_elements, contact_g, vec_xi, shape_func_val = (
                self._select_contact_elements(
                    nodes, crd, elm_ref, elm_nodes_crd, tangents, normals, length, dim
                )
            )
            new_contact_list = dict(zip(nodes[id_slave], contact_elements))
        else:
            id_slave = np.array(
                [i for i, nd in enumerate(nodes) if contact_list.get(nd) is not None],
                dtype=np.intp,
            )
            contact_elements = np.array(
                [contact_list[nd] for nd in nodes[id_slave]], dtype=np.intp
            )
            slave_crd = crd[nodes[id_slave]]
            if dim == 1:  # surf.elm_type == 'lin2':
                vec_xi = (
                    1
                    / length[contact_elements]
                    * (
                        (slave_crd - elm_nodes_crd[contact_elements, 0, :])
                        * tangents[contact_elements]
                    ).sum(axis=1)
                )
            else:  # surf.elm_type == 'tri3':
                # work only for tri3 face
                # need to be corrected
                tg = tangents[contact_elements]
                vec_xi = np.linalg.solve(
                    tg @ tg.transpose([0, 2, 1]),
                    np.sum(
                        (slave_crd - elm_nodes_crd[contact_elements, 0, :])[:, None, :]
                        * tg,
                        axis=2,
                    )[..., np.newaxis],
                )[..., 0]

            # contact points in global coordinates
            shape_func_val = elm_ref.shape_function(vec_xi)
            contact_point = (
                shape_func_val[:, np.newaxis, :] @ elm_nodes_crd[contact_elements]
            )[:, 0, :]

            # algebric distance from the possible elements
            contact_g = (
                (slave_crd - contact_point)[:, np.newaxis, :]
                @ normals[contact_elements][:, :, np.newaxis]
            )[:, 0, 0]

        if len(contact_elements) == 0:
            self.global_matrix = sparse.csr_array(
                (self.space.nvar * surf.n_nodes, self.space.nvar * surf.n_nodes)
            )
            self.global_vector = 0
            return {}  # empty contact_list

        n1 = normals[contact_elements]
        a1 = tangents[contact_elements]

        # Need to build several matrices
        #   - matrix that compute g (algebric normal distance) from u knowing vec_xi (Ns in eq 9.18 p 239)
        contact_nodes = np.column_stack(
            (nodes[id_slave], surf_elements[contact_elements])
        )
        sorted_indices = contact_nodes.argsort(axis=1)

        indices = (
            np.take_along_axis(contact_nodes, sorted_indices, axis=1)[:, np.newaxis, :]
            + first_indices_disp
        ).ravel()

        def _sorted_data(coef, vec):
            # (coef[sorted_indices] * vec.reshape(-1, 1)).ravel() for each contact
            coef = np.take_along_axis(coef, sorted_indices, axis=1)
            return (coef[:, np.newaxis, :] * vec[:, :, np.newaxis]).ravel()

        coef_s = np.column_stack((np.ones(len(contact_elements)), -shape_func_val))
        data_Ns = _sorted_data(coef_s, n1)
        if dim == 1:
            data_Ts = _sorted_data(coef_s, a1)
            coef_0 = np.broadcast_to(np.array([0, -1, 1]), contact_nodes.shape)
            data_N0s = _sorted_data(coef_0, n1)
        else:
            # constant value for tri3
            shape_func_deriv_val = np.array(elm_ref.shape_function_derivative(vec_xi))
            data_T1 = _sorted_data(coef_s, a1[:, 0])
            data_T2 = _sorted_data(coef_s, a1[:, 1])
            zeros = np.zeros((len(contact_elements), 1))
            data_N1 = _sorted_data(np.hstack((zeros, -shape_func_deriv_val[:, 0])), n1)
            data_N2 = _sorted_data(np.hstack((zeros, -shape_func_deriv_val[:, 1])), n1)

        shape = (len(contact_elements), self.space.nvar * surf.n_nodes)
        indptr = np.arange(0, len(indices) + 1, n_dof_contact)
        Ns = sparse.csr_array((data_Ns, indices, indptr), shape=shape)

        if dim == 1:
            N0s = sparse.csr_array((data_N0s, indices, indptr), shape=shape)
            Ts = sparse.csr_array((data_Ts, indices, indptr), shape=shape)
        else:
            T1 = sparse.csr_array((data_T1, indices, indptr), shape=shape)
            T2 = sparse.csr_array((data_T2, indices, indptr), shape=shape)
            N1 = sparse.csr_array((data_N1, indices, indptr), shape=shape)
            N2 = sparse.csr_array((data_N2, indices, indptr), shape=shape)

        # contact_g = contact_g* (contact_g< self.clearance) #remove negative value
        # if (contact_g > self.clearance).any():
        #     print('Warning, contact have been loosing')

        Fcontact, eps = self.run_normal_law(contact_g - self.clearance)

        # Apply 2πr circumferential weighting for axisymmetric problems
        if is_axi:
            axi_weight = 2 * np.pi * crd[nodes[id_slave], 0]
            Fcontact = Fcontact * axi_weight
            if np.isscalar(eps):
                eps = eps * axi_weight
            else:
                eps = eps * axi_weight

        if dim == 1:
            F_div_l = sparse.diags(
                Fcontact / length[contact_elements], format="csr"
            )  # F = -eps*g
            g_div_l = sparse.diags(contact_g / length[contact_elements], format="csr")

            # TO DO integrate 9.36 eq from wriggers 2006 with fd._sparse lib to improve building perfomance

            if not (np.isscalar(eps)):
                mat_eps = sparse.diags(eps, format="csr")
                self.global_matrix = (
                    Ns.T @ mat_eps @ Ns
                    + N0s.T @ F_div_l @ Ts
                    + Ts.T @ F_div_l @ N0s
                    + N0s.T @ (F_div_l @ g_div_l) @ N0s
                )
            else:
                # self.global_matrix = eps*( Ns.T@Ns - N0s.T@g_div_l@Ts -
                #                       Ts.T@g_div_l@N0s - N0s.T@(g_div_l@g_div_l)@N0s)

                self.global_matrix = (
                    eps * Ns.T @ Ns
                    + N0s.T @ F_div_l @ Ts
                    + Ts.T @ F_div_l @ N0s
                    + N0s.T @ (F_div_l @ g_div_l) @ N0s
                )

        else:  # dim== 2
            # mat_eps_g = sparse.diags(contact_g * eps, format='csr')
            mat_eps_g = sparse.csr_array(
                (
                    contact_g * eps,
                    np.arange(len(contact_g)),
                    np.arange(len(contact_g) + 1),
                )
            )
            mat_eps_g_g = sparse.csr_array(
                (contact_g * contact_g * eps, mat_eps_g.indices, mat_eps_g.indptr)
            )
            reciprocal_metric = np.linalg.inv(
                tangents[contact_elements]
                @ tangents[contact_elements].transpose([0, 2, 1])
            )
            # for one el: reciprocal_metric = np.linalg.inv(tangents[el] @ tangents[el].T)   #[[a^11, a^12], [a^21, a^22]]

            # eps should be multiplied by the element surface for better results

            if not (np.isscalar(eps)):
                mat_eps = sparse.diags(eps, format="csr")

                self.global_matrix = (
                    Ns.T @ mat_eps @ Ns
                    - N1.T @ mat_eps_g @ T1
                    - N2.T @ mat_eps_g @ T2
                    - T1.T @ mat_eps_g @ N1
                    - T2.T @ mat_eps_g @ N2
                    - N1.T
                    @ sparse.csr_array(
                        (
                            mat_eps_g_g.data * reciprocal_metric[:, 0, 0],
                            mat_eps_g.indices,
                            mat_eps_g.indptr,
                        )
                    )
                    @ N1
                    - N1.T
                    @ sparse.csr_array(
                        (
                            mat_eps_g_g.data * reciprocal_metric[:, 0, 1],
                            mat_eps_g.indices,
                            mat_eps_g.indptr,
                        )
                    )
                    @ N2
                    - N2.T
                    @ sparse.csr_array(
                        (
                            mat_eps_g_g.data * reciprocal_metric[:, 0, 1],
                            mat_eps_g.indices,
                            mat_eps_g.indptr,
                        )
                    )
                    @ N1
                    - N2.T
                    @ sparse.csr_array(
                        (
                            mat_eps_g_g.data * reciprocal_metric[:, 1, 1],
                            mat_eps_g.indices,
                            mat_eps_g.indptr,
                        )
                    )
                    @ N2
                )
            else:
                self.global_matrix = (
                    eps * Ns.T @ Ns
                    - N1.T @ mat_eps_g @ T1
                    - N2.T @ mat_eps_g @ T2
                    - T1.T @ mat_eps_g @ N1
                    - T2.T @ mat_eps_g @ N2
                    - N1.T
                    @ sparse.csr_array(
                        (
                            mat_eps_g_g.data * reciprocal_metric[:, 0, 0],
                            mat_eps_g.indices,
                            mat_eps_g.indptr,
                        )
                    )
                    @ N1
                    - N1.T
                    @ sparse.csr_array(
                        (
                            mat_eps_g_g.data * reciprocal_metric[:, 0, 1],
                            mat_eps_g.indices,
                            mat_eps_g.indptr,
                        )
                    )
                    @ N2
                    - N2.T
                    @ sparse.csr_array(
                        (
                            mat_eps_g_g.data * reciprocal_metric[:, 0, 1],
                            mat_eps_g.indices,
                            mat_eps_g.indptr,
                        )
                    )
                    @ N1
                    - N2.T
                    @ sparse.csr_array(
                        (
                            mat_eps_g_g.data * reciprocal_metric[:, 1, 1],
                            mat_eps_g.indices,
                            mat_eps_g.indptr,
                        )
                    )
                    @ N2
                )

        self.global_vector = Ns.T @ Fcontact

        # self.sv['contact_elements'] = contact_elements   #would be better to get force at nodes and includes node 2 node
        # self.sv['Fcontact'] = Fcontact
        # self.sv['contact_g'] = contact_g
        # print(contact_elements)
        # print(contact_g)

        if update_contact:
            return new_contact_list
        else:
            return contact_list

        # voir eq 9.35 et 9.36 (page 241 du pdf) avec def 9.18 et 9.19 page 239

    def _select_contact_elements(
        self, nodes, crd, elm_ref, elm_nodes_crd, tangents, normals, length, dim
    ):
        """Search the master element in contact with each slave node.

        Each slave node is projected on all its candidate elements given by
        :py:meth:`global_search` (all slave nodes at once). For each slave
        node, the selected element is the admissible candidate with the
        largest algebraic gap (first one in case of equality).

        Returns
        -------
        id_slave, elements, gaps, vec_xi, shape_func_val: numpy arrays
            Index (in ``nodes``) of the slave nodes in contact, and the
            element, gap, local coordinates and shape function values of
            the associated contact points.
        """
        list_possible_elements = self.global_search()
        n_candidates = np.fromiter(
            (len(pe) for pe in list_possible_elements),
            dtype=np.intp,
            count=len(list_possible_elements),
        )
        if n_candidates.sum() == 0:
            empty = np.zeros(0, dtype=np.intp)
            return empty, empty, np.zeros(0), None, None

        # (slave node, candidate element) pairs, in the slave node order
        pair_slave = np.repeat(np.arange(len(n_candidates)), n_candidates)
        pair_el = np.concatenate(
            [np.asarray(pe, dtype=np.intp) for pe in list_possible_elements if len(pe)]
        )
        slave_crd = crd[nodes[pair_slave]]

        # orthogonal projection on the element plane in node coordinates
        if dim == 1:  # surf.elm_type = 'lin2'
            vec_xi = (
                1
                / length[pair_el]
                * ((slave_crd - elm_nodes_crd[pair_el, 0, :]) * tangents[pair_el]).sum(
                    axis=1
                )
            )
        else:  # surf.elm_type == 'tri3':
            # work only for tri3 face
            vec_xi = np.linalg.solve(
                (tangents[pair_el] @ tangents[pair_el].transpose([0, 2, 1])),
                np.sum(
                    (
                        (slave_crd - elm_nodes_crd[pair_el, 0, :]).reshape(-1, 1, 3)
                        * tangents[pair_el]
                    ),
                    axis=2,
                )[..., np.newaxis],
            )

        # contact points in global coordinates
        shape_func_val = elm_ref.shape_function(vec_xi)
        contact_points = (shape_func_val[:, np.newaxis, :] @ elm_nodes_crd[pair_el])[
            :, 0, :
        ]

        # algebric distance from the possible elements
        g = ((slave_crd - contact_points) * normals[pair_el]).sum(axis=1)

        # element that may be in contact (ie vec_xi inside the element)
        tol = self.tol
        if dim == 1:  # surf.elm_type == 'lin2':
            inside = (vec_xi + tol >= 0) * (vec_xi - tol <= 1)
        else:  # surf.elm_type == 'tri3':
            inside = (
                (vec_xi[:, 0] + tol >= 0)
                * (vec_xi[:, 1] + tol >= 0)
                * (1 - vec_xi[:, 0] - vec_xi[:, 1] + tol >= 0)
            )[:, 0]

        if self.max_penetration is None:
            in_gap = g < self.clearance
        else:
            in_gap = (g < self.clearance) * (g > -self.max_penetration)

        if dim == 1:
            admissible = inside & in_gap
        else:
            # tri3: an element containing the projection is admissible as
            # soon as one candidate of the same slave node is in the gap range.
            has_candidates = n_candidates > 0
            starts = (np.cumsum(n_candidates) - n_candidates)[has_candidates]
            any_in_gap = np.zeros(len(n_candidates), dtype=bool)
            any_in_gap[has_candidates] = np.logical_or.reduceat(in_gap, starts)
            admissible = inside & any_in_gap[pair_slave]

        # for each slave node: first admissible candidate with the largest gap
        # (argmax semantics, nan being the largest value)
        cand = np.flatnonzero(admissible)
        g_cand = g[cand]
        is_nan = np.isnan(g_cand)
        cand = cand[np.lexsort((cand, -g_cand, ~is_nan, pair_slave[cand]))]
        slave_cand = pair_slave[cand]
        best = (
            cand[np.r_[True, slave_cand[1:] != slave_cand[:-1]]] if len(cand) else cand
        )

        return (
            pair_slave[best],
            pair_el[best],
            g[best],
            vec_xi[best],
            shape_func_val[best],
        )

    def set_disp(self, disp):
        if np.isscalar(disp) and disp == 0:
            self.current = self
        else:
            new_crd = self.mesh.nodes + disp.T
            if self.current == self:
                # initialize a new
                new_mesh = copy(self.mesh)
                new_mesh.nodes = new_crd
                new_assembly = copy(self)
                new_assembly.mesh = new_mesh
                self.current = new_assembly
            else:
                self.current.mesh.nodes = new_crd

    def initialize(self, pb):
        # self.update(problem)
        # initialize the contact list
        # self.contact_list_xi = {}
        self.sv["contact_list"] = self.current.contact_search(
            {}, True
        )  # initialize contact state
        self.sv_start = dict(self.sv)
        self.update_contact = True

    def set_start(self, problem):
        # set_start should update the tangent matrix. Here the current state is kept
        # self.sv['contact_list'] = self.current.contact_search({}, True) #initialize contact state
        self.sv_start = dict(
            self.sv
        )  # create a new dict with alias inside (not deep copy)
        self.update_contact = True

    def to_start(self, pb):
        self.set_disp(pb.get_disp())
        self.sv = self.current.sv = dict(self.sv_start)

        self.current.contact_search(
            self.sv["contact_list"], update_contact=False
        )  # initialize global_matrix and global_vector
        # here the tangent matrix is recomputed that may lead to a slight difference from the 1rst attempt.
        # self.current.assemble_global_mat(compute)
        self.update_contact = True

    def update(self, pb, compute="all"):
        self.set_disp(pb.get_disp())
        self.sv["contact_list"] = self.current.contact_search(
            self.sv["contact_list"], self.update_contact
        )
        if self.contact_search_once:
            self.update_contact = False

    def run_normal_law(self, g):
        if self.normal_law == "linear":
            return self._linear_law(g)
        else:  # bilinear law
            return self._bilinear_law(g)

    def _bilinear_law(self, g):
        contact_g = g
        Fc0 = (
            self.eps_a * self.limit_soft_contact
        )  # self.limit_soft_contact should be > 0
        eps = (contact_g <= -self.limit_soft_contact) * self.eps_n + (
            contact_g > -self.limit_soft_contact
        ) * self.eps_a
        Fcontact = (-self.eps_n * (contact_g + self.limit_soft_contact) + Fc0) * (
            contact_g <= -self.limit_soft_contact
        ) - (self.eps_a * contact_g) * (contact_g > -self.limit_soft_contact)

        return Fcontact, eps

    def _linear_law(self, g):
        Fcontact = -self.eps_n * np.array(g)
        return Fcontact, self.eps_n

    def _nearest_node_bucket_sort(self):
        """Find a list of elements that may be in contact for all slave nodes.

        Same as :py:meth:`_nearest_node`, but the nearest master node of a
        slave node is only searched in the neighbor buckets of a regular
        grid (bucket size given by the ``bucket_size`` attribute).
        """
        mesh = self.mesh
        bounding_box = mesh.bounding_box
        w = self.bucket_size
        nearest_neighbors = _bucket_nearest_master(
            np.asarray(mesh.nodes),
            bounding_box[0],
            (bounding_box.size // w).astype(int),  # n of buckets along each direction
            w,
            self.space.ndim,
            self.master_nodes,
            np.asarray(self.slave_nodes),
            self.max_dist,
            (np.array([0, 1]), np.array([-1, 0]), np.array([-1, 0, 1])),
            self_contact=False,
        )
        return self._elements_of_nearest_nodes(nearest_neighbors)

    def _elements_of_nearest_nodes(self, nearest_neighbors):
        """Elements containing the nearest master node of each slave node.

        ``nearest_neighbors[i]`` is the nearest master node of the ith slave
        node, or -1 if no master node is close enough (empty list).
        """
        found = np.flatnonzero(nearest_neighbors != -1)
        possible_elements = [[] for _ in range(len(nearest_neighbors))]
        for i, elm in zip(
            found,
            _elements_of_nodes(
                np.asarray(self.mesh.elements), nearest_neighbors[found].astype(int)
            ),
        ):
            possible_elements[i] = elm
        return possible_elements

    def _nearest_master_nodes(self, exclude_self=False):
        """Nearest master node of all slave nodes.

        Returns the index (in ``master_nodes``) of the nearest master node
        and the corresponding distance for each slave node. The result is
        the one of a brute force search: ``argmin`` of
        ``np.linalg.norm(slave - masters, axis=1)``, ie same distance values
        and lowest index among equidistant nodes. A kd-tree gives a few
        candidate master nodes, that contain all the nodes whose distance
        may be the minimum, and the exact distances of the candidates are
        compared. If ``exclude_self`` is True, the ith master node is ignored
        for the ith slave node (self contact, where both lists are the same).
        """
        crd = np.asarray(self.mesh.nodes)
        slave_crd = crd[np.asarray(self.slave_nodes, dtype=np.intp)]
        master_crd = crd[self.master_nodes]
        n_slave = len(slave_crd)
        n_master = len(master_crd)
        trial_node_indice = np.empty(n_slave, dtype=np.intp)
        min_dist = np.empty(n_slave)

        # The kd-tree needs finite coordinates, and its tie margin is the one
        # of float64 distances: brute force for all the slave nodes otherwise.
        use_tree = (
            crd.dtype == np.float64
            and n_slave > 0
            and np.isfinite(slave_crd).all()
            and np.isfinite(master_crd).all()
        )
        if use_tree:
            k = min(n_master, 8 + int(exclude_self))
            kd_dist, cand = spatial.cKDTree(master_crd).query(slave_crd, k=k)
            kd_dist = kd_dist.reshape(n_slave, k)
            cand = cand.reshape(n_slave, k)
            if exclude_self:
                is_self = cand == np.arange(n_slave)[:, np.newaxis]
                d_ref = np.where(is_self, np.inf, kd_dist).min(axis=1)
            else:
                d_ref = kd_dist[:, 0]
            # Slaves whose kth candidate is as close as the nearest one may
            # have other equidistant master nodes: brute force for these.
            complete = (kd_dist[:, -1] > d_ref * (1 + 1e-8)) | (k == n_master)
        else:
            cand = np.zeros((n_slave, 1), dtype=np.intp)
            is_self = np.zeros((n_slave, 1), dtype=bool)
            complete = np.zeros(n_slave, dtype=bool)

        rows = np.flatnonzero(complete)
        dist = np.linalg.norm(
            slave_crd[rows, np.newaxis, :] - master_crd[cand[rows]], axis=2
        )
        if exclude_self:
            dist[is_self[rows]] = np.inf
        min_dist[rows] = dist.min(axis=1)
        trial_node_indice[rows] = np.where(
            dist == min_dist[rows, np.newaxis], cand[rows], n_master
        ).min(axis=1)

        # brute force for the remaining slave nodes, by blocks to limit the
        # memory
        rows = np.flatnonzero(~complete)
        block = max(1, 2**20 // max(1, n_master))
        for start in range(0, len(rows), block):
            sub = rows[start : start + block]
            dist = np.linalg.norm(slave_crd[sub, np.newaxis, :] - master_crd, axis=2)
            if exclude_self:
                keep = sub < n_master
                dist[np.flatnonzero(keep), sub[keep]] = np.inf
            trial = dist.argmin(axis=1)
            trial_node_indice[sub] = trial
            min_dist[sub] = np.take_along_axis(dist, trial[:, np.newaxis], axis=1)[:, 0]
        return trial_node_indice, min_dist

    def _nearest_node(self):
        """Find a list of elements that may be in contact for all slave nodes.

        This method find the nearest neighbor of all slave nodes and return the
        associated elements if the nearest_neighbor is near enough
        (criterion givien by the max_dist attribute).

        Warning: Master and slave surfaces should be disjoint. If not it SelfContact should be used instead.

        Returns
        -------
        possible_elements: list[list[int]]
        possible_elements[i] is the list of indices of the master surface elements
        that may be in contact with the ith slave nodes
        are given in possible_elements[i]. If no element can be
        in contact, possible_elements[i] is an empty list.

        """
        trial_node_indice, min_dist = self._nearest_master_nodes()
        # to improve performance, ignore contact if distance to the closest
        # node is to high
        nearest_neighbors = np.where(
            min_dist > self.max_dist, -1, self.master_nodes[trial_node_indice]
        )
        return self._elements_of_nearest_nodes(nearest_neighbors)


class SelfContact(Contact):
    """Self contact Assembly (ie contact of a geomtry between itself) based on a node 2 surface formulation"""

    def __init__(
        self,
        mesh: Mesh,
        normal_law: str = "linear",
        extract_surface: bool = False,
        search_algorithm: str = "search_nearest",  #'bucket', #'search_nearest'
        name: str = "Self contact",
        space: ModelingSpace | None = None,
    ):
        """
         Assembly related to surface 2 surface contact, using a node 2 surface formulation
         with a penality method.

         Parameters
         ----------

        mesh: fedoo.Mesh
             Mesh of the geometry where the self contact is created.
             Can be only the mesh of the surface or the mesh of the full object.
             If the full object mesh is given, the extract_surface argument should be set to True.
         normal_law: str in {'linear', 'bilinear'}, default = 'linear'
             Type of contact law for the normal contact.
         name: str
             The name of contact assembly
         space: ModelingSpace
             Modeling space associated to the weakform. If None is specified, the active ModelingSpace is considered.

         Notes
         --------------------------
         Several attributes can be modified to change the contact behavior:
           * eps_n: contact penalty parameter. Need to be adjusted depending of the rigidity of the material in contact.
           * clearance: Contact clearance to adjust the two surfaces.
           * contact_search_once: If True (default) the elmement in contact are only searched at the begining of iteration.
             It avoid oscilations between contact and non contact state during the NR iterations.
           * tol: Tolerance for possible slide of a node outside an element.
           * max_dist: Max distance from nodes at which contact is considered.
           * eps_a: Penalty parameter for soft contact in bilinear contact law (only used if bilinear law is choosen).
           * limit_soft_contact: For bilinear contact law, define the penetration limit between soft contact
             (stiffness eps_a), and hard contact (stiffness eps_n) - only used if bilinear law is choosen.

        """

        if extract_surface:
            mesh = extract_surface_mesh(mesh)

        nodes = np.unique(mesh.elements)

        super().__init__(
            nodes,
            mesh,
            normal_law,
            search_algorithm,
            name=name,
            space=space,
        )

    def _nearest_node(self):
        """Find a list of elements that may be in contact for all slave nodes.

        This method find the nearest neighbor of all slave nodes and return the
        associated elements if the nearest_neighbor is near enough
        (criterion givien by the max_dist attribute).

        Returns
        -------
        possible_elements: list[list[int]]
        possible_elements[i] is the list of indices of the master surface elements
        that may be in contact with the ith slave nodes
        are given in possible_elements[i]. If no element can be
        in contact, possible_elements[i] is an empty list.

        """
        # nearest node without considering the distance of slave_node
        # between itself
        trial_node_indice, min_dist = self._nearest_master_nodes(exclude_self=True)
        slave_nodes = np.asarray(self.slave_nodes)
        close = np.flatnonzero(~(min_dist > self.max_dist))
        nearest_neighbors = self.master_nodes[trial_node_indice[close]]
        elements = np.asarray(self.mesh.elements)
        elm_nearest = _elements_of_nodes(elements, nearest_neighbors)
        elm_slave = _elements_of_nodes(elements, slave_nodes[close])

        # to improve performance, ignore contact if distance to the closest
        # node is to high
        possible_elements = [[] for _ in range(len(slave_nodes))]
        for i, elm_nn, elm_sl in zip(close, elm_nearest, elm_slave):
            # exclude the elements that contain the slave node
            possible_elements[i] = np.setdiff1d(elm_nn, elm_sl, True)
        return possible_elements

    def _nearest_node_bucket_sort(self):
        mesh = self.mesh
        bounding_box = mesh.bounding_box
        w = self.bucket_size
        slave_nodes = np.asarray(self.slave_nodes)
        nearest_neighbors = _bucket_nearest_master(
            np.asarray(mesh.nodes),
            bounding_box[0],
            (bounding_box.size // w).astype(int),  # n of buckets along each direction
            w,
            self.space.ndim,
            self.master_nodes,
            slave_nodes,
            self.max_dist,
            (np.array([0, 1]), np.array([0, -1]), np.array([0, -1, 1])),
            self_contact=True,
        )

        possible_elements = self._elements_of_nearest_nodes(nearest_neighbors)
        # exclusde elements that containts the slave_node
        elements = np.asarray(mesh.elements)
        for i in np.flatnonzero(nearest_neighbors != -1):
            elm = possible_elements[i]
            possible_elements[i] = elm[
                (elements[elm] != slave_nodes[i]).sum(axis=1) != 0
            ]
        return possible_elements
