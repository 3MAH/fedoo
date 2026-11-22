"""Vectorized penalty contact search.

The contact search of :py:class:`fedoo.constraint.Contact` and
:py:class:`fedoo.constraint.SelfContact` processes all the slave nodes at
once. These tests check that it gives exactly (bitwise) the results of the
former per-node implementation, kept below as reference. In the reference
bucket search, the uninitialized ``np.empty`` array of nearest nodes is
replaced by -1 (no nearest node), which is the value now used.
"""

import warnings

import numpy as np
import pytest
from scipy import sparse, spatial

import fedoo as fd
from fedoo.constraint.contact import Contact, SelfContact
from fedoo.lib_elements.element_list import get_element


# ---------------------------------------------------------------------------
# Reference implementation (former per-node loops)
# ---------------------------------------------------------------------------
def _ref_contact_search(self, contact_list={}, update_contact=True):
    nodes = self.slave_nodes
    surf = self.mesh
    if update_contact:
        new_contact_list = {}
    elm_ref = get_element(surf.elm_type)(1)
    elm_nodes_crd = surf.nodes[surf.elements]
    if surf.elm_type == "lin2":
        local_frame = elm_ref.get_local_frame(
            elm_nodes_crd, elm_ref.get_gp_elm_coordinates(1)
        )
        tangents = local_frame[:, 0, 0]
        normals = local_frame[:, 0, 1]
        dim = 1
        length = np.linalg.norm(elm_nodes_crd[:, 1, :] - elm_nodes_crd[:, 0, :], axis=1)
    else:
        tangents = elm_nodes_crd[:, 1:, :] - elm_nodes_crd[:, [0], :]
        normals = np.cross(tangents[:, 0, :], tangents[:, 1, :])
        normals = normals / np.linalg.norm(normals, axis=1).reshape(-1, 1)
        dim = 2
    indices = []
    contact_elements = []
    contact_g = []
    is_axi = self.space.is_axisymmetric
    if is_axi:
        contact_r = []
    data_Ns = []
    if dim == 1:
        data_N0s = []
        data_Ts = []
        data_T0s = []
    else:
        data_N1 = []
        data_N2 = []
        data_T1 = []
        data_T2 = []
    first_indices_disp = (
        np.array(self.space.get_rank_vector("Disp")).reshape(-1, 1) * surf.n_nodes
    )
    n_dof_contact = (surf.elements.shape[1] + 1) * len(first_indices_disp)
    if update_contact:
        list_possible_elements = self.global_search()
    for i_nd, slave_node in enumerate(nodes):
        if update_contact:
            possible_elements = list_possible_elements[i_nd]
            if not len(possible_elements):
                continue
            if dim == 1:
                vec_xi = (
                    1
                    / length[possible_elements]
                    * (
                        (
                            surf.nodes[slave_node]
                            - elm_nodes_crd[possible_elements, 0, :]
                        )
                        * tangents[possible_elements]
                    ).sum(axis=1)
                )
            else:
                vec_xi = np.linalg.solve(
                    tangents[possible_elements]
                    @ tangents[possible_elements].transpose([0, 2, 1]),
                    np.sum(
                        (
                            surf.nodes[slave_node]
                            - elm_nodes_crd[possible_elements, 0, :]
                        ).reshape(-1, 1, 3)
                        * tangents[possible_elements],
                        axis=2,
                    )[..., np.newaxis],
                )
            shape_func_val = elm_ref.shape_function(vec_xi)
            contact_points = (
                shape_func_val[:, np.newaxis, :] @ elm_nodes_crd[possible_elements]
            ).squeeze()
            g = (
                (surf.nodes[slave_node] - contact_points) * normals[possible_elements]
            ).sum(axis=1)
            if self.max_penetration is None:
                if (g > self.clearance).all():
                    continue
            elif np.logical_or(g > self.clearance, g < -self.max_penetration).all():
                continue
            if dim == 1:
                test = (vec_xi + self.tol >= 0) * (vec_xi - self.tol <= 1)
            else:
                test = (
                    (vec_xi[:, 0] + self.tol >= 0)
                    * (vec_xi[:, 1] + self.tol >= 0)
                    * (1 - vec_xi[:, 0] - vec_xi[:, 1] + self.tol >= 0)
                )
            if self.max_penetration is None:
                test = np.where(test * (g < self.clearance))[0]
            else:
                test = np.where(
                    test * (g < self.clearance) * (g > -self.max_penetration)
                )[0]
            if len(test) == 0:
                continue
            if len(test) == 1:
                id_el = test[0]
            else:
                id_el = test[g[test].argmax()]
            g = g[id_el]
            el = possible_elements[id_el]
            vec_xi = vec_xi[id_el]
            shape_func_val = shape_func_val[id_el]
            new_contact_list[slave_node] = el
        else:
            el = contact_list.get(slave_node)
            if el is None:
                continue
            if dim == 1:
                vec_xi = (
                    1
                    / length[el]
                    * (
                        (surf.nodes[slave_node] - elm_nodes_crd[el, 0, :])
                        * tangents[el]
                    ).sum()
                )
            else:
                vec_xi = np.linalg.solve(
                    tangents[el] @ tangents[el].T,
                    np.sum(
                        (surf.nodes[slave_node] - elm_nodes_crd[el, 0, :]).reshape(1, 3)
                        * tangents[el],
                        axis=1,
                    ),
                ).reshape(1, -1)
            shape_func_val = elm_ref.shape_function(vec_xi)[0]
            contact_point = shape_func_val @ elm_nodes_crd[el]
            g = (surf.nodes[slave_node] - contact_point) @ normals[el]
        n1 = normals[el]
        a1 = tangents[el]
        contact_g.append(g)
        contact_elements.append(el)
        if is_axi:
            contact_r.append(surf.nodes[slave_node, 0])
        contact_nodes = np.hstack(([slave_node], surf.elements[el]))
        sorted_indices = contact_nodes.argsort()
        indices.extend(
            list((contact_nodes[sorted_indices] + first_indices_disp).ravel())
        )
        data_Ns.extend(
            list(
                (
                    np.hstack(([1], -shape_func_val))[sorted_indices]
                    * n1.reshape(-1, 1)
                ).ravel()
            )
        )
        if dim == 1:
            data_Ts.extend(
                list(
                    (
                        np.hstack(([1], -shape_func_val))[sorted_indices]
                        * a1.reshape(-1, 1)
                    ).ravel()
                )
            )
            data_N0s.extend(
                list((np.array([0, -1, 1])[sorted_indices] * n1.reshape(-1, 1)).ravel())
            )
            data_T0s.extend(
                list((np.array([0, -1, 1])[sorted_indices] * a1.reshape(-1, 1)).ravel())
            )
        else:
            shape_func_deriv_val = elm_ref.shape_function_derivative([vec_xi])[0]
            data_T1.extend(
                list(
                    (
                        np.hstack(([1], -shape_func_val))[sorted_indices]
                        * a1[0].reshape(-1, 1)
                    ).ravel()
                )
            )
            data_T2.extend(
                list(
                    (
                        np.hstack(([1], -shape_func_val))[sorted_indices]
                        * a1[1].reshape(-1, 1)
                    ).ravel()
                )
            )
            data_N1.extend(
                list(
                    (
                        np.hstack(([0], -shape_func_deriv_val[0]))[sorted_indices]
                        * n1.reshape(-1, 1)
                    ).ravel()
                )
            )
            data_N2.extend(
                list(
                    (
                        np.hstack(([0], -shape_func_deriv_val[1]))[sorted_indices]
                        * n1.reshape(-1, 1)
                    ).ravel()
                )
            )
    if not contact_elements:
        self.global_matrix = sparse.csr_array(
            (self.space.nvar * surf.n_nodes, self.space.nvar * surf.n_nodes)
        )
        self.global_vector = 0
        return {}
    shape = (len(contact_elements), self.space.nvar * surf.n_nodes)
    indptr = np.arange(0, len(indices) + 1, n_dof_contact)
    Ns = sparse.csr_array((data_Ns, indices, indptr), shape=shape)
    if dim == 1:
        N0s = sparse.csr_array((data_N0s, indices, indptr), shape=shape)
        Ts = sparse.csr_array((data_Ts, indices, indptr), shape=shape)
        T0s = sparse.csr_array((data_Ns, indices, indptr), shape=shape)
    else:
        T1 = sparse.csr_array((data_T1, indices, indptr), shape=shape)
        T2 = sparse.csr_array((data_T2, indices, indptr), shape=shape)
        N1 = sparse.csr_array((data_N1, indices, indptr), shape=shape)
        N2 = sparse.csr_array((data_N2, indices, indptr), shape=shape)
    contact_g = np.array(contact_g)
    Fcontact, eps = self.run_normal_law(contact_g - self.clearance)
    if is_axi:
        axi_weight = 2 * np.pi * np.array(contact_r)
        Fcontact = Fcontact * axi_weight
        if np.isscalar(eps):
            eps = eps * axi_weight
        else:
            eps = eps * axi_weight
    if dim == 1:
        F_div_l = sparse.diags(Fcontact / length[contact_elements], format="csr")
        g_div_l = sparse.diags(contact_g / length[contact_elements], format="csr")
        if not np.isscalar(eps):
            mat_eps = sparse.diags(eps, format="csr")
            self.global_matrix = (
                Ns.T @ mat_eps @ Ns
                + N0s.T @ F_div_l @ Ts
                + Ts.T @ F_div_l @ N0s
                + N0s.T @ (F_div_l @ g_div_l) @ N0s
            )
        else:
            self.global_matrix = (
                eps * Ns.T @ Ns
                + N0s.T @ F_div_l @ Ts
                + Ts.T @ F_div_l @ N0s
                + N0s.T @ (F_div_l @ g_div_l) @ N0s
            )
    else:
        mat_eps_g = sparse.csr_array(
            (contact_g * eps, np.arange(len(contact_g)), np.arange(len(contact_g) + 1))
        )
        mat_eps_g_g = sparse.csr_array(
            (contact_g * contact_g * eps, mat_eps_g.indices, mat_eps_g.indptr)
        )
        reciprocal_metric = np.linalg.inv(
            tangents[contact_elements] @ tangents[contact_elements].transpose([0, 2, 1])
        )
        if not np.isscalar(eps):
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
    if update_contact:
        return new_contact_list
    else:
        return contact_list


def _ref_nearest_node(self):
    possible_elements = []
    for slave_node in self.slave_nodes:
        dist_slave_nodes = np.linalg.norm(
            self.mesh.nodes[slave_node] - self.mesh.nodes[self.master_nodes], axis=1
        )
        trial_node_indice = dist_slave_nodes.argmin()
        if dist_slave_nodes[trial_node_indice] > self.max_dist:
            possible_elements.append([])
        else:
            nearest_neighbors = self.master_nodes[trial_node_indice]
            possible_elements.append(
                (self.mesh.elements == nearest_neighbors).sum(axis=1).nonzero()[0]
            )
    return possible_elements


def _ref_nearest_node_bucket_sort(self):
    ndim = self.space.ndim
    mesh = self.mesh
    master_nodes = self.master_nodes
    slave_nodes = self.slave_nodes
    max_dist = self.max_dist
    w = self.bucket_size
    n_buckets = (mesh.bounding_box.size // w).astype(int)
    bucket_crd = ((mesh.nodes[master_nodes] - mesh.bounding_box[0]) / w).astype(int)
    if ndim == 2:
        bucket_id = bucket_crd[:, 0] + bucket_crd[:, 1] * n_buckets[0]
    else:
        bucket_id = (
            bucket_crd[:, 0]
            + bucket_crd[:, 1] * n_buckets[0]
            + bucket_crd[:, 2] * n_buckets[0] * n_buckets[1]
        )
    sorted_indices = bucket_id.argsort()
    temp = [0] + [
        i
        for i in range(1, len(bucket_id))
        if bucket_id[sorted_indices[i]] != bucket_id[sorted_indices[i - 1]]
    ]
    bucket_master = {
        bucket_id[sorted_indices[temp[i]]]: master_nodes[
            sorted_indices[temp[i] : temp[i + 1]]
        ]
        for i in range(0, len(temp) - 1)
    }
    bucket_crd = ((mesh.nodes[slave_nodes] - mesh.bounding_box[0]) / w).astype(int)
    if ndim == 2:
        bucket_id = bucket_crd[:, 0] + bucket_crd[:, 1] * n_buckets[0]
    else:
        bucket_id = (
            bucket_crd[:, 0]
            + bucket_crd[:, 1] * n_buckets[0]
            + bucket_crd[:, 2] * n_buckets[0] * n_buckets[1]
        )
    sorted_indices = bucket_id.argsort()
    temp = [0] + [
        i
        for i in range(1, len(bucket_id))
        if bucket_id[sorted_indices[i]] != bucket_id[sorted_indices[i - 1]]
    ]
    bucket_crd = {
        bucket_id[sorted_indices[temp[i]]]: bucket_crd[sorted_indices[temp[i]]]
        for i in range(0, len(temp) - 1)
    }
    bucket = {
        bucket_id[sorted_indices[temp[i]]]: sorted_indices[temp[i] : temp[i + 1]]
        for i in range(0, len(temp) - 1)
    }
    side_m = np.array([0, 1])
    side_p = np.array([-1, 0])
    inside = np.array([-1, 0, 1])
    array_neighbors = {
        i: [
            side_m
            if bucket_crd[i][dd] == 0
            else side_p
            if bucket_crd[i][dd] == n_buckets[dd] - 1
            else inside
            for dd in range(ndim)
        ]
        for i in bucket
    }
    if ndim == 2:
        neighbors_buckets = {
            i: (
                i
                + array_neighbors[i][0]
                + n_buckets[0] * array_neighbors[i][1].reshape(-1, 1)
            ).flatten()
            for i in bucket
        }
    else:
        neighbors_buckets = {
            i: (
                i
                + array_neighbors[i][0]
                + n_buckets[0] * array_neighbors[i][1].reshape(-1, 1)
                + n_buckets[0] * n_buckets[1] * array_neighbors[i][2].reshape(-1, 1, 1)
            ).flatten()
            for i in bucket
        }
    nearest_neighbors = np.full(len(slave_nodes), -1.0)
    for id_b in bucket:
        neighbor_master_nodes = np.array(
            sum([list(bucket_master.get(i, [])) for i in neighbors_buckets[id_b]], [])
        )
        if len(neighbor_master_nodes) == 0:
            continue
        dist_slave_nodes = spatial.distance_matrix(
            mesh.nodes[slave_nodes[bucket[id_b]]], mesh.nodes[neighbor_master_nodes]
        )
        trial_node_indice = dist_slave_nodes.argmin(axis=1)
        min_dist = np.take_along_axis(
            dist_slave_nodes, np.expand_dims(trial_node_indice, axis=-1), axis=-1
        ).ravel()
        nearest_neighbors[bucket[id_b]] = -1 + (min_dist <= max_dist) * (
            neighbor_master_nodes[trial_node_indice] + 1
        )
    return [
        [] if nn == -1 else (mesh.elements == nn).sum(axis=1).nonzero()[0]
        for nn in nearest_neighbors
    ]


def _ref_nearest_element_ipctk(self):
    cands = self._ipctk.Candidates()
    cands.build(
        self._ipctk_cmesh,
        self._ipctk_build_vertices(),
        min(self.max_dist, self._ipctk_inflation),
        broad_phase=self._ipctk.HashGrid(),
    )
    elems = self._ipctk_compact_elements
    cvid_to_slave = self._ipctk_cvid_to_slave
    vid_to_orig = self._ipctk_vid_to_orig
    is_self = isinstance(self, SelfContact)
    result = [[] for _ in range(len(self.slave_nodes))]
    if self._ipctk_ndim == 2:
        for c in cands.ev_candidates:
            si = cvid_to_slave[c.vertex_id]
            if si < 0:
                continue
            if is_self and vid_to_orig[c.vertex_id] in elems[c.edge_id]:
                continue
            result[si].append(c.edge_id)
    else:
        for c in cands.fv_candidates:
            si = cvid_to_slave[c.vertex_id]
            if si < 0:
                continue
            if is_self and vid_to_orig[c.vertex_id] in elems[c.face_id]:
                continue
            result[si].append(c.face_id)
    return result


def _ref_self_nearest_node(self):
    possible_elements = []
    for id_nd, slave_node in enumerate(self.slave_nodes):
        dist_slave_nodes = np.linalg.norm(
            self.mesh.nodes[slave_node] - self.mesh.nodes[self.master_nodes], axis=1
        )
        trial_nodes_indices = []
        if id_nd != 0:
            trial_nodes_indices.append(dist_slave_nodes[:id_nd].argmin())
        if id_nd != len(self.master_nodes) - 1:
            trial_nodes_indices.append(
                dist_slave_nodes[id_nd + 1 :].argmin() + id_nd + 1
            )
        trial_node_indice = trial_nodes_indices[
            np.argmin(dist_slave_nodes[trial_nodes_indices])
        ]
        if dist_slave_nodes[trial_node_indice] > self.max_dist:
            possible_elements.append([])
        else:
            nearest_neighbors = self.master_nodes[trial_node_indice]
            elm_with_slave_nd = (
                (self.mesh.elements == slave_node).sum(axis=1).nonzero()[0]
            )
            possible_elements.append(
                np.setdiff1d(
                    (self.mesh.elements == nearest_neighbors).sum(axis=1).nonzero()[0],
                    elm_with_slave_nd,
                    True,
                )
            )
    return possible_elements


def _ref_self_nearest_node_bucket_sort(self):
    ndim = self.space.ndim
    mesh = self.mesh
    master_nodes = self.master_nodes
    slave_nodes = self.slave_nodes
    max_dist = self.max_dist
    w = self.bucket_size
    n_buckets = (mesh.bounding_box.size // w).astype(int)
    bucket_crd = ((mesh.nodes[master_nodes] - mesh.bounding_box[0]) / w).astype(int)
    if ndim == 2:
        bucket_id = bucket_crd[:, 0] + bucket_crd[:, 1] * n_buckets[0]
    else:
        bucket_id = (
            bucket_crd[:, 0]
            + bucket_crd[:, 1] * n_buckets[0]
            + bucket_crd[:, 2] * n_buckets[0] * n_buckets[1]
        )
    sorted_indices = bucket_id.argsort()
    temp = [0] + [
        i
        for i in range(1, len(bucket_id))
        if bucket_id[sorted_indices[i]] != bucket_id[sorted_indices[i - 1]]
    ]
    bucket_master = {
        bucket_id[sorted_indices[temp[i]]]: master_nodes[
            sorted_indices[temp[i] : temp[i + 1]]
        ]
        for i in range(0, len(temp) - 1)
    }
    bucket_crd = ((mesh.nodes[slave_nodes] - mesh.bounding_box[0]) / w).astype(int)
    if ndim == 2:
        bucket_id = bucket_crd[:, 0] + bucket_crd[:, 1] * n_buckets[0]
    else:
        bucket_id = (
            bucket_crd[:, 0]
            + bucket_crd[:, 1] * n_buckets[0]
            + bucket_crd[:, 2] * n_buckets[0] * n_buckets[1]
        )
    sorted_indices = bucket_id.argsort()
    temp = [0] + [
        i
        for i in range(1, len(bucket_id))
        if bucket_id[sorted_indices[i]] != bucket_id[sorted_indices[i - 1]]
    ]
    bucket_crd = {
        bucket_id[sorted_indices[temp[i]]]: bucket_crd[sorted_indices[temp[i]]]
        for i in range(0, len(temp) - 1)
    }
    bucket = {
        bucket_id[sorted_indices[temp[i]]]: sorted_indices[temp[i] : temp[i + 1]]
        for i in range(0, len(temp) - 1)
    }
    side_m = np.array([0, 1])
    side_p = np.array([0, -1])
    inside = np.array([0, -1, 1])
    array_neighbors = {
        i: [
            side_m
            if bucket_crd[i][dd] == 0
            else side_p
            if bucket_crd[i][dd] == n_buckets[dd] - 1
            else inside
            for dd in range(ndim)
        ]
        for i in bucket
    }
    if ndim == 2:
        neighbors_buckets = {
            i: (
                i
                + array_neighbors[i][0]
                + n_buckets[0] * array_neighbors[i][1].reshape(-1, 1)
            ).flatten()
            for i in bucket
        }
    else:
        neighbors_buckets = {
            i: (
                i
                + array_neighbors[i][0]
                + n_buckets[0] * array_neighbors[i][1].reshape(-1, 1)
                + n_buckets[0] * n_buckets[1] * array_neighbors[i][2].reshape(-1, 1, 1)
            ).flatten()
            for i in bucket
        }
    nearest_neighbors = np.full(len(slave_nodes), -1.0)
    for id_b in bucket:
        neighbor_master_nodes = np.array(
            sum([list(bucket_master.get(i, [])) for i in neighbors_buckets[id_b]], [])
        )
        if len(neighbor_master_nodes) == 0:
            continue
        dist_slave_nodes = spatial.distance_matrix(
            mesh.nodes[slave_nodes[bucket[id_b]]], mesh.nodes[neighbor_master_nodes]
        )
        np.fill_diagonal(dist_slave_nodes, np.inf)
        trial_node_indice = dist_slave_nodes.argmin(axis=1)
        min_dist = np.take_along_axis(
            dist_slave_nodes, np.expand_dims(trial_node_indice, axis=-1), axis=-1
        ).ravel()
        nearest_neighbors[bucket[id_b]] = -1 + (min_dist <= max_dist) * (
            neighbor_master_nodes[trial_node_indice] + 1
        )
    return [
        []
        if nn == -1
        else (
            (mesh.elements == nn).sum(axis=1) * (mesh.elements != slave_nd).sum(axis=1)
        ).nonzero()[0]
        for slave_nd, nn in zip(slave_nodes, nearest_neighbors)
    ]


class _RefContact(Contact):
    contact_search = _ref_contact_search
    _nearest_node = _ref_nearest_node
    _nearest_node_bucket_sort = _ref_nearest_node_bucket_sort
    _nearest_element_ipctk = _ref_nearest_element_ipctk


class _RefSelfContact(SelfContact):
    contact_search = _ref_contact_search
    _nearest_node = _ref_self_nearest_node
    _nearest_node_bucket_sort = _ref_self_nearest_node_bucket_sort
    _nearest_element_ipctk = _ref_nearest_element_ipctk


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------
def _search(contact, disp1, disp2):
    """Global search and contact search (update, fixed and update again)."""
    out = []
    contact.set_disp(disp1)
    cur = contact.current
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        out.append([np.asarray(p, dtype=int) for p in cur.global_search()])
        cl = cur.contact_search({}, True)
        out.append(_state(cur, cl))
        contact.set_disp(disp2)
        cl2 = contact.current.contact_search(cl, False)
        out.append(_state(contact.current, cl2))
        cl3 = contact.current.contact_search(cl, True)
        out.append(_state(contact.current, cl3))
    return out


def _state(cur, contact_list):
    K = cur.global_matrix.tocsr()
    return (
        {int(k): int(v) for k, v in contact_list.items()},
        K.data.copy(),
        K.indices.copy(),
        K.indptr.copy(),
        np.atleast_1d(np.asarray(cur.global_vector, dtype=float)).copy(),
    )


def _assert_same(new, ref):
    lpe_new, lpe_ref = new[0], ref[0]
    assert len(lpe_new) == len(lpe_ref)
    for a, b in zip(lpe_new, lpe_ref):
        assert np.array_equal(a, b)
    n_contacts = 0
    for s_new, s_ref in zip(new[1:], ref[1:]):
        assert s_new[0] == s_ref[0]
        n_contacts += len(s_ref[0])
        for a, b in zip(s_new[1:], s_ref[1:]):
            assert np.array_equal(a, b)
    assert n_contacts > 0  # the case does test some contacts


def _make_pair(cls_new, cls_ref, *args, setup=None, **kargs):
    pair = []
    for cls in (cls_new, cls_ref):
        c = cls(*args, **kargs)
        if setup is not None:
            setup(c)
        pair.append(c)
    return pair


_ALGORITHMS = ["bucket", "search_nearest"]
try:
    import ipctk  # noqa: F401

    _ALGORITHMS.append("ipctk")
except ImportError:
    pass


def _settings(variant):
    def setup(c):
        if variant == 1:
            c.clearance = 0.002
            c.tol = 0.05
        elif variant == 2:
            c.max_penetration = 0.01

    return setup


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("space", ["2D", "2Daxi"])
@pytest.mark.parametrize("algorithm", _ALGORITHMS)
@pytest.mark.parametrize("law", ["linear", "bilinear"])
@pytest.mark.parametrize("variant", [0, 1, 2])
def test_contact_2d_matches_reference(space, algorithm, law, variant):
    fd.ModelingSpace(space)
    fd.weakform.StressEquilibrium(fd.constitutivelaw.ElasticIsotrop(1.0, 0.3))
    mesh_rect = fd.mesh.rectangle_mesh(
        nx=11, ny=21, x_min=0, x_max=1, y_min=0, y_max=1, elm_type="quad4"
    )
    mesh_rect.element_sets["rect"] = np.arange(mesh_rect.n_elements)
    mesh_disk = fd.mesh.disk_mesh(radius=0.5, nr=8, nt=8, elm_type="quad4")
    mesh_disk.nodes += np.array([1.5, 0.48])
    mesh_disk.element_sets["disk"] = np.arange(mesh_disk.n_elements)
    mesh = fd.Mesh.stack(mesh_rect, mesh_disk)
    rect_nodes = np.unique(mesh.extract_elements("rect").elements)
    nodes_right = np.intersect1d(mesh.find_nodes("X", 1), rect_nodes)
    surf = fd.mesh.extract_surface(mesh.extract_elements("disk"))
    disk_nodes = np.setdiff1d(np.arange(mesh.n_nodes), rect_nodes)

    def setup(c):
        c.max_dist = 1
        c.eps_n = 5e5
        _settings(variant)(c)

    new, ref = _make_pair(
        Contact, _RefContact, nodes_right, surf, law, algorithm, setup=setup
    )
    rng = np.random.default_rng(variant)
    d1 = np.zeros((2, mesh.n_nodes))
    d1[0, disk_nodes] = -0.03
    d1 += 1e-3 * rng.standard_normal(d1.shape)
    d2 = d1 + 2e-3 * rng.standard_normal(d1.shape)
    _assert_same(_search(new, d1, d2), _search(ref, d1, d2))


@pytest.mark.parametrize("algorithm", _ALGORITHMS)
def test_self_contact_2d_matches_reference(algorithm):
    fd.ModelingSpace("2D")
    fd.weakform.StressEquilibrium(fd.constitutivelaw.ElasticIsotrop(1.0, 0.3))
    mesh = fd.mesh.hole_plate_mesh(nr=11, nt=11, length=100, height=100, radius=45)
    surf = fd.mesh.extract_surface(mesh)

    def setup(c):
        c.max_dist = 5

    new, ref = _make_pair(
        SelfContact,
        _RefSelfContact,
        surf,
        "linear",
        search_algorithm=algorithm,
        setup=setup,
    )
    rng = np.random.default_rng(1)
    y = mesh.nodes[:, 1]
    d1 = np.zeros((2, mesh.n_nodes))
    d1[1] = -0.45 * (y - y.min())
    d1 += 0.05 * rng.standard_normal(d1.shape)
    d2 = d1 + 0.05 * rng.standard_normal(d1.shape)
    _assert_same(_search(new, d1, d2), _search(ref, d1, d2))


@pytest.mark.parametrize("algorithm", _ALGORITHMS)
@pytest.mark.parametrize("variant", [0, 1, 2])
def test_contact_3d_matches_reference(algorithm, variant):
    fd.ModelingSpace("3D")
    fd.weakform.StressEquilibrium(fd.constitutivelaw.ElasticIsotrop(1.0, 0.3))
    box1 = fd.mesh.box_mesh(
        nx=7,
        ny=7,
        nz=3,
        x_min=0,
        x_max=1,
        y_min=0,
        y_max=1,
        z_min=0,
        z_max=0.5,
        elm_type="hex8",
    )
    box2 = fd.mesh.box_mesh(
        nx=5,
        ny=5,
        nz=3,
        x_min=0.2,
        x_max=0.8,
        y_min=0.2,
        y_max=0.8,
        z_min=0.5,
        z_max=0.9,
        elm_type="hex8",
    )
    mesh = fd.Mesh.stack(box1, box2)
    n1 = box1.n_nodes
    surf = fd.mesh.extract_surface(
        mesh.extract_elements(np.arange(box1.n_elements)), quad2tri=True
    )
    slaves = np.arange(n1, mesh.n_nodes)
    slaves = slaves[np.isclose(mesh.nodes[slaves, 2], 0.5)]

    def setup(c):
        c.max_dist = 0.3
        c.eps_n = 1e5
        _settings(variant)(c)

    new, ref = _make_pair(
        Contact, _RefContact, slaves, surf, "bilinear", algorithm, setup=setup
    )
    rng = np.random.default_rng(variant)
    d1 = np.zeros((3, mesh.n_nodes))
    d1[2, n1:] = -0.01
    d1 += 2e-3 * rng.standard_normal(d1.shape)
    d2 = d1 + 2e-3 * rng.standard_normal(d1.shape)
    _assert_same(_search(new, d1, d2), _search(ref, d1, d2))


@pytest.mark.parametrize("algorithm", _ALGORITHMS)
def test_self_contact_3d_matches_reference(algorithm):
    fd.ModelingSpace("3D")
    fd.weakform.StressEquilibrium(fd.constitutivelaw.ElasticIsotrop(1.0, 0.3))
    mesh = fd.mesh.box_mesh(nx=6, ny=6, nz=6, elm_type="hex8")
    surf = fd.mesh.extract_surface(mesh, quad2tri=True)

    def setup(c):
        c.max_dist = 0.5

    new, ref = _make_pair(
        SelfContact,
        _RefSelfContact,
        surf,
        "linear",
        search_algorithm=algorithm,
        setup=setup,
    )
    rng = np.random.default_rng(2)
    z = mesh.nodes[:, 2]
    d1 = np.zeros((3, mesh.n_nodes))
    d1[2] = -0.9 * (z - z.min())
    d1 += 0.02 * rng.standard_normal(d1.shape)
    d2 = d1 + 0.01 * rng.standard_normal(d1.shape)
    _assert_same(_search(new, d1, d2), _search(ref, d1, d2))


@pytest.mark.parametrize("exclude_self", [False, True])
def test_nearest_master_nodes_ties(exclude_self):
    """Equidistant nodes (regular grid): lowest index, as a brute force."""
    fd.ModelingSpace("3D")
    fd.weakform.StressEquilibrium(fd.constitutivelaw.ElasticIsotrop(1.0, 0.3))
    mesh = fd.mesh.box_mesh(nx=5, ny=5, nz=5, elm_type="hex8")
    surf = fd.mesh.extract_surface(mesh, quad2tri=True)
    if exclude_self:
        c = SelfContact(surf, search_algorithm="search_nearest")
    else:
        slaves = np.arange(mesh.n_nodes)  # interior nodes too
        c = Contact(slaves, surf, search_algorithm="search_nearest")
    trial, min_dist = c._nearest_master_nodes(exclude_self)

    crd = mesh.nodes
    for i, nd in enumerate(np.asarray(c.slave_nodes)):
        dist = np.linalg.norm(crd[nd] - crd[c.master_nodes], axis=1)
        if exclude_self:
            dist[i] = np.inf
        assert trial[i] == dist.argmin()
        assert min_dist[i] == dist[dist.argmin()]


@pytest.mark.parametrize("case", ["nan", "float32"])
@pytest.mark.parametrize("algorithm", ["bucket", "search_nearest"])
def test_global_search_without_kdtree_matches_reference(case, algorithm):
    """Non finite or single precision coordinates: same candidates as before."""
    fd.ModelingSpace("3D")
    fd.weakform.StressEquilibrium(fd.constitutivelaw.ElasticIsotrop(1.0, 0.3))
    mesh = fd.mesh.box_mesh(nx=5, ny=5, nz=5, elm_type="hex8")
    surf = fd.mesh.extract_surface(mesh, quad2tri=True)
    if case == "float32":
        surf.nodes = surf.nodes.astype(np.float32)
    elif algorithm == "search_nearest":
        surf.nodes = surf.nodes.copy()
        surf.nodes[3, 1] = np.nan  # a diverged iterate must not raise
    new, ref = _make_pair(
        SelfContact, _RefSelfContact, surf, search_algorithm=algorithm
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_new, res_ref = new.global_search(), ref.global_search()
    assert len(res_new) == len(res_ref)
    for a, b in zip(res_new, res_ref):
        assert np.array_equal(np.asarray(a, dtype=int), np.asarray(b, dtype=int))
