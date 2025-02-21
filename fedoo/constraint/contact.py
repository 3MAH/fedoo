from __future__ import annotations

# from fedoo.core.base   import ProblemBase
import numpy as np
from fedoo.core.base import AssemblyBase

# from fedoo.core.boundary_conditions import BCBase, MPC, ListBC
from fedoo.core.base import ProblemBase
from fedoo.core.mesh import MeshBase, Mesh
from fedoo.lib_elements.element_list import get_element
from scipy import sparse, spatial
from fedoo.core.modelingspace import ModelingSpace
from fedoo.mesh import extract_surface as extract_surface_mesh
from copy import copy

import time


class Contact(AssemblyBase):
    """Contact Assembly based on a node 2 surface formulation"""

    def __init__(
        self,
        slave_nodes: list[int] | Mesh,
        surface_mesh: Mesh,
        normal_law: str = "linear",
        search_algorithm: str = "bucket",  #'bucket', #'search_nearest'
        space: ModelingSpace | None = None,
        name: str = "Contact nodes 2 surface",
    ):
        """
        Assembly related to surface 2 surface contact, using a node 2 surface formulation
        with a penality method.

        Parameters
        ----------

        slave_nodes: numpy array of int or Mesh
            List of nodes indices: nodes that belong to the slave surface
            or Mesh of the slave surface (in this case, the slave node indices are
            extracted).
        surface_mesh: fedoo.Mesh
            Mesh of the master surface
        normal_law: str in {'linear', 'bilinear'}, default = 'linear'
            Type of contact law for the normal contact.
        space: ModelingSpace
            Modeling space associated to the weakform. If None is specified, the active ModelingSpace is considered.
        name: str
            The name of contact assembly

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
            # by default bucket_size should be set to max sqrt(2)*max(edge_size)
            if self.mesh.elements.shape[1] == 2:
                max_edge_size = np.linalg.norm(
                    self.mesh.nodes[self.mesh.elements[:, 1]]
                    - self.mesh.nodes[self.mesh.elements[:, 0]],
                    axis=1,
                ).max()
            elif self.mesh.elements.shape[1] == 3:
                max_edge_size = np.max(
                    [
                        np.linalg.norm(
                            self.mesh.nodes[self.mesh.elements[:, ind[1]]]
                            - self.mesh.nodes[self.mesh.elements[:, ind[0]]],
                            axis=1,
                        ).max()
                        for ind in [[0, 1], [1, 2], [2, 0]]
                    ]
                )
            else:
                return NotImplemented
            self.bucket_size = np.sqrt(2) * max_edge_size
            # self.bucket_size = self.mesh.bounding_box.size.max()/10 #bucket size #bounding box includes all nodes
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
        # self.bc_type = 'Contact'
        # BCBase.__init__(self, name)
        # self._update_during_inc = 1

        self.normal_law = normal_law.lower()
        """Name of the contact law. The normal law can be run using the "run_normal_law" method."""

    # def __repr__(self):
    #     list_str = ['Contact:']
    #     if self.name != "": list_str.append("name = '{}'".format(self.name))

    #     return "\n".join(list_str)

    def global_search(self):
        if self.search_algorithm == "bucket":
            return self._nearest_node_bucket_sort()
        else:  #'search_nearest'
            return self._nearest_node()

    def assemble_global_mat(self, compute="all"):
        pass

    def contact_search(self, contact_list={}, update_contact=True):
        nodes = self.slave_nodes
        surf = self.mesh  # mesh of the surface
        if update_contact:
            # update contact connection
            new_contact_list = {}

        # look for contact
        # brut force, compute the distance between nodes and all elements,
        # no vectorization, not optimized

        # get the normal surface on the center of the elements for each element on
        # the master sufrace
        elm_ref = get_element(surf.elm_type)(
            1
        )  # 1 gauss point to compute the local base (center of the element)
        elm_nodes_crd = surf.nodes[surf.elements]

        if surf.elm_type == "lin2":
            local_frame = elm_ref.GetLocalFrame(
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

        indices = []
        contact_elements = []
        contact_g = []

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
        )  # column vector with the first indice for each disp component
        n_dof_contact = (surf.elements.shape[1] + 1) * len(
            first_indices_disp
        )  # number of dof involved in a contact point

        list_nodes = np.unique(surf.elements)

        if update_contact:
            t0 = time.time()
            list_possible_elements = self.global_search()
            # print('temps: ', time.time()-t0)

        for i_nd, slave_node in enumerate(nodes):
            # if self.no_slip and slave_node in contact_list:
            #     el = contact_list.get(slave_node) #read the element in contact
            #     #vec_xi remain the same. Contact isn't allowed to slip along the element:
            #     vec_xi = self.contact_list_xi[slave_node]

            #     #contact points in global coordinates
            #     shape_func_val = elm_ref.ShapeFunction(vec_xi)[0]
            #     contact_point = (shape_func_val @ elm_nodes_crd[el])

            #     #algebric distance from the possible elements
            #     g = (surf.nodes[slave_node] - contact_point) @ normals[el]

            #     if update_contact:
            #         if (self.no_contact_loss or g<self.clearance):
            #             new_contact_list[slave_node] = el

            #         else: #contact_loss - slave_node no more in contact
            #             del self.contact_list_xi[slave_node]
            #             continue

            #     # if g>self.clearance: g = self.clearance

            #     n1 = normals[el]
            #     a1 = tangents[el]

            # elif update_contact:

            if update_contact:
                possible_elements = list_possible_elements[i_nd]

                if not (len(possible_elements)):  # not(possible_elements):
                    # no convenient nodes found (empty list)
                    continue

                # orthogonal projection on the element plane in node coordinates
                if dim == 1:  # surf.elm_type = 'lin2'
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
                else:  # surf.elm_type == 'tri3':
                    # work only for tri3 face
                    vec_xi = np.linalg.solve(
                        (
                            tangents[possible_elements]
                            @ tangents[possible_elements].transpose([0, 2, 1])
                        ),
                        np.sum(
                            (
                                (
                                    surf.nodes[slave_node]
                                    - elm_nodes_crd[possible_elements, 0, :]
                                ).reshape(-1, 1, 3)
                                * tangents[possible_elements]
                            ),
                            axis=2,
                        ),
                    )

                # contact points in global coordinates
                shape_func_val = elm_ref.ShapeFunction(vec_xi)
                contact_points = (
                    shape_func_val[:, np.newaxis, :] @ elm_nodes_crd[possible_elements]
                ).squeeze()

                # algebric distance from the possible elements
                g = (
                    (surf.nodes[slave_node] - contact_points)
                    * normals[possible_elements]
                ).sum(axis=1)

                if self.max_penetration is None:
                    if (g > self.clearance).all():
                        continue
                else:
                    if np.logical_or(
                        g > self.clearance, g < -self.max_penetration
                    ).all():
                        continue

                # #element that may be in contact (ie vec_xi inside the element) and where g<0
                # test = (vec_xi >= 0) * (vec_xi<=1) #id of elements where there may be contact
                if dim == 1:  # surf.elm_type == 'lin2':
                    test = (vec_xi + self.tol >= 0) * (
                        vec_xi - self.tol <= 1
                    )  # id of elements where there may be contact
                else:  # surf.elm_type == 'tri3':
                    test = (
                        (vec_xi[:, 0] + self.tol >= 0)
                        * (vec_xi[:, 1] + self.tol >= 0)
                        * (1 - vec_xi[:, 0] - vec_xi[:, 1] + self.tol >= 0)
                    )  # id of elements where there may be contact

                # test = (vec_xi+self.tol >= 0) * (vec_xi-self.tol<=1) #id of elements where there may be contact
                if self.max_penetration is None:
                    test = np.where(test * (g < self.clearance))[0]
                else:
                    test = np.where(
                        test * (g < self.clearance) * (g > -self.max_penetration)
                    )[0]

                # if len(test) == 0 and slave_node in contact_list:
                #     # test = (vec_xi+self.tol/length[possible_elements] >= 0) * (vec_xi-self.tol/length[possible_elements]<=1) #id of elements where there may be contact
                #     test = (vec_xi+self.tol >= 0) * (vec_xi-self.tol<=1) #id of elements where there may be contact
                #     test = np.where(test*(g<self.clearance))[0]

                if len(test) == 0:
                    continue

                if len(test) == 1:
                    id_el = test[0]
                else:  # len(test)==2
                    # if slave_node in contact_list and contact_list[slave_node] in np.array(possible_elements)[test]:
                    #     id_el = possible_elements.index(contact_list[slave_node])
                    # else:
                    #     #choose the nearest element
                    # id_el = test[np.abs(g[test]).argmin()] #dont work with non zero clearance
                    id_el = test[g[test].argmax()]

                # id_el before selecting an id_el, we could also test if the the slave node is inside an element

                # contact is established
                g = g[id_el]
                el = possible_elements[id_el]
                vec_xi = vec_xi[id_el]
                # if self.no_slip:
                #     self.contact_list_xi[slave_node] = vec_xi
                shape_func_val = shape_func_val[id_el]
                new_contact_list[slave_node] = el

            else:
                el = contact_list.get(slave_node)  # read the element in contact
                if el is None:
                    continue

                if dim == 1:  # surf.elm_type == 'lin2':
                    vec_xi = (
                        1
                        / length[el]
                        * (
                            (surf.nodes[slave_node] - elm_nodes_crd[el, 0, :])
                            * tangents[el]
                        ).sum()
                    )
                else:  # surf.elm_type == 'tri3':
                    # work only for tri3 face

                    # need to be corrected
                    vec_xi = np.linalg.solve(
                        tangents[el] @ tangents[el].T,
                        np.sum(
                            (surf.nodes[slave_node] - elm_nodes_crd[el, 0, :]).reshape(
                                1, 3
                            )
                            * tangents[el],
                            axis=1,
                        ),
                    ).reshape(1, -1)

                # contact points in global coordinates
                shape_func_val = elm_ref.ShapeFunction(vec_xi)[0]
                contact_point = shape_func_val @ elm_nodes_crd[el]

                # algebric distance from the possible elements
                g = (surf.nodes[slave_node] - contact_point) @ normals[el]
                # if g>self.clearance: g = self.clearance

            n1 = normals[el]
            a1 = tangents[el]

            contact_g.append(g)
            contact_elements.append(el)

            # if vec_xi > 1:
            #     shape_func_val = np.array([0.,1.])
            # elif vec_xi < 0:
            #     shape_func_val = np.array([1.,0.])

            # Need to build several matrices
            #   - matrix that compute g (algebric normal distance) from u knowing vec_xi (Ns in eq 9.18 p 239)
            #   - matrix that

            # col
            # row
            # n1 = normals[el]
            # a1 = tangents[el]

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
                    list(
                        (
                            np.array([0, -1, 1])[sorted_indices] * n1.reshape(-1, 1)
                        ).ravel()
                    )
                )
                data_T0s.extend(
                    list(
                        (
                            np.array([0, -1, 1])[sorted_indices] * a1.reshape(-1, 1)
                        ).ravel()
                    )
                )
            else:
                shape_func_deriv_val = elm_ref.ShapeFunctionDerivative(
                    [vec_xi]
                )[
                    0
                ]  # constant value for tri3, could be compute outside the loop for an arbitrary vec_xi
                data_T1.extend(
                    list(
                        (
                            np.hstack(([1], -shape_func_val))[sorted_indices]
                            * (a1[0]).reshape(-1, 1)
                        ).ravel()
                    )
                )
                data_T2.extend(
                    list(
                        (
                            np.hstack(([1], -shape_func_val))[sorted_indices]
                            * (a1[1]).reshape(-1, 1)
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

        if not (contact_elements):
            self.global_matrix = sparse.csr_array(
                (self.space.nvar * surf.n_nodes, self.space.nvar * surf.n_nodes)
            )
            self.global_vector = 0
            return {}  # empty contact_list

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
        # contact_g = contact_g* (contact_g< self.clearance) #remove negative value
        # if (contact_g > self.clearance).any():
        #     print('Warning, contact have been loosing')

        Fcontact, eps = self.run_normal_law(contact_g - self.clearance)
        # print(Fcontact)

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
        # if self.update_contact:
        #     print(self.sv['contact_list'])
        if self.contact_search_once:
            self.update_contact = False

        # self.current.assemble_global_mat(compute)

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
        ndim = self.space.ndim
        mesh = self.mesh
        master_nodes = self.master_nodes
        slave_nodes = self.slave_nodes
        max_dist = self.max_dist

        w = self.bucket_size
        n_buckets = (mesh.bounding_box.size // w).astype(
            int
        )  # n of buckets along each direction

        # bucket_sort master_nodes
        bucket_crd = ((mesh.nodes[master_nodes] - mesh.bounding_box[0]) / w).astype(
            int
        )  # bucket coordinates (ie nth bucket in each direction) for master nodes
        if ndim == 2:
            bucket_id = bucket_crd[:, 0] + bucket_crd[:, 1] * n_buckets[0]
        else:  # ndim == 3
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
        }  # bucket[i] contain the master nodes in the ith bucket

        # bucket_sort slave_nodes
        bucket_crd = ((mesh.nodes[slave_nodes] - mesh.bounding_box[0]) / w).astype(
            int
        )  # bucket coordinates (ie nth bucket in each direction) for slave nodes
        if ndim == 2:
            bucket_id = bucket_crd[:, 0] + bucket_crd[:, 1] * n_buckets[0]
        else:  # ndim == 3
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
        }  # store the bucket crd for each bucket id
        bucket = {
            bucket_id[sorted_indices[temp[i]]]: sorted_indices[temp[i] : temp[i + 1]]
            for i in range(0, len(temp) - 1)
        }  # contain the slave_nodes index for the ith bucket. Use self.slave_nodes[bucket_id] to retrive the corresponding node indices

        # build dict that contains all neighbors buckets
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
        else:  # ndim == 3:
            neighbors_buckets = {
                i: (
                    i
                    + array_neighbors[i][0]
                    + n_buckets[0] * array_neighbors[i][1].reshape(-1, 1)
                    + n_buckets[0]
                    * n_buckets[1]
                    * array_neighbors[i][2].reshape(-1, 1, 1)
                ).flatten()
                for i in bucket
            }

        nearest_neighbors = np.empty(len(slave_nodes))

        for id_b in bucket:
            # closest_node technique:
            neighbor_master_nodes = np.array(
                sum(
                    [list(bucket_master.get(i, [])) for i in neighbors_buckets[id_b]],
                    [],
                )
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
        possible_elements = []
        for slave_node in self.slave_nodes:
            dist_slave_nodes = np.linalg.norm(
                self.mesh.nodes[slave_node] - self.mesh.nodes[self.master_nodes], axis=1
            )
            trial_node_indice = dist_slave_nodes.argmin()
            if dist_slave_nodes[trial_node_indice] > self.max_dist:
                # to improve performance, ignore contact if distance to the closest node is to high
                possible_elements.append([])
            else:
                nearest_neighbors = self.master_nodes[trial_node_indice]
                # possible_elements.append([el for el in range(self.mesh.n_elements) if nearest_neighbors in self.mesh.elements[el] and slave_node not in self.mesh.elements[el]])
                possible_elements.append(
                    (self.mesh.elements == nearest_neighbors).sum(axis=1).nonzero()[0]
                )
                # possible_elements.append([el for el in range(self.mesh.n_elements) if nearest_neighbors in self.mesh.elements[el]]) #very slow

        return possible_elements


class SelfContact(Contact):
    """Self contact Assembly (ie contact of a geomtry between itself) based on a node 2 surface formulation"""

    def __init__(
        self,
        mesh: Mesh,
        normal_law: str = "linear",
        extract_surface: bool = False,
        search_algorithm: str = "search_nearest",  #'bucket', #'search_nearest'
        space: ModelingSpace | None = None,
        name: str = "Self contact",
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
         space: ModelingSpace
             Modeling space associated to the weakform. If None is specified, the active ModelingSpace is considered.
         name: str
             The name of contact assembly

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

        super().__init__(nodes, mesh, normal_law, search_algorithm, space, name)

    # def global_search(self):
    #     """Find a list of elements that may be in contact for all slave nodes.

    #     This method find the nearest neighbor of all slave nodes and return the
    #     associated elements if the nearest_neighbor is near enough
    #     (criterion givien by the max_dist attribute).

    #     Returns
    #     -------
    #     possible_elements: list[list[int]]
    #     possible_elements[i] is the list of indices of the master surface elements
    #     that may be in contact with the ith slave nodes
    #     are given in possible_elements[i]. If no element can be
    #     in contact, possible_elements[i] is an empty list.
    #     """
    #     return self.nearest_node_self()

    #     possible_elements = []
    #     for id_nd,slave_node in enumerate(self.slave_nodes):
    #         dist_slave_nodes = np.linalg.norm(self.mesh.nodes[slave_node]-self.mesh.nodes[self.master_nodes], axis=1)

    #         #get the indice with the minimal distance without considering the distance of slave_node between itself
    #         trial_nodes_indices = []
    #         if id_nd != 0:
    #             trial_nodes_indices.append(dist_slave_nodes[:id_nd].argmin())

    #         if id_nd != len(self.master_nodes)-1:
    #             trial_nodes_indices.append(dist_slave_nodes[id_nd+1:].argmin() + id_nd+1)

    #         trial_node_indice = trial_nodes_indices[np.argmin(dist_slave_nodes[trial_nodes_indices])]

    #         if dist_slave_nodes[trial_node_indice] > self.max_dist:
    #             #to improve performance, ignore contact if distance to the closest node is to high
    #             possible_elements.append([])
    #         else:

    #             nearest_neighbors = self.master_nodes[trial_node_indice]
    #             possible_elements.append([el for el in range(self.mesh.n_elements) if nearest_neighbors in self.mesh.elements[el] and slave_node not in self.mesh.elements[el]])
    #             # possible_elements.append([el for el in range(self.mesh.n_elements) if nearest_neighbors in self.mesh.elements[el]])

    #     return possible_elements

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
        possible_elements = []
        for id_nd, slave_node in enumerate(self.slave_nodes):
            dist_slave_nodes = np.linalg.norm(
                self.mesh.nodes[slave_node] - self.mesh.nodes[self.master_nodes], axis=1
            )

            # get the indice with the minimal distance without considering the distance of slave_node between itself
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
                # to improve performance, ignore contact if distance to the closest node is to high
                possible_elements.append([])
            else:
                nearest_neighbors = self.master_nodes[trial_node_indice]
                # possible_elements.append([el for el in range(self.mesh.n_elements) if nearest_neighbors in self.mesh.elements[el] and slave_node not in self.mesh.elements[el]]) #very slow
                elm_with_slave_nd = (
                    (self.mesh.elements == slave_node).sum(axis=1).nonzero()[0]
                )
                possible_elements.append(
                    np.setdiff1d(
                        (self.mesh.elements == nearest_neighbors)
                        .sum(axis=1)
                        .nonzero()[0],
                        elm_with_slave_nd,
                        True,
                    )
                )

        return possible_elements

    def _nearest_node_bucket_sort(self):
        ndim = self.space.ndim
        mesh = self.mesh
        master_nodes = self.master_nodes
        slave_nodes = self.slave_nodes
        max_dist = self.max_dist

        w = self.bucket_size
        n_buckets = (mesh.bounding_box.size // w).astype(
            int
        )  # n of buckets along each direction

        # bucket_sort master_nodes
        bucket_crd = ((mesh.nodes[master_nodes] - mesh.bounding_box[0]) / w).astype(
            int
        )  # bucket coordinates (ie nth bucket in each direction) for master nodes
        if ndim == 2:
            bucket_id = bucket_crd[:, 0] + bucket_crd[:, 1] * n_buckets[0]
        else:  # ndim == 3
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
        }  # bucket[i] contain the master nodes in the ith bucket

        # bucket_sort slave_nodes
        bucket_crd = ((mesh.nodes[slave_nodes] - mesh.bounding_box[0]) / w).astype(
            int
        )  # bucket coordinates (ie nth bucket in each direction) for slave nodes
        if ndim == 2:
            bucket_id = bucket_crd[:, 0] + bucket_crd[:, 1] * n_buckets[0]
        else:  # ndim == 3
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
        }  # store the bucket crd for each bucket id
        bucket = {
            bucket_id[sorted_indices[temp[i]]]: sorted_indices[temp[i] : temp[i + 1]]
            for i in range(0, len(temp) - 1)
        }  # contain the slave_nodes index for the ith bucket. Use self.slave_nodes[bucket_id] to retrive the corresponding node indices

        # build dict that contains all neighbors buckets
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
        else:  # ndim == 3:
            neighbors_buckets = {
                i: (
                    i
                    + array_neighbors[i][0]
                    + n_buckets[0] * array_neighbors[i][1].reshape(-1, 1)
                    + n_buckets[0]
                    * n_buckets[1]
                    * array_neighbors[i][2].reshape(-1, 1, 1)
                ).flatten()
                for i in bucket
            }

        nearest_neighbors = np.empty(len(slave_nodes))

        for id_b in bucket:
            # closest_node technique:
            neighbor_master_nodes = np.array(
                sum(
                    [list(bucket_master.get(i, [])) for i in neighbors_buckets[id_b]],
                    [],
                )
            )

            if len(neighbor_master_nodes) == 0:
                continue

            dist_slave_nodes = spatial.distance_matrix(
                mesh.nodes[slave_nodes[bucket[id_b]]], mesh.nodes[neighbor_master_nodes]
            )
            np.fill_diagonal(
                dist_slave_nodes, np.inf
            )  # to avoid selecting self as nearest_neibourgh
            trial_node_indice = dist_slave_nodes.argmin(axis=1)

            min_dist = np.take_along_axis(
                dist_slave_nodes, np.expand_dims(trial_node_indice, axis=-1), axis=-1
            ).ravel()
            nearest_neighbors[bucket[id_b]] = -1 + (min_dist <= max_dist) * (
                neighbor_master_nodes[trial_node_indice] + 1
            )

        # return [[] if nn == -1 else (mesh.elements == nn).sum(axis=1).nonzero()[0] for nn in nearest_neighbors]

        # exclusde elements that containts the slave_node
        return [
            []
            if nn == -1
            else (
                (mesh.elements == nn).sum(axis=1)
                * (mesh.elements != slave_nd).sum(axis=1)
            ).nonzero()[0]
            for slave_nd, nn in zip(slave_nodes, nearest_neighbors)
        ]


# Have not been tested for now
class NodeContact(AssemblyBase):
    """Class that define a node 2 node contact"""

    def __init__(
        self, mesh, nodes1, nodes2, space=None, name="Contact nodes 2 surface"
    ):
        """
        In development.
        """
        if space is None:
            space = ModelingSpace.get_active()
        AssemblyBase.__init__(self, name, space)

        self.nodes1 = nodes1
        self.nodes2 = nodes2
        self.mesh = mesh

        self.current = self

        self.eps_n = 1e7
        """ Contact penalty parameter. """

        self.max_dist = 0.5
        """ Max distance from nodes at which contact is considered"""

        self.sv = {}
        """ Dictionary of state variables associated to the associated for the current problem."""
        self.sv_start = {}

    def assemble_global_mat(self, compute="all"):
        pass

    def contact_search(self):
        nodes1 = self.nodes1
        nodes2 = self.nodes2
        if update_contact:
            # update contact connection
            new_contact_list = {}

        # look for contact
        contact_nodes = []
        contact_g = []

        # matrix for special treatment -> node 2 node if two nodes are close
        # should be move in a node_2_node contact assembly
        Xs = []
        indices_n2n = []
        data_n2n = []
        # end

        list_nodes = np.unique(surf.elements)

        for nd in nodes1:
            dist_nodes = np.linalg.norm(mesh.nodes[nd] - mesh.nodes[nodes2], axis=1)
            trial_node_indice = dist_nodes.argmin()

            if dist_nodes[trial_node_indice] >= self.max_dist:
                # to improve performance, ignore contact if distance to the closest node is to high
                continue

            # asses which elements are in contact
            list_contact_nodes = np.where(dist_nodes < self.max_dist)[0]

            for nd2 in list_contact_nodes:
                t = mesh.nodes[nd] - mesh.nodes[nd2]  # x1-x2
                Xs.extend(t)

                # n2n_global_vector[[slave_node,slave_node+surf.n_nodes]] = -self.eps_nn*t
                # n2n_global_vector[[master_node,master_node+surf.n_nodes]] = self.eps_nn*t
                normal = t / np.linalg.norm(t)

                contact_nodes = np.array([nd, nd2])

                sorted_indices = contact_nodes.argsort()
                indices_n2n.extend(
                    list(
                        (
                            contact_nodes[sorted_indices]
                            + np.array([[0], [mesh.n_nodes]])
                        ).ravel()
                    )
                )
                data_n2n.extend(list(np.tile(np.array([-1, 1])[sorted_indices], 2)))

        M_n2n = sparse.csr_array(
            (data_n2n, indices_n2n, np.arange(0, len(indices_n2n) + 1, 2)),
            shape=(len(indices_n2n) // 2, self.space.nvar * surf.n_nodes),
        )

        # contact law -> put it in function
        # Fc0 = self.eps_a*self.clearance
        # eps = (contact_g <= 0) * self.eps_n + (contact_g > 0) * self.eps_a
        # Fcontact = (-self.eps_n * contact_g + Fc0) * (contact_g <= 0) \
        #            +(self.eps_a * (self.clearance - contact_g)) * (contact_g > 0)

        self.global_matrix = (-self.eps_n) * M_n2n.T @ M_n2n

        self.global_vector = self.eps_n * M_n2n.T @ np.array(Xs)

        # voir eq 9.35 et 9.36 (page 241 du pdf) avec def 9.18 et 9.19 page 239

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
        self.current.contact_search()  # initialize contact state
        self.sv_start = dict(self.sv)

    def set_start(self, problem):
        self.sv_start = dict(
            self.sv
        )  # create a new dict with alias inside (not deep copy)

    def to_start(self, pb):
        self.sv = dict(self.sv_start)
        self.set_disp(pb.get_disp())
        self.current.contact_search()  # initialize global_matrix and global_vector

    def update(self, pb, compute="all"):
        self.set_disp(pb.get_disp())
        self.current.contact_search()


# def global_search_bucket_sort(self):
#     possible_elements = []

#     #bucket_sort
#     w = self.mesh.bounding_box.size.max()/10 #bucket size
#     n_buckets =  (self.mesh.bounding_box.size // w).astype(int)

#     temp = ((self.mesh.nodes[self.slave_nodes] - self.mesh.bounding_box[0])/w).astype(int)
#     bucket_id = temp[:,0] + temp[:,1] * n_buckets[0]  + temp[:,2] * n_buckets[0]*n_buckets[1] #only work in 3D -> improve

#     sorted_indices = bucket_id.argsort()

#     temp = [0] + [i for i in range(1,len(bucket_id)) if bucket_id[sorted_indices[i]] != bucket_id[sorted_indices[i-1]]]

#     bucket = {bucket_id[sorted_indices[temp[i]]]: sorted_indices[temp[i]:temp[i+1]] for i in range(0, len(temp)-1)} #bucket[2] contain the ith bucket -> perhaps best to include all bucket coordinates (list of list for instance)

#     neighbors_buckets = {i: (i+np.array([-1,0,1])+n_buckets[0]*np.array([-1,0,1]).reshape(-1,1) + n_buckets[0]*n_buckets[1]*np.array([-1,0,1]).reshape(-1,1,1)).flatten() for i in bucket}

#     for id_b in bucket:
#         #closest_node technique:
#         slave_node = bucket[id_b]
#         master_nodes = sum([list(bucket.get(i,[])) for i in neighbors_buckets[id_b]],[])

#         # dist_slave_nodes = np.linalg.norm(self.mesh.nodes[slave_node]-self.mesh.nodes[master_nodes], axis=1)
#         dist_slave_nodes = self.mesh.nodes[slave_node] @ self.mesh.nodes[master_nodes].T
#         trial_node_indice = dist_slave_nodes.argmin()
#         if dist_slave_nodes[trial_node_indice] > self.max_dist:
#             #to improve performance, ignore contact if distance to the closest node is to high
#             possible_elements.append([])
#         else:

#             nearest_neighbors = self.master_nodes[trial_node_indice]
#             # possible_elements.append([el for el in range(self.mesh.n_elements) if nearest_neighbors in self.mesh.elements[el] and slave_node not in self.mesh.elements[el]])
#             possible_elements.append([el for el in range(self.mesh.n_elements) if nearest_neighbors in self.mesh.elements[el]])
