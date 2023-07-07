#### En cours de dÃ©veloppement


# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 15:27:43 2020

@author: Etienne
"""
# from fedoo.core.base   import ProblemBase 
import numpy as np
from fedoo.core.base import AssemblyBase
# from fedoo.core.boundary_conditions import BCBase, MPC, ListBC
from fedoo.core.base import ProblemBase
from fedoo.core.mesh import MeshBase, Mesh
from fedoo.lib_elements.element_list import * 
from scipy import sparse
from fedoo.core.modelingspace import ModelingSpace
from copy import copy

def extract_surface_mesh(mesh):
    if mesh.elm_type in ['quad4','quad8','quad9']:
        faces_in_elm = [[0,1],[1,2],[2,3],[3,0]]
        n_face_nodes = 2
        face_elm_type = 'lin2'
    elif mesh.elm_type in ['tri3','tri6']:
        faces_in_elm = [[0,1],[1,2],[2,0]]
        n_face_nodes = 2
        face_elm_type = 'lin2'
    else: 
        raise NotImplementedError()
        
    n_faces_in_elm = len(faces_in_elm)
    list_faces = mesh.elements[:,faces_in_elm] #shape = (n_elements, n_elm_faces, n_node_per_faces)    
    
    test = np.sort(list_faces.reshape(-1,2), axis=1)
    ind_sorted = np.lexsort(tuple((test[:,i-1] for i in range(n_face_nodes,0,-1))))
    # sorted_faces = [frozenset(face) for face in test[ind_sorted]]
    sorted_faces = test[ind_sorted]
    
    surf_elements = [] #list of the surface elements (ie exterior faces of the initial mesh)
    ind_element = [] #ind_element[i] is the indice of the element who contains the face surf_elements[i]
    i=1
    while i < len(sorted_faces):            
        if (sorted_faces[i] != sorted_faces[i-1]).any():     
            surf_elements.append(list(sorted_faces[i-1]))
            ind_element.append(ind_sorted[i-1]//n_faces_in_elm)
            
            if i == len(sorted_faces)-1:
                #if i is the last element, we should append it also
                surf_elements.append(list(sorted_faces[i]))    
                ind_element.append(ind_sorted[i]//n_faces_in_elm)
                            
        else:                              
            while i<len(sorted_faces) and (sorted_faces[i] == sorted_faces[i-1]).all():
                i+=1
        i+=1
            
    surf_elements = np.array(surf_elements)
    
    #Check if normal are well defined (outside the volume)
    #idea: compute the distance of a point of the elm that is not the face itself    
    nodes_inside=[np.setdiff1d(mesh.elements[ind_element[i]], face)[0] for i,face in enumerate(surf_elements)]
        
    element_face = get_element(face_elm_type)(1)
    
    elm_nodes_crd = mesh.nodes[surf_elements]
    local_frame = element_face.GetLocalFrame(elm_nodes_crd, element_face.get_gp_elm_coordinates(1))

    tangent_axis = local_frame[:,0,0]
    normal_axis = local_frame[:,0,1]
    
    # The two following lines work only for 2 node 1d element    # length = np.linalg.norm(mesh.nodes[surf.elements[:,1]] - mesh.nodes[surf.elements[:,0]], axis = 1)
    length = np.linalg.norm(elm_nodes_crd[:,1,:] - elm_nodes_crd[:,0,:], axis = 1)    
    vec_xi = 1/length*((mesh.nodes[nodes_inside] - elm_nodes_crd[:,0,:]) * tangent_axis).sum(axis=1)

    #for all elements
    shape_func_val = element_face.ShapeFunction(vec_xi)
    contact_points = (shape_func_val[:, np.newaxis, :] @ elm_nodes_crd).squeeze()
    #or equivalent : 
    # contact_points = np.sum(shape_func_val[..., np.newaxis] * surf.nodes[surf.elements], axis = 1)
    #or (only for 1D element)
    # contact_points = shape_func_val[:,0].reshape(-1,1)*surf.nodes[surf.elements[:,0]]+  \
    #                  shape_func_val[:,1].reshape(-1,1)*surf.nodes[surf.elements[:,1]]
    g = ((mesh.nodes[nodes_inside] - contact_points) * normal_axis).sum(axis=1)
    mask = np.where(g>0)[0] #where normal need to be revered
    surf_elements[mask] = surf_elements[mask,::-1] 
    
    return Mesh(mesh.nodes, surf_elements, face_elm_type)





class Contact(AssemblyBase):
    """Class that define contact based on a node 2 surface formulation"""
    
    def __init__(self, nodes, surface_mesh, space = None, name = "Contact nodes 2 surface"):
        """   
        In development.             
        """
        if space is None: space = ModelingSpace.get_active()            
        AssemblyBase.__init__(self, name, space)

        self.nodes = nodes
        self.mesh = surface_mesh
        
        self.current = self 
        
        self.eps_a = 1e4
        self.eps_n = 1e7
        # self.eps_nn = 1e7
        """ Contact penalty parameter. """
        
        self.max_dist = 0.5
        """ Max distance from nodes at which contact is considered""" 
        
        self.dist_switch_n2n = None
        """ Distance from nodes at which the model swich to node 2 node contact"""
        
        self.clearance = 0.0
        """ Distance at which we consider contact for adjustement and stabilisation"""
        
        self.tol = 0.15
        """ Tol for possible slide of a node outside an element"""
        
        self.sv = {}
        """ Dictionary of state variables associated to the associated for the current problem."""
        self.sv_start = {}
        # self.bc_type = 'Contact'
        # BCBase.__init__(self, name)        
        # self._update_during_inc = 1
        
    # def __repr__(self):
    #     list_str = ['Contact:']
    #     if self.name != "": list_str.append("name = '{}'".format(self.name))
        
    #     return "\n".join(list_str)
    


    def assemble_global_mat(self, compute = 'all'):
        pass
        
    def contact_search(self, contact_list = {}, update_contact = True):
        nodes = self.nodes
        surf = self.mesh #mesh of the surface
        if update_contact: 
            #update contact connection
            new_contact_list = {}
   
        #look for contact
        #brut force, compute the distance between nodes and all elements, 
        #no vectorization, not optimized
        
        #get the normal surface on the center of the elements for each element on
        #the master sufrace
        elm_ref = get_element(surf.elm_type)(1) #1 gauss point to compute the local base (center of the element)
        elm_nodes_crd = surf.nodes[surf.elements]        
        
        local_frame = elm_ref.GetLocalFrame(elm_nodes_crd, elm_ref.get_gp_elm_coordinates(1))

        tangent_axis = local_frame[:,0,0]
        normal_axis = local_frame[:,0,1]
        
        # The following lines work only for 2 node 1d element
        # length = np.linalg.norm(mesh.nodes[surf.elements[:,1]] - mesh.nodes[surf.elements[:,0]], axis = 1)
        length = np.linalg.norm(elm_nodes_crd[:,1,:] - elm_nodes_crd[:,0,:], axis = 1)    

        indices = []
        contact_elements = []
        contact_g = []

        data_Ns = []
        data_N0s = []
        data_Ts = []
        data_T0s = []

        #matrix for special treatment -> node 2 node if two nodes are close
        #should be move in a node_2_node contact assembly 
        Xs= []
        indices_n2n = []
        data_n2n = []
        contact_n2n = [] #list of nodes in contact in n2n 
        #end 
        
        list_nodes = np.unique(surf.elements)               
            
        for slave_node in nodes:
            
            if update_contact:
                dist_slave_nodes = np.linalg.norm(surf.nodes[slave_node]-surf.nodes[list_nodes], axis=1)          
                trial_node_indice = dist_slave_nodes.argmin()               
                                
                if dist_slave_nodes[trial_node_indice] > self.max_dist: 
                    #to improve performance, ignore contact if distance to the closest node is to high
                    continue                       
                    
                #asses which elements can be in contact
                trial_node = list_nodes[trial_node_indice]
                possible_elements = [el for el in range(surf.n_elements) if trial_node in surf.elements[el]]
            
                #orthogonal projection on the element plane in node coordinates
                vec_xi = 1/length[possible_elements]*((surf.nodes[slave_node] - elm_nodes_crd[possible_elements,0,:]) * tangent_axis[possible_elements]).sum(axis=1)
                
                
                #contact points in global coordinates
                shape_func_val = elm_ref.ShapeFunction(vec_xi)
                contact_points = (shape_func_val[:, np.newaxis, :] @ elm_nodes_crd[possible_elements]).squeeze()
        
                #algebric distance from the possible elements
                g = ((surf.nodes[slave_node] - contact_points) * normal_axis[possible_elements]).sum(axis=1)            
                
                # #element that may be in contact (ie vec_xi inside the element) and where g<0  
                
                # to check
                # if slave_node in contact_list: 
                    # test = (vec_xi+self.tol/length[possible_elements] >= 0) * (vec_xi-self.tol/length[possible_elements]<=1) #id of elements where there may be contact
                test = (vec_xi+self.tol >= 0) * (vec_xi-self.tol<=1) #id of elements where there may be contact
                # else: 
                #     test = (vec_xi >= 0) * (vec_xi<=1) #id of elements where there may be contact
                #end to check
                
                test = np.where(test*(g<self.clearance))[0]
                            
    
                
                if len(test)==0:
                    continue
                
                if len(test)==1:
                    id_el = test[0]
                else: #len(test)==2
                    
                    # if slave_node in contact_list and contact_list[slave_node] in np.array(possible_elements)[test]:
                    #     id_el = possible_elements.index(contact_list[slave_node])
                    # else:                    
                    #     #choose the nearest element
                    id_el = test[np.abs(g[test]).argmin()]
                #id_el before selecting an id_el, we should test if the the slave node is inside an element                                        
                
                # # #if one nodes is very close consider a n2n link
                # if self.dist_switch_n2n is not None and dist_slave_nodes[trial_node_indice] < self.dist_switch_n2n+ g[id_el]:
                #     # print('n2n')
                    
                #     #in this case, there is a node close.
                #     #treat the contact as node to node to avoid change of normal direction between close el
                #     master_node = list_nodes[trial_node_indice]
                                    
                #     # print(master_node)
                #     # print(slave_node)
                    
                #     t = (surf.nodes[master_node] - surf.nodes[slave_node]) #x1-x2
                #     # print('t:', t)
                #     # print(np.linalg.norm(t))
                #     Xs.extend(t)
                    
                #     # n2n_global_vector[[slave_node,slave_node+surf.n_nodes]] = -self.eps_nn*t
                #     # n2n_global_vector[[master_node,master_node+surf.n_nodes]] = self.eps_nn*t
                #     # n1 = t/np.linalg.norm(t)
                    
                                  
                #     contact_nodes = np.array([slave_node, master_node])                                              
                    
                #     sorted_indices = contact_nodes.argsort()
                #     indices_n2n.extend(list((contact_nodes[sorted_indices] + np.array([[0], [surf.n_nodes]])).ravel()))
                #     data_n2n.extend(list(np.tile(np.array([-1,1])[sorted_indices],2)))         
                    
                #     contact_n2n.append(master_node)
                    
                # else:
    
                # for id_el in range(len(g)): 
                
                # contact is established
                g = g[id_el]
                el = possible_elements[id_el]
                vec_xi = vec_xi[id_el]
                shape_func_val = shape_func_val[id_el]
                new_contact_list[slave_node] = el                
            
            else:
                el = contact_list.get(slave_node) #read the element in contact
                if el is None: continue
                vec_xi = 1/length[el]*((surf.nodes[slave_node] - elm_nodes_crd[el,0,:]) * tangent_axis[el]).sum()
                                
                #contact points in global coordinates
                shape_func_val = elm_ref.ShapeFunction(vec_xi)[0]
                contact_point = (shape_func_val @ elm_nodes_crd[el])
        
                #algebric distance from the possible elements
                g = (surf.nodes[slave_node] - contact_point) @ normal_axis[el]      

            contact_g.append(g)
            contact_elements.append(el)
            
            if vec_xi > 1:
                shape_func_val = np.array([0.,1.])
            elif vec_xi < 0:
                shape_func_val = np.array([1.,0.])
        
            # Need to build several matrices
            #   - matrix that compute g (algebric normal distance) from u knowing vec_xi (Ns in eq 9.18 p 239)
            #   - matrix that 
            
            # col
            # row
            n1 = normal_axis[contact_elements[-1]]
            a1 = tangent_axis[contact_elements[-1]]
            
            
            contact_nodes = np.hstack(([slave_node],surf.elements[contact_elements[-1]]))
            sorted_indices = contact_nodes.argsort()
            
            indices.extend(list((contact_nodes[sorted_indices] + np.array([[0], [surf.n_nodes]])).ravel()))
            data_Ns.extend(list((np.hstack(([1],-shape_func_val))[sorted_indices] * n1.reshape(-1,1)).ravel()))
            data_Ts.extend(list((np.hstack(([1],-shape_func_val))[sorted_indices] * a1.reshape(-1,1)).ravel()))
            
            data_N0s.extend(list((np.array([0,-1,1])[sorted_indices] * n1.reshape(-1,1)).ravel())) 
            data_T0s.extend(list((np.array([0,-1,1])[sorted_indices] * a1.reshape(-1,1)).ravel())) 
                            
            
        shape = (len(contact_elements), self.space.nvar*surf.n_nodes)
        indptr = np.arange(0,len(indices)+1, 6)
        Ns = sparse.csr_array((data_Ns, indices, indptr), shape=shape)
        N0s = sparse.csr_array((data_N0s, indices, indptr), shape=shape)  
        Ts = sparse.csr_array((data_Ts, indices, indptr), shape=shape)
        T0s = sparse.csr_array((data_Ns, indices, indptr), shape=shape)

        M_n2n = sparse.csr_array((data_n2n, indices_n2n, np.arange(0,len(indices_n2n)+1, 2)), 
                                 shape=(len(indices_n2n)//2, self.space.nvar*surf.n_nodes))

        contact_g = np.array(contact_g)            
        # contact_g = contact_g* (contact_g< self.clearance) #remove negative value
        # if (contact_g > self.clearance).any(): 
        #     print('Warning, contact have been loosing')
        
        #contact law -> put it in function
        Fc0 = self.eps_a*self.clearance
        eps = (contact_g <= 0) * self.eps_n + (contact_g > 0) * self.eps_a 
        Fcontact = (-self.eps_n * contact_g + Fc0) * (contact_g <= 0) \
                   +(self.eps_a * (self.clearance - contact_g)) * (contact_g > 0) 
        
        #or 
        # Fcontact = -self.eps_n * np.array(contact_g)

        Fcontact_div_l = sparse.diags(Fcontact/length[contact_elements], format='csr')
        g_div_l = sparse.diags(contact_g/length[contact_elements], format='csr')
        #TO DO integrate 9.36 eq from wriggers 2006 with fd._sparse lib to improve building perfomance            
        self.global_matrix = self.eps_n*( Ns.T@Ns - N0s.T@g_div_l@Ts -  
                                          Ts.T@g_div_l@N0s - N0s.T@(g_div_l@g_div_l)@N0s)
        
        # self.global_matrix = self.eps_n*( Ns.T@Ns ) + N0s.T@Fcontact_div_l@Ts +  \
        #                                   Ts.T@Fcontact_div_l@N0s + N0s.T@(Fcontact_div_l@g_div_l)@N0s
        # self.global_matrix = self.global_matrix + self.eps_nn*  M_n2n.T@M_n2n
        # self.global_matrix = self.eps_nn*  M_n2n.T@M_n2n
        # self.global_matrix = self.eps_n*( Ns.T@Ns)
        
        
        # print('nodes:', contact_n2n)
        # print('elements:', contact_elements)
        
        # self.global_vector = Ns.T@Fcontact - self.eps_nn * M_n2n.T@np.array(Xs)
        self.global_vector = Ns.T@Fcontact 
        # print('toto: ',self.global_vector[[65, 121]])
        self.sv['contact_elements'] = contact_elements   #would be better to get force at nodes and includes node 2 node
        self.sv['Fcontact'] = Fcontact
        self.sv['contact_g'] = contact_g
        if update_contact:
            return new_contact_list
        else:
            return contact_list
            
        #voir eq 9.35 et 9.36 (page 241 du pdf) avec def 9.18 et 9.19 page 239
            

    def set_disp(self, disp):
        if disp is 0: self.current = self
        else:
            new_crd = self.mesh.nodes + disp.T
            if self.current == self:
                #initialize a new 
                new_mesh = copy(self.mesh)
                new_mesh.nodes = new_crd
                new_assembly = copy(self)                                                    
                new_assembly.mesh = new_mesh
                self.current = new_assembly
            else: 
                self.current.mesh.nodes = new_crd
        
 
    def initialize(self, pb):
        # self.update(problem)
        #initialize the contact list
        self.sv['contact_list'] = self.current.contact_search({}, True) #initialize contact state
        self.sv_start = dict(self.sv)
        self.update_contact = True
        
      
    def set_start(self, problem):
        #set_start should update the tangent matrix. Here the current state is kept
        self.sv_start = dict(self.sv) #create a new dict with alias inside (not deep copy)
        # self.sv['contact_list'] = self.current.contact_search({}, True) #initialize contact state
        self.update_contact = True
    
    
    def to_start(self, pb):
        self.sv = dict(self.sv_start)
        self.set_disp(pb.get_disp())
        self.current.contact_search(self.sv['contact_list'], update_contact = False) #initialize global_matrix and global_vector
        #here the tangent matrix is recomputed that may lead to a slight difference from the 1rst attempt.
        # self.current.assemble_global_mat(compute)
        self.update_contact = True
  
    def update(self, pb, compute = 'all'):        
        self.set_disp(pb.get_disp())
        self.sv['contact_list'] = self.current.contact_search(self.sv['contact_list'], self.update_contact)
        if self.update_contact:
            print(self.sv['contact_list'])
        self.update_contact = False
        
        # self.current.assemble_global_mat(compute)
