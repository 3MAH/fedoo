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
from fedoo.lib_elements.element_list import get_element 
from scipy import sparse
from fedoo.core.modelingspace import ModelingSpace
from copy import copy


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
        
        self.eps_a = 1e5
        self.eps_n = 1e7
        # self.eps_nn = 1e7
        """ Contact penalty parameter. """
        
        self.max_dist = 0.5
        """ Max distance from nodes at which contact is considered""" 
        
        # self.dist_switch_n2n = None
        # """ Distance from nodes at which the model swich to node 2 node contact"""
        
        self.clearance = 0.
        """ Distance at which we consider contact for adjustement and stabilisation"""
        
        self.tol = 0.1
        """ Tol for possible slide of a node outside an element"""
        
        self.contact_search_once = False
        """Only search contact at the begining of iteration"""
        
        
        self.sv = {}
        """ Dictionary of state variables associated to the associated for the current problem."""
        self.sv_start = {}
        # self.bc_type = 'Contact'
        # BCBase.__init__(self, name)        
        # self._update_during_inc = 1
        
        #default contact law
        if self.clearance == 0:
            self.contact_law = self.linear_law
        else:
            self.contact_law = self.bilinear_law
            
        
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
                
                if (g>self.clearance).all():                    
                    continue                                                                
                
                # #element that may be in contact (ie vec_xi inside the element) and where g<0  
                # test = (vec_xi >= 0) * (vec_xi<=1) #id of elements where there may be contact
                test = (vec_xi+self.tol >= 0) * (vec_xi-self.tol<=1) #id of elements where there may be contact

                # test = (vec_xi+self.tol >= 0) * (vec_xi-self.tol<=1) #id of elements where there may be contact
                test = np.where(test*(g<self.clearance))[0]
                                
                # if len(test) == 0 and slave_node in contact_list: 
                #     # test = (vec_xi+self.tol/length[possible_elements] >= 0) * (vec_xi-self.tol/length[possible_elements]<=1) #id of elements where there may be contact
                #     test = (vec_xi+self.tol >= 0) * (vec_xi-self.tol<=1) #id of elements where there may be contact
                #     test = np.where(test*(g<self.clearance))[0]
    
                
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
                    
                #id_el before selecting an id_el, we could also test if the the slave node is inside an element                                        
                
                
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
                if g>self.clearance: g = self.clearance
                


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
        
        Fcontact, eps = self.contact_law(contact_g)
        # print(Fcontact)
        F_div_l = sparse.diags(Fcontact/length[contact_elements], format='csr') #F = -eps*g
        g_div_l = sparse.diags(contact_g/length[contact_elements], format='csr')

        #TO DO integrate 9.36 eq from wriggers 2006 with fd._sparse lib to improve building perfomance            
        # self.global_matrix = self.eps_n @ ( Ns.T@Ns - N0s.T@g_div_l@Ts -  
        #                            Ts.T@g_div_l@N0s - N0s.T@(g_div_l@g_div_l)@N0s)

        if not(np.isscalar(eps)): 
            mat_eps = sparse.diags(eps, format='csr')
            self.global_matrix = ( Ns.T@mat_eps@Ns + N0s.T@F_div_l@Ts +  
                                       Ts.T@F_div_l@N0s + N0s.T@(F_div_l@g_div_l)@N0s)
        else:
            # self.global_matrix = eps*( Ns.T@Ns - N0s.T@g_div_l@Ts -  
            #                       Ts.T@g_div_l@N0s - N0s.T@(g_div_l@g_div_l)@N0s)
        
            self.global_matrix = ( eps*Ns.T@Ns + N0s.T@F_div_l@Ts +  
                                        Ts.T@F_div_l@N0s + N0s.T@(F_div_l@g_div_l)@N0s)
        
        
        self.global_vector = Ns.T@Fcontact 
        
        self.sv['contact_elements'] = contact_elements   #would be better to get force at nodes and includes node 2 node
        self.sv['Fcontact'] = Fcontact
        self.sv['contact_g'] = contact_g
        # print(contact_elements)
        # print(contact_g)
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
        if self.contact_search_once:
            self.update_contact = False
        
        # self.current.assemble_global_mat(compute)


    
    def bilinear_law(self, g):
        contact_g = g        
        Fc0 = self.eps_a*self.clearance
        eps = (contact_g <= 0) * self.eps_n + (contact_g > 0) * self.eps_a 
        Fcontact = (-self.eps_n * contact_g + Fc0) * (contact_g <= 0) \
                   +(self.eps_a * (self.clearance - contact_g)) * (contact_g > 0) 
        
        return Fcontact, eps
    
    def linear_law(self, g):
        Fcontact = -self.eps_n * np.array(g)
        return Fcontact, self.eps_n



        


#Have not been tested for now
class NodeContact(AssemblyBase):
    """Class that define a node 2 node contact"""
    
    def __init__(self, mesh, nodes1, nodes2, space = None, name = "Contact nodes 2 surface"):
        """   
        In development.             
        """
        if space is None: space = ModelingSpace.get_active()            
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


    def assemble_global_mat(self, compute = 'all'):
        pass
        
    def contact_search(self):
        nodes1 = self.nodes1
        nodes2 = self.nodes2
        if update_contact: 
            #update contact connection
            new_contact_list = {}
   
        #look for contact
        contact_nodes = []
        contact_g = []

        #matrix for special treatment -> node 2 node if two nodes are close
        #should be move in a node_2_node contact assembly 
        Xs= []
        indices_n2n = []
        data_n2n = []
        #end 
        
        list_nodes = np.unique(surf.elements)               
            
        for nd in nodes1:
            
            dist_nodes = np.linalg.norm(mesh.nodes[nd]-mesh.nodes[nodes2], axis=1)          
            trial_node_indice = dist_nodes.argmin()               
                            
            if dist_nodes[trial_node_indice] >= self.max_dist: 
                #to improve performance, ignore contact if distance to the closest node is to high
                continue                       
                
            #asses which elements are in contact
            list_contact_nodes = np.where(dist_nodes<self.max_dist)[0]                
            
            for nd2 in list_contact_nodes:
                              
                                
                t = (mesh.nodes[nd] - mesh.nodes[nd2]) #x1-x2
                Xs.extend(t)
                
                # n2n_global_vector[[slave_node,slave_node+surf.n_nodes]] = -self.eps_nn*t
                # n2n_global_vector[[master_node,master_node+surf.n_nodes]] = self.eps_nn*t
                normal = t/np.linalg.norm(t)
                
                              
                contact_nodes = np.array([nd, nd2])                                              
                
                sorted_indices = contact_nodes.argsort()
                indices_n2n.extend(list((contact_nodes[sorted_indices] + np.array([[0], [mesh.n_nodes]])).ravel()))
                data_n2n.extend(list(np.tile(np.array([-1,1])[sorted_indices],2)))         
                               
        M_n2n = sparse.csr_array((data_n2n, indices_n2n, np.arange(0,len(indices_n2n)+1, 2)), 
                                 shape=(len(indices_n2n)//2, self.space.nvar*surf.n_nodes))

        #contact law -> put it in function
        # Fc0 = self.eps_a*self.clearance
        # eps = (contact_g <= 0) * self.eps_n + (contact_g > 0) * self.eps_a 
        # Fcontact = (-self.eps_n * contact_g + Fc0) * (contact_g <= 0) \
        #            +(self.eps_a * (self.clearance - contact_g)) * (contact_g > 0) 
        

        
        self.global_matrix = (-self.eps_n)*  M_n2n.T@M_n2n
        
        
        self.global_vector = self.eps_n * M_n2n.T@np.array(Xs)

            
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
        self.current.contact_search() #initialize contact state
        self.sv_start = dict(self.sv)
        
      
    def set_start(self, problem):
        self.sv_start = dict(self.sv) #create a new dict with alias inside (not deep copy)
    
    
    def to_start(self, pb):
        self.sv = dict(self.sv_start)
        self.set_disp(pb.get_disp())
        self.current.contact_search() #initialize global_matrix and global_vector

  
    def update(self, pb, compute = 'all'):        
        self.set_disp(pb.get_disp())
        self.current.contact_search()




