#### En cours de développement


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
    
    # The two following lines work only for 2 node 1d element    # lenght = np.linalg.norm(mesh.nodes[surf.elements[:,1]] - mesh.nodes[surf.elements[:,0]], axis = 1)
    lenght = np.linalg.norm(elm_nodes_crd[:,1,:] - elm_nodes_crd[:,0,:], axis = 1)    
    vec_xi = 1/lenght*((mesh.nodes[nodes_inside] - elm_nodes_crd[:,0,:]) * tangent_axis).sum(axis=1)

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
    
    def __init__(self, nodes, surface_mesh, name = "Contact nodes 2 surface"):
        """   
        In development.             
        """

        self.nodes = nodes
        self.mesh = surface_mesh
        
        AssemblyBase.__init__(self, name)
        # self.bc_type = 'Contact'
        # BCBase.__init__(self, name)        
        # self._update_during_inc = 1
        
    # def __repr__(self):
    #     list_str = ['Contact:']
    #     if self.name != "": list_str.append("name = '{}'".format(self.name))
        
    #     return "\n".join(list_str)
    
    def initialize(self, problem, t0=0.):
        pass
        # for i,var in enumerate(self.var_cd):
        #     if isinstance(var, str):
        #         self.var_cd[i] = problem.space.variable_rank(var)
                


    def assemble_global_mat(self, compute = 'all'):
        nodes = self.nodes
        surf = self.mesh #mesh of the surface
                
        #look for contact
        #brut force, compute the distance between nodes and all elements
        
        #get the normal surface on the center of the elements for each element on
        #the master sufrace
        elm_ref = get_element(surf.elm_type)(1) #1 gauss point to compute the local base (center of the element)
        elm_nodes_crd = surf.nodes[surf.elements]        
        
        local_frame = elm_ref.GetLocalFrame(elm_nodes_crd, elm_ref.get_gp_elm_coordinates(1))

        tangent_axis = local_frame[:,0,0]
        normal_axis = local_frame[:,0,1]
        
        # The following lines work only for 2 node 1d element
        # lenght = np.linalg.norm(mesh.nodes[surf.elements[:,1]] - mesh.nodes[surf.elements[:,0]], axis = 1)
        lenght = np.linalg.norm(elm_nodes_crd[:,1,:] - elm_nodes_crd[:,0,:], axis = 1)    

        for node in nodes:
            vec_xi = 1/lenght*((surf.nodes[node] - elm_nodes_crd[:,0,:]) * tangent_axis).sum(axis=1)
            possible = np.where((vec_xi >= 0) * (vec_xi<=1))[0] #id of elements where there may be contact
    
            #for all elements
            shape_func_val = element_face.ShapeFunction(vec_xi[possible])
            contact_points = (shape_func_val[:, np.newaxis, :] @ elm_nodes_crd[possible]).squeeze()

    
            g = ((surf.nodes[node] - contact_points) * normal_axis[possible]).sum(axis=1)
            contact_elements = possible[np.where(g<0)[0]]
            
            #voir eq 9.35 et 9.36 (page 241 du pdf) avec def 9.18 et 9.19 page 239



        
        # self.global_matrix = ??
        # self.global_vector = ??

                
        
        
  
    
    
    
    # for i, face in enumerate(surf_elements):        
    #     elm = ind_element[i] #elm the id of the element associated with face
    #     node_inside = np.setdiff1d(elm, face)[0] #this node should be inside the volume
        
    #     element_face = fd.lib_elements.element_list.get_element(face_elm_type)(0)
        
    #     elm_nodes_crd = mesh.nodes[surf.elements]
    #     local_frame = element_face.GetLocalFrame(elm_nodes_crd, element_face.get_gp_elm_coordinates(1))

    #     tangent_axis = local_frame[:,0,0]
    #     normal_axis = local_frame[:,0,1]
    #     # lenght = np.linalg.norm(mesh.nodes[surf.elements[:,1]] - mesh.nodes[surf.elements[:,0]], axis = 1)
    #     lenght = np.linalg.norm(elm_nodes_crd[:,1,:] - mesh.nodes[surf.elements[:,0,:]], axis = 1)
        
    #     vec_xi = 1/lenght*((mesh.nodes[node_inside] - (elm_nodes_crd[:,0,:]) * tangent_axis).sum(axis=1)
    #     # possible = np.where((vec_xi >= 0) * (vec_xi<=1))[0] #id of elements where there may be contact

    #     # #for all elements
    #     # shape_func_val = element.ShapeFunction(vec_xi[possible])
    #     # contact_points = (shape_func_val[:, np.newaxis, :] @ surf.nodes[surf.elements[possible]]).squeeze()
    #     # #or equivalent : 
    #     # # contact_points = np.sum(shape_func_val[..., np.newaxis] * surf.nodes[surf.elements], axis = 1)
    #     # #or (only for 1D element)
    #     # # contact_points = shape_func_val[:,0].reshape(-1,1)*surf.nodes[surf.elements[:,0]]+  \
    #     # #                  shape_func_val[:,1].reshape(-1,1)*surf.nodes[surf.elements[:,1]]
    #     # g = ((mesh.nodes[nodes_contact[0]] - contact_points) * normal_axis[possible]).sum(axis=1)


    #     # contact_elements = possible[np.where(g<0)[0]]

        

        
    
    
    
    
    
            
    
    # element_faces = [[frozenset(face) for face in elm_faces] for elm_faces in list_faces]    
    # list_faces = {frozenset(face) for face in list_faces.reshape(-1,n_face_nodes)}

    
    
    
    
    # surf = fd.Mesh.from_pyvista(mesh.to_pyvista().extract_feature_edges(
    #     boundary_edges=True, non_manifold_edges=False, feature_edges=False, manifold_edges=False).cast_to_unstructured_grid())
    # if mesh.ndim == 2:
    #     surf.nodes = surf.nodes[:,:2]

    # element = fd.lib_elements.element_list.get_element(surf.elm_type)(0)
    # local_frame = element.GetLocalFrame(surf.nodes[surf.elements], element.get_gp_elm_coordinates(1))

    # #check the normal orientation

    # nodes_contact = mesh.add_nodes([L+0.01, h/2]) #add only one nodes
    # # nodes_contact = mesh.add_nodes([0.5, 0.01]) #add only one nodes

    # contact = fd.constraint.Contact(nodes_contact, surf)
    # element = fd.lib_elements.element_list.get_element(surf.elm_type)(0)
    # local_frame = element.GetLocalFrame(surf.nodes[surf.elements], element.get_gp_elm_coordinates(1))

    # tangent_axis = local_frame[:,0,0]
    # normal_axis = local_frame[:,0,1]
    # lenght = np.linalg.norm(surf.nodes[surf.elements[:,1]] - surf.nodes[surf.elements[:,0]], axis = 1)
    # vec_xi = 1/lenght*((mesh.nodes[nodes_contact[0]] - surf.nodes[surf.elements[:,0]]) * tangent_axis).sum(axis=1)
    # possible = np.where((vec_xi >= 0) * (vec_xi<=1))[0] #id of elements where there may be contact

    # #for all elements
    # shape_func_val = element.ShapeFunction(vec_xi[possible])
    # contact_points = (shape_func_val[:, np.newaxis, :] @ surf.nodes[surf.elements[possible]]).squeeze()
    # #or equivalent : 
    # # contact_points = np.sum(shape_func_val[..., np.newaxis] * surf.nodes[surf.elements], axis = 1)
    # #or (only for 1D element)
    # # contact_points = shape_func_val[:,0].reshape(-1,1)*surf.nodes[surf.elements[:,0]]+  \
    # #                  shape_func_val[:,1].reshape(-1,1)*surf.nodes[surf.elements[:,1]]
    # g = ((mesh.nodes[nodes_contact[0]] - contact_points) * normal_axis[possible]).sum(axis=1)


    # contact_elements = possible[np.where(g<0)[0]]





# class Contact(BCBase):
#     """Class that define contact based on a node 2 surface formulation"""
    
#     def __init__(self, nodes, surface_mesh, name = "Contact nodes 2 surface"):
#         """   
#         In development.             
#         """

#         self.nodes = nodes
#         self.mesh = surface_mesh
        
#         self.bc_type = 'Contact'
#         BCBase.__init__(self, name)        
#         self._update_during_inc = 1
        
#     def __repr__(self):
#         list_str = ['Contact:']
#         if self.name != "": list_str.append("name = '{}'".format(self.name))
        
#         return "\n".join(list_str)
    
#     def initialize(self, problem):
#         pass
#         # for i,var in enumerate(self.var_cd):
#         #     if isinstance(var, str):
#         #         self.var_cd[i] = problem.space.variable_rank(var)
                

#     def generate(self, problem, t_fact=1, t_fact_old=None):
                
#         # mesh = problem.mesh                     
#         nodes = self.nodes
#         surf = self.mesh #surface mesh
        
#         res = ListBC()
#         #look for contact
#         #brut force, compute the distance between nodes and all elements
        
#         #get the normal surface on the center of the elements for each element on
#         #the master sufrace
#         element = get_element(surf.elm_type)
#         element.GetLocalFrame(surf.nodes[surf.elements], vec_xi)
        
        
        
        
        
#         res.initialize(problem)        
#         return res.generate(problem, t_fact, t_fact_old)
