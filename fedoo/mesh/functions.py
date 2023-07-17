# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 16:09:45 2023

@author: eprulier
"""
import numpy as np
from fedoo.core.mesh import Mesh
from fedoo.mesh.simple import line_mesh_1D
from fedoo.lib_elements.element_list import get_element 

try:
    import pyvista as pv
    # import vtk
    USE_PYVISTA = True
except:
    USE_PYVISTA = False

def extract_surface(mesh):
    """Build a mesh of the surface of a given 2D mesh.

    This function ensure that the normal of the surface elements are 
    oriented outside the volume.
    For now only usable for 2D meshes.     

    Parameters
    ----------
    mesh : fd.Mesh
        Mesh from which we want to extract the surface

    Returns
    -------
    Mesh
        Surface Mesh

    """
    
    # It can be used in 2D, and then the returned mesh will be composed of linear 
    # elements. If a 3D mesh is given, the surface mesh will be composed of 
    # 2D shell elements.
    
    if mesh.elm_type in ['tet4', 'tet10', 'hex8', 'hex20']:
        if USE_PYVISTA:
            #the normal orientation seems ok, but need to be ensure
            return Mesh.from_pyvista(mesh.to_pyvista.extract_surface())        
        else:
            raise NameError('Pyvista not found. Extraction of volume surface need pyvista.')
    
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

def extrude(mesh, extrude_path, n_nodes, use_local_frame = False, name = ""):
    
    if isinstance(extrude_path, Mesh): 
        mesh1 = extrude_path
    else:
        if np.isscalar(extrude_path):
            if mesh.elm_type == 'lin3': elm_type = 'lin3'
            else: elm_type = 'lin2'
                
            mesh1 = line_mesh_1D(n_nodes, x_min=0, x_max=extrude_path, elm_type = elm_type, name = "")
    
    n_el1 = mesh1.n_elements ;
    n_el = n_el1*mesh.n_elements
    n_elm_nd0 = mesh.n_elm_nodes
    n_elm_nd1 = mesh1.n_elm_nodes
    elm = np.zeros((n_el,n_elm_nd1*n_elm_nd0), dtype=int)      

    if mesh.elm_type == 'lin2': 
        dim_mesh = 1
        if mesh1.elm_type == 'lin2': 
            type_elm = 'quad4'    
            dim_mesh1 = 1
            for i in range(mesh.n_elements):
                elm[i*n_el1:(i+1)*n_el1 , [0,1]] = mesh1.elements + mesh.elements[i,0]*mesh1.n_nodes
                elm[i*n_el1:(i+1)*n_el1 , [3,2]] = mesh1.elements + mesh.elements[i,1]*mesh1.n_nodes
        elif mesh1.elm_type == 'quad4': 
            dim_mesh1 = 2
            type_elm = 'hex8'
            for i in range(mesh.n_elements):                
                elm[i*n_el1:(i+1)*n_el1 , 0:n_elm_nd1         ] = mesh1.elements + mesh.elements[i,0]*mesh1.n_nodes
                elm[i*n_el1:(i+1)*n_el1 , n_elm_nd1:2*n_elm_nd1] = mesh1.elements + mesh.elements[i,1]*mesh1.n_nodes                        
        else: raise NameError('Element not implemented')

    elif mesh.elm_type == 'lin3':     #need verification because the node numerotation for lin2 has changed             
        dim_mesh = 1
        if mesh1.elm_type == 'lin3': #mesh1 and mesh are lin3 elements
            dim_mesh1 = 1
            type_elm = 'quad9'
            for i in range(mesh.n_elements): #éléments 1D à 3 noeuds (pour le moment uniquement pour générer des éléments quad9)
                elm[i*n_el1:(i+1)*n_el1 , [0,4,1] ] = mesh1.elements + mesh.elements[i,0]*mesh1.n_nodes
                elm[i*n_el1:(i+1)*n_el1 , [7,8,5] ] = mesh1.elements + mesh.elements[i,1]*mesh1.n_nodes
                elm[i*n_el1:(i+1)*n_el1 , [3,6,2] ] = mesh1.elements + mesh.elements[i,2]*mesh1.n_nodes
        else: raise NameError('Element not implemented')
        
    elif mesh.elm_type == 'quad4':
        dim_mesh = 2
        if mesh1.elm_type == 'lin2':
            dim_mesh1 = 1
            type_elm = 'hex8'                        
            for i in range(n_el1):                
                elm[i::n_el1 , 0:n_elm_nd0         ] = mesh.elements*mesh1.n_nodes + mesh1.elements[i,0]
                elm[i::n_el1 , n_elm_nd0:2*n_elm_nd0] = mesh.elements*mesh1.n_nodes + mesh1.elements[i,1]
        else: raise NameError('Element not implemented')
        
    elif mesh.elm_type == 'tri3':
        dim_mesh = 2
        if mesh1.elm_type == 'lin2':
            dim_mesh1 = 1
            type_elm = 'wed6'                        
            for i in range(n_el1):                
                elm[i::n_el1 , 0:n_elm_nd0         ] = mesh.elements*mesh1.n_nodes + mesh1.elements[i,0]
                elm[i::n_el1 , n_elm_nd0:2*n_elm_nd0] = mesh.elements*mesh1.n_nodes + mesh1.elements[i,1]                
    else: raise NameError('Element not implemented') 
    
    if use_local_frame == False:       
        Ncrd = mesh1.n_nodes * mesh.n_nodes
#                crd = np.c_[np.tile(mesh1.nodes[:,:dim_mesh1],(mesh.n_nodes,1)), \
#                            np.reshape([np.ones((mesh1.n_nodes,1))*mesh.nodes[i,:dim_mesh] for i in range(mesh.n_nodes)] ,(Ncrd,-1)) ] 
        crd = np.c_[np.reshape([np.ones((mesh1.n_nodes,1))*mesh.nodes[i,:dim_mesh] for i in range(mesh.n_nodes)] ,(Ncrd,-1)), \
                    np.tile(mesh1.nodes[:,:dim_mesh1],(mesh.n_nodes,1))] 
    elif dim_mesh == 1: #dim_mesh is the thickness
        crd = np.zeros((mesh1.n_nodes*mesh.n_nodes, np.shape(mesh1.nodes)[1]))
        for i in range(mesh.n_nodes):
            crd[i*mesh1.n_nodes:(i+1)*mesh1.n_nodes,:] = mesh1.nodes + mesh1.local_frame[:,-1,:]*mesh.nodes[i][0]
    else: return NotImplemented
    
    return Mesh(crd, elm, type_elm, name =name)                        
   