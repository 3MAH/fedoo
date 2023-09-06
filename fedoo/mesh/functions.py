# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 16:09:45 2023

@author: eprulier
"""
import numpy as np
from fedoo.core.mesh import Mesh
from fedoo.mesh.simple import line_mesh_1D
from fedoo.lib_elements.element_list import get_element, get_node_elm_coordinates

try:
    import pyvista as pv
    # import vtk
    USE_PYVISTA = True
except:
    USE_PYVISTA = False

def extract_surface(mesh):
    """Build a mesh of the surface of a given 2D or 3D mesh.

    This function ensure that the normal of the surface elements are 
    oriented outside the volume.
    The returned surface mesh is based on tri3 elements for initial 3D meshes, 
    and on lin2 elements for initial 2D meshes.    

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
    
    # if mesh.elm_type in ['tet4', 'tet10', 'hex8', 'hex20']:
    #     if USE_PYVISTA:
    #         #the normal orientation seems ok, but need to be checked
    #         return Mesh.from_pyvista(mesh.to_pyvista().extract_surface())        
    #     else:
    #         raise NameError('Pyvista not found. Extraction of volume surface need pyvista.')
    
    if mesh.elm_type in ['quad4','quad8','quad9']:
        faces_in_elm = [[0,1],[1,2],[2,3],[3,0]]
        n_face_nodes = 2
        face_elm_type = 'lin2'
        dim = 1
    elif mesh.elm_type in ['tri3','tri6']:
        faces_in_elm = [[0,1],[1,2],[2,0]]
        n_face_nodes = 2
        face_elm_type = 'lin2'
        dim = 1
    elif mesh.elm_type in ['tet4', 'tet10']:
        faces_in_elm = [[0,1,2],[1,2,3],[2,3,0], [3,0,1]]
        n_face_nodes = 3
        face_elm_type = 'tri3'
        dim = 2
    elif mesh.elm_type in ['hex8', 'hex20']:
        faces_in_elm = [[0,1,2,3],[5,4,7,6],[0,4,5,1],[2,6,7,3],[1,5,6,2],[4,0,3,7]]
        n_face_nodes = 4
        face_elm_type = 'quad4'
        dim = 2
    # elif mesh.elm_type in ['hex8', 'hex20']:
    #     faces_in_elm = [[0,1,2], [0,2,3], [5,4,7], [5,7,6], [0,4,5], [0,5,1], 
    #                     [2,6,7], [2,7,3], [1,5,6], [1,6,2], [4,0,3], [4,3,7]]
    #     n_face_nodes = 3
    #     face_elm_type = 'tri3'
    #     dim = 2
    else: 
        raise NotImplementedError()
        
    n_faces_in_elm = len(faces_in_elm)
    list_faces = mesh.elements[:,faces_in_elm].reshape(-1,n_face_nodes) #shape = (n_elements, n_elm_faces, n_node_per_faces) before reshape
    
    test = np.sort(list_faces, axis=1)
    ind_sorted = np.lexsort(tuple((test[:,i-1] for i in range(n_face_nodes,0,-1))))
    # sorted_faces = [frozenset(face) for face in test[ind_sorted]]
    sorted_faces = test[ind_sorted]
    
    surf_elements = [] #list of the surface elements (ie exterior faces of the initial mesh)
    ind_element = [] #ind_element[i] is the indice of the element who contains the face surf_elements[i]
    i=1
    while i < len(sorted_faces):            
        if (sorted_faces[i] != sorted_faces[i-1]).any():     
            # surf_elements.append(list(sorted_faces[i-1]))
            surf_elements.append(list(list_faces[ind_sorted[i-1]]))
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
        
    nodes_inside=[np.setdiff1d(mesh.elements[ind_element[i]], face)[0] for i,face in enumerate(surf_elements)]

    if face_elm_type == 'quad4':
        #divide each quad into 2 tri
        face_elm_type = 'tri3' 
        surf_elements = np.hstack((surf_elements[:,:3], surf_elements[:,[0,2,3]])).reshape(-1,3)
        # surf_elements = np.vstack((surf_elements[:, :3], surf_elements[:,[0,2,3]]))
        nodes_inside = np.column_stack((nodes_inside, nodes_inside)).reshape(-1)
    
    #Check if normal are well defined (outside the volume)
    #idea: compute the distance of a point of the elm that is not the face itself    
        
    element_face = get_element(face_elm_type)(1)
    
    elm_nodes_crd = mesh.nodes[surf_elements]
    
    if dim == 1:
        local_frame = element_face.GetLocalFrame(elm_nodes_crd, element_face.get_gp_elm_coordinates(1))
    
        tangent_axis = local_frame[:,0,0]
        normal_axis = local_frame[:,0,-1]
        
        # The two following lines work only for 2 node 1d element    # length = np.linalg.norm(mesh.nodes[surf.elements[:,1]] - mesh.nodes[surf.elements[:,0]], axis = 1)
        length = np.linalg.norm(elm_nodes_crd[:,1,:] - elm_nodes_crd[:,0,:], axis = 1)    
        vec_xi = 1/length*((mesh.nodes[nodes_inside] - elm_nodes_crd[:,0,:]) * tangent_axis).sum(axis=1)
    else: #dim == 2
        #work only for tri3 face
        
        # tangent1 = elm_nodes_crd[:,1,:] - elm_nodes_crd[:,0,:]
        # tangent2 = elm_nodes_crd[:,2,:] - elm_nodes_crd[:,0,:]
        
        tangent = elm_nodes_crd[:,1:,:] - elm_nodes_crd[:,[0],:] #tangent[:,i,:] gives the ith tangent axis i in [0,1]. tangent are not orthogonal
        
        vec_xi = np.linalg.solve((tangent @ tangent.transpose([0,2,1])),       
                        np.sum(((mesh.nodes[nodes_inside] - elm_nodes_crd[:,0,:]).reshape(-1,1,3) * tangent), axis=2))
        
        # local_frame = element_face.GetLocalFrame(elm_nodes_crd, element_face.get_gp_elm_coordinates(1))
        # normal_axis = local_frame[:,0,-1]
        normal_axis = np.cross(tangent[:,0,:],tangent[:,1,:])
        normal_axis = normal_axis/np.linalg.norm(normal_axis,axis=1).reshape(-1,1)

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
    """
    Build a volume or surface mesh from the extrusion of a surface or wire mesh. 
    
    Parameters
    ----------
    mesh : fedoo.Mesh
        The mesh to extrude
    extrude_path : float, tuple[float] or fedoo.Mesh
        The path along which the mesh will be extruded. 
        extrude_path can either be:
        - a float: extrude_path is the extrusion thickness.
        - a tuple: extrude_path is the min and max coordinates values along the thickness.
        - a fedoo.Mesh with line elements: define the path along which the mesh is extruded.        
    n_nodes : int
        number of nodes in the extrusion direction. n_nodes is ignored if extrude_path is a Mesh.
    use_local_frame : bool
        If True, the extrusion use the nodal local_frame of the extrude_path Mesh 
        (if available). The default is False.
    name : str, optional
        The name of the final Mesh.

    Returns
    -------
    Mesh object
    """
    
    if isinstance(extrude_path, Mesh): 
        mesh1 = extrude_path
    else:
        if np.isscalar(extrude_path):
            if mesh.elm_type == 'lin3': elm_type = 'lin3'
            else: elm_type = 'lin2'
                
            mesh1 = line_mesh_1D(n_nodes, x_min=0, x_max=extrude_path, elm_type = elm_type, name = "")
        elif isinstance(extrude_path, tuple) and len(tuple) == 2: #assume iterable 
            mesh1 = line_mesh_1D(n_nodes, x_min=extrude_path[0], x_max=extrude_path[1], elm_type = elm_type, name = "")
        else:
            raise NameError('extrude_path argument not understood. ')
    
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


def change_elm_type(mesh, elm_type, name=""):
    """
    Attempt to change the type of element of the given mesh with the given elm_type.
    
    Work only if the two elements are compatible. This function may change the 
    order of the nodes and elements.    
        
    Parameters
    ----------
    mesh : fedoo.Mesh        
        The mesh to modify
    elm_type : str
        New element type
    name : str, optional
        name of the new mesh.

    Returns
    -------
    fedoo.Mesh
    
    Notes
    ------
    This function is costly and not optimized. It should be avoided for high dimension 
    meshes.
    
    """
    elm_ref = get_element(mesh.elm_type)(0)
    xi_nd = get_node_elm_coordinates(elm_type)
    
    if len(xi_nd) <= elm_ref.n_nodes: 
        #the new elm_type is of lower order thant the previous one
        return Mesh(mesh.nodes, mesh.elements[:,:elm_ref.n_nodes], elm_type, name=name)
        
    xi_nd = xi_nd[elm_ref.n_nodes:]
    shape_func_val = elm_ref.ShapeFunction(xi_nd)
    
    new_nodes_crd = shape_func_val @ mesh.nodes[mesh.elements]
    new_nodes = np.vstack((mesh.nodes,new_nodes_crd.reshape(-1, mesh.ndim)))
    
    new_nodes_ind = np.arange(mesh.n_nodes,mesh.n_nodes+mesh.n_elements*new_nodes_crd.shape[1])
    new_nodes_ind = new_nodes_ind.reshape(mesh.n_elements, -1)

    new_elm = np.empty((mesh.n_elements,elm_ref.n_nodes+len(xi_nd)), dtype=int)
    new_elm[:,:elm_ref.n_nodes] = mesh.elements
    new_elm[:,elm_ref.n_nodes:] = new_nodes_ind
    
    new_mesh = Mesh(new_nodes, new_elm, elm_type, name = name)

    new_mesh.merge_nodes(new_mesh.find_coincident_nodes()) #very slow strategy
    
    #○r using pyvista. try to see if it is more efficient
    # new_mesh = new_mesh.to_pyvista().clean(tol=1e-6, remove_unused_points=False)
    # new_mesh = new_mesh.from_pyvista()
    return new_mesh
    
    