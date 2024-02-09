import numpy as np
from fedoo.core.mesh import Mesh
from fedoo.mesh.simple import line_mesh, line_mesh_cylindric
from fedoo.mesh.functions import quad2tri, change_elm_type


def structured_mesh_2D(data, edge1, edge2, edge3, edge4, elm_type = 'quad4', method = 0, ndim = None, name =""):
#     #edge1 and edge3 should have the same lenght 
#     #edge2 and edge4 should have the same lenght
#     #last node of edge1 should be the first of edge2 and so on...
#     if no name is defined, the name is the same as crd
    
    if hasattr(data,'elm_type'): #data is a mesh        
        if data.elements is None: elm = []
        else: 
            elm = list(data.elements)
            elm_type = data.elm_type
        crd = data.nodes
        if name == "": name = data.name
    else: 
        elm = []
        crd = data
    
    edge3 = edge3[::-1]
    edge4 = edge4[::-1]
    x1 = crd[edge1] ; x2 = crd[edge2] ; x3 = crd[edge3] ; x4 = crd[edge4]
    new_crd = list(crd.copy())
    grid = np.empty((len(x1), len(x2)))
    grid[0,:] = edge4 ; grid[-1,:] = edge2
    grid[:,0] =  edge1 ; grid[:,-1] = edge3
    
    N = len(new_crd)
    coef1 = np.linspace(0,1,len(x1)).reshape(-1,1)
    coef2 = np.linspace(0,1,len(x2)).reshape(-1,1)
    
    for i in range(1,len(x1)-1):  
        if method == 0:     
            #intesection of lines dawn between nodes of oposite edges             
            pos = ( (x1[i,0]*x3[i,1]-x1[i,1]*x3[i,0])*(x2-x4)-(x1[i]-x3[i])*(x2[:,0]*x4[:,1]-x2[:,1]*x4[:,0]).reshape(-1,1) ) \
                / ( (x1[i,0]-x3[i,0])*(x2[:,1]-x4[:,1])-(x1[i,1]-x3[i,1])*(x2[:,0]-x4[:,0]) ).reshape(-1,1)                
            # px= ( (x1[i]*y3[i]-y1[i]*x3[i])*(x2-x4)-(x1[i]-x3[i])*(x2*y4-y2*x4) ) / ( (x1[i]-x3[i])*(y2-y4)-(y1[i]-y3[i])*(x2-x4) )         
            # py= ( (x1[i]*y3[i]-y1[i]*x3[i])*(y2-y4)-(y1[i]-y3[i])*(x2*y4-y2*x4) ) / ( (x1[i]-x3[i])*(y2-y4)-(y1[i]-y3[i])*(x2-x4) )
        elif method == 1:    
            #nodes are regularly distributed between edge1 and  edge3 and moved in the perpendicular direction to be 
            #as closed as possible to a regular distribution between edge2 and edge4
            if i == 1:
                # pos = x4
                vec = (x2 - pos)/np.linalg.norm(x2 - pos, axis=1).reshape(-1,1)
                
            #prediction of position
            pos1 = x1[i]*(1-coef2) + x3[i]*coef2
            #uncentainty of position (pos1 - pos2) where pos2 is the position usging another method
            dpos = x2*(coef1[i]) + x4*(1-coef1[i]) - pos1 #uncertainty about the best node position
            #direction vector normalized (pos1-pos_old)
            # vec = (x2 - pos)/np.linalg.norm(x2 - pos, axis=1).reshape(-1,1)
            #pos is modified only in the direction vec
            pos = np.sum(vec * dpos, axis=1).reshape(-1,1)*vec + pos1
        elif method == 2:
            #nodes are regularly distributed between edge1 and  edge3
            pos = x1[i]*(1-coef2) + x3[i]*coef2
        elif method == 3:
            #nodes are regularly distributed between edge 2 and edge4
            pos = x2*(coef1[i]) + x4*(1-coef1[i])        
            

        new_crd += list(pos[1:-1])
        # new_crd += list(np.c_[px[1:-1],py[1:-1]])
        grid[i,1:-1] = np.arange(N,len(new_crd),1)
        N = len(new_crd)        

    nx = grid.shape[0] ; ny = grid.shape[1]
    
    if elm_type == 'quad4':
        elm += [[grid[i,j], grid[i+1,j],grid[i+1,j+1], grid[i,j+1]] for j in range(ny-1) for i in range(nx-1)]            
    elif elm_type == 'quad9':                
        elm += [[grid[i,j],grid[i+2,j],grid[i+2,j+2],grid[i,j+2],grid[i+1,j],grid[i+2,j+1],grid[i+1,j+2],grid[i,j+1],grid[i+1,j+1]] for j in range(0,ny-2,2) for i in range(0,nx-2,2)]
    elif elm_type == 'tri3':    
        for j in range(ny-1):
            elm += [[grid[i,j],grid[i+1,j],grid[i,j+1]] for i in range(nx-1)]
            elm += [[grid[i+1,j],grid[i+1,j+1],grid[i,j+1]] for i in range(nx-1)]
    elif elm_type == 'tri6':
        for j in range(0,ny-2,2):
            elm += [[grid[i,j],grid[i+2,j],grid[i,j+2], grid[i+1,j],grid[i+1,j+1],grid[i,j+1]] for i in range(0,nx-2,2)]
            elm += [[grid[i+2,j],grid[i+2,j+2],grid[i,j+2], grid[i+2,j+1],grid[i+1,j+2],grid[i+1,j+1]] for i in range(0,nx-2,2)]
    else:
        raise NameError("'{}' elements are not implemented".format(elm_type))
    
    elm = np.array(elm, dtype=int)
    return Mesh(np.array(new_crd), elm, elm_type, ndim=ndim, name=name)


def generate_nodes(mesh, N, data, type_gen = 'straight'):
    """
    Add regularly espaced nodes to an existing mesh between to existing nodes.
    
    This function serve to generated structured meshes. 
    To create a 2D stuctured mesh: 
        - Create and mesh with only sigular nodes that will serve to build the edges
        - Use the generate_nodes functions to add some nodes to the edge
        - Use the structured_mesh_2D from the set of nodes corresponding the egdes to build the final mesh.

    Parameters
    ----------
    mesh : Mesh
        the existing mesh
    N : int
        Number of generated nodes.
    data : list or tuple
        if type_gen == 'straight', data should contain the indices of the starting (data[0]) and ending (data[1]).
        if type_gen == 'circular', data should contain the indices of the starting (data[0]) and ending (data[1]) nodes and the coordinates of the center of the circle (data[2])
    type_gen : str in {'straight', 'circular'}
        Type of line generated. The default is 'straight'.

    Returns
    -------
    np.ndarray[int]
        array containing indices of the new generated nodes

    """
    #if type_gen == 'straight' -> data = (node1, node2)
    #if type_gen == 'circular' -> data = (node1, node2, (center_x, center_y))
    crd = mesh.nodes    
    if type_gen == 'straight':
        node1 = data[0] ; node2 = data[1]
        listNodes = mesh.add_nodes(line_mesh(N, crd[node1], crd[node2]).nodes[1:-1])
        return np.array([node1]+list(listNodes)+[node2])
    if type_gen == 'circular':
        nd1 = data[0] ; nd2 = data[1] ; c = data[2]
        c = np.array(c)
        R = np.linalg.norm(crd[nd1]-c)
        assert np.abs(R-np.linalg.norm(crd[nd2]-c))<R*1e-4, "Final nodes is not on the circle"
        # (crd[nd1]-c)
        theta_min = np.arctan2(crd[nd1,1]-c[1],crd[nd1,0]-c[0])
        theta_max = np.arctan2(crd[nd2,1]-c[1],crd[nd2,0]-c[0])
        m = line_mesh_cylindric(N, R, theta_min, theta_max) #circular mesh
        listNodes = mesh.add_nodes(m.nodes[1:-1]+c)
        return np.array([nd1]+list(listNodes)+[nd2])


def hole_plate_mesh(nr=11, nt=11, length=100, height=100, radius=20, elm_type = 'quad4', sym= False, include_node_sets = True, ndim = None, name =""):
    """
    Create a mesh of a 2D plate with a hole  

    Parameters
    ----------
    nr, nt : int
        Numbers of nodes in the radial and tangent direction from the hole (default = 11). 
        nt is the number of nodes of the half of an exterior edge
    length, height : int,float
        The length and height of the plate (default : 100).
    radius : int,float
        The radius of the hole (default : 20).

    elm_type : {'quad4', 'quad9', 'tri3', 'tri6'}
        The type of the element generated (default='quad4')
    Sym : bool 
        Sym = True, if only the returned mesh assume symetric condition and 
        only the quarter of the plate is returned (default=False)
    include_node_sets : bool
        if True (default), the boundary nodes are included in the mesh node_sets dict.
        
    Returns
    -------
    Mesh
        The generated geometry in Mesh format. See the Mesh class for more details.        
    
    See Also
    --------
    line_mesh : 1D mesh of a line    
    rectangle_mesh : Surface mesh of a rectangle
    """   
    if elm_type in ['quad9', 'tri6']: 
        nr = nr//2*2+1 #in case nr is not initially odd
        nt = nt//2*2+1 #in case nt is not initially odd
    elif elm_type == 'quad8': 
        return change_elm_type(
                  hole_plate_mesh(nr, nt, length, height, radius, 'quad9', sym, include_node_sets, ndim, name),
                  'quad8',
                )
    elif elm_type not in ['quad4', 'tri3']: raise NameError('Non compatible element shape')
    
    if isinstance(radius, tuple):
        ellipse = True
        ellipse_radius = np.array(radius)
        radius = 1
    else: ellipse = False
        
    
    L = length/2
    h = height/2
    m = Mesh(np.array([[radius,0],[L,0],[L,h],[0,h],[0,radius],[radius*np.cos(np.pi/4),radius*np.sin(np.pi/4)]]))
    edge4 = generate_nodes(m,nt,(5,0,(0,0)), type_gen = 'circular')
    edge7 = generate_nodes(m,nt,(5,4,(0,0)), type_gen = 'circular')
    if ellipse: 
        m.nodes[edge4]*=ellipse_radius
        m.nodes[edge7[1:]]*=ellipse_radius
            
    edge1 = generate_nodes(m,nr,(0,1))
    edge2 = generate_nodes(m,nt,(1,2))
    edge3 = generate_nodes(m,nr,(2,5))
    
    edge5 = generate_nodes(m,nr,(4,3))
    edge6 = generate_nodes(m,nt,(3,2))
    
    m = structured_mesh_2D(m, edge1, edge2, edge3, edge4, elm_type = elm_type, method=3)
    m = structured_mesh_2D(m, edge5, edge6, edge3, edge7, elm_type = elm_type, method=3, ndim = ndim, name=name)
    
    if sym:      
        if include_node_sets:
            m.node_sets.update({'hole_edge': list(edge4[:0:-1])+list(edge7), 'right': list(edge2), 
                            'left_sym': list(edge5), 'top': list(edge6), 'bottom_sym': list(edge1)})
    else:
        nnd = m.n_nodes
        crd = m.nodes.copy()
        crd[:,0] = -m.nodes[:,0]
        m2 = Mesh(crd, m.elements, m.elm_type)
        m = Mesh.stack(m,m2)
                
        crd = m.nodes.copy()
        crd[:,1] = -m.nodes[:,1]
        m2 = Mesh(crd, m.elements, m.elm_type)
        m = Mesh.stack(m,m2, name=name)
        
        if include_node_sets:
            m.node_sets['top'] = list((edge6+nnd)[:0:-1])+list(edge6)
            m.node_sets['bottom'] = list((edge6+3*nnd)[:0:-1])+list(edge6+2*nnd)
            m.node_sets['right'] = list((edge2+2*nnd)[:0:-1]) + list(edge2)
            m.node_sets['left'] = list((edge2+3*nnd)[:0:-1]) + list(edge2+nnd) 
            edge_hole = np.hstack((edge4[:0:-1],edge7))
            m.node_sets['hole_edge'] = list(edge_hole) + list(edge_hole[-2::-1]+nnd) + \
                                       list(edge_hole[1:]+3*nnd) + list(edge_hole[-2:0:-1]+2*nnd)   
                
        node_to_merge = np.vstack((np.c_[edge5, edge5+nnd], 
                                   np.c_[edge5+2*nnd, edge5+3*nnd],
                                   np.c_[edge1, edge1+2*nnd],
                                   np.c_[edge1+nnd, edge1+3*nnd]))                                   
        
        m.merge_nodes(node_to_merge)    
    
    return m
    

def disk_mesh(radius=0.5, nx=11, ny=11, elm_type = 'quad4', ndim = None, name =""):
    if elm_type == 'quad8': 
        return change_elm_type(
                  disk_mesh(radius, nx, ny, 'quad9', ndim, name),
                  'quad8',
                )
    m = hole_plate_mesh(nx, ny, 0.5*radius, 0.5*radius, radius, elm_type)
    hole_edge = m.node_sets['hole_edge']

    m = structured_mesh_2D(m, m.node_sets['right'], 
                           m.node_sets['top'][::-1],
                           m.node_sets['left'][::-1],
                           m.node_sets['bottom'], elm_type, ndim = ndim, name=name)
    m.node_sets = {'boundary': hole_edge}
    
    return m
    # if elm_type == 'quad4': return m
    # elif elm_type == 'tri3': return quad2tri(m)
    # else: raise NameError('Non compatible element shape')
