"""This module contains functions to generate simple meshes"""

from fedoo.core.mesh import Mesh
from fedoo.core.modelingspace import ModelingSpace
import itertools

# import scipy as sp
import numpy as np

# utility fuctions
# Only Functions are declared here !!
def stack(mesh1,mesh2, name = ""):
        """
        Make the spatial stack of two mesh objects which have the same element shape. 
        This function doesn't merge coindicent Nodes. 
        For that purpose, use the Mesh methods 'find_coincident_nodes' and 'merge_nodes'
        on the resulting Mesh. 
                
        Return 
        ---------
        Mesh object with is the spacial stack of mesh1 and mesh2
        """
        return Mesh.stack(mesh1,mesh2,name)


def rectangle_mesh(nx=11, ny=11, x_min=0, x_max=1, y_min=0, y_max=1, elm_type = 'quad4', ndim = None, name =""):
    """
    Create a rectangular Mesh

    Parameters
    ----------
    nx, ny : int
        Numbers of nodes in the x and y axes (default = 11).
    x_min, x_max, y_min, y_max : int,float
        The boundary of the square (default : 0, 1, 0, 1).
    elm_type : {'tri3', 'quad4', 'quad8', 'quad9'}
        The type of the element generated (default='quad4')

        * 'tri3' -- 3 node linear triangular mesh
        * 'tri6' -- 6 node linear triangular mesh
        * 'quad4' -- 4 node quadrangular mesh
        * 'quad8' -- 8 node quadrangular mesh (à tester)      
        * 'quad9' -- 9 node quadrangular mesh

    Returns
    -------
    Mesh
        The generated geometry in Mesh format. See the Mesh class for more details.

    See Also
    --------
    line_mesh : 1D mesh of a line    
    rectangle_mesh : Surface mesh of a rectangle
    box_mesh : Volume mesh of a box 
    grid_mesh_cylindric : Surface mesh of a grid in cylindrical coodrinate 
    line_mesh_cylindric : Line mesh in cylindrical coordinate
    """
        
    if elm_type == 'quad9' or elm_type == 'tri6': 
        nx=int(nx//2*2+1) ; ny=int(ny//2*2+1) #pour nombre impair de noeuds 
    X,Y = np.meshgrid(np.linspace(x_min,x_max,nx),np.linspace(y_min,y_max,ny))    
    crd = np.c_[np.reshape(X,(-1,1)),np.reshape(Y,(-1,1))]
    if elm_type == 'quad8':
        dx = (x_max-x_min)/(nx-1.) ; dy = (y_max-y_min)/(ny-1.)
        X,Y = np.meshgrid(np.linspace(x_min+dx/2.,x_max-dx/2.,nx-1),np.linspace(y_min,y_max,ny))    
        crd2 = np.c_[np.reshape(X,(-1,1)),np.reshape(Y,(-1,1))]
        X,Y = np.meshgrid(np.linspace(x_min,x_max,nx),np.linspace(y_min+dy/2,y_max-dy/2,ny-1))    
        crd3 = np.c_[np.reshape(X,(-1,1)),np.reshape(Y,(-1,1))]
        crd = np.vstack((crd,crd2,crd3))
        elm = [[nx*j+i,nx*j+i+1,nx*(j+1)+i+1,nx*(j+1)+i, nx*ny+(nx-1)*j+i, nx*ny+(nx-1)*ny+nx*j+i+1 , nx*ny+(nx-1)*(j+1)+i, nx*ny+(nx-1)*ny+nx*j+i] for j in range(0,ny-1) for i in range(0,nx-1)]
    elif elm_type == 'quad4':
        elm = [[nx*j+i,nx*j+i+1,nx*(j+1)+i+1,nx*(j+1)+i] for j in range(ny-1) for i in range(nx-1)]            
    elif elm_type == 'quad9':                
        elm = [[nx*j+i,nx*j+i+2,nx*(j+2)+i+2,nx*(j+2)+i,nx*j+i+1,nx*(j+1)+i+2,nx*(j+2)+i+1,nx*(j+1)+i,nx*(j+1)+i+1] for j in range(0,ny-2,2) for i in range(0,nx-2,2)]
    elif elm_type == 'tri3':
        elm = []    
        for j in range(ny-1):
            elm += [[nx*j+i,nx*j+i+1,nx*(j+1)+i] for i in range(nx-1)]
            elm += [[nx*j+i+1,nx*(j+1)+i+1,nx*(j+1)+i] for i in range(nx-1)]
    elif elm_type == 'tri6':
        elm = []          
        for j in range(0,ny-2,2):
            elm += [[nx*j+i,nx*j+i+2,nx*(j+2)+i, nx*j+i+1,nx*(j+1)+i+1,nx*(j+1)+i] for i in range(0,nx-2,2)]
            elm += [[nx*j+i+2,nx*(j+2)+i+2,nx*(j+2)+i, nx*(j+1)+i+2,nx*(j+2)+i+1,nx*(j+1)+i+1] for i in range(0,nx-2,2)]
            
    elm = np.array(elm)

    returned_mesh = Mesh(crd, elm, elm_type, ndim, name)
    if elm_type != 'quad8':
        N = returned_mesh.n_nodes
        returned_mesh.add_node_set([nd for nd in range(nx)], 'bottom')
        returned_mesh.add_node_set([nd for nd in range(N-nx,N)], 'top')
        returned_mesh.add_node_set([nd for nd in range(0,N,nx)], 'left')
        returned_mesh.add_node_set([nd for nd in range(nx-1,N,nx)], 'right')
    else: 
        print('Warning: no boundary set of nodes defined for quad8 elements')    

    return returned_mesh
    

def grid_mesh_cylindric(nr=11, nt=11, r_min=0, r_max=1, theta_min=0, theta_max=1, elm_type = 'quad4', init_rep_loc = 0, ndim = None, name = ""):  
    """
    Create a mesh as a regular grid in cylindrical coordinate

    Parameters
    ----------
    nr, nt : int
        Numbers of nodes in the r and theta axes (default = 11).
    x_min, x_max, y_min, y_max : int,float
        The boundary of the square (default : 0, 1, 0, 1).
    elm_type : {'tri3', 'quad4', 'quad8', 'quad9'}
        The type of the element generated (default='quad4')
        * 'tri3' -- 3 node linear triangular mesh
        * 'quad4' -- 4 node quadrangular mesh
        * 'quad8' -- 8 node quadrangular mesh (à tester)      
        * 'quad9' -- 9 node quadrangular mesh
    init_rep_loc : {0, 1} 
        if init_rep_loc is set to 1, the local basis is initialized with the global basis.

    Returns
    -------
    Mesh
        The generated geometry in Mesh format. See the Mesh class for more details.
        
    See Also
    --------
    line_mesh : 1D mesh of a line    
    rectangle_mesh : Surface mesh of a rectangle
    box_mesh : Volume mesh of a box     
    line_mesh_cylindric : Line mesh in cylindrical coordinate    
    """
    
    if theta_min<theta_max: 
        m = rectangle_mesh(nr, nt, r_min, r_max, theta_min, theta_max, elm_type, ndim, name)
    else: 
        m = rectangle_mesh(nr, nt, r_min, r_max, theta_max, theta_min, elm_type, ndim, name)

    r = m.nodes[:,0]
    theta = m.nodes[:,1]
    crd = np.c_[r*np.cos(theta) , r*np.sin(theta)] 
    returned_mesh = Mesh(crd, m.elements, elm_type, ndim, name)
    returned_mesh.local_frame = m.local_frame

    if theta_min<theta_max: 
        returned_mesh.add_node_set(m.node_sets["left"], "bottom")
        returned_mesh.add_node_set(m.node_sets["right"], "top")
        returned_mesh.add_node_set(m.node_sets["bottom"], "left")
        returned_mesh.add_node_set(m.node_sets["top"], "right")
    else: 
        returned_mesh.add_node_set(m.node_sets["left"], "bottom")
        returned_mesh.add_node_set(m.node_sets["right"], "top")
        returned_mesh.add_node_set(m.node_sets["top"], "left")
        returned_mesh.add_node_set(m.node_sets["bottom"], "right")
    
    return returned_mesh
    

def line_mesh_1D(n_nodes=11, x_min=0, x_max=1, elm_type = 'lin2', name = ""):            
    """
    Create the Mesh of a straight line with corrdinates in 1D. 

    Parameters
    ----------
    n_nodes : int
        Numbers of nodes (default = 11).
    x_min, x_max : int,float
        The boundary of the line (default : 0, 1).
    elm_type : {'lin2', 'lin3', 'lin4'}
        The shape of the elements (default='lin2')

        * 'lin2' -- 2 node line
        * 'lin3' -- 3 node line
        * 'lin4' -- 4 node line  

    Returns
    -------
    Mesh
        The generated geometry in Mesh format. See the Mesh class for more details.
    
    See Also
    --------     
    line_mesh : Mesh of a line whith choosen dimension
    rectangle_mesh : Surface mesh of a rectangle
    box_mesh : Volume mesh of a box 
    grid_mesh_cylindric : Surface mesh of a grid in cylindrical coodrinate 
    line_mesh_cylindric : Line mesh in cylindrical coordinate
    """
    if elm_type == 'lin2': #1D element with 2 nodes
        crd = np.c_[np.linspace(x_min,x_max,n_nodes)] #Nodes coordinates 
        elm = np.c_[range(n_nodes-1), np.arange(1,n_nodes)] #Elements
    elif elm_type == 'lin3': #1D element with 3 nodes
        n_nodes = n_nodes//2*2+1 #In case N is not initially odd
        crd = np.c_[np.linspace(x_min,x_max,n_nodes)] #Nodes coordinates 
        elm = np.c_[np.arange(0,n_nodes-2,2), np.arange(1,n_nodes-1,2), np.arange(2,n_nodes,2)] #Elements
    elif elm_type == 'lin4':
        n_nodes = n_nodes//3*3+1 
        crd = np.c_[np.linspace(x_min,x_max,n_nodes)] #Nodes coordinates
        elm = np.c_[np.arange(0,n_nodes-3,3), np.arange(1,n_nodes-2,3), np.arange(2,n_nodes-1,3), np.arange(3,n_nodes,3)] #Elements 

    returned_mesh = Mesh(crd, elm, elm_type, name=name)
    returned_mesh.add_node_set([0], "left")
    returned_mesh.add_node_set([n_nodes-1], "right")
    
    return returned_mesh


def line_mesh(n_nodes=11, x_min=0, x_max=1, elm_type = 'lin2', ndim = None, name = ""):    
    """
    Create the Mesh of a straight line

    Parameters
    ----------
    n_nodes : int
        Numbers of nodes (default = 11).
    x_min, x_max : int,float,list
        The boundary of the line as scalar (1D) or list (default : 0, 1).
    elm_type : {'lin2', 'lin3', 'lin4'}
        The shape of the elements (default='lin2')
        * 'lin2' -- 2 node line
        * 'lin3' -- 3 node line
        * 'lin4' -- 4 node line  

    Returns
    -------
    Mesh
        The generated geometry in Mesh format. See the Mesh class for more details.
        
    See Also
    --------     
    rectangle_mesh : Surface mesh of a rectangle
    box_mesh : Volume mesh of a box 
    grid_mesh_cylindric : Surface mesh of a grid in cylindrical coodrinate 
    line_mesh_cylindric : Line mesh in cylindrical coordinate
    """
    if np.isscalar(x_min):
        m = line_mesh_1D(n_nodes,x_min,x_max,elm_type,name)    
        # if ModelingSpace.get_dimension() in ['2Dplane', '2Dstress'] : dim = 2
        # else: dim = 3
        crd = np.c_[m.nodes, np.zeros((n_nodes,ndim-1))]
        elm = m.elements
        returned_mesh = Mesh(crd,elm,elm_type, name = name)
        returned_mesh.add_node_set(m.node_sets["left"], "left")
        returned_mesh.add_node_set(m.node_sets["right"], "right")
    else: 
        m = line_mesh_1D(n_nodes,0.,1.,elm_type,name)    
        crd = m.nodes
        crd = (np.array(x_max)-np.array(x_min))*crd+np.array(x_min)
        elm = m.elements
        returned_mesh = Mesh(crd,elm,elm_type, name = name)
        returned_mesh.add_node_set(m.node_sets["left"], "left")
        returned_mesh.add_node_set(m.node_sets["right"], "right")
        
    return returned_mesh
    

def line_mesh_cylindric(nt=11, r=1, theta_min=0, theta_max=3.14, elm_type = 'lin2', init_rep_loc = 0, ndim = None, name = ""):
    """
    Create the mesh of a curved line based on cylindrical coordinates  

    Parameters
    ----------
    nt : int
        Numbers of nodes along the angular coordinate (default = 11).
    theta_min, theta_max : int,float
        The boundary of the line defined by the angular coordinate (default : 0, 3.14).
    elm_type : {'lin2', 'lin3', 'lin4'}
        The shape of the elements (default='lin2')
        * 'lin2' -- 2 node line
        * 'lin3' -- 3 node line
        * 'lin4' -- 4 node line  
    init_rep_loc : {0, 1} 
        if init_rep_loc is set to 1, the local frame is initialized with the cylindrical local basis.

    Returns
    -------
    Mesh
        The generated geometry in Mesh format. See the Mesh class for more details.
        
    See Also
    --------     
    line_mesh : Mesh of a line whith choosen dimension
    rectangle_mesh : Surface mesh of a rectangle
    box_mesh : Volume mesh of a box 
    grid_mesh_cylindric : Surface mesh of a grid in cylindrical coodrinate 
    line_mesh_cylindric : Line mesh in cylindrical coordinate
    """
    # init_rep_loc = 1 si on veut initialiser le repère local (0 par défaut)
    m = line_mesh_1D(nt,theta_min,theta_max,elm_type,name)       
    theta = m.nodes[:,0]
    elm = m.elements  

    crd = np.c_[r*np.cos(theta), r*np.sin(theta)]
        
    returned_mesh = Mesh(crd,elm,elm_type, ndim, name)
    returned_mesh.add_node_set(m.node_sets["left"], "left")
    returned_mesh.add_node_set(m.node_sets["right"], "right")
    
    if init_rep_loc: 
        local_frame = np.array( [ [[np.sin(t),-np.cos(t)],[np.cos(t),np.sin(t)]] for t in theta ] ) 
    
    return returned_mesh


def box_mesh(nx=11, ny=11, nz=11, x_min=0, x_max=1, y_min=0, y_max=1, z_min=0, z_max=1, elm_type = 'hex8', name = ""):      
    """
    Create the mesh of a box  

    Parameters
    ----------
    nx, ny, nz : int
        Numbers of nodes in the x, y and z axes (default = 11).
    x_min, x_max, y_min, y_max, z_min, z_max : int,float
        The boundary of the box (default : 0, 1, 0, 1, 0, 1).
    elm_type : {'hex8', 'hex20'}
        The type of the element generated (default='hex8')
        * 'hex8' -- 8 node hexahedron
        * 'hex20' -- 20 node second order hexahedron

    Returns
    -------
    Mesh
        The generated geometry in Mesh format. See the Mesh class for more details.        
    
    See Also
    --------
    line_mesh : 1D mesh of a line    
    rectangle_mesh : Surface mesh of a rectangle
    grid_mesh_cylindric : Surface mesh of a grid in cylindrical coodrinate 
    line_mesh_cylindric : Line mesh in cylindrical coord
    """
            
    Y,Z,X = np.meshgrid(np.linspace(y_min,y_max,ny),np.linspace(z_min,z_max,nz),np.linspace(x_min,x_max,nx))    
    crd = np.c_[np.reshape(X,(-1,1)),np.reshape(Y,(-1,1)),np.reshape(Z,(-1,1))]
    
    if elm_type == 'hex20':
        dx = (x_max-x_min)/(nx-1.) ; dy = (y_max-y_min)/(ny-1.) ; dz = (z_max-z_min)/(nz-1.)
        Y,Z,X = np.meshgrid(np.linspace(y_min,y_max,ny),np.linspace(z_min,z_max,nz),np.linspace(x_min+dx/2.,x_max+dx/2.,nx-1, endpoint=False))    
        crd2 = np.c_[np.reshape(X,(-1,1)),np.reshape(Y,(-1,1)),np.reshape(Z,(-1,1))]           
        Y,Z,X = np.meshgrid(np.linspace(y_min,y_max,ny),np.linspace(z_min+dz/2.,z_max+dz/2.,nz-1, endpoint=False),np.linspace(x_min,x_max,nx))    
        crd3 = np.c_[np.reshape(X,(-1,1)),np.reshape(Y,(-1,1)),np.reshape(Z,(-1,1))]        
        Y,Z,X = np.meshgrid(np.linspace(y_min+dy/2.,y_max+dy/2.,ny-1, endpoint=False),np.linspace(z_min,z_max,nz),np.linspace(x_min,x_max,nx))    
        crd4 = np.c_[np.reshape(X,(-1,1)),np.reshape(Y,(-1,1)),np.reshape(Z,(-1,1))]
     
        crd = np.vstack((crd,crd2,crd3,crd4))
        
        elm = [[nx*j+i+(k*nx*ny), nx*j+i+1+(k*nx*ny), nx*(j+1)+i+1+(k*nx*ny), nx*(j+1)+i+(k*nx*ny), \
                nx*j+i+(k*nx*ny)+nx*ny, nx*j+i+1+(k*nx*ny)+nx*ny, nx*(j+1)+i+1+(k*nx*ny)+nx*ny, nx*(j+1)+i+(k*nx*ny)+nx*ny, \
                nx*ny*nz+(nx-1)*j+i+k*(nx-1)*ny,nx*j+i+1+nx*ny*nz+(nx-1)*ny*nz+(nz-1)*nx*ny+(k*nx*(ny-1)),nx*ny*nz+(nx-1)*(j+1)+i+k*(nx-1)*ny,nx*j+i+nx*ny*nz+(nx-1)*ny*nz+(nz-1)*nx*ny+(k*nx*(ny-1)), \
                nx*ny*nz+(nx-1)*ny*nz+nx*j+i+k*ny*nx,nx*ny*nz+(nx-1)*ny*nz+nx*j+i+1+k*ny*nx,nx*ny*nz+(nx-1)*ny*nz+nx+nx*j+i+1+k*ny*nx,nx*ny*nz+(nx-1)*ny*nz+nx+nx*j+i+k*ny*nx, \
                nx*ny*nz+(nx-1)*ny+(nx-1)*j+i+k*(nx-1)*ny,nx*j+i+1+nx*ny*nz+(nx-1)*ny*nz+(nz-1)*nx*ny+nx*(ny-1)+(k*nx*(ny-1)),nx*ny*nz+(nx-1)*ny+(nx-1)*(j+1)+i+k*(nx-1)*ny,nx*j+i+nx*ny*nz+(nx-1)*ny*nz+(nz-1)*nx*ny+nx*(ny-1)+(k*nx*(ny-1))] for k in range(nz-1) for j in range(ny-1) for i in range(nx-1)] 
                
        bottom = [nd for nd in range(ny*nx)]+list(range(nx*ny*nz,(nx-1)*ny+nx*ny*nz))+list(range(nx*ny*nz+(nx-1)*ny*nz+(nz-1)*nx*ny, nx*ny*nz+(nx-1)*ny*nz+(nz-1)*nx*ny+(ny-1)*nx))
        top = [nd for nd in range((nz-1)*ny*nx,nz*nx*ny)]+list(range(nx*ny*nz+(nx-1)*ny*(nz-1),nx*ny*nz+(nx-1)*ny*nz))+list(range(nx*ny*nz+(nx-1)*ny*nz+(nz-1)*nx*ny+(ny-1)*nx*(nz-1), nx*ny*nz+(nx-1)*ny*nz+(nz-1)*nx*ny+(ny-1)*nx*nz))
        left = list(itertools.chain.from_iterable([range(i*nx*ny,i*nx*ny+nx*ny,nx) for i in range(nz)]))+list(range(nx*ny*nz+(nx-1)*ny*nz, nx*ny*nz+(nx-1)*ny*nz+(nz-1)*nx*ny,nx))+list(range(nx*ny*nz+(nx-1)*ny*nz+(nz-1)*nx*ny, nx*ny*nz+(nx-1)*ny*nz+(nz-1)*nx*ny+nx*(ny-1)*nz,nx))
        right = list(itertools.chain.from_iterable([range(i*nx*ny+nx-1,i*nx*ny+nx*ny,nx) for i in range(nz)]))+list(range(nx*ny*nz+(nx-1)*ny*nz+nx-1,nx*ny*nz+(nx-1)*ny*nz+(nz-1)*nx*ny,nx))+list(range(nx*ny*nz+(nx-1)*ny*nz+(nz-1)*nx*ny+nx-1, nx*ny*nz+(nx-1)*ny*nz+(nz-1)*nx*ny+nx*(ny-1)*nz,nx))
        front = list(itertools.chain.from_iterable([range(i*nx*ny,i*nx*ny+nx) for i in range(nz)]))+list(itertools.chain.from_iterable([range(i*(nx-1)*ny+nx*ny*nz, i*(nx-1)*ny+nx*ny*nz+nx-1) for i in range(nz)]))+list(itertools.chain.from_iterable([range(i*nx*ny+nx*ny*nz+(nx-1)*ny*nz, i*nx*ny+nx*ny*nz+(nx-1)*ny*nz+nx) for i in range(nz-1)]))
        back = list(itertools.chain.from_iterable([range(i*nx*ny+nx*ny-nx,i*nx*ny+nx*ny) for i in range(nz)]))+list(itertools.chain.from_iterable([range(i*(nx-1)*ny+nz*nx*ny+(nx-1)*(ny-1), i*(nx-1)*ny+nz*nx*ny+(nx-1)*(ny-1)+nx-1) for i in range(nz)]))+list(itertools.chain.from_iterable([range(i*nx*ny+nz*nx*ny+(nx-1)*ny*nz+(ny-1)*nx,i*nx*ny+nz*nx*ny+(nx-1)*ny*nz+(ny-1)*nx+nx) for i in range(nz-1)]))
        
    elif elm_type == 'hex8':
        elm = [[nx*j+i+(k*nx*ny),nx*j+i+1+(k*nx*ny),nx*(j+1)+i+1+(k*nx*ny),nx*(j+1)+i+(k*nx*ny),nx*j+i+(k*nx*ny)+nx*ny,nx*j+i+1+(k*nx*ny)+nx*ny,nx*(j+1)+i+1+(k*nx*ny)+nx*ny,nx*(j+1)+i+(k*nx*ny)+nx*ny] for k in range(nz-1) for j in range(ny-1) for i in range(nx-1)]     

        front = list(itertools.chain.from_iterable([range(i*nx*ny,i*nx*ny+nx) for i in range(nz)]))      # [item for sublist in bas for item in sublist] #flatten a list
        back = list(itertools.chain.from_iterable([range(i*nx*ny+nx*ny-nx,i*nx*ny+nx*ny) for i in range(nz)]))
        left = list(itertools.chain.from_iterable([range(i*nx*ny,i*nx*ny+nx*ny,nx) for i in range(nz)]))
        right = list(itertools.chain.from_iterable([range(i*nx*ny+nx-1,i*nx*ny+nx*ny,nx) for i in range(nz)]))
        bottom = [nd for nd in range(ny*nx)]
        top = [nd for nd in range((nz-1)*ny*nx,nz*nx*ny)]

    else:
        raise NameError('Element not implemented. Only support hex8 and hex20 elements')
        
    N = np.shape(crd)[0]
    elm = np.array(elm)
    
    returned_mesh = Mesh(crd, elm, elm_type, name = name)  
    for i, ndSet in enumerate([right, left, top, bottom, front, back]):
        ndSetId = ('right', 'left', 'top', 'bottom', 'front', 'back')[i]
        returned_mesh.add_node_set(ndSet, ndSetId)
                 
    return returned_mesh


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
    elif elm_type == 'quad8':
        raise NameError("'quad8' elements are not implemented")
    
    elm = np.array(elm, dtype=int)
    return Mesh(np.array(new_crd), elm, elm_type, ndim, name)


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
        (crd[nd1]-c)
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
    L = length/2
    h = height/2
    m = Mesh(np.array([[radius,0],[L,0],[L,h],[0,h],[0,radius],[radius*np.cos(np.pi/4),radius*np.sin(np.pi/4)]]))
    edge1 = generate_nodes(m,nr,(0,1))
    edge2 = generate_nodes(m,nt,(1,2))
    edge3 = generate_nodes(m,nr,(2,5))
    edge4 = generate_nodes(m,nt,(5,0,(0,0)), type_gen = 'circular')
    
    edge5 = generate_nodes(m,nr,(4,3))
    edge6 = generate_nodes(m,nt,(3,2))
    edge7 = generate_nodes(m,nt,(5,4,(0,0)), type_gen = 'circular')
    
    m = structured_mesh_2D(m, edge1, edge2, edge3, edge4, elm_type = 'quad4', method=3)
    m = structured_mesh_2D(m, edge5, edge6, edge3, edge7, elm_type = 'quad4', method=3, ndim = ndim, name=name)
    
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
        
    if elm_type == 'quad4': return m
    elif elm_type == 'tri3': return quad2tri(m)
    else: raise NameError('Non compatible element shape')
    

def disk_mesh(radius=0.5, nx=11, ny=11, elm_type = 'quad4', ndim = None, name =""):
    m = hole_plate_mesh(nx, ny, 0.5*radius, 0.5*radius, radius, elm_type = 'quad4')
    hole_edge = m.node_sets['hole_edge']

    m = structured_mesh_2D(m, m.node_sets['right'], 
                           m.node_sets['top'][::-1],
                           m.node_sets['left'][::-1],
                           m.node_sets['bottom'], elm_type = 'quad4', ndim = ndim, name=name)
    m.node_sets = {'boundary': hole_edge}
    
    if elm_type == 'quad4': return m
    elif elm_type == 'tri3': return quad2tri(m)
    else: raise NameError('Non compatible element shape')

    


def quad2tri(mesh):
    assert mesh.elm_type == 'quad4', "element shape should be 'quad4' for quad2tri"
    crd = mesh.nodes
    elm = mesh.elements
    return Mesh(crd, np.vstack( [elm[:,0:3], elm[:,[0,2,3]]]), 'tri3')
                
if __name__=="__main__":
    import math
    a = line_mesh_cylindric(11, 1, 0, math.pi, 'lin2', init_rep_loc = 0)
    b = line_mesh_cylindric(11, 1, 0, math.pi, 'lin4', init_rep_loc = 1)

    print(b.nodes)

