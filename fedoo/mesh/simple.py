"""This module contains functions to generate simple meshes"""

from fedoo.core.mesh import Mesh
from fedoo.utilities.modelingspace import ModelingSpace
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


def rectangle_mesh(Nx=11, Ny=11, x_min=0, x_max=1, y_min=0, y_max=1, ElementShape = 'quad4', ndim = None, name =""):
    """
    Create a rectangular Mesh

    Parameters
    ----------
    Nx, Ny : int
        Numbers of nodes in the x and y axes (default = 11).
    x_min, x_max, y_min, y_max : int,float
        The boundary of the square (default : 0, 1, 0, 1).
    ElementShape : {'tri3', 'quad4', 'quad8', 'quad9'}
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
        
    if ElementShape == 'quad9' or ElementShape == 'tri6': 
        Nx=int(Nx//2*2+1) ; Ny=int(Ny//2*2+1) #pour nombre impair de noeuds 
    X,Y = np.meshgrid(np.linspace(x_min,x_max,Nx),np.linspace(y_min,y_max,Ny))    
    crd = np.c_[np.reshape(X,(-1,1)),np.reshape(Y,(-1,1))]
    if ElementShape == 'quad8':
        dx = (x_max-x_min)/(Nx-1.) ; dy = (y_max-y_min)/(Ny-1.)
        X,Y = np.meshgrid(np.linspace(x_min+dx/2.,x_max-dx/2.,Nx-1),np.linspace(y_min,y_max,Ny))    
        crd2 = np.c_[np.reshape(X,(-1,1)),np.reshape(Y,(-1,1))]
        X,Y = np.meshgrid(np.linspace(x_min,x_max,Nx),np.linspace(y_min+dy/2,y_max-dy/2,Ny-1))    
        crd3 = np.c_[np.reshape(X,(-1,1)),np.reshape(Y,(-1,1))]
        crd = np.vstack((crd,crd2,crd3))
        elm = [[Nx*j+i,Nx*j+i+1,Nx*(j+1)+i+1,Nx*(j+1)+i, Nx*Ny+(Nx-1)*j+i, Nx*Ny+(Nx-1)*Ny+Nx*j+i+1 , Nx*Ny+(Nx-1)*(j+1)+i, Nx*Ny+(Nx-1)*Ny+Nx*j+i] for j in range(0,Ny-1) for i in range(0,Nx-1)]
    elif ElementShape == 'quad4':
        elm = [[Nx*j+i,Nx*j+i+1,Nx*(j+1)+i+1,Nx*(j+1)+i] for j in range(Ny-1) for i in range(Nx-1)]            
    elif ElementShape == 'quad9':                
        elm = [[Nx*j+i,Nx*j+i+2,Nx*(j+2)+i+2,Nx*(j+2)+i,Nx*j+i+1,Nx*(j+1)+i+2,Nx*(j+2)+i+1,Nx*(j+1)+i,Nx*(j+1)+i+1] for j in range(0,Ny-2,2) for i in range(0,Nx-2,2)]
    elif ElementShape == 'tri3':
        elm = []    
        for j in range(Ny-1):
            elm += [[Nx*j+i,Nx*j+i+1,Nx*(j+1)+i] for i in range(Nx-1)]
            elm += [[Nx*j+i+1,Nx*(j+1)+i+1,Nx*(j+1)+i] for i in range(Nx-1)]
    elif ElementShape == 'tri6':
        elm = []          
        for j in range(0,Ny-2,2):
            elm += [[Nx*j+i,Nx*j+i+2,Nx*(j+2)+i, Nx*j+i+1,Nx*(j+1)+i+1,Nx*(j+1)+i] for i in range(0,Nx-2,2)]
            elm += [[Nx*j+i+2,Nx*(j+2)+i+2,Nx*(j+2)+i, Nx*(j+1)+i+2,Nx*(j+2)+i+1,Nx*(j+1)+i+1] for i in range(0,Nx-2,2)]
            
    elm = np.array(elm)

    returned_mesh = Mesh(crd, elm, ElementShape, ndim, name)
    if ElementShape != 'quad8':
        N = returned_mesh.n_nodes
        returned_mesh.add_node_set([nd for nd in range(Nx)], 'bottom')
        returned_mesh.add_node_set([nd for nd in range(N-Nx,N)], 'top')
        returned_mesh.add_node_set([nd for nd in range(0,N,Nx)], 'left')
        returned_mesh.add_node_set([nd for nd in range(Nx-1,N,Nx)], 'right')
    else: 
        print('Warning: no boundary set of nodes defined for quad8 elements')    

    return returned_mesh
    

def grid_mesh_cylindric(Nr=11, Ntheta=11, r_min=0, r_max=1, theta_min=0, theta_max=1, ElementShape = 'quad4', init_rep_loc = 0, ndim = None, name = ""):  
    """
    Create a mesh as a regular grid in cylindrical coordinate

    Parameters
    ----------
    Nx, Ny : int
        Numbers of nodes in the x and y axes (default = 11).
    x_min, x_max, y_min, y_max : int,float
        The boundary of the square (default : 0, 1, 0, 1).
    ElementShape : {'tri3', 'quad4', 'quad8', 'quad9'}
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
        m = rectangle_mesh(Nr, Ntheta, r_min, r_max, theta_min, theta_max, ElementShape, ndim, name)
    else: 
        m = rectangle_mesh(Nr, Ntheta, r_min, r_max, theta_max, theta_min, ElementShape, ndim, name)

    r = m.nodes[:,0]
    theta = m.nodes[:,1]
    crd = np.c_[r*np.cos(theta) , r*np.sin(theta)] 
    returned_mesh = Mesh(crd, m.elements, ElementShape, ndim, name)
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
    

def line_mesh_1D(N=11, x_min=0, x_max=1, ElementShape = 'lin2', name = ""):            
    """
    Create the Mesh of a straight line with corrdinates in 1D. 

    Parameters
    ----------
    N : int
        Numbers of nodes (default = 11).
    x_min, x_max : int,float
        The boundary of the line (default : 0, 1).
    ElementShape : {'lin2', 'lin3', 'lin4'}
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
    if ElementShape == 'lin2': #1D element with 2 nodes
        crd = np.c_[np.linspace(x_min,x_max,N)] #Nodes coordinates 
        elm = np.c_[range(N-1), np.arange(1,N)] #Elements
    elif ElementShape == 'lin3': #1D element with 3 nodes
        N = N//2*2+1 #In case N is not initially odd
        crd = np.c_[np.linspace(x_min,x_max,N)] #Nodes coordinates 
        elm = np.c_[np.arange(0,N-2,2), np.arange(1,N-1,2), np.arange(2,N,2)] #Elements
    elif ElementShape == 'lin4':
        N = N//3*3+1 
        crd = np.c_[np.linspace(x_min,x_max,N)] #Nodes coordinates
        elm = np.c_[np.arange(0,N-3,3), np.arange(1,N-2,3), np.arange(2,N-1,3), np.arange(3,N,3)] #Elements 

    returned_mesh = Mesh(crd, elm, ElementShape, name=name)
    returned_mesh.add_node_set([0], "left")
    returned_mesh.add_node_set([N-1], "right")
    
    return returned_mesh


def line_mesh(N=11, x_min=0, x_max=1, ElementShape = 'lin2', ndim = None, name = ""):    
    """
    Create the Mesh of a straight line

    Parameters
    ----------
    N : int
        Numbers of nodes (default = 11).
    x_min, x_max : int,float,list
        The boundary of the line as scalar (1D) or list (default : 0, 1).
    ElementShape : {'lin2', 'lin3', 'lin4'}
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
        m = line_mesh_1D(N,x_min,x_max,ElementShape,name)    
        # if ModelingSpace.GetDimension() in ['2Dplane', '2Dstress'] : dim = 2
        # else: dim = 3
        crd = np.c_[m.nodes, np.zeros((N,ndim-1))]
        elm = m.elements
        returned_mesh = Mesh(crd,elm,ElementShape, name = name)
        returned_mesh.add_node_set(m.node_sets["left"], "left")
        returned_mesh.add_node_set(m.node_sets["right"], "right")
    else: 
        m = line_mesh_1D(N,0.,1.,ElementShape,name)    
        crd = m.nodes
        crd = (np.array(x_max)-np.array(x_min))*crd+np.array(x_min)
        elm = m.elements
        returned_mesh = Mesh(crd,elm,ElementShape, name = name)
        returned_mesh.add_node_set(m.node_sets["left"], "left")
        returned_mesh.add_node_set(m.node_sets["right"], "right")
        
    return returned_mesh
    

def line_mesh_cylindric(Ntheta=11, r=1, theta_min=0, theta_max=3.14, ElementShape = 'lin2', init_rep_loc = 0, ndim = None, name = ""):
    """
    Create the mesh of a curved line based on cylindrical coordinates  

    Parameters
    ----------
    Ntheta : int
        Numbers of nodes along the angular coordinate (default = 11).
    theta_min, theta_max : int,float
        The boundary of the line defined by the angular coordinate (default : 0, 3.14).
    ElementShape : {'lin2', 'lin3', 'lin4'}
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
    m = line_mesh_1D(Ntheta,theta_min,theta_max,ElementShape,name)       
    theta = m.nodes[:,0]
    elm = m.elements  

    crd = np.c_[r*np.cos(theta), r*np.sin(theta)]
        
    returned_mesh = Mesh(crd,elm,ElementShape, ndim, name)
    returned_mesh.add_node_set(m.node_sets["left"], "left")
    returned_mesh.add_node_set(m.node_sets["right"], "right")
    
    if init_rep_loc: 
        local_frame = np.array( [ [[np.sin(t),-np.cos(t)],[np.cos(t),np.sin(t)]] for t in theta ] ) 
    
    return returned_mesh


def box_mesh(Nx=11, Ny=11, Nz=11, x_min=0, x_max=1, y_min=0, y_max=1, z_min=0, z_max=1, ElementShape = 'hex8', name = ""):      
    """
    Create the mesh of a box  

    Parameters
    ----------
    Nx, Ny, Nz : int
        Numbers of nodes in the x, y and z axes (default = 11).
    x_min, x_max, y_min, y_max, z_min, z_max : int,float
        The boundary of the box (default : 0, 1, 0, 1, 0, 1).
    ElementShape : {'hex8', 'hex20'}
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
            
    Y,Z,X = np.meshgrid(np.linspace(y_min,y_max,Ny),np.linspace(z_min,z_max,Nz),np.linspace(x_min,x_max,Nx))    
    crd = np.c_[np.reshape(X,(-1,1)),np.reshape(Y,(-1,1)),np.reshape(Z,(-1,1))]
    
    if ElementShape == 'hex20':
        dx = (x_max-x_min)/(Nx-1.) ; dy = (y_max-y_min)/(Ny-1.) ; dz = (z_max-z_min)/(Nz-1.)
        Y,Z,X = np.meshgrid(np.linspace(y_min,y_max,Ny),np.linspace(z_min,z_max,Nz),np.linspace(x_min+dx/2.,x_max+dx/2.,Nx-1, endpoint=False))    
        crd2 = np.c_[np.reshape(X,(-1,1)),np.reshape(Y,(-1,1)),np.reshape(Z,(-1,1))]           
        Y,Z,X = np.meshgrid(np.linspace(y_min,y_max,Ny),np.linspace(z_min+dz/2.,z_max+dz/2.,Nz-1, endpoint=False),np.linspace(x_min,x_max,Nx))    
        crd3 = np.c_[np.reshape(X,(-1,1)),np.reshape(Y,(-1,1)),np.reshape(Z,(-1,1))]        
        Y,Z,X = np.meshgrid(np.linspace(y_min+dy/2.,y_max+dy/2.,Ny-1, endpoint=False),np.linspace(z_min,z_max,Nz),np.linspace(x_min,x_max,Nx))    
        crd4 = np.c_[np.reshape(X,(-1,1)),np.reshape(Y,(-1,1)),np.reshape(Z,(-1,1))]
     
        crd = np.vstack((crd,crd2,crd3,crd4))
        
        elm = [[Nx*j+i+(k*Nx*Ny), Nx*j+i+1+(k*Nx*Ny), Nx*(j+1)+i+1+(k*Nx*Ny), Nx*(j+1)+i+(k*Nx*Ny), \
                Nx*j+i+(k*Nx*Ny)+Nx*Ny, Nx*j+i+1+(k*Nx*Ny)+Nx*Ny, Nx*(j+1)+i+1+(k*Nx*Ny)+Nx*Ny, Nx*(j+1)+i+(k*Nx*Ny)+Nx*Ny, \
                Nx*Ny*Nz+(Nx-1)*j+i+k*(Nx-1)*Ny,Nx*j+i+1+Nx*Ny*Nz+(Nx-1)*Ny*Nz+(Nz-1)*Nx*Ny+(k*Nx*(Ny-1)),Nx*Ny*Nz+(Nx-1)*(j+1)+i+k*(Nx-1)*Ny,Nx*j+i+Nx*Ny*Nz+(Nx-1)*Ny*Nz+(Nz-1)*Nx*Ny+(k*Nx*(Ny-1)), \
                Nx*Ny*Nz+(Nx-1)*Ny*Nz+Nx*j+i+k*Ny*Nx,Nx*Ny*Nz+(Nx-1)*Ny*Nz+Nx*j+i+1+k*Ny*Nx,Nx*Ny*Nz+(Nx-1)*Ny*Nz+Nx+Nx*j+i+1+k*Ny*Nx,Nx*Ny*Nz+(Nx-1)*Ny*Nz+Nx+Nx*j+i+k*Ny*Nx, \
                Nx*Ny*Nz+(Nx-1)*Ny+(Nx-1)*j+i+k*(Nx-1)*Ny,Nx*j+i+1+Nx*Ny*Nz+(Nx-1)*Ny*Nz+(Nz-1)*Nx*Ny+Nx*(Ny-1)+(k*Nx*(Ny-1)),Nx*Ny*Nz+(Nx-1)*Ny+(Nx-1)*(j+1)+i+k*(Nx-1)*Ny,Nx*j+i+Nx*Ny*Nz+(Nx-1)*Ny*Nz+(Nz-1)*Nx*Ny+Nx*(Ny-1)+(k*Nx*(Ny-1))] for k in range(Nz-1) for j in range(Ny-1) for i in range(Nx-1)] 
                
        bottom = [nd for nd in range(Ny*Nx)]+list(range(Nx*Ny*Nz,(Nx-1)*Ny+Nx*Ny*Nz))+list(range(Nx*Ny*Nz+(Nx-1)*Ny*Nz+(Nz-1)*Nx*Ny, Nx*Ny*Nz+(Nx-1)*Ny*Nz+(Nz-1)*Nx*Ny+(Ny-1)*Nx))
        top = [nd for nd in range((Nz-1)*Ny*Nx,Nz*Nx*Ny)]+list(range(Nx*Ny*Nz+(Nx-1)*Ny*(Nz-1),Nx*Ny*Nz+(Nx-1)*Ny*Nz))+list(range(Nx*Ny*Nz+(Nx-1)*Ny*Nz+(Nz-1)*Nx*Ny+(Ny-1)*Nx*(Nz-1), Nx*Ny*Nz+(Nx-1)*Ny*Nz+(Nz-1)*Nx*Ny+(Ny-1)*Nx*Nz))
        left = list(itertools.chain.from_iterable([range(i*Nx*Ny,i*Nx*Ny+Nx*Ny,Nx) for i in range(Nz)]))+list(range(Nx*Ny*Nz+(Nx-1)*Ny*Nz, Nx*Ny*Nz+(Nx-1)*Ny*Nz+(Nz-1)*Nx*Ny,Nx))+list(range(Nx*Ny*Nz+(Nx-1)*Ny*Nz+(Nz-1)*Nx*Ny, Nx*Ny*Nz+(Nx-1)*Ny*Nz+(Nz-1)*Nx*Ny+Nx*(Ny-1)*Nz,Nx))
        right = list(itertools.chain.from_iterable([range(i*Nx*Ny+Nx-1,i*Nx*Ny+Nx*Ny,Nx) for i in range(Nz)]))+list(range(Nx*Ny*Nz+(Nx-1)*Ny*Nz+Nx-1,Nx*Ny*Nz+(Nx-1)*Ny*Nz+(Nz-1)*Nx*Ny,Nx))+list(range(Nx*Ny*Nz+(Nx-1)*Ny*Nz+(Nz-1)*Nx*Ny+Nx-1, Nx*Ny*Nz+(Nx-1)*Ny*Nz+(Nz-1)*Nx*Ny+Nx*(Ny-1)*Nz,Nx))
        front = list(itertools.chain.from_iterable([range(i*Nx*Ny,i*Nx*Ny+Nx) for i in range(Nz)]))+list(itertools.chain.from_iterable([range(i*(Nx-1)*Ny+Nx*Ny*Nz, i*(Nx-1)*Ny+Nx*Ny*Nz+Nx-1) for i in range(Nz)]))+list(itertools.chain.from_iterable([range(i*Nx*Ny+Nx*Ny*Nz+(Nx-1)*Ny*Nz, i*Nx*Ny+Nx*Ny*Nz+(Nx-1)*Ny*Nz+Nx) for i in range(Nz-1)]))
        back = list(itertools.chain.from_iterable([range(i*Nx*Ny+Nx*Ny-Nx,i*Nx*Ny+Nx*Ny) for i in range(Nz)]))+list(itertools.chain.from_iterable([range(i*(Nx-1)*Ny+Nz*Nx*Ny+(Nx-1)*(Ny-1), i*(Nx-1)*Ny+Nz*Nx*Ny+(Nx-1)*(Ny-1)+Nx-1) for i in range(Nz)]))+list(itertools.chain.from_iterable([range(i*Nx*Ny+Nz*Nx*Ny+(Nx-1)*Ny*Nz+(Ny-1)*Nx,i*Nx*Ny+Nz*Nx*Ny+(Nx-1)*Ny*Nz+(Ny-1)*Nx+Nx) for i in range(Nz-1)]))
        
    elif ElementShape == 'hex8':
        elm = [[Nx*j+i+(k*Nx*Ny),Nx*j+i+1+(k*Nx*Ny),Nx*(j+1)+i+1+(k*Nx*Ny),Nx*(j+1)+i+(k*Nx*Ny),Nx*j+i+(k*Nx*Ny)+Nx*Ny,Nx*j+i+1+(k*Nx*Ny)+Nx*Ny,Nx*(j+1)+i+1+(k*Nx*Ny)+Nx*Ny,Nx*(j+1)+i+(k*Nx*Ny)+Nx*Ny] for k in range(Nz-1) for j in range(Ny-1) for i in range(Nx-1)]     

        front = list(itertools.chain.from_iterable([range(i*Nx*Ny,i*Nx*Ny+Nx) for i in range(Nz)]))      # [item for sublist in bas for item in sublist] #flatten a list
        back = list(itertools.chain.from_iterable([range(i*Nx*Ny+Nx*Ny-Nx,i*Nx*Ny+Nx*Ny) for i in range(Nz)]))
        left = list(itertools.chain.from_iterable([range(i*Nx*Ny,i*Nx*Ny+Nx*Ny,Nx) for i in range(Nz)]))
        right = list(itertools.chain.from_iterable([range(i*Nx*Ny+Nx-1,i*Nx*Ny+Nx*Ny,Nx) for i in range(Nz)]))
        bottom = [nd for nd in range(Ny*Nx)]
        top = [nd for nd in range((Nz-1)*Ny*Nx,Nz*Nx*Ny)]

    else:
        raise NameError('Element not implemented. Only support hex8 and hex20 elements')
        
    N = np.shape(crd)[0]
    elm = np.array(elm)
    
    returned_mesh = Mesh(crd, elm, ElementShape, name = name)  
    for i, ndSet in enumerate([right, left, top, bottom, front, back]):
        ndSetId = ('right', 'left', 'top', 'bottom', 'front', 'back')[i]
        returned_mesh.add_node_set(ndSet, ndSetId)
                 
    return returned_mesh


def structured_mesh_2D(data, Edge1, Edge2, Edge3, Edge4, ElementShape = 'quad4', ndim = None, name =""):
#     #Edge1 and Edge3 should have the same lenght 
#     #Edge2 and Edge4 should have the same lenght
#     #last node of Edge1 should be the first of Edge2 and so on...
#     if no name is defined, the name is the same as crd
    
    if hasattr(data,'elm_type'): #data is a mesh        
        if data.elements is None: elm = []
        else: 
            elm = list(data.elements)
            ElementShape = data.elm_type
        crd = data.nodes
        if name == "": name = data.name
    else: 
        elm = []
        crd = data
        
    x1 = crd[Edge1,0] ; x2 = crd[Edge2,0] ; x3 = crd[Edge3,0][::-1] ; x4 = crd[Edge4,0][::-1]
    y1 = crd[Edge1,1] ; y2 = crd[Edge2,1] ; y3 = crd[Edge3,1][::-1] ; y4 = crd[Edge4,1][::-1] 
    new_crd = list(crd.copy())
    grid = np.empty((len(x1), len(x2)))
    grid[0,:] = Edge4[::-1] ; grid[-1,:] = Edge2
    grid[:,0] =  Edge1 ; grid[:,-1] = Edge3[::-1]
    
    N = len(new_crd)
    for i in range(1,len(x1)-1):                    
        px= ( (x1[i]*y3[i]-y1[i]*x3[i])*(x2-x4)-(x1[i]-x3[i])*(x2*y4-y2*x4) ) / ( (x1[i]-x3[i])*(y2-y4)-(y1[i]-y3[i])*(x2-x4) )         
        py= ( (x1[i]*y3[i]-y1[i]*x3[i])*(y2-y4)-(y1[i]-y3[i])*(x2*y4-y2*x4) ) / ( (x1[i]-x3[i])*(y2-y4)-(y1[i]-y3[i])*(x2-x4) )
        new_crd += list(np.c_[px[1:-1],py[1:-1]])
        grid[i,1:-1] = np.arange(N,len(new_crd),1)
        N = len(new_crd)        

    Nx = grid.shape[0] ; Ny = grid.shape[1]
    
    if ElementShape == 'quad4':
        elm += [[grid[i,j], grid[i+1,j],grid[i+1,j+1], grid[i,j+1]] for j in range(Ny-1) for i in range(Nx-1)]            
    elif ElementShape == 'quad9':                
        elm += [[grid[i,j],grid[i+2,j],grid[i+2,j+2],grid[i,j+2],grid[i+1,j],grid[i+2,j+1],grid[i+1,j+2],grid[i,j+1],grid[i+1,j+1]] for j in range(0,Ny-2,2) for i in range(0,Nx-2,2)]
    elif ElementShape == 'tri3':    
        for j in range(Ny-1):
            elm += [[grid[i,j],grid[i+1,j],grid[i,j+1]] for i in range(Nx-1)]
            elm += [[grid[i+1,j],grid[i+1,j+1],grid[i,j+1]] for i in range(Nx-1)]
    elif ElementShape == 'tri6':
        for j in range(0,Ny-2,2):
            elm += [[grid[i,j],grid[i+2,j],grid[i,j+2], grid[i+1,j],grid[i+1,j+1],grid[i,j+1]] for i in range(0,Nx-2,2)]
            elm += [[grid[i+2,j],grid[i+2,j+2],grid[i,j+2], grid[i+2,j+1],grid[i+1,j+2],grid[i+1,j+1]] for i in range(0,Nx-2,2)]
    elif ElementShape == 'quad8':
        raise NameError("'quad8' elements are not implemented")
    
    elm = np.array(elm, dtype=int)
    return Mesh(np.array(new_crd), elm, ElementShape, ndim, name)


def generate_nodes(mesh, N, data, typeGen = 'straight'):
    #if typeGen == 'straight' -> data = (node1, node2)
    #if typeGen == 'circular' -> data = (node1, node2, (center_x, center_y))
    crd = mesh.nodes    
    if typeGen == 'straight':
        node1 = data[0] ; node2 = data[1]
        listNodes = mesh.add_nodes(line_mesh(N, crd[node1], crd[node2]).nodes[1:-1])
        return np.array([node1]+list(listNodes)+[node2])
    if typeGen == 'circular':
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


def hole_plate_mesh(Nx=11, Ny=11, Lx=100, Ly=100, R=20, ElementShape = 'quad4', sym= True, ndim = None, name =""):
    """
    Create a mesh of a 2D plate with a hole  

    Parameters
    ----------
    Nx, Ny : int
        Numbers of nodes in the x and y axes (default = 11).
    Lx, Ly : int,float
        The lenght of the plate in the x and y axes (default : 100).
    R : int,float
        The radius of the hole (default : 20).

    ElementShape : {'quad4', 'quad9', 'tri3', 'tri6'}
        The type of the element generated (default='quad4')
    Sym : bool 
        Sym = True, if only the returned mesh assume symetric condition and 
        only the quarter of the plate is returned (default=True)
    Returns
    -------
    Mesh
        The generated geometry in Mesh format. See the Mesh class for more details.        
    
    See Also
    --------
    line_mesh : 1D mesh of a line    
    rectangle_mesh : Surface mesh of a rectangle
    """   

    
    if sym == True:
        L = Lx/2
        h = Ly/2
        m = Mesh(np.array([[R,0],[L,0],[L,h],[0,h],[0,R],[R*np.cos(np.pi/4),R*np.sin(np.pi/4)]]))
        Edge1 = generate_nodes(m,Nx,(0,1))
        Edge2 = generate_nodes(m,Ny,(1,2))
        Edge3 = generate_nodes(m,Nx,(2,5))
        Edge4 = generate_nodes(m,Ny,(5,0,(0,0)), typeGen = 'circular')
        
        Edge5 = generate_nodes(m,Nx,(4,3))
        Edge6 = generate_nodes(m,Ny,(3,2))
        Edge7 = generate_nodes(m,Ny,(5,4,(0,0)), typeGen = 'circular')
        
        m = structured_mesh_2D(m, Edge1, Edge2, Edge3, Edge4, ElementShape = 'quad4')
        m = structured_mesh_2D(m, Edge5, Edge6, Edge3, Edge7, ElementShape = 'quad4', ndim = ndim, name=name)
    else: 
        L = Lx/2
        h = Ly/2
        m = Mesh(np.array([[R,0],[L,0],[L,h],[0,h],[0,R],[R*np.cos(np.pi/4),R*np.sin(np.pi/4)]]))
        Edge1 = generate_nodes(m,Nx,(0,1))
        Edge2 = generate_nodes(m,Ny,(1,2))
        Edge3 = generate_nodes(m,Nx,(2,5))
        Edge4 = generate_nodes(m,Ny,(5,0,(0,0)), typeGen = 'circular')
        
        Edge5 = generate_nodes(m,Nx,(4,3))
        Edge6 = generate_nodes(m,Ny,(3,2))
        Edge7 = generate_nodes(m,Ny,(5,4,(0,0)), typeGen = 'circular')
        
        m = structured_mesh_2D(m, Edge1, Edge2, Edge3, Edge4, ElementShape = 'quad4')
        m = structured_mesh_2D(m, Edge5, Edge6, Edge3, Edge7, ElementShape = 'quad4', ndim = ndim)
        
        nnd = m.n_nodes
        crd = m.nodes.copy()
        crd[:,0] = -m.nodes[:,0]
        m2 = Mesh(crd, m.elements, m.elm_type)
        m = Mesh.stack(m,m2)
        
        crd = m.nodes.copy()
        crd[:,1] = -m.nodes[:,1]
        m2 = Mesh(crd, m.elements, m.elm_type)
        m = Mesh.stack(m,m2, name=name)
        
        node_to_merge = np.vstack((np.c_[Edge5, Edge5+nnd], 
                                   np.c_[Edge5+2*nnd, Edge5+3*nnd],
                                   np.c_[Edge1, Edge1+2*nnd],
                                   np.c_[Edge1+nnd, Edge1+3*nnd]))
                                   
        
        m.merge_nodes(node_to_merge)
        
    if ElementShape == 'quad4': return m
    elif ElementShape == 'tri3': return quad2tri(m)
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

