from fedoo.libMesh.Mesh import Mesh
from fedoo.libUtil.Dimension import ProblemDimension
import itertools

# import scipy as sp
import numpy as np

# utility fuctions
# Only Functions are declared here !!

def RectangleMesh(Nx=11, Ny=11, x_min=0, x_max=1, y_min=0, y_max=1, ElementShape = 'quad4', ID =""):
    """
    Define the mesh of a rectangle.

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
    LineMesh : 1D mesh of a line    
    RectangleMesh : Surface mesh of a rectangle
    BoxMesh : Volume mesh of a box 
    GridMeshCylindric : Surface mesh of a grid in cylindrical coodrinate 
    LineMeshCylindric : Line mesh in cylindrical coordinate
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

    ReturnedMesh = Mesh(crd, elm, ElementShape, None, ID)
    if ElementShape != 'quad8':
        N = ReturnedMesh.GetNumberOfNodes()
        ReturnedMesh.AddSetOfNodes([nd for nd in range(Nx)], 'bottom')
        ReturnedMesh.AddSetOfNodes([nd for nd in range(N-Nx,N)], 'top')
        ReturnedMesh.AddSetOfNodes([nd for nd in range(0,N,Nx)], 'left')
        ReturnedMesh.AddSetOfNodes([nd for nd in range(Nx-1,N,Nx)], 'right')
    else: 
        print('Warning: no boundary set of nodes defined for quad8 elements')    

    return ReturnedMesh
    

def GridMeshCylindric(Nr=11, Ntheta=11, r_min=0, r_max=1, theta_min=0, theta_max=1, ElementShape = 'quad4', init_rep_loc = 0, ID = ""):  
    """
    Define the mesh as a grid in cylindrical coordinate

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
    LineMesh : 1D mesh of a line    
    RectangleMesh : Surface mesh of a rectangle
    BoxMesh : Volume mesh of a box     
    LineMeshCylindric : Line mesh in cylindrical coordinate    
    """
    
    if theta_min<theta_max: 
        m = RectangleMesh(Nr, Ntheta, r_min, r_max, theta_min, theta_max, ElementShape, ID)
    else: 
        m = RectangleMesh(Nr, Ntheta, r_min, r_max, theta_max, theta_min, ElementShape, ID)

    r = m.GetNodeCoordinates()[:,0]
    theta = m.GetNodeCoordinates()[:,1]
    crd = np.c_[r*np.cos(theta) , r*np.sin(theta)] 
    ReturnedMesh = Mesh(crd, m.GetElementTable(), ElementShape, m.GetLocalFrame(), ID)

    if theta_min<theta_max: 
        ReturnedMesh.AddSetOfNodes(m.GetSetOfNodes("left"), "bottom")
        ReturnedMesh.AddSetOfNodes(m.GetSetOfNodes("right"), "top")
        ReturnedMesh.AddSetOfNodes(m.GetSetOfNodes("bottom"), "left")
        ReturnedMesh.AddSetOfNodes(m.GetSetOfNodes("top"), "right")
    else: 
        ReturnedMesh.AddSetOfNodes(m.GetSetOfNodes("left"), "bottom")
        ReturnedMesh.AddSetOfNodes(m.GetSetOfNodes("right"), "top")
        ReturnedMesh.AddSetOfNodes(m.GetSetOfNodes("top"), "left")
        ReturnedMesh.AddSetOfNodes(m.GetSetOfNodes("bottom"), "right")
    
    return ReturnedMesh
    
def LineMesh1D(N=11, x_min=0, x_max=1, ElementShape = 'lin2', ID = ""):            
    """
    Define the mesh of a line with corrdinates in 1D. 

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
    LineMesh : Mesh of a line whith choosen dimension
    RectangleMesh : Surface mesh of a rectangle
    BoxMesh : Volume mesh of a box 
    GridMeshCylindric : Surface mesh of a grid in cylindrical coodrinate 
    LineMeshCylindric : Line mesh in cylindrical coordinate
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

    ReturnedMesh = Mesh(crd, elm, ElementShape, None, ID)
    ReturnedMesh.AddSetOfNodes([0], "left")
    ReturnedMesh.AddSetOfNodes([N-1], "right")
    
    return ReturnedMesh

def LineMesh(N=11, x_min=0, x_max=1, ElementShape = 'lin2', ID = ""):    
    """
    Define the mesh of a line  

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
    RectangleMesh : Surface mesh of a rectangle
    BoxMesh : Volume mesh of a box 
    GridMeshCylindric : Surface mesh of a grid in cylindrical coodrinate 
    LineMeshCylindric : Line mesh in cylindrical coordinate
    """
    if np.isscalar(x_min):
        m = LineMesh1D(N,x_min,x_max,ElementShape,ID)    
        if ProblemDimension.Get() in ['2Dplane', '2Dstress'] : dim = 2
        else: dim = 3
        crd = np.c_[m.GetNodeCoordinates(), np.zeros((N,dim-1))]
        elm = m.GetElementTable()
        ReturnedMesh = Mesh(crd,elm,ElementShape, None, ID)
        ReturnedMesh.AddSetOfNodes(m.GetSetOfNodes("left"), "left")
        ReturnedMesh.AddSetOfNodes(m.GetSetOfNodes("right"), "right")
    else: 
        m = LineMesh1D(N,0.,1.,ElementShape,ID)    
        crd = m.GetNodeCoordinates()
        crd = (np.array(x_max)-np.array(x_min))*crd+np.array(x_min)
        elm = m.GetElementTable()
        ReturnedMesh = Mesh(crd,elm,ElementShape, None, ID)
        ReturnedMesh.AddSetOfNodes(m.GetSetOfNodes("left"), "left")
        ReturnedMesh.AddSetOfNodes(m.GetSetOfNodes("right"), "right")
        
    return ReturnedMesh
    
def LineMeshCylindric(Ntheta=11, r=1, theta_min=0, theta_max=3.14, ElementShape = 'lin2', init_rep_loc = 0, ID = ""):
    """
    Define the mesh of a line in cylindrical coordinates  

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
    LineMesh : Mesh of a line whith choosen dimension
    RectangleMesh : Surface mesh of a rectangle
    BoxMesh : Volume mesh of a box 
    GridMeshCylindric : Surface mesh of a grid in cylindrical coodrinate 
    LineMeshCylindric : Line mesh in cylindrical coordinate
    """
    # init_rep_loc = 1 si on veut initialiser le repère local (0 par défaut)
    m = LineMesh1D(Ntheta,theta_min,theta_max,ElementShape,ID)       
    theta = m.GetNodeCoordinates()[:,0]
    elm = m.GetElementTable()

    LocalFrame = np.array( [ [[np.sin(t),-np.cos(t)],[np.cos(t),np.sin(t)]] for t in theta ] )

    crd = np.c_[r*np.cos(theta), r*np.sin(theta)]
        
    ReturnedMesh = Mesh(crd,elm,ElementShape, LocalFrame, ID)
    ReturnedMesh.AddSetOfNodes(m.GetSetOfNodes("left"), "left")
    ReturnedMesh.AddSetOfNodes(m.GetSetOfNodes("right"), "right")
    
    return ReturnedMesh

def BoxMesh(Nx=11, Ny=11, Nz=11, x_min=0, x_max=1, y_min=0, y_max=1, z_min=0, z_max=1, ElementShape = 'hex8', ID = ""):      
    """
    Define the mesh of a box  

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
    LineMesh : 1D mesh of a line    
    RectangleMesh : Surface mesh of a rectangle
    GridMeshCylindric : Surface mesh of a grid in cylindrical coodrinate 
    LineMeshCylindric : Line mesh in cylindrical coord
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
                
        bottom = [nd for nd in range(Ny*Nx)]+range(Nx*Ny*Nz,(Nx-1)*Ny+Nx*Ny*Nz)+range(Nx*Ny*Nz+(Nx-1)*Ny*Nz+(Nz-1)*Nx*Ny, Nx*Ny*Nz+(Nx-1)*Ny*Nz+(Nz-1)*Nx*Ny+(Ny-1)*Nx)
        top = [nd for nd in range((Nz-1)*Ny*Nx,Nz*Nx*Ny)]+range(Nx*Ny*Nz+(Nx-1)*Ny*(Nz-1),Nx*Ny*Nz+(Nx-1)*Ny*Nz)+range(Nx*Ny*Nz+(Nx-1)*Ny*Nz+(Nz-1)*Nx*Ny+(Ny-1)*Nx*(Nz-1), Nx*Ny*Nz+(Nx-1)*Ny*Nz+(Nz-1)*Nx*Ny+(Ny-1)*Nx*Nz)
        left = list(itertools.chain.from_iterable([range(i*Nx*Ny,i*Nx*Ny+Nx*Ny,Nx) for i in range(Nz)]))+range(Nx*Ny*Nz+(Nx-1)*Ny*Nz, Nx*Ny*Nz+(Nx-1)*Ny*Nz+(Nz-1)*Nx*Ny,Nx)+range(Nx*Ny*Nz+(Nx-1)*Ny*Nz+(Nz-1)*Nx*Ny, Nx*Ny*Nz+(Nx-1)*Ny*Nz+(Nz-1)*Nx*Ny+Nx*(Ny-1)*Nz,Nx)
        right = list(itertools.chain.from_iterable([range(i*Nx*Ny+Nx-1,i*Nx*Ny+Nx*Ny,Nx) for i in range(Nz)]))+range(Nx*Ny*Nz+(Nx-1)*Ny*Nz+Nx-1,Nx*Ny*Nz+(Nx-1)*Ny*Nz+(Nz-1)*Nx*Ny,Nx)+range(Nx*Ny*Nz+(Nx-1)*Ny*Nz+(Nz-1)*Nx*Ny+Nx-1, Nx*Ny*Nz+(Nx-1)*Ny*Nz+(Nz-1)*Nx*Ny+Nx*(Ny-1)*Nz,Nx)
        front = list(itertools.chain.from_iterable([range(i*Nx*Ny,i*Nx*Ny+Nx) for i in range(Nz)]))+list(itertools.chain.from_iterable([range(i*(Nx-1)*Ny+Nx*Ny*Nz, i*(Nx-1)*Ny+Nx*Ny*Nz+Nx-1) for i in range(Nz)]))+list(itertools.chain.from_iterable([range(i*Nx*Ny+Nx*Ny*Nz+(Nx-1)*Ny*Nz, i*Nx*Ny+Nx*Ny*Nz+(Nx-1)*Ny*Nz+Nx) for i in range(Nz-1)]))
        back = list(itertools.chain.from_iterable([range(i*Nx*Ny+Nx*Ny-Nx,i*Nx*Ny+Nx*Ny) for i in range(Nz)]))+list(itertools.chain.from_iterable([range(i*(Nx-1)*Ny+Nz*Nx*Ny+(Nx-1)*(Ny-1), i*(Nx-1)*Ny+Nz*Nx*Ny+(Nx-1)*(Ny-1)+Nx-1) for i in range(Nz)]))+list(itertools.chain.from_iterable([range(i*Nx*Ny+Nz*Nx*Ny+(Nx-1)*Ny*Nz+(Ny-1)*Nx,i*Nx*Ny+Nz*Nx*Ny+(Nx-1)*Ny*Nz+(Ny-1)*Nx+Nx) for i in range(Nz-1)]))
        
    if ElementShape == 'hex8':
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
    
    ReturnedMesh = Mesh(crd, elm, ElementShape, None, ID)  
    for i, ndSet in enumerate([right, left, top, bottom, front, back]):
        ndSetId = ('right', 'left', 'top', 'bottom', 'front', 'back')[i]
        ReturnedMesh.AddSetOfNodes(ndSet, ndSetId)
                 
    return ReturnedMesh




def GridStructuredMesh2D(data, Edge1, Edge2, Edge3, Edge4, ElementShape = 'quad4', ID =""):
#     #Edge1 and Edge3 should have the same lenght 
#     #Edge2 and Edge4 should have the same lenght
#     #last node of Edge1 should be the first of Edge2 and so on...
#     if no ID is defined, the ID is the same as crd
    
    if hasattr(data,'GetElementTable'): #data is a mesh        
        if data.GetElementTable() is None: elm = []
        else: 
            elm = list(data.GetElementTable())
            ElementShape = data.GetElementShape()
        crd = data.GetNodeCoordinates()
        if ID == "": ID = data.GetID()
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
    return Mesh(np.array(new_crd), elm, ElementShape, None, ID)



# def EmptyMesh(ElementShape = 'quad4', ID=""):
#     return Mesh(np.array([]), np.array([], dtype=int), ElementShape, None, ID)

def GenerateNodes(mesh, N, data, typeGen = 'straight'):
    #if typeGen == 'straight' -> data = (node1, node2)
    #if typeGen == 'circular' -> data = (node1, node2, (center_x, center_y))
    crd = mesh.GetNodeCoordinates()    
    if typeGen == 'straight':
        node1 = data[0] ; node2 = data[1]
        listNodes = mesh.AddNodes(LineMesh(N, crd[node1], crd[node2]).GetNodeCoordinates()[1:-1])
        return np.array([node1]+list(listNodes)+[node2])
    if typeGen == 'circular':
        nd1 = data[0] ; nd2 = data[1] ; c = data[2]
        c = np.array(c)
        R = np.linalg.norm(crd[nd1]-c)
        assert np.abs(R-np.linalg.norm(crd[nd2]-c))<R*1e-4, "Final nodes is not on the circle"
        (crd[nd1]-c)
        theta_min = np.arctan2(crd[nd1,1]-c[1],crd[nd1,0]-c[0])
        theta_max = np.arctan2(crd[nd2,1]-c[1],crd[nd2,0]-c[0])
        m = LineMeshCylindric(N, R, theta_min, theta_max) #circular mesh
        listNodes = mesh.AddNodes(m.GetNodeCoordinates()[1:-1]+c)
        return np.array([nd1]+list(listNodes)+[nd2])

                      
if __name__=="__main__":
    import math
    a = LineMeshCylindric(11, 1, 0, math.pi, 'lin2', init_rep_loc = 0)
    b = LineMeshCylindric(11, 1, 0, math.pi, 'lin4', init_rep_loc = 1)

    print(b.GetNodeCoordinates())

