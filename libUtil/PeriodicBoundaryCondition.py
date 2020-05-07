# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 15:27:43 2020

@author: Etienne
"""
# from fedoo.libProblem.ProblemBase   import ProblemBase 
from fedoo.libProblem.BoundaryCondition import BoundaryCondition
import numpy as np
from fedoo.libMesh.Mesh import MeshBase

def DefinePeriodicBoundaryCondition(mesh, NodeEps, VarEps, dim='3D', tol=1e-8, ProblemID = 'MainProblem'):
    """
    Parameters
    ----------
    crd : mesh ID or mesh object
        A periodic mesh to apply the periodic boundary conditions
    NodeEps : lise of int
        NodeEps is a list containing the node index of strain tensor component (virtual node)
        In 2D: [EpsXX, EpsYY, EpsXY]
        In 3D: [EpsXX, EpsYY, EpsZZ, EpsYZ, EpsXZ, EpsXY]
    VarEps : list of string
        VarEps is a list containing the variable id used for each component
    dim : '2D' or '3D', optional
        This parameter is used to define if the periodicity is in 2D or in 3D.
        A periodicity in 2D can be associated with a 3D mesh. 
        The default is '2D'.
    tol : float, optional
        Tolerance for the position of nodes. The default is 1e-8.
    ProblemID : ProblemID on which the boundary conditions are applied
        The default is 'MainProblem'.

    Returns
    -------
    None.

    """
    #TODO: add set to the mesh and don't compute the set if the set are already present
    if dim in ['2D','2d']: dim = 2
    if dim in ['3D','3d']: dim = 3

    if isinstance(mesh, str): mesh = MeshBase.GetAll()[mesh]
    crd = mesh.GetNodeCoordinates()
    xmax = np.max(crd[:,0]) ; xmin = np.min(crd[:,0])
    ymax = np.max(crd[:,1]) ; ymin = np.min(crd[:,1])
    if dim == 3:                        
        zmax = np.max(crd[:,2]) ; zmin = np.min(crd[:,2])
  
    if dim == 2:                       
        #Define node sets
        left   = np.where( (np.abs(crd[:,0] - xmin) < tol) * (np.abs(crd[:,1] - ymin) > tol) * (np.abs(crd[:,1] - ymax) > tol))[0]
        right  = np.where( (np.abs(crd[:,0] - xmax) < tol) * (np.abs(crd[:,1] - ymin) > tol) * (np.abs(crd[:,1] - ymax) > tol))[0]
        bottom = np.where( (np.abs(crd[:,1] - ymin) < tol) * (np.abs(crd[:,0] - xmin) > tol) * (np.abs(crd[:,0] - xmax) > tol))[0]
        top    = np.where( (np.abs(crd[:,1] - ymax) < tol) * (np.abs(crd[:,0] - xmin) > tol) * (np.abs(crd[:,0] - xmax) > tol))[0]
    
        corner_lb = np.where((np.abs(crd[:,0] - xmin) < tol) * (np.abs(crd[:,1] - ymin) < tol))[0] #corner left bottom
        corner_rb = np.where((np.abs(crd[:,0] - xmax) < tol) * (np.abs(crd[:,1] - ymin) < tol))[0] #corner right bottom
        corner_lt = np.where((np.abs(crd[:,0] - xmin) < tol) * (np.abs(crd[:,1] - ymax) < tol))[0] #corner left top
        corner_rt = np.where((np.abs(crd[:,0] - xmax) < tol) * (np.abs(crd[:,1] - ymax) < tol))[0] #corner right top

        lelf = left[np.argsort(crd[left,1])]
        right = right[np.argsort(crd[right,1])]
        top = top[np.argsort(crd[top,0])]
        bottom = bottom[np.argsort(crd[bottom,0])]
        
        BoundaryCondition('MPC', ['DispX','DispX',VarEps[0]], [np.full_like(right, 1), np.full_like(left, -1), np.full_like(right, -(xmax-xmin))], [right,left,np.full_like(right, NodeEps[0])], ProblemID = ProblemID)
        BoundaryCondition('MPC', ['DispY','DispY',VarEps[2]], [np.full_like(right,1), np.full_like(left,-1), np.full_like(right,-0.5*(xmax-xmin))], [right,left,np.full_like(right,NodeEps[2])], ProblemID = ProblemID)
        BoundaryCondition('MPC', ['DispX','DispX',VarEps[2]], [np.full_like(top,1), np.full_like(bottom,-1), np.full_like(top,-0.5*(ymax-ymin))], [top,bottom,np.full_like(top,NodeEps[2])], ProblemID = ProblemID)
        BoundaryCondition('MPC', ['DispY','DispY',VarEps[1]], [np.full_like(top,1), np.full_like(bottom,-1), np.full_like(top,-(ymax-ymin))], [top,bottom,np.full_like(top,NodeEps[1])], ProblemID = ProblemID)
        
        #elimination of DOF from edge left/top -> edge left/bottom
        BoundaryCondition('MPC', ['DispY','DispY',VarEps[1]], [np.full_like(corner_lt,1), np.full_like(corner_lb,-1), np.full_like(corner_lt,-(ymax-ymin))], [corner_lt, corner_lb, np.full_like(corner_lt,NodeEps[1])], ProblemID = ProblemID)
        BoundaryCondition('MPC', ['DispX','DispX',VarEps[2]], [np.full_like(corner_lt,1), np.full_like(corner_lb,-1), np.full_like(corner_lt,-0.5*(ymax-ymin))], [corner_lt, corner_lb, np.full_like(corner_lt,NodeEps[2])], ProblemID = ProblemID)
        #elimination of DOF from edge right/bottom -> edge left/bottom
        BoundaryCondition('MPC', ['DispX','DispX',VarEps[0]], [np.full_like(corner_rb,1), np.full_like(corner_lb,-1), np.full_like(corner_rb,-(xmax-xmin))], [corner_rb, corner_lb, np.full_like(corner_lt,NodeEps[0])], ProblemID = ProblemID)
        BoundaryCondition('MPC', ['DispY','DispY',VarEps[2]], [np.full_like(corner_rb,1), np.full_like(corner_lb,-1), np.full_like(corner_rb,-0.5*(xmax-xmin))], [corner_rb, corner_lb, np.full_like(corner_lt, NodeEps[2])], ProblemID = ProblemID)
        #elimination of DOF from edge right/top -> edge left/bottom
        BoundaryCondition('MPC', ['DispX','DispX',VarEps[0],VarEps[2]], [np.full_like(corner_rt,1), np.full_like(corner_lb,-1), np.full_like(corner_rt,-(xmax-xmin)), np.full_like(corner_rt,-0.5*(ymax-ymin))], [corner_rt, corner_lb, np.full_like(corner_rt,NodeEps[0]), np.full_like(corner_rt,NodeEps[2])], ProblemID = ProblemID)
        BoundaryCondition('MPC', ['DispY','DispY',VarEps[2],VarEps[1]], [np.full_like(corner_rt,1), np.full_like(corner_lb,-1), np.full_like(corner_rt,-0.5*(xmax-xmin)), np.full_like(corner_rt,-(ymax-ymin))], [corner_rt, corner_lb, np.full_like(corner_rt,NodeEps[2]), np.full_like(corner_rt,NodeEps[1])], ProblemID = ProblemID)                
                                       
                                       
        # Problem.BoundaryCondition('MPC', 'DispX', [right*0+1, left*0-1], [right,left], -Exx*(xmax-xmin)+right*0)
        # # Problem.BoundaryCondition('MPC', 'DispY', [right*0+1, left*0-1], [right,left], -Exy*(xmax-xmin)+right*0)
        # # Problem.BoundaryCondition('MPC', 'DispX', [top*0+1, bottom*0-1], [top,bottom], -Exy*(ymax-ymin)+top*0)
        # # Problem.BoundaryCondition('MPC', 'DispY', [top*0+1, bottom*0-1], [top,bottom], -Eyy*(ymax-ymin)+top*0)
        # Problem.BoundaryCondition('Dirichlet','DispX', 0, center)
        # Problem.BoundaryCondition('Dirichlet','DispY', 0, center)
        
        #StrainNode[0] - 'DispX' is a virtual dof for EXX
        #StrainNode[0] - 'DispY' is a virtual dof for EXY
        #StrainNode[1] - 'DispX' is a non used node -> need to be blocked
        #StrainNode[1] - 'DispY' is a virtual dof for EYY
        
    elif dim == 3:                        
        
        #Define node sets
        left   = np.where( np.abs(crd[:,0] - xmin) < tol )[0]
        right  = np.where( np.abs(crd[:,0] - xmax) < tol )[0]
        bottom = np.where( np.abs(crd[:,1] - ymin) < tol )[0]
        top    = np.where( np.abs(crd[:,1] - ymax) < tol )[0]
        behind = np.where( np.abs(crd[:,2] - zmin) < tol )[0]
        front  = np.where( np.abs(crd[:,2] - zmax) < tol )[0]
        
        #l = left, r = right, b = bottom, t = top, f = front, d = behind
        edge_lb = np.intersect1d(left , bottom, assume_unique=True)
        edge_lt = np.intersect1d(left , top   , assume_unique=True)
        edge_rb = np.intersect1d(right, bottom, assume_unique=True)
        edge_rt = np.intersect1d(right, top   , assume_unique=True)
        
        edge_bd = np.intersect1d(bottom, behind, assume_unique=True)
        edge_bf = np.intersect1d(bottom, front , assume_unique=True)
        edge_td = np.intersect1d(top   , behind, assume_unique=True)
        edge_tf = np.intersect1d(top   , front , assume_unique=True)
        
        edge_ld = np.intersect1d(left , behind, assume_unique=True)
        edge_lf = np.intersect1d(left , front , assume_unique=True)
        edge_rd = np.intersect1d(right, behind, assume_unique=True)
        edge_rf = np.intersect1d(right, front , assume_unique=True)
        
        #sort edges (required to assign the good pair of nodes)
        edge_lb = edge_lb[np.argsort(crd[edge_lb,2])]
        edge_lt = edge_lt[np.argsort(crd[edge_lt,2])]
        edge_rb = edge_rb[np.argsort(crd[edge_rb,2])]
        edge_rt = edge_rt[np.argsort(crd[edge_rt,2])]
        
        edge_bd = edge_bd[np.argsort(crd[edge_bd,0])][1:-1] #without corners
        edge_bf = edge_bf[np.argsort(crd[edge_bf,0])][1:-1] #without corners
        edge_td = edge_td[np.argsort(crd[edge_td,0])][1:-1] #without corners
        edge_tf = edge_tf[np.argsort(crd[edge_tf,0])][1:-1] #without corners
        
        edge_ld = edge_ld[np.argsort(crd[edge_ld,1])][1:-1] #without corners
        edge_lf = edge_lf[np.argsort(crd[edge_lf,1])][1:-1] #without corners
        edge_rd = edge_rd[np.argsort(crd[edge_rd,1])][1:-1] #without corners
        edge_rf = edge_rf[np.argsort(crd[edge_rf,1])][1:-1] #without corners
        
       
        #extract corner from left and right edges
        corner_lbd = edge_lb[[0]]
        corner_lbf = edge_lb[[-1]]
        corner_ltd = edge_lt[[0]]
        corner_ltf = edge_lt[[-1]]
        corner_rbd = edge_rb[[0]]
        corner_rbf = edge_rb[[-1]]
        corner_rtd = edge_rt[[0]]
        corner_rtf = edge_rt[[-1]]
        edge_lb = edge_lb[1:-1]
        edge_lt = edge_lt[1:-1]
        edge_rb = edge_rb[1:-1]
        edge_rt = edge_rt[1:-1]
        
        all_edges = np.hstack((edge_lb, edge_lt, edge_rb, edge_rt, edge_bd, edge_bf, 
                               edge_td, edge_tf, edge_ld, edge_lf, edge_rd, edge_rf, 
                               corner_lbd, corner_lbf, corner_ltd, corner_ltf, 
                               corner_rbd, corner_rbf, corner_rtd, corner_rtf))
        
        left   = np.setdiff1d(left  , all_edges, assume_unique=True)
        right  = np.setdiff1d(right , all_edges, assume_unique=True)
        bottom = np.setdiff1d(bottom, all_edges, assume_unique=True)
        top    = np.setdiff1d(top   , all_edges, assume_unique=True)
        behind = np.setdiff1d(behind, all_edges, assume_unique=True)
        front  = np.setdiff1d(front , all_edges, assume_unique=True)
               
        #sort adjacent faces to ensure node correspondance
        decimal_round = int(-np.log10(tol)-1)
        left   = left  [np.lexsort((crd[left  ,1], crd[left  ,2].round(decimal_round)))]
        right  = right [np.lexsort((crd[right ,1], crd[right ,2].round(decimal_round)))]
        bottom = bottom[np.lexsort((crd[bottom,0], crd[bottom,2].round(decimal_round)))]
        top    = top   [np.lexsort((crd[top   ,0], crd[top   ,2].round(decimal_round)))]
        behind = behind[np.lexsort((crd[behind,0], crd[behind,1].round(decimal_round)))]
        front  = front [np.lexsort((crd[front ,0], crd[front ,1].round(decimal_round)))]
                                        
        #now apply periodic boudary conditions        
        dx = xmax-xmin ; dy = ymax-ymin ; dz = zmax-zmin        
        #[EpsXX, EpsYY, EpsZZ, EpsYZ, EpsXZ, EpsXY]
        #Left/right faces
        BoundaryCondition('MPC', ['DispX','DispX',VarEps[0]], [np.full_like(right,1), np.full_like(left, -1), np.full_like(right, -dx)]  , [right,left,np.full_like(right, NodeEps[0])], ProblemID = ProblemID)
        BoundaryCondition('MPC', ['DispY','DispY',VarEps[5]], [np.full_like(right,1), np.full_like(left,-1), np.full_like(right,-0.5*dx)], [right,left,np.full_like(right,NodeEps[5])], ProblemID = ProblemID)
        BoundaryCondition('MPC', ['DispZ','DispZ',VarEps[4]], [np.full_like(right,1), np.full_like(left,-1), np.full_like(right,-0.5*dx)], [right,left,np.full_like(right,NodeEps[4])], ProblemID = ProblemID)
        #top/bottom faces
        BoundaryCondition('MPC', ['DispX','DispX',VarEps[5]], [np.full_like(top,1), np.full_like(bottom,-1), np.full_like(top,-0.5*dy)], [top,bottom,np.full_like(top,NodeEps[5])], ProblemID = ProblemID)
        BoundaryCondition('MPC', ['DispY','DispY',VarEps[1]], [np.full_like(top,1), np.full_like(bottom,-1), np.full_like(top, -dy)]   , [top,bottom,np.full_like(top,NodeEps[1])], ProblemID = ProblemID)
        BoundaryCondition('MPC', ['DispZ','DispZ',VarEps[3]], [np.full_like(top,1), np.full_like(bottom,-1), np.full_like(top,-0.5*dy)], [top,bottom,np.full_like(top,NodeEps[3])], ProblemID = ProblemID)
        #front/behind faces
        BoundaryCondition('MPC', ['DispX','DispX',VarEps[4]], [np.full_like(front,1), np.full_like(behind,-1), np.full_like(front,-0.5*dz)], [front,behind,np.full_like(front,NodeEps[4])], ProblemID = ProblemID)
        BoundaryCondition('MPC', ['DispY','DispY',VarEps[3]], [np.full_like(front,1), np.full_like(behind,-1), np.full_like(front,-0.5*dz)], [front,behind,np.full_like(front,NodeEps[3])], ProblemID = ProblemID)
        BoundaryCondition('MPC', ['DispZ','DispZ',VarEps[2]], [np.full_like(front,1), np.full_like(behind,-1), np.full_like(front,-dz)    ], [front,behind,np.full_like(front,NodeEps[2])], ProblemID = ProblemID)
                
        #elimination of DOF from edge left/top -> edge left/bottom
        BoundaryCondition('MPC', ['DispX','DispX',VarEps[5]], [np.full_like(edge_lt,1), np.full_like(edge_lb,-1), np.full_like(edge_lt,-0.5*dy)], [edge_lt, edge_lb, np.full_like(edge_lt,NodeEps[5])], ProblemID = ProblemID)
        BoundaryCondition('MPC', ['DispY','DispY',VarEps[1]], [np.full_like(edge_lt,1), np.full_like(edge_lb,-1), np.full_like(edge_lt,-dy)    ], [edge_lt, edge_lb, np.full_like(edge_lt,NodeEps[1])], ProblemID = ProblemID)
        BoundaryCondition('MPC', ['DispZ','DispZ',VarEps[3]], [np.full_like(edge_lt,1), np.full_like(edge_lb,-1), np.full_like(edge_lt,-0.5*dy)], [edge_lt, edge_lb, np.full_like(edge_lt,NodeEps[3])], ProblemID = ProblemID)        
        #elimination of DOF from edge right/bottom -> edge left/bottom
        BoundaryCondition('MPC', ['DispX','DispX',VarEps[0]], [np.full_like(edge_rb,1), np.full_like(edge_lb,-1), np.full_like(edge_rb,-dx)    ], [edge_rb, edge_lb, np.full_like(edge_lt,NodeEps[0])], ProblemID = ProblemID)
        BoundaryCondition('MPC', ['DispY','DispY',VarEps[5]], [np.full_like(edge_rb,1), np.full_like(edge_lb,-1), np.full_like(edge_rb,-0.5*dx)], [edge_rb, edge_lb, np.full_like(edge_lt,NodeEps[5])], ProblemID = ProblemID)
        BoundaryCondition('MPC', ['DispZ','DispZ',VarEps[4]], [np.full_like(edge_rb,1), np.full_like(edge_lb,-1), np.full_like(edge_rb,-0.5*dx)], [edge_rb, edge_lb, np.full_like(edge_lt,NodeEps[4])], ProblemID = ProblemID)        
        #elimination of DOF from edge right/top -> edge left/bottom
        BoundaryCondition('MPC', ['DispX','DispX',VarEps[0],VarEps[5]], [np.full_like(edge_rt,1), np.full_like(edge_lb,-1), np.full_like(edge_rt,-dx), np.full_like(edge_rt,-0.5*dy)], [edge_rt, edge_lb, np.full_like(edge_rt,NodeEps[0]), np.full_like(edge_rt,NodeEps[5])], ProblemID = ProblemID)
        BoundaryCondition('MPC', ['DispY','DispY',VarEps[5],VarEps[1]], [np.full_like(edge_rt,1), np.full_like(edge_lb,-1), np.full_like(edge_rt,-0.5*dx), np.full_like(edge_rt,-dy)], [edge_rt, edge_lb, np.full_like(edge_rt,NodeEps[5]), np.full_like(edge_rt,NodeEps[1])], ProblemID = ProblemID)                
        BoundaryCondition('MPC', ['DispZ','DispZ',VarEps[4],VarEps[3]], [np.full_like(edge_rt,1), np.full_like(edge_lb,-1), np.full_like(edge_rt,-0.5*dx), np.full_like(edge_rt,-0.5*dy)], [edge_rt, edge_lb, np.full_like(edge_rt,NodeEps[4]), np.full_like(edge_rt,NodeEps[3])], ProblemID = ProblemID)                
                                       
        #elimination of DOF from edge top/behind -> edge bottom/behind
        BoundaryCondition('MPC', ['DispX','DispX',VarEps[5]], [np.full_like(edge_td,1), np.full_like(edge_bd,-1), np.full_like(edge_td,-0.5*dy)], [edge_td, edge_bd, np.full_like(edge_td,NodeEps[5])], ProblemID = ProblemID)
        BoundaryCondition('MPC', ['DispY','DispY',VarEps[1]], [np.full_like(edge_td,1), np.full_like(edge_bd,-1), np.full_like(edge_td,-dy)    ], [edge_td, edge_bd, np.full_like(edge_td,NodeEps[1])], ProblemID = ProblemID)
        BoundaryCondition('MPC', ['DispZ','DispZ',VarEps[3]], [np.full_like(edge_td,1), np.full_like(edge_bd,-1), np.full_like(edge_td,-0.5*dy)], [edge_td, edge_bd, np.full_like(edge_td,NodeEps[3])], ProblemID = ProblemID)        
        #elimination of DOF from edge bottom/front -> edge bottom/behind
        BoundaryCondition('MPC', ['DispX','DispX',VarEps[4]], [np.full_like(edge_bf,1), np.full_like(edge_bd,-1), np.full_like(edge_bf,-0.5*dz)], [edge_bf, edge_bd, np.full_like(edge_bf,NodeEps[4])], ProblemID = ProblemID)
        BoundaryCondition('MPC', ['DispY','DispY',VarEps[3]], [np.full_like(edge_bf,1), np.full_like(edge_bd,-1), np.full_like(edge_bf,-0.5*dz)], [edge_bf, edge_bd, np.full_like(edge_bf,NodeEps[3])], ProblemID = ProblemID)
        BoundaryCondition('MPC', ['DispZ','DispZ',VarEps[2]], [np.full_like(edge_bf,1), np.full_like(edge_bd,-1), np.full_like(edge_bf,-dz)    ], [edge_bf, edge_bd, np.full_like(edge_bf,NodeEps[2])], ProblemID = ProblemID)        
        #elimination of DOF from edge top/front -> edge bottom/behind
        BoundaryCondition('MPC', ['DispX','DispX',VarEps[5],VarEps[4]], [np.full_like(edge_tf,1), np.full_like(edge_bd,-1), np.full_like(edge_tf,-0.5*dy), np.full_like(edge_tf,-0.5*dz)], [edge_tf, edge_bd, np.full_like(edge_tf,NodeEps[5]), np.full_like(edge_tf,NodeEps[4])], ProblemID = ProblemID)
        BoundaryCondition('MPC', ['DispY','DispY',VarEps[1],VarEps[3]], [np.full_like(edge_tf,1), np.full_like(edge_bd,-1), np.full_like(edge_tf,-dy)    , np.full_like(edge_tf,-0.5*dz)], [edge_tf, edge_bd, np.full_like(edge_tf,NodeEps[1]), np.full_like(edge_tf,NodeEps[3])], ProblemID = ProblemID)                
        BoundaryCondition('MPC', ['DispZ','DispZ',VarEps[3],VarEps[2]], [np.full_like(edge_tf,1), np.full_like(edge_bd,-1), np.full_like(edge_tf,-0.5*dy), np.full_like(edge_tf,-dz)    ], [edge_tf, edge_bd, np.full_like(edge_tf,NodeEps[3]), np.full_like(edge_tf,NodeEps[2])], ProblemID = ProblemID)                
   
        #elimination of DOF from edge right/behind -> edge left/behind
        BoundaryCondition('MPC', ['DispX','DispX',VarEps[0]], [np.full_like(edge_rd,1), np.full_like(edge_ld,-1), np.full_like(edge_rd,-dx)    ], [edge_rd, edge_ld, np.full_like(edge_ld,NodeEps[0])], ProblemID = ProblemID)
        BoundaryCondition('MPC', ['DispY','DispY',VarEps[5]], [np.full_like(edge_rd,1), np.full_like(edge_ld,-1), np.full_like(edge_rd,-0.5*dx)], [edge_rd, edge_ld, np.full_like(edge_ld,NodeEps[5])], ProblemID = ProblemID)
        BoundaryCondition('MPC', ['DispZ','DispZ',VarEps[4]], [np.full_like(edge_rd,1), np.full_like(edge_ld,-1), np.full_like(edge_rd,-0.5*dx)], [edge_rd, edge_ld, np.full_like(edge_ld,NodeEps[4])], ProblemID = ProblemID)        
        #elimination of DOF from edge left/front -> edge left/behind
        BoundaryCondition('MPC', ['DispX','DispX',VarEps[4]], [np.full_like(edge_lf,1), np.full_like(edge_ld,-1), np.full_like(edge_rd,-0.5*dz)], [edge_lf, edge_ld, np.full_like(edge_ld,NodeEps[4])], ProblemID = ProblemID)
        BoundaryCondition('MPC', ['DispY','DispY',VarEps[3]], [np.full_like(edge_lf,1), np.full_like(edge_ld,-1), np.full_like(edge_rd,-0.5*dz)], [edge_lf, edge_ld, np.full_like(edge_ld,NodeEps[3])], ProblemID = ProblemID)
        BoundaryCondition('MPC', ['DispZ','DispZ',VarEps[2]], [np.full_like(edge_lf,1), np.full_like(edge_ld,-1), np.full_like(edge_rd,-dz)    ], [edge_lf, edge_ld, np.full_like(edge_ld,NodeEps[2])], ProblemID = ProblemID)        
        #elimination of DOF from edge right/front -> edge left/behind
        BoundaryCondition('MPC', ['DispX','DispX',VarEps[0],VarEps[4]], [np.full_like(edge_rf,1), np.full_like(edge_ld,-1), np.full_like(edge_rf,-dx    ), np.full_like(edge_rf,-0.5*dz)], [edge_rf, edge_ld, np.full_like(edge_rf,NodeEps[0]), np.full_like(edge_rf,NodeEps[4])], ProblemID = ProblemID)
        BoundaryCondition('MPC', ['DispY','DispY',VarEps[5],VarEps[3]], [np.full_like(edge_rf,1), np.full_like(edge_ld,-1), np.full_like(edge_rf,-0.5*dx), np.full_like(edge_rf,-0.5*dz)], [edge_rf, edge_ld, np.full_like(edge_rf,NodeEps[5]), np.full_like(edge_rf,NodeEps[3])], ProblemID = ProblemID)                
        BoundaryCondition('MPC', ['DispZ','DispZ',VarEps[4],VarEps[2]], [np.full_like(edge_rf,1), np.full_like(edge_ld,-1), np.full_like(edge_rf,-0.5*dx), np.full_like(edge_rf,-dz)    ], [edge_rf, edge_ld, np.full_like(edge_rf,NodeEps[4]), np.full_like(edge_rf,NodeEps[2])], ProblemID = ProblemID)                
        
        # #### CORNER ####
        #elimination of DOF from corner right/bottom/behind (corner_rbd) -> corner left/bottom/behind (corner_lbd) 
        BoundaryCondition('MPC', ['DispX','DispX',VarEps[0]], [np.full_like(corner_rbd,1), np.full_like(corner_lbd,-1), np.full_like(corner_rbd,-dx)    ], [corner_rbd, corner_lbd, np.full_like(corner_rbd,NodeEps[0])], ProblemID = ProblemID)
        BoundaryCondition('MPC', ['DispY','DispY',VarEps[5]], [np.full_like(corner_rbd,1), np.full_like(corner_lbd,-1), np.full_like(corner_rbd,-0.5*dx)], [corner_rbd, corner_lbd, np.full_like(corner_rbd,NodeEps[5])], ProblemID = ProblemID)
        BoundaryCondition('MPC', ['DispZ','DispZ',VarEps[4]], [np.full_like(corner_rbd,1), np.full_like(corner_lbd,-1), np.full_like(corner_rbd,-0.5*dx)], [corner_rbd, corner_lbd, np.full_like(corner_rbd,NodeEps[4])], ProblemID = ProblemID) 
        #elimination of DOF from corner left/top/behind (corner_ltd) -> corner left/bottom/behind (corner_lbd) 
        BoundaryCondition('MPC', ['DispX','DispX',VarEps[5]], [np.full_like(corner_ltd,1), np.full_like(corner_lbd,-1), np.full_like(corner_ltd,-0.5*dy)], [corner_ltd, corner_lbd, np.full_like(corner_ltd,NodeEps[5])], ProblemID = ProblemID)
        BoundaryCondition('MPC', ['DispY','DispY',VarEps[1]], [np.full_like(corner_ltd,1), np.full_like(corner_lbd,-1), np.full_like(corner_ltd,-dy)    ], [corner_ltd, corner_lbd, np.full_like(corner_ltd,NodeEps[1])], ProblemID = ProblemID)
        BoundaryCondition('MPC', ['DispZ','DispZ',VarEps[3]], [np.full_like(corner_ltd,1), np.full_like(corner_lbd,-1), np.full_like(corner_ltd,-0.5*dy)], [corner_ltd, corner_lbd, np.full_like(corner_ltd,NodeEps[3])], ProblemID = ProblemID)
        #elimination of DOF from corner left/bottom/front (corner_lbf) -> corner left/bottom/behind (corner_lbd) 
        BoundaryCondition('MPC', ['DispX','DispX',VarEps[4]], [np.full_like(corner_lbf,1), np.full_like(corner_lbd,-1), np.full_like(corner_lbf,-0.5*dz)], [corner_lbf, corner_lbd, np.full_like(corner_lbf,NodeEps[4])], ProblemID = ProblemID)
        BoundaryCondition('MPC', ['DispY','DispY',VarEps[3]], [np.full_like(corner_lbf,1), np.full_like(corner_lbd,-1), np.full_like(corner_lbf,-0.5*dz)], [corner_lbf, corner_lbd, np.full_like(corner_lbf,NodeEps[3])], ProblemID = ProblemID)
        BoundaryCondition('MPC', ['DispZ','DispZ',VarEps[2]], [np.full_like(corner_lbf,1), np.full_like(corner_lbd,-1), np.full_like(corner_lbf,-dz)    ], [corner_lbf, corner_lbd, np.full_like(corner_lbf,NodeEps[2])], ProblemID = ProblemID)
        #elimination of DOF from corner right/top/behind (corner_rtd) -> corner left/bottom/behind (corner_lbd) 
        BoundaryCondition('MPC', ['DispX','DispX',VarEps[0],VarEps[5]], [np.full_like(corner_rtd,1), np.full_like(corner_lbd,-1), np.full_like(corner_rtd,-dx)    , np.full_like(corner_rtd,-0.5*dy)], [corner_rtd, corner_lbd, np.full_like(corner_rtd,NodeEps[0]), np.full_like(corner_rtd,NodeEps[5])], ProblemID = ProblemID)
        BoundaryCondition('MPC', ['DispY','DispY',VarEps[5],VarEps[1]], [np.full_like(corner_rtd,1), np.full_like(corner_lbd,-1), np.full_like(corner_rtd,-0.5*dx), np.full_like(corner_rtd,-dy)    ], [corner_rtd, corner_lbd, np.full_like(corner_rtd,NodeEps[5]), np.full_like(corner_rtd,NodeEps[1])], ProblemID = ProblemID)
        BoundaryCondition('MPC', ['DispZ','DispZ',VarEps[4],VarEps[3]], [np.full_like(corner_rtd,1), np.full_like(corner_lbd,-1), np.full_like(corner_rtd,-0.5*dx), np.full_like(corner_rtd,-0.5*dy)], [corner_rtd, corner_lbd, np.full_like(corner_rtd,NodeEps[4]), np.full_like(corner_rtd,NodeEps[3])], ProblemID = ProblemID)
        #elimination of DOF from corner left/top/front (corner_ltf) -> corner left/bottom/behind (corner_lbd) 
        BoundaryCondition('MPC', ['DispX','DispX',VarEps[5],VarEps[4]], [np.full_like(corner_ltf,1), np.full_like(corner_lbd,-1), np.full_like(corner_ltf,-0.5*dy), np.full_like(corner_ltf,-0.5*dz)], [corner_ltf, corner_lbd, np.full_like(corner_ltf,NodeEps[5]), np.full_like(corner_ltf,NodeEps[4])], ProblemID = ProblemID)
        BoundaryCondition('MPC', ['DispY','DispY',VarEps[1],VarEps[3]], [np.full_like(corner_ltf,1), np.full_like(corner_lbd,-1), np.full_like(corner_ltf,-dy)    , np.full_like(corner_ltf,-0.5*dz)], [corner_ltf, corner_lbd, np.full_like(corner_ltf,NodeEps[1]), np.full_like(corner_ltf,NodeEps[3])], ProblemID = ProblemID)
        BoundaryCondition('MPC', ['DispZ','DispZ',VarEps[3],VarEps[2]], [np.full_like(corner_ltf,1), np.full_like(corner_lbd,-1), np.full_like(corner_ltf,-0.5*dy), np.full_like(corner_ltf,-dz)    ], [corner_ltf, corner_lbd, np.full_like(corner_ltf,NodeEps[3]), np.full_like(corner_ltf,NodeEps[2])], ProblemID = ProblemID)
        #elimination of DOF from corner right/bottom/front (corner_rbf) -> corner left/bottom/behind (corner_lbd) 
        BoundaryCondition('MPC', ['DispX','DispX',VarEps[0],VarEps[4]], [np.full_like(corner_rbf,1), np.full_like(corner_lbd,-1), np.full_like(corner_rbf,-dx)    , np.full_like(corner_rbf,-0.5*dz)], [corner_rbf, corner_lbd, np.full_like(corner_rbf,NodeEps[0]), np.full_like(corner_rbf,NodeEps[4])], ProblemID = ProblemID)
        BoundaryCondition('MPC', ['DispY','DispY',VarEps[5],VarEps[3]], [np.full_like(corner_rbf,1), np.full_like(corner_lbd,-1), np.full_like(corner_rbf,-0.5*dx), np.full_like(corner_rbf,-0.5*dz)], [corner_rbf, corner_lbd, np.full_like(corner_rbf,NodeEps[5]), np.full_like(corner_rbf,NodeEps[3])], ProblemID = ProblemID)
        BoundaryCondition('MPC', ['DispZ','DispZ',VarEps[4],VarEps[2]], [np.full_like(corner_rbf,1), np.full_like(corner_lbd,-1), np.full_like(corner_rbf,-0.5*dx), np.full_like(corner_rbf,-dz)    ], [corner_rbf, corner_lbd, np.full_like(corner_rbf,NodeEps[4]), np.full_like(corner_rbf,NodeEps[2])], ProblemID = ProblemID)
        
        
        
        
        #elimination of DOF from corner right/top/front (corner_rtf) -> corner left/bottom/behind (corner_lbd) 
        BoundaryCondition('MPC', ['DispX','DispX',VarEps[0],VarEps[5], VarEps[4]], 
                                  [np.full_like(corner_rtf,1), np.full_like(corner_lbd,-1), np.full_like(corner_rtf,-dx)    , np.full_like(corner_rtf,-0.5*dy), np.full_like(corner_rtf,-0.5*dz)], 
                                  [corner_rtf, corner_lbd, np.full_like(corner_rtf,NodeEps[0]), np.full_like(corner_rtf,NodeEps[5]), np.full_like(corner_rtf,NodeEps[4])], ProblemID = ProblemID)
        BoundaryCondition('MPC', ['DispY','DispY',VarEps[5],VarEps[1], VarEps[3]], 
                                  [np.full_like(corner_rtf,1), np.full_like(corner_lbd,-1), np.full_like(corner_rtf,-0.5*dx), np.full_like(corner_rtf,-dy)    , np.full_like(corner_rtf,-0.5*dz)], 
                                  [corner_rtf, corner_lbd, np.full_like(corner_rtf,NodeEps[5]), np.full_like(corner_rtf,NodeEps[1]), np.full_like(corner_rtf,NodeEps[3])], ProblemID = ProblemID)
        BoundaryCondition('MPC', ['DispZ','DispZ',VarEps[4],VarEps[3], VarEps[2]], 
                                  [np.full_like(corner_rtf,1), np.full_like(corner_lbd,-1), np.full_like(corner_rtf,-0.5*dx), np.full_like(corner_rtf,-0.5*dy), np.full_like(corner_rtf,-dz)    ], 
                                  [corner_rtf, corner_lbd, np.full_like(corner_rtf,NodeEps[4]), np.full_like(corner_rtf,NodeEps[2]), np.full_like(corner_rtf,NodeEps[1])], ProblemID = ProblemID)

        
        
        #corner set OK
        #[EpsXX, EpsYY, EpsZZ, EpsYZ, EpsXZ, EpsXY]








    elif dim == 'test':                        
        VarEps = [VarEps[0], VarEps[1], None, None, None, VarEps[2]]
        NodeEps = [NodeEps[0], NodeEps[1], None, None, None, NodeEps[2]]        

        #Define node sets
        left   = np.where( np.abs(crd[:,0] - xmin) < tol )[0]
        right  = np.where( np.abs(crd[:,0] - xmax) < tol )[0]
        bottom = np.where( np.abs(crd[:,1] - ymin) < tol )[0]
        top    = np.where( np.abs(crd[:,1] - ymax) < tol )[0]
        # behind = np.where( np.abs(crd[:,2] - zmin) < tol )[0]
        # front  = np.where( np.abs(crd[:,2] - zmax) < tol )[0]
        
        #l = left, r = right, b = bottom, t = top, f = front, d = behind
        edge_lb = np.intersect1d(left , bottom, assume_unique=True)
        edge_lt = np.intersect1d(left , top   , assume_unique=True)
        edge_rb = np.intersect1d(right, bottom, assume_unique=True)
        edge_rt = np.intersect1d(right, top   , assume_unique=True)
        
        # edge_bd = np.intersect1d(bottom, behind, assume_unique=True)
        # edge_bf = np.intersect1d(bottom, front , assume_unique=True)
        # edge_td = np.intersect1d(top   , behind, assume_unique=True)
        # edge_tf = np.intersect1d(top   , front , assume_unique=True)
        
        # edge_ld = np.intersect1d(left , behind, assume_unique=True)
        # edge_lf = np.intersect1d(left , front , assume_unique=True)
        # edge_rd = np.intersect1d(right, behind, assume_unique=True)
        # edge_rf = np.intersect1d(right, front , assume_unique=True)
        
        #sort edges (required to assign the good pair of nodes)
        edge_lb = edge_lb[np.argsort(crd[edge_lb,2])]
        edge_lt = edge_lt[np.argsort(crd[edge_lt,2])]
        edge_rb = edge_rb[np.argsort(crd[edge_rb,2])]
        edge_rt = edge_rt[np.argsort(crd[edge_rt,2])]
        
        # edge_bd = edge_bd[np.argsort(crd[edge_bd,0])][1:-1] #without corners
        # edge_bf = edge_bf[np.argsort(crd[edge_bf,0])][1:-1] #without corners
        # edge_td = edge_td[np.argsort(crd[edge_td,0])][1:-1] #without corners
        # edge_tf = edge_tf[np.argsort(crd[edge_tf,0])][1:-1] #without corners
        
        # edge_ld = edge_ld[np.argsort(crd[edge_ld,1])][1:-1] #without corners
        # edge_lf = edge_lf[np.argsort(crd[edge_lf,1])][1:-1] #without corners
        # edge_rd = edge_rd[np.argsort(crd[edge_rd,1])][1:-1] #without corners
        # edge_rf = edge_rf[np.argsort(crd[edge_rf,1])][1:-1] #without corners
        
        # #extract corner from left and right edges
        # corner_lbd = edge_lb[[0]]
        # corner_lbf = edge_lb[[-1]]
        # corner_ltd = edge_lt[[0]]
        # corner_ltf = edge_lt[[-1]]
        # corner_rbd = edge_rb[[0]]
        # corner_rbf = edge_rb[[-1]]
        # corner_rtd = edge_rt[[0]]
        # corner_rtf = edge_rt[[-1]]
        # edge_lb = edge_lb[1:-1]
        # edge_lt = edge_lt[1:-1]
        # edge_rb = edge_rb[1:-1]
        # edge_rt = edge_rt[1:-1]
        
        # all_edges = np.hstack((edge_lb, edge_lt, edge_rb, edge_rt, edge_bd, edge_bf, 
        #                        edge_td, edge_tf, edge_ld, edge_lf, edge_rd, edge_rf, 
        #                        corner_lbd, corner_lbf, corner_ltd, corner_ltf, 
        #                        corner_rbd, corner_rbf, corner_rtd, corner_rtf))



        all_edges = np.hstack((edge_lb, edge_lt, edge_rb, edge_rt))
        ####
        
        # intersect1d n'est pas la bonne fonction 
        # trouver une fonction qui enlève les valeurs qui sont dans all_edges
        # setdiff1d est sans doute la solution
        
        ####
        left   = np.setdiff1d(left  , all_edges, assume_unique=False)
        right  = np.setdiff1d(right , all_edges, assume_unique=False)
        bottom = np.setdiff1d(bottom, all_edges, assume_unique=False)
        top    = np.setdiff1d(top   , all_edges, assume_unique=False)
        # behind = np.intersect1d(behind, all_edges, assume_unique=True)
        # front  = np.intersect1d(front , all_edges, assume_unique=True)
        
        #sort adjacent faces to ensure node correspondance
        decimal_round = int(-np.log10(tol)-1)
        left   = left  [np.lexsort((crd[left  ,1], crd[left  ,2].round(decimal_round)))]
        right  = right [np.lexsort((crd[right ,1], crd[right ,2].round(decimal_round)))]
        bottom = bottom[np.lexsort((crd[bottom,0], crd[bottom,2].round(decimal_round)))]
        top    = top   [np.lexsort((crd[top   ,0], crd[top   ,2].round(decimal_round)))]
        # behind = behind[np.lexsort((crd[behind,0], crd[behind,1].round(decimal_round)))]
        # front  = front [np.lexsort((crd[front ,0], crd[front ,1].round(decimal_round)))]

        #now apply periodic boudary conditions        
        dx = xmax-xmin ; dy = ymax-ymin #; dz = zmax-zmin        
        #[EpsXX, EpsYY, EpsZZ, EpsYZ, EpsXZ, EpsXY]
        #Left/right faces
        BoundaryCondition('MPC', ['DispX','DispX',VarEps[0]], [np.full_like(right,1), np.full_like(left, -1), np.full_like(right, -dx)]  , [right,left,np.full_like(right, NodeEps[0])], ProblemID = ProblemID)
        BoundaryCondition('MPC', ['DispY','DispY',VarEps[5]], [np.full_like(right,1), np.full_like(left,-1), np.full_like(right,-0.5*dx)], [right,left,np.full_like(right,NodeEps[5])], ProblemID = ProblemID)
        # BoundaryCondition('MPC', ['DispZ','DispZ',VarEps[4]], [np.full_like(right,1), np.full_like(left,-1), np.full_like(right,-0.5*dx)], [right,left,np.full_like(right,NodeEps[4])], ProblemID = ProblemID)
        #top/bottom faces
        BoundaryCondition('MPC', ['DispX','DispX',VarEps[5]], [np.full_like(top,1), np.full_like(bottom,-1), np.full_like(top,-0.5*dy)], [top,bottom,np.full_like(top,NodeEps[5])], ProblemID = ProblemID)
        BoundaryCondition('MPC', ['DispY','DispY',VarEps[1]], [np.full_like(top,1), np.full_like(bottom,-1), np.full_like(top, -dy)]   , [top,bottom,np.full_like(top,NodeEps[1])], ProblemID = ProblemID)
        # BoundaryCondition('MPC', ['DispZ','DispZ',VarEps[3]], [np.full_like(top,1), np.full_like(bottom,-1), np.full_like(top,-0.5*dy)], [top,bottom,np.full_like(top,NodeEps[3])], ProblemID = ProblemID)
        # #front/behind faces
        # BoundaryCondition('MPC', ['DispX','DispX',VarEps[4]], [np.full_like(front,1), np.full_like(behind,-1), np.full_like(front,-0.5*dz)], [front,behind,np.full_like(front,NodeEps[4])], ProblemID = ProblemID)
        # BoundaryCondition('MPC', ['DispY','DispY',VarEps[3]], [np.full_like(front,1), np.full_like(behind,-1), np.full_like(front,-0.5*dz)], [front,behind,np.full_like(front,NodeEps[3])], ProblemID = ProblemID)
        # BoundaryCondition('MPC', ['DispZ','DispZ',VarEps[2]], [np.full_like(front,1), np.full_like(behind,-1), np.full_like(front,-dz)    ], [front,behind,np.full_like(front,NodeEps[2])], ProblemID = ProblemID)
        
        
        #elimination of DOF from edge left/top -> edge left/bottom
        BoundaryCondition('MPC', ['DispX','DispX',VarEps[5]], [np.full_like(edge_lt,1), np.full_like(edge_lb,-1), np.full_like(edge_lt,-0.5*dy)], [edge_lt, edge_lb, np.full_like(edge_lt,NodeEps[5])], ProblemID = ProblemID)
        BoundaryCondition('MPC', ['DispY','DispY',VarEps[1]], [np.full_like(edge_lt,1), np.full_like(edge_lb,-1), np.full_like(edge_lt,-dy)    ], [edge_lt, edge_lb, np.full_like(edge_lt,NodeEps[1])], ProblemID = ProblemID)
        # BoundaryCondition('MPC', ['DispZ','DispZ',VarEps[3]], [np.full_like(edge_lt,1), np.full_like(edge_lb,-1), np.full_like(edge_lt,-0.5*dy)], [edge_lt, edge_lb, np.full_like(edge_lt,NodeEps[3])], ProblemID = ProblemID)        
        # elimination of DOF from edge right/bottom -> edge left/bottom
        BoundaryCondition('MPC', ['DispX','DispX',VarEps[0]], [np.full_like(edge_rb,1), np.full_like(edge_lb,-1), np.full_like(edge_rb,-dx)    ], [edge_rb, edge_lb, np.full_like(edge_lt,NodeEps[0])], ProblemID = ProblemID)
        BoundaryCondition('MPC', ['DispY','DispY',VarEps[5]], [np.full_like(edge_rb,1), np.full_like(edge_lb,-1), np.full_like(edge_rb,-0.5*dx)], [edge_rb, edge_lb, np.full_like(edge_lt,NodeEps[5])], ProblemID = ProblemID)
        # BoundaryCondition('MPC', ['DispZ','DispZ',VarEps[4]], [np.full_like(edge_rb,1), np.full_like(edge_lb,-1), np.full_like(edge_rb,-0.5*dx)], [edge_rb, edge_lb, np.full_like(edge_lt,NodeEps[4])], ProblemID = ProblemID)        
        #elimination of DOF from edge right/top -> edge left/bottom
        BoundaryCondition('MPC', ['DispX','DispX',VarEps[0],VarEps[5]], [np.full_like(edge_rt,1), np.full_like(edge_lb,-1), np.full_like(edge_rt,-dx), np.full_like(edge_rt,-0.5*dy)], [edge_rt, edge_lb, np.full_like(edge_rt,NodeEps[0]), np.full_like(edge_rt,NodeEps[5])], ProblemID = ProblemID)
        BoundaryCondition('MPC', ['DispY','DispY',VarEps[5],VarEps[1]], [np.full_like(edge_rt,1), np.full_like(edge_lb,-1), np.full_like(edge_rt,-0.5*dx), np.full_like(edge_rt,-dy)], [edge_rt, edge_lb, np.full_like(edge_rt,NodeEps[5]), np.full_like(edge_rt,NodeEps[1])], ProblemID = ProblemID)                
        # BoundaryCondition('MPC', ['DispZ','DispZ',VarEps[4],VarEps[3]], [np.full_like(edge_rt,1), np.full_like(edge_lb,-1), np.full_like(edge_rt,-0.5*dx), np.full_like(edge_rt,-0.5*dy)], [edge_rt, edge_lb, np.full_like(edge_rt,NodeEps[4]), np.full_like(edge_rt,NodeEps[3])], ProblemID = ProblemID)                
                                       
        # #elimination of DOF from edge top/behind -> edge bottom/behind
        # BoundaryCondition('MPC', ['DispX','DispX',VarEps[5]], [np.full_like(edge_td,1), np.full_like(edge_bd,-1), np.full_like(edge_td,-0.5*dy)], [edge_td, edge_bd, np.full_like(edge_td,NodeEps[5])], ProblemID = ProblemID)
        # BoundaryCondition('MPC', ['DispY','DispY',VarEps[1]], [np.full_like(edge_td,1), np.full_like(edge_bd,-1), np.full_like(edge_td,-dy)    ], [edge_td, edge_bd, np.full_like(edge_td,NodeEps[1])], ProblemID = ProblemID)
        # BoundaryCondition('MPC', ['DispZ','DispZ',VarEps[3]], [np.full_like(edge_td,1), np.full_like(edge_bd,-1), np.full_like(edge_td,-0.5*dy)], [edge_td, edge_bd, np.full_like(edge_td,NodeEps[3])], ProblemID = ProblemID)        
        # #elimination of DOF from edge bottom/front -> edge bottom/behind
        # BoundaryCondition('MPC', ['DispX','DispX',VarEps[4]], [np.full_like(edge_bf,1), np.full_like(edge_bd,-1), np.full_like(edge_bf,-0.5*dz)], [edge_bf, edge_bd, np.full_like(edge_bf,NodeEps[4])], ProblemID = ProblemID)
        # BoundaryCondition('MPC', ['DispY','DispY',VarEps[3]], [np.full_like(edge_bf,1), np.full_like(edge_bd,-1), np.full_like(edge_bf,-0.5*dz)], [edge_bf, edge_bd, np.full_like(edge_bf,NodeEps[3])], ProblemID = ProblemID)
        # BoundaryCondition('MPC', ['DispZ','DispZ',VarEps[2]], [np.full_like(edge_bf,1), np.full_like(edge_bd,-1), np.full_like(edge_bf,-dz)    ], [edge_bf, edge_bd, np.full_like(edge_bf,NodeEps[2])], ProblemID = ProblemID)        
        # #elimination of DOF from edge top/front -> edge bottom/behind
        # BoundaryCondition('MPC', ['DispX','DispX',VarEps[5],VarEps[4]], [np.full_like(edge_tf,1), np.full_like(edge_bd,-1), np.full_like(edge_tf,-0.5*dy), np.full_like(edge_tf,-0.5*dz)], [edge_tf, edge_bd, np.full_like(edge_tf,NodeEps[5]), np.full_like(edge_tf,NodeEps[4])], ProblemID = ProblemID)
        # BoundaryCondition('MPC', ['DispY','DispY',VarEps[1],VarEps[3]], [np.full_like(edge_tf,1), np.full_like(edge_bd,-1), np.full_like(edge_tf,-dy)    , np.full_like(edge_tf,-0.5*dz)], [edge_tf, edge_bd, np.full_like(edge_tf,NodeEps[1]), np.full_like(edge_tf,NodeEps[3])], ProblemID = ProblemID)                
        # BoundaryCondition('MPC', ['DispZ','DispZ',VarEps[3],VarEps[2]], [np.full_like(edge_tf,1), np.full_like(edge_bd,-1), np.full_like(edge_tf,-0.5*dy), np.full_like(edge_tf,-dz)    ], [edge_tf, edge_bd, np.full_like(edge_tf,NodeEps[3]), np.full_like(edge_tf,NodeEps[2])], ProblemID = ProblemID)                
   
        # #elimination of DOF from edge right/behind -> edge left/behind
        # BoundaryCondition('MPC', ['DispX','DispX',VarEps[0]], [np.full_like(edge_rd,1), np.full_like(edge_ld,-1), np.full_like(edge_rd,-dx)    ], [edge_rd, edge_ld, np.full_like(edge_ld,NodeEps[0])], ProblemID = ProblemID)
        # BoundaryCondition('MPC', ['DispY','DispY',VarEps[5]], [np.full_like(edge_rd,1), np.full_like(edge_ld,-1), np.full_like(edge_rd,-0.5*dx)], [edge_rd, edge_ld, np.full_like(edge_ld,NodeEps[5])], ProblemID = ProblemID)
        # BoundaryCondition('MPC', ['DispZ','DispZ',VarEps[4]], [np.full_like(edge_rd,1), np.full_like(edge_ld,-1), np.full_like(edge_rd,-0.5*dx)], [edge_rd, edge_ld, np.full_like(edge_ld,NodeEps[4])], ProblemID = ProblemID)        
        # #elimination of DOF from edge left/front -> edge left/behind
        # BoundaryCondition('MPC', ['DispX','DispX',VarEps[4]], [np.full_like(edge_lf,1), np.full_like(edge_ld,-1), np.full_like(edge_rd,-0.5*dz)], [edge_lf, edge_ld, np.full_like(edge_ld,NodeEps[4])], ProblemID = ProblemID)
        # BoundaryCondition('MPC', ['DispY','DispY',VarEps[3]], [np.full_like(edge_lf,1), np.full_like(edge_ld,-1), np.full_like(edge_rd,-0.5*dz)], [edge_lf, edge_ld, np.full_like(edge_ld,NodeEps[3])], ProblemID = ProblemID)
        # BoundaryCondition('MPC', ['DispZ','DispZ',VarEps[2]], [np.full_like(edge_lf,1), np.full_like(edge_ld,-1), np.full_like(edge_rd,-dz)    ], [edge_lf, edge_ld, np.full_like(edge_ld,NodeEps[2])], ProblemID = ProblemID)        
        # #elimination of DOF from edge right/front -> edge left/behind
        # BoundaryCondition('MPC', ['DispX','DispX',VarEps[0],VarEps[4]], [np.full_like(edge_rf,1), np.full_like(edge_ld,-1), np.full_like(edge_rf,-dx    ), np.full_like(edge_rf,-0.5*dz)], [edge_rf, edge_ld, np.full_like(edge_rf,NodeEps[0]), np.full_like(edge_rf,NodeEps[4])], ProblemID = ProblemID)
        # BoundaryCondition('MPC', ['DispY','DispY',VarEps[5],VarEps[3]], [np.full_like(edge_rf,1), np.full_like(edge_ld,-1), np.full_like(edge_rf,-0.5*dx), np.full_like(edge_rf,-0.5*dz)], [edge_rf, edge_ld, np.full_like(edge_rf,NodeEps[5]), np.full_like(edge_rf,NodeEps[3])], ProblemID = ProblemID)                
        # BoundaryCondition('MPC', ['DispZ','DispZ',VarEps[4],VarEps[2]], [np.full_like(edge_rf,1), np.full_like(edge_ld,-1), np.full_like(edge_rf,-0.5*dx), np.full_like(edge_rf,-dz)    ], [edge_rf, edge_ld, np.full_like(edge_rf,NodeEps[4]), np.full_like(edge_rf,NodeEps[2])], ProblemID = ProblemID)                
        
        # #### CORNER ####
        # #elimination of DOF from corner right/bottom/behind (corner_rbd) -> corner left/bottom/behind (corner_lbd) 
        # BoundaryCondition('MPC', ['DispX','DispX',VarEps[0]], [np.full_like(corner_rbd,1), np.full_like(corner_lbd,-1), np.full_like(corner_rbd,-dx)    ], [corner_rbd, corner_lbd, np.full_like(corner_rbd,NodeEps[0])], ProblemID = ProblemID)
        # BoundaryCondition('MPC', ['DispY','DispY',VarEps[5]], [np.full_like(corner_rbd,1), np.full_like(corner_lbd,-1), np.full_like(corner_rbd,-0.5*dx)], [corner_rbd, corner_lbd, np.full_like(corner_rbd,NodeEps[5])], ProblemID = ProblemID)
        # BoundaryCondition('MPC', ['DispZ','DispZ',VarEps[4]], [np.full_like(corner_rbd,1), np.full_like(corner_lbd,-1), np.full_like(corner_rbd,-0.5*dx)], [corner_rbd, corner_lbd, np.full_like(corner_rbd,NodeEps[4])], ProblemID = ProblemID) 
        # #elimination of DOF from corner left/top/behind (corner_ltd) -> corner left/bottom/behind (corner_lbd) 
        # BoundaryCondition('MPC', ['DispX','DispX',VarEps[5]], [np.full_like(corner_ltd,1), np.full_like(corner_lbd,-1), np.full_like(corner_ltd,-0.5*dy)], [corner_ltd, corner_lbd, np.full_like(corner_ltd,NodeEps[5])], ProblemID = ProblemID)
        # BoundaryCondition('MPC', ['DispY','DispY',VarEps[1]], [np.full_like(corner_ltd,1), np.full_like(corner_lbd,-1), np.full_like(corner_ltd,-dy)    ], [corner_ltd, corner_lbd, np.full_like(corner_ltd,NodeEps[1])], ProblemID = ProblemID)
        # BoundaryCondition('MPC', ['DispZ','DispZ',VarEps[3]], [np.full_like(corner_ltd,1), np.full_like(corner_lbd,-1), np.full_like(corner_ltd,-0.5*dy)], [corner_ltd, corner_lbd, np.full_like(corner_ltd,NodeEps[3])], ProblemID = ProblemID)
        # #elimination of DOF from corner left/bottom/front (corner_lbf) -> corner left/bottom/behind (corner_lbd) 
        # BoundaryCondition('MPC', ['DispX','DispX',VarEps[4]], [np.full_like(corner_lbf,1), np.full_like(corner_lbd,-1), np.full_like(corner_lbf,-0.5*dz)], [corner_lbf, corner_lbd, np.full_like(corner_lbf,NodeEps[4])], ProblemID = ProblemID)
        # BoundaryCondition('MPC', ['DispY','DispY',VarEps[3]], [np.full_like(corner_lbf,1), np.full_like(corner_lbd,-1), np.full_like(corner_lbf,-0.5*dz)], [corner_lbf, corner_lbd, np.full_like(corner_lbf,NodeEps[3])], ProblemID = ProblemID)
        # BoundaryCondition('MPC', ['DispZ','DispZ',VarEps[2]], [np.full_like(corner_lbf,1), np.full_like(corner_lbd,-1), np.full_like(corner_lbf,-dz)    ], [corner_lbf, corner_lbd, np.full_like(corner_lbf,NodeEps[2])], ProblemID = ProblemID)
        # #elimination of DOF from corner right/top/behind (corner_rtd) -> corner left/bottom/behind (corner_lbd) 
        # BoundaryCondition('MPC', ['DispX','DispX',VarEps[0],VarEps[5]], [np.full_like(corner_rtd,1), np.full_like(corner_lbd,-1), np.full_like(corner_rtd,-dx)    , np.full_like(corner_rtd,-0.5*dy)], [corner_rtd, corner_lbd, np.full_like(corner_rtd,NodeEps[0]), np.full_like(corner_rtd,NodeEps[5])], ProblemID = ProblemID)
        # BoundaryCondition('MPC', ['DispY','DispY',VarEps[5],VarEps[1]], [np.full_like(corner_rtd,1), np.full_like(corner_lbd,-1), np.full_like(corner_rtd,-0.5*dx), np.full_like(corner_rtd,-dy)    ], [corner_rtd, corner_lbd, np.full_like(corner_rtd,NodeEps[5]), np.full_like(corner_rtd,NodeEps[1])], ProblemID = ProblemID)
        # BoundaryCondition('MPC', ['DispZ','DispZ',VarEps[4],VarEps[3]], [np.full_like(corner_rtd,1), np.full_like(corner_lbd,-1), np.full_like(corner_rtd,-0.5*dx), np.full_like(corner_rtd,-0.5*dy)], [corner_rtd, corner_lbd, np.full_like(corner_rtd,NodeEps[4]), np.full_like(corner_rtd,NodeEps[3])], ProblemID = ProblemID)
        # #elimination of DOF from corner left/top/front (corner_ltf) -> corner left/bottom/behind (corner_lbd) 
        # BoundaryCondition('MPC', ['DispX','DispX',VarEps[5],VarEps[4]], [np.full_like(corner_ltf,1), np.full_like(corner_lbd,-1), np.full_like(corner_ltf,-0.5*dy), np.full_like(corner_ltf,-0.5*dz)], [corner_ltf, corner_lbd, np.full_like(corner_ltf,NodeEps[5]), np.full_like(corner_ltf,NodeEps[4])], ProblemID = ProblemID)
        # BoundaryCondition('MPC', ['DispY','DispY',VarEps[1],VarEps[3]], [np.full_like(corner_ltf,1), np.full_like(corner_lbd,-1), np.full_like(corner_ltf,-dy)    , np.full_like(corner_ltf,-0.5*dz)], [corner_ltf, corner_lbd, np.full_like(corner_ltf,NodeEps[1]), np.full_like(corner_ltf,NodeEps[3])], ProblemID = ProblemID)
        # BoundaryCondition('MPC', ['DispZ','DispZ',VarEps[3],VarEps[2]], [np.full_like(corner_ltf,1), np.full_like(corner_lbd,-1), np.full_like(corner_ltf,-0.5*dy), np.full_like(corner_ltf,-dz)    ], [corner_ltf, corner_lbd, np.full_like(corner_ltf,NodeEps[3]), np.full_like(corner_ltf,NodeEps[2])], ProblemID = ProblemID)
        # #elimination of DOF from corner right/bottom/front (corner_rbf) -> corner left/bottom/behind (corner_lbd) 
        # BoundaryCondition('MPC', ['DispX','DispX',VarEps[0],VarEps[4]], [np.full_like(corner_rbf,1), np.full_like(corner_lbd,-1), np.full_like(corner_rbf,-dx)    , np.full_like(corner_rbf,-0.5*dz)], [corner_rbf, corner_lbd, np.full_like(corner_rbf,NodeEps[5]), np.full_like(corner_rbf,NodeEps[4])], ProblemID = ProblemID)
        # BoundaryCondition('MPC', ['DispY','DispY',VarEps[5],VarEps[3]], [np.full_like(corner_rbf,1), np.full_like(corner_lbd,-1), np.full_like(corner_rbf,-0.5*dx), np.full_like(corner_rbf,-0.5*dz)], [corner_rbf, corner_lbd, np.full_like(corner_rbf,NodeEps[1]), np.full_like(corner_rbf,NodeEps[3])], ProblemID = ProblemID)
        # BoundaryCondition('MPC', ['DispZ','DispZ',VarEps[4],VarEps[2]], [np.full_like(corner_rbf,1), np.full_like(corner_lbd,-1), np.full_like(corner_rbf,-0.5*dx), np.full_like(corner_rbf,-dz)    ], [corner_rbf, corner_lbd, np.full_like(corner_rbf,NodeEps[3]), np.full_like(corner_rbf,NodeEps[2])], ProblemID = ProblemID)
        # #elimination of DOF from corner right/top/front (corner_rtf) -> corner left/bottom/behind (corner_lbd) 
        # BoundaryCondition('MPC', ['DispX','DispX',VarEps[0],VarEps[5], VarEps[4]], 
        #                          [np.full_like(corner_rtf,1), np.full_like(corner_lbd,-1), np.full_like(corner_rtf,-dx)    , np.full_like(corner_rtf,-0.5*dy), np.full_like(corner_rtf,-0.5*dz)], 
        #                          [corner_rtf, corner_lbd, np.full_like(corner_rtf,NodeEps[0]), np.full_like(corner_rtf,NodeEps[5]), np.full_like(corner_rtf,NodeEps[4])], ProblemID = ProblemID)
        # BoundaryCondition('MPC', ['DispY','DispY',VarEps[5],VarEps[1], VarEps[3]], 
        #                          [np.full_like(corner_rtf,1), np.full_like(corner_lbd,-1), np.full_like(corner_rtf,-0.5*dx), np.full_like(corner_rtf,-dy)    , np.full_like(corner_rtf,-0.5*dz)], 
        #                          [corner_rtf, corner_lbd, np.full_like(corner_rtf,NodeEps[5]), np.full_like(corner_rtf,NodeEps[1]), np.full_like(corner_rtf,NodeEps[3])], ProblemID = ProblemID)
        # BoundaryCondition('MPC', ['DispZ','DispZ',VarEps[4],VarEps[3], VarEps[2]], 
        #                          [np.full_like(corner_rtf,1), np.full_like(corner_lbd,-1), np.full_like(corner_rtf,-0.5*dx), np.full_like(corner_rtf,-0.5*dy), np.full_like(corner_rtf,-dz)    ], 
        #                          [corner_rtf, corner_lbd, np.full_like(corner_rtf,NodeEps[4]), np.full_like(corner_rtf,NodeEps[2]), np.full_like(corner_rtf,NodeEps[1])], ProblemID = ProblemID)

        
        
        #[EpsXX, EpsYY, EpsZZ, EpsYZ, EpsXZ, EpsXY]
                            