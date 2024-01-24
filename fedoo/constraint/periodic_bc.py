# from fedoo.core.base   import ProblemBase 
import numpy as np
from fedoo.core.boundary_conditions import BCBase, MPC, ListBC
from fedoo.core.base import ProblemBase
from fedoo.core.mesh import MeshBase
    
class PeriodicBC(BCBase):
    """Class defining periodic boundary conditions"""

    def __init__(self, node_cd, var_cd, dim=None, tol=1e-8, meshperio = True, name = "Periodicity"):
        """   
        Create a perdiodic boundary condition object using several multi-points constraints.
        Some constraint driver (cd) dof  are used to define mean strain or mean displacement gradient.
        The dof associated to a contraint driver is difined by the node indice (defined in node_cd) 
        and the associated variable (defined in var_vd).
        
        The constraint drivers can be defined in several way
            * [Eps_XX] or [[Eps_XX]] -> strain dof for 1D periodicity
            * [Eps_XX, Eps_YY, 2*Eps_XY] -> strain dof for 2D periodicity using Voigt notation for shear components
            * [Eps_XX, Eps_YY, Eps_ZZ, 2*Eps_XY, 2*Eps_XZ, 2*Eps_YZ] -> strain dof for 3D periodicity using Voigt notation for shear components
            * [[DU_XX, DU_XY], [DU_YX, DU_YY]] -> gradient of displacement for 2D periodicity
            * [[DU_XX, DU_XY, DU_XZ], [DU_YX, DU_YY, DU_YZ], [DU_ZX, DU_ZY, DU_ZZ]] 
            -> gradient of displacement for 3D periodicity                       
        
        Parameters
        ----------
        node_cd : list of nodes, or list of list of nodes 
            Nodes used as constraint drivers for each strain component. The dof used as contraint drivers are defined in var_cd.
        var_cd : list of str, or list of list of str.
            Variables used as constraint drivers. The len of lists should be the same as node_cd.
        dim : int in [1,2,3] (default = assess from the constraint drivers dimension)
            Number of dimensions with periodic condition. 
            If dim = 1: the periodicity is assumed along the x coordinate
            If dim = 2: the periodicity is assumed along x and y coordinates            
            
        tol : float, optional
            Tolerance for the periodic nodes detection. The default is 1e-8.
        name : str, optional
            Name of the created boundary condition. The default is "Periodicity".
        
        
        Remarks
        ---------------
        
        * The boundary condition object needs to be used with a problem associated to 
        a periodic mesh. 
        * The periodic nodes are automatically detected using the given tolerance (tol). 
        * The nodes of the Xp (x=xmax), Yp (y=ymax) and Zp (z=zmax) faces are 
        eliminated from the system (slave nodes) and can't be used in another mpc.
        
        Example
        ---------
        
        .. code-block:: python
            
            import fedoo as fd
            
            mesh = fd.mesh.box_mesh()        
            
            #add nodes not associated to any element for constraint driver
            node_cd = fd.Mesh["Domain2"].add_nodes(crd_center, 3)  
            
            list_strain_nodes = [StrainNodes[0], StrainNodes[0], StrainNodes[0],
                                    StrainNodes[1], StrainNodes[1], StrainNodes[1]]
            list_strain_var = ['DispX', 'DispY', 'DispZ','DispX', 'DispY', 'DispZ']
            
            # or using the displacement gradient formulation (in this case the shear strain are true strain component): 
            # list_strain_nodes = [[StrainNodes[0], StrainNodes[1], StrainNodes[1]],
            #                      [StrainNodes[1], StrainNodes[0], StrainNodes[1]],
            #                      [StrainNodes[1], StrainNodes[1], StrainNodes[0]]]
            # list_strain_var = [['DispX', 'DispX', 'DispY'],
            #                    ['DispX', 'DispY', 'DispZ'],
            #                    ['DispY', 'DispZ', 'DispZ']]
            
            bc_periodic = fd.homogen.PeriodicBC(list_strain_nodes, list_strain_var) 
            
        """
    
        self.shear_coef = 1
        if np.isscalar(node_cd[0]):
            self.shear_coef = 0.5
            if len(node_cd) == 1: 
                if dim is None: dim = 1
                var_cd = [var_cd]
                node_cd = [node_cd]
            elif len(node_cd) == 3: 
                if dim is None: dim = 2
                node_cd = [[node_cd[0], node_cd[2]],
                            [node_cd[2], node_cd[1]]]
                var_cd =  [[var_cd[0], var_cd[2]],
                            [var_cd[2], var_cd[1]]]
                
            elif len(node_cd) == 6:
                if dim is None: dim = 3 
                node_cd = [[node_cd[0], node_cd[3], node_cd[4]],
                            [node_cd[3], node_cd[1], node_cd[5]],
                            [node_cd[4], node_cd[5], node_cd[2]]]
                var_cd =  [[var_cd[0], var_cd[3], var_cd[4]],
                            [var_cd[3], var_cd[1], var_cd[5]],
                            [var_cd[4], var_cd[5], var_cd[2]]]
            else: 
                raise NameError('Lenght of node_cd and var_cd should be 1,3 or 6')
        
        elif dim is None: 
            dim = len(node_cd[0])       
                
        self.node_cd = node_cd
        self.var_cd = var_cd
        self.dim = dim #dimension of periodicity (1, 2 or 3)
        self.tol = tol
        self.bc_type = 'PeriodicBC'
        BCBase.__init__(self, name)
        
        self.meshperio = meshperio
        
        
    def __repr__(self):
        list_str = ['{}D Periodic Boundary Condition:'.format(self.dim)]
        if self.name != "": list_str.append("name = '{}'".format(self.name))
        
        return "\n".join(list_str)


    def _prepare_periodic_lists(self)
        """ 
        Prepare the lists knowing that :
        # Xm = left
        # Xp = right
        # Ym = bottom  
        # Yp = top
        # Zm = back
        # Zp = fron

        """

        xmax = np.max(crd[:,0]) ; xmin = np.min(crd[:,0])
        ymax = np.max(crd[:,1]) ; ymin = np.min(crd[:,1])
        if self.dim == 3:                        
            zmax = np.max(crd[:,2]) ; zmin = np.min(crd[:,2])

        face_Xm   = np.where( np.abs(crd[:,0] - xmin) < tol )[0]
        face_Xp  = np.where( np.abs(crd[:,0] - xmax) < tol )[0]
        
        if self.dim > 1:
            Ym = np.where( np.abs(crd[:,1] - ymin) < tol )[0]
            Yp    = np.where( np.abs(crd[:,1] - ymax) < tol )[0]
            
            #extract edges/corners from the intersection of faces
            edge_XmYm  = np.intersect1d(Xm , Ym, assume_unique=True)
            edge_XmYp     = np.intersect1d(Xm , Yp   , assume_unique=True)
            edge_XpYm = np.intersect1d(Xp, Ym, assume_unique=True)
            edge_XpYp    = np.intersect1d(Xp, Yp   , assume_unique=True)

            if self.dim > 2: #or dim == 3 
                face_Zm = np.where( np.abs(crd[:,2] - zmin) < tol )[0]
                face_Zp  = np.where( np.abs(crd[:,2] - zmax) < tol )[0]  
                                
                #extract edges/corners from the intersection of faces
                edge_YmZm = np.intersect1d(Ym, Zm, assume_unique=True)
                edge_YmZp = np.intersect1d(Ym, Zp , assume_unique=True)
                edge_YpZm = np.intersect1d(Yp   , Zm, assume_unique=True)
                edge_YpZp = np.intersect1d(Yp   , Zp , assume_unique=True)
                
                edge_XmZm  = np.intersect1d(Xm , Zm, assume_unique=True)
                edge_XmZp  = np.intersect1d(Xm , Zp , assume_unique=True)
                edge_XpZm  = np.intersect1d(Xp, Zm, assume_unique=True)
                edge_XpZp  = np.intersect1d(Xp, Zp , assume_unique=True)        
                
                #extract corners from the intersection of edges
                corner_XmYmZm  = np.intersect1d(edge_XmYm , edge_YmZm, assume_unique=True) 
                corner_XmYmZp  = np.intersect1d(edge_XmYm , edge_YmZp, assume_unique=True)
                corner_XmYpZm  = np.intersect1d(edge_XmYp , edge_YpZm, assume_unique=True)
                corner_XmYpZp  = np.intersect1d(edge_XmYp , edge_YpZp, assume_unique=True)
                corner_XpYmZm  = np.intersect1d(edge_XpYm , edge_YmZm, assume_unique=True)
                corner_XpYmZp  = np.intersect1d(edge_XpYm , edge_YmZp, assume_unique=True)
                corner_XpYpZm  = np.intersect1d(edge_XpYp , edge_YpZm, assume_unique=True)
                corner_XpYpZp  = np.intersect1d(edge_XpYp , edge_YpZp, assume_unique=True)
        
                
                # Remove nodes that beloing to several sets
                all_corners = np.hstack((corner_XmYmZm, corner_XmYmZp, corner_XmYpZm, corner_XmYpZp, 
                                            corner_XpYmZm, corner_XpYmZp, corner_XpYpZm, corner_XpYpZp))

                edge_XmYm = np.setdiff1d(edge_XmYm, all_corners, assume_unique=True)
                edge_XmYp = np.setdiff1d(edge_XmYp, all_corners, assume_unique=True)
                edge_XpYm = np.setdiff1d(edge_XpYm, all_corners, assume_unique=True)
                edge_XpYp = np.setdiff1d(edge_XpYp, all_corners, assume_unique=True)

                edge_YmZm = np.setdiff1d(edge_YmZm, all_corners, assume_unique=True)
                edge_YmZp = np.setdiff1d(edge_YmZp, all_corners, assume_unique=True)
                edge_YpZm = np.setdiff1d(edge_YpZm, all_corners, assume_unique=True)
                edge_YpZp = np.setdiff1d(edge_YpZp, all_corners, assume_unique=True)

                edge_XmZm = np.setdiff1d(edge_XmZm, all_corners, assume_unique=True)
                edge_XmZp = np.setdiff1d(edge_XmZp, all_corners, assume_unique=True)
                edge_XpZm = np.setdiff1d(edge_XpZm, all_corners, assume_unique=True)
                edge_XpZp = np.setdiff1d(edge_XpZp, all_corners, assume_unique=True)
            
            
                all_edges = np.hstack((edge_XmYm, edge_XmYp, edge_XpYm, edge_XpYp, edge_YmZm, edge_YmZp, 
                                        edge_YpZm, edge_YpZp, edge_XmZm, edge_XmZp, edge_XpZm, edge_XpZp, 
                                        all_corners))
        
            else: #dim = 2
                all_edges = np.hstack((edge_XmYm, edge_XmYp, edge_XpYm, edge_XpYp))
        
            face_Xm   = np.setdiff1d(Xm  , all_edges, assume_unique=True)
            face_Xp  = np.setdiff1d(Xp , all_edges, assume_unique=True)
            face_Ym = np.setdiff1d(Ym, all_edges, assume_unique=True)
            face_Yp    = np.setdiff1d(Yp   , all_edges, assume_unique=True)
        
            if mesh.ndim > 2: #if there is a z coordinate
                #sort edges (required to assign the good pair of nodes)
                edge_XmYm = edge_XmYm[np.argsort(crd[edge_XmYm,2])]
                edge_XmYp = edge_XmYp[np.argsort(crd[edge_XmYp,2])]
                edge_XpYm = edge_XpYm[np.argsort(crd[edge_XpYm,2])]
                edge_XpYp = edge_XpYp[np.argsort(crd[edge_XpYp,2])]
        
            if self.dim > 2: 
                face_Zm = np.setdiff1d(Zm, all_edges, assume_unique=True)
                face_Zp  = np.setdiff1d(Zp , all_edges, assume_unique=True)
                
                edge_YmZm = edge_YmZm[np.argsort(crd[edge_YmZm,0])] 
                edge_YmZp = edge_YmZp[np.argsort(crd[edge_YmZp,0])]
                edge_YpZm = edge_YpZm[np.argsort(crd[edge_YpZm,0])]
                edge_YpZp = edge_YpZp[np.argsort(crd[edge_YpZp,0])]
                
                edge_XmZm = edge_XmZm[np.argsort(crd[edge_XmZm,1])]
                edge_XmZp = edge_XmZp[np.argsort(crd[edge_XmZp,1])]
                edge_XpZm = edge_XpZm[np.argsort(crd[edge_XpZm,1])]
                edge_XpZp = edge_XpZp[np.argsort(crd[edge_XpZp,1])]
                
        #sort adjacent faces to ensure node correspondance
        if mesh.ndim == 2: 
            face_Xm   = face_Xm   [np.argsort(crd[face_Xm  ,1])]
            face_Xp  = face_Xp  [np.argsort(crd[face_Xp ,1])]
            if self.dim > 1:
                face_Ym = face_Ym [np.argsort(crd[face_Ym,0])]
                face_Yp    = face_Yp    [np.argsort(crd[face_Yp   ,0])]
            
        elif mesh.ndim > 2: 
            decimal_round = int(-np.log10(tol)-1)
            face_Xm   = Xm  [np.lexsort((crd[face_Xm  ,1], crd[face_Xm  ,2].round(decimal_round)))]
            face_Xp  = face_Xp [np.lexsort((crd[face_Xp ,1], crd[face_Xp ,2].round(decimal_round)))]
            if self.dim > 1:
                face_Ym = face_Ym[np.lexsort((crd[face_Ym,0], crd[face_Ym,2].round(decimal_round)))]
                face_Yp    = face_Yp   [np.lexsort((crd[face_Yp   ,0], crd[face_Yp   ,2].round(decimal_round)))]
            if self.dim > 2: 
                face_Zm = face_Zm[np.lexsort((crd[face_Zm,0], crd[face_Zm,1].round(decimal_round)))]
                face_Zp  = face_Zp [np.lexsort((crd[face_Zp ,0], crd[face_Zp ,1].round(decimal_round)))]        

        return {
            'face_Xm': (face_Xm),
            'face_Ym': (face_Ym),
            'face_Zm': (face_Zm),                        
            'face_Xp': (face_Xp),
            'face_Yp': (face_Yp),
            'face_Zp': (face_Zp),        
            'edge_XmYm': (edge_XmYm),
            'edge_XmZm': (edge_XmZm),
            'edge_YmZm': (edge_YmZm),
            'edge_XpYm': (edge_XpYm),
            'edge_XpYp': (edge_XpYp),
            'edge_XmYp': (edge_XmYp),
            'edge_XpZm': (edge_XpZm),
            'edge_XpZp': (edge_XpZp),
            'edge_XmZp': (edge_XmZp),
            'edge_YpZm': (edge_YpZm),
            'edge_YpZp': (edge_YpZp),
            'edge_YmZp': (edge_YmZp),
            'corner_XmYmZm': (corner_XmYmZm),
            'corner_XmYmZp': (corner_XmYmZp),
            'corner_XmYpZm': (corner_XmYpZm),
            'corner_XmYpZp': (corner_XmYpZp),
            'corner_XpYmZm': (corner_XpYmZm),
            'corner_XpYmZp': (corner_XpYmZp),
            'corner_XpYpZm': (corner_XpYpZm),
            'corner_XpYpZp': (corner_XpYpZp),
        }

    def _list_MPC_periodic(slef, node_cd, var_cd, d_rve, dic_faces_edges_periodic):

add dx, dy, dz        


        face_Xm = dic_faces_edges_periodic('face_Xm')
        face_Ym = dic_faces_edges_periodic('face_Ym')
        face_Zm = dic_faces_edges_periodic('face_Zm')                
        face_Xp = dic_faces_edges_periodic('face_Xp')
        face_Yp = dic_faces_edges_periodic('face_Yp')
        face_Zp = dic_faces_edges_periodic('face_Zp')
        edge_XmYm = dic_faces_edges_periodic('edge_XmYm')
        edge_XmZm = dic_faces_edges_periodic('edge_XmZm')
        edge_YmZm = dic_faces_edges_periodic('edge_YmZm')
        edge_XpYm = dic_faces_edges_periodic('edge_XpYm')
        edge_XpYp = dic_faces_edges_periodic('edge_XpYp')
        edge_XmYp = dic_faces_edges_periodic('edge_XmYp')
        edge_XpZm = dic_faces_edges_periodic('edge_XpZm')
        edge_XpZp = dic_faces_edges_periodic('edge_XpZp')
        edge_XmZp = dic_faces_edges_periodic('edge_XmZp')
        edge_YpZm = dic_faces_edges_periodic('edge_YpZm')
        edge_YpZp = dic_faces_edges_periodic('edge_YpZp')
        edge_YmZp = dic_faces_edges_periodic('edge_YmZp')
        corner_XmYmZm = dic_faces_edges_periodic('corner_XmYmZm')
        corner_XmYmZp = dic_faces_edges_periodic('corner_XmYmZp')
        corner_XmYpZm = dic_faces_edges_periodic('corner_XmYpZm')
        corner_XmYpZp = dic_faces_edges_periodic('corner_XmYpZp')
        corner_XpYmZm = dic_faces_edges_periodic('corner_XpYmZm')
        corner_XpYmZp = dic_faces_edges_periodic('corner_XpYmZp')
        corner_XpYpZm = dic_faces_edges_periodic('corner_XpYpZm')
        corner_XpYpZp = dic_faces_edges_periodic('corner_XpYpZp')


        res = ListBC()    
        #face_Xm/Xp faces (DispX)
        res.append(MPC([face_Xp,face_Xm,np.full_like(face_Xp,node_cd[0][0])], ['DispX','DispX',var_cd[0][0]], [np.full_like(face_Xp,1), np.full_like(face_Xm,-1), np.full_like(face_Xp,-dx, dtype=float)]))
        
        if self.dim > 1:
            #face_Xm/face_Xp faces (DispY)
            res.append(MPC([face_Xp,face_Xm,np.full_like(face_Xp,node_cd[1][0])], ['DispY','DispY',var_cd[1][0]], [np.full_like(face_Xp,1), np.full_like(face_Xm,-1), np.full_like(face_Xp,-sc*dx, dtype=float)]))
            
            #face_Yp/face_Ym faces (DispX and DispY)
            res.append(MPC([face_Yp,face_Ym,np.full_like(face_Yp,node_cd[0][1])], ['DispX','DispX',var_cd[0][1]], [np.full_like(face_Yp,1), np.full_like(face_Ym,-1), np.full_like(face_Yp,-sc*dy, dtype=float)]))
            res.append(MPC([face_Yp,face_Ym,np.full_like(face_Yp,node_cd[1][1])], ['DispY','DispY',var_cd[1][1]], [np.full_like(face_Yp,1), np.full_like(face_Ym,-1), np.full_like(face_Yp,-dy, dtype=float)]))
            
            #elimination of DOF from edge Xm/Yp -> edge Xm/Ym (DispX, DispY)
            res.append(MPC([edge_XmYp, edge_XmYm, np.full_like(edge_XmYp,node_cd[0][1])], ['DispX','DispX',var_cd[0][1]], [np.full_like(edge_XmYp,1), np.full_like(edge_XmYm,-1), np.full_like(edge_XmYp,-sc*dy, dtype=float)]))
            res.append(MPC([edge_XmYp, edge_XmYm, np.full_like(edge_XmYp,node_cd[1][1])], ['DispY','DispY',var_cd[1][1]], [np.full_like(edge_XmYp,1), np.full_like(edge_XmYm,-1), np.full_like(edge_XmYp,-dy, dtype=float)]))
            #elimination of DOF from edge face_Xp/Ym -> edge Xm/Ym (DispX, DispY)
            res.append(MPC([edge_XpYm, edge_XmYm, np.full_like(edge_XmYp,node_cd[0][0])], ['DispX','DispX',var_cd[0][0]], [np.full_like(edge_XpYm,1), np.full_like(edge_XmYm,-1), np.full_like(edge_XpYm,-dx, dtype=float)]))
            res.append(MPC([edge_XpYm, edge_XmYm, np.full_like(edge_XmYp,node_cd[1][0])], ['DispY','DispY',var_cd[1][0]], [np.full_like(edge_XpYm,1), np.full_like(edge_XmYm,-1), np.full_like(edge_XpYm,-sc*dx, dtype=float)]))
            #elimination of DOF from edge Xp/Yp -> edge Xm/Ym (DispX, DispY)
            res.append(MPC([edge_XpYp, edge_XmYm, np.full_like(edge_XpYp,node_cd[0][0]), np.full_like(edge_XpYp,node_cd[0][1])], ['DispX','DispX',var_cd[0][0],var_cd[0][1]], [np.full_like(edge_XpYp,1), np.full_like(edge_XmYm,-1), np.full_like(edge_XpYp,-dx, dtype=float), np.full_like(edge_XpYp,-sc*dy, dtype=float)]))
            res.append(MPC([edge_XpYp, edge_XmYm, np.full_like(edge_XpYp,node_cd[1][0]), np.full_like(edge_XpYp,node_cd[1][1])], ['DispY','DispY',var_cd[1][0],var_cd[1][1]], [np.full_like(edge_XpYp,1), np.full_like(edge_XmYm,-1), np.full_like(edge_XpYp,-sc*dx, dtype=float), np.full_like(edge_XpYp,-dy, dtype=float)]))

            
        if self.dim > 2:
            #DispZ for Xm/Xp faces
            res.append(MPC([face_Xp,face_Xm,np.full_like(face_Xp,node_cd[2][0])], ['DispZ','DispZ',var_cd[2][0]], [np.full_like(face_Xp,1), np.full_like(face_Xm,-1), np.full_like(face_Xp,-sc*dx, dtype=float)]))
            #DispZ for Yp/Ym faces
            res.append(MPC([face_Yp,face_Ym,np.full_like(face_Yp,node_cd[2][1])], ['DispZ','DispZ',var_cd[2][1]], [np.full_like(face_Yp,1), np.full_like(face_Ym,-1), np.full_like(face_Yp,-sc*dy, dtype=float)]))
        
            #elimination of DOF from edge Xm/Yp -> edge Xm/Ym (DispZ)
            res.append(MPC([edge_XmYp, edge_XmYm, np.full_like(edge_XmYp,node_cd[2][1])], ['DispZ','DispZ',var_cd[2][1]], [np.full_like(edge_XmYp,1), np.full_like(edge_XmYm,-1), np.full_like(edge_XmYp,-sc*dy, dtype=float)]))
            #elimination of DOF from edge Xp/Ym -> edge Xm/Ym (DispZ)
            res.append(MPC([edge_XpYm, edge_XmYm, np.full_like(edge_XmYp,node_cd[2][0])], ['DispZ','DispZ',var_cd[2][0]], [np.full_like(edge_XpYm,1), np.full_like(edge_XmYm,-1), np.full_like(edge_XpYm,-sc*dx, dtype=float)]))
            #elimination of DOF from edge Xp/Yp -> edge Xm/Ym (DispZ)
            res.append(MPC([edge_XpYp, edge_XmYm, np.full_like(edge_XpYp,node_cd[2][0]), np.full_like(edge_XpYp,node_cd[2][1])], ['DispZ','DispZ',var_cd[2][0],var_cd[2][1]], [np.full_like(edge_XpYp,1), np.full_like(edge_XmYm,-1), np.full_like(edge_XpYp,-sc*dx, dtype=float), np.full_like(edge_XpYp,-sc*dy, dtype=float)])) 

            
            #Zp/Zm faces
            res.append(MPC([face_Zp,face_Zm,np.full_like(face_Zp,node_cd[0][2])], ['DispX','DispX',var_cd[0][2]], [np.full_like(face_Zp,1), np.full_like(face_Zm,-1), np.full_like(face_Zp,-sc*dz, dtype=float)]))
            res.append(MPC([face_Zp,face_Zm,np.full_like(face_Zp,node_cd[1][2])], ['DispY','DispY',var_cd[1][2]], [np.full_like(face_Zp,1), np.full_like(face_Zm,-1), np.full_like(face_Zp,-sc*dz, dtype=float)]))
            res.append(MPC([face_Zp,face_Zm,np.full_like(face_Zp,node_cd[2][2])], ['DispZ','DispZ',var_cd[2][2]], [np.full_like(face_Zp,1), np.full_like(face_Zm,-1), np.full_like(face_Zp,-dz, dtype=float)]))
                
            #elimination of DOF from edge Yp/Zm -> edge Ym/Zm
            res.append(MPC([edge_YpZm, edge_YmZm, np.full_like(edge_YpZm,node_cd[0][1])], ['DispX','DispX',var_cd[0][1]], [np.full_like(edge_YpZm,1), np.full_like(edge_YmZm,-1), np.full_like(edge_YpZm,-sc*dy, dtype=float)]))
            res.append(MPC([edge_YpZm, edge_YmZm, np.full_like(edge_YpZm,node_cd[1][1])], ['DispY','DispY',var_cd[1][1]], [np.full_like(edge_YpZm,1), np.full_like(edge_YmZm,-1), np.full_like(edge_YpZm,-dy, dtype=float)]))
            res.append(MPC([edge_YpZm, edge_YmZm, np.full_like(edge_YpZm,node_cd[2][1])], ['DispZ','DispZ',var_cd[2][1]], [np.full_like(edge_YpZm,1), np.full_like(edge_YmZm,-1), np.full_like(edge_YpZm,-sc*dy, dtype=float)]))
            #elimination of DOF from edge Ym/Zp -> edge Ym/Zm
            res.append(MPC([edge_YmZp, edge_YmZm, np.full_like(edge_YmZp,node_cd[0][2])], ['DispX','DispX',var_cd[0][2]], [np.full_like(edge_YmZp,1), np.full_like(edge_YmZm,-1), np.full_like(edge_YmZp,-sc*dz, dtype=float)]))
            res.append(MPC([edge_YmZp, edge_YmZm, np.full_like(edge_YmZp,node_cd[1][2])], ['DispY','DispY',var_cd[1][2]], [np.full_like(edge_YmZp,1), np.full_like(edge_YmZm,-1), np.full_like(edge_YmZp,-sc*dz, dtype=float)]))
            res.append(MPC([edge_YmZp, edge_YmZm, np.full_like(edge_YmZp,node_cd[2][2])], ['DispZ','DispZ',var_cd[2][2]], [np.full_like(edge_YmZp,1), np.full_like(edge_YmZm,-1), np.full_like(edge_YmZp,-dz, dtype=float)]))
            #elimination of DOF from edge Yp/Zp -> edge Ym/Zm
            res.append(MPC([edge_YpZp, edge_YmZm, np.full_like(edge_YpZp,node_cd[0][1]), np.full_like(edge_YpZp,node_cd[0][2])], ['DispX','DispX',var_cd[0][1],var_cd[0][2]], [np.full_like(edge_YpZp,1), np.full_like(edge_YmZm,-1), np.full_like(edge_YpZp,-sc*dy, dtype=float), np.full_like(edge_YpZp,-sc*dz, dtype=float)]))
            res.append(MPC([edge_YpZp, edge_YmZm, np.full_like(edge_YpZp,node_cd[1][1]), np.full_like(edge_YpZp,node_cd[1][2])], ['DispY','DispY',var_cd[1][1],var_cd[1][2]], [np.full_like(edge_YpZp,1), np.full_like(edge_YmZm,-1), np.full_like(edge_YpZp,-dy, dtype=float),    np.full_like(edge_YpZp,-sc*dz, dtype=float)]))
            res.append(MPC([edge_YpZp, edge_YmZm, np.full_like(edge_YpZp,node_cd[2][1]), np.full_like(edge_YpZp,node_cd[2][2])], ['DispZ','DispZ',var_cd[2][1],var_cd[2][2]], [np.full_like(edge_YpZp,1), np.full_like(edge_YmZm,-1), np.full_like(edge_YpZp,-sc*dy, dtype=float), np.full_like(edge_YpZp,-dz, dtype=float)]))
            
            #elimination of DOF from edge Xp/Zm -> edge Xm/Zm
            res.append(MPC([edge_XpZm, edge_XmZm, np.full_like(edge_XmZm,node_cd[0][0])], ['DispX','DispX',var_cd[0][0]], [np.full_like(edge_XpZm,1), np.full_like(edge_XmZm,-1), np.full_like(edge_XpZm,-dx, dtype=float)]))
            res.append(MPC([edge_XpZm, edge_XmZm, np.full_like(edge_XmZm,node_cd[1][0])], ['DispY','DispY',var_cd[1][0]], [np.full_like(edge_XpZm,1), np.full_like(edge_XmZm,-1), np.full_like(edge_XpZm,-sc*dx, dtype=float)]))
            res.append(MPC([edge_XpZm, edge_XmZm, np.full_like(edge_XmZm,node_cd[2][0])], ['DispZ','DispZ',var_cd[2][0]], [np.full_like(edge_XpZm,1), np.full_like(edge_XmZm,-1), np.full_like(edge_XpZm,-sc*dx, dtype=float)]))
            #elimination of DOF from edge Xm/Zp -> edge Xm/Zm
            res.append(MPC([edge_XmZp, edge_XmZm, np.full_like(edge_XmZm,node_cd[0][2])], ['DispX','DispX',var_cd[0][2]], [np.full_like(edge_XmZp,1), np.full_like(edge_XmZm,-1), np.full_like(edge_XpZm,-sc*dz, dtype=float)]))
            res.append(MPC([edge_XmZp, edge_XmZm, np.full_like(edge_XmZm,node_cd[1][2])], ['DispY','DispY',var_cd[1][2]], [np.full_like(edge_XmZp,1), np.full_like(edge_XmZm,-1), np.full_like(edge_XpZm,-sc*dz, dtype=float)]))
            res.append(MPC([edge_XmZp, edge_XmZm, np.full_like(edge_XmZm,node_cd[2][2])], ['DispZ','DispZ',var_cd[2][2]], [np.full_like(edge_XmZp,1), np.full_like(edge_XmZm,-1), np.full_like(edge_XpZm,-dz, dtype=float)]))
            #elimination of DOF from edge Xp/Zp -> edge Xm/Zm
            res.append(MPC([edge_XpZp, edge_XmZm, np.full_like(edge_XpZp,node_cd[0][0]), np.full_like(edge_XpZp,node_cd[0][2])], ['DispX','DispX',var_cd[0][0],var_cd[0][2]], [np.full_like(edge_XpZp,1), np.full_like(edge_XmZm,-1), np.full_like(edge_XpZp,-dx, dtype=float),    np.full_like(edge_XpZp,-sc*dz, dtype=float)]))
            res.append(MPC([edge_XpZp, edge_XmZm, np.full_like(edge_XpZp,node_cd[1][0]), np.full_like(edge_XpZp,node_cd[1][2])], ['DispY','DispY',var_cd[1][0],var_cd[1][2]], [np.full_like(edge_XpZp,1), np.full_like(edge_XmZm,-1), np.full_like(edge_XpZp,-sc*dx, dtype=float), np.full_like(edge_XpZp,-sc*dz, dtype=float)]))
            res.append(MPC([edge_XpZp, edge_XmZm, np.full_like(edge_XpZp,node_cd[2][0]), np.full_like(edge_XpZp,node_cd[2][2])], ['DispZ','DispZ',var_cd[2][0],var_cd[2][2]], [np.full_like(edge_XpZp,1), np.full_like(edge_XmZm,-1), np.full_like(edge_XpZp,-sc*dx, dtype=float), np.full_like(edge_XpZp,-dz, dtype=float)]))
            
            # #### CORNER ####
            #elimination of DOF from corner Xp/Ym/Zm (XpYmZm) -> corner Xm/Ym/Zm (XmYmZm) 
            res.append(MPC([corner_XpYmZm, corner_XmYmZm, np.full_like(corner_XpYmZm,node_cd[0][0])], ['DispX','DispX',var_cd[0][0]], [np.full_like(corner_XpYmZm,1), np.full_like(corner_XmYmZm,-1), np.full_like(corner_XpYmZm,-dx, dtype=float)]))
            res.append(MPC([corner_XpYmZm, corner_XmYmZm, np.full_like(corner_XpYmZm,node_cd[1][0])], ['DispY','DispY',var_cd[1][0]], [np.full_like(corner_XpYmZm,1), np.full_like(corner_XmYmZm,-1), np.full_like(corner_XpYmZm,-sc*dx, dtype=float)]))
            res.append(MPC([corner_XpYmZm, corner_XmYmZm, np.full_like(corner_XpYmZm,node_cd[2][0])], ['DispZ','DispZ',var_cd[2][0]], [np.full_like(corner_XpYmZm,1), np.full_like(corner_XmYmZm,-1), np.full_like(corner_XpYmZm,-sc*dx, dtype=float)]))
            #elimination of DOF from corner Xm/Yp/Zm (XmYpZm) -> corner Xm/Ym/Zm (XmYmZm) 
            res.append(MPC([corner_XmYpZm, corner_XmYmZm, np.full_like(corner_XmYpZm,node_cd[0][1])], ['DispX','DispX',var_cd[0][1]], [np.full_like(corner_XmYpZm,1), np.full_like(corner_XmYmZm,-1), np.full_like(corner_XmYpZm,-sc*dy, dtype=float)]))
            res.append(MPC([corner_XmYpZm, corner_XmYmZm, np.full_like(corner_XmYpZm,node_cd[1][1])], ['DispY','DispY',var_cd[1][1]], [np.full_like(corner_XmYpZm,1), np.full_like(corner_XmYmZm,-1), np.full_like(corner_XmYpZm,-dy, dtype=float)]))
            res.append(MPC([corner_XmYpZm, corner_XmYmZm, np.full_like(corner_XmYpZm,node_cd[2][1])], ['DispZ','DispZ',var_cd[2][1]], [np.full_like(corner_XmYpZm,1), np.full_like(corner_XmYmZm,-1), np.full_like(corner_XmYpZm,-sc*dy, dtype=float)]))
            #elimination of DOF from corner Xm/Ym/Zp (XmYmZp) -> corner Xm/Ym/Zm (XmYmZm) 
            res.append(MPC([corner_XmYmZp, corner_XmYmZm, np.full_like(corner_XmYmZp,node_cd[0][2])], ['DispX','DispX',var_cd[0][2]], [np.full_like(corner_XmYmZp,1), np.full_like(corner_XmYmZm,-1), np.full_like(corner_XmYmZp,-sc*dz, dtype=float)]))
            res.append(MPC([corner_XmYmZp, corner_XmYmZm, np.full_like(corner_XmYmZp,node_cd[1][2])], ['DispY','DispY',var_cd[1][2]], [np.full_like(corner_XmYmZp,1), np.full_like(corner_XmYmZm,-1), np.full_like(corner_XmYmZp,-sc*dz, dtype=float)]))
            res.append(MPC([corner_XmYmZp, corner_XmYmZm, np.full_like(corner_XmYmZp,node_cd[2][2])], ['DispZ','DispZ',var_cd[2][2]], [np.full_like(corner_XmYmZp,1), np.full_like(corner_XmYmZm,-1), np.full_like(corner_XmYmZp,-dz, dtype=float)]))
            #elimination of DOF from corner Xp/Yp/Zm (XpYpZm) -> corner Xm/Ym/Zm (XmYmZm) 
            res.append(MPC([corner_XpYpZm, corner_XmYmZm, np.full_like(corner_XpYpZm,node_cd[0][0]), np.full_like(corner_XpYpZm,node_cd[0][1])], ['DispX','DispX',var_cd[0][0],var_cd[0][1]], [np.full_like(corner_XpYpZm,1), np.full_like(corner_XmYmZm,-1), np.full_like(corner_XpYpZm,-dx, dtype=float),    np.full_like(corner_XpYpZm,-sc*dy, dtype=float)]))
            res.append(MPC([corner_XpYpZm, corner_XmYmZm, np.full_like(corner_XpYpZm,node_cd[1][0]), np.full_like(corner_XpYpZm,node_cd[1][1])], ['DispY','DispY',var_cd[1][0],var_cd[1][1]], [np.full_like(corner_XpYpZm,1), np.full_like(corner_XmYmZm,-1), np.full_like(corner_XpYpZm,-sc*dx, dtype=float), np.full_like(corner_XpYpZm,-dy, dtype=float)]))
            res.append(MPC([corner_XpYpZm, corner_XmYmZm, np.full_like(corner_XpYpZm,node_cd[2][0]), np.full_like(corner_XpYpZm,node_cd[2][1])], ['DispZ','DispZ',var_cd[2][0],var_cd[2][1]], [np.full_like(corner_XpYpZm,1), np.full_like(corner_XmYmZm,-1), np.full_like(corner_XpYpZm,-sc*dx, dtype=float), np.full_like(corner_XpYpZm,-sc*dy, dtype=float)]))
            #elimination of DOF from corner Xm/Yp/Zp (XmYpZp) -> corner Xm/Ym/Zm (XmYmZm) 
            res.append(MPC([corner_XmYpZp, corner_XmYmZm, np.full_like(corner_XmYpZp,node_cd[0][1]), np.full_like(corner_XmYpZp,node_cd[0][2])], ['DispX','DispX',var_cd[0][1],var_cd[0][2]], [np.full_like(corner_XmYpZp,1), np.full_like(corner_XmYmZm,-1), np.full_like(corner_XmYpZp,-sc*dy, dtype=float), np.full_like(corner_XmYpZp,-sc*dz, dtype=float)]))
            res.append(MPC([corner_XmYpZp, corner_XmYmZm, np.full_like(corner_XmYpZp,node_cd[1][1]), np.full_like(corner_XmYpZp,node_cd[1][2])], ['DispY','DispY',var_cd[1][1],var_cd[1][2]], [np.full_like(corner_XmYpZp,1), np.full_like(corner_XmYmZm,-1), np.full_like(corner_XmYpZp,-dy, dtype=float),    np.full_like(corner_XmYpZp,-sc*dz, dtype=float)]))
            res.append(MPC([corner_XmYpZp, corner_XmYmZm, np.full_like(corner_XmYpZp,node_cd[2][1]), np.full_like(corner_XmYpZp,node_cd[2][2])], ['DispZ','DispZ',var_cd[2][1],var_cd[2][2]], [np.full_like(corner_XmYpZp,1), np.full_like(corner_XmYmZm,-1), np.full_like(corner_XmYpZp,-sc*dy, dtype=float), np.full_like(corner_XmYpZp,-dz, dtype=float)]))
            #elimination of DOF from corner Xp/Ym/Zp (XpYmZp) -> corner Xm/Ym/Zm (XmYmZm) 
            res.append(MPC([corner_XpYmZp, corner_XmYmZm, np.full_like(corner_XpYmZp,node_cd[0][0]), np.full_like(corner_XpYmZp,node_cd[0][2])], ['DispX','DispX',var_cd[0][0],var_cd[0][2]], [np.full_like(corner_XpYmZp,1), np.full_like(corner_XmYmZm,-1), np.full_like(corner_XpYmZp,-dx, dtype=float),    np.full_like(corner_XpYmZp,-sc*dz, dtype=float)]))
            res.append(MPC([corner_XpYmZp, corner_XmYmZm, np.full_like(corner_XpYmZp,node_cd[1][0]), np.full_like(corner_XpYmZp,node_cd[1][2])], ['DispY','DispY',var_cd[1][0],var_cd[1][2]], [np.full_like(corner_XpYmZp,1), np.full_like(corner_XmYmZm,-1), np.full_like(corner_XpYmZp,-sc*dx, dtype=float), np.full_like(corner_XpYmZp,-sc*dz, dtype=float)]))
            res.append(MPC([corner_XpYmZp, corner_XmYmZm, np.full_like(corner_XpYmZp,node_cd[2][0]), np.full_like(corner_XpYmZp,node_cd[2][2])], ['DispZ','DispZ',var_cd[2][0],var_cd[2][2]], [np.full_like(corner_XpYmZp,1), np.full_like(corner_XmYmZm,-1), np.full_like(corner_XpYmZp,-sc*dx, dtype=float), np.full_like(corner_XpYmZp,-dz, dtype=float)]))
            
                    
            #elimination of DOF from corner Xp/Yp/Zp (XpYpZp) -> corner Xm/Ym/Zm (XmYmZm) 
            res.append(MPC([corner_XpYpZp, corner_XmYmZm, np.full_like(corner_XpYpZp,node_cd[0][0]), np.full_like(corner_XpYpZp,node_cd[0][1]), np.full_like(corner_XpYpZp,node_cd[0][2])], 
                                        ['DispX','DispX',var_cd[0][0],var_cd[0][1], var_cd[0][2]], 
                                        [np.full_like(corner_XpYpZp,1), np.full_like(corner_XmYmZm,-1), np.full_like(corner_XpYpZp,-dx, dtype=float), np.full_like(corner_XpYpZp,-sc*dy, dtype=float), np.full_like(corner_XpYpZp,-sc*dz, dtype=float)]))
            res.append(MPC([corner_XpYpZp, corner_XmYmZm, np.full_like(corner_XpYpZp,node_cd[1][0]), np.full_like(corner_XpYpZp,node_cd[1][1]), np.full_like(corner_XpYpZp,node_cd[1][2])],
                                        ['DispY','DispY',var_cd[1][0],var_cd[1][1], var_cd[1][2]], 
                                        [np.full_like(corner_XpYpZp,1), np.full_like(corner_XmYmZm,-1), np.full_like(corner_XpYpZp,-sc*dx, dtype=float), np.full_like(corner_XpYpZp,-dy, dtype=float), np.full_like(corner_XpYpZp,-sc*dz, dtype=float)]))
            res.append(MPC([corner_XpYpZp, corner_XmYmZm, np.full_like(corner_XpYpZp,node_cd[2][0]), np.full_like(corner_XpYpZp,node_cd[2][1]), np.full_like(corner_XpYpZp,node_cd[2][2])],
                                        ['DispZ','DispZ',var_cd[2][0],var_cd[2][1], var_cd[2][2]], 
                                        [np.full_like(corner_XpYpZp,1), np.full_like(corner_XmYmZm,-1), np.full_like(corner_XpYpZp,-sc*dx, dtype=float), np.full_like(corner_XpYpZp,-sc*dy, dtype=float), np.full_like(corner_XpYpZp,-dz, dtype=float)]))



    def _list_MPC_non_periodic_node_distance(node_cd, var_cd, d_rve, dic_closest_points_on_boundaries):

        res = ListBC()        

        face_Xm = dic_closest_points_on_boundaries['face_Xm']
        face_Xp = dic_closest_points_on_boundaries['face_Xp']


        face_Xm = dic_faces_edges_periodic('face_Xm')
        face_Ym = dic_faces_edges_periodic('face_Ym')
        face_Zm = dic_faces_edges_periodic('face_Zm')                
        face_Xp = dic_faces_edges_periodic('face_Xp')
        face_Yp = dic_faces_edges_periodic('face_Yp')
        face_Zp = dic_faces_edges_periodic('face_Zp')
        edge_XmYm = dic_faces_edges_periodic('edge_XmYm')
        edge_XmZm = dic_faces_edges_periodic('edge_XmZm')
        edge_YmZm = dic_faces_edges_periodic('edge_YmZm')
        edge_XpYm = dic_faces_edges_periodic('edge_XpYm')
        edge_XpYp = dic_faces_edges_periodic('edge_XpYp')
        edge_XmYp = dic_faces_edges_periodic('edge_XmYp')
        edge_XpZm = dic_faces_edges_periodic('edge_XpZm')
        edge_XZp = dic_faces_edges_periodic('edge_XpZp')
        edge_XmZp = dic_faces_edges_periodic('edge_XmZp')
        edge_YpZm = dic_faces_edges_periodic('edge_YpZm')
        edge_YpZp = dic_faces_edges_periodic('edge_YpZp')
        edge_YmZp = dic_faces_edges_periodic('edge_YmZp')

        node_cd = [[node_cd[0], node_cd[3], node_cd[4]],
                    [node_cd[3], node_cd[1], node_cd[5]],
                    [node_cd[4], node_cd[5], node_cd[2]]]

        var_cd =  [[var_cd[0], var_cd[3], var_cd[4]],
                    [var_cd[3], var_cd[1], var_cd[5]],
                    [var_cd[4], var_cd[5], var_cd[2]]]

        face_Xp_1_asarray = np.asarray(face_Xp[1])

        list_node_sets = np.concatenate((face_Xp[0], face_Xm, np.full_like(face_Xm,node_cd[0][0], dtype=object)), axis=1)
        list_variables = ['DispX' for i in range(0,face_Xp_1_asarray.shape[1])] + ['DispX'] + [var_cd[0][0]]

        face_Xp_1_asarray = np.asarray(face_Xp[1])
        dimensions_to_factors_rescaled = face_Xp_1_asarray/np.sum(face_Xp_1_asarray, axis=1).reshape(-1,1)
        list_factors = np.concatenate((dimensions_to_factors_rescaled, np.full_like(face_Xm,-1), np.full_like(face_Xm,-d_rve[0], dtype=float)), axis=1)

        res.append(MPC(list_node_sets, list_variables,list_factors))


    def initialize(self, problem):
        mesh = problem.mesh                
        tol = self.tol
        var_cd = self.var_cd
        node_cd = self.node_cd           

        crd = mesh.nodes        
        list_var = problem.space.list_variables() #list of variable id defined in the active modeling space

        dx = xmax-xmin 
        if self.dim > 1: dy = ymax-ymin 
        if self.dim > 2: dz = zmax-zmin                
        
        sc = self.shear_coef
        res = ListBC()        

        #if rot DOF are used, apply continuity of the rotational dof
        list_rot_var = []
        if 'RotX' in list_var: list_rot_var.append('RotX')
        if 'RotY' in list_var: list_rot_var.append('RotY')
        if 'RotZ' in list_var: list_rot_var.append('RotZ')
        
        #also applied continuity to non used disp component
        if self.dim < 3 and 'DispZ' in list_var: list_rot_var.append('DispZ')
        if self.dim == 1 and 'DispY' in list_var: list_rot_var.append('DispY')
        
        for var in list_rot_var: 
            #### FACES ####
            res.append(MPC([face_Xp,face_Xm], [var,var], [np.full_like(face_Xp,1), np.full_like(face_Xm,-1)]))
            if self.dim>1:
                res.append(MPC([face_Yp,face_Ym], [var,var], [np.full_like(face_Yp,1), np.full_like(face_Ym,-1)]))
            if self.dim>2: 
                res.append(MPC([face_Zp,face_Zm], [var,var], [np.full_like(face_Zp,1), np.full_like(face_Zm,-1)]))
                        
            #### EDGES ####
            if self.dim>1: 
                res.append(MPC([edge_XmYp, edge_XmYm], [var,var], [np.full_like(edge_XmYp,1), np.full_like(edge_XmYm,-1)]))
                res.append(MPC([edge_XpYm, edge_XmYm], [var,var], [np.full_like(edge_XpYm,1), np.full_like(edge_XmYm,-1)]))        
                res.append(MPC([edge_XpYp, edge_XmYm], [var,var], [np.full_like(edge_XpYp,1), np.full_like(edge_XmYm,-1)]))  
            
            if self.dim >2: 
                res.append(MPC([edge_YpZm, edge_YmZm], [var,var], [np.full_like(edge_YpZm,1), np.full_like(edge_YmZm,-1)]))
                res.append(MPC([edge_YmZp, edge_YmZm], [var,var], [np.full_like(edge_YmZp,1), np.full_like(edge_YmZm,-1)]))        
                res.append(MPC([edge_YpZp, edge_YmZm], [var,var], [np.full_like(edge_YpZp,1), np.full_like(edge_YmZm,-1)]))  
            
                res.append(MPC([edge_XpZm, edge_XmZm], [var,var], [np.full_like(edge_XpZm,1), np.full_like(edge_XmZm,-1)]))
                res.append(MPC([edge_XmZp, edge_XmZm], [var,var], [np.full_like(edge_XmZp,1), np.full_like(edge_XmZm,-1)]))
                res.append(MPC([edge_XpZp, edge_XmZm], [var,var], [np.full_like(edge_XpZp,1), np.full_like(edge_XmZm,-1)]))  
                                
                #### CORNERS ####
                res.append(MPC([corner_XpYmZm, corner_XmYmZm], [var,var], [np.full_like(corner_XpYmZm,1), np.full_like(corner_XmYmZm,-1)]))
                res.append(MPC([corner_XmYpZm, corner_XmYmZm], [var,var], [np.full_like(corner_XmYpZm,1), np.full_like(corner_XmYmZm,-1)]))
                res.append(MPC([corner_XmYmZp, corner_XmYmZm], [var,var], [np.full_like(corner_XmYmZp,1), np.full_like(corner_XmYmZm,-1)]))
                res.append(MPC([corner_XpYpZm, corner_XmYmZm], [var,var], [np.full_like(corner_XpYpZm,1), np.full_like(corner_XmYmZm,-1)]))
                res.append(MPC([corner_XmYpZp, corner_XmYmZm], [var,var], [np.full_like(corner_XmYpZp,1), np.full_like(corner_XmYmZm,-1)]))
                res.append(MPC([corner_XpYmZp, corner_XmYmZm], [var,var], [np.full_like(corner_XpYmZp,1), np.full_like(corner_XmYmZm,-1)]))
                res.append(MPC([corner_corner_XpYpZp, XmYmZm], [var,var], [np.full_like(corner_XpYpZp,1), np.full_like(corner_XmYmZm,-1)]))

        res.initialize(problem)
        self.list_mpc = res

    def generate(self, problem, t_fact=1, t_fact_old=None):
        return self.list_mpc.generate(problem, t_fact, t_fact_old)

