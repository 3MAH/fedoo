"""Fedoo Mesh object"""

import numpy as np

# from fedoo.util.Coordinate import Coordinate
from fedoo.core.base import MeshBase
from fedoo.lib_elements.element_list import get_DefaultNbPG, get_element
from scipy import sparse

from os.path import splitext
try:
    import pyvista as pv
    # import vtk
    USE_PYVISTA = True
except:
    USE_PYVISTA = False


class Mesh(MeshBase):    
    """
    Fedoo Mesh object.
        
    Parameters
    ----------
    
    nodes: numpy array of float
        List of nodes coordinates. nodes[i] is the coordinate of the ith node.
    elements: numpy array of int
        Elements table. elements[i] define the nodes associated to the ith element.
    elm_type: str
        Type of the element. The type of the element should be coherent with the shape of elements.
    ndim:
        Dimension of the mesh. By default, ndim is deduced from the nodes coordinates using ndim = nodes.shape[1]    
    name: str
        The name of the mesh
        
    
    Alternative constructor
    --------------------------
    
    It is possible to build a mesh using the following static methods:
      * Mesh.read(filename, name) to load a mesh from a file 
      * Mesh.from_pyvista(pvmesh, name) to build a mesh from pyvista UnstructuredGrid object 
    
    
    Example
    --------
    
    Create a one element mesh from a 2d mesh in a 3d space:
        
      >>> import fedoo as fd
      >>> import numpy as np      
      >>> nodes = np.array([[0,0],[1,0],[1,1],[0,1]])    
      >>> elm = np.array([0,1,2,3])
      >>> mesh = fd.Mesh(nodes, elm, 'quad4', ndim = 3, name = 'unit square mesh')
    """
    
    
    def __init__(self, nodes, elements=None, elm_type=None, ndim = None, name = ""):
        MeshBase.__init__(self, name)
        self.nodes = nodes #node coordinates     
        """ List of nodes coordinates: nodes[i] gives the coordinates of the ith node."""
        self.elements = elements #element table
        """ List of elements: elements[i] gives the nodes associated to the ith element."""
        self.elm_type = elm_type
        """Type of the element (str). The type of the element should be coherent with the shape of elements."""
        self.node_sets = {} #node on the boundary for instance
        """Dict containing node sets associated to the mesh"""
        self.element_sets = {}
        """Dict containing element sets associated to the mesh"""
        self.local_frame = None #contient le repere locale (3 vecteurs unitaires) en chaque noeud. Vaut 0 si pas de rep locaux definis

        if ndim is None: ndim = self.nodes.shape[1]
        elif ndim > self.nodes.shape[1]:
            dim_add = ndim-self.nodes.shape[1]
            self.nodes = np.c_[self.nodes, np.zeros((self.n_nodes, dim_add))]
            # if ndim == 3 and local_frame is not None:
            #     local_frame_temp = np.zeros((self.n_nodes,3,3))
            #     local_frame_temp[:,:2,:2] = self.local_frame
            #     local_frame_temp[:,2,2]   = 1
            #     self.local_frame = local_frame_temp
        
        if ndim == 1: self.crd_name = ('X')
        elif self.nodes.shape[1] == 2: self.crd_name = ('X', 'Y')
        # elif n == '2Dplane' or n == '2Dstress': self.crd_name = ('X', 'Y')
        else: self.crd_name = ('X', 'Y', 'Z')
        
        self._saved_gausspoint2node_mat = {}
        self._saved_node2gausspoint_mat = {}
        self._saved_gaussian_quadrature_mat = {}
        self._elm_interpolation = {}
        self._sparse_structure = {}
        
    
    @staticmethod
    def from_pyvista(pvmesh, name = ""):
        """Build a Mesh from a pyvista UnstructuredGrid mesh. 
        Node and element data are not copied.
        For now, only mesh with single element type may be imported.
        Multi-element meshes will be integrated later. """        
        if USE_PYVISTA:     
            if len(pvmesh.cells_dict) != 1: return NotImplemented
            
            elm_type =  {3:'lin2',
                         5:'tri3',
                         9:'quad4',
                         10:'tet4',
                         12:'hex8',
                         13:'wed6',
                         14:'pyr5',
                         21:'lin3',
                         22:'tri6',
                         23:'quad8', 
                         24:'tet10',
                         25:'hex20'
                         }.get(pvmesh.celltypes[0], None)
                                   
            if elm_type is None: raise NameError('Element Type '+ str(elm_type) + ' not available in pyvista')            
            
            elm = list(pvmesh.cells_dict.values())[0]
            # elm = pvmesh.cells.reshape(-1,pvmesh.cells[0]+1)[1:]            
            
            return Mesh(pvmesh.points, elm, elm_type, name=name)
        else:
            raise NameError('Pyvista not installed.')
            
    
    @staticmethod
    def read(filename, name = ""):
        """Build a Mesh from a mesh. This function use the pyvista load method. 
        The file type is inferred from the file name.
        For now, only mesh with single element type may be imported.
        Multi-element meshes will be integrated later. """     
        if USE_PYVISTA:   
            mesh = Mesh.from_pyvista(pv.read(filename), name=name)
            return mesh
        else:
            raise NameError('Pyvista not installed.')

        
    
    def add_node_set(self,node_indices,name):
        """        
        Add a set of nodes to the Mesh
        
        Parameters
        ----------
        node indices : list or 1D numpy.array
            A list of node indices
        name : str
            name of the set of nodes            
        """
        self.node_sets[name] = node_indices
        
        
    def add_element_set(self,element_indices,name):
        """        
        Add a set of elements to the Mesh
        
        Parameters
        ----------
        element_indices : list or 1D numpy.array
            A list of node indexes
        name : str
            name of the set of nodes            
        """
        self.element_sets[name] = element_indices
        
           
    def add_nodes(self, coordinates = None, nb_added = None):
        """
        Add some nodes to the node list.
        
        The new nodes are not liked to any element.

        Parameters
        ----------
        coordinates : np.ndarray
            The coordinates of the new nodes.             
        nb_added : int
            Number of new nodes
            By default, the value is deduced from she shape of coordinates
            nb_added is used only for creating several nodes with the same coordinates
        """
        n_nodes_old = self.n_nodes
        if nb_added is None and coordinates is None: nb_added = 1
        if nb_added is None:
            self.nodes = np.vstack((self.nodes, coordinates))
        else:
            if coordinates is None:
                self.nodes = np.vstack((self.nodes, 
                    np.zeros([nb_added,self.nodes.shape[1]])))
            else:
                self.nodes = np.vstack((self.nodes, 
                    np.tile(coordinates,(nb_added,1))))

        return np.arange(n_nodes_old,self.n_nodes)

    def add_internal_nodes(self, nb_added):
        new_nodes = self.add_nodes(nb_added=self.n_elements*nb_added)
        self.elements = np.c_[self.elements, new_nodes]
        self.reset_interpolation()
        
    
    # warning , this method must be static
    @staticmethod
    def stack(mesh1,mesh2, name = ""):       
        """
        *Static method* - Make the spatial stack of two mesh objects which have the same element shape. 
        This function doesn't merge coindicent Nodes. 
        For that purpose, use the Mesh methods 'find_coincident_nodes' and 'merge_nodes'
        on the resulting Mesh. 
                
        Return 
        ---------
        Mesh object with is the spacial stack of mesh1 and mesh2
        """
        if isinstance(mesh1, str): mesh1 = Mesh.get_all()[mesh1]
        if isinstance(mesh2, str): mesh2 = Mesh.get_all()[mesh2]
        
        if mesh1.elm_type != mesh2.elm_type:    
            raise NameError("Can only stack meshes with the same element shape")
            
        n_nodes = mesh1.n_nodes
        n_elements = mesh1.n_elements
         
        new_crd = np.r_[mesh1.nodes , mesh2.nodes]
        new_elm = np.r_[mesh1.elements , mesh2.elements + n_nodes]
        
        new_ndSets = dict(mesh1.node_sets)
        for key in mesh2.node_sets:
            if key in mesh1.node_sets:
                new_ndSets[key] = np.r_[mesh1.node_sets[key], np.array(mesh2.node_sets[key]) + n_nodes]
            else:
                new_ndSets[key] = np.array(mesh2.node_sets[key]) + n_nodes                                  
        
        new_elSets = dict(mesh1.element_sets)
        for key in mesh2.element_sets:
            if key in mesh1.element_sets:
                new_elSets[key] = np.r_[mesh1.element_sets[key], np.array(mesh2.element_sets[key]) + n_elements]
            else:
                new_elSets[key] = np.array(mesh2.element_sets[key]) + n_elements    
                   
        mesh3 = Mesh(new_crd, new_elm, mesh1.elm_type, name = name)
        mesh3.node_sets = new_ndSets
        mesh3.element_sets = new_elSets
        return mesh3
    

    def find_coincident_nodes(self,tol=1e-8):
        """ 
        Find some nodes with the same position considering a tolerance given by the argument tol. 
        return an array of shape (number_coincident_nodes, 2) where each line is a pair of nodes that are at the same position.
        These pairs of nodes can be merged using :
            meshObject.merge_nodes(meshObject.find_coincident_nodes()) 
            
        where meshObject is the Mesh object containing merged coincidentNodes.
        """
        n_nodes = self.n_nodes
        decimal_round = int(-np.log10(tol)-1)
        crd = self.nodes.round(decimal_round) #round coordinates to match tolerance
        ind_sorted   = np.lexsort((crd[:  ,2], crd[:  ,1], crd[:  ,0]))

        ind_coincident = np.where(np.linalg.norm(crd[ind_sorted[:-1]]-crd[ind_sorted[1:]], axis = 1) == 0)[0] #indices of the first coincident nodes
        return np.array([ind_sorted[ind_coincident], ind_sorted[ind_coincident+1]]).T
 
    
    def merge_nodes(self,node_couples):
        """ 
        Merge some nodes 
        The total number and the id of nodes are modified
        """
        n_nodes = self.n_nodes
        nds_del = node_couples[:,1] #list des noeuds a supprimer
        nds_kept = node_couples[:,0] #list des noeuds a conserver
         
        unique_nodes, ordre = np.unique(nds_del, return_index=True)
        assert len(unique_nodes) == len(nds_del), "A node can't be deleted 2 times"
        # ordre = np.argsort(nds_del)
        j=0 
        new_num = np.zeros(n_nodes,dtype = 'int')
        for nd in range(n_nodes):    
            if j<len(nds_del) and nd==nds_del[ordre[j]]: 
                #test if some nodes are equal to deleted node among the kept nodes. If required update the kept nodes values
                deleted_nodes = np.where(nds_kept == nds_del[ordre[j]])[0] #index of nodes to kept that are deleted and need to be updated to their new values
                nds_kept[deleted_nodes] = nds_kept[ordre[j]]
                j+=1
            else: new_num[nd] = nd-j           
        new_num[nds_del] = new_num[node_couples[:,0]]        
        list_nd_new = [nd for nd in range(n_nodes) if not(nd in nds_del)]                                     
        self.elements = new_num[self.elements]
        for key in self.node_sets:
            self.node_sets[key] = new_num[self.node_sets[key]]         
        self.nodes = self.nodes[list_nd_new]  
        self.reset_interpolation()
    

    def remove_nodes(self, index_nodes):    
        """ 
        Remove some nodes and associated element.
        
        The total number and the id of nodes are modified.
        """
        nds_del = np.unique(index_nodes)
        n_nodes = self.n_nodes
        
        list_nd_new = [nd for nd in range(n_nodes) if not(nd in nds_del)]
        self.nodes = self.nodes[list_nd_new]  
                
        new_num = np.zeros(n_nodes,dtype = 'int')
        new_num[list_nd_new] = np.arange(len(list_nd_new))

        #delete element associated with deleted nodes
        deleted_elm = np.where(np.isin(self.elements, nds_del))[0]        
        
        mask = np.ones(len(self.elements) , dtype=bool)
        mask[deleted_elm] = False
        self.elements = self.elements[mask]
        
        self.elements = new_num[self.elements]

        for key in self.node_sets:
            self.node_sets[key] = new_num[self.node_sets[key]]
        
        self.reset_interpolation()
        return new_num
    
    
    def find_isolated_nodes(self):  
        """ 
        Return the nodes that are not associated with any element. 

        Return
        -------------        
        1D array containing the indexes of the non used nodes.
        If all elements are used, return an empty array.        
        """
        return np.setdiff1d(np.arange(self.n_nodes), np.unique(self.elements.flatten()))
    
    
    def remove_isolated_nodes(self):  
        """ 
        Remove the nodes that are not associated with any element. 
        
        The total number and the id of nodes are modified
        
        Return : NumberOfRemovedNodes int 
            the number of removed nodes (int).         
        """
        index_non_used_nodes = np.setdiff1d(np.arange(self.n_nodes), np.unique(self.elements.flatten()))
        self.remove_nodes(index_non_used_nodes)
        self.reset_interpolation()
        return len(index_non_used_nodes)
    
        
    def translate(self,disp):
        """
        Translate the mesh using the given displacement vector
        The disp vector should be on the form [u, v, w]
        """
        self.nodes = self.nodes + disp.T        
        
    
    def extract_elements(self,SetOfElementKey, name =""):
        """
        Return a new mesh from the set of elements defined by SetOfElementKey
        """
        new_SetOfElements = {}
        ListElm = self.element_sets[SetOfElementKey]
        for key in self.element_sets:
            new_SetOfElements[key] = np.array([el for el in self.element_sets[key] if el in ListElm])       
        
        subMesh = Mesh(self.nodes, self.elements[ListElm], self.elm_type, self.local_frame, name =name)                
        return subMesh    
       
    
    def nearest_node(self, X):
        """
        Return the index of the nearst node from the given position X
        
        Parameters
        ----------
        X : 1D np.ndarray
            Coordinates of a point. len(X) should be 3 in 3D or 2 in 3D.
            
        Returns
        -------
        The index of the nearest node to X 
        """
        return np.linalg.norm(self.nodes-X, axis=1).argmin()
    
        
    def find_nodes(self, selection_criterion, value=0, tol=1e-6):
        """
        Return a list of nodes from a given selection criterion

        Parameters
        ----------
       : str
            selection criterion used to select the returned nodes
            possibilities are: 
            - 'X': select nodes with a specified x coordinate value
            - 'Y': select nodes with a specified y coordinate value
            - 'Z': select nodes with a specified z coordinate value
            - 'XY' : select nodes with specified x and y coordinates values
            - 'XZ' : select nodes with specified x and z coordinates values
            - 'YZ' : select nodes with specified y and z coordinates values
            - 'Point': Distance to a point            

        value : scalar or list of scalar of numpy array
            - if selection_criterion in ['X', 'Y', 'Z'] value should be a scalar
            - if selection_criterion in ['XY', 'XZ', 'YZ'] value should be a list (or array) containing 2 scalar which are the coordinates in the given plane
            - if selection_criterion in ['point'] value should be a list (or array) containing 2 scalars (for 2D problem) or 3 scalars (for 3D problems) which are the coordinates of the point.
            
        tol : float
            Tolerance of the given criterion
            
        Returns
        -------
        List of node index
        """
        assert np.isscalar(tol), "tol should be a scalar"
        if selection_criterion in ['X','Y','Z']:
            assert np.isscalar(value), "value should be a scalar for selection_criterion = " + selection_criterion
            if selection_criterion == 'X':
                return np.where(np.abs(self.nodes[:,0]-value) < tol)[0]
            elif selection_criterion == 'Y':
                return np.where(np.abs(self.nodes[:,1]-value) < tol)[0]
            elif selection_criterion == 'Z':
                return np.where(np.abs(self.nodes[:,2]-value) < tol)[0]
        elif selection_criterion == 'XY':
            return np.where(np.linalg.norm(self.nodes[:,:2]-value, axis=1) < tol)[0]
        elif selection_criterion == 'XZ':
            return np.where(np.linalg.norm(self.nodes[:,::2]-value, axis=1) < tol)[0]
        elif selection_criterion == 'YZ':
            return np.where(np.linalg.norm(self.nodes[:,1:]-value, axis=1) < tol)[0]        
        elif selection_criterion.lower() == 'point':
            return np.where(np.linalg.norm(self.nodes-value, axis=1) < tol)[0]
        else:
            raise NameError("selection_criterion should be 'X','Y','Z' or 'point'")

    def copy(self):
        return Mesh(self.nodes.copy(),self.elements.copy(), self.elm_type)
    
    def to_pyvista(self):
        if USE_PYVISTA:            
            cell_type =  {'lin2':3,
                          'tri3':5,
                          'quad4':9,
                          'tet4':10,
                          'hex8':12,
                          'wed6':13,
                          'pyr5':14,
                          'lin3':21,
                          'tri6':22,
                          'quad8':23,           
                          'tet10':24,
                          'hex20':25
                          }.get(self.elm_type, None)
            if cell_type is None: raise NameError('Element Type '+ str(self.elm_type) + ' not available in pyvista')

            elm = np.empty((self.elements.shape[0], self.elements.shape[1]+1), dtype=int)
            elm[:,0] = self.elements.shape[1]
            elm[:,1:] = self.elements
            crd = self.nodes
            
            if crd.shape[1]<3:
                crd = np.hstack((crd, np.zeros((crd.shape[0], 3-crd.shape[1]))))
                
            return pv.UnstructuredGrid(elm.ravel(),  np.full(len(elm),cell_type, dtype=int), crd)
        else:
            raise NameError('Pyvista not installed.')
            

    def save(self, filename, binary=True):
        """        
        Save the mesh object to file. This function use the save function of the pyvista UnstructuredGrid object
        
        Parameters
        ----------
        filename : str
            Filename of output file including the path. Writer type is inferred from the extension of the filename. If no extension is set, 'vtk' is assumed. 
        binary : bool, optional
            If True, write as binary. Otherwise, write as ASCII.
        """
        extension = splitext(filename)[1]
        if extension == '': 
            filename = filename + '.vtk'

        self.to_pyvista().save(filename, binary)
    
    
    def plot(self,show_edges=True):
        """Simple plot function using pyvista"""
        if USE_PYVISTA:            
            self.to_pyvista().plot(show_edges=show_edges)
        else:
            raise NameError('Pyvista not installed.')


    def init_interpolation(self,n_elm_gp = None):
        if n_elm_gp is None: n_elm_gp = get_DefaultNbPG(self.elm_type)
        
        n_nodes = self.n_nodes        
        n_elements = self.n_elements                                         
        n_elm_nd = self.n_elm_nodes #number of nodes associated to each element
        
        #-------------------------------------------------------------------
        #Initialise the geometrical interpolation
        #-------------------------------------------------------------------   
        # elm_interpol = get_element(self.elm_type)(n_elm_gp, mesh=self) #initialise element interpolation
        elm_interpol = get_element(self.elm_type)(n_elm_gp) #initialise element interpolation
        n_elm_dof = len(elm_interpol.xi_nd) #number of dof used in the geometrical interpolation for each element - for isoparametric elements n_elm_dof = n_elm_nd

        elm_geom = self.elements[:,:n_elm_dof] #element table restrictied to geometrical dof

        n_elm_with_nd = np.bincount(elm_geom.reshape(-1)) #len(n_elm_with_nd) = n_nodes #number of elements connected to each node        
                
        #-------------------------------------------------------------------
        # Compute the array containing row and col indices used to assemble the sparse matrices
        #-------------------------------------------------------------------          
        row = np.empty((n_elements, n_elm_gp, n_elm_nd)) ; col = np.empty((n_elements, n_elm_gp, n_elm_nd))                
        row[:] = np.arange(n_elements).reshape((-1,1,1)) + np.arange(n_elm_gp).reshape(1,-1,1)*n_elements 
        col[:] = self.elements.reshape((n_elements,1,n_elm_nd))
        #row_geom/col_geom: row and col indices using only the dof used in the geometrical interpolation (col = col_geom if geometrical and variable interpolation are the same)
        row_geom = np.reshape(row[...,:n_elm_dof], -1) ; col_geom = np.reshape(col[...,:n_elm_dof], -1)
        
        #-------------------------------------------------------------------
        # Assemble the matrix that compute the node values from pg based on the geometrical shape functions (no angular dof for ex)    
        #-------------------------------------------------------------------                                
        PGtoNode = np.linalg.pinv(elm_interpol.ShapeFunctionPG) #pseudo-inverse of NodeToPG
        dataPGtoNode = PGtoNode.T.reshape((1,n_elm_gp,n_elm_dof))/n_elm_with_nd[elm_geom].reshape((n_elements,1,n_elm_dof)) #shape = (n_elements, n_elm_gp, n_elm_nd)   
        self._saved_gausspoint2node_mat[n_elm_gp] = sparse.coo_matrix((dataPGtoNode.reshape(-1),(col_geom,row_geom)), shape=(n_nodes,n_elements*n_elm_gp) ).tocsr() #matrix to compute the node values from pg using the geometrical shape functions 

        #-------------------------------------------------------------------
        # Assemble the matrix that compute the pg values from nodes using the geometrical shape functions (no angular dof for ex)    
        #-------------------------------------------------------------------             
        dataNodeToPG = np.empty((n_elements, n_elm_gp, n_elm_dof))
        dataNodeToPG[:] = elm_interpol.ShapeFunctionPG.reshape((1,n_elm_gp,n_elm_dof)) 
        self._saved_node2gausspoint_mat[n_elm_gp] = sparse.coo_matrix((np.reshape(dataNodeToPG,-1),(row_geom,col_geom)), shape=(n_elements*n_elm_gp, n_nodes) ).tocsr() #matrix to compute the pg values from nodes using the geometrical shape functions (no angular dof)
        
        # save some data related to interpolation for potentiel future use
        self._elm_interpolation[n_elm_gp] = elm_interpol       
        self._sparse_structure[n_elm_gp] = (row.reshape(-1), col.reshape(-1))
        self._elements_geom = elm_geom #dont depend on n_elm_gp
        
    
    def _compute_gaussian_quadrature_mat(self, n_elm_gp = None):        
        if n_elm_gp is None: n_elm_gp = get_DefaultNbPG(self.elm_type)
        if n_elm_gp not in self._saved_gaussian_quadrature_mat:
            self.init_interpolation(n_elm_gp)        
                
        elm_interpol = self._elm_interpolation[n_elm_gp]
        vec_xi = elm_interpol.xi_pg #coordinate of points of gauss in element coordinate (xi)        
        elm_interpol.ComputeJacobianMatrix(self.nodes[self._elements_geom], vec_xi, self.local_frame) #compute elm_interpol.JacobianMatrix, elm_interpol.detJ and elm_interpol.inverseJacobian

        #-------------------------------------------------------------------
        # Compute the diag matrix used for the gaussian quadrature
        #-------------------------------------------------------------------  
        gaussianQuadrature = (elm_interpol.detJ * elm_interpol.w_pg).T.reshape(-1) 
        self._saved_gaussian_quadrature_mat[n_elm_gp] = sparse.diags(gaussianQuadrature, 0, format='csr') #matrix to get the gaussian quadrature (integration over each element)        

    
    def _get_gausspoint2node_mat(self, n_elm_gp=None): 
        if n_elm_gp is None: n_elm_gp = get_DefaultNbPG(self.elm_type)
        if not(n_elm_gp in self._saved_gausspoint2node_mat):
            self.init_interpolation(n_elm_gp)        
        return self._saved_gausspoint2node_mat[n_elm_gp]
    
    
    def _get_node2gausspoint_mat(self, n_elm_gp=None): 
        if n_elm_gp is None: n_elm_gp = get_DefaultNbPG(self.elm_type)   
        if not(n_elm_gp in self._saved_node2gausspoint_mat):
            self.init_interpolation(n_elm_gp)                
        return self._saved_node2gausspoint_mat[n_elm_gp]


    def _get_gaussian_quadrature_mat(self, n_elm_gp=None): 
        if n_elm_gp is None: n_elm_gp = get_DefaultNbPG(self.elm_type)
        if not(n_elm_gp in self._saved_gaussian_quadrature_mat):
            self._compute_gaussian_quadrature_mat(n_elm_gp)
        return self._saved_gaussian_quadrature_mat[n_elm_gp]


    def determine_data_type(self, data, n_elm_gp=None):                       
        if n_elm_gp is None: n_elm_gp = get_DefaultNbPG(self.elm_type)
 
        test = 0
        if data.shape[-1] == self.n_nodes: 
            data_type = 'Node' #fonction définie aux noeuds   
            test+=1               
        if data.shape[-1]  == self.n_elements: 
            data_type = 'Element' #fonction définie aux éléments
            test += 1
        if data.shape[-1] == n_elm_gp*self.n_elements:
            data_type = 'GaussPoint'
            test += 1
        assert test, "Error: data doesn't match with the number of nodes, number of elements or number of gauss points."
        if test>1: "Warning: kind of data is confusing. " + data_type +" values choosen."
        return data_type  
    

    def data_to_gausspoint(self, data, n_elm_gp=None):         
        """
        Convert a field array (node values or element values) to gauss points.
        data: array containing the field (node or element values)
        return: array containing the gausspoint field 
        The shape of the array is tested.
        """               
        if n_elm_gp is None: n_elm_gp = get_DefaultNbPG(self.elm_type)           
        data_type = self.determine_data_type(data, n_elm_gp)       

        if data_type == 'Node': 
            return self._get_node2gausspoint_mat(n_elm_gp) @ data
        if data_type == 'Element':
            if len(np.shape(data)) == 1: return np.tile(data.copy(),n_elm_gp) #why use copy here ?
            else: return np.tile(data.copy(),[n_elm_gp,1])            
        return data #in case data contains already PG values


    def convert_data(self, data, convert_from=None, convert_to='GaussPoint', n_elm_gp=None):
       if np.isscalar(data): return data

       if n_elm_gp is None: n_elm_gp = get_DefaultNbPG(self.elm_type)
       
       if convert_from is None: convert_from = self.determine_data_type(data, n_elm_gp)
           
       assert (convert_from in ['Node','GaussPoint','Element']) and (convert_to in ['Node','GaussPoint','Element']), "only possible to convert 'Node', 'Element' and 'GaussPoint' values"
       
       if convert_from == convert_to: return data    
       
       data = data.T
       if convert_from == 'Node': 
           data = self._get_node2gausspoint_mat(n_elm_gp) @ data
       elif convert_from == 'Element':             
           if len(np.shape(data)) == 1: data = np.tile(data.copy(),n_elm_gp)
           else: data = np.tile(data.copy(),[n_elm_gp,1])
           
       # from here data should be defined at 'PG'
       if convert_to == 'Node': 
           return (self._get_gausspoint2node_mat(n_elm_gp) @ data).T
       elif convert_to == 'Element': 
           return (np.sum(np.split(data, n_elm_gp),axis=0) / n_elm_gp).T
       else: return data.T


    def integrate_field(self, field, type_field = None, n_elm_gp = None):        
        assert type_field in ['Node','GaussPoint','Element', None], "TypeField should be 'Node', 'Element' or 'GaussPoint' values"        
        if n_elm_gp is None: 
            if type_field == 'GaussPoint': n_elm_gp = len(field)//self.n_elements
            else: n_elm_gp = get_DefaultNbPG(self.elm_type)               
                            
        return sum(self._get_gaussian_quadrature_mat(n_elm_gp) @ self.data_to_gausspoint(field,n_elm_gp))


    def reset_interpolation(self):
        """Remove all the saved data related to field interpolation. 
        This method should be used when modifying the element table or when removing or adding nodes.
        """
        self._saved_gausspoint2node_mat = {}
        self._saved_node2gausspoint_mat = {}
        self._saved_gaussian_quadrature_mat = {}
        self._elm_interpolation = {}
        self._sparse_structure = {}

    @property
    def n_nodes(self):
        """
        Total number of nodes in the Mesh        
        """
        return len(self.nodes)
    
    @property
    def n_elements(self):
        """
        Total number of elements in the Mesh        
        """
        return len(self.elements)


    @property
    def n_elm_nodes(self):
        """
        Number of nodes associated to each element 
        """
        return self.elements.shape[1]
    

    @property
    def ndim(self):
        """
        Dimension of the mesh       
        """
        return self.nodes.shape[1]
    
    @property
    def bounding_box(self):
        return BoundingBox(self)


    # def GetCoordinatename(self):
    #     return self.crd_name
    
    # def SetCoordinatename(self,ListCoordinatename):        
    #     self.crd_name = ListCoordinatename
    
    
    
    
    
    #
    # To be developed later
    #
    # def InititalizeLocalFrame(self):
    #     """
    #     Following the mesh geometry and the element shape, a local frame is initialized on each nodes
    #     """
#        elmRef = self.elm_type(1)        
#        rep_loc = np.zeros((self.__n_nodes,np.shape(self.nodes)[1],np.shape(self.nodes)[1]))   
#        for e in self.elements:
#            if self.__localBasis == None: rep_loc[e] += elmRef.getRepLoc(self.nodes[e], elmRef.xi_nd)
#            else: rep_loc[e] += elmRef.getRepLoc(self.nodes[e], elmRef.xi_nd, self.__rep_loc[e]) 
#
#        rep_loc = np.array([rep_loc[nd]/len(np.where(self.elements==nd)[0]) for nd in range(self.__n_nodes)])
#        rep_loc = np.array([ [r/linalg.norm(r) for r in rep] for rep in rep_loc])
#        self__.localBasis = rep_loc


    #
    # development
    #
#     def GetElementLocalFrame(self): #Précalcul des opérateurs dérivés suivant toutes les directions (optimise les calculs en minimisant le nombre de boucle)               
#         #initialisation
#         elmRef = eval(self.elm_type)(1) #only 1 gauss point for returning one local Frame per element
               
#         elm = self.elements
#         crd = self.nodes
        
# #        elmGeom.ComputeJacobianMatrix(crd[elm_geom], vec_xi, localFrame) #elmRef.JacobianMatrix, elmRef.detJ, elmRef.inverseJacobian
#         return elmRef.GetLocalFrame(crd[elm], elmRef.xi_pg, self.local_frame) #array of shape (n_elements, n_elm_gp, nb of vectors in basis = dim, dim)


    # def bounding_box(self, return_center = False):
    #     """
    #     Return the cordinate of the left/bottom/behind (Xmin) and the right/top/front (Xmax) corners

    #     Parameters
    #     ----------
    #     return_center : bool, optional (default = False)
    #         if return_center = True, also return the coordinate of the center of the bounding box

    #     Returns
    #     -------
    #     - Xmin: numpy array of float containing the coordinates of the left/bottom/behind corner
    #     - Xmax: numpy array of float containing the coordinates of the right/top/front corner
    #     - Xcenter (if return_center = True): numpy array of float containing the coordinates of the center
    #     """
    #     Xmin = self.nodes.min(axis=0)
    #     Xmax = self.nodes.max(axis=0)
    #     if return_center == False: 
    #         return Xmin, Xmax
    #     else: 
    #         return Xmin, Xmax, (Xmin+Xmax)/2
    
    
    # def GetSetOfNodes(self,SetOfId):
    #     """
    #     Return the set of nodes whose name is SetOfId
        
    #     Parameters
    #     ----------
    #     SetOfId : str
    #         name of the set of nodes

    #     Returns
    #     -------
    #     list or 1D numpy array containing node indexes
    #     """
    #     return self.node_sets[SetOfId]
        
    # def GetSetOfElements(self,SetOfId):
    #     """
    #     Return the set of elements whose name is SetOfId
        
    #     Parameters
    #     ----------
    #     SetOfId : str
    #         name of the set of elements

    #     Returns
    #     -------
    #     list or 1D numpy array containing element indexes
    #     """
    #     return self.element_sets[SetOfId]

    # def RemoveSetOfNodes(self,SetOfId):
    #     """
    #     Remove the set of nodes whose name is SetOfId from the Mesh
        
    #     Parameters
    #     ----------
    #     SetOfId : str
    #         name of the set of nodes
    #     """
    #     del self.node_sets[SetOfId]
        
    # def RemoveSetOfElements(self,SetOfId):
    #     """
    #     Remove the set of elements whose name is SetOfId from the Mesh
        
    #     Parameters
    #     ----------
    #     SetOfId : str
    #         name of the set of elements
    #     """
    #     del self.element_sets[SetOfId]

    # def ListSetOfNodes(self):
    #     """
    #     Return a list containing the name (str) of all set of nodes defined in the Mesh.
    #     """
    #     return [key for key in self.node_sets]

    # def ListSetOfElements(self):    
    #     """
    #     Return a list containing the name (str) of all set of elements defined in the Mesh.
    #     """
    #     return [key for key in self.element_sets]
                                
    # def GetElementShape(self):
    #     """
    #     Return the element shape (ie, type of element) of the Mesh. 
        
    #     Parameters
    #     ----------
    #     SetOfId : str
    #         name of the set of nodes
            
    #     The element shape if defined as a str according the the list of available element shape.
    #     For instance, the element shape may be: 'lin2', 'tri3', 'tri6', 'quad4', 'quad8', 'quad9', 'hex8', ...
        
    #     Remark
    #     ----------
    #     The element shape associated to the Mesh is only used for geometrical interpolation and may be different from the one used in the Assembly object.
    #     """
    #     return self.elm_type
    
    # def SetElementShape(self, value):
    #     """
    #     Change the element shape (ie, type of element) of the Mesh.
        
        
    #     The element shape if defined as a str according the the list of available element shape.
    #     For instance, the element shape may be: 'lin2', 'tri3', 'tri6', 'quad4', 'quad8', 'quad9', 'hex8', ...
        
    #     Remark
    #     ----------
    #     The element shape associated to the Mesh is only used for geometrical interpolation and may be different from the one used in the Assembly object.
    #     """
    #     self.elm_type = value
    

class BoundingBox(list):
    def __init__(self, m):
        if isinstance(m, Mesh):
            m = [m.nodes.min(axis=0), m.nodes.max(axis=0)]
        elif isinstance(m,list):
            if len(m) != 2:                
                raise NameError('list lenght for BoundingBox object must be 2')        
        else:
            raise NameError('Can only create BoundingBox from Mesh object or list of 2 points')
                
        list.__init__(self,m)
    
    @property
    def xmin(self):
        """
        return xmin
        """
        return self[0][0]

    @property
    def xmax(self):
        """
        return xmax
        """
        return self[1][0]

    @property
    def ymin(self):
        """
        return ymin
        """
        return self[0][1]

    @property
    def ymax(self):
        """
        return ymax
        """
        return self[1][1]
    
    @property
    def zmin(self):
        """
        return zmin
        """
        return self[0][2]

    @property
    def zmax(self):
        """
        return zmax
        """
        return self[1][2]

    @property
    def center(self):
        """
        return the center of the bounding box
        """
        return (self[0] + self[1])/2
    
    
    @property
    def volume(self):
        """
        return the volume of the bounding box
        """
        return (self[1]-self[0]).prod()               

if __name__=="__main__":
    import scipy as sp
    a = Mesh(sp.array([[0,0,0],[1,0,0]]), sp.array([[0,1]]),'lin2')
    print(a.nodes)



