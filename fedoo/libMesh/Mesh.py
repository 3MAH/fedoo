#import scipy as sp
import numpy as np

from fedoo.libUtil.ModelingSpace import ModelingSpace
# from fedoo.libUtil.Coordinate import Coordinate
from fedoo.libMesh.MeshBase import MeshBase
from fedoo.libElement import *

def Create(NodeCoordinates, ElementTable, ElementShape, LocalFrame=None, ID = ""):        
    return Mesh(NodeCoordinates, ElementTable, ElementShape, LocalFrame, ID)

class Mesh(MeshBase):
    def __init__(self, NodeCoordinates, ElementTable=None, ElementShape=None, LocalFrame=None, ID = ""):
        MeshBase.__init__(self, ID)
        self.__NodeCoordinates = NodeCoordinates #node coordinates            
        self.__ElementTable = ElementTable #element
        self.__ElementShape = ElementShape
        self.__SetOfNodes = {} #node on the boundary for instance
        self.__SetOfElements = {}
        self.__LocalFrame = LocalFrame #contient le repere locale (3 vecteurs unitaires) en chaque noeud. Vaut 0 si pas de rep locaux definis

        n = ModelingSpace.GetDimension()
        N = self.__NodeCoordinates.shape[0]
                
        if self.__NodeCoordinates.shape[1] == 1: self.__CoordinateID = ('X')
        elif self.__NodeCoordinates.shape[1] == 2: self.__CoordinateID = ('X', 'Y')
        elif n == '2Dplane' or n == '2Dstress': self.__CoordinateID = ('X', 'Y')
        else: self.__CoordinateID = ('X', 'Y', 'Z')
        
        if n == '3D' and self.__NodeCoordinates.shape[1] == 2:
            self.__NodeCoordinates = np.c_[self.__NodeCoordinates, np.zeros(N)]
            if LocalFrame != None:
                LocalFrameTemp = np.zeros((N,3,3))
                LocalFrameTemp[:,:2,:2] = self.__LocalFrame
                LocalFrameTemp[:,2,2]   = 1
                self.__LocalFrame = LocalFrameTemp
   
    def AddSetOfNodes(self,NodeIndexes,SetOfId):
        """        
        Add a set of nodes to the Mesh
        
        Parameters
        ----------
        NodeIndexes : list or 1D numpy.array
            A list of node indexes
        SetOfId : str
            ID of the set of nodes            
        """
        self.__SetOfNodes[SetOfId] = NodeIndexes
        
    def AddSetOfElements(self,ElementIndexes,SetOfId):
        """        
        Add a set of elements to the Mesh
        
        Parameters
        ----------
        ElementIndexes : list or 1D numpy.array
            A list of node indexes
        SetOfId : str
            ID of the set of nodes            
        """
        self.__SetOfElements[SetOfId] = ElementIndexes

    def GetSetOfNodes(self,SetOfId):
        """
        Return the set of nodes whose ID is SetOfId
        
        Parameters
        ----------
        SetOfId : str
            ID of the set of nodes

        Returns
        -------
        list or 1D numpy array containing node indexes
        """
        return self.__SetOfNodes[SetOfId]
        
    def GetSetOfElements(self,SetOfId):
        """
        Return the set of elements whose ID is SetOfId
        
        Parameters
        ----------
        SetOfId : str
            ID of the set of elements

        Returns
        -------
        list or 1D numpy array containing element indexes
        """
        return self.__SetOfElements[SetOfId]

    def RemoveSetOfNodes(self,SetOfId):
        """
        Remove the set of nodes whose ID is SetOfId from the Mesh
        
        Parameters
        ----------
        SetOfId : str
            ID of the set of nodes
        """
        del self.__SetOfNodes[SetOfId]
        
    def RemoveSetOfElements(self,SetOfId):
        """
        Remove the set of elements whose ID is SetOfId from the Mesh
        
        Parameters
        ----------
        SetOfId : str
            ID of the set of elements
        """
        del self.__SetOfElements[SetOfId]

    def ListSetOfNodes(self):
        """
        Return a list containing the ID (str) of all set of nodes defined in the Mesh.
        """
        return [key for key in self.__SetOfNodes]

    def ListSetOfElements(self):    
        """
        Return a list containing the ID (str) of all set of elements defined in the Mesh.
        """
        return [key for key in self.__SetOfElements]
                            
    def GetNumberOfNodes(self):
        """
        Return the total number of nodes in the Mesh        
        """
        return len(self.__NodeCoordinates)
    
    def GetNumberOfElements(self):
        """
        Return the total number of elements in the Mesh        
        """
        return len(self.__ElementTable)
    
    def GetElementShape(self):
        """
        Return the element shape (ie, type of element) of the Mesh. 
        
        Parameters
        ----------
        SetOfId : str
            ID of the set of nodes
            
        The element shape if defined as a str according the the list of available element shape.
        For instance, the element shape may be: 'lin2', 'tri3', 'tri6', 'quad4', 'quad8', 'quad9', 'hex8', ...
        
        Remark
        ----------
        The element shape associated to the Mesh is only used for geometrical interpolation and may be different from the one used in the Assembly object.
        """
        return self.__ElementShape
    
    def SetElementShape(self, value):
        """
        Change the element shape (ie, type of element) of the Mesh.
        
        
        The element shape if defined as a str according the the list of available element shape.
        For instance, the element shape may be: 'lin2', 'tri3', 'tri6', 'quad4', 'quad8', 'quad9', 'hex8', ...
        
        Remark
        ----------
        The element shape associated to the Mesh is only used for geometrical interpolation and may be different from the one used in the Assembly object.
        """
        self.__ElementShape = value
    
    def GetNodeCoordinates(self):
        return self.__NodeCoordinates

    def SetNodeCoordinates(self,a):
        self.__NodeCoordinates = a
        
    def AddNodes(self, Coordinates = None, NumberOfNewNodes = None):
        """
        Add some nodes to the node list.
        
        The new nodes are not liked to any element.

        Parameters
        ----------
        Coordinates : np.ndarray
            The coordinates of the new nodes.             
        NumberOfNewNodes : int
            Number of new nodes
            By default, the value is deduced from she shape of Coordinates
            NumberOfNewNodes is used only for creating several nodes with the same coordinates
        """
        NbNd_old = self.GetNumberOfNodes()
        if NumberOfNewNodes is None and Coordinates is None: NumberOfNewNodes = 1
        if NumberOfNewNodes is None:
            self.__NodeCoordinates = np.vstack((self.__NodeCoordinates, Coordinates))
        else:
            if Coordinates is None:
                self.__NodeCoordinates = np.vstack((self.__NodeCoordinates, 
                    np.zeros([NumberOfNewNodes,self.__NodeCoordinates.shape[1]])))
            else:
                self.__NodeCoordinates = np.vstack((self.__NodeCoordinates, 
                    np.tile(Coordinates,(NumberOfNewNodes,1))))

        return np.arange(NbNd_old,self.GetNumberOfNodes())

    def AddInternalNodes(self, numberOfInternalNodes):
        newNodes = self.AddNodes(NumberOfNewNodes=self.GetNumberOfElements()*numberOfInternalNodes)
        self.__ElementTable = np.c_[self.__ElementTable, newNodes]
        
    def GetElementTable(self):
        return self.__ElementTable
    
    def GetLocalFrame(self):
        return self.__LocalFrame
    
    def GetCoordinateID(self):
        return self.__CoordinateID
    
    def SetCoordinateID(self,ListCoordinateID):        
        self.__CoordinateID = ListCoordinateID
    
    # warning , this method must be static
    @staticmethod
    def Stack(Mesh1,Mesh2, ID = ""):
        """
        *Static method* - Make the spatial stack of two mesh objects which have the same element shape.        
        
        Same as the function Stack in the Mesh module.
        
        Return 
        ---------
        Mesh object with is the spacial stack of Mesh1 and Mesh2
        """
        if isinstance(Mesh1, str): Mesh1 = Mesh.GetAll()[Mesh1]
        if isinstance(Mesh2, str): Mesh2 = Mesh.GetAll()[Mesh2]
        
        if Mesh1.GetElementShape() != Mesh2.GetElementShape():    
            raise NameError("Can only stack meshes with the same element shape")
            
        Nnd = Mesh1.GetNumberOfNodes()
        Nel = Mesh1.GetNumberOfElements()
         
        new_crd = np.r_[Mesh1.__NodeCoordinates , Mesh2.__NodeCoordinates]
        new_elm = np.r_[Mesh1.__ElementTable , Mesh2.__ElementTable + Nnd]
        
        new_ndSets = dict(Mesh1.__SetOfNodes)
        for key in Mesh2.__SetOfNodes:
            if key in Mesh1.__SetOfNodes:
                new_ndSets[key] = np.r_[Mesh1.__SetOfNodes[key], np.array(Mesh2.__SetOfNodes[key]) + Nnd]
            else:
                new_ndSets[key] = np.array(Mesh2.__SetOfNodes[key]) + Nnd                                  
        
        new_elSets = dict(Mesh1.__SetOfElements)
        for key in Mesh2.__SetOfElements:
            if key in Mesh1.__SetOfElements:
                new_elSets[key] = np.r_[Mesh1.__SetOfElements[key], np.array(Mesh2.__SetOfElements[key]) + Nel]
            else:
                new_elSets[key] = np.array(Mesh2.__SetOfElements[key]) + Nel    
                   
        Mesh3 = Mesh(new_crd, new_elm, Mesh1.__ElementShape, ID = ID)
        Mesh3.__SetOfNodes = new_ndSets
        Mesh3.__SetOfElements = new_elSets
        return Mesh3

    def FindCoincidentNodes(self,tol=1e-8):
        """ 
        Find some nodes with the same position considering a tolerance given by the argument tol. 
        return an array of shape (numberOfCoincidentNodes, 2) where each line is a pair of nodes that are at the same position.
        These pairs of nodes can be merged using :
            meshObject.MergeNodes(meshObject.findCoincidentNodes()) 
            
        where meshObject is the Mesh object containing merged coincidentNodes.
        """
        Nnd = self.GetNumberOfNodes()
        decimal_round = int(-np.log10(tol)-1)
        crd = self.__NodeCoordinates.round(decimal_round) #round coordinates to match tolerance
        ind_sorted   = np.lexsort((crd[:  ,2], crd[:  ,1], crd[:  ,0]))

        ind_coincident = np.where(np.linalg.norm(crd[ind_sorted[:-1]]-crd[ind_sorted[1:]], axis = 1) == 0)[0] #indices of the first coincident nodes
        return np.array([ind_sorted[ind_coincident], ind_sorted[ind_coincident+1]]).T
 
    
    def MergeNodes(self,IndexCouples):
        """ 
        Merge some nodes 
        The total number and the id of nodes are modified
        """
        Nnd = self.GetNumberOfNodes()
        nds_del = IndexCouples[:,1] #list des noeuds a supprimer
        nds_kept = IndexCouples[:,0] #list des noeuds a conserver
         
        unique_nodes, ordre = np.unique(nds_del, return_index=True)
        assert len(unique_nodes) == len(nds_del), "A node can't be deleted 2 times"
        # ordre = np.argsort(nds_del)
        j=0 
        new_num = np.zeros(Nnd,dtype = 'int')
        for nd in range(Nnd):    
            if j<len(nds_del) and nd==nds_del[ordre[j]]: 
                #test if some nodes are equal to deleted node among the kept nodes. If required update the kept nodes values
                indDelNodes = np.where(nds_kept == nds_del[ordre[j]])[0] #index of nodes to kept that are deleted and need to be updated to their new values
                nds_kept[indDelNodes] = nds_kept[ordre[j]]
                j+=1
            else: new_num[nd] = nd-j           
        new_num[nds_del] = new_num[IndexCouples[:,0]]        
        list_nd_new = [nd for nd in range(Nnd) if not(nd in nds_del)]                                     
        self.__ElementTable = new_num[self.__ElementTable]
        for key in self.__SetOfNodes:
            self.__SetOfNodes[key] = new_num[self.__SetOfNodes[key]]         
        self.__NodeCoordinates = self.__NodeCoordinates[list_nd_new]  
    

    def RemoveNodes(self, index_nodes):    
        """ 
        Remove some nodes and associated element.
        
        The total number and the id of nodes are modified.
        """
        nds_del = np.unique(index_nodes)
        Nnd = self.GetNumberOfNodes()
        
        list_nd_new = [nd for nd in range(Nnd) if not(nd in nds_del)]
        self.__NodeCoordinates = self.__NodeCoordinates[list_nd_new]  
                
        new_num = np.zeros(Nnd,dtype = 'int')
        new_num[list_nd_new] = np.arange(len(list_nd_new))

        #delete element associated with deleted nodes
        deleted_elm = np.where(np.isin(self.__ElementTable, nds_del))[0]        
        
        Mask = np.ones(len(self.__ElementTable) , dtype=bool)
        Mask[deleted_elm] = False
        self.__ElementTable = self.__ElementTable[Mask]
        
        self.__ElementTable = new_num[self.__ElementTable]

        for key in self.__SetOfNodes:
            self.__SetOfNodes[key] = new_num[self.__SetOfNodes[key]]
            
        return new_num
    
    def FindNonUsedNodes(self):  
        """ 
        Return the nodes that are not associated with any element. 

        Return
        -------------        
        1D array containing the indexes of the non used nodes.
        If all elements are used, return an empty array.        
        """
        return np.setdiff1d(np.arange(self.GetNumberOfNodes()), np.unique(self.__ElementTable.flatten()))
    
    def RemoveNonUsedNodes(self):  
        """ 
        Remove the nodes that are not associated with any element. 
        
        The total number and the id of nodes are modified
        
        Return : NumberOfRemovedNodes int 
            the number of removed nodes (int).         
        """
        index_non_used_nodes = np.setdiff1d(np.arange(self.GetNumberOfNodes()), np.unique(self.__ElementTable.flatten()))
        self.RemoveNodes(index_non_used_nodes)
        return len(index_non_used_nodes)
        
    def Translate(self,Vector):
        """
        Translate the mesh along a given vector        
        """
        self.__NodeCoordinates = self.__NodeCoordinates + Vector        
    
    def ExtractSetOfElements(self,SetOfElementKey, ID=""):
        """
        Return a new mesh from the set of elements defined by SetOfElementKey
        """
        new_SetOfElements = {}
        ListElm = self.__SetOfElements[SetOfElementKey]
        for key in self.__SetOfElements:
            new_SetOfElements[key] = np.array([el for el in self.__SetOfElements[key] if el in ListElm])       
        
        subMesh = Mesh(self.__NodeCoordinates, self.__ElementTable[ListElm], self.__ElementShape, self.__LocalFrame, ID=ID)                
        return subMesh    
       
    #
    # To be developed later
    #
    def InititalizeLocalFrame(self):
        """
        Following the mesh geometry and the element shape, a local frame is initialized on each nodes
        """
#        elmRef = self.__ElementShape(1)        
#        rep_loc = np.zeros((self.__Nnd,np.shape(self.__NodeCoordinates)[1],np.shape(self.__NodeCoordinates)[1]))   
#        for e in self.__ElementTable:
#            if self.__localBasis == None: rep_loc[e] += elmRef.getRepLoc(self.__NodeCoordinates[e], elmRef.xi_nd)
#            else: rep_loc[e] += elmRef.getRepLoc(self.__NodeCoordinates[e], elmRef.xi_nd, self.__rep_loc[e]) 
#
#        rep_loc = np.array([rep_loc[nd]/len(np.where(self.__ElementTable==nd)[0]) for nd in range(self.__Nnd)])
#        rep_loc = np.array([ [r/linalg.norm(r) for r in rep] for rep in rep_loc])
#        self__.localBasis = rep_loc


    #
    # development
    #
    def GetElementLocalFrame(self): #Précalcul des opérateurs dérivés suivant toutes les directions (optimise les calculs en minimisant le nombre de boucle)               
        #initialisation
        elmRef = eval(self.__ElementShape)(1) #only 1 gauss point for returning one local Frame per element
               
        elm = self.__ElementTable
        crd = self.__NodeCoordinates
        
#        elmGeom.ComputeJacobianMatrix(crd[elm_geom], vec_xi, localFrame) #elmRef.JacobianMatrix, elmRef.detJ, elmRef.inverseJacobian
        return elmRef.GetLocalFrame(crd[elm], elmRef.xi_pg, self.__LocalFrame) #array of shape (Nel, nb_pg, nb of vectors in basis = dim, dim)



def GetAll():
    return Mesh.GetAll()

def Stack(Mesh1,Mesh2, ID = ""):
        """
        Make the spatial stack of two mesh objects which have the same element shape.        
        This function doesn't merge coindicent Nodes. 
        For that purpose, use the Mesh methods 'FindCoincidentNodes' and 'MergeNodes'
        on the resulting Mesh. 
        
        Return 
        ---------
        Mesh object with the spacial stack of Mesh1 and Mesh2
        """
        return Mesh.Stack(Mesh1,Mesh2,ID)

if __name__=="__main__":
    import scipy as sp
    a = Mesh(sp.array([[0,0,0],[1,0,0]]), sp.array([[0,1]]),'lin2')
    print(a.GetNodeCoordinates())

