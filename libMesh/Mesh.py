#import scipy as sp
import numpy as np

from fedoo.libUtil.Dimension import ProblemDimension
from fedoo.libUtil.Coordinate import Coordinate
from fedoo.libMesh.MeshBase import *
from fedoo.libElement import *

def Create(NodeCoordinates, ElementTable, ElementShape, LocalFrame=None, ID = ""):        
    return Mesh(NodeCoordinates, ElementTable, ElementShape, LocalFrame, ID)

class Mesh(MeshBase):
    def __init__(self, NodeCoordinates, ElementTable, ElementShape, LocalFrame=None, ID = ""):
        MeshBase.__init__(self, ID)
        self.__NodeCoordinates = NodeCoordinates #node coordinates            
        self.__ElementTable = ElementTable #element
        self.__ElementShape = ElementShape
        self.__SetOfNodes = {} #node on the boundary for instance
        self.__SetOfElements = {}
        self.__LocalFrame = LocalFrame #contient le repere locale (3 vecteurs unitaires) en chaque noeud. Vaut 0 si pas de rep locaux definis

        n = ProblemDimension.Get()
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
        self.__SetOfNodes[SetOfId] = NodeIndexes
        
    def AddSetOfElements(self,ElementIndexes,SetOfId):
        self.__SetOfElements[SetOfId] = ElementIndexes

    def GetSetOfNodes(self,SetOfId):
        return self.__SetOfNodes[SetOfId]
        
    def GetSetOfElements(self,SetOfId):
        return self.__SetOfElements[SetOfId]

    def RemoveSetOfNodes(self,SetOfId):
        del self.__SetOfNodes[SetOfId]
        
    def RemoveSetOfElements(self,SetOfId):
        del self.__SetOfElements[SetOfId]

    def ListSetOfNodes(self):
        return [key for key in self.__SetOfNodes]

    def ListSetOfElements(self):    
        return [key for key in self.__SetOfElements]
                            
    def GetNumberOfNodes(self):
        return len(self.__NodeCoordinates)
    
    def GetNumberOfElements(self):
        return len(self.__ElementTable)
    
    def GetElementShape(self):
        return self.__ElementShape
    
    def SetElementShape(self, value):
        self.__ElementShape = value
    
    def GetNodeCoordinates(self):
        return self.__NodeCoordinates

    def SetNodeCoordinates(self,a):
        self.__NodeCoordinates = a
        
    def AddNodes(self, Coordinates = None, NumberOfNewNodes = None):
        """
        Add some nodes to the node list 
        The new nodes are not liked to any element

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
        TODO make the spatial stack of two mesh objects which have the same element shape
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
    
    def MergeNodes(self,IndexCouples):
        """ 
        Merge some nodes 
        The total number and the id of nodes are modified
        """
        Nnd = self.GetNumberOfNodes()
        nds_del = IndexCouples[:,1] #list des noeuds a supprimer
        ordre = np.argsort(nds_del)
        j=0 
        new_num = np.zeros(Nnd,dtype = 'int')
        for nd in range(Nnd):            
            if j<len(nds_del) and nd==nds_del[ordre[j]]: j+=1                
            else: new_num[nd] = nd-j           
        new_num[nds_del] = new_num[IndexCouples[:,0]]        
        list_nd_new = [nd for nd in range(Nnd) if not(nd in nds_del)]                                     
        self.__ElementTable = new_num[self.__ElementTable]
        for key in self.__SetOfNodes:
            self.__SetOfNodes[key] = new_num[self.__SetOfNodes[key]]         
        self.__NodeCoordinates = self.__NodeCoordinates[list_nd_new]  
    

    def RemoveNodes(self, index_nodes):    
        """ 
        Remove some nodes and associated element
        The total number and the id of nodes are modified
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
    
        
        
    def Translate(self,Vector):
        """
        Translate the mesh along a given vector        
        """
        self.__NodeCoordinates = self.__NodeCoordinates + Vector        
    
    def ExtractSetOfElements(self,SetOfElementKey):
        """
        Return a new mesh from the set of elements defined by SetOfElementKey
        """
        new_SetOfElements = {}
        ListElm = self.__SetOfElements[SetOfElementKey]
        for key in self.__SetOfElements:
            new_SetOfElements[key] = np.array([el for el in self.__SetOfElements[key] if el in ListElm])       
        
        subMesh = Mesh(self.__NodeCoordinates, self.__ElementTable[ListElm], self.__ElementShape, self.__LocalFrame)                
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

if __name__=="__main__":
    import scipy as sp
    a = Mesh(sp.array([[0,0,0],[1,0,0]]), sp.array([[0,1]]),'lin2')
    print(a.GetNodeCoordinates())

