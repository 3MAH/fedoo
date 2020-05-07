from fedoo.libMesh.MeshBase import MeshBase
from fedoo.libUtil.Variable import Variable
from fedoo.libMesh.Mesh import Mesh as MeshFEM
import numpy as np

class MeshPGD(MeshBase): #class pour définir des maillages sous forme séparées
    """     
    PGD.Mesh(mesh1, mesh2, ....)    

    A MaillageS object represents a mesh under a separated form.     
    The global mesh is defined with a tensorial product of submeshes contained
    in the attribute maillage.
    Each submeshes has its own coordinate system.     
    ----------
   
    """      
 
    def __init__(self, *args, **kargs):
        if 'ID' in kargs: ID = kargs['ID']
        else: ID = 'PGDMesh'
        
        if not isinstance(ID, str): assert 0, "An ID must be a string"
        
        MeshBase.__init__(self, ID)
        self.__ListMesh = [MeshBase.GetAll()[m] if isinstance(m,str) else m for m in args]
        
        listCrdID = [crdid for m in self.__ListMesh for crdid in m.GetCoordinateID() ]
        if len(set(listCrdID)) != len(listCrdID): 
            print("Warning: some coordinate ID are defined in more than one mesh")
        
        self.__SetOfNodes = {} #node on the boundary for instance
        self.__SetOfElements = {}
        self.__SpecificVariableRank = {} #to define specific variable rank for each submesh (used for the PGD)
    
    def _SetSpecificVariableRank(self, idmesh, idvar, specific_rank):
        #idmesh : the id of any submesh
        #idvar : variable id that is given by Variable.GetRank(name) 
        #        if idvar == 'default': define the default value for all variables
        #specific_rank : rank considered for the PGD assembly   
        #no specific rank can be defined if there is a change of basis in the pysicial mesh related to coordinates 'X', 'Y' and 'Z'
        
        assert isinstance(idmesh, int), 'idmesh must an integer, not a ' + str(type(idmesh))
        assert idvar == 'default' or isinstance(idvar, int),  'idvar must an integer or "default"'
        assert isinstance(specific_rank, int), 'specific_rank must an integer, not a ' + str(type(idmesh))
        
        if idmesh in self.__SpecificVariableRank:
            self.__SpecificVariableRank[idmesh][idvar] =  specific_rank
        else: self.__SpecificVariableRank[idmesh] = {idvar: specific_rank}
        
    def _GetSpecificVariableRank(self, idmesh, idvar):
        assert isinstance(idmesh, int), 'idmesh must an integer, not a ' + str(type(idmesh))
        assert idvar == 'default' or isinstance(idvar, int),  'idvar must an integer or "default"'
        
        if idmesh in self.__SpecificVariableRank:
            if idvar in self.__SpecificVariableRank[idmesh]:
                return self.__SpecificVariableRank[idmesh][idvar]
            elif 'default' in self.__SpecificVariableRank[idmesh]:
                return self.__SpecificVariableRank[idmesh]['default']
        else: return idvar

    def _GetSpecificNumberOfVariables(self, idmesh):
        assert isinstance(idmesh, int), 'idmesh must an integer, not a ' + str(type(idmesh))
        if idmesh in self.__SpecificVariableRank:
            return max(self.__SpecificVariableRank[idmesh].values())+1
        else: return Variable.GetNumberOfVariable()
    
    def GetDimension(self):
        return len(self.__ListMesh)
    
    def GetListMesh(self):
        return self.__ListMesh
    
    def AddSetOfNodes(self, listNodeIndexes, listSubMesh = None, ID=None):
        """
        The Set Of Nodes in Mesh PGD object are used to defined the boundary conditions.
        There is two ways of defining a SetOfNodes:
        
        PGD.Mesh.AddSetOfNodes([nodeIndexes_1,...,nodeIndexes_n ], ID=SetOfID)
            * nodeIndexes_i a list of node indexe cooresponding to the ith subMesh (as defined in the constructor of the PGD.Mesh object)
            * nodeIndexes_i can also be set to "all" to indicate that all the nodes have to be included
            * SetOfID is the ID of the SetOf
            
        PGD.Mesh.AddSetOfNodes([nodeIndexes_1,...,nodeIndexes_n ], [subMesh_1,...,subMesh_n], ID=SetOfID)
            * nodeIndexes_i a list of node indexe cooresponding to the mesh indicated in subMesh_i
            * subMesh_i can be either a mesh ID (str object) or a Mesh object
            * the keyword "all" is NOT available when the subMesh are indicated.
            * If a subMesh is not included in listSubMesh, all the Nodes are considered
            * SetOfID is the ID of the SetOf
        """
        if ID == None:
            num = 1
            while "NodeSet"+str(num) in self.__SetOfNodes: 
                num += 1
            ID = "NodeSet"+str(num)                
        
        if listSubMesh is None:
            if len(listNodeIndexes) != len(self.__ListMesh):
                assert 0, "The lenght of the Node Indexes List must be equal to the number of submeshes"
            listSubMesh = [i for i in range(len(self.__ListMesh)) if not(listNodeIndexes[i] is 'all' or listNodeIndexes[i] is 'ALL')]
            listNodeIndexes = [NodeIndexes for NodeIndexes in listNodeIndexes if not(NodeIndexes is 'all' or NodeIndexes is 'ALL')]
        else:
            listSubMesh = [self.__ListMesh.index(MeshBase.GetAll()[m]) if isinstance(m,str) else self.__ListMesh.index(m) for m in listSubMesh]
                
        self.__SetOfNodes[ID] = [listSubMesh, listNodeIndexes]
        
    def AddSetOfElements(self, listElementIndexes, listSubMesh = None, ID=None):
        """
        See the documention of AddSetOfNodes. 
        AddSetOfElements is a similar method for Elements.
        """
        if ID == None:
            num = 1
            while "ElementSet"+str(num) in self.__SetOfNodes: 
                num += 1
            ID = "ElementSet"+str(num)                
        
        if listSubMesh is None:
            if len(listElementIndexes) != len(self.__ListMesh):
                assert 0, "The lenght of the Node Indexes List must be equal to the number of submeshes"
            listSubMesh = [self.__ListMesh[i] for i in range(len(self.__ListMesh)) if listElementIndexes[i] not in ['all','ALL']]
            listElementIndexes = [ElmIndexes for ElmIndexes in ListElementIndexes if ElmIndexes not in ['all','ALL']]
        else:
            listSubMesh = [Mesh.GetAll()[m] if isinstance(m,str) else m for m in listSubMesh]
        
        self.__SetOfElements[ID] = [listSubMesh, listElementIndexes]

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
    
    def FindCoordinateID(self, crd):
        """
        Try to find a coordinate in the submeshes. 
        Return the position of the mesh in the list MeshPGD.GetListMesh() or None if the coordinate is not found
        """
        for idmesh, mesh in enumerate(self.__ListMesh):
            if crd in mesh.GetCoordinateID():
                return idmesh
        return None    
    
    def GetNumberOfNodes(self):
        """
        Return a list containing the number of nodes for all submeshes
        """
        return [m.GetNumberOfNodes() for m in self.__ListMesh]
    
    def GetNumberOfElements(self):
        """
        Return a list containing the number of nodes for all submeshes
        """
        return [m.GetNumberOfElements() for m in self.__ListMesh]

    @staticmethod        
    def Create(*args, **kargs):        
        return MeshPGD(*args, **kargs)
        
                                        
#    def __getitem__(self, key):
#        return self.__ListMesh[key]
        
#    def append(self, mesh):
#        """Add a new mesh (submesh of the separated mesh) in the maillage attribute"""
#        self.__ListMesh.append(mesh)
#
#    def __len__(self): 
#        return len(self.__ListMesh) # taille de la liste                             


            
        


    def ExtractFullMesh(self, ID = "FullMesh", useLocalFrame = False):
        if len(self.__ListMesh) == 3: 
            return NotImplemented
#            mesh1 = self.__ListMesh[0] ; mesh2 = self.__ListMesh[1] ; mesh3 = self.__ListMesh[2]
#            if mesh1.__ElementShape == 'lin2' and mesh2.__ElementShape == 'lin2' and mesh3.__ElementShape == 'lin2':
#                elmType = 'hex8'                
#                
#            else: 'element doesnt exist'
        elif len(self.__ListMesh) == 2:            
            mesh1 = self.__ListMesh[1] ; mesh0 = self.__ListMesh[0]
            
            Nel1 = mesh1.GetNumberOfElements() ; Nel0 = mesh0.GetNumberOfElements() 
            Nel = Nel1*Nel0
            ndInElm1 = np.shape(mesh1.GetElementTable())[1] 
            ndInElm0 = np.shape(mesh0.GetElementTable())[1]
            elm = np.zeros((Nel,ndInElm1*ndInElm0), dtype=int)
           
            if mesh0.GetElementShape() == 'lin2': #mesh0 is 'lin2'
                dim_mesh0 = 1
                if mesh1.GetElementShape() == 'lin2': 
                    type_elm = 'quad4'    
                    dim_mesh1 = 1
                    for i in range(Nel0):
                        elm[i*Nel1:(i+1)*Nel1 , [0,1]] = mesh1.GetElementTable() + mesh0.GetElementTable()[i,0]*mesh1.GetNumberOfNodes()
                        elm[i*Nel1:(i+1)*Nel1 , [3,2]] = mesh1.GetElementTable() + mesh0.GetElementTable()[i,1]*mesh1.GetNumberOfNodes()
                elif mesh1.GetElementShape() == 'quad4': 
                    dim_mesh1 = 2
                    type_elm = 'hex8'
                    for i in range(Nel0):                
                        elm[i*Nel1:(i+1)*Nel1 , 0:ndInElm1         ] = mesh1.GetElementTable() + mesh0.GetElementTable()[i,0]*mesh1.GetNumberOfNodes()
                        elm[i*Nel1:(i+1)*Nel1 , ndInElm1:2*ndInElm1] = mesh1.GetElementTable() + mesh0.GetElementTable()[i,1]*mesh1.GetNumberOfNodes()                        
                else: raise NameError('Element not implemented')

            elif mesh0.GetElementShape() == 'lin3':     #need verification because the node numerotation for lin2 has changed             
                dim_mesh0 = 1
                if mesh1.GetElementShape() == 'lin3': #mesh1 and mesh0 are lin3 elements
                    dim_mesh1 = 1
                    type_elm = 'quad9'
                    for i in range(Nel0): #éléments 1D à 3 noeuds (pour le moment uniquement pour générer des éléments quad9)
                        elm[i*Nel1:(i+1)*Nel1 , [0,4,1] ] = mesh1.GetElementTable() + mesh0.GetElementTable()[i,0]*mesh1.GetNumberOfNodes()
                        elm[i*Nel1:(i+1)*Nel1 , [7,8,5] ] = mesh1.GetElementTable() + mesh0.GetElementTable()[i,1]*mesh1.GetNumberOfNodes()
                        elm[i*Nel1:(i+1)*Nel1 , [3,6,2] ] = mesh1.GetElementTable() + mesh0.GetElementTable()[i,2]*mesh1.GetNumberOfNodes()
                else: raise NameError('Element not implemented')
                
            elif mesh0.GetElementShape() == 'quad4':
                dim_mesh0 = 2
                if mesh1.GetElementShape() == 'lin2':
                    dim_mesh1 = 1
                    type_elm = 'hex8'                        
                    for i in range(Nel1):                
                        elm[i::Nel1 , 0:ndInElm0         ] = mesh0.GetElementTable()*mesh1.GetNumberOfNodes() + mesh1.GetElementTable()[i,0]
                        elm[i::Nel1 , ndInElm0:2*ndInElm0] = mesh0.GetElementTable()*mesh1.GetNumberOfNodes() + mesh1.GetElementTable()[i,1]
                            
            else: raise NameError('Element not implemented') 
            
            if useLocalFrame == False:       
                Ncrd = mesh1.GetNumberOfNodes() * mesh0.GetNumberOfNodes()
#                crd = np.c_[np.tile(mesh1.GetNodeCoordinates()[:,:dim_mesh1],(mesh0.GetNumberOfNodes(),1)), \
#                            np.reshape([np.ones((mesh1.GetNumberOfNodes(),1))*mesh0.GetNodeCoordinates()[i,:dim_mesh0] for i in range(mesh0.GetNumberOfNodes())] ,(Ncrd,-1)) ] 
                crd = np.c_[np.reshape([np.ones((mesh1.GetNumberOfNodes(),1))*mesh0.GetNodeCoordinates()[i,:dim_mesh0] for i in range(mesh0.GetNumberOfNodes())] ,(Ncrd,-1)), \
                            np.tile(mesh1.GetNodeCoordinates()[:,:dim_mesh1],(mesh0.GetNumberOfNodes(),1))] 
            elif dim_mesh0 == 1: #dim_mesh0 is the thickness
                crd = np.zeros((mesh1.GetNumberOfNodes()*mesh0.GetNumberOfNodes(), np.shape(mesh1.GetNodeCoordinates())[1]))
                for i in range(mesh0.GetNumberOfNodes()):
                    crd[i*mesh1.GetNumberOfNodes():(i+1)*mesh1.GetNumberOfNodes(),:] = mesh1.GetNodeCoordinates() + mesh1.GetLocalFrame()[:,-1,:]*mesh0.GetNodeCoordinates()[i][0]
            else: return NotImplemented
            
            return MeshFEM(crd, elm, type_elm, ID=ID)                        
        
        elif len(self.__ListMesh) == 1 : return self.__ListMesh[0]
        else: raise NameError("FullMesh can only be extracted from Separated Mesh of dimenson <= 3")
            
        
