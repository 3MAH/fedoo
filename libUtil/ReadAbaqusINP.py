from fedoo.libMesh.Mesh import Mesh
from fedoo.libProblem.BoundaryCondition import BoundaryCondition
import numpy as np

class ReadINP:

    def __init__(self, *args):
        self.filename = args[0].strip()
        keyword = []
        inp = []
        for filename in args:
            filename = filename.strip()
            # if meshID == None:
            #     meshID = filename
            #     if meshID[-4:].lower() == '.msh':
            #         meshID = meshID[:-4]
            # mesh = None
               
            #print 'Reading file',`filename`
            f = open(filename,'r')
            txt = f.read()
            f.close()
            txt = txt.lower().replace(',', '\t')
            txt = txt.split('\n')
            txt = [line.strip() for line in txt] #remove unwanted spaces
            txt = [line for line in txt if line != '' and line[:2] != '**'] #remove empty line and comments
            inp.extend(txt)
        
        keyword = [(i ,line) for i,line in enumerate(inp) if (line[0]=='*' and line[1]!='*')]
        keyword.append((len(inp), '*end'))
        
        NodeData = []
        ElmData = []
        NodeDataName = []
        ElmDataName = []    
        
        Element = []
        NodeSet = {}
        ElementSet = {}
        NodeCoordinate = None
        NodeNumber = None
        
        Equation = {} #dict where entry are nb terms on the multi point constraint equation
        
        for k in range(len(keyword)):
            nline = keyword[k][0]
            key = keyword[k][1]
    
            if key[0:5] == '*node': 
                # inp[nline+1:keyword[k+1][0]]
                crd = np.array([inp[line].split() for line in range(nline+1,keyword[k+1][0])], dtype = float) 
                NodeNumber = crd[:,0]
                NodeCoordinate = crd[:,1:]
                
                # numnode0 = int(msh[0].split()[0]) #0 or 1, in the mesh format the first node is 0
                # #a conversion is required if the msh file begin with another number
                # #The numbering of nodes is assumed to be continuous (p. ex 1,2,3,....)
                # crd = np.array([msh[nd].split()[1:] for nd in range(Nb_nodes)], dtype = float)             
            
            elif key[0:8] == '*element': 
                celltype = key[8:].strip()[4:].replace('=','').strip()
    
                if celltype in ['cpe3', 'cps3', 'cpe3h']: 
                    fedooElm = 'tri3'
                elif celltype in ['cpe4', 'cpe4h', 'cpe4i', 'cpe4ih', 'cpe4r', 'cpe4rh',  
                                  'cps4', 'cps4i', 'cps4r']: 
                    fedooElm = 'quad4'
                elif celltype in ['cpe6', 'cpe6h', 'cpe6m', 'cpe6mh', 'cps6', 'cps6m']:
                    fedooElm = 'tri6'
                elif celltype in ['cpe8', 'cpe8h', 'cpe8r', 'cpe8rh', 'cps8', 'cps8r']:
                    fedooElm = 'quad8'    
                elif celltype[:4] in ['c3d4']: 
                    fedooElm = 'tet4'
                elif celltype[:4] in ['c3d8']: 
                    fedooElm = 'hex8'
                else: fedooElm = None 
                
    
                elm = np.array([inp[line].split() for line in range(nline+1,keyword[k+1][0])], dtype = float) 
                Element.append({})
                Element[-1]['ElementNumber'] = elm[:,0]
                Element[-1]['ElementTable'] = elm[:,1:]
                Element[-1]['ElementType'] = fedooElm
    
            elif key[0:5] == '*nset': 
                idSet = key[5:].strip()[4:].replace('=','').strip()
                if 'unsorted' in idSet: idSet = idSet.replace('unsorted','') #option ignored
                if 'generate' in idSet:
                    idSet = idSet.replace('generate','')
                    temp = [inp[line].split() for line in range(nline+1,keyword[k+1][0])]
                    nset = []
                    for line in temp: 
                        if len(line)>1: incr = int(line[2])
                        else: incr = 1
                        nset.extend(list(range(int(line[0]), int(line[1])+1, incr)))
                    nset = np.array(nset)
                else:
                    nset = np.array([nd for line in range(nline+1,keyword[k+1][0]) for nd in inp[line].split()], dtype = int)        
                idSet = idSet.strip()
                NodeSet[idSet] = nset
                
            elif key[0:6] == '*elset': 
                idSet = key[6:].strip()[5:].replace('=','').strip()  
                if 'unsorted' in idSet: idSet = idSet.replace('unsorted','') #option ignored                
                if 'generate' in idSet:
                    idSet = idSet.replace('generate','')
                    temp = [inp[line].split() for line in range(nline+1,keyword[k+1][0])]
                    elset = []
                    for line in temp: 
                        if len(line)>1: incr = int(line[2])
                        else: incr = 1
                        elset.extend(list(range(int(line[0]), int(line[1])+1, incr)))
                    elset = np.array(elset)
                else:
                    elset = np.array([el for line in range(nline+1,keyword[k+1][0]) for el in inp[line].split() ], dtype = int)        
                idSet = idSet.strip()
                ElementSet[idSet] = elset    
            
            elif key[0:9] == '*equation': 
                # Equation.append(np.hstack([inp[line].split() for line in range(nline+1,keyword[k+1][0])]).astype(float))
                nTerms = int(inp[nline+1])
                eq = np.hstack([inp[line].split() for line in range(nline+2,keyword[k+1][0])]).astype(float)
                listVar = tuple(eq[1::3].astype(int))
                if not( listVar in Equation): Equation[listVar] = []
                Equation[listVar].append(eq)                
                
                # if not(nTerms in Equation): Equation[nTerms] = []
                # Equation[nTerms].append(np.hstack([inp[line].split() for line in range(nline+2,keyword[k+1][0])]).astype(float))

                            
        self.Equation = Equation #for debug
        self.__Equation = Equation
        self.__Element = Element
        self.__NodeSet = NodeSet
        self.__ElementSet = ElementSet
        self.__NodeCoordinate = NodeCoordinate
        self.__NodeNumber = NodeNumber
        
        ConvertNodeDict = dict(zip(NodeNumber, list(range(0,len(NodeNumber)))))
        self.__ConvertNode = np.vectorize(ConvertNodeDict.get) #function
        
        # self.ConvertNodeDict = ConvertNodeDict
        # self.ConvertNode = self.__ConvertNode
        # self.Element = Element
    
    def toMesh(self, meshID = None):     
        if meshID == None:
            meshID = self.filename
            if meshID[-4:].lower() == '.inp': meshID = meshID[:-4]           
        for count,dict_elm in enumerate(self.__Element):
            if len(self.__Element) < 2: importedMeshName = meshID
            else: importedMeshName = meshID+str(count)
            
            elm = self.__ConvertNode(dict_elm['ElementTable'])
            Mesh(self.__NodeCoordinate, elm, dict_elm['ElementType'], ID = importedMeshName) 
            #add set of nodes
            for SetOfId,NodeIndexes in enumerate(self.__NodeSet):
                Mesh.GetAll()[importedMeshName].AddSetOfNodes(self.__ConvertNode(NodeIndexes),SetOfId)
            
            ElementNumber = dict_elm['ElementNumber']
            ConvertElementDict = dict(zip(ElementNumber, list(range(0,len(ElementNumber)))))        
            ConvertElement = np.vectorize(ConvertElementDict.get) #function
            for SetOfId,ElementIndexes in enumerate(self.__ElementSet):
                Temp = ConvertElement(ElementIndexes)                
                Mesh.GetAll()[importedMeshName].AddSetOfElements(Temp[Temp != None].astype(int),SetOfId)
            

    def applyBoundaryCondition(self, ProblemID = "MainProblem"):
        for listVar in self.__Equation:
            eq = np.array(self.__Equation[listVar])
            BoundaryCondition('MPC', listVar, eq[:,2::3], eq[:,0::3].astype(int), ProblemID = ProblemID)

            
    
    
    
    

# def ImportFromVTK(filename, meshID = None):
#     filename = filename.strip()

#     if meshID == None:
#         meshID = filename
#         if meshID[-4:].lower() == '.vtk': meshID = meshID[:-4]
        
#     #print 'Reading file',`filename`
#     f = open(filename,'r')
#     vtk = f.read()
#     f.close()
#     vtk = vtk.split('\n')
#     vtk = [line.strip() for line in vtk if line.strip() != '']   
    
#     l = vtk.pop(0)
#     fileversion = l.replace(' ','').lower()
#     if not fileversion == '#vtkdatafileversion2.0':
#         print ('File %s is not in VTK 2.0 format, got %s' % (filename, fileversion))
#         print (' but continuing anyway..')
#     header = vtk.pop(0)
#     format = vtk.pop(0).lower()
#     if format not in ['ascii','binary']:
#         raise ValueError('Expected ascii|binary but got %s'%(format))
#     if format == 'binary':
#         raise NotImplementedError('reading vtk binary format')
        
#     l = vtk.pop(0).lower()     
#     if l[0:7] != 'dataset':
#         raise ValueError('expected dataset but got %s'%(l[0:7]))
#     if l[-17:] != 'unstructured_grid':
#         raise NotImplementedError('Only unstructured grid are implemented')
   
#     point_data = False
#     cell_data = False   
#     NodeData = []
#     ElmData = []
#     NodeDataName = []
#     ElmDataName = []    
   
#     # à partir de maintenant il n'y a plus d'ordre. il faut tout tester. 
#     while vtk != []:
#         l = vtk.pop(0).split()
#         if l[0].lower() == 'points':
#             Nb_nodes = int(l[1])
#             #l[2] est considéré comme float dans tous les cas
#             crd = np.array([vtk[nd].split() for nd in range(Nb_nodes)], dtype = float) 
#             del vtk[0:Nb_nodes]
                         
#         elif l[0].lower() == 'cells':
#             Nb_el = int(l[1])
#             cells = vtk[0:Nb_el]
#             del vtk[0:Nb_el]
             
#         elif l[0].lower() == 'cell_types':
#             Nb_el = int(l[1])
#             celltype_all = np.array(vtk[0:Nb_el], dtype=int)
#             del vtk[0:Nb_el]
                     
#         elif l[0].lower() == 'point_data':
#             Nb_nodes = int(l[1])
#             point_data = True ; cell_data = False
            
#         elif l[0].lower() == 'cell_data':
#             Nb_el = int(l[1])
#             point_data = False ; cell_data = True          
            
            
#         if l[0].lower() == 'scalars' or l[0].lower() == 'vectors':   
#             name = l[1] #l[2] est considéré comme float dans tous les cas
#             if l[0].lower() == 'scalars': 
#                 vtk.pop(0) #lookup_table not implemented
#                 ncol = int(l[3])
#             elif l[0].lower() == 'vectors': ncol = 3
                
#             if point_data == True:
#                 NodeData.append(np.reshape(np.array(' '.join([vtk[ii] for ii in range(Nb_nodes)]).split(), dtype=float) , (-1,ncol)))                                
# #                NodeData.append(np.array([vtk[ii].split() for ii in range(Nb_nodes)], dtype = float))
#                 NodeDataName.append(name)
#                 del vtk[0:Nb_nodes]
#             elif cell_data == True:
#                 ElmData.append(np.reshape(np.array(' '.join([vtk[ii] for ii in range(Nb_el)]).split(), dtype=float) , (-1,ncol)))
# #                ElmData.append(np.array([vtk[ii].split() for ii in range(Nb_el)], dtype = float))
#                 print(np.shape(ElmData))
#                 ElmDataName.append(name)                                
#                 del vtk[0:Nb_el]
#             else: Print('Warning: Data ignored')
            
#         if l[0].lower() == 'tensors': 
#             Print('Warning: tensor data not implemented. Data ignored')
#             if point_data == True:
#                 del vtk[0:Nb_nodes]
#             elif cell_data == True:
#                 del vtk[0:Nb_el]            
    
#     #Traitement des éléments
#     count = 0
#     for celltype in list(np.unique(celltype_all)):         
#         list_el = np.where(celltype_all == celltype)[0]
#         elm =  np.array([cells[el].split()[1:] for el in list_el], dtype = int) 
#         type_elm = {'3':'lin2',
#                     '5':'tri3',
#                     '9':'quad4',
#                     '10':'tet4',
#                     '12':'hex8',
#                     '21':'lin3',
#                     '22':'tri6',
#                     '23':'quad8',           
#                     '24':'tet10',
#                     '25':'hex20'
#                     }.get(str(celltype))
#                       #not implemented '13':wed6 - '14':pyr5
#                       #vtk format doesnt support quad9

#         if type_elm == None: print('Warning : Elements type {} is not implemeted!'.format(celltype)) #element ignored
#         else:            
#             if len(list(np.unique(celltype_all))) == 1:
#                 importedMeshName = meshID
#             else: importedMeshName = meshID+str(count)
                
#             print('Mesh imported: "' + importedMeshName + '" with elements ' + type_elm)
#             Mesh(crd, elm, type_elm, ID = importedMeshName)
#             count+=1

#     return NodeData, NodeDataName, ElmData, ElmDataName