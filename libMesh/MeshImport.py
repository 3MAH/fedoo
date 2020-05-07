from fedoo.libMesh.Mesh import Mesh
import numpy as np

def ImportFromFile(filename, meshID = None):
    if filename[-4:].lower() == '.msh':
        return ImportFromMSH(filename, meshID)
    elif filename[-4:].lower() == '.vtk':
        return ImportFromVTK(filename, meshID)    
    else: assert 0, "Only .vtk and .msh file can be imported"

def ImportFromMSH(filename, meshID = None):

    filename = filename.strip()

    if meshID == None:
        meshID = filename
        if meshID[-4:].lower() == '.msh':
            meshID = meshID[:-4]
    mesh = None
       
    #print 'Reading file',`filename`
    f = open(filename,'r')
    msh = f.read()
    f.close()
    msh = msh.split('\n')
    msh = [line.strip() for line in msh if line.strip() != '']   
    
    l = msh.pop(0)
    if l.lower()!='$meshformat': raise NameError('Unknown file format')
    l = msh.pop(0).lower().split() 
    #versionnumber, file-type, data-size
    l = msh.pop(0) #$EndMeshFormat
    
    NodeData = []
    ElmData = []
    NodeDataName = []
    ElmDataName = []    
                
    while msh != []:       
        l = msh.pop(0).lower() 
        
        if l == '$nodes': 
            Nb_nodes = int(msh.pop(0))
            
            numnode0 = int(msh[0].split()[0]) #0 or 1, in the mesh format the first node is 0
            #a conversion is required if the msh file begin with another number
            #The numbering of nodes is assumed to be continuous (p. ex 1,2,3,....)
            crd = np.array([msh[nd].split()[1:] for nd in range(Nb_nodes)], dtype = float) 
            
            del msh[0:Nb_nodes]
            msh.pop(0) #$EndNodes
        
        elif l == '$elements': 
            Nb_el = int(msh.pop(0))
            cells = [msh[el].split()[1:] for el in range(Nb_el)] 
            del msh[0:Nb_el]
            
            celltype_all = np.array([cells[el][0] for el in range(Nb_el)], int)
            Nb_tag = int(cells[0][1]) #assume to be constant
#            if np.linalg.norm(cells[:,1] - Nb_tag) != 0:  
#                raise NameError('Only uniform number of Tags are readable')
                        
            if Nb_tag < 2: raise NameError('A minimum of 2 tags is required')
            elif Nb_tag > 2: print('Warning: only the second tag is read')
            
            msh.pop(0) #$EndElements
                        
            #Tags = [cells[el][2:Nb_tag] for el in range(Nb_el)] #not used
            #PhysicalEntity = np.array([cells[el][2] for el in range(Nb_el)]) #fist tag not used for now
                        
            Geom_all = np.array([cells[el][3] for el in range(Nb_el)], int)                  
            list_geom = list(np.unique(Geom_all))
                        
        elif l == '$nodedata' or l == '$elementdata':
            nb_str_tag = int(msh.pop(0))
            if l == '$nodedata':
                NodeDataName += [str(msh.pop(0))] #the first string tag is the name of data
            else: 
                ElmDataName += [str(msh.pop(0))]
            del msh[0:nb_str_tag-1] #remove unused string tags
            nb_real_tag = int(msh.pop(0))
            del msh[0:nb_real_tag]#remove unused real tags
            nb_int_tag = int(msh.pop(0))
            del msh[0:nb_int_tag]#remove unused int tags
                        
            if l == '$nodedata':
                if len(msh[0].split()) == 2:
                    NodeData += [np.array([msh[nd].split()[1] for nd in range(Nb_nodes)], dtype=float)]
                elif len(msh[0].split()) > 3:
                    NodeData += [np.array([msh[nd].split()[1:] for nd in range(Nb_nodes)], dtype=float)]                    
                del msh[0:Nb_nodes]
            else:
                if len(msh[0].split()) == 2:
                    ElmData += [np.array([msh[el].split()[1] for el in range(Nb_el)], dtype=float)]
                elif len(msh[0].split()) > 3:
                    ElmData += [np.array([msh[el].split()[1:] for el in range(Nb_el)], dtype=float)]
                del msh[0:Nb_el]
                    
                                      
    count = 0
    for celltype in list(np.unique(celltype_all)):     
        type_elm = None
        list_el = np.where(celltype_all == celltype)[0]
        elm =  np.array([cells[el][2+Nb_tag:] for el in list_el], int) - numnode0

        type_elm = {'1':'lin2',
                    '2':'tri3',
                    '3':'quad4',
                    '4':'tet4',
                    '5':'hex8',
                    '8':'lin3',
                    '9':'tri6',
                    '10':'quad9',                    
                    '11':'tet10',
                    '16':'quad8',           
                    '17':'hex20'
                    }.get(str(celltype))
                      #not implemented '6':wed6 - '7':pyr5

        GeometricalEntity = []
        for geom in list_geom:
            GeometricalEntity.append([i for i in range(len(list_el)) if Geom_all[list_el[i]]==geom ])
                   
        if type_elm == None: print('Warning : Elements type {} is not implemeted!'.format(celltype)) #element ignored
        else:          
            if len(list(np.unique(celltype_all))) == 1:
                importedMeshName = meshID
            else: importedMeshName = meshID+str(count)
                
            print('Mesh imported: "' + importedMeshName + '" with elements ' + type_elm)
            Mesh(crd, elm, type_elm, ID = importedMeshName)            
            #Rajouter GeometricalEntity en elSet
            count+=1

#    res= MeshData(mesh)
#    res.NodeData = NodeData
#    res.NodeDataName = NodeDataName
#    res.ElmData = ElmData
#    res.ElmDataName = ElmDataName
#    res.GeometricalEntity = GeometricalEntity
    return NodeData, NodeDataName, ElmData, ElmDataName       
    
    
    
    

def ImportFromVTK(filename, meshID = None):
    filename = filename.strip()

    if meshID == None:
        meshID = filename
        if meshID[-4:].lower() == '.vtk': meshID = meshID[:-4]
        
    #print 'Reading file',`filename`
    f = open(filename,'r')
    vtk = f.read()
    f.close()
    vtk = vtk.split('\n')
    vtk = [line.strip() for line in vtk if line.strip() != '']   
    
    l = vtk.pop(0)
    fileversion = l.replace(' ','').lower()
    if not fileversion == '#vtkdatafileversion2.0':
        print ('File %s is not in VTK 2.0 format, got %s' % (filename, fileversion))
        print (' but continuing anyway..')
    header = vtk.pop(0)
    format = vtk.pop(0).lower()
    if format not in ['ascii','binary']:
        raise ValueError('Expected ascii|binary but got %s'%(format))
    if format == 'binary':
        raise NotImplementedError('reading vtk binary format')
        
    l = vtk.pop(0).lower()     
    if l[0:7] != 'dataset':
        raise ValueError('expected dataset but got %s'%(l[0:7]))
    if l[-17:] != 'unstructured_grid':
        raise NotImplementedError('Only unstructured grid are implemented')
   
    point_data = False
    cell_data = False   
    NodeData = []
    ElmData = []
    NodeDataName = []
    ElmDataName = []    
   
    # à partir de maintenant il n'y a plus d'ordre. il faut tout tester. 
    while vtk != []:
        l = vtk.pop(0).split()
        if l[0].lower() == 'points':
            Nb_nodes = int(l[1])
            #l[2] est considéré comme float dans tous les cas
            crd = np.array([vtk[nd].split() for nd in range(Nb_nodes)], dtype = float) 
            del vtk[0:Nb_nodes]
                         
        elif l[0].lower() == 'cells':
            Nb_el = int(l[1])
            cells = vtk[0:Nb_el]
            del vtk[0:Nb_el]
             
        elif l[0].lower() == 'cell_types':
            Nb_el = int(l[1])
            celltype_all = np.array(vtk[0:Nb_el], dtype=int)
            del vtk[0:Nb_el]
                     
        elif l[0].lower() == 'point_data':
            Nb_nodes = int(l[1])
            point_data = True ; cell_data = False
            
        elif l[0].lower() == 'cell_data':
            Nb_el = int(l[1])
            point_data = False ; cell_data = True          
            
            
        if l[0].lower() == 'scalars' or l[0].lower() == 'vectors':   
            name = l[1] #l[2] est considéré comme float dans tous les cas
            if l[0].lower() == 'scalars': 
                vtk.pop(0) #lookup_table not implemented
                ncol = int(l[3])
            elif l[0].lower() == 'vectors': ncol = 3
                
            if point_data == True:
                NodeData.append(np.reshape(np.array(' '.join([vtk[ii] for ii in range(Nb_nodes)]).split(), dtype=float) , (-1,ncol)))                                
#                NodeData.append(np.array([vtk[ii].split() for ii in range(Nb_nodes)], dtype = float))
                NodeDataName.append(name)
                del vtk[0:Nb_nodes]
            elif cell_data == True:
                ElmData.append(np.reshape(np.array(' '.join([vtk[ii] for ii in range(Nb_el)]).split(), dtype=float) , (-1,ncol)))
#                ElmData.append(np.array([vtk[ii].split() for ii in range(Nb_el)], dtype = float))
                print(np.shape(ElmData))
                ElmDataName.append(name)                                
                del vtk[0:Nb_el]
            else: Print('Warning: Data ignored')
            
        if l[0].lower() == 'tensors': 
            Print('Warning: tensor data not implemented. Data ignored')
            if point_data == True:
                del vtk[0:Nb_nodes]
            elif cell_data == True:
                del vtk[0:Nb_el]            
    
    #Traitement des éléments
    count = 0
    for celltype in list(np.unique(celltype_all)):         
        list_el = np.where(celltype_all == celltype)[0]
        elm =  np.array([cells[el].split()[1:] for el in list_el], dtype = int) 
        type_elm = {'3':'lin2',
                    '5':'tri3',
                    '9':'quad4',
                    '10':'tet4',
                    '12':'hex8',
                    '21':'lin3',
                    '22':'tri6',
                    '23':'quad8',           
                    '24':'tet10',
                    '25':'hex20'
                    }.get(str(celltype))
                      #not implemented '13':wed6 - '14':pyr5
                      #vtk format doesnt support quad9

        if type_elm == None: print('Warning : Elements type {} is not implemeted!'.format(celltype)) #element ignored
        else:            
            if len(list(np.unique(celltype_all))) == 1:
                importedMeshName = meshID
            else: importedMeshName = meshID+str(count)
                
            print('Mesh imported: "' + importedMeshName + '" with elements ' + type_elm)
            Mesh(crd, elm, type_elm, ID = importedMeshName)
            count+=1

    return NodeData, NodeDataName, ElmData, ElmDataName