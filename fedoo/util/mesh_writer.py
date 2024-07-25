import numpy as np
from fedoo.core.mesh import Mesh


def write_vtk(dataset, filename="test.vtk", gp_data_to_node=True):
    # if multi_mesh == True: raise NotImplementedError('multi_mesh not implemented')

    datatype = "UNSTRUCTURED_GRID"
    mesh = dataset.mesh
    type_elm = mesh.elm_type

    try:  # get the number of nodes per element (in case  there is additional internal nodes)
        nb_nd_elm = str(int(type_elm[-2:]))
    except:
        nb_nd_elm = str(int(type_elm[-1]))

    #        nb_nd_elm = str(np.shape(elm)[1]) #Number of nodes per element

    elm = mesh.elements[:, : int(nb_nd_elm)]

    cell_type = {
        "lin2": "3",
        "tri3": "5",
        "quad4": "9",
        "tet4": "10",
        "hex8": "12",
        "wed6": "13",
        "pyr5": "14",
        "lin3": "21",
        "tri6": "22",
        "quad8": "23",
        "tet10": "24",
        "hex20": "25",
    }.get(type_elm)
    if cell_type == None:
        raise NotImplementedError("{} is not available in vtk".format(type_elm))

    ret = ["# vtk DataFile Version 3.0", "Some data", "ASCII"]

    # POINT
    ret += ["DATASET {}".format(datatype), "POINTS {} float".format(mesh.n_nodes)]

    if mesh.ndim == 2:
        ret += [
            " ".join([str(xx) for xx in line] + ["0.0"]) for line in mesh.nodes
        ]  # add a third dimension
    elif mesh.ndim == 3:
        ret += [" ".join([str(xx) for xx in line]) for line in mesh.nodes]
    else:
        raise NameError(
            "Error in the dimension of nodes coordinates - only 2D or 3D available"
        )

    # CELLS
    ret += [
        "CELLS {} {}".format(
            len(elm), (np.array(np.shape(elm)) + np.array([0, 1])).prod()
        )
    ]
    ret += [" ".join([nb_nd_elm] + [str(nd) for nd in line]) for line in elm]
    ret += ["CELL_TYPES {}".format(len(elm))]
    ret += [cell_type for i in range(mesh.n_elements)]

    # POINT_DATA
    # Convert gausspoint data to node data if gp_data_to_node == True
    if gp_data_to_node:
        node_data = {}
        for field in dataset.gausspoint_data:
            node_data[field] = dataset.get_data(field, data_type="Node")
        node_data.update(dataset.node_data)
    else:
        node_data = dataset.node_data
    if len(node_data) > 0:
        ret += ["POINT_DATA {}".format(mesh.n_nodes)]
        for data_name, data in node_data.items():
            # only scalar for now Vector and Tensor data may be developped
            if len(np.shape(data)) == 1:  # Scalar data
                ret += ["SCALARS {} float 1".format(data_name)]
                ret += ["LOOKUP_TABLE default"]  # to be developped if needed
                ret += list(data.astype(str))
            elif len(np.shape(data)) == 2:  # vector data
                data = data.T
                if np.shape(data)[1] == 3:  # vecteur
                    ret += ["VECTORS {} float".format(data_name)]
                    ret += [" ".join(val.astype(str)) for val in data]
                #                        ret += [' '.join('%.2f' % i for i in val) for val in data]
                else:
                    ret += ["SCALARS {} float {}".format(data_name, np.shape(data)[1])]
                    ret += ["LOOKUP_TABLE default"]  # to be developped if needed
                    ret += [" ".join(val.astype(str)) for val in data]
            elif len(np.shape(data)) == 3:  # tensor data
                data = data.transpose(2, 0, 1)
                ret += ["TENSORS {} float".format(data_name)]
                ret += [
                    "\n".join([" ".join(line.astype(str)) for line in val]) + "\n "
                    for val in data
                ]
            else:
                raise NameError("Data size mismatch")

    # CELL_DATA
    if len(dataset.element_data) > 0:
        ret += ["CELL_DATA {}".format(mesh.n_elements)]
        for data_name, data in dataset.element_data.items():
            if len(np.shape(data)) == 1:  # Scalar data
                ret += ["SCALARS {} float 1".format(data_name)]
                ret += ["LOOKUP_TABLE default"]  # to be developped if needed
                ret += list(data.astype(str))
            elif len(np.shape(data)) == 2:
                data = data.T
                if np.shape(data)[1] == 3:  # vector data
                    ret += ["VECTORS {} float".format(data_name)]
                    ret += [" ".join(val.astype(str)) for val in data]
                else:
                    ret += ["SCALARS {} float {}".format(data_name, np.shape(data)[1])]
                    ret += ["LOOKUP_TABLE default"]  # to be developped if needed
                    ret += [" ".join(val.astype(str)) for val in data]
            elif len(np.shape(data)) == 3:  # tensor data
                data = data.transpose(2, 0, 1)
                ret += ["TENSORS {} float".format(data_name)]
                ret += [
                    "\n".join([" ".join(line.astype(str)) for line in val]) + "\n "
                    for val in data
                ]
            else:
                raise NameError("Data size mismatch")

    f = open(filename, "w")
    f.write("\n".join(ret))
    f.close()


def write_msh(dataset, filename="test.msh", gp_data_to_node=True):
    # if self.multi_mesh == True: raise NotImplementedError('multi_mesh not implemented')

    mesh = dataset.mesh
    type_elm = mesh.elm_type

    try:  # get the number of nodes per element (in case  there is additional internal nodes)
        nb_nd_elm = int(type_elm[-2:])
    except:
        nb_nd_elm = int(type_elm[-1])

    #        nb_nd_elm = str(np.shape(elm)[1]) #Number of nodes per element

    elm = mesh.elements[:, :nb_nd_elm]

    if type_elm == "tet10":
        elm = elm[:, [1, 2, 3, 4, 5, 6, 7, 9, 8]]
    elif type_elm == "hex20":
        elm = elm[
            :, [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 16, 9, 17, 10, 18, 19, 12, 15, 13, 14]
        ]  # convert node order from fedoo mesh to vtk

    type_el = {
        "lin2": "1",
        "tri3": "2",
        "quad4": "3",
        "tet4": "4",
        "hex8": "5",
        "wed6": "6",
        "pyr5": "7",
        "lin3": "8",
        "tri6": "9",
        "quad9": "10",
        "tet10": "11",
        "quad8": "16",
        "hex20": "17",
    }.get(type_elm)

    GeometricalEntity = []
    #        if self.mesh.geometricalEntity != []:
    #            GeometricalEntity = np.zeros(self.mesh.mesh.n_elements,dtype=int)
    #            for ii,listelm in enumerate(self.mesh.geometricalEntity):
    #                GeometricalEntity[listelm] = ii

    if type_el == None:
        raise NotImplementedError("{} is not available in msh".format(type_elm))

    ret = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$Nodes", str(mesh.n_nodes)]
    if np.shape(mesh.nodes)[1] == 2:
        ret += [
            " ".join([str(nd)] + [str(xx) for xx in mesh.nodes[nd]] + ["0.0"])
            for nd in range(mesh.n_nodes)
        ]
    elif np.shape(mesh.nodes)[1] == 3:
        ret += [
            " ".join([str(nd)] + [str(xx) for xx in mesh.nodes[nd]])
            for nd in range(mesh.n_nodes)
        ]
    else:
        raise NameError(
            "Error in the dimension of nodes coordinates - only 2D or 3D available"
        )

    ret += ["$EndNodes", "$Elements", str(mesh.n_elements)]
    if GeometricalEntity == []:
        ret += [
            " ".join([str(el), type_el, "2", "1", "0"] + [str(nd) for nd in elm[el]])
            for el in range(mesh.n_elements)
        ]
    else:
        ret += [
            " ".join(
                [str(el), type_el, "2", "1", str(GeometricalEntity[el])]
                + [str(nd) for nd in elm[el]]
            )
            for el in range(mesh.n_elements)
        ]
    ret += ["$EndElements"]

    # Convert gausspoint data to node data if gp_data_to_node == True
    if gp_data_to_node:
        node_data = {}
        for field in dataset.gausspoint_data:
            node_data[field] = dataset.get_data(field, data_type="Node")
        node_data.update(dataset.node_data)
    else:
        node_data = dataset.node_data

    for data_name, data in node_data.items():
        ret += ["$NodeData", "1", data_name]  # NumerOfStringTag, Name of Data
        if len(np.shape(data)) == 1:  # Scalar data
            ret += [
                "1",
                "0.0",
                "3",
                "0",
                "1",
                str(mesh.n_nodes),
            ]  # NumberOfRealTag, time value, NumberOfIntegerTag, time step, number of scalar component (1, 3 or 9), Number of Nodes
            ret += [" ".join([str(nd), str(data[nd])]) for nd in range(mesh.n_nodes)]
        elif len(np.shape(data)) == 2:  # Vector data
            data = data.T
            ret += [
                "1",
                "0.0",
                "3",
                "0",
                str(np.shape(data)[1]),
                str(mesh.n_nodes),
            ]  # NumberOfRealTag, time value, NumberOfIntegerTag, time step, number of scalar component (1, 3 or 9), Number of Nodes
            ret += [
                " ".join([str(nd)] + list(data[nd].astype(str)))
                for nd in range(mesh.n_nodes)
            ]
        elif len(np.shape(data)) == 3:  # Tensor data
            data = data.transpose(2, 0, 1)
            ret += [
                "1",
                "0.0",
                "3",
                "0",
                "9",
                str(mesh.n_nodes),
            ]  # NumberOfRealTag, time value, NumberOfIntegerTag, time step, number of scalar component (1, 3 or 9), Number of Nodes
            ret += [
                " ".join(
                    [str(nd)] + list(data[nd][0].astype(str)),
                    +list(data[nd][1].astype(str)),
                    +list(data[nd][2].astype(str)),
                )
                for nd in range(mesh.n_nodes)
            ]
        else:
            raise NameError("Data size mismatch")
        ret += ["$EndNodeData"]

    for data_name, data in dataset.element_data.items():
        ret += ["$ElementData", "1", data_name]  # NumerOfStringTag, Name of Data
        if len(np.shape(data)) == 1:  # Scalar data
            ret += [
                "1",
                "0.0",
                "3",
                "0",
                "1",
                str(mesh.n_elements),
            ]  # NumberOfRealTag, time value, NumberOfIntegerTag, time step, number of scalar component (1, 3 or 9), Number of Nodes
            ret += [" ".join([str(el), str(data[el])]) for el in range(mesh.n_elements)]
        elif len(np.shape(data)) == 2:  # Vector data
            data = data.T
            ret += [
                "1",
                "0.0",
                "3",
                "0",
                str(np.shape(data)[1]),
                str(mesh.n_elements),
            ]  # NumberOfRealTag, time value, NumberOfIntegerTag, time step, number of scalar component (1, 3 or 9), Number of Nodes
            ret += [
                " ".join([str(el)] + list(data[el].astype(str)))
                for el in range(mesh.n_elements)
            ]
        elif len(np.shape(data)) == 3:  # Tensor data
            data = data.transpose(2, 0, 1)
            ret += [
                "1",
                "0.0",
                "3",
                "0",
                "9",
                str(mesh.n_elements),
            ]  # NumberOfRealTag, time value, NumberOfIntegerTag, time step, number of scalar component (1, 3 or 9), Number of Nodes
            ret += [
                " ".join(
                    [str(el)] + list(data[el][0].astype(str)),
                    +list(data[el][1].astype(str)),
                    +list(data[el][2].astype(str)),
                )
                for el in range(mesh.n_elements)
            ]

        else:
            raise NameError("Data size mismatch")
        ret += ["$EndElementData"]

    # Si besoin on peu faire pareil $ElementNodeData

    f = open(filename, "w")
    f.write("\n".join(ret))
    f.close()


# class ExportData:
#     def __init__(self, mesh, multiMesh = False):
#         if isinstance(mesh, str):
#             mesh = Mesh.get_all()[mesh]

#         self.mesh = mesh
#         self.multi_mesh = multiMesh #True if all the mesh defined in mesh._elm and mesh._type are included
#         self.NodeData = []
#         self.ElmData = []
#         self.NodeDataName = []
#         self.ElmDataName = []
#         self.format = 'ascii'
#         self.header = 'Some data'

#     def addNodeData(self, Data, Name=None):
#         if len(Data) != self.mesh.n_nodes:
#             raise NameError('Data dimension doesnt match the number of nodes')
#         if len(Data.shape) == 1: Data = np.c_[Data]
#         if Data.shape[1] == 2:
#             Data = np.c_[Data,np.zeros(self.mesh.n_nodes)]
#         self.NodeData += [np.array(Data)]
#         if Name==None:
#             Name = 'Data_{}'.format(len(self.NodeData))
#         self.NodeDataName += [Name]

#     def addElmData(self, Data, Name=None):
#         if len(Data) != self.mesh.n_elements:
#             raise NameError('Data dimension doesnt match the number of elements')
#         self.ElmData+= [Data]
#         if Name==None:
#             Name = 'Data_{}'.format(len(self.ElmData))
#         self.ElmDataName += [Name]

#     def toVTK(self, filename='test.vtk'):
#         if self.multi_mesh == True: raise NotImplementedError('multi_mesh not implemented')

#         datatype = 'UNSTRUCTURED_GRID'
#         type_elm = self.mesh.elm_type
#         try: #get the number of nodes per element (in case  there is additional internal nodes)
#             nb_nd_elm =  str(int(type_elm[-2:]))
#         except:
#             nb_nd_elm =  str(int(type_elm[-1]))

# #        nb_nd_elm = str(np.shape(elm)[1]) #Number of nodes per element

#         crd = self.mesh.nodes
#         elm = self.mesh.elements[:,:int(nb_nd_elm)]
#         Ncrd = self.mesh.n_nodes
#         Nel = self.mesh.n_elements

#         if type_elm == 'hex20':
#             elm = elm[:,[0,1,2,3,4,5,6,7,8,9,10,11,16,17,18,19,12,13,14,15]]

#         cell_type =  {'lin2':'3',
#                       'tri3':'5',
#                       'quad4':'9',
#                       'tet4':'10',
#                       'hex8':'12',
#                       'wed6':'13',
#                       'pyr5':'14',
#                       'lin3':'21',
#                       'tri6':'22',
#                       'quad8':'23',
#                       'tet10':'24',
#                       'hex20':'25'
#                       }.get(type_elm)
#         if cell_type == None: raise NotImplementedError('{} is not available in vtk'.format(type_elm))

#         ret = ['# vtk DataFile Version 3.0',
#                    self.header,
#                    self.format.upper()
#                    ]

#         #POINT
#         ret += ['DATASET {}'.format(datatype),
#                 'POINTS {} float'.format(Ncrd)]

#         if np.shape(crd)[1] == 2:
#             ret += [' '.join([str(xx) for xx in line]+['0.0']) for line in crd]
#         elif np.shape(crd)[1] == 3:
#             ret += [' '.join([str(xx) for xx in line]) for line in crd]
#         else:
#             raise NameError('Error in the dimension of nodes coordinates - only 2D or 3D available')


#         #CELLS
#         ret += ['CELLS {} {}'.format(len(elm), (np.array(np.shape(elm))+np.array([0,1])).prod())]
#         ret += [' '.join([nb_nd_elm]+[str(nd) for nd in line]) for line in elm]
#         ret += ['CELL_TYPES {}'.format(len(elm))]
#         ret += [cell_type for i in range(Nel)]

#         #POINT_DATA
#         if self.NodeData != []:
#             ret += ['POINT_DATA {}'.format(Ncrd)]
#             for i in range(len(self.NodeDataName)):
#                 # only scalar for now Vector and Tensor data may be developped
#                 if len(np.shape(self.NodeData[i])) == 1: #Scalar data
#                     ret += ['SCALARS {} float 1'.format(self.NodeDataName[i])]
#                     ret += ['LOOKUP_TABLE default'] #to be developped if needed
#                     ret += list(self.NodeData[i].astype(str))
#                 elif len(np.shape(self.NodeData[i])) == 2: #vector data
#                     if np.shape(self.NodeData[i])[1]==3: #vecteur
#                         ret += ['VECTORS {} float'.format(self.NodeDataName[i])]
#                         ret += [' '.join(val.astype(str)) for val in self.NodeData[i]]
# #                        ret += [' '.join('%.2f' % i for i in val) for val in self.NodeData[i]]
#                     else:
#                         ret += ['SCALARS {} float {}'.format(self.NodeDataName[i],np.shape(self.NodeData[i])[1])]
#                         ret += ['LOOKUP_TABLE default'] #to be developped if needed
#                         ret += [' '.join(val.astype(str)) for val in self.NodeData[i]]
#                 elif len(np.shape(self.NodeData[i])) == 3: #tensor data
#                     ret += ['TENSORS {} float'.format(self.NodeDataName[i])]
#                     ret += ['\n'.join([' '.join(line.astype(str)) for line in val])+'\n ' for val in self.NodeData[i]]
#                 else: raise NameError('Data size mismatch')


#         #CELL_DATA
#         if self.ElmData != []:
#             ret += ['CELL_DATA {}'.format(Nel)]
#             for i in range(len(self.ElmDataName)):
#                 if len(np.shape(self.ElmData[i])) == 1: #Scalar data
#                     ret += ['SCALARS {} float 1'.format(self.ElmDataName[i])]
#                     ret += ['LOOKUP_TABLE default'] #to be developped if needed
#                     ret += list(self.ElmData[i].astype(str))
#                 elif len(np.shape(self.ElmData[i])) == 2:
#                     if np.shape(self.ElmData[i])[1]==3: #vector data
#                         ret += ['VECTORS {} float'.format(self.ElmDataName[i])]
#                         ret += [' '.join(val.astype(str)) for val in self.ElmData[i]]
#                     else:
#                         ret += ['SCALARS {} float {}'.format(self.ElmDataName[i],np.shape(self.ElmData[i])[1])]
#                         ret += ['LOOKUP_TABLE default'] #to be developped if needed
#                         ret += [' '.join(val.astype(str)) for val in self.ElmData[i]]
#                 elif len(np.shape(self.ElmData[i])) == 3: #tensor data
#                     ret += ['TENSORS {} float'.format(self.ElmDataName[i])]
#                     ret += ['\n'.join([' '.join(line.astype(str)) for line in val])+'\n ' for val in self.ElmData[i]]
#                 else: raise NameError('Data size mismatch')

#         f = open(filename,'w')
#         f.write('\n'.join(ret))
#         f.close()


#     def toMSH(self, filename='test.msh'):
#         if self.multi_mesh == True: raise NotImplementedError('multi_mesh not implemented')

#         type_elm = self.mesh.elm_type
#         try: #get the number of nodes per element (in case  there is additional internal nodes)
#             nb_nd_elm =  int(type_elm[-2:])
#         except:
#             nb_nd_elm =  int(type_elm[-1])

# #        nb_nd_elm = str(np.shape(elm)[1]) #Number of nodes per element

#         crd = self.mesh.nodes
#         elm = self.mesh.elements[:,:nb_nd_elm]
#         Ncrd = self.mesh.n_nodes
#         Nel = self.mesh.n_elements

#         if type_elm == 'tet10':
#             elm = elm[:, [1,2,3,4,5,6,7,9,8]]

#         type_el = {   'lin2':'1',
#                       'tri3':'2',
#                       'quad4':'3',
#                       'tet4':'4',
#                       'hex8':'5',
#                       'wed6':'6',
#                       'pyr5':'7',
#                       'lin3':'8',
#                       'tri6':'9',
#                       'quad9':'10',
#                       'tet10':'11',
#                       'quad8':'16',
#                       'hex20':'17',
#                       }.get(type_elm)

#         GeometricalEntity = []
# #        if self.mesh.geometricalEntity != []:
# #            GeometricalEntity = np.zeros(self.mesh.Nel,dtype=int)
# #            for ii,listelm in enumerate(self.mesh.geometricalEntity):
# #                GeometricalEntity[listelm] = ii

#         if type_el == None: raise NotImplementedError('{} is not available in msh'.format(type_elm))

#         ret = ['$MeshFormat',
#                '2.2 0 8',
#                '$EndMeshFormat',
#                '$Nodes',
#                 str(Ncrd)]
#         if np.shape(crd)[1] == 2:
#             ret += [' '.join([str(nd)]+[str(xx) for xx in crd[nd]] + ['0.0'])  for nd in range(Ncrd)]
#         elif np.shape(crd)[1] == 3:
#             ret += [' '.join([str(nd)]+[str(xx) for xx in crd[nd]])  for nd in range(Ncrd)]
#         else:
#             raise NameError('Error in the dimension of nodes coordinates - only 2D or 3D available')

#         ret += ['$EndNodes',
#                '$Elements',
#                 str(Nel)]
#         if GeometricalEntity == []:
#             ret += [' '.join([str(el), type_el, '2', '1', '0']+[str(nd) for nd in elm[el]])  for el in range(Nel)]
#         else:
#             ret += [' '.join([str(el), type_el, '2', '1', str(GeometricalEntity[el])]+[str(nd) for nd in elm[el]])  for el in range(Nel)]
#         ret += ['$EndElements']

#         for k in range (len(self.NodeDataName)) :
#             ret += ['$NodeData', '1', self.NodeDataName[k]]  #NumerOfStringTag, Name of Data
#             if len(np.shape(self.NodeData[k])) == 1: #Scalar data
#                 ret += ['1', '0.0', '3', '0', '1', str(Ncrd)] #NumberOfRealTag, time value, NumberOfIntegerTag, time step, number of scalar component (1, 3 or 9), Number of Nodes
#                 ret += [' '.join([str(nd),str(self.NodeData[k][nd])])  for nd in range(Ncrd)]
#             elif len(np.shape(self.NodeData[k])) == 2: #Vector data
#                 ret += ['1', '0.0', '3', '0', str(np.shape(self.NodeData[k])[1]), str(Ncrd)] #NumberOfRealTag, time value, NumberOfIntegerTag, time step, number of scalar component (1, 3 or 9), Number of Nodes
#                 ret += [' '.join([str(nd)]+list(self.NodeData[k][nd].astype(str)))  for nd in range(Ncrd)]
#             elif len(np.shape(self.NodeData[k])) == 3: #Tensor data
#                 ret += ['1', '0.0', '3', '0', '9', str(Ncrd)] #NumberOfRealTag, time value, NumberOfIntegerTag, time step, number of scalar component (1, 3 or 9), Number of Nodes
#                 ret += [' '.join([str(nd)]+list(self.NodeData[k][nd][0].astype(str)), \
#                                           +list(self.NodeData[k][nd][1].astype(str)), \
#                                           +list(self.NodeData[k][nd][2].astype(str))) for nd in range(Ncrd)]
#             else: raise NameError('Data size mismatch')
#             ret += ['$EndNodeData']

#         for k in range (len(self.ElmDataName)) :
#             ret += ['$ElementData', '1', self.ElmDataName[k]]  #NumerOfStringTag, Name of Data
#             if len(np.shape(self.ElmData[k])) == 1: #Scalar data
#                 ret += ['1', '0.0', '3', '0', '1', str(Nel)] #NumberOfRealTag, time value, NumberOfIntegerTag, time step, number of scalar component (1, 3 or 9), Number of Nodes
#                 ret += [' '.join([str(el),str(self.ElmData[k][el])])  for el in range(Nel)]
#             elif len(np.shape(self.ElmData[k])) == 2: #Vector data
#                 ret += ['1', '0.0', '3', '0', str(np.shape(self.ElmData[k])[1]), str(Nel)] #NumberOfRealTag, time value, NumberOfIntegerTag, time step, number of scalar component (1, 3 or 9), Number of Nodes
#                 ret += [' '.join([str(el)]+list(self.ElmData[k][el].astype(str)))  for el in range(Nel)]
#             elif len(np.shape(self.ElmData[k])) == 3: #Tensor data
#                 ret += ['1', '0.0', '3', '0', '9', str(Nel)] #NumberOfRealTag, time value, NumberOfIntegerTag, time step, number of scalar component (1, 3 or 9), Number of Nodes
#                 ret += [' '.join([str(el)]+list(self.ElmData[k][el][0].astype(str)), \
#                                           +list(self.ElmData[k][el][1].astype(str)), \
#                                           +list(self.ElmData[k][el][2].astype(str))) for el in range(Nel)]

#             else: raise NameError('Data size mismatch')
#             ret += ['$EndElementData']

#         # Si besoin on peu faire pareil $ElementNodeData

#         f = open(filename,'w')
#         f.write('\n'.join(ret))
#         f.close()


# def plot_msh(self):
#     from subprocess import Popen
#     self.toMSH()
#     Popen("gmsh test.msh", stdin=None, stdout=None, stderr=None)

# def plot_vtk(self):
#     from subprocess import Popen
#     self.toVTK()
#     Popen('"C:\\Program Files\\ParaView 5.2.0-RC2-Qt4-OpenGL2-Windows-64bit\\bin\\paraview" test.vtk', stdin=None, stdout=None, stderr=None)


# def ImportMSH(filename):
#    mesh = None
#
#    filename = filename.strip()
#    if filename[-4:].lower()!='.msh':
#        filename += '.msh'
#
#    #print 'Reading file',`filename`
#    f = open(filename,'r')
#    msh = f.read()
#    f.close()
#    msh = msh.split('\n')
#    msh = [line.strip() for line in msh if line.strip() != '']
#
#    l = msh.pop(0)
#    if l.lower()!='$meshformat': raise NameError('Unknown file format')
#    l = msh.pop(0).lower().split()
#    #versionnumber, file-type, data-size
#    l = msh.pop(0) #$EndMeshFormat
#
#    NodeData = []
#    ElmData = []
#    NodeDataName = []
#    ElmDataName = []
#
#    while msh != []:
#        l = msh.pop(0).lower()
#
#        if l == '$nodes':
#            Nb_nodes = int(msh.pop(0))
#
#            numnode0 = int(msh[0].split()[0]) #0 or 1, in the mesh format the first node is 0
#            #a conversion is required if the msh file begin with another number
#            #The numbering of nodes is assumed to be continuous (p. ex 1,2,3,....)
#            crd = np.array([msh[nd].split()[1:] for nd in range(Nb_nodes)], dtype = float)
#
#            del msh[0:Nb_nodes]
#            msh.pop(0) #$EndNodes
#
#        elif l == '$elements':
#            Nb_el = int(msh.pop(0))
#            cells = [msh[el].split()[1:] for el in range(Nb_el)]
#            del msh[0:Nb_el]
#
#            celltype_all = np.array([cells[el][0] for el in range(Nb_el)], int)
#            Nb_tag = int(cells[0][1]) #assume to be constant
##            if np.linalg.norm(cells[:,1] - Nb_tag) != 0:
##                raise NameError('Only uniform number of Tags are readable')
#
#            if Nb_tag < 2: raise NameError('A minimum of 2 tags is required')
#            elif Nb_tag > 2: print('Warning: only the second tag is read')
#
#            msh.pop(0) #$EndElements
#
#            #Tags = [cells[el][2:Nb_tag] for el in range(Nb_el)] #not used
#            #PhysicalEntity = np.array([cells[el][2] for el in range(Nb_el)]) #fist tag not used for now
#
#            Geom_all = np.array([cells[el][3] for el in range(Nb_el)], int)
#            list_geom = list(np.unique(Geom_all))
#
#        elif l == '$nodedata' or l == '$elementdata':
#            nb_str_tag = int(msh.pop(0))
#            if l == '$nodedata':
#                NodeDataName += [str(msh.pop(0))] #the first string tag is the name of data
#            else:
#                ElmDataName += [str(msh.pop(0))]
#            del msh[0:nb_str_tag-1] #remove unused string tags
#            nb_real_tag = int(msh.pop(0))
#            del msh[0:nb_real_tag]#remove unused real tags
#            nb_int_tag = int(msh.pop(0))
#            del msh[0:nb_int_tag]#remove unused int tags
#
#            if l == '$nodedata':
#                if len(msh[0].split()) == 2:
#                    NodeData += [np.array([msh[nd].split()[1] for nd in range(Nb_nodes)], dtype=float)]
#                elif len(msh[0].split()) > 3:
#                    NodeData += [np.array([msh[nd].split()[1:] for nd in range(Nb_nodes)], dtype=float)]
#                del msh[0:Nb_nodes]
#            else:
#                if len(msh[0].split()) == 2:
#                    ElmData += [np.array([msh[el].split()[1] for el in range(Nb_el)], dtype=float)]
#                elif len(msh[0].split()) > 3:
#                    ElmData += [np.array([msh[el].split()[1:] for el in range(Nb_el)], dtype=float)]
#                del msh[0:Nb_el]
#
#
#    count = 0
#    for celltype in list(np.unique(celltype_all)):
#        type_elm = None
#        list_el = np.where(celltype_all == celltype)[0]
#        elm =  np.array([cells[el][2+Nb_tag:] for el in list_el], int) - numnode0
#        if celltype in [1,2,3,4,5,8,9,10,16,11,17]:
#            type_elm = celltype
#            #some elements are not implemented wed6, pyr5, ...
#
#        GeometricalEntity = []
#        for geom in list_geom:
#            GeometricalEntity.append([i for i in range(len(list_el)) if Geom_all[list_el[i]]==geom ])
#
#        if type_elm == None: print('Warning : Elements type {} is not implemeted!'.format(celltype)) #element ignored
#        elif count==0:
#            mesh = Maillage_EF(crd, elm, type_elm, bord=[], rep_loc=None, structured_mesh = 0, n_elm_gp = None, geometricalEntity = GeometricalEntity)
#            count+=1
#        else:
#            mesh.addMesh(elm, type_elm, rep_loc=None, structured_mesh = 0, n_elm_gp = None, geometricalEntity = GeometricalEntity)
#            count+=1
#
#    res= MeshData(mesh)
#    res.NodeData = NodeData
#    res.NodeDataName = NodeDataName
#    res.ElmData = ElmData
#    res.ElmDataName = ElmDataName
##    res.GeometricalEntity = GeometricalEntity
#    return res
#


# def ImportVTK(filename):
#    mesh = None
#
#    filename = filename.strip()
#    if filename[-4:].lower()!='.vtk':
#        filename += '.vtk'
#
#    #print 'Reading file',`filename`
#    f = open(filename,'r')
#    vtk = f.read()
#    f.close()
#    vtk = vtk.split('\n')
#    vtk = [line.strip() for line in vtk if line.strip() != '']
#
#    l = vtk.pop(0)
#    fileversion = l.replace(' ','').lower()
#    if not fileversion == '#vtkdatafileversion2.0':
#        print ('File %s is not in VTK 2.0 format, got %s' % (filename, fileversion))
#        print (' but continuing anyway..')
#    header = vtk.pop(0)
#    format = vtk.pop(0).lower()
#    if format not in ['ascii','binary']:
#        raise ValueError('Expected ascii|binary but got %s'%(format))
#    if format == 'binary':
#        raise NotImplementedError('reading vtk binary format')
#
#    l = vtk.pop(0).lower()
#    if l[0:7] != 'dataset':
#        raise ValueError('expected dataset but got %s'%(l[0:7]))
#    if l[-17:] != 'unstructured_grid':
#        raise NotImplementedError('Only unstructured grid are implemented')
#
#    point_data = False
#    cell_data = False
#    NodeData = []
#    ElmData = []
#    NodeDataName = []
#    ElmDataName = []
#
#    # à partir de maintenant il n'y a plus d'ordre. il faut tout tester.
#    while vtk != []:
#        l = vtk.pop(0).split()
#        if l[0].lower() == 'points':
#            Nb_nodes = int(l[1])
#            #l[2] est considéré comme float dans tous les cas
#            crd = np.array([vtk[nd].split() for nd in range(Nb_nodes)], dtype = float)
#            del vtk[0:Nb_nodes]
#
#        elif l[0].lower() == 'cells':
#            Nb_el = int(l[1])
#            cells = vtk[0:Nb_el]
#            del vtk[0:Nb_el]
#
#        elif l[0].lower() == 'cell_types':
#            Nb_el = int(l[1])
#            celltype_all = np.array(vtk[0:Nb_el], dtype=int)
#            del vtk[0:Nb_el]
#
#        elif l[0].lower() == 'point_data':
#            Nb_nodes = int(l[1])
#            point_data = True ; cell_data = False
#
#        elif l[0].lower() == 'cell_data':
#            Nb_el = int(l[1])
#            point_data = False ; cell_data = True
#
#
#        if l[0].lower() == 'scalars' or l[0].lower() == 'vectors':
#            name = l[1] #l[2] est considéré comme float dans tous les cas
#            if l[0].lower() == 'scalars':
#                vtk.pop(0) #lookup_table not implemented
#                ncol = int(l[3])
#            elif l[0].lower() == 'vectors': ncol = 3
#
#            if point_data == True:
#                NodeData.append(np.reshape(np.array(' '.join([vtk[ii] for ii in range(Nb_nodes)]).split(), dtype=float) , (-1,ncol)))
##                NodeData.append(np.array([vtk[ii].split() for ii in range(Nb_nodes)], dtype = float))
#                NodeDataName.append(name)
#                del vtk[0:Nb_nodes]
#            elif cell_data == True:
#                ElmData.append(np.reshape(np.array(' '.join([vtk[ii] for ii in range(Nb_el)]).split(), dtype=float) , (-1,ncol)))
##                ElmData.append(np.array([vtk[ii].split() for ii in range(Nb_el)], dtype = float))
#                print(np.shape(ElmData))
#                ElmDataName.append(name)
#                del vtk[0:Nb_el]
#            else: Print('Warning: Data ignored')
#
#        if l[0].lower() == 'tensors':
#            Print('Warning: tensor data not implemented. Data ignored')
#            if point_data == True:
#                del vtk[0:Nb_nodes]
#            elif cell_data == True:
#                del vtk[0:Nb_el]
#
#    #Traitement des éléments
#    count = 0
#    for celltype in list(np.unique(celltype_all)):
#        list_el = np.where(celltype_all == celltype)[0]
#        elm =  np.array([cells[el].split()[1:] for el in list_el], dtype = int)
#        type_elm = {'3':1,
#                    '5':2,
#                    '9':3,
#                    '10':4,
#                    '12':5,
#                    '21':8,
#                    '22':9,
#                    '23':16,
#                    '24':11,
#                    '25':17
#                    }.get(str(celltype))
#                      #not implemented '13':wed6 - '14':pyr5
#        if type_elm == None: print('Warning : Elements type {} is not implemeted!'.format(celltype)) #ignored
#        elif count==0:
#            mesh = Maillage_EF(crd, elm, type_elm, bord=[], rep_loc=None, structured_mesh = 0, n_elm_gp = None)
#            count+=1
#        else:
#            mesh.addMesh(elm, type_elm, rep_loc=None, structured_mesh = 0, n_elm_gp = None)
#            count+=1
#
#    res= MeshData(mesh)
#    res.NodeData = NodeData
#    res.NodeDataName = NodeDataName
#    res.ElmData = ElmData
#    res.ElmDataName = ElmDataName
#    return res
#
#
# def plot_mesh(mesh, visu = 'msh', elset=None):
#    if elset != None:
#        import copy
#        mesh = copy.deepcopy(mesh)
#        mesh.elm = mesh.elm[elset] ; mesh.Nel = len(mesh.elm) ; mesh.geometricalEntity = []
#    from subprocess import Popen
#
#    if visu == 'msh':
#        MeshData(mesh).toMSH()
#        Popen("gmsh test.msh", stdin=None, stdout=None, stderr=None)
#    else:
#        MeshData(mesh).toVTK()
#        Popen('"C:\\Program Files\\ParaView 5.2.0-RC2-Qt4-OpenGL2-Windows-64bit\\bin\\paraview" test.vtk', stdin=None, stdout=None, stderr=None)
#
