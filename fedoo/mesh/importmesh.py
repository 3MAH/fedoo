"""This module contains functions to import fedoo mesh from files"""

from __future__ import annotations

from fedoo.core.mesh import Mesh, MultiMesh
import numpy as np


def import_file(filename: str, name: str = "") -> Mesh:
    """Import a mesh from a file.

    Mesh.read should be prefered in most cases.

    Parameters
    ----------
    filename : str
        Name of file to import. Should be a ".msh" or a ".vtk" ascii files.
    name : str, optional
        Name of the imported Mesh. The default is "".

    Returns
    -------
    Mesh object
    """
    if filename[-4:].lower() == ".msh":
        return import_msh(filename, name)
    elif filename[-4:].lower() == ".vtk":
        return import_vtk(filename, name)
    else:
        assert 0, "Only .vtk and .msh file can be imported"


def import_msh(
    filename: str, name: str = "", mesh_type: list[str] = ["curve", "surface", "volume"]
) -> Mesh:
    """Import a mesh from a msh file (gmsh format).

    Mesh.read should be prefered in most cases.

    Parameters
    ----------
    filename : str
        Name of file to import. Should be a ".msh" or a ".vtk" ascii files.
    name : str, optional
        Name of the imported Mesh. The default is "".
    mesh_type : list of str in {'curve', 'surface', 'volume'}
        Type of geometries to import. Default = ['curve', 'surface', 'volume'],
        ie import all meshes.

    Returns
    -------
    Mesh object
    """

    if isinstance(mesh_type, str):
        mesh_type = [mesh_type]

    possible_element_type = []
    if "curve" in mesh_type:
        possible_element_type.extend([("1", "lin2"), ("8", "lin3")])
    if "surface" in mesh_type:
        possible_element_type.extend(
            [
                ("2", "tri3"),
                ("3", "quad4"),
                ("9", "tri6"),
                ("10", "quad9"),
                ("16", "quad8"),
            ]
        )
    if "volume" in mesh_type:
        possible_element_type.extend(
            [("4", "tet4"), ("5", "hex8"), ("11", "tet10"), ("17", "hex20")]
        )
    possible_element_type = dict(possible_element_type)

    # possible_element_type = {
    #             '1':'lin2',
    #             '2':'tri3',
    #             '3':'quad4',
    #             '4':'tet4',
    #             '5':'hex8',
    #             '8':'lin3',
    #             '9':'tri6',
    #             '10':'quad9',
    #             '11':'tet10',
    #             '16':'quad8',
    #             '17':'hex20'}

    filename = filename.strip()

    if name == None:
        name = filename
        if name[-4:].lower() == ".msh":
            name = name[:-4]
    mesh = None

    # print 'Reading file',`filename`
    f = open(filename, "r")
    msh = f.read()
    f.close()
    msh = msh.split("\n")
    msh = [line.strip() for line in msh if line.strip() != ""]

    l = msh.pop(0)
    if l.lower() != "$meshformat":
        raise NameError("Unknown file format")
    l = msh.pop(0).lower().split()  # versionnumber, file-type, data-size

    # Check version
    version = l[0]
    assert version in ["4.1", "2.2"], "Only support 2.2 and 4.1 version of gmsh format"

    l = msh.pop(0)  # $EndMeshFormat

    NodeData = []
    ElmData = []
    NodeDataName = []
    ElmDataName = []

    PhysicalNames = None
    # pointEntities = {}
    curveEntities = {}
    surfaceEntities = {}
    volumeEntities = {}

    curvePhysicalParts = {}
    surfacePhysicalParts = {}
    volumePhysicalParts = {}

    while msh != []:
        l = msh.pop(0).lower()

        if l == "$physicalnames":
            # create a dict PhysicalNames which contains a str name for every physical tags
            numPhysicalNames = int(msh.pop(0))
            PhysicalNames = dict(
                [
                    msh[physical_name].split()[1:3]
                    for physical_name in range(numPhysicalNames)
                ]
            )
            del msh[0:numPhysicalNames]

            msh.pop(0)  # $EndPhysicalNames

        elif l == "$entities" and version == "4.1":
            l = msh.pop(0).split()
            numPoints = int(l[0])
            numCurves = int(l[1])
            numSurfaces = int(l[2])
            numVolumes = int(l[3])

            # #read point entities
            # for i in range(numPoints):
            #     l = msh.pop(0).split()
            #     pointEntities[l[0]] =  l[5:5+int(l[4])]

            del msh[0:numPoints]

            if "curve" in mesh_type:
                # read line entities
                for i in range(numCurves):
                    l = msh.pop(0).split()
                    curveEntities[l[0]] = l[8 : 8 + int(l[7])]
            else:
                del msh[0:numCurves]

            if "surface" in mesh_type:
                # read surface entities
                for i in range(numSurfaces):
                    l = msh.pop(0).split()
                    surfaceEntities[l[0]] = l[8 : 8 + int(l[7])]
            else:
                del msh[0:numSurfaces]

            if "volume" in mesh_type:
                # read volume entitires
                for i in range(numVolumes):
                    l = msh.pop(0).split()
                    volumeEntities[l[0]] = l[8 : 8 + int(l[7])]
            else:
                del msh[0:numVolumes]

            msh.pop(0)  # $EndEntities

        elif l == "$nodes":
            if version == "2.2":
                Nb_nodes = int(msh.pop(0))

                numnode0 = int(
                    msh[0].split()[0]
                )  # 0 or 1, in the mesh format the first node is 0
                # a conversion is required if the msh file begin with another number
                # The numbering of nodes is assumed to be continuous (p. ex 1,2,3,....)
                crd = np.array(
                    [msh[nd].split()[1:] for nd in range(Nb_nodes)], dtype=float
                )

                del msh[0:Nb_nodes]
                msh.pop(0)  # $EndNodes
            elif version == "4.1":
                l = msh.pop(0).split()
                NbEntityBlocks = int(l[0])
                Nb_nodes = int(l[1])
                numnode0 = int(l[2])
                assert (
                    int(l[3]) == Nb_nodes - numnode0 + 1
                ), "Only possible to import msh file with continuous node numbering"
                crd = np.empty((Nb_nodes, 3))
                for i in range(NbEntityBlocks):
                    l = msh.pop(0).split()
                    # entityDim = l[0]
                    # entityTag = l[1] #id of the entity

                    assert (
                        int(l[2]) == 0
                    ), "Parametric coordinates are not implemented in this msh reader"
                    numNodesInBlock = int(l[3])
                    if numNodesInBlock != 0:
                        idnodes = np.array(msh[:numNodesInBlock], dtype=int) - numnode0
                        crd[idnodes] = np.array(
                            [
                                nd_crd.split()
                                for nd_crd in msh[numNodesInBlock : 2 * numNodesInBlock]
                            ],
                            dtype=float,
                        )
                        del msh[0 : 2 * numNodesInBlock]

                msh.pop(0)  # $EndNodes

        elif l == "$elements":
            if version == "2.2":
                Nb_el = int(msh.pop(0))
                cells = [msh[el].split()[1:] for el in range(Nb_el)]
                del msh[0:Nb_el]

                celltype_all = np.array([cells[el][0] for el in range(Nb_el)], int)
                Nb_tag = int(cells[0][1])  # assume to be constant
                #            if np.linalg.norm(cells[:,1] - Nb_tag) != 0:
                #                raise NameError('Only uniform number of Tags are readable')

                if Nb_tag < 2:
                    raise NameError("A minimum of 2 tags is required")
                elif Nb_tag > 2:
                    print("Warning: only the second tag is read")

                msh.pop(0)  # $EndElements

                # Tags = [cells[el][2:Nb_tag] for el in range(Nb_el)] #not used
                # PhysicalEntity = np.array([cells[el][2] for el in range(Nb_el)]) #fist tag not used for now

                Geom_all = np.array([cells[el][3] for el in range(Nb_el)], int)
                list_geom = list(np.unique(Geom_all))

            elif version == "4.1":
                l = msh.pop(0).split()
                NbEntityBlocks = int(l[0])
                Nb_el = int(l[1])
                # numel0 = int(l[2]) #usefull ?
                element_all = {}

                for i in range(NbEntityBlocks):
                    l = msh.pop(0).split()
                    entityDim = l[
                        0
                    ]  # 0 for point, 1 for curve, 2 for surface, 3 for volume
                    entityTag = l[1]  # id of the entity
                    elementType = l[2]
                    numElementsInBlock = int(l[3])

                    if entityDim == 1:
                        Entities = curveEntities
                    elif entityDim == 2:
                        Entities = surfaceEntities
                    elif entityDim == 3:
                        Entities = volumeEntities
                    else:
                        Entities = {}

                    if (
                        numElementsInBlock != 0
                        and elementType in possible_element_type.keys()
                    ):
                        if elementType not in element_all:
                            element_all[elementType] = [
                                [],
                                {},
                            ]  # [elementTable, elementSet]

                        idel0 = len(element_all[elementType][0])
                        element_all[elementType][0].extend(
                            [msh[el].split()[1:] for el in range(numElementsInBlock)]
                        )

                        list_elm_entity = list(
                            range(idel0, len(element_all[elementType][0]))
                        )

                        element_all[elementType][1]["Entity" + entityTag] = (
                            list_elm_entity
                        )

                        # add physical parts elset
                        if entityTag in Entities:
                            for physicalTag in Entities[entityTag]:
                                if physicalTag in PhysicalNames:
                                    el_name = PhysicalNames["physicalTag"]
                                else:
                                    el_name = "PhysicalPart" + physicalTag

                                if el_name not in element_all[elementType][1]:
                                    element_all[elementType][1][el_name] = (
                                        list_elm_entity
                                    )
                                else:
                                    element_all[elementType][1][el_name].append(
                                        list_elm_entity
                                    )

                        del msh[0:numElementsInBlock]

                    elif numElementsInBlock != 0:
                        del msh[0:numElementsInBlock]
                        # print('Warning : Elements type {} is not implemeted!'.format(elementType)) #element ignored

        elif l == "$nodedata" or l == "$elementdata":
            nb_str_tag = int(msh.pop(0))
            if l == "$nodedata":
                NodeDataName += [
                    str(msh.pop(0))
                ]  # the first string tag is the name of data
            else:
                ElmDataName += [str(msh.pop(0))]
            del msh[0 : nb_str_tag - 1]  # remove unused string tags
            nb_real_tag = int(msh.pop(0))
            del msh[0:nb_real_tag]  # remove unused real tags
            nb_int_tag = int(msh.pop(0))
            del msh[0:nb_int_tag]  # remove unused int tags

            if l == "$nodedata":
                if len(msh[0].split()) == 2:
                    NodeData += [
                        np.array(
                            [msh[nd].split()[1] for nd in range(Nb_nodes)], dtype=float
                        )
                    ]
                elif len(msh[0].split()) > 3:
                    NodeData += [
                        np.array(
                            [msh[nd].split()[1:] for nd in range(Nb_nodes)], dtype=float
                        )
                    ]
                del msh[0:Nb_nodes]
            else:
                if len(msh[0].split()) == 2:
                    ElmData += [
                        np.array(
                            [msh[el].split()[1] for el in range(Nb_el)], dtype=float
                        )
                    ]
                elif len(msh[0].split()) > 3:
                    ElmData += [
                        np.array(
                            [msh[el].split()[1:] for el in range(Nb_el)], dtype=float
                        )
                    ]
                del msh[0:Nb_el]

    count = 0
    if version == "2.2":
        if len(list(np.unique(celltype_all))) > 1:
            multi_mesh = True
            list_mesh = []
        else:
            multi_mesh = False

        for celltype in list(np.unique(celltype_all)):
            type_elm = None
            list_el = np.where(celltype_all == celltype)[0]
            elm = np.array([cells[el][2 + Nb_tag :] for el in list_el], int) - numnode0

            type_elm = possible_element_type.get(str(celltype))
            # not implemented '6':wed6 - '7':pyr5

            if (
                type_elm == "tet10"
            ):  # swap axes to account for different numbering schemes
                elm[:, [8, 9]] = elm[:, [9, 8]]
            elif (
                type_elm == "hex20"
            ):  # change order to account for different numbering schemes
                elm[:, 9:20] = elm[:, [11, 13, 9, 16, 18, 19, 17, 10, 12, 14, 15]]

            GeometricalEntity = []
            for geom in list_geom:
                GeometricalEntity.append(
                    [i for i in range(len(list_el)) if Geom_all[list_el[i]] == geom]
                )

            if type_elm is None:
                print(
                    "Warning : Elements type {} is not implemeted!".format(celltype)
                )  # element ignored
            else:
                if multi_mesh:
                    mesh_name = name + str(count)
                else:
                    mesh_name = name

                imported_mesh = Mesh(crd, elm, type_elm, name=mesh_name)

                # Rajouter GeometricalEntity en elSet

                if multi_mesh:
                    list_mesh.append(imported_mesh)
                count += 1

    elif version == "4.1":
        if len(element_all) > 1:
            multi_mesh = True
            list_mesh = []
        else:
            multi_mesh = False

        for elementType in element_all:
            elm = np.array(element_all[elementType][0], int) - numnode0

            type_elm = possible_element_type.get(str(elementType))
            # not implemented '6':wed6 - '7':pyr5

            if (
                type_elm == "tet10"
            ):  # swap axes to account for different numbering schemes
                elm[:, [8, 9]] = elm[:, [9, 8]]

            if multi_mesh:
                mesh_name = name + str(count)
            else:
                mesh_name = name

            imported_mesh = Mesh(crd, elm, type_elm, name=mesh_name)
            # add entity set of elements
            for elset in element_all[elementType][1]:
                imported_mesh.add_element_set(element_all[elementType][1][elset], elset)

            if multi_mesh:
                list_mesh.append(imported_mesh)

            # print('Mesh imported: "' + importedMeshName + '" with elements ' + type_elm)
            count += 1

    if multi_mesh:
        imported_mesh = MultiMesh.from_mesh_list(list_mesh, name)

    return imported_mesh
    # return NodeData, NodeDataName, ElmData, ElmDataName


def import_vtk(filename: str, name: str = "") -> Mesh:
    """Import a mesh from a vtk file (gmsh format).

    Mesh.read should be prefered in most cases.

    Parameters
    ----------
    filename : str
        Name of file to import. Should be a ".msh" or a ".vtk" ascii files.
    name : str, optional
        Name of the imported Mesh. The default is "".

    Returns
    -------
    Mesh object
    """

    filename = filename.strip()

    if name == None:
        name = filename
        if name[-4:].lower() == ".vtk":
            name = name[:-4]

    # print 'Reading file',`filename`
    f = open(filename, "r")
    vtk = f.read()
    f.close()
    vtk = vtk.split("\n")
    vtk = [line.strip() for line in vtk if line.strip() != ""]

    l = vtk.pop(0)
    fileversion = l.replace(" ", "").lower()
    if not fileversion == "#vtkdatafileversion2.0":
        print("File %s is not in VTK 2.0 format, got %s" % (filename, fileversion))
        print(" but continuing anyway..")
    header = vtk.pop(0)
    format = vtk.pop(0).lower()
    if format not in ["ascii", "binary"]:
        raise ValueError("Expected ascii|binary but got %s" % (format))
    if format == "binary":
        raise NotImplementedError("reading vtk binary format")

    l = vtk.pop(0).lower()
    if l[0:7] != "dataset":
        raise ValueError("expected dataset but got %s" % (l[0:7]))
    if l[-17:] != "unstructured_grid":
        raise NotImplementedError("Only unstructured grid are implemented")

    point_data = False
    cell_data = False
    NodeData = []
    ElmData = []
    NodeDataName = []
    ElmDataName = []

    # à partir de maintenant il n'y a plus d'ordre. il faut tout tester.
    while vtk != []:
        l = vtk.pop(0).split()
        if l[0].lower() == "points":
            Nb_nodes = int(l[1])
            # l[2] est considéré comme float dans tous les cas
            crd = np.array([vtk[nd].split() for nd in range(Nb_nodes)], dtype=float)
            del vtk[0:Nb_nodes]

        elif l[0].lower() == "cells":
            Nb_el = int(l[1])
            cells = vtk[0:Nb_el]
            del vtk[0:Nb_el]

        elif l[0].lower() == "cell_types":
            Nb_el = int(l[1])
            celltype_all = np.array(vtk[0:Nb_el], dtype=int)
            del vtk[0:Nb_el]

        elif l[0].lower() == "point_data":
            Nb_nodes = int(l[1])
            point_data = True
            cell_data = False

        elif l[0].lower() == "cell_data":
            Nb_el = int(l[1])
            point_data = False
            cell_data = True

        if l[0].lower() == "scalars" or l[0].lower() == "vectors":
            name = l[1]  # l[2] est considéré comme float dans tous les cas
            if l[0].lower() == "scalars":
                vtk.pop(0)  # lookup_table not implemented
                ncol = int(l[3])
            elif l[0].lower() == "vectors":
                ncol = 3

            if point_data == True:
                NodeData.append(
                    np.reshape(
                        np.array(
                            " ".join([vtk[ii] for ii in range(Nb_nodes)]).split(),
                            dtype=float,
                        ),
                        (-1, ncol),
                    )
                )
                #                NodeData.append(np.array([vtk[ii].split() for ii in range(Nb_nodes)], dtype = float))
                NodeDataName.append(name)
                del vtk[0:Nb_nodes]
            elif cell_data == True:
                ElmData.append(
                    np.reshape(
                        np.array(
                            " ".join([vtk[ii] for ii in range(Nb_el)]).split(),
                            dtype=float,
                        ),
                        (-1, ncol),
                    )
                )
                #                ElmData.append(np.array([vtk[ii].split() for ii in range(Nb_el)], dtype = float))
                print(np.shape(ElmData))
                ElmDataName.append(name)
                del vtk[0:Nb_el]
            else:
                print("Warning: Data ignored")

        if l[0].lower() == "tensors":
            print("Warning: tensor data not implemented. Data ignored")
            if point_data == True:
                del vtk[0:Nb_nodes]
            elif cell_data == True:
                del vtk[0:Nb_el]

    # Traitement des éléments
    count = 0
    if len(list(np.unique(celltype_all))):
        multi_mesh = True
        list_mesh = []
    else:
        multi_mesh = False

    for celltype in list(np.unique(celltype_all)):
        list_el = np.where(celltype_all == celltype)[0]
        elm = np.array([cells[el].split()[1:] for el in list_el], dtype=int)
        type_elm = {
            "3": "lin2",
            "5": "tri3",
            "9": "quad4",
            "10": "tet4",
            "12": "hex8",
            "21": "lin3",
            "22": "tri6",
            "23": "quad8",
            "24": "tet10",
            "25": "hex20",
        }.get(str(celltype))
        # not implemented '13':wed6 - '14':pyr5
        # vtk format doesnt support quad9

        if type_elm == None:
            print(
                "Warning : Elements type {} is not implemeted!".format(celltype)
            )  # element ignored
        else:
            if multi_mesh:
                mesh_name = name + str(count)
            else:
                mesh_name = name

            # print('Mesh imported: "' + mesh_name + '" with elements ' + type_elm)
            imported_mesh = Mesh(crd, elm, type_elm, name=mesh_name)
            count += 1

    if multi_mesh:
        list_mesh.append(imported_mesh)

    return imported_mesh
    # return NodeData, NodeDataName, ElmData, ElmDataName
