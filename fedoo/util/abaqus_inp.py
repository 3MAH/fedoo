from fedoo.core.mesh import Mesh

# from fedoo.core.base import BoundaryCondition
import numpy as np


class ReadINP:
    def __init__(self, *args):
        self.filename = args[0].strip()
        keyword = []
        inp = []
        for filename in args:
            filename = filename.strip()
            # if meshname == None:
            #     meshname = filename
            #     if meshname[-4:].lower() == '.msh':
            #         meshname = meshname[:-4]
            # mesh = None

            # print 'Reading file',`filename`
            f = open(filename, "r")
            txt = f.read()
            f.close()
            txt = txt.lower().replace(",", "\t")
            txt = txt.split("\n")
            txt = [line.strip() for line in txt]  # remove unwanted spaces
            txt = [
                line for line in txt if line != "" and line[:2] != "**"
            ]  # remove empty line and comments
            inp.extend(txt)

        keyword = [
            (i, line)
            for i, line in enumerate(inp)
            if (line[0] == "*" and line[1] != "*")
        ]
        keyword.append((len(inp), "*end"))

        NodeData = []
        ElmData = []
        NodeDataName = []
        ElmDataName = []

        Element = []
        NodeSet = {}
        ElementSet = {}
        NodeCoordinate = None
        NodeNumber = None

        Equation = {}  # dict where entry are nb terms on the multi point constraint equation

        for k in range(len(keyword)):
            nline = keyword[k][0]
            key = keyword[k][1]

            if key[0:5] == "*node":
                # inp[nline+1:keyword[k+1][0]]
                crd = np.array(
                    [inp[line].split() for line in range(nline + 1, keyword[k + 1][0])],
                    dtype=float,
                )
                NodeNumber = crd[:, 0]
                NodeCoordinate = crd[:, 1:]
                print(NodeCoordinate)

                # numnode0 = int(msh[0].split()[0]) #0 or 1, in the mesh format the first node is 0
                # #a conversion is required if the msh file begin with another number
                # #The numbering of nodes is assumed to be continuous (p. ex 1,2,3,....)
                # crd = np.array([msh[nd].split()[1:] for nd in range(Nb_nodes)], dtype = float)

            elif key[0:8] == "*element":
                celltype = key[8:].strip()[4:].replace("=", "").strip()

                if celltype[:4] in ["cpe3", "cps3"]:  # , 'cpe3h'
                    fedooElm = "tri3"
                elif celltype[:4] in [
                    "cpe4",
                    "cps4",
                ]:  #'cpe4h', 'cpe4i', 'cpe4ih', 'cpe4r', 'cpe4rh', 'cps4i', 'cps4r'
                    fedooElm = "quad4"
                elif celltype[:4] in [
                    "cpe6",
                    "cps6",
                ]:  #'cpe6h', 'cpe6m', 'cpe6mh','cps6m'
                    fedooElm = "tri6"
                elif celltype[:4] in [
                    "cpe8",
                    "cps8",
                ]:  #'cpe8h', 'cpe8r', 'cpe8rh', 'cps8r'
                    fedooElm = "quad8"
                elif celltype[:4] in ["c3d4"]:
                    fedooElm = "tet4"
                elif celltype[:4] in ["c3d8"]:
                    fedooElm = "hex8"
                else:
                    fedooElm = None

                elm = np.array(
                    [inp[line].split() for line in range(nline + 1, keyword[k + 1][0])],
                    dtype=float,
                )
                Element.append({})
                Element[-1]["ElementNumber"] = elm[:, 0]
                Element[-1]["ElementTable"] = elm[:, 1:]
                Element[-1]["ElementType"] = fedooElm

            elif key[0:5] == "*nset":
                idSet = key[5:].strip()[4:].replace("=", "").strip()
                if "unsorted" in idSet:
                    idSet = idSet.replace("unsorted", "")  # option ignored
                if "generate" in idSet:
                    idSet = idSet.replace("generate", "")
                    temp = [
                        inp[line].split()
                        for line in range(nline + 1, keyword[k + 1][0])
                    ]
                    nset = []
                    for line in temp:
                        if len(line) > 1:
                            incr = int(line[2])
                        else:
                            incr = 1
                        nset.extend(list(range(int(line[0]), int(line[1]) + 1, incr)))
                    nset = np.array(nset)
                else:
                    nset = np.array(
                        [
                            nd
                            for line in range(nline + 1, keyword[k + 1][0])
                            for nd in inp[line].split()
                        ],
                        dtype=int,
                    )
                idSet = idSet.strip()
                NodeSet[idSet] = nset

            elif key[0:6] == "*elset":
                idSet = key[6:].strip()[5:].replace("=", "").strip()
                if "unsorted" in idSet:
                    idSet = idSet.replace("unsorted", "")  # option ignored
                if "generate" in idSet:
                    idSet = idSet.replace("generate", "")
                    temp = [
                        inp[line].split()
                        for line in range(nline + 1, keyword[k + 1][0])
                    ]
                    elset = []
                    for line in temp:
                        if len(line) > 1:
                            incr = int(line[2])
                        else:
                            incr = 1
                        elset.extend(list(range(int(line[0]), int(line[1]) + 1, incr)))
                    elset = np.array(elset)
                else:
                    elset = np.array(
                        [
                            el
                            for line in range(nline + 1, keyword[k + 1][0])
                            for el in inp[line].split()
                        ],
                        dtype=int,
                    )
                idSet = idSet.strip()
                ElementSet[idSet] = elset

            elif key[0:9] == "*equation":
                # Equation.append(np.hstack([inp[line].split() for line in range(nline+1,keyword[k+1][0])]).astype(float))
                nTerms = int(inp[nline + 1])
                eq = np.hstack(
                    [inp[line].split() for line in range(nline + 2, keyword[k + 1][0])]
                ).astype(float)
                listVar = tuple(eq[1::3].astype(int))
                if not (listVar in Equation):
                    Equation[listVar] = []
                Equation[listVar].append(eq)

                # if not(nTerms in Equation): Equation[nTerms] = []
                # Equation[nTerms].append(np.hstack([inp[line].split() for line in range(nline+2,keyword[k+1][0])]).astype(float))

        # self.Equation = Equation #for debug
        self.__Equation = Equation
        self.__Element = Element
        self.__NodeSet = NodeSet
        self.__ElementSet = ElementSet
        self.__NodeCoordinate = NodeCoordinate
        self.__NodeNumber = NodeNumber

        ConvertNodeDict = dict(zip(NodeNumber, list(range(0, len(NodeNumber)))))
        self.__ConvertNode = np.vectorize(ConvertNodeDict.get)  # function

        # self.ConvertNodeDict = ConvertNodeDict
        # self.ConvertNode = self.__ConvertNode
        # self.Element = Element

    def toMesh(self, meshname=None):
        if meshname == None:
            meshname = self.filename
            if meshname[-4:].lower() == ".inp":
                meshname = meshname[:-4]
        for count, dict_elm in enumerate(self.__Element):
            if len(self.__Element) < 2:
                importedMeshName = meshname
            else:
                importedMeshName = meshname + str(count)

            elm = self.__ConvertNode(dict_elm["ElementTable"])
            mesh = Mesh(
                self.__NodeCoordinate,
                elm,
                dict_elm["ElementType"],
                name=importedMeshName,
            )
            # add set of nodes
            for SetOfId in self.__NodeSet:
                NodeIndexes = self.__NodeSet[SetOfId]
                mesh.add_node_set(self.__ConvertNode(NodeIndexes), SetOfId)

            ElementNumber = dict_elm["ElementNumber"]
            ConvertElementDict = dict(
                zip(ElementNumber, list(range(0, len(ElementNumber))))
            )
            ConvertElement = np.vectorize(ConvertElementDict.get)  # function
            for SetOfId in self.__ElementSet:
                ElementIndexes = self.__ElementSet[SetOfId]
                Temp = ConvertElement(ElementIndexes)
                mesh.add_element_set(Temp[Temp != None].astype(int), SetOfId)

        return mesh

    # def applyBoundaryCondition(self, Problemname = "MainProblem"):
    #     for listVar in self.__Equation:
    #         eq = np.array(self.__Equation[listVar])
    #         BoundaryCondition('MPC', listVar, eq[:,2::3], eq[:,0::3].astype(int), Problemname = Problemname)
