from fedoo.core.base import MeshBase
from fedoo.core.mesh import Mesh as MeshFEM
import numpy as np


class MeshPGD(MeshBase):  # class pour définir des maillages sous forme séparées
    """
    PGD.Mesh(mesh1, mesh2, ....)

    A MaillageS object represents a mesh under a separated form.
    The global mesh is defined with a tensorial product of submeshes contained
    in the attribute maillage.
    Each submeshes has its own coordinate system.
    ----------

    """

    def __init__(self, *args, **kargs):
        if "name" in kargs:
            name = kargs["name"]
        else:
            name = "PGDMesh"

        if not isinstance(name, str):
            assert 0, "An name must be a string"

        MeshBase.__init__(self, name)
        self.__ListMesh = [
            MeshBase.get_all()[m] if isinstance(m, str) else m for m in args
        ]

        listCrdname = [crdid for m in self.__ListMesh for crdid in m.crd_name]
        if len(set(listCrdname)) != len(listCrdname):
            print("Warning: some coordinate name are defined in more than one mesh")

        self.node_sets = {}  # node on the boundary for instance
        self.element_sets = {}
        self.__SpecificVariableRank = {}  # to define specific variable rank for each submesh (used for the PGD)

    def _SetSpecificVariableRank(self, idmesh, idvar, specific_rank):
        # idmesh : the id of any submesh
        # idvar : variable id that is given by the ModelingSpace variable_rank(name) method
        #        if idvar == 'default': define the default value for all variables
        # specific_rank : rank considered for the PGD assembly
        # no specific rank can be defined if there is a change of basis in the physicial mesh related to coordinates 'X', 'Y' and 'Z'

        assert isinstance(idmesh, int), "idmesh must an integer, not a " + str(
            type(idmesh)
        )
        assert idvar == "default" or isinstance(
            idvar, int
        ), 'idvar must an integer or "default"'
        assert isinstance(specific_rank, int), (
            "specific_rank must an integer, not a " + str(type(idmesh))
        )

        if idmesh in self.__SpecificVariableRank:
            self.__SpecificVariableRank[idmesh][idvar] = specific_rank
        else:
            self.__SpecificVariableRank[idmesh] = {idvar: specific_rank}

    def _GetSpecificVariableRank(self, idmesh, idvar):
        assert isinstance(idmesh, int), "idmesh must an integer, not a " + str(
            type(idmesh)
        )
        assert idvar == "default" or isinstance(
            idvar, int
        ), 'idvar must an integer or "default"'

        if idmesh in self.__SpecificVariableRank:
            if idvar in self.__SpecificVariableRank[idmesh]:
                return self.__SpecificVariableRank[idmesh][idvar]
            elif "default" in self.__SpecificVariableRank[idmesh]:
                return self.__SpecificVariableRank[idmesh]["default"]
        else:
            return idvar

    def _GetSpecificNumberOfVariables(self, idmesh, nvar):
        assert isinstance(idmesh, int), "idmesh must an integer, not a " + str(
            type(idmesh)
        )
        if idmesh in self.__SpecificVariableRank:
            return max(self.__SpecificVariableRank[idmesh].values()) + 1
        else:
            return nvar

    def get_dimension(self):
        return len(self.__ListMesh)

    def GetListMesh(self):
        return self.__ListMesh

    def add_node_set(self, listNodeIndexes, listSubMesh=None, name=None):
        """
        The Set Of Nodes in Mesh PGD object are used to defined the boundary conditions.
        There is two ways of defining a SetOfNodes:

        PGD.Mesh.add_node_set([nodeIndexes_1,...,nodeIndexes_n ], name =SetOfname)
            * nodeIndexes_i a list of node indexe cooresponding to the ith subMesh (as defined in the constructor of the PGD.Mesh object)
            * nodeIndexes_i can also be set to "all" to indicate that all the nodes have to be included
            * SetOfname is the name of the SetOf

        PGD.Mesh.add_node_set([nodeIndexes_1,...,nodeIndexes_n ], [subMesh_1,...,subMesh_n], name =SetOfname)
            * nodeIndexes_i a list of node indexe cooresponding to the mesh indicated in subMesh_i
            * subMesh_i can be either a mesh name (str object) or a Mesh object
            * the keyword "all" is NOT available when the subMesh are indicated.
            * If a subMesh is not included in listSubMesh, all the Nodes are considered
            * SetOfname is the name of the SetOf
        """
        if name == None:
            num = 1
            while "NodeSet" + str(num) in self.node_sets:
                num += 1
            name = "NodeSet" + str(num)

        if listSubMesh is None:
            if len(listNodeIndexes) != len(self.__ListMesh):
                assert 0, "The lenght of the Node Indexes List must be equal to the number of submeshes"
            listSubMesh = [
                i
                for i in range(len(self.__ListMesh))
                if not (
                    np.array_equal(listNodeIndexes[i], "all")
                    or np.array_equal(listNodeIndexes[i], "ALL")
                )
            ]
            listNodeIndexes = [
                NodeIndexes
                for NodeIndexes in listNodeIndexes
                if not (
                    np.array_equal(NodeIndexes, "all")
                    or np.array_equal(NodeIndexes, "ALL")
                )
            ]
        else:
            # listSubMesh = [self.__ListMesh.index(MeshBase.get_all()[m]) if isinstance(m,str) else self.__ListMesh.index(m) for m in listSubMesh]
            listSubMesh = [
                (
                    self.__ListMesh.index(MeshBase.get_all()[m])
                    if isinstance(m, str)
                    else m
                    if isinstance(m, int)
                    else self.__ListMesh.index(m)
                )
                for m in listSubMesh
            ]

        self.node_sets[name] = [listSubMesh, listNodeIndexes]

    def add_element_set(self, listElementIndexes, listSubMesh=None, name=None):
        """
        See the documention of add_node_set.
        add_element_set is a similar method for Elements.
        """
        if name == None:
            num = 1
            while "ElementSet" + str(num) in self.node_sets:
                num += 1
            name = "ElementSet" + str(num)

        if listSubMesh is None:
            if len(listElementIndexes) != len(self.__ListMesh):
                assert 0, "The lenght of the Node Indexes List must be equal to the number of submeshes"
            listSubMesh = [
                self.__ListMesh[i]
                for i in range(len(self.__ListMesh))
                if listElementIndexes[i] not in ["all", "ALL"]
            ]
            listElementIndexes = [
                ElmIndexes
                for ElmIndexes in listElementIndexes
                if ElmIndexes not in ["all", "ALL"]
            ]
        else:
            listSubMesh = [
                MeshBase.get_all()[m]
                if isinstance(m, str)
                else m
                if isinstance(m, int)
                else self.__ListMesh.index(m)
                for m in listSubMesh
            ]

        self.element_sets[name] = [listSubMesh, listElementIndexes]

    def GetSetOfNodes(self, SetOfId):
        return self.node_sets[SetOfId]

    def GetSetOfElements(self, SetOfId):
        return self.element_sets[SetOfId]

    def RemoveSetOfNodes(self, SetOfId):
        del self.node_sets[SetOfId]

    def RemoveSetOfElements(self, SetOfId):
        del self.element_sets[SetOfId]

    def ListSetOfNodes(self):
        return [key for key in self.node_sets]

    def ListSetOfElements(self):
        return [key for key in self.element_sets]

    def FindCoordinatename(self, crd):
        """
        Try to find a coordinate in the submeshes.
        Return the position of the mesh in the list MeshPGD.GetListMesh() or None if the coordinate is not found
        """
        for idmesh, mesh in enumerate(self.__ListMesh):
            if crd in mesh.crd_name:
                return idmesh
        return None

    def GetNumberOfNodes(self):
        """
        Return a list containing the number of nodes for all submeshes
        """
        return [m.n_nodes for m in self.__ListMesh]

    def GetNumberOfElements(self):
        """
        Return a list containing the number of nodes for all submeshes
        """
        return [m.n_elements for m in self.__ListMesh]

    @staticmethod
    def create(*args, **kargs):
        return MeshPGD(*args, **kargs)

    #    def __getitem__(self, key):
    #        return self.__ListMesh[key]

    #    def append(self, mesh):
    #        """Add a new mesh (submesh of the separated mesh) in the maillage attribute"""
    #        self.__ListMesh.append(mesh)
    #
    #    def __len__(self):
    #        return len(self.__ListMesh) # taille de la liste

    def ExtractFullMesh(self, name="FullMesh", useLocalFrame=False):
        if len(self.__ListMesh) == 3:
            return NotImplemented
        #            mesh1 = self.__ListMesh[0] ; mesh2 = self.__ListMesh[1] ; mesh3 = self.__ListMesh[2]
        #            if mesh1.elm_type == 'lin2' and mesh2.elm_type == 'lin2' and mesh3.elm_type == 'lin2':
        #                elmType = 'hex8'
        #
        #            else: 'element doesnt exist'
        elif len(self.__ListMesh) == 2:
            mesh1 = self.__ListMesh[1]
            mesh0 = self.__ListMesh[0]

            Nel1 = mesh1.n_elements
            Nel0 = mesh0.n_elements
            Nel = Nel1 * Nel0
            ndInElm1 = np.shape(mesh1.elements)[1]
            ndInElm0 = np.shape(mesh0.elements)[1]
            elm = np.zeros((Nel, ndInElm1 * ndInElm0), dtype=int)

            if mesh0.elm_type == "lin2":  # mesh0 is 'lin2'
                dim_mesh0 = 1
                if mesh1.elm_type == "lin2":
                    type_elm = "quad4"
                    dim_mesh1 = 1
                    for i in range(Nel0):
                        elm[i * Nel1 : (i + 1) * Nel1, [0, 1]] = (
                            mesh1.elements + mesh0.elements[i, 0] * mesh1.n_nodes
                        )
                        elm[i * Nel1 : (i + 1) * Nel1, [3, 2]] = (
                            mesh1.elements + mesh0.elements[i, 1] * mesh1.n_nodes
                        )
                elif mesh1.elm_type == "quad4":
                    dim_mesh1 = 2
                    type_elm = "hex8"
                    for i in range(Nel0):
                        elm[i * Nel1 : (i + 1) * Nel1, 0:ndInElm1] = (
                            mesh1.elements + mesh0.elements[i, 0] * mesh1.n_nodes
                        )
                        elm[i * Nel1 : (i + 1) * Nel1, ndInElm1 : 2 * ndInElm1] = (
                            mesh1.elements + mesh0.elements[i, 1] * mesh1.n_nodes
                        )
                else:
                    raise NameError("Element not implemented")

            elif (
                mesh0.elm_type == "lin3"
            ):  # need verification because the node numerotation for lin2 has changed
                dim_mesh0 = 1
                if mesh1.elm_type == "lin3":  # mesh1 and mesh0 are lin3 elements
                    dim_mesh1 = 1
                    type_elm = "quad9"
                    for i in range(
                        Nel0
                    ):  # éléments 1D à 3 noeuds (pour le moment uniquement pour générer des éléments quad9)
                        elm[i * Nel1 : (i + 1) * Nel1, [0, 4, 1]] = (
                            mesh1.elements + mesh0.elements[i, 0] * mesh1.n_nodes
                        )
                        elm[i * Nel1 : (i + 1) * Nel1, [7, 8, 5]] = (
                            mesh1.elements + mesh0.elements[i, 1] * mesh1.n_nodes
                        )
                        elm[i * Nel1 : (i + 1) * Nel1, [3, 6, 2]] = (
                            mesh1.elements + mesh0.elements[i, 2] * mesh1.n_nodes
                        )
                else:
                    raise NameError("Element not implemented")

            elif mesh0.elm_type == "quad4":
                dim_mesh0 = 2
                if mesh1.elm_type == "lin2":
                    dim_mesh1 = 1
                    type_elm = "hex8"
                    for i in range(Nel1):
                        elm[i::Nel1, 0:ndInElm0] = (
                            mesh0.elements * mesh1.n_nodes + mesh1.elements[i, 0]
                        )
                        elm[i::Nel1, ndInElm0 : 2 * ndInElm0] = (
                            mesh0.elements * mesh1.n_nodes + mesh1.elements[i, 1]
                        )

            else:
                raise NameError("Element not implemented")

            if useLocalFrame == False:
                Ncrd = mesh1.n_nodes * mesh0.n_nodes
                #                crd = np.c_[np.tile(mesh1.nodes[:,:dim_mesh1],(mesh0.n_nodes,1)), \
                #                            np.reshape([np.ones((mesh1.n_nodes,1))*mesh0.nodes[i,:dim_mesh0] for i in range(mesh0.n_nodes)] ,(Ncrd,-1)) ]
                crd = np.c_[
                    np.reshape(
                        [
                            np.ones((mesh1.n_nodes, 1)) * mesh0.nodes[i, :dim_mesh0]
                            for i in range(mesh0.n_nodes)
                        ],
                        (Ncrd, -1),
                    ),
                    np.tile(mesh1.nodes[:, :dim_mesh1], (mesh0.n_nodes, 1)),
                ]
            elif dim_mesh0 == 1:  # dim_mesh0 is the thickness
                crd = np.zeros(
                    (mesh1.n_nodes * mesh0.n_nodes, np.shape(mesh1.nodes)[1])
                )
                for i in range(mesh0.n_nodes):
                    crd[i * mesh1.n_nodes : (i + 1) * mesh1.n_nodes, :] = (
                        mesh1.nodes + mesh1.local_frame[:, -1, :] * mesh0.nodes[i][0]
                    )
            else:
                return NotImplemented

            return MeshFEM(crd, elm, type_elm, name=name)

        elif len(self.__ListMesh) == 1:
            return self.__ListMesh[0]
        else:
            raise NameError(
                "FullMesh can only be extracted from Separated Mesh of dimenson <= 3"
            )

    @property
    def n_nodes(self):
        return self.GetNumberOfNodes()
