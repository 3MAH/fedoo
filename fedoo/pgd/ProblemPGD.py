import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg

# from fedoo.pgd.problemPGD.BoundaryConditionPGD import *
from fedoo.core.base import ProblemBase
from fedoo.core.boundary_conditions import BoundaryCondition
from fedoo.pgd.SeparatedArray import SeparatedArray, SeparatedOnes, SeparatedZeros
from fedoo.problem.explicit_dynamic import _ExplicitDynamicBase
from fedoo.problem.linear import _LinearBase
from fedoo.problem.newmark import _NewmarkBase
from fedoo.problem.non_linear import _NonLinearBase

# ===============================================================================
# Classes permettant de définir un problème sous forme discrète (forme séparée)
# ===============================================================================


class ProblemPGD(ProblemBase):
    def __init__(self, A, B, D, Mesh, name="MainProblem", space=None):
        # the problem is AX = B + D
        ProblemBase.__init__(self, name, space)

        nvar = self.space.nvar
        self.mesh = Mesh
        self.__NumberOfSubspace = Mesh.get_dimension()

        listNumberOfNodes = Mesh.n_nodes  # list of number of nodes for all submesh
        # self.__ModelingSpace is a list of number of DoF for each subspace
        self.__ModelingSpace = [
            Mesh._GetSpecificNumberOfVariables(idmesh, nvar) * listNumberOfNodes[idmesh]
            for idmesh in range(self.__NumberOfSubspace)
        ]
        # self.__ModelingSpace = self.__A.GetShape()

        self.__A = A
        # if self.__A != 0: self.__A.tocsr() #just in case A is in another format as csr

        if np.isscalar(B) and B == 0:
            self.__B = self._new_vect_dof()
        else:
            self.__B = B

        self.__D = D

        self.__DofBlocked = None
        self.__DofFree = None
        #    self.__A_BoundaryCondition = None
        self.__X = 0
        self.__Xbc = 0  # for boundary conditions

    # ===============================================================================
    # Internal Functions
    # ===============================================================================
    def _new_vect_dof(
        self,
    ):  # initialize a vector (force vector for instance) being giving the stiffness matrix
        return SeparatedZeros(self.__ModelingSpace)

    def _set_vect_component(
        self, vector, name, value
    ):  # initialize a vector (force vector for instance) being giving the stiffness matrix
        assert isinstance(name, str), "argument error"

        if name.lower() == "all":
            vector = value
        else:
            return NotImplemented

    #
    #            i = self.space.variable_rank(name)
    #
    #            n = Problem.__Mesh.n_nodes
    #
    #            NewmarkPGD.__Xdotdot[i*n : (i+1)*n] = value

    def _get_vect_component(self, vector, name):
        assert isinstance(name, str), "argument error"

        if name.lower() == "all":
            return vector

        else:
            return NotImplemented

    def SecTerm_RS(
        self, R, d1
    ):  # détermine le second terme du problème restreint à la dimension d1 et connaissant R
        prod_aux_3 = sp.ones((R.nbTerm(), self.__C.nbTerm()))
        for d2 in list(range(d1)) + list(range(d1 + 1, len(self.__C))):
            prod_aux_3 = prod_aux_3 * (sp.dot(R.data[d2].T, self.__C.data[d2]))
        return sp.reshape(
            sp.dot(prod_aux_3, self.__C.data[d1].T) @ self.__MatCB[d1], (-1, 1)
        )

    def calcMat_RS(
        self, R, d1
    ):  # détermine la matrice équivalente sur la dimension d1 connaissant R
        N_d1 = self.__A.GetShape()[d1]

        # ===============================================================================
        #         PGD standard(galerkin)
        # ===============================================================================
        # Possibilité d'optimisation à tester
        # 1 - utiliser un tableau de sparse matrix pour vectorisé les opérations
        # 2 - faire un assemblage de toutes les matrices creuses dans une grande matrice creuse par bloc pour vectoriser les opérations

        # MatCB = self.__MatCB[ddcalc]
        # self.__X.data[ddcalc][self.__DofFree[ddcalc].reshape(-1,1),termToChange]  = np.reshape(self._ProblemBase__Solve(MatCB[ddcalc].T @ M @ MatCB[ddcalc], self.__MatCB.T @ V ) , (len(termToChange), -1)).T

        MatCB = self.__MatCB[d1]

        if R.nbTerm() == 1:
            Mat_K = sparse.csr_matrix((N_d1, N_d1))
            for kk in range(self.__A.NumberOfOperators()):
                prod_aux = 1
                for d2 in list(range(d1)) + list(
                    range(d1 + 1, self.__A.get_dimension())
                ):
                    prod_aux = prod_aux * sp.dot(
                        R.data[d2].T, self.__A.data[kk][d2] * R.data[d2]
                    )
                Mat_K = Mat_K + self.__A.data[kk][d1] * float(prod_aux)
            return MatCB.T @ Mat_K @ MatCB
        else:
            nbTerm = R.nbTerm()
            Mat_K = [
                [sparse.csr_matrix((N_d1, N_d1)) for j in range(nbTerm)]
                for i in range(nbTerm)
            ]
            for kk in range(self.__A.NumberOfOperators()):
                prod_aux = 1
                for d2 in list(range(d1)) + list(
                    range(d1 + 1, self.__A.get_dimension())
                ):
                    prod_aux = prod_aux * sp.dot(
                        R.data[d2].T, self.__A.data[kk][d2] * R.data[d2]
                    )
                Mat_K = [
                    [
                        Mat_K[i][j] + self.__A.data[kk][d1] * prod_aux[i, j]
                        for j in range(nbTerm)
                    ]
                    for i in range(nbTerm)
                ]

            Mat_K = [
                [MatCB.T @ Mat_K[i][j] @ MatCB for j in range(nbTerm)]
                for i in range(nbTerm)
            ]
            Mat_K = sparse.bmat(Mat_K, format="csr")
            return Mat_K

            # Mat_K  = sparse.csr_matrix((R.nbTerm()*N_d1 , R.nbTerm()*N_d1))
            # for kk in range(self.__A.NumberOfOperators()):
            #     prod_aux = 1
            #     for d2 in list(range(d1))+list(range(d1+1,self.__A.get_dimension())):
            #         prod_aux = prod_aux * sp.dot(R.data[d2].T,self.__A.data[kk][d2]*R.data[d2])
            #     Mat_K = Mat_K + sparse.bmat([[self.__A.data[kk][d1] * prod_aux[i,j] for j in range(R.nbTerm())] for i in range(R.nbTerm())])
            # return Mat_K

    #    def copy(self):
    #        return self.__class__(self)

    def SecTerm_Alpha(
        self, FF, BB
    ):  # détermine le second terme du calcul des alpha connaissant FF
        V = 1
        for dd in range(BB.dim):
            if sparse.issparse(
                FF.data[dd]
            ):  # bug fonction dot pour sparse matrice (utile pour calcul FE)
                V = V * (FF.data[dd].T * BB.data[dd])
            else:
                V = V * sp.dot(FF.data[dd].T, BB.data[dd])
        return sp.sum(V, 1)

    def calcMat_Alpha(
        self, FF
    ):  # détermine la matrice équivalente sur la dimension pour le calcul des alpha
        # traitement différent si FF contient des matrices creuses
        nbFF = FF.nbTerm()
        if sparse.issparse(
            FF.data[0]
        ):  # si FF contient des matrices creuses (cas particulier)
            Mat_K = sparse.csr_matrix((nbFF, nbFF))
            for Op in self.__A.data:
                M = FF.data[0].T * Op[0] * FF.data[0]
                for dd in range(1, self.__A.dim):
                    M = sparse.csr_matrix.multiply(
                        M, FF.data[dd].T * Op[dd] * FF.data[dd]
                    )
                Mat_K = Mat_K + M
            return Mat_K  # au format csr
        else:  # si FF contient des matrices pleines (cas général)
            Mat_K = 0
            for Op in self.__A.data:
                M = 1
                for dd in range(self.__A.get_dimension()):
                    M = M * sp.dot(FF.data[dd].T, Op[dd] * FF.data[dd])
                Mat_K += M
            return Mat_K

    def get_B(self):
        return self.__B

    def set_D(self, D):
        self.__D = D

    def get_X(self):
        return self.__X

    def get_Xbc(self):
        return self.__Xbc

    def get_dof_solution(self, name="all"):
        assert isinstance(name, str), "argument error"

        if name.lower() == "all":
            return self.__X + self.__Xbc

        if self.__Xbc == 0:
            return self.__X.GetVariable(self.space.variable_rank(name), self.mesh)
        else:
            return self.__X.GetVariable(
                self.space.variable_rank(name), self.mesh
            ) + self.__Xbc.GetVariable(self.space.variable_rank(name), self.mesh)

    def set_DoFSolution(self, name, value):
        assert isinstance(name, str), "argument error"

        if name.lower() == "all":
            self.__X = value
        else:
            return NotImplemented

    #            i = self.space.variable_rank(name)
    #
    #            self.__X[i*n : (i+1)*n] = value

    def SetInitialBCToCurrent(self):
        pass  # do nothing. Only applyed for FEM problem

    # ===============================================================================
    # Fonction to build the PGD solution
    # ===============================================================================

    def ComputeResidualNorm(self, err_0=None):
        if err_0 == None:
            err_0 = 1
            #            if self.err_0 == None: err_0 = 1
            #            else: err_0 = self.err_0
        if self.__Xbc == 0 and self.__X == 0:
            res = self.__B + self.__D
        else:
            res = self.__B + self.__D - self.__A * (self.__Xbc + self.__X)

        # CL à intégrer???
        for dd in range(self.mesh.get_dimension()):
            res.data[dd][self.__DofBlocked[dd]] = 0
        return res.norm(nbvar=self.space.nvar) / err_0

    def GetResidual(self):
        if self.__Xbc == 0 and self.__X == 0:
            res = self.__B + self.__D
        else:
            res = self.__B + self.__D - self.__A * (self.__Xbc + self.__X)

        # CL à intégrer???
        for dd in range(self.mesh.get_dimension()):
            res.data[dd][self.__DofBlocked[dd]] = 0
        return res

    def update_pgd(self, termToChange, ddcalc="all"):  # extended PGD
        if ddcalc == "all":
            for nMesh in range(self.mesh.get_dimension()):
                self.updatePGD(termToChange, nMesh)
            return

        termToKeep = [t for t in range(self.__X.nbTerm()) if not t in termToChange]

        # --- Initialisation du second membre (BB/CC/SecTerm) --- #
        if termToKeep == []:
            self.__C = self.__B + self.__D - self.__A * self.__Xbc
        else:
            self.__C = (
                self.__B
                + self.__D
                - self.__A * (self.__X.getTerm(termToKeep) + self.__Xbc)
            )

            #        for dd in range(self.__NumberOfSubspace): self.__C.data[dd][self.__CL[dd],:] = 0

        RS = self.__X.getTerm(termToChange)  # value of the solution to update
        M = self.calcMat_RS(RS, ddcalc)
        V = self.SecTerm_RS(RS, ddcalc)

        NbDoF = self.__C.shape[ddcalc]
        DofFree = np.hstack(
            [self.__DofFree[ddcalc] + i * NbDoF for i in range(len(termToChange))]
        )

        self.__X.data[ddcalc][self.__DofFree[ddcalc].reshape(-1, 1), termToChange] = (
            np.reshape(self._solve(M, V), (len(termToChange), -1)).T
        )
        # self.__X.data[ddcalc][self.__DofFree[ddcalc].reshape(-1,1),termToChange]  = np.reshape(self._ProblemBase__Solve(M[DofFree.reshape(-1,1),DofFree], V[DofFree] ) , (len(termToChange), -1)).T

    def update_alpha(self):
        BB = SeparatedArray(self.__B + self.__D - self.__A * self.__Xbc)
        alpha = sp.c_[
            linalg.solve(self.calcMat_Alpha(self.__X), self.SecTerm_Alpha(self.__X, BB))
        ]
        #        alpha = self.solve_Alpha(self.__X,SeparatedArray(self.__B - self.__A*self.__Xbc))
        self.__X.data[0] = (
            sp.tile(alpha.T, (self.__ModelingSpace[0], 1)) * self.__X.data[0]
        )

    def AddNewTerm(self, numberOfTerm=1, value=None, variable="all"):
        if variable != "all":
            return NotImplemented
        if value == None:
            self.__X += SeparatedArray(
                [np.random.random((nn, numberOfTerm)) for nn in self.__ModelingSpace]
            )
        elif isinstance(value, (int, float)):
            self.__X += value * SeparatedOnes(self.__ModelingSpace, numberOfTerm)
        elif isinstance(value, SeparatedArray):
            for t in range(numberOfTerm):
                self.__X += value.getTerm(t % value.nbTerm())
        # for boundary conditions
        for dd in range(self.mesh.get_dimension()):
            self.__X.data[dd][self.__DofBlocked[dd]] = 0

    # ===============================================================================
    # Fonctions for Boundary Conditions
    # ===============================================================================
    # TODO
    # verifier l'utlisation de var dans boundary conditions PGD
    # reprendre les conditions aux limites en incluant les méthodes de pénalités pour des conditions aux limites plus exotiques
    # verifier qu'il n'y a pas de probleme lié au CL sur les ddl inutiles
    def apply_boundary_conditions(self, t_fact=1, t_fact_old=None):
        meshPGD = self.mesh
        shapeX = self.__ModelingSpace
        X = self.__X
        Xbc = 0  # SeparatedZeros(shapeX)
        F = 0
        nvar = self.space.nvar

        DofB = [np.array([]) for i in self.__ModelingSpace]
        dimBC = None  # dimension requiring a modification of Xbc - dimBC = None if Xbc is never modified, dimBC = dd if only the dimension dd is modified, dimBC = 'many' if there is more than one dimension

        MPC = False
        data = [[] for i in self.__ModelingSpace]
        row = [[] for i in self.__ModelingSpace]
        col = [[] for i in self.__ModelingSpace]

        Nnd = [
            meshPGD.GetListMesh()[d].n_nodes for d in range(meshPGD.get_dimension())
        ]  # number of nodes in each dimensions
        Nvar = [
            meshPGD._GetSpecificNumberOfVariables(d, nvar)
            for d in range(meshPGD.get_dimension())
        ]

        for e in self.bc.generate(self, t_fact, t_fact_old):
            # SetOfNodesForBC = meshPGD.node_sets[e.SetOfname]

            if e.bc_type == "Neumann":
                SetOfNodesForBC = e.node_set
                Value = e._current_value
                if Value == 0:
                    continue  # dans ce cas, pas de force à ajouter
                    #                dd = SetOfNodesForBC[0][0]
                    #                index = SetOfNodesForBC[1][0]
                var = [
                    meshPGD._GetSpecificVariableRank(d, e.variable)
                    for d in range(meshPGD.get_dimension())
                ]  # specific variable rank related to the submesh dd

                # item = the index of nodes in each subspace (slice if all nodes are included)
                item = [
                    slice(Nnd[d] * var[d], Nnd[d] * (var[d] + 1))
                    for d in range(meshPGD.get_dimension())
                ]
                for i, d in enumerate(SetOfNodesForBC[0]):
                    index = np.array(SetOfNodesForBC[1][i], dtype=int)
                    item[d] = var[d] * Nnd[d] + index

                if isinstance(Value, np.ndarray):
                    Value = SeparatedArray([Value.reshape(-1, 1)])
                if isinstance(Value, SeparatedArray):
                    if len(Value) != meshPGD.get_dimension():
                        if len(Value) == len(SetOfNodesForBC[1]):
                            nbt = Value.nbTerm()
                            Value = SeparatedArray(
                                [
                                    Value.data[SetOfNodesForBC[0].index(d)]
                                    if d in SetOfNodesForBC[0]
                                    else np.ones((1, nbt))
                                    for d in range(len(shapeX))
                                ]
                            )
                        else:
                            assert 0, "Dimension doesn't match"

                if np.isscalar(F) and F == 0:
                    if isinstance(Value, (float, int, np.floating)):
                        Fadd = SeparatedZeros(shapeX)
                        # for d in range(meshPGD.get_dimension()): Fadd.data[d][item[d]] = Value
                        Fadd.data[0][item[0]] = Value
                        for d in range(1, meshPGD.get_dimension()):
                            Fadd.data[d][item[d]] = 1.0
                    else:
                        Fadd = SeparatedZeros(shapeX, nbTerm=Value.nbTerm())
                        Fadd.data[0][item[0]] = Value.data[d]
                        for d in range(1, meshPGD.get_dimension()):
                            Fadd.data[d][item[d]] = Value.data[d]
                    F = F + Fadd
                else:
                    F.__setitem__(tuple(item), Value)

            elif e.bc_type == "Dirichlet":
                SetOfNodesForBC = e.node_set
                Value = e._current_value
                if len(SetOfNodesForBC[1]) == 1 and isinstance(
                    Value, (int, float, np.floating, np.ndarray)
                ):  # The BC can be applied on only 1 subspace
                    dd = SetOfNodesForBC[0][0]
                    index = np.array(SetOfNodesForBC[1][0], dtype=int)
                    var = meshPGD._GetSpecificVariableRank(
                        dd, e.variable
                    )  # specific variable rank related to the submesh dd
                    GlobalIndex = (var * Nnd[dd] + index).astype(int)

                    if isinstance(Value, np.ndarray):
                        Value = Value.reshape(-1, 1)

                    DofB[dd] = np.hstack((DofB[dd], GlobalIndex))
                    if dimBC is None:  # initialization of Xbc
                        if not np.array_equal(
                            Value, 0
                        ):  # modification of the second term Xbc
                            dimBC = dd
                            Xbc = SeparatedArray(
                                [
                                    np.ones((shapeX[d], 1))
                                    if d != dd
                                    else np.zeros((shapeX[d], 1))
                                    for d in range(len(shapeX))
                                ]
                            )
                            Xbc.data[dd][GlobalIndex] = Value
                    elif dd == dimBC:  # in this case, the definition of Xbc is trivial
                        Xbc.data[dd][GlobalIndex] = Value
                    else:  # many dimension required the modification of BC
                        dimBC = "many"
                        Xbc_old = Xbc.copy()
                        Xbc_old.data[dd] = 0 * Xbc_add.data[dd]
                        Xbc_old.data[dd][GlobalIndex] = Xbc.data[dd][GlobalIndex]
                        Xbc_add = SeparatedArray(
                            [
                                np.ones((shapeX[d], 1))
                                if d != dd
                                else np.zeros((shapeX[d], 1))
                                for d in range(len(shapeX))
                            ]
                        )
                        Xbc_add.data[dd][GlobalIndex] = Value
                        Xbc = Xbc + Xbc_add - Xbc_old

                else:  # a penatly method is required
                    return NotImplemented

            elif e.bc_type == "MPC":
                # SetOfNodesForBC_Master = [meshPGD.node_sets[setofid] for setofid in e.SetOfnameMaster]
                SetOfNodesForBC_Master = e.list_node_sets[1:]
                SetOfNodesForBC = e.list_node_sets[0]

                # test if The BC can be applied on only 1 subspace, ie if each setofnodes is defined only on 1 same subspace
                if (
                    len(SetOfNodesForBC[1]) == 1
                    and all(len(setof[1]) == 1 for setof in SetOfNodesForBC_Master)
                    and all(
                        setof[0][0] == SetOfNodesForBC[0][0]
                        for setof in SetOfNodesForBC_Master
                    )
                ):
                    # isinstance(Value, (int,float,np.floating,np.ndarray)):

                    dd = SetOfNodesForBC[0][0]  # the subspace involved
                    Index = np.array(SetOfNodesForBC[1][0], dtype=int)
                    IndexMaster = np.array(
                        [setof[1][0] for setof in SetOfNodesForBC_Master], dtype=int
                    )

                    # global index for the slave nodes (eliminated nodes)
                    var = meshPGD._GetSpecificVariableRank(
                        dd, e.list_variables[0]
                    )  # specific variable rank related to the submesh dd
                    GlobalIndex = (var * Nnd[dd] + np.array(Index)).astype(int)

                    # add the eliminated node to the list of eliminated nodes
                    DofB[dd] = np.hstack((DofB[dd], GlobalIndex))

                    MPC = True  # need to compute a MPC change of base matrix

                    # Value treatment
                    if isinstance(Value, np.ndarray):
                        Value = Value.reshape(-1, 1)

                    if dimBC is None:  # initialization of Xbc
                        if not np.array_equal(
                            Value, 0
                        ):  # modification of the second term Xbc
                            dimBC = dd
                            Xbc = SeparatedArray(
                                [
                                    np.ones((shapeX[d], 1))
                                    if d != dd
                                    else np.zeros((shapeX[d], 1))
                                    for d in range(len(shapeX))
                                ]
                            )
                            Xbc.data[dd][GlobalIndex] = Value
                    elif dd == dimBC:  # in this case, the definition of Xbc is trivial
                        Xbc.data[dd][GlobalIndex] = Value
                    else:  # many dimension required the modification of BC
                        dimBC = "many"
                        Xbc_old = Xbc.copy()
                        Xbc_old.data[dd] = 0 * Xbc_add.data[dd]
                        Xbc_old.data[dd][GlobalIndex] = Xbc.data[dd][GlobalIndex]
                        Xbc_add = SeparatedArray(
                            [
                                np.ones((shapeX[d], 1))
                                if d != dd
                                else np.zeros((shapeX[d], 1))
                                for d in range(len(shapeX))
                            ]
                        )
                        Xbc_add.data[dd][GlobalIndex] = Value
                        Xbc = Xbc + Xbc_add - Xbc_old

                    nbFact = len(e._factors)

                    # shape self.Factor should be nbFact*nbMPC
                    # shape self.Index should be nbMPC
                    # shape self.IndexMaster should be nbFact*nbMPC
                    data[dd].append(np.array(e._factors.T).ravel())
                    row[dd].append(
                        (GlobalIndex.reshape(-1, 1) * np.ones(nbFact)).ravel()
                    )
                    col[dd].append(
                        (IndexMaster + np.c_[e.list_variables[1:]] * Nnd[dd]).T.ravel()
                    )

                else:  # a penatly method is required
                    return NotImplemented

            else:
                assert 0, "Boundary type non recognized"

        #        if F == 0: F = SeparatedZeros(shapeX)

        DofB = [
            np.unique(dofb).astype(int) for dofb in DofB
        ]  # bloqued DoF for all the submeshes
        DofL = [
            np.setdiff1d(range(shapeX[d]), DofB[d]).astype(int)
            for d in range(meshPGD.get_dimension())
        ]  # free dof for all the submeshes

        if X != 0:
            for d in range(meshPGD.get_dimension()):
                X.data[dd][DofB[d]] = 0

        # build matrix MPC
        if MPC:
            # Treating the case where MPC includes some blocked nodes as master nodes
            # M is a matrix such as Ublocked = M@U + Uimp
            # Compute M + M@M

            listM = [
                sparse.coo_matrix(
                    (np.hstack(data[d]), (np.hstack(row[d]), np.hstack(col[d]))),
                    shape=(Nvar[d] * Nnd[d], Nvar[d] * Nnd[d]),
                )
                if len(data[d]) > 0
                else sparse.coo_matrix((Nvar[d] * Nnd[d], Nvar[d] * Nnd[d]))
                for d in range(meshPGD.get_dimension())
            ]

            Xbc = SeparatedArray(
                [
                    Xbc.data[d] + listM[d] @ Xbc.data[d]
                    for d in range(meshPGD.get_dimension())
                ]
            )
            listM = [(M + M @ M).tocoo() for M in listM]

            data = [M.data for M in listM]
            row = [M.row for M in listM]
            col = [M.col for M in listM]

            # modification col numbering from DofL to np.arange(len(DofL))
            for d in range(meshPGD.get_dimension()):
                if len(DofB[d]) > 0:  # no change if there is no blocked dof
                    changeInd = np.full(
                        Nvar[d] * Nnd[d], np.nan
                    )  # mettre des nan plutôt que des zeros pour générer une erreur si pb
                    changeInd[DofL[d]] = np.arange(len(DofL[d]))
                    col[d] = changeInd[
                        np.hstack(col[d])
                    ]  # need hstack here ? Not sure because it should have already been done
                    mask = np.logical_not(np.isnan(col[d]))  # mask to delete nan value

                    col[d] = col[d][mask]
                    row[d] = row[d][mask]
                    data[d] = data[d][mask]

        # #adding identity for free nodes
        col = [
            np.hstack((col[d], np.arange(len(DofL[d]))))
            for d in range(meshPGD.get_dimension())
        ]
        row = [np.hstack((row[d], DofL[d])) for d in range(meshPGD.get_dimension())]

        data = [
            np.hstack((data[d], np.ones(len(DofL[d]))))
            for d in range(meshPGD.get_dimension())
        ]

        MatCB = [
            sparse.coo_matrix(
                (data[d], (row[d], col[d])), shape=(Nvar[d] * Nnd[d], len(DofL[d]))
            ).tocsr()
            for d in range(meshPGD.get_dimension())
        ]

        self.__X = X
        self.__Xbc = Xbc
        self.__B = F
        self.__DofBlocked = DofB
        self.__DofFree = DofL
        self.__MatCB = MatCB

    ### Functions related to boundary contidions
    def BoundaryCondition(
        self,
        bc_type,
        Var,
        Value,
        Index,
        Constant=None,
        timeEvolution=None,
        initialValue=None,
        name="",
    ):
        """
        Define some boundary conditions

        Parameters
        ----------
        bc_type : str
            Type of boundary conditions : 'Dirichlet', 'Neumann' or 'MPC' for multipoint constraints.
        Var : str, list of str, or list of int
            variable name (str) or list of variable name or for MPC only, list of variable rank
        Value : scalar or array or list of scalars or list of array
            Variable final value (Dirichlet) or force Value (Neumann) or list of factor (MPC)
            For Neumann and Dirichlet, if Var is a list of str, Value may be :
                (i) scalar if the same Value is applied for all Variable
                (ii) list of scalars, if the scalar values are different for all Variable (in this case the len of Value should be equal to the lenght of Var)
                (iii) list of arrays, if the scalar Value is potentially different for all variables and for all indexes. In this case, Value[num_var][i] should give the value of the num_var variable related to the node i.
        Index : list of int, str, list of list of int, list of str
            For FEM Problem with Neumann/Dirichlet BC: Nodes Index (list of int)
            For FEM Problem with MPC: list Node Indexes (list of list of int)
            For PGD Problem with Neumann/Dirichlet BC: SetOfname (type str) defining a set of Nodes of the reference mesh
            For PGD Problem with MPC: list of SetOfname (str)
        Constant : scalar, optional
            For MPC only, constant value on the equation
        timeEvolution : function
            Function that gives the temporal evolution of the BC Value (applyed as a factor to the specified BC). The function y=f(x) where x in [0,1] and y in [0,1]. For x, 0 denote the begining of the step and 1 the end.
        initialValue : float, array or None
            if None, the initialValue is keep to the current state.
            if scalar value: The initialValue is the same for all dof defined in BC
            if array: the len of the array should be = to the number of dof defined in the BC

            Default: None
        name : str, optional
            Define an name for the Boundary Conditions. Default is "No name". The same name may be used for several BC.

        Returns
        -------
        None.

        Remark
        -------
        To define many MPC in one operation, use array where each line define a single MPC
        """
        if isinstance(Var, str) and Var not in self.space.list_variables():
            # we assume that Var is a Vector
            try:
                Var = [
                    self.space.variable_name(var_rank)
                    for var_rank in self.space.get_vector(Var)
                ]
            except:
                raise NameError("Unknown variable name")

        if isinstance(Var, list) and bc_type != "MPC":
            if np.isscalar(Value):
                Value = [Value for var in Var]
            for i, var in enumerate(Var):
                self.bc.append(
                    BoundaryCondition(
                        bc_type, var, Value[i], Index, timeEvolution, initialValue, name
                    )
                )
        else:
            if bc_type == "MPC":
                self.bc.append(
                    MPC(Var, Value, Index, Constant, timeEvolution, initialValue, name)
                )
            else:
                self.bc.append(
                    BoundaryCondition(
                        bc_type, Var, Value, Index, timeEvolution, initialValue, name
                    )
                )


##    def CL_ml(self, dd, QQ): #conditions aux limites via des multiplicateurs de lagrange
##        #QQ matrice définissant les CL
##        #à développer
##        pass
#
#    def linkNodes(self, dd, nds, var): #lient des noeuds (CL périodiques)
#        """
#        Apply Periodic Boundary Conditions
#
#        Parameters
#        ----------
#        dd : int
#            id of the considered space
#        nds : list of list or arrays of integers
#            Each element on the list is a list of two nodes to link
#        var : int (default = 0)
#            id of variable to applied the boundary conditions
#
#        See also
#        ----------
#        addCL : Method used to apply Dirichelet boundary conditions
#        """
#
#        #vérifier le fonctionnement avec EqD_min
#        #Non optimisée
#        N = self.N[dd] #nombre de noeuds
#        self.CL[dd].extend(var*N + sp.array(nds)[:,1])
#
#        for nd in nds:
#            for Op in self.AA.data:
#                Op[dd] = Op[dd].tolil()
#                Op[dd][var*N +nd[0],:] += Op[dd][var*N + nd[1],:]
#                Op[dd].rows[var*N + nd[1]]=[]
#                Op[dd].data[var*N + nd[1]]=[]
#            self.AA[0][dd].rows[var*N + nd[1]] = [var*N + nd[0], var*N + nd[1]]
#            self.AA[0][dd].data[var*N + nd[1]] = [-1, 1]
#
#
#
#
#
#
# class EqD_min(EqD): #Equation discrete avec minimisation du résidu
#    def SecTerm_RS(self,R,d1): #détermine le vecteur équivalent sur la dimension d1 connaissant R
#        Vec_F = sp.zeros((self.CC.NN[d1], 1))
#        for kk in range(self.AA.NumberOfOperators()):
#            prod_aux_3 = sp.ones((1, self.CC.nbTerm()))
#            for d2 in range(d1)+range(d1+1,len(self.CC)):
#                prod_aux_3 = prod_aux_3 * (sp.dot(R.data[d2].T,self.AA.data[kk][d2].T*self.CC.data[d2]))
#            Vec_F += sp.dot(self.AA.data[kk][d1].T*self.CC.data[d1], prod_aux_3.T)
#        return Vec_F
#
#    def SecTerm_Alpha(self,FF,BB): #détermine le vecteur équivalent pour calcul des alpha
#        # ne marche pas !!!
#        SecM = 0
#        for Op in self.AA.data:
#            V=1
#            for dd in range(BB.dim):
#                V = V*sp.dot(FF.data[dd].T , Op[dd].T*BB.data[dd])
#            SecM += sp.sum(V,1)
#        return SecM
#
#    def calcMat_RS(self, R, d1): #détermine la matrice équivalente sur la dimension d1 connaissant R
#        N_d1 = self.AA.NN[d1]
#        Mat_K  = sparse.csr_matrix((N_d1 , N_d1))
#
#        #===============================================================================
#        #         Symétrisation de l'opérateur (minimisation du résidu)
#        #===============================================================================
#        prod_aux = sp.ones((self.AA.NumberOfOperators(), self.AA.NumberOfOperators()))
#        AA_R = self.AA*R #on n'est pas obligé de calculer AA_R pour la dimension d1 (à optimiser)
#        for d2 in range(d1)+range(d1+1,self.AA.dim):
#            prod_aux = prod_aux * sp.dot(AA_R.data[d2].T , AA_R.data[d2])
#        for k1 in range(self.AA.NumberOfOperators()):
#            Mat_K = Mat_K + (self.AA.data[k1][d1].T * self.AA.data[k1][d1]) * float(prod_aux[k1][k1])
#            for k2 in range(k1+1,self.AA.NumberOfOperators()):
#                M = self.AA.data[k2][d1].T*self.AA.data[k1][d1]
#                Mat_K = Mat_K + (M+M.T) * float(prod_aux[k1][k2])
#        return Mat_K
#
#    def MatEquiv_Alpha(self, FF): #détermine la matrice équivalente pour le calcul des alpha
#        # ne marche pas !!!
#        #Pas optimisé
#        Mat_K = 0
#        for Op in self.AA.data:
#            for Op2 in self.AA.data:
#                M=1
#                for dd in range(FF.dim):
#                    M = M*sp.dot(FF.data[dd].T * Op2[dd].T , Op[dd]*FF.data[dd])
#                Mat_K +=M
#        return Mat_K
#
#


class Linear(_LinearBase, ProblemPGD):
    pass


class ExplicitDynamic(_ExplicitDynamicBase, ProblemPGD):
    pass


class NonLinear(_NonLinearBase, ProblemPGD):
    pass


class Newmark(_NewmarkBase, ProblemPGD):
    pass


# def get_Xbc(): return ProblemPGD.get_all()["MainProblem"].get_Xbc()
# def ComputeResidualNorm(err_0=None): return ProblemPGD.get_all()["MainProblem"].ComputeResidualNorm(err_0)
# def GetResidual(): return ProblemPGD.get_all()["MainProblem"].GetResidual()
# def updatePGD(termToChange, ddcalc='all'): return ProblemPGD.get_all()["MainProblem"].updatePGD(termToChange, ddcalc)
# def updateAlpha(): return ProblemPGD.get_all()["MainProblem"].updateAlpha()
# def AddNewTerm(numberOfTerm = 1, value = None, variable = 'all'): return ProblemPGD.get_all()["MainProblem"].AddNewTerm(numberOfTerm, value, variable)
