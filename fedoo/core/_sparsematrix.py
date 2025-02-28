import numpy as np
from scipy import sparse
from numbers import Number


class _BlocSparse:
    def __init__(
        self,
        nbBlocRow,
        nbBlocCol,
        nbpg=None,
        savedBlocStructure=None,
        assume_sym=False,
    ):
        # savedBlocStructure are data from similar structured blocsparse that avoid time consuming operations
        self.data = [[None for i in range(nbBlocCol)] for j in range(nbBlocRow)]
        self.isempty = True
        if savedBlocStructure is None:
            # array for bloc coo structure
            self.col = None
            self.row = None

            # shape of a single bloc
            self.blocShape = None

            # matrix to convert coo data to csr data
            self.Matrix_convertCOOtoCSR = None

            # array for csr structure (bloc or whole depending on the choosen model)
            self.indices_csr = None
            self.indptr_csr = None
        else:
            self.col = savedBlocStructure["col"]
            self.row = savedBlocStructure["row"]
            self.blocShape = savedBlocStructure["blocShape"]
            self.Matrix_convertCOOtoCSR = savedBlocStructure["Matrix_convertCOOtoCSR"]
            self.indices_csr = savedBlocStructure["indices_csr"]
            self.indptr_csr = savedBlocStructure["indptr_csr"]

        self.nbBlocRow = nbBlocRow
        self.nbBlocCol = nbBlocCol
        self.nbpg = nbpg
        self.__assume_sym = assume_sym

    #    def addToBloc(self, Mat, rowBloc, colBloc):
    #        #Mat should be a scipy matrix using the csr format
    #        bloc = self.data[rowBloc][colBloc]
    #        if np.isscalar(bloc) and bloc == 0:
    #            self.data[rowBloc][colBloc] = Mat.copy()
    #            self.data_coo[rowBloc][colBloc] = self.data[row][col].tocoo(copy = False)
    #            #row are sorted for data_coo
    #        else:
    #            bloc.data = bloc.data + Mat.data

    def addToBlocATB(self, A, B, coef, rowBloc, colBloc, mat_lumping=False):
        # A and B should be scipy matrix using the csr format and with the same number of column per row for each row
        # A and coef may be a list. In this case compute sum([coef[ii]*A[ii] for ii in range(len(A))]).T @ B

        if self.__assume_sym and rowBloc > colBloc:
            # if self.__assume_sym, only compute bloc belonging to inf triangular matrix
            return

        self.isempty = False

        n_elm_gp = self.nbpg
        NnzColPerRowB = B.indptr[
            1
        ]  # number of non zero column per line for csr matrix B

        if not isinstance(A, list):
            NnzColPerRowA = A.indptr[
                1
            ]  # number of non zero column per line for csr matrix A

            if not (isinstance(coef, Number)):
                coef = coef.reshape(-1, 1, 1)

            if self.nbpg is None:
                new_data = (coef * A.data.reshape(-1, NnzColPerRowA, 1)) @ (
                    B.data.reshape(-1, 1, NnzColPerRowB)
                )  # at each PG we build a nbNode x nbNode matrix
            else:
                new_data = (coef * A.data.reshape(-1, NnzColPerRowA, 1)).reshape(
                    n_elm_gp, -1, NnzColPerRowA
                ).transpose((1, 2, 0)) @ B.data.reshape(
                    n_elm_gp, -1, NnzColPerRowB
                ).transpose(
                    1, 0, 2
                )  # at each element we build a nbNode x nbNode matrix

            if mat_lumping:
                if self.data[rowBloc][colBloc] is None:
                    self.data[rowBloc][colBloc] = np.zeros_like(new_data)

                list_ind_row = np.arange(new_data.shape[1])
                self.data[rowBloc][colBloc][:, list_ind_row, list_ind_row] += (
                    new_data.sum(axis=2)
                )
            else:
                if self.data[rowBloc][colBloc] is None:
                    self.data[rowBloc][colBloc] = new_data
                else:
                    self.data[rowBloc][colBloc] += new_data

        else:
            NnzColPerRowA = A[0].indptr[
                1
            ]  # number of non zero column per line for csr matrix A
            listCoef = coef  # alias
            listA = A  # alias
            A = listA[0]  # to compute col and row

            # coef_A_data = sum([coef[ii]*listA[ii].data.reshape(-1,NnzColPerRowA,1) if isinstance(coef, Number) else \
            #                    coef[ii].reshape(-1,1,1)*listA[ii].data.reshape(-1,NnzColPerRowA,1) for ii in range(len(listA))])

            coef_A_data = 0
            for ii, A in enumerate(listA):
                coef = listCoef[ii]
                if not (isinstance(coef, Number)):
                    coef = coef.reshape(-1, 1, 1)
                coef_A_data += coef * A.data.reshape(-1, NnzColPerRowA, 1)

            if self.nbpg is None:
                new_data = coef_A_data @ (
                    B.data.reshape(-1, 1, NnzColPerRowB)
                )  # at each PG we build a nbNode x nbNode matrix
            else:
                new_data = coef_A_data.reshape(n_elm_gp, -1, NnzColPerRowA).transpose(
                    (1, 2, 0)
                ) @ B.data.reshape(n_elm_gp, -1, NnzColPerRowB).transpose(
                    1, 0, 2
                )  # at each element we build a nbNode x nbNode matrix

            if mat_lumping:  # only the diag terms are non zero, with the sum of row values. Non diag terms are set to zeros but not removed to allow fast addition with non lumped matrix)
                if self.data[rowBloc][colBloc] is None:
                    self.data[rowBloc][colBloc] = np.zeros_like(new_data)

                list_ind_row = np.arange(new_data.shape[1])
                self.data[rowBloc][colBloc][:, list_ind_row, list_ind_row] += (
                    new_data.sum(axis=2)
                )  # set value only to diag terms
            else:
                if self.data[rowBloc][colBloc] is None:
                    self.data[rowBloc][colBloc] = new_data
                else:
                    self.data[rowBloc][colBloc] += new_data

        if self.col is None:
            # column indieces of A defined in A.indices are the row indices in final matrix
            # column indieces of B defined in B.indices are the column indices in final matrix
            if self.nbpg is None:
                self.row = (
                    A.indices.reshape(-1, NnzColPerRowA, 1)
                    @ np.ones((1, NnzColPerRowB), np.int32)
                ).ravel()
                self.col = (
                    np.ones((NnzColPerRowA, 1), np.int32)
                    @ B.indices.reshape(-1, 1, NnzColPerRowB)
                ).ravel()
                self.blocShape = (A.shape[1], B.shape[1])
            else:
                NelA = A.shape[0] // self.nbpg
                NelB = B.shape[0] // self.nbpg
                self.row = (
                    A.indices[0 : NelA * NnzColPerRowA].reshape(NelA, NnzColPerRowA, 1)
                    @ np.ones((1, NnzColPerRowB), np.int32)
                ).ravel()
                self.col = (
                    np.ones((NnzColPerRowA, 1), np.int32)
                    @ B.indices[0 : NelB * NnzColPerRowB].reshape(
                        NelB, 1, NnzColPerRowB
                    )
                ).ravel()
                self.blocShape = (A.shape[1], B.shape[1])

    def tocsr(
        self,
    ):  # should be improved in some way, perhaps by using a cpp script
        # import time
        # start=time.time()
        if self.isempty:
            return 0

        method = 1
        if method == 0:
            assert not (
                self.__assume_sym
            ), "method = 0 for sparse matrix building can't be used with the assume_sym option. Contact developer"
            ResDat = np.array(
                [
                    self.data[i][j]
                    for i in range(self.nbBlocRow)
                    for j in range(self.nbBlocCol)
                    if self.data[i][j] is not None
                ]
            ).ravel()
            ResRow = np.array(
                [
                    self.row + i * self.blocShape[0]
                    for i in range(self.nbBlocRow)
                    for j in range(self.nbBlocCol)
                    if self.data[i][j] is not None
                ],
                np.int32,
            ).ravel()
            ResCol = np.array(
                [
                    self.col + j * self.blocShape[1]
                    for i in range(self.nbBlocRow)
                    for j in range(self.nbBlocCol)
                    if self.data[i][j] is not None
                ],
                np.int32,
            ).ravel()

            Res = sparse.coo_matrix(
                (ResDat, (ResRow, ResCol)),
                shape=(
                    self.blocShape[0] * self.nbBlocRow,
                    self.blocShape[1] * self.nbBlocCol,
                ),
                copy=False,
            ).tocsr()
        elif method in [1, 2]:
            if self.Matrix_convertCOOtoCSR is None:
                # create alias
                if method == 1:
                    # compute intermediate csr matrix for each block
                    row_coo = self.row
                    col_coo = self.col
                    shape_coo = self.blocShape
                elif method == 2:
                    row_coo = np.array(
                        [
                            self.row + i * self.blocShape[0]
                            for i in range(self.nbBlocRow)
                            for j in range(self.nbBlocCol)
                            if self.data[i][j] is not None
                        ],
                        np.int32,
                    ).ravel()
                    col_coo = np.array(
                        [
                            self.col + j * self.blocShape[1]
                            for i in range(self.nbBlocRow)
                            for j in range(self.nbBlocCol)
                            if self.data[i][j] is not None
                        ],
                        np.int32,
                    ).ravel()
                    shape_coo = [
                        self.blocShape[0] * self.nbBlocRow,
                        self.blocShape[1] * self.nbBlocCol,
                    ]

                # compute convert matrix  that convert coo data to csr data
                data = np.ones(len(row_coo), dtype=np.int32)
                ref = (
                    row_coo.astype(np.int64) * shape_coo[1] + col_coo
                )  # ref for sorting by row indices then col indices in each row
                sorted_ind = ref.argsort()
                indices = sorted_ind

                # compute indptr of the convert matrix in csr format
                (val, ind_unique, count) = np.unique(
                    ref, return_index=True, return_counts=True
                )  # count the unique values that should be present in the csr data array (the duplicated values will be sumed)
                indptr = np.empty(len(val) + 1, dtype=np.int32)
                indptr[0] = 0
                np.cumsum(count, out=indptr[1:])

                self.Matrix_convertCOOtoCSR = sparse.csr_matrix(
                    (data, indices, indptr), shape=(len(val), len(col_coo))
                )

                # compute indices and indptr the final block csr matrix
                self.indices_csr = col_coo[ind_unique]

                nb_nnz_row = np.bincount(
                    row_coo[ind_unique], minlength=shape_coo[0]
                )  # nb_nnz_row[i] is the number of non zero term in row i.
                self.indptr_csr = np.empty(shape_coo[0] + 1, dtype=np.int32)
                self.indptr_csr[0] = 0
                np.cumsum(nb_nnz_row, out=self.indptr_csr[1:])

            if method == 1:
                if self.__assume_sym == True:
                    self.data = [
                        [
                            self.data[i][j]
                            if i <= j
                            else self.data[j][i].transpose(0, 2, 1)
                            for j in range(self.nbBlocCol)
                        ]
                        for i in range(self.nbBlocRow)
                    ]

                blocks = [
                    [
                        sparse.csr_matrix(
                            (
                                self.Matrix_convertCOOtoCSR @ self.data[i][j].ravel(),
                                self.indices_csr,
                                self.indptr_csr,
                            ),
                            shape=(self.blocShape[0], self.blocShape[1]),
                            copy=False,
                        )
                        if self.data[i][j] is not None
                        else sparse.csr_matrix((self.blocShape[0], self.blocShape[1]))
                        for j in range(self.nbBlocCol)
                    ]
                    for i in range(self.nbBlocRow)
                ]
                Res = sparse.bmat(blocks, format="csr")

            elif method == 2:
                assert not (
                    self.__assume_sym
                ), "method = 2 for sparse matrix building can't be used with the assume_sym option. Contact developer"
                data_coo = np.array(
                    [
                        self.data[i][j]
                        for i in range(self.nbBlocRow)
                        for j in range(self.nbBlocCol)
                        if self.data[i][j] is not None
                    ]
                ).ravel()
                Res = sparse.csr_matrix(
                    (
                        self.Matrix_convertCOOtoCSR @ data_coo,
                        self.indices_csr,
                        self.indptr_csr,
                    ),
                    shape=(
                        self.blocShape[0] * self.nbBlocRow,
                        self.blocShape[1] * self.nbBlocCol,
                    ),
                )

        # print(time.time()-start)
        # Res.eliminate_zeros()
        return Res

    def get_BlocStructure(self):
        # return data that may be reuse with other blocsparse
        return {
            "col": self.col,
            "row": self.row,
            "blocShape": self.blocShape,
            "Matrix_convertCOOtoCSR": self.Matrix_convertCOOtoCSR,
            "indices_csr": self.indices_csr,
            "indptr_csr": self.indptr_csr,
        }

        # ResDat = np.array([self.data[i][j] for i in range(self.nbBlocRow) for j in range(self.nbBlocCol) if self.data[i][j] is not None]).ravel()
        # ResRow = np.array([self.row+i*self.blocShape[0] for i in range(self.nbBlocRow) for j in range(self.nbBlocCol) if self.data[i][j] is not None], np.int32).ravel()
        # ResCol = np.array([self.col+j*self.blocShape[1] for i in range(self.nbBlocRow) for j in range(self.nbBlocCol) if self.data[i][j] is not None], np.int32).ravel()
        # Res = sparse.coo_matrix((ResDat, (ResRow,ResCol)), shape=(self.blocShape[0]*self.nbBlocRow, self.blocShape[1]*self.nbBlocCol), copy = False).tocsr()
        # # Res.data.round(10, Res.data)
        # Res.eliminate_zeros()
        # return Res


class _BlocSparseOld:
    def __init__(self, nbBlocRow, nbBlocCol):
        self.data = [[None for i in range(nbBlocCol)] for j in range(nbBlocRow)]
        self.col = None
        self.row = None
        self.blocShape = None
        self.nbBlocRow = nbBlocRow
        self.nbBlocCol = nbBlocCol

    def addToBlocATB(self, A, B, coef, rowBloc, colBloc):
        # A and B should be scipy matrix using the csr format and with the same number of column per row for each row (computed used GetElementaryOp of the Assembly class)
        # Add a matrix A.T * Mat_coef * B to the bloc defined by (rowBloc, colBloc)
        # Mat_coef is a diag matrix whose diag is given in the variable "coef"

        NnzColPerRowA = A.indptr[
            1
        ]  # number of non zero column per line for csr matrix A
        NnzColPerRowB = B.indptr[
            1
        ]  # number of non zero column per line for csr matrix B

        if not (isinstance(coef, Number)):
            coef = coef.reshape(-1, 1, 1)

        if self.data[rowBloc][colBloc] is None:
            # self.data[rowBloc][colBloc] = (coef * A.data.reshape(-1,NnzColPerRowA,1)).reshape(n_elm_gp,-1,NnzColPerRowA).transpose((1,2,0)) @ B.data.reshape(n_elm_gp,-1,NnzColPerRowB).transpose(1,0,2) #at each element we build a nbNode x nbNode matrix
            temp = A.data.reshape(-1, NnzColPerRowA, 1) @ B.data.reshape(
                -1, 1, NnzColPerRowB
            )  # at each PG we build a nbNode x nbNode matrix
            self.data[rowBloc][colBloc] = coef * temp
        else:
            # self.data[rowBloc][colBloc] += (coef * A.data.reshape(-1,NnzColPerRowA,1)).reshape(n_elm_gp,-1,NnzColPerRowA).transpose((1,2,0)) @ B.data.reshape(n_elm_gp,-1,NnzColPerRowB).transpose(1,0,2) #at element PG we build a nbNode x nbNode matrix
            temp = A.data.reshape(-1, NnzColPerRowA, 1) @ B.data.reshape(
                -1, 1, NnzColPerRowB
            )  # at each PG we build a nbNode x nbNode matrix
            self.data[rowBloc][colBloc] += coef * temp

        if self.col is None:
            # column indices of A defined in A.indices are the row indices in final matrix
            # column indices of B defined in B.indices are the column indices in final matrix
            self.row = (
                A.indices.reshape(-1, NnzColPerRowA, 1)
                @ np.ones((1, NnzColPerRowB), np.int32)
            ).ravel()
            self.col = (
                np.ones((NnzColPerRowA, 1), np.int32)
                @ B.indices.reshape(-1, 1, NnzColPerRowB)
            ).ravel()
            self.blocShape = (A.shape[1], B.shape[1])

        return temp

    def addToBloc(self, Arr, coef, rowBloc, colBloc):
        # Arr should be an array of size (totalNbPG = n_elm_gp*Nel, NndPerEl, NndPerEl)
        # This Arr gives the local assembled matrix associated to each PG
        # It should be multiplied by the factor coefficient associated to each PG
        # The global assembly is performed using a coo matrix format with self.row ans self.col vector previously computed by addToBlocATB
        # For that purpose, use the "toCSR" method

        if not (isinstance(coef, Number)):
            coef = coef.reshape(-1, 1, 1)

        if self.data[rowBloc][colBloc] is None:
            self.data[rowBloc][colBloc] = coef * Arr
        else:
            self.data[rowBloc][colBloc] += coef * Arr

    def toCSR(self):
        ResDat = np.array(
            [
                self.data[i][j]
                for i in range(self.nbBlocRow)
                for j in range(self.nbBlocCol)
                if self.data[i][j] is not None
            ]
        ).ravel()
        ResRow = np.array(
            [
                self.row + i * self.blocShape[0]
                for i in range(self.nbBlocRow)
                for j in range(self.nbBlocCol)
                if self.data[i][j] is not None
            ],
            np.int32,
        ).ravel()
        ResCol = np.array(
            [
                self.col + j * self.blocShape[1]
                for i in range(self.nbBlocRow)
                for j in range(self.nbBlocCol)
                if self.data[i][j] is not None
            ],
            np.int32,
        ).ravel()

        Res = sparse.coo_matrix(
            (ResDat, (ResRow, ResCol)),
            shape=(
                self.blocShape[0] * self.nbBlocRow,
                self.blocShape[1] * self.nbBlocCol,
            ),
            copy=False,
        ).tocsr()
        # Res.data.round(10, Res.data)
        Res.eliminate_zeros()
        return Res


def bloc_matrix(M, nb_bloc, position):
    var = position[1]
    var_vir = position[0]
    M = M.tocsr()

    indptr = np.zeros(np.shape(M)[0] * nb_bloc[0] + 1, dtype=int)
    indptr[var_vir * np.shape(M)[0] : (var_vir + 1) * np.shape(M)[0] + 1] = M.indptr
    indptr[(var_vir + 1) * np.shape(M)[0] + 1 :] = indptr[
        (var_vir + 1) * np.shape(M)[0]
    ]

    MatBloc = sparse.csr_matrix(
        (M.data, M.indices + var * np.shape(M)[1], indptr),
        shape=(np.shape(M)[0] * nb_bloc[0], np.shape(M)[1] * nb_bloc[1]),
    )

    return MatBloc


def ColumnBlocMatrix(listBloc, nb_bloc, position):
    order = list(np.array(position).argsort())
    position.sort()
    listBloc = [listBloc[ii].tocsr() for ii in order]

    NbRowPerBloc = np.shape(listBloc[0])[0]
    indices = np.hstack(Mat.indices for Mat in listBloc)
    data = np.hstack(Mat.data for Mat in listBloc)

    indptr = np.empty(NbRowPerBloc * nb_bloc + 1, dtype=int)

    ind = 0  # indice defining the begining of the bloc in indptr
    for var in range(nb_bloc):
        if var in position:
            Mat = listBloc[position.index(var)]
            indptr[var * NbRowPerBloc : (var + 1) * NbRowPerBloc + 1] = Mat.indptr + ind
            ind += len(Mat.indices)
        else:
            indptr[var * NbRowPerBloc : (var + 1) * NbRowPerBloc + 1] = ind

    return sparse.csr_matrix(
        (data, indices, indptr),
        shape=(NbRowPerBloc * nb_bloc, np.shape(listBloc[0])[1]),
    )


def RowBlocMatrix(listBloc, nb_bloc, position, coef):
    return sum(
        [
            bloc_matrix(coef[ii] * listBloc[ii], (1, nb_bloc), (0, position[ii]))
            for ii in range(len(listBloc))
        ]
    )


#    return sum([bloc_matrix(listBloc[ii], (1,nb_bloc), (0,position[ii])) if coef[ii] == 1 \
#                else coef[ii]*bloc_matrix(listBloc[ii], (1,nb_bloc), (0,position[ii])) for ii in range(len(listBloc))])
#
