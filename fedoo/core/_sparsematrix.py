import sys

import numpy as np
from scipy import sparse
from numbers import Number

from fedoo.core import base

try:  # optional compiled kernels, used when numba is installed
    from fedoo.core._numba_kernels import gauss_sum_atb as _gauss_sum_atb_numba
    from fedoo.core._numba_kernels import gather_sum as _gather_sum_numba
    from fedoo.core._numba_kernels import csr_matvec as _csr_matvec_numba
    from fedoo.core._numba_kernels import (
        gauss_sum_atb_jobs as _gauss_sum_atb_jobs_numba,
    )
    from fedoo.core._numba_kernels import (
        elementary_operator_values as _elementary_operator_values_numba,
    )
    import numba as _numba
except ImportError:
    _elementary_operator_values_numba = None
    _gauss_sum_atb_numba = None
    _gather_sum_numba = None
    _csr_matvec_numba = None
    _gauss_sum_atb_jobs_numba = None


def _performance_cores():
    """Number of performance cores of a hybrid Apple chip, None elsewhere.

    The efficiency cores slow down short parallel regions (MUMPS, assembly
    kernels), whose work is split before it starts: they are left out of the
    default thread counts.
    """
    if sys.platform != "darwin":
        return None
    try:
        import ctypes
        import ctypes.util

        libc = ctypes.CDLL(ctypes.util.find_library("c"))
        value, size = ctypes.c_int(0), ctypes.c_size_t(ctypes.sizeof(ctypes.c_int))
        if libc.sysctlbyname(
            b"hw.perflevel0.physicalcpu",
            ctypes.byref(value),
            ctypes.byref(size),
            None,
            0,
        ):
            return None
        return value.value if value.value > 0 else None
    except Exception:
        return None


_DEFAULT_NUMBA_THREADS = None


def _set_numba_threads():
    """Number of threads of the numba kernels: fedoo's N_THREADS if set, else
    the performance cores (hybrid Apple chips) or numba's default."""
    global _DEFAULT_NUMBA_THREADS
    max_threads = _numba.config.NUMBA_NUM_THREADS
    if base.N_THREADS > 0:
        n_threads = base.N_THREADS
    else:
        if _DEFAULT_NUMBA_THREADS is None:
            _DEFAULT_NUMBA_THREADS = _performance_cores() or max_threads
        n_threads = _DEFAULT_NUMBA_THREADS
    _numba.set_num_threads(min(n_threads, max_threads))


def _csr_matvec(M, x):
    """M @ x for a csr matrix M and a vector x, by rows in parallel (numba)
    when numba is installed."""
    if (
        _csr_matvec_numba is None
        or not sparse.issparse(M)
        or M.format != "csr"
        or M.dtype != np.float64
        or np.ndim(x) != 1
        or np.asarray(x).dtype != np.float64
    ):
        return M @ x
    out = np.empty(M.shape[0])
    _set_numba_threads()
    _csr_matvec_numba(M.indptr, M.indices, M.data, np.ascontiguousarray(x), out)
    return out


def _csr_rmatvec(M, x):
    """M.T @ x for a csr matrix M and a vector x, in parallel (numba) when
    numba is installed. The transposed csr matrix is computed once and kept
    on M: M must not be modified in place afterwards (as the elementary
    operators of an assembly, rebuilt when the mesh changes)."""
    if _csr_matvec_numba is None or not sparse.issparse(M) or M.format != "csr":
        return M.T @ x
    transposed = getattr(M, "_fedoo_transposed", None)
    if transposed is None:
        pattern = getattr(M, "_fedoo_pattern", None)
        if pattern is not None:
            perm_t, indices_t, indptr_t = pattern.transposed()
            transposed = sparse.csr_matrix(
                (M.data[perm_t], indices_t, indptr_t),
                shape=(M.shape[1], M.shape[0]),
                copy=False,
            )
        else:
            transposed = M.T.tocsr()
            transposed.sort_indices()
        M._fedoo_transposed = transposed
    return _csr_matvec(transposed, x)


class _CsrPattern:
    """Conversion of coo values on given (row, col) indices to csr values:
    a permutation, when the coo indices have no duplicate."""

    def __init__(self, row, col, shape):
        self.row, self.col, self.shape = row, col, shape
        n = len(row)
        # coo values = their position + 1, exact in float64
        m = sparse.coo_matrix(
            (np.arange(1, n + 1, dtype=np.float64), (row, col)), shape=shape
        ).tocsr()
        m.sort_indices()
        self.valid = m.nnz == n  # no duplicate: the conversion is a permutation
        self.perm = (m.data - 1).astype(np.int64)
        self.indices, self.indptr = m.indices, m.indptr
        self._transposed = None

    def transposed(self):
        """(perm, indices, indptr) of the transposed matrix, from the csr values."""
        if self._transposed is None:
            nnz = len(self.indices)
            mt = sparse.csr_matrix(
                (np.arange(1, nnz + 1, dtype=np.float64), self.indices, self.indptr),
                shape=self.shape,
            ).T.tocsr()
            mt.sort_indices()
            self._transposed = ((mt.data - 1).astype(np.int64), mt.indices, mt.indptr)
        return self._transposed


_CSR_PATTERNS = {}  # (id(row), id(col), shape) -> _CsrPattern, the last few


def _csr_pattern(row, col, shape):
    """The _CsrPattern of the coo indices (row, col), cached for the same
    row and col arrays."""
    key = (id(row), id(col), tuple(shape))
    pattern = _CSR_PATTERNS.get(key)
    if pattern is None or pattern.row is not row or pattern.col is not col:
        pattern = _CsrPattern(row, col, tuple(shape))
        if len(_CSR_PATTERNS) >= 8:
            _CSR_PATTERNS.pop(next(iter(_CSR_PATTERNS)))
        _CSR_PATTERNS[key] = pattern
    return pattern


def _csr_from_pattern(values, pattern):
    """csr matrix of values given in the csr order of pattern."""
    res = sparse.csr_matrix(
        (values, pattern.indices, pattern.indptr), shape=pattern.shape, copy=False
    )
    res.has_sorted_indices = True
    res._fedoo_pattern = pattern
    return res


def _coo_to_csr(data, row, col, shape):
    """sparse.coo_matrix((data, (row, col)), shape).tocsr(), with sorted
    indices. For the same row and col arrays (e.g. the elementary operators of
    a mesh whose nodes move), the conversion is computed once."""
    pattern = _csr_pattern(row, col, shape)
    if not pattern.valid:
        res = sparse.coo_matrix((data, (row, col)), shape=shape).tocsr()
        res.sort_indices()
        return res
    res = sparse.csr_matrix(
        (np.asarray(data).reshape(-1)[pattern.perm], pattern.indices, pattern.indptr),
        shape=shape,
        copy=False,
    )
    res.has_sorted_indices = True
    res._fedoo_pattern = pattern
    return res


def _gauss_sum_atb(coefs, a_list, b, out=None):
    """Element matrices sum_g outer(sum_k coefs[k][g, e] a_list[k][g, e], b[g, e]).

    a_list[k]: (n_gp, n_elm, nA), b: (n_gp, n_elm, nB), coefs[k]: scalar or
    (n_gp, n_elm). Returns an (n_elm, nA, nB) array, or adds the element
    matrices to ``out`` (in place) if given. Uses the numba kernel when numba
    is installed, numpy otherwise.
    """
    if _gauss_sum_atb_numba is not None and b.dtype == np.float64:
        kernel_coefs, kernel_a = [], []
        for coef, a in zip(coefs, a_list):
            if isinstance(coef, Number):
                kernel_coefs.append(np.full(a.shape[:2], float(coef)))
            else:
                kernel_coefs.append(np.ascontiguousarray(coef, dtype=np.float64))
            kernel_a.append(np.ascontiguousarray(a, dtype=np.float64))
        accumulate = out is not None
        if out is None:
            out = np.empty((b.shape[1], a_list[0].shape[2], b.shape[2]))
        _set_numba_threads()
        _gauss_sum_atb_numba(
            tuple(kernel_coefs),
            tuple(kernel_a),
            np.ascontiguousarray(b),
            out,
            accumulate,
        )
        return out
    coef_a = 0
    for coef, a in zip(coefs, a_list):
        coef_a = coef_a + (
            coef * a if isinstance(coef, Number) else coef[..., None] * a
        )
    new_data = coef_a.transpose(1, 2, 0) @ b.transpose(1, 0, 2)
    if out is None:
        return new_data
    out += new_data
    return out


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

            # global csr structures of the numba path, by bloc layout
            self._gather_cache = {}
        else:
            self.col = savedBlocStructure["col"]
            self.row = savedBlocStructure["row"]
            self.blocShape = savedBlocStructure["blocShape"]
            self.Matrix_convertCOOtoCSR = savedBlocStructure["Matrix_convertCOOtoCSR"]
            self.indices_csr = savedBlocStructure["indices_csr"]
            self.indptr_csr = savedBlocStructure["indptr_csr"]
            self._gather_cache = savedBlocStructure.setdefault("gather_cache", {})

        self.nbBlocRow = nbBlocRow
        self.nbBlocCol = nbBlocCol
        self.nbpg = nbpg
        self.__assume_sym = assume_sym
        # element matrix products waiting to be computed together (numba)
        self._jobs = []

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
        if self.nbpg is None or mat_lumping:
            self._flush()  # these paths read the blocs

        n_elm_gp = self.nbpg
        NnzColPerRowB = B.indptr[
            1
        ]  # number of non zero column per line for csr matrix B

        if not isinstance(A, list):
            NnzColPerRowA = A.indptr[
                1
            ]  # number of non zero column per line for csr matrix A

            if self.nbpg is None:
                if not (isinstance(coef, Number)):
                    coef = coef.reshape(-1, 1, 1)
                new_data = (coef * A.data.reshape(-1, NnzColPerRowA, 1)) @ (
                    B.data.reshape(-1, 1, NnzColPerRowB)
                )  # at each PG we build a nbNode x nbNode matrix
            else:
                # at each element we build a nbNode x nbNode matrix, added in
                # place to the bloc (unless lumped)
                new_data = self._add_gauss_sum_atb(
                    rowBloc,
                    colBloc,
                    [coef if isinstance(coef, Number) else coef.reshape(n_elm_gp, -1)],
                    [A.data.reshape(n_elm_gp, -1, NnzColPerRowA)],
                    B.data.reshape(n_elm_gp, -1, NnzColPerRowB),
                    mat_lumping,
                )

            if new_data is None:
                pass
            elif mat_lumping:
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

            if self.nbpg is None:
                coef_A_data = 0
                for ii, A in enumerate(listA):
                    coef = listCoef[ii]
                    if not (isinstance(coef, Number)):
                        coef = coef.reshape(-1, 1, 1)
                    coef_A_data += coef * A.data.reshape(-1, NnzColPerRowA, 1)
                new_data = coef_A_data @ (
                    B.data.reshape(-1, 1, NnzColPerRowB)
                )  # at each PG we build a nbNode x nbNode matrix
            else:
                # at each element we build a nbNode x nbNode matrix, added in
                # place to the bloc (unless lumped)
                new_data = self._add_gauss_sum_atb(
                    rowBloc,
                    colBloc,
                    [
                        c if isinstance(c, Number) else c.reshape(n_elm_gp, -1)
                        for c in listCoef
                    ],
                    [a.data.reshape(n_elm_gp, -1, NnzColPerRowA) for a in listA],
                    B.data.reshape(n_elm_gp, -1, NnzColPerRowB),
                    mat_lumping,
                )
                A = listA[-1]  # as left by the numpy loop, to compute col and row

            if new_data is None:
                pass
            elif mat_lumping:  # only the diag terms are non zero, with the sum of row values. Non diag terms are set to zeros but not removed to allow fast addition with non lumped matrix)
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

    def _add_gauss_sum_atb(self, rowBloc, colBloc, coefs, a_list, b, mat_lumping):
        """Add the element matrices of _gauss_sum_atb to the bloc (rowBloc,
        colBloc), or return them if mat_lumping. With numba, the products are
        kept and computed all together by _flush (one pass over the
        elements); the bloc is then None until _flush."""
        if (
            not mat_lumping
            and _gauss_sum_atb_jobs_numba is not None
            and b.dtype == np.float64
            and all(a.dtype == np.float64 and a.shape == b.shape for a in a_list)
        ):
            if self._jobs and self._jobs[0][4].shape != b.shape:
                self._flush()  # a batch shares the shape of the operators
            self._jobs.append((rowBloc, colBloc, coefs, a_list, b))
            return None
        self._flush()
        new_data = _gauss_sum_atb(
            coefs,
            a_list,
            b,
            out=None if mat_lumping else self.data[rowBloc][colBloc],
        )
        if mat_lumping:
            return new_data
        self.data[rowBloc][colBloc] = new_data
        return None

    def _flush(self):
        """Compute the products kept by _add_gauss_sum_atb, in one numba call."""
        jobs, self._jobs = self._jobs, []
        if not jobs:
            return

        def key(x):  # same values: same memory, shape and strides
            return (x.__array_interface__["data"][0], x.shape, x.strides)

        ops, op_index = [], {}
        coefs, coef_index = [], {}
        n_gp, n_elm, n = jobs[0][4].shape

        def op_id(x):
            k = key(x)
            if k not in op_index:
                op_index[k] = len(ops)
                ops.append(np.ascontiguousarray(x, dtype=np.float64))
            return op_index[k]

        def coef_id(c):
            if isinstance(c, Number):
                k = ("scalar", float(c))
                if k not in coef_index:
                    coef_index[k] = len(coefs)
                    coefs.append(np.full((n_gp, n_elm), float(c)))
            else:
                k = key(c)
                if k not in coef_index:
                    coef_index[k] = len(coefs)
                    coefs.append(
                        np.ascontiguousarray(
                            np.broadcast_to(c, (n_gp, n_elm)), dtype=np.float64
                        )
                    )
            return coef_index[k]

        blocs, bloc_index = [], {}
        job_out, job_b, job_acc, term_ptr, term_a, term_coef = [], [], [], [0], [], []
        initialized = []
        for row_bloc, col_bloc, job_coefs, a_list, b in jobs:
            if (row_bloc, col_bloc) not in bloc_index:
                bloc_index[(row_bloc, col_bloc)] = len(blocs)
                blocs.append((row_bloc, col_bloc))
                initialized.append(self.data[row_bloc][col_bloc] is not None)
            o = bloc_index[(row_bloc, col_bloc)]
            job_out.append(o)
            job_acc.append(initialized[o])
            initialized[o] = True
            job_b.append(op_id(b))
            for c, a in zip(job_coefs, a_list):
                term_a.append(op_id(a))
                term_coef.append(coef_id(c))
            term_ptr.append(len(term_a))

        out = np.empty((len(blocs), n_elm, n, n))
        for o, (row_bloc, col_bloc) in enumerate(blocs):
            if self.data[row_bloc][col_bloc] is not None:
                out[o] = self.data[row_bloc][col_bloc]
        _set_numba_threads()
        # coefs and ops: C contiguous float64 arrays, alive during the call
        _gauss_sum_atb_jobs_numba(
            np.array([c.ctypes.data for c in coefs], dtype=np.int64),
            np.array([op.ctypes.data for op in ops], dtype=np.int64),
            n_gp,
            n_elm,
            n,
            np.array(job_out, dtype=np.int64),
            np.array(job_b, dtype=np.int64),
            np.array(job_acc, dtype=np.bool_),
            np.array(term_ptr, dtype=np.int64),
            np.array(term_a, dtype=np.int64),
            np.array(term_coef, dtype=np.int64),
            out,
        )
        for o, (row_bloc, col_bloc) in enumerate(blocs):
            self.data[row_bloc][col_bloc] = out[o]

    def tocsr(
        self,
    ):  # should be improved in some way, perhaps by using a cpp script
        # import time
        # start=time.time()
        if self.isempty:
            return 0
        self._flush()

        if _gather_sum_numba is not None:
            res = self._tocsr_gather()
            if res is not None:
                return res

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
                            else (
                                self.data[j][i].transpose(0, 2, 1)
                                if self.data[j][i] is not None
                                else None
                            )
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

    def _sources(self):
        """The computed blocs and, for each bloc of the matrix, its source.

        Returns (computed, layout): computed lists the (i, j) of the blocs
        holding element matrices, layout lists (i, j, k, transposed) for each
        non empty bloc of the matrix, k being its index in computed. With
        assume_sym, a bloc below the diagonal is the transposed bloc above.
        """
        computed, layout = [], []
        for i in range(self.nbBlocRow):
            for j in range(self.nbBlocCol):
                if self.__assume_sym and i > j:
                    continue
                if self.data[i][j] is not None:
                    computed.append((i, j))
        index = {ij: k for k, ij in enumerate(computed)}
        for i in range(self.nbBlocRow):
            for j in range(self.nbBlocCol):
                if self.__assume_sym and i > j:
                    if (j, i) in index:
                        layout.append((i, j, index[(j, i)], True))
                elif (i, j) in index:
                    layout.append((i, j, index[(i, j)], False))
        return computed, tuple(layout)

    def _gather_structure(self, layout, n_a, n_b):
        """Global csr pattern of the matrix and, for each of its entries, the
        element matrix values to sum: src[ptr[k]:ptr[k + 1]] for the entry k.
        A source value g is the value g % size of the computed bloc g // size,
        size being the number of values of a bloc.

        All the blocs share the coo pattern (self.row, self.col): only this
        pattern is sorted, the global one follows by index arithmetic. A row
        of the global matrix is a local row r of the bloc row i, made of the
        entries of r in each bloc (i, j), in the order of j.
        """
        size = len(self.row)
        n_rows_bloc, n_cols_bloc = self.blocShape
        n_cols = n_cols_bloc * self.nbBlocCol

        # pattern of a bloc: unique entries u (sorted by row, then column)
        # and their sources order[first[u]:first[u] + n_src[u]]
        key = self.row.astype(np.int64) * n_cols_bloc + self.col
        order = np.argsort(key, kind="stable")
        key = key[order]
        first = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
        n_src = np.diff(np.r_[first, size])
        key = key[first]
        loc_row, loc_col = np.divmod(key, n_cols_bloc)
        row_count = np.bincount(loc_row, minlength=n_rows_bloc)
        row_start = np.r_[0, np.cumsum(row_count)[:-1]]
        n_loc = len(key)
        if any(transposed for *_, transposed in layout):
            # value (e, a, b) of a bloc below the diagonal: (e, b, a) of the
            # computed bloc above
            elm, local = np.divmod(np.arange(size, dtype=np.int64), n_a * n_b)
            a, b = np.divmod(local, n_b)
            flat_t = elm * (n_a * n_b) + b * n_b + a

        indices, entry_u, entry_bloc, row_counts = [], [], [], []
        for i in range(self.nbBlocRow):
            blocs = [(j, k, t) for ii, j, k, t in layout if ii == i]
            n_j = len(blocs)
            if n_j == 0:
                row_counts.append(np.zeros(n_rows_bloc, dtype=np.int64))
                continue
            # position of the entry (u, j) among the entries of the bloc row
            u = np.arange(n_loc)
            pos = (
                n_j * row_start[loc_row][:, None]
                + np.arange(n_j)[None, :] * row_count[loc_row][:, None]
                + (u - row_start[loc_row])[:, None]
            ).ravel()
            u_of = np.empty(n_loc * n_j, dtype=np.int64)
            j_of = np.empty(n_loc * n_j, dtype=np.int64)
            u_of[pos] = np.repeat(u, n_j)
            j_of[pos] = np.tile(np.arange(n_j), n_loc)
            col_offset = np.array([j * n_cols_bloc for j, _, _ in blocs])
            indices.append(loc_col[u_of] + col_offset[j_of])
            entry_u.append(u_of)
            entry_bloc.append(np.array([(k, t) for _, k, t in blocs])[j_of])
            row_counts.append(n_j * row_count)
        indices = np.concatenate(indices)
        entry_u = np.concatenate(entry_u)
        entry_bloc = np.concatenate(entry_bloc)
        row_counts = np.concatenate(row_counts)

        # sources of each global entry, in the order of the bloc pattern
        count = n_src[entry_u]
        ptr = np.r_[0, np.cumsum(count)]
        offset = np.arange(ptr[-1]) - np.repeat(ptr[:-1], count)
        flat = order[np.repeat(first[entry_u], count) + offset]
        transposed = np.repeat(entry_bloc[:, 1].astype(bool), count)
        if transposed.any():
            flat[transposed] = flat_t[flat[transposed]]
        src = np.repeat(entry_bloc[:, 0], count) * size + flat

        idx_dtype = np.int32 if max(len(indices), n_cols) < 2**31 else np.int64
        src_dtype = np.int32 if len(src) < 2**31 else np.int64
        indptr = np.zeros(len(row_counts) + 1, dtype=idx_dtype)
        np.cumsum(row_counts, out=indptr[1:])
        return (
            src.astype(src_dtype),
            ptr.astype(src_dtype),
            indices.astype(idx_dtype),
            indptr,
        )

    def _tocsr_gather(self):
        """The global csr matrix summed in parallel (numba) from the element
        matrices of all the blocs, on a pattern computed once per bloc layout.
        The returned matrices share their indices and indptr arrays, so that
        an unchanged pattern is the same objects. None if not applicable."""
        computed, layout = self._sources()
        sources = [self.data[i][j] for i, j in computed]
        if not sources or any(
            s.dtype != np.float64 or s.ndim != 3 or s.size != len(self.row)
            for s in sources
        ):
            return None
        n_a, n_b = sources[0].shape[1:]
        if any(t for *_, t in layout) and n_a != n_b:
            return None
        structure = self._gather_cache.get(layout)
        if structure is None:
            structure = self._gather_structure(layout, n_a, n_b)
            self._gather_cache[layout] = structure
        src, ptr, indices, indptr = structure
        data = np.empty(len(indices))
        _set_numba_threads()
        _gather_sum_numba(
            tuple(np.ascontiguousarray(s).reshape(-1) for s in sources),
            src,
            ptr,
            data,
        )
        res = sparse.csr_matrix(
            (data, indices, indptr),
            shape=(
                self.blocShape[0] * self.nbBlocRow,
                self.blocShape[1] * self.nbBlocCol,
            ),
            copy=False,
        )
        res.has_sorted_indices = True
        return res

    def get_BlocStructure(self):
        # return data that may be reuse with other blocsparse
        return {
            "col": self.col,
            "row": self.row,
            "blocShape": self.blocShape,
            "Matrix_convertCOOtoCSR": self.Matrix_convertCOOtoCSR,
            "indices_csr": self.indices_csr,
            "indptr_csr": self.indptr_csr,
            "gather_cache": self._gather_cache,
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


def bloc_matrix(M, nb_bloc, position, shape=None):
    var = position[1]
    var_vir = position[0]
    M = M.tocsr()
    if shape is None:
        shape = (M.shape[0] * nb_bloc[0], M.shape[1] * nb_bloc[1])

    indptr = np.zeros(shape[0] + 1, dtype=int)
    indptr[var_vir * M.shape[0] : (var_vir + 1) * M.shape[0] + 1] = M.indptr
    indptr[(var_vir + 1) * M.shape[0] + 1 :] = indptr[(var_vir + 1) * M.shape[0]]

    MatBloc = sparse.csr_matrix(
        (M.data, M.indices + var * np.shape(M)[1], indptr),
        shape=shape,
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


def RowBlocMatrix(listBloc, nb_bloc, position, coef, n_cols=None):
    if n_cols is None:
        shape = None
    else:
        shape = (listBloc[0].shape[0], n_cols)

    return sum(
        [
            bloc_matrix(
                coef[ii] * listBloc[ii],
                (1, nb_bloc),
                (0, position[ii]),
                shape=shape,
            )
            for ii in range(len(listBloc))
        ]
    )


#    return sum([bloc_matrix(listBloc[ii], (1,nb_bloc), (0,position[ii])) if coef[ii] == 1 \
#                else coef[ii]*bloc_matrix(listBloc[ii], (1,nb_bloc), (0,position[ii])) for ii in range(len(listBloc))])
#


def scatter_dense_block(block, dof_indices, shape):
    """Scatter a dense ``(k, k)`` block onto a global sparse matrix.

    ``dof_indices`` (length ``k``) gives the global row/column position of each
    local DOF: entry ``block[i, j]`` lands at
    ``(dof_indices[i], dof_indices[j])``. Returns a
    :class:`scipy.sparse.csr_matrix` of the requested ``shape``.
    """
    idx = np.asarray(dof_indices, dtype=int)
    k = len(idx)
    rows = np.repeat(idx, k)
    cols = np.tile(idx, k)
    return sparse.csr_matrix(
        (np.asarray(block, dtype=float).ravel(), (rows, cols)), shape=shape
    )
