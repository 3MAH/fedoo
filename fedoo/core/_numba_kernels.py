"""Compiled (numba) kernels of the assembly, used when numba is installed.

Importing this module raises ImportError without numba: the callers then keep
their numpy implementation.
"""

import numpy as np
from numba import carray, njit, prange, types
from numba.extending import intrinsic


@njit(parallel=True, cache=True, fastmath={"contract"})
def gauss_sum_atb(coefs, a_list, b, out, accumulate):
    """out[e] (+)= sum_g outer(sum_k coefs[k][g, e] * a_list[k][g, e], b[g, e]).

    a_list: tuple of (n_gp, n_elm, nA) arrays, b: (n_gp, n_elm, nB) array,
    coefs: tuple of (n_gp, n_elm) arrays. out: (n_elm, nA, nB). The operations
    follow the order of the numpy code: sum of the coef * a terms, sum over the
    Gauss points, accumulation. The fused multiply-add contraction matches
    numpy's matmul: for a single term, the result is the numpy one bit for bit
    for the 8 x 8 blocs of hex8. With several terms, numpy sums them without
    fused multiply-add, so the results agree to rounding only.
    """
    n_gp, n_elm, nb = b.shape
    na = a_list[0].shape[2]
    # chunks of elements: the work arrays are allocated once per chunk
    n_chunks = min(n_elm, 64)
    for chunk in prange(n_chunks):
        e0 = n_elm * chunk // n_chunks
        e1 = n_elm * (chunk + 1) // n_chunks
        k = np.empty((na, nb))
        s = np.empty(na)
        for e in range(e0, e1):
            k[:, :] = 0.0
            for g in range(n_gp):
                s[:] = 0.0
                for m in range(len(a_list)):
                    a = a_list[m]
                    c = coefs[m][g, e]
                    for i in range(na):
                        s[i] += c * a[g, e, i]
                for i in range(na):
                    si = s[i]
                    for j in range(nb):
                        k[i, j] += si * b[g, e, j]
            if accumulate:
                for i in range(na):
                    for j in range(nb):
                        out[e, i, j] += k[i, j]
            else:
                for i in range(na):
                    for j in range(nb):
                        out[e, i, j] = k[i, j]


@njit(parallel=True, cache=True)
def gather_sum(sources, src, ptr, out):
    """out[k] = sum of the source values src[ptr[k]:ptr[k + 1]], in that order.

    sources: tuple of 1D arrays of the same size n; the source value g is
    sources[g // n][g % n]. Used to sum the element matrices of all the blocs
    directly into the data array of the global csr matrix.
    """
    size = sources[0].size
    for k in prange(len(out)):
        s = 0.0
        for p in range(ptr[k], ptr[k + 1]):
            g = src[p]
            s += sources[g // size][g % size]
        out[k] = s


@njit(parallel=True, cache=True)
def csr_matvec(indptr, indices, data, x, out):
    """out = A @ x for the csr matrix A = (data, indices, indptr), by rows."""
    for i in prange(len(out)):
        s = 0.0
        for p in range(indptr[i], indptr[i + 1]):
            s += data[p] * x[indices[p]]
        out[i] = s


@intrinsic
def _float64_pointer(typingctx, address):
    """The address (int64) of float64 values as a pointer, for carray."""
    sig = types.CPointer(types.float64)(types.int64)

    def codegen(context, builder, signature, args):
        return builder.inttoptr(
            args[0], context.get_value_type(types.CPointer(types.float64))
        )

    return sig, codegen


@njit(parallel=True, cache=True, fastmath={"contract"})
def gauss_sum_atb_jobs(
    coef_address,
    op_address,
    n_gp,
    n_elm,
    n,
    job_out,
    job_b,
    job_acc,
    term_ptr,
    term_a,
    term_coef,
    out,
):
    """Several gauss_sum_atb in one pass over the elements.

    For each job c, in order: out[job_out[c], e] (+)= sum_g outer(sum_t
    coef_t[g, e] * op_a[g, e], op_b[g, e]) with t in term_ptr[c]:term_ptr[c + 1],
    coef_t the (n_gp, n_elm) array at coef_address[term_coef[t]], op_a and op_b
    the (n_gp, n_elm, n) arrays at op_address[term_a[t]] and
    op_address[job_b[c]], accumulated when job_acc[c]. out: (n_out, n_elm, n, n).

    The arrays are given by the address of their values: they must be C
    contiguous float64 arrays of these shapes, kept alive by the caller. (A
    list of arrays would be reference counted at each item access, which the
    threads contend for.) The operations of a job are those of gauss_sum_atb,
    in the same order, so the results are the same. The jobs are done chunk
    of elements by chunk of elements, so that the operator values of a chunk
    are read from memory once for all the jobs.
    """
    nb = n
    n_chunks = min(n_elm, 256)
    max_size = (n_elm + n_chunks - 1) // n_chunks
    for chunk in prange(n_chunks):
        e0 = n_elm * chunk // n_chunks
        e1 = n_elm * (chunk + 1) // n_chunks
        s = np.empty((max_size, n_gp, n))
        k = np.empty((n, nb))
        for c in range(len(job_out)):
            # s[e, g] = sum_t coef_t[g, e] * op_a[g, e], terms in order
            s[:, :, :] = 0.0
            for t in range(term_ptr[c], term_ptr[c + 1]):
                a = carray(_float64_pointer(op_address[term_a[t]]), (n_gp, n_elm, n))
                cf = carray(_float64_pointer(coef_address[term_coef[t]]), (n_gp, n_elm))
                for e in range(e0, e1):
                    for g in range(n_gp):
                        c_ge = cf[g, e]
                        for i in range(n):
                            s[e - e0, g, i] += c_ge * a[g, e, i]
            b = carray(_float64_pointer(op_address[job_b[c]]), (n_gp, n_elm, nb))
            o = job_out[c]
            acc = job_acc[c]
            for e in range(e0, e1):
                k[:, :] = 0.0
                for g in range(n_gp):
                    for i in range(n):
                        si = s[e - e0, g, i]
                        for j in range(nb):
                            k[i, j] += si * b[g, e, j]
                if acc:
                    for i in range(n):
                        for j in range(nb):
                            out[o, e, i, j] += k[i, j]
                else:
                    for i in range(n):
                        for j in range(nb):
                            out[o, e, i, j] = k[i, j]


@njit(parallel=True, cache=True, fastmath={"contract"})
def elementary_operator_values(inv_jac, dshape, shape, perm, n_gp, n_nodes, out):
    """Values, in csr order, of the elementary operators of a mesh: out[0] the
    shape functions, out[1 + i] the derivatives d/dx_i = inv_jac[..., i, :] @
    dshape, for the coo entries q = (e * n_gp + g) * n_nodes + k (element e,
    gauss point g, local node k) that the csr position p holds (q = perm[p]).

    inv_jac: (n_elm, n_gp, d, d_ref); dshape: (n_gp, d_ref, n_nodes);
    shape: (m, n_gp, n_nodes) with m = 1 (same for all elements) or n_elm.
    """
    d = inv_jac.shape[2]
    d_ref = inv_jac.shape[3]
    shared = shape.shape[0] == 1
    for p in prange(perm.shape[0]):
        q = perm[p]
        k = q % n_nodes
        eg = q // n_nodes
        g = eg % n_gp
        e = eg // n_gp
        out[0, p] = shape[0 if shared else e, g, k]
        for i in range(d):
            s = 0.0
            for c in range(d_ref):
                s += inv_jac[e, g, i, c] * dshape[g, c, k]
            out[1 + i, p] = s


@njit(parallel=True, cache=True)
def small_det(a, out):
    """out[k] = det(a[k]) for a (n, d, d) array, d = 2 or 3, in closed form.

    Same operations, in the same order, as the numpy closed form of
    fedoo.lib_elements.element_base._small_det: the results are identical.
    """
    if a.shape[1] == 2:
        for k in prange(a.shape[0]):
            out[k] = a[k, 0, 0] * a[k, 1, 1] - a[k, 0, 1] * a[k, 1, 0]
    else:
        for k in prange(a.shape[0]):
            out[k] = (
                a[k, 0, 0] * (a[k, 1, 1] * a[k, 2, 2] - a[k, 1, 2] * a[k, 2, 1])
                + a[k, 0, 1] * (a[k, 1, 2] * a[k, 2, 0] - a[k, 1, 0] * a[k, 2, 2])
                + a[k, 0, 2] * (a[k, 1, 0] * a[k, 2, 1] - a[k, 1, 1] * a[k, 2, 0])
            )


@njit(parallel=True, cache=True)
def small_inv(a, out):
    """out[k] = inv(a[k]) for a (n, d, d) array, d = 2 or 3, by the adjugate.

    Same operations as the numpy closed form of _small_inv (identical
    results). Returns False if a determinant is zero (out is then partial).
    """
    n = a.shape[0]
    singular = 0
    if a.shape[1] == 2:
        for k in prange(n):
            d = a[k, 0, 0] * a[k, 1, 1] - a[k, 0, 1] * a[k, 1, 0]
            if d == 0.0:
                singular += 1
                continue
            out[k, 0, 0] = a[k, 1, 1] / d
            out[k, 0, 1] = -a[k, 0, 1] / d
            out[k, 1, 0] = -a[k, 1, 0] / d
            out[k, 1, 1] = a[k, 0, 0] / d
    else:
        for k in prange(n):
            c00 = a[k, 1, 1] * a[k, 2, 2] - a[k, 1, 2] * a[k, 2, 1]
            c10 = a[k, 1, 2] * a[k, 2, 0] - a[k, 1, 0] * a[k, 2, 2]
            c20 = a[k, 1, 0] * a[k, 2, 1] - a[k, 1, 1] * a[k, 2, 0]
            d = a[k, 0, 0] * c00 + a[k, 0, 1] * c10 + a[k, 0, 2] * c20
            if d == 0.0:
                singular += 1
                continue
            out[k, 0, 0] = c00 / d
            out[k, 1, 0] = c10 / d
            out[k, 2, 0] = c20 / d
            out[k, 0, 1] = (a[k, 0, 2] * a[k, 2, 1] - a[k, 0, 1] * a[k, 2, 2]) / d
            out[k, 1, 1] = (a[k, 0, 0] * a[k, 2, 2] - a[k, 0, 2] * a[k, 2, 0]) / d
            out[k, 2, 1] = (a[k, 0, 1] * a[k, 2, 0] - a[k, 0, 0] * a[k, 2, 1]) / d
            out[k, 0, 2] = (a[k, 0, 1] * a[k, 1, 2] - a[k, 0, 2] * a[k, 1, 1]) / d
            out[k, 1, 2] = (a[k, 0, 2] * a[k, 1, 0] - a[k, 0, 0] * a[k, 1, 2]) / d
            out[k, 2, 2] = (a[k, 0, 0] * a[k, 1, 1] - a[k, 0, 1] * a[k, 1, 0]) / d
    return singular == 0
