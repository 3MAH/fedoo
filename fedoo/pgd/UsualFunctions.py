# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 10:29:20 2018

@author: etienne
"""

from fedoo.pgd.SeparatedArray import SeparatedOnes, SeparatedArray
import numpy as np
import scipy.sparse as sparse
from numbers import Number


def mult(U, V, max_norm_err=1e-4):
    if isinstance(U, Number):
        return mult(V, U)

    if V == 0:
        return 0
    if isinstance(V, Number):
        return SeparatedArray(
            [np.c_[a * self.data[0]]]
            + [np.c_[self.data[dd]] for dd in range(1, self.dim)]
        )
    elif isinstance(V, SeparatedArray):  # implémentation rapide à optimiser
        res = 0
        for ii in range(U.nbTerm()):
            for jj in range(V.nbTerm()):
                #                norm = np.array([np.dot(U.data[dd][:,ii], V.data[dd][:,jj]) for dd in range(U.dim)]).prod()

                Uiijj = SeparatedArray(
                    [
                        np.c_[U.data[dd][:, ii] * V.data[dd][:, jj]]
                        for dd in range(U.dim)
                    ]
                )
                if Uiijj.norm() > max_norm_err:
                    res += SeparatedArray(
                        [
                            np.c_[self.data[dd][:, ii] * a.data[dd][:, jj]]
                            for dd in range(self.dim)
                        ]
                    )
        return res
    else:
        return NotImplemented


def inv(U, max_iter=1000, max_norm_err=1e-4, max_iter_RS=5):
    # resolution of the equation V*U = 1 using the PGD, and return V
    shape = U.shape
    dim = U.dim
    V = 0
    RHS = SeparatedOnes(
        shape
    )  # right hand side of the system modified at each iteration (PGD algorithm)
    err_0 = RHS.norm()

    for iter in range(max_iter):
        #        print('ITER: '+str(iter))
        R = SeparatedArray(1, shape)
        R2 = R * R

        err_RS = 1
        comp_nR = 0

        while err_RS > 1e-5:
            #            print(err_RS)

            comp_nR += 1
            #            print('comp_nR: '+str(comp_nR))

            R_old = SeparatedArray(R)
            for d1 in range(dim):
                #                print('dim_calc: '+str(d1))
                prod_aux_lhs = 1
                prod_aux_rhs = 1
                for d2 in list(range(d1)) + list(range(d1 + 1, dim)):
                    prod_aux_lhs = prod_aux_lhs * (np.dot(R2.data[d2].T, U.data[d2]))
                    prod_aux_rhs = prod_aux_rhs * (np.dot(R.data[d2].T, RHS.data[d2]))

                R.data[d1] = np.dot(RHS.data[d1], prod_aux_rhs.T) / np.dot(
                    U.data[d1], prod_aux_lhs.T
                )
                R2.data[d1] = R.data[d1] * R.data[d1]

            #            R.regularise() ; R2 = R*R

            err_RS = (R - R_old).norm() / R.norm()
            if comp_nR > max_iter_RS:
                break

        RHS = RHS - R * U
        if RHS.nbTerm() > 100:
            RHS = RHS.reductionPGD()
        #        RHS.reductionPGD()
        V = V + R

        norm_err = RHS.norm() / err_0
        #        print(norm_err)
        if norm_err < max_norm_err:
            break

    return V


def power(U, p, max_iter=1000, max_norm_err=1e-4, max_iter_RS=10):
    # PGD resolution of the equation (X+a)V'- pV = 0 with V(0) = a**p and U=X+a
    # the solution is V = (X+a)**p = U**p
    # a new dimension is added for the initial condition ie Xc = X*k with k in [0,1]
    # initial condition: V(kX) = a**p if k=0

    a = 1
    V0 = a**p
    dk = 1e-3
    k = np.arange(0, 1, dk)
    nb_k = k.shape[0]

    data = [-a * np.ones(nb_k), a * np.ones(nb_k)]
    M1 = sparse.spdiags(data, [-1, 0], nb_k, nb_k, format="csr")

    data = [-p * dk - k[1:], k]
    M2 = sparse.diags(data, [-1, 0], (nb_k, nb_k), format="csr")

    ##    #======= test ==========
    #    M2 = M1.T*M2
    #    M1 = M1.T*M1
    ##    #======= fin test ==========

    # Defining X = (U-a) with the new dimension k
    X = SeparatedArray([np.ones((1, U.nbTerm() + 1))] + (U - a).data)
    RHS = SeparatedArray([X.data[1] * p * dk] + X.data[2:]) + a
    RHS = SeparatedArray(
        [np.zeros((nb_k, RHS.nbTerm()))] + RHS.data
    )  # right hand side of the system modified at each iteration (PGD algorithm)
    RHS.data[0][0] = V0  # for the inial value when k = 0 => (kX+a)**p = a**p

    shape = RHS.shape
    dim = RHS.dim
    err_0 = RHS.norm()

    V = 0

    for iter in range(max_iter):
        print("ITER: " + str(iter))
        #        R = SeparatedArray([np.zeros((nb_k, 1))] + U.getTerm(iter%U.nbTerm()).data)
        R = SeparatedArray(1, shape)
        R2 = R * R

        err_RS = 1
        comp_nR = 0

        while err_RS > 1e-5:
            comp_nR += 1
            #            print('comp_nR: '+str(comp_nR))

            R_old = SeparatedArray(R)
            for d1 in range(dim):
                #                print('dim_calc: '+str(d1))
                prod_aux_lhs_1 = 1
                prod_aux_lhs_2 = 1
                prod_aux_rhs = 1
                for d2 in list(range(d1)) + list(range(d1 + 1, dim)):
                    prod_aux_rhs = prod_aux_rhs * (np.dot(R.data[d2].T, RHS.data[d2]))
                    if d2 == 0:
                        prod_aux_lhs_1 = prod_aux_lhs_1 * (
                            np.dot(R.data[d2].T, M1 * R.data[d2])
                        )
                        prod_aux_lhs_2 = (
                            prod_aux_lhs_2
                            * X.data[d2]
                            * (np.dot(R.data[d2].T, (M2 * R.data[d2])))
                        )
                    else:
                        prod_aux_lhs_1 = prod_aux_lhs_1 * (
                            np.dot(R.data[d2].T, R.data[d2])
                        )
                        prod_aux_lhs_2 = prod_aux_lhs_2 * (
                            np.dot(R2.data[d2].T, X.data[d2])
                        )

                if d1 == 0:
                    #                    M = np.dot(X.data[d1],prod_aux_lhs_2.T)*M2 + prod_aux_lhs_1[0]*M1

                    data = [
                        -a * prod_aux_lhs_1[0]
                        - np.dot(X.data[d1], prod_aux_lhs_2.T)[0] * (p * dk + k[1:]),
                        a * prod_aux_lhs_1[0]
                        + np.dot(X.data[d1], prod_aux_lhs_2.T)[0] * k,
                    ]
                    M = sparse.diags(data, [-1, 0], (nb_k, nb_k), format="csr")
                    R.data[d1] = sparse.linalg.spsolve(
                        M, np.dot(RHS.data[d1], prod_aux_rhs.T)
                    ).reshape(-1, 1)

                ##                    #======= test ==========
                #                    M = M1.T*sparse.diags(data, [-1,0], (nb_k, nb_k), format="csr")
                #                    R.data[d1] = sparse.linalg.spsolve(M,M1.T*np.dot(RHS.data[d1], prod_aux_rhs.T) ).reshape(-1,1)
                ##                    #======= fin test ==========

                else:
                    R.data[d1] = np.dot(RHS.data[d1], prod_aux_rhs.T) / (
                        prod_aux_lhs_1 + np.dot(X.data[d1], prod_aux_lhs_2.T)
                    )
                    R2.data[d1] = R.data[d1] * R.data[d1]

            R.regularise()
            R2 = R * R

            err_RS = (R - R_old).norm() / R.norm()
            #            print(err_RS)

            if comp_nR > max_iter_RS:
                break

        RHS = (
            RHS
            - SeparatedArray([M1 * R.data[0]] + R.data[1:])
            - X * SeparatedArray([M2 * R.data[0]] + R.data[1:])
        )

        #        RHS.sort()
        if RHS.nbTerm() > 100:
            RHS.sort()
        #            RHS.regularise()
        #            RHS = RHS.reductionPGD(max_norm_err = 1e-2)
        #            RHS = RHS.reduction2D()
        #            RHS.reduction()

        print(RHS.nbTerm())
        V = V + R

        norm_err = RHS.norm() / err_0
        print(norm_err)
        if norm_err < max_norm_err:
            break

    #    return V
    res = SeparatedArray(V.data[1:])
    res.data[1] = res.data[1] * V.data[0][-1]
    return res


def exp(U, max_iter=1000, max_norm_err=1e-4, max_iter_RS=100):
    # resolution of the equation V'- V = 0 with V(0) = 1 using the PGD
    # a new dimension is added for the initial condition ie Uc = U*k with k in [0,1]
    # initial condition: V(kU) = 1 if k=0

    V = 0
    dk = 1e-3
    nb_k = int(1 // dk)

    DX_plus_1 = (
        SeparatedArray([np.ones((1, U.nbTerm()))] + [dk * U.data[0]] + U.data[1:]) + 1
    )

    RHS = SeparatedArray(
        [np.zeros((nb_k, DX_plus_1.nbTerm()))] + DX_plus_1.data[1:]
    )  # right hand side of the system modified at each iteration (PGD algorithm)
    RHS.data[0][0] = 1  # for the inial value when k = 0 => exp (kU) = 1

    shape = RHS.shape
    dim = RHS.dim
    err_0 = RHS.norm()

    for iter in range(max_iter):
        print("ITER: " + str(iter))
        R = SeparatedArray([np.zeros((nb_k, 1))] + U.getTerm(iter % U.nbTerm()).data)
        # R = SeparatedArray(1,shape)
        R2 = R * R

        err_RS = 1
        comp_nR = 0

        while err_RS > 1e-5:
            comp_nR += 1
            #            print('comp_nR: '+str(comp_nR))

            R_old = SeparatedArray(R)
            for d1 in range(dim):
                #                print('dim_calc: '+str(d1))
                prod_aux_lhs_x1 = 1
                prod_aux_lhs_x0 = 1
                prod_aux_rhs = 1
                for d2 in list(range(d1)) + list(range(d1 + 1, dim)):
                    prod_aux_rhs = prod_aux_rhs * (np.dot(R.data[d2].T, RHS.data[d2]))
                    prod_aux_lhs_x1 = prod_aux_lhs_x1 * (
                        np.dot(R.data[d2].T, R.data[d2])
                    )
                    if d2 == 0:
                        prod_aux_lhs_x0 = (
                            -prod_aux_lhs_x0
                            * DX_plus_1.data[d2]
                            * (np.dot(R.data[d2][1:].T, R.data[d2][0:-1]))
                        )
                    #                        prod_aux_lhs_x0 = - prod_aux_lhs_x0 * (np.dot(R.data[d2][1:].T, R.data[d2][0:-1] ))
                    else:
                        prod_aux_lhs_x0 = prod_aux_lhs_x0 * (
                            np.dot(R2.data[d2].T, DX_plus_1.data[d2])
                        )

                if d1 == 0:
                    data = [
                        (-np.dot(DX_plus_1.data[d1], prod_aux_lhs_x0.T))[0]
                        * np.ones(nb_k),
                        prod_aux_lhs_x1[0] * np.ones(nb_k),
                    ]
                    M = sparse.spdiags(data, [-1, 0], nb_k, nb_k, format="csr")
                    R.data[d1] = sparse.linalg.spsolve(
                        M, np.dot(RHS.data[d1], prod_aux_rhs.T)
                    ).reshape(-1, 1)
                else:
                    R.data[d1] = np.dot(RHS.data[d1], prod_aux_rhs.T) / (
                        prod_aux_lhs_x1 + np.dot(DX_plus_1.data[d1], prod_aux_lhs_x0.T)
                    )
                    R2.data[d1] = R.data[d1] * R.data[d1]

            R.regularise()
            R2 = R * R

            err_RS = (R - R_old).norm() / R.norm()
            #            print(err_RS)

            if comp_nR > max_iter_RS:
                break

        R_x0 = SeparatedArray(R)
        R_x0.data[0][1:] = R.data[0][0:-1]
        R_x0.data[0][0] = 0

        RHS = RHS - R + DX_plus_1 * R_x0
        #        if RHS.nbTerm()>100:
        #            RHS.regularise()
        #            RHS = RHS.reductionPGD()
        #            RHS = RHS.reduction2D()
        #            RHS.reduction()
        V = V + R

        norm_err = RHS.norm() / err_0
        print(norm_err)
        if norm_err < max_norm_err:
            break

    #    return V
    res = SeparatedArray(V.data[1:])
    res.data[1] = res.data[1] * V.data[0][-1]
    return res


# # Doesn't work well
def sqrt(U, max_iter=1000, max_norm_err=1e-4, max_iter_RS=5):
    # resolution of the equation V*V = U using the PGD, and return V
    shape = U.shape
    dim = U.dim
    V = U.getTerm(0)
    V.data = [np.sqrt(data_dd) for data_dd in V.data]
    RHS = U  # right hand side of the system modified at each iteration (PGD algorithm)
    err_0 = RHS.norm()

    for iter in range(max_iter):
        print("ITER: " + str(iter))
        R = SeparatedArray(1, shape)
        R2 = R * R

        err_RS = 1
        comp_nR = 0

        while err_RS > 1e-5:
            #            print(err_RS)

            comp_nR += 1
            print("comp_nR: " + str(comp_nR))

            R_old = SeparatedArray(R)
            for d1 in range(dim):
                #                print('dim_calc: '+str(d1))
                prod_aux_lhs = 1
                prod_aux_rhs = 1
                for d2 in list(range(d1)) + list(range(d1 + 1, dim)):
                    prod_aux_lhs = prod_aux_lhs * (np.dot(R2.data[d2].T, V.data[d2]))
                    prod_aux_rhs = prod_aux_rhs * (np.dot(R.data[d2].T, RHS.data[d2]))

                R.data[d1] = np.dot(RHS.data[d1], prod_aux_rhs.T) / np.dot(
                    V.data[d1], prod_aux_lhs.T
                )
                R2.data[d1] = R.data[d1] * R.data[d1]

            #            R.regularise() ; R2 = R*R

            err_RS = (R - R_old).norm() / R.norm()
            if comp_nR > max_iter_RS:
                break

        V = V + R
        RHS = (U - V * V).reduction()  # a ameliorer
        #        if RHS.nbTerm()>50:
        #            RHS = RHS.reductionPGD()
        #        RHS.reductionPGD()

        norm_err = RHS.norm() / err_0
        print(norm_err)
        if norm_err < max_norm_err:
            break

    return V


def divide(U1, U2, max_iter=1000, max_norm_err=1e-4, max_iter_RS=5):
    # resolution of the equation V*U2 = U1 using the PGD, and return V

    if isinstance(U2, Number):
        return U1 / U2
    if isinstance(U1, Number):
        return inv(U2) * U1

    shape = U1.shape
    dim = U1.dim
    assert U1.shape == U2.shape, "Can only divide SeparatedArray of the same shape"
    V = 0
    RHS = U1  # right hand side of the system modified at each iteration (PGD algorithm)
    err_0 = RHS.norm()

    for iter in range(max_iter):
        #        print('ITER: '+str(iter))
        R = SeparatedArray(1, shape)
        R2 = R * R

        err_RS = 1
        comp_nR = 0

        while err_RS > 1e-5:
            #            print(err_RS)

            comp_nR += 1
            #            print('comp_nR: '+str(comp_nR))

            R_old = SeparatedArray(R)
            for d1 in range(dim):
                #                print('dim_calc: '+str(d1))
                prod_aux_lhs = 1
                prod_aux_rhs = 1
                for d2 in list(range(d1)) + list(range(d1 + 1, dim)):
                    prod_aux_lhs = prod_aux_lhs * (np.dot(R2.data[d2].T, U2.data[d2]))
                    prod_aux_rhs = prod_aux_rhs * (np.dot(R.data[d2].T, RHS.data[d2]))

                R.data[d1] = np.dot(RHS.data[d1], prod_aux_rhs.T) / np.dot(
                    U2.data[d1], prod_aux_lhs.T
                )
                R2.data[d1] = R.data[d1] * R.data[d1]

            #            R.regularise() ; R2 = R*R

            err_RS = (R - R_old).norm() / R.norm()
            if comp_nR > max_iter_RS:
                break

        RHS = RHS - R * U2
        if RHS.nbTerm() > 100:
            RHS = RHS.reductionPGD()
        #        RHS.reductionPGD()
        V = V + R

        norm_err = RHS.norm() / err_0
        #        print(norm_err)
        if norm_err < max_norm_err:
            break

    return V
