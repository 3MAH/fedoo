import numpy as np
from scipy import linalg
from numbers import (
    Number,
)  # class de base qui permet de tester si un type est numérique


# ===============================================================================
# Définition des fonctions, opérateurs et équations discrètes
# ===============================================================================
class SeparatedArray:  # Fonction discrete sous forme séparée
    """
    Discrete separated representation of a function

    This can be instantiated in several ways:
        SeparatedArray(num, shape)
            create a constant object with the value num and the given shape
        SeparatedArray(FD)
            if FD is a SeparatedArray object, return a copy of FD
        SeparatedArray([Array1, Array2, ...])
            Create a SeparatedArray object where the number of dimensions is the
            lenght of the list. The ith term of the sum is given
            by the ith column of each array.
        SeparatedArray(Array)
            if Array of dimension 2: equivalent to SeparatedArray([Array])
            if Array of dimension 1: build a SeparatedArray with 1 term and 1 dimension

    Attributes
    ----------
    data : list of array
        Data for each separated dimension.
    dim : int
        Number of separated dimensions
    shape : list
        Number of nodes in each separated dimension


    Notes
    -----

    SeparatedArray object can be used in arithmetic operations: it support
    addition, subtraction, multiplication and division

    Data Structure
        The data are contained in the data attribute
        The ith column of the jth array contains the descrete values
        of the ith function over the jth dimension
    """

    def __add__(self, A):  # somme
        if isinstance(A, SeparatedArray):
            if self.dim == A.dim and self.shape == A.shape:
                return SeparatedArray(
                    [np.c_[self.data[dd], A.data[dd]] for dd in range(self.dim)]
                )
            else:
                raise NameError("Dimensions doesnt match")
        elif isinstance(A, Number) == 1:
            if A == 0:
                return self
            else:
                return self + SeparatedArray(A, self.shape)
        else:
            return NotImplemented

    def __div__(self, a):  # division (type numérique uniquement)
        if isinstance(a, Number) == 1:
            return SeparatedArray(
                [np.c_[self.data[0] / a]]
                + [np.c_[self.data[dd]] for dd in range(1, self.dim)]
            )
        else:
            return NotImplemented

    def __truediv__(self, a):  # division (type numérique uniquement)
        return self.__div__(a)

    def __getitem__(self, key):
        if not (isinstance(key, tuple)):
            key = [key]  # on renvoie les fonctions définient sur la dimension key
        if len(key) < self.dim:
            key = list(key) + [
                slice(None) for i in range(self.dim - len(key))
            ]  # si il manque des variables on prend tout l'intervale

        ind = []
        dim = []
        # ind contient les indices ou plages d'indices sous forme de liste
        # dim contient les dimensions sur lesquels on génère un tableau
        for ii, kk in enumerate(key):
            if isinstance(kk, (Number, np.floating)):
                ind.append(kk)
            elif isinstance(kk, slice):
                ind.append(kk)
                dim.append(
                    ii
                )  # ind.append(range(self.shape[ii]).__getitem__(kk)); dim.append(ii)
            elif isinstance(kk, list) or (
                isinstance(kk, np.ndarray) and len(kk.shape) == 1
            ):
                ind.append(kk)
                dim.append(ii)
            else:
                assert 0, "Only support numeric, slice, list or np.ndarray key"

        if len(dim) == 0:
            return np.sum(
                np.prod([self.data[dd][key[dd], :] for dd in range(self.dim)], 0)
            )
        elif len(dim) == 1:
            if len(self.data) == 1:
                return np.sum(self.data[0], 1)
            else:
                return np.dot(
                    self.data[dim[0]][ind[dim[0]], :],
                    np.prod(
                        [
                            self.data[dd][key[dd], :]
                            for dd in list(range(dim[0]))
                            + list(range(dim[0] + 1, self.dim))
                        ],
                        0,
                    ),
                )
        elif len(dim) == 2:
            temp = np.prod(
                [
                    self.data[dd][key[dd], :]
                    for dd in list(range(dim[0]))
                    + list(range(dim[0] + 1, dim[1]))
                    + list(range(dim[1] + 1, self.dim))
                ],
                0,
            )
            return np.dot(
                temp * self.data[dim[0]][ind[dim[0]], :],
                self.data[dim[1]][ind[dim[1]], :].T,
            )
        elif len(dim) == 3:
            res = np.zeros(
                (len(ind[dim[0]]), len(ind[dim[1]]), len(ind[dim[2]]))
            )  # initialisation du tableau
            for ii, kk in enumerate(ind[dim[2]]):
                res[:, :, ii] = self[:, :, kk]  # à vérifier
            return res

        else:
            return NotImplemented

    def __setitem__(self, key, value):
        if not (isinstance(key, tuple)):
            key = [key]  # on renvoie les fonctions définient sur la dimension key
        key = list(key)
        if len(key) < self.dim:
            key = list(key) + [
                slice(None) for i in range(self.dim - len(key))
            ]  # si il manque des variables on prend tout l'intervale
        if isinstance(value, list):
            value = np.ndarray(value)

        dim = []  # dim contient les dimensions sur lesquels on génère un tableau
        for ii, kk in enumerate(key):
            if isinstance(kk, list) or isinstance(kk, np.ndarray):
                key[ii] = np.array(kk, dtype=int).reshape(-1)
                if len(key[ii]) == 1:
                    key[ii] = key[ii][0]
            if not (isinstance(kk, (Number, np.floating))):
                dim.append(ii)

        if (
            len(dim) == 0
        ):  # if only one element is modified in the global array. Value must be an int or a float
            assert isinstance(
                value, (Number, np.floating)
            ), "Size of assigning values doesn't match in SeparatedArray.__setitem__"
            FFadd = SeparatedZeros(self.shape)  # SeparatedArray to add
            old_value = np.sum(
                np.prod([self.data[dd][key[dd]] for dd in range(self.dim)], 0)
            )  # old value
            for dd, kk in enumerate(key):
                if dd == 0:
                    FFadd.data[dd][kk] = value - old_value
                else:
                    FFadd.data[dd][kk] = 1.0

        elif len(dim) == 1:
            if isinstance(value, np.ndarray):
                value = value.reshape(-1)
            FFadd = SeparatedZeros(self.shape)  # SeparatedArray to add
            old_value = np.dot(
                self.data[dim[0]][key[dim[0]]],
                np.prod(
                    [
                        self.data[dd][key[dd], :]
                        for dd in list(range(dim[0]))
                        + list(range(dim[0] + 1, self.dim))
                    ],
                    0,
                ),
            )  # old values
            for dd, kk in enumerate(key):
                if dd == dim[0]:
                    FFadd.data[dd][kk] = np.c_[value - old_value]
                else:
                    FFadd.data[dd][kk] = 1.0

        elif len(dim) >= 2:
            if isinstance(value, np.ndarray):
                #                FFadd = ConvertArraytoSeparatedArray(value-SeparatedArray.__getitem__(self,tuple(key)))
                value = ConvertArraytoSeparatedArray(value)

            FFadd = SeparatedZeros(self.shape, nbTerm=self.nbTerm())
            for ii, kk in enumerate(key):
                FFadd.data[ii][kk] = self.data[ii][kk]

            if isinstance(value, (Number, np.floating)):
                FFadd = value - FFadd
            elif isinstance(value, SeparatedArray):
                FFadd_value = SeparatedZeros(self.shape, nbTerm=value.nbTerm())

                if len(value) != len(self):
                    if len(value) == len(dim):
                        nbt = value.nbTerm()
                        value = SeparatedArray(
                            [
                                value.data[dim.index(d)]
                                if d in dim
                                else np.ones((1, nbt))
                                for d in range(len(self))
                            ]
                        )
                    else:
                        assert 0, "Dimension doesn't match"

                for dd, kk in enumerate(key):
                    FFadd_value.data[dd][kk] = value.data[dd]
                #                        if dd in dim:

                FFadd = FFadd_value - FFadd

        self.data = (self + FFadd).data

    #    def __setitem__(self, key, value):
    #        if not(isinstance(key, tuple)): key = [key] #on renvoie les fonctions définient sur la dimension key
    #        if len(key) < self.dim: key = list(key)+[slice(None) for i in range(self.dim - len(key))] #si il manque des variables on prend tout l'intervale
    #
    #        ind = []; dim = []
    #        #ind contient les indices ou plages d'indices sous forme de liste
    #        #dim contient les dimensions sur lesquels on génère un tableau
    #        for ii, kk in enumerate(key):
    #            print(kk)
    #            if isinstance(kk, (Number, np.floating)): ind.append(kk)
    #            elif isinstance(kk, slice): ind.append(kk) ;  dim.append(ii) #ind.append(list(range(self.shape[ii]).__getitem__(kk))); dim.append(ii)
    #            elif isinstance(kk, list) or (isinstance(kk,np.ndarray) and len(kk.shape) == 1): \
    #                ind.append(kk) ; dim.append(ii)
    #
    #        if len(dim) == 0: #if only one element is modified. Value must be an int or a float
    #            assert isinstance(value, (Number, np.floating)), "Size of assigning values doesn't match in SeparatedArray.__setitem__"
    #            FFadd = SeparatedZeros(self.shape) #SeparatedArray to add
    #            old_value = np.sum(np.prod([self.data[dd][key[dd], :] for dd in range(self.dim)] , 0)) #old value
    #            for dd, kk in enumerate(ind):
    #                if dd == 0: FFadd.data[dd][kk] = value - old_value
    #                else: FFadd.data[dd][kk] = 1.
    #
    #        elif len(dim) == 1:
    #            FFadd = SeparatedZeros(self.shape) #SeparatedArray to add
    #            np.prod([self.data[dd][key[dd], :] for dd in list(range(dim[0]))+list(range(dim[0]+1,self.dim))] , 0)
    #
    #            old_value = np.dot(self.data[dim[0]][ind[dim[0]],:], np.prod([self.data[dd][key[dd], :] for dd in list(range(dim[0]))+list(range(dim[0]+1,self.dim))] , 0) ) #old values
    #            for dd, kk in enumerate(ind):
    #                if isinstance(kk,Number): FFadd.data[dd][kk] = 1.
    #                else:
    #                    FFadd.data[dd][kk] = np.c_[value - old_value]
    #        elif len(dim) >= 2:
    #            if isinstance(value,SeparatedArray):
    #                return NotImplemented
    ##                toadd = SeparatedArray_zeros(value.shape, nbTerm=self.nbTerm())
    ##                d1 = 0
    ##                for dd,kk in enumerate(ind):
    ##                    if isinstance(kk, int): FFadd.data[dd][kk] = 1.
    ##                    else:
    ##                        FFadd.data[dd][kk] = toadd.data[d1]
    ##                        d1 += 1
    ##                toadd = value - self.
    #            else: #elif isinstance(value, np.ndarray):
    #                toadd = ConvertArraytoSeparatedArray(value-SeparatedArray.__getitem__(self,tuple(key))) #object SeparatedArray to add with dim=2
    #
    #            FFadd = SeparatedArray_zeros(self.shape, nbTerm = toadd.nbTerm()) #SeparatedArray to add
    #            d1 = 0
    #            for dd,kk in enumerate(ind):
    #                if isinstance(kk, Number): FFadd.data[dd][kk] = 1.
    #                else:
    #                    FFadd.data[dd][kk] = toadd.data[d1]
    #                    d1 += 1
    #
    #        else: return NotImplemented
    #        self.data = (self+FFadd).data

    def __init__(self, listD, N=0):
        if isinstance(listD, Number):
            self.data = [np.ones((nn, 1)) for nn in N]
            self.data[0] = listD * self.data[0]
        elif isinstance(listD, SeparatedArray):
            self.data = [Fd.copy() for Fd in listD.data]
        elif isinstance(listD, np.ndarray):
            self.data = [np.c_[listD]]
        elif isinstance(listD, list):
            self.data = [np.array(Fd.copy()) for Fd in listD]
        self.updateShape()

    def __len__(self):  # retourne le nombre de dimensions
        return len(self.data)  # taille de la liste

    def __mul__(self, a):  # multiplication
        return self.multiply(a)

    def __neg__(self):  # retourne l'opposé
        return SeparatedArray(
            [np.c_[-self.data[0]]] + [np.c_[self.data[dd]] for dd in range(1, self.dim)]
        )

    def __radd__(self, A):  # somme à droite
        return self + A

    def __rmul__(self, a):  # multiplication à droite
        return self * a

    def __rsub__(self, A):  # soustraction à droite (avec un type numérique uniquement)
        if A == 0:
            return -self
        else:
            return -self + SeparatedArray(A, self.shape)

    def __repr__(self):
        res = "SeparatedArray of shape " + str(self.shape)
        for dd, mat in enumerate(self.data):
            res += "\n \nColumn vectors related to subspace " + str(dd) + "\n"
            res += str(mat)
        return res

    #                str(self.data) # affiché avec la fonction print

    def __sub__(self, A):  # soustraction
        if isinstance(A, SeparatedArray) == 1:
            if self.dim == A.dim:
                return SeparatedArray(
                    [np.c_[self.data[0], -A.data[0]]]
                    + [np.c_[self.data[dd], A.data[dd]] for dd in range(1, self.dim)]
                )
            else:
                raise NameError("Dimensions doesnt match")
        elif isinstance(A, Number) == 1:
            if A == 0:
                return self
            else:
                return self + SeparatedArray(-A, self.shape)
        else:
            return NotImplemented

    def copy(self):
        """Return a copy of the object"""
        return self.__class__(self)

    def extrude(self, Nnd):  # on rajoute une dimension par extrusion
        """
        Add a new dimension by extrusion
        (copying the same values along a new dimension)
        This function modfify the object (it doesn't return a new object)

        If Nnd is an int:
            Add a new dimension with Nnd nodes

        If Nnd is a list:
            Add len(Nnd) new dimensions. The number of new nodes for each
            dimension is in the list Nnd
        """
        if isinstance(Nnd, int):
            self.data += [np.ones((Nnd, self.nbTerm()))]
            self.dim = len(self.data)
            self.shape += [Nnd]
        elif isinstance(Nnd, list) == 1:
            for nn in Nnd:
                self.extrude(nn)
        return self

    def getTerm(self, ii):  # retourne uniquement le ième term de la somme
        """
        Return only the term ii of the sum related to the
        separated representation
        """
        return SeparatedArray([np.c_[self.data[dd][:, ii]] for dd in range(self.dim)])

    def nbTerm(self):
        """
        Return the number of terms in the sum related to the
        separated representation
        """
        return np.shape(self.data[0])[1]

    def norm(self, nbvar=1):
        """
        Return the euclidean norm
        If the separated representation contains several variables, the norm can
        be computed indepandantly for each variable using nbvar > 1.
        """
        return np.sqrt(self.squareNorm(nbvar))

    def multiply(self, a):
        """
        SeparatedArray.multiply(a)

        If a is a scalar: multiply all the terms of the first dimension by a
        If a is a SeparatedArray with the same shape, return the element-wise multiplication
        The number of terms of the returned SeparatedArray object can be high (=a.nbTerm()*self.nbTerm())
        """
        if a == 0:
            return 0
        if isinstance(a, Number):
            return SeparatedArray(
                [np.c_[a * self.data[0]]]
                + [np.c_[self.data[dd]] for dd in range(1, self.dim)]
            )
        elif isinstance(a, SeparatedArray):  # implémentation rapide à optimiser
            res = 0
            for ii in range(self.nbTerm()):
                for jj in range(a.nbTerm()):
                    res += SeparatedArray(
                        [
                            np.c_[self.data[dd][:, ii] * a.data[dd][:, jj]]
                            for dd in range(self.dim)
                        ]
                    )
            return res
        else:
            return NotImplemented

    def argsort(self):
        norme = 1
        for dd in range(self.dim):
            norme *= np.linalg.norm(self.data[dd], np.inf, axis=0)
        return norme.argsort()

    def sort(self):
        new_arg = self.argsort()
        for dd in range(self.dim):
            self.data[dd] = self.data[dd][:, new_arg]

    def tensorProd(
        self, FF, modify=False
    ):  # à vérifier dans le cas où le nombre de terme est différent de 1 pour les deux objets SeparatedArray
        """
        SeparatedArray.tensorProd(FF, modify = False)

        If FF is a SeparatedArray: Return the tensor product of two SeparatedArray objects
            The returned object is a SeparatedArray of dimension self.dim+FF.dim

        If FF is a scalar: Return the product a*self. The dimension is unchanged
        if modify == True: the object is modified
        """
        if FF == 0 and modify != True:
            return 0
        if isinstance(FF, Number):
            res = [np.c_[FF * self.data[0]]] + [
                np.c_[self.data[dd]] for dd in range(1, self.dim)
            ]
        elif isinstance(FF, SeparatedArray):
            nbTermFF = FF.nbTerm()
            res = [np.tile(self.data[ii], ((1, nbTermFF))) for ii in range(self.dim)]
            res += [
                np.hstack(
                    [ff[:, [col]] * np.ones(self.nbTerm()) for col in range(nbTermFF)]
                )
                for ff in FF.data
            ]
        else:
            return NotImplemented
        if modify == True:
            self.data = res
            self.updateShape()
            return self
        else:
            return SeparatedArray(res)

    # ===============================================================================
    # Fonctions de réduction
    # ===============================================================================
    # Fonction pour réduire la somme (#a tester si dim>2)
    #    def reduction(self, max_iter = 1000, max_norm_err = 1e-4, max_iter_RS = 5, nbvar=1):
    #        if nbvar == 1: self.reductionPGD2(max_iter, max_norm_err)
    #        else:
    #            NN=np.array(self.shape)//nbvar
    #            CC = [ SeparatedArray( [self.data[dd][var*NN[dd]:(var+1)*NN[dd],:] for dd in range(self.dim) ]).reductionPGD2(max_iter,max_norm_err, max_iter_RS = 2) for var in range(nbvar) ]
    ##            for var in range(nbvar): CC[var].reduction(max_iter,max_norm_err)
    #            nt = np.sum(np.array([CC[var].nbTerm() for var in range(nbvar)]))
    #            self.data = [np.zeros((nbvar*NN[dd], nt)) for dd in range(self.dim)]
    #            t = 0
    #            for var in range(nbvar):
    #                for dd in range(self.dim):
    #                    self.data[dd][var*NN[dd]:(var+1)*NN[dd],t:t+CC[var].nbTerm()] = CC[var].data[dd]
    #                t+=CC[var].nbTerm()
    #        return self

    # Fonction pour réduire la somme (#a tester si dim>2)
    def reduction(self, max_iter=1000, max_norm_err=1e-4, max_iter_RS=5, nbvar=1):
        if nbvar == 1:
            self.reductionPGD2(max_iter, max_norm_err)
        else:
            NN = np.array(self.shape) // nbvar
            CC = [
                SeparatedArray(
                    [
                        self.data[dd][var * NN[dd] : (var + 1) * NN[dd], :]
                        for dd in range(self.dim)
                    ]
                ).reductionPGD2(max_iter, max_norm_err, max_iter_RS=2)
                for var in range(nbvar)
            ]
            #            for var in range(nbvar): CC[var].reduction(max_iter,max_norm_err)
            nt = np.array([CC[var].nbTerm() for var in range(nbvar)]).max()

            self.data = [np.zeros((nbvar * NN[dd], nt)) for dd in range(self.dim)]
            for var in range(nbvar):
                for dd in range(self.dim):
                    self.data[dd][
                        var * NN[dd] : (var + 1) * NN[dd], 0 : CC[var].nbTerm()
                    ] = CC[var].data[dd]
        return self

    def reduction_multivar(
        self, nbvar=0, max_iter=1000, max_norm_err=1e-4, max_iter_RS=5, max_norm_RS=1e-5
    ):
        dim = self.dim
        NN = np.array(self.shape) // nbvar
        #        CC = [ [self.data[dd][var*NN[dd]:(var+1)*NN[dd],:] for dd in range(self.dim) ]
        #            for var in range(nbvar): CC[var].reduction(max_iter,max_norm_err)
        #        nt = np.array([CC[var].nbTerm() for var in range(nbvar)]).max()
        #        self.data = [np.zeros((nbvar*NN[dd], nt)) for dd in range(self.dim)]
        #        for var in range(nbvar):
        #            for dd in range(self.dim):
        #                self.data[dd][var*NN[dd]:(var+1)*NN[dd],0:CC[var].nbTerm()] = CC[var].data[dd]

        #        NN=self.shape ; dim = self.dim
        err_0 = self.norm(nbvar=nbvar)  # changer la norm
        CC = SeparatedArray(self)
        R = 0
        nbR = 0

        for iter in range(max_iter):
            #            print("iter : " + str(iter))
            if iter != 0:
                R_iter = SeparatedArray(R)
            else:
                R_iter = 0
            R += SeparatedArray(1, NN * nbvar)
            nbR += 1
            err_RS = 1
            comp_nR = 0

            while err_RS > max_norm_RS:
                comp_nR += 1
                R_old = SeparatedArray(R)

                for var in range(nbvar):
                    #            if iter<CC.nbTerm(): R += SeparatedArray(CC.getTerm(iter))
                    #            else: R += SeparatedArray(1,NN)
                    #            nbR += 1

                    for d1 in range(dim):
                        prod_aux = 1
                        prod_aux_3 = 1
                        for d2 in list(range(d1)) + list(range(d1 + 1, dim)):
                            prod_aux = prod_aux * np.dot(
                                R.data[d2][var * NN[d2] : (var + 1) * NN[d2]].T,
                                R.data[d2][var * NN[d2] : (var + 1) * NN[d2]],
                            )
                            prod_aux_3 = prod_aux_3 * (
                                np.dot(
                                    R.data[d2][var * NN[d2] : (var + 1) * NN[d2]].T,
                                    CC.data[d2][var * NN[d2] : (var + 1) * NN[d2]],
                                )
                            )
                        Vec = np.dot(
                            CC.data[d1][var * NN[d1] : (var + 1) * NN[d1]], prod_aux_3.T
                        )
                        prod_aux = linalg.inv(prod_aux)
                        #                        R.data[d1][var*NN[d1]:(var+1)*NN[d1]] = np.dot(Vec,prod_aux.T)
                        R.data[d1][var * NN[d1] : (var + 1) * NN[d1]] = np.dot(
                            Vec, prod_aux
                        )

                #                    R.regularise()

                err_RS = (R - R_old).norm(nbvar=nbvar) / err_0
                #                print(err_RS)
                if comp_nR > max_iter_RS:
                    break

            #            print(np.dot(prod_aux,prod_aux.T))

            #            norm_err = (CC-R).norm(nbvar = nbvar)/err_0
            norm_err = (R - R_iter).norm(nbvar=nbvar) / err_0

            print(
                str(iter)
                + " - "
                + str(comp_nR)
                + " - "
                + str(norm_err)
                + " - "
                + str(err_RS)
            )
            if norm_err < max_norm_err:
                break

        self.data = R.data
        return R

    def reductionPGD2(self, max_iter=1000, max_norm_err=1e-4, max_iter_RS=5):
        NN = self.shape
        dim = self.dim
        err_0 = self.norm()
        CC = SeparatedArray(self)
        R = 0
        nbR = 0

        for iter in range(max_iter):
            #            if iter<CC.nbTerm(): R += SeparatedArray(CC.getTerm(iter))
            #            else: R += SeparatedArray(1,NN)
            #            nbR += 1
            R += SeparatedArray(1, NN)
            nbR += 1
            err_RS = 1
            comp_nR = 0
            while err_RS > 1e-5:
                comp_nR += 1
                R_old = SeparatedArray(R)
                for d1 in range(dim):
                    prod_aux = 1
                    prod_aux_3 = 1
                    for d2 in list(range(d1)) + list(range(d1 + 1, dim)):
                        prod_aux = prod_aux * np.dot(R.data[d2].T, R.data[d2])
                        prod_aux_3 = prod_aux_3 * (np.dot(R.data[d2].T, CC.data[d2]))
                    Vec = np.dot(CC.data[d1], prod_aux_3.T)
                    prod_aux = linalg.inv(prod_aux)
                    R.data[d1] = np.dot(Vec, prod_aux.T)

                #                R.regularise()

                err_RS = (R - R_old).norm() / R.norm()
                #                print(err_RS)
                if comp_nR > max_iter_RS:
                    break

            norm_err = (CC - R).norm() / err_0
            #            print(str(iter)+' - '+str(comp_nR)+' - '+str(norm_err))
            if norm_err < max_norm_err:
                break

        self.data = R.data
        return R

    def reductionPGD(self, max_iter=1000, max_norm_err=1e-4, GG=0):
        NN = self.shape
        dim = self.dim
        err_0 = self.norm()
        CC = SeparatedArray(self)
        if GG == 0:
            self.data = []
        else:
            self.data = GG.data

        for iter in range(max_iter):
            ntC = CC.nbTerm()
            R = SeparatedArray(1, NN)
            err_RS = 1
            comp_nR = 0

            while err_RS > 1e-5:
                comp_nR += 1
                R_old = SeparatedArray(R)
                for d1 in range(dim):
                    prod_aux = 1
                    prod_aux_3 = 1
                    for d2 in list(range(d1)) + list(range(d1 + 1, dim)):
                        prod_aux = prod_aux * np.dot(R.data[d2].T, R.data[d2])
                        prod_aux_3 = prod_aux_3 * (np.dot(R.data[d2].T, CC.data[d2]))

                    R.data[d1] = np.dot(CC.data[d1], prod_aux_3.T) / float(prod_aux)
                R.regularise()

                err_RS = (R - R_old).norm() / R.norm()
                #                print(err_RS)
                if comp_nR > 50:
                    break

            CC = CC - R
            if self.data == []:
                self.data = list(R.data)
            else:
                self.data = [np.c_[self.data[dd], R.data[dd]] for dd in range(self.dim)]
            norm_err = CC.norm() / err_0
            #            print(norm_err)
            if norm_err < max_norm_err:
                break

        return self

    def reduction2D(self, max_norm_err=1e-4, pod=1):  # ne marche pas si dim>2
        if self.dim == 2:
            if pod == 0:
                if self.shape[0] < self.shape[1]:
                    CC = np.dot(self.data[1], self.data[0].T)
                    self.data[1] = CC
                    self.data[0] = np.identity(np.shape(CC)[1])
                else:
                    CC = np.dot(self.data[0], self.data[1].T)
                    self.data[0] = CC
                    self.data[1] = np.identity(np.shape(CC)[1])
            else:
                CC = np.dot(self.data[0], self.data[1].T)
                U, S, V = linalg.svd(CC, False)

                # nombre de valeurs propres retenues
                i = 1
                while S[i] / S[0] > max_norm_err:
                    i += 1
                    if i == len(S):
                        break
                nbeig = i + 1
                self.data[0] = U[:, 0:nbeig] * S[0:nbeig]
                self.data[1] = V[0:nbeig, :].T
        else:
            np.disp("Warning: reduction_POD dont work if dim != 2")
        return self

    def regularise(self, nbvar=1):
        """
        In the separated representation, the norm of the functions related
        to all the dimensions expeted the first (the dimension with index 0) is
        set to 1. The norm of functions over the first dimension are modified.

        Numerical problems can occure due to the non uniqueness of
        the separated representation.
        The aim of this method is to provide a separated representation with
        where the norms of every functions are coherent.

        The norms can be computed using many variables defining nbvar.
        See SeparatedArray.norm for more details.
        """

        NN = np.array(self.shape) // nbvar
        for var in range(nbvar):
            for dd in range(1, self.dim):
                norme_dd = np.sum(
                    np.sqrt(
                        np.dot(
                            self.data[dd][var * NN[dd] : (var + 1) * NN[dd], :].T,
                            self.data[dd][var * NN[dd] : (var + 1) * NN[dd], :],
                        )
                    )
                )
                self.data[0][var * NN[dd] : (var + 1) * NN[dd], :] = (
                    self.data[0][var * NN[dd] : (var + 1) * NN[dd], :] * norme_dd
                )
                self.data[dd][var * NN[dd] : (var + 1) * NN[dd], :] = (
                    self.data[dd][var * NN[dd] : (var + 1) * NN[dd], :] / norme_dd
                )

    def squareNorm(self, nbvar=1):
        """
        Return the squared euclidean norm
        """
        # calcule la norme ||self||²
        ntC = np.shape(self.data[0])[1]
        norme2 = 0
        for var in range(nbvar):
            prod_aux = np.ones((ntC, ntC))
            for dd in range(self.dim):
                prod_aux = prod_aux * np.dot(
                    self.data[dd][
                        var * self.shape[dd] // nbvar : (var + 1)
                        * self.shape[dd]
                        // nbvar,
                        :,
                    ].T,
                    self.data[dd][
                        var * self.shape[dd] // nbvar : (var + 1)
                        * self.shape[dd]
                        // nbvar,
                        :,
                    ],
                )
            norme2 += abs(np.sum(np.sum(prod_aux)))
        return norme2

    #    def GetVariable(self, var, nbvar):
    #        return SeparatedArray([self.data[dd][var*self.shape[dd]//nbvar:(var+1)*self.shape[dd]//nbvar,:] for dd in range(self.dim)])

    def GetVariable(self, var, meshPGD):
        specificVariable = [
            meshPGD._GetSpecificVariableRank(dd, var) for dd in range(self.dim)
        ]
        NN = meshPGD.n_nodes
        return SeparatedArray(
            [
                self.data[dd][
                    specificVariable[dd] * NN[dd] : (specificVariable[dd] + 1) * NN[dd],
                    :,
                ]
                for dd in range(self.dim)
            ]
        )

    def swap_space(self, ind1, ind2=0):
        """
        Modify the order of spaces

        Parameters
        ----------
        ind1:
            if ind1 is a list, it contains the new index of each space
                for example ind1[2] is the new index of the space 2
                ind2 is not used in this case
            if ind1 is an integer, the function swap the 2 spaces defined by
                the index ind1 and ind2
        """
        if isinstance(ind1, int):
            if ind1 == ind2:
                return self
            ind1 = (
                range(ind1)
                + [ind2]
                + range(ind1 + 1, ind2)
                + [ind1]
                + range(ind2 + 1, self.dim)
            )
        # sinon, c'est que ind1 contient la liste des indices dans le nouvel ordre
        self.data = [self.data[ind] for ind in ind1]
        self.updateShape()
        return self

    def tile(self, nb_rep, dd=None):
        """
        Modify the SeparatedArray object by repeating the number of times given by reps.

        There is two ways to use this function:

        tile(nb_rep) :
            In that case, the lenght of nb_rep must be equal to self.dim
            nb_rep[ii] contains the number of repetition for the dimension ii
        tile(nb_rep, dd) :
            In that case, the lenght of nb_rep must be equal to the one of dd
            nb_rep[ii] of the number of repetition of the dimension dd[ii]


        Parameters
        ----------
        ind1: list
            The number of repetition along each dimension or along
            the dimensions defined in dd
        dd: list
            If needed, the index of the dimensions to repeat
        """
        # nb_rep est un tableau contenant le nombre de répétitions - dd est un tableau contenant les dimensions à répéter
        if dd == None:
            dd = range(len(nb_rep))
        for ii in range(len(nb_rep)):
            self.data[dd[ii]] = np.tile(self.data[dd[ii]], ((nb_rep[ii], 1)))
        self.updateShape()
        return self

    def toSeparatedArray(self):
        pass

    def updateShape(self):
        """Actualise the shape of the object if there is some modifications"""
        self.dim = len(self.data)
        self.shape = [np.shape(self.data[d1])[0] for d1 in range(self.dim)]


# ===============================================================================
# Fonctions renvoyant des objets FoncD
# ===============================================================================
def ConvertArraytoSeparatedArray(FF):  # ne marche que pour les fonctions 2D
    if isinstance(FF, list):
        dim = len(np.shape(FF.data[0]))
        nbvar = len(FF)
        temp = [ConvertArraytoSeparatedArray(FF.data[var]) for var in range(nbvar)]
        return SeparatedArray(
            [np.vstack([temp[var][dd] for var in range(nbvar)]) for dd in range(dim)]
        )
    elif isinstance(FF, np.ndarray):
        NN = np.shape(FF)
        if NN[0] < NN[1]:
            return SeparatedArray([np.identity(NN[0]), FF.T])
        else:
            return SeparatedArray([FF, np.identity(NN[1])])
    else:
        raise NameError("Not implemented error")


def SeparatedOnes(shape, nbTerm=1):
    """
    Return a new FoncD of given shape filled with ones.
    Please refer to the documentation for FoncD_zeros for further details.

    See also
    ----------
    FoncD_zeros
    """
    return SeparatedArray([np.ones((nn, nbTerm)) for nn in shape])


#    return SeparatedArray(1,shape)


def SeparatedZeros(shape, nbTerm=1):
    """
    Return a new FoncD of given shape filled with zeros.
    If var is None, this function return a FoncD.
    If var is a list, it returns a FoncDVect. Then, shape contains the shape
    of a variable, not the final shape of the full object.

    Parameters
    ----------
    shape: list of ints
        Shape of the new object for example [2,3].
    var: list defining the variables
        see FoncDVect for more details
    nbTerm: int
        Number of terms in the sum (all terms are filled with 0)
        By default nbTerm is 1

    See also
    ----------
    FoncD_ones, FoncDVect
    """
    return SeparatedArray([np.zeros((nn, nbTerm)) for nn in shape])


def MergeSeparatedArray(F1, F2):
    if F1.dim != F2.dim:
        raise NameError("Dimensions doesnt match")
    elif F1.nbTerm() != F2.nbTerm():
        raise NameError("Not implemented error")
    return SeparatedArray(
        [np.vstack((F1.data[dd], F2.data[dd])) for dd in range(F1.dim)]
    )
