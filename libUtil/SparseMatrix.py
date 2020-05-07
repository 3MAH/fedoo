# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 15:15:15 2019

@author: Etienne
"""

import numpy as np
from scipy import sparse
from numbers import Number 


class _BlocSparse():
    
    def __init__(self, nbBlocRow, nbBlocCol,nbpg=None):
        self.data = [[0 for i in range(nbBlocCol)] for j in range(nbBlocRow)]
        self.col = None
        self.row = None
        self.blocShape = None
        self.nbBlocRow = nbBlocRow
        self.nbBlocCol = nbBlocCol
        self.nbpg=nbpg
        
#    def addToBloc(self, Mat, rowBloc, colBloc):
#        #Mat should be a scipy matrix using the csr format
#        bloc = self.data[rowBloc][colBloc]
#        if bloc is 0: 
#            self.data[rowBloc][colBloc] = Mat.copy()
#            self.data_coo[rowBloc][colBloc] = self.data[row][col].tocoo(copy = False)
#            #row are sorted for data_coo
#        else: 
#            bloc.data = bloc.data + Mat.data

    def addToBloc(self, A, B, coef, rowBloc, colBloc):
        #A and B should a scipy matrix using the csr format and with the same number of column per row for each row
        
        NnzColPerRowA = A.indptr[1] #number of non zero column per line for csr matrix A
        NnzColPerRowB = B.indptr[1] #number of non zero column per line for csr matrix B
        nb_pg = self.nbpg
                
        if not(isinstance(coef, Number)): coef = coef.reshape(-1,1,1)
        
        if self.data[rowBloc][colBloc] is 0: 
            if self.nbpg is None:
                self.data[rowBloc][colBloc] = (coef * A.data.reshape(-1,NnzColPerRowA,1) @ (B.data.reshape(-1,1,NnzColPerRowB))) #at each PG we build a nbNode x nbNode matrix
            else:
                self.data[rowBloc][colBloc] = (coef * A.data.reshape(-1,NnzColPerRowA,1)).reshape(nb_pg,-1,NnzColPerRowA).transpose((1,2,0)) @ B.data.reshape(nb_pg,-1,NnzColPerRowB).transpose(1,0,2) #at each PG we build a nbNode x nbNode matrix

        
        else:
            if self.nbpg is None:
                self.data[rowBloc][colBloc] += (coef * A.data.reshape(-1,NnzColPerRowA) @ (B.data.reshape(-1,1,NnzColPerRowB)))           
            else:
                self.data[rowBloc][colBloc] += (coef * A.data.reshape(-1,NnzColPerRowA,1)).reshape(nb_pg,-1,NnzColPerRowA).transpose((1,2,0)) @ B.data.reshape(nb_pg,-1,NnzColPerRowB).transpose(1,0,2) #at each PG we build a nbNode x nbNode matrix
        
        if self.col is None:
            # column indieces of A defined in A.indices are the row indices in final matrix
            # column indieces of B defined in B.indices are the column indices in final matrix
            if self.nbpg is None:
                self.row = (A.indices.reshape(-1,NnzColPerRowA,1) @ np.ones((1,NnzColPerRowB), np.int32)).ravel()
                self.col = (np.ones((NnzColPerRowA,1), np.int32) @ B.indices.reshape(-1,1,NnzColPerRowB) ).ravel()
                self.blocShape = (A.shape[1], B.shape[1])
            else:
                NelA = A.shape[0]//self.nbpg
                NelB = B.shape[0]//self.nbpg
                self.row = (A.indices[0:NelA*NnzColPerRowA].reshape(NelA,NnzColPerRowA,1) @ np.ones((1,NnzColPerRowB), np.int32)).ravel()
                self.col = (np.ones((NnzColPerRowA,1), np.int32) @ B.indices[0:NelB*NnzColPerRowB].reshape(NelB,1,NnzColPerRowB) ).ravel()
                self.blocShape = (A.shape[1], B.shape[1])                
               
    def toCSR(self):    
        ResDat = np.array([self.data[i][j] for i in range(self.nbBlocRow) for j in range(self.nbBlocCol) if self.data[i][j] is not 0]).ravel()
        ResRow = np.array([self.row+i*self.blocShape[0] for i in range(self.nbBlocRow) for j in range(self.nbBlocCol) if self.data[i][j] is not 0], np.int32).ravel()
        ResCol = np.array([self.col+j*self.blocShape[1] for i in range(self.nbBlocRow) for j in range(self.nbBlocCol) if self.data[i][j] is not 0], np.int32).ravel()
        Res = sparse.coo_matrix((ResDat, (ResRow,ResCol)), shape=(self.blocShape[0]*self.nbBlocRow, self.blocShape[1]*self.nbBlocCol), copy = False).tocsr()
#        Res = Res2.tocsr()
#        del Res2 
        Res.data.round(10, Res.data)
        Res.eliminate_zeros()
        return Res





def bloc_matrix(M, nb_bloc, position):
                
    var = position[1]
    var_vir = position[0]      
    M = M.tocsr()
    
    indptr = np.zeros(np.shape(M)[0]*nb_bloc[0]+1, dtype = int)
    indptr[var_vir*np.shape(M)[0]:(var_vir+1)*np.shape(M)[0]+1] = M.indptr
    indptr[(var_vir+1)*np.shape(M)[0]+1 : ] = indptr[(var_vir+1)*np.shape(M)[0]]
    
    MatBloc = sparse.csr_matrix((M.data, M.indices + var*np.shape(M)[1], indptr), shape = (np.shape(M)[0]*nb_bloc[0],np.shape(M)[1]*nb_bloc[1])) 

    return MatBloc

def ColumnBlocMatrix(listBloc, nb_bloc, position):
                
    order = list(np.array(position).argsort())
    position.sort()
    listBloc = [listBloc[ii].tocsr() for ii in order]
    
    NbRowPerBloc = np.shape(listBloc[0])[0]
    indices = np.hstack(Mat.indices for Mat in listBloc) 
    data    = np.hstack(Mat.data for Mat in listBloc) 
    
    indptr = np.empty(NbRowPerBloc*nb_bloc+1, dtype = int)

    ind = 0 #indice defining the begining of the bloc in indptr
    for var in range(nb_bloc):        
        if var in position:            
            Mat = listBloc[position.index(var)]                  
            indptr[var*NbRowPerBloc:(var+1)*NbRowPerBloc+1] = Mat.indptr + ind
            ind += len(Mat.indices)            
        else:
            indptr[var*NbRowPerBloc:(var+1)*NbRowPerBloc+1] = ind          
    
    return sparse.csr_matrix((data, indices, indptr), shape = (NbRowPerBloc*nb_bloc,np.shape(listBloc[0])[1])) 


def RowBlocMatrix(listBloc, nb_bloc, position, coef):    
    return sum([bloc_matrix(coef[ii]*listBloc[ii], (1,nb_bloc), (0,position[ii])) for ii in range(len(listBloc))])
#    return sum([bloc_matrix(listBloc[ii], (1,nb_bloc), (0,position[ii])) if coef[ii] == 1 \
#                else coef[ii]*bloc_matrix(listBloc[ii], (1,nb_bloc), (0,position[ii])) for ii in range(len(listBloc))])
#        