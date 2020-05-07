from fedoo.libAssembly.AssemblyBase import AssemblyBase
from fedoo.libUtil.Variable import *
from fedoo.libUtil.Dimension import ProblemDimension
from fedoo.libUtil.Coordinate import Coordinate
from fedoo.libUtil.PostTreatement import listStressTensor, listStrainTensor
from fedoo.libMesh.Mesh import Mesh
from fedoo.libElement import *
from fedoo.libWeakForm.WeakForm import WeakForm
from fedoo.libConstitutiveLaw.ConstitutiveLaw import ConstitutiveLaw
from fedoo.libUtil.GradOperator import GetGradOperator
from fedoo.libUtil.SparseMatrix import _BlocSparse as BlocSparse
from fedoo.libUtil.SparseMatrix import RowBlocMatrix

from scipy import sparse
import numpy as np
from numbers import Number 
import time

def Create(weakForm, mesh="", elementType="", ID="", **kargs):        
    return Assembly(weakForm, mesh, elementType, ID, **kargs)
               
class Assembly(AssemblyBase):
    __saveOperator = {}       
    __saveMatrixChangeOfBasis = {}   
    __saveMatGaussianQuadrature = {} 
    __saveNodeToPGMatrix = {}
    __savePGtoNodeMatrix = {}
           
    def __init__(self,weakForm, mesh="", elementType="", ID="", **kargs):        
#        t0 = time.time()
        
        if isinstance(weakForm, str):
            weakForm = WeakForm.GetAll()[weakForm]

        if isinstance(mesh, str):
            mesh = Mesh.GetAll()[mesh]
                
        AssemblyBase.__init__(self, ID)
        
        self.__MeshChange = kargs.pop('MeshChange', False)        
        self.__Mesh = mesh   
        self.__weakForm = weakForm
        self.__elmType= elementType #.lower()        
        self.__nb_pg = kargs.pop('nb_pg', None)
        if self.__nb_pg is None: self.__nb_pg = GetDefaultNbPG(elementType, mesh)
                    
        #print('Finite element operator for Assembly "' + ID + '" built in ' + str(time.time()-t0) + ' seconds')
        
        self.computeMatrixMethod = 'new'

    def ComputeGlobalMatrix(self, compute = 'all'):
        """
        Compute the global matrix and global vector related to the assembly
        if compute = 'all', compute the global matrix and vector
        if compute = 'matrix', compute only the matrix
        if compute = 'vector', compute only the vector
        """
        computeMatrixMethod = self.computeMatrixMethod
        
        nb_pg = self.__nb_pg
        mesh = self.__Mesh
        
        if self.__MeshChange == True: 
            if mesh.GetID() in Assembly.__saveMatrixChangeOfBasis: del Assembly.__saveMatrixChangeOfBasis[mesh.GetID()]
            Assembly.PreComputeElementaryOperators(mesh, self.__elmType, nb_pg=nb_pg)
                 
        nvar = Variable.GetNumberOfVariable()
        wf = self.__weakForm.GetDifferentialOperator(mesh)                

        MatGaussianQuadrature = Assembly.__GetGaussianQuadratureMatrix(mesh, self.__elmType, nb_pg=nb_pg)
        MatrixChangeOfBasis = Assembly.__GetChangeOfBasisMatrix(mesh)
        
        if computeMatrixMethod == 'new':
            MM = BlocSparse(nvar, nvar, self.__nb_pg)
            
            #sl contains list of slice object that contains the dimension for each variable
            #size of VV and sl must be redefined for case with change of basis
            VV = 0
            nbNodes = self.__Mesh.GetNumberOfNodes()            
            sl = [slice(i*nbNodes, (i+1)*nbNodes) for i in range(nvar)] 
            
            for ii in range(len(wf.op)):
                if compute == 'matrix' and wf.op[ii] is 1: continue
                if compute == 'vector' and wf.op[ii] is not 1: continue
            
                coef_vir = [1] ; var_vir = [wf.op_vir[ii].u] #list in case there is an angular variable
                if not(Variable.GetDerivative(var_vir[0]) is None): 
                    var_vir.append(Variable.GetDerivative(var_vir[0])[0])
                    coef_vir.append(Variable.GetDerivative(var_vir[0])[1])
#                Matvir = (RowBlocMatrix(Assembly.__GetElementaryOp(mesh, wf.op_vir[ii], self.__elmType, nb_pg=nb_pg), nvar, var_vir, coef_vir) * MatrixChangeOfBasis).T
                #check how it appens with change of variable and rotation dof
                #add coef_vir somewhere
                Matvir = Assembly.__GetElementaryOp(mesh, wf.op_vir[ii], self.__elmType, nb_pg=nb_pg)

                if isinstance(wf.coef[ii], Number): 
                    coef_PG = wf.coef[ii]*MatGaussianQuadrature.data #MatGaussianQuadrature.data is the diagonal of MatGaussianQuadrature
                else:
                    coef_PG = Assembly.__ConvertToGaussPoints(mesh, wf.coef[ii][:], self.__elmType, nb_pg=nb_pg)*MatGaussianQuadrature.data                                                 
    
                if wf.op[ii] == 1: #only virtual operator -> compute a vector 
                    if VV is 0: VV = np.zeros((self.__Mesh.GetNumberOfNodes() * nvar))
                    VV[sl[var_vir[0]]] = VV[sl[var_vir[0]]] - coef_vir[0] * Matvir[0].T * (coef_PG) #this line may be optimized
                        
                else: #virtual and real operators -> compute a matrix
                    coef = [1] ; var = [wf.op[ii].u] #list in case there is an angular variable
                    if not(Variable.GetDerivative(var[0]) is None):     
                        var.append(Variable.GetDerivative(var[0])[0])
                        coef.append(Variable.GetDerivative(var[0])[1])
#                    Mat    =  RowBlocMatrix(Assembly.__GetElementaryOp(mesh, wf.op[ii], self.__elmType, nb_pg=nb_pg), nvar, var, coef)         * MatrixChangeOfBasis             
                    Mat    =  Assembly.__GetElementaryOp(mesh, wf.op[ii], self.__elmType, nb_pg=nb_pg)
    
                    #Possibility to increase performance for multivariable case 
                    #the structure should be the same for derivative dof, so the blocs could be computed altogether
                    for i in range(len(Mat)):
                        for j in range(len(Matvir)):
                            MM.addToBloc(Matvir[j], Mat[i], (coef[i]*coef_vir[j]) * coef_PG, var_vir[j], var[i])
            
            if compute != 'vector': 
                if MatrixChangeOfBasis is 1: 
                    self.SetMatrix(MM.toCSR()*MatrixChangeOfBasis) #format csr         
                else: 
                    self.SetMatrix(MatrixChangeOfBasis.T * MM.toCSR() * MatrixChangeOfBasis) #format csr         
            if compute != 'matrix': 
                if VV is 0: self.SetVector(0)
                elif MatrixChangeOfBasis is 1: self.SetVector(VV) #numpy array
                else: self.SetVector(MatrixChangeOfBasis.T * VV)                     
            
        elif computeMatrixMethod == 'old':
            MM = 0
            VV = 0
            
            for ii in range(len(wf.op)):
                if compute == 'matrix' and wf.op[ii] is 1: continue
                if compute == 'vector' and wf.op[ii] is not 1: continue
            
                coef_vir = [1] ; var_vir = [wf.op_vir[ii].u] #list in case there is an angular variable
                if not(Variable.GetDerivative(var_vir[0]) is None): 
                    var_vir.append(Variable.GetDerivative(var_vir[0])[0])
                    coef_vir.append(Variable.GetDerivative(var_vir[0])[1])
                Matvir = (RowBlocMatrix(Assembly.__GetElementaryOp(mesh, wf.op_vir[ii], self.__elmType, nb_pg=nb_pg), nvar, var_vir, coef_vir) * MatrixChangeOfBasis).T
    
                if wf.op[ii] == 1: #only virtual operator -> compute a vector 
                    if isinstance(wf.coef[ii], Number): 
                        VV = VV - wf.coef[ii]*Matvir * MatGaussianQuadrature.data
                    else:
                        coef_PG = Assembly.__ConvertToGaussPoints(mesh, wf.coef[ii][:], self.__elmType, nb_pg=nb_pg)*MatGaussianQuadrature.data                             
                        VV = VV - Matvir * (coef_PG)
                        
                else: #virtual and real operators -> compute a matrix
                    coef = [1] ; var = [wf.op[ii].u] #list in case there is an angular variable
                    if not(Variable.GetDerivative(var[0]) is None):     
                        var.append(Variable.GetDerivative(var[0])[0])
                        coef.append(Variable.GetDerivative(var[0])[1])
                    Mat    =  RowBlocMatrix(Assembly.__GetElementaryOp(mesh, wf.op[ii], self.__elmType, nb_pg=nb_pg), nvar, var, coef)         * MatrixChangeOfBasis             
    
                    if isinstance(wf.coef[ii], Number): #and self.op_vir[ii] != 1: 
                        MM = MM + wf.coef[ii]*Matvir * MatGaussianQuadrature * Mat  
                    else:
                        coef_PG = Assembly.__ConvertToGaussPoints(mesh, wf.coef[ii][:], self.__elmType, nb_pg=nb_pg)                    
                        CoefMatrix = sparse.csr_matrix( (MatGaussianQuadrature.data*coef_PG, MatGaussianQuadrature.indices, MatGaussianQuadrature.indptr), shape = MatGaussianQuadrature.shape)   
                        MM = MM + Matvir * CoefMatrix * Mat                

#            MM = MM.tocsr()
#            MM.eliminate_zeros()
            if compute != 'vector': self.SetMatrix(MM) #format csr         
            if compute != 'matrix': self.SetVector(VV) #numpy array
    
    def SetMesh(self, mesh):
        self.__Mesh = mesh

    def GetMesh(self):
        return self.__Mesh
    
    def GetMatrixChangeOfBasis(self):
        return Assembly.__GetChangeOfBasisMatrix(self.__Mesh)

    def Update(self, pb, time=None, compute = 'all'):
        """
        Update the associated weak form and assemble the global matrix
        Parameters: 
            - pb: a Problem object containing the Dof values
            - time: the current time        
        """
        outValues = self.__weakForm.Update(self, pb, time)
        self.ComputeGlobalMatrix(compute)
        return outValues

    def ResetTimeIncrement(self):
        """
        Reset the current time increment (internal variable in the constitutive equation)
        Doesn't assemble the new global matrix. Use the Update method for that purpose.
        """
        self.__weakForm.ResetTimeIncrement()        

    def NewTimeIncrement(self):
        """
        Apply the modification to the constitutive equation required at each change of time increment. 
        Generally used to increase non reversible internal variable
        Doesn't assemble the new global matrix. Use the Update method for that purpose.
        """
        self.__weakForm.NewTimeIncrement()        
 
    def Reset(self):
        """
        Reset the assembly to it's initial state.
        Internal variable in the constitutive equation are reinitialized 
        and stored global matrix and vector are deleted
        """
        self.__weakForm.Reset()    
        self.deleteGlobalMatrix()
      
    @staticmethod         
    def PreComputeElementaryOperators(mesh, elementType, nb_pg = None, **kargs): #Précalcul des opérateurs dérivés suivant toutes les directions (optimise les calculs en minimisant le nombre de boucle)               
        #initialisation    
        if nb_pg is None: NumberOfGaussPoint = GetDefaultNbPG(elementType, mesh)
        else: NumberOfGaussPoint = nb_pg
        
        objElement = eval(elementType)

        if isinstance(objElement, dict):               
            for val in set(objElement.values()):               
                Assembly.PreComputeElementaryOperators(mesh, val, nb_pg, **kargs)
            return

        elmRef = objElement(NumberOfGaussPoint)                          
               
        Nnd = mesh.GetNumberOfNodes()
        Nel = mesh.GetNumberOfElements()
        elm = mesh.GetElementTable()
        nNd_elm = np.shape(elm)[1]
        crd = mesh.GetNodeCoordinates()
        dim = ProblemDimension.GetDoF()
        
        if NumberOfGaussPoint == 0: # in this case, it is a finite difference mesh
            # we compute the operators directly from the element library
            OP = elmRef.computeOperator(crd,elm)
            Assembly.__saveMatGaussianQuadrature[(mesh.GetID(),NumberOfGaussPoint)] = sparse.identity(OP[0][0].shape[0], 'd', format= 'csr') #No gaussian quadrature in this case : nodal identity matrix
            Assembly.__savePGtoNodeMatrix[(mesh.GetID(), NumberOfGaussPoint)] = 1  #no need to translate between pg and nodes because no pg 
            Assembly.__saveNodeToPGMatrix[(mesh.GetID(), NumberOfGaussPoint)] = 1                                    
            Assembly.__saveMatrixChangeOfBasis[mesh.GetID()] = 1 # No change of basis:  MatrixChangeOfBasis = 1 
            Assembly.__saveOperator[(mesh.GetID(),elementType,NumberOfGaussPoint)] = OP #elmRef.computeOperator(crd,elm)
            return                                

        elmRefGeom = eval(mesh.GetElementShape())(NumberOfGaussPoint)
        nNd_elm_geom = len(elmRefGeom.xi_nd)
        elm_geom = elm[:,:nNd_elm_geom]

        localFrame = mesh.GetLocalFrame()           
        nb_elm_nd = np.bincount(elm_geom.reshape(-1)) #len(nb_elm_nd) = Nnd
        
        vec_xi = elmRef.xi_pg

        PGtoNode = np.linalg.pinv(elmRefGeom.ShapeFunctionPG) #pseudo-inverse of NodeToPG
#        PGtoNode = np.linalg.inv(np.dot(elmRef.GeometricalShapeFunctionPG.T , elmRef.GeometricalShapeFunctionPG)) 
#        PGtoNode = np.dot(PGtoNode , elmRef.GeometricalShapeFunctionPG.T) #inverse of the NodeToPG matrix (built from the values of the shapeFuctions at PG) based on the least square method
        
        elmRefGeom.ComputeJacobianMatrix(crd[elm_geom], vec_xi, localFrame) #elmRef.JacobianMatrix, elmRef.detJ, elmRef.inverseJacobian
        derivativePG = np.matmul(elmRefGeom.inverseJacobian , elmRef.ShapeFunctionDerivativePG)

        nb_dir_deriv = derivativePG.shape[-2] 
        nop = nb_dir_deriv+1 #nombre d'opérateur à discrétiser
        if hasattr(elmRef,'ShapeFunctionSecondDerivativePG'):
            #TODO : only work for beam. Consider revising in the future
#            secondDerivativePG = np.matmul(elmRefGeom.inverseJacobian , elmRef.ShapeFunctionSecondDerivativePG)
            secondDerivativePG = np.matmul(elmRefGeom.inverseJacobian**2 , elmRef.ShapeFunctionSecondDerivativePG)
            
                        
            nop += nb_dir_deriv
            computeSecondDerivativeOp = True
        else: computeSecondDerivativeOp = False

        NbDoFperNode = np.shape(elmRef.ShapeFunctionPG)[1]//nNd_elm
        if NbDoFperNode > 1: #for bernoulli beam (of plate in the future)        
            AngularDoF = True 
        else:
            AngularDoF = False
                       
        range_nbPG = np.arange(NumberOfGaussPoint)        
         

        if Assembly.__GetChangeOfBasisMatrix(mesh) is 1: ChangeOfBasis = False
        else: 
            ChangeOfBasis = True
            range_nNd_elm = np.arange(nNd_elm)
                 
        #-------------------------------------------------------------------
        # Assemblage
        #-------------------------------------------------------------------  
        gaussianQuadrature = (elmRefGeom.detJ * elmRefGeom.w_pg).T.reshape(-1) 

        row = np.empty((Nel, NumberOfGaussPoint, nNd_elm)) ; col = np.empty((Nel, NumberOfGaussPoint, nNd_elm)) ; col2 = np.empty((Nel, NumberOfGaussPoint, nNd_elm))                
        data = [[np.empty((Nel, NumberOfGaussPoint, nNd_elm)) for j in range(NbDoFperNode)] for i in range(nop)] 
        dataNodeToPG = np.empty((Nel, NumberOfGaussPoint, nNd_elm_geom))

        row[:] = np.arange(Nel).reshape((-1,1,1)) + range_nbPG.reshape(1,-1,1)*Nel 
        col[:] = elm.reshape((Nel,1,nNd_elm))
        if ChangeOfBasis: col2[:] = np.arange(Nel).reshape((-1,1,1)) + range_nNd_elm.reshape((1,1,-1))*Nel 

        dataPGtoNode = PGtoNode.T.reshape((1,NumberOfGaussPoint,nNd_elm_geom))/nb_elm_nd[elm_geom].reshape((Nel,1,nNd_elm_geom)) #shape = (Nel, NumberOfGaussPoint, nNd_elm)   
        dataNodeToPG[:] = elmRefGeom.ShapeFunctionPG.reshape((1,NumberOfGaussPoint,nNd_elm_geom))    
        data[0][0][:] = elmRef.ShapeFunctionPG[:,:nNd_elm].reshape((1,NumberOfGaussPoint,nNd_elm))

        for dir_deriv in range(nb_dir_deriv):
            data[dir_deriv+1][0][:] = derivativePG[..., dir_deriv, :nNd_elm]
            if computeSecondDerivativeOp:
                data[1 + nb_dir_deriv + dir_deriv][0][:] = secondDerivativePG[...,dir_deriv, :nNd_elm]
       
        if AngularDoF: #angular dof for C1 elements
            for j in range(1,NbDoFperNode):
                data[0][j][:] = elmRef.ShapeFunctionPG[:,j*nNd_elm:(j+1)*nNd_elm].reshape((1,NumberOfGaussPoint,nNd_elm))
                for dir_deriv in range(nb_dir_deriv):
                    data[dir_deriv+1][j][:] = derivativePG[...,dir_deriv, j*nNd_elm:(j+1)*nNd_elm]
                    if computeSecondDerivativeOp:
                        data[1 + nb_dir_deriv + dir_deriv][j][:] = secondDerivativePG[...,dir_deriv, j*nNd_elm:(j+1)*nNd_elm]      
        
        row_geom = np.reshape(row[...,:nNd_elm_geom], -1) ; col_geom = np.reshape(col[...,:nNd_elm_geom], -1)
        row = np.reshape(row,-1) ; col = np.reshape(col,-1) ; col2 = np.reshape(col2,-1)                            
        
        if ChangeOfBasis: Ncol = Nel * nNd_elm
        else: 
            Ncol = Nnd
            col2 = col

        op_dd = [ [sparse.coo_matrix((data[i][j].reshape(-1),(row,col2)), shape=(Nel*NumberOfGaussPoint , Ncol) ).tocsr() for j in range(NbDoFperNode) ] for i in range(nop)]        

#        data = [sparse.diags(gaussianQuadrature, 0, format='csr')] #matrix to get the gaussian quadrature (integration over each element)
        Assembly.__saveMatGaussianQuadrature[(mesh.GetID(),NumberOfGaussPoint)] = sparse.diags(gaussianQuadrature, 0, format='csr') #matrix to get the gaussian quadrature (integration over each element)        
        #matrix to compute the node values from pg
#        data.extend([sparse.coo_matrix((sp.reshape(dataPGtoNode,-1),(col,row)), shape=(Nel * nNd_elm , Nel*NumberOfGaussPoint) )])
        Assembly.__savePGtoNodeMatrix[(mesh.GetID(), NumberOfGaussPoint)] = sparse.coo_matrix((dataPGtoNode.reshape(-1),(col_geom,row_geom)), shape=(Nnd,Nel*NumberOfGaussPoint) ).tocsr() #matrix to compute the node values from pg using the geometrical shape functions 
        #matrix to compute the pg values from nodes using the geometrical shape functions (no angular dof)
        Assembly.__saveNodeToPGMatrix[(mesh.GetID(), NumberOfGaussPoint)] = sparse.coo_matrix((sp.reshape(dataNodeToPG,-1),(row_geom,col_geom)), shape=(Nel*NumberOfGaussPoint, Nnd) ).tocsr() #matrix to compute the pg values from nodes using the geometrical shape functions (no angular dof)

        
        data = {0: op_dd[0]} #data is a dictionnary
        for i in range(nb_dir_deriv): 
            data[1, i] = op_dd[i+1]
            if computeSecondDerivativeOp:
                data[2,i] = op_dd[i+1+nb_dir_deriv]
        Assembly.__saveOperator[(mesh.GetID(),elementType,NumberOfGaussPoint)] = data   
        
    @staticmethod
    def __GetElementaryOp(mesh, deriv, elementType, nb_pg=None): #calcul la discrétision relative à un seul opérateur dérivé   
        if nb_pg is None: nb_pg = GetDefaultNbPG(elementType, mesh)

        if isinstance(eval(elementType), dict):
            elementDict = eval(elementType)
            elementType = elementDict.get(Variable.GetName(deriv.u))
            if elementType is None: elementType = elementDict.get('default')
                
        if not((mesh.GetID(),elementType,nb_pg) in Assembly.__saveOperator):
            Assembly.PreComputeElementaryOperators(mesh, elementType, nb_pg)
        
        data = Assembly.__saveOperator[(mesh.GetID(),elementType,nb_pg)]

        if deriv.ordre == 0 and 0 in data:
            return data[0]
        
        #extract the mesh coordinate that corespond to coordinate rank given in deriv.x     
        ListMeshCoordinateIDRank = [Coordinate.GetRank(crdID) for crdID in mesh.GetCoordinateID()]
        if deriv.x in ListMeshCoordinateIDRank: xx= ListMeshCoordinateIDRank.index(deriv.x)
        else: return data[0] #if the coordinate doesnt exist, return operator without derivation
                         
        if (deriv.ordre, xx) in data:
            return data[deriv.ordre, xx]
        else: assert 0, "Operator unavailable"            

    @staticmethod
    def __GetGaussianQuadratureMatrix(mesh, elementType, nb_pg=None): #calcul la discrétision relative à un seul opérateur dérivé   
        if nb_pg is None: nb_pg = GetDefaultNbPG(elementType, mesh)        
        if not((mesh.GetID(),nb_pg) in Assembly.__saveMatGaussianQuadrature):
            Assembly.PreComputeElementaryOperators(mesh, elementType, nb_pg)
        return Assembly.__saveMatGaussianQuadrature[(mesh.GetID(),nb_pg)]
    
    @staticmethod
    def __GetGaussianPointToNodeMatrix(mesh, elementType, nb_pg=None): #calcul la discrétision relative à un seul opérateur dérivé   
        if nb_pg is None: nb_pg = GetDefaultNbPG(elementType, mesh)        
        if not((mesh.GetID(),nb_pg) in Assembly.__savePGtoNodeMatrix):
            Assembly.PreComputeElementaryOperators(mesh, elementType, nb_pg)        
        return Assembly.__savePGtoNodeMatrix[(mesh.GetID(),nb_pg)]
    
    @staticmethod
    def __GetNodeToGaussianPointMatrix(mesh, elementType, nb_pg=None): #calcul la discrétision relative à un seul opérateur dérivé   
        if nb_pg is None: nb_pg = GetDefaultNbPG(elementType, mesh)
        if not((mesh.GetID(),nb_pg) in Assembly.__saveNodeToPGMatrix):
            Assembly.PreComputeElementaryOperators(mesh, elementType, nb_pg)
        
        return Assembly.__saveNodeToPGMatrix[(mesh.GetID(),nb_pg)]
    

    @staticmethod
    def __GetChangeOfBasisMatrix(mesh): # change of basis matrix for beam or plate elements
        if not(mesh.GetID()) in Assembly.__saveMatrixChangeOfBasis:        
            ### change of basis treatment for beam or plate elements
            MatrixChangeOfBasis = 1
            computeMatrixChangeOfBasis = False

            Nnd = mesh.GetNumberOfNodes()
            Nel = mesh.GetNumberOfElements()
            elm = mesh.GetElementTable()
            nNd_elm = np.shape(elm)[1]            
            crd = mesh.GetNodeCoordinates()
            dim = ProblemDimension.GetDoF()
            localFrame = mesh.GetLocalFrame()
            elmRefGeom = eval(mesh.GetElementShape())()
    #        xi_nd = elmRefGeom.xi_nd
            xi_nd = GetNodePositionInElementCoordinates(mesh.GetElementShape(), nNd_elm) #function to define

            if 'X' in mesh.GetCoordinateID() and 'Y' in mesh.GetCoordinateID(): #if not in physical space, no change of variable
                for nameVector in Variable.ListVector():
                    if Variable.GetVectorCoordinateSystem(nameVector) == 'global':
                        if computeMatrixChangeOfBasis == False:
                            range_nNd_elm = np.arange(nNd_elm) 
                            computeMatrixChangeOfBasis = True
                            Nvar = Variable.GetNumberOfVariable()
                            listGlobalVector = []  ; listLocalVariable = list(range(Nvar))
    #                        MatrixChangeOfBasis = sparse.lil_matrix((Nvar*Nel*nNd_elm, Nvar*Nnd)) #lil is very slow because it change the sparcity of the structure
                        listGlobalVector.append(Variable.GetVector(nameVector))                
                        listLocalVariable = [i for i in listLocalVariable if not(i in listGlobalVector[-1])]
                #Data to build MatrixChangeOfBasis with coo sparse format
                if computeMatrixChangeOfBasis:
                    rowMCB = np.empty((len(listGlobalVector)*Nel, nNd_elm, dim,dim))
                    colMCB = np.empty((len(listGlobalVector)*Nel, nNd_elm, dim,dim))
                    dataMCB = np.empty((len(listGlobalVector)*Nel, nNd_elm, dim,dim))
                    LocalFrameEl = elmRefGeom.GetLocalFrame(crd[elm], xi_nd, localFrame) #array of shape (Nel, nb_nd, nb of vectors in basis = dim, dim)
                    for ivec, vec in enumerate(listGlobalVector):
                        dataMCB[ivec*Nel:(ivec+1)*Nel] = LocalFrameEl                    
                        rowMCB[ivec*Nel:(ivec+1)*Nel] = np.arange(Nel).reshape(-1,1,1,1) + range_nNd_elm.reshape(1,-1,1,1)*Nel + np.array(vec).reshape(1,1,-1,1)*(Nel*nNd_elm)
                        colMCB[ivec*Nel:(ivec+1)*Nel] = elm.reshape(Nel,nNd_elm,1,1) + np.array(vec).reshape(1,1,1,-1)*Nnd        
    
            if computeMatrixChangeOfBasis:
                MatrixChangeOfBasis = sparse.coo_matrix((sp.reshape(dataMCB,-1),(sp.reshape(rowMCB,-1),sp.reshape(colMCB,-1))), shape=(Nel*nNd_elm*Nvar, Nnd*Nvar))
                for var in listLocalVariable:  
                    MatrixChangeOfBasis = MatrixChangeOfBasis.tolil()
                    MatrixChangeOfBasis[ range(var*Nel*nNd_elm , (var+1)*Nel*nNd_elm)  ,  range(var*Nel*nNd_elm , (var+1)*Nel*nNd_elm) ] = 1                    
                MatrixChangeOfBasis = MatrixChangeOfBasis.tocsr()
            
            Assembly.__saveMatrixChangeOfBasis[mesh.GetID()] = MatrixChangeOfBasis   
            return MatrixChangeOfBasis
        
        return Assembly.__saveMatrixChangeOfBasis[mesh.GetID()]

    @staticmethod
    def __GetResultGaussPoints(mesh, operator, U, elementType, nb_pg=None):  #return the results at GaussPoints      
        res = 0
        nvar = Variable.GetNumberOfVariable()

        MatrixChangeOfBasis = Assembly.__GetChangeOfBasisMatrix(mesh)
        
        for ii in range(len(operator.op)):
            var = [operator.op[ii].u] ; coef = [1] 
            if not(Variable.GetDerivative(var[0]) is None):     
                var.append(Variable.GetDerivative(var[0])[0])
                coef.append(Variable.GetDerivative(var[0])[1])
            assert operator.op_vir[ii]==1, "Operator virtual are only required to build FE operators, but not to get element results"

            if isinstance(operator.coef[ii], Number): coef_PG = operator.coef[ii]                 
            else: coef_PG = Assembly.__ConvertToGaussPoints(mesh, operator.coef[ii][:], elementType, nb_pg)

            res += coef_PG * (RowBlocMatrix(Assembly.__GetElementaryOp(mesh, operator.op[ii], elementType, nb_pg) , nvar, var, coef) * MatrixChangeOfBasis * U)
        return res

    @staticmethod
    def __ConvertToGaussPoints(mesh, values, elementType, nb_pg=None):         
        """
        Convert an array of values related to a specific mesh (Nodal values, Element Values or Points of Gauss values) to the gauss points
        mesh: the considered Mesh object
        values: array containing the values (nodal or element value)
        The shape of the array is tested.
        """       
        NumberOfGaussPointValues = Assembly.__saveMatGaussianQuadrature[(mesh.GetID(),nb_pg)].shape[0]
        test = 0
        if len(values) == mesh.GetNumberOfNodes(): 
            typeOfValues = 'Node' #fonction définie aux noeuds   
            test+=1               
        if len(values) == mesh.GetNumberOfElements(): 
            typeOfValues = 'Element' #fonction définie aux éléments
            test += 1
        if len(values) == NumberOfGaussPointValues:
            typeOfValues = 'PG'
            test += 1
        assert test, "Error: data doesn't match with the number of nodes, number of elements or number of gauss points."
        if test>1: "Warning: kind of data is confusing. " + typeOfValues +" values choosen."

        if typeOfValues == 'Node': 
            return Assembly.__GetNodeToGaussianPointMatrix(mesh, elementType, nb_pg) * values
        if typeOfValues == 'Element':
            NumberOfGaussPoints = NumberOfGaussPointValues//len(values)
            if len(np.shape(values)) == 1: return np.tile(values.copy(),NumberOfGaussPoints)
            else: return np.tile(values.copy(),[NumberOfGaussPoints,1])
            
        return values
                
    def GetElementResult(self, operator, U):
        """
        Not a Static Method.

        Return some element results based on the finite element discretization of 
        a differential operator on a mesh being given the dof results and the type of elements.
        
        Parameters
        ----------
        mesh: string or Mesh 
            If mesh is a string, it should be a meshID.
            Define the mesh to get the results from
            
        operator: OpDiff
            Differential operator defining the required results
         
        U: numpy.ndarray
            Vector containing all the DoF solution 
            
        Return: numpy.ndarray
            A Vector containing the values on each element. 
            It is computed using an arithmetic mean of the values from gauss points
            The vector lenght is the number of element in the mesh              
        """
                
        res = Assembly.__GetResultGaussPoints(self.__Mesh, operator, U, self.__elmType, self.__nb_pg)
        NumberOfGaussPoint = res.shape[0]//self.__Mesh.GetNumberOfElements()
        return np.reshape(res, (NumberOfGaussPoint,-1)).sum(0) / NumberOfGaussPoint

    def GetGaussPointResult(self, operator, U):
        """
        Return some results at element Gauss points based on the finite element discretization of 
        a differential operator on a mesh being given the dof results and the type of elements.
        
        Parameters
        ----------           
        operator: OpDiff
            Differential operator defining the required results
         
        U: numpy.ndarray
            Vector containing all the DoF solution 
            
        Return: numpy.ndarray
            A Vector containing the values on each point of gauss for each element. 
            The vector lenght is the number of element time the number of Gauss points per element
        """
        return Assembly.__GetResultGaussPoints(self.__Mesh, operator, U, self.__elmType, self.__nb_pg)        

    def GetNodeResult(self, operator, U):
        """
        Not a Static Method.

        Return some node results based on the finite element discretization of 
        a differential operator on a mesh being given the dof results and the type of elements.
        
        Parameters
        ----------
        operator: OpDiff
            Differential operator defining the required results
         
        U: numpy.ndarray
            Vector containing all the DoF solution         
            
        Return: numpy.ndarray            
            A Vector containing the values on each node. 
            An interpolation is used to get the node values from the gauss point values on each element. 
            After that, an arithmetic mean is used to compute a single node value from all adjacent elements.
            The vector lenght is the number of nodes in the mesh  
        """
        
        GaussianPointToNodeMatrix = Assembly.__GetGaussianPointToNodeMatrix(self.__Mesh, self.__elmType, self.__nb_pg)
        res = Assembly.__GetResultGaussPoints(self.__Mesh, operator, U, self.__elmType, self.__nb_pg)
        return GaussianPointToNodeMatrix * res        

    def ConvertData(self, data, convertFrom, convertTo):
        assert (convertFrom in ['Node','GaussPoint','Element']) and (convertTo in ['Node','GaussPoint','Element']), "only possible to convert 'Node', 'Element' and 'GaussPoint' values"
        if convertFrom == convertTo: return data       
        if convertFrom == 'Node': 
            data = Assembly.__GetNodeToGaussianPointMatrix(self.__Mesh, self.__elmType, self.__nb_pg) * data
            convertFrom = 'GaussPoint'
        elif convertFrom == 'Element':             
            NumberOfGaussPoints = self.__nb_pg # Assembly.__saveMatGaussianQuadrature[(self.__Mesh.GetID(),self.__nb_pg)].shape[0]//len(data)
            if len(np.shape(data)) == 1: data = np.tile(data.copy(),NumberOfGaussPoints)
            else: data = np.tile(data.copy(),[NumberOfGaussPoints,1])
            convertFrom = 'GaussPoint'            
        # from here convertFrom should be 'PG'
        if convertTo == 'Node': 
            return Assembly.__GetGaussianPointToNodeMatrix(self.__Mesh, self.__elmType, self.__nb_pg) * data 
        elif convertTo == 'Element': 
            NumberOfGaussPoint = self.__nb_pg #data.shape[0]//self.__Mesh.GetNumberOfElements()
            return np.reshape(data, (NumberOfGaussPoint,-1)).sum(0) / NumberOfGaussPoint
        else: return data 
            
            

    def GetStressTensor(self, U, constitutiveLaw, Type="Nodal"):
        """
        Not a static method.
        Return the Stress Tensor of an assembly using the Voigt notation as a python list. 
        The total displacement field and a ConstitutiveLaw have to be given.
        
        Can only be used for linear constitutive law. 
        For non linear ones, use the GetStress method of the ConstitutiveLaw object.

        Options : 
        - Type :"Nodal", "Element" or "GaussPoint" integration (default : "Nodal")

        See GetNodeResult, GetElementResult and GetGaussPointResult.

        example : 
        S = SpecificAssembly.GetStressTensor(Problem.Problem.GetDoFSolution('all'), SpecificConstitutiveLaw)
        """
        if isinstance(constitutiveLaw, str):
            constitutiveLaw = ConstitutiveLaw.GetAll()[constitutiveLaw]

        if Type == "Nodal":
            return listStressTensor([self.GetNodeResult(e, U) if e!=0 else np.zeros(self.__Mesh.GetNumberOfNodes()) for e in constitutiveLaw.GetStressOperator()])
        
        elif Type == "Element":
            return listStressTensor([self.GetElementResult(e, U) if e!=0 else np.zeros(self.__Mesh.GetNumberOfElements()) for e in constitutiveLaw.GetStressOperator()])
        
        elif Type == "GaussPoint":
            NumberOfGaussPointValues = self.__Mesh.GetNumberOfElements() * self.__nb_pg #Assembly.__saveOperator[(self.__Mesh.GetID(), self.__elmType, self.__nb_pg)][0].shape[0]
            return listStressTensor([self.GetGaussPointResult(e, U) if e!=0 else np.zeros(NumberOfGaussPointValues) for e in constitutiveLaw.GetStressOperator()])
        
        else:
            assert 0, "Wrong argument for Type: use 'Nodal', 'Element', or 'GaussPoint'"
        
        
    def GetStrainTensor(self, U, Type="Nodal", nlgeom = True):
        """
        Not a static method.
        Return the Green Lagrange Strain Tensor of an assembly using the Voigt notation as a python list. 
        The total displacement field has to be given.
        see GetNodeResults and GetElementResults

        Options : 
        - Type :"Nodal", "Element" or "GaussPoint" integration (default : "Nodal")
        - nlgeom = True or False if the strain tensor account for geometrical non-linearities
        if nlgeom = False, the Strain Tensor is assumed linear (default : True)

        example : 
        S = SpecificAssembly.GetStrainTensor(Problem.Problem.GetDoFSolution('all'))
        """        

        GradValues = self.GetGradTensor(U, Type)
        
        if nlgeom == False:
            Strain  = [GradValues[i][i] for i in range(3)] 
            Strain += [GradValues[1][2] + GradValues[2][1], GradValues[0][2] + GradValues[2][0], GradValues[0][1] + GradValues[1][0]]
        else:            
            Strain  = [GradValues[i][i] + 0.5*sum([GradValues[k][i]**2 for k in range(3)]) for i in range(3)] 
            Strain += [GradValues[1][2] + GradValues[2][1] + sum([GradValues[k][1]*GradValues[k][2] for k in range(3)])]
            Strain += [GradValues[0][2] + GradValues[2][0] + sum([GradValues[k][0]*GradValues[k][2] for k in range(3)])]
            Strain += [GradValues[0][1] + GradValues[1][0] + sum([GradValues[k][0]*GradValues[k][1] for k in range(3)])] 
        
        return listStrainTensor(Strain)
    
    def GetGradTensor(self, U, Type = "Nodal"):
        """
        Not a static method.
        Return the Gradient Tensor of a vector (generally displacement given by Problem.GetDofSolution('all')
        as a list of list of numpy array
        The total displacement field U has to be given (under a flatten numpy array)
        see GetNodeResults and GetElementResults

        Options : 
        - Type :"Nodal", "Element" or "GaussPoint" integration (default : "Nodal")
        """        
        GradOperator = GetGradOperator()        

        if Type == "Nodal":
            return [ [self.GetNodeResult(op, U) if op != 0 else np.zeros(self.__Mesh.GetNumberOfNodes()) for op in line_op] for line_op in GradOperator]
            
        elif Type == "Element":
            return [ [self.GetElementResult(op, U) if op!=0 else np.zeros(self.__Mesh.GetNumberOfElements()) for op in line_op] for line_op in GradOperator]        
        
        elif Type == "GaussPoint":
            NumberOfGaussPointValues = self.__nb_pg * self.__Mesh.GetNumberOfElements() #Assembly.__saveMatGaussianQuadrature[(self.__Mesh.GetID(), self.__nb_pg)].shape[0]
            return [ [self.GetGaussPointResult(op, U) if op!=0 else np.zeros(NumberOfGaussPointValues) for op in line_op] for line_op in GradOperator]        
        else:
            assert 0, "Wrong argument for Type: use 'Nodal', 'Element', or 'GaussPoint'"

    def GetExternalForces(self, U, Nvar=None):
        """
        Not a static method.
        Return the nodal Forces and moments in global coordinates related to a specific assembly considering the DOF solution given in U
        The resulting forces are the sum of :
        - External forces (associated to Neumann boundary conditions)
        - Nodal reaction (associated to Dirichelet boundary conditions)
        - Inertia forces 
        
        Return an array whose columns are Fx, Fy, Fz, Mx, My and Mz.         
                    
        example : 
        S = SpecificAssembly.GetNodalForces(Problem.Problem.GetDoFSolution('all'))

        an optionnal parameter is allowed to have extenal forces for other types of simulation with no beams !
        """
        if Nvar is None: Nvar = Variable.GetNumberOfVariable()
        return np.reshape(self.GetMatrix() * U - self.GetVector(), (Nvar,-1)).T                        
#        return np.reshape(self.GetMatrix() * U, (Nvar,-1)).T                        

        

#    def GetInternalForces(self, U, CoordinateSystem = 'global'): 
#        """
#        Not a static method.
#        Only available for 2 nodes beam element
#        Return the element internal Forces and moments related to a specific assembly considering the DOF solution given in U.
#        Return array whose columns are Fx, Fy, Fz, Mx, My and Mz. 
#        
#        Parameter: if CoordinateSystem == 'local' the result is given in the local coordinate system
#                   if CoordinateSystem == 'global' the result is given in the global coordinate system (default)
#        """
#        
##        operator = self.__weakForm.GetDifferentialOperator(self.__Mesh)
#        operator = self.__weakForm.GetGeneralizedStress()
#        res = [self.GetElementResult(operator[i], U) for i in range(5)]
#        return res
#        

                 
        
#        res = np.reshape(res,(6,-1)).T
#        Nel = mesh.GetNumberOfElements()
#        res = (res[Nel:,:]-res[0:Nel:,:])/2
#        res = res[:, [Variable.GetRank('DispX'), Variable.GetRank('DispY'), Variable.GetRank('DispZ'), \
#                              Variable.GetRank('ThetaX'), Variable.GetRank('ThetaY'), Variable.GetRank('ThetaZ')]]         
#        
#        if CoordinateSystem == 'local': return res
#        elif CoordinateSystem == 'global': 
#            #require a transformation between local and global coordinates on element
#            #classical MatrixChangeOfBasis transform only toward nodal values
#            elmRef = eval(self.__Mesh.GetElementShape())(1)#one pg  with the geometrical element
#            vec = [0,1,2] ; dim = 3
#       
#            #Data to build MatrixChangeOfBasisElement with coo sparse format
#            crd = mesh.GetNodeCoordinates() ; elm = mesh.GetElementTable()
#            rowMCB = np.empty((Nel, 1, dim,dim))
#            colMCB = np.empty((Nel, 1, dim,dim))            
#            rowMCB[:] = np.arange(Nel).reshape(-1,1,1,1) + np.array(vec).reshape(1,1,-1,1)*Nel # [[id_el + var*Nel] for var in vec]    
#            colMCB[:] = np.arange(Nel).reshape(-1,1,1,1) + np.array(vec).reshape(1,1,1,-1)*Nel # [id_el+Nel*var for var in vec]
#            dataMCB = elmRef.GetLocalFrame(crd[elm], elmRef.xi_pg, mesh.GetLocalFrame()) #array of shape (Nel, nb_pg=1, nb of vectors in basis = dim, dim)                        
#
#            MatrixChangeOfBasisElement = sparse.coo_matrix((sp.reshape(dataMCB,-1),(sp.reshape(rowMCB,-1),sp.reshape(colMCB,-1))), shape=(dim*Nel, dim*Nel)).tocsr()
#            
#            F = np.reshape( MatrixChangeOfBasisElement.T * np.reshape(res[:,0:3].T, -1)  ,  (3,-1) ).T
#            C = np.reshape( MatrixChangeOfBasisElement.T * np.reshape(res[:,3:6].T, -1)  ,  (3,-1) ).T
#            return np.hstack((F,C))            





    def GetInternalForces(self, U, CoordinateSystem = 'global'): 
        """
        Not a static method.
        Only available for 2 nodes beam element
        Return the element internal Forces and moments related to a specific assembly considering the DOF solution given in U.
        Return array whose columns are Fx, Fy, Fz, Mx, My and Mz. 
        
        Parameter: if CoordinateSystem == 'local' the result is given in the local coordinate system
                   if CoordinateSystem == 'global' the result is given in the global coordinate system (default)
        """
        
        operator = self.__weakForm.GetDifferentialOperator(self.__Mesh)
        mesh = self.__Mesh
        nvar = Variable.GetNumberOfVariable()
        MatrixChangeOfBasis = Assembly.__GetChangeOfBasisMatrix(mesh, self.__elmType)

        MatGaussianQuadrature = Assembly.__GetGaussianQuadratureMatrix(mesh, self.__elmType)

        res = 0        
        for ii in range(len(operator.op)):
            var = [operator.op[ii].u] ; coef = [1]
            var_vir = [operator.op_vir[ii].u] ; coef_vir = [1]

            if not(Variable.GetDerivative(var[0]) is None):     
                var.append(Variable.GetDerivative(var[0])[0])
                coef.append(Variable.GetDerivative(var[0])[1])
            if not(Variable.GetDerivative(var_vir[0]) is None): 
                var_vir.append(Variable.GetDerivative(var_vir[0])[0]) 
                coef_vir.append(Variable.GetDerivative(var_vir[0])[1])            

            Mat    =  RowBlocMatrix(Assembly.__GetElementaryOp(mesh, operator.op[ii], self.__elmType), nvar, var, coef)        
            Matvir =  RowBlocMatrix(Assembly.__GetElementaryOp(mesh, operator.op_vir[ii], self.__elmType), nvar, var_vir, coef_vir).T 

            if isinstance(operator.coef[ii], Number): #and self.op_vir[ii] != 1: 
                res = res + operator.coef[ii]*Matvir * MatGaussianQuadrature * Mat * MatrixChangeOfBasis * U                         
        
        res = np.reshape(res,(6,-1)).T
        Nel = mesh.GetNumberOfElements()
        res = (res[Nel:,:]-res[0:Nel:,:])/2
        res = res[:, [Variable.GetRank('DispX'), Variable.GetRank('DispY'), Variable.GetRank('DispZ'), \
                              Variable.GetRank('ThetaX'), Variable.GetRank('ThetaY'), Variable.GetRank('ThetaZ')]]         
        
        if CoordinateSystem == 'local': return res
        elif CoordinateSystem == 'global': 
            #require a transformation between local and global coordinates on element
            #classical MatrixChangeOfBasis transform only toward nodal values
            elmRef = eval(self.__Mesh.GetElementShape())(1)#one pg  with the geometrical element
            vec = [0,1,2] ; dim = 3
       
            #Data to build MatrixChangeOfBasisElement with coo sparse format
            crd = mesh.GetNodeCoordinates() ; elm = mesh.GetElementTable()
            rowMCB = np.empty((Nel, 1, dim,dim))
            colMCB = np.empty((Nel, 1, dim,dim))            
            rowMCB[:] = np.arange(Nel).reshape(-1,1,1,1) + np.array(vec).reshape(1,1,-1,1)*Nel # [[id_el + var*Nel] for var in vec]    
            colMCB[:] = np.arange(Nel).reshape(-1,1,1,1) + np.array(vec).reshape(1,1,1,-1)*Nel # [id_el+Nel*var for var in vec]
            dataMCB = elmRef.GetLocalFrame(crd[elm], elmRef.xi_pg, mesh.GetLocalFrame()) #array of shape (Nel, nb_pg=1, nb of vectors in basis = dim, dim)                        

            MatrixChangeOfBasisElement = sparse.coo_matrix((sp.reshape(dataMCB,-1),(sp.reshape(rowMCB,-1),sp.reshape(colMCB,-1))), shape=(dim*Nel, dim*Nel)).tocsr()
            
            F = np.reshape( MatrixChangeOfBasisElement.T * np.reshape(res[:,0:3].T, -1)  ,  (3,-1) ).T
            C = np.reshape( MatrixChangeOfBasisElement.T * np.reshape(res[:,3:6].T, -1)  ,  (3,-1) ).T
            return np.hstack((F,C))            



#def GetDefaultNumberOfPG(elmType, mesh):
#    nb_pg = GetDefaultNbPG(elmType)   
#    if nb_pg is None: 
#        nb_pg = GetDefaultNbPG(mesh.GetElementShape())
#    if nb_pg is None:
#        raise NameError('Element unknown: no default number of integration points')
#    return nb_pg