#simcoon compatible

from fedoo.libAssembly.AssemblyBase import AssemblyBase
from fedoo.libUtil.ModelingSpace import ModelingSpace
from fedoo.libUtil.PostTreatement import listStressTensor, listStrainTensor
from fedoo.libMesh.Mesh import Mesh
from fedoo.libElement import *
from fedoo.libWeakForm.WeakForm import WeakForm
from fedoo.libConstitutiveLaw.ConstitutiveLaw import ConstitutiveLaw
from fedoo.libUtil.GradOperator import GetGradOperator
from fedoo.libUtil.SparseMatrix import _BlocSparse as BlocSparse
from fedoo.libUtil.SparseMatrix import _BlocSparseOld as BlocSparseOld #required for 'old' computeMatrixMehtod
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
    __associatedVariables = {} #dict containing all associated variables (rotational dof for C1 elements) for elementType
           
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
        if elementType == "": elementType = mesh.GetElementShape()
        self.__elmType= elementType #.lower()

        #determine the type of coordinate system used for vector of variables (displacement for instance). This type may be specified in element (under dict form only)        
        #TypeOfCoordinateSystem may be 'local' or 'global'. If 'local' variables are used, a change of variable is required
        #If TypeOfCoordinateSystemis not specified in the element, 'global' value (no change of basis) is considered by default
        if isinstance(eval(elementType), dict):
            self.__TypeOfCoordinateSystem = eval(elementType).get('__TypeOfCoordinateSystem', 'global')                
        else: self.__TypeOfCoordinateSystem = 'global'

        self.__nb_pg = kargs.pop('nb_pg', None)
        if self.__nb_pg is None: self.__nb_pg = GetDefaultNbPG(elementType, mesh)

        self.__saveBlocStructure = None #use to save data about the sparse structure and avoid time consuming recomputation
        #print('Finite element operator for Assembly "' + ID + '" built in ' + str(time.time()-t0) + ' seconds')        
        self.computeMatrixMethod = 'new' #computeMatrixMethod = 'old' and 'very_old' only used for debug purpose        

    def ComputeGlobalMatrix(self, compute = 'all'):
        """
        Compute the global matrix and global vector related to the assembly
        if compute = 'all', compute the global matrix and vector
        if compute = 'matrix', compute only the matrix
        if compute = 'vector', compute only the vector
        if compute = 'none', compute nothing
        """                        
        if compute == 'none': return
        
        computeMatrixMethod = self.computeMatrixMethod
        
        nb_pg = self.__nb_pg
        mesh = self.__Mesh        
        
        if self.__MeshChange == True:             
            if mesh.GetID() in Assembly.__saveMatrixChangeOfBasis: del Assembly.__saveMatrixChangeOfBasis[mesh.GetID()]            
            Assembly.PreComputeElementaryOperators(mesh, self.__elmType, nb_pg=nb_pg)
                 
        nvar = ModelingSpace.GetNumberOfVariable()
        wf = self.__weakForm.GetDifferentialOperator(mesh)      

        MatGaussianQuadrature = Assembly.__GetGaussianQuadratureMatrix(mesh, self.__elmType, nb_pg=nb_pg)
        MatrixChangeOfBasis = Assembly.__GetChangeOfBasisMatrix(mesh, self.__TypeOfCoordinateSystem)        
        associatedVariables = Assembly.__GetAssociatedVariables(self.__elmType) #for element requiring many variable such as beam with disp and rot dof        
                             
        if computeMatrixMethod == 'new': 
            
            intRef = wf.sort()
            #sl contains list of slice object that contains the dimension for each variable
            #size of VV and sl must be redefined for case with change of basis
            VV = 0
            nbNodes = self.__Mesh.GetNumberOfNodes()            
            sl = [slice(i*nbNodes, (i+1)*nbNodes) for i in range(nvar)] 
            
            if nb_pg == 0: #if finite difference elements, don't use BlocSparse                              
                blocks = [[None for i in range(nvar)] for j in range(nvar)]
                self.__saveBlocStructure = 0 #don't save block structure for finite difference mesh
                    
                Matvir = Assembly.__GetElementaryOp(mesh, wf.op_vir[0], self.__elmType, nb_pg=0)[0].T #should be identity matrix restricted to nodes used in the finite difference mesh
                
                for ii in range(len(wf.op)):
                    if compute == 'matrix' and wf.op[ii] is 1: continue
                    if compute == 'vector' and wf.op[ii] is not 1: continue
                
                    if ii > 0 and intRef[ii] == intRef[ii-1]: #if same operator as previous with different coef, add the two coef
                        coef_PG += wf.coef[ii]
                    else: coef_PG = wf.coef[ii]   #coef_PG = nodal values (finite diffirences)
                    
                    if ii < len(wf.op)-1 and intRef[ii] == intRef[ii+1]: #if operator similar to the next, continue 
                        continue
                                                    
                    var_vir = wf.op_vir[ii].u
                    assert wf.op_vir[ii].ordre == 0, "This weak form is not compatible with finite difference mesh"
                                        
                    if wf.op[ii] == 1: #only virtual operator -> compute a vector which is the nodal values
                        if VV is 0: VV = np.zeros((self.__Mesh.GetNumberOfNodes() * nvar))
                        VV[sl[var_vir[i]]] = VV[sl[var_vir[i]]] - (coef_PG) 
                            
                    else: #virtual and real operators -> compute a matrix
                        var = wf.op[ii].u   
                        if isinstance(coef_PG, Number): coef_PG = coef_PG * np.ones_like(MatGaussianQuadrature.data)
                        CoefMatrix = sparse.csr_matrix( (coef_PG, MatGaussianQuadrature.indices, MatGaussianQuadrature.indptr), shape = MatGaussianQuadrature.shape)   
                        Mat    =  Assembly.__GetElementaryOp(mesh, wf.op[ii], self.__elmType, nb_pg=nb_pg)[0]
                        
                        if blocks[var_vir][var] is None: 
                            blocks[var_vir][var] = Matvir @ CoefMatrix @ Mat
                        else:                    
                            blocks[var_vir][var].data += (Matvir @ CoefMatrix @ Mat).data
                            
                blocks = [[b if b is not None else sparse.csr_matrix((nbNodes,nbNodes)) \
                          for b in blocks_row] for blocks_row in blocks ]        
                MM = sparse.bmat(blocks, format ='csr')
                
            else:
                MM = BlocSparse(nvar, nvar, self.__nb_pg, self.__saveBlocStructure)
                
                for ii in range(len(wf.op)):
                    if compute == 'matrix' and wf.op[ii] is 1: continue
                    if compute == 'vector' and wf.op[ii] is not 1: continue
                
                    if isinstance(wf.coef[ii], Number) or len(wf.coef[ii])==1: 
                        #if nb_pg == 0, coef_PG = nodal values (finite diffirences)
                        coef_PG = wf.coef[ii] 
                    else:
                        coef_PG = Assembly.__ConvertToGaussPoints(mesh, wf.coef[ii][:], self.__elmType, nb_pg=nb_pg)                                                 
                    
                    if ii > 0 and intRef[ii] == intRef[ii-1]: #if same operator as previous with different coef, add the two coef
                        coef_PG_sum += coef_PG
                    else: coef_PG_sum = coef_PG   
                    
                    if ii < len(wf.op)-1 and intRef[ii] == intRef[ii+1]: #if operator similar to the next, continue 
                        continue
                                    
                    coef_PG = coef_PG_sum * MatGaussianQuadrature.data #MatGaussianQuadrature.data is the diagonal of MatGaussianQuadrature
                
                    coef_vir = [1] ; var_vir = [wf.op_vir[ii].u] #list in case there is an angular variable
                                   
                    if var_vir[0] in associatedVariables:
                        var_vir.extend(associatedVariables[var_vir[0]][0])
                        coef_vir.extend(associatedVariables[var_vir[0]][1])                   
                              
    #                Matvir = (RowBlocMatrix(Assembly.__GetElementaryOp(mesh, wf.op_vir[ii], self.__elmType, nb_pg=nb_pg), nvar, var_vir, coef_vir) * MatrixChangeOfBasis).T
                    #check how it appens with change of variable and rotation dof
                    Matvir = Assembly.__GetElementaryOp(mesh, wf.op_vir[ii], self.__elmType, nb_pg=nb_pg)
                    
                    if wf.op[ii] == 1: #only virtual operator -> compute a vector 
                        if VV is 0: VV = np.zeros((self.__Mesh.GetNumberOfNodes() * nvar))
                        for i in range(len(Matvir)):
                            VV[sl[var_vir[i]]] = VV[sl[var_vir[i]]] - coef_vir[i] * Matvir[i].T * (coef_PG) #this line may be optimized
                            
                    else: #virtual and real operators -> compute a matrix
                        coef = [1] ; var = [wf.op[ii].u] #list in case there is an angular variable                
                        if var[0] in associatedVariables:
                            var.extend(associatedVariables[var[0]][0])
                            coef.extend(associatedVariables[var[0]][1])                                             
    
    #                    Mat    =  RowBlocMatrix(Assembly.__GetElementaryOp(mesh, wf.op[ii], self.__elmType, nb_pg=nb_pg), nvar, var, coef)         * MatrixChangeOfBasis             
                        Mat    =  Assembly.__GetElementaryOp(mesh, wf.op[ii], self.__elmType, nb_pg=nb_pg)
        
                        #Possibility to increase performance for multivariable case 
                        #the structure should be the same for derivative dof, so the blocs could be computed altogether
                        for i in range(len(Mat)):
                            for j in range(len(Matvir)):
                                MM.addToBlocATB(Matvir[j], Mat[i], (coef[i]*coef_vir[j]) * coef_PG, var_vir[j], var[i])
                
            if compute != 'vector':
                if MatrixChangeOfBasis is 1: 
                    self.SetMatrix(MM.tocsr()) #format csr         
                else: 
                    self.SetMatrix(MatrixChangeOfBasis.T * MM.tocsr() * MatrixChangeOfBasis) #format csr         
            if compute != 'matrix': 
                if VV is 0: self.SetVector(0)
                elif MatrixChangeOfBasis is 1: self.SetVector(VV) #numpy array
                else: self.SetVector(MatrixChangeOfBasis.T * VV)                     

            if self.__saveBlocStructure is None: self.__saveBlocStructure = MM.GetBlocStructure()        

        elif computeMatrixMethod == 'old': #keep a lot in memory, not very efficient in a memory point of view. May be slightly more rapid in some cases                            
        
            intRef = wf.sort() #intRef = list of integer for compareason (same int = same operator with different coef)            
            
            if (mesh.GetID(), self.__elmType, nb_pg) not in Assembly.__saveOperator:
                Assembly.__saveOperator[(mesh.GetID(), self.__elmType, nb_pg)] = {}
            saveOperator = Assembly.__saveOperator[(mesh.GetID(), self.__elmType, nb_pg)]
            
            #list_elementType contains the id of the element associated with every variable
            #list_elementType could be stored to avoid reevaluation 
            if isinstance(eval(self.__elmType), dict):
                elementDict = eval(self.__elmType)
                list_elementType = [elementDict.get(ModelingSpace.GetVariableName(i))[0] for i in range(nvar)]
                list_elementType = [elementDict.get('__default') if elmtype is None else elmtype for elmtype in list_elementType]
            else: list_elementType = [self.__elmType for i in range(nvar)]
            
            if 'blocShape' not in saveOperator:
                saveOperator['blocShape'] = saveOperator['colBlocSparse'] = saveOperator['rowBlocSparse'] = None
            
            #MM not used if only compute vector
            MM = BlocSparseOld(nvar, nvar)
            MM.col = saveOperator['colBlocSparse'] #col indices for bloc to build coo matrix with BlocSparse
            MM.row = saveOperator['rowBlocSparse'] #row indices for bloc to build coo matrix with BlocSparse
            MM.blocShape = saveOperator['blocShape'] #shape of one bloc in BlocSparse
            
            #sl contains list of slice object that contains the dimension for each variable
            #size of VV and sl must be redefined for case with change of basis
            VV = 0
            nbNodes = self.__Mesh.GetNumberOfNodes()            
            sl = [slice(i*nbNodes, (i+1)*nbNodes) for i in range(nvar)] 
            
            for ii in range(len(wf.op)):                   
                if compute == 'matrix' and wf.op[ii] is 1: continue
                if compute == 'vector' and wf.op[ii] is not 1: continue
            
                if isinstance(wf.coef[ii], Number) or len(wf.coef[ii])==1: 
                    coef_PG = wf.coef[ii] #MatGaussianQuadrature.data is the diagonal of MatGaussianQuadrature
                else:
                    coef_PG = Assembly.__ConvertToGaussPoints(mesh, wf.coef[ii][:], self.__elmType, nb_pg=nb_pg)                                                 
                
                if ii > 0 and intRef[ii] == intRef[ii-1]: #if same operator as previous with different coef, add the two coef
                    coef_PG_sum += coef_PG
                else: coef_PG_sum = coef_PG   
                
                if ii < len(wf.op)-1 and intRef[ii] == intRef[ii+1]: #if operator similar to the next, continue 
                    continue
                
                coef_PG = coef_PG_sum * MatGaussianQuadrature.data 
                            
                coef_vir = [1] ; var_vir = [wf.op_vir[ii].u] #list in case there is an angular variable
                               
                if var_vir[0] in associatedVariables:
                    var_vir.extend(associatedVariables[var_vir[0]][0])
                    coef_vir.extend(associatedVariables[var_vir[0]][1])         
                                                 
                if wf.op[ii] == 1: #only virtual operator -> compute a vector 
                                            
                    Matvir = Assembly.__GetElementaryOp(mesh, wf.op_vir[ii], self.__elmType, nb_pg=nb_pg)         
                    if VV is 0: VV = np.zeros((self.__Mesh.GetNumberOfNodes() * nvar))
                    for i in range(len(Matvir)):
                        VV[sl[var_vir[i]]] = VV[sl[var_vir[i]]] - coef_vir[i] * Matvir[i].T * (coef_PG) #this line may be optimized
                        
                else: #virtual and real operators -> compute a matrix
                    coef = [1] ; var = [wf.op[ii].u] #list in case there is an angular variable                
                    if var[0] in associatedVariables:
                        var.extend(associatedVariables[var[0]][0])
                        coef.extend(associatedVariables[var[0]][1])                                                                                     
                    
                    tupleID = (list_elementType[wf.op_vir[ii].u], wf.op_vir[ii].x, wf.op_vir[ii].ordre, list_elementType[wf.op[ii].u], wf.op[ii].x, wf.op[ii].ordre) #tuple to identify operator
                    if tupleID in saveOperator:
                        MatvirT_Mat = saveOperator[tupleID] #MatvirT_Mat is an array that contains usefull data to build the matrix MatvirT*Matcoef*Mat where Matcoef is a diag coefficient matrix. MatvirT_Mat is build with BlocSparse class
                    else: 
                        MatvirT_Mat = None
                        saveOperator[tupleID] = [[None for i in range(len(var))] for j in range(len(var_vir))]
                        Matvir = Assembly.__GetElementaryOp(mesh, wf.op_vir[ii], self.__elmType, nb_pg=nb_pg)         
                        Mat = Assembly.__GetElementaryOp(mesh, wf.op[ii], self.__elmType, nb_pg=nb_pg)                                               

                    for i in range(len(var)):
                        for j in range(len(var_vir)):
                            if MatvirT_Mat is not None:           
                                MM.addToBloc(MatvirT_Mat[j][i], (coef[i]*coef_vir[j]) * coef_PG, var_vir[j], var[i])                                     
                            else:  
                                saveOperator[tupleID][j][i] = MM.addToBlocATB(Matvir[j], Mat[i], (coef[i]*coef_vir[j]) * coef_PG, var_vir[j], var[i])
                                if saveOperator['colBlocSparse'] is None: 
                                    saveOperator['colBlocSparse'] = MM.col
                                    saveOperator['rowBlocSparse'] = MM.row
                                    saveOperator['blocShape'] = MM.blocShape
                               
            if compute != 'vector': 
                if MatrixChangeOfBasis is 1: 
                    self.SetMatrix(MM.toCSR()) #format csr         
                else: 
                    self.SetMatrix(MatrixChangeOfBasis.T * MM.toCSR() * MatrixChangeOfBasis) #format csr         
            if compute != 'matrix': 
                if VV is 0: self.SetVector(0)
                elif MatrixChangeOfBasis is 1: self.SetVector(VV) #numpy array
                else: self.SetVector(MatrixChangeOfBasis.T * VV)         
        
        
        elif computeMatrixMethod == 'very_old':
            MM = 0
            VV = 0
            
            for ii in range(len(wf.op)):
                if compute == 'matrix' and wf.op[ii] is 1: continue
                if compute == 'vector' and wf.op[ii] is not 1: continue
            
                coef_vir = [1] ; var_vir = [wf.op_vir[ii].u] #list in case there is an angular variable      
                if var_vir[0] in associatedVariables:
                    var_vir.extend(associatedVariables[var_vir[0]][0])
                    coef_vir.extend(associatedVariables[var_vir[0]][1])     
                     
                Matvir = (RowBlocMatrix(Assembly.__GetElementaryOp(mesh, wf.op_vir[ii], self.__elmType, nb_pg=nb_pg), nvar, var_vir, coef_vir) * MatrixChangeOfBasis).T
    
                if wf.op[ii] == 1: #only virtual operator -> compute a vector 
                    if isinstance(wf.coef[ii], Number): 
                        VV = VV - wf.coef[ii]*Matvir * MatGaussianQuadrature.data
                    else:
                        coef_PG = Assembly.__ConvertToGaussPoints(mesh, wf.coef[ii][:], self.__elmType, nb_pg=nb_pg)*MatGaussianQuadrature.data                             
                        VV = VV - Matvir * (coef_PG)
                        
                else: #virtual and real operators -> compute a matrix
                    coef = [1] ; var = [wf.op[ii].u] #list in case there is an angular variable                  
                    if var[0] in associatedVariables:
                        var.extend(associatedVariables[var[0]][0])
                        coef.extend(associatedVariables[var[0]][1])     
                                    
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
    
    def GetWeakForm(self):
        return self.__weakForm
           
    def GetNumberOfGaussPoints(self):
        return self.__nb_pg
    
    def GetMatrixChangeOfBasis(self):
        return Assembly.__GetChangeOfBasisMatrix(self.__Mesh, self.__TypeOfCoordinateSystem)
    

    def Initialize(self, pb, initialTime=0.):
        """
        Initialize the associated weak form and assemble the global matrix with the elastic matrix
        Parameters: 
            - initialTime: the initial time        
        """
        self.__weakForm.Initialize(self, pb, initialTime)
        self.ComputeGlobalMatrix()

    def Update(self, pb, dtime=None, compute = 'all'):
        """
        Update the associated weak form and assemble the global matrix
        Parameters: 
            - pb: a Problem object containing the Dof values
            - time: the current time        
        """
        self.__weakForm.Update(self, pb, dtime)
        self.ComputeGlobalMatrix(compute)

    def ResetTimeIncrement(self):
        """
        Reset the current time increment (internal variable in the constitutive equation)
        Doesn't assemble the new global matrix. Use the Update method for that purpose.
        """
        self.__weakForm.ResetTimeIncrement()
        self.ComputeGlobalMatrix(compute='all')

    def NewTimeIncrement(self):
        """
        Apply the modification to the constitutive equation required at each change of time increment. 
        Generally used to increase non reversible internal variable
        Doesn't assemble the new global matrix. Use the Update method for that purpose.
        """
        self.__weakForm.NewTimeIncrement() #should update GetH() method to return elastic rigidity matrix for prediction        
        self.ComputeGlobalMatrix(compute='matrix')
 
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
        #-------------------------------------------------------------------
        #Initialisation   
        #-------------------------------------------------------------------
        if nb_pg is None: NumberOfGaussPoint = GetDefaultNbPG(elementType, mesh)
        else: NumberOfGaussPoint = nb_pg
                  
        Nnd = mesh.GetNumberOfNodes()
        Nel = mesh.GetNumberOfElements()
        elm = mesh.GetElementTable()
        nNd_elm = np.shape(elm)[1]
        crd = mesh.GetNodeCoordinates()
        dim = ModelingSpace.GetDoF()
        
        if isinstance(eval(elementType), dict):
            TypeOfCoordinateSystem = eval(elementType).get('__TypeOfCoordinateSystem', 'global')                
        else: TypeOfCoordinateSystem = 'global'

        
        #-------------------------------------------------------------------
        #Case of finite difference mesh    
        #-------------------------------------------------------------------        
        if NumberOfGaussPoint == 0: # in this case, it is a finite difference mesh
            # we compute the operators directly from the element library
            elmRef = eval(elementType)(NumberOfGaussPoint)
            OP = elmRef.computeOperator(crd,elm)
            Assembly.__saveMatGaussianQuadrature[(mesh.GetID(),NumberOfGaussPoint)] = sparse.identity(OP[0][0].shape[0], 'd', format= 'csr') #No gaussian quadrature in this case : nodal identity matrix
            Assembly.__savePGtoNodeMatrix[(mesh.GetID(), NumberOfGaussPoint)] = 1  #no need to translate between pg and nodes because no pg 
            Assembly.__saveNodeToPGMatrix[(mesh.GetID(), NumberOfGaussPoint)] = 1                                    
            Assembly.__saveMatrixChangeOfBasis[mesh.GetID()] = 1 # No change of basis:  MatrixChangeOfBasis = 1 #this line could be deleted because the coordinate should in principle defined as 'global' 
            Assembly.__saveOperator[(mesh.GetID(),elementType,NumberOfGaussPoint)] = OP #elmRef.computeOperator(crd,elm)
            return                                

        #-------------------------------------------------------------------
        #Initialise the geometrical interpolation
        #-------------------------------------------------------------------   
        elmRefGeom = eval(mesh.GetElementShape())(NumberOfGaussPoint, mesh=mesh) #initialise element
        nNd_elm_geom = len(elmRefGeom.xi_nd) #number of dof used in the geometrical interpolation
        elm_geom = elm[:,:nNd_elm_geom] 

        localFrame = mesh.GetLocalFrame()
        nb_elm_nd = np.bincount(elm_geom.reshape(-1)) #len(nb_elm_nd) = Nnd #number of element connected to each node        
        vec_xi = elmRefGeom.xi_pg #coordinate of points of gauss in element coordinate (xi)
        
        elmRefGeom.ComputeJacobianMatrix(crd[elm_geom], vec_xi, localFrame) #compute elmRefGeom.JacobianMatrix, elmRefGeom.detJ and elmRefGeom.inverseJacobian

        #-------------------------------------------------------------------
        # Compute the diag matrix used for the gaussian quadrature
        #-------------------------------------------------------------------  
        gaussianQuadrature = (elmRefGeom.detJ * elmRefGeom.w_pg).T.reshape(-1) 
        Assembly.__saveMatGaussianQuadrature[(mesh.GetID(),NumberOfGaussPoint)] = sparse.diags(gaussianQuadrature, 0, format='csr') #matrix to get the gaussian quadrature (integration over each element)        

        #-------------------------------------------------------------------
        # Compute the array containing row and col indices used to assemble the sparse matrices
        #-------------------------------------------------------------------          
        range_nbPG = np.arange(NumberOfGaussPoint)                 
        if Assembly.__GetChangeOfBasisMatrix(mesh, TypeOfCoordinateSystem) is 1: ChangeOfBasis = False
        else: 
            ChangeOfBasis = True
            range_nNd_elm = np.arange(nNd_elm)
        
        row = np.empty((Nel, NumberOfGaussPoint, nNd_elm)) ; col = np.empty((Nel, NumberOfGaussPoint, nNd_elm))                
        row[:] = np.arange(Nel).reshape((-1,1,1)) + range_nbPG.reshape(1,-1,1)*Nel 
        col[:] = elm.reshape((Nel,1,nNd_elm))
        #row_geom/col_geom: row and col indices using only the dof used in the geometrical interpolation (col = col_geom if geometrical and variable interpolation are the same)
        row_geom = np.reshape(row[...,:nNd_elm_geom], -1) ; col_geom = np.reshape(col[...,:nNd_elm_geom], -1)
        
        if ChangeOfBasis: 
            col = np.empty((Nel, NumberOfGaussPoint, nNd_elm))
            col[:] = np.arange(Nel).reshape((-1,1,1)) + range_nNd_elm.reshape((1,1,-1))*Nel 
            Ncol = Nel * nNd_elm
        else: 
            Ncol = Nnd                      
        row = np.reshape(row,-1) ; col = np.reshape(col,-1)  

        #-------------------------------------------------------------------
        # Assemble the matrix that compute the node values from pg based on the geometrical shape functions (no angular dof for ex)    
        #-------------------------------------------------------------------                                
        PGtoNode = np.linalg.pinv(elmRefGeom.ShapeFunctionPG) #pseudo-inverse of NodeToPG
        dataPGtoNode = PGtoNode.T.reshape((1,NumberOfGaussPoint,nNd_elm_geom))/nb_elm_nd[elm_geom].reshape((Nel,1,nNd_elm_geom)) #shape = (Nel, NumberOfGaussPoint, nNd_elm)   
        Assembly.__savePGtoNodeMatrix[(mesh.GetID(), NumberOfGaussPoint)] = sparse.coo_matrix((dataPGtoNode.reshape(-1),(col_geom,row_geom)), shape=(Nnd,Nel*NumberOfGaussPoint) ).tocsr() #matrix to compute the node values from pg using the geometrical shape functions 

        #-------------------------------------------------------------------
        # Assemble the matrix that compute the pg values from nodes using the geometrical shape functions (no angular dof for ex)    
        #-------------------------------------------------------------------             
        dataNodeToPG = np.empty((Nel, NumberOfGaussPoint, nNd_elm_geom))
        dataNodeToPG[:] = elmRefGeom.ShapeFunctionPG.reshape((1,NumberOfGaussPoint,nNd_elm_geom)) 
        Assembly.__saveNodeToPGMatrix[(mesh.GetID(), NumberOfGaussPoint)] = sparse.coo_matrix((sp.reshape(dataNodeToPG,-1),(row_geom,col_geom)), shape=(Nel*NumberOfGaussPoint, Nnd) ).tocsr() #matrix to compute the pg values from nodes using the geometrical shape functions (no angular dof)

        #-------------------------------------------------------------------
        # Build the list of elementType to assemble (some beam element required several elementType in function of the variable)
        #-------------------------------------------------------------------        
        objElement = eval(elementType)
        if isinstance(objElement, dict):
            listElementType = set([objElement[key][0] for key in objElement.keys() if key[:2]!='__' or key == '__default'])               
        else: 
            listElementType =  [elementType]
        
        #-------------------------------------------------------------------
        # Assembly of the elementary operators for each elementType 
        #-------------------------------------------------------------------      
        for elementType in listElementType: 
            elmRef = eval(elementType)(NumberOfGaussPoint, mesh = mesh, elmGeom = elmRefGeom)
            nb_dir_deriv = 0
            if hasattr(elmRef,'ShapeFunctionDerivativePG'):
                derivativePG = elmRefGeom.inverseJacobian @ elmRef.ShapeFunctionDerivativePG #derivativePG = np.matmul(elmRefGeom.inverseJacobian , elmRef.ShapeFunctionDerivativePG)
                nb_dir_deriv = derivativePG.shape[-2] 
            nop = nb_dir_deriv+1 #nombre d'opérateur à discrétiser
    
            NbDoFperNode = np.shape(elmRef.ShapeFunctionPG)[-1]//nNd_elm
            
            data = [[np.empty((Nel, NumberOfGaussPoint, nNd_elm)) for j in range(NbDoFperNode)] for i in range(nop)] 
    
            for j in range(0,NbDoFperNode):
                data[0][j][:] = elmRef.ShapeFunctionPG[...,j*nNd_elm:(j+1)*nNd_elm].reshape((-1,NumberOfGaussPoint,nNd_elm)) #same as dataNodeToPG matrix if geometrical shape function are the same as interpolation functions
                for dir_deriv in range(nb_dir_deriv):
                    data[dir_deriv+1][j][:] = derivativePG[...,dir_deriv, j*nNd_elm:(j+1)*nNd_elm]
                        
            op_dd = [ [sparse.coo_matrix((data[i][j].reshape(-1),(row,col)), shape=(Nel*NumberOfGaussPoint , Ncol) ).tocsr() for j in range(NbDoFperNode) ] for i in range(nop)]        
                
            data = {0: op_dd[0]} #data is a dictionnary
            for i in range(nb_dir_deriv): 
                data[1, i] = op_dd[i+1]

            Assembly.__saveOperator[(mesh.GetID(),elementType,NumberOfGaussPoint)] = data   
    
    @staticmethod
    def __GetElementaryOp(mesh, deriv, elementType, nb_pg=None): 
        #Gives a list of sparse matrix that convert node values for one variable to the pg values of a simple derivative op (for instance d/dz)
        #The list contains several element if the elementType include several variable (dof variable in beam element). In other case, the list contains only one matrix
        #The variables are not considered. For a global use, the resulting matrix should be assembled in a block matrix with the nodes values for all variables
        if nb_pg is None: nb_pg = GetDefaultNbPG(elementType, mesh)

        if isinstance(eval(elementType), dict):
            elementDict = eval(elementType)
            elementType = elementDict.get(ModelingSpace.GetVariableName(deriv.u))
            if elementType is None: elementType = elementDict.get('__default')
            elementType = elementType[0]
            
        if not((mesh.GetID(),elementType,nb_pg) in Assembly.__saveOperator):
            Assembly.PreComputeElementaryOperators(mesh, elementType, nb_pg)
        
        data = Assembly.__saveOperator[(mesh.GetID(),elementType,nb_pg)]

        if deriv.ordre == 0 and 0 in data:
            return data[0]
        
        #extract the mesh coordinate that corespond to coordinate rank given in deriv.x     
        ListMeshCoordinateIDRank = [ModelingSpace.GetCoordinateRank(crdID) for crdID in mesh.GetCoordinateID()]
        if deriv.x in ListMeshCoordinateIDRank: xx= ListMeshCoordinateIDRank.index(deriv.x)
        else: return data[0] #if the coordinate doesnt exist, return operator without derivation
                         
        if (deriv.ordre, xx) in data:
            return data[deriv.ordre, xx]
        else: assert 0, "Operator unavailable"      
        
    # @staticmethod
    # def __GetElementaryOp2(mesh, deriv_vir, deriv, elementType, nb_pg=None): 
    #     #Gives a list of sparse matrix that convert node values for one variable to the pg values of a simple derivative op (for instance d/dz)
    #     #The list contains several element if the elementType include several variable (dof variable in beam element). In other case, the list contains only one matrix
    #     #The variables are not considered. For a global use, the resulting matrix should be assembled in a block matrix with the nodes values for all variables
    #     if nb_pg is None: nb_pg = GetDefaultNbPG(elementType, mesh)

    #     if isinstance(eval(elementType), dict):
    #         elementDict = eval(elementType)
    #         elementType = elementDict.get(ModelingSpace.GetVariableName(deriv.u))[0]
    #         if elementType is None: elementType = elementDict.get('__default')
            
    #         elementType_vir = elementDict.get(ModelingSpace.GetVariableName(deriv_vir.u))[0]
    #         if elementType_vir is None: elementType = elementDict.get('__default')                

    #     else: elementType_vir = elementType


    #     if not((mesh.GetID(),elementType,elementType_vir, nb_pg) in Assembly.__saveOperator):        
    #         if not((mesh.GetID(),elementType,nb_pg) in Assembly.__saveOperator):
    #             Assembly.PreComputeElementaryOperators(mesh, elementType, nb_pg)
                    
    #         if (elementType != elementType_vir) and ((mesh.GetID(),elementType,nb_pg) not in Assembly.__saveOperator):
    #             Assembly.PreComputeElementaryOperators(mesh, elementType_vir, nb_pg)
            
    #         data = Assembly.__saveOperator[(mesh.GetID(),elementType,nb_pg)]
    
    #         if deriv.ordre == 0 and 0 in data:
    #             return data[0]
            
    #         #extract the mesh coordinate that corespond to coordinate rank given in deriv.x     
    #         ListMeshCoordinateIDRank = [ModelingSpace.GetCoordinateRank(crdID) for crdID in mesh.GetCoordinateID()]
    #         if deriv.x in ListMeshCoordinateIDRank: xx= ListMeshCoordinateIDRank.index(deriv.x)
    #         else: return data[0] #if the coordinate doesnt exist, return operator without derivation
                             
    #         if (deriv.ordre, xx) in data:
    #             return data[deriv.ordre, xx]
    #         else: assert 0, "Operator unavailable"        

    @staticmethod
    def __GetGaussianQuadratureMatrix(mesh, elementType, nb_pg=None): #calcul la discrétision relative à un seul opérateur dérivé   
        if nb_pg is None: nb_pg = GetDefaultNbPG(elementType, mesh)        
        if not((mesh.GetID(),nb_pg) in Assembly.__saveMatGaussianQuadrature):
            Assembly.PreComputeElementaryOperators(mesh, elementType, nb_pg)
        return Assembly.__saveMatGaussianQuadrature[(mesh.GetID(),nb_pg)]

    @staticmethod    
    def __GetAssociatedVariables(elementType): #add the associated variables (rotational dof for C1 elements) of the current element        
        if elementType not in Assembly.__associatedVariables:
            objElement = eval(elementType)
            if isinstance(objElement, dict):            
                Assembly.__associatedVariables[elementType] = {ModelingSpace.GetVariableRank(key): 
                                       [[ModelingSpace.GetVariableRank(v) for v in val[1][1::2]],
                                        val[1][0::2]] for key,val in objElement.items() if len(val)>1 and key in ModelingSpace.ListVariable()} 
                    # val[1][0::2]] for key,val in objElement.items() if key in ModelingSpace.ListVariable() and len(val)>1}
            else: Assembly.__associatedVariables[elementType] = {}
        return Assembly.__associatedVariables[elementType] 
    
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
    def __GetChangeOfBasisMatrix(mesh, TypeOfCoordinateSystem): # change of basis matrix for beam or plate elements
        
        if TypeOfCoordinateSystem == 'global': return 1
        if mesh.GetID() not in Assembly.__saveMatrixChangeOfBasis:        
            ### change of basis treatment for beam or plate elements
            ### Compute the change of basis matrix for vector defined in ModelingSpace.ListVector()
            MatrixChangeOfBasis = 1
            computeMatrixChangeOfBasis = False

            Nnd = mesh.GetNumberOfNodes()
            Nel = mesh.GetNumberOfElements()
            elm = mesh.GetElementTable()
            nNd_elm = np.shape(elm)[1]            
            crd = mesh.GetNodeCoordinates()
            dim = ModelingSpace.GetDoF()
            localFrame = mesh.GetLocalFrame()
            elmRefGeom = eval(mesh.GetElementShape())(mesh=mesh)
    #        xi_nd = elmRefGeom.xi_nd
            xi_nd = GetNodePositionInElementCoordinates(mesh.GetElementShape(), nNd_elm) #function to define

            if 'X' in mesh.GetCoordinateID() and 'Y' in mesh.GetCoordinateID(): #if not in physical space, no change of variable                
                for nameVector in ModelingSpace.ListVector():
                    if computeMatrixChangeOfBasis == False:
                        range_nNd_elm = np.arange(nNd_elm) 
                        computeMatrixChangeOfBasis = True
                        Nvar = ModelingSpace.GetNumberOfVariable()
                        listGlobalVector = []  ; listScalarVariable = list(range(Nvar))
#                        MatrixChangeOfBasis = sparse.lil_matrix((Nvar*Nel*nNd_elm, Nvar*Nnd)) #lil is very slow because it change the sparcity of the structure
                    listGlobalVector.append(ModelingSpace.GetVector(nameVector)) #vector that need to be change in local coordinate            
                    listScalarVariable = [i for i in listScalarVariable if not(i in listGlobalVector[-1])] #scalar variable that doesnt need to be converted
                #Data to build MatrixChangeOfBasis with coo sparse format
                if computeMatrixChangeOfBasis:
                    rowMCB = np.empty((len(listGlobalVector)*Nel, nNd_elm, dim,dim))
                    colMCB = np.empty((len(listGlobalVector)*Nel, nNd_elm, dim,dim))
                    dataMCB = np.empty((len(listGlobalVector)*Nel, nNd_elm, dim,dim))
                    LocalFrameEl = elmRefGeom.GetLocalFrame(crd[elm], xi_nd, localFrame) #array of shape (Nel, nb_nd, nb of vectors in basis = dim, dim)
                    for ivec, vec in enumerate(listGlobalVector):
                        # dataMCB[ivec*Nel:(ivec+1)*Nel] = LocalFrameEl[:,:,:dim,:dim]                  
                        dataMCB[ivec*Nel:(ivec+1)*Nel] = LocalFrameEl                  
                        rowMCB[ivec*Nel:(ivec+1)*Nel] = np.arange(Nel).reshape(-1,1,1,1) + range_nNd_elm.reshape(1,-1,1,1)*Nel + np.array(vec).reshape(1,1,-1,1)*(Nel*nNd_elm)
                        colMCB[ivec*Nel:(ivec+1)*Nel] = elm.reshape(Nel,nNd_elm,1,1) + np.array(vec).reshape(1,1,1,-1)*Nnd        
    
                    if len(listScalarVariable) > 0:
                        #add the component from scalar variables (ie variable not requiring a change of basis)
                        dataMCB = sp.hstack( (dataMCB.reshape(-1), sp.ones(len(listScalarVariable)*Nel*nNd_elm) )) #no change of variable so only one value adding in dataMCB

                        rowMCB_loc = np.empty((len(listScalarVariable)*Nel, nNd_elm))
                        colMCB_loc = np.empty((len(listScalarVariable)*Nel, nNd_elm))
                        for ivar, var in enumerate(listScalarVariable):
                            rowMCB_loc[ivar*Nel:(ivar+1)*Nel] = np.arange(Nel).reshape(-1,1) + range_nNd_elm.reshape(1,-1)*Nel + var*(Nel*nNd_elm)
                            colMCB_loc[ivar*Nel:(ivar+1)*Nel] = elm + var*Nnd        
                        
                        rowMCB = sp.hstack( (rowMCB.reshape(-1), rowMCB_loc.reshape(-1)))
                        colMCB = sp.hstack( (colMCB.reshape(-1), colMCB_loc.reshape(-1)))
                        
                        MatrixChangeOfBasis = sparse.coo_matrix((dataMCB,(rowMCB,colMCB)), shape=(Nel*nNd_elm*Nvar, Nnd*Nvar))                   
                    else:
                        MatrixChangeOfBasis = sparse.coo_matrix((dataMCB.reshape(-1),(rowMCB.reshape(-1),colMCB.reshape(-1))), shape=(Nel*nNd_elm*Nvar, Nnd*Nvar))
                    
                    MatrixChangeOfBasis = MatrixChangeOfBasis.tocsr()                     
            
            Assembly.__saveMatrixChangeOfBasis[mesh.GetID()] = MatrixChangeOfBasis   
            return MatrixChangeOfBasis

        return Assembly.__saveMatrixChangeOfBasis[mesh.GetID()]

    @staticmethod
    def __GetResultGaussPoints(mesh, operator, U, elementType, nb_pg=None):  #return the results at GaussPoints      
        #TODO : can be accelerated by avoiding RowBlocMatrix (need to be checked) -> For each elementary 
        # 1 - at the very begining, compute Uloc = MatrixChangeOfBasis * U 
        # 2 - reshape Uloc to separate each var Uloc = Uloc.reshape(var, -1)
        # 3 - in the loop : res += coef_PG * (Assembly.__GetElementaryOp(mesh, operator.op[ii], elementType, nb_pg) , nvar, var, coef) * Uloc[var]
        
        res = 0
        nvar = ModelingSpace.GetNumberOfVariable()
        
        if isinstance(eval(elementType), dict):
            TypeOfCoordinateSystem = eval(elementType).get('__TypeOfCoordinateSystem', 'global')                
        else: TypeOfCoordinateSystem = 'global'

        MatrixChangeOfBasis = Assembly.__GetChangeOfBasisMatrix(mesh, TypeOfCoordinateSystem)
        associatedVariables = Assembly.__GetAssociatedVariables(elementType)
        
        for ii in range(len(operator.op)):
            var = [operator.op[ii].u] ; coef = [1] 
            
            if var[0] in associatedVariables:
                var.extend(associatedVariables[var[0]][0])
                coef.extend(associatedVariables[var[0]][1])     
    
            assert operator.op_vir[ii]==1, "Operator virtual are only required to build FE operators, but not to get element results"

            if isinstance(operator.coef[ii], Number): coef_PG = operator.coef[ii]                 
            else: coef_PG = Assembly.__ConvertToGaussPoints(mesh, operator.coef[ii][:], elementType, nb_pg)

            res += coef_PG * (RowBlocMatrix(Assembly.__GetElementaryOp(mesh, operator.op[ii], elementType, nb_pg) , nvar, var, coef) * MatrixChangeOfBasis * U)
        return res

    @staticmethod
    def __ConvertToGaussPoints(mesh, data, elementType, nb_pg=None):         
        """
        Convert an array of values related to a specific mesh (Nodal values, Element Values or Points of Gauss values) to the gauss points
        mesh: the considered Mesh object
        data: array containing the values (nodal or element value)
        The shape of the array is tested.
        """               
        if nb_pg is None: nb_pg = GetDefaultNbPG(elementType, mesh)             
        dataType = DetermineDataType(data, mesh, nb_pg)       

        if dataType == 'Node': 
            return Assembly.__GetNodeToGaussianPointMatrix(mesh, elementType, nb_pg) * data
        if dataType == 'Element':
            if len(np.shape(data)) == 1: return np.tile(data.copy(),nb_pg)
            else: return np.tile(data.copy(),[nb_pg,1])            
        return data #in case data contains already PG values
                
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
        
    def ConvertData(self, data, convertFrom=None, convertTo='GaussPoint'):
        return ConvertData(data, self.__Mesh, convertFrom, convertTo, self.__elmType, self.__nb_pg)
            
    def IntegrateField(self, Field, TypeField = 'GaussPoint'):
        assert TypeField in ['Node','GaussPoint','Element'], "TypeField should be 'Node', 'Element' or 'GaussPoint' values"
        Field = self.ConvertData(Field, TypeField, 'GaussPoint')
        return sum(Assembly.__GetGaussianQuadratureMatrix(self.__Mesh, self.__elmType, self.__nb_pg)@Field)

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
        
        
    def GetStrainTensor(self, U, Type="Nodal", nlgeom = None):
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

        if nlgeom is None: 
            if hasattr(self.__weakForm, 'nlgeom'): nlgeom = self.__weakForm.nlgeom
            else: nlgeom = False
            
        GradValues = self.GetGradTensor(U, Type)
        
        if nlgeom == False:
            Strain  = [GradValues[i][i] for i in range(3)] 
            Strain += [GradValues[0][1] + GradValues[1][0], GradValues[0][2] + GradValues[2][0], GradValues[1][2] + GradValues[2][1]]
        else:            
            Strain  = [GradValues[i][i] + 0.5*sum([GradValues[k][i]**2 for k in range(3)]) for i in range(3)] 
            Strain += [GradValues[0][1] + GradValues[1][0] + sum([GradValues[k][0]*GradValues[k][1] for k in range(3)])]             
            Strain += [GradValues[0][2] + GradValues[2][0] + sum([GradValues[k][0]*GradValues[k][2] for k in range(3)])]
            Strain += [GradValues[1][2] + GradValues[2][1] + sum([GradValues[k][1]*GradValues[k][2] for k in range(3)])]
        
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
        if Nvar is None: Nvar = ModelingSpace.GetNumberOfVariable()
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
#        res = res[:, [ModelingSpace.GetVariableRank('DispX'), ModelingSpace.GetVariableRank('DispY'), ModelingSpace.GetVariableRank('DispZ'), \
#                              ModelingSpace.GetVariableRank('ThetaX'), ModelingSpace.GetVariableRank('ThetaY'), ModelingSpace.GetVariableRank('ThetaZ')]]         
#        
#        if CoordinateSystem == 'local': return res
#        elif CoordinateSystem == 'global': 
#            #require a transformation between local and global coordinates on element
#            #classical MatrixChangeOfBasis transform only toward nodal values
#            elmRef = eval(self.__Mesh.GetElementShape())(1, mesh=mesh)#one pg  with the geometrical element
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
        nvar = ModelingSpace.GetNumberOfVariable()
        dim = ModelingSpace.GetDoF()
        MatrixChangeOfBasis = Assembly.__GetChangeOfBasisMatrix(mesh, self.__TypeOfCoordinateSystem)

        MatGaussianQuadrature = Assembly.__GetGaussianQuadratureMatrix(mesh, self.__elmType)
        associatedVariables = Assembly.__GetAssociatedVariables(self.__elmType)
        
        #TODO: use the computeGlobalMatrix() method to compute sum(operator.coef[ii]*Matvir * MatGaussianQuadrature * Mat)
        #add options in computeGlobalMatrix() to (i): dont save the computed matrix, (ii): neglect the ChangeOfBasis Matrix
        res = 0        
        for ii in range(len(operator.op)):
            var = [operator.op[ii].u] ; coef = [1]
            var_vir = [operator.op_vir[ii].u] ; coef_vir = [1]

            if var[0] in associatedVariables:
                var.extend(associatedVariables[var[0]][0])
                coef.extend(associatedVariables[var[0]][1])     
            if var_vir[0] in associatedVariables:
                var_vir.extend(associatedVariables[var_vir[0]][0])
                coef_vir.extend(associatedVariables[var_vir[0]][1])             

            Mat    =  RowBlocMatrix(Assembly.__GetElementaryOp(mesh, operator.op[ii], self.__elmType), nvar, var, coef)        
            Matvir =  RowBlocMatrix(Assembly.__GetElementaryOp(mesh, operator.op_vir[ii], self.__elmType), nvar, var_vir, coef_vir).T 

            if isinstance(operator.coef[ii], Number): #and self.op_vir[ii] != 1: 
                res = res + operator.coef[ii]*Matvir * MatGaussianQuadrature * Mat * MatrixChangeOfBasis * U   
            else:
                return NotImplemented                      
        
        res = np.reshape(res,(nvar,-1)).T
        
        Nel = mesh.GetNumberOfElements()
        res = (res[Nel:2*Nel,:]-res[0:Nel:,:])/2
        
        # if dim == 3:
        #     res = res[:, [ModelingSpace.GetVariableRank('DispX'), ModelingSpace.GetVariableRank('DispY'), ModelingSpace.GetVariableRank('DispZ'), \
        #                   ModelingSpace.GetVariableRank('RotX'), ModelingSpace.GetVariableRank('RotY'), ModelingSpace.GetVariableRank('RotZ')]]   
        # else: 
        #     res = res[:, [ModelingSpace.GetVariableRank('DispX'), ModelingSpace.GetVariableRank('DispY'), ModelingSpace.GetVariableRank('RotZ')]]   
        
        if CoordinateSystem == 'local': return res
        elif CoordinateSystem == 'global': 
            #require a transformation between local and global coordinates on element
            #classical MatrixChangeOfBasis transform only toward nodal values
            elmRef = eval(self.__Mesh.GetElementShape())(1, mesh=mesh)#one pg  with the geometrical element            
            if dim == 3: vec = [0,1,2] 
            else: vec = [0,1]
       
            #Data to build MatrixChangeOfBasisElement with coo sparse format
            crd = mesh.GetNodeCoordinates() ; elm = mesh.GetElementTable()
            rowMCB = np.empty((Nel, 1, dim,dim))
            colMCB = np.empty((Nel, 1, dim,dim))            
            rowMCB[:] = np.arange(Nel).reshape(-1,1,1,1) + np.array(vec).reshape(1,1,-1,1)*Nel # [[id_el + var*Nel] for var in vec]    
            colMCB[:] = np.arange(Nel).reshape(-1,1,1,1) + np.array(vec).reshape(1,1,1,-1)*Nel # [id_el+Nel*var for var in vec]
            dataMCB = elmRef.GetLocalFrame(crd[elm], elmRef.xi_pg, mesh.GetLocalFrame()) #array of shape (Nel, nb_pg=1, nb of vectors in basis = dim, dim)                        

            MatrixChangeOfBasisElement = sparse.coo_matrix((sp.reshape(dataMCB,-1),(sp.reshape(rowMCB,-1),sp.reshape(colMCB,-1))), shape=(dim*Nel, dim*Nel)).tocsr()
            
            F = np.reshape( MatrixChangeOfBasisElement.T * np.reshape(res[:,0:dim].T, -1)  ,  (dim,-1) ).T
            if dim == 3: 
                C = np.reshape( MatrixChangeOfBasisElement.T * np.reshape(res[:,3:6].T, -1)  ,  (3,-1) ).T
            else: C = res[:,2]
            
            return np.c_[F,C] #np.hstack((F,C))            



def ConvertData(data, mesh, convertFrom=None, convertTo='GaussPoint', elmType=None, nb_pg =None):        
    if isinstance(data, Number): return data
    
    if isinstance(mesh, str): mesh = Mesh.GetAll()[mesh]
    if elmType is None: elmType = mesh.GetElementShape()
    if nb_pg is None: nb_pg = GetDefaultNbPG(elmType, mesh)
    
    if isinstance(data, (listStrainTensor, listStressTensor)):        
        try:
            return type(data)(ConvertData(data.asarray().T, mesh, convertFrom, convertTo, elmType, nb_pg).T)
        except:
            NotImplemented
    
    if convertFrom is None: convertFrom = DetermineDataType(data, mesh, nb_pg)
        
    assert (convertFrom in ['Node','GaussPoint','Element']) and (convertTo in ['Node','GaussPoint','Element']), "only possible to convert 'Node', 'Element' and 'GaussPoint' values"
    
    if convertFrom == convertTo: return data       
    if convertFrom == 'Node': 
        data = Assembly._Assembly__GetNodeToGaussianPointMatrix(mesh, elmType, nb_pg) * data
        convertFrom = 'GaussPoint'
    elif convertFrom == 'Element':             
        if len(np.shape(data)) == 1: data = np.tile(data.copy(),nb_pg)
        else: data = np.tile(data.copy(),[nb_pg,1])
        convertFrom = 'GaussPoint'
        
    # from here convertFrom should be 'PG'
    if convertTo == 'Node': 
        return Assembly._Assembly__GetGaussianPointToNodeMatrix(mesh, elmType, nb_pg) * data 
    elif convertTo == 'Element': 
        return np.sum(np.split(data, nb_pg),axis=0) / nb_pg
    else: return data 

def DetermineDataType(data, mesh, nb_pg):               
        if isinstance(mesh, str): mesh = Mesh.GetAll()[mesh]
        if nb_pg is None: nb_pg = GetDefaultNbPG(elmType, mesh)
 
        test = 0
        if len(data) == mesh.GetNumberOfNodes(): 
            dataType = 'Node' #fonction définie aux noeuds   
            test+=1               
        if len(data) == mesh.GetNumberOfElements(): 
            dataType = 'Element' #fonction définie aux éléments
            test += 1
        if len(data) == nb_pg*mesh.GetNumberOfElements():
            dataType = 'GaussPoint'
            test += 1
        assert test, "Error: data doesn't match with the number of nodes, number of elements or number of gauss points."
        if test>1: "Warning: kind of data is confusing. " + dataType +" values choosen."
        return dataType        