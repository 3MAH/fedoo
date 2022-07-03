#simcoon compatible

from fedoo.libAssembly.AssemblyBase import AssemblyBase, AssemblySum
from fedoo.libUtil.PostTreatement import listStressTensor, listStrainTensor
from fedoo.libMesh.Mesh import Mesh
from fedoo.libElement import *
from fedoo.libWeakForm.WeakForm import WeakForm
from fedoo.libConstitutiveLaw.ConstitutiveLaw import ConstitutiveLaw
from fedoo.libAssembly.SparseMatrix import _BlocSparse as BlocSparse
from fedoo.libAssembly.SparseMatrix import _BlocSparseOld as BlocSparseOld #required for 'old' computeMatrixMehtod
from fedoo.libAssembly.SparseMatrix import RowBlocMatrix

from scipy import sparse
import numpy as np
from numbers import Number 
from copy import copy
import time


def Create(weakform, mesh="", elementType="", name ="", **kargs): 
    if isinstance(weakform, str):
        weakform = WeakForm.get_all()[weakform]
        
    if hasattr(weakform, 'list_weakform') and weakform.assembly_options is None: #WeakFormSum object
        list_weakform = weakform.list_weakform
        
        if isinstance(mesh, str): mesh = Mesh.get_all()[mesh]
        if elementType == "": elementType = mesh.elm_type
                
        #get lists of some non compatible assembly_options items for each weakform in list_weakform
        list_nb_gp = [wf.assembly_options.get('nb_gp', GetDefaultNbPG(elementType, mesh)) for wf in list_weakform]
        list_assume_sym = [wf.assembly_options.get('assume_sym', False) for wf in list_weakform]
        list_prop = list(zip(list_nb_gp, list_assume_sym))
        list_diff_prop = list(set(list_prop)) #list of different non compatible properties that required separated assembly
        
        if len(list_diff_prop) == 1: #only 1 assembly is required
            #update assembly_options
            prop = list_diff_prop[0]
            weakform.assembly_options = {'nb_gp': prop[0] , 'assume_sym': prop[1]}
            weakform.assembly_options['mat_lumping'] = [wf.assembly_options.get('mat_lumping',False) for wf in weakform.list_weakform]
            return Assembly(weakform, mesh, elementType, name, **kargs)
        
        else: #we need to create and sum several assemblies
            list_assembly = []
            for prop in list_diff_prop:
                l_wf = [list_weakform[i] for i,p in enumerate(list_prop) if p == prop] #list_weakform with compatible properties
                if len(l_wf) == 1:
                    wf = l_wf[0] #standard weakform. No WeakFormSum required
                else:
                    #create a new WeakFormSum object
                    wf = WeakFormSum(l_wf) #to modify : add automatic name
                    #define the assembly_options dict of the new weakform
                    wf.assembly_options = {'nb_gp': prop[0] , 'assume_sym': prop[1]}
                    wf.assembly_options['assume_sym'] = [w.assembly_options['assume_sym'] for w in l_wf]
                list_assembly.append(Assembly(wf, mesh, elementType, "", **kargs))
        
        # list_assembly = [Assembly(wf, mesh, elementType, "", **kargs) for wf in weakform.list_weakform]
        kargs['assembly_output'] = kargs.get('assembly_output', list_assembly[0])
        return AssemblySum(list_assembly, name, **kargs)       
    
    return Assembly(weakform, mesh, elementType, name, **kargs)
               
class Assembly(AssemblyBase):
    _saved_elementary_operator = {} 
    _saved_change_of_basis_mat = {}   
    _saved_gaussian_quadrature_mat = {} 
    _saved_node2gausspoint_mat = {}
    _saved_gausspoint2node_mat = {}
    _saved_associated_variables = {} #dict containing all associated variables (rotational dof for C1 elements) for elementType
           
    def __init__(self,weakform, mesh="", elementType="", name ="", **kargs):                      
        
        if isinstance(weakform, str):
            weakform = WeakForm.get_all()[weakform]
        
        if weakform.assembly_options is None:
            #should be a non compatible WeakFormSum object
            raise NameError('Some Assembly associated to WeakFormSum object can only be created using the Create function')

        if isinstance(mesh, str):
            mesh = Mesh.get_all()[mesh]
            
        if isinstance(weakform, WeakForm):
            self._weakform = weakform
            AssemblyBase.__init__(self, name, weakform.space)
        else: #weakform should be a ModelingSpace object
            assert hasattr(weakform, 'list_variable') and hasattr(weakform, 'list_coordinate'),\
                'WeakForm not understood'
            self._weakform = None
            AssemblyBase.__init__(self, name, space = weakform)
        
        #attributes to set assembly related to current (deformed) configuration
        #used for update lagrangian method. 
        self.current = self 
        
        self.__MeshChange = kargs.pop('MeshChange', False)        
        self.__Mesh = mesh   
        if elementType == "": elementType = mesh.elm_type
        self.__elmType= elementType #.lower()
        self.__TypeOfCoordinateSystem = self._GetTypeOfCoordinateSystem()

        self.__nb_gp = kargs.pop('nb_gp', None)
        if self.__nb_gp is None: 
            self.__nb_gp = weakform.assembly_options.get('nb_gp', GetDefaultNbPG(elementType, mesh))
        
        self.assume_sym = weakform.assembly_options.get('assume_sym', False)
        self.mat_lumping = weakform.assembly_options.get('mat_lumping', False)
        
        self.__saveBlocStructure = None #use to save data about the sparse structure and avoid time consuming recomputation
        self.computeMatrixMethod = 'new' #computeMatrixMethod = 'old' and 'very_old' only used for debug purpose        
        self.__factorize_op = True #option for debug purpose (should be set to True for performance)        


    def ComputeGlobalMatrix(self, compute = 'all'):
        """
        Compute the global matrix and global vector related to the assembly
        if compute = 'all', compute the global matrix and vector
        if compute = 'matrix', compute only the matrix
        if compute = 'vector', compute only the vector
        if compute = 'none', compute nothing
        """                        
        if compute == 'none': return
        
        # t0 = time.time()

        computeMatrixMethod = self.computeMatrixMethod
        
        nb_gp = self.__nb_gp
        mesh = self.__Mesh        
        
        if self.__MeshChange == True:             
            if mesh in Assembly._saved_change_of_basis_mat: del Assembly._saved_change_of_basis_mat[mesh]            
            self.PreComputeElementaryOperators()
                 
        nvar = self.space.nvar
        wf = self._weakform.GetDifferentialOperator(mesh)      
        
        MatGaussianQuadrature = self._get_gaussian_quadrature_mat()        
        MatrixChangeOfBasis = self.get_change_of_basis_mat()        
        associatedVariables = self._get_associated_variables() #for element requiring many variable such as beam with disp and rot dof        

        if computeMatrixMethod == 'new': 
            
            intRef, sorted_indices = wf.sort()
            #sl contains list of slice object that contains the dimension for each variable
            #size of VV and sl must be redefined for case with change of basis
            VV = 0
            nbNodes = self.__Mesh.n_nodes            
            sl = [slice(i*nbNodes, (i+1)*nbNodes) for i in range(nvar)] 
            
            if nb_gp == 0: #if finite difference elements, don't use BlocSparse                              
                blocks = [[None for i in range(nvar)] for j in range(nvar)]
                self.__saveBlocStructure = 0 #don't save block structure for finite difference mesh
                    
                Matvir = self._get_elementary_operator(wf.op_vir[0], nb_gp=0)[0].T #should be identity matrix restricted to nodes used in the finite difference mesh
                
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
                        if VV is 0: VV = np.zeros((self.__Mesh.n_nodes * nvar))
                        VV[sl[var_vir[i]]] = VV[sl[var_vir[i]]] - (coef_PG) 
                            
                    else: #virtual and real operators -> compute a matrix
                        var = wf.op[ii].u   
                        if isinstance(coef_PG, Number): coef_PG = coef_PG * np.ones_like(MatGaussianQuadrature.data)
                        CoefMatrix = sparse.csr_matrix( (coef_PG, MatGaussianQuadrature.indices, MatGaussianQuadrature.indptr), shape = MatGaussianQuadrature.shape)   
                        Mat    =  self._get_elementary_operator(wf.op[ii])[0]
                        
                        if blocks[var_vir][var] is None: 
                            blocks[var_vir][var] = Matvir @ CoefMatrix @ Mat
                        else:                    
                            blocks[var_vir][var].data += (Matvir @ CoefMatrix @ Mat).data
                            
                blocks = [[b if b is not None else sparse.csr_matrix((nbNodes,nbNodes)) \
                          for b in blocks_row] for blocks_row in blocks ]        
                MM = sparse.bmat(blocks, format ='csr')
                
            else:
                MM = BlocSparse(nvar, nvar, self.__nb_gp, self.__saveBlocStructure, assume_sym = self.assume_sym)
                listMatvir = listCoef_PG = None
                
                sum_coef = False #bool that indicate if operator are the same and can be sum
                
                if hasattr(self._weakform, '_list_mat_lumping'):
                    change_mat_lumping = True  #use different mat_lumping option for each operator                             
                else:
                    change_mat_lumping = False
                    mat_lumping = self.mat_lumping                

                for ii in range(len(wf.op)):                                        
                        
                    if compute == 'matrix' and wf.op[ii] is 1: continue
                    if compute == 'vector' and wf.op[ii] is not 1: continue
                    
                    if wf.op[ii] is not 1 and self.assume_sym and wf.op[ii].u < wf.op_vir[ii].u:
                        continue                
                    
                    if change_mat_lumping:
                        mat_lumping = self._weakform._list_mat_lumping[sorted_indices[ii]]                        
                    
                    if isinstance(wf.coef[ii], Number) or len(wf.coef[ii])==1: 
                        #if nb_gp == 0, coef_PG = nodal values (finite diffirences)
                        coef_PG = wf.coef[ii] 
                    else:
                        coef_PG = self._convert_to_gausspoints(wf.coef[ii][:])                                                 
                    
                    # if ii > 0 and intRef[ii] == intRef[ii-1]: #if same operator as previous with different coef, add the two coef
                    if sum_coef: #if same operator as previous with different coef, add the two coef
                        coef_PG_sum += coef_PG
                        sum_coef = False
                    else: coef_PG_sum = coef_PG   
                    
                    if ii < len(wf.op)-1 and intRef[ii] == intRef[ii+1]: #if operator similar to the next, continue
                        if not(change_mat_lumping) or mat_lumping == self._weakform._list_mat_lumping[sorted_indices[ii+1]]: 
                            sum_coef = True
                            continue
                                    
                    coef_PG = coef_PG_sum * MatGaussianQuadrature.data #MatGaussianQuadrature.data is the diagonal of MatGaussianQuadrature
    #                Matvir = (RowBlocMatrix(self._get_elementary_operator(wf.op_vir[ii]), nvar, var_vir, coef_vir) * MatrixChangeOfBasis).T
                    #check how it appens with change of variable and rotation dof
                    
                    Matvir = self._get_elementary_operator(wf.op_vir[ii])
                    
                    if listMatvir is not None: #factorization of real operator (sum of virtual operators)
                        listMatvir = [listMatvir[j]+[Matvir[j]] for j in range(len(Matvir))] 
                        listCoef_PG = listCoef_PG + [coef_PG]  
                        
                    if ii < len(wf.op)-1 and wf.op[ii] != 1 and wf.op[ii+1] != 1 and self.__factorize_op == True:
                        #if it possible, factorization of op to increase assembly performance (sum of several op_vir)                        
                        factWithNextOp = [wf.op[ii].u, wf.op[ii].x, wf.op[ii].ordre, wf.op_vir[ii].u] == [wf.op[ii+1].u, wf.op[ii+1].x, wf.op[ii+1].ordre, wf.op_vir[ii+1].u] #True if factorization is possible with next op                                            
                        if factWithNextOp:
                            if listMatvir is None: 
                                listMatvir = [[Matvir[j]] for j in range(len(Matvir))] #initialization of listMatvir and listCoef_PG
                                listCoef_PG = [coef_PG]
                            continue #si factorization possible -> go to next op, all the factorizable operators will treat together                        
                    else: factWithNextOp = False

                    coef_vir = [1] ; var_vir = [wf.op_vir[ii].u] #list in case there is an angular variable                                                   
                    if var_vir[0] in associatedVariables:
                        var_vir.extend(associatedVariables[var_vir[0]][0])
                        coef_vir.extend(associatedVariables[var_vir[0]][1])   
                    
                    if wf.op[ii] == 1: #only virtual operator -> compute a vector 
                        if VV is 0: VV = np.zeros((self.__Mesh.n_nodes * nvar))
                        for i in range(len(Matvir)):
                            VV[sl[var_vir[i]]] = VV[sl[var_vir[i]]] - coef_vir[i] * Matvir[i].T * (coef_PG) #this line may be optimized
                            
                    else: #virtual and real operators -> compute a matrix                                         
                        coef = [1] ; var = [wf.op[ii].u] #list in case there is an angular variable                
                        if var[0] in associatedVariables:
                            var.extend(associatedVariables[var[0]][0])
                            coef.extend(associatedVariables[var[0]][1])                                             
    
    #                    Mat    =  RowBlocMatrix(self._get_elementary_operator(wf.op[ii]), nvar, var, coef)         * MatrixChangeOfBasis             
                        Mat    =  self._get_elementary_operator(wf.op[ii])
        
                        #Possibility to increase performance for multivariable case 
                        #the structure should be the same for derivative dof, so the blocs could be computed altogether
                        if listMatvir is None:
                            for i in range(len(Mat)):
                                for j in range(len(Matvir)):
                                    MM.addToBlocATB(Matvir[j], Mat[i], (coef[i]*coef_vir[j]) * coef_PG, var_vir[j], var[i], mat_lumping)
                        else:
                            for i in range(len(Mat)):
                                for j in range(len(Matvir)):
                                    MM.addToBlocATB(listMatvir[j], Mat[i], [(coef[i]*coef_vir[j]) * coef_PG for coef_PG in listCoef_PG], var_vir[j], var[i], mat_lumping)                        
                            listMatvir = None
                            listCoef_PG = None
                        
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
            
            if (mesh, self.__elmType, nb_gp) not in Assembly._saved_elementary_operator:
                Assembly._saved_elementary_operator[(mesh, self.__elmType, nb_gp)] = {}
            saveOperator = Assembly._saved_elementary_operator[(mesh, self.__elmType, nb_gp)]
            
            #list_elementType contains the id of the element associated with every variable
            #list_elementType could be stored to avoid reevaluation 
            if isinstance(eval(self.__elmType), dict):
                elementDict = eval(self.__elmType)
                list_elementType = [elementDict.get(self.space.variable_name(i))[0] for i in range(nvar)]
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
            nbNodes = self.__Mesh.n_nodes            
            sl = [slice(i*nbNodes, (i+1)*nbNodes) for i in range(nvar)] 
            
            for ii in range(len(wf.op)):                   
                if compute == 'matrix' and wf.op[ii] is 1: continue
                if compute == 'vector' and wf.op[ii] is not 1: continue
            
                if isinstance(wf.coef[ii], Number) or len(wf.coef[ii])==1: 
                    coef_PG = wf.coef[ii] #MatGaussianQuadrature.data is the diagonal of MatGaussianQuadrature
                else:
                    coef_PG = self._convert_to_gausspoints(wf.coef[ii][:])                                                 
                
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
                                            
                    Matvir = self._get_elementary_operator(wf.op_vir[ii])         
                    if VV is 0: VV = np.zeros((self.__Mesh.n_nodes * nvar))
                    for i in range(len(Matvir)):
                        VV[sl[var_vir[i]]] = VV[sl[var_vir[i]]] - coef_vir[i] * Matvir[i].T * (coef_PG) #this line may be optimized
                        
                else: #virtual and real operators -> compute a matrix
                    coef = [1] ; var = [wf.op[ii].u] #list in case there is an angular variable                
                    if var[0] in associatedVariables:
                        var.extend(associatedVariables[var[0]][0])
                        coef.extend(associatedVariables[var[0]][1])                                                                                     
                    
                    tuplename = (list_elementType[wf.op_vir[ii].u], wf.op_vir[ii].x, wf.op_vir[ii].ordre, list_elementType[wf.op[ii].u], wf.op[ii].x, wf.op[ii].ordre) #tuple to identify operator
                    if tuplename in saveOperator:
                        MatvirT_Mat = saveOperator[tuplename] #MatvirT_Mat is an array that contains usefull data to build the matrix MatvirT*Matcoef*Mat where Matcoef is a diag coefficient matrix. MatvirT_Mat is build with BlocSparse class
                    else: 
                        MatvirT_Mat = None
                        saveOperator[tuplename] = [[None for i in range(len(var))] for j in range(len(var_vir))]
                        Matvir = self._get_elementary_operator(wf.op_vir[ii])         
                        Mat = self._get_elementary_operator(wf.op[ii])

                    for i in range(len(var)):
                        for j in range(len(var_vir)):
                            if MatvirT_Mat is not None:           
                                MM.addToBloc(MatvirT_Mat[j][i], (coef[i]*coef_vir[j]) * coef_PG, var_vir[j], var[i])                                     
                            else:  
                                saveOperator[tuplename][j][i] = MM.addToBlocATB(Matvir[j], Mat[i], (coef[i]*coef_vir[j]) * coef_PG, var_vir[j], var[i])
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
                     
                Matvir = (RowBlocMatrix(self._get_elementary_operator(wf.op_vir[ii]), nvar, var_vir, coef_vir) * MatrixChangeOfBasis).T
    
                if wf.op[ii] == 1: #only virtual operator -> compute a vector 
                    if isinstance(wf.coef[ii], Number): 
                        VV = VV - wf.coef[ii]*Matvir * MatGaussianQuadrature.data
                    else:
                        coef_PG = self._convert_to_gausspoints(wf.coef[ii][:])*MatGaussianQuadrature.data                             
                        VV = VV - Matvir * (coef_PG)
                        
                else: #virtual and real operators -> compute a matrix
                    coef = [1] ; var = [wf.op[ii].u] #list in case there is an angular variable                  
                    if var[0] in associatedVariables:
                        var.extend(associatedVariables[var[0]][0])
                        coef.extend(associatedVariables[var[0]][1])     
                                    
                    Mat    =  RowBlocMatrix(self._get_elementary_operator(wf.op[ii]), nvar, var, coef)         * MatrixChangeOfBasis             
    
                    if isinstance(wf.coef[ii], Number): #and self.op_vir[ii] != 1: 
                        MM = MM + wf.coef[ii]*Matvir * MatGaussianQuadrature * Mat  
                    else:
                        coef_PG = self._convert_to_gausspoints(wf.coef[ii][:])                    
                        CoefMatrix = sparse.csr_matrix( (MatGaussianQuadrature.data*coef_PG, MatGaussianQuadrature.indices, MatGaussianQuadrature.indptr), shape = MatGaussianQuadrature.shape)   
                        MM = MM + Matvir * CoefMatrix * Mat                

#            MM = MM.tocsr()
#            MM.eliminate_zeros()
            if compute != 'vector': self.SetMatrix(MM) #format csr         
            if compute != 'matrix': self.SetVector(VV) #numpy array
            
        # print('temps : ', print(compute), ' - ', time.time()- t0)
    
    def SetMesh(self, mesh):
        self.__Mesh = mesh

    def GetMesh(self):
        return self.__Mesh
    
    def GetWeakForm(self):
        return self._weakform
           
    def GetNumberOfGaussPoints(self):
        return self.__nb_gp
    
    def get_change_of_basis_mat(self):
        if self.__TypeOfCoordinateSystem == 'global': return 1
        
        mesh = self.__Mesh
        if mesh not in Assembly._saved_change_of_basis_mat:        
            ### change of basis treatment for beam or plate elements
            ### Compute the change of basis matrix for vector defined in self.space.list_vector()
            MatrixChangeOfBasis = 1
            computeMatrixChangeOfBasis = False

            Nnd = mesh.n_nodes
            Nel = mesh.n_elements
            elm = mesh.elements
            nNd_elm = np.shape(elm)[1]            
            crd = mesh.nodes
            dim = self.space.ndim
            localFrame = mesh.local_frame
            elmRefGeom = eval(mesh.elm_type)(mesh=mesh)
    #        xi_nd = elmRefGeom.xi_nd
            xi_nd = GetNodePositionInElementCoordinates(mesh.elm_type, nNd_elm) #function to define

            if 'X' in mesh.crd_name and 'Y' in mesh.crd_name: #if not in physical space, no change of variable                
                for nameVector in self.space.list_vector():
                    if computeMatrixChangeOfBasis == False:
                        range_nNd_elm = np.arange(nNd_elm) 
                        computeMatrixChangeOfBasis = True
                        nvar = self.space.nvar
                        listGlobalVector = []  ; listScalarVariable = list(range(nvar))
#                        MatrixChangeOfBasis = sparse.lil_matrix((nvar*Nel*nNd_elm, nvar*Nnd)) #lil is very slow because it change the sparcity of the structure
                    listGlobalVector.append(self.space.get_vector(nameVector)) #vector that need to be change in local coordinate            
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
                        dataMCB = np.hstack( (dataMCB.reshape(-1), np.ones(len(listScalarVariable)*Nel*nNd_elm) )) #no change of variable so only one value adding in dataMCB

                        rowMCB_loc = np.empty((len(listScalarVariable)*Nel, nNd_elm))
                        colMCB_loc = np.empty((len(listScalarVariable)*Nel, nNd_elm))
                        for ivar, var in enumerate(listScalarVariable):
                            rowMCB_loc[ivar*Nel:(ivar+1)*Nel] = np.arange(Nel).reshape(-1,1) + range_nNd_elm.reshape(1,-1)*Nel + var*(Nel*nNd_elm)
                            colMCB_loc[ivar*Nel:(ivar+1)*Nel] = elm + var*Nnd        
                        
                        rowMCB = np.hstack( (rowMCB.reshape(-1), rowMCB_loc.reshape(-1)))
                        colMCB = np.hstack( (colMCB.reshape(-1), colMCB_loc.reshape(-1)))
                        
                        MatrixChangeOfBasis = sparse.coo_matrix((dataMCB,(rowMCB,colMCB)), shape=(Nel*nNd_elm*nvar, Nnd*nvar))                   
                    else:
                        MatrixChangeOfBasis = sparse.coo_matrix((dataMCB.reshape(-1),(rowMCB.reshape(-1),colMCB.reshape(-1))), shape=(Nel*nNd_elm*nvar, Nnd*nvar))
                    
                    MatrixChangeOfBasis = MatrixChangeOfBasis.tocsr()                     
            
            Assembly._saved_change_of_basis_mat[mesh] = MatrixChangeOfBasis   
            return MatrixChangeOfBasis

        return Assembly._saved_change_of_basis_mat[mesh]


    # def InitializeConstitutiveLaw(self, assembly, pb, initialTime=0.):
    #     if hasattr(self,'nlgeom'): nlgeom = self.nlgeom
    #     else: nlgeom=False
    #     constitutivelaw = self.GetConstitutiveLaw()
        
    #     if constitutivelaw is not None:
    #         if isinstance(constitutivelaw, list):
    #             for cl in constitutivelaw:
    #                 cl.Initialize(assembly, pb, initialTime, nlgeom)
    #         else:
    #             constitutivelaw.Initialize(assembly, pb, initialTime, nlgeom)
 
    def Initialize(self, pb, initialTime=0.):
        """
        Initialize the associated weak form and assemble the global matrix with the elastic matrix
        Parameters: 
            - initialTime: the initial time        
        """        
        if self._weakform.GetConstitutiveLaw() is not None:
            if hasattr(self._weakform,'nlgeom'): nlgeom = self._weakform.nlgeom
            else: nlgeom=False
            self._weakform.GetConstitutiveLaw().Initialize(self, pb, initialTime, nlgeom)
        
        self._weakform.Initialize(self, pb, initialTime)
                
    def InitTimeIncrement(self, pb, dtime):
        self._weakform.InitTimeIncrement(self, pb, dtime)
        self.ComputeGlobalMatrix() 
        #no need to compute vector if the previous iteration has converged and (dtime hasn't changed or dtime isn't used in the weakform)
        #in those cases, self.ComputeGlobalMatrix(compute = 'matrix') should be more efficient

    def Update(self, pb, dtime=None, compute = 'all'):
        """
        Update the associated weak form and assemble the global matrix
        Parameters: 
            - pb: a Problem object containing the Dof values
            - time: the current time        
        """
        if self._weakform.GetConstitutiveLaw() is not None:
            self._weakform.GetConstitutiveLaw().Update(self, pb, dtime)
        self._weakform.Update(self, pb, dtime)
        self.current.ComputeGlobalMatrix(compute)

    def ResetTimeIncrement(self):
        """
        Reset the current time increment (internal variable in the constitutive equation)
        Doesn't assemble the new global matrix. Use the Update method for that purpose.
        """
        if self._weakform.GetConstitutiveLaw() is not None:
            self._weakform.GetConstitutiveLaw().ResetTimeIncrement()
        self._weakform.ResetTimeIncrement()
        # self.ComputeGlobalMatrix(compute='all')

    def NewTimeIncrement(self):
        """
        Apply the modification to the constitutive equation required at each change of time increment. 
        Generally used to increase non reversible internal variable
        Doesn't assemble the new global matrix. Use the Update method for that purpose.
        """
        if self._weakform.GetConstitutiveLaw() is not None:
            self._weakform.GetConstitutiveLaw().NewTimeIncrement()
        self._weakform.NewTimeIncrement() #should update GetH() method to return elastic rigidity matrix for prediction        
        # self.ComputeGlobalMatrix(compute='matrix')
 
    def Reset(self):
        """
        Reset the assembly to it's initial state.
        Internal variable in the constitutive equation are reinitialized 
        and stored global matrix and vector are deleted
        """
        if self._weakform.GetConstitutiveLaw() is not None:
            self._weakform.GetConstitutiveLaw().Reset()
        self._weakform.Reset()    
        self.deleteGlobalMatrix()

    @staticmethod
    def delete_memory():
        """
        Static method of the Assembly class. 
        Erase all the static variables of the Assembly object. 
        Stored data, are data that are used to compute the global assembly in an
        efficient way. 
        However, the stored data may cause errors if the mesh is modified. In this case, the data should be recomputed, 
        but it is not done by default. In this case, deleting the memory should 
        resolve the problem. 
        
        -----------
        Remark : it the MeshChange argument is set to True when creating the Assembly object, the
        memory will be recomputed by default, which may cause a decrease in assembling performances
        """
        Assembly._saved_elementary_operator = {} 
        Assembly._saved_change_of_basis_mat = {}   
        Assembly._saved_gaussian_quadrature_mat = {} 
        Assembly._saved_node2gausspoint_mat = {}
        Assembly._saved_gausspoint2node_mat = {}
        Assembly._saved_associated_variables = {} #dict containing all associated variables (rotational dof for C1 elements) for elementType
        
        
    def PreComputeElementaryOperators(self,nb_gp = None): #Précalcul des opérateurs dérivés suivant toutes les directions (optimise les calculs en minimisant le nombre de boucle)               
        #-------------------------------------------------------------------
        #Initialisation   
        #-------------------------------------------------------------------
        mesh = self.__Mesh
        elementType = self.__elmType
        if nb_gp is None: NumberOfGaussPoint = self.__nb_gp
        else: NumberOfGaussPoint = nb_gp
                  
        Nnd = mesh.n_nodes
        Nel = mesh.n_elements
        elm = mesh.elements
        nNd_elm = np.shape(elm)[1]
        crd = mesh.nodes
        
        #-------------------------------------------------------------------
        #Case of finite difference mesh    
        #-------------------------------------------------------------------        
        if NumberOfGaussPoint == 0: # in this case, it is a finite difference mesh
            # we compute the operators directly from the element library
            elmRef = eval(elementType)(NumberOfGaussPoint)
            OP = elmRef.computeOperator(crd,elm)
            Assembly._saved_gaussian_quadrature_mat[(mesh,NumberOfGaussPoint)] = sparse.identity(OP[0][0].shape[0], 'd', format= 'csr') #No gaussian quadrature in this case : nodal identity matrix
            Assembly._saved_gausspoint2node_mat[(mesh, NumberOfGaussPoint)] = 1  #no need to translate between pg and nodes because no pg 
            Assembly._saved_node2gausspoint_mat[(mesh, NumberOfGaussPoint)] = 1                                    
            Assembly._saved_change_of_basis_mat[mesh] = 1 # No change of basis:  MatrixChangeOfBasis = 1 #this line could be deleted because the coordinate should in principle defined as 'global' 
            Assembly._saved_elementary_operator[(mesh,elementType,NumberOfGaussPoint)] = OP #elmRef.computeOperator(crd,elm)
            return                                

        #-------------------------------------------------------------------
        #Initialise the geometrical interpolation
        #-------------------------------------------------------------------   
        elmRefGeom = eval(mesh.elm_type)(NumberOfGaussPoint, mesh=mesh) #initialise element
        nNd_elm_geom = len(elmRefGeom.xi_nd) #number of dof used in the geometrical interpolation
        elm_geom = elm[:,:nNd_elm_geom] 

        localFrame = mesh.local_frame
        nb_elm_nd = np.bincount(elm_geom.reshape(-1)) #len(nb_elm_nd) = Nnd #number of element connected to each node        
        vec_xi = elmRefGeom.xi_pg #coordinate of points of gauss in element coordinate (xi)
        
        elmRefGeom.ComputeJacobianMatrix(crd[elm_geom], vec_xi, localFrame) #compute elmRefGeom.JacobianMatrix, elmRefGeom.detJ and elmRefGeom.inverseJacobian

        #-------------------------------------------------------------------
        # Compute the diag matrix used for the gaussian quadrature
        #-------------------------------------------------------------------  
        gaussianQuadrature = (elmRefGeom.detJ * elmRefGeom.w_pg).T.reshape(-1) 
        Assembly._saved_gaussian_quadrature_mat[(mesh,NumberOfGaussPoint)] = sparse.diags(gaussianQuadrature, 0, format='csr') #matrix to get the gaussian quadrature (integration over each element)        

        #-------------------------------------------------------------------
        # Compute the array containing row and col indices used to assemble the sparse matrices
        #-------------------------------------------------------------------          
        range_nbPG = np.arange(NumberOfGaussPoint)                 
        if self.get_change_of_basis_mat() is 1: ChangeOfBasis = False
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
        Assembly._saved_gausspoint2node_mat[(mesh, NumberOfGaussPoint)] = sparse.coo_matrix((dataPGtoNode.reshape(-1),(col_geom,row_geom)), shape=(Nnd,Nel*NumberOfGaussPoint) ).tocsr() #matrix to compute the node values from pg using the geometrical shape functions 

        #-------------------------------------------------------------------
        # Assemble the matrix that compute the pg values from nodes using the geometrical shape functions (no angular dof for ex)    
        #-------------------------------------------------------------------             
        dataNodeToPG = np.empty((Nel, NumberOfGaussPoint, nNd_elm_geom))
        dataNodeToPG[:] = elmRefGeom.ShapeFunctionPG.reshape((1,NumberOfGaussPoint,nNd_elm_geom)) 
        Assembly._saved_node2gausspoint_mat[(mesh, NumberOfGaussPoint)] = sparse.coo_matrix((np.reshape(dataNodeToPG,-1),(row_geom,col_geom)), shape=(Nel*NumberOfGaussPoint, Nnd) ).tocsr() #matrix to compute the pg values from nodes using the geometrical shape functions (no angular dof)

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
                data[1, i] = op_dd[i+1] #as index and indptr should be the same, perhaps it will be more memory efficient to only store the data field

            Assembly._saved_elementary_operator[(mesh,elementType,NumberOfGaussPoint)] = data   
    
    def _get_elementary_operator(self, deriv, nb_gp=None): 
        #Gives a list of sparse matrix that convert node values for one variable to the pg values of a simple derivative op (for instance d/dz)
        #The list contains several element if the elementType include several variable (dof variable in beam element). In other case, the list contains only one matrix
        #The variables are not considered. For a global use, the resulting matrix should be assembled in a block matrix with the nodes values for all variables
        if nb_gp is None: nb_gp = self.__nb_gp

        elementType = self.__elmType
        mesh = self.__Mesh
        
        if isinstance(eval(elementType), dict):
            elementDict = eval(elementType)
            elementType = elementDict.get(self.space.variable_name(deriv.u))
            if elementType is None: elementType = elementDict.get('__default')
            elementType = elementType[0]
            
        if not((mesh,elementType,nb_gp) in Assembly._saved_elementary_operator):
            self.PreComputeElementaryOperators(nb_gp)
        
        data = Assembly._saved_elementary_operator[(mesh,elementType,nb_gp)]

        if deriv.ordre == 0 and 0 in data:
            return data[0]
        
        #extract the mesh coordinate that corespond to coordinate rank given in deriv.x     
        ListMeshCoordinatenameRank = [self.space.coordinate_rank(crdname) for crdname in mesh.crd_name if crdname in self.space.list_coordinate()]
        if deriv.x in ListMeshCoordinatenameRank: xx= ListMeshCoordinatenameRank.index(deriv.x)
        else: return data[0] #if the coordinate doesnt exist, return operator without derivation (for PGD)
                         
        if (deriv.ordre, xx) in data:
            return data[deriv.ordre, xx]
        else: assert 0, "Operator unavailable"      
              

    def _get_gaussian_quadrature_mat(self): #calcul la discrétision relative à un seul opérateur dérivé   
        mesh = self.__Mesh
        nb_gp = self.__nb_gp
        if not((mesh,nb_gp) in Assembly._saved_gaussian_quadrature_mat):
            self.PreComputeElementaryOperators()
        return Assembly._saved_gaussian_quadrature_mat[(mesh,nb_gp)]

    def _get_associated_variables(self): #associated variables (rotational dof for C1 elements) of elementType        
        elementType = self.__elmType
        if elementType not in Assembly._saved_associated_variables:
            objElement = eval(elementType)
            if isinstance(objElement, dict):            
                Assembly._saved_associated_variables[elementType] = {self.space.variable_rank(key): 
                                       [[self.space.variable_rank(v) for v in val[1][1::2]],
                                        val[1][0::2]] for key,val in objElement.items() if len(val)>1 and key in self.space.list_variable()} 
                    # val[1][0::2]] for key,val in objElement.items() if key in self.space.list_variable() and len(val)>1}
            else: Assembly._saved_associated_variables[elementType] = {}
        return Assembly._saved_associated_variables[elementType] 

    def _GetTypeOfCoordinateSystem(self): 
        #determine the type of coordinate system used for vector of variables (displacement for instance). This type may be specified in element (under dict form only)        
        #TypeOfCoordinateSystem may be 'local' or 'global'. If 'local' variables are used, a change of variable is required
        #If TypeOfCoordinateSystemis not specified in the element, 'global' value (no change of basis) is considered by default
        if isinstance(eval(self.__elmType), dict):
            return eval(self.__elmType).get('__TypeOfCoordinateSystem', 'global')                
        else: 
            return 'global'
    
    def _get_gausspoint2node_mat(self, nb_gp=None): #calcul la discrétision relative à un seul opérateur dérivé   
        if nb_gp is None: nb_gp = self.__nb_gp     
        if not((self.__Mesh,nb_gp) in Assembly._saved_gausspoint2node_mat):
            self.PreComputeElementaryOperators(nb_gp)        
        return Assembly._saved_gausspoint2node_mat[(self.__Mesh,nb_gp)]
    
    def _get_node2gausspoint_mat(self, nb_gp=None): #calcul la discrétision relative à un seul opérateur dérivé   
        if nb_gp is None: nb_gp = self.__nb_gp     
        if not((self.__Mesh,nb_gp) in Assembly._saved_node2gausspoint_mat):
            Assembly.PreComputeElementaryOperators(nb_gp)
        
        return Assembly._saved_node2gausspoint_mat[(self.__Mesh,nb_gp)]
    
    def _convert_to_gausspoints(self, data, nb_gp=None):         
        """
        Convert an array of values related to a specific mesh (Nodal values, Element Values or Points of Gauss values) to the gauss points
        mesh: the considered Mesh object
        data: array containing the values (nodal or element value)
        The shape of the array is tested.
        """               
        if nb_gp is None: nb_gp = self.__nb_gp            
        dataType = DetermineDataType(data, self.__Mesh, nb_gp)       

        if dataType == 'Node': 
            return self._get_node2gausspoint_mat(nb_gp) * data
        if dataType == 'Element':
            if len(np.shape(data)) == 1: return np.tile(data.copy(),nb_gp)
            else: return np.tile(data.copy(),[nb_gp,1])            
        return data #in case data contains already PG values
                
    def GetElementResult(self, operator, U):
        """
        Return some element results based on the finite element discretization of 
        a differential operator on a mesh being given the dof results and the type of elements.
        
        Parameters
        ----------
        mesh: string or Mesh 
            If mesh is a string, it should be a meshname.
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
                
        res = self.GetGaussPointResult(operator, U)
        NumberOfGaussPoint = res.shape[0]//self.__Mesh.n_elements
        return np.reshape(res, (NumberOfGaussPoint,-1)).sum(0) / NumberOfGaussPoint

    def GetGaussPointResult(self, operator, U, nb_gp = None):
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
        
        #TODO : can be accelerated by avoiding RowBlocMatrix (need to be checked) -> For each elementary 
        # 1 - at the very begining, compute Uloc = MatrixChangeOfBasis * U 
        # 2 - reshape Uloc to separate each var Uloc = Uloc.reshape(var, -1)
        # 3 - in the loop : res += coef_PG * (Assembly._get_elementary_operator(mesh, operator.op[ii], elementType, nb_gp) , nvar, var, coef) * Uloc[var]
        
        res = 0
        nvar = self.space.nvar
        
        mesh = self.__Mesh 
        elementType = self.__elmType
        if nb_gp is None: nb_gp = self.__nb_gp
        
        MatrixChangeOfBasis = self.get_change_of_basis_mat()
        associatedVariables = self._get_associated_variables()    
        
        for ii in range(len(operator.op)):
            var = [operator.op[ii].u] ; coef = [1] 
            
            if var[0] in associatedVariables:
                var.extend(associatedVariables[var[0]][0])
                coef.extend(associatedVariables[var[0]][1])     
    
            assert operator.op_vir[ii]==1, "Operator virtual are only required to build FE operators, but not to get element results"

            if isinstance(operator.coef[ii], Number): coef_PG = operator.coef[ii]                 
            else: coef_PG = Assembly._convert_to_gausspoints(mesh, operator.coef[ii][:], elementType, nb_gp)

            res += coef_PG * (RowBlocMatrix(self._get_elementary_operator(operator.op[ii], nb_gp) , nvar, var, coef) * MatrixChangeOfBasis * U)
        
        return res
        

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
        
        GaussianPointToNodeMatrix = self._get_gausspoint2node_mat()
        res = self.GetGaussPointResult(operator, U)
        return GaussianPointToNodeMatrix * res        
                
    def ConvertData(self, data, convertFrom=None, convertTo='GaussPoint'):
        
        if isinstance(data, Number): return data
        
        nb_gp = self.__nb_gp        
        
        if isinstance(data, (listStrainTensor, listStressTensor)):        
            try:
                return type(data)(self.ConvertData(data.asarray().T, convertFrom, convertTo).T)
            except:
                NotImplemented
        
        if convertFrom is None: convertFrom = DetermineDataType(data, self.__Mesh, nb_gp)
            
        assert (convertFrom in ['Node','GaussPoint','Element']) and (convertTo in ['Node','GaussPoint','Element']), "only possible to convert 'Node', 'Element' and 'GaussPoint' values"
        
        if convertFrom == convertTo: return data       
        if convertFrom == 'Node': 
            data = self._get_node2gausspoint_mat() * data
        elif convertFrom == 'Element':             
            if len(np.shape(data)) == 1: data = np.tile(data.copy(),nb_gp)
            else: data = np.tile(data.copy(),[nb_gp,1])
            
        # from here data should be defined at 'PG'
        if convertTo == 'Node': 
            return self._get_gausspoint2node_mat() * data 
        elif convertTo == 'Element': 
            return np.sum(np.split(data, nb_gp),axis=0) / nb_gp
        else: return data 
        
            
    def IntegrateField(self, Field, TypeField = 'GaussPoint'):
        assert TypeField in ['Node','GaussPoint','Element'], "TypeField should be 'Node', 'Element' or 'GaussPoint' values"
        Field = self.ConvertData(Field, TypeField, 'GaussPoint')
        return sum(self._get_gaussian_quadrature_mat()@Field)

    # def GetStressTensor(self, U, constitutiveLaw, Type="Nodal"):
    #     """
    #     Not a static method.
    #     Return the Stress Tensor of an assembly using the Voigt notation as a python list. 
    #     The total displacement field and a ConstitutiveLaw have to be given.
        
    #     Can only be used for linear constitutive law. 
    #     For non linear ones, use the GetStress method of the ConstitutiveLaw object.

    #     Options : 
    #     - Type :"Nodal", "Element" or "GaussPoint" integration (default : "Nodal")

    #     See GetNodeResult, GetElementResult and GetGaussPointResult.

    #     example : 
    #     S = SpecificAssembly.GetStressTensor(Problem.Problem.GetDoFSolution('all'), SpecificConstitutiveLaw)
    #     """
    #     if isinstance(constitutiveLaw, str):
    #         constitutiveLaw = ConstitutiveLaw.get_all()[constitutiveLaw]

    #     if Type == "Nodal":
    #         return listStressTensor([self.GetNodeResult(e, U) if e!=0 else np.zeros(self.__Mesh.n_nodes) for e in constitutiveLaw.GetStressOperator()])
        
    #     elif Type == "Element":
    #         return listStressTensor([self.GetElementResult(e, U) if e!=0 else np.zeros(self.__Mesh.n_elements) for e in constitutiveLaw.GetStressOperator()])
        
    #     elif Type == "GaussPoint":
    #         NumberOfGaussPointValues = self.__Mesh.n_elements * self.__nb_gp #Assembly._saved_elementary_operator[(self.__Mesh, self.__elmType, self.__nb_gp)][0].shape[0]
    #         return listStressTensor([self.GetGaussPointResult(e, U) if e!=0 else np.zeros(NumberOfGaussPointValues) for e in constitutiveLaw.GetStressOperator()])
        
    #     else:
    #         assert 0, "Wrong argument for Type: use 'Nodal', 'Element', or 'GaussPoint'"
        
    
    def set_disp(self, disp):
        if disp is 0: self.current = self
        else:
            new_crd = self.GetMesh().nodes + disp.T
            if self.current == self:
                #initialize a new assembly
                new_mesh = copy(self.GetMesh())
                new_mesh.nodes = new_crd
                new_assembly = copy(self)                                                    
                new_assembly.SetMesh(new_mesh)   
                self.current = new_assembly
            else: 
                self.current.GetMesh().nodes = new_crd
                
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
            if hasattr(self._weakform, 'nlgeom'): nlgeom = self._weakform.nlgeom
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
        Return the Gradient Tensor of a vector (generally displacement given by Problem.GetDofSolution('all')
        as a list of list of numpy array
        The total displacement field U has to be given as a flatten numpy array
        see GetNodeResults and GetElementResults

        Options : 
        - Type :"Nodal", "Element" or "GaussPoint" integration (default : "Nodal")
        """        
        grad_operator = self.space.op_grad_u()        

        if Type == "Nodal":
            return [ [self.GetNodeResult(op, U) if op != 0 else np.zeros(self.__Mesh.n_nodes) for op in line_op] for line_op in grad_operator]
            
        elif Type == "Element":
            return [ [self.GetElementResult(op, U) if op!=0 else np.zeros(self.__Mesh.n_elements) for op in line_op] for line_op in grad_operator]        
        
        elif Type == "GaussPoint":
            NumberOfGaussPointValues = self.__nb_gp * self.__Mesh.n_elements #Assembly._saved_gaussian_quadrature_mat[(self.__Mesh, self.__nb_gp)].shape[0]
            return [ [self.GetGaussPointResult(op, U) if op!=0 else np.zeros(NumberOfGaussPointValues) for op in line_op] for line_op in grad_operator]        
        else:
            assert 0, "Wrong argument for Type: use 'Nodal', 'Element', or 'GaussPoint'"

    def GetExternalForces(self, U, nvar=None):
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
        if nvar is None: nvar = self.space.nvar
        return np.reshape(self.GetMatrix() * U - self.GetVector(), (nvar,-1)).T                        
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
##        operator = self._weakform.GetDifferentialOperator(self.__Mesh)
#        operator = self._weakform.GetGeneralizedStress()
#        res = [self.GetElementResult(operator[i], U) for i in range(5)]
#        return res
#        

                 
        
#        res = np.reshape(res,(6,-1)).T
#        Nel = mesh.n_elements
#        res = (res[Nel:,:]-res[0:Nel:,:])/2
#        res = res[:, [self.space.variable_rank('DispX'), self.space.variable_rank('DispY'), self.space.variable_rank('DispZ'), \
#                              self.space.variable_rank('ThetaX'), self.space.variable_rank('ThetaY'), self.space.variable_rank('ThetaZ')]]         
#        
#        if CoordinateSystem == 'local': return res
#        elif CoordinateSystem == 'global': 
#            #require a transformation between local and global coordinates on element
#            #classical MatrixChangeOfBasis transform only toward nodal values
#            elmRef = eval(self.__Mesh.elm_type)(1, mesh=mesh)#one pg  with the geometrical element
#            vec = [0,1,2] ; dim = 3
#       
#            #Data to build MatrixChangeOfBasisElement with coo sparse format
#            crd = mesh.nodes ; elm = mesh.elements
#            rowMCB = np.empty((Nel, 1, dim,dim))
#            colMCB = np.empty((Nel, 1, dim,dim))            
#            rowMCB[:] = np.arange(Nel).reshape(-1,1,1,1) + np.array(vec).reshape(1,1,-1,1)*Nel # [[id_el + var*Nel] for var in vec]    
#            colMCB[:] = np.arange(Nel).reshape(-1,1,1,1) + np.array(vec).reshape(1,1,1,-1)*Nel # [id_el+Nel*var for var in vec]
#            dataMCB = elmRef.GetLocalFrame(crd[elm], elmRef.xi_pg, mesh.local_frame) #array of shape (Nel, nb_gp=1, nb of vectors in basis = dim, dim)                        
#
#            MatrixChangeOfBasisElement = sparse.coo_matrix((np.reshape(dataMCB,-1),(np.reshape(rowMCB,-1),np.reshape(colMCB,-1))), shape=(dim*Nel, dim*Nel)).tocsr()
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
        
        operator = self._weakform.GetDifferentialOperator(self.__Mesh)
        mesh = self.__Mesh
        nvar = self.space.nvar
        dim = self.space.ndim
        MatrixChangeOfBasis = self.get_change_of_basis_mat()

        MatGaussianQuadrature = self._get_gaussian_quadrature_mat()
        associatedVariables = self._get_associated_variables()
        
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

            Mat    =  RowBlocMatrix(self._get_elementary_operator(operator.op[ii]), nvar, var, coef)        
            Matvir =  RowBlocMatrix(self._get_elementary_operator(operator.op_vir[ii]), nvar, var_vir, coef_vir).T 

            if isinstance(operator.coef[ii], Number): #and self.op_vir[ii] != 1: 
                res = res + operator.coef[ii]*Matvir * MatGaussianQuadrature * Mat * MatrixChangeOfBasis * U   
            else:
                return NotImplemented                      
        
        res = np.reshape(res,(nvar,-1)).T
        
        Nel = mesh.n_elements
        res = (res[Nel:2*Nel,:]-res[0:Nel:,:])/2
        
        # if dim == 3:
        #     res = res[:, [self.space.variable_rank('DispX'), self.space.variable_rank('DispY'), self.space.variable_rank('DispZ'), \
        #                   self.space.variable_rank('RotX'), self.space.variable_rank('RotY'), self.space.variable_rank('RotZ')]]   
        # else: 
        #     res = res[:, [self.space.variable_rank('DispX'), self.space.variable_rank('DispY'), self.space.variable_rank('RotZ')]]   
        
        if CoordinateSystem == 'local': return res
        elif CoordinateSystem == 'global': 
            #require a transformation between local and global coordinates on element
            #classical MatrixChangeOfBasis transform only toward nodal values
            elmRef = eval(self.__Mesh.elm_type)(1, mesh=mesh)#one pg  with the geometrical element            
            if dim == 3: vec = [0,1,2] 
            else: vec = [0,1]
       
            #Data to build MatrixChangeOfBasisElement with coo sparse format
            crd = mesh.nodes ; elm = mesh.elements
            rowMCB = np.empty((Nel, 1, dim,dim))
            colMCB = np.empty((Nel, 1, dim,dim))            
            rowMCB[:] = np.arange(Nel).reshape(-1,1,1,1) + np.array(vec).reshape(1,1,-1,1)*Nel # [[id_el + var*Nel] for var in vec]    
            colMCB[:] = np.arange(Nel).reshape(-1,1,1,1) + np.array(vec).reshape(1,1,1,-1)*Nel # [id_el+Nel*var for var in vec]
            dataMCB = elmRef.GetLocalFrame(crd[elm], elmRef.xi_pg, mesh.local_frame) #array of shape (Nel, nb_gp=1, nb of vectors in basis = dim, dim)                        

            MatrixChangeOfBasisElement = sparse.coo_matrix((np.reshape(dataMCB,-1),(np.reshape(rowMCB,-1),np.reshape(colMCB,-1))), shape=(dim*Nel, dim*Nel)).tocsr()
            
            F = np.reshape( MatrixChangeOfBasisElement.T * np.reshape(res[:,0:dim].T, -1)  ,  (dim,-1) ).T
            if dim == 3: 
                C = np.reshape( MatrixChangeOfBasisElement.T * np.reshape(res[:,3:6].T, -1)  ,  (3,-1) ).T
            else: C = res[:,2]
            
            return np.c_[F,C] #np.hstack((F,C))            


    def copy(self, new_id = ""):
        """
        Return a raw deep copy of the assembly without keeping current state (internal variable).

        Parameters
        ----------
        new_id : TYPE, optional
            The name of the created constitutive law. The default is "".

        Returns
        -------
        The copy of the assembly
        """
        new_wf = self._weakform.copy()
        
        return Assembly(new_wf, self.__Mesh, self.__elmType, new_id)
    
    @property
    def elm_type(self):
        return self.__elmType
    
def delete_memory():
    Assembly.delete_memory()
    

# def ConvertData(data, mesh, convertFrom=None, convertTo='GaussPoint', elmType=None, nb_gp =None):        
#     if isinstance(data, Number): return data
    
#     if isinstance(mesh, str): mesh = Mesh.get_all()[mesh]
#     if elmType is None: elmType = mesh.elm_type
#     if nb_gp is None: nb_gp = GetDefaultNbPG(elmType, mesh)
    
#     if isinstance(data, (listStrainTensor, listStressTensor)):        
#         try:
#             return type(data)(ConvertData(data.asarray().T, mesh, convertFrom, convertTo, elmType, nb_gp).T)
#         except:
#             NotImplemented
    
#     if convertFrom is None: convertFrom = DetermineDataType(data, mesh, nb_gp)
        
#     assert (convertFrom in ['Node','GaussPoint','Element']) and (convertTo in ['Node','GaussPoint','Element']), "only possible to convert 'Node', 'Element' and 'GaussPoint' values"
    
#     if convertFrom == convertTo: return data       
#     if convertFrom == 'Node': 
#         data = Assembly._Assembly_get_node2gausspoint_mat(mesh, elmType, nb_gp) * data
#         convertFrom = 'GaussPoint'
#     elif convertFrom == 'Element':             
#         if len(np.shape(data)) == 1: data = np.tile(data.copy(),nb_gp)
#         else: data = np.tile(data.copy(),[nb_gp,1])
#         convertFrom = 'GaussPoint'
        
#     # from here convertFrom should be 'PG'
#     if convertTo == 'Node': 
#         return Assembly._Assembly_get_gausspoint2node_mat(mesh, elmType, nb_gp) * data 
#     elif convertTo == 'Element': 
#         return np.sum(np.split(data, nb_gp),axis=0) / nb_gp
#     else: return data 

def DetermineDataType(data, mesh, nb_gp):               
        if isinstance(mesh, str): mesh = Mesh.get_all()[mesh]
        if nb_gp is None: nb_gp = GetDefaultNbPG(elmType, mesh)
 
        test = 0
        if len(data) == mesh.n_nodes: 
            dataType = 'Node' #fonction définie aux noeuds   
            test+=1               
        if len(data) == mesh.n_elements: 
            dataType = 'Element' #fonction définie aux éléments
            test += 1
        if len(data) == nb_gp*mesh.n_elements:
            dataType = 'GaussPoint'
            test += 1
        assert test, "Error: data doesn't match with the number of nodes, number of elements or number of gauss points."
        if test>1: "Warning: kind of data is confusing. " + dataType +" values choosen."
        return dataType        