from fedoo.libAssembly.AssemblyBase import AssemblyBase
from fedoo.libAssembly.Assembly import Assembly as AssemblyFEM
from fedoo.libAssembly.Assembly import RowBlocMatrix
from fedoo.libUtil.Variable import Variable
from fedoo.libUtil.StrainOperator import GetStrainOperator
from fedoo.libMesh.Mesh import Mesh as MeshFEM
from fedoo.libElement import *
from fedoo.libWeakForm.WeakForm import WeakForm
from fedoo.libPGD.SeparatedOperator import SeparatedOperator
from fedoo.libPGD.SeparatedArray import SeparatedArray

from scipy import sparse
import numpy as np
from numbers import Number 
               
class AssemblyPGD(AssemblyFEM):
            
      
    def __init__(self,weakForm, mesh="", ID=""):
        #mesh should be of type PGD.Mesh

        if isinstance(weakForm, str):            
            weakForm = WeakForm.GetAll()[weakForm]

        if isinstance(mesh, str):
            mesh = MeshFEM.GetAll()[mesh]
                                
        AssemblyBase.__init__(self, ID)        

        self.__weakForm = weakForm
        self.__Mesh = mesh #should be a MeshPGD object 
        self.__listElementType = [m.GetElementShape() for m in mesh.GetListMesh()] #ElementType for every subMesh defined in self.__Mesh
        self.__listNumberOfGaussPoints = [GetDefaultNbPG(eltype) for eltype in self.__listElementType] #Nb_pg for every subMesh defined in self.__Mesh (default value)

    def ComputeGlobalMatrix(self):
        mesh = self.__Mesh
        dim = mesh.GetDimension()

        wf = self.__weakForm.GetDifferentialOperator(mesh)  
        nvar = [mesh._GetSpecificNumberOfVariables(idmesh) for idmesh in range(dim)]       
        
        AA = []  
        BB = 0
        
        for ii in range(len(wf.op)):     

            if wf.op[ii] == 1: #only virtual operator -> compute a separated array
                BBadd = []
            else:#virtual and real operators -> compute a separated operator 
                if isinstance(wf.coef[ii], SeparatedArray):
                    nb_term_coef = wf.coef[ii].nbTerm()
                    AA += [[] for term in range(nb_term_coef)]
                else: AA.append([]) 
                            
            for dd, subMesh in enumerate(mesh.GetListMesh()):

                elmType= self.__listElementType[dd] 
                nb_pg = self.__listNumberOfGaussPoints[dd]                
                MatGaussianQuadrature = AssemblyPGD._Assembly__GetGaussianQuadratureMatrix(subMesh, elmType, nb_pg)   
                MatrixChangeOfBasis = AssemblyPGD._Assembly__GetChangeOfBasisMatrix(subMesh)                             
                
                coef_vir = [1]                
                var_vir = [mesh._GetSpecificVariableRank (dd, wf.op_vir[ii].u)] #list in case there is an angular variable
                if 'X' in subMesh.GetCoordinateID(): #test if the subMesh is related to the spatial coordinates (Variable derivative are only for spatial derivative in beam or shell models)                        
                    if not(Variable.GetDerivative(wf.op_vir[ii].u) is None): 
                        var_vir.append(mesh._GetSpecificVariableRank (dd, Variable.GetDerivative(wf.op_vir[ii].u)[0]) )
                        coef_vir.append(Variable.GetDerivative(wf.op_vir[ii].u)[1])
                Matvir = (RowBlocMatrix(AssemblyPGD._Assembly__GetElementaryOp(subMesh, wf.op_vir[ii], elmType, nb_pg), nvar[dd], var_vir, coef_vir ) * MatrixChangeOfBasis).T               

                if wf.op[ii] == 1: #only virtual operator -> compute a separated array                                         
                    if isinstance(wf.coef[ii], (Number, np.floating)): #and self.op_vir[ii] != 1: 
                        if dd == 0: BBadd.append( wf.coef[ii]*Matvir * MatGaussianQuadrature.data.reshape(-1,1) )
                        else: BBadd.append( Matvir * MatGaussianQuadrature.data.reshape(-1,1) )                        
                    elif isinstance(wf.coef[ii], SeparatedArray):
                        coef_PG = AssemblyPGD._Assembly__ConvertToGaussPoints(subMesh, wf.coef[ii].data[dd], elmType, nb_pg)                      
                        BBadd.append( Matvir * (MatGaussianQuadrature.data.reshape(-1,1) * coef_PG))                                              
                    
                else: #virtual and real operators -> compute a separated operator 
                    coef = [1]
                    var = [mesh._GetSpecificVariableRank (dd, wf.op[ii].u)] #list in case there is an angular variable                
                    if 'X' in subMesh.GetCoordinateID(): #test if the subMesh is related to the spatial coordinates (Variable derivative are only for spatial derivative in beam or shell models)
                        if not(Variable.GetDerivative(wf.op[ii].u) is None):     
                            var.append(mesh._GetSpecificVariableRank (dd, Variable.GetDerivative(wf.op[ii].u)[0]) )
                            coef.append(Variable.GetDerivative(wf.op[ii].u)[1])                                                                             
                    Mat    =  RowBlocMatrix(AssemblyPGD._Assembly__GetElementaryOp(subMesh, wf.op[ii], elmType, nb_pg), nvar[dd], var, coef)         * MatrixChangeOfBasis 

                                                                                                             
                    if isinstance(wf.coef[ii], (Number, np.floating)): #and self.op_vir[ii] != 1: 
                        if dd == 0: AA[-1].append( wf.coef[ii]*Matvir * MatGaussianQuadrature * Mat )
                        else: AA[-1].append( Matvir * MatGaussianQuadrature * Mat )
                    elif isinstance(wf.coef[ii], SeparatedArray):
                        coef_PG = AssemblyPGD._Assembly__ConvertToGaussPoints(subMesh, wf.coef[ii].data[dd], elmType, nb_pg)
                        
                        for kk in range(nb_term_coef):
                            #CoefMatrix is a diag matrix that includes the gaussian quadrature coefficients and the value of wf.coef at gauss points                                
                            CoefMatrix = sparse.csr_matrix( (MatGaussianQuadrature.data*coef_PG[:,kk], MatGaussianQuadrature.indices, MatGaussianQuadrature.indptr), shape = MatGaussianQuadrature.shape)                        
                            AA[-nb_term_coef+kk].append( Matvir * CoefMatrix * Mat)                    
        
            if wf.op[ii] == 1:
                BB = BB - SeparatedArray(BBadd)
        
        if AA == []: self.SetMatrix(0)
        else: self.SetMatrix(SeparatedOperator(AA)) 
        self.SetVector(BB) 

    def SetMesh(self, mesh):
        self.__Mesh = mesh

    def GetMesh(self):
        return self.__Mesh
    
    def SetElementType(self, listElementType, listSubMesh = None):
        """
        Define the Type of Element used for the finite element assembly of each subMesh
        Example of available element type: 'lin2', 'beam', 'tri6', ...
        
        PGD.Assembly.SetElementType([ElementType_1,...,ElementType_n ])
            * ElementType_i is a list of ElementType cooresponding to the ith subMesh 
              (as defined in the constructor of the PGD.Mesh object related to the Assembly)            
            
        PGD.Assembly.SetElementType([ElementType_1,...,ElementType_n ], [subMesh_1,...,subMesh_n] )
            * ElementType_i is a list of ElementType cooresponding to the mesh indicated in subMesh_i
            * subMesh_i can be either a mesh ID (str object) or a Mesh object
            * If a subMesh is not included in listSubMesh, the ElementType for assembly is not modified (based on the geometrical element shape by default)
        """ 
        
        if listSubMesh is None:
            if len(listElementType) != len(self.__Mesh.GetListMesh()):
                assert 0, "The lenght of the Element Type List must be equal to the number of submeshes"
            self.__listElementType = [ElementType for ElementType in listElementType]
            self.__listNumberOfGaussPoints = [GetDefaultNbPG(self.__listElementType[dd], self.__Mesh.GetListMesh()[dd]) for dd in range(len(self.__listElementType))] #Nb_pg for every subMesh defined in self.__Mesh (default value)            
        else:
            for i,m in enumerate(listSubMesh):                
                if isinstance(m, str): m = MeshFEM.GetAll()[m]
                dd = self.__Mesh.GetListMesh().index(m)
                self.__listElementType[dd] = listElementType[i]
                self.__listNumberOfGaussPoints[dd] = GetDefaultNbPG(listElementType[i], m)
                
    def SetNumberOfGaussPoints(self, listNumberOfGaussPoints, listSubMesh = None):
        """
        Define the number of Gauss Points used for the finite element assembly of each subMesh
        The specified number of gauss points should be compatible with the elements defined by PGD.Assembly.SetElementType
        If NumberOfGaussPoints is set to None a default value related to the specified element is used 
        
        PGD.Assembly.NumberOfGaussPoints([NumberOfGaussPoints_1,...,NumberOfGaussPoints_n ])
            * NumberOfGaussPoints_i is a list of NumberOfGaussPoints cooresponding to the ith subMesh 
              (as defined in the constructor of the PGD.Mesh object related to the Assembly)            
            
        PGD.Assembly.NumberOfGaussPoints([NumberOfGaussPoints_1,...,NumberOfGaussPoints_n ], [subMesh_1,...,subMesh_n] )
            * NumberOfGaussPoints_i is a list of NumberOfGaussPoints cooresponding to the mesh indicated in subMesh_i
            * subMesh_i can be either a mesh ID (str object) or a Mesh object
            * If a subMesh is not included in listSubMesh, the NumberOfGaussPoints for assembly is not modified        
        """ 
        
        if listSubMesh is None:
            if len(listNumberOfGaussPoints) != len(self.__Mesh.GetListMesh()):
                assert 0, "The lenght of the Element Type List must be equal to the number of submeshes"
            self.__listNumberOfGaussPoints = listNumberOfGaussPoints
        else:
            for i,m in enumerate(listSubMesh):                
                if isinstance(m, str): m = Mesh.GetAll()[m]
                self.__listNumberOfGaussPoints[self.__Mesh.GetListMesh().index(m)] = listNumberOfGaussPoints[i]    

    @staticmethod
    def __GetResultGaussPoints(mesh, operator, U, list_elementType, list_nb_pg=None):  #return the results at GaussPoints      
        res = 0
        nvar = [mesh._GetSpecificNumberOfVariables(idmesh) for idmesh in range(mesh.GetDimension())]  
        if  list_nb_pg is None: 
            list_nb_pg = [GetDefaultNbPG(self.__listElementType[dd], self.__Mesh.GetListMesh()[dd]) for dd in range(len(self.__listElementType))]
            
        
        for ii in range(len(operator.op)):
            if isinstance(operator.coef[ii], Number): coef_PG = operator.coef[ii]
            else: coef_PG = []
            res_add =  []
                
            for dd, subMesh in enumerate(mesh.GetListMesh()):                                
                var = [mesh._GetSpecificVariableRank (dd, operator.op[ii].u)]
                coef = [1]
                if 'X' in subMesh.GetCoordinateID(): #test if the subMesh is related to the spatial coordinates
                    if not(Variable.GetDerivative(operator.op[ii].u) is None): 
                        var.append(mesh._GetSpecificVariableRank (dd, Variable.GetDerivative(operator.op[ii].u)[0]) )
                        coef.append(Variable.GetDerivative(operator.op[ii].u)[1])
                assert operator.op_vir[ii]==1, "Operator virtual are only required to build FE operators, but not to get element results"
                
                if isinstance(coef_PG, list):
                    coef_PG.append(AssemblyPGD._Assembly__ConvertToGaussPoints(subMesh, operator.coef[ii].data[dd], list_elementType[dd], list_nb_pg[dd]))
                
                MatrixChangeOfBasis = AssemblyPGD._Assembly__GetChangeOfBasisMatrix(subMesh, list_elementType[dd], list_nb_pg[dd])                                                           
                res_add.append(RowBlocMatrix(AssemblyPGD._Assembly__GetElementaryOp(subMesh, operator.op[ii], list_elementType[dd], list_nb_pg[dd]) , nvar[dd], var, coef) * MatrixChangeOfBasis * U.data[dd])
            
            if isinstance(coef_PG, list): coef_PG = SeparatedArray(Coef_PG)
            res = res + coef_PG*SeparatedArray(res_add)
                                        
        return res


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

        list_nb_elm = self.__Mesh.GetNumberOfElements()
        res = AssemblyPGD.__GetResultGaussPoints(self.__Mesh, operator, U, self.__listElementType, self.__listNumberOfGaussPoints)
        NumberOfGaussPoint = [res.data[dd].shape[0]//list_nb_elm[dd] for dd in range(len(res))]
                
        return SeparatedArray([np.reshape(res.data[dd], (NumberOfGaussPoint[dd],list_nb_elm[dd],-1)).sum(0) / NumberOfGaussPoint[dd] for dd in range(len(res))])

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
        GaussianPointToNodeMatrix = SeparatedOperator([[AssemblyPGD._Assembly__GetGaussianPointToNodeMatrix(subMesh, self.__listElementType[dd], self.__listNumberOfGaussPoints[dd]) for dd, subMesh in enumerate(self.__Mesh.GetListMesh())]])
        res = AssemblyPGD.__GetResultGaussPoints(self.__Mesh, operator, U, self.__listElementType, self.__listNumberOfGaussPoints)
        return GaussianPointToNodeMatrix * res 


    def GetStressTensor(self, U, constitutiveLaw, IntegrationType="Nodal"):
        """
        Not a static method.
        Return the Stress Tensor of an assembly using the Voigt notation as a python list. 
        The total displacement field and a ConstitutiveLaw have to be given.
        see GetNodeResults and GetElementResults.

        Options : 
        - IntegrationType :"Nodal" or "Element" integration (default : "Nodal")

        example : 
        S = SpecificAssembly.GetStressTensor(Problem.Problem.GetDoFSolution('all'), SpecificConstitutiveLaw)
        """
        if isinstance(constitutiveLaw, str):
            constitutiveLaw = ConstitutiveLaw.GetAll()[constitutiveLaw]

        if IntegrationType == "Nodal":            
            return [self.GetNodeResult(e, U) if e!=0 else Separatedzeros(self.__Mesh.GetNumberOfNodes()) for e in constitutiveLaw.GetStress()]
        
        elif IntegrationType == "Element":
            return [self.GetElementResult(e, U) if e!=0 else Separatedzeros(self.__Mesh.GetNumberOfElements()) for e in constitutiveLaw.GetStress()]
        
        else:
            assert 0, "Wrong argument for IntegrationType"

    def GetStrainTensor(self, U, IntegrationType="Nodal"):
        """
        Not a static method.
        Return the Strain Tensor of an assembly using the Voigt notation as a python list. 
        The total displacement field and a ConstitutiveLaw have to be given.
        see GetNodeResults and GetElementResults.

        Options : 
        - IntegrationType :"Nodal" or "Element" integration (default : "Nodal")

        example : 
        S = SpecificAssembly.GetStressTensor(Problem.Problem.GetDoFSolution('all'), SpecificConstitutiveLaw)
        """

        if IntegrationType == "Nodal":
            return [self.GetNodeResult(e, U) if e!=0 else Separatedzeros(self.__Mesh.GetNumberOfNodes()) for e in GetStrainOperator()[0]]
        
        elif IntegrationType == "Element":
            return [self.GetElementResult(e, U) if e!=0 else Separatedzeros(self.__Mesh.GetNumberOfElements()) for e in GetStrainOperator()[0]]
        
        else:
            assert 0, "Wrong argument for IntegrationType"



    def GetExternalForces(self, U, NumberOfVariable=None):
        """
        Not a static method.
        Return the nodal Forces and moments in global coordinates related to a specific assembly considering the DOF solution given in U
        The resulting forces are the sum of :
            - External forces (associated to Neumann boundary conditions)
            - Nodal reaction (associated to Dirichelet boundary conditions)
            - Inertia forces 
        
        Return a list of separated array [Fx, Fy, Fz, Mx, My, Mz].   
                    
        example : 
        S = SpecificAssembly.GetNodalForces(PGD.Problem.GetDoFSolution('all'))
        """

        ExtForce = self.GetMatrix() * U
        if NumberOfVariable==None:
            return [ExtForce.GetVariable(var, self.__Mesh) for var in range(Variable.GetNumberOfVariable())]
        else:
            return [ExtForce.GetVariable(var, self.__Mesh) for var in range(NumberOfVariable)]    
    
    
    @staticmethod        
    def Create(weakForm, mesh="", ID=""):        
        return AssemblyPGD(weakForm, mesh, ID)