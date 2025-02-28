from fedoo.core.base import AssemblyBase
from fedoo.core.assembly import Assembly as AssemblyFEM
from fedoo.core._sparsematrix import RowBlocMatrix

# from fedoo.util.Variable import Variable
# from fedoo.util.StrainOperator import GetStrainOperator
from fedoo.util.voigt_tensors import StrainTensorList
from fedoo.core.mesh import Mesh as MeshFEM
from fedoo.lib_elements.element_list import get_default_n_gp
from fedoo.core.weakform import WeakFormBase
from fedoo.pgd.SeparatedOperator import SeparatedOperator
from fedoo.pgd.SeparatedArray import SeparatedArray, SeparatedZeros
from fedoo.core.base import ConstitutiveLaw

from scipy import sparse
import numpy as np
from numbers import Number


class AssemblyPGD(AssemblyFEM):
    def __init__(self, weakForm, mesh="", name=""):
        # mesh should be of type PGD.Mesh

        if isinstance(weakForm, str):
            weakForm = WeakFormBase[weakForm]

        if isinstance(mesh, str):
            mesh = MeshFEM.get_all()[mesh]

        AssemblyBase.__init__(self, name, weakForm.space)

        self.weakform = weakForm
        self.mesh = mesh  # should be a MeshPGD object
        self.__listElementType = [
            m.elm_type for m in mesh.GetListMesh()
        ]  # ElementType for every subMesh defined in self.mesh
        self.__listNumberOfGaussPoints = [
            get_default_n_gp(eltype) for eltype in self.__listElementType
        ]  # Nb_pg for every subMesh defined in self.mesh (default value)
        # [get_default_n_gp(self.__listElementType[dd], self.mesh.GetListMesh()[dd]) for dd in range(len(self.__listElementType))]

        self.__listAssembly = [
            AssemblyFEM(
                weakForm,
                m,
                self.__listElementType[i],
                n_elm_gp=self.__listNumberOfGaussPoints[i],
            )
            for i, m in enumerate(mesh.GetListMesh())
        ]

        self.sv = {}
        """ Dictionary of state variables associated to the associated for the current problem."""
        self.sv_start = {}

        self._pb = None

    def assemble_global_mat(self, compute="all"):
        """
        Compute the global matrix and global vector using a separated representation
        if compute = 'all', compute the global matrix and vector
        if compute = 'matrix', compute only the matrix
        if compute = 'vector', compute only the vector
        """
        if compute == "none":
            return

        mesh = self.mesh
        dim = mesh.get_dimension()

        wf = self.weakform.get_weak_equation(self, self._pb)
        nvar = [
            mesh._GetSpecificNumberOfVariables(idmesh, self.space.nvar)
            for idmesh in range(dim)
        ]

        AA = []
        BB = 0

        for ii in range(len(wf.op)):
            if compute == "matrix" and wf.op[ii] == 1:
                continue
            if compute == "vector" and not wf.op[ii] == 1:
                continue

            if wf.op[ii] == 1:  # only virtual operator -> compute a separated array
                BBadd = []
            else:  # virtual and real operators -> compute a separated operator
                if isinstance(wf.coef[ii], SeparatedArray):
                    nb_term_coef = wf.coef[ii].nbTerm()
                    AA += [[] for term in range(nb_term_coef)]
                else:
                    AA.append([])

            for dd, subMesh in enumerate(mesh.GetListMesh()):
                elmType = self.__listElementType[dd]
                n_elm_gp = self.__listNumberOfGaussPoints[dd]
                MatGaussianQuadrature = self._get_gaussian_quadrature_mat(dd)
                MatrixChangeOfBasis = self.__listAssembly[dd].get_change_of_basis_mat()
                associatedVariables = self._get_associated_variables(dd)

                coef_vir = [1]
                var_vir = [
                    mesh._GetSpecificVariableRank(dd, wf.op_vir[ii].u)
                ]  # list in case there is an angular variable

                # if 'X' in subMesh.crd_name: #test if the subMesh is related to the spatial coordinates (Variable derivative are only for spatial derivative in beam or shell models)
                if wf.op_vir[ii].u in associatedVariables:
                    var_vir.extend(
                        [
                            mesh._GetSpecificVariableRank(dd, v)
                            for v in associatedVariables[wf.op_vir[ii].u][0]
                        ]
                    )
                    coef_vir.extend(associatedVariables[wf.op_vir[ii].u][1])
                # if not(Variable.get_Derivative(wf.op_vir[ii].u) is None):
                #     var_vir.append(mesh._GetSpecificVariableRank (dd, Variable.get_Derivative(wf.op_vir[ii].u)[0]) )
                #     coef_vir.append(Variable.get_Derivative(wf.op_vir[ii].u)[1])

                Matvir = (
                    RowBlocMatrix(
                        self.__listAssembly[dd]._get_elementary_operator(wf.op_vir[ii]),
                        nvar[dd],
                        var_vir,
                        coef_vir,
                    )
                    * MatrixChangeOfBasis
                ).T

                if wf.op[ii] == 1:  # only virtual operator -> compute a separated array
                    if isinstance(
                        wf.coef[ii], (Number, np.floating)
                    ):  # and self.op_vir[ii] != 1:
                        if dd == 0:
                            BBadd.append(
                                wf.coef[ii]
                                * Matvir
                                * MatGaussianQuadrature.data.reshape(-1, 1)
                            )
                        else:
                            BBadd.append(
                                Matvir * MatGaussianQuadrature.data.reshape(-1, 1)
                            )
                    elif isinstance(wf.coef[ii], SeparatedArray):
                        coef_PG = self.__listAssembly[dd]._convert_to_gausspoints(
                            wf.coef[ii].data[dd]
                        )
                        BBadd.append(
                            Matvir
                            * (MatGaussianQuadrature.data.reshape(-1, 1) * coef_PG)
                        )

                else:  # virtual and real operators -> compute a separated operator
                    coef = [1]
                    var = [
                        mesh._GetSpecificVariableRank(dd, wf.op[ii].u)
                    ]  # list in case there is an angular variable

                    # if 'X' in subMesh.crd_name: #test if the subMesh is related to the spatial coordinates (Variable derivative are only for spatial derivative in beam or shell models)
                    if wf.op[ii].u in associatedVariables:
                        var.extend(
                            [
                                mesh._GetSpecificVariableRank(dd, v)
                                for v in associatedVariables[wf.op[ii].u][0]
                            ]
                        )
                        coef.extend(associatedVariables[wf.op[ii].u][1])
                    # if not(Variable.get_Derivative(wf.op[ii].u) is None):
                    #     var.append(mesh._GetSpecificVariableRank (dd, Variable.get_Derivative(wf.op[ii].u)[0]) )
                    #     coef.append(Variable.get_Derivative(wf.op[ii].u)[1])

                    Mat = (
                        RowBlocMatrix(
                            self.__listAssembly[dd]._get_elementary_operator(wf.op[ii]),
                            nvar[dd],
                            var,
                            coef,
                        )
                        * MatrixChangeOfBasis
                    )

                    if isinstance(
                        wf.coef[ii], (Number, np.floating)
                    ):  # and self.op_vir[ii] != 1:
                        if dd == 0:
                            AA[-1].append(
                                wf.coef[ii] * Matvir * MatGaussianQuadrature * Mat
                            )
                        else:
                            AA[-1].append(Matvir * MatGaussianQuadrature * Mat)
                    elif isinstance(wf.coef[ii], SeparatedArray):
                        coef_PG = self.__listAssembly[dd]._convert_to_gausspoints(
                            wf.coef[ii].data[dd]
                        )

                        for kk in range(nb_term_coef):
                            # CoefMatrix is a diag matrix that includes the gaussian quadrature coefficients and the value of wf.coef at gauss points
                            CoefMatrix = sparse.csr_matrix(
                                (
                                    MatGaussianQuadrature.data * coef_PG[:, kk],
                                    MatGaussianQuadrature.indices,
                                    MatGaussianQuadrature.indptr,
                                ),
                                shape=MatGaussianQuadrature.shape,
                            )
                            AA[-nb_term_coef + kk].append(Matvir * CoefMatrix * Mat)

            if wf.op[ii] == 1:
                BB = BB - SeparatedArray(BBadd)

        if compute != "vector":
            if AA == []:
                self.global_matrix = 0
            else:
                self.global_matrix = SeparatedOperator(AA)
        if compute != "matrix":
            self.global_vector = BB

    def SetElementType(self, listElementType, listSubMesh=None):
        """
        Define the Type of Element used for the finite element assembly of each subMesh
        Example of available element type: 'lin2', 'beam', 'tri6', ...

        PGD.Assembly.SetElementType([ElementType_1,...,ElementType_n ])
            * ElementType_i is a list of ElementType cooresponding to the ith subMesh
              (as defined in the constructor of the PGD.Mesh object related to the Assembly)

        PGD.Assembly.SetElementType([ElementType_1,...,ElementType_n ], [subMesh_1,...,subMesh_n] )
            * ElementType_i is a list of ElementType cooresponding to the mesh indicated in subMesh_i
            * subMesh_i can be either a mesh name (str object) or a Mesh object
            * If a subMesh is not included in listSubMesh, the ElementType for assembly is not modified (based on the geometrical element shape by default)
        """

        if listSubMesh is None:
            if len(listElementType) != len(self.mesh.GetListMesh()):
                assert 0, "The lenght of the Element Type List must be equal to the number of submeshes"
            self.__listElementType = [ElementType for ElementType in listElementType]
            self.__listNumberOfGaussPoints = [
                get_default_n_gp(
                    self.__listElementType[dd], self.mesh.GetListMesh()[dd]
                )
                for dd in range(len(self.__listElementType))
            ]  # Nb_pg for every subMesh defined in self.mesh (default value)
        else:
            for i, m in enumerate(listSubMesh):
                if isinstance(m, str):
                    m = MeshFEM.get_all()[m]
                dd = self.mesh.GetListMesh().index(m)
                self.__listElementType[dd] = listElementType[i]
                self.__listNumberOfGaussPoints[dd] = get_default_n_gp(
                    listElementType[i], m
                )

        self.__listAssembly = [
            AssemblyFEM(
                self.weakform,
                m,
                self.__listElementType[i],
                n_elm_gp=self.__listNumberOfGaussPoints[i],
            )
            for i, m in enumerate(self.mesh.GetListMesh())
        ]

    def SetNumberOfGaussPoints(self, listNumberOfGaussPoints, listSubMesh=None):
        """
        Define the number of Gauss Points used for the finite element assembly of each subMesh
        The specified number of gauss points should be compatible with the elements defined by PGD.Assembly.SetElementType
        If NumberOfGaussPoints is set to None a default value related to the specified element is used

        PGD.Assembly.NumberOfGaussPoints([NumberOfGaussPoints_1,...,NumberOfGaussPoints_n ])
            * NumberOfGaussPoints_i is a list of NumberOfGaussPoints cooresponding to the ith subMesh
              (as defined in the constructor of the PGD.Mesh object related to the Assembly)

        PGD.Assembly.NumberOfGaussPoints([NumberOfGaussPoints_1,...,NumberOfGaussPoints_n ], [subMesh_1,...,subMesh_n] )
            * NumberOfGaussPoints_i is a list of NumberOfGaussPoints cooresponding to the mesh indicated in subMesh_i
            * subMesh_i can be either a mesh name (str object) or a Mesh object
            * If a subMesh is not included in listSubMesh, the NumberOfGaussPoints for assembly is not modified
        """

        if listSubMesh is None:
            if len(listNumberOfGaussPoints) != len(self.mesh.GetListMesh()):
                assert 0, "The lenght of the Element Type List must be equal to the number of submeshes"
            self.__listNumberOfGaussPoints = listNumberOfGaussPoints
        else:
            for i, m in enumerate(listSubMesh):
                if isinstance(m, str):
                    m = MeshFEM.get_all()[m]
                self.__listNumberOfGaussPoints[self.mesh.GetListMesh().index(m)] = (
                    listNumberOfGaussPoints[i]
                )

        self.__listAssembly = [
            AssemblyFEM(
                self.weakform,
                m,
                self.__listElementType[i],
                n_elm_gp=self.__listNumberOfGaussPoints[i],
            )
            for i, m in enumerate(self.mesh.GetListMesh())
        ]

    def _get_associated_variables(self, idmesh):
        return self.__listAssembly[idmesh]._get_associated_variables()

    def _get_gaussian_quadrature_mat(self, idmesh):
        return self.__listAssembly[idmesh]._get_gaussian_quadrature_mat()

    # def _test_if_local_csys(self, idmesh):
    #     return self.__listAssembly[idmesh]._test_if_local_csys()

    def get_element_results(self, operator, U):
        """
        Not a Static Method.

        Return some element results based on the finite element discretization of
        a differential operator on a mesh being given the dof results and the type of elements.

        Parameters
        ----------
        mesh: string or Mesh
            If mesh is a string, it should be a meshname.
            Define the mesh to get the results from

        operator: DiffOp
            Differential operator defining the required results

        U: numpy.ndarray
            Vector containing all the DoF solution

        Return: numpy.ndarray
            A Vector containing the values on each element.
            It is computed using an arithmetic mean of the values from gauss points
            The vector lenght is the number of element in the mesh
        """

        list_nb_elm = self.mesh.n_elements
        res = self.get_gp_results(operator, U)
        NumberOfGaussPoint = [
            res.data[dd].shape[0] // list_nb_elm[dd] for dd in range(len(res))
        ]

        return SeparatedArray(
            [
                np.reshape(
                    res.data[dd], (NumberOfGaussPoint[dd], list_nb_elm[dd], -1)
                ).sum(0)
                / NumberOfGaussPoint[dd]
                for dd in range(len(res))
            ]
        )

    def get_gp_results(self, operator, U):
        """
        Return some results at element Gauss points based on the finite element discretization of
        a differential operator on a mesh being given the dof results and the type of elements.

        Parameters
        ----------
        operator: DiffOp
            Differential operator defining the required results

        U: numpy.ndarray
            Vector containing all the DoF solution

        Return: numpy.ndarray
            A Vector containing the values on each point of gauss for each element.
            The vector lenght is the number of element time the number of Gauss points per element
        """

        mesh = self.mesh
        list_n_elm_gp = self.__listNumberOfGaussPoints
        res = 0
        nvar = [
            mesh._GetSpecificNumberOfVariables(idmesh, self.space.nvar)
            for idmesh in range(mesh.get_dimension())
        ]

        for ii in range(len(operator.op)):
            if isinstance(operator.coef[ii], Number):
                coef_PG = operator.coef[ii]
            else:
                coef_PG = []
            res_add = []

            for dd, subMesh in enumerate(mesh.GetListMesh()):
                var = [mesh._GetSpecificVariableRank(dd, operator.op[ii].u)]
                coef = [1]

                # if 'X' in subMesh.crd_name: #test if the subMesh is related to the spatial coordinates
                associatedVariables = self._get_associated_variables(dd)

                if operator.op[ii].u in associatedVariables:
                    var.extend(
                        [
                            mesh._GetSpecificVariableRank(dd, v)
                            for v in associatedVariables[operator.op[ii].u][0]
                        ]
                    )
                    coef.extend(associatedVariables[operator.op[ii].u][1])
                # if not(Variable.get_Derivative(operator.op[ii].u) is None):
                #     var.append(mesh._GetSpecificVariableRank (dd, Variable.get_Derivative(operator.op[ii].u)[0]) )
                #     coef.append(Variable.get_Derivative(operator.op[ii].u)[1])

                assert (
                    operator.op_vir[ii] == 1
                ), "Operator virtual are only required to build FE operators, but not to get element results"

                if isinstance(coef_PG, list):
                    coef_PG.append(
                        self.__listAssembly[dd]._convert_to_gausspoints(
                            operator.coef[ii].data[dd]
                        )
                    )

                MatrixChangeOfBasis = self.__listAssembly[dd].get_change_of_basis_mat()
                res_add.append(
                    RowBlocMatrix(
                        self.__listAssembly[dd]._get_elementary_operator(
                            operator.op[ii]
                        ),
                        nvar[dd],
                        var,
                        coef,
                    )
                    * MatrixChangeOfBasis
                    * U.data[dd]
                )

            if isinstance(coef_PG, list):
                coef_PG = SeparatedArray(coef_PG)
            res = res + coef_PG * SeparatedArray(res_add)

        return res

    def get_node_results(self, operator, U):
        """
        Not a Static Method.

        Return some node results based on the finite element discretization of
        a differential operator on a mesh being given the dof results and the type of elements.

        Parameters
        ----------
        operator: DiffOp
            Differential operator defining the required results

        U: numpy.ndarray
            Vector containing all the DoF solution

        Return: numpy.ndarray
            A Vector containing the values on each node.
            An interpolation is used to get the node values from the gauss point values on each element.
            After that, an arithmetic mean is used to compute a single node value from all adjacent elements.
            The vector lenght is the number of nodes in the mesh
        """
        GaussianPointToNodeMatrix = SeparatedOperator(
            [
                [
                    self.__listAssembly[dd].mesh._get_gausspoint2node_mat()
                    for dd in range(len(self.mesh.GetListMesh()))
                ]
            ]
        )
        res = self.get_gp_results(operator, U)
        return GaussianPointToNodeMatrix * res

    def GetStressTensor(self, U, constitutivelaw, IntegrationType="Node"):
        """
        Not a static method.
        Return the Stress Tensor of an assembly using the Voigt notation as a python list.
        The total displacement field and a ConstitutiveLaw have to be given.
        see get_node_resultss and get_element_resultss.

        Options :
        - IntegrationType :"Node" or "Element" integration (default : "Node")

        example :
        S = SpecificAssembly.GetStressTensor(Problem.Problem.get_dof_solution('all'), SpecificConstitutiveLaw)
        """
        if isinstance(constitutivelaw, str):
            constitutivelaw = ConstitutiveLaw.get_all()[constitutivelaw]

        if IntegrationType == "Node":
            return [
                self.get_node_results(e, U)
                if e != 0
                else SeparatedZeros(self.mesh.n_nodes)
                for e in constitutivelaw.get_stress()
            ]

        elif IntegrationType == "Element":
            return [
                self.get_element_results(e, U)
                if e != 0
                else SeparatedZeros(self.mesh.n_elements)
                for e in constitutivelaw.get_stress()
            ]

        else:
            assert 0, "Wrong argument for IntegrationType"

    def get_strain(self, U, IntegrationType="Node"):
        """
        Not a static method.
        Return the Strain Tensor of an assembly using the Voigt notation as a python list.
        The total displacement field and a ConstitutiveLaw have to be given.
        see get_node_resultss and get_element_resultss.

        Options :
        - IntegrationType :"Node" or "Element" integration (default : "Node")

        example :
        S = SpecificAssembly.GetStressTensor(Problem.Problem.get_dof_solution('all'), SpecificConstitutiveLaw)
        """

        if IntegrationType == "Node":
            return StrainTensorList(
                [
                    self.get_node_results(e, U)
                    if e != 0
                    else SeparatedZeros(self.mesh.n_nodes)
                    for e in self.space.op_strain()
                ]
            )

        elif IntegrationType == "Element":
            return StrainTensorList(
                [
                    self.get_element_results(e, U)
                    if e != 0
                    else SeparatedZeros(self.mesh.n_elements)
                    for e in self.space.op_strain()
                ]
            )

        else:
            assert 0, "Wrong argument for IntegrationType"

    def get_ext_forces(self, U, NumberOfVariable=None):
        """
        Not a static method.
        Return the nodal Forces and moments in global coordinates related to a specific assembly considering the DOF solution given in U
        The resulting forces are the sum of :
            - External forces (associated to Neumann boundary conditions)
            - Nodal reaction (associated to Dirichelet boundary conditions)
            - Inertia forces

        Return a list of separated array [Fx, Fy, Fz, Mx, My, Mz].

        example :
        S = SpecificAssembly.GetNodalForces(PGD.Problem.get_dof_solution('all'))
        """

        ExtForce = self.get_global_matrix() * U
        if NumberOfVariable is None:
            return [
                ExtForce.GetVariable(var, self.mesh) for var in range(self.space.nvar)
            ]
        else:
            return [
                ExtForce.GetVariable(var, self.mesh) for var in range(NumberOfVariable)
            ]

    @staticmethod
    def create(weakForm, mesh="", name=""):
        return AssemblyPGD(weakForm, mesh, name)
