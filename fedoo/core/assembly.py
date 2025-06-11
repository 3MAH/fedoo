from copy import copy

import numpy as np
from scipy import sparse

from fedoo.core._sparsematrix import RowBlocMatrix
from fedoo.core._sparsematrix import _BlocSparse as BlocSparse
from fedoo.core._sparsematrix import (
    _BlocSparseOld as BlocSparseOld,
)  # required for 'old' _assembly_method
from fedoo.core.assembly_sum import AssemblySum
from fedoo.core.base import AssemblyBase
from fedoo.core.mesh import Mesh
from fedoo.core.weakform import WeakFormBase, _AssemblyOptions, WeakFormSum
from fedoo.lib_elements.element_list import (
    get_default_n_gp,
    get_element,
    get_node_elm_coordinates,
)
from fedoo.util.voigt_tensors import StrainTensorList, StressTensorList


class Assembly(AssemblyBase):
    """
    Fedoo Assembly object.

    This class is one of the main object of fedoo that is dedicated to all that
    is related to global matrices assembly. Basically, an Assembly object is
    build upon a weakform (equation written using a weak formulation), a mesh,
    a type of element and a number of integration points (gauss points).

    Parameters
    ----------
    weakform: WeakForm instance
        weakform associated to the assembly
    mesh: Mesh instance
        domain over which the weakform should be integrated
    elm_type: str
        Type of the element used for the field interpolation. This element may
        be different that the one used for the geometrical interpolation
        defined in mesh.elm_type.
    name: str
        The name of the assembly
    n_elm_gp: number of gauss points per element for the numerical integration.
        To use with caution. By default, this value is set automatically for
        each element type. A non default number of integration points may be
        forced using this argument.

    Notes
    -----
    To launch the assembling, use the method "assemble_global_mat()"
    Then, the assembled global matrix and global vector are stored in the
    attributes "global_matrix" and "global_vector".
    """

    _saved_elementary_operators = {}
    _saved_change_of_basis_mat = {}
    # _saved_node2gausspoint_mat = {}
    # _saved_gausspoint2node_mat = {}
    _saved_associated_variables = {}  # dict containing all associated variables (rotational dof for C1 elements) for elm_type

    def __init__(self, weakform, mesh="", elm_type="", name="", **kargs):
        if isinstance(weakform, str):
            weakform = WeakFormBase.get_all()[weakform]

        if weakform.assembly_options is None:
            # should be a non compatible WeakFormSum object
            raise NameError(
                "Some Assembly associated to WeakFormSum object can only be created using the Create function"
            )

        if isinstance(mesh, str):
            mesh = Mesh.get_all()[mesh]
        if not type(mesh) == Mesh:
            if hasattr(mesh, "mesh_dict"):
                raise TypeError(
                    "Can't create an assembly based on a MultiMesh object. "
                    "For that purpose, create separated assemblies for each "
                    "element type and sum them together."
                )
            else:
                raise TypeError("mesh should refers to a fedoo.Mesh object")

        if isinstance(weakform, WeakFormBase):
            self.weakform = weakform
            AssemblyBase.__init__(self, name, weakform.space)
        else:  # weakform should be a ModelingSpace object
            assert hasattr(weakform, "list_variable") and hasattr(
                weakform, "list_coordinate"
            ), "WeakForm not understood"
            self.weakform = None
            AssemblyBase.__init__(self, name, space=weakform)

        # attributes to set assembly related to current (deformed) configuration
        # used for update lagrangian method.
        self.current = self

        self.meshChange = kargs.pop("MeshChange", False)
        self.mesh = mesh
        if elm_type == "":
            elm_type = weakform.assembly_options.get(
                "elm_type", mesh.elm_type, mesh.elm_type
            )  # change elm_type if it was specified in assembly_options
        self.elm_type = elm_type.lower()

        self.n_elm_gp = kargs.pop("n_elm_gp", None)
        if self.n_elm_gp is None:
            self.n_elm_gp = weakform.assembly_options.get(
                "n_elm_gp", elm_type, get_default_n_gp(elm_type, mesh)
            )

        self.assume_sym = weakform.assembly_options.get("assume_sym", elm_type, False)
        self.mat_lumping = weakform.assembly_options.get("mat_lumping", elm_type, False)

        self._use_local_csys = self._test_if_local_csys()
        self._element_local_frame = None

        self._saved_bloc_structure = None  # use to save data about the sparse structure and avoid time consuming recomputation
        self._assembly_method = (
            "new"  # _assembly_method = 'old' and 'very_old' only used for debug purpose
        )
        self.__factorize_op = (
            True  # option for debug purpose (should be set to True for performance)
        )

        self.sv = {}
        """ Dictionary of state variables associated to the current problem."""
        self.sv_start = {}
        self.sv_type = {}  # type of values (between 'Node', 'Element' and 'GaussPoint'. default = 'GaussPoint' if field not present in sv_type)

        self._nlgeom = None

        self._pb = None

    def __add__(self, another_assembly):
        return Assembly.sum(self, another_assembly)

    def assemble_global_mat(self, compute="all"):
        """
        Compute the global matrix and global vector related to the assembly
        if compute = 'all', compute the global matrix and vector
        if compute = 'matrix', compute only the matrix
        if compute = 'vector', compute only the vector
        if compute = 'none', compute nothing
        """
        if compute == "none":
            return

        # t0 = time.time()

        _assembly_method = self._assembly_method

        n_elm_gp = self.n_elm_gp

        if (
            self.meshChange == True
        ):  # only node position change is considered here. For change of sparsity, use also, self.mesh.init_interpolation
            if self.mesh in Assembly._saved_change_of_basis_mat:
                del Assembly._saved_change_of_basis_mat[self.mesh]
            self.compute_elementary_operators()

        nvar = self.space.nvar
        wf = self.weakform.get_weak_equation(self, self._pb)

        mat_gaussian_quadrature = self._get_gaussian_quadrature_mat()
        mat_change_of_basis = self.get_change_of_basis_mat()
        associatedVariables = (
            self._get_associated_variables()
        )  # for element requiring many variable such as beam with disp and rot dof

        if _assembly_method == "new":
            same_op_as_next, sorted_indices = wf.sort()
            # sl contains list of slice object that contains the dimension for each variable
            # size of VV and sl must be redefined for case with change of basis
            VV = 0

            # number of col for each bloc
            if np.isscalar(mat_change_of_basis) and mat_change_of_basis == 1:
                n_bloc_cols = self.mesh.n_nodes
            else:
                n_bloc_cols = self.mesh.n_elements * self.mesh.n_elm_nodes
                # or: mat_change_of_basis.shape[0]//nvar
            sl = [slice(i * n_bloc_cols, (i + 1) * n_bloc_cols) for i in range(nvar)]

            if n_elm_gp == 0:  # if finite difference elements, don't use BlocSparse
                blocks = [[None for i in range(nvar)] for j in range(nvar)]
                self._saved_bloc_structure = (
                    0  # don't save block structure for finite difference mesh
                )

                Matvir = self._get_elementary_operator(
                    wf.op_vir[0], n_elm_gp=0
                )[
                    0
                ].T  # should be identity matrix restricted to nodes used in the finite difference mesh

                for ii in range(len(wf.op)):
                    if compute == "matrix" and wf.op[ii] == 1:
                        continue
                    if compute == "vector" and not (wf.op[ii] == 1):
                        continue

                    if (
                        ii > 0 and same_op_as_next[ii - 1]
                    ):  # if same operator as previous with different coef, add the two coef
                        coef_PG += wf.coef[ii]
                    else:
                        coef_PG = wf.coef[
                            ii
                        ]  # coef_PG = nodal values (finite differences)

                    if (
                        ii < len(wf.op) - 1 and same_op_as_next[ii]
                    ):  # if operator similar to the next, continue
                        continue

                    var_vir = wf.op_vir[ii].u
                    assert (
                        wf.op_vir[ii].ordre == 0
                    ), "This weak form is not compatible with finite difference mesh"

                    if (
                        wf.op[ii] == 1
                    ):  # only virtual operator -> compute a vector which is the nodal values
                        if np.isscalar(VV) and VV == 0:
                            VV = np.zeros((n_bloc_cols * nvar))
                        VV[sl[var_vir[ii]]] = VV[sl[var_vir[ii]]] - (coef_PG)

                    else:  # virtual and real operators -> compute a matrix
                        var = wf.op[ii].u
                        if np.isscalar(coef_PG):
                            coef_PG = coef_PG * np.ones_like(
                                mat_gaussian_quadrature.data
                            )
                        CoefMatrix = sparse.csr_matrix(
                            (
                                coef_PG,
                                mat_gaussian_quadrature.indices,
                                mat_gaussian_quadrature.indptr,
                            ),
                            shape=mat_gaussian_quadrature.shape,
                        )
                        Mat = self._get_elementary_operator(wf.op[ii])[0]

                        if blocks[var_vir][var] is None:
                            blocks[var_vir][var] = Matvir @ CoefMatrix @ Mat
                        else:
                            blocks[var_vir][var].data += (
                                Matvir @ CoefMatrix @ Mat
                            ).data

                blocks = [
                    [
                        b
                        if b is not None
                        else sparse.csr_matrix((self.mesh.n_nodes, self.mesh.n_nodes))
                        for b in blocks_row
                    ]
                    for blocks_row in blocks
                ]
                MM = sparse.bmat(blocks, format="csr")

            else:
                MM = BlocSparse(
                    nvar,
                    nvar,
                    self.n_elm_gp,
                    self._saved_bloc_structure,
                    assume_sym=self.assume_sym,
                )
                listMatvir = listCoef_PG = None

                sum_coef = (
                    False  # bool that indicate if operator are the same and can be sum
                )

                if hasattr(self.weakform, "_list_mat_lumping"):
                    change_mat_lumping = (
                        True  # use different mat_lumping option for each operator
                    )
                else:
                    change_mat_lumping = False
                    mat_lumping = self.mat_lumping

                for ii in range(len(wf.op)):
                    if (
                        compute == "matrix"
                        and np.isscalar(wf.op[ii])
                        and wf.op[ii] == 1
                    ):
                        continue
                    if compute == "vector" and not wf.op[ii] == 1:
                        continue

                    if (
                        not wf.op[ii] == 1
                        and self.assume_sym
                        and wf.op[ii].u < wf.op_vir[ii].u
                    ):
                        continue

                    if change_mat_lumping:
                        mat_lumping = self.weakform._list_mat_lumping[
                            sorted_indices[ii]
                        ]

                    if np.isscalar(wf.coef[ii]) or len(wf.coef[ii]) == 1:
                        # if n_elm_gp == 0, coef_PG = nodal values (finite diffirences)
                        coef_PG = wf.coef[ii]
                    else:
                        coef_PG = self.mesh.data_to_gausspoint(wf.coef[ii][:], n_elm_gp)

                    # if ii > 0 and intRef[ii] == intRef[ii-1]: #if same operator as previous with different coef, add the two coef
                    if sum_coef:  # if same operator as previous with different coef, add the two coef
                        coef_PG_sum += coef_PG
                        sum_coef = False
                    else:
                        coef_PG_sum = coef_PG

                    if (
                        ii < len(wf.op) - 1 and same_op_as_next[ii]
                    ):  # if operator similar to the next, continue
                        if (
                            not (change_mat_lumping)
                            or mat_lumping
                            == self.weakform._list_mat_lumping[sorted_indices[ii + 1]]
                        ):
                            sum_coef = True
                            continue

                    coef_PG = (
                        coef_PG_sum * mat_gaussian_quadrature.data
                    )  # mat_gaussian_quadrature.data is the diagonal of mat_gaussian_quadrature
                    #                Matvir = (RowBlocMatrix(self._get_elementary_operator(wf.op_vir[ii]), nvar, var_vir, coef_vir) * mat_change_of_basis).T
                    # check how it appens with change of variable and rotation dof

                    Matvir = self._get_elementary_operator(wf.op_vir[ii])

                    if (
                        listMatvir is not None
                    ):  # factorization of real operator (sum of virtual operators)
                        listMatvir = [
                            listMatvir[j] + [Matvir[j]] for j in range(len(Matvir))
                        ]
                        listCoef_PG = listCoef_PG + [coef_PG]

                    if (
                        ii < len(wf.op) - 1
                        and wf.op[ii] != 1
                        and wf.op[ii + 1] != 1
                        and self.__factorize_op == True
                    ):
                        # if it possible, factorization of op to increase assembly performance (sum of several op_vir)
                        factWithNextOp = [
                            wf.op[ii].u,
                            wf.op[ii].x,
                            wf.op[ii].ordre,
                            wf.op_vir[ii].u,
                        ] == [
                            wf.op[ii + 1].u,
                            wf.op[ii + 1].x,
                            wf.op[ii + 1].ordre,
                            wf.op_vir[ii + 1].u,
                        ]  # True if factorization is possible with next op
                        if factWithNextOp:
                            if listMatvir is None:
                                listMatvir = [
                                    [Matvir[j]] for j in range(len(Matvir))
                                ]  # initialization of listMatvir and listCoef_PG
                                listCoef_PG = [coef_PG]
                            continue  # si factorization possible -> go to next op, all the factorizable operators will treat together
                    else:
                        factWithNextOp = False

                    coef_vir = [1]
                    var_vir = [
                        wf.op_vir[ii].u
                    ]  # list in case there is an angular variable
                    if var_vir[0] in associatedVariables:
                        var_vir.extend(associatedVariables[var_vir[0]][0])
                        coef_vir.extend(associatedVariables[var_vir[0]][1])

                    if wf.op[ii] == 1:  # only virtual operator -> compute a vector
                        if np.isscalar(VV) and VV == 0:
                            VV = np.zeros((n_bloc_cols * nvar))
                        for i in range(len(Matvir)):
                            try:
                                VV[sl[var_vir[i]]] = VV[sl[var_vir[i]]] - coef_vir[
                                    i
                                ] * Matvir[i].T * (
                                    coef_PG
                                )  # this line may be optimized
                            except:
                                pass

                    else:  # virtual and real operators -> compute a matrix
                        coef = [1]
                        var = [wf.op[ii].u]  # list in case there is an angular variable
                        if var[0] in associatedVariables:
                            var.extend(associatedVariables[var[0]][0])
                            coef.extend(associatedVariables[var[0]][1])

                        #                    Mat    =  RowBlocMatrix(self._get_elementary_operator(wf.op[ii]), nvar, var, coef)         * mat_change_of_basis
                        Mat = self._get_elementary_operator(wf.op[ii])

                        # Possibility to increase performance for multivariable case
                        # the structure should be the same for derivative dof, so the blocs could be computed altogether
                        if listMatvir is None:
                            for i in range(len(Mat)):
                                for j in range(len(Matvir)):
                                    MM.addToBlocATB(
                                        Matvir[j],
                                        Mat[i],
                                        (coef[i] * coef_vir[j]) * coef_PG,
                                        var_vir[j],
                                        var[i],
                                        mat_lumping,
                                    )
                        else:
                            for i in range(len(Mat)):
                                for j in range(len(Matvir)):
                                    MM.addToBlocATB(
                                        listMatvir[j],
                                        Mat[i],
                                        [
                                            (coef[i] * coef_vir[j]) * coef_PG
                                            for coef_PG in listCoef_PG
                                        ],
                                        var_vir[j],
                                        var[i],
                                        mat_lumping,
                                    )
                            listMatvir = None
                            listCoef_PG = None

            if compute != "vector":
                if np.isscalar(mat_change_of_basis) and mat_change_of_basis == 1:
                    self.global_matrix = MM.tocsr()  # format csr
                else:
                    self.global_matrix = (
                        mat_change_of_basis.T * MM.tocsr() * mat_change_of_basis
                    )  # format csr
            if compute != "matrix":
                if np.isscalar(VV) and VV == 0:
                    self.global_vector = 0
                elif np.isscalar(mat_change_of_basis) and mat_change_of_basis == 1:
                    self.global_vector = VV  # numpy array
                else:
                    self.global_vector = mat_change_of_basis.T * VV

            if self._saved_bloc_structure is None:
                self._saved_bloc_structure = MM.get_BlocStructure()

        elif _assembly_method == "old":
            # keep a lot in memory, not very efficient in a memory point of view. May be slightly more rapid in some cases
            # Don't work with alias variables
            intRef = wf.sort()  # intRef = list of integer for compareason (same int = same operator with different coef)

            if (
                self.mesh,
                self.elm_type,
                n_elm_gp,
            ) not in Assembly._saved_elementary_operators:
                Assembly._saved_elementary_operators[
                    (self.mesh, self.elm_type, n_elm_gp)
                ] = {}
            saveOperator = Assembly._saved_elementary_operators[
                (self.mesh, self.elm_type, n_elm_gp)
            ]

            # list_elm_type contains the id of the element associated with every variable
            # list_elm_type could be stored to avoid reevaluation
            if hasattr(get_element(self.elm_type), "get_elm_type"):
                element = get_element(self.elm_type)
                list_elm_type = [
                    element.get_elm_type(self.space.variable_name(i))
                    for i in range(nvar)
                ]  # will not work for alias variable...
            else:
                list_elm_type = [self.elm_type for i in range(nvar)]

            if "blocShape" not in saveOperator:
                saveOperator["blocShape"] = saveOperator[
                    "colBlocSparse"
                ] = saveOperator["rowBlocSparse"] = None

            # MM not used if only compute vector
            MM = BlocSparseOld(nvar, nvar)
            MM.col = saveOperator[
                "colBlocSparse"
            ]  # col indices for bloc to build coo matrix with BlocSparse
            MM.row = saveOperator[
                "rowBlocSparse"
            ]  # row indices for bloc to build coo matrix with BlocSparse
            MM.blocShape = saveOperator["blocShape"]  # shape of one bloc in BlocSparse

            # sl contains list of slice object that contains the dimension for each variable
            # size of VV and sl must be redefined for case with change of basis
            VV = 0
            nbNodes = self.mesh.n_nodes
            sl = [slice(i * nbNodes, (i + 1) * nbNodes) for i in range(nvar)]

            for ii in range(len(wf.op)):
                if compute == "matrix" and np.isscalar(wf.op[ii]) and wf.op[ii] == 1:
                    continue
                if compute == "vector" and not wf.op[ii] == 1:
                    continue

                if np.isscalar(wf.coef[ii]) or len(wf.coef[ii]) == 1:
                    coef_PG = wf.coef[
                        ii
                    ]  # mat_gaussian_quadrature.data is the diagonal of mat_gaussian_quadrature
                else:
                    coef_PG = self.mesh.data_to_gausspoint(wf.coef[ii][:], n_elm_gp)

                if (
                    ii > 0 and intRef[ii] == intRef[ii - 1]
                ):  # if same operator as previous with different coef, add the two coef
                    coef_PG_sum += coef_PG
                else:
                    coef_PG_sum = coef_PG

                if (
                    ii < len(wf.op) - 1 and intRef[ii] == intRef[ii + 1]
                ):  # if operator similar to the next, continue
                    continue

                coef_PG = coef_PG_sum * mat_gaussian_quadrature.data

                coef_vir = [1]
                var_vir = [wf.op_vir[ii].u]  # list in case there is an angular variable

                if var_vir[0] in associatedVariables:
                    var_vir.extend(associatedVariables[var_vir[0]][0])
                    coef_vir.extend(associatedVariables[var_vir[0]][1])

                if wf.op[ii] == 1:  # only virtual operator -> compute a vector
                    Matvir = self._get_elementary_operator(wf.op_vir[ii])
                    if np.isscalar(VV) and VV == 0:
                        VV = np.zeros((self.mesh.n_nodes * nvar))
                    for i in range(len(Matvir)):
                        VV[sl[var_vir[i]]] = VV[sl[var_vir[i]]] - coef_vir[i] * Matvir[
                            i
                        ].T * (coef_PG)  # this line may be optimized

                else:  # virtual and real operators -> compute a matrix
                    coef = [1]
                    var = [wf.op[ii].u]  # list in case there is an angular variable
                    if var[0] in associatedVariables:
                        var.extend(associatedVariables[var[0]][0])
                        coef.extend(associatedVariables[var[0]][1])

                    tuplename = (
                        list_elm_type[wf.op_vir[ii].u],
                        wf.op_vir[ii].x,
                        wf.op_vir[ii].ordre,
                        list_elm_type[wf.op[ii].u],
                        wf.op[ii].x,
                        wf.op[ii].ordre,
                    )  # tuple to identify operator
                    if tuplename in saveOperator:
                        MatvirT_Mat = saveOperator[
                            tuplename
                        ]  # MatvirT_Mat is an array that contains usefull data to build the matrix MatvirT*Matcoef*Mat where Matcoef is a diag coefficient matrix. MatvirT_Mat is build with BlocSparse class
                    else:
                        MatvirT_Mat = None
                        saveOperator[tuplename] = [
                            [None for i in range(len(var))] for j in range(len(var_vir))
                        ]
                        Matvir = self._get_elementary_operator(wf.op_vir[ii])
                        Mat = self._get_elementary_operator(wf.op[ii])

                    for i in range(len(var)):
                        for j in range(len(var_vir)):
                            if MatvirT_Mat is not None:
                                MM.addToBloc(
                                    MatvirT_Mat[j][i],
                                    (coef[i] * coef_vir[j]) * coef_PG,
                                    var_vir[j],
                                    var[i],
                                )
                            else:
                                saveOperator[tuplename][j][i] = MM.addToBlocATB(
                                    Matvir[j],
                                    Mat[i],
                                    (coef[i] * coef_vir[j]) * coef_PG,
                                    var_vir[j],
                                    var[i],
                                )
                                if saveOperator["colBlocSparse"] is None:
                                    saveOperator["colBlocSparse"] = MM.col
                                    saveOperator["rowBlocSparse"] = MM.row
                                    saveOperator["blocShape"] = MM.blocShape

            if compute != "vector":
                if np.isscalar(mat_change_of_basis) and mat_change_of_basis == 1:
                    self.global_matrix = MM.toCSR()  # format csr
                else:
                    self.global_matrix = (
                        mat_change_of_basis.T * MM.toCSR() * mat_change_of_basis
                    )  # format csr
            if compute != "matrix":
                if np.isscalar(VV) and VV == 0:
                    self.global_vector = 0
                elif np.isscalar(mat_change_of_basis) and mat_change_of_basis == 1:
                    self.global_vector = VV  # numpy array
                else:
                    self.global_vector = mat_change_of_basis.T * VV

        elif _assembly_method == "very_old":
            MM = 0
            VV = 0

            for ii in range(len(wf.op)):
                if compute == "matrix" and np.isscalar(wf.op[ii]) and wf.op[ii] == 1:
                    continue
                if compute == "vector" and not wf.op[ii] == 1:
                    continue

                coef_vir = [1]
                var_vir = [wf.op_vir[ii].u]  # list in case there is an angular variable
                if var_vir[0] in associatedVariables:
                    var_vir.extend(associatedVariables[var_vir[0]][0])
                    coef_vir.extend(associatedVariables[var_vir[0]][1])

                Matvir = (
                    RowBlocMatrix(
                        self._get_elementary_operator(wf.op_vir[ii]),
                        nvar,
                        var_vir,
                        coef_vir,
                    )
                    * mat_change_of_basis
                ).T

                if wf.op[ii] == 1:  # only virtual operator -> compute a vector
                    if np.isscalar(wf.coef[ii]):
                        VV = VV - wf.coef[ii] * Matvir * mat_gaussian_quadrature.data
                    else:
                        coef_PG = (
                            self.mesh.data_to_gausspoint(wf.coef[ii][:], n_elm_gp)
                            * mat_gaussian_quadrature.data
                        )
                        VV = VV - Matvir * (coef_PG)

                else:  # virtual and real operators -> compute a matrix
                    coef = [1]
                    var = [wf.op[ii].u]  # list in case there is an angular variable
                    if var[0] in associatedVariables:
                        var.extend(associatedVariables[var[0]][0])
                        coef.extend(associatedVariables[var[0]][1])

                    Mat = (
                        RowBlocMatrix(
                            self._get_elementary_operator(wf.op[ii]),
                            nvar,
                            var,
                            coef,
                        )
                        * mat_change_of_basis
                    )

                    if np.isscalar(wf.coef[ii]):  # and self.op_vir[ii] != 1:
                        MM = MM + wf.coef[ii] * Matvir * mat_gaussian_quadrature * Mat
                    else:
                        coef_PG = self.mesh.data_to_gausspoint(wf.coef[ii][:], n_elm_gp)
                        CoefMatrix = sparse.csr_matrix(
                            (
                                mat_gaussian_quadrature.data * coef_PG,
                                mat_gaussian_quadrature.indices,
                                mat_gaussian_quadrature.indptr,
                            ),
                            shape=mat_gaussian_quadrature.shape,
                        )
                        MM = MM + Matvir * CoefMatrix * Mat

            #            MM = MM.tocsr()
            #            MM.eliminate_zeros()
            if compute != "vector":
                self.global_matrix = MM  # format csr
            if compute != "matrix":
                self.global_vector = VV  # numpy array

        # print('temps : ', print(compute), ' - ', time.time()- t0)

    def get_change_of_basis_mat(self):
        ### change of basis treatment for beam or plate elements
        ### Compute the change of basis matrix for vector defined in self.space.list_vectors()
        if not (self._use_local_csys):
            return 1

        mesh = self.mesh

        if mesh in Assembly._saved_change_of_basis_mat:
            return Assembly._saved_change_of_basis_mat[mesh]

        mat_change_of_basis = 1
        compute_mat_change_of_basis = False

        n_nd = mesh.n_nodes
        n_el = mesh.n_elements
        elm = mesh.elements
        n_elm_nodes = np.shape(elm)[1]
        dim = self.space.ndim
        local_frame = mesh.local_frame

        if (
            "X" in mesh.crd_name and "Y" in mesh.crd_name
        ):  # if not in physical space, no change of variable
            for nameVector in self.space.list_vectors():
                # if compute_mat_change_of_basis == False and len(self.space.get_vector(nameVector))>1:
                if compute_mat_change_of_basis == False:
                    range_n_elm_nodes = np.arange(n_elm_nodes)
                    compute_mat_change_of_basis = True
                    nvar = self.space.nvar
                    listGlobalVector = []
                    listScalarVariable = list(range(nvar))
                #                        mat_change_of_basis = sparse.lil_matrix((nvar*n_el*n_elm_nodes, nvar*n_nd)) #lil is very slow because it change the sparcity of the structure

                rank_vector = self.space.get_rank_vector(nameVector)
                if rank_vector[0] not in listGlobalVector:
                    # ignore alias vectors. just test the 1st var
                    listGlobalVector.append(rank_vector)
                    # vector that need to be change in local coordinate
                    listScalarVariable = [
                        i for i in listScalarVariable if not (i in listGlobalVector[-1])
                    ]  # scalar variable that doesnt need to be converted
            # Data to build mat_change_of_basis with coo sparse format
            if compute_mat_change_of_basis:
                # get element local basis
                if self._element_local_frame is None:
                    if self.n_elm_gp not in mesh._elm_interpolation:
                        mesh.init_interpolation(self.n_elm_gp)

                    elmRefGeom = mesh._elm_interpolation[self.n_elm_gp]
                    # elmRefGeom = get_element(mesh.elm_type)(mesh=mesh)

                    xi_nd = get_node_elm_coordinates(mesh.elm_type, n_elm_nodes)
                    local_frame_el = elmRefGeom.GetLocalFrame(
                        mesh.nodes[mesh._elements_geom], xi_nd, local_frame
                    )  # array of shape (n_el, nb_nd, nb of vectors in basis = dim, dim)
                else:
                    local_frame_el = self._element_local_frame

                rowMCB = np.empty((len(listGlobalVector) * n_el, n_elm_nodes, dim, dim))
                colMCB = np.empty((len(listGlobalVector) * n_el, n_elm_nodes, dim, dim))
                dataMCB = np.empty(
                    (len(listGlobalVector) * n_el, n_elm_nodes, dim, dim)
                )

                for ivec, vec in enumerate(listGlobalVector):
                    # dataMCB[ivec*n_el:(ivec+1)*n_el] = local_frame_el[:,:,:dim,:dim]
                    dataMCB[ivec * n_el : (ivec + 1) * n_el] = local_frame_el
                    rowMCB[ivec * n_el : (ivec + 1) * n_el] = (
                        np.arange(n_el).reshape(-1, 1, 1, 1)
                        + range_n_elm_nodes.reshape(1, -1, 1, 1) * n_el
                        + np.array(vec).reshape(1, 1, -1, 1) * (n_el * n_elm_nodes)
                    )
                    colMCB[ivec * n_el : (ivec + 1) * n_el] = (
                        elm.reshape(n_el, n_elm_nodes, 1, 1)
                        + np.array(vec).reshape(1, 1, 1, -1) * n_nd
                    )

                if len(listScalarVariable) > 0:
                    # add the component from scalar variables (ie variable not requiring a change of basis)
                    dataMCB = np.hstack(
                        (
                            dataMCB.reshape(-1),
                            np.ones(len(listScalarVariable) * n_el * n_elm_nodes),
                        )
                    )  # no change of variable so only one value adding in dataMCB

                    rowMCB_loc = np.empty((len(listScalarVariable) * n_el, n_elm_nodes))
                    colMCB_loc = np.empty((len(listScalarVariable) * n_el, n_elm_nodes))
                    for ivar, var in enumerate(listScalarVariable):
                        rowMCB_loc[ivar * n_el : (ivar + 1) * n_el] = (
                            np.arange(n_el).reshape(-1, 1)
                            + range_n_elm_nodes.reshape(1, -1) * n_el
                            + var * (n_el * n_elm_nodes)
                        )
                        colMCB_loc[ivar * n_el : (ivar + 1) * n_el] = elm + var * n_nd

                    rowMCB = np.hstack((rowMCB.reshape(-1), rowMCB_loc.reshape(-1)))
                    colMCB = np.hstack((colMCB.reshape(-1), colMCB_loc.reshape(-1)))

                    mat_change_of_basis = sparse.coo_matrix(
                        (dataMCB, (rowMCB, colMCB)),
                        shape=(n_el * n_elm_nodes * nvar, n_nd * nvar),
                    )
                else:
                    mat_change_of_basis = sparse.coo_matrix(
                        (
                            dataMCB.reshape(-1),
                            (rowMCB.reshape(-1), colMCB.reshape(-1)),
                        ),
                        shape=(n_el * n_elm_nodes * nvar, n_nd * nvar),
                    )

                mat_change_of_basis = mat_change_of_basis.tocsr()

        Assembly._saved_change_of_basis_mat[mesh] = mat_change_of_basis
        return mat_change_of_basis

    def initialize(self, pb):
        """
        Initialize the associated weak form and assemble the global matrix
        with the elastic matrix.

        Parameters:
            - pb: the problem associated to the assembly
        """
        self.weakform.initialize(self, pb)

        if self.weakform.constitutivelaw is not None:
            self.weakform.constitutivelaw.initialize(self, pb)

        self._pb = pb  # set the associated problem
        self.sv_start = dict(
            self.sv
        )  # initialization in case sv in modified by weakform.initialize

    def set_start(self, pb):
        """
        Apply the modification to the constitutive equation required at each new time increment.
        Generally used to increase non reversible internal variable
        Assemble the new global matrix.
        """
        self.weakform.set_start(self, pb)
        if self.weakform.constitutivelaw is not None:
            self.weakform.constitutivelaw.set_start(
                self, pb
            )  # should update GetH() method to return elastic rigidity matrix for prediction

        self.sv_start = dict(
            self.sv
        )  # create a new dict with alias inside (not deep copy)

        self.current.assemble_global_mat("all")
        # no need to compute vector if the previous iteration has converged and (dt hasn't changed or dt isn't used in the weakform)
        # in those cases, self.assemble_global_mat(compute = 'matrix') should be more efficient
        # save statev start values

    def update(self, pb, compute="all"):
        """
        Update the associated weak form and assemble the global matrix
        Parameters:
            - pb: a Problem object containing the Dof values
        """
        self.weakform.update(self, pb)
        if self.weakform.constitutivelaw is not None:
            self.weakform.constitutivelaw.update(self, pb)
        self.weakform.update_2(self, pb)
        self.current.assemble_global_mat(compute)

    def to_start(self, pb):
        """
        reset the current time increment (internal variable in the constitutive equation)
        Doesn't assemble the new global matrix. Use the Update method for that purpose.
        """
        self.weakform.to_start(self, pb)
        if self.weakform.constitutivelaw is not None:
            self.weakform.constitutivelaw.to_start(self, pb)

        # replace statev with the start values
        self.sv = self.current.sv = dict(self.sv_start)

        self.current.assemble_global_mat("all")

    def reset(self):
        """
        reset the assembly to it's initial state.
        Internal variable in the constitutive equation are reinitialized
        and stored global matrix and vector are deleted
        """
        self.weakform.reset()
        if self.weakform.constitutivelaw is not None:
            self.weakform.constitutivelaw.reset()
        self.delete_global_mat()
        # self.current.delete_global_mat()
        self.current = self

        # remove all state variables
        self.sv = {}
        self.sv_start = {}
        self.sv_type = {}

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
        Assembly._saved_elementary_operators = {}
        Assembly._saved_change_of_basis_mat = {}
        # Assembly._saved_gaussian_quadrature_mat = {}
        # Assembly._saved_node2gausspoint_mat = {}
        # Assembly._saved_gausspoint2node_mat = {}
        Assembly._saved_associated_variables = {}  # dict containing all associated variables (rotational dof for C1 elements) for elm_type

    def compute_elementary_operators(
        self, n_elm_gp=None
    ):  # Précalcul des opérateurs dérivés suivant toutes les directions (optimise les calculs en minimisant le nombre de boucle)
        # -------------------------------------------------------------------
        # Initialisation
        # -------------------------------------------------------------------
        mesh = self.mesh
        elm_type = self.elm_type
        if n_elm_gp is None:
            n_elm_gp = self.n_elm_gp

        n_elements = mesh.n_elements
        elements = mesh.elements
        nodes = mesh.nodes
        n_elm_nodes = (
            mesh.n_elm_nodes
        )  # number of nodes associated to each element including internal nodes

        # -------------------------------------------------------------------
        # Case of finite difference mesh
        # -------------------------------------------------------------------
        if n_elm_gp == 0:  # in this case, it is a finite difference mesh
            # we compute the operators directly from the element library
            elmRef = get_element(elm_type)(n_elm_gp)
            OP = elmRef.computeOperator(nodes, elements)
            mesh._saved_gaussian_quadrature_mat[n_elm_gp] = sparse.identity(
                OP[0, 0][0].shape[0], "d", format="csr"
            )  # No gaussian quadrature in this case : nodal identity matrix
            mesh._saved_gausspoint2node_mat[n_elm_gp] = (
                1  # no need to translate between pg and nodes because no pg
            )
            mesh._saved_node2gausspoint_mat[n_elm_gp] = 1
            Assembly._saved_change_of_basis_mat[mesh] = (
                1  # No change of basis:  mat_change_of_basis = 1 #this line could be deleted because the coordinate should in principle defined as 'global'
            )
            Assembly._saved_elementary_operators[(mesh, elm_type, n_elm_gp)] = (
                OP  # elmRef.computeOperator(nodes,elements)
            )
            return

        # -------------------------------------------------------------------
        # Initialize the geometrical interpolation (gaussian quadrature, jacobian matrix, ...)
        # -------------------------------------------------------------------
        if n_elm_gp not in mesh._elm_interpolation:
            mesh.init_interpolation(n_elm_gp)

        mesh._compute_gaussian_quadrature_mat(n_elm_gp)
        elmRefGeom = mesh._elm_interpolation[n_elm_gp]

        # -------------------------------------------------------------------
        # Compute the array containing row and col indices used to assemble the sparse matrices
        # -------------------------------------------------------------------
        row, col = mesh._sparse_structure[
            n_elm_gp
        ]  # row and col have been computed in the mesh.init_interpolation method

        mat_change_of_basis = self.get_change_of_basis_mat()
        if np.isscalar(mat_change_of_basis) and mat_change_of_basis == 1:
            # ChangeOfBasis = False
            n_col = mesh.n_nodes
        else:
            # ChangeOfBasis = True -> modify col vector
            col = np.empty((n_elements, n_elm_gp, n_elm_nodes))
            col[:] = (
                np.arange(n_elements).reshape((-1, 1, 1))
                + np.arange(n_elm_nodes).reshape((1, 1, -1)) * n_elements
            )
            col = col.reshape(-1)
            n_col = n_elements * n_elm_nodes

        # -------------------------------------------------------------------
        # Build the list of elm_type to assemble (some beam element required several elm_type in function of the variable)
        # -------------------------------------------------------------------
        objElement = get_element(elm_type)
        if hasattr(objElement, "get_all_elm_type"):
            list_elm_type = objElement.get_all_elm_type()
        else:
            list_elm_type = [objElement]

        # -------------------------------------------------------------------
        # Assembly of the elementary operators for each elm_type
        # -------------------------------------------------------------------
        for elm_type in list_elm_type:
            elmRef = elm_type(n_elm_gp, elmGeom=elmRefGeom, assembly=self)

            n_interpol_nodes = elmRef.n_nodes  # number of nodes used in the element interpolation (may be different from mesh.n_elm_nodes)

            # special treatment so that elements can have several different
            # shape functions for a same node (different kind of interpolation)
            # that may be used with different x values (op_deriv.x)
            # required for 3D hourglass
            n_diff_interpolations = 1
            if isinstance(elmRef.ShapeFunctionPG, list):
                n_diff_interpolations = len(elmRef.ShapeFunctionPG)
                shape_functions = elmRef.ShapeFunctionPG
                NbDoFperNode = elmRef.ShapeFunctionPG[0].shape[-1] // n_interpol_nodes
            else:
                shape_functions = [elmRef.ShapeFunctionPG]
                NbDoFperNode = elmRef.ShapeFunctionPG.shape[-1] // n_interpol_nodes
            # end special treatment

            nb_dir_deriv = 0
            if hasattr(elmRef, "ShapeFunctionDerivativePG"):
                derivativePG = (
                    elmRefGeom.inverseJacobian @ elmRef.ShapeFunctionDerivativePG
                )  # derivativePG = np.matmul(elmRefGeom.inverseJacobian , elmRef.ShapeFunctionDerivativePG)
                nb_dir_deriv = derivativePG.shape[-2]
            nop = (
                nb_dir_deriv + n_diff_interpolations
            )  # nombre d'opérateur à discrétiser

            data = [
                [
                    np.empty((n_elements, n_elm_gp, n_elm_nodes))
                    for j in range(NbDoFperNode)
                ]
                for i in range(nop)
            ]

            for j in range(0, NbDoFperNode):
                for i in range(n_diff_interpolations):  # i should be mainly 0
                    data[i][j][:, :, :n_interpol_nodes] = shape_functions[i][
                        ..., j * n_interpol_nodes : (j + 1) * n_interpol_nodes
                    ].reshape(
                        (-1, n_elm_gp, n_interpol_nodes)
                    )  # same as dataNodeToPG matrix if geometrical shape function are the same as interpolation functions
                for dir_deriv in range(nb_dir_deriv):
                    data[dir_deriv + n_diff_interpolations][j][
                        :, :, :n_interpol_nodes
                    ] = derivativePG[
                        ...,
                        dir_deriv,
                        j * n_interpol_nodes : (j + 1) * n_interpol_nodes,
                    ]

            op_dd = [
                [
                    sparse.coo_matrix(
                        (data[i][j].reshape(-1), (row, col)),
                        shape=(n_elements * n_elm_gp, n_col),
                    ).tocsr()
                    for j in range(NbDoFperNode)
                ]
                for i in range(nop)
            ]

            data = {(0, i): op_dd[i] for i in range(n_diff_interpolations)}
            for i in range(nb_dir_deriv):
                data[1, i] = op_dd[
                    i + n_diff_interpolations
                ]  # as index and indptr should be the same, perhaps it will be more memory efficient to only store the data field

            Assembly._saved_elementary_operators[(mesh, elm_type.name, n_elm_gp)] = data

    def _get_elementary_operator(self, deriv, n_elm_gp=None):
        # Gives a list of sparse matrix that convert node values for one variable to the pg values of a simple derivative op (for instance d/dz)
        # The list contains several element if the elm_type include several variable (dof variable in beam element). In other case, the list contains only one matrix
        # The variables are not considered. For a global use, the resulting matrix should be assembled in a block matrix with the nodes values for all variables
        if n_elm_gp is None:
            n_elm_gp = self.n_elm_gp

        elm_type = self.elm_type
        mesh = self.mesh

        if hasattr(get_element(elm_type), "get_elm_type"):
            elm_type = get_element(elm_type).get_elm_type(deriv.u_name).name

        if not ((mesh, elm_type, n_elm_gp) in Assembly._saved_elementary_operators):
            self.compute_elementary_operators(n_elm_gp)

        data = Assembly._saved_elementary_operators[(mesh, elm_type, n_elm_gp)]

        if deriv.ordre == 0:
            # in this case deriv.x should be 0 execpt if several interpolations
            # are used. In this case, deriv.x is and arbitrary id not related
            # to any coordinate
            xx = deriv.x
        else:
            # extract the mesh coordinate that corespond to coordinate rank given in deriv.x
            ListMeshCoordinatenameRank = [
                self.space.coordinate_rank(crdname)
                for crdname in mesh.crd_name
                if crdname in self.space.list_coordinates()
            ]
            if deriv.x in ListMeshCoordinatenameRank:
                xx = ListMeshCoordinatenameRank.index(deriv.x)
            else:
                # for PGD only (will be probably deprecated):
                # if the coordinate doesnt exist, return operator without
                # derivation ()
                return data[0, 0]
        if (deriv.ordre, xx) in data:
            return data[deriv.ordre, xx]
        else:
            pass
            assert 0, "Operator unavailable"

    def _get_gaussian_quadrature_mat(
        self,
    ):  # calcul la discrétision relative à un seul opérateur dérivé
        if not (self.n_elm_gp in self.mesh._saved_gaussian_quadrature_mat):
            self.compute_elementary_operators()
        return self.mesh._saved_gaussian_quadrature_mat[self.n_elm_gp]

    def _get_associated_variables(
        self,
    ):  # associated variables (rotational dof for C1 elements) of elm_type
        # based on variable rank, ie dont make differences between alias
        elm_type = self.elm_type
        if elm_type not in Assembly._saved_associated_variables:
            objElement = get_element(elm_type)
            if hasattr(objElement, "associated_variables"):
                Assembly._saved_associated_variables[elm_type] = {
                    self.space.variable_rank(key): [
                        [self.space.variable_rank(v) for v in val[1::2]],
                        val[0::2],
                    ]
                    for key, val in objElement.associated_variables.items()
                    if key in self.space.list_variables()
                }
                # Assembly._saved_associated_variables[elm_type] = {self.space.variable_rank(key):
                #                        [[self.space.variable_rank(v) for v in val[1][1::2]],
                #                         val[1][0::2]] for key,val in objElement.items() if key in self.space.list_variables() and len(val)>1}
                # val[1][0::2]] for key,val in objElement.items() if key in self.space.list_variables() and len(val)>1}
            else:
                Assembly._saved_associated_variables[elm_type] = {}
        return Assembly._saved_associated_variables[elm_type]

    def _test_if_local_csys(self):
        # determine the type of coordinate system used for vector of variables (displacement for instance). This type may be specified in element (under dict form only with elm_dict['__local_csys'] = True)
        # return True if the variables are defined in a local coordinate system, or False if global variables are used. If local variables are used, a change of variable is required
        # If '__local_csys' is not specified in the element, 'global' value (no change of basis) is considered by default
        element = get_element(self.elm_type)
        if hasattr(element, "local_csys"):
            return element.local_csys
        else:
            return False

        # if isinstance(get_element(self.elm_type), dict):
        #     return get_element(self.elm_type).get('__local_csys', False)
        # else:
        #     return False

    def get_element_results(self, operator, U):
        """
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

        res = self.get_gp_results(operator, U)
        n_elm_gp = res.shape[0] // self.mesh.n_elements
        return np.reshape(res, (n_elm_gp, -1)).sum(0) / n_elm_gp

    def get_gp_results(self, operator, U, n_elm_gp=None, use_local_dof=False):
        """
        Return some results at element Gauss points based on the finite element discretization of
        a differential operator on a mesh being given the dof results and the type of elements.

        Parameters
        ----------
        operator: DiffOp
            Differential operator defining the required results
        U: numpy.ndarray
            Vector containing all the DoF solution
        use_local_dof: bool, default: False
            if True, U is interpreted as local dof for each node in element
            The dof should be ordered so that U.reshape(mesh.n_elm_nodes, space.nvar, mesh.n_elements)[i,var,el]
            gives the value of the variable var (variable indice) for the ith nodes of the element el.

        Return: numpy.ndarray
            A Vector containing the values on each point of gauss for each element.
            The vector lenght is the number of element time the number of Gauss points per element
        """
        # TODO : can be accelerated by avoiding RowBlocMatrix (need to be checked) -> For each elementary
        # 1 - reshape U to separate each var U = U.reshape(var, -1)
        # 2 - in the loop : res += coef_PG * (Assembly._get_elementary_operator(mesh, operator.op[ii], elm_type, n_elm_gp) , nvar, var, coef) * U[var]

        res = 0
        nvar = self.space.nvar

        mesh = self.mesh
        elm_type = self.elm_type
        if n_elm_gp is None:
            n_elm_gp = self.n_elm_gp

        if not (use_local_dof):
            mat_change_of_basis = self.get_change_of_basis_mat()
            if not (np.isscalar(mat_change_of_basis) and mat_change_of_basis == 1):
                U = mat_change_of_basis @ U  # U local

        associatedVariables = self._get_associated_variables()

        for ii in range(len(operator.op)):
            var = [operator.op[ii].u]
            coef = [1]

            if var[0] in associatedVariables:
                var.extend(associatedVariables[var[0]][0])
                coef.extend(associatedVariables[var[0]][1])

            assert (
                operator.op_vir[ii] == 1
            ), "Operator virtual are only required to build FE operators, but not to get element results"

            if np.isscalar(operator.coef[ii]):
                coef_PG = operator.coef[ii]
            else:
                coef_PG = self.mesh.data_to_gausspoint(operator.coef[ii][:], n_elm_gp)

            res += coef_PG * (
                RowBlocMatrix(
                    self._get_elementary_operator(operator.op[ii], n_elm_gp),
                    nvar,
                    var,
                    coef,
                )
                @ U
            )

        return res  # equivalent to self._get_assembled_operator(operator, n_elm_gp) @ U but more optimized

    def _get_assembled_operator(self, operator, n_elm_gp=None):
        # not very optimized (sum of RowBlocMatrix use scipy add operator that is not efficient (change sparse structure))
        nvar = self.space.nvar

        mesh = self.mesh
        elm_type = self.elm_type
        if n_elm_gp is None:
            n_elm_gp = self.n_elm_gp

        mat_change_of_basis = self.get_change_of_basis_mat()
        associatedVariables = self._get_associated_variables()

        matrix = 0
        for ii in range(len(operator.op)):
            var = [operator.op[ii].u]
            coef = [1]

            if var[0] in associatedVariables:
                var.extend(associatedVariables[var[0]][0])
                coef.extend(associatedVariables[var[0]][1])

            assert (
                operator.op_vir[ii] == 1
            ), "Operator virtual are only required to build FE operators, but not to get element results"

            if np.isscalar(mat_change_of_basis) and mat_change_of_basis == 1:
                M = RowBlocMatrix(
                    self._get_elementary_operator(operator.op[ii], n_elm_gp),
                    nvar,
                    var,
                    coef,
                )
            else:
                M = (
                    RowBlocMatrix(
                        self._get_elementary_operator(operator.op[ii], n_elm_gp),
                        nvar,
                        var,
                        coef,
                    )
                    * mat_change_of_basis
                )

            if np.isscalar(operator.coef[ii]):
                M.data *= operator.coef[ii]
                matrix += M
            else:
                mat_gaussian_quadrature = self._get_gaussian_quadrature_mat()
                coef_PG = self.mesh.data_to_gausspoint(operator.coef[ii][:], n_elm_gp)
                coef_matrix = sparse.csr_matrix(
                    (
                        coef_PG,
                        mat_gaussian_quadrature.indices,
                        mat_gaussian_quadrature.indptr,
                    ),
                    shape=mat_gaussian_quadrature.shape,
                )
                matrix += coef_matrix @ M

        return matrix

    def get_node_results(self, operator, U):
        """
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

        res = self.get_gp_results(operator, U)
        return self.mesh._get_gausspoint2node_mat(self.n_elm_gp) @ res

    def convert_data(self, data, convert_from=None, convert_to="GaussPoint"):
        if isinstance(data, (StrainTensorList, StressTensorList)):
            return data.convert(self, convert_from, convert_to)

        return self.mesh.convert_data(data, convert_from, convert_to, self.n_elm_gp)

    def integrate_field(self, field, type_field=None):
        return self.mesh.integrate_field(field, type_field, self.n_elm_gp)

    def set_disp(self, disp):
        if np.isscalar(disp) and disp == 0:
            self.current = self
        else:
            new_crd = self.mesh.nodes + disp.T
            if self.current == self:
                # initialize a new assembly
                new_mesh = copy(self.mesh)
                new_mesh.nodes = new_crd
                new_mesh._saved_gaussian_quadrature_mat = {}
                new_assembly = copy(self)
                new_assembly.mesh = new_mesh
                self.current = new_assembly
            else:
                self.current.mesh.nodes = new_crd
                self.current.mesh._saved_gaussian_quadrature_mat = {}

                if self.current.mesh in self._saved_change_of_basis_mat:
                    del self._saved_change_of_basis_mat[self.current.mesh]

                for k in tuple(self._saved_elementary_operators):
                    if k[0] == self.current.mesh:
                        self._saved_elementary_operators.pop(k)

    def get_strain(self, U, Type="Node", nlgeom=None):
        """
        Not a static method.
        Return the Green Lagrange Strain Tensor of an assembly using the Voigt notation as a python list.
        The total displacement field has to be given.
        see get_node_results and get_element_results

        Options :
        - Type :"Node", "Element" or "GaussPoint" integration (default : "Node")
        - nlgeom = True or False if the strain tensor account for geometrical non-linearities
        if nlgeom = False, the Strain Tensor is assumed linear (default : True)

        example :
        S = SpecificAssembly.get_strain(Problem.Problem.get_dof_solution('all'))
        """

        if nlgeom is None:
            if hasattr(self.weakform, "nlgeom"):
                nlgeom = self.weakform.nlgeom
            else:
                nlgeom = False

        GradValues = self.get_grad_disp(U, Type)

        if nlgeom == False:
            Strain = [GradValues[i][i] for i in range(3)]
            Strain += [
                GradValues[0][1] + GradValues[1][0],
                GradValues[0][2] + GradValues[2][0],
                GradValues[1][2] + GradValues[2][1],
            ]
        else:
            Strain = [
                GradValues[i][i] + 0.5 * sum([GradValues[k][i] ** 2 for k in range(3)])
                for i in range(3)
            ]
            Strain += [
                GradValues[0][1]
                + GradValues[1][0]
                + sum([GradValues[k][0] * GradValues[k][1] for k in range(3)])
            ]
            Strain += [
                GradValues[0][2]
                + GradValues[2][0]
                + sum([GradValues[k][0] * GradValues[k][2] for k in range(3)])
            ]
            Strain += [
                GradValues[1][2]
                + GradValues[2][1]
                + sum([GradValues[k][1] * GradValues[k][2] for k in range(3)])
            ]

        return StrainTensorList(Strain)

    def get_grad_disp(self, U, Type="Node"):
        """
        Return the Gradient Tensor of a vector (generally displacement given by Problem.get_DofSolution('all')
        as a list of list of numpy array
        The total displacement field U has to be given as a flatten numpy array
        see get_node_results and get_element_resultss

        Options :
        - Type :"Node", "Element" or "GaussPoint" integration (default : "Node")
        """
        grad_operator = self.space.op_grad_u()

        if Type == "Node":
            return [
                [
                    self.get_node_results(op, U)
                    if op != 0
                    else np.zeros(self.mesh.n_nodes)
                    for op in line_op
                ]
                for line_op in grad_operator
            ]

        elif Type == "Element":
            return [
                [
                    self.get_element_results(op, U)
                    if op != 0
                    else np.zeros(self.mesh.n_elements)
                    for op in line_op
                ]
                for line_op in grad_operator
            ]

        elif Type == "GaussPoint":
            return [
                [
                    self.get_gp_results(op, U)
                    if op != 0
                    else np.zeros(self.n_gauss_points)
                    for op in line_op
                ]
                for line_op in grad_operator
            ]
        else:
            assert 0, "Wrong argument for Type: use 'Node', 'Element', or 'GaussPoint'"

    #     def get_ext_forces(self, U, nvar=None):
    #         """
    #         Not a static method.
    #         Return the nodal Forces and moments in global coordinates related to a specific assembly considering the DOF solution given in U
    #         The resulting forces are the sum of :
    #         - External forces (associated to Neumann boundary conditions)
    #         - Node reaction (associated to Dirichelet boundary conditions)
    #         - Inertia forces

    #         Return an array whose columns are Fx, Fy, Fz, Mx, My and Mz.

    #         example :
    #         S = SpecificAssembly.get_ext_forces(Problem.Problem.get_dof_solution('all'))
    #         """
    #         if nvar is None: nvar = self.space.nvar
    #         return np.reshape(self.get_global_matrix() * U - self.get_global_vector(), (nvar,-1))
    # #        return np.reshape(self.get_global_matrix() * U, (Nvar,-1)).T

    #    def get_int_forces(self, U, CoordinateSystem = 'global'):
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
    ##        operator = self.weakform.get_weak_equation(self.mesh)
    #        operator = self.weakform.get_generalized_stress()
    #        res = [self.get_element_results(operator[i], U) for i in range(5)]
    #        return res
    #

    #        res = np.reshape(res,(6,-1)).T
    #        n_el = mesh.n_elements
    #        res = (res[n_el:,:]-res[0:n_el:,:])/2
    #        res = res[:, [self.space.variable_rank('DispX'), self.space.variable_rank('DispY'), self.space.variable_rank('DispZ'), \
    #                              self.space.variable_rank('ThetaX'), self.space.variable_rank('ThetaY'), self.space.variable_rank('ThetaZ')]]
    #
    #        if CoordinateSystem == 'local': return res
    #        elif CoordinateSystem == 'global':
    #            #require a transformation between local and global coordinates on element
    #            #classical mat_change_of_basis transform only toward nodal values
    #            elmRef = get_element(self.mesh.elm_type)(1, mesh=mesh)#one pg  with the geometrical element
    #            vec = [0,1,2] ; dim = 3
    #
    #            #Data to build mat_change_of_basis_el with coo sparse format
    #            crd = mesh.nodes ; elm = mesh.elements
    #            rowMCB = np.empty((n_el, 1, dim,dim))
    #            colMCB = np.empty((n_el, 1, dim,dim))
    #            rowMCB[:] = np.arange(n_el).reshape(-1,1,1,1) + np.array(vec).reshape(1,1,-1,1)*n_el # [[id_el + var*n_el] for var in vec]
    #            colMCB[:] = np.arange(n_el).reshape(-1,1,1,1) + np.array(vec).reshape(1,1,1,-1)*n_el # [id_el+n_el*var for var in vec]
    #            dataMCB = elmRef.GetLocalFrame(crd[elm], elmRef.xi_pg, mesh.local_frame) #array of shape (n_el, n_elm_gp=1, nb of vectors in basis = dim, dim)
    #
    #            mat_change_of_basisElement = sparse.coo_matrix((np.reshape(dataMCB,-1),(np.reshape(rowMCB,-1),np.reshape(colMCB,-1))), shape=(dim*n_el, dim*n_el)).tocsr()
    #
    #            F = np.reshape( mat_change_of_basis_el.T * np.reshape(res[:,0:3].T, -1)  ,  (3,-1) ).T
    #            C = np.reshape( mat_change_of_basis_el.T * np.reshape(res[:,3:6].T, -1)  ,  (3,-1) ).T
    #            return np.hstack((F,C))

    def get_int_forces(self, U, CoordinateSystem="global"):
        """
        Only available for 2 nodes beam element
        Return the element internal Forces and moments related to a specific assembly considering the DOF solution given in U.
        Return array whose columns are Fx, Fy, Fz, Mx, My and Mz.

        Parameter: if CoordinateSystem == 'local' the result is given in the local coordinate system
                   if CoordinateSystem == 'global' the result is given in the global coordinate system (default)
        """

        operator = self.weakform.get_weak_equation(self, None)
        mesh = self.mesh
        nvar = self.space.nvar
        dim = self.space.ndim
        mat_change_of_basis = self.get_change_of_basis_mat()

        mat_gaussian_quadrature = self._get_gaussian_quadrature_mat()
        associatedVariables = self._get_associated_variables()

        # TODO: use the computeGlobalMatrix() method to compute sum(operator.coef[ii]*Matvir * mat_gaussian_quadrature * Mat)
        # add options in computeGlobalMatrix() to (i): dont save the computed matrix, (ii): neglect the ChangeOfBasis Matrix
        res = 0
        for ii in range(len(operator.op)):
            var = [operator.op[ii].u]
            coef = [1]
            var_vir = [operator.op_vir[ii].u]
            coef_vir = [1]

            if var[0] in associatedVariables:
                var.extend(associatedVariables[var[0]][0])
                coef.extend(associatedVariables[var[0]][1])
            if var_vir[0] in associatedVariables:
                var_vir.extend(associatedVariables[var_vir[0]][0])
                coef_vir.extend(associatedVariables[var_vir[0]][1])

            Mat = RowBlocMatrix(
                self._get_elementary_operator(operator.op[ii]), nvar, var, coef
            )
            Matvir = RowBlocMatrix(
                self._get_elementary_operator(operator.op_vir[ii]),
                nvar,
                var_vir,
                coef_vir,
            ).T

            if np.isscalar(operator.coef[ii]):  # and self.op_vir[ii] != 1:
                res = (
                    res
                    + operator.coef[ii]
                    * Matvir
                    * mat_gaussian_quadrature
                    * Mat
                    * mat_change_of_basis
                    * U
                )
            else:
                return NotImplemented

        res = np.reshape(res, (nvar, -1)).T

        n_el = mesh.n_elements
        res = (res[n_el : 2 * n_el, :] - res[0:n_el:, :]) / 2

        # if dim == 3:
        #     res = res[:, [self.space.variable_rank('DispX'), self.space.variable_rank('DispY'), self.space.variable_rank('DispZ'), \
        #                   self.space.variable_rank('RotX'), self.space.variable_rank('RotY'), self.space.variable_rank('RotZ')]]
        # else:
        #     res = res[:, [self.space.variable_rank('DispX'), self.space.variable_rank('DispY'), self.space.variable_rank('RotZ')]]

        if CoordinateSystem == "local":
            return res
        elif CoordinateSystem == "global":
            # require a transformation between local and global coordinates on element
            # classical mat_change_of_basis transform only toward nodal values
            elmRef = get_element(self.mesh.elm_type)(
                1, assembly=assembly
            )  # one pg  with the geometrical element
            if dim == 3:
                vec = [0, 1, 2]
            else:
                vec = [0, 1]

            # Data to build mat_change_of_basis_el with coo sparse format
            crd = mesh.nodes
            elm = mesh.elements
            rowMCB = np.empty((n_el, 1, dim, dim))
            colMCB = np.empty((n_el, 1, dim, dim))
            rowMCB[:] = (
                np.arange(n_el).reshape(-1, 1, 1, 1)
                + np.array(vec).reshape(1, 1, -1, 1) * n_el
            )  # [[id_el + var*n_el] for var in vec]
            colMCB[:] = (
                np.arange(n_el).reshape(-1, 1, 1, 1)
                + np.array(vec).reshape(1, 1, 1, -1) * n_el
            )  # [id_el+n_el*var for var in vec]
            dataMCB = elmRef.GetLocalFrame(
                crd[elm], elmRef.xi_pg, mesh.local_frame
            )  # array of shape (n_el, n_elm_gp=1, nb of vectors in basis = dim, dim)

            mat_change_of_basis_el = sparse.coo_matrix(
                (
                    np.reshape(dataMCB, -1),
                    (np.reshape(rowMCB, -1), np.reshape(colMCB, -1)),
                ),
                shape=(dim * n_el, dim * n_el),
            ).tocsr()

            F = np.reshape(
                mat_change_of_basis_el.T * np.reshape(res[:, 0:dim].T, -1),
                (dim, -1),
            ).T
            if dim == 3:
                C = np.reshape(
                    mat_change_of_basis_el.T * np.reshape(res[:, 3:6].T, -1),
                    (3, -1),
                ).T
            else:
                C = res[:, 2]

            return np.c_[F, C]  # np.hstack((F,C))

    def copy(self, new_id=""):
        """
        Return a raw copy of the assembly without keeping current state (internal variable).

        Parameters
        ----------
        new_id : TYPE, optional
            The name of the created constitutive law. The default is "".

        Returns
        -------
        The copy of the assembly
        """
        return Assembly(self.weakform, self.mesh, self.elm_type, new_id)

    @property
    def n_gauss_points(self):
        """
        Returns
        -------
        int
            The total number of integration points (ie Gauss points) associated to the assembly.
            n_gauss_points is the total number of Gauss points whereas n_elm_gp gives only he numbre of gauss points per element:
            n_gauss_points = mesh.n_elements + assembly.n_elm_gp.
        """
        return self.mesh.n_elements * self.n_elm_gp

    @property
    def state_variables(self):
        """Alias for the sv dict containing the state variables."""
        return self.sv

    @staticmethod
    def sum(*listAssembly, name="", **kargs):
        """
        Return a new assembly which is a sum of N assembly.
        Assembly.sum(assembly1, assembly2, ..., assemblyN, name ="", reload = [1,4] )

        The N first arguments are the assembly to be summed.
        name is the name of the created assembly:
        reload: a list of indices for subassembly that are recomputed at each time the summed assembly
        is Launched. Default is 'all' (equivalent to all indices).
        """
        return AssemblySum(list(listAssembly), name, **kargs)

    @staticmethod
    def create(weakform, mesh="", elm_type="", name="", **kargs):
        if isinstance(weakform, str):
            weakform = WeakFormBase[weakform]

        if isinstance(mesh, str):
            mesh = Mesh[mesh]
        if not type(mesh) == Mesh:
            if hasattr(mesh, "mesh_dict"):
                raise TypeError(
                    "Can't create an assembly based on a MultiMesh object. "
                    "For that purpose, create separated assemblies for each "
                    "element type and sum them together."
                )
            else:
                raise TypeError("mesh should refers to a fedoo.Mesh object")

        if (
            hasattr(weakform, "list_weakform") and weakform.assembly_options is None
        ):  # WeakFormSum object
            list_weakform = weakform.list_weakform

            # get lists of some non compatible assembly_options items for each weakform in list_weakform
            list_elm_type = [
                elm_type
                if elm_type != ""
                else wf.assembly_options.get("elm_type", mesh.elm_type, mesh.elm_type)
                for wf in list_weakform
            ]
            list_n_elm_gp = [
                wf.assembly_options.get(
                    "n_elm_gp", list_elm_type[i], get_default_n_gp(elm_type, mesh)
                )
                for i, wf in enumerate(list_weakform)
            ]
            list_assume_sym = [
                wf.assembly_options.get("assume_sym", list_elm_type[i], False)
                for i, wf in enumerate(list_weakform)
            ]
            list_prop = list(zip(list_n_elm_gp, list_assume_sym, list_elm_type))
            list_diff_prop = list(
                set(list_prop)
            )  # list of different non compatible properties that required separated assembly

            if len(list_diff_prop) == 1:  # only 1 assembly is required
                # update assembly_options
                prop = list_diff_prop[0]
                elm_type = prop[2]
                weakform.assembly_options = _AssemblyOptions()
                weakform.assembly_options["n_elm_gp", elm_type] = prop[0]
                weakform.assembly_options["assume_sym", elm_type] = prop[1]
                weakform.assembly_options["mat_lumping", elm_type] = [
                    wf.assembly_options.get("mat_lumping", elm_type, False)
                    for wf in weakform.list_weakform
                ]
                return Assembly(weakform, mesh, elm_type, name, **kargs)

            else:  # we need to create and sum several assemblies
                list_assembly = []
                for prop in list_diff_prop:
                    l_wf = [
                        list_weakform[i] for i, p in enumerate(list_prop) if p == prop
                    ]  # list_weakform with compatible properties
                    elm_type = prop[2]
                    if len(l_wf) == 1:
                        wf = l_wf[0]  # standard weakform. No WeakFormSum required
                    else:
                        # create a new WeakFormSum object
                        wf = WeakFormSum(l_wf)
                        # define the assembly_options of the new weakform
                        wf.assembly_options = _AssemblyOptions()
                        wf.assembly_options["n_elm_gp", elm_type] = prop[0]
                        wf.assembly_options["assume_sym", elm_type] = prop[1]
                        wf.assembly_options["mat_lumping", elm_type] = [
                            w.assembly_options.get("mat_lumping", elm_type, False)
                            for w in l_wf
                        ]

                    list_assembly.append(Assembly(wf, mesh, elm_type, "", **kargs))
                    if list_weakform[0] in l_wf:
                        assembly_output = list_assembly[
                            -1
                        ]  # by default, the assembly used for output is the one associated to the 1st weakform

            # list_assembly = [Assembly(wf, mesh, elm_type, "", **kargs) for wf in weakform.list_weakform]
            kargs["assembly_output"] = kargs.get("assembly_output", assembly_output)
            return AssemblySum(list_assembly, name, **kargs)

        else:
            return Assembly(weakform, mesh, elm_type, name, **kargs)


def delete_memory():
    Assembly.delete_memory()
