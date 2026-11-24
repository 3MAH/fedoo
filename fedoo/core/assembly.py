from copy import copy
from numbers import Number

import numpy as np
from scipy import sparse

from fedoo.core._sparsematrix import (
    RowBlocMatrix,
    _WeightedCoef,
    _coo_to_csr,
    _csr_matvec,
    _csr_matvec_multi,
    _csr_rmatvec,
    _csr_rmatvec_multi,
)
from fedoo.core._sparsematrix import _BlocSparse as BlocSparse
from fedoo.core.assembly_sum import AssemblySum
from fedoo.core.base import AssemblyBase
from fedoo.core.diffop import DiffOp, _Derivative, _mul_coef
from fedoo.core.mesh import Mesh
from fedoo.core.weakform import WeakFormBase, _AssemblyOptions, WeakFormSum
from fedoo.lib_elements.element_list import (
    get_default_n_gp,
    get_element,
    get_node_elm_coordinates,
)
from fedoo.util.localframe import as_local_frame
from fedoo.util.voigt_tensors import StrainTensorList, StressTensorList


def _transposed_products(terms):
    """Matvir.T @ values for the (Matvir, values, ...) terms, the terms with
    the same operator computed together (see _csr_rmatvec_multi); None for a
    term left to the term by term computation of assemble_global_mat."""
    products = [None] * len(terms)
    groups = {}  # id(Matvir) -> (Matvir, [term index])
    for k, term in enumerate(terms):
        groups.setdefault(id(term[0]), (term[0], []))[1].append(k)
    for matrix, indices in groups.values():
        if len(indices) < 2:
            continue
        try:
            results = _csr_rmatvec_multi(matrix, [terms[k][1] for k in indices])
        except Exception:
            # term by term, as before (where an error leaves the term out)
            continue
        if results is not None:
            for k, res in zip(indices, results):
                products[k] = res
    return products


_LOCAL_DOF_COLS = {}  # (n_elements, n_elm_gp, n_elm_nodes) -> col, the last few


def _local_dof_col(n_elements, n_elm_gp, n_elm_nodes):
    """Column (local dof e + node * n_elements) of the coo entries of the
    elementary operators with a change of basis. The same array for the same
    sizes, so that its conversion to csr is cached (see _coo_to_csr)."""
    key = (n_elements, n_elm_gp, n_elm_nodes)
    col = _LOCAL_DOF_COLS.get(key)
    if col is None:
        col = np.empty((n_elements, n_elm_gp, n_elm_nodes))
        col[:] = (
            np.arange(n_elements).reshape((-1, 1, 1))
            + np.arange(n_elm_nodes).reshape((1, 1, -1)) * n_elements
        )
        col = col.reshape(-1)
        col.flags.writeable = False
        if len(_LOCAL_DOF_COLS) >= 8:
            _LOCAL_DOF_COLS.pop(next(iter(_LOCAL_DOF_COLS)))
        _LOCAL_DOF_COLS[key] = col
    return col


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
    # Associated-variable ranks depend on both the element interpolation and
    # the ModelingSpace in which those variables were declared.
    _saved_associated_variables = {}
    # index of a space coordinate among the mesh coordinates (see
    # _mesh_coordinate_index)
    _saved_coordinate_index = {}

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

        self.mesh_change = kargs.pop("mesh_change", False)
        if elm_type == "" or elm_type is None:
            elm_type = weakform.assembly_options.get(
                "elm_type", mesh.elm_type, mesh.elm_type
            )  # change elm_type if it was specified in assembly_options
        self.elm_type = elm_type.lower()
        self.mesh = mesh

        self.n_elm_gp = kargs.pop("n_elm_gp", None)
        if self.n_elm_gp is None:
            self.n_elm_gp = weakform.assembly_options.get(
                "n_elm_gp", elm_type, get_default_n_gp(elm_type, mesh)
            )

        # ``None`` means that the weak form has not resolved this option yet.
        # Formulation-dependent defaults are selected during initialization;
        # an explicit boolean assigned before then is preserved.
        self.assume_sym = weakform.assembly_options.get("assume_sym", elm_type)
        self.mat_lumping = weakform.assembly_options.get("mat_lumping", elm_type, False)

        self._use_local_csys = self._test_if_local_csys()
        self._element_local_frame = None
        self._initial_element_local_frame = None

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

        self.sv_component = {}
        # allow to define output accessible with problem.get_results.
        # for instance: sv_component[component_name] = (sv_name, indice)
        # indice may be an int or a slice

        self._nlgeom = None

        self._pb = None

    def set_element_local_frame(
        self,
        frames=None,
        location="Element",
        *,
        guide=None,
        guide_direction=None,
    ):
        """Define the initial local frame of beam or shell elements.

        The element geometry fixes one axis of the frame. For a beam, the
        local x axis is aligned with the element tangent. For a shell, the
        local z axis is aligned with the surface normal. Consequently, a
        supplied frame does not override these geometrical directions; it
        determines the rotation of the remaining two axes around them.

        ``guide`` provides a simpler way to define this remaining orientation.
        For beams it approximates the local y axis by default, or the local z
        axis with ``guide_direction="z"``. For shells it approximates the
        local x axis. The guide is projected onto the plane perpendicular to
        the geometrically fixed axis, and the last axis is then completed to
        form an orthonormal, right-handed frame.

        ``frames`` and ``guide`` may be uniform, elemental, or nodal. Nodal
        values are interpolated to the element centres before the geometrical
        projection is applied.

        This method must be called before problem initialization. A Gauss-point
        field is intentionally not accepted because an element rigid frame is
        independent of the integration rule.
        """
        if self._pb is not None:
            raise RuntimeError(
                "element local frames must be set before problem initialization"
            )
        if frames is not None and guide is not None:
            raise ValueError("provide either frames or guide, not both")
        if frames is None and guide is None:
            self._initial_element_local_frame = None
            self._element_local_frame = None
            return self

        if location not in ("Node", "Element"):
            raise ValueError("location must be 'Node' or 'Element'")

        if frames is not None:
            frames = np.asarray(as_local_frame(frames, self.mesh.ndim))
            frames = frames.reshape(-1, self.mesh.ndim, self.mesh.ndim)
            expected = self.mesh.n_nodes if location == "Node" else self.mesh.n_elements
            if len(frames) not in (1, expected):
                raise ValueError(
                    f"{location} local frames require one or {expected} matrices, "
                    f"got {len(frames)}"
                )
            guide_axis = 2 if guide_direction == "z" and self.mesh.ndim == 3 else 1
            if len(frames) == 1:
                frame_guide = frames[0, guide_axis]
                guide_location = None
            else:
                frame_guide = frames[:, guide_axis]
                guide_location = location

            # Shell frames use their x axis as guide; beam frames use y (or z).
            elm_ref = get_element(self.mesh.elm_type)
            if hasattr(elm_ref, "geometry_elm"):
                elm_ref = elm_ref.geometry_elm
            elm_ref = elm_ref(1)
            topology_dim = np.asarray(
                elm_ref.shape_function_derivative(elm_ref.get_gp_elm_coordinates(1))
            ).shape[1]
            if topology_dim == 2 and self.mesh.ndim == 3:
                frame_guide = frames[0, 0] if len(frames) == 1 else frames[:, 0]
                guide_direction = "x"
            elif topology_dim != 1:
                raise ValueError(
                    "element local frames are only supported for beam and shell meshes"
                )
        else:
            frame_guide = guide
            guide_location = location

        resolved = self.mesh.get_element_local_frame(
            guide=frame_guide,
            location=guide_location,
            guide_direction=guide_direction,
        )
        stored = np.asarray(resolved)[:, np.newaxis, :, :]
        self._initial_element_local_frame = stored.copy()
        self._element_local_frame = stored.copy()
        return self

    def get_element_local_frame(self):
        """Return the current rigid frame, one rotation matrix per element."""
        if self._element_local_frame is None:
            return self.mesh.get_element_local_frame()
        frames = np.asarray(self._element_local_frame)
        if frames.ndim == 4:
            if frames.shape[1] != 1:
                raise ValueError("an element rigid frame must be unique per element")
            return frames[:, 0]
        return frames

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
            self.mesh_change == True
        ):  # only node position change is considered here. For change of sparsity, use also, self.mesh.init_interpolation
            for k in tuple(Assembly._saved_change_of_basis_mat):
                if k[0] == self.mesh:
                    Assembly._saved_change_of_basis_mat.pop(k)
            self.compute_elementary_operators()

        nvar = self.space.nvar
        wf = self.weakform.get_weak_equation(self, self._pb)
        if wf == 0:
            if compute != "vector":
                self.global_matrix = 0
            if compute != "matrix":
                self.global_vector = 0
            return

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
                use_change_of_basis = False
            else:
                n_bloc_cols = self.mesh.n_elements * self.mesh.n_elm_nodes
                use_change_of_basis = True
            sl = [slice(i * n_bloc_cols, (i + 1) * n_bloc_cols) for i in range(nvar)]

            # total number of dof for the operator (with pb global dof)
            if use_change_of_basis or self._pb is None or self._pb.n_global_dof == 0:
                n_dof = n_bloc_cols * nvar
                global_mat_resize = False
            else:
                n_dof = n_bloc_cols * nvar + self._pb.n_global_dof
                global_mat_resize = True  # resize to account for global dof

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
                        coef_PG = (
                            coef_PG + wf.coef[ii]
                        )  # not in place: coefs may be shared
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
                        if np.array_equal(VV, 0):
                            VV = np.zeros(n_dof)
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
                        (
                            b
                            if b is not None
                            else sparse.csr_matrix(
                                (self.mesh.n_nodes, self.mesh.n_nodes)
                            )
                        )
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
                vector_terms = []  # (Matvir, coef, var_vir, coef_vir) of VV

                sum_coef = (
                    False  # bool that indicate if operator are the same and can be sum
                )
                weighted_coefs = {}  # id(coef) -> (coef, its _WeightedCoef)

                if hasattr(self.weakform, "_list_mat_lumping"):
                    change_mat_lumping = (
                        True  # use different mat_lumping option for each operator
                    )
                else:
                    change_mat_lumping = False
                    mat_lumping = self.mat_lumping

                # operators equal to 1 (virtual operator only: vector terms)
                op_is_one = [op == 1 for op in wf.op]
                gauss_weights = mat_gaussian_quadrature.data

                for ii in range(len(wf.op)):
                    if compute == "matrix" and op_is_one[ii]:
                        continue
                    if compute == "vector" and not op_is_one[ii]:
                        continue

                    if (
                        not op_is_one[ii]
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
                        coef_PG = self.mesh.data_to_gausspoint(wf.coef[ii], n_elm_gp)

                    # if ii > 0 and intRef[ii] == intRef[ii-1]: #if same operator as previous with different coef, add the two coef
                    if sum_coef:  # if same operator as previous with different coef, add the two coef
                        coef_parts.append(coef_PG)  # summed when weighted
                        sum_coef = False
                    else:
                        coef_parts = [coef_PG]

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

                    # sum of coef_parts times the gauss weights, computed when
                    # read (or by the numba kernel); a shared coef: once
                    if len(coef_parts) == 1:
                        coef_PG_sum = coef_parts[0]
                        weighted = weighted_coefs.get(id(coef_PG_sum))
                        if weighted is not None and weighted[0] is coef_PG_sum:
                            coef_PG = weighted[1]
                        else:
                            coef_PG = _WeightedCoef(coef_parts, gauss_weights)
                            if not np.isscalar(coef_PG_sum):
                                weighted_coefs[id(coef_PG_sum)] = (coef_PG_sum, coef_PG)
                    else:
                        coef_PG = _WeightedCoef(coef_parts, gauss_weights)
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
                        and not op_is_one[ii]
                        and not op_is_one[ii + 1]
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

                    if op_is_one[ii]:  # only virtual operator -> compute a vector
                        if np.isscalar(VV):  # VV = 0
                            VV = np.zeros(n_dof)
                        # the products are computed after the loop, in this order
                        values = coef_PG.values()
                        for i in range(len(Matvir)):
                            vector_terms.append(
                                (Matvir[i], values, var_vir[i], coef_vir[i])
                            )

                    else:  # virtual and real operators -> compute a matrix
                        coef = [1]
                        var = [wf.op[ii].u]  # list in case there is an angular variable
                        if var[0] in associatedVariables:
                            var.extend(associatedVariables[var[0]][0])
                            coef.extend(associatedVariables[var[0]][1])

                        Mat = self._get_elementary_operator(wf.op[ii])

                        # Possibility to increase performance for multivariable case
                        # the structure should be the same for derivative dof, so the blocs could be computed altogether
                        if listMatvir is None:
                            for i in range(len(Mat)):
                                for j in range(len(Matvir)):
                                    MM.addToBlocATB(
                                        Matvir[j],
                                        Mat[i],
                                        _mul_coef(coef[i] * coef_vir[j], coef_PG),
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
                                            _mul_coef(coef[i] * coef_vir[j], coef_PG)
                                            for coef_PG in listCoef_PG
                                        ],
                                        var_vir[j],
                                        var[i],
                                        mat_lumping,
                                    )
                            listMatvir = None
                            listCoef_PG = None

                # VV[sl[var_vir]] -= coef_vir * Matvir.T @ coef, in the order of
                # the terms; the products with the same operator together
                products = _transposed_products(vector_terms)
                for (Matvir_i, values, var_vir_i, coef_vir_i), product in zip(
                    vector_terms, products
                ):
                    try:
                        if product is None:
                            product = _csr_rmatvec(Matvir_i, values)
                        VV[sl[var_vir_i]] -= _mul_coef(coef_vir_i, product)
                    except:
                        pass

            if compute != "vector":
                if use_change_of_basis:
                    global_matrix = None
                    if isinstance(MM, BlocSparse):
                        # element matrices rotated element by element
                        global_matrix = MM._tocsr_change_of_basis(
                            mat_change_of_basis, self.mesh.n_nodes
                        )
                    if global_matrix is None:
                        global_matrix = (
                            mat_change_of_basis.T * MM.tocsr() * mat_change_of_basis
                        )
                    self.global_matrix = global_matrix  # format csr
                else:
                    self.global_matrix = MM.tocsr()  # format csr
                    if global_mat_resize:
                        self.global_matrix.resize(n_dof, n_dof)
            if compute != "matrix":
                if np.array_equal(VV, 0):
                    self.global_vector = 0
                elif use_change_of_basis:
                    self.global_vector = mat_change_of_basis.T * VV
                else:
                    self.global_vector = VV  # numpy array

            if self._saved_bloc_structure is None:
                self._saved_bloc_structure = MM.get_BlocStructure()

        elif _assembly_method == "very_old":
            # Only for debug purpose. No optim. Dont work with global dof. Not tested
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
                        coef_PG = self.mesh.data_to_gausspoint(wf.coef[ii], n_elm_gp)
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

        change_of_basis_key = self._change_of_basis_cache_key()
        if change_of_basis_key in Assembly._saved_change_of_basis_mat:
            return Assembly._saved_change_of_basis_mat[change_of_basis_key]

        mat_change_of_basis = 1
        compute_mat_change_of_basis = False

        n_nd = mesh.n_nodes
        n_el = mesh.n_elements
        elm = mesh.elements
        n_elm_nodes = np.shape(elm)[1]
        dim = self.space.ndim

        if (
            "X" in mesh.crd_name and "Y" in mesh.crd_name
        ):  # if not in physical space, no change of variable
            for nameVector in self.space.list_vectors():
                # if compute_mat_change_of_basis == False and len(self.space.get_vector(nameVector))>1:
                if compute_mat_change_of_basis == False:
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
                    local_frame_el = elmRefGeom.get_local_frame(
                        mesh.nodes[mesh._elements_geom], xi_nd
                    )
                    #### warninge ! to improve
                    # local_frame_el.shape is (n_el, nb_gp, dim, dim)
                    # but array of shape (n_el, nb_nd, nb vectors in basis = dim, dim)
                    # is expected since the local frame is applied at node dofs.
                    # work for planar element but not for element with curvature.
                    # Simple idea: use interpolation at nodes with n_elm_gp = 0
                    ############################
                else:
                    local_frame_el = self._element_local_frame

                if self._pb is not None:
                    n_global_dof = self._pb.n_global_dof
                else:
                    n_global_dof = 0

                dataMCB = np.empty(
                    (len(listGlobalVector) * n_el, n_elm_nodes, dim, dim)
                )
                for ivec, vec in enumerate(listGlobalVector):
                    # dataMCB[ivec*n_el:(ivec+1)*n_el] = local_frame_el[:,:,:dim,:dim]
                    dataMCB[ivec * n_el : (ivec + 1) * n_el] = local_frame_el
                dataMCB = dataMCB.reshape(-1)
                if len(listScalarVariable) > 0:
                    # add the component from scalar variables (ie variable not requiring a change of basis)
                    dataMCB = np.hstack(
                        (
                            dataMCB,
                            np.ones(len(listScalarVariable) * n_el * n_elm_nodes),
                        )
                    )  # no change of variable so only one value adding in dataMCB

                rowMCB, colMCB = self._change_of_basis_indices(
                    mesh, listGlobalVector, listScalarVariable, dim
                )
                # csr conversion of the coo matrix, cached for the same indices
                mat_change_of_basis = _coo_to_csr(
                    dataMCB,
                    rowMCB,
                    colMCB,
                    (n_el * n_elm_nodes * nvar, n_nd * nvar + n_global_dof),
                )

        Assembly._saved_change_of_basis_mat[change_of_basis_key] = mat_change_of_basis
        return mat_change_of_basis

    @staticmethod
    def _change_of_basis_indices(mesh, list_vectors, list_scalars, dim):
        """Row and column indices of the coo entries of the change of basis
        matrix (the values change with the local frames). Kept on the mesh,
        for the same variables, while its element table is unchanged (a copy
        is compared, so that a table modified in place is detected)."""
        elm, n_nd = mesh.elements, mesh.n_nodes
        key = (
            n_nd,
            tuple(tuple(vec) for vec in list_vectors),
            tuple(list_scalars),
            dim,
        )
        saved = getattr(mesh, "_saved_change_of_basis_indices", None)
        if saved is None:
            saved = mesh._saved_change_of_basis_indices = {}
        cached = saved.get(key)
        if cached is not None and np.array_equal(cached[0], elm):
            return cached[1], cached[2]

        n_el, n_elm_nodes = elm.shape
        range_n_elm_nodes = np.arange(n_elm_nodes)
        rowMCB = np.empty((len(list_vectors) * n_el, n_elm_nodes, dim, dim))
        colMCB = np.empty((len(list_vectors) * n_el, n_elm_nodes, dim, dim))
        for ivec, vec in enumerate(list_vectors):
            rowMCB[ivec * n_el : (ivec + 1) * n_el] = (
                np.arange(n_el).reshape(-1, 1, 1, 1)
                + range_n_elm_nodes.reshape(1, -1, 1, 1) * n_el
                + np.array(vec).reshape(1, 1, -1, 1) * (n_el * n_elm_nodes)
            )
            colMCB[ivec * n_el : (ivec + 1) * n_el] = (
                elm.reshape(n_el, n_elm_nodes, 1, 1)
                + np.array(vec).reshape(1, 1, 1, -1) * n_nd
            )
        rowMCB, colMCB = rowMCB.reshape(-1), colMCB.reshape(-1)

        if len(list_scalars) > 0:
            # variables that don't require a change of basis
            rowMCB_loc = np.empty((len(list_scalars) * n_el, n_elm_nodes))
            colMCB_loc = np.empty((len(list_scalars) * n_el, n_elm_nodes))
            for ivar, var in enumerate(list_scalars):
                rowMCB_loc[ivar * n_el : (ivar + 1) * n_el] = (
                    np.arange(n_el).reshape(-1, 1)
                    + range_n_elm_nodes.reshape(1, -1) * n_el
                    + var * (n_el * n_elm_nodes)
                )
                colMCB_loc[ivar * n_el : (ivar + 1) * n_el] = elm + var * n_nd
            rowMCB = np.hstack((rowMCB, rowMCB_loc.reshape(-1)))
            colMCB = np.hstack((colMCB, colMCB_loc.reshape(-1)))

        rowMCB.flags.writeable = False
        colMCB.flags.writeable = False
        saved[key] = (np.array(elm), rowMCB, colMCB)
        return rowMCB, colMCB

    def _change_of_basis_cache_key(self):
        """Return the cache key for the assembly change-of-basis matrix."""
        n_global_dof = 0 if self._pb is None else self._pb.n_global_dof
        return (
            self.mesh,
            self.space,
            self._use_local_csys,
            self.mesh._local_frame_cache_key(self._element_local_frame),
            n_global_dof,
        )

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
        # in principle, need to compute 'matrix' only if tangent stiffness has been updated to elastic.
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
        Initial values set with set_initial_statev() are deleted and must be
        reapplied before the next initialization.
        """
        self.weakform.reset()
        if self.weakform.constitutivelaw is not None:
            self.weakform.constitutivelaw.reset()
        self.delete_global_mat()
        # self.current.delete_global_mat()
        self.current = self
        self._element_local_frame = (
            None
            if self._initial_element_local_frame is None
            else self._initial_element_local_frame.copy()
        )

        # remove all state variables
        self._initial_statev_components = set()
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

        Note: if the mesh_change argument is set to True when creating the Assembly object, the
        memory will be recomputed by default, which may cause a decrease in assembling performances
        """
        Assembly._saved_elementary_operators = {}
        Assembly._saved_change_of_basis_mat = {}
        # Assembly._saved_gaussian_quadrature_mat = {}
        # Assembly._saved_node2gausspoint_mat = {}
        # Assembly._saved_gausspoint2node_mat = {}
        Assembly._saved_associated_variables = {}
        Assembly._saved_coordinate_index = {}

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
            mesh._saved_gaussian_quadrature_mat[
                mesh._gaussian_quadrature_cache_key(n_elm_gp)
            ] = sparse.identity(
                OP[0, 0][0].shape[0], "d", format="csr"
            )  # No gaussian quadrature in this case : nodal identity matrix
            mesh._saved_gausspoint2node_mat[n_elm_gp] = (
                1  # no need to translate between pg and nodes because no pg
            )
            mesh._saved_node2gausspoint_mat[n_elm_gp] = 1
            Assembly._saved_change_of_basis_mat[self._change_of_basis_cache_key()] = (
                1  # No change of basis: mat_change_of_basis = 1
            )
            Assembly._saved_elementary_operators[
                (
                    mesh,
                    elm_type,
                    n_elm_gp,
                    self._use_local_csys,
                    mesh._local_frame_cache_key(self._element_local_frame),
                )
            ] = OP
            return

        # -------------------------------------------------------------------
        # Initialize the geometrical interpolation (gaussian quadrature, jacobian matrix, ...)
        # -------------------------------------------------------------------
        if n_elm_gp not in mesh._elm_interpolation:
            mesh.init_interpolation(n_elm_gp)

        mesh._compute_gaussian_quadrature_mat(n_elm_gp, self._element_local_frame)
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
            col = _local_dof_col(n_elements, n_elm_gp, n_elm_nodes)
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
            if isinstance(elmRef.shape_function_gp, list):
                n_diff_interpolations = len(elmRef.shape_function_gp)
                shape_functions = elmRef.shape_function_gp
                NbDoFperNode = elmRef.shape_function_gp[0].shape[-1] // n_interpol_nodes
            else:
                shape_functions = [elmRef.shape_function_gp]
                NbDoFperNode = elmRef.shape_function_gp.shape[-1] // n_interpol_nodes
            # end special treatment

            nb_dir_deriv = 0
            if hasattr(elmRef, "shape_function_derivative_gp"):
                inv_jacobian_matrix = elmRefGeom.inv_jacobian_matrix
                if elmRef.use_center_jacobian:
                    # Reduced interpolation operators must use the geometry
                    # transformation evaluated at the element center.
                    mesh._compute_gaussian_quadrature_mat(1, self._element_local_frame)
                    inv_jacobian_matrix = mesh._elm_interpolation[1].inv_jacobian_matrix
                op_dd = self._elementary_operators_numba(
                    elmRef,
                    inv_jacobian_matrix,
                    shape_functions,
                    NbDoFperNode,
                    n_interpol_nodes,
                    n_elm_nodes,
                    n_elm_gp,
                    row,
                    col,
                    (n_elements * n_elm_gp, n_col),
                    np.isscalar(mat_change_of_basis) and mat_change_of_basis == 1,
                )
                if op_dd is not None:  # computed in csr order by a numba kernel
                    data = {(0, 0): op_dd[0]}
                    for i in range(len(op_dd) - 1):
                        data[1, i] = op_dd[i + 1]
                    Assembly._saved_elementary_operators[
                        (
                            mesh,
                            elm_type.name,
                            n_elm_gp,
                            self._use_local_csys,
                            mesh._local_frame_cache_key(self._element_local_frame),
                        )
                    ] = data
                    continue
                derivativePG = inv_jacobian_matrix @ elmRef.shape_function_derivative_gp
                nb_dir_deriv = derivativePG.shape[-2]
            nop = (
                nb_dir_deriv + n_diff_interpolations
            )  # nombre d'opérateur à discrétiser

            # zeros, not empty: a sub-element of a CombinedElement (linear
            # pressure on a quadratic mesh) fills only its n_interpol_nodes
            # columns, the others must be exact zeros.
            data = [
                [
                    np.zeros((n_elements, n_elm_gp, n_elm_nodes))
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
                    _coo_to_csr(
                        data[i][j].reshape(-1),
                        row,
                        col,
                        (n_elements * n_elm_gp, n_col),
                    )
                    for j in range(NbDoFperNode)
                ]
                for i in range(nop)
            ]

            data = {(0, i): op_dd[i] for i in range(n_diff_interpolations)}
            for i in range(nb_dir_deriv):
                data[1, i] = op_dd[
                    i + n_diff_interpolations
                ]  # as index and indptr should be the same, perhaps it will be more memory efficient to only store the data field

            Assembly._saved_elementary_operators[
                (
                    mesh,
                    elm_type.name,
                    n_elm_gp,
                    self._use_local_csys,
                    mesh._local_frame_cache_key(self._element_local_frame),
                )
            ] = data

    @staticmethod
    def _elementary_operators_numba(
        elmRef,
        inv_jacobian_matrix,
        shape_functions,
        n_dof_per_node,
        n_interpol_nodes,
        n_elm_nodes,
        n_elm_gp,
        row,
        col,
        shape,
        no_change_of_basis,
    ):
        """Elementary operators [N, d/dx_0, d/dx_1, ...] (one list item per
        operator, each a list over the dof of a node) computed directly in
        csr order by a numba kernel, for the common case: one interpolation,
        one dof per node, all the element nodes interpolated, no change of
        basis, coo indices without duplicate. None otherwise (or without
        numba), and the general path is used."""
        from fedoo.core._sparsematrix import (
            _elementary_operator_values_numba,
            _csr_pattern,
            _csr_from_pattern,
            _numba_threads,
        )

        if _elementary_operator_values_numba is None:
            return None
        if not (
            no_change_of_basis
            and n_dof_per_node == 1
            and len(shape_functions) == 1
            and n_interpol_nodes == n_elm_nodes
            and not elmRef.use_center_jacobian
        ):
            return None
        dshape = np.asarray(elmRef.shape_function_derivative_gp)
        inv_jac = np.asarray(inv_jacobian_matrix)
        if (
            dshape.ndim != 3
            or dshape.shape[0] != n_elm_gp
            or dshape.shape[2] != n_elm_nodes
            or inv_jac.ndim != 4
            or inv_jac.shape[1] != n_elm_gp
            or inv_jac.shape[3] != dshape.shape[1]
        ):
            return None
        n_elm = inv_jac.shape[0]
        sf = np.asarray(shape_functions[0])
        if sf.shape[-1] != n_elm_nodes or sf.size % (n_elm_gp * n_elm_nodes):
            return None
        sf = sf.reshape(-1, n_elm_gp, n_elm_nodes)
        if sf.shape[0] not in (1, n_elm) or len(row) != n_elm * n_elm_gp * n_elm_nodes:
            return None
        pattern = _csr_pattern(row, col, shape)
        if not pattern.valid:
            return None
        # one array per operator: a row of a 2d array would be copied by
        # scipy (data much smaller than its base)
        out = tuple(np.empty(len(pattern.perm)) for _ in range(1 + inv_jac.shape[2]))
        with _numba_threads(len(pattern.perm) * len(out)):
            _elementary_operator_values_numba(
                np.ascontiguousarray(inv_jac, dtype=np.float64),
                np.ascontiguousarray(dshape, dtype=np.float64),
                np.ascontiguousarray(sf, dtype=np.float64),
                pattern.perm,
                n_elm_gp,
                n_elm_nodes,
                out,
            )
        return [[_csr_from_pattern(values, pattern)] for values in out]

    def _get_elementary_operator(self, deriv, n_elm_gp=None):
        # Gives a list of sparse matrix that convert node values for one variable to the pg values of a simple derivative op (for instance d/dz)
        # The list contains several element if the elm_type include several variable (dof variable in beam element). In other case, the list contains only one matrix
        # The variables are not considered. For a global use, the resulting matrix should be assembled in a block matrix with the nodes values for all variables
        if n_elm_gp is None:
            n_elm_gp = self.n_elm_gp

        elm_type = self.elm_type
        mesh = self.mesh

        element = get_element(elm_type)
        if hasattr(element, "get_elm_type"):
            elm_type = element.get_elm_type(deriv.u_name).name

        key = (
            mesh,
            elm_type,
            n_elm_gp,
            self._use_local_csys,
            mesh._local_frame_cache_key(self._element_local_frame),
        )
        data = Assembly._saved_elementary_operators.get(key)
        if data is None:
            self.compute_elementary_operators(n_elm_gp)
            data = Assembly._saved_elementary_operators[key]

        if deriv.ordre == 0:
            # in this case deriv.x should be 0 execpt if several interpolations
            # are used. In this case, deriv.x is and arbitrary id not related
            # to any coordinate
            xx = deriv.x
        else:
            # extract the mesh coordinate that corespond to coordinate rank given in deriv.x
            xx = self._mesh_coordinate_index(deriv.x)
            if xx is None:
                # for PGD only (will be probably deprecated):
                # if the coordinate doesnt exist, return operator without
                # derivation ()
                return data[0, 0]
        if (deriv.ordre, xx) in data:
            return data[deriv.ordre, xx]
        else:
            assert 0, "Operator unavailable"

    def _mesh_coordinate_index(self, x):
        """Index, among the mesh coordinates that are coordinates of the
        modeling space, of the coordinate of rank x; None if absent. Cached
        by (mesh coordinate names, space, number of space coordinates, x)."""
        key = (self.mesh.crd_name, self.space, self.space.ncrd, x)
        try:
            return Assembly._saved_coordinate_index[key]
        except KeyError:
            pass
        except TypeError:  # unhashable key: no cache
            key = None
        ranks = [
            self.space.coordinate_rank(crdname)
            for crdname in self.mesh.crd_name
            if crdname in self.space.list_coordinates()
        ]
        index = ranks.index(x) if x in ranks else None
        if key is not None:
            Assembly._saved_coordinate_index[key] = index
        return index

    def _get_gaussian_quadrature_mat(
        self,
    ):  # calcul la discrétision relative à un seul opérateur dérivé
        cache_key = self.mesh._gaussian_quadrature_cache_key(
            self.n_elm_gp, self._element_local_frame
        )
        if cache_key not in self.mesh._saved_gaussian_quadrature_mat:
            self.compute_elementary_operators()
        return self.mesh._saved_gaussian_quadrature_mat[cache_key]

    def _get_associated_variables(
        self,
    ):  # associated variables (rotational dof for C1 elements) of elm_type
        # based on variable rank, ie dont make differences between alias
        elm_type = self.elm_type
        cache_key = (elm_type, self.space)
        if cache_key not in Assembly._saved_associated_variables:
            objElement = get_element(elm_type)
            if hasattr(objElement, "associated_variables"):
                Assembly._saved_associated_variables[cache_key] = {
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
                Assembly._saved_associated_variables[cache_key] = {}
        return Assembly._saved_associated_variables[cache_key]

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
        # WARNING : work if the length of U is not consistent with n_gp
        # because we force the good shape with RowBlockMatrix
        res = 0
        nvar = self.space.nvar
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

            # RowBlocMatrix(ops, nvar, var, coef, n_cols=len(U)) @ U without the
            # bloc matrix: each operator acts on the values of its variable
            values = 0
            for op, v, c in zip(
                self._get_elementary_operator(operator.op[ii], n_elm_gp), var, coef
            ):
                n = op.shape[1]
                if np.isscalar(values) and isinstance(c, Number) and c == 1:
                    # a new array without -0. (sums from +0.): it is 0 + itself
                    values = _csr_matvec(op, U[v * n : (v + 1) * n])
                else:
                    values = values + _mul_coef(
                        c, _csr_matvec(op, U[v * n : (v + 1) * n])
                    )
            res += _mul_coef(coef_PG, values)

        return res  # equivalent to self._get_assembled_operator(operator, n_elm_gp) @ U but more optimized

    def _get_assembled_operator(self, operator, n_elm_gp=None):
        # not very optimized (sum of RowBlocMatrix use scipy add operator that is not efficient (change sparse structure))
        nvar = self.space.nvar

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

    def operator_apply(self, wf, nodal_vector, use_local_dof=False):
        """Apply a discrete nodal vector to the trial space of a bilinear operator.

        This function performs a contraction of a bilinear form a(v, u*) with
        a known field 'v'. It replaces the trial function (v) with the
        interpolated field defined by the nodal_vector, while keeping the
        virtual/test field (u*) symbolic.

        Parameters
        ----------
        Args:
        wf : DiffOp
            A bilinear opearator associated with a weakform that may be used
            to build a global matrix (e.g., stiffness or mass matrix).

        nodal_vector : array_like
            The discrete values (Degrees of Freedom) used to instantiate
            the trial field. By default, the vector is interpreted in the
            global nodal basis.
        use_local_dof : bool, default: False
            If True, nodal_vector is interpreted as local element dofs, ordered
            by variable blocks with ``n_elements * n_elm_nodes`` values each.

        Return:
            DiffOp : An operator containing only the virtual (test) operator,
                ready for vector assembly.
        """
        new_wf = type(wf)([], [], [])
        mat_change_of_basis = self.get_change_of_basis_mat()
        if use_local_dof:
            local_nodal_vector = nodal_vector
        elif np.isscalar(mat_change_of_basis) and mat_change_of_basis == 1:
            local_nodal_vector = nodal_vector
        else:
            n_nodal_dof = self.space.nvar * self.mesh.n_nodes
            n_global_dof = mat_change_of_basis.shape[1]
            if len(nodal_vector) == n_global_dof:
                local_nodal_vector = mat_change_of_basis @ nodal_vector
            elif len(nodal_vector) == n_nodal_dof and n_nodal_dof <= n_global_dof:
                padded_vector = np.zeros(n_global_dof)
                padded_vector[:n_nodal_dof] = nodal_vector
                local_nodal_vector = mat_change_of_basis @ padded_vector
            else:
                raise ValueError(
                    "nodal_vector length is inconsistent with this assembly: "
                    f"got {len(nodal_vector)}, expected {n_nodal_dof} or "
                    f"{n_global_dof}. Pass use_local_dof=True for local "
                    "element dofs."
                )

        associatedVariables = (
            self._get_associated_variables()
        )  # for element requiring many variable such as beam with disp and rot dof
        for ii in range(len(wf.op)):
            if np.isscalar(wf.op[ii]) and wf.op[ii] == 1:
                continue

            if ii == 0 or wf.op[ii] != wf.op[ii - 1]:
                # operator is not the same as previous. recompute
                var = [wf.op[ii].u]
                coef = [1]
                if var[0] in associatedVariables:
                    # add rot dof for instance
                    var.extend(associatedVariables[var[0]][0])
                    coef.extend(associatedVariables[var[0]][1])

                elementary_operator = self._get_elementary_operator(wf.op[ii])
                n_operator_cols = elementary_operator[0].shape[1]
                gp_values = (
                    elementary_operator[0]
                    @ local_nodal_vector[
                        var[0] * n_operator_cols : (var[0] + 1) * n_operator_cols
                    ]
                )
                if len(var) > 1:
                    for jj, v in enumerate(var[1:]):
                        gp_values += (
                            elementary_operator[jj + 1]
                            @ local_nodal_vector[
                                v * n_operator_cols : (v + 1) * n_operator_cols
                            ]
                        )
                # or:
                # gp_values = self.get_gp_results(type(wf)([wf.op[ii]], [1], [1]), nodal_vector)

            # add gp_values to wf.coef[ii]
            new_wf.op.append(1)
            new_wf.op_vir.append(wf.op_vir[ii])
            if np.isscalar(wf.coef[ii]) or len(wf.coef[ii]) == 1:
                # if n_elm_gp == 0, coef_PG = nodal values (finite diffirences)
                new_wf.coef.append(wf.coef[ii] * gp_values)
            else:  # coef is an array. convert to gp if needed.
                new_wf.coef.append(
                    self.mesh.data_to_gausspoint(wf.coef[ii][:]) * gp_values
                )

        return new_wf

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

    def convert_data(
        self, data, convert_from=None, convert_to="GaussPoint", method=None
    ):
        if isinstance(data, (StrainTensorList, StressTensorList)):
            return data.convert(self, convert_from, convert_to, method)

        return self.mesh.convert_data(
            data, convert_from, convert_to, self.n_elm_gp, method
        )

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
                new_mesh._saved_gausspoint2node_l2 = {}
                new_assembly = copy(self)
                new_assembly.mesh = new_mesh
                self.current = new_assembly
            else:
                self.current.mesh.nodes = new_crd
                self.current.mesh._saved_gaussian_quadrature_mat = {}
                self.current.mesh._saved_gausspoint2node_l2 = {}

                for k in tuple(self._saved_change_of_basis_mat):
                    if k[0] == self.current.mesh:
                        self._saved_change_of_basis_mat.pop(k)

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
            gp_results = self._gp_results_by_operator(grad_operator, U)
            return [
                [
                    (
                        # as get_node_results
                        self.mesh._get_gausspoint2node_mat(self.n_elm_gp)
                        @ gp_results[i][j]
                        if op != 0
                        else np.zeros(self.mesh.n_nodes)
                    )
                    for j, op in enumerate(line_op)
                ]
                for i, line_op in enumerate(grad_operator)
            ]

        elif Type == "Element":
            gp_results = self._gp_results_by_operator(grad_operator, U)
            n_elements = self.mesh.n_elements
            return [
                [
                    (
                        # as get_element_results
                        np.reshape(
                            gp_results[i][j],
                            (gp_results[i][j].shape[0] // n_elements, -1),
                        ).sum(0)
                        / (gp_results[i][j].shape[0] // n_elements)
                        if op != 0
                        else np.zeros(self.mesh.n_elements)
                    )
                    for j, op in enumerate(line_op)
                ]
                for i, line_op in enumerate(grad_operator)
            ]

        elif Type == "GaussPoint":
            gp_results = self._gp_results_by_operator(grad_operator, U)
            return [
                [
                    (gp_results[i][j] if op != 0 else np.zeros(self.n_gauss_points))
                    for j, op in enumerate(line_op)
                ]
                for i, line_op in enumerate(grad_operator)
            ]
        else:
            assert 0, "Wrong argument for Type: use 'Node', 'Element', or 'GaussPoint'"

    def _gp_results_by_operator(self, operators, U):
        """[[self.get_gp_results(op, U) for op in line] for line in operators]
        (None for the operators equal to 0). The simple operators (one
        derivative of a variable without associated variable, coefficient 1,
        no change of basis) that share an elementary operator are applied to
        their variables together, the operator being read once (numba): same
        values."""
        res = [[None for op in line] for line in operators]
        simple = {}  # id(elementary operator) -> (operator, [(i, j, values)])
        mat_change_of_basis = self.get_change_of_basis_mat()
        if np.isscalar(mat_change_of_basis) and mat_change_of_basis == 1:
            associated_variables = self._get_associated_variables()
            for i, line in enumerate(operators):
                for j, op in enumerate(line):
                    if (
                        isinstance(op, DiffOp)
                        and len(op.op) == 1
                        and isinstance(op.op[0], _Derivative)
                        and np.isscalar(op.op_vir[0])
                        and op.op_vir[0] == 1
                        and isinstance(op.coef[0], Number)
                        and op.coef[0] == 1
                        and op.op[0].u not in associated_variables
                    ):
                        matrix = self._get_elementary_operator(op.op[0])[0]
                        n = matrix.shape[1]
                        u = op.op[0].u
                        simple.setdefault(id(matrix), (matrix, []))[1].append(
                            (i, j, U[u * n : (u + 1) * n])
                        )
        for matrix, items in simple.values():
            values = _csr_matvec_multi(matrix, [x for _, _, x in items])
            if values is None:
                continue
            # get_gp_results: 0 + (a new matvec array, without -0.): itself
            for (i, j, _), v in zip(items, values):
                res[i][j] = v
        for i, line in enumerate(operators):
            for j, op in enumerate(line):
                if res[i][j] is None and op != 0:
                    res[i][j] = self.get_gp_results(op, U)
        return res

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
            dataMCB = elmRef.get_local_frame(
                crd[elm], elmRef.xi_pg
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

    def _save_sv(self):
        """shallow copy of state variables in _sv_saved attribute."""
        self._sv_saved = dict(self.sv)

    def _load_sv(self):
        """reload a saved copy of state variables."""
        if not hasattr(self, "_sv_saved"):
            raise RuntimeError("no saved state variables")
        # self.sv = dict(self._sv_saved)
        self.sv.clear()
        self.sv.update(self._sv_saved)
        del self._sv_saved

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
                (
                    elm_type
                    if elm_type != ""
                    else wf.assembly_options.get(
                        "elm_type", mesh.elm_type, mesh.elm_type
                    )
                )
                for wf in list_weakform
            ]
            list_n_elm_gp = [
                wf.assembly_options.get(
                    "n_elm_gp", list_elm_type[i], get_default_n_gp(elm_type, mesh)
                )
                for i, wf in enumerate(list_weakform)
            ]
            raw_assume_sym = [
                wf.assembly_options.get("assume_sym", list_elm_type[i])
                for i, wf in enumerate(list_weakform)
            ]
            # Preserve the historical grouping behavior: an unset option is
            # grouped with the conservative non-symmetric path. The resulting
            # assembly keeps ``None`` only when every grouped form is unset.
            list_assume_sym = [
                False if value is None else value for value in raw_assume_sym
            ]
            list_prop = list(zip(list_n_elm_gp, list_assume_sym, list_elm_type))
            list_diff_prop = []
            for prop in list_prop:
                if prop not in list_diff_prop:
                    list_diff_prop.append(prop)
            # list of different non compatible properties that require separated assembly

            if len(list_diff_prop) == 1:  # only 1 assembly is required
                # update assembly_options
                prop = list_diff_prop[0]
                elm_type = prop[2]
                weakform.assembly_options = _AssemblyOptions()
                weakform.assembly_options["n_elm_gp", elm_type] = prop[0]
                weakform.assembly_options["assume_sym", elm_type] = (
                    None if all(value is None for value in raw_assume_sym) else prop[1]
                )
                weakform.assembly_options["mat_lumping", elm_type] = [
                    wf.assembly_options.get("mat_lumping", elm_type, False)
                    for wf in weakform.list_weakform
                ]
                return Assembly(weakform, mesh, elm_type, name, **kargs)

            else:  # we need to create and sum several assemblies
                list_assembly = []
                for prop in list_diff_prop:
                    indices = [i for i, value in enumerate(list_prop) if value == prop]
                    l_wf = [
                        list_weakform[i] for i in indices
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
                        wf.assembly_options["assume_sym", elm_type] = (
                            None
                            if all(raw_assume_sym[i] is None for i in indices)
                            else prop[1]
                        )
                        wf.assembly_options["mat_lumping", elm_type] = [
                            w.assembly_options.get("mat_lumping", elm_type, False)
                            for w in l_wf
                        ]

                    list_assembly.append(Assembly(wf, mesh, elm_type, "", **kargs))
            # list_assembly = [Assembly(wf, mesh, elm_type, "", **kargs) for wf in weakform.list_weakform]
            return AssemblySum(list_assembly, name, **kargs)

        else:
            return Assembly(weakform, mesh, elm_type, name, **kargs)


def delete_memory():
    Assembly.delete_memory()
