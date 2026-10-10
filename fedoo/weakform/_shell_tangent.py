"""Analytical operators for the Jacobian of the corotational shell residual.

All columns refer to LOCAL spatial increments. Ordinary assembly applies the
current change of basis on both sides. No residual perturbations or separately
assembled stiffness corrections are used here.
"""

from dataclasses import dataclass

import numpy as np

from fedoo.core.diffop import DiffOp
from fedoo.lib_elements.element_list import CombinedElement, get_element, get_all
from fedoo.weakform._beam_tangent import _log_derivative, _skew

_VARIABLES = ("DispX", "DispY", "DispZ", "RotX", "RotY", "RotZ")
_BATCH_SIZE = 64


def initialize_shell_tangent(wf, assembly):
    """Add nodal evaluation aliases to a private interpolation variant.

    A frame/centroid variation couples all nodes, including nodes beyond an
    integration point's usual shape-function support. Constant nodal evaluation
    fields express these functionals as ordinary DiffOps. They share existing
    DOFs and do not change the physical shell interpolation or default elements.
    """
    element = get_element(assembly.elm_type)
    name = (
        element.name
        if element.name.endswith("_consistent")
        else element.name + "_consistent"
    )
    if name not in get_all():
        variant = CombinedElement(
            name,
            element.base_elm,
            default_n_gp=assembly.n_elm_gp,
            local_csys=True,
            n_nodes=element.n_nodes,
            dict_elm_type=dict(element.dict_elm_type),
            associated_variables=dict(element.associated_variables),
        )
        variant.geometry_elm = element.geometry_elm
        for node in range(element.n_nodes):

            def shape_function(self, xi, node=node):
                values = np.zeros((len(xi), self.n_nodes))
                values[:, node] = 1
                return values

            def __init__(self, n_elm_gp, **kwargs):
                element.geometry_elm.__init__(self, n_elm_gp)

            interpolation = type(
                f"_{name}_node_{node}",
                (element.geometry_elm,),
                {
                    "name": f"_{name}_node_{node}",
                    "__init__": __init__,
                    "shape_function": shape_function,
                },
            )
            for variable in _VARIABLES:
                variant.set_variable_interpolation(
                    f"_ShellNode{node}{variable}", interpolation
                )
    for node in range(element.n_nodes):
        for variable in _VARIABLES:
            wf.space.variable_alias(f"_ShellNode{node}{variable}", variable)
    assembly.elm_type = name
    assembly.assume_sym = False
    ne, nn = assembly.mesh.n_elements, assembly.mesh.n_elm_nodes
    assembly.sv["_ShellTangentDof"] = np.zeros((ne, nn * wf.space.nvar))
    # A fixed penalty scale avoids differentiating H (a material third derivative).
    stiffness = wf.constitutivelaw.get_shell_stiffness_matrix()[2][2]
    assembly.sv["_ShellDrillReference"] = np.array(
        assembly.convert_data(stiffness), copy=True
    )


def _nodal_fields(wf, n_nodes):
    """Reuse invariant nodal aliases; no coefficient or trial state is cached."""
    ranks = tuple(wf.space.variable_rank(variable) for variable in _VARIABLES)
    key = (n_nodes, wf.space.nvar, ranks)
    cache = getattr(wf, "_shell_nodal_fields", None)
    if cache is None:
        cache = wf._shell_nodal_fields = {}
    if key not in cache:
        columns = (np.asarray(ranks)[:, None] * n_nodes + np.arange(n_nodes)).ravel()
        fields = [0] * (n_nodes * wf.space.nvar)
        for variable, rank in zip(_VARIABLES, ranks):
            for node in range(n_nodes):
                fields[rank * n_nodes + node] = wf.space.variable(
                    f"_ShellNode{node}{variable}"
                )
        cache[key] = fields, columns
    return cache[key]


def _nodal_operator(wf, coefficients, n_nodes):
    """sum_j a_j u_j using aliases, in Gauss-major coefficient ordering."""
    fields, columns = _nodal_fields(wf, n_nodes)
    active = columns[np.any(coefficients, axis=(0, 1))[columns]]
    if not len(active):
        return 0
    # Reduce and reorder all nodal coefficients together rather than scanning
    # and flattening each node separately. Only Python operator objects require
    # a list comprehension; their invariant descriptors are already cached.
    values = coefficients[..., active].transpose(2, 1, 0).reshape(len(active), -1)
    operators = [fields[column].op[0] for column in active]
    return DiffOp(operators, [1] * len(operators), list(values))


def _normalize(vector, derivative):
    """Derivative of v/|v|; derivative shape (element, column, xyz)."""
    norm = np.linalg.norm(vector, axis=-1)
    if np.any(norm < 1e-12):
        raise ValueError(
            "Consistent shell tangent requires a nondegenerate element frame."
        )
    unit = vector / norm[:, None]
    projection = np.sum(unit[:, None] * derivative, axis=-1)
    return unit, (derivative - unit[:, None] * projection[..., None]) / norm[
        :, None, None
    ]


def _kinematics(wf, assembly, selected, q):
    """Differentiate centroid, geometric normal, fitted frame and local DOFs.

    For the in-plane fit, the optimal proper 2D rotation has angle
    atan2(C_21-C_12, C_11+C_22). Differentiate that rotation, not SVD vectors.
    This also covers the reflection correction in the existing frame update.
    """
    mesh = assembly.mesh
    element = get_element(assembly.elm_type)
    geometry = element.geometry_elm(1)
    frame = assembly.get_element_local_frame()[selected]
    nodes = mesh.nodes[mesh.elements[selected]]
    ne, nn = nodes.shape[:2]
    nd = nn * wf.space.nvar
    dx = np.zeros((ne, nd, nn, 3))
    theta = np.zeros_like(dx)
    ranks = np.asarray([wf.space.variable_rank(variable) for variable in _VARIABLES])
    columns = ranks.reshape(2, 3, 1) * nn + np.arange(nn)
    # Each nodal increment contributes the corresponding local-frame axis.
    # Advanced indexing fills all axes and nodes in two batched assignments.
    dx[:, columns[0], np.arange(nn)] = frame[:, :, None]
    theta[:, columns[1], np.arange(nn)] = frame[:, :, None]
    rel = nodes - nodes.mean(axis=1, keepdims=True)
    drel = dx - dx.mean(axis=2, keepdims=True)

    # Exact normal used by mesh.get_element_local_frame(): geometry at its center.
    dn = np.asarray(geometry.shape_function_derivative_gp)[0]
    jacobian = np.einsum("in,enj->eij", dn, nodes)
    djacobian = np.einsum("in,ecnj->ecij", dn, dx)
    normal = np.cross(jacobian[:, 0], jacobian[:, 1])
    dnormal = np.cross(djacobian[:, :, 0], jacobian[:, None, 1]) + np.cross(
        jacobian[:, None, 0], djacobian[:, :, 1]
    )
    z, dz = _normalize(normal, dnormal)
    guide = frame[:, 0]
    dguide = np.einsum("ei,ecij->ecj", guide, _skew(theta.mean(axis=2)))
    dot = np.sum(guide * z, axis=-1)
    ddot = np.sum(dguide * z[:, None] + guide[:, None] * dz, axis=-1)
    projected = guide - dot[:, None] * z
    dprojected = dguide - ddot[..., None] * z[:, None] - dot[:, None, None] * dz
    x, dx_frame = _normalize(projected, dprojected)
    y = np.cross(z, x)
    dy_frame = np.cross(dz, x[:, None]) + np.cross(z[:, None], dx_frame)

    if wf.true_drilling_rotation:
        reference = assembly.sv["_InitialNodeLocalPos"][selected, :, :2]
        local = np.stack(
            [np.einsum("eni,ei->en", rel, x), np.einsum("eni,ei->en", rel, y)], axis=-1
        )
        dlocal = np.stack(
            [
                np.einsum("ecni,ei->ecn", drel, x)
                + np.einsum("eni,eci->ecn", rel, dx_frame),
                np.einsum("ecni,ei->ecn", drel, y)
                + np.einsum("eni,eci->ecn", rel, dy_frame),
            ],
            axis=-1,
        )
        covariance = np.einsum("eni,enj->eij", local, reference)
        dcovariance = np.einsum("ecni,enj->ecij", dlocal, reference)
        a = covariance[:, 0, 0] + covariance[:, 1, 1]
        b = covariance[:, 1, 0] - covariance[:, 0, 1]
        da = dcovariance[:, :, 0, 0] + dcovariance[:, :, 1, 1]
        db = dcovariance[:, :, 1, 0] - dcovariance[:, :, 0, 1]
        denominator = a**2 + b**2
        if np.any(denominator < 1e-24):
            raise ValueError(
                "Consistent shell tangent requires a unique in-plane frame fit."
            )
        angle = np.arctan2(b, a)
        dangle = (a[:, None] * db - b[:, None] * da) / denominator[:, None]
        c, s = np.cos(angle), np.sin(angle)
        fitted_x = c[:, None] * x + s[:, None] * y
        fitted_y = -s[:, None] * x + c[:, None] * y
        fitted_dx = (
            c[:, None, None] * dx_frame
            + s[:, None, None] * dy_frame
            + fitted_y[:, None] * dangle[..., None]
        )
        fitted_dy = (
            -s[:, None, None] * dx_frame
            + c[:, None, None] * dy_frame
            - fitted_x[:, None] * dangle[..., None]
        )
        dx_frame, dy_frame = fitted_dx, fitted_dy
    dframe = np.stack([dx_frame, dy_frame, dz], axis=2)
    spin_matrix = dframe @ frame[:, None].swapaxes(-1, -2)
    spin = spin_matrix[..., [2, 0, 1], [1, 2, 0]]
    dposition = np.einsum("ecij,enj->ecni", dframe, rel) + np.einsum(
        "eij,ecnj->ecni", frame, drel
    )
    dq = np.zeros((ne, nd, nd))
    dq[:, columns[0]] = dposition.transpose(0, 3, 2, 1)
    local_theta = np.einsum("eij,ecnj->ecni", frame, theta)
    rotation = q[:, columns[1]].transpose(0, 2, 1)
    drotation = np.einsum(
        "enij,ecnj->ecni", _log_derivative(rotation), spin[:, :, None] + local_theta
    )
    dq[:, columns[1]] = drotation.transpose(0, 3, 2, 1)
    return dx, dframe, dposition, dq, spin


def _projected_inverse_derivatives(geometry, selected, dposition):
    """Differentiate cached projected geometry using local nodal variations.

    Since sum(N_,alpha)=0, centroid terms vanish from delta_J. Thus projecting
    delta_J_global and differentiating the frame is exactly equivalent to
    differentiating the local nodal positions already computed by _kinematics.
    """
    dn = np.asarray(geometry.shape_function_derivative_gp)
    inverse = geometry.inv_jacobian_matrix[selected]
    dprojected = np.einsum("gin,ecnj->ecgij", dn, dposition[..., :2])
    dinverse = -inverse[:, None] @ dprojected @ inverse[:, None]
    return inverse, dinverse


def _relative_area_derivative(geometry, nodes, frame, disp_columns, nd):
    """Physical surface-area variation in local displacement/rotation columns."""
    dn = np.asarray(geometry.shape_function_derivative_gp)
    jacobian = np.einsum("gin,enj->egij", dn, nodes)
    cross = np.cross(jacobian[:, :, 0], jacobian[:, :, 1])
    # delta(log dA) = a^alpha . delta(a_alpha). Its nodal derivative
    # only contains translations; rotations do not change physical area.
    dual = (
        np.stack(
            [np.cross(jacobian[:, :, 1], cross), np.cross(cross, jacobian[:, :, 0])],
            axis=2,
        )
        / np.sum(cross**2, axis=-1)[..., None, None]
    )
    local_dual = dual @ frame[:, None].swapaxes(-1, -2)
    area_translation = np.einsum("egij,gin->egjn", local_dual, dn)
    relative_area = np.zeros((len(nodes), nd, len(dn)))
    relative_area[:, disp_columns] = area_translation.transpose(0, 2, 3, 1)
    return relative_area


@dataclass
class _InterpolationData:
    geometry: object
    center_geometry: object
    records: list
    interpolations: dict
    frame: np.ndarray
    nodes: np.ndarray
    disp_columns: np.ndarray
    coordinate_directions: np.ndarray


def _prepare_interpolation(wf, assembly):
    """Read each interpolation once per tangent evaluation, outside batches.

    Only references to current assembly operators are retained, for this call.
    Trial-dependent geometry and MITC data are never cached across updates.
    """
    mesh = assembly.mesh
    element = get_element(assembly.elm_type)
    ne, ng, nn = mesh.n_elements, assembly.n_elm_gp, mesh.n_elm_nodes
    records, interpolations, values_cache = [], {}, {}
    operators = wf.generalized_strain_operator() + [wf.drill_constraint_operator()]
    for strain, op in enumerate(operators):
        for derivative, coefficient in zip(op.op, op.coef):
            block = assembly._get_elementary_operator(derivative)[0]
            if id(block) not in values_cache:
                if block.nnz == ne * ng * nn and np.all(np.diff(block.indptr) == nn):
                    values = block.data.reshape(ng, ne, nn).transpose(1, 0, 2)
                else:
                    rows = (
                        np.arange(ne)[:, None, None] + ne * np.arange(ng)[None, :, None]
                    )
                    cols = (
                        np.arange(ne)[:, None, None] + ne * np.arange(nn)[None, None, :]
                    )
                    rows, cols = np.broadcast_arrays(rows, cols)
                    values = np.asarray(block[rows.ravel(), cols.ravel()]).reshape(
                        ne, ng, nn
                    )
                values_cache[id(block)] = values
            interpolation_type = element.get_elm_type(derivative.u_name)
            records.append(
                (
                    strain,
                    derivative.u,
                    coefficient,
                    derivative.ordre,
                    derivative.x,
                    interpolation_type,
                    values_cache[id(block)],
                )
            )
            if derivative.ordre and interpolation_type not in interpolations:
                interpolations[interpolation_type] = interpolation_type(
                    ng, elmGeom=mesh._elm_interpolation[ng], assembly=assembly
                )
    geometry = mesh._elm_interpolation[ng]
    center = (
        mesh._elm_interpolation[1]
        if any(
            interpolation.use_center_jacobian
            for interpolation in interpolations.values()
        )
        else None
    )
    ranks = np.asarray(wf.space.get_rank_vector("Disp"))
    # Unit variations of local x/y coordinates, shared by every element.
    coordinate_directions = np.zeros((1, 2 * nn, nn, 2))
    indices = np.arange(2 * nn)
    coordinate_directions[:, indices, indices % nn, indices // nn] = 1
    return _InterpolationData(
        geometry,
        center,
        records,
        interpolations,
        assembly.get_element_local_frame(),
        mesh.nodes[mesh.elements],
        ranks[:, None] * nn + np.arange(nn),
        coordinate_directions,
    )


def _interpolation_derivatives(wf, data, selected, dposition, q, stress):
    """Read B from assembly and differentiate the actual FI/SRI/MITC operators.

    MITC covariant interpolation is linear in local nodal coordinates. Applying
    its existing compute_parametric_b to their variation gives its analytical
    derivative, including all tying-point metrics, without duplicating formulas.

    Differentiate first in the 2*nn local in-plane coordinate directions X.
    Contract B_,X with q and stress before applying dX/du. This avoids expanding
    interpolation derivatives in all displacement/rotation columns; dX/du
    still includes the complete centroid and rotating-frame variations.
    """
    geometry = data.geometry
    nodes = data.nodes[selected]
    nb, nn = nodes.shape[:2]
    ng, nd = len(geometry.xi_pg), nn * wf.space.nvar
    directions = data.coordinate_directions
    nc = 2 * nn
    inverse, dinverse = _projected_inverse_derivatives(geometry, selected, directions)
    b = np.zeros((nb, ng, 9, nd))
    # Only (delta_B) q and (delta_B).T s are needed. Contract each nonzero
    # interpolation block directly instead of storing a mostly zero 5D delta_B.
    db_q = np.zeros((nb, nc, ng, 9))
    interpolation_work = np.zeros((nb, ng, nd, nc))
    variation_cache = {}
    center_derivatives = None
    for (
        strain,
        rank,
        coefficient,
        order,
        axis,
        interpolation_type,
        values,
    ) in data.records:
        columns = slice(rank * nn, (rank + 1) * nn)
        b[:, :, strain, columns] += coefficient * values[selected]
        if not order:
            continue
        if interpolation_type not in variation_cache:
            interpolation = data.interpolations[interpolation_type]
            inv, dinv = inverse, dinverse
            if interpolation.use_center_jacobian:
                if center_derivatives is None:
                    center_derivatives = _projected_inverse_derivatives(
                        data.center_geometry,
                        selected,
                        directions,
                    )
                inv, dinv = center_derivatives
            reference = np.asarray(interpolation.shape_function_derivative_gp)
            reference = reference[selected] if reference.ndim == 4 else reference[None]
            variation = dinv @ reference[:, None]
            if hasattr(interpolation, "compute_parametric_b"):
                coordinates = directions[0, ..., interpolation.axis_idx]
                bx, by = interpolation.compute_parametric_b(geometry.xi_pg, coordinates)
                dreference = interpolation.sign * np.stack([bx, by], axis=2).reshape(
                    1, nc, ng, 2, nn
                )
                variation += inv[:, None] @ dreference
            variation_cache[interpolation_type] = variation
        variation = variation_cache[interpolation_type]
        block_variation = variation[..., axis, :]
        if coefficient != 1:
            block_variation = coefficient * block_variation
        db_q[..., strain] += np.einsum("ecgn,en->ecg", block_variation, q[:, columns])
        interpolation_work[:, :, columns] += (
            block_variation.transpose(0, 2, 3, 1) * stress[:, :, strain, None, None]
        )
    # Chain rule: (B_,X q) dX/du and (B_,X.T stress) dX/du.
    pullback = dposition[..., :2].transpose(0, 3, 2, 1).reshape(nb, nc, nd)
    db_q = db_q.transpose(0, 2, 3, 1) @ pullback[:, None]
    interpolation_work = interpolation_work @ pullback[:, None]
    relative_area = _relative_area_derivative(
        geometry, nodes, data.frame[selected], data.disp_columns, nd
    )
    return b, db_q, interpolation_work, relative_area


@dataclass
class ShellTangentOperators:
    strain_correction: np.ndarray
    interpolation_work: np.ndarray
    relative_area: np.ndarray
    frame_rotation: np.ndarray
    resultants: np.ndarray
    n_nodes: int

    def field(self, wf, coefficients):
        """Convert combined Gauss-point coefficients to a nodal DiffOp."""
        return _nodal_operator(wf, coefficients, self.n_nodes)

    def virtual_work(self, wf, coefficients):
        """sum_j v(coefficients[:, :, :, j]) delta_u_j."""
        fields, columns = _nodal_fields(wf, self.n_nodes)
        active = np.any(coefficients, axis=(0, 1))
        physical = np.zeros(len(fields), dtype=bool)
        physical[columns] = True
        active &= physical[:, None] & physical[None]
        # Trial-major ordering matches the previous column-by-column expansion.
        trial, virtual = np.nonzero(active.T)
        if not len(trial):
            return 0
        values = (
            coefficients[:, :, virtual, trial]
            .transpose(2, 1, 0)
            .reshape(len(trial), -1)
        )
        return DiffOp(
            [fields[column].op[0] for column in trial],
            [fields[row].op[0] for row in virtual],
            list(values),
        )


def shell_tangent_operators(wf, assembly, resultants):
    """Return differential operators for the four weak-form corrections.

    The ninth strain/resultant is the drilling constraint/penalty reaction.
    All interpolation and force-direction terms therefore include drilling.
    """
    mesh = assembly.mesh
    ne, ng, nn = mesh.n_elements, assembly.n_elm_gp, mesh.n_elm_nodes
    nd = nn * wf.space.nvar
    q = assembly.sv["_ShellTangentDof"]
    stress = np.stack(
        [
            np.broadcast_to(assembly.convert_data(s), (ne * ng,)).reshape(ng, ne).T
            for s in resultants
        ],
        axis=-1,
    )
    correction = np.empty((ne, ng, 9, nd))
    interpolation_work = np.empty((ne, ng, nd, nd))
    relative_area = np.empty((ne, ng, nd))
    local_force = np.empty((ne, ng, nd))
    spin = np.empty((ne, 3, nd))
    interpolation_data = _prepare_interpolation(wf, assembly)
    for start in range(0, ne, _BATCH_SIZE):
        selected = slice(start, start + _BATCH_SIZE)
        dx, dframe, dposition, dq, frame_spin = _kinematics(
            wf, assembly, selected, q[selected]
        )
        b, db_q, work, darea = _interpolation_derivatives(
            wf,
            interpolation_data,
            selected,
            dposition,
            q[selected],
            stress[selected],
        )
        correction[selected] = b @ (dq - np.eye(nd))[:, None] + db_q
        interpolation_work[selected] = work
        relative_area[selected] = darea.swapaxes(1, 2)
        local_force[selected] = np.einsum("egin,egi->egn", b, stress[selected])
        spin[selected] = frame_spin.swapaxes(1, 2)
    # Contract the three frame-spin products before expanding into DiffOps.
    # Expanding first creates up to 3 * nd**2 terms with duplicate operators.
    frame_rotation = np.zeros_like(interpolation_work)
    vector_ranks = np.asarray(
        [wf.space.get_rank_vector(vector) for vector in ("Disp", "Rot")]
    )
    vector_columns = nn * vector_ranks[:, None] + np.arange(nn)[None, :, None]
    # f.T skew(delta_omega) = (skew(f) delta_omega).T. Rotate both
    # displacement and moment vectors at every node in one batched product.
    frame_rotation[:, :, vector_columns] = (
        _skew(local_force[:, :, vector_columns]) @ spin[:, None, None, None]
    )
    return ShellTangentOperators(
        np.moveaxis(correction, 2, 0),
        interpolation_work,
        relative_area,
        frame_rotation,
        stress,
        nn,
    )
