"""Analytical linearization of the existing two-node UL beam residual.

The force law is the historical rotated local force, not a newly introduced
energy-conjugate corotational law. Its Jacobian need not be symmetric.
All derivatives below are with respect to spatial incremental rotations,
as used by the nonlinear solver, rather than additive total rotation vectors.
"""

from dataclasses import dataclass

import numpy as np

_TANGENT_BATCH_SIZE = 2048


def _skew(v):
    """Skew matrices for vectors with arbitrary leading batch dimensions."""
    x, y, z = np.moveaxis(v, -1, 0)
    zero = np.zeros_like(x)
    return np.stack([zero, -z, y, z, zero, -x, -y, x, zero], axis=-1).reshape(
        v.shape[:-1] + (3, 3)
    )


def _log_derivative(v):
    """Inverse left Jacobian of SO(3), with its small-angle expansion."""
    angle = np.linalg.norm(v, axis=-1)
    a = np.asarray(1 / 12 + angle**2 / 720 + angle**4 / 30240)
    regular = angle >= 1e-4
    theta = angle[regular]
    a[regular] = (1 - 0.5 * theta / np.tan(0.5 * theta)) / theta**2
    s = _skew(v)
    return np.eye(3) - 0.5 * s + a[..., None, None] * (s @ s)


def _strain_coefficients(wf, assembly, element_slice):
    """Read B from assembly's cached beam interpolation operators.

    Gauss rows are ordered (point, element); local columns (node, element).
    Associated-variable blocks carry the coupled displacement/rotation
    interpolation, exactly as in Assembly.get_gp_results.
    """
    ne, ng = assembly.mesh.n_elements, assembly.n_elm_gp
    elements = np.arange(ne)[element_slice]
    rows = elements[:, None, None] + ne * np.arange(ng)[None, :, None]
    cols = elements[:, None, None] + ne * np.arange(2)[None, None, :]
    rows, cols = np.broadcast_arrays(rows, cols)
    b = np.zeros((len(elements), ng, 6, 2 * wf.space.nvar))
    associated = assembly._get_associated_variables()
    for strain, op in enumerate(wf.space.op_beam_strain()):
        if op == 0:
            continue
        for derivative, coefficient in zip(op.op, op.coef):
            variables, signs = [derivative.u], [1]
            if derivative.u in associated:
                variables += list(associated[derivative.u][0])
                signs += list(associated[derivative.u][1])
            blocks = assembly._get_elementary_operator(derivative)
            for block, variable, sign in zip(blocks, variables, signs):
                values = np.asarray(block[rows.ravel(), cols.ravel()]).reshape(
                    len(elements), ng, 2
                )
                b[:, :, strain, 2 * variable : 2 * variable + 2] += (
                    coefficient * sign * values
                )
    return b


def _strain_matrices(wf, assembly, length, element_slice=slice(None)):
    """Reuse B and differentiate its interpolation with respect to length.

    B itself comes from Assembly. Only B_L needs explicit formulas here,
    since ordinary beam interpolation does not provide a length derivative.
    """
    mesh = assembly.mesh
    ng = assembly.n_elm_gp
    xi = mesh._elm_interpolation[ng].xi_pg[:, 0][None, :]
    L = length[:, None]
    b = _strain_coefficients(wf, assembly, element_slice)
    db = np.zeros_like(b)
    ranks = {name: wf.space.variable_rank(name) for name in wf.space.list_variables()}

    def put(row, name, derivatives):
        rank = ranks[name]
        db[:, :, row, 2 * rank : 2 * rank + 2] = np.stack(derivatives, axis=-1)

    put(0, "DispX", [1 / L**2, -1 / L**2])
    if wf.space.ndim == 3:
        put(3, "RotX", [1 / L**2, -1 / L**2])
    for bend, shear, disp, rot, inertia, sign in [
        (5, 1, "DispY", "RotZ", wf.properties.Izz, 1),
        (4, 2, "DispZ", "RotY", wf.properties.Iyy, -1),
    ]:
        if disp not in ranks:
            continue
        k = wf.properties.k
        phi = (
            np.zeros_like(L)
            if np.isscalar(k) and k == 0
            else np.broadcast_to(
                24 * inertia * (1 + wf.properties.material.nu) / (k * wf.properties.A),
                (mesh.n_elements,),
            )[element_slice, None]
            / L**2
        )
        dp = -2 * phi / L
        c = 1 / (1 + phi)
        dc = -(c**2) * dp
        d = c / L
        dd = dc / L - c / L**2
        dv = 6 * (dc / L**2 - 2 * c / L**3) * (2 * xi - 1)
        t1 = 6 * xi - 4 - phi
        t2 = 6 * xi - 2 + phi
        put(bend, disp, [sign * dv, -sign * dv])
        put(bend, rot, [dd * t1 - d * dp, dd * t2 + d * dp])
        p = phi * c
        deriv_p = dp * c + phi * dc
        put(
            shear,
            disp,
            [-deriv_p / L + p / L**2, deriv_p / L - p / L**2],
        )
        put(
            shear,
            rot,
            [-sign * deriv_p / 2, -sign * deriv_p / 2],
        )
    return b, db


def _kinematic_derivatives(wf, length, frame, q):
    """Differentiate length, local DOFs and frame w.r.t. LOCAL increments.

    ``dframe[e, c]`` is the frame derivative for column c. For a local nodal
    rotation Q = R A, dQ Q.T = dR R.T + R [dtheta] R.T: A cancels. This
    avoids multiplying nodal rotation matrices for every tangent column.
    """
    dim = wf.space.ndim
    nv = wf.space.nvar
    nd = 2 * nv
    ne = len(length)
    disp = wf.space.get_rank_vector("Disp")
    rot = (
        [wf.space.variable_rank("RotZ")]
        if dim == 2
        else wf.space.get_rank_vector("Rot")
    )
    # Columns are LOCAL nodal increments. Assembly already applies T on
    # both sides of the weak form; no dense nodal transformation is needed.
    dx = np.zeros((ne, dim, nd))
    angle = np.zeros((ne, nd, 3))
    for axis, rank in enumerate(disp):
        dx[:, :, 2 * rank] = -frame[:, axis]
        dx[:, :, 2 * rank + 1] = frame[:, axis]
    for axis, rank in enumerate(rot):
        if dim == 2:
            angle[:, 2 * rank : 2 * rank + 2, 2] = 1
        else:
            angle[:, 2 * rank : 2 * rank + 2] = frame[:, None, axis]
    dl = np.einsum("ei,eic->ec", frame[:, 0], dx)
    dq = np.zeros((ne, nd, nd))
    dq[:, 2 * disp[0] + 1] = dl
    dframe = np.zeros((ne, nd, dim, dim))
    if dim == 2:
        da = np.einsum("ei,eic->ec", frame[:, 1], dx) / length[:, None]
        dframe[:, :, 0] = frame[:, None, 1] * da[..., None]
        dframe[:, :, 1] = -frame[:, None, 0] * da[..., None]
        for node in range(2):
            index = 2 * rot[0] + node
            dq[:, index] = -da
            dq[:, index, index] += 1
    else:
        x = frame[:, None, 0]
        y = frame[:, None, 1]
        z = frame[:, None, 2]
        dframe[:, :, 0] = (dx.swapaxes(-1, -2) - x * dl[..., None]) / length[
            :, None, None
        ]
        # Exactly the trial-frame convention used in BeamEquilibrium.update.
        dz_trial = np.einsum("ei,ecij->ecj", frame[:, 2], _skew(0.5 * angle))
        dy_trial = np.cross(dz_trial, x) + np.cross(z, dframe[:, :, 0])
        dframe[:, :, 1] = dy_trial - y * np.sum(y * dy_trial, axis=-1, keepdims=True)
        dframe[:, :, 2] = np.cross(dframe[:, :, 0], y) + np.cross(x, dframe[:, :, 1])
        omega = np.stack(
            [
                np.sum(dframe[:, :, 2] * y, axis=-1),
                np.sum(dframe[:, :, 0] * z, axis=-1),
                np.sum(dframe[:, :, 1] * x, axis=-1),
            ],
            axis=-1,
        )
        # R [theta] R.T = [R theta]; only the axial vector is needed.
        node_omega = np.einsum("eij,ecj->eci", frame, angle)
        for node in range(2):
            indices = 2 * np.asarray(rot) + node
            total_omega = omega.copy()
            total_omega[:, indices] += node_omega[:, indices]
            jacobian = _log_derivative(q[:, indices])
            dq[:, indices, :] = jacobian @ total_omega.swapaxes(-1, -2)
    return dq, dl, dframe


def _field_operator(wf, coefficients, length, n_gp):
    """A nodal linear functional expressed as midpoint and chord derivative.

    a1*u1 + a2*u2 = (a1+a2)*mean(u) + L/2*(a2-a1)*du/ds.
    These are independent nodal fields; the ordinary beam strain operators
    retain their coupled bending interpolation.
    """
    ne = len(length)
    coefficients = np.broadcast_to(coefficients, (ne, n_gp, 2 * wf.space.nvar))
    result = 0
    for variable in ("DispX", "DispY", "DispZ", "RotX", "RotY", "RotZ"):
        if variable not in wf.space.list_variables():
            continue
        rank = wf.space.variable_rank(variable)
        a1, a2 = coefficients[:, :, 2 * rank], coefficients[:, :, 2 * rank + 1]
        mean = a1 + a2
        difference = (a2 - a1) * length[:, None] / 2
        if np.any(mean):
            result += wf.space.variable("_BeamMean" + variable) * mean.T.ravel()
        if np.any(difference):
            result += (
                wf.space.derivative("_BeamMean" + variable, "X") * difference.T.ravel()
            )
    return result


@dataclass
class BeamTangentOperators:
    """Trial/virtual operators for the four residual derivatives.

    strain_correction = B (delta_q - delta_u_local) + B_L q delta_L
    interpolation_work = (B_L v_local).T s
    relative_length = delta_L / L
    frame_work[j] = (B S_j v_local).T s
    frame_spin[j] = axial(delta_R R.T)[j]
    """

    strain_correction: list
    interpolation_work: object
    delta_length: object
    relative_length: object
    frame_work: list
    frame_spin: list


def beam_tangent_operators(wf, assembly, data):
    """Prepare local differential operators; beam.py writes the weak form.

    R comes from the current assembly, shared with ordinary beam assembly.
    Its sparse nodal transformation T is applied there, not rebuilt here.
    Derivatives are batched; no element stiffness matrix is constructed.
    """
    q = data[0]
    frames = assembly.get_element_local_frame()
    mesh = assembly.mesh
    length = np.linalg.norm(
        mesh.nodes[mesh.elements[:, 1]] - mesh.nodes[mesh.elements[:, 0]], axis=1
    )
    ne, ng = mesh.n_elements, assembly.n_elm_gp
    nd = 2 * wf.space.nvar
    updated_strain = np.empty((ne, ng, 6, nd))
    length_virtual = np.empty((ne, ng, nd))
    local_force = np.empty((ne, ng, nd))
    spin_coefficients = np.empty((ne, 1 if wf.space.ndim == 2 else 3, nd))
    stress = np.zeros((ne, ng, 6))
    values = assembly.sv["BeamStress"]
    if not (np.isscalar(values) and values == 0):
        for i, value in enumerate(values):
            stress[:, :, i] = (
                value if np.isscalar(value) else np.asarray(value).reshape(ng, ne).T
            )
    for start in range(0, ne, _TANGENT_BATCH_SIZE):
        selected = slice(start, start + _TANGENT_BATCH_SIZE)
        b, db = _strain_matrices(wf, assembly, length[selected], selected)
        dq, dl, dframe = _kinematic_derivatives(
            wf, length[selected], frames[selected], q[selected]
        )
        # dq is already differentiated with respect to local increments.
        dbq = (db @ q[selected, None, :, None])[..., 0]
        updated_strain[selected] = (
            b @ (dq - np.eye(nd))[:, None] + dbq[..., None] * dl[:, None, None]
        )
        length_virtual[selected] = np.einsum("egrn,egr->egn", db, stress[selected])
        local_force[selected] = np.einsum("egrn,egr->egn", b, stress[selected])
        spin = dframe @ frames[selected, None].swapaxes(-1, -2)
        axial = (
            spin[..., 1, 0, None]
            if wf.space.ndim == 2
            else spin[..., [2, 0, 1], [1, 2, 0]]
        )
        spin_coefficients[selected] = axial.swapaxes(-1, -2)

    strain_correction = [
        _field_operator(wf, updated_strain[:, :, i], length, ng) for i in range(6)
    ]
    relative_length = wf.space.derivative("_BeamMeanDispX", "X")
    delta_length = relative_length * np.tile(length, ng)
    interpolation_work = _field_operator(wf, length_virtual, length, ng)

    dim = wf.space.ndim
    groups = [wf.space.get_rank_vector("Disp")]
    if dim == 3:
        groups.append(wf.space.get_rank_vector("Rot"))
    generators = np.array([[[0.0, -1.0], [1.0, 0.0]]]) if dim == 2 else _skew(np.eye(3))
    frame_work, frame_spin = [], []
    for component, generator in enumerate(generators):
        virtual_coefficients = np.zeros_like(local_force)
        for ranks in groups:
            for node in range(2):
                indices = 2 * np.asarray(ranks) + node
                virtual_coefficients[:, :, indices] = (
                    local_force[:, :, indices] @ generator
                )
        frame_work.append(_field_operator(wf, virtual_coefficients, length, ng))
        frame_spin.append(
            _field_operator(wf, spin_coefficients[:, component, None], length, ng)
        )
    return BeamTangentOperators(
        strain_correction,
        interpolation_work,
        delta_length,
        relative_length,
        frame_work,
        frame_spin,
    )
