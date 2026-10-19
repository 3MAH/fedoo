"""Geometrically exact (Simo-Reissner) two-node beam kinematics.

Crisfield-Jelenic interpolation of the rotations, reduced to one point: the
relative rotation of the element ``psi = log(L1.T L2)`` gives the material
curvature ``K = psi / L0`` and the midpoint triad ``Lm = L1 exp(psi / 2)``
gives the material strain ``Gamma = Lm.T d - E1``, ``d`` the chord over its
reference length. Every quantity is written in the reference element frame,
whose local DOFs are the ones the assembly works with: there, the nodal
triads of the reference configuration are the identity.

Derivatives are taken with respect to spatial spin increments
(``L <- exp(dtheta) L``), as used by the nonlinear solver. The stress
contracted second derivative ``G`` is assembled from exact directional
derivatives of ``B``; the tangent is the exact Jacobian of the residual.
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from fedoo.util.so3 import jacobian_coefficients as _coefficients
from fedoo.util.so3 import skew as _skew


def _jacobians(psi):
    """J(psi / 2), J(psi)^-1 and the data of their directional derivatives."""
    eye = np.eye(3)
    s = _skew(psi)
    s2 = s @ s
    a, alpha, _, _, _, _ = _coefficients(np.linalg.norm(psi, axis=-1))
    half = 0.5 * psi
    h = _skew(half)
    h2 = h @ h
    _, _, b, beta, c, gamma = _coefficients(np.linalg.norm(half, axis=-1))
    jinv = eye - 0.5 * s + a[..., None, None] * s2
    jhalf = eye + b[..., None, None] * h + c[..., None, None] * h2
    return jinv, jhalf, (s, s2, a, alpha), (h, h2, b, beta, c, gamma)


def _d_jinv(psi, data, dpsi):
    """Directional derivative of J(psi)^-1 along dpsi (batched directions)."""
    s, s2, a, alpha = data
    ds = _skew(dpsi)
    dot = np.einsum("...i,...ci->...c", psi, dpsi)
    return (
        -0.5 * ds
        + a[..., None, None, None] * (ds @ s[..., None, :, :] + s[..., None, :, :] @ ds)
        + (alpha[..., None] * dot)[..., None, None] * s2[..., None, :, :]
    )


def _d_jhalf(psi, data, dpsi):
    """Directional derivative of J(psi / 2) along dpsi (batched directions)."""
    h, h2, b, beta, c, gamma = data
    dh = _skew(0.5 * dpsi)
    dot = np.einsum("...i,...ci->...c", 0.5 * psi, 0.5 * dpsi)
    return (
        b[..., None, None, None] * dh
        + c[..., None, None, None] * (dh @ h[..., None, :, :] + h[..., None, :, :] @ dh)
        + (beta[..., None] * dot)[..., None, None] * h[..., None, :, :]
        + (gamma[..., None] * dot)[..., None, None] * h2[..., None, :, :]
    )


@dataclass
class ExactBeamKinematics:
    """Element strains and their first and second derivatives.

    strain : (ne, 6) material strains in the fedoo order
        (Gamma_1, Gamma_2, Gamma_3, K_1, K_2, K_3).
    b : (ne, 6, nd) d strain / d local DOFs, DOF index 2 * rank + node.
    db : (ne, nd, 6, nd) directional derivative of b, one direction per DOF,
        or None when the tangent is not needed.
    """

    strain: np.ndarray
    b: np.ndarray
    db: np.ndarray = None


def _ranks(space):
    disp = list(space.get_rank_vector("Disp"))
    if space.ndim == 3:
        rot = list(space.get_rank_vector("Rot"))
    else:
        rot = [space.variable_rank("RotZ")]
    return disp, rot


def exact_beam_kinematics(space, length, u_local, rotations, need_tangent=True):
    """Strains, B and dB of two-node Simo-Reissner elements.

    Parameters
    ----------
    space : ModelingSpace
    length : (ne,) reference lengths.
    u_local : (ne, 2, dim) nodal displacements in the reference element frame.
    rotations : (ne, 2) nodal rotation angles in 2D, or (ne, 2, 3, 3) nodal
        rotation matrices in the reference element frame in 3D.
    need_tangent : bool, default=True
        If False, ``db`` is not computed (residual-only evaluation).
    """
    if space.ndim == 2:
        return _kinematics_2d(space, length, u_local, rotations, need_tangent)
    return _kinematics_3d(space, length, u_local, rotations, need_tangent)


def _kinematics_2d(space, length, u_local, theta, need_tangent=True):
    ne = len(length)
    nd = 2 * space.nvar
    disp, (rot,) = _ranks(space)
    L = length[:, None]
    d = (u_local[:, 1] - u_local[:, 0]) / L
    d[:, 0] += 1
    theta_m = 0.5 * (theta[:, 0] + theta[:, 1])
    c, s = np.cos(theta_m), np.sin(theta_m)
    gamma_1 = c * d[:, 0] + s * d[:, 1] - 1
    gamma_2 = -s * d[:, 0] + c * d[:, 1]

    strain = np.zeros((ne, 6))
    strain[:, 0] = gamma_1
    strain[:, 1] = gamma_2
    strain[:, 5] = (theta[:, 1] - theta[:, 0]) / length

    b = np.zeros((ne, 6, nd))
    axial = np.stack([c, s], axis=-1) / L
    transverse = np.stack([-s, c], axis=-1) / L
    for node, sign in ((0, -1), (1, 1)):
        for axis, rank in enumerate(disp):
            b[:, 0, 2 * rank + node] = sign * axial[:, axis]
            b[:, 1, 2 * rank + node] = sign * transverse[:, axis]
        b[:, 0, 2 * rot + node] = 0.5 * gamma_2
        b[:, 1, 2 * rot + node] = -0.5 * (gamma_1 + 1)
        b[:, 5, 2 * rot + node] = sign / length
    if not need_tangent:
        return ExactBeamKinematics(strain, b)

    # direction j moves theta_m by 1/2 if j is a rotation DOF
    dtheta_m = np.zeros((ne, nd))
    dtheta_m[:, 2 * rot : 2 * rot + 2] = 0.5
    db = np.zeros((ne, nd, 6, nd))
    for node, sign in ((0, -1), (1, 1)):
        for axis, rank in enumerate(disp):
            db[:, :, 0, 2 * rank + node] = sign * transverse[:, None, axis] * dtheta_m
            db[:, :, 1, 2 * rank + node] = -sign * axial[:, None, axis] * dtheta_m
        db[:, :, 0, 2 * rot + node] = 0.5 * b[:, 1]
        db[:, :, 1, 2 * rot + node] = -0.5 * b[:, 0]
    return ExactBeamKinematics(strain, b, db)


def _kinematics_3d(space, length, u_local, triads, need_tangent=True):
    ne = len(length)
    nd = 2 * space.nvar
    disp, rot = _ranks(space)
    eye = np.eye(3)
    L = length[:, None]
    lambda_1 = triads[:, 0]
    lambda_1_t = lambda_1.swapaxes(-1, -2)
    relative = lambda_1_t @ triads[:, 1]
    psi = Rotation.from_matrix(relative).as_rotvec()
    lambda_m = lambda_1 @ Rotation.from_rotvec(0.5 * psi).as_matrix()
    lambda_m_t = lambda_m.swapaxes(-1, -2)
    d = (u_local[:, 1] - u_local[:, 0]) / L
    d[:, 0] += 1

    jinv, jhalf, jinv_data, jhalf_data = _jacobians(psi)
    # dtheta_m = (I - T/2) dtheta_1 + T/2 dtheta_2
    t_mat = lambda_1 @ jhalf @ jinv @ lambda_1_t
    curvature_op = jinv @ lambda_1_t / L[..., None]  # d K / d theta_2
    skew_d = _skew(d)

    strain = np.zeros((ne, 6))
    strain[:, :3] = np.einsum("eij,ej->ei", lambda_m_t, d)
    strain[:, 0] -= 1
    strain[:, 3:] = psi / L

    def blocks(lm_t, sd, t, k_op):
        """B blocks (u1, u2, theta1, theta2), each (..., 6, 3)."""
        zero = np.zeros(lm_t.shape[:-2] + (3, 3))
        upper_u = lm_t / L.reshape(L.shape[:1] + (1,) * (lm_t.ndim - 1))
        theta_1 = lm_t @ sd @ (eye - 0.5 * t)
        theta_2 = lm_t @ sd @ (0.5 * t)
        return (
            np.concatenate([-upper_u, zero], axis=-2),
            np.concatenate([upper_u, zero], axis=-2),
            np.concatenate([theta_1, -k_op], axis=-2),
            np.concatenate([theta_2, k_op], axis=-2),
        )

    b = np.zeros((ne, 6, nd))
    for block, (ranks, node) in zip(
        blocks(lambda_m_t, skew_d, t_mat, curvature_op),
        ((disp, 0), (disp, 1), (rot, 0), (rot, 1)),
    ):
        b[:, :, 2 * np.asarray(ranks) + node] = block
    if not need_tangent:
        return ExactBeamKinematics(strain, b)

    # Directions: one unit increment per local DOF
    directions = np.eye(nd)
    du1 = directions[:, 2 * np.asarray(disp)]
    du2 = directions[:, 2 * np.asarray(disp) + 1]
    dth1 = directions[:, 2 * np.asarray(rot)]
    dth2 = directions[:, 2 * np.asarray(rot) + 1]
    dd = (du2 - du1)[None] / L[:, None]  # (ne, nd, 3)
    dpsi = np.einsum("eij,cj->eci", jinv @ lambda_1_t, dth2 - dth1)
    djinv = _d_jinv(psi, jinv_data, dpsi)
    djhalf = _d_jhalf(psi, jhalf_data, dpsi)
    skew_dth1 = _skew(dth1)[None]
    # lambda_1 -> exp(dth1) lambda_1 changes lambda_1.T by -lambda_1.T [dth1]
    dl1_t = -lambda_1_t[:, None] @ skew_dth1
    dt_mat = (
        skew_dth1 @ t_mat[:, None]
        - t_mat[:, None] @ skew_dth1
        + lambda_1[:, None]
        @ (djhalf @ jinv[:, None] + jhalf[:, None] @ djinv)
        @ lambda_1_t[:, None]
    )
    dtheta_m = np.einsum("eij,cj->eci", eye - 0.5 * t_mat, dth1) + np.einsum(
        "eij,cj->eci", 0.5 * t_mat, dth2
    )
    dlm_t = -lambda_m_t[:, None] @ _skew(dtheta_m)
    dk_op = (djinv @ lambda_1_t[:, None] + jinv[:, None] @ dl1_t) / L[:, None, :, None]

    # Product rule on each factor of the B blocks
    lm_t = lambda_m_t[:, None]
    sd = skew_d[:, None]
    t = t_mat[:, None]
    dsd = _skew(dd)
    zero = np.zeros((ne, nd, 3, 3))
    upper_u = dlm_t / L[:, None, :, None]
    dtheta_1 = (
        dlm_t @ sd @ (eye - 0.5 * t)
        + lm_t @ dsd @ (eye - 0.5 * t)
        - lm_t @ sd @ (0.5 * dt_mat)
    )
    dtheta_2 = (
        dlm_t @ sd @ (0.5 * t) + lm_t @ dsd @ (0.5 * t) + lm_t @ sd @ (0.5 * dt_mat)
    )
    dblocks = (
        np.concatenate([-upper_u, zero], axis=-2),
        np.concatenate([upper_u, zero], axis=-2),
        np.concatenate([dtheta_1, -dk_op], axis=-2),
        np.concatenate([dtheta_2, dk_op], axis=-2),
    )
    db = np.zeros((ne, nd, 6, nd))
    for block, (ranks, node) in zip(
        dblocks, ((disp, 0), (disp, 1), (rot, 0), (rot, 1))
    ):
        db[:, :, :, 2 * np.asarray(ranks) + node] = block
    return ExactBeamKinematics(strain, b, db)


def element_operators(kinematics, stress, tangent):
    """Residual density B.T s and tangent density B.T C B + G per Gauss point.

    stress : (ng, ne, 6) material resultants (N, M).
    tangent : (ng, ne, 6, 6) section tangent.
    Returns (ng, ne, nd) and (ng, ne, nd, nd), in local nodal DOFs. Without
    tangent (``tangent`` or ``kinematics.db`` None), the stiffness is None.
    """
    b = kinematics.b
    force = np.einsum("eia,gei->gea", b, stress)
    if tangent is None or kinematics.db is None:
        return force, None
    material = np.einsum("eia,geij,ejb->geab", b, tangent, b)
    # G[a, c] = d(B.T s)_a / d q_c at fixed s
    geometric = np.einsum("ecia,gei->geac", kinematics.db, stress)
    return force, material + geometric


def to_field_basis(length, force, stiffness, nvar):
    """Express nodal operators on the (midpoint, chord derivative) basis.

    Per variable, u1 = m - L/2 u' and u2 = m + L/2 u' (see _field_operator).
    """
    nd = 2 * nvar
    ne = len(length)
    p_inv = np.zeros((ne, nd, nd))
    for rank in range(nvar):
        p_inv[:, 2 * rank, 2 * rank] = 1
        p_inv[:, 2 * rank + 1, 2 * rank] = 1
        p_inv[:, 2 * rank, 2 * rank + 1] = -length / 2
        p_inv[:, 2 * rank + 1, 2 * rank + 1] = length / 2
    force = np.einsum("eba,geb->gea", p_inv, force)
    if stiffness is None:
        return force, None
    return force, p_inv.swapaxes(-1, -2)[None] @ stiffness @ p_inv[None]
