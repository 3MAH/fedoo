"""Finite-difference consistency of the finite-strain global tangent.

Guards two historical defects of the 3D finite-strain path:

1. ``_init_nl_strain_op_vir`` compared ``space.ndim`` (an int) to the string
   ``"3D"``, so every 3D problem silently used the 2D-truncated geometric
   stiffness: all initial-stress terms involving a Z derivative (S33, S13,
   S23 slots and the k=Z rows) were missing. The tangent error was O(sigma),
   grew with load, and destroyed Newton on soft (bending) modes -- e.g. the
   NEOHC cantilever stalled at 0.82 L with a contraction rate ~0.85.

2. The UL log-corate branch used the umat box tangent divided by J as the
   spatial tangent. The tangent consistent with the Cauchy residual plus the
   standard initial-stress term is the Lie (Truesdell) one:
   box -> dS/dE -> DSDE_2_Dsigma_LieDD.

The check: on a hex8 box at 10% stretch, the assembled tangent must match the
central finite difference of the assembled global vector to ~1e-6 in relative
directional norm. With defect (1) the error is ~3e-3; machine-accurate
assembly gives ~1e-8.

The ELISO / EPICP cases (and the log_R corate) additionally require simcoon's
EXACT spectral log-box tangent transport (Daleckii-Krein maps in Lt_convert,
shipped after 2.0.0b1): with the earlier first-order (frozen-spin) transport
they sit at ~2e-3. They are gated by a runtime capability probe rather than a
version compare (see _simcoon_exact_log_transport).

HISTORY TRANSPORT (resolved 2026-09-01): the missing term at large rotation
was NOT the kernel's rotated-history sensitivity but the FRAME of the box
tangent inside simcoon's exact transport: the spatial-frame Lt was applied
to the material-frame d(ln U) without the polar-rotation conjugation
R^T (Lt : R dh R^T) R -- invisible for isotropic Lt (ELISO) or R ~ I
(stretch tests), O(||Lt_dev|| * theta_R) for a plastified box tangent under
accumulated rotation.  With the conjugated map plus the exact polar frame
increment DR = R1 R0^T in simcoon's log_R kinematics, the multi-increment
assembled tangent is FD-exact (~2e-8 through gamma = 0.3 committed shear)
and EPICP Newton stays flat at 4 iters/increment through gamma = 0.4
(before: 5 -> 18 over 0 -> 0.2, subiter exhaustion ~0.3).  The
multi-increment case is gated by _simcoon_exact_history_transport.

KNOWN LIMIT: with corate "log" (XBM) the frame increment is the XBM spin
integral, which is not exactly equivariant under superposed rotation; with
plastic history this leaves a genuine O(||EP|| * dtheta) tangent residual
(~1e-4 at gamma1 = 0.1, ~1e-3 at 0.3 -- harmless to Newton in practice:
iteration counts stay flat).  The multi-increment test therefore locks
corate log_R only.
"""

import functools

import numpy as np
import pytest
from scipy.linalg import logm, sqrtm

simcoon = pytest.importorskip("simcoon")

import fedoo as fd

MU, KAPPA = 3.0, 150.0
STRETCH = 0.10
# EPICP: E, nu, alpha, sigma_Y, k, m -- 10% stretch drives far beyond the
# 200 MPa yield (yield strain ~0.3%), so the box tangent is the evolving
# elastoplastic one, which is what the exact transport is really for.
EPICP_PROPS = [70000.0, 0.3, 1e-5, 200.0, 1000.0, 0.3]

_VOIGT = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]


def _sig_to_mat(s):
    """Voigt stress 6-vector -> symmetric 3x3 matrix."""
    return np.array([[s[0], s[3], s[4]], [s[3], s[1], s[5]], [s[4], s[5], s[2]]])


def _box_tangent_pk2_fd_error(call, F):
    """Rel. error of Lt_convert("DsigmaDe_2_DSDE") vs central FD of S(E).

    ``call(F) -> (s6, sig_mat, Lt)`` is a one-shot material-point umat
    evaluation returning the Cauchy stress (6-vector and matrix) and the
    corotational box tangent. Shared by both capability probes.
    """

    def pk2(E):
        F_ = np.real(sqrtm(2 * E + np.eye(3)))
        _, sig_mat, _ = call(F_)
        Fi = np.linalg.inv(F_)
        return np.linalg.det(F_) * Fi @ sig_mat @ Fi.T

    E0 = 0.5 * (F.T @ F - np.eye(3))
    s6, _, Lt = call(F)
    dsde = simcoon.Lt_convert(
        np.asfortranarray(Lt.reshape(6, 6, 1)),
        np.asfortranarray(F.reshape(3, 3, 1)),
        np.asfortranarray(np.asarray(s6).reshape(6, 1)),
        "DsigmaDe_2_DSDE",
    )[:, :, 0]
    h = 1e-7
    fd_mat = np.zeros((6, 6))
    for j, (k, l) in enumerate(_VOIGT):
        dE = np.zeros((3, 3))
        dE[k, l] = dE[l, k] = h
        dS = (pk2(E0 + dE) - pk2(E0 - dE)) / (2 * h)
        col = np.array([dS[0, 0], dS[1, 1], dS[2, 2], dS[0, 1], dS[0, 2], dS[1, 2]])
        fd_mat[:, j] = col if j < 3 else col / 2.0
    return np.abs(dsde - fd_mat).max() / np.abs(fd_mat).max()


def _directional_fd_error(pb, assembly, free, n_dir=3, eps=1e-8):
    """Max rel. error of K vs central FD of the assembled global vector.

    Probes ``n_dir`` random directions restricted to the ``free`` dofs,
    around the current converged state of ``pb``.
    """
    U0 = np.asarray(pb._U + (0 if np.isscalar(pb._dU) else pb._dU)).copy()
    pb._U = U0.copy()
    pb._dU = 0
    assembly.update(pb, "all")
    K = assembly.current.get_global_matrix().tocsr()

    rng = np.random.default_rng(0)
    errors = []
    for _ in range(n_dir):
        delta = np.zeros(K.shape[0])
        delta[free] = rng.standard_normal(len(free))
        delta /= np.abs(delta).max()
        r = []
        for sgn in (1, -1):
            pb._U = U0 + sgn * eps * delta
            pb._dU = 0
            assembly.update(pb, "vector")
            r.append(np.asarray(assembly.current.get_global_vector()).ravel().copy())
        fd_dir = (r[0] - r[1]) / (2 * eps)
        Kd = np.asarray(K @ delta).ravel()
        # fedoo assembles -residual in the global vector
        errors.append(
            np.linalg.norm(fd_dir[free] + Kd[free]) / np.linalg.norm(Kd[free])
        )
    return max(errors)


@functools.lru_cache(maxsize=1)
def _simcoon_exact_log_transport():
    """True when simcoon ships the exact log-box tangent transport.

    Capability probe (material point): one-shot ELISO at a rotation-free
    stretched state; the chain umat-box -> Lt_convert("DsigmaDe_2_DSDE") must
    match the FD of S(E) to ~1e-9. The first-order (frozen-spin) transport of
    simcoon <= 2.0.0b1 gives ~2e-3 here. A version compare is not usable:
    the installed metadata may lag the fix (dev builds report 0+unknown).
    TODO: once the simcoon release containing the fix has a number (TBD),
    bump the pyproject pin to it and replace this probe by that requirement.
    """
    props = np.asfortranarray(np.array([[8.7], [0.49], [1e-5]]))

    def call(F):
        F_ = np.asfortranarray(F.reshape(3, 3, 1))
        F0 = np.asfortranarray(np.eye(3).reshape(3, 3, 1))
        z6 = np.zeros((6, 1), order="F")
        DR = np.asfortranarray(np.eye(3).reshape(3, 3, 1))
        statev = np.zeros((1, 1), order="F")
        wm = np.zeros((4, 1), order="F")
        temp = np.zeros((1, 1), order="F")
        C = F.T @ F
        w, v = np.linalg.eigh(C)
        e = v @ np.diag(0.5 * np.log(w)) @ v.T
        de = np.asfortranarray(
            np.array(
                [e[0, 0], e[1, 1], e[2, 2], 2 * e[0, 1], 2 * e[0, 2], 2 * e[1, 2]]
            ).reshape(6, 1)
        )
        sig, _, _, Lt = simcoon.umat(
            "ELISO",
            z6,
            de,
            F0,
            F_,
            z6,
            DR,
            props,
            statev,
            0.0,
            1.0,
            wm,
            temp,
            ndi=3,
            tangent_mode=1,
        )
        s = sig[:, 0]
        return s, _sig_to_mat(s), (Lt[:, :, 0] if Lt.ndim == 3 else Lt)

    Fb = np.array([[1.08, 0.04, 0.0], [0.0, 0.96, 0.0], [0.0, 0.0, 0.99]])
    Fb = np.real(sqrtm(Fb @ Fb.T))
    return _box_tangent_pk2_fd_error(call, Fb) < 1e-6


requires_exact_transport = pytest.mark.skipif(
    not _simcoon_exact_log_transport(),
    reason=(
        "installed simcoon lacks the exact log-box tangent transport "
        "(first-order transport <= 2.0.0b1); bump the simcoon pin to the "
        "release containing it (number TBD) and drop this guard"
    ),
)


@functools.lru_cache(maxsize=1)
def _simcoon_exact_history_transport():
    """True when the exact transport also carries the rotation of the box.

    Discriminating probe (material point): ONE virgin EPICP increment of
    simple shear gamma = 0.3 (rotation-carrying, plastified, so the box
    tangent is anisotropic while the history is still zero).  The chain
    umat-box -> Lt_convert("DsigmaDe_2_DSDE") must match the FD of S(E) to
    ~1e-8.  Without the polar conjugation of the box inside the exact map
    (simcoon <= the 2026-09-01 fix) this sits at ~5e-3 even though
    _simcoon_exact_log_transport passes (its probe is rotation-free).
    """
    props = np.asfortranarray(np.c_[EPICP_PROPS].astype(float))

    def call(F):
        lnV = 0.5 * np.real(logm(F @ F.T))
        de = np.array(
            [
                lnV[0, 0],
                lnV[1, 1],
                lnV[2, 2],
                2 * lnV[0, 1],
                2 * lnV[0, 2],
                2 * lnV[1, 2],
            ]
        )
        F0 = np.asfortranarray(np.eye(3).reshape(3, 3, 1))
        _, DR, _ = simcoon.objective_rate(
            "log_R", F0, np.asfortranarray(F.reshape(3, 3, 1)), 1.0, False
        )
        z6 = np.zeros((6, 1), order="F")
        sig, _, _, Lt = simcoon.umat(
            "EPICP",
            z6,
            np.asfortranarray(de.reshape(6, 1)),
            F0,
            np.asfortranarray(F.reshape(3, 3, 1)),
            z6,
            np.asfortranarray(DR),
            props,
            np.zeros((8, 1), order="F"),
            0.0,
            1.0,
            np.zeros((4, 1), order="F"),
            np.zeros((1, 1), order="F"),
            ndi=3,
            tangent_mode=2,
        )
        s = sig[:, 0]
        return s, _sig_to_mat(s), (Lt[:, :, 0] if Lt.ndim == 3 else Lt)

    F1 = np.eye(3)
    F1[0, 1] = 0.3
    return _box_tangent_pk2_fd_error(call, F1) < 1e-6


requires_exact_history_transport = pytest.mark.skipif(
    not _simcoon_exact_history_transport(),
    reason=(
        "installed simcoon lacks the rotation-conjugated exact transport + "
        "exact polar log_R frame increment (2026-09-01 fix); bump the "
        "simcoon pin to the release containing it (number TBD) and drop "
        "this guard"
    ),
)


def _fd_tangent_error(nlgeom, law="NEOHC", corate=None):
    fd.Assembly.delete_memory()
    fd.ModelingSpace("3D")
    mesh = fd.mesh.box_mesh(nx=2, ny=2, nz=2, elm_type="hex8", name="box")

    if law == "NEOHC":
        material = fd.constitutivelaw.Simcoon("NEOHC", [MU, KAPPA], name="law")
    elif law == "ELISO":
        # hypoelastic path (no _Lt_from_F): exercises the generic UL
        # box -> dS/dE -> Lie conversion
        material = fd.constitutivelaw.Simcoon("ELISO", [8.7, 0.49, 1e-5], name="law")
    else:  # EPICP: evolving elastoplastic box tangent, loaded beyond yield
        material = fd.constitutivelaw.Simcoon("EPICP", EPICP_PROPS, name="law")
        # FD around a converged state measures the derivative of the
        # algorithmic (return-mapping) update: compare against the
        # Simo-Hughes algorithmic tangent, not the continuum one.
        material.tangent_mode = 2
    wf = fd.weakform.StressEquilibrium(material, nlgeom=nlgeom)
    wf.geometric_stiffness = True
    if corate is not None:
        wf.corate = corate
    assembly = fd.Assembly.create(wf, mesh, name="asm")

    pb = fd.problem.NonLinear("asm")
    pb.set_nr_criterion("Displacement", err0=1.0, tol=1e-10, max_subiter=30)

    bottom = mesh.find_nodes("Z", mesh.bounding_box.zmin)
    top = mesh.find_nodes("Z", mesh.bounding_box.zmax)
    pb.bc.add("Dirichlet", bottom, "DispZ", 0)
    pb.bc.add("Dirichlet", [0], "Disp", 0)
    pb.bc.add("Dirichlet", top, "DispZ", STRETCH)

    if law == "EPICP":
        # Plasticity: solve ONE increment and do NOT commit it (no set_start),
        # so sv_start stays virgin and the FD point sits DEEP in the plastic
        # branch (de = 10%). Committing would leave the FD state exactly on
        # the yield surface (de = 0): a central difference then straddles the
        # elastic/plastic kink and measures ~(C_e + C_ep)/2, not the tangent.
        pb.initialize()
        pb.tmax = 1.0
        pb.dtime = 1.0
        pb.set_start()
        pb.time = 0.0
        convergence, _, _ = pb.solve_time_increment()
        assert convergence, "EPICP single increment did not converge"
    else:
        pb.nlsolve(dt=0.2, tmax=1.0, update_dt=True, print_info=0)

    n_nodes = mesh.n_nodes
    blocked = {0, n_nodes, 2 * n_nodes}
    for n in np.concatenate([bottom, top]):
        blocked.add(int(n) + 2 * n_nodes)
    free = np.array([i for i in range(3 * n_nodes) if i not in blocked])
    return _directional_fd_error(pb, assembly, free)


_CASES = [
    # (law, corate). NEOHC's Lt is baked by inverting the transport, so it is
    # exact with any simcoon and needs no capability gate.
    pytest.param("NEOHC", "log", id="NEOHC-log"),
    pytest.param("ELISO", "log", marks=requires_exact_transport, id="ELISO-log"),
    pytest.param("ELISO", "log_r", marks=requires_exact_transport, id="ELISO-logR"),
    pytest.param("EPICP", "log", marks=requires_exact_transport, id="EPICP-log"),
    pytest.param("EPICP", "log_r", marks=requires_exact_transport, id="EPICP-logR"),
]


@pytest.mark.parametrize("law, corate", _CASES)
@pytest.mark.parametrize("nlgeom", ["TL", "UL"])
def test_finite_strain_tangent_is_fd_consistent(nlgeom, law, corate):
    err = _fd_tangent_error(nlgeom, law, corate)
    assert err < 1e-6, (
        f"{law} {nlgeom} corate={corate} global tangent inconsistent with FD "
        f"(rel error {err:.3e}): check the 3D geometric stiffness operator "
        "and the UL tangent conversion (box -> dS/dE -> Lie)"
    )


def _fd_history_tangent_error(gamma1, corate, n_inc=4, dgamma=0.05):
    """FD consistency WITH committed plastic history (rotating simple shear).

    EPICP cube, UL: n_inc committed increments to gamma1 (set_start after
    each: stress rotated, statev stored), then ONE more increment solved
    WITHOUT set_start; central FD of the assembled vector around its
    converged endpoint, exactly like _fd_tangent_error.
    """
    fd.Assembly.delete_memory()
    fd.ModelingSpace("3D")
    # nx=3: with both z-faces fully driven (shear), the mid-layer nodes are
    # the free dofs the FD probes (nx=2 would leave none).
    mesh = fd.mesh.box_mesh(nx=3, ny=3, nz=3, elm_type="hex8", name="box")

    material = fd.constitutivelaw.Simcoon("EPICP", EPICP_PROPS, name="law")
    material.tangent_mode = 2
    wf = fd.weakform.StressEquilibrium(material, nlgeom="UL")
    wf.geometric_stiffness = True
    wf.corate = corate
    assembly = fd.Assembly.create(wf, mesh, name="asm")

    pb = fd.problem.NonLinear("asm")
    pb.set_nr_criterion("Displacement", err0=1.0, tol=1e-10, max_subiter=30)

    bottom = mesh.find_nodes("Z", mesh.bounding_box.zmin)
    top = mesh.find_nodes("Z", mesh.bounding_box.zmax)
    zmax = mesh.bounding_box.zmax
    gamma_total = gamma1 + dgamma
    pb.bc.add("Dirichlet", bottom, "Disp", 0)
    pb.bc.add("Dirichlet", top, "DispX", gamma_total * zmax)
    pb.bc.add("Dirichlet", top, "DispY", 0)
    pb.bc.add("Dirichlet", top, "DispZ", 0)

    pb.initialize()
    pb.tmax = 1.0
    t1 = gamma1 / gamma_total
    pb.dtime = t1 / n_inc
    pb.set_start()
    pb.time = 0.0
    for i in range(n_inc):
        conv, _, _ = pb.solve_time_increment()
        assert conv, f"committed increment {i + 1} did not converge"
        pb.set_start()
        pb.time = (i + 1) * pb.dtime

    pb.dtime = 1.0 - t1
    conv, _, _ = pb.solve_time_increment()
    assert conv, "final uncommitted increment did not converge"

    n_nodes = mesh.n_nodes
    blocked = set()
    for n in np.concatenate([bottom, top]):
        for r in range(3):
            blocked.add(int(n) + r * n_nodes)
    free = np.array([i for i in range(3 * n_nodes) if i not in blocked])
    return _directional_fd_error(pb, assembly, free)


@requires_exact_history_transport
def test_finite_strain_tangent_with_plastic_history():
    # corate log_R only: the XBM ("log") frame increment is not exactly
    # equivariant, leaving a documented O(||EP|| * dtheta) residual (see the
    # module docstring KNOWN LIMIT) -- log_R is the exact-transport corate.
    err = _fd_history_tangent_error(0.2, "log_r")
    assert err < 1e-6, (
        f"UL EPICP corate=log_r with committed plastic history: global "
        f"tangent inconsistent with FD (rel error {err:.3e}): check the "
        "polar conjugation in simcoon's exact transport and the log_R "
        "frame increment DR = R1 R0^T"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
