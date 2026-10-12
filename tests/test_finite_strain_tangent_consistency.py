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

The ELISO / EPICP cases (and the log_R corate) use Simcoon 2.1's exact
spectral log-box tangent transport (Daleckii-Krein maps in Lt_convert).

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
multi-increment case checks this transport with committed plastic history.

KNOWN LIMIT: with corate "log" (XBM) the frame increment is the XBM spin
integral, which is not exactly equivariant under superposed rotation; with
plastic history this leaves a genuine O(||EP|| * dtheta) tangent residual
(~1e-4 at gamma1 = 0.1, ~1e-3 at 0.3 -- harmless to Newton in practice:
iteration counts stay flat).  The multi-increment test therefore locks
corate log_R only.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.linalg import logm, sqrtm

simcoon = pytest.importorskip("simcoon")

import fedoo as fd  # noqa: E402
from fedoo.util.voigt_tensors import StressTensorList, StrainTensorList  # noqa: E402

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
    for j, (k, m) in enumerate(_VOIGT):
        dE = np.zeros((3, 3))
        dE[k, m] = dE[m, k] = h
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


def test_simcoon_exact_log_transport():
    """Check the exact log-box tangent transport required from Simcoon 2.1.

    Material-point check: one-shot ELISO at a rotation-free
    stretched state; the chain umat-box -> Lt_convert("DsigmaDe_2_DSDE") must
    match the FD of S(E) to ~1e-9. The first-order (frozen-spin) transport of
    earlier Simcoon versions gives ~2e-3 here.
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
    assert _box_tangent_pk2_fd_error(call, Fb) < 1e-6


def test_simcoon_exact_history_transport():
    """Check that the exact transport carries the rotation of the box.

    Discriminating probe (material point): ONE virgin EPICP increment of
    simple shear gamma = 0.3 (rotation-carrying, plastified, so the box
    tangent is anisotropic while the history is still zero).  The chain
    umat-box -> Lt_convert("DsigmaDe_2_DSDE") must match the FD of S(E) to
    ~1e-8.  Without the polar conjugation of the box inside the exact map
    (simcoon <= the 2026-09-01 fix) this sits at ~5e-3 even though
    the rotation-free log transport check passes.
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
    assert _box_tangent_pk2_fd_error(call, F1) < 1e-6


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
    elif law == "FEDOO_ELISO":
        material = fd.constitutivelaw.ElasticIsotrop(8.7, 0.49, name="law")
    else:  # EPICP: evolving elastoplastic box tangent, loaded beyond yield
        material = fd.constitutivelaw.Simcoon("EPICP", EPICP_PROPS, name="law")
        # FD around a converged state measures the derivative of the
        # algorithmic (return-mapping) update: compare against the
        # Simo-Hughes algorithmic tangent, not the continuum one.
        material.tangent_mode = 2
    wf = fd.weakform.StressEquilibrium(material, nlgeom=nlgeom)
    if corate is not None:
        wf.corate = corate
    assembly = fd.Assembly.create(wf, mesh, name="asm")

    pb = fd.problem.NonLinear("asm")
    pb.set_nr_criterion("Displacement", err0=1.0, tol=1e-10, max_subiter=30)

    bottom = mesh.find_nodes("Z", mesh.bounding_box.zmin)
    top = mesh.find_nodes("Z", mesh.bounding_box.zmax)
    rotation_anchor = int(
        np.intersect1d(
            bottom,
            np.intersect1d(
                mesh.find_nodes("X", mesh.bounding_box.xmax),
                mesh.find_nodes("Y", mesh.bounding_box.ymin),
            ),
        )[0]
    )
    pb.bc.add("Dirichlet", bottom, "DispZ", 0)
    pb.bc.add("Dirichlet", [0], "Disp", 0)
    # Remove the remaining rigid rotation about Z without restraining the
    # homogeneous lateral contraction (the anchor lies on Y = ymin = 0).
    pb.bc.add("Dirichlet", [rotation_anchor], "DispY", 0)
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
        convergence, subiter, error = pb.solve_time_increment()
        assert convergence, (
            "EPICP single increment did not converge: "
            f"subiter={subiter}, error={error:.16e}"
        )
    else:
        pb.nlsolve(dt=0.2, tmax=1.0, update_dt=True, print_info=0)

    n_nodes = mesh.n_nodes
    blocked = {0, n_nodes, 2 * n_nodes, rotation_anchor + n_nodes}
    for n in np.concatenate([bottom, top]):
        blocked.add(int(n) + 2 * n_nodes)
    free = np.array([i for i in range(3 * n_nodes) if i not in blocked])
    return _directional_fd_error(pb, assembly, free)


_CASES = [
    # (law, corate).
    pytest.param("NEOHC", "log", id="NEOHC-log"),
    # the hyperelastic box is built in the corate sim.umat is given: fedoo must pass its own
    # (before, it got log_R's box and converted it with the Jaumann/GN map: 1e-4 in UL)
    pytest.param("NEOHC", "jaumann", id="NEOHC-jaumann"),
    pytest.param("NEOHC", "gn", id="NEOHC-gn"),
    pytest.param("ELISO", "log", id="ELISO-log"),
    pytest.param("ELISO", "log_r", id="ELISO-logR"),
    pytest.param(
        "FEDOO_ELISO",
        "log",
        id="FedooELISO-log",
    ),
    pytest.param(
        "FEDOO_ELISO",
        "log_r",
        id="FedooELISO-logR",
    ),
    pytest.param("EPICP", "log", id="EPICP-log"),
    pytest.param("EPICP", "log_r", id="EPICP-logR"),
]


# UL runs the rate formulation, with a corate; TL runs the total one, which has none (the
# native law keeps the log_r rate update there)
_FORMULATION_CASES = [
    pytest.param("UL", *c.values, id=f"UL-{c.id}") for c in _CASES
] + [
    pytest.param("TL", law, None, id=f"TL-{law}")
    for law in ("NEOHC", "ELISO", "FEDOO_ELISO", "EPICP")
]


@pytest.mark.parametrize("nlgeom, law, corate", _FORMULATION_CASES)
def test_finite_strain_tangent_is_fd_consistent(nlgeom, law, corate):
    err = _fd_tangent_error(nlgeom, law, corate)
    assert err < 1e-6, (
        f"{law} {nlgeom} corate={corate} global tangent inconsistent with FD "
        f"(rel error {err:.3e}): check the 3D geometric stiffness operator "
        "and the UL tangent conversion (box -> dS/dE -> Lie)"
    )


def test_native_elastic_matches_simcoon_eliso_at_finite_volume():
    """Native elasticity uses the same Cauchy/box convention as ELISO."""

    E, nu = 8.7, 0.3
    F = np.diag([1.10, 0.97, 1.03])
    F_field = np.asfortranarray(F[:, :, None])
    identity = np.asfortranarray(np.eye(3)[:, :, None])
    strain = simcoon.Log_strain(F_field, True, False)
    _, rotation, _ = simcoon.objective_rate("log_R", identity, F_field, 1.0, False)
    zeros = np.zeros((6, 1), order="F")
    simcoon_stress, _, _, simcoon_tangent = simcoon.umat(
        "ELISO",
        zeros,
        np.asfortranarray(strain),
        identity,
        F_field,
        zeros,
        np.asfortranarray(rotation),
        np.asfortranarray([[E], [nu], [1.0e-5]]),
        np.zeros((1, 1), order="F"),
        0.0,
        1.0,
        np.zeros((4, 1), order="F"),
        np.zeros((1, 1), order="F"),
        ndi=3,
        tangent_mode=1,
    )

    class MaterialPointAssembly:
        _nlgeom = "UL"
        n_gauss_points = 1
        space = SimpleNamespace(get_dimension=lambda: "3D")

        def __init__(self):
            self.sv = {
                "Strain": StrainTensorList(strain),
                "F": F_field,
            }

        @staticmethod
        def convert_data(value):
            return value

    assembly = MaterialPointAssembly()
    native = fd.constitutivelaw.ElasticIsotrop(E, nu)
    native.update(assembly, None)

    np.testing.assert_allclose(
        assembly.sv["Stress"].asarray(), simcoon_stress, rtol=1.0e-12, atol=1.0e-12
    )
    np.testing.assert_allclose(
        assembly.sv["TangentMatrix"],
        simcoon_tangent[:, :, 0],
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_heterogeneous_finite_strain_initializes_F_for_all_phases():
    """A UMAT phase must not leave a native phase with a singular initial F."""

    fd.Assembly.delete_memory()
    fd.ModelingSpace("2D")
    mesh = fd.mesh.rectangle_mesh(nx=3, ny=2, elm_type="quad4")
    mesh.element_sets["umat"] = np.array([0])
    mesh.element_sets["native"] = np.array([1])

    umat = fd.constitutivelaw.Simcoon("ELISO", [8.7, 0.3, 1.0e-5])
    native = fd.constitutivelaw.ElasticIsotrop(8.7, 0.3)
    material = fd.constitutivelaw.Heterogeneous((umat, native), ("umat", "native"))
    weakform = fd.weakform.StressEquilibrium(material, nlgeom="UL")
    assembly = fd.Assembly.create(weakform, mesh)
    problem = fd.problem.NonLinear(assembly)

    problem.initialize()

    expected = np.repeat(np.eye(3)[:, :, None], assembly.n_gauss_points, axis=2)
    np.testing.assert_allclose(assembly.sv_start["F"], expected)


def test_convert_tangent_false_preserves_user_tangent_in_tl():
    """The bypass keeps dS/dE untouched while still updating PK2 stress."""

    fd.ModelingSpace("3D")
    material = fd.constitutivelaw.ElasticIsotrop(8.7, 0.3)
    weakform = fd.weakform.StressEquilibrium(
        material, nlgeom="TL", convert_tangent=False
    )
    tangent = np.arange(36.0).reshape(6, 6)
    stress = StressTensorList(np.arange(1.0, 7.0).reshape(6, 1))
    F = np.asfortranarray(np.diag([1.1, 0.9, 1.05])[:, :, None])
    assembly = SimpleNamespace(
        _nlgeom="TL",
        sv={"Stress": stress, "TangentMatrix": tangent, "F": F},
    )
    problem = SimpleNamespace(_line_search_update=False)

    weakform.update_2(assembly, problem)

    assert assembly.sv["TangentMatrix"] is tangent
    np.testing.assert_allclose(
        assembly.sv["PK2"].asarray(), stress.cauchy_to_pk2(F).asarray()
    )


def _fd_history_tangent_error(gamma1, corate, n_inc=4, dgamma=0.05, nlgeom="UL"):
    """FD consistency WITH committed plastic history (rotating simple shear).

    EPICP cube: n_inc committed increments to gamma1 (set_start after
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
    wf = fd.weakform.StressEquilibrium(material, nlgeom=nlgeom)
    if (
        corate is not None
    ):  # None: nlgeom="TL", the total formulation, which has no rate
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


def _solved_stretch(nlgeom, corate):
    """EPICP box stretched past yield in one increment; return its assembly."""
    fd.Assembly.delete_memory()
    fd.ModelingSpace("3D")
    mesh = fd.mesh.box_mesh(nx=2, ny=2, nz=2, elm_type="hex8", name="box")
    material = fd.constitutivelaw.Simcoon("EPICP", EPICP_PROPS, name="law")
    material.tangent_mode = 2
    wf = fd.weakform.StressEquilibrium(material, nlgeom=nlgeom)
    if (
        corate is not None
    ):  # None: nlgeom="TL", the total formulation, which has no rate
        wf.corate = corate
    assembly = fd.Assembly.create(wf, mesh, name="asm")
    pb = fd.problem.NonLinear("asm")
    pb.set_nr_criterion("Displacement", err0=1.0, tol=1e-10, max_subiter=30)
    bottom = mesh.find_nodes("Z", mesh.bounding_box.zmin)
    top = mesh.find_nodes("Z", mesh.bounding_box.zmax)
    pb.bc.add("Dirichlet", bottom, "Disp", 0)
    pb.bc.add("Dirichlet", top, "DispZ", STRETCH)
    pb.nlsolve(dt=0.5, tmax=1.0, update_dt=False, print_info=0)
    return assembly


@pytest.mark.parametrize(
    "nlgeom, corate", [("UL", "log_r"), ("UL", "jaumann"), ("TL", None)]
)
def test_tangent_returned_by_the_law_matches_the_weakform_conversion(
    nlgeom, corate, monkeypatch
):
    """sim.umat(tangent_output=...) gives what update_2 computed from the box tangent."""
    fused = _solved_stretch(nlgeom, corate)
    assert fused._tangent_converted
    monkeypatch.setattr(fd.constitutivelaw.Simcoon, "supports_tangent_output", False)
    chained = _solved_stretch(nlgeom, corate)
    assert not chained._tangent_converted
    np.testing.assert_allclose(
        fused.sv["Stress"].asarray(),
        chained.sv["Stress"].asarray(),
        rtol=1e-12,
        atol=1e-12,
    )
    Lt = chained.sv["TangentMatrix"]
    np.testing.assert_allclose(
        fused.sv["TangentMatrix"], Lt, rtol=0, atol=1e-10 * np.abs(Lt).max()
    )


# ---------------------------------------------------------------------------------------------
# The committed state. A converged Cauchy stress lives in the end configuration and is
# committed (and written to the results) as it is; the start state of the next increment is
# transported with THAT increment's rotation, when the law is called.
# ---------------------------------------------------------------------------------------------


def _homogeneous_shear_states(nlgeom, corate, gamma, n_inc, law="ELISO", history=None):
    """Every node driven in simple shear, n_inc increments solved one by one. Returns the
    stress of the last increment as converged, then as committed by set_start. ``history``
    collects the converged stress of every increment."""
    fd.Assembly.delete_memory()
    fd.ModelingSpace("3D")
    mesh = fd.mesh.box_mesh(nx=2, ny=2, nz=2, elm_type="hex8", name="box")
    if law == "ELISO":
        material = fd.constitutivelaw.Simcoon("ELISO", [70000.0, 0.3, 1e-5], name="law")
    else:
        material = fd.constitutivelaw.Simcoon("EPICP", EPICP_PROPS, name="law")
        material.tangent_mode = 2
    wf = fd.weakform.StressEquilibrium(material, nlgeom=nlgeom)
    if (
        corate is not None
    ):  # None: nlgeom="TL", the total formulation, which has no rate
        wf.corate = corate
    assembly = fd.Assembly.create(wf, mesh, name="asm")
    pb = fd.problem.NonLinear("asm")
    pb.set_nr_criterion("Displacement", err0=1.0, tol=1e-10, max_subiter=30)
    nodes = np.arange(mesh.n_nodes)
    pb.bc.add("Dirichlet", nodes, "DispX", gamma * mesh.nodes[:, 2])
    pb.bc.add("Dirichlet", nodes, "DispY", 0)
    pb.bc.add("Dirichlet", nodes, "DispZ", 0)
    pb.initialize()
    pb.tmax = 1.0
    pb.dtime = 1.0 / n_inc
    pb.set_start()
    pb.time = 0.0
    for i in range(n_inc):
        conv, _, _ = pb.solve_time_increment()
        assert conv
        converged = assembly.sv["Stress"].asarray().copy()
        if history is not None:
            history.append(converged)
        work = np.asarray(assembly.sv["Wm"]).copy()
        pb.set_start()
        pb.time = (i + 1) * pb.dtime
    return converged, assembly.sv["Stress"].asarray().copy(), work, assembly


def _elastic_shear_reference(gamma):
    F = np.array([[1.0, 0.0, gamma], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    b, N = np.linalg.eigh(F @ F.T)
    lnV = N @ np.diag(0.5 * np.log(b)) @ N.T
    strain = np.array(
        [lnV[0, 0], lnV[1, 1], lnV[2, 2], 2 * lnV[0, 1], 2 * lnV[0, 2], 2 * lnV[1, 2]]
    )
    return simcoon.L_iso([70000.0, 0.3], "Enu") @ strain, strain


@pytest.mark.parametrize(
    "nlgeom, corate",
    [("UL", c) for c in ("log_r", "log", "log_r_inc", "log_inc", "jaumann", "gn")]
    + [("TL", None)],
)
def test_committed_stress_is_the_converged_stress(nlgeom, corate):
    """set_start used to rotate the converged stress once more by the increment's rotation:
    7 % on s_xx here, and that was the stress written to the result files."""
    converged, committed, _, assembly = _homogeneous_shear_states(
        nlgeom, corate, 0.6, 12
    )
    np.testing.assert_array_equal(committed, converged)
    if nlgeom == "TL":
        np.testing.assert_allclose(
            np.asarray(assembly.sv["PK2"].asarray()),
            np.asarray(assembly.sv["Stress"].cauchy_to_pk2(assembly.sv["F"]).asarray()),
            atol=1e-9,
        )


@pytest.mark.parametrize("corate", ["log_r", "log"])
def test_elastic_shear_is_exact_with_the_total_log_strain(corate):
    """Isotropic elasticity on ln V recomputed from F: tau = L : ln V at any step, as
    converged and as committed, with a work equal to the stored energy at second order."""
    exact, strain = _elastic_shear_reference(0.6)
    scale = np.abs(exact).max()
    errors = []
    for n_inc in (12, 48):
        _, committed, work, _ = _homogeneous_shear_states("UL", corate, 0.6, n_inc)
        np.testing.assert_allclose(committed[:, 0], exact, atol=1e-10 * scale)
        errors.append(abs(work[0, 0] - 0.5 * exact @ strain))
    assert errors[1] < errors[0] / 8.0


def test_incremental_log_strain_converges_to_the_total_one():
    """ "log_inc" accumulates the logarithmic strain increment by increment. The start strain
    must be carried by the rotation of the increment being solved; left in the start
    configuration it gave 4 % of error on this stress at 12 increments instead of 0.1 %.
    The increment is first order or closed form depending on the simcoon version."""
    exact, _ = _elastic_shear_reference(0.6)
    scale = np.abs(exact).max()
    err = [
        np.abs(
            _homogeneous_shear_states("UL", "log_inc", 0.6, n)[0][:, 0] - exact
        ).max()
        for n in (12, 48)
    ]
    assert err[0] < 2e-3 * scale
    assert err[1] < max(err[0] / 3.0, 1e-9 * scale)


def _objective_rate_returns_the_solver_increment():
    """simcoon's objective_rate used to return its own strain increment, which did not
    accumulate to ln V under "log_R". The corrected one is closed form under "log": one
    increment from the identity gives ln V exactly."""
    F = np.array([[1.0, 0.0, 0.6], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    de = np.asarray(simcoon.objective_rate("log", np.eye(3), F, 1.0, True)[0]).ravel()
    return np.allclose(de, _elastic_shear_reference(0.6)[1], rtol=0.0, atol=1e-10)


def test_incremental_log_r_follows_the_total_one_through_plasticity():
    """ "log_r_inc" and "log_r" are the same law integrated from an increment or from the total
    strain: through a plastic simple shear their stresses converge to each other."""
    if not _objective_rate_returns_the_solver_increment():
        pytest.skip(
            "this simcoon returns a log_R increment that does not accumulate to ln V"
        )
    reference = _homogeneous_shear_states("UL", "log_r", 0.6, 192, law="EPICP")[0][:, 0]
    err = [
        np.abs(
            _homogeneous_shear_states("UL", "log_r_inc", 0.6, n, law="EPICP")[0][:, 0]
            - reference
        ).max()
        for n in (12, 48)
    ]
    assert err[1] < err[0] / 3.0
    assert err[1] < 6e-3 * np.abs(reference).max()


def test_start_stress_is_transported_with_the_current_rotation():
    """The law receives the committed stress carried by the rotation of the increment being
    solved. It used to receive it carried by the rotation of the previous increment."""
    from simcoon import Rotation

    calls = []
    original = fd.constitutivelaw.Simcoon._call_umat

    def spy(self, etot, detot, F0, F1, stress, DR, *args, **kwargs):
        calls.append((np.array(stress), np.array(DR)))
        return original(self, etot, detot, F0, F1, stress, DR, *args, **kwargs)

    history = []
    fd.constitutivelaw.Simcoon._call_umat = spy
    try:
        _homogeneous_shear_states("UL", "log_r", 0.6, 6, law="EPICP", history=history)
    finally:
        fd.constitutivelaw.Simcoon._call_umat = original
    # last call = last iteration of the last increment, started from the 5th converged state
    stress, DR = calls[-1]
    expected = Rotation.from_matrix(DR.transpose(2, 0, 1)).apply_stress(history[-2])
    assert np.abs(history[-2]).max() > 100.0
    assert np.abs(expected - history[-2]).max() > 1.0  # the rotation is not negligible
    np.testing.assert_allclose(stress, expected, atol=1e-9)


# ---------------------------------------------------------------------------------------------
# nlgeom="TL": the total formulation, the strain-space framework of Miehe, Apel and Lambrecht
# (2002). The law works on ln U on the reference frame and on the stress T dual to it, where
# nlgeom="UL" (the rate formulation) works on ln V and the Kirchhoff stress. The tangent is
# P^T (dT/dH) P + T:L, whose geometric term has no counterpart in the log_r map.
# ---------------------------------------------------------------------------------------------


# The total formulation needs a simcoon that provides it (after 2.1); with an older one
# nlgeom="TL" keeps the log_r rate update.
_needs_total_formulation = pytest.mark.skipif(
    not hasattr(simcoon, "total_formulation_serves"),
    reason="the installed simcoon has no total formulation",
)


@_needs_total_formulation
def test_total_formulation_tangent_with_plastic_history():
    """Committed plastic shear increments, then the tangent of one more: T is not coaxial
    with U here, so the geometric term T:L and the dual-stress recovery are both loaded.
    (The log_r rate update that TL keeps with simcoon 2.1 does not pass this check: its dS/dE
    is not symmetric in a non-coaxial state, which the TL assembly assumes.)"""
    err = _fd_history_tangent_error(0.2, None, nlgeom="TL")
    assert err < 1e-6, f"TL EPICP with plastic history: FD error {err:.3e}"


def _homogeneous_shear(nlgeom, corate, gamma=0.6, n_inc=12, law="EPICP"):
    """Every node driven in simple shear: one homogeneous F, n_inc equal increments."""
    fd.Assembly.delete_memory()
    fd.ModelingSpace("3D")
    mesh = fd.mesh.box_mesh(nx=2, ny=2, nz=2, elm_type="hex8", name="box")
    if law == "EPICP":
        material = fd.constitutivelaw.Simcoon("EPICP", EPICP_PROPS, name="law")
    elif law == "NEOHC":
        material = fd.constitutivelaw.Simcoon("NEOHC", [1000.0, 5000.0], name="law")
    elif law == "MODUL":
        from simcoon.modular import elastoplastic_model

        E, nu, alpha, sigma_Y, k, m = EPICP_PROPS
        material = fd.constitutivelaw.Simcoon.from_modular(
            elastoplastic_model(E=E, nu=nu, sigma_Y=sigma_Y, k=k, m=m, alpha=alpha),
            tangent_mode=2,
            name="law",
        )
    else:
        material = fd.constitutivelaw.Simcoon("ELISO", [70000.0, 0.3, 1e-5], name="law")
    material.tangent_mode = 2
    wf = fd.weakform.StressEquilibrium(material, nlgeom=nlgeom)
    if (
        corate is not None
    ):  # None: nlgeom="TL", the total formulation, which has no rate
        wf.corate = corate
    assembly = fd.Assembly.create(wf, mesh, name="asm")
    pb = fd.problem.NonLinear("asm")
    pb.set_nr_criterion("Displacement", err0=1.0, tol=1e-10, max_subiter=30)
    nodes = np.arange(mesh.n_nodes)
    pb.bc.add("Dirichlet", nodes, "DispX", gamma * mesh.nodes[:, 2])
    pb.bc.add("Dirichlet", nodes, "DispY", 0)
    pb.bc.add("Dirichlet", nodes, "DispZ", 0)
    pb.nlsolve(dt=1.0 / n_inc, tmax=1.0, update_dt=False, print_info=0)
    return assembly


@_needs_total_formulation
def test_total_formulation_reproduces_the_simcoon_solver():
    """Homogeneous plastic simple shear in TL: the Gauss-point state is the one simcoon's own
    solver integrates under its total formulation, with the same increments."""
    from simcoon.solver import Block, StepMeca

    gamma, n_inc = 0.6, 12
    assembly = _homogeneous_shear("TL", None, gamma, n_inc)
    F = [1.0, gamma, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    step = StepMeca(
        control="F", value=F, time=1.0, ninc=n_inc, Dn_init=1.0, Dn_mini=1e-6
    )
    # fedoo shears x along z: simcoon's 1-2 plane is fedoo's 1-3 plane
    ref = simcoon.solver.solve(
        Block(steps=[step], control_type="F"),
        "EPICP",
        np.array(EPICP_PROPS),
        8,
        T_init=0.0,
        kinematic_form="total",
    )["Stress"][:, -1]
    stress = assembly.sv["Stress"].asarray()
    expected = np.array([ref[0], ref[2], ref[1], ref[4], ref[3], ref[5]])
    np.testing.assert_allclose(
        stress, expected[:, None] * np.ones((1, stress.shape[1])), atol=1e-6
    )
    # ln U, not ln V, is what the law was fed
    U = np.array([[1.0, 0.0, gamma], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    c, N = np.linalg.eigh(U.T @ U)
    H = N @ np.diag(0.5 * np.log(c)) @ N.T
    np.testing.assert_allclose(
        assembly.sv["Strain"].asarray()[:, 0],
        [H[0, 0], H[1, 1], H[2, 2], 2 * H[0, 1], 2 * H[0, 2], 2 * H[1, 2]],
        atol=1e-12,
    )


def test_total_formulation_elastic_shear_is_exact_at_any_step():
    """Isotropic elasticity in simple shear: tau = L : ln V. The strain is a state function of
    F, so one increment gives it to round-off."""
    gamma = 0.6
    F = np.array([[1.0, 0.0, gamma], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    b, N = np.linalg.eigh(F @ F.T)
    lnV = N @ np.diag(0.5 * np.log(b)) @ N.T
    exact = simcoon.L_iso([70000.0, 0.3], "Enu") @ np.array(
        [lnV[0, 0], lnV[1, 1], lnV[2, 2], 2 * lnV[0, 1], 2 * lnV[0, 2], 2 * lnV[1, 2]]
    )
    scale = np.abs(exact).max()
    for n_inc in (1, 12):
        stress = (
            _homogeneous_shear("TL", None, gamma, n_inc, law="ELISO")
            .sv["Stress"]
            .asarray()
        )
        np.testing.assert_allclose(
            stress, exact[:, None] * np.ones((1, stress.shape[1])), atol=1e-10 * scale
        )


@_needs_total_formulation
def test_a_modular_material_runs_the_total_formulation_in_tl():
    """A modular J2 material is the EPICP box: in TL it runs the total formulation and gives
    the same stress."""
    modular = _homogeneous_shear("TL", None, law="MODUL")
    assert modular.weakform.kinematic_form == "total"
    stress = modular.sv["Stress"].asarray().copy()
    reference = _homogeneous_shear("TL", None, law="EPICP").sv["Stress"].asarray()
    np.testing.assert_allclose(stress, reference, atol=1e-6 * np.abs(reference).max())


def _polar_rotation(F):
    c, N = np.linalg.eigh(F.T @ F)
    return F @ N @ np.diag(1.0 / np.sqrt(c)) @ N.T


@_needs_total_formulation
def test_the_two_formulations_read_one_box_differently():
    """Plastic simple shear, where the stress is not coaxial with the stretch. The box and its
    strain are the same in TL (total) and in UL with log_r (rate): its output is T in one and
    the rotated Kirchhoff stress in the other, R^T tau_rate R = T_total, whatever the step.
    The two Cauchy stresses therefore differ, and the gap does not close with the step."""
    gamma = 0.6
    R = _polar_rotation(np.array([[1.0, 0.0, gamma], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
    gap = []
    for n_inc in (12, 48):
        total = _homogeneous_shear("TL", None, gamma, n_inc)
        s_total = total.sv["Stress"]
        assert (
            total.weakform.kinematic_form == "total" and total.weakform.corate is None
        )
        T = s_total.cauchy_to_log_stress_u(total.sv["F"])
        np.testing.assert_allclose(
            T.log_stress_u_to_cauchy(total.sv["F"]).asarray(),
            s_total.asarray(),
            atol=1e-9,
        )
        T = T.asarray()
        s_total = s_total.asarray().copy()
        rate = _homogeneous_shear("UL", "log_r", gamma, n_inc)
        assert rate.weakform.kinematic_form == "rate"
        s_rate = rate.sv["Stress"].asarray()
        box_rate = R.T @ _sig_to_mat(s_rate[:, 0]) @ R  # J = 1 in simple shear
        np.testing.assert_allclose(
            _sig_to_mat(T[:, 0]), box_rate, atol=1e-6 * np.abs(s_rate).max()
        )
        gap.append(np.abs(s_total - s_rate).max())
        assert gap[-1] > 1e-3 * np.abs(s_rate).max()
    assert gap[1] > 0.8 * gap[0]


@_needs_total_formulation
def test_a_law_built_on_F_is_the_same_in_both_formulations():
    """NEOHC is a function of F alone: TL (total) and UL (rate) give the same stress."""
    total = _homogeneous_shear("TL", None, 0.6, 4, law="NEOHC")
    assert total.weakform.kinematic_form == "total"
    s_total = total.sv["Stress"].asarray().copy()
    s_rate = (
        _homogeneous_shear("UL", "log_r", 0.6, 4, law="NEOHC").sv["Stress"].asarray()
    )
    np.testing.assert_allclose(s_total, s_rate, atol=1e-8 * np.abs(s_rate).max())


@_needs_total_formulation
def test_the_total_lagrangian_method_takes_no_corate():
    with pytest.raises(ValueError, match="no rate"):
        _homogeneous_shear("TL", "log_r", n_inc=1)


def test_a_law_without_the_total_formulation_keeps_the_rate_update_in_tl():
    """A native law does not declare the total formulation: TL runs it as before."""
    fd.Assembly.delete_memory()
    fd.ModelingSpace("3D")
    mesh = fd.mesh.box_mesh(nx=2, ny=2, nz=2, elm_type="hex8", name="box")
    material = fd.constitutivelaw.ElasticIsotrop(8.7, 0.3, name="law")
    wf = fd.weakform.StressEquilibrium(material, nlgeom="TL")
    fd.Assembly.create(wf, mesh, name="asm")
    pb = fd.problem.NonLinear("asm")
    nodes = np.arange(mesh.n_nodes)
    pb.bc.add("Dirichlet", nodes, "Disp", 0)
    pb.bc.add("Dirichlet", nodes, "DispX", 0.1 * mesh.nodes[:, 2])
    pb.nlsolve(dt=1.0, tmax=1.0, update_dt=False, print_info=0)
    assert wf.kinematic_form == "rate" and wf.corate == "log_r"


@_needs_total_formulation
def test_log_measures_are_results_of_either_formulation():
    """LogStrainU, LogStrainV and LogStressU express a run in the measures of the other
    formulation: ln V = R ln U R^T, and T is the stress dual to ln U."""
    gamma = 0.6
    R = _polar_rotation(np.array([[1.0, 0.0, gamma], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
    eng = np.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])
    fields = ["Strain", "LogStrainU", "LogStrainV", "LogStressU"]
    for nlgeom, corate in (("TL", None), ("UL", "log_r")):
        assembly = _homogeneous_shear(nlgeom, corate, gamma, 4)
        data = (
            fd.Problem.get_active()
            .get_results("asm", fields, "GaussPoint")
            .gausspoint_data
        )
        lnU, lnV = data["LogStrainU"][:, 0], data["LogStrainV"][:, 0]
        # the strain of the law: ln U under the total formulation, ln V under the rate one
        np.testing.assert_allclose(
            data["Strain"][:, 0], lnU if nlgeom == "TL" else lnV, atol=1e-12
        )
        np.testing.assert_allclose(
            R @ _sig_to_mat(lnU * eng) @ R.T, _sig_to_mat(lnV * eng), atol=1e-12
        )
        np.testing.assert_allclose(
            data["LogStressU"],
            assembly.sv["Stress"].cauchy_to_log_stress_u(assembly.sv["F"]).asarray(),
            atol=1e-12,
        )
    # a strain accumulated by a rate that does not integrate to a function of F
    _homogeneous_shear("UL", "jaumann", gamma, 2)
    with pytest.raises(NameError, match="state function"):
        fd.Problem.get_active().get_results("asm", ["LogStrainU"], "GaussPoint")


@_needs_total_formulation
def test_a_law_simcoon_does_not_serve_keeps_the_rate_update_in_tl():
    """simcoon decides which of its laws run under the total formulation: a small-strain-only
    law (here the SMA model) keeps the log_r rate update in TL instead of being refused."""
    assert fd.constitutivelaw.Simcoon("EPICP", EPICP_PROPS).supports_total_formulation
    assert fd.constitutivelaw.Simcoon(
        "NEOHC", [1000.0, 5000.0]
    ).supports_total_formulation
    for name in ("SMADI", "ZENER", "HYPOO"):
        assert not simcoon.total_formulation_serves(name)


def test_corate_can_be_unset_and_is_refused_only_with_the_total_formulation():
    """The corate belongs to the rate formulation. In TL it is refused when the law runs the
    total formulation, and accepted when the law keeps the rate update (a native law)."""
    fd.Assembly.delete_memory()
    fd.ModelingSpace("3D")
    mesh = fd.mesh.box_mesh(nx=2, ny=2, nz=2, elm_type="hex8", name="box")
    native = fd.constitutivelaw.ElasticIsotrop(8.7, 0.3, name="law")
    wf = fd.weakform.StressEquilibrium(native, nlgeom="TL")
    wf.corate = "log"
    fd.Assembly.create(wf, mesh, name="asm")
    pb = fd.problem.NonLinear("asm")
    pb.bc.add("Dirichlet", np.arange(mesh.n_nodes), "Disp", 0)
    pb.nlsolve(dt=1.0, tmax=1.0, update_dt=False, print_info=0)
    assert wf.kinematic_form == "rate" and wf.corate == "log"
    wf.corate = None
    assert wf.corate == "log_r"
