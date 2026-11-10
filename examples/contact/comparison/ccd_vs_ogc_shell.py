r"""
CCD vs OGC step filtering on shells (sphere pressed on a plate)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A thin spherical shell is pushed down onto a clamped square plate, both
meshed with ``tri3`` shell elements. The same IPC barrier contact is solved
twice; only the way the Newton-Raphson step is kept intersection-free
changes:

- **CCD** (``use_ccd=True``, default): continuous collision detection gives
  the largest step fraction :math:`\alpha \in (0, 1]` that keeps all
  surfaces apart, and the *whole* displacement increment is scaled by
  :math:`\alpha`, followed by an energy backtracking.
- **OGC** (``use_ogc=True``): the Offset Geometric Contact trust region
  clamps the move of *each vertex* separately, so vertices far from the
  contact zone take their full Newton step.

OGC filters vertex positions, so it is only valid when every node of the
mesh lies on the contact surface: shells and membranes. On a solid mesh,
interior nodes would not be filtered and ``IPCContact`` raises a
``ValueError``; use CCD there.

The shells are linear (``nlgeom=False``): the only nonlinearity is the
contact, which keeps the comparison focused on the step filter.

Contact is detected between the shell *mid-surfaces* with
:math:`\hat{d} = 0.05` mm, whereas the shells are 0.5 mm (sphere) and
1.0 mm (plate) thick: the half-thicknesses are not offset, so the material
volumes of the two shells interpenetrate at contact. This is acceptable for
comparing the step filters, not for a physical contact pressure.

.. note::
   This example requires the ``ipctk`` package (``pip install fedoo[ipc]``).
"""

from time import time

import fedoo as fd
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

###############################################################################
# Parameters
# ----------
# Units: N, mm, MPa.

E, NU = 70e3, 0.3
RADIUS = 10.0  # sphere radius
T_SPHERE = 0.5  # sphere shell thickness
T_PLATE = 1.0  # plate thickness
HALF_WIDTH = 30.0  # plate half width
GAP = 0.25  # initial gap between the sphere and the plate
PUSH = 1.5  # downward displacement imposed on the top of the sphere
DHAT = 0.05  # absolute barrier activation distance (< GAP)


###############################################################################
# Shell meshes
# ------------
# A single ``tri3`` mesh holds both shells (``Mesh.__add__`` stacks the nodes
# without merging), so that the IPC contact surface is the full mesh: every
# node is a surface node, as required by OGC.


def build_mesh():
    sphere = fd.Mesh.from_pyvista(
        pv.Sphere(
            radius=RADIUS,
            center=(0.0, 0.0, GAP + RADIUS),
            theta_resolution=24,
            phi_resolution=24,
        )
    )
    plate = fd.mesh.rectangle_mesh(
        31,
        31,
        -HALF_WIDTH,
        HALF_WIDTH,
        -HALF_WIDTH,
        HALF_WIDTH,
        elm_type="tri3",
        ndim=3,
    )
    mesh = sphere + plate
    mesh.element_sets["sphere"] = np.arange(sphere.n_elements)
    mesh.element_sets["plate"] = np.arange(sphere.n_elements, mesh.n_elements)
    return mesh, sphere.n_nodes


###############################################################################
# Problem builder
# ---------------
# The two shells have different thicknesses, so each gets its own shell
# section and assembly, built on the extracted elements (which keep the
# global node numbering of ``mesh``). The plate edges are clamped; the top cap
# of the sphere is driven downward.


def build_problem(method):
    mesh, n_sphere_nodes = build_mesh()
    fd.ModelingSpace("3D")

    material = fd.constitutivelaw.ElasticIsotrop(E, NU)
    sphere_wf = fd.weakform.PlateEquilibrium(
        fd.constitutivelaw.ShellHomogeneous(material, T_SPHERE)
    )
    plate_wf = fd.weakform.PlateEquilibrium(
        fd.constitutivelaw.ShellHomogeneous(material, T_PLATE)
    )
    structure = fd.Assembly.create(
        sphere_wf, mesh.extract_elements("sphere")
    ) + fd.Assembly.create(plate_wf, mesh.extract_elements("plate"))

    contact = fd.constraint.IPCContact(
        mesh,
        surface_mesh=mesh,
        dhat=DHAT,
        dhat_is_relative=False,
        use_ccd=(method == "CCD"),
        use_ogc=(method == "OGC"),
    )

    pb = fd.problem.NonLinear(structure + contact)

    x, y, z = mesh.nodes.T
    plate_nodes = np.arange(n_sphere_nodes, mesh.n_nodes)
    edges = plate_nodes[
        np.isclose(np.abs(x[plate_nodes]), HALF_WIDTH)
        | np.isclose(np.abs(y[plate_nodes]), HALF_WIDTH)
    ]
    cap = np.where(z > z.max() - 1.0)[0]

    pb.bc.add("Dirichlet", edges, "Disp", 0)
    pb.bc.add("Dirichlet", edges, "Rot", 0)
    pb.bc.add("Dirichlet", cap, "Disp", [0, 0, -PUSH])
    # Reference IPC Newton setting (displacement criterion).
    pb.set_nr_criterion("Displacement", tol=5e-3, max_subiter=15)
    return pb, structure, cap


###############################################################################
# Instrumented solve
# ------------------
# ``solve_time_increment`` is wrapped to record the number of Newton-Raphson
# iterations of each converged increment; the callback records the force
# needed to push the cap.


def run(method):
    pb, structure, cap = build_problem(method)
    hist = {"time": [], "force": [], "nr_iters": []}

    solve_increment = pb.solve_time_increment

    def counted(max_subiter=None, tol_nr=None):
        convergence, nr_iter, err = solve_increment(max_subiter, tol_nr)
        if convergence:
            hist["nr_iters"].append(nr_iter)
        return convergence, nr_iter, err

    pb.solve_time_increment = counted

    def track(pb):
        hist["time"].append(pb.time)
        hist["force"].append(-np.sum(pb.get_ext_forces("Disp")[2, cap]))

    t0 = time()
    pb.nlsolve(dt=0.05, tmax=1, update_dt=True, print_info=0, callback=track)
    hist["wall_time"] = time() - t0
    return pb, structure, hist


histories = {}
for method in ("CCD", "OGC"):
    pb, structure, histories[method] = run(method)

print(f"{'':6s}{'increments':>12s}{'total NR':>10s}{'time (s)':>10s}")
for method, h in histories.items():
    print(
        f"{method:6s}{len(h['nr_iters']):12d}{sum(h['nr_iters']):10d}"
        f"{h['wall_time']:10.2f}"
    )

###############################################################################
# Comparison
# ----------
# Both filters solve the same barrier problem, so the force curves must
# coincide: the force is zero until the gap closes at
# :math:`t \approx` ``GAP / PUSH``, then grows as both shells deform. The
# difference lies in the Newton-Raphson iteration counts.

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for (method, h), marker in zip(histories.items(), ("o", "s")):
    axes[0].plot(
        np.array(h["time"]) * PUSH, h["force"], marker + "-", ms=3, label=method
    )
    axes[1].plot(np.arange(1, len(h["nr_iters"]) + 1), h["nr_iters"], marker + "-")
axes[0].set_xlabel("Imposed displacement of the cap (mm)")
axes[0].set_ylabel("Pushing force (N)")
axes[0].legend()
axes[1].set_xlabel("Increment")
axes[1].set_ylabel("NR iterations")
for ax in axes:
    ax.grid(True, alpha=0.3)
fig.tight_layout()

###############################################################################
# Deformed shells at the end of the OGC run.

res = pb.get_results(structure, ["Disp"])
res.plot("Disp", "Z", "Node", scale=1, azimuth=30, elevation=20)
