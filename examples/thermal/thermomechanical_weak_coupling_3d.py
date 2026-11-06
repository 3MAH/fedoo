"""
Weakly coupled thermo-mechanical analysis of a gyroid lattice
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Staggered (weak) thermo-mechanical coupling on a 3D gyroid lattice: a transient
:class:`fedoo.weakform.HeatEquation` problem and a small-strain
:class:`fedoo.weakform.StressEquilibrium` problem with thermal expansion are
advanced increment by increment with
:py:meth:`fedoo.problem.NonLinear.solve_time_increment`. At each increment the
converged temperature is copied into the mechanical assembly. The +X face is
heated to :math:`T = 100` during the first phase and cooled back to
:math:`T = 0` during the second one; all faces are clamped.

The material values are illustrative: the thermal properties are given in SI
units while the mechanical ones are in MPa, and the thermal expansion
coefficient is exaggerated to amplify the effect.

About 15 000 nodes and 2 x 100 increments of both problems (about 30 min). The
last section reads the saved thermal and mechanical results and animates the
temperature on the deformed lattice.
"""

from pathlib import Path

import fedoo as fd
import numpy as np
import pyvista as pv

###############################################################################
# Mesh and boundary node sets
# ---------------------------

try:
    example_dir = Path(__file__).resolve().parent
except NameError:  # interactive session
    example_dir = Path.cwd()
mesh_file = example_dir.parents[1] / "util" / "meshes" / "gyroid.msh"
fd.mesh.import_file(str(mesh_file), name="Domain")
meshname = "Domain"
filename_res = str(example_dir / "results" / "thermo_meca_nl")

mesh = fd.Mesh[meshname]

# node sets for boundary conditions
Xmin, Xmax = mesh.bounding_box
left = mesh.find_nodes("X", Xmin[0])
right = mesh.find_nodes("X", Xmax[0])
bottom = mesh.find_nodes("Y", Xmin[1])
top = mesh.find_nodes("Y", Xmax[1])
back = mesh.find_nodes("Z", Xmin[2])
front = mesh.find_nodes("Z", Xmax[2])


boundary = np.unique(np.hstack((bottom, top, left, right, back, front)))

###############################################################################
# Thermal problem
# ---------------
# Each problem has its own modeling space. The thermal values are
# illustrative: the conductivity is about ten times that of a steel and the
# specific heat about 1000 times smaller.

thermal_space = fd.ModelingSpace("3D")


K = 500  # thermal conductivity, W/(m.K) (illustrative value)
c = 0.500  # specific heat, J/(kg.K) (illustrative value)
rho = 7800  # density, kg/m3
thermal_law = fd.constitutivelaw.ThermalProperties(K, c, rho, name="ThermalLaw")
wf_th = fd.weakform.HeatEquation(thermal_law, space=thermal_space)
assemb = fd.Assembly.create(wf_th, meshname, name="Assembling_T")

pb_th = fd.problem.NonLinear("Assembling_T")
pb_th.set_time_integrator(fd.time.FIRST_ORDER, fd.time.BackwardEuler())

pb_th.set_nr_criterion("Displacement", tol=5e-2, max_subiter=5, err0=100)

###############################################################################
# Mechanical problem
# ------------------
# Small strain problem (MPa units) with thermal expansion. ``mat`` selects the
# law: 0 elastic (Simcoon ``ELISO``), 1 Simcoon ``EPICP``, 2 native
# :class:`fedoo.constitutivelaw.ElastoPlasticity`.

mech_space = fd.ModelingSpace("3D")

E = 200e3
nu = 0.3
alpha = 1e-3  # thermal expansion coefficient (large, to amplify the effect)
NLGEOM = False
mat = 0
if mat == 0:
    props = np.array([E, nu, alpha])
    mechanical_law = fd.constitutivelaw.Simcoon("ELISO", props, name="MechanicalLaw")
elif mat == 1 or mat == 2:
    Re = 300
    k = 1000
    m = 1
    if mat == 1:
        props = np.array([E, nu, alpha, Re, k, m])
        mechanical_law = fd.constitutivelaw.Simcoon(
            "EPICP", props, name="MechanicalLaw"
        )
    elif mat == 2:
        mechanical_law = fd.constitutivelaw.ElastoPlasticity(
            E, nu, Re, name="MechanicalLaw"
        )
        mechanical_law.set_hardening_function("power", h=k, beta=m)
else:
    mechanical_law = fd.constitutivelaw.ElasticIsotrop(E, nu, name="MechanicalLaw")

wf_mech = fd.weakform.StressEquilibrium("MechanicalLaw", nlgeom=NLGEOM)

fd.Assembly.create(wf_mech, meshname, name="Assembling_M")

pb_m = fd.problem.NonLinear("Assembling_M")
pb_m.set_nr_criterion("Displacement", tol=1e-2, err0=1)

###############################################################################
# Outputs and boundary conditions
# -------------------------------
# The results are written in the ``results`` folder next to this script. The
# temperature of the +X face rises linearly to 100 during the first half of
# the heating phase, then stays constant. All the faces are clamped.

res_m = pb_m.add_output(
    filename_res + "_me",
    "Assembling_M",
    ["Disp", "Stress", "Stress_vm"],
)

res_th = pb_th.add_output(filename_res + "_th", "Assembling_T", ["Temp"])


def timeEvolution(timeFactor):
    if timeFactor < 0.5:
        return 2 * timeFactor
    else:
        return 1


pb_th.bc.add("Dirichlet", right, "Temp", 100, time_func=timeEvolution, name="temp")

pb_m.bc.add("Dirichlet", boundary, "Disp", 0)

###############################################################################
# Heating phase: staggered solution
# ---------------------------------
# At each increment the thermal problem is solved first, then its converged
# temperature at the Gauss points is copied into the state variables of the
# mechanical assembly before the mechanical increment is solved.

nb_iter = 100
tmax = 30
dt = tmax / nb_iter

pb_th.initialize()
pb_m.initialize()

pb_th.tmax = tmax
pb_m.tmax = tmax
pb_th.dtime = pb_m.dtime = dt

pb_th.set_start()
pb_m.set_start()

for i in range(nb_iter):
    pb_th.time = pb_m.time = time = i * dt
    convergence, nbNRiter, normRes = pb_th.solve_time_increment()
    if not (convergence):
        print("WARNING: iteration", i, "has not converged: err =", normRes)
    pb_th.set_start()
    pb_th.save_results(i)

    print(
        "Iter {} - Therm - Time: {:.5f} - NR iter: {} - Err: {:.5f}".format(
            i, time, nbNRiter, normRes
        )
    )
    pb_m.assembly.sv["Temp"] = pb_th.assembly.sv["Temp"]

    convergence, nbNRiter, normRes = pb_m.solve_time_increment()
    assert convergence, "mechanical problem has not converged"
    pb_m.set_start()
    pb_m.save_results(i)

    print(
        "Iter {} - Mech - Time: {:.5f} - NR iter: {} - Err: {:.5f}".format(
            i, time, nbNRiter, normRes
        )
    )

###############################################################################
# Cooling phase
# -------------
# The imposed temperature goes from its current value (100) to 0 with the same
# time function. Without ``start_value="Current"``, the Dirichlet increments
# would be computed from a zero start value and the face would stay at 100.
pb_th.bc[0].change_value(0, start_value="Current")

pb_th.t0 = tmax
pb_th.tmax = 2 * tmax
pb_m.t0 = tmax
pb_m.tmax = 2 * tmax

for i in range(nb_iter, 2 * nb_iter):
    # the time must be advanced and each increment closed by set_start, as in the
    # first phase, otherwise every increment is solved again from the same state
    pb_th.time = pb_m.time = time = i * dt
    convergence, nbNRiter, normRes = pb_th.solve_time_increment()
    if not (convergence):
        print("WARNING: iteration", i, "has not converged: err =", normRes)
    pb_th.set_start()
    pb_th.save_results(i)

    print(
        "Iter {} - Therm - Time: {:.5f} - NR iter: {} - Err: {:.5f}".format(
            i, time, nbNRiter, normRes
        )
    )
    pb_m.assembly.sv["Temp"] = pb_th.assembly.sv["Temp"]

    convergence, nbNRiter, normRes = pb_m.solve_time_increment()
    assert convergence, "mechanical problem has not converged"
    pb_m.set_start()
    pb_m.save_results(i)

    print(
        "Iter {} - Mech - Time: {:.5f} - NR iter: {} - Err: {:.5f}".format(
            i, time, nbNRiter, normRes
        )
    )

###############################################################################
# Animation
# ---------
# ``res_m.write_movie("results/stress_field", "Stress_vm")`` writes a movie of
# a single result file. To show the temperature on the deformed lattice
# (displacement amplified 5 times), both result files are read back with
# :func:`fedoo.read_data`, the mechanical displacement is copied into the
# thermal data set, and the frames are written into
# ``results/thermomechanical_weak_coupling_3d.gif`` with a rotating camera.
# ``DataSet.plot`` moves the nodes by the ``Disp`` field when ``scale`` is not
# zero.

thermal_results = fd.read_data(filename_res + "_th.fdh5")
mechanical_results = fd.read_data(filename_res + "_me.fdh5")
n_frames = min(thermal_results.n_iter, mechanical_results.n_iter)
frames = range(0, n_frames, 2)  # one saved increment out of two

plotter = pv.Plotter(window_size=(800, 600), off_screen=True)
plotter.set_background("white")
plotter.open_gif(
    str(example_dir / "results" / "thermomechanical_weak_coupling_3d.gif"), fps=20
)
for frame, iteration in enumerate(frames):
    thermal_results.load(iteration)
    mechanical_results.load(iteration)
    thermal_results.node_data["Disp"] = mechanical_results.node_data["Disp"]
    thermal_results.plot(
        "Temp",
        data_type="Node",
        scale=5,
        show=False,
        show_edges=True,
        clim=(0, 100),
        scalar_bar_args={
            "title_font_size": 20,
            "label_font_size": 16,
            "color": "black",
        },
        title="",
        name="thermo-mechanical",
        plotter=plotter,
        lock_view=frame > 0,
    )
    if frame == 0:
        plotter.camera.SetFocalPoint(thermal_results.mesh.bounding_box.center)
        plotter.camera.position = (-2.090, 1.758, 1.708)
    plotter.camera.Azimuth(360 / len(frames))
    plotter.write_frame()
plotter.close()
