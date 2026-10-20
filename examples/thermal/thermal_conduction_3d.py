"""
Transient heat conduction in a gyroid lattice
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Transient heat conduction in a 3D gyroid lattice (tet4 mesh read from
``util/meshes/gyroid.msh``) with :class:`fedoo.weakform.HeatEquation`. The +X face
is suddenly brought to :math:`T = 100` (step time function); the other faces are
insulated and the initial temperature is zero.

About 15 000 nodes and 100 time increments (about 30 s).
"""

import os
from pathlib import Path

import fedoo as fd

###############################################################################
# Mesh and thermal law
# --------------------
# The material values are illustrative and are not those of a real material:
# the conductivity is about ten times that of a steel and the specific heat
# about 1000 times smaller, so that the heat diffuses quickly in the lattice.

fd.ModelingSpace("3D")

meshname = "Domain"
nb_iter = 100

try:
    example_dir = Path(__file__).resolve().parent
except NameError:  # interactive session
    example_dir = Path.cwd()
mesh_file = example_dir.parents[1] / "util" / "meshes" / "gyroid.msh"
fd.mesh.import_file(str(mesh_file), name=meshname)

mesh = fd.Mesh[meshname]

K = 500  # thermal conductivity, W/(m.K) (illustrative value)
c = 0.500  # specific heat, J/(kg.K) (illustrative value)
rho = 7800  # density, kg/m3
material = fd.constitutivelaw.ThermalProperties(K, c, rho, name="ThermalLaw")
wf = fd.weakform.HeatEquation(material)
assemb = fd.Assembly.create(wf, meshname, name="Assembling")

# node sets for boundary conditions
Xmin, Xmax = mesh.bounding_box
right = mesh.find_nodes("X", Xmax[0])

###############################################################################
# Problem, time integrator and boundary conditions
# ------------------------------------------------

pb = fd.problem.NonLinear("Assembling")
pb.set_time_integrator(fd.time.FIRST_ORDER, fd.time.BackwardEuler())
pb.set_nr_criterion("Displacement", tol=5e-2, max_subiter=5, err0=100)

if not (os.path.isdir("results")):
    os.mkdir("results")
results = pb.add_output("results/thermal3D", "Assembling", ["Temp"])

tmax = 10


def step_function(t_fact):
    """Impose the full value as soon as the first increment starts."""
    if t_fact == 0:
        return 0
    else:
        return 1


pb.bc.add("Dirichlet", right, "Temp", 100, time_func=step_function)

###############################################################################
# Solve and plot the temperature at an intermediate time
# ------------------------------------------------------
# The saved increment 46 is loaded and plotted. An animation of the whole
# history can be written with ``results.write_movie("results/thermal3D", "Temp")``.

pb.nlsolve(dt=tmax / nb_iter, tmax=tmax, update_dt=True)

cpos = [
    (-2.090457552750125, 1.7582929402632352, 1.707926514944027),
    (0.20739316009534275, -0.2296587829717462, -0.38339561081860574),
    (0.42357673667356105, -0.37693638734293083, 0.8237121512068624),
]

results.load(46)
pl = results.plot("Temp", show=False)
pl.camera_position = cpos
pl.show()
