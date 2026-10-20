"""
Rigid tie constraint
~~~~~~~~~~~~~~~~~~~~

Example of a 2D rigid tie with :class:`fedoo.constraint.RigidTie2D`. The bottom
face of a square is clamped and a rigid body rotation of 45° is imposed on the
top face through the ``RigidRotZ`` degree of freedom added by the constraint,
in finite strain (plane strain).
"""

import fedoo as fd
import numpy as np

fd.ModelingSpace("2D")  # plane strain assumption

NLGEOM = True
E = 200e3
nu = 0.3

mesh = fd.mesh.rectangle_mesh()
material = fd.constitutivelaw.ElasticIsotrop(E, nu, name="ConstitutiveLaw")

wf = fd.weakform.StressEquilibrium(material, nlgeom=NLGEOM, name="wf")

assembly = fd.Assembly.create("wf", mesh)

# node set for boundary conditions
bottom = mesh.find_nodes("Y", mesh.bounding_box.ymin)
top = mesh.find_nodes("Y", mesh.bounding_box.ymax)

pb = fd.problem.NonLinear(assembly)
pb.set_nr_criterion("Displacement", err0=1, tol=1e-2, max_subiter=5)

results = pb.add_output(
    "results/rigid_tie_example", assembly, ["Disp", "Stress", "Strain", "Fext"]
)

pb.bc.add(fd.constraint.RigidTie2D(top))

pb.bc.add("Dirichlet", bottom, "Disp", 0)
pb.bc.add("Dirichlet", "RigidRotZ", -np.pi / 4)  # Rigid rotation of the top

pb.nlsolve(dt=0.2, tmax=1, update_dt=True, print_info=1, interval_output=0.2)

###############################################################################
# Plot the YY stress component on the deformed configuration.

results.plot("Stress", component="YY", data_type="Node", show=True)

###############################################################################
# A movie of the whole loading can be written with
# ``results.write_movie("results/rigid_tie_example", "Stress", "vm", "Node")``
# (requires ``imageio`` and ``imageio-ffmpeg``).
