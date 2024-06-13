"""
3D Canteleaver Beam with geometric nonlinearities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import fedoo as fd
import numpy as np
import time

# --------------- Pre-Treatment --------------------------------------------------------

fd.ModelingSpace("3D")

# Units: N, mm, MPa
mesh = fd.mesh.box_mesh(
    nx=31,
    ny=7,
    nz=7,
    x_min=0,
    x_max=1000,
    y_min=0,
    y_max=100,
    z_min=0,
    z_max=100,
    elm_type="hex8",
    name="Domain",
)

# Material definition
fd.constitutivelaw.ElasticIsotrop(200e3, 0.3, name="ElasticLaw")
wf = fd.weakform.StressEquilibrium("ElasticLaw", nlgeom=True)

# Assembly (print the time required for assembling)
assemb = fd.Assembly.create(wf, mesh, "hex8", name="Assembling")

# Type of problem
pb = fd.problem.NonLinear("Assembling")

# Boundary conditions
nodes_left = mesh.find_nodes("X", mesh.bounding_box.xmin)
nodes_load = mesh.find_nodes(
    f"X=={mesh.bounding_box.xmax} and Y=={mesh.bounding_box.ymax}"
)

pb.bc.add("Dirichlet", nodes_left, "Disp", 0)
pb.bc.add("Dirichlet", nodes_load, "DispY", -500)

# --------------- Solve --------------------------------------------------------
pb.nlsolve(dt=0.2)

# #--------------- Post-Treatment -----------------------------------------------
res = pb.get_results("Assembling", ["Stress", "Disp"])
res.plot("Stress", "XX", "Node", show_edges=False)
