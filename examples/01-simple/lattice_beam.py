"""
Beam Lattice Structure
~~~~~~~~~~~~~~~~~~~~~~
"""

import fedoo as fd
import numpy as np

E = 1e5
nu = 0.3
radius=1
k = 0.9  # reduce section. k=0 to neglect the shear effect
n = 10  # n_nodes
ndim=3 
if ndim==2:
    fd.ModelingSpace("2D")
else:
    fd.ModelingSpace("3D")

material = fd.constitutivelaw.ElasticIsotrop(E, nu)
beam_props = fd.constitutivelaw.BeamCircular(material, radius, k=k)

# Extract a Lattice Mesh from a 3D mesh
mesh = fd.mesh.extract_edges(fd.mesh.box_mesh(n,n,n))
# mesh = fd.mesh.extract_edges(fd.mesh.rectangle_mesh(n,n, ndim=ndim))

nodes_left = mesh.find_nodes("X", mesh.bounding_box.xmin)
nodes_right = mesh.find_nodes("X", mesh.bounding_box.xmax)

wf = fd.weakform.BeamEquilibrium(beam_props)
# assembly = fd.Assembly.create(wf, mesh, "bernoullibeam")
assembly = fd.Assembly.create(wf, mesh)

pb = fd.problem.NonLinear(assembly, nlgeom=True)
# results = pb.add_output("test", assembly, ["Disp", "Rot", "BeamStress", "BeamStrain"])
results = pb.add_output("test", assembly, ["Disp", "Rot"])

pb.bc.add("Dirichlet", nodes_left, ["Disp", "Rot"], 0)
pb.bc.add("Dirichlet", nodes_right, "DispY", -1)
# pb.set_nr_criterion(max_subiter=5, err0=None)
pb.nlsolve(dt=0.02, print_info=1)

# Post treatment
results.plot("Disp", 0)
results.write_movie("lattice", "Disp", "Y", rot_azimuth=0.2)
