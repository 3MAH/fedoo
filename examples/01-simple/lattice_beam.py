"""
Beam Lattice Structure
~~~~~~~~~~~~~~~~~~~~~~
"""

import fedoo as fd
import numpy as np

E = 1e5
nu = 0.3
radius=1
k = 0 # reduce section. k=0 to neglect the shear effect
n = 6 # n_nodes
ndim=3
if ndim==2:
    fd.ModelingSpace("2D")
else:
    fd.ModelingSpace("3D")

material = fd.constitutivelaw.ElasticIsotrop(E, nu)
beam_props = fd.constitutivelaw.BeamCircular(material, radius, k=k)

# Extract a Lattice Mesh from a 3D mesh
if ndim == 2:
    mesh = fd.mesh.extract_edges(fd.mesh.rectangle_mesh(n,n, x_max=100, y_max=100, ndim=ndim))
else:
    mesh = fd.mesh.extract_edges(fd.mesh.box_mesh(n,n,n, x_max=100, y_max=100, z_max=100))

nodes_left = mesh.find_nodes("X", mesh.bounding_box.xmin)
nodes_right = mesh.find_nodes("X", mesh.bounding_box.xmax)

wf = fd.weakform.BeamEquilibrium(beam_props)

# assembly = fd.Assembly.create(wf, mesh, "bernoullibeam")
assembly = fd.Assembly.create(wf, mesh)

pb = fd.problem.NonLinear(assembly, nlgeom=True)
results = pb.add_output("test", assembly, ["Disp", "Rot", "BeamStress", "BeamStrain"])

pb.bc.add("Dirichlet", nodes_left, ["Disp", "Rot"], 0)
# pb.bc.add("Dirichlet", nodes_right, "DispY", -200)
pb.bc.add("Dirichlet", nodes_right, "DispY", -50)
# pb.bc.add("Dirichlet", nodes_right, "DispX", 2000)
# pb.bc.add("Dirichlet", np.arange(mesh.n_nodes), "DispY", 0)

pb.set_nr_criterion(max_subiter=4, err0=None, tol=1e-4)
pb.nlsolve(dt=0.02, update_dt=True, print_info=1)

# Post treatment
# results.plot("BeamStress", 0)
# results.write_movie("lattice", "BeamStress", "X", rot_azimuth=0)



results = pb.get_results(assembly, ["Disp", "Rot", "BeamStress"])
results.plot("BeamStress", 0)

# results.plot("Rot", 0, node_labels=True)

