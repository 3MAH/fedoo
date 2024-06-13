##############################################################################
# Very simple example to get the homogenized elastic stiffness matrix using  #
# periodic boundary conditions from a periodic mesh                          #
##############################################################################

import fedoo as fd
import numpy as np

# --------------- Pre-Treatment --------------------------------------------------------
fd.ModelingSpace("3D")

mesh = fd.mesh.import_file("../../util/meshes/octet_truss.msh")["tet4"]

# Material definition
material = fd.constitutivelaw.ElasticIsotrop(1e5, 0.3)
wf = fd.weakform.StressEquilibrium(material)

# Assembly
assembly = fd.Assembly.create(wf, mesh)

L_eff = fd.homogen.get_homogenized_stiffness(assembly, meshperio=True)

np.set_printoptions(precision=3, suppress=True)
print("L_eff = ", L_eff)

# # Get the equivalent homogen isotropic properties using simcoon function L_iso_props
# from simcoon import simmit as sim
# props_test_eff = sim.L_iso_props(L_eff)
# print('props', props_test_eff) #gives E and nu
