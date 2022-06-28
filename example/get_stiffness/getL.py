from fedoo import *
import numpy as np
import time
from simcoon import simmit as sim
import os

#--------------- Pre-Treatment --------------------------------------------------------
Util.ProblemDimension("3D")

Mesh.import_file('octet_surf.msh', meshname = "Domain")

meshname = "Domain2"

umat_name = 'ELISO'
props = np.array([[1e5, 0.3, 1]])
nstatev = 1

L = sim.L_iso(1e5, 0.3, 'Enu')
props_test = sim.L_iso_props(L)
print('props', props_test)

L_eff = Homogen.GetHomogenizedStiffness(meshname, L, meshperio=True)

np.set_printoptions(precision=3, suppress=True)
print('L_eff = ', L_eff)

props_test_eff = sim.L_iso_props(L_eff)
print('props', props_test_eff)
