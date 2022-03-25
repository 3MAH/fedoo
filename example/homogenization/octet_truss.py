from fedoo import *
import numpy as np
import time
from simcoon import simmit as sim
import os

#--------------- Pre-Treatment --------------------------------------------------------
Util.ProblemDimension("3D")

Mesh.ImportFromFile('octet_surf.msh', meshID = "Domain")

meshID = "Domain2"

umat_name = 'ELISO'
props = np.array([[1e5, 0.3, 1]])
nstatev = 1

L = sim.L_iso(1e5, 0.3, 'Enu')
props_test = sim.L_iso_props(L)
print('props', props_test)

solver_type = 0
corate_type = 2

path_data = 'data'
path_results = 'results'
path_file = 'path.txt'
outputfile = 'results_ELISO.txt'
outputdatfile = 'output.dat'

Homogen.SolverUnitCell(meshID, umat_name, props, nstatev, solver_type, corate_type, path_data, path_results, path_file, outputfile, outputdatfile, meshperio=True)
