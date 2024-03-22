from fedoo import fd
import numpy as np
import time
from simcoon import simmit as sim
import os

#--------------- Pre-Treatment --------------------------------------------------------
fd.ModelingSpace("3D")

mesh = fd.mesh.import_file('../../util/meshes/octet_truss.msh')['tet4']

umat_name = 'ELISO'
props = np.array([1e5, 0.3, 1]) # 

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

homogen.get_resultsUnitCell(meshname, umat_name, props, nstatev, solver_type, corate_type, path_data, path_results, path_file, outputfile, outputdatfile, meshperio=True)
