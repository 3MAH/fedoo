import fedoo as fd
import numpy as np
import os

# Define the Modeling Space - Here 2D problem with plane stress assumption.
fd.ModelingSpace("2Dplane", "macro_space")
fd.ModelingSpace("3D", "micro_space")

# Generate a simple structured mesh "Domain" (plate with a hole).
mesh_macro = fd.mesh.hole_plate_mesh(
    nr=3,
    nt=3,
    length=100,
    height=100,
    radius=20,
    elm_type="quad4",
    sym=True,
    name="macro",
)

mesh_micro = fd.mesh.import_file("octet_surf.msh", name="micro")["tet4"]

# fd.util.meshPlot2d("macro")

# Define an elastic isotropic material with E = 2e5MPa et nu = 0.3 (steel)

E = 200e3
nu = 0.3
Re = 300
k = 1000
alpha = 1e-5  # ???
m = 0.25
uimp = -5

props = np.array([E, nu, alpha, Re, k, m])
Material = fd.constitutivelaw.Simcoon("EPICP", props, name="ConstitutiveLaw")

fd.weakform.StressEquilibrium("ConstitutiveLaw", name="micro_wf", space="micro_space")

micro_assembly = fd.Assembly.create("micro_wf", mesh_micro)

micro_cells = fd.constitutivelaw.FE2(micro_assembly, name="FEM")

# Create the weak formulation of the mechanical equilibrium equation
fd.weakform.StressEquilibrium("FEM", name="WeakForm")

# Create a global assembly
macro_assembly = fd.Assembly.create(
    "WeakForm", "macro", name="Assembly", MeshChange=True
)

# Define a new static problem
pb = fd.problem.NonLinear("Assembly")
# Problem.set_nr_criterion("Displacement")

# create a 'result' folder and set the desired ouputs
if not (os.path.isdir("results")):
    os.mkdir("results")
res = pb.add_output("results/FE2", "Assembly", ["Disp", "Stress", "Strain", "Wm"])

# output result for a random micro cell (here for the 5th integration point)
# Warning : the results of micro_cells are automatically saved at each NR iteration (several iteration per time iteration)
pb.initialize()
res_micro = micro_cells.list_problem[5].add_output(
    "results/micro_cell", micro_cells.list_assembly[5], ["Disp", "Stress", "Strain"]
)

# Definition of the set of nodes for boundary conditions
crd = mesh_macro.nodes
left = np.where(crd[:, 0] == np.min(crd[:, 0]))[0]
right = np.where(crd[:, 0] == np.max(crd[:, 0]))[0]
bottom = np.where(crd[:, 1] == np.min(crd[:, 1]))[0]

# Boundary conditions
# symetry condition on left (ux = 0)
pb.bc.add("Dirichlet", left, "DispX", 0)
# symetry condition on bottom edge (ux = 0)
pb.bc.add("Dirichlet", bottom, "DispY", 0)
# displacement on right (ux=0.1mm)
pb.bc.add("Dirichlet", right, "DispX", 0.1)

# Solve problem
pb.nlsolve()
