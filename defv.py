import fedoo as fd
from tools_fea import read_props
import numpy as np


fd.ModelingSpace("3D")

props_init = read_props("simuEF/params_sma_init.txt")
mesh = fd.mesh.box_mesh(
    nx=2,
    ny=2,
    nz=2,  # 2 nodes/dir → 1 hex8 element, 8 corner nodes
    x_min=0,
    x_max=1,
    y_min=0,
    y_max=1,
    z_min=0,
    z_max=1,
    elm_type="hex8",
    name="Domain",
)
material = fd.constitutivelaw.Simcoon("SMAUT", props_init)

# material = fd.constitutivelaw.ElasticIsotrop(E, nu, name="Material")
wf = fd.weakform.StressEquilibrium(material, nlgeom=False)
assembly = fd.Assembly.create(wf, mesh, name="Assembly")
print(assembly.sv["Temp"])
temp = 300.0
if isinstance(temp, float):
    assembly.sv["Temp"] = 300.0

pb = fd.problem.NonLinear(assembly)

left = mesh.find_nodes("X", mesh.bounding_box.xmin)
right = mesh.find_nodes("X", mesh.bounding_box.xmax)
volume = mesh.bounding_box.volume

pb.bc.add("Dirichlet", "left", "DispX", -5e-1)
pb.bc.add("Dirichlet", "right", "Disp", 0)

ref_node = mesh.nearest_node(mesh.bounding_box.center)

pb.bc.add("Dirichlet", ref_node, "Disp", 0)

volume = mesh.bounding_box.volume  # = 1.0 mm³ for the unit cube

pb.nlsolve(dt=0.2, tmax=1, t0=0, update_dt=True, print_info=1, interval_output=0.02)
