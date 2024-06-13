import fedoo as fd
import numpy as np

space = fd.ModelingSpace("2D")

space.new_variable("Vx")
space.new_variable("Vy")
space.new_vector("V", ["Vx", "Vy"])
space.new_variable("P")  # pressure

# create simple DiffOp objects
dVx_dX = space.derivative("Vx", "X")
dVy_dY = space.derivative("Vy", "Y")
dVy_dX = space.derivative("Vy", "X")
dVx_dY = space.derivative("Vx", "Y")
P = space.variable("P")

mu = 10  # viscosity
L1 = 100
L2 = 200
R = L1 / 5
Vleft = 1

D = [dVx_dX, dVy_dY, dVx_dY + dVy_dX]

# Momentum conservation equation
eq1 = (
    D[0].virtual * (-P + 2 * mu * D[0])
    + D[1].virtual * (-P + 2 * mu * D[1])
    + 0.5 * D[2].virtual * D[2]
)

# Mass conservation equation
eq2 = P.virtual * (dVx_dX + dVy_dY)

# weakform
wf = fd.WeakForm(eq1 + eq2)

# Or by summing the weakforms
# wf1 = fd.WeakForm(D[0].virtual * (-P+2*mu*D[0]) + D[1].virtual * (-P+2*mu*D[1]) + 0.5*D[2].virtual * D[2])
# wf2 = fd.WeakForm(P.virtual * (dVx_dX + dVy_dY))
# wf = wf1 + wf2

mesh1 = fd.mesh.hole_plate_mesh(
    length=L1, height=L1, radius=R, elm_type="tri3", sym=False
)
mesh2 = fd.mesh.rectangle_mesh(
    nx=41,
    ny=21,
    x_min=L1 / 2,
    x_max=L1 / 2 + L2,
    y_min=-L1 / 2,
    y_max=L1 / 2,
    elm_type="tri3",
)

mesh = fd.Mesh.stack(mesh1, mesh2)
mesh.merge_nodes(mesh.find_coincident_nodes())

hole = mesh.find_nodes(
    "Point", np.array([0, 0]), R + 1e-6
)  # contient les noeuds bulles !

internal_nodes = mesh.add_internal_nodes(1)

elm_type = "p1+/p1"
new_elm = fd.lib_elements.element_list.CombinedElement("p1+/p1", "tri3")
new_elm.set_variable_interpolation("Vx", "tri3bubble")
new_elm.set_variable_interpolation("Vy", "tri3bubble")

fd.Assembly.create(wf, mesh, elm_type=elm_type, name="assembling")

pb = fd.problem.Linear("assembling")

left = mesh.find_nodes("X", mesh.bounding_box.xmin)
right = mesh.find_nodes("X", mesh.bounding_box.xmax)
bottom = mesh.find_nodes("Y", mesh.bounding_box.ymin)
top = mesh.find_nodes("Y", mesh.bounding_box.ymax)
# hole = mesh.find_nodes('Point', np.array([0,0]), 20) #contient les noeuds bulles !

# pb.bc.add("Dirichlet", left, ["Vx", "Vy"], [1, 0])
pb.bc.add("Dirichlet", left, "Vx", Vleft)
# pb.bc.add("Dirichlet", right, "Vy", 0)
pb.bc.add("Dirichlet", bottom, "V", 0)
pb.bc.add("Dirichlet", top, "V", 0)
pb.bc.add("Dirichlet", internal_nodes, "P", 0)  # remove non used pressure central dof
pb.bc.add("Dirichlet", hole, "V", 0)  # remove non used pressure central dof

pb.apply_boundary_conditions()

pb.solve()


# plot with streamlines with pyvista.

# streamlines are not included in the plot function embeded in fedoo, then we need to
# add them by hand

res = pb.get_results("assembling", ["V", "P"])

# res.plot("P")

# Vnorm = np.linalg.norm(res.node_data['V'], axis=0)

# pl = res.plot("V", component = "norm", show = False)
pl = res.plot("P", component=0, show_edges=False, show=False)

# get velocity associated to physical nodes and add a dim to allow 3d treatment
pl.mesh["velocity"] = np.c_[
    res["V"][:, : mesh.n_physical_nodes].T, np.zeros(mesh.n_physical_nodes)
]

line_streamlines = pl.mesh.streamlines(
    "velocity",
    pointa=(-L1 / 2 + 1e-5, -L1 / 2 + 1e-5, 0),
    pointb=(-L1 / 2 + 1e-5, L1 - 1e-5, 0),
    n_points=20,
    max_time=1000.0,
    # compute_vorticity=False,
)


pl.add_mesh(line_streamlines.tube(radius=L1 / 100))
pl.show()
