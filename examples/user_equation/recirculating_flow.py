import fedoo as fd
import numpy as np

fd.Assembly.delete_memory()

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

mu = 0.1  # viscosity

D = [dVx_dX, dVy_dY, dVx_dY + dVy_dX]

# Momentum conservation equation
eq1 = (
    D[0].virtual * (-P + 2 * mu * D[0])
    + D[1].virtual * (-P + 2 * mu * D[1])
    + 0.5 * D[2].virtual * D[2]
)
# wf1 = fd.WeakForm(D[0].virtual * (-P+2*mu*D[0]) + D[1].virtual * (-P+2*mu*D[1]) + 0.5*D[2].virtual * D[2])

# Mass conservation equation
eq2 = P.virtual * (dVx_dX + dVy_dY)
# wf2 = fd.WeakForm(P.virtual * (dVx_dX + dVy_dY))

# wf = wf1 + wf2
wf = fd.WeakForm(eq1 + eq2)

mesh = fd.mesh.rectangle_mesh(elm_type="tri3")
internal_nodes = mesh.add_internal_nodes(1)

elm_type = "p1+/p1"
new_elm = fd.lib_elements.element_list.CombinedElement("p1+/p1", "tri3")
new_elm.set_variable_interpolation("Vx", "tri3bubble")
new_elm.set_variable_interpolation("Vy", "tri3bubble")

fd.Assembly.create(wf, mesh, elm_type=elm_type, name="assembling")

pb = fd.problem.Linear("assembling")

pb.bc.add("Dirichlet", "left", "V", 0)
pb.bc.add("Dirichlet", "right", "V", 0)
pb.bc.add("Dirichlet", "bottom", "V", 0)
pb.bc.add("Dirichlet", "top", ["Vx", "Vy"], [1.0, 0.0])
pb.bc.add("Dirichlet", internal_nodes, "P", 0)  # remove non used pressure central dof

pb.apply_boundary_conditions()

pb.solve()

# plot with streamlines with pyvista.

# streamlines are not included in the plot function embeded in fedoo, then we need to
# add them by hand

res = pb.get_results("assembling", ["V", "P"])
Vnorm = np.linalg.norm(res.node_data["V"], axis=0)

pl = res.plot("V", component="norm", show=False)

pl.mesh["vectors"] = np.c_[
    res["V"][:, : mesh.n_physical_nodes].T, np.zeros(mesh.n_physical_nodes)
]

line_streamlines = pl.mesh.streamlines(
    "vectors",
    pointa=(0.5, 0, 0),
    pointb=(0.5, 1, 0),
    n_points=10,
    # max_time=10.0,
    # compute_vorticity=False,
)


pl.add_mesh(line_streamlines.tube(radius=0.005))
pl.show()


# res.plot("V")
