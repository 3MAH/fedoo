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


#### need to build a weakform object to allow adding the initial state ?
D = [dVx_dX, dVy_dY, dVx_dY + dVy_dX]
rho = 1  # density
# Steady momentum conservation equation
eq_steady = (
    D[0].virtual * (-P + 2 * mu * D[0])
    + D[1].virtual * (-P + 2 * mu * D[1])
    + 0.5 * D[2].virtual * D[2]
)

# Mass conservation equation
eq_mass = P.virtual * (dVx_dX + dVy_dY)

# weakform
wf1 = fd.WeakForm(eq_mass + eq_steady)
#### need to get modified


# time derivative. Build a weakform to update
class TimeDerivative(fd.core.weakform.WeakFormBase):
    def __init__(self, rho, name=None, space=None):
        fd.core.weakform.WeakFormBase.__init__(self, name, space)

        # self.space.new_variable("Vx")
        # self.space.new_variable("Vy")
        # self.space.new_vector("V", ["Vx","Vy"])

        self.__dtime = 0
        self.rho = rho
        self.assembly_options["mat_lumping"] = (
            True  # use mat lumping for time derivative
        )

    def initialize(self, assembly, pb, t0=0.0):
        if not (np.isscalar(pb.get_dof_solution())):
            self.__v_start = assembly.convert_data(
                pb.get_dof_solution("V"), convert_from="Node", convert_to="GaussPoint"
            )
            self.__v = self.__v_start
        else:
            self.__v_start = self.__v = 0

    def update(self, assembly, pb, dtime):
        self.__v = assembly.convert_data(
            pb.get_dof_solution("V"), convert_from="Node", convert_to="GaussPoint"
        )

    def set_start(self, assembly, pb, dtime):  # new time increment
        self.__dtime = dtime
        self.__v_start = self.__v

    def get_weak_equation(self, mesh=None):
        op_vx = self.space.variable("Vx")
        op_vy = self.space.variable("Vy")

        # steady state should not include the following term
        if self.__dtime != 0:
            return 1 / self.__dtime * self.rho * (
                op_vx.virtual * op_vx
                + op_vx.virtual * (self.__v[0] - self.__v_start[0])
            ) + 1 / self.__dtime * self.rho * (
                op_vy.virtual * op_vy
                + op_vy.virtual * (self.__v[1] - self.__v_start[1])
            )
        else:
            return 0


# now add convection term. Build a non linear weak form
class Convection(fd.core.weakform.WeakFormBase):
    def __init__(self, rho, name=None, space=None):
        fd.core.weakform.WeakFormBase.__init__(self, name, space)

        # self.space.new_variable("Vx")
        # self.space.new_variable("Vy")
        # self.space.new_vector("V", ["Vx","Vy"])

        self.rho = rho

    def initialize(self, assembly, pb, t0=0.0):
        if not (np.isscalar(pb.get_dof_solution())):
            self.__v_start = assembly.convert_data(
                pb.get_dof_solution("V"), convert_from="Node", convert_to="GaussPoint"
            )
            self.__v = self.__v_start
        else:
            self.__v_start = self.__v = 0

    def update(self, assembly, pb, dtime):
        self.__v = assembly.convert_data(
            pb.get_dof_solution("V"), convert_from="Node", convert_to="GaussPoint"
        )

    def set_start(self, assembly, pb, dtime):  # new time increment
        self.__dtime = dtime
        self.__v_start = self.__v

    def get_weak_equation(self, mesh=None):
        Vx = self.space.variable("Vx")
        Vy = self.space.variable("Vy")
        dVx_dX = space.derivative("Vx", "X")
        dVy_dY = space.derivative("Vy", "Y")
        dVy_dX = space.derivative("Vy", "X")
        dVx_dY = space.derivative("Vx", "Y")

        Vx_old = self.__v[0]
        Vy_old = self.__v[1]

        Uold_grad_U = [
            Vx_old * dVx_dX + Vy_old * dVx_dY,
            Vx_old * dVy_dX + Vy_old * dVy_dY,
        ]  # to multiply by [Vx.virtual, Vy.virtual]
        # U_grad_Uold = [Vx * dVx_dX_old + Vy * dVx_dY_old, Vx * dVy_dX_old + Vy * dVy_dY_old]
        # U_grad_U_0 = [Vx_old * dVx_dX_old + Vy_old_old * dVx_dY_old, Vx_old * dVy_dX_old + Vy_old * dVy_dY_old]

        return self.rho * (Uold_grad_U)


wf2 = fd.core.weakform.WeakFormSum(
    [TimeDerivative(rho, "dVdt"), Convection(rho, "convection")]
)
wf = wf1 + wf2

# mesh = fd.mesh.rectangle_mesh(elm_type = "tri3")
mesh1 = fd.mesh.hole_plate_mesh(radius=20, elm_type="tri3", sym=False)
mesh2 = fd.mesh.rectangle_mesh(
    nx=41, ny=21, x_min=50, x_max=250, y_min=-50, y_max=50, elm_type="tri3"
)

mesh = fd.Mesh.stack(mesh1, mesh2)
mesh.find_coincident_nodes()
mesh.merge_nodes(mesh.find_coincident_nodes())


hole = mesh.find_nodes("Point", np.array([0, 0]), 21)  # contient les noeuds bulles !

internal_nodes = mesh.add_internal_nodes(1)

elm_type = "p1+/p1"
new_elm = fd.lib_elements.element_list.CombinedElement("p1+/p1", "tri3")
new_elm.set_variable_interpolation("Vx", "tri3bubble")
new_elm.set_variable_interpolation("Vy", "tri3bubble")

fd.Assembly.create(wf, mesh, elm_type=elm_type, name="assembling")

pb = fd.problem.NonLinear("assembling")

left = mesh.find_nodes("X", mesh.bounding_box.xmin)
right = mesh.find_nodes("X", mesh.bounding_box.xmax)
bottom = mesh.find_nodes("Y", mesh.bounding_box.ymin)
top = mesh.find_nodes("Y", mesh.bounding_box.ymax)
# hole = mesh.find_nodes('Point', np.array([0,0]), 20) #contient les noeuds bulles !

# pb.bc.add("Dirichlet", left, ["Vx", "Vy"], [1, 0])
pb.bc.add("Dirichlet", left, "Vx", 1)
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
# Vnorm = np.linalg.norm(res.node_data['V'], axis=0)

# pl = res.plot("V", component = "norm", show = False)
pl = res.plot("P", component=0, show=False)

pl.mesh["vectors"] = np.c_[res["V"].T, np.zeros(mesh.n_nodes)]

line_streamlines = pl.mesh.streamlines(
    "vectors",
    pointa=(-50, -50, 0),
    pointb=(-50, 50, 0),
    n_points=20,
    max_time=10000.0,
    # compute_vorticity=False,
)


pl.add_mesh(line_streamlines.tube(radius=1))
pl.show()


# res.plot("V")
