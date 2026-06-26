import fedoo as fd
import numpy as np
from fedoo.core.weakform import WeakFormBase


# Create a Non-Linear weak form that return a linearized weak form at the current state
# ------------------------
class NavierStokes(WeakFormBase):
    """
    Navier-Stokes weak formulation for incompressible Newtonian fluids.

    This implementation uses a mixed velocity/pressure formulation.
    It supports transient and non-linear convection terms.

    Parameters
    ----------
    rho: float or array
        Density of the fluid.
    mu: float or array
        Dynamic viscosity of the fluid.
    name: str, optional
        Name of the weak form.
    space: ModelingSpace, optional
        Modeling space.
    """

    def __init__(self, rho, mu, name="", space=None):
        super().__init__(name, space)
        self.rho = rho
        self.mu = mu

        # Velocity variables
        self.space.new_variable("Vx")
        self.space.new_variable("Vy")
        if self.space.ndim == 3:
            self.space.new_variable("Vz")
            self.space.new_vector("V", ["Vx", "Vy", "Vz"])
        else:
            self.space.new_vector("V", ["Vx", "Vy"])

        # Pressure variable
        self.space.new_variable("P")

        # Internal operators for updating sv
        self._op_v = [
            self.space.variable(name) for name in ["Vx", "Vy", "Vz"][: self.space.ndim]
        ]
        self._op_p = self.space.variable("P")

        self._grad_ops = []
        for i in range(self.space.ndim):
            var_name = ["Vx", "Vy", "Vz"][i]
            for j in range(self.space.ndim):
                crd_name = ["X", "Y", "Z"][j]
                self._grad_ops.append(self.space.derivative(var_name, crd_name))

    def initialize(self, assembly, pb):
        """Initialize state variables."""
        assembly.sv["V"] = np.zeros((self.space.ndim, assembly.n_gauss_points))
        assembly.sv["P"] = np.zeros(assembly.n_gauss_points)
        assembly.sv["gradV"] = np.zeros((self.space.ndim**2, assembly.n_gauss_points))
        self._v_start = np.zeros((self.space.ndim, assembly.n_gauss_points))

    def set_start(self, assembly, pb):
        """Called at the beginning of a new time step."""
        self._v_start = assembly.sv["V"].copy()

    def update(self, assembly, pb):
        """Update current values from solution."""
        sol = pb.get_dof_solution()
        if np.isscalar(sol) and sol == 0:
            return

        v_results = []
        for op in self._op_v:
            v_results.append(assembly.get_gp_results(op, sol))
        assembly.sv["V"] = np.array(v_results)
        assembly.sv["P"] = assembly.get_gp_results(self._op_p, sol)

        grads = []
        for op in self._grad_ops:
            grads.append(assembly.get_gp_results(op, sol))
        assembly.sv["gradV"] = np.array(grads)

    def get_weak_equation(self, assembly, pb):
        """Build the weak equation including tangent and residual."""
        ndim = self.space.ndim

        # Increments (operators)
        V = [self.space.variable(["Vx", "Vy", "Vz"][i]) for i in range(ndim)]
        P = self.space.variable("P")

        # grad_V_inc[i][j] = dVi / dXj
        grad_V_inc = [[None for _ in range(ndim)] for _ in range(ndim)]
        for i in range(ndim):
            for j in range(ndim):
                grad_V_inc[i][j] = self.space.derivative(
                    ["Vx", "Vy", "Vz"][i], ["X", "Y", "Z"][j]
                )

        # Current state
        V_curr = assembly.sv["V"]
        P_curr = assembly.sv["P"]
        gradV_curr = assembly.sv["gradV"].reshape(ndim, ndim, -1)

        # Time step
        dt = pb.dtime

        DiffOp = 0

        # 1. Momentum Equations (Conservation of linear momentum)
        for i in range(ndim):
            # Viscous + Pressure part
            # Residual: 2*mu*eps(V)_ij * delta_eps_ij - P*delta_v_i,i
            # Actually we use: mu*(gradVi,j + gradVj,i)*delta_gradVi,j - P*delta_gradVi,i

            # Viscous terms
            for j in range(ndim):
                # Tangent: mu * (gradVi,j + gradVj,i)
                # But in incremental form, we sum over all j
                # Term from 2*mu*eps_ij * delta_eps_ij
                # = mu * (Vi,j + Vj,i) * delta_Vi,j

                # Residual part
                res_visc = self.mu * (gradV_curr[i, j] + gradV_curr[j, i])
                # Tangent part
                tan_visc = self.mu * (grad_V_inc[i][j] + grad_V_inc[j][i])

                DiffOp += grad_V_inc[i][j].virtual * (res_visc + tan_visc)

            # Pressure part in momentum equation: -P * div(delta_v)
            # = -P * delta_Vi,i
            DiffOp += grad_V_inc[i][i].virtual * (-P_curr - P)

            # Convection part: rho * (V . grad) V
            # Residual: rho * sum_j ( Vj_curr * gradVi,j_curr )
            res_conv = self.rho * sum(V_curr[j] * gradV_curr[i, j] for j in range(ndim))
            # Tangent: rho * sum_j ( dVj * gradVi,j_curr + Vj_curr * dgradVi,j )
            tan_conv = self.rho * sum(
                V[j] * gradV_curr[i, j] + V_curr[j] * grad_V_inc[i][j]
                for j in range(ndim)
            )

            # Time derivative: rho * dV/dt
            if dt != 0:
                res_time = (self.rho / dt) * (V_curr[i] - self._v_start[i])
                tan_time = (self.rho / dt) * V[i]
            else:
                res_time = 0
                tan_time = 0

            DiffOp += V[i].virtual * (res_conv + tan_conv + res_time + tan_time)

        # 2. Continuity Equation (Conservation of mass: div(V) = 0)
        # Residual: div(V_curr)
        res_cont = sum(gradV_curr[i, i] for i in range(ndim))
        # Tangent: div(dV)
        tan_cont = sum(grad_V_inc[i][i] for i in range(ndim))

        DiffOp += P.virtual * (res_cont + tan_cont)

        # Axisymmetric case
        if self.space._dimension == "2Daxi":
            rr = assembly.sv["_R_gausspoints"]
            # Add extra term for divergence in cylindrical coordinates: Vr/r
            # Vr is Vx (radial)
            res_cont_axi = V_curr[0] / rr
            tan_cont_axi = V[0] / rr
            DiffOp += P.virtual * (res_cont_axi + tan_cont_axi)

            # Also extra viscous term: sigma_theta = -P + 2*mu*Vr/r
            res_visc_axi = 2 * self.mu * V_curr[0] / rr
            tan_visc_axi = 2 * self.mu * V[0] / rr
            # Virtual work: sigma_theta * delta_eps_theta = sigma_theta * delta_Vr / r
            DiffOp += V[0].virtual * (
                1 / rr * (res_visc_axi + tan_visc_axi - P_curr - P)
            )

            DiffOp = DiffOp * ((2 * np.pi) * rr)

        return DiffOp


# etup Modeling Space
# -------------------
space = fd.ModelingSpace("2D")

# 2. Fluid Properties
# -------------------
rho = 10.0  # density
mu = 0.1  # dynamic viscosity (low value to see some convection effects)

# Navier-Stokes Weak Formulation
# ------------------------------
# We use the new NavierStokes weak form from the library
wf = NavierStokes(rho, mu)

# Mesh Generation
# ---------------
# Flow around a cylinder (hole in a plate)
L1 = 100
R = 10
mesh1 = fd.mesh.hole_plate_mesh(
    length=L1, height=L1, radius=R, elm_type="tri3", sym=False
)
# Add a wake region
mesh2 = fd.mesh.rectangle_mesh(
    nx=40,
    ny=20,
    x_min=L1 / 2,
    x_max=L1 / 2 + 200,
    y_min=-L1 / 2,
    y_max=L1 / 2,
    elm_type="tri3",
)

mesh = fd.Mesh.stack(mesh1, mesh2)
mesh.merge_nodes(mesh.find_coincident_nodes())

# Element type for stable mixed formulation (P1+ / P1)
# ----------------------------------------------------
# P1+ elements (linear triangles with bubble functions) for velocity
# P1 elements for pressure
internal_nodes = mesh.add_internal_nodes(1)

elm_type = "p1+/p1"
new_elm = fd.lib_elements.element_list.CombinedElement(elm_type, "tri3")
new_elm.set_variable_interpolation("Vx", "tri3bubble")
new_elm.set_variable_interpolation("Vy", "tri3bubble")
new_elm.set_variable_interpolation("P", "tri3")

# Assembly and Problem
# --------------------
assembly = fd.Assembly.create(wf, mesh, elm_type=elm_type, name="assembling")
pb = fd.problem.NonLinear("assembling")
res = pb.add_output("navier_stockes", assembly, ["P", "V"])

# Boundary Conditions
# -------------------
left = mesh.find_nodes("X", mesh.bounding_box.xmin)
right = mesh.find_nodes("X", mesh.bounding_box.xmax)
bottom = mesh.find_nodes("Y", mesh.bounding_box.ymin)
top = mesh.find_nodes("Y", mesh.bounding_box.ymax)
hole = mesh.find_nodes("Point", np.array([0, 0]), R + 1e-3)

# Inlet velocity
pb.bc.add("Dirichlet", left, "Vx", 1.0)
pb.bc.add("Dirichlet", left, "Vy", 0.0)

# No-slip on top and bottom (walls)
pb.bc.add("Dirichlet", bottom, "V", 0.0)
pb.bc.add("Dirichlet", top, "V", 0.0)

# No-slip on the cylinder
pb.bc.add("Dirichlet", hole, "V", 0.0)

# Pressure reference (fix P at one internal node per element where P is not defined)
# Actually, we just need to fix one P value in the whole domain for incompressibility
# but since P is not defined on bubble nodes, we fix them to 0 as in original example.
pb.bc.add("Dirichlet", internal_nodes, "P", 0.0)

# Solve
# -----
pb.nlsolve(dt=1, tmax=100, update_dt=True)

# Post-Processing
# ---------------
# res = pb.get_results("assembling", ["V", "P"])

# Plot Pressure
pl = res.plot("P", show=False, title="Pressure")

# Add Streamlines
# We need to map nodal results to physical mesh for pyvista streamlines
# (excluding internal bubble nodes)
phys_nodes = mesh.n_nodes
v_phys = res["V"].T  # [:, :phys_nodes].T
coords_phys = mesh.nodes[:phys_nodes]

pl.mesh["velocity"] = np.c_[v_phys, np.zeros(phys_nodes)]
line_streamlines = pl.mesh.streamlines(
    "velocity",
    pointa=(mesh.bounding_box.xmin + 1, mesh.bounding_box.ymin + 1, 0),
    pointb=(mesh.bounding_box.xmin + 1, mesh.bounding_box.ymax - 1, 0),
    n_points=25,
    max_time=500.0,
)

pl.add_mesh(line_streamlines.tube(radius=0.5), color="white")
pl.show()
