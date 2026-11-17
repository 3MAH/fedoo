"""
Dynamic crack propagation by node release in a pre-strained strip
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A plane-stress strip (half model, symmetry on the crack plane ``Y = ymin``)
is first stretched statically. The crack then advances at a constant speed
``coeff * c_R`` by releasing one node of the crack plane per time step
(linear implicit Newmark dynamics). The dynamic energy release rate
``GId`` is computed from the elastic and kinetic energy balance and compared
with the static value ``Gqs``.

The mesh is imported from ``Broberg2D_10000_10000.msh`` (10 x 10 domain,
uniform spacing ``MeshSize`` on the first 500 crack-plane intervals, graded
beyond). About 500 linear time steps are solved.

The model is a symmetric half of the strip, so the elastic and kinetic
energies are those of one half: the printed ``Gqs`` and ``GId`` are half of
the energy release rate of the full strip. Their ratio, the quantity of
interest here, is unaffected.
"""

import fedoo as fd
import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

###############################################################################
# Input parameters
# ----------------

coeff = 0.2  # crack velocity / Rayleigh wave speed
COEFF = 1  # number of time steps per released node
MeshSize = 1e-3  # based on the work of C. Fond

Epsilon = 0.01  # Initial static strain
E = 1.5e9  # Young Modulus
nu = 0.33  # Poisson ratio
rho = 1000  # Density

# Newmark parameters: gamma = 1/2 gives no numerical damping, beta = 1/2
# is unconditionally stable (the average-acceleration rule is beta = 1/4).
gamma = 0.5
beta = 0.5

###############################################################################
# Derived parameters
# ------------------

TransverseWaveSpeed = (E / (2 * rho * (1 + nu))) ** (1 / 2)
# Fit of c_R / c_T as a function of nu (plane strain). In plane stress the
# exact ratio corresponds to nu / (1 + nu); kept as is (about 1 % faster).
RayleighWaveSpeed = (
    0.874032 + 0.200396 * nu - 0.0756704 * (nu**2)
) * TransverseWaveSpeed
CrackVelocity = RayleighWaveSpeed * coeff

dt = MeshSize / CrackVelocity / COEFF  # StepTime

###############################################################################
# Mesh, material and assembly
# ---------------------------

fd.ModelingSpace("2Dstress")

meshID = "Domain"
mesh = fd.mesh.import_file(
    os.path.join(SCRIPT_DIR, "Broberg2D_10000_10000.msh"), name=meshID
)
mesh.nodes = mesh.nodes[:, :2]

material = fd.constitutivelaw.ElasticIsotrop(E, nu, name="ElasticLaw")
material.set_density(rho)

# weak form and assembly
fd.weakform.StressEquilibrium("ElasticLaw", name="StressEquilibrium")

fd.Assembly.create("StressEquilibrium", mesh, "tri3", name="StiffAssembling")

length = mesh.bounding_box.size_x
height = mesh.bounding_box.size_y

ImposedDisplacement = Epsilon * height


Xmin = mesh.find_nodes("X", mesh.bounding_box.xmin)
Xmax = mesh.find_nodes("X", mesh.bounding_box.xmax)
Ymin = mesh.find_nodes("Y", mesh.bounding_box.ymin)
Ymax = mesh.find_nodes("Y", mesh.bounding_box.ymax)
Ymin = Ymin[
    np.argsort(mesh.nodes[Ymin][:, 0])
]  # to get a list of nodes sorted from xmin to xmax

###############################################################################
# Static pre-strain
# -----------------

pb = fd.problem.Linear("StiffAssembling")

pb.bc.add("Dirichlet", Xmax, "DispX", 0)
pb.bc.add("Dirichlet", Xmin, "DispX", 0)
pb.bc.add("Dirichlet", Ymax, "DispY", ImposedDisplacement)

deb = pb.bc.add("Dirichlet", Ymin, "DispY", 0, name="deb")

pb.apply_boundary_conditions()
pb.solve()

U = pb.get_dof_solution()
# Static energy release rate (half model: half of the full-strip value)
Gqs = pb.get_elastic_energy() / (length)

###############################################################################
# Dynamic crack propagation by node release
# -----------------------------------------

Ecin, Eela = [], []

pb = fd.problem.LinearNewmark(
    assembly="StiffAssembling",
    time_step=dt,
    integrator=fd.time.Newmark(beta=beta, gamma=gamma),
)
pb.set_initial_displacement("all", U)

pb.set_initial_velocity("all", 0)
pb.set_initial_acceleration("all", 0)
pb.initialize()

os.makedirs(os.path.join(SCRIPT_DIR, "results"), exist_ok=True)
res = pb.add_output(
    os.path.join(SCRIPT_DIR, "results", "Broberg2D_Dynamic"),
    "StiffAssembling",
    ["Disp", "Stress"],
)

pb.bc.add("Dirichlet", Xmax, "DispX", 0)
pb.bc.add("Dirichlet", Xmin, "DispX", 0)
pb.bc.add("Dirichlet", Ymax, "DispY", ImposedDisplacement)

deb = pb.bc.add("Dirichlet", Ymin, "DispY", 0, name="deb")

# The crack advances at constant speed only where the crack-plane spacing is
# MeshSize (first 500 intervals); beyond, the mesh is graded and GId (divided
# by MeshSize) would be meaningless.
n_release = int(np.sum(np.isclose(np.diff(mesh.nodes[Ymin, 0]), MeshSize, rtol=1e-3)))
n_steps = n_release * COEFF
for i in range(n_steps):
    print("\r v = {}cr - {:.1f}%".format(coeff, (i + 1) / n_steps * 100), end="")

    if i % COEFF == 0:
        deb.node_set = deb.node_set[1:]

    pb.apply_boundary_conditions()
    pb.solve()
    # update_weakform=True refreshes the stress field (otherwise the saved
    # Stress stays equal to the static pre-strain); compute="none" keeps the
    # cached linear operators.
    pb.update(compute="none", update_weakform=True)
    pb.save_results(i)

    Ecin.append(pb.get_kinetic_energy())
    Eela.append(pb.get_elastic_energy())

Ecin = np.array(Ecin)
Eela = np.array(Eela)
# Dynamic energy release rate from the energy balance (half model, like Gqs)
GId = (-(Eela[1:] - Eela[:-1]) - (Ecin[1:] - Ecin[:-1])) / (MeshSize)

print("\nGqs (static) =", Gqs)
print(
    "mean GId / Gqs over the last half of the propagation =",
    np.mean(GId[len(GId) // 2 :]) / Gqs,
)

###############################################################################
# Normalized dynamic energy release rate
# --------------------------------------

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(np.arange(1, len(GId) + 1), GId / Gqs, lw=1)
ax.set_xlabel("Released node")
ax.set_ylabel(r"$G_{Id} / G_{qs}$")
ax.set_title(f"Crack speed {coeff} $c_R$")
ax.grid(True, alpha=0.3)
fig.tight_layout()
plt.show()
