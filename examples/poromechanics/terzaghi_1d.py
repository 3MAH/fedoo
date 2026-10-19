"""
Terzaghi 1D consolidation (small strain)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A column of saturated porous material with a linear elastic skeleton is
loaded by a constant pressure on its top face. The top face is drained
(``PorePressure = 0``), the bottom and lateral faces are sealed, and the
lateral displacements are blocked (oedometric, i.e. 1D, consolidation).

When the load is applied the fluid cannot escape instantly: it carries the
load and the pore pressure jumps to its undrained value. The fluid then
drains through the top face and the load is progressively transferred to
the skeleton. The pore pressure at the sealed bottom face is compared with
the classical Terzaghi series solution.

The model uses Biot poroelasticity: the total Cauchy stress is
:math:`\\sigma = \\sigma_{\\mathrm{eff}}(\\varepsilon) - \\alpha p \\mathbf{I}`
and the fluid mass balance is

.. math::

   \\frac{1}{M}\\dot p + \\alpha\\, \\dot\\varepsilon_v
   - \\nabla \\cdot \\left( \\frac{k}{\\mu_f} \\nabla p \\right) = 0.

See :doc:`/poromechanics` for details.
"""

import numpy as np
import matplotlib.pyplot as plt

import fedoo as fd

###############################################################################
# Material parameters and analytical solution
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# With a single drainage path of length :math:`L`, the closed-form solution
# uses the oedometric modulus :math:`M_{oed} = K + 4G/3`, the consolidation
# coefficient :math:`c_v = (k/\mu_f) / (\alpha^2/M_{oed} + 1/M)` and the time
# factor :math:`T = c_v t / L^2`.

L = 1.0  # column height [m]
E, nu = 1.0e6, 0.3  # drained skeleton [Pa]
M = 1.0e8  # Biot modulus [Pa] (nearly incompressible fluid)
alpha = 1.0  # Biot coefficient
k = 1.0e-7  # permeability
mu_f = 1.0  # fluid viscosity
sigma0 = 1.0e3  # applied surface load [Pa]

G = E / (2.0 * (1.0 + nu))
K = E / (3.0 * (1.0 - 2.0 * nu))
M_oed = K + 4.0 * G / 3.0
c_v = (k / mu_f) / (alpha**2 / M_oed + 1.0 / M)
p0 = alpha * M * sigma0 / (M_oed + alpha**2 * M)  # undrained pore pressure
settlement_drained = -sigma0 * L / M_oed


def terzaghi_bottom_pressure(T, nterms=300):
    """Normalized pore pressure p(z=0, t) / p0 at the sealed face."""
    m = np.arange(nterms)
    Mm = (np.pi / 2.0) * (2 * m + 1)
    return np.sum((2.0 / Mm) * ((-1.0) ** m) * np.exp(-(Mm**2) * T))


###############################################################################
# Mesh, constitutive laws and weak form
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The skeleton law is a standard mechanical law. The fluid properties are
# grouped in :py:class:`~fedoo.constitutivelaw.PoroFluidProperties`. The
# :py:class:`~fedoo.weakform.PoroMechanics` weak form couples the momentum
# balance (unknowns ``Disp`` and the skeleton ``Pressure`` multiplier) with
# the fluid mass balance (unknown ``PorePressure``).
# :py:class:`~fedoo.weakform.PoroMechanicsSimple` is the two-field
# alternative, recommended when the model has a free-traction face.

fd.ModelingSpace("3D")

nz = 20
mesh = fd.mesh.box_mesh(
    nx=2,
    ny=2,
    nz=nz + 1,
    x_min=0,
    x_max=0.1,
    y_min=0,
    y_max=0.1,
    z_min=0,
    z_max=L,
    elm_type="hex8",
)

skeleton = fd.constitutivelaw.ElasticIsotrop(E, nu)
fluid = fd.constitutivelaw.PoroFluidProperties(
    permeability=k,
    fluid_viscosity=mu_f,
    biot_coefficient=alpha,
    biot_modulus=M,
    initial_porosity=0.5,
)
wf = fd.weakform.PoroMechanics(skeleton, fluid, bulk_modulus=K, nlgeom=False)
poro_assembly = fd.Assembly.create(wf, mesh)

###############################################################################
# Loading and boundary conditions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The surface load is applied as a step: setting ``initial_pressure`` equal
# to the final pressure removes the default linear ramp over the time step.

top = mesh.find_nodes("Z", L)
bottom = mesh.find_nodes("Z", 0.0)
side_x = np.unique(
    np.concatenate([mesh.find_nodes("X", 0.0), mesh.find_nodes("X", 0.1)])
)
side_y = np.unique(
    np.concatenate([mesh.find_nodes("Y", 0.0), mesh.find_nodes("Y", 0.1)])
)

load = fd.constraint.Pressure.from_nodes(mesh, top, sigma0, initial_pressure=sigma0)

pb = fd.problem.NonLinear(poro_assembly + load)
pb.set_nr_criterion("Displacement", tol=1e-4, max_subiter=30)

pb.bc.add("Dirichlet", side_x, "DispX", 0.0)
pb.bc.add("Dirichlet", side_y, "DispY", 0.0)
pb.bc.add("Dirichlet", bottom, "DispZ", 0.0)
pb.bc.add("Dirichlet", top, "PorePressure", 0.0)  # drained top face

###############################################################################
# Time integration
# ~~~~~~~~~~~~~~~~
# The time increments are driven by hand to record the bottom pore pressure,
# the top settlement and a few pressure profiles along the column.

dt = 0.2
nb_steps = 60
snapshot_steps = [0, 4, 14, 29, 59]

pb.initialize()
pb.tmax = dt * nb_steps
pb.dtime = dt
pb.set_start()

column = np.where((mesh.nodes[:, 0] == 0.0) & (mesh.nodes[:, 1] == 0.0))[0]
column = column[np.argsort(mesh.nodes[column, 2])]

p_bottom = np.zeros(nb_steps)
uz_top = np.zeros(nb_steps)
profiles = {}
for step in range(nb_steps):
    pb.time = step * dt
    convergence, nb_nr, res = pb.solve_time_increment()
    if not convergence:
        raise RuntimeError(f"Step {step}: Newton-Raphson failed (res={res:g})")
    pb.set_start()

    p = pb.get_dof_solution("PorePressure")
    p_bottom[step] = np.mean(p[bottom])
    uz_top[step] = np.mean(pb.get_dof_solution("DispZ")[top])
    if step in snapshot_steps:
        profiles[step] = p[column].copy()

###############################################################################
# Comparison with the Terzaghi solution
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

t = (np.arange(nb_steps) + 1) * dt
T = c_v * t / L**2
T_fine = np.linspace(1e-3, T[-1], 300)
series = [terzaghi_bottom_pressure(Ti) for Ti in T_fine]

print(f"Undrained pore pressure: fedoo {p_bottom[0]:.4g} Pa, analytical {p0:.4g} Pa")
print(
    f"Final settlement: fedoo {uz_top[-1]:.4g} m, "
    f"drained analytical {settlement_drained:.4g} m"
)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

ax = axes[0]
ax.plot(T_fine, series, "k-", label="Terzaghi series")
ax.plot(T, p_bottom / p0, "o", ms=4, label="fedoo")
ax.set_xlabel(r"time factor $T = c_v t / L^2$")
ax.set_ylabel(r"$p(z=0) / p_0$")
ax.set_title("Pore pressure at the sealed bottom")
ax.legend()
ax.grid(True)

ax = axes[1]
z = mesh.nodes[column, 2]
for step, p_z in profiles.items():
    ax.plot(p_z / p0, z, label=f"T = {T[step]:.2f}")
ax.set_xlabel(r"$p / p_0$")
ax.set_ylabel("z [m]")
ax.set_title("Pore pressure profiles")
ax.legend()
ax.grid(True)

fig.tight_layout()
plt.show()
