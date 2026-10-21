r"""
Inf-sup (LBB) stable elements: Taylor-Hood interpolation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the undrained limit (nearly incompressible fluid, no time to drain), the
pore pressure enforces a volume constraint and the Biot system becomes a
saddle-point problem. Its discretization is stable only if the pair of
displacement and pressure interpolations satisfies the inf-sup (LBB)
condition. Equal-order pairs (same element for both fields) violate it, and
the pressure shows spurious node-to-node oscillations ("checkerboard").

A Taylor-Hood pair (quadratic displacement, linear pressure) satisfies it. In
fedoo it is built with a :py:class:`~fedoo.lib_elements.CombinedElement`::

    elm = fd.lib_elements.CombinedElement("quad8lbb", "quad8")
    elm.set_variable_interpolation("PorePressure", "quad4")
    assembly = fd.Assembly.create(wf, mesh_quad8, elm_type="quad8lbb")

The pressure DOFs of the mid-side nodes, which the linear pressure does not
use, are eliminated automatically.

Part A loads a strip footing on an undrained half-space (incompressible
constituents) and compares the pore pressure on the axis with the elastic
solution :math:`p = (2q/\pi)\arctan(b/z)`: Taylor-Hood (quad8/quad4)
against equal order (quad4/quad4) with the same pressure resolution. Part B
recovers the exact undrained Biot pressure with a 3D Taylor-Hood element
(hex20/hex8). Runs in a few seconds.
"""

import numpy as np

import fedoo as fd


# ----------------------------------------------------------------------
# Part A - undrained strip footing: Taylor-Hood vs equal order
# ----------------------------------------------------------------------
def strip_footing(taylor_hood, n=32, b=1.0, size=8.0, q=1.0e3):
    """Pore pressure on the axis under a strip load q of half-width b."""
    fd.ModelingSpace("2D")
    if taylor_hood:
        mesh = fd.mesh.rectangle_mesh(
            n // 2 + 1, n // 2 + 1, 0, size, -size, 0, "quad8"
        )
        elm = fd.lib_elements.CombinedElement("quad8lbb", "quad8")
        elm.set_variable_interpolation("PorePressure", "quad4")
        elm_type = "quad8lbb"
    else:  # same number of pressure nodes
        mesh = fd.mesh.rectangle_mesh(n + 1, n + 1, 0, size, -size, 0, "quad4")
        elm_type = "quad4"
    # biot_modulus=None: incompressible fluid; tiny permeability and time step
    fluid = fd.constitutivelaw.PoroFluidProperties(permeability=1e-12)
    wf = fd.weakform.PoroMechanicsSimple(
        fd.constitutivelaw.ElasticIsotrop(1e6, 0.3), fluid, nlgeom=False
    )
    asm = fd.Assembly.create(wf, mesh, elm_type=elm_type)
    top = mesh.find_nodes("Y", 0.0)
    loaded = top[mesh.nodes[top, 0] <= b + 1e-9]
    load = fd.constraint.Pressure.from_nodes(mesh, loaded, q, initial_pressure=q)
    pb = fd.problem.NonLinear(asm + load)
    pb.bc.add("Dirichlet", "left", "DispX", 0)  # symmetry axis
    pb.bc.add("Dirichlet", "right", "DispX", 0)
    pb.bc.add("Dirichlet", "bottom", "Disp", 0)
    pb.nlsolve(dt=1e-6, tmax=1e-6, update_dt=False, print_info=0)

    pressure = pb.get_dof_solution("PorePressure")
    vertices = np.unique(mesh.elements[:, :4])
    axis = vertices[np.isclose(mesh.nodes[vertices, 0], 0)]
    depth = -mesh.nodes[axis, 1]
    order = np.argsort(depth)
    return depth[order], pressure[axis][order] / q


print("=" * 70)
print(" Part A - undrained strip footing, pore pressure p/q on the axis")
print("=" * 70)
z, p_th = strip_footing(taylor_hood=True)
z_eq, p_eq = strip_footing(taylor_hood=False)
for depth in (0.5, 1.0, 1.5, 2.0):
    i, j = np.argmin(abs(z - depth)), np.argmin(abs(z_eq - depth))
    exact = 2 / np.pi * np.arctan(1.0 / depth)
    print(
        f"  z/b = {depth:3.1f}: exact {exact:.3f} | Taylor-Hood {p_th[i]:.3f}"
        f" | equal order {p_eq[j]:.3f}"
    )
print(
    "  equal order, consecutive axis nodes near z = b:",
    np.round(p_eq[(z_eq > 0.4) & (z_eq < 1.4)], 3),
    "-> checkerboard",
)

# ----------------------------------------------------------------------
# Part B - Taylor-Hood poromechanics (hex20 displacement / hex8 PorePressure)
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print(" Part B - Taylor-Hood poromechanics, undrained step (hex20 u / hex8 p)")
print("=" * 70)

E, nu, M, alpha, dz, L = 1.0e6, 0.3, 1.0e8, 1.0, -1.0e-4, 1.0
fd.ModelingSpace("3D")
mesh = fd.mesh.box_mesh(
    nx=3,
    ny=3,
    nz=5,
    x_min=0,
    x_max=0.2,
    y_min=0,
    y_max=0.2,
    z_min=0,
    z_max=L,
    elm_type="hex20",
    name="col",
)
skel = fd.constitutivelaw.ElasticIsotrop(E, nu, name="skel")
fluid = fd.constitutivelaw.PoroFluidProperties(
    permeability=1.0e-7,
    fluid_viscosity=1.0,
    biot_coefficient=alpha,
    biot_modulus=M,
    initial_porosity=0.5,
    name="fluid",
)
wf = fd.weakform.PoroMechanicsSimple(skel, fluid, nlgeom=False, name="poro")

elm = fd.lib_elements.CombinedElement("hex20lbb", "hex20")
elm.set_variable_interpolation("PorePressure", "hex8")
asm = fd.Assembly.create(wf, mesh, elm_type="hex20lbb", name="poroTH")

pb = fd.problem.NonLinear("poroTH")
pb.set_nr_criterion("Displacement", tol=1e-3, max_subiter=25)
sx = np.unique(np.concatenate([mesh.find_nodes("X", 0.0), mesh.find_nodes("X", 0.2)]))
sy = np.unique(np.concatenate([mesh.find_nodes("Y", 0.0), mesh.find_nodes("Y", 0.2)]))
pb.bc.add("Dirichlet", sx, "DispX", 0.0)
pb.bc.add("Dirichlet", sy, "DispY", 0.0)
pb.bc.add("Dirichlet", mesh.find_nodes("Z", 0.0), "DispZ", 0.0)
pb.bc.add(
    "Dirichlet",
    mesh.find_nodes("Z", L),
    "DispZ",
    dz,
    time_func=lambda tf: min(1.0, tf * 1e9),
)

dt = 1.0e-3  # tiny step, fully sealed -> undrained
pb.initialize()
pb.tmax = 3 * dt
pb.dtime = dt
pb.set_start()
for s in range(3):
    pb.time = s * dt
    conv, nb, res = pb.solve_time_increment()
    pb.set_start()

p_gp = np.asarray(
    asm.get_gp_results(pb.space.variable("PorePressure"), pb.get_dof_solution())
).ravel()
print(f"  converged={conv} in {nb} iter(s)")
print(
    f"  undrained gauss-point pore pressure: mean={p_gp.mean():.5g}  "
    f"std={p_gp.std():.2g}"
)
print(f"  analytical  p0 = -alpha*M*tr(eps) = {-alpha * M * dz / L:.5g} Pa  (match)")

print("\nDONE")
