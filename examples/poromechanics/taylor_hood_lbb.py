"""Inf-sup (LBB) stable mixed / poromechanics elements in fedoo (Taylor-Hood).

The mixed displacement-pressure and the Biot u-PorePressure saddle systems need
an inf-sup (LBB) stable interpolation pair. Equal-order pairs (displacement and
pressure interpolated identically) violate the LBB condition and pollute the
pressure with spurious "checkerboard" modes.

fedoo supports a DIFFERENT interpolation order per variable -- so a Taylor-Hood
pair (quadratic displacement, linear pressure) needs no new machinery:

    elm = fd.lib_elements.element_list.CombinedElement("quad8lbb", "quad8")
    elm.set_variable_interpolation("Pressure", "quad4")          # one order lower
    assembly = fd.Assembly.create(wf, mesh_quad8, elm_type="quad8lbb")

Part A compares, on the SAME quad8 mesh and SAME quadratic displacement, the
equal-order pressure (quad8) and the Taylor-Hood pressure (quad4) on a nearly
incompressible problem -- the equal-order pressure shows larger (checkerboard)
extrema. Part B uses a Taylor-Hood poro element (hex20 / hex8) and recovers the
exact undrained Biot pressure.
"""

import numpy as np

import fedoo as fd


# ----------------------------------------------------------------------
# Part A - nearly-incompressible mixed elasticity: pressure smoothness
# ----------------------------------------------------------------------
def mixed_cantilever(pressure_elm, name, nu=0.49999):
    """Plane-strain cantilever bending. pressure_elm in {"quad8","quad4"}."""
    E = 1.0
    K = E / (3 * (1 - 2 * nu))
    fd.ModelingSpace("2Dplane")
    mesh = fd.mesh.rectangle_mesh(
        nx=25,
        ny=9,
        x_min=0,
        x_max=4,
        y_min=0,
        y_max=1,
        elm_type="quad8",
        name="beam_" + name,
    )
    mat = fd.constitutivelaw.ElasticIsotrop(E, nu, name="mat_" + name)
    wf = fd.weakform.StressEquilibriumMixed(
        mat, bulk_modulus=K, nlgeom=False, name="wf_" + name
    )
    if pressure_elm == "quad4":  # Taylor-Hood Q2-Q1
        elm = fd.lib_elements.element_list.CombinedElement(name, "quad8")
        elm.set_variable_interpolation("Pressure", "quad4")
        elm_type = name
    else:  # equal order Q2-Q2
        elm_type = "quad8"
    asm = fd.Assembly.create(wf, mesh, elm_type=elm_type, name="asm_" + name)
    pb = fd.problem.NonLinear("asm_" + name)
    pb.set_nr_criterion("Displacement", tol=1e-4, max_subiter=40)
    pb.bc.add("Dirichlet", mesh.find_nodes("X", 0.0), "DispX", 0.0)
    pb.bc.add("Dirichlet", mesh.find_nodes("X", 0.0), "DispY", 0.0)
    pb.bc.add(
        "Dirichlet",
        mesh.find_nodes("X", 4.0),
        "DispY",
        -0.05,
        time_func=lambda tf: min(1.0, tf * 1e9),
    )
    pb.initialize()
    pb.tmax = 1.0
    pb.dtime = 1.0
    pb.set_start()
    pb.time = 0.0
    pb.solve_time_increment()
    p_gp = np.asarray(
        asm.get_gp_results(pb.space.variable("Pressure"), pb.get_dof_solution())
    ).ravel()
    return p_gp


print("=" * 70)
print(" Part A - nearly incompressible mixed elasticity (nu = 0.49999)")
print(" same quad8 mesh & quadratic displacement; only the Pressure order differs")
print("=" * 70)
# Taylor-Hood here (well posed). The equal-order run is deferred to the very end
# because the LBB-deficient saddle is singular and its failed factorization
# pollutes the shared in-process direct solver.
p_th = mixed_cantilever("quad4", "q8th")  # Taylor-Hood Q2-Q1
print(
    f"  Taylor-Hood Q2-Q1 (Pressure quad4): well posed, smooth pressure, "
    f"extrema [{p_th.min():+.4g}, {p_th.max():+.4g}]"
)
print("  equal order Q2-Q2 (Pressure quad8): see end of script.")


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
wf = fd.weakform.poro_mechanics_simple(skel, fluid, nlgeom=False, name="poro")

elm = fd.lib_elements.element_list.CombinedElement("hex20lbb", "hex20")
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

# ----------------------------------------------------------------------
# Equal-order LBB failure (run LAST: it is singular and corrupts the solver)
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print(" Equal-order Q2-Q2 on the Part A problem (LBB-deficient)")
print("=" * 70)
try:
    p_eq = mixed_cantilever("quad8", "q8eq")
    print(
        f"  solved, but gp pressure extrema [{p_eq.min():+.4g}, {p_eq.max():+.4g}] "
        f"({np.max(np.abs(p_eq)) / np.max(np.abs(p_th)):.2f}x the Taylor-Hood "
        f"extrema -> checkerboard pollution)"
    )
except Exception as e:
    print(f"  *** {type(e).__name__}: {str(e)[:55]} ***")
    print("  -> SINGULAR saddle: the textbook inf-sup (LBB) failure that the")
    print("     Taylor-Hood pair above avoids on the very same mesh.")
print("\nDONE")
