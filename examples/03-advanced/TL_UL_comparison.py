"""
Total Lagrangian vs Updated Lagrangian comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example compares the Total Lagrangian (TL) and Updated Lagrangian (UL)
formulations on two material types:

- **EPICP**: elasto-plastic isotropic with combined power-law hardening
  (hypoelastic, tangent defined w.r.t. corotational strain rate).
- **NEOHC**: compressible Neo-Hookean hyperelastic
  (tangent defined w.r.t. the deformation gradient F).

Both formulations solve the same physical problem and should give identical
results.  The example measures execution time to highlight the efficiency
difference: UL is typically faster because TL requires an extra tangent
conversion step (Cauchy tangent -> PK2/Green-Lagrange tangent) and the
assembly of the geometric stiffness matrix.
"""

import numpy as np
from time import perf_counter
import fedoo as fd

# %%
# Common mesh
# -----------
# A 3-D cantilever beam is used for both tests. The left face is clamped and
# loading is applied on the right end.

fd.ModelingSpace("3D")

# beam dimensions (mm)
Lx, Ly, Lz = 500, 50, 50
nx, ny, nz = 21, 5, 5


def make_mesh():
    """Create a fresh beam mesh (needed because each run resets the model)."""
    return fd.mesh.box_mesh(
        nx=nx, ny=ny, nz=nz,
        x_min=0, x_max=Lx, y_min=0, y_max=Ly, z_min=0, z_max=Lz,
        elm_type="hex8", name="Domain",
    )


# %%
# Helper: run a single simulation
# --------------------------------

def run(nlgeom, material_setup, disp_bc, dt=0.05, tol=1e-4, label=""):
    """Run a nonlinear simulation and return (tip_x, tip_y, time, n_iter).

    Parameters
    ----------
    nlgeom : str
        "UL" or "TL".
    material_setup : callable
        Function that creates the constitutive law (called after ModelingSpace).
    disp_bc : dict
        Boundary condition dict, e.g. {"DispY": -100} applied to the load set.
    dt, tol : float
        Time step and Newton-Raphson tolerance.
    label : str
        Label for printing.
    """
    fd.ModelingSpace("3D")
    mesh = make_mesh()

    material_setup()
    wf = fd.weakform.StressEquilibrium("Mat", nlgeom=nlgeom)
    assemb = fd.Assembly.create(wf, mesh, "hex8", name="Assembling")
    pb = fd.problem.NonLinear("Assembling", nlgeom=nlgeom)

    nodes_left = mesh.find_nodes("X", mesh.bounding_box.xmin)
    nodes_right = mesh.find_nodes("X", mesh.bounding_box.xmax)
    nodes_load = mesh.find_nodes(
        f"X=={mesh.bounding_box.xmax} and Y=={mesh.bounding_box.ymax}"
    )

    pb.bc.add("Dirichlet", nodes_left, "Disp", 0)
    for comp, val in disp_bc.items():
        pb.bc.add("Dirichlet", nodes_load, comp, val)

    t0 = perf_counter()
    pb.nlsolve(dt=dt, tol_nr=tol, print_info=1, max_subiter=10)
    elapsed = perf_counter() - t0

    d = pb.get_disp()
    tip_x = np.mean(d[0][nodes_right])
    tip_y = np.mean(d[1][nodes_right])
    return tip_x, tip_y, elapsed


# %%
# Test 1 -- Elasto-plastic bending (EPICP)
# -----------------------------------------
# Steel-like material under bending: significant plasticity develops at the
# clamped end. The imposed displacement is moderate enough for both
# formulations to converge with the same time stepping.
#
# Material parameters:
#
# .. math::
#    \sigma_\text{eq} = \sigma_y + k\, p^m
#
# with :math:`E = 200\,000` MPa, :math:`\nu = 0.3`,
# :math:`\sigma_y = 300` MPa, :math:`k = 1000` MPa, :math:`m = 0.3`.

print("=" * 65)
print("TEST 1: Elasto-plastic bending (EPICP)")
print("  Cantilever beam 500x50x50 mm, DispY = -100 mm at tip top edge")
print("=" * 65)


def epicp_material():
    E, nu, alpha = 200e3, 0.3, 1e-5
    sigma_y, k, m = 300, 1000, 0.3
    props = np.array([E, nu, alpha, sigma_y, k, m])
    fd.constitutivelaw.Simcoon("EPICP", props, name="Mat")


x_ul, y_ul, t_ul = run("UL", epicp_material, {"DispY": -100}, dt=0.05)
x_tl, y_tl, t_tl = run("TL", epicp_material, {"DispY": -100}, dt=0.05)

print(f"\n  {'':18s} {'tip X (mm)':>12s} {'tip Y (mm)':>12s} {'time (s)':>10s}")
print(f"  {'UL':18s} {x_ul:12.4f} {y_ul:12.4f} {t_ul:10.2f}")
print(f"  {'TL':18s} {x_tl:12.4f} {y_tl:12.4f} {t_tl:10.2f}")
print(f"  {'Diff (%)':18s} "
      f"{abs(x_tl - x_ul) / max(abs(x_ul), 1e-10) * 100:11.4f}% "
      f"{abs(y_tl - y_ul) / abs(y_ul) * 100:11.4f}%")
print(f"  Time ratio TL/UL: {t_tl / t_ul:.2f}x")

# %%
# Test 2 -- Hyperelastic bending (NEOHC)
# ----------------------------------------
# Compressible Neo-Hookean rubber-like material under the same bending
# loading. The hyperelastic tangent is defined with respect to the
# deformation gradient F (Lie derivative form), which is the natural form
# for the Total Lagrangian conversion.
#
# Material parameters: :math:`\mu = 80` MPa, :math:`\kappa = 200` MPa.

print(f"\n{'=' * 65}")
print("TEST 2: Hyperelastic bending (NEOHC)")
print("  Cantilever beam 500x50x50 mm, DispY = -100 mm at tip top edge")
print("=" * 65)


def neohc_material():
    mu, kappa = 80, 200
    props = np.array([mu, kappa])
    fd.constitutivelaw.Simcoon("NEOHC", props, name="Mat")


x_ul2, y_ul2, t_ul2 = run("UL", neohc_material, {"DispY": -100}, dt=0.05)
x_tl2, y_tl2, t_tl2 = run("TL", neohc_material, {"DispY": -100}, dt=0.05)

print(f"\n  {'':18s} {'tip X (mm)':>12s} {'tip Y (mm)':>12s} {'time (s)':>10s}")
print(f"  {'UL':18s} {x_ul2:12.4f} {y_ul2:12.4f} {t_ul2:10.2f}")
print(f"  {'TL':18s} {x_tl2:12.4f} {y_tl2:12.4f} {t_tl2:10.2f}")
print(f"  {'Diff (%)':18s} "
      f"{abs(x_tl2 - x_ul2) / max(abs(x_ul2), 1e-10) * 100:11.4f}% "
      f"{abs(y_tl2 - y_ul2) / abs(y_ul2) * 100:11.4f}%")
print(f"  Time ratio TL/UL: {t_tl2 / t_ul2:.2f}x")

# %%
# Summary
# -------

print(f"\n{'=' * 65}")
print("SUMMARY")
print("=" * 65)
print("""
Both formulations give the same physical result (displacements match to
within fractions of a percent).

The Updated Lagrangian (UL) formulation is faster because it avoids:
  - the tangent modulus conversion (Cauchy -> PK2 via simcoon Lt_convert),
  - the geometric stiffness matrix assembly.

The speed advantage of UL is more pronounced for hypoelastic materials
(EPICP) than for hyperelastic (NEOHC) because the tangent conversion is
more costly when it involves stress correction terms.

Recommendation: prefer UL for production runs; use TL when the reference-
configuration formulation is specifically required (e.g. certain coupled
multi-physics problems).
""")
