"""
Thick cylinder inflation: F_theta-theta verification (2Daxi)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pinning benchmark for the canonical hoop deformation gradient
:math:`F_{\\theta\\theta} = r_{\\text{current}} / R_{\\text{reference}}`
in a finite-strain updated-lagrangian axisymmetric problem.

A thick-walled annulus is inflated by prescribing a radial Dirichlet
displacement on the outer boundary (the inner boundary is clamped
radially). The simulation runs in ``2Daxi`` with ``nlgeom='UL'``, then
the example reads the deformation gradient stored on the assembly's
state vector and asserts that the hoop component matches the textbook
relation :math:`F_{\\theta\\theta} = r/R` (Bonet & Wood, Box 8.3;
Holzapfel Sec. 2.5).

The assertion is what protects against regressions in
``_comp_grad_disp``: an earlier fedoo implementation divided ``u_r``
by ``r_current`` instead of the reference radius ``R``, giving the
small-strain limit ``1 + u_r/r`` rather than the correct ``r/R``. At
the inflation level used here (10-30% hoop stretch) the two formulas
disagree by several percent, so this example fails loudly if the bug
returns.

This example also demonstrates how to extract the per-Gauss-point
deformation gradient from ``assembly.sv``.
"""

import fedoo as fd
import numpy as np

# ---------------------------------------------------------------------------
# 1. Mesh: a thick annulus in the (r, z) plane.
# ---------------------------------------------------------------------------
# r in [A, B], z in [0, H]. The symmetry axis is Y in 2D (i.e. r = X column).
A, B = 1.0, 2.0
H = 0.5

fd.ModelingSpace("2Daxi")
mesh = fd.mesh.rectangle_mesh(nx=21, ny=11, x_min=A, x_max=B, y_min=0.0, y_max=H)

# ---------------------------------------------------------------------------
# 2. Linear elastic isotropic constitutive law.
# ---------------------------------------------------------------------------
# Linear elasticity is sufficient: the F_theta-theta = r/R relation is a pure
# kinematic identity, independent of the constitutive law. We only need the
# solver to converge to the prescribed boundary motion.
E = 1.0e3
nu = 0.3
material = fd.constitutivelaw.ElasticIsotrop(E, nu)

wf = fd.weakform.StressEquilibrium(material)
assembly = fd.Assembly.create(wf, mesh)

# ---------------------------------------------------------------------------
# 3. Updated-Lagrangian non-linear problem.
# ---------------------------------------------------------------------------
pb = fd.problem.NonLinear(assembly, nlgeom="UL")
pb.set_nr_criterion("Displacement", tol=1e-6, max_subiter=15)

# Inner radius: clamped radially. Top and bottom: clamped axially to keep
# the problem axisymmetric and 2D (no plane-strain ambiguity).
left = mesh.node_sets["left"]  # r = A
right = mesh.node_sets["right"]  # r = B
bottom = mesh.node_sets["bottom"]
top = mesh.node_sets["top"]

# Inner radius radially fixed -> drives a non-uniform F_theta-theta(R).
pb.bc.add("Dirichlet", left, "DispX", 0.0)
# Symmetry / axial confinement to keep things 2D.
pb.bc.add("Dirichlet", bottom, "DispY", 0.0)
pb.bc.add("Dirichlet", top, "DispY", 0.0)
# Outer radius pulled outward by delta -> a rich F_theta-theta field.
delta = 0.6  # 30% radial expansion at r = B
pb.bc.add("Dirichlet", right, "DispX", delta)

pb.nlsolve(dt=0.1, tmax=1.0, update_dt=True, print_info=0)

# ---------------------------------------------------------------------------
# 4. Verify F_theta-theta = r_current / R_reference at every gauss point.
# ---------------------------------------------------------------------------
# F is stored as (3, 3, n_gauss_points), Fortran-ordered.
F = assembly.sv["F"]
F_tt = F[2, 2, :]

# R0 is the reference radius captured at problem initialize.
R0 = assembly.sv["_R0_gausspoints"]

# Current r at gauss points: same interpolation, but on the deformed mesh.
r_current = assembly.current.mesh.convert_data(
    assembly.current.mesh.nodes[:, 0],
    "Node",
    "GaussPoint",
    n_elm_gp=assembly.n_elm_gp,
)

F_tt_expected = r_current / R0

err_max = np.max(np.abs(F_tt - F_tt_expected))
print(f"Inflation: delta = {delta}, max |F_tt - r/R| = {err_max:.3e}")
print(f"Hoop stretch range: [{F_tt.min():.4f}, {F_tt.max():.4f}]")
print(f"Reference radius range: [{R0.min():.4f}, {R0.max():.4f}]")

# Tight tolerance: F_theta-theta is computed *exactly* from u_r and R0 by
# `_comp_grad_disp`; any discrepancy beyond fp noise indicates a regression.
assert err_max < 1e-10, (
    f"F_theta-theta drifted from r/R by {err_max:.3e}; "
    "this is a regression in fedoo/weakform/stress_equilibrium._comp_grad_disp."
)

# ---------------------------------------------------------------------------
# 5. Negative control: the *buggy* formula 1 + u_r / r_current would give a
#    different (incorrect) F_theta-theta. Show how badly it would have
#    differed at the stretches reached here.
# ---------------------------------------------------------------------------
u_r_gp = r_current - R0
F_tt_buggy = 1.0 + u_r_gp / r_current  # the pre-fix formula
disagreement = np.max(np.abs(F_tt_buggy - F_tt_expected))
print(
    "Disagreement between the (correct) r/R and the (buggy) 1 + u_r/r "
    f"formulas at this stretch: {disagreement:.3e} "
    f"({100 * disagreement / F_tt_expected.max():.2f}% of peak hoop stretch)."
)
