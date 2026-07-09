"""
Finite-strain Neo-Hookean cantilever (rigid cap)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A slender, nearly-incompressible cylinder is clamped at its base and bent by a
transverse load applied to a rigid cap tied to its top face. It is loaded well
into the large-deflection, finite-strain regime (tip deflection of order the
cylinder length, ~40 % local strain at the clamped base).

This exercises the updated-lagrangian hyperelastic path of
:class:`fedoo.weakform.StressEquilibrium` with a simcoon ``NEOHC`` (compressible
Neo-Hookean) law. The strain energy is

.. math::
    W = \\tfrac{\\mu}{2}(\\bar I_1 - 3) + \\kappa (J \\ln J - J + 1),

with :math:`\\mu = E/2(1+\\nu)` the shear modulus and
:math:`\\kappa = E/3(1-2\\nu)` the bulk modulus.

The rigid cap is a :class:`fedoo.constraint.RigidTie`: it ties the whole top face
to six global rigid-body DOFs (``RigidDispX/Y/Z``, ``RigidRotX/Y/Z``). Here the
cap is driven by prescribing its transverse displacement ``RigidDispX``.

The mesh is read from an Abaqus ``.inp`` deck. Two are provided and give the same
result: a **linear** hex8 mesh solved with reduced integration
(:func:`fedoo.weakform.StressEquilibriumRI`, which avoids volumetric locking at
:math:`\\nu = 0.49`), and a **quadratic** hex20 mesh solved with full integration
(:class:`fedoo.weakform.StressEquilibrium`).

Displacement control is used because static *force* control is ill-conditioned
for such a flexible structure: the cap's transverse stiffness is tiny next to the
internal stiffness, so a force increment demands a large displacement jump.
Loading by force is done through implicit dynamics
(:class:`fedoo.weakform.implicit_dynamic.ImplicitDynamic`), where the cap's
inertia/damping regularise that soft mode.
"""

import os

import numpy as np

import fedoo as fd

# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------
L, R = 0.05, 0.0025  # cylinder length / radius [m]
E, nu = 60e6, 0.49  # Young's modulus [Pa], Poisson's ratio
mu = E / (2 * (1 + nu))
kappa = E / (3 * (1 - 2 * nu))

U_CAP = 0.03  # prescribed transverse cap displacement [m] (~0.6 L)

# "linear"    -> hex8,  reduced integration (StressEquilibriumRI); faster
# "quadratic" -> hex20, full integration    (StressEquilibrium)
MESH = "linear"

MESH_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../util/meshes",
    "cyl08_hexa_lin.inp" if MESH == "linear" else "cyl08_hexa_quad.inp",
)

fd.ModelingSpace("3D")


def read_cylinder_inp(path):
    """Read a C3D8 / C3D20 Abaqus .inp into a fedoo hex8 / hex20 Mesh.

    ``fd.Mesh.read`` handles this directly where the meshio-based reader is
    available; otherwise fall back to meshio (pulled in via ``pyvista[io]``).
    """
    try:
        return fd.Mesh.read(path)
    except (NotImplementedError, NameError, ModuleNotFoundError):
        import meshio  # noqa: PLC0415

        m = meshio.read(path)
        if "hexahedron20" in m.cells_dict:
            return fd.Mesh(m.points, m.cells_dict["hexahedron20"], "hex20")
        return fd.Mesh(m.points, m.cells_dict["hexahedron"], "hex8")


mesh = read_cylinder_inp(MESH_FILE)
z = mesh.nodes[:, 2]
bottom = mesh.find_nodes("Z", z.min())  # clamped base
top = mesh.find_nodes("Z", z.max())  # tied to the rigid cap

# --------------------------------------------------------------------------
# Material, weak form, assembly
# --------------------------------------------------------------------------
material = fd.constitutivelaw.Simcoon("NEOHC", [mu, kappa], name="neohookean")

if mesh.elm_type == "hex8":
    # reduced integration + hourglass control avoids volumetric locking
    wf = fd.weakform.StressEquilibriumRI(material, nlgeom="UL")
    stress_wf = wf.list_weakform[0]
else:  # hex20: full integration
    wf = fd.weakform.StressEquilibrium(material, nlgeom="UL")
    stress_wf = wf
# initial-stress stiffness, needed in the tangent under large rotations
stress_wf.geometric_stiffness = True

assembly = fd.Assembly.create(wf, mesh, name="assembly")

# --------------------------------------------------------------------------
# Problem, boundary conditions, solve (displacement control)
# --------------------------------------------------------------------------
pb = fd.problem.NonLinear(assembly)
pb.set_nr_criterion("Displacement", err0=1.0, tol=1e-3, max_subiter=20)

results = pb.add_output("neohookean_cantilever", assembly, ["Disp", "Stress", "Strain"])

pb.bc.add(fd.constraint.RigidTie(top))  # rigid cap on the top face
pb.bc.add("Dirichlet", bottom, "Disp", 0)  # clamp the base
pb.bc.add("Dirichlet", "RigidDispX", U_CAP)  # drive the cap transversely

pb.nlsolve(dt=0.05, tmax=1.0, update_dt=True, print_info=1, interval_output=0.05)

# --------------------------------------------------------------------------
# Post-processing
# --------------------------------------------------------------------------
results.load(results.n_iter - 1)  # last (fully-loaded) increment
disp = results.get_data("Disp", None, "Node")
print(f"max |displacement| : {np.linalg.norm(disp, axis=0).max():.4e} m  (L = {L})")
print(f"max von Mises stress: {results.get_data('Stress', 'vm', 'Node').max():.4e} Pa")

results.plot("Stress", component="vm", data_type="Node", show=True)
