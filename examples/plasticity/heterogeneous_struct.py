"""
Matrix with a stiff inclusion: three ways to assign materials
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A square plate (plane stress) contains a circular inclusion five times stiffer
than the matrix. Both materials are linear elastic and the problem is solved
with :class:`fedoo.problem.Linear`. The example shows three ways to give
different materials to different element sets:

1. one assembly per material, summed into a global assembly;
2. a single :class:`fedoo.constitutivelaw.ElasticIsotrop` law whose Young
   modulus is an array with one value per element;
3. a :class:`fedoo.constitutivelaw.Heterogeneous` law that maps one law to
   each element set.

The three methods give the same solution. Change ``method`` to switch between
them. See the elastoplastic heterogeneous structure example for the nonlinear
version.
"""

import fedoo as fd
import numpy as np

###############################################################################
# Mesh of the matrix and of the inclusion
# ---------------------------------------
# The plate with a hole and the disk are meshed separately, then merged into a
# single mesh. The coincident nodes of the interface are merged so that the
# inclusion is perfectly bonded to the matrix.

mesh = fd.mesh.hole_plate_mesh(
    nr=11, nt=11, length=100, height=100, radius=20, elm_type="quad4", name="Domain"
)

mesh.element_sets["matrix"] = np.arange(0, mesh.n_elements)

mesh_disk = fd.mesh.disk_mesh(20, 11, 11)
mesh_disk.element_sets["inclusion"] = np.arange(0, mesh_disk.n_elements)

mesh = mesh + mesh_disk

# glue the inclusion to the matrix
mesh.merge_nodes(np.c_[mesh.node_sets["hole_edge"], mesh.node_sets["boundary"]])

###############################################################################
# Materials and assembly
# ----------------------
# Matrix: :math:`E = 2 \times 10^4`, inclusion: :math:`E = 10^5`, both with
# :math:`\nu = 0.3`.

# 2D problem with the plane stress assumption.
fd.ModelingSpace("2Dstress")

method = 1  # 1: sum of assemblies, 2: array properties, 3: Heterogeneous law
if method == 1:
    # method 1: one assembly per material, then sum of the assemblies
    material1 = fd.constitutivelaw.ElasticIsotrop(2e4, 0.3)
    material2 = fd.constitutivelaw.ElasticIsotrop(1e5, 0.3)

    # weak formulation of the mechanical equilibrium equation
    wf1 = fd.weakform.StressEquilibrium(material1)
    wf2 = fd.weakform.StressEquilibrium(material2)

    # global assembly
    assemb1 = fd.Assembly.create(wf1, mesh.extract_elements("matrix"))
    assemb2 = fd.Assembly.create(wf2, mesh.extract_elements("inclusion"))

    assembly = assemb1 + assemb2

elif method == 2:
    # method 2: constitutive law with one property value per element.
    # Simpler, but only works if both materials use the same kind of law.
    E = np.empty(mesh.n_elements)
    E[mesh.element_sets["matrix"]] = 2e4
    E[mesh.element_sets["inclusion"]] = 1e5

    material = fd.constitutivelaw.ElasticIsotrop(E, 0.3)
    wf = fd.weakform.StressEquilibrium(material)
    assembly = fd.Assembly.create(wf, mesh)

elif method == 3:
    # method 3: heterogeneous constitutive law
    material1 = fd.constitutivelaw.ElasticIsotrop(2e4, 0.3)
    material2 = fd.constitutivelaw.ElasticIsotrop(1e5, 0.3)

    material = fd.constitutivelaw.Heterogeneous(
        (material1, material2), ("matrix", "inclusion")
    )

    wf = fd.weakform.StressEquilibrium(material)
    assembly = fd.Assembly.create(wf, mesh)

###############################################################################
# Linear problem and boundary conditions
# --------------------------------------
# The left edge is clamped and a displacement :math:`u_x = 20` is prescribed
# on the right edge (with :math:`u_y = 0`).

pb = fd.problem.Linear(assembly)

left = mesh.find_nodes("X", mesh.bounding_box.xmin)
right = mesh.find_nodes("X", mesh.bounding_box.xmax)

pb.bc.add("Dirichlet", left, "Disp", 0)
pb.bc.add("Dirichlet", right, "Disp", [20, 0])

pb.apply_boundary_conditions()

pb.solve()

###############################################################################
# Von Mises stress
# ----------------
# With method 1, each assembly has its own results, plotted in the same
# window.

field_plot = "Stress"
component = "vm"
if method == 1:
    # stress, strain and displacement (nodal values) of each assembly
    res1 = pb.get_results(assemb1, ["Stress", "Strain", "Disp"])
    res2 = pb.get_results(assemb2, ["Stress", "Strain", "Disp"])

    pl = res2.plot(field_plot, component=component, show=False)
    res1.plot(field_plot, component=component, plotter=pl)

if method == 2 or method == 3:
    res = pb.get_results(assembly, ["Stress", "Strain", "Disp"])
    res.plot(field_plot, component=component)
