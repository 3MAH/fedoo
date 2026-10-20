"""
Elastoplastic matrix with an elastic inclusion (Heterogeneous law)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Nonlinear version of the "matrix with a stiff inclusion" example. The matrix
follows the J2 elastoplastic law with power-law hardening of Simcoon
(``EPICP``) and the circular inclusion is linear elastic. The plate (plane
stress) is stretched by 20 % in finite strain (``nlgeom=True``) with
:class:`fedoo.problem.NonLinear`.

The two laws are combined in a single
:class:`fedoo.constitutivelaw.Heterogeneous` law (``method = 3``), or put in
two assemblies that are summed (``method = 1``).
"""

import fedoo as fd
import numpy as np

###############################################################################
# Mesh of the matrix and of the inclusion
# ---------------------------------------
# The plate with a hole and the disk are meshed separately, then merged into a
# single mesh. The coincident nodes of the interface are merged so that the
# inclusion is perfectly bonded to the matrix.

# 2D problem with the plane stress assumption.
fd.ModelingSpace("2Dstress")

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
# Matrix: ``EPICP`` with :math:`E = 2 \times 10^5`, :math:`\nu = 0.3` and a
# yield stress of 200. Inclusion: linear elastic, :math:`E = 10^5`.

method = 3  # 1: sum of assemblies, 3: Heterogeneous law
if method == 1:
    # method 1: one assembly per material, then sum of the assemblies
    material2 = fd.constitutivelaw.ElasticIsotrop(1e5, 0.3)
    props = np.array([2e5, 0.3, 1e-5, 200, 1000, 0.3])  # E, nu, alpha, Re, k, m
    material1 = fd.constitutivelaw.Simcoon("EPICP", props)

    # weak formulation of the mechanical equilibrium equation
    wf1 = fd.weakform.StressEquilibrium(material1, nlgeom=True)
    wf2 = fd.weakform.StressEquilibrium(material2, nlgeom=True)

    # global assembly
    assemb1 = fd.Assembly.create(wf1, mesh.extract_elements("matrix"))
    assemb2 = fd.Assembly.create(wf2, mesh.extract_elements("inclusion"))

    assembly = assemb1 + assemb2

elif method == 3:
    # method 3: heterogeneous constitutive law
    material2 = fd.constitutivelaw.ElasticIsotrop(1e5, 0.3)
    props = np.array([2e5, 0.3, 1e-5, 200, 1000, 0.3])  # E, nu, alpha, Re, k, m
    material1 = fd.constitutivelaw.Simcoon("EPICP", props)

    material = fd.constitutivelaw.Heterogeneous(
        (material1, material2), ("matrix", "inclusion")
    )

    wf = fd.weakform.StressEquilibrium(material, nlgeom=True)
    assembly = fd.Assembly.create(wf, mesh)

###############################################################################
# Nonlinear problem and boundary conditions
# -----------------------------------------
# The left edge is clamped and a displacement :math:`u_x = 20` is prescribed
# on the right edge (with :math:`u_y = 0`).

pb = fd.problem.NonLinear(assembly)

left = mesh.find_nodes("X", mesh.bounding_box.xmin)
right = mesh.find_nodes("X", mesh.bounding_box.xmax)

pb.bc.add("Dirichlet", left, "Disp", 0)
pb.bc.add("Dirichlet", right, "Disp", [20, 0])

# results of each increment (written in the "results" folder), used to build
# the animation of the gallery page
if method == 3:
    pb.add_output("results/heterogeneous_struct_nl", assembly, ["Disp", "Stress"])

pb.nlsolve(dt=0.1, print_info=1)

###############################################################################
# Von Mises stress at the end of the loading
# ------------------------------------------

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
