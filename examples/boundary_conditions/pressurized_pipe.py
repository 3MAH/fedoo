r"""
Pressure constraint
~~~~~~~~~~~~~~~~~~~

This example illustrates the use of :class:`fedoo.constraint.Pressure` for the
simulation of a pipe (outer radius 200 mm, thickness 20 mm) under an internal
pressure of 2 MPa and an external pressure of 0.1 MPa. The problem is treated
in 2D with the plane strain assumption. For this thin pipe, the hoop stress is
close to :math:`(p_i r_i - p_e r_e)/t \approx 17` MPa; the plotted field is the
hydrostatic pressure :math:`-\mathrm{tr}(\boldsymbol{\sigma})/3`.
"""

import fedoo as fd

# Pressure in MPa
external_pressure = 0.1
internal_pressure = 2
radius = 200
thickness = 20

fd.ModelingSpace("2D")  # plane strain assumption
mesh = fd.mesh.hollow_disk_mesh(radius, thickness, nr=5, nt=41)

material = fd.constitutivelaw.ElasticIsotrop(200e3, 0.3)

wf = fd.weakform.StressEquilibrium(material)
solid_assembly = fd.Assembly.create(wf, mesh)

###############################################################################
# To find the nodes belonging to the boundary of the pipe, the
# :py:meth:`fedoo.Mesh.find_nodes` method is used with the "Distance" criterion
# from the center of the pipe whose coordinates are [0, 0].

ext_nodes = mesh.find_nodes("Distance", ([0, 0], radius))
int_nodes = mesh.find_nodes("Distance", ([0, 0], radius - thickness))

###############################################################################
# The pressure requires to build a weakform over a surface Mesh. To
# automatically build the required surface mesh from a set of nodes, we use
# the :py:meth:`fedoo.constraint.Pressure.from_nodes` constructor.
#
# Alternatively, we can use the
# :py:meth:`fedoo.constraint.Pressure.from_elements` constructor from a set of
# elements to extract the external surface.
#
# .. note::
#   The from_nodes and from_elements constructors can't be used to apply a
#   pressure over a shell structure because, as the shell mesh is a surface
#   mesh, these constructors will extract linear mesh of the boundaries.
#   To apply a pressure over a shell geometry, the
#   :py:class:`fedoo.constraint.Pressure` constructor needs to be called.
#   See the example
#   :ref:`sphx_glr_examples_basic_spherical_shell_compression.py`


ext_pressure = fd.constraint.Pressure.from_nodes(mesh, ext_nodes, external_pressure)
int_pressure = fd.constraint.Pressure.from_nodes(mesh, int_nodes, internal_pressure)

###############################################################################
# Define a problem from the solid and pressure assemblies
# The 3 assemblies are summed.
#
# The pipe is otherwise free: three point constraints remove the rigid-body
# motions (two translations and one rotation). As the loading is axisymmetric,
# the reactions on these nodes vanish and the stress field is not affected.

pb = fd.problem.Linear(solid_assembly + ext_pressure + int_pressure)
pb.bc.add("Dirichlet", mesh.nearest_node([radius, 0]), "DispY", 0)
pb.bc.add("Dirichlet", mesh.nearest_node([-radius, 0]), "DispY", 0)
pb.bc.add("Dirichlet", mesh.nearest_node([0, radius]), "DispX", 0)
pb.solve()

pb.get_results(solid_assembly, "Stress").plot("Stress", "pressure", "Node")
