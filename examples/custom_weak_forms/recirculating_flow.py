r"""
Lid-driven cavity (Stokes flow) with a user-defined weak form
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Steady incompressible Stokes flow in a unit square cavity whose top wall
moves at a unit velocity, written with the differential operators of a
:class:`fedoo.ModelingSpace` and a :class:`fedoo.WeakForm`. The mixed problem
uses the stable ``p1+/p1`` element (bubble-enriched velocity, linear
pressure). The weak form is the one of the Stokes example around a cylinder:

.. math::

    \int_\Omega \left( 2\mu\,\mathbf{D}(\mathbf{v}) : \mathbf{D}(\delta\mathbf{v})
    - p\,\nabla\cdot\delta\mathbf{v} \right) d\Omega = 0, \qquad
    \int_\Omega \delta p\,\nabla\cdot\mathbf{v}\, d\Omega = 0.
"""

import fedoo as fd
import numpy as np

fd.Assembly.delete_memory()

space = fd.ModelingSpace("2D")

space.new_variable("Vx")
space.new_variable("Vy")
space.new_vector("V", ["Vx", "Vy"])
space.new_variable("P")  # pressure

# create simple DiffOp objects
dVx_dX = space.derivative("Vx", "X")
dVy_dY = space.derivative("Vy", "Y")
dVy_dX = space.derivative("Vy", "X")
dVx_dY = space.derivative("Vx", "Y")
P = space.variable("P")

mu = 0.1  # dynamic viscosity

# strain rate in Voigt notation: D_xx, D_yy, gamma_xy = 2 D_xy
D = [dVx_dX, dVy_dY, dVx_dY + dVy_dX]

# Momentum conservation equation
eq1 = (
    D[0].virtual * (-P + 2 * mu * D[0])
    + D[1].virtual * (-P + 2 * mu * D[1])
    + mu * D[2].virtual * D[2]
)

# Mass conservation equation
eq2 = P.virtual * (dVx_dX + dVy_dY)

wf = fd.WeakForm(eq1 + eq2)

###############################################################################
# Mesh, mixed element and boundary conditions
# -------------------------------------------
# The pressure dof of the bubble nodes is unused and set to zero. All the
# velocities are prescribed on the boundary, so the pressure is only defined up
# to a constant (the bubble nodes do not fix it): it is set to zero at the
# bottom-left corner.
#
# The two top corners belong both to a side wall (:math:`\mathbf{v} = 0`) and
# to the lid (:math:`v_x = 1`). They are removed from the side walls, so that
# they move with the lid (the "leaky" lid-driven cavity); two Dirichlet
# conditions with different values on the same dof would emit a
# ``BoundaryConditionWarning``. This velocity discontinuity makes the pressure
# singular at the top corners.

mesh = fd.mesh.rectangle_mesh(elm_type="tri3")
internal_nodes = mesh.add_internal_nodes(1)

elm_type = "p1+/p1"
new_elm = fd.lib_elements.CombinedElement(elm_type, "tri3")
new_elm.set_variable_interpolation("Vx", "tri3bubble")
new_elm.set_variable_interpolation("Vy", "tri3bubble")

fd.Assembly.create(wf, mesh, elm_type=elm_type, name="assembling")

pb = fd.problem.Linear("assembling")

top = mesh.node_sets["top"]
pb.bc.add("Dirichlet", np.setdiff1d(mesh.node_sets["left"], top), "V", 0)
pb.bc.add("Dirichlet", np.setdiff1d(mesh.node_sets["right"], top), "V", 0)
pb.bc.add("Dirichlet", "bottom", "V", 0)
pb.bc.add("Dirichlet", "top", ["Vx", "Vy"], [1.0, 0.0])
pb.bc.add("Dirichlet", internal_nodes, "P", 0)  # unused pressure dof of the bubbles
pb.bc.add("Dirichlet", [mesh.nearest_node([0, 0])], "P", 0)  # pressure level

pb.apply_boundary_conditions()

pb.solve()

###############################################################################
# Velocity magnitude and streamlines
# ----------------------------------
# Streamlines are not part of the fedoo plot function: they are added with
# PyVista on the plotted mesh (``res.meshplot``).

res = pb.get_results("assembling", ["V", "P"])

pl = res.plot("V", component="norm", show=False)

# Add a zero out-of-plane component for PyVista's 3D streamline filter.
res.meshplot["vectors"] = np.c_[res["V"].T, np.zeros(mesh.n_nodes)]

line_streamlines = res.meshplot.streamlines(
    "vectors",
    pointa=(0.5, 0, 0),
    pointb=(0.5, 1, 0),
    n_points=10,
)


pl.add_mesh(line_streamlines.tube(radius=0.005))
pl.show()
