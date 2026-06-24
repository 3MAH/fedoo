"""
Nodal Hessian recovery on a fedoo mesh
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compute the recovered nodal gradient and Hessian of an analytic scalar
field on a 2D mesh using ``fedoo.recover_gradient`` and
``fedoo.recover_hessian``. The output is written to a vtk file so you can
inspect it in ParaView.

Pair with ``mmgpy.metrics.create_metric_from_hessian`` (and
``fedoo.to_upper_diagonal`` to get mmg's row-major upper-triangular
packing) to drive anisotropic adaptive remeshing.
"""

import numpy as np
import fedoo as fd

###############################################################################
# Build a 2D mesh and an analytic scalar field with a known closed-form Hessian.
mesh = fd.mesh.rectangle_mesh(nx=51, ny=51, elm_type="tri3")
x, y = mesh.nodes.T

# f(x, y) = sin(2*pi*x) * sin(2*pi*y)
# d2f/dx2 = -4*pi^2 * sin(2*pi*x) * sin(2*pi*y)
# d2f/dy2 = -4*pi^2 * sin(2*pi*x) * sin(2*pi*y)
# d2f/dxdy =  4*pi^2 * cos(2*pi*x) * cos(2*pi*y)
field = np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)

###############################################################################
# Recover gradient and Hessian. Both calls reuse fedoo's cached
# GP-to-Node projection matrix and shape-derivative tables - everything is
# vectorized, no per-vertex Python loop.
g = fd.recover_gradient(mesh, field)  # (n_nodes, 2)
H = fd.recover_hessian(mesh, field)  # (n_nodes, 2, 2)

# Compare against the analytic Hessian on interior nodes.
k = (2 * np.pi) ** 2
H_xx_exact = -k * np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)
H_xy_exact = +k * np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y)

interior = (x > 0.1) & (x < 0.9) & (y > 0.1) & (y < 0.9)
err_xx = np.abs(H[interior, 0, 0] - H_xx_exact[interior]).max()
err_xy = np.abs(H[interior, 0, 1] - H_xy_exact[interior]).max()
print(f"max |H_xx - exact| (interior) = {err_xx:.3e}")
print(f"max |H_xy - exact| (interior) = {err_xy:.3e}")

###############################################################################
# Save a vtk file with the recovered fields for visualisation in ParaView.
ds = fd.DataSet(
    mesh,
    data={
        "field": field,
        "grad": g.T,  # (2, n_nodes) for fedoo's vector layout
        "H_xx": H[:, 0, 0],
        "H_yy": H[:, 1, 1],
        "H_xy": H[:, 0, 1],
    },
    data_type="node",
)
ds.save("hessian_recovery.vtk")
print("Wrote hessian_recovery.vtk")

###############################################################################
# To drive an mmg adaptive remesh, pack the Hessian into mmg's
# row-major upper-triangular layout and feed it to
# ``mmgpy.metrics.create_metric_from_hessian``::
#
#     metric = fd.to_upper_diagonal(H)        # (n_nodes, 3) for 2D
#     # mmgpy.metrics.create_metric_from_hessian(metric, target_error=1e-3, ...)
