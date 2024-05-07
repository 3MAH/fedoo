
import fedoo as fd

space = fd.ModelingSpace("2D")

U = space.new_variable("U")

dU_dX = space.derivative("U","X")
dU_dY = space.derivative("U","Y")

wf = fd.WeakForm(dU_dX.virtual*dU_dX + dU_dY.virtual*dU_dY, name = "Poisson Equation")
mesh = fd.mesh.rectangle_mesh()
# mesh = fd.mesh.box_mesh()

fd.Assembly.create(wf, mesh, name = 'assembling')

pb = fd.problem.Linear('assembling')
pb.bc.add("Dirichlet", "left", "U", 0)
pb.bc.add("Dirichlet", "right", "U", 1)

pb.apply_boundary_conditions()

pb.solve()

res = pb.get_results('assembling', ["U"])
res.plot("U", 'GaussPoint')