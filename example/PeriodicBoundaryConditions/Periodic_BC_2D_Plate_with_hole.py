import fedoo as fd 
import numpy as np

#------------------------------------------------------------------------------
# Dimension of the problem
#------------------------------------------------------------------------------
fd.ModelingSpace("2Dstress")

#------------------------------------------------------------------------------
# Definition of the Geometry 
#------------------------------------------------------------------------------
mesh = fd.mesh.hole_plate_mesh(name = 'Domain')

#alternative mesh below (uncomment the line)
# Mesh.rectangle_mesh(Nx=51, Ny=51, x_min=-50, x_max=50, y_min=-50, y_max=50, ElementShape = 'quad4', name ="Domain")
type_el = mesh.elm_type
fd.util.mesh_plot_2d(mesh) #plot the mesh (using matplotlib)

#------------------------------------------------------------------------------
# Set of nodes for boundary conditions
#------------------------------------------------------------------------------
crd = mesh.nodes 
xmax = np.max(crd[:,0]) ; xmin = np.min(crd[:,0])
ymax = np.max(crd[:,1]) ; ymin = np.min(crd[:,1])

center = mesh.nearest_node(mesh.bounding_box.center)

#------------------------------------------------------------------------------
# Adding virtual nodes related the macroscopic strain
#------------------------------------------------------------------------------
StrainNodes = mesh.add_nodes(np.zeros(crd.shape[1]),2) 
#The position of the virtual node has no importance (the position is arbitrary set to [0,0,0])
#For a problem in 2D with a 2D periodicity, we need 3 independant strain component 
#2 nodes (with 2 dof per node in 2D) are required

#The dof 'DispX' of the node StrainNodes[0] will be arbitrary associated to the EXX strain component
#The dof 'DispY' of the node StrainNodes[1] will be arbitrary associated to the EYY strain component
#The dof 'DispY' of the node StrainNodes[0] will be arbitrary associated to the EXY strain component
#The dof 'DispX' of the node StrainNodes[1] is not used and will be blocked to avoid singularity

#------------------------------------------------------------------------------
#Material definition
#------------------------------------------------------------------------------
fd.constitutivelaw.ElasticIsotrop(1e5, 0.3, name = 'ElasticLaw')

#------------------------------------------------------------------------------
#Mechanical weak formulation
#------------------------------------------------------------------------------
fd.weakform.StressEquilibrium("ElasticLaw")

#------------------------------------------------------------------------------
#Global Matrix assembly
#------------------------------------------------------------------------------
fd.Assembly.create("ElasticLaw", mesh, type_el, name="Assembling") 

#------------------------------------------------------------------------------
#Static problem based on the just defined assembly
#------------------------------------------------------------------------------
pb = fd.problem.Linear("Assembling")

#------------------------------------------------------------------------------
#Boundary conditions
#------------------------------------------------------------------------------
#Macroscopic strain component to enforce
Exx = 0
Eyy = 0
Exy = 0.1

#Add some multipoint constraint for periodic conditions associated to the defined strain dof
pb.bc.add(fd.constraint.PeriodicBC([StrainNodes[0], StrainNodes[1], StrainNodes[0]], ['DispX', 'DispY', 'DispY']))
# fd.homogen.DefinePeriodicBoundaryCondition(mesh, [StrainNodes[0], StrainNodes[1], StrainNodes[0]], ['DispX', 'DispY', 'DispY'], dim='2D')

#Mean strain: Dirichlet (strain) or Neumann (associated mean stress) can be enforced
pb.bc.add('Dirichlet',[StrainNodes[0]], 'DispX', Exx) #EpsXX
pb.bc.add('Dirichlet',[StrainNodes[0]], 'DispY', Exy) #EpsXY
# Problem.bc.add('Neumann','DispX', 1e4, [StrainNodes[0]]) #EpsXY

pb.bc.add('Dirichlet',[StrainNodes[1]], 'DispX', 0) #nothing (blocked to avoir singularity)
pb.bc.add('Dirichlet',[StrainNodes[1]], 'DispY', Eyy) #EpsYY

#Block one node to avoid singularity
pb.bc.add('Dirichlet', center, 'Disp', 0)

pb.apply_boundary_conditions()

#------------------------------------------------------------------------------
#Solve
#------------------------------------------------------------------------------
pb.set_solver('CG') #Preconditioned Conjugate Gradient
pb.solve()

#------------------------------------------------------------------------------
# Post-treatment
#------------------------------------------------------------------------------
res = pb.get_results("Assembling", ['Disp','Stress'])

#plot the deformed mesh with the shear stress (component=3). 
res.plot('Stress', 'XY', 'Node')
# simple matplotlib alternative if pyvista is not installed:
#fd.util.field_plot_2d("Assembling", disp = pb.get_dof_solution(), dataname = 'Stress', component=3, scale_factor = 1, plot_edge = True, nb_level = 6, type_plot = "smooth")

# print the macroscopic strain tensor and stress tensor
print('Strain tensor ([Exx, Eyy, Exy]): ', [pb.get_disp('DispX')[-2], pb.get_disp('DispY')[-1], pb.get_disp('DispY')[-2]])
#Compute the mean stress 
#Get the stress tensor (PG values)

# mesh.get_volume() #surface of domain without the void (hole)
Surf = (xmax-xmin)*(ymax-ymin) #total surface of the domain
MeanStress = [1/Surf*mesh.integrate_field(res['Stress'][i]) for i in [0,1,3]]

print('Stress tensor ([Sxx, Syy, Sxy]): ', MeanStress)


