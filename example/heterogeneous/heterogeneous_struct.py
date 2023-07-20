import fedoo as fd
import numpy as np

#Generate a mesh with a spheric inclusion inside
mesh = fd.mesh.hole_plate_mesh(nr=11, nt=11, length=100, height=100, radius=20, \
	elm_type = 'quad4', name ="Domain")

mesh.element_sets['matrix'] = np.arange(0,mesh.n_elements)

mesh_disk = fd.mesh.disk_mesh(20, 11, 11)
mesh_disk.element_sets['inclusion'] = np.arange(0,mesh_disk.n_elements)
    
mesh = mesh + mesh_disk

#glue the inclusion to the matrix
mesh.merge_nodes(np.c_[mesh.node_sets['hole_edge'], mesh.node_sets['boundary']])


#Define the Modeling Space - Here 2D problem with plane stress assumption.
fd.ModelingSpace("2Dstress") 

method = 1
if method == 1:
    ### method 1: sum assembly 
    #Define an elastic isotropic material with 
    material1 = fd.constitutivelaw.ElasticIsotrop(2e4, 0.3) 
    material2 = fd.constitutivelaw.ElasticIsotrop(1e5, 0.3) 
    # props = np.array([1e5, 0.3, 1e-5, 300,1000,0.3]) #E, nu, alpha, Re,k,m
    # material2 = fd.constitutivelaw.Simcoon("EPICP", props)
    
    #Create the weak formulation of the mechanical equilibrium equation
    wf1 = fd.weakform.StressEquilibrium(material1) 
    wf2 = fd.weakform.StressEquilibrium(material2) 
    
    #Create a global assembly
    assemb1 = fd.Assembly.create(wf1, mesh.extract_elements('matrix')) 
    assemb2 = fd.Assembly.create(wf2, mesh.extract_elements('inclusion')) 
    
    assembly = assemb1 + assemb2

elif method == 2:
    ### method 2: use constitutive law with array properties parameters
    
    # More simple but only work if the material have the same kind of material
    
    #Define young modulus for each element
    E = np.empty(mesh.n_elements)
    E[mesh.element_sets['matrix']] = 2e4
    E[mesh.element_sets['inclusion']] = 1e5
    
    material = fd.constitutivelaw.ElasticIsotrop(E, 0.3) 
    wf = fd.weakform.StressEquilibrium(material) 
    assembly = fd.Assembly.create(wf, mesh) 

elif method == 3: 
    
    ### method 3: build an heterogeneous constitutive law
    material1 = fd.constitutivelaw.ElasticIsotrop(2e4, 0.3) 
    material2 = fd.constitutivelaw.ElasticIsotrop(1e5, 0.3) 
    # props = np.array([1e5, 0.3, 1e-5, 300,1000,0.3]) #E, nu, alpha, Re,k,m
    # material2 = fd.constitutivelaw.Simcoon("EPICP", props)
    
    material = fd.constitutivelaw.Heterogeneous((material1, material2), ('matrix', 'inclusion'))
    
    wf = fd.weakform.StressEquilibrium(material) 
    assembly = fd.Assembly.create(wf, mesh) 
    



#Define a new static problem
pb = fd.problem.Linear(assembly)

#Definition of the set of nodes for boundary conditions
left = mesh.find_nodes('X',mesh.bounding_box.xmin)
right = mesh.find_nodes('X',mesh.bounding_box.xmax)
bottom = mesh.find_nodes('Y', mesh.bounding_box.ymin)

#Boundary conditions
#symetry condition on left (ux = 0)
pb.bc.add('Dirichlet', left, 'Disp',    0 ) 
#symetry condition on bottom edge (ux = 0)
# pb.bc.add('Dirichlet', bottom, 'DispY',    0 ) 
#displacement on right (ux=0.1mm)
pb.bc.add('Dirichlet', right, 'Disp', [20,0] ) 

pb.apply_boundary_conditions()

#Solve problem
pb.solve()

#---------- Post-Treatment ----------
field_plot = 'Stress'
component = 'vm'
if method == 1:
    #Get the stress tensor, strain tensor, and displacement (nodal values)
    res1 = pb.get_results(assemb1, ['Stress','Strain', 'Disp'])
    res2 = pb.get_results(assemb2, ['Stress','Strain', 'Disp'])
    
    pl = res2.plot(field_plot,component=component, show=False)
    res1.plot(field_plot, component=component, plotter=pl)
    # res2.plot(field_plot, 'Node',component=component)
    # res1.plot(field_plot, 'Node', component=component)


if method == 2 or method == 3:
    res = pb.get_results(assembly, ['Stress','Strain', 'Disp'])
    res.plot(field_plot, component=component)
