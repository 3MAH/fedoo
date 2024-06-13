# -*- coding: utf-8 -*-


###pep8 name converter for fedoo

import os, fnmatch


def findReplace(dirs, find, replace, filePattern):
    for directory in dirs:
        for path, dirs, files in os.walk(os.path.abspath(directory)):
            for filename in fnmatch.filter(files, filePattern):
                if filename not in ["fedoo2pep8.py", "fedoo2pep8_script.py"]:
                    filepath = os.path.join(path, filename)
                    with open(filepath, encoding="utf-8") as f:
                        s = f.read()
                    s = s.replace(find, replace)
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(s)


def findReplace_change(directories, find, replace, change_type=0, filePattern="*.py"):
    # change_type = 0 - change ( ... ) by [ ... ] after the name
    for directory in directories:
        for path, dirs, files in os.walk(os.path.abspath(directory)):
            for filename in fnmatch.filter(files, filePattern):
                if filename not in ["fedoo2pep8.py", "fedoo2pep8_script.py"]:
                    filepath = os.path.join(path, filename)
                    with open(filepath, encoding="utf-8") as f:
                        s = f.read()

                    ind_start = s.find(find)
                    while ind_start != -1:
                        ind_end = ind_start + len(find)
                        if change_type == 0 and (
                            s[ind_end] == "(" or s[ind_end + 1] == "("
                        ):
                            ind_end = s.find(")", ind_end) + 1

                            change_s = (
                                s[ind_start:ind_end]
                                .replace(find, replace)
                                .replace("(", "[")
                                .replace(")", "]")
                            )

                            s = s[:ind_start] + change_s + s[ind_end:]
                        else:
                            assert 0, "error, check the file"
                        ind_start = s.find(find)

                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(s)


dirs = ["../example", "../tests"]
# dirs = ["../tests"]
rep = {}

# # Mesh
# rep["GetAll"] = "get_all"
# rep[".GetNodeCoordinates()"] = ".nodes"
# rep[".GetElementTable()"] = ".elements"
# rep[".GetElementShape()"] = ".elm_type"
# rep[".GetNumberOfElements()"] = ".n_elements"
# rep[".GetNumberOfNodes()"] = ".n_nodes"
# rep["AddSetOfNodes"] = "add_node_set"
# rep["AddSetOfElements"] = "add_element_set"
# rep[".AddNodes"] = ".add_nodes"
# rep[".FindNodes"] = ".find_nodes"
# rep[".Translate"] = ".translate"
# rep["GetNearestNode"] = "nearest_node"
# rep["GetBoundingBox"] = "bounding_box"
# rep[".GetCoordinateID()"] = ".crd_name"

# rep["AddInternalNodes"] = "add_internal_nodes"
# rep["Mesh.Stack"] = "Mesh.stack"
# rep[".MergeNodes"] = ".merge_nodes"
# rep[".RemoveNodes"] = ".remove_nodes"
# rep["FindCoincidentNodes"] = "find_coincident_nodes"
# rep["ExtractSetOfElements"] = "extract_elements"
# rep["FindNonUsedNodes"] = "find_isolated_nodes"
# rep["RemoveNonUsedNodes"] = "remove_isolated_nodes"

# #Meshtools
# rep["RectangleMesh"] = "rectangle_mesh"
# rep["GridMeshCylindric"] = "grid_mesh_cylindric"
# rep["LineMesh1D"] = "line_mesh_1D"
# rep["LineMeshCylindric"] = "line_mesh_cylindric"
# rep["LineMesh"] = "line_mesh"
# rep["BoxMesh"] = "box_mesh"
# rep["GridStructuredMesh2D"] = "structured_mesh_2D"
# rep["GenerateNodes"] = "generate_nodes"
# rep["HolePlateMesh"] = "hole_plate_mesh"

# #MeshImport
# rep["ImportFromFile"] = "import_file"
# rep["ImportFromMSH"] = "import_msh"
# rep["ImportFromVTK"] = "import_vtk"

# rep["ID"] = "name"


# for key in rep:
#     findReplace(dirs, key, rep[key], "*.py")

# findReplace_change(dirs, ".GetSetOfNodes", ".node_sets", 0)
# findReplace_change(dirs, ".GetSetOfElements", ".element_sets", 0)


# ==================================================
# Assembly module
# ==================================================
# rep["GetMatrixChangeOfBasis"] = "get_change_of_basis_mat"
# rep["DeleteMemory"] = "delete_memory"
# rep["computeMatrixMethod"] = "_assembly_method"
# rep["ComputeGlobalMatrix"] = "assemble_global_mat"
# rep[".GetMesh()"] = ".mesh"
# # SetMesh non modifié
# rep[".GetWeakForm()"] = ".weakform"
# rep[".Initialize"] = ".initialize"
# rep[".GetNumberOfGaussPoints()"] = ".n_elm_gp"
# rep["nb_pg"] = "n_elm_gp"
# rep["GetElementResult"] = "get_element_results"
# rep["GetGaussPointResult"] = "get_gp_results"
# rep["GetNodeResult"] = "get_node_results"
# rep["ConvertData"] = "convert_data"
# rep["IntegrateField"] = "integrate_field"
# rep["GetStrainTensor"] = "get_strain"
# rep["GetGradTensor"] = "get_grad_disp"
# rep["GetExternalForces"] = "get_ext_forces"
# rep["GetInternalForces"] = "get_int_forces"
# rep["DetermineDataType"] = "determine_data_type"
# rep["deleteGlobalMatrix"] = "delete_global_mat"
# rep["GetMatrix"] = "global_matrix"
# rep["GetVector"] = "global_vector"
# rep["Assembly.Create"] = "Assembly.create"


# ==================================================
# Util module
# ==================================================
# rep["Util.ProblemDimension"] = "ModelingSpace"


# ==================================================
# Change module name
# ==================================================
# rep["fd.ConstitutiveLaw"] = "fd.constitutivelaw"
# rep["fd.WeakForm"] = "fd.weakform"
# rep["fd.Assembly"] = "fd.assembly"
# rep["fd.Problem"] = "fd.problem"
# rep["fd.Homogen"] = "fd.homogen"
# rep["fd.PGD"] = "fd.pgd"


# ==================================================
# Problem
# ==================================================

# rep["GetActive"] = "get_active"
# rep["SetActive"] = "set_active"
# rep["SetSolver"] = "set_solver"
# rep["GetDisp("] = "get_disp("
# rep["GetRot("] = "get_rot("
# rep["GetTemp("] = "get_temp("
# rep["AddOutput"] = "add_output"
# rep["SaveResults"] = "save_results"
# rep["GetResults"] = "get_results"
# rep["ApplyBoundaryCondition"] = "apply_boundary_conditions"

# rep["NLSolve("] = "nlsolve("
# rep["Solve("] = "solve("

# rep["NLsolve("] = "nlsolve("
# rep[".BoundaryCondition("] = ".bc.add("
# rep[".Static("] = ".Linear("
# rep[".NonLinearStatic("] = ".NonLinear("
# rep["SetNewtonRaphsonErrorCriterion"] = "set_nr_criterion"
# rep["GetDoFSolution"] = "get_dof_solution"

# ==================================================
# WeakForm
# ==================================================
rep["InternalForce"] = "StressEquilibrium"


# ==================================================
# Other
# ==================================================
# rep["arrayStressTensor"] = "StressTensorArray"
# rep["listStressTensor"] = "StressTensorList"
# rep["listStrainTensor"] = "StrainTensorList"


for key in rep:
    findReplace(dirs, key, rep[key], "*.py")

assert 0

# WeakForm


rep["GetDisp"] = "get_disp"
rep["GetRot"] = "get_rot"
rep["GetTemp"] = "get_temp"


rep["ChangeAssembly"] = "change_assembly"
rep["SetNewtonRaphsonErrorCriterion"] = "set_error_criterion"

rep["NewtonRaphsonError"] = "newton_raphson_error"
rep["NewtonRaphsonIncr"] = "newton_raphson_increment"
rep["ResetLoadFactor"] = "reset_load_factor"
rep["NLSolve"] = "nlsolve"


rep["GetElasticEnergy"] = "get_elastic_energy"
rep["GetNodalElasticEnergy"] = "get_nodal_elastic_energy"

rep["AddOutput"] = "add_output"
rep["SaveResults"] = "save_results"
rep["GetResults"] = "get_results"
rep["ApplyBoundaryCondition"] = "apply_boundary_condition"


rep["SetA"] = "set_A"
rep["GetA"] = "get_A"
rep["SetB"] = "set_B"
rep["GetB"] = "get_B"
rep["SetD"] = "set_D"
rep["GetD"] = "get_D"
rep["GetX"] = "get_X"

rep["GetMesh"] = "get_mesh"


rep["GetDoFSolution"] = "get_dof_solution"
rep["SetDoFSolution"] = "set_dof_solution"  # usefull ???

rep["SetInitialBCToCurrent"] = "set_initial_bc_to_current"
rep["GetVectorComponent"] = "get_vector_component"

rep["GetVelocity"] = "get_velocity"  # deprecated
rep["GetAcceleration"] = "get_acceleration"  # deprecated
rep["__Xdot"] = "__acceleration"
rep["__Xdotdot"] = "__velocity"
rep["__Acceleration"] = "__acceleration"
rep["__Velocity"] = "__velocity"


rep["SetInitialDisplacement"] = "set_initial_displacement"
rep["SetInitialVelocity"] = "set_initial_velocity"
rep["SetInitialAcceleration"] = "set_initial_acceleration"
rep["SetRayleighDamping"] = "set_rayleigh_damping"
rep["GetExternalForces"] = "get_external_forces"
rep["GetKineticEnergy"] = "get_kinetic_energy"
rep["GetDampingPower"] = "get_damping_power"
rep["UpdateStiffness"] = "update_stiffness"

rep["GetXbc"] = "get_Xbc"
rep["ComputeResidualNorm"] = "residual_norm"
rep["GetResidual"] = "get_residual"
rep["UpdatePGD"] = "update_pgd"
rep["UpdateAlpha"] = "update_alpha"
rep["AddNewTerm"] = "add_new_term"

rep[".Initialize"] = "initialize"
rep["def Initialize"] = "def initialize"
rep["Reset"] = "reset"
rep[".Update"] = ".update"
rep["def Update"] = "def update"

rep["UniqueBoundaryCondition"] = (
    "qsdfqsfhjqsdljkfdhqsdlkfjhqsdflkhjqsddlqjksh"  # save UniqueBoundaryCondition
)
rep["BoundaryCondition"] = "boundary_condition"
rep["qsdfqsfhjqsdljkfdhqsdlkfjhqsdflkhjqsddlqjksh"] = (
    "UniqueBoundaryCondition"  # reload UniqueBoundaryCondition
)

rep["Solve"] = "solve"


#### TODO remove Xdot and Xdotdot from newmark problems
# rep["GetXdot"] = "get_Xdot"
# rep["GetXdotdot"] = "get_Xdotdot"


# ========================
# Script changes that should be done by hand
# ========================
# - Bounding box retrun a class with xmin, xmax, ymin, ... and center properties
#  This is not compatible with the previous syntax
