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


dirs = ["../fedoo"]
# dirs = ["../tests"]
rep = {}


# rep[".GetActive()"] = ".get_active()"
# rep[".GetDimension()"] = ".get_dimension()"
# rep[".list_variable"] = ".list_variables"
# rep[".list_coordinate"] = ".list_coordinates"
# rep[".list_vector"] = ".list_vectors"
# rep["fedoo.utilities.modelingspace"] = "fedoo.core.modelingspace"

# rep["OpDiff"] = "DiffOp"

# rep["fedoo.core._mechanical3d import Mechanical3D"] = "fedoo.core.mechanical3d import Mechanical3D"
# rep["fedoo.weakform.weakform import WeakForm"] = "fedoo.core.base import WeakForm"
# rep["fedoo.weakform.weakform   import WeakForm"] = "fedoo.core.base import WeakForm"
# rep["fedoo.problem.ProblemBase"] = "fedoo.core.base"
# rep["from fedoo.problem.Problem import"] = "from fedoo.core.problem import"
# rep["fedoo.utilities."] = "fedoo.util."
# rep["fedoo.util.dataset"] = "fedoo.core.dataset"


# rep["SeparatedLocalFrame"] = "separated_local_frame"
# rep["GlobalLocalFrame"] = "global_local_frame"


# Problem


# rep["GetDisp("] = "get_disp("
# rep["GetRot("] = "get_rot("
# rep["GetTemp("] = "get_temp("


# rep["AddOutput"] = "add_output"
# rep["SaveResults"] = "save_results"
# rep["GetResults"] = "get_results"
# rep["ApplyBoundaryCondition"] = "apply_boundary_conditions"


# rep["GetBC"] = "get_bc"
# rep["RemoveBC"] = "remove_bc"
# rep["PrintBC"] = "print_bc"

# rep["ChangeAssembly"] = "change_assembly"
# rep["SetNewtonRaphsonErrorCriterion"] = "set_error_criterion"

# rep["NewtonRaphsonError"] = "newton_raphson_error"
# rep["NewtonRaphsonIncr"] = "newton_raphson_increment"
# rep["ResetLoadFactor"] = "reset_load_factor"
# rep["NLSolve"] = "nlsolve"


# rep["GetElasticEnergy"] = "get_elastic_energy"
# rep["GetNodalElasticEnergy"] = "get_nodal_elastic_energy"


# rep["SetA"] = "set_A"
# rep["GetA"] = "get_A"
# rep["SetB"] = "set_B"
# rep["GetB"] = "get_B"
# rep["SetD"] = "set_D"
# rep["GetD"] = "get_D"
# rep["GetX"] = "get_X"


# rep["GetDoFSolution"] = "get_dof_solution"
# rep["SetDoFSolution"] = "set_dof_solution" #usefull ???


# rep["SetInitialBCToCurrent"] = "set_initial_bc_to_current"
# rep["GetVectorComponent"] = "get_vector_component"

# rep["GetVelocity"] = "get_velocity" #deprecated
# rep["GetAcceleration"] = "get_acceleration" #deprecated
# rep["__Xdot"] = "__acceleration"
# rep["__Xdotdot"] = "__velocity"
# rep["__Acceleration"] = "__acceleration"
# rep["__Velocity"] = "__velocity"


# rep["SetInitialDisplacement"] = "set_initial_displacement"
# rep["SetInitialVelocity"] = "set_initial_velocity"
# rep["SetInitialAcceleration"] = "set_initial_acceleration"
# rep["SetRayleighDamping"] = "set_rayleigh_damping"
# rep["GetExternalForces"] = "get_external_forces"
# rep["GetKineticEnergy"] = "get_kinetic_energy"
# rep["GetDampingPower"] = "get_damping_power"
# rep["UpdateStiffness"] = "update_stiffness"

# rep["GetXbc"] = "get_Xbc"
# rep["ComputeResidualNorm"] = "residual_norm"
# rep["GetResidual"] = "get_residual"
# rep["UpdatePGD"] = "update_pgd"
# rep["UpdateAlpha"] = "update_alpha"
# rep["AddNewTerm"] = "add_new_term"

# rep[".Initialize"] = "initialize"
# rep["def Initialize"] = "def initialize"
# rep["Reset"] = "reset"
# rep[".Update"] = ".update"
# rep["def Update"] = "def update"

# rep["UniqueBoundaryCondition"] = "qsdfqsfhjqsdljkfdhqsdlkfjhqsdflkhjqsddlqjksh" #save UniqueBoundaryCondition
# rep["BoundaryCondition"] = "boundary_condition"
# rep["qsdfqsfhjqsdljkfdhqsdlkfjhqsdflkhjqsddlqjksh"] = "UniqueBoundaryCondition" #reload UniqueBoundaryCondition

# rep["UniqueBoundaryCondition"] = "BoundaryCondition"

# rep[".SetD("] = "._D = "

# rep[".GetA()"] = "._A"
# rep[".GetB()"] = "._B"
# rep[".GetD()"] = "._D"

# rep["GetPKII("] = "get_pk2("
# rep["GetKirchhoff("] = "get_kirchhoff("
# rep["GetCauchy("] = "get_cauchy("
# rep["GetStrain("] = "get_strain("
# rep["GetStatev("] = "get_statev("
# rep["GetWm("] = "get_wm("
# rep["GetStress("] = "get_stress("

# rep["get_disp_gradient("] = "get_disp_grad("
#     if self.__currentGradDisp is 0: return 0
#     else: return self.__currentGradDisp

# def GetTangentMatrix(self):


# rep["SolveTimeIncrement"] = "solve_time_increment"
# rep["GetDoFSolution"] = "get_dof_solution"
# rep["NewtonRaphsonIncrement"] = "nr_increment"
# rep["NewtonRaphsonError"] = "nr_error"
# rep["get_DifferentialOperator"] = "get_weak_equation"

# rep["GetNodePositionInElementCoordinates"] = "get_node_elm_coordinates"
rep["GetTangentMatrix"] = "get_tangent_matrix"

for key in rep:
    findReplace(dirs, key, rep[key], "*.py")

assert 0

# WeakForm


#### TODO remove Xdot and Xdotdot from newmark problems
# rep["GetXdot"] = "get_Xdot"
# rep["GetXdotdot"] = "get_Xdotdot"


# ========================
# Script changes that should be done by hand
# ========================
# - Bounding box retrun a class with xmin, xmax, ymin, ... and center properties
#  This is not compatible with the previous syntax
