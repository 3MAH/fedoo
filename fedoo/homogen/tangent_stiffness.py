# derive de ConstitutiveLaw
# This law should be used with an InternalForce WeakForm

from fedoo.core.base import MeshBase as Mesh
from fedoo.constitutivelaw import ElasticAnisotropic
from fedoo.core.base import ConstitutiveLaw
from fedoo.weakform.stress_equilibrium import StressEquilibrium
from fedoo.core.assembly import Assembly
from fedoo.core.problem import Problem
from fedoo.problem.linear import Linear
from fedoo.core.base import ProblemBase
from fedoo.constraint.periodic_bc import PeriodicBC
import numpy as np
import os
import time


def get_homogenized_stiffness(assemb, meshperio=True, **kargs):
    # Definition of the set of nodes for boundary conditions
    if isinstance(assemb, str):
        assemb = Assembly.get_all()[assemb]

    # Type of problem
    pb = Linear(assemb)
    pb.set_A(assemb.get_global_matrix())

    C = get_tangent_stiffness(pb, meshperio, **kargs)

    return C


def get_tangent_stiffness(pb=None, meshperio=True, **kargs):
    #################### PERTURBATION METHODE #############################
    solver = kargs.get("solver", "direct")
    solver_type = kargs.get("solver_type", None)
    pc_type = kargs.get("pc_type", None)

    if pb is None:
        pb = ProblemBase.get_active()
    elif isinstance(pb, str):
        pb = ProblemBase.get_all()[pb]
    mesh = pb.mesh

    crd = mesh.nodes

    crd_center = mesh.bounding_box.center
    center = [np.linalg.norm(crd - crd_center, axis=1).argmin()]

    ndim = pb.space.ndim

    if ndim == 3:
        BC_perturb = np.eye(6)
        # BC_perturb[3:6,3:6] *= 2 #2xEXY
    else:  # ndim == 2
        BC_perturb = np.eye(3)

    DStrain = []
    DStress = []

    if "_perturbation" in pb.get_all() and Problem["_perturbation"].mesh is not mesh:
        # if required an option could be added to delete '_perturbation' in case the mesh may change
        print(
            'WARNING: delete old "_perturbation" problem that is related to another mesh'
        )
        del pb.get_all()["_perturbation"]

    if "_perturbation" not in pb.get_all():
        # initialize perturbation problem
        pb_post_tt = Problem(0, 0, 0, mesh, name="_perturbation")
        pb_post_tt.set_solver(solver, solver_type=solver_type, pc_type=pc_type)

        pb.make_active()

        # Shall add other conditions later on
        pb_post_tt.bc.add(
            PeriodicBC(
                "small_strain",
                meshperio=meshperio,
            )
        )

        pb_post_tt.bc.add(
            "Dirichlet",
            center,
            list(pb.space.list_variables()),
            0,
            name="center",
        )
    else:
        pb_post_tt = Problem["_perturbation"]

    pb_post_tt.set_A(pb.get_A())
    pb.bc.remove("_Strain")

    # typeBC = 'Dirichlet' #doesn't work with meshperio = False
    typeBC = "Neumann"

    for i in range(len(BC_perturb)):
        pb_post_tt.bc.add(
            typeBC,
            "E_xx",
            BC_perturb[i][0],
            start_value=0,
            name="_Strain",
        )  # EpsXX
        pb_post_tt.bc.add(
            typeBC,
            "E_yy",
            BC_perturb[i][1],
            start_value=0,
            name="_Strain",
        )  # EpsYY
        if ndim == 3:
            pb_post_tt.bc.add(
                typeBC,
                "E_zz",
                BC_perturb[i][2],
                start_value=0,
                name="_Strain",
            )  # EpsZZ
            pb_post_tt.bc.add(
                typeBC,
                "E_xy",
                BC_perturb[i][3],
                start_value=0,
                name="_Strain",
            )  # EpsXY
            pb_post_tt.bc.add(
                typeBC,
                "E_xz",
                BC_perturb[i][4],
                start_value=0,
                name="_Strain",
            )  # EpsXZ
            pb_post_tt.bc.add(
                typeBC,
                "E_yz",
                BC_perturb[i][5],
                start_value=0,
                name="_Strain",
            )  # EpsYZ
        else:
            pb_post_tt.bc.add(
                typeBC,
                "E_xy",
                BC_perturb[i][2],
                start_value=0,
                name="_Strain",
            )  # EpsXY

        pb_post_tt.apply_boundary_conditions()

        pb_post_tt.solve()
        if typeBC == "Neumann":
            X = pb_post_tt.get_X()  # alias
            list_res = DStrain
        else:
            X = pb_post_tt.get_ext_forces()  # F
            list_res = DStress

        if ndim == 3:
            list_res.append(
                np.array(
                    [
                        pb_post_tt._get_vect_component(X, "E_xx")[0],
                        pb_post_tt._get_vect_component(X, "E_yy")[0],
                        pb_post_tt._get_vect_component(X, "E_zz")[0],
                        pb_post_tt._get_vect_component(X, "E_xy")[0],
                        pb_post_tt._get_vect_component(X, "E_xz")[0],
                        pb_post_tt._get_vect_component(X, "E_yz")[0],
                    ]
                )
            )
        else:  # ndim == 2
            list_res.append(
                np.array(
                    [
                        pb_post_tt._get_vect_component(X, "E_xx")[0],
                        pb_post_tt._get_vect_component(X, "E_yy")[0],
                        pb_post_tt._get_vect_component(X, "E_xy")[0],
                    ]
                )
            )

    volume = mesh.bounding_box.volume
    if typeBC == "Neumann":
        C = np.linalg.inv(np.array(DStrain).T) / volume
    else:
        C = np.array(DStress).T / volume

    return C
