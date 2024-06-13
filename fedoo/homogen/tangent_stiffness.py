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
    mesh = assemb.mesh

    if "_StrainNodes" in mesh.node_sets:
        crd = mesh.nodes[:-2]
    else:
        crd = mesh.nodes

    type_el = mesh.elm_type
    xmax = np.max(crd[:, 0])
    xmin = np.min(crd[:, 0])
    ymax = np.max(crd[:, 1])
    ymin = np.min(crd[:, 1])
    zmax = np.max(crd[:, 2])
    zmin = np.min(crd[:, 2])
    crd_center = (np.array([xmin, ymin, zmin]) + np.array([xmax, ymax, zmax])) / 2
    center = [np.linalg.norm(crd - crd_center, axis=1).argmin()]

    BC_perturb = np.eye(6)
    # BC_perturb[3:6,3:6] *= 2 #2xEXY

    DStrain = []
    DStress = []

    if "_StrainNodes" in mesh.node_sets:
        StrainNodes = mesh.node_sets["_StrainNodes"]
        remove_strain = False
    else:
        StrainNodes = mesh.add_nodes(
            crd_center, 2
        )  # add virtual nodes for macro strain
        mesh.add_node_set(StrainNodes, "_StrainNodes")
        remove_strain = True

    # Type of problem
    pb = Linear(assemb)

    C = get_tangent_stiffness(pb, meshperio, **kargs)
    if remove_strain:
        mesh.remove_nodes(StrainNodes)
        del mesh.node_sets["_StrainNodes"]

    # del pb.get_all()['_perturbation'] #erase the perturbation problem in case of homogenized stiffness is required for another mesh

    return C


def get_homogenized_stiffness_2(mesh, L, meshperio=True, Problemname=None, **kargs):
    print(
        "WARNING: get_homogenized_stiffness_2 will be deleted in future versions of fedoo. Use get_homogenized_stiffness instead after building an Assembly object."
    )
    #################### PERTURBATION METHODE #############################

    solver = kargs.get("solver", "direct")

    # Definition of the set of nodes for boundary conditions
    if isinstance(mesh, str):
        mesh = Mesh.get_all()[mesh]

    if "_StrainNodes" in mesh.node_sets:
        crd = mesh.nodes[:-2]
    else:
        crd = mesh.nodes

    type_el = mesh.elm_type
    # type_el = 'hex20'
    xmax = np.max(crd[:, 0])
    xmin = np.min(crd[:, 0])
    ymax = np.max(crd[:, 1])
    ymin = np.min(crd[:, 1])
    zmax = np.max(crd[:, 2])
    zmin = np.min(crd[:, 2])
    crd_center = (np.array([xmin, ymin, zmin]) + np.array([xmax, ymax, zmax])) / 2
    center = [np.linalg.norm(crd - crd_center, axis=1).argmin()]

    BC_perturb = np.eye(6)
    # BC_perturb[3:6,3:6] *= 2 #2xEXY

    DStrain = []
    DStress = []

    if "_StrainNodes" in mesh.node_sets:
        StrainNodes = mesh.node_sets["_StrainNodes"]
    else:
        StrainNodes = mesh.add_nodes(
            crd_center, 2
        )  # add virtual nodes for macro strain
        mesh.add_node_set(StrainNodes, "_StrainNodes")

    ElasticAnisotropic(L, name="ElasticLaw")

    # Assembly
    StressEquilibrium("ElasticLaw")
    Assembly("ElasticLaw", mesh, type_el, name="Assembling")

    # Type of problem
    pb = Linear("Assembling")

    pb_post_tt = Problem(0, 0, 0, mesh, name="_perturbation")
    pb_post_tt.set_solver(solver)
    pb_post_tt.set_A(pb.get_A())

    # Shall add other conditions later on
    pb_post_tt.bc.add(
        PeriodicBC(
            [
                StrainNodes[0],
                StrainNodes[0],
                StrainNodes[0],
                StrainNodes[1],
                StrainNodes[1],
                StrainNodes[1],
            ],
            ["DispX", "DispY", "DispZ", "DispX", "DispY", "DispZ"],
            dim=3,
            meshperio=meshperio,
        )
    )

    pb_post_tt.bc.add("Dirichlet", center, "Disp", 0, name="center")

    pb_post_tt.apply_boundary_conditions()

    # typeBC = 'Dirichlet' #doesn't work with meshperio = False
    typeBC = "Neumann"

    for i in range(6):
        pb_post_tt.bc.remove("_Strain")
        pb_post_tt.bc.add(
            typeBC,
            [StrainNodes[0]],
            "DispX",
            BC_perturb[i][0],
            start_value=0,
            name="_Strain",
        )  # EpsXX
        pb_post_tt.bc.add(
            typeBC,
            [StrainNodes[0]],
            "DispY",
            BC_perturb[i][1],
            start_value=0,
            name="_Strain",
        )  # EpsYY
        pb_post_tt.bc.add(
            typeBC,
            [StrainNodes[0]],
            "DispZ",
            BC_perturb[i][2],
            start_value=0,
            name="_Strain",
        )  # EpsZZ
        pb_post_tt.bc.add(
            typeBC,
            [StrainNodes[1]],
            "DispX",
            BC_perturb[i][3],
            start_value=0,
            name="_Strain",
        )  # EpsXY
        pb_post_tt.bc.add(
            typeBC,
            [StrainNodes[1]],
            "DispY",
            BC_perturb[i][4],
            start_value=0,
            name="_Strain",
        )  # EpsXZ
        pb_post_tt.bc.add(
            typeBC,
            [StrainNodes[1]],
            "DispZ",
            BC_perturb[i][5],
            start_value=0,
            name="_Strain",
        )  # EpsYZ

        pb_post_tt.apply_boundary_conditions()

        pb_post_tt.solve()

        X = pb_post_tt.get_X()  # alias

        if typeBC == "Neumann":
            DStrain.append(
                np.array(
                    [
                        pb_post_tt._get_vect_component(X, "DispX")[StrainNodes[0]],
                        pb_post_tt._get_vect_component(X, "DispY")[StrainNodes[0]],
                        pb_post_tt._get_vect_component(X, "DispZ")[StrainNodes[0]],
                        pb_post_tt._get_vect_component(X, "DispX")[StrainNodes[1]],
                        pb_post_tt._get_vect_component(X, "DispY")[StrainNodes[1]],
                        pb_post_tt._get_vect_component(X, "DispZ")[StrainNodes[1]],
                    ]
                )
            )
        else:
            F = pb_post_tt.get_ext_forces()
            F = F.reshape(3, -1)
            stress = [F[0, -2], F[1, -2], F[2, -2], F[0, -1], F[1, -1], F[2, -1]]

            DStress.append(stress)

    if typeBC == "Neumann":
        C = np.linalg.inv(np.array(DStrain).T)
    else:
        C = np.array(DStress).T

    return C


def get_tangent_stiffness(pb=None, meshperio=True, **kargs):
    #################### PERTURBATION METHODE #############################
    solver = kargs.get("solver", "direct")

    if pb is None:
        pb = ProblemBase.get_active()
    elif isinstance(pb, str):
        pb = ProblemBase.get_all()[pb]
    mesh = pb.mesh

    if "_StrainNodes" in mesh.node_sets:
        crd = mesh.nodes[:-2]
    else:
        crd = mesh.nodes

    xmax = np.max(crd[:, 0])
    xmin = np.min(crd[:, 0])
    ymax = np.max(crd[:, 1])
    ymin = np.min(crd[:, 1])
    zmax = np.max(crd[:, 2])
    zmin = np.min(crd[:, 2])
    crd_center = (np.array([xmin, ymin, zmin]) + np.array([xmax, ymax, zmax])) / 2
    center = [np.linalg.norm(crd - crd_center, axis=1).argmin()]

    BC_perturb = np.eye(6)
    # BC_perturb[3:6,3:6] *= 2 #2xEXY

    DStrain = []
    DStress = []

    if "_StrainNodes" in mesh.node_sets:
        StrainNodes = mesh.node_sets["_StrainNodes"]
        remove_strain = False
        A = pb.get_A()
    else:
        StrainNodes = mesh.add_nodes(
            crd_center, 2
        )  # add virtual nodes for macro strain
        mesh.add_node_set(StrainNodes, "_StrainNodes")
        remove_strain = True
        A = pb.get_A().copy()
        A.resize(np.array(pb.get_A().shape) + 6)
    # StrainNodes=[len(crd),len(crd)+1] #last 2 nodes

    if "_perturbation" in pb.get_all() and Problem["_perturbation"].mesh is not mesh:
        # if required an option could be added to delete '_perturbation' in case the mesh may change
        print(
            'WARNING: delete old "_perturbation" problem that is related to another mesh'
        )
        del pb.get_all()["_perturbation"]

    if "_perturbation" not in pb.get_all():
        # initialize perturbation problem
        pb_post_tt = Problem(0, 0, 0, mesh, name="_perturbation")
        pb_post_tt.set_solver(solver)

        pb.make_active()

        # Shall add other conditions later on
        pb_post_tt.bc.add(
            PeriodicBC(
                [
                    StrainNodes[0],
                    StrainNodes[0],
                    StrainNodes[0],
                    StrainNodes[1],
                    StrainNodes[1],
                    StrainNodes[1],
                ],
                ["DispX", "DispY", "DispZ", "DispX", "DispY", "DispZ"],
                dim=3,
                meshperio=meshperio,
            )
        )

        pb_post_tt.bc.add("Dirichlet", center, "Disp", 0, name="center")
    else:
        pb_post_tt = Problem["_perturbation"]

    pb_post_tt.set_A(pb.get_A())

    # typeBC = 'Dirichlet' #doesn't work with meshperio = False
    typeBC = "Neumann"

    pb_post_tt.apply_boundary_conditions()

    for i in range(6):
        pb_post_tt.bc.add(
            typeBC,
            [StrainNodes[0]],
            "DispX",
            BC_perturb[i][0],
            start_value=0,
            name="_Strain",
        )  # EpsXX
        pb_post_tt.bc.add(
            typeBC,
            [StrainNodes[0]],
            "DispY",
            BC_perturb[i][1],
            start_value=0,
            name="_Strain",
        )  # EpsYY
        pb_post_tt.bc.add(
            typeBC,
            [StrainNodes[0]],
            "DispZ",
            BC_perturb[i][2],
            start_value=0,
            name="_Strain",
        )  # EpsZZ
        pb_post_tt.bc.add(
            typeBC,
            [StrainNodes[1]],
            "DispX",
            BC_perturb[i][3],
            start_value=0,
            name="_Strain",
        )  # EpsXY
        pb_post_tt.bc.add(
            typeBC,
            [StrainNodes[1]],
            "DispY",
            BC_perturb[i][4],
            start_value=0,
            name="_Strain",
        )  # EpsXZ
        pb_post_tt.bc.add(
            typeBC,
            [StrainNodes[1]],
            "DispZ",
            BC_perturb[i][5],
            start_value=0,
            name="_Strain",
        )  # EpsYZ

        pb_post_tt.apply_boundary_conditions()

        pb_post_tt.solve()
        X = pb_post_tt.get_X()  # alias
        if typeBC == "Neumann":
            DStrain.append(
                np.array(
                    [
                        pb_post_tt._get_vect_component(X, "DispX")[StrainNodes[0]],
                        pb_post_tt._get_vect_component(X, "DispY")[StrainNodes[0]],
                        pb_post_tt._get_vect_component(X, "DispZ")[StrainNodes[0]],
                        pb_post_tt._get_vect_component(X, "DispX")[StrainNodes[1]],
                        pb_post_tt._get_vect_component(X, "DispY")[StrainNodes[1]],
                        pb_post_tt._get_vect_component(X, "DispZ")[StrainNodes[1]],
                    ]
                )
            )
        else:
            F = pb_post_tt.get_ext_forces()
            F = F.reshape(3, -1)
            stress = [F[0, -2], F[1, -2], F[2, -2], F[0, -1], F[1, -1], F[2, -1]]

            DStress.append(stress)

        pb_post_tt.bc.remove("_Strain")

    volume = mesh.bounding_box.volume
    if typeBC == "Neumann":
        C = np.linalg.inv(np.array(DStrain).T) / volume
    else:
        C = np.array(DStress).T / volume

    if remove_strain:
        mesh.remove_nodes(StrainNodes)
        mesh.RemoveSetOfNodes("_StrainNodes")

    return C
