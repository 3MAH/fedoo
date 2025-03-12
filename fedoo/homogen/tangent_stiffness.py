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
    crd_center = mesh.bounding_box.center
    center = [np.linalg.norm(crd - crd_center, axis=1).argmin()]

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

    if "_StrainNodes" in mesh.node_sets:
        crd = mesh.nodes[:-2]
    else:
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

        raise NotImplementedError(
            "A bug has been identified in this function.\
                                  Contact a developer if you need it."
        )
        # bug to solve: the resize is not sufficient. It affect the node numbering
        A.resize(np.array(pb.get_A().shape) + 2 * pb.space.nvar)
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
        pb_post_tt.set_solver(solver, solver_type=solver_type, pc_type=pc_type)

        pb.make_active()

        # Shall add other conditions later on
        if ndim == 3:
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
        else:
            pb_post_tt.bc.add(
                PeriodicBC(
                    [
                        StrainNodes[0],
                        StrainNodes[0],
                        StrainNodes[1],
                    ],
                    ["DispX", "DispY", "DispX"],
                    dim=2,
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

    # typeBC = 'Dirichlet' #doesn't work with meshperio = False
    typeBC = "Neumann"

    if pb.space.nvar > ndim:
        # if rot dof
        pb_post_tt.bc.add(
            "Dirichlet",
            [StrainNodes[0], StrainNodes[1]],
            [
                var
                for var in pb_post_tt.space.list_variables()
                if var not in ["DispX", "DispY", "DispZ"]
            ],
            0,
        )
    if ndim == 2:
        pb_post_tt.bc.add("Dirichlet", [StrainNodes[1]], "DispY", 0)

    for i in range(len(BC_perturb)):
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
        if ndim == 3:
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
        else:
            pb_post_tt.bc.add(
                typeBC,
                [StrainNodes[1]],
                "DispX",
                BC_perturb[i][2],
                start_value=0,
                name="_Strain",
            )  # EpsXY

        pb_post_tt.apply_boundary_conditions()

        pb_post_tt.solve()
        X = pb_post_tt.get_X()  # alias
        if typeBC == "Neumann":
            if ndim == 3:
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
            else:  # ndim == 2
                DStrain.append(
                    np.array(
                        [
                            pb_post_tt._get_vect_component(X, "DispX")[StrainNodes[0]],
                            pb_post_tt._get_vect_component(X, "DispY")[StrainNodes[0]],
                            pb_post_tt._get_vect_component(X, "DispX")[StrainNodes[1]],
                        ]
                    )
                )
        else:
            F = pb_post_tt.get_ext_forces()
            F = F.reshape(ndim, -1)
            if ndim == 3:
                stress = [F[0, -2], F[1, -2], F[2, -2], F[0, -1], F[1, -1], F[2, -1]]
            else:
                stress = [F[0, -2], F[1, -2], F[0, -1]]

            DStress.append(stress)

        pb_post_tt.bc.remove("_Strain")

    volume = mesh.bounding_box.volume
    if typeBC == "Neumann":
        C = np.linalg.inv(np.array(DStrain).T) / volume
    else:
        C = np.array(DStress).T / volume

    if remove_strain:
        mesh.remove_nodes(StrainNodes)
        del mesh.node_sets["_StrainNodes"]

    return C
