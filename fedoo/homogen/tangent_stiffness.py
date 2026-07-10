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
from fedoo.constraint.mean_value_constraint import MeanValueConstraint
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
    rigid_body_constraint = kargs.get("rigid_body_constraint", "pin")
    if rigid_body_constraint not in ("pin", "mean"):
        raise ValueError(
            'rigid_body_constraint should be "pin" (block the node nearest '
            'to the center) or "mean" (constrain the mean displacement '
            "with lagrange multipliers)."
        )
    is_direct_solver = (
        not isinstance(solver, str)
        or solver.lower() in {"direct", "direct_scipy", "pardiso", "mumps"}
        or (
            solver.lower() == "petsc"
            and solver_type is not None
            and solver_type.lower() == "preonly"
            and pc_type is not None
            and pc_type.lower() in {"lu", "cholesky"}
        )
    )
    if rigid_body_constraint == "mean" and not is_direct_solver:
        # the mean constraint borders the system with zero-diagonal lagrange
        # rows (saddle point): iterative solvers diverge / return NaN.
        raise ValueError(
            'rigid_body_constraint="mean" requires a direct solver (the '
            f"lagrange multipliers make the system indefinite); got solver={solver!r}."
        )

    if pb is None:
        pb = ProblemBase.get_active()
    elif isinstance(pb, str):
        pb = ProblemBase.get_all()[pb]
    mesh = pb.mesh

    ndim = pb.space.ndim

    if ndim == 3:
        BC_perturb = np.eye(6)
        # BC_perturb[3:6,3:6] *= 2 #2xEXY
    else:  # ndim == 2
        BC_perturb = np.eye(3)

    DStrain = []
    DStress = []

    if "_perturbation" in pb.get_all() and (
        Problem["_perturbation"].mesh is not mesh
        or getattr(Problem["_perturbation"], "_rigid_body_constraint", "pin")
        != rigid_body_constraint
    ):
        # if required an option could be added to delete '_perturbation' in case the mesh may change
        print(
            'WARNING: delete old "_perturbation" problem that is related to '
            "another mesh or rigid body constraint"
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
                dic_closest_points_on_boundaries=kargs.get(
                    "dic_closest_points_on_boundaries", None
                ),
            )
        )

        if rigid_body_constraint == "pin":
            # block the node nearest to the RVE center
            center = [
                np.linalg.norm(mesh.nodes - mesh.bounding_box.center, axis=1).argmin()
            ]
            pb_post_tt.bc.add(
                "Dirichlet",
                center,
                list(pb.space.list_variables()),
                0,
                name="center",
            )
        else:  # rigid_body_constraint == "mean"
            constraint = MeanValueConstraint(mesh, "Disp", name="_MeanDisp")
            constraint.initialize(pb_post_tt)
            pb_post_tt._mean_disp_constraint = constraint
        pb_post_tt._rigid_body_constraint = rigid_body_constraint
    else:
        pb_post_tt = Problem["_perturbation"]

    A = pb.get_A()
    if A is None and hasattr(pb, "assembly"):
        A = pb.assembly.get_global_matrix()
        pb.set_A(A)
    if rigid_body_constraint == "mean":
        # add the lagrange multiplier rows that enforce mean(disp) = 0
        A = A.copy()
        A.resize(pb_post_tt.n_dof, pb_post_tt.n_dof)
        A = A + pb_post_tt._mean_disp_constraint.get_global_matrix()
    pb_post_tt.set_A(A)
    pb.bc.remove("_Strain")

    # typeBC = 'Dirichlet' #doesn't work with meshperio = False
    typeBC = "Neumann"

    # Reuse factorization across perturbations: A and the constraint
    # reduction matrix _MatCB are constant in this loop (only Neumann BCs
    # on global E_xx/yy/.. dofs change, affecting only B). Avoids re-
    # factorizing on each perturbation when a direct backend (pypardiso,
    # python-mumps or petsc) is available.
    reuse_factor = False
    try:
        pb_post_tt.set_reuse_factorization(True)
        reuse_factor = True
    except RuntimeError:
        pass  # no direct backend available, fall back to per-iteration solves

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

    if reuse_factor:
        pb_post_tt.set_reuse_factorization(False)

    volume = mesh.bounding_box.volume
    if typeBC == "Neumann":
        C = np.linalg.inv(np.array(DStrain).T) / volume
    else:
        C = np.array(DStress).T / volume

    return C
