import numpy as np
import pytest

import fedoo as fd


def _solve_periodic(use_lm, suffix):
    fd.ModelingSpace("2Dstress")
    mesh = fd.mesh.hole_plate_mesh(name=f"periodic_mesh_{suffix}")
    material = fd.constitutivelaw.ElasticIsotrop(
        1e5, 0.3, name=f"periodic_material_{suffix}"
    )
    weakform = fd.weakform.StressEquilibrium(material, name=f"periodic_wf_{suffix}")
    assembly = fd.Assembly.create(weakform, mesh, name=f"periodic_assembly_{suffix}")
    periodic = fd.constraint.PeriodicBC(periodicity_type="small_strain")
    if use_lm:
        assembly += fd.LagrangeMultiplierAssembly(
            mesh, periodic, name=f"PeriodicLM_{suffix}"
        )
    pb = fd.problem.Linear(assembly, name=f"periodic_problem_{suffix}")
    pb.set_solver("direct_scipy")
    if not use_lm:
        pb.bc.add(periodic)
    pb.bc.add("Dirichlet", ["E_xx", "E_xy", "E_yy"], [0.0, 0.1, 0.0])
    center = mesh.nearest_node(mesh.bounding_box.center)
    pb.bc.add("Dirichlet", center, "Disp", 0.0)
    pb.solve()
    return pb.get_dof_solution("Disp"), pb.get_ext_forces("MeanStrain")


def _make_nonlinear_model(dimension, suffix, mesh):
    fd.ModelingSpace(dimension)
    material = fd.constitutivelaw.ElasticIsotrop(2e5, 0.3, name=f"material_{suffix}")
    weakform = fd.weakform.StressEquilibrium(material, nlgeom=True, name=f"wf_{suffix}")
    assembly = fd.Assembly.create(weakform, mesh, name=f"assembly_{suffix}")
    return assembly


def _solve_rigid_tie_2d(use_lm, suffix):
    fd.ModelingSpace("2D")
    mesh = fd.mesh.rectangle_mesh(name=f"rigid2d_mesh_{suffix}")
    assembly = _make_nonlinear_model("2D", suffix, mesh)
    bottom = mesh.find_nodes("Y", mesh.bounding_box.ymin)
    top = mesh.find_nodes("Y", mesh.bounding_box.ymax)
    constraint = fd.constraint.RigidTie2D(top)
    if use_lm:
        assembly += fd.LagrangeMultiplierAssembly(
            mesh, constraint, name=f"RigidTie2DLM_{suffix}"
        )
    pb = fd.problem.NonLinear(assembly, name=f"problem_{suffix}")
    pb.set_solver("direct_scipy")
    pb.set_nr_criterion("Displacement", err0=1, tol=1e-5, max_subiter=40)
    if not use_lm:
        pb.bc.add(constraint)
    pb.bc.add("Dirichlet", bottom, "Disp", 0.0)
    pb.bc.add("Dirichlet", "RigidRotZ", -np.pi / 4)
    pb.nlsolve(dt=0.2, tmax=1.0, update_dt=False, print_info=0)
    return pb.get_dof_solution("Disp"), pb.get_ext_forces("RigidRotZ")


def _solve_face_constraint(kind, use_lm, suffix):
    fd.ModelingSpace("3D")
    mesh = fd.mesh.box_mesh(
        nx=3,
        ny=3,
        nz=3,
        x_min=0.0,
        x_max=2.0,
        y_min=-1.0,
        y_max=1.0,
        z_min=-1.0,
        z_max=1.0,
        elm_type="hex8",
        name=f"face_mesh_{suffix}",
    )
    material = fd.constitutivelaw.ElasticIsotrop(
        2e5, 0.3, name=f"face_material_{suffix}"
    )
    weakform = fd.weakform.StressEquilibrium(
        material, nlgeom="UL", name=f"face_wf_{suffix}"
    )
    assembly = fd.Assembly.create(
        weakform, mesh, "hex8", name=f"face_assembly_{suffix}"
    )
    left = mesh.find_nodes("X", mesh.bounding_box.xmin)
    right = mesh.find_nodes("X", mesh.bounding_box.xmax)
    if kind == "rigid":
        constraint = fd.constraint.RigidTie(right)
        driver = "RigidRotX"
    else:
        surface = fd.mesh.extract_surface(mesh, node_set=right, reduce_order=False)
        constraint = fd.constraint.MeanMotion(surface, components=["Rot"])
        driver = "MeanRotX"
    if use_lm:
        assembly += fd.LagrangeMultiplierAssembly(
            mesh, constraint, name=f"FaceLM_{suffix}"
        )
    pb = fd.problem.NonLinear(assembly, nlgeom="UL", name=f"face_problem_{suffix}")
    pb.set_solver("direct_scipy")
    pb.set_nr_criterion("Displacement", tol=1e-3, max_subiter=20)
    if not use_lm:
        pb.bc.add(constraint)
    pb.bc.add("Dirichlet", left, "Disp", 0.0)
    pb.bc.add("Dirichlet", driver, np.pi / 3)
    pb.nlsolve(dt=0.1, tmax=1.0, update_dt=False, print_info=0)
    return pb.get_dof_solution("Disp"), pb.get_ext_forces(driver)


@pytest.mark.parametrize(
    "solver,case",
    [
        (_solve_periodic, "periodic"),
        (_solve_rigid_tie_2d, "rigid2d"),
    ],
)
def test_example_constraint_matches_lagrange_multiplier(solver, case):
    displacement_mpc, reaction_mpc = solver(False, f"{case}_mpc")
    displacement_lm, reaction_lm = solver(True, f"{case}_lm")

    relative_displacement_error = np.linalg.norm(
        displacement_lm - displacement_mpc
    ) / np.linalg.norm(displacement_mpc)
    assert relative_displacement_error < 2e-2, relative_displacement_error
    reaction_scale = max(np.max(np.abs(reaction_mpc)), 1.0)
    assert np.allclose(
        reaction_lm, reaction_mpc, rtol=2e-5, atol=2e-10 * reaction_scale
    )


@pytest.mark.parametrize("kind", ["rigid", "mean"])
def test_torsion_face_constraint_matches_lagrange_multiplier(kind):
    displacement_mpc, reaction_mpc = _solve_face_constraint(kind, False, f"{kind}_mpc")
    displacement_lm, reaction_lm = _solve_face_constraint(kind, True, f"{kind}_lm")

    relative_displacement_error = np.linalg.norm(
        displacement_lm - displacement_mpc
    ) / np.linalg.norm(displacement_mpc)
    assert relative_displacement_error < 1e-2, relative_displacement_error
    reaction_error = np.linalg.norm(reaction_lm - reaction_mpc) / max(
        np.linalg.norm(reaction_mpc), 1.0
    )
    assert reaction_error < 1e-2, reaction_error


if __name__ == "__main__":
    pytest.main([__file__])
