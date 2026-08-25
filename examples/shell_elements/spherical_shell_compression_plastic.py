"""Small elastoplastic compression test of a ping-pong ball.

This reduced example exercises ``ShellHomogeneousNonLinear`` with the
Simcoon ``EPICP`` plane-stress constitutive law. The shell kinematics are
corotational, while strains passed to the material points remain small.
"""

import numpy as np
import pyvista as pv

import fedoo as fd


def run_simulation(
    mesh_resolution=8,
    pressure=2.0,
    n_steps=2,
    print_info=1,
):
    """Run a deliberately small pressure-compression simulation."""
    young_modulus = 2_000.0  # MPa
    poisson_ratio = 0.37
    yield_stress = 35.0  # MPa
    hardening_modulus = 100.0  # MPa
    hardening_exponent = 0.3
    radius = 20.0  # mm
    thickness = 0.45  # mm

    fd.ModelingSpace("3D")
    sphere = pv.Sphere(
        radius=radius,
        theta_resolution=mesh_resolution,
        phi_resolution=mesh_resolution,
    )
    mesh = fd.Mesh.from_pyvista(sphere)

    properties = np.array(
        [
            young_modulus,
            poisson_ratio,
            0.0,
            yield_stress,
            hardening_modulus,
            hardening_exponent,
        ]
    )
    material = fd.constitutivelaw.Simcoon("EPICP", properties)
    material.tangent_mode = 1
    shell = fd.constitutivelaw.ShellHomogeneousNonLinear(
        material,
        thickness,
        n_thickness_points=3,
        k=5 / 6,
    )

    weakform = fd.weakform.PlateEquilibrium(shell, nlgeom=True)
    assembly = fd.Assembly.create(weakform, mesh)

    loaded_elements = mesh.find_elements(
        f"Z>{mesh.bounding_box.zmax-3} or " f"Z<{mesh.bounding_box.zmin+3}"
    )
    pressure_load = fd.constraint.Pressure(
        mesh.extract_elements(loaded_elements),
        pressure,
    )

    problem = fd.problem.NonLinear(assembly, nlgeom=True)
    problem.bc.add(pressure_load)

    nodes = mesh.nodes
    node_a = int(np.argmin(nodes[:, 0]))
    node_b = int(np.argmax(nodes[:, 0]))
    node_c = int(np.argmax(nodes[:, 1]))
    problem.bc.add("Dirichlet", node_a, "Disp", 0)
    problem.bc.add("Dirichlet", node_b, ["DispY", "DispZ"], 0)
    problem.bc.add("Dirichlet", node_c, "DispZ", 0)
    problem.set_nr_criterion(
        "Displacement",
        tol=5e-3,
        max_subiter=15,
        adaptive_stiffness=True,
    )
    problem.nlsolve(
        dt=1 / n_steps,
        tmax=1,
        update_dt=False,
        print_info=print_info,
    )
    return problem, assembly, shell


if __name__ == "__main__":
    problem, assembly, shell = run_simulation()
    displacement = problem.get_disp()
    deformed = assembly.mesh.to_pyvista()
    deformed.points += displacement.T
    deformed.plot(show_edges=True)
