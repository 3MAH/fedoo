#
# Plate element to model the canteleaver beam using different kind of plate elements
#

import numpy as np
import pytest
import fedoo as fd

# Define the combinations to test: (geom_type, element_formulation, test_nlgeom, ref_sol)
# This includes Full Integration (p...), Selective Reduced Integration (...sri), and MITC (...mitc)
TEST_CONFIGURATIONS = [
    ("quad4", "pquad4", False, -7.687578514010123),
    (
        "quad4",
        "pquad4sri",
        True,
        -18.779474568978365,
    ),  # Testing nlgeom=True on this one for performance
    ("quad4", "pquad4mitc", False, -19.619460304260063),
    ("quad8", "pquad8", False, -19.629203972874315),
    ("quad8", "pquad8mitc", False, -19.630934258132694),
    ("quad9", "pquad9", False, -19.61106447027928),
    ("quad9", "pquad9mitc", False, -19.625844872041974),
    ("tri3", "ptri3", False, -8.490748579541856),
    ("tri3", "ptri3sri", False, -14.09480247634044),
    ("tri3", "ptri3mitc", False, -19.602557388258848),
    ("tri6", "ptri6", False, -19.580589719182846),
    ("tri6", "ptri6mitc", False, -19.58058970493779),
]

# Angles to test: 0 (flat), and a arbitrary 3D rotation to test unaligned space
ROTATION_ANGLES = [
    (0, 0, 0),  # Perfectly aligned with XY plane
    (30, 45, 60),  # Rotated in 3D space
]


def get_rotation_matrix(alpha_deg, beta_deg, gamma_deg):
    """Generates a 3D rotation matrix from intrinsic Euler angles (in degrees)."""
    alpha = np.radians(alpha_deg)
    beta = np.radians(beta_deg)
    gamma = np.radians(gamma_deg)

    # Rotation around Z, then Y, then X
    R_z = np.array(
        [
            [np.cos(alpha), -np.sin(alpha), 0],
            [np.sin(alpha), np.cos(alpha), 0],
            [0, 0, 1],
        ]
    )
    R_y = np.array(
        [[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]]
    )
    R_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(gamma), -np.sin(gamma)],
            [0, np.sin(gamma), np.cos(gamma)],
        ]
    )

    return R_x @ R_y @ R_z


@pytest.mark.parametrize(
    "geom_elm_type, fedoo_elm_type, nlgeom, ref_sol", TEST_CONFIGURATIONS
)
@pytest.mark.parametrize("angles", ROTATION_ANGLES)
def test_plate_element(geom_elm_type, fedoo_elm_type, nlgeom, ref_sol, angles):
    fd.Assembly.delete_memory()
    fd.ModelingSpace("3D")

    E = 1e5
    nu = 0.3
    L = 100
    h = 20
    thickness = 1
    F_magnitude = -10

    # 1. Material and Section Setup
    fd.constitutivelaw.ElasticIsotrop(E, nu, name="Material")
    fd.constitutivelaw.ShellHomogeneous("Material", thickness, k=1, name="PlateSection")

    # 2. Mesh Generation
    mesh = fd.mesh.rectangle_mesh(
        51, 11, 0, L, -h / 2, h / 2, geom_elm_type, ndim=3, name="plate"
    )

    # 3. Rotate Mesh to test unaligned space
    R = get_rotation_matrix(*angles)
    # Update the physical positions of the nodes in the mesh
    mesh.nodes[:, :3] = (R @ mesh.nodes[:, :3].T).T

    # 4. Identify Node Sets
    nodes_left = mesh.node_sets["left"]
    nodes_right = mesh.node_sets["right"]

    # To find the center node on the right, we project back to original coordinates
    # or use distance calculation from the expected centerline.
    orig_coords = (np.linalg.inv(R) @ mesh.nodes[nodes_right, :3].T).T
    node_right_center = nodes_right[(orig_coords[:, 1] ** 2).argmin()]

    # 5. Weakform and Assembly
    fd.weakform.PlateEquilibrium("PlateSection", name="WFplate")
    fd.Assembly.create("WFplate", "plate", fedoo_elm_type, name="plate")

    # 6. Problem Definition (Switching to NonLinear if nlgeom=True)
    if nlgeom:
        # Assuming fedoo uses LargeDisplacement or similar for nlgeom,
        # modify this line to match your specific fedoo non-linear problem syntax if different.
        pb = fd.problem.NonLinear("plate", nlgeom=True)
    else:
        pb = fd.problem.Linear("plate")

    # 7. Boundary Conditions
    # For a rotated plate, standard global DispX/DispY/DispZ will over-constrain or misalign
    # if we don't project them, OR we can define them in local coordinates if Fedoo supports it.
    # Assuming standard Dirichlet clamped end:
    pb.bc.add("Dirichlet", nodes_left, ["Disp", "Rot"], 0)

    # Rotate the Force Vector to align with the transformed plate's "Z" (thickness) direction
    # Local force was [0, 0, F_magnitude]
    local_force = np.array([0, 0, F_magnitude])
    global_force = R @ local_force

    pb.bc.add("Neumann", node_right_center, "DispX", global_force[0])
    pb.bc.add("Neumann", node_right_center, "DispY", global_force[1])
    pb.bc.add("Neumann", node_right_center, "DispZ", global_force[2])

    # 8. Solve
    if nlgeom:
        pb.nlsolve(dt=0.25)
    else:
        pb.solve()

    # 9. Validation
    # Extract global displacements and project back to local Z to verify deflection
    disp_global = np.array(
        [
            pb.get_disp("DispX")[node_right_center],
            pb.get_disp("DispY")[node_right_center],
            pb.get_disp("DispZ")[node_right_center],
        ]
    )

    disp_local = np.linalg.inv(R) @ disp_global
    local_disp_z = disp_local[2]

    if nlgeom:
        assert np.abs(local_disp_z - ref_sol) < 1e-2
    else:
        assert np.abs(local_disp_z - ref_sol) < 1e-7


if __name__ == "__main__":
    pytest.main([__file__])
