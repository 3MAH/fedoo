import fedoo as fd
import numpy as np
import pyvista as pv

STRAIN_RANGES = [-0.5, 0.5]


def hole_plate_mesh(
    n_nodes_x: int = 21,
    n_nodes_y: int = 21,
    plate_width: float = 100,
    plate_height: float = 100,
    hole_radius: float = 20,
) -> fd.Mesh:
    assert plate_width == plate_height, "Plate height and width must be the same"
    mesh = fd.mesh.hole_plate_mesh(
        (n_nodes_x // 2) + 1,
        (n_nodes_y // 2) + 1,
        length=plate_width,
        height=plate_height,
        radius=hole_radius,
        elm_type="quad4",
        sym=False,
        ndim=None,
    )
    return mesh


def compute_mechanical_fields(
    mesh: fd.Mesh,
    eps_xx: float,
    eps_yy: float,
    gamma_xy: float,
    young_modulus: float = 1e5,
    poisson_ratio: float = 0.3,
) -> np.ndarray:
    fd.Assembly.delete_memory()
    # --------------- Pre-Treatment -------------------------------------------
    space = fd.ModelingSpace("2Dstress")

    type_el = mesh.elm_type
    center = mesh.nearest_node(mesh.bounding_box.center)

    strain_nodes = mesh.add_virtual_nodes(2)

    material = fd.constitutivelaw.ElasticIsotrop(young_modulus, poisson_ratio)

    # Assembly
    wf = fd.weakform.StressEquilibrium(material)

    assemb = fd.Assembly.create(wf, mesh, type_el)

    # Type of problem
    pb = fd.problem.Linear(assemb)

    # Shall add other conditions later on
    bc_periodic = fd.constraint.PeriodicBC(
        [strain_nodes[0], strain_nodes[1], strain_nodes[0]],
        ["DispX", "DispY", "DispY"],
        dim=2,
    )
    pb.bc.add(bc_periodic)

    pb.bc.add("Dirichlet", strain_nodes[1], "DispX", 0)
    pb.bc.add("Dirichlet", center, "Disp", 0, name="center")

    pb.apply_boundary_conditions()

    pb.bc.remove("_Strain")
    pb.bc.add(
        "Dirichlet",
        [strain_nodes[0]],
        "DispX",
        eps_xx,
        start_value=0,
        name="_Strain",
    )  # EpsXX
    pb.bc.add(
        "Dirichlet",
        [strain_nodes[1]],
        "DispY",
        eps_yy,
        start_value=0,
        name="_Strain",
    )  # EpsYY
    pb.bc.add(
        "Dirichlet",
        [strain_nodes[0]],
        "DispY",
        gamma_xy,
        start_value=0,
        name="_Strain",
    )  # 2EpsXY

    pb.apply_boundary_conditions()

    pb.solve()
    i = 0

    res = pb.get_results(assemb, ["Disp", "Stress", "Strain"])
    # mean_stress

    stress_field_per_node = pb.get_results(assemb, "Stress", "Node")["Stress"]
    strain_field_per_node = pb.get_results(assemb, "Strain", "Node")["Strain"]
    # Filter XX, YY, XY
    xx_yy_xy_indices = [0, 1, 3]
    stress_field_per_node = stress_field_per_node[xx_yy_xy_indices]
    strain_field_per_node = strain_field_per_node[xx_yy_xy_indices]

    volume = mesh.bounding_box.volume
    # Compute mean stress
    """
    mean_stress = [
        mesh.integrate_field(res["Stress", i], type_field="GaussPoint") / volume
        for i in xx_yy_xy_indices
    ]
    """
    # Debug
    # for component in ["XX", "YY", "XY", "vm"]:
    #    pb.get_results(assemb, "Stress", "Node").plot(
    #        "Stress", component=component
    #    )
    # Remove virtual nodes added by fedoo
    displacement = pb.get_results(assemb, "Disp", "Node").get_data("Disp").T
    von_mises_stress = pb.get_results(assemb, "Stress", "Node").get_data(
        "Stress", component="vm"
    )
    return stress_field_per_node, displacement, von_mises_stress


def mesh_updater(
    strain_xx: float, strain_yy: float, strain_xy: float, hole_radius: float
) -> pv.PolyData:
    mesh = hole_plate_mesh(hole_radius=hole_radius)
    (
        stress_field_per_node,
        displacement,
        von_mises_stress,
    ) = compute_mechanical_fields(mesh, strain_xx, strain_yy, strain_xy)
    mesh = mesh.to_pyvista()
    # mesh.point_data.set_scalars(stress_x, "stress_x")
    # mesh.point_data.set_scalars(stress_y, "stress_y")
    # mesh.point_data.set_scalars(stress_field_per_node.T[:, 2], "stress_xy")
    mesh.point_data.set_scalars(von_mises_stress, "von_mises")
    mesh.points[:, :-1] += displacement
    return mesh


pl = pv.Plotter()


class StressRoutine:
    def __init__(self, mesh: pv.PolyData):
        self.output = mesh  # Expected PyVista mesh type
        # default parameters
        self.kwargs = {
            "strain_xx": 0.025,
            "strain_yy": 0.025,
            "strain_xy": 0.025,
            "hole_radius": 20,
        }

    def __call__(self, param, value):
        self.kwargs[param] = value
        self.update()

    def update(self):
        # This is where you call your simulation
        result = mesh_updater(**self.kwargs)
        result.ComputeBounds()
        center = result.center
        pl.camera.SetFocalPoint(center)
        pl.camera.position = tuple(center + np.array([0, 0, 2 * result.length]))
        pl.camera.up = tuple([0, 1, 0])
        scalar_field = result.point_data["von_mises"]
        pl.update_scalar_bar_range([scalar_field.min(), scalar_field.max()])
        self.output.copy_from(result)
        return


start_mesh = mesh_updater(0.03, -0.025, 0.025, 20)
engine = StressRoutine(start_mesh)

start_mesh.ComputeBounds()
center = start_mesh.center
pl.camera.SetFocalPoint(center)
pl.camera.position = tuple(center + np.array([0, 0, 2 * start_mesh.length]))
pl.camera.up = tuple([0, 1, 0])
pl.set_background("Gray")
sargs = dict(color="white")
pl.add_mesh(
    start_mesh,
    show_edges=True,
    cmap="jet",
    scalars="von_mises",
    scalar_bar_args=sargs,
)


pl.add_slider_widget(
    callback=lambda value: engine("strain_xx", value),
    rng=STRAIN_RANGES,
    value=0.025,
    title="Mean Strain X",
    pointa=(0.025, 0.9),
    pointb=(0.14, 0.9),
    style="modern",
    interaction_event="always",
)


pl.add_slider_widget(
    callback=lambda value: engine("strain_yy", value),
    rng=STRAIN_RANGES,
    value=0.025,
    title="Mean Strain Y",
    pointa=(0.28, 0.9),
    pointb=(0.42, 0.9),
    style="modern",
    interaction_event="always",
)

pl.add_slider_widget(
    callback=lambda value: engine("strain_xy", value),
    rng=STRAIN_RANGES,
    value=0.025,
    title="Mean Strain XY",
    pointa=(0.57, 0.9),
    pointb=(0.71, 0.9),
    style="modern",
    interaction_event="always",
)

pl.add_slider_widget(
    callback=lambda value: engine("hole_radius", value),
    rng=[5, 45],
    value=20,
    title="Hole Radius",
    pointa=(0.85, 0.9),
    pointb=(0.98, 0.9),
    style="modern",
    interaction_event="always",
)


pl.show()
