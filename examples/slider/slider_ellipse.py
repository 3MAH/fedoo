import fedoo as fd
import numpy as np
import pyvista as pv

XMAX = 100
YMAX = 10
FORCE = 500


def compute_mechanical_fields(
    mesh: fd.Mesh,
    field_name: str,
    young_modulus: float = 1e5,
    poisson_ratio: float = 0.3,
) -> np.ndarray:
    fd.Assembly.delete_memory()
    # --------------- Pre-Treatment --------------------------------------------------------
    space = fd.ModelingSpace("2Dstress")

    type_el = mesh.elm_type

    material = fd.constitutivelaw.ElasticIsotrop(young_modulus, poisson_ratio)

    # Assembly
    wf = fd.weakform.StressEquilibrium(material)

    assemb = fd.Assembly.create(wf, mesh, type_el)

    # Type of problem
    pb = fd.problem.Linear(assemb)

    pb.bc.add("Dirichlet", "bottom", "Disp", 0)
    pb.bc.add("Dirichlet", "left", "DispX", 0)
    pb.bc.add("Dirichlet", "right", "DispX", 0)
    pb.bc.add("Dirichlet", "top", "Disp", [0, -1, 0])

    pb.apply_boundary_conditions()

    pb.solve()
    i = 0

    # res = pb.get_results(assemb, ["Disp", "Stress", "Strain"])

    displacement = pb.get_results(assemb, "Disp", "Node").get_data("Disp").T

    if field_name == "DispY":
        field = displacement[:, 1]
    elif field_name == "Von-Mises":
        field = pb.get_results(assemb, "Stress", "Node").get_data(
            "Stress", component="vm"
        )
    elif field_name == "Stress_XX":
        field = pb.get_results(assemb, "Stress", "Node").get_data(
            "Stress", component="XX"
        )
    elif field_name == "Stress_YY":
        field = pb.get_results(assemb, "Stress", "Node").get_data(
            "Stress", component="YY"
        )
    elif field_name == "Stress_XY":
        field = pb.get_results(assemb, "Stress", "Node").get_data(
            "Stress", component="XY"
        )
    elif field_name == "Div_Stress_norm":
        Stress_node = pb.get_results(assemb, "Stress", "Node")["Stress"]
        div_sigma = np.c_[
            assemb.get_node_results(space.op_div_u(), Stress_node[[0, 3]].reshape(-1)),
            assemb.get_node_results(space.op_div_u(), Stress_node[[3, 1]].reshape(-1)),
        ]
        field = np.linalg.norm(div_sigma, axis=1)
    return displacement, field


def mesh_updater(
    nx: int, a: float, b: float, elm_type: str, field_name: str = "Von-Mises"
) -> pv.PolyData:
    mesh = fd.mesh.hole_plate_mesh(
        nx,
        nx,
        100,
        100,
        (a, b),
        elm_type,
        False,
    )
    # mesh = fd.mesh.rectangle_mesh(nx, ny, 0, 100, 0, 10, elm_type)

    (
        displacement,
        field,
    ) = compute_mechanical_fields(mesh, field_name)
    mesh = mesh.to_pyvista()
    # mesh.point_data.set_scalars(stress_x, "stress_x")
    # mesh.point_data.set_scalars(stress_y, "stress_y")
    # mesh.point_data.set_scalars(stress_field_per_node.T[:, 2], "stress_xy")
    mesh.point_data.set_scalars(field, "field")
    mesh.points[:, :-1] += displacement
    return mesh


pl = pv.Plotter()


class StressRoutine:
    def __init__(self, mesh: pv.PolyData):
        self.output = mesh  # Expected PyVista mesh type
        # default parameters
        self.kwargs = {
            "nx": 21,
            # "ny": 11,
            "a": 30,
            "b": 10,
            "elm_type": "quad4",
        }
        self.quadratic = 0
        self.field_names = [
            "Von-Mises",
            "DispY",
            "Stress_XX",
            "Stress_YY",
            "Stress_XY",
            "Div_Stress_norm",
        ]
        self.id_field = 0

    def __call__(self, param, value):
        if param == "elm_type":
            if value:
                pl.actors["elm_type"].input = "tri"
                if self.quadratic:
                    self.kwargs["elm_type"] = "tri6"
                else:
                    self.kwargs["elm_type"] = "tri3"
            else:
                pl.actors["elm_type"].input = "quad"
                if self.quadratic:
                    self.kwargs["elm_type"] = "quad9"
                else:
                    self.kwargs["elm_type"] = "quad4"
        elif param == "show_edges":
            pl.actors["mesh"].GetProperty().show_edges = value
        elif param == "quadratic":
            self.quadratic = value
            if value:
                pl.actors["quadratic"].input = "quadratic"
                if self.kwargs["elm_type"][:3] == "tri":
                    self.kwargs["elm_type"] = "tri6"
                else:
                    self.kwargs["elm_type"] = "quad9"
            else:
                pl.actors["quadratic"].input = "linear"
                if self.kwargs["elm_type"][:3] == "tri":
                    self.kwargs["elm_type"] = "tri3"
                else:
                    self.kwargs["elm_type"] = "quad4"
        elif param == "field":
            self.id_field += 1
            if self.id_field == len(self.field_names):
                self.id_field = 0
            pl.actors["field"].input = self.field_names[self.id_field]
        else:
            self.kwargs[param] = value
        self.update()

    def update(self):
        # This is where you call your simulation
        self.kwargs["field_name"] = self.field_names[self.id_field]
        result = mesh_updater(**self.kwargs)
        result.ComputeBounds()
        center = result.center
        pl.camera.SetFocalPoint(center)
        pl.camera.position = tuple(center + np.array([0, 0, 2 * result.length]))
        pl.camera.up = tuple([0, 1, 0])
        scalar_field = result.point_data["field"]
        pl.update_scalar_bar_range([scalar_field.min(), scalar_field.max()])
        self.output.copy_from(result)
        return


start_mesh = mesh_updater(21, 30, 10, "quad4", "Von-Mises")
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
    scalars="field",
    scalar_bar_args=sargs,
    name="mesh",
)
# pl.set_background('White')

pl.add_slider_widget(
    callback=lambda value: engine("nx", int(value)),
    rng=[2, 50],
    value=21,
    title="Number of nodes",
    pointa=(0.025, 0.9),
    pointb=(0.14, 0.9),
    style="modern",
    fmt="%0.0f",
    interaction_event="always",
)

# pl.add_slider_widget(
#     callback=lambda value: engine("ny", int(value)),
#     rng=[2,30],
#     value=5,
#     title="Number of y-nodes",
#     pointa=(0.28, 0.9),
#     pointb=(0.42, 0.9),
#     style="modern",
#     fmt="%0.0f",
#     interaction_event="always",
# )

pl.add_slider_widget(
    callback=lambda value: engine("a", value),
    rng=[1, 40],
    value=5,
    title="x radius",
    pointa=(0.28, 0.9),
    pointb=(0.42, 0.9),
    style="modern",
    fmt="%0.0f",
    interaction_event="always",
)

pl.add_slider_widget(
    callback=lambda value: engine("b", value),
    rng=[1, 40],
    value=5,
    title="y radius",
    pointa=(0.57, 0.9),
    pointb=(0.71, 0.9),
    style="modern",
    fmt="%0.0f",
    interaction_event="always",
)

pl.add_checkbox_button_widget(
    callback=lambda value: engine("field", value),
    value=False,
    position=(10, 130),
    color_on="grey",
    size=40,
)


pl.add_text(
    "Von_Mises", (70, 140), font_size=10, name="field"
)  # use viewport = True for relative coordinates


pl.add_checkbox_button_widget(
    callback=lambda value: engine("elm_type", value),
    value=False,
    position=(10, 90),
    color_on="grey",
    size=40,
)

pl.add_text(
    "quad", (70, 100), font_size=10, name="elm_type"
)  # use viewport = True for relative coordinates

pl.add_checkbox_button_widget(
    callback=lambda value: engine("quadratic", value),
    value=False,
    position=(10, 50),
    color_on="grey",
    size=40,
)

pl.add_text(
    "linear", (70, 60), font_size=10, name="quadratic"
)  # use viewport = True for relative coordinates


pl.add_checkbox_button_widget(
    callback=lambda value: engine("show_edges", value),
    value=True,
    position=(10, 10),
    size=40,
)

pl.add_text(
    "show_edges", (70, 20), font_size=10
)  # use viewport = True for relative coordinates

try:
    pl.add_logo_widget("fedOOLogos.png")
except:
    pass

# pl.add_slider_widget(
#     callback=lambda value: engine("strain_xy", value),
#     rng=STRAIN_RANGES,
#     value=0.025,
#     title="Mean Strain XY",
#     pointa=(0.57, 0.9),
#     pointb=(0.71, 0.9),
#     style="modern",
#     interaction_event="always",
# )

# pl.add_slider_widget(
#     callback=lambda value: engine("hole_radius", value),
#     rng=[5, 45],
#     value=20,
#     title="Hole Radius",
#     pointa=(0.85, 0.9),
#     pointb=(0.98, 0.9),
#     style="modern",
#     interaction_event="always",
# )


pl.show()
