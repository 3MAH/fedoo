"""Natural frequencies and mode shapes of a cantilever."""

import fedoo as fd


fd.ModelingSpace("2Dplane")

mesh = fd.mesh.rectangle_mesh(
    nx=21,
    ny=5,
    x_min=0.0,
    x_max=1.0,
    y_min=-0.05,
    y_max=0.05,
    elm_type="quad4",
)

steel = fd.constitutivelaw.ElasticIsotrop(210e9, 0.3)
steel.set_density(7800.0)
weakform = fd.weakform.StressEquilibrium(steel)
assembly = fd.Assembly.create(weakform, mesh)

modal = fd.problem.Modal(assembly)
clamped = mesh.find_nodes("X", mesh.bounding_box.xmin)
modal.bc.add("Dirichlet", clamped, "Disp", 0.0)
results = modal.add_output(
    "cantilever_modes.fdh5", assembly, ["Disp", "Strain", "Stress"]
)
modal.solve(n_modes=6, sigma=0.0)

print("Natural frequencies [Hz]:")
for mode, frequency in enumerate(modal.frequencies, start=1):
    print(f"  mode {mode}: {frequency:.6g}")

# The selected mode is exposed through the usual problem result API.
modal.set_mode(0, scale=0.02)
first_mode = modal.get_results(assembly, ["Disp", "Strain"])

# Open the multi-frame result with fd.viewer(results). Each frame is one mode.
