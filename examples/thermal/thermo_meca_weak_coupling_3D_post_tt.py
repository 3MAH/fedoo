"""Create a movie from the weak thermo-mechanical coupling results.

Run ``thermo_meca_weak_coupling_3D.py`` first to generate the thermal and
mechanical result files used by this post-processing script.
"""

from pathlib import Path

import fedoo as fd
import pyvista as pv


def write_coupled_movie(
    result_prefix=None,
    output_filename=None,
    n_frames=None,
):
    """Plot temperature on the amplified deformed configuration."""
    example_dir = Path(__file__).resolve().parent
    if result_prefix is None:
        result_prefix = example_dir / "results" / "thermo_meca_nl"
    else:
        result_prefix = Path(result_prefix)

    if output_filename is None:
        output_filename = result_prefix.with_suffix(".mp4")
    else:
        output_filename = Path(output_filename)

    thermal_results = fd.read_data(f"{result_prefix}_th.fdh5")
    mechanical_results = fd.read_data(f"{result_prefix}_me.fdh5")

    if thermal_results.mesh.n_nodes != mechanical_results.mesh.n_nodes:
        raise ValueError("Thermal and mechanical result meshes do not match.")

    available_frames = min(thermal_results.n_iter, mechanical_results.n_iter)
    if n_frames is None:
        n_frames = available_frames
    else:
        n_frames = min(n_frames, available_frames)
    if n_frames < 1:
        raise ValueError("The result files do not contain any frame.")

    plotter = pv.Plotter(window_size=(1024, 768), off_screen=True)
    plotter.set_background("white")
    plotter.open_movie(str(output_filename), framerate=24, quality=4)

    scalar_bar_args = {
        "title_font_size": 20,
        "label_font_size": 16,
        "color": "black",
    }

    for iteration in range(n_frames):
        thermal_results.load(iteration)
        mechanical_results.load(iteration)

        # DataSet.plot uses the nodal displacement field to update the displayed
        # coordinates when scale is non-zero.
        thermal_results.node_data["Disp"] = mechanical_results.node_data["Disp"]
        thermal_results.plot(
            "Temp",
            data_type="Node",
            scale=5,
            show=False,
            show_edges=True,
            clim=(0, 100),
            scalar_bar_args=scalar_bar_args,
            title="",
            name="thermo-mechanical",
            plotter=plotter,
            lock_view=iteration > 0,
        )

        if iteration == 0:
            plotter.camera.SetFocalPoint(thermal_results.mesh.bounding_box.center)
            plotter.camera.position = (
                -2.090457552750125,
                1.7582929402632352,
                1.707926514944027,
            )

        plotter.camera.Azimuth(360 / n_frames)
        plotter.write_frame()

    plotter.close()
    return output_filename


if __name__ == "__main__":
    write_coupled_movie()
