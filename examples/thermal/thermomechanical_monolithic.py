"""Transient heating and thermal expansion, solved together in small strain.

Run from the checkout with:
    python -m examples.thermal.thermomechanical_monolithic
    python -m examples.thermal.thermomechanical_monolithic --simcoon
    python -m examples.thermal.thermomechanical_monolithic --nlgeom TL
    python -m examples.thermal.thermomechanical_monolithic --nlgeom UL

SI units; 2D plane strain with unit out-of-plane thickness. The left edge is
clamped and kept at 293.15 K; the right edge ramps to 343.15 K in one second.
"""

import argparse
import numpy as np
import fedoo as fd


def run(use_simcoon=False, output=None, nlgeom=False):
    if use_simcoon and nlgeom:
        raise NotImplementedError(
            "The Simcoon umat_T adapter currently supports small strain."
        )
    fd.ModelingSpace("2Dplane")
    mesh = fd.mesh.rectangle_mesh(
        nx=21, ny=5, x_max=0.02, y_max=0.004, elm_type="quad4"
    )
    E, nu, alpha = 70e9, 0.3, 23e-6
    rho, cp, conductivity, T0 = 2700.0, 900.0, 180.0, 293.15
    if use_simcoon:
        material = fd.constitutivelaw.SimcoonThermoMechanical(
            "ELISO",
            [rho, cp, E, nu, alpha],
            conductivity,
            reference_temperature=T0,
        )
    else:
        material = fd.constitutivelaw.ElasticIsotrop(
            E,
            nu,
            thermal=fd.constitutivelaw.ThermalProperties(
                conductivity, cp, rho, alpha=alpha, reference_temperature=T0
            ),
        )
    weakform = fd.weakform.StressEquilibrium(
        material, nlgeom=nlgeom
    ) + fd.weakform.HeatEquation(material, nlgeom="TL" if nlgeom else False)
    assembly = fd.Assembly.create(weakform, mesh)
    pb = fd.problem.NonLinear(assembly)
    pb.set_time_integrator(fd.time.FIRST_ORDER, fd.time.BackwardEuler())
    pb.set_dof("Temp", T0)
    pb.set_nr_criterion(
        "Fields",
        fields={
            "Disp": {"atol": 1e-5, "rtol": 1e-8},
            "Temp": {"atol": 1e-6, "rtol": 1e-8},
        },
    )
    left = mesh.find_nodes("X", 0)
    right = mesh.find_nodes("X", 0.02)
    pb.bc.add("Dirichlet", left, "Disp", 0)
    pb.bc.add("Dirichlet", left, "Temp", T0, start_value=T0)
    pb.bc.add("Dirichlet", right, "Temp", T0 + 50, start_value=T0)
    if output is not None:
        pb.add_output(output, assembly, ["Disp", "Temp", "Stress"])
    pb.nlsolve(dt=0.1, tmax=1.0, update_dt=False, print_info=0)
    return pb, assembly


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--simcoon", action="store_true", help="Use Simcoon's thermal ELISO UMAT"
    )
    parser.add_argument("--output", help="Optional FDH5 output filename")
    parser.add_argument(
        "--nlgeom",
        choices=["TL", "UL"],
        default=False,
        help="Finite-strain native elasticity; heat always uses TL",
    )
    args = parser.parse_args()
    problem, _ = run(args.simcoon, args.output, args.nlgeom)
    print(
        f"Temperature: {np.min(problem.get_temp()):.3f} .. {np.max(problem.get_temp()):.3f} K"
    )
    print(f"Maximum displacement: {np.max(problem.get_disp('DispX')):.6g} m")
