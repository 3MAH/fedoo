"""Taylor-Hood (LBB-stable) poromechanics elements via fedoo CombinedElement.

fedoo supports per-variable interpolation order, so an inf-sup-stable Taylor-Hood
pair (quadratic displacement, linear pore pressure) is built without new
machinery:

    elm = fd.lib_elements.element_list.CombinedElement("hex20lbb", "hex20")
    elm.set_variable_interpolation("PorePressure", "hex8")
    fd.Assembly.create(wf, mesh_hex20, elm_type="hex20lbb")

This guards the Taylor-Hood poro path: it must converge and reproduce the exact
undrained Biot coupling magnitude p0 = -alpha*M*tr(eps) at the gauss points.

(The qualitative inf-sup *benefit* — equal-order Q2-Q2 gives a singular saddle
while Q2-Q1 is well posed — is shown in examples/poromechanics/taylor_hood_lbb.py;
it deliberately triggers a singular factorization and so is not a pytest case.)
"""

import numpy as np

import fedoo as fd


def test_taylor_hood_poro_undrained_magnitude():
    """hex20/hex8 Taylor-Hood poro: undrained p = -alpha*M*tr(eps), exactly."""
    E, nu, M, alpha, dz, L = 1.0e6, 0.3, 1.0e8, 1.0, -1.0e-4, 1.0

    fd.ModelingSpace("3D")
    mesh = fd.mesh.box_mesh(
        nx=3,
        ny=3,
        nz=5,
        x_min=0,
        x_max=0.2,
        y_min=0,
        y_max=0.2,
        z_min=0,
        z_max=L,
        elm_type="hex20",
        name="ColTH",
    )
    skel = fd.constitutivelaw.ElasticIsotrop(E, nu, name="SkTH")
    fluid = fd.constitutivelaw.PoroFluidProperties(
        permeability=1.0e-7,
        fluid_viscosity=1.0,
        biot_coefficient=alpha,
        biot_modulus=M,
        initial_porosity=0.5,
        name="FlTH",
    )
    wf = fd.weakform.poro_mechanics_simple(skel, fluid, nlgeom=False, name="PTH")

    # Taylor-Hood: quadratic displacement (hex20), linear PorePressure (hex8)
    elm = fd.lib_elements.element_list.CombinedElement("hex20lbb", "hex20")
    elm.set_variable_interpolation("PorePressure", "hex8")
    asm = fd.Assembly.create(wf, mesh, elm_type="hex20lbb", name="ATHPoro")

    pb = fd.problem.NonLinear("ATHPoro")
    pb.set_nr_criterion("Displacement", tol=1e-3, max_subiter=25)
    sx = np.unique(
        np.concatenate([mesh.find_nodes("X", 0.0), mesh.find_nodes("X", 0.2)])
    )
    sy = np.unique(
        np.concatenate([mesh.find_nodes("Y", 0.0), mesh.find_nodes("Y", 0.2)])
    )
    pb.bc.add("Dirichlet", sx, "DispX", 0.0)
    pb.bc.add("Dirichlet", sy, "DispY", 0.0)
    pb.bc.add("Dirichlet", mesh.find_nodes("Z", 0.0), "DispZ", 0.0)
    pb.bc.add(
        "Dirichlet",
        mesh.find_nodes("Z", L),
        "DispZ",
        dz,
        time_func=lambda tf: min(1.0, tf * 1e9),
    )

    # tiny dt -> negligible diffusion; fully sealed (no PorePressure BC) -> undrained
    dt = 1.0e-3
    pb.initialize()
    pb.tmax = 3 * dt
    pb.dtime = dt
    pb.set_start()
    for s in range(3):
        pb.time = s * dt
        conv, nb, res = pb.solve_time_increment()
        assert conv, f"Taylor-Hood poro did not converge (res={res:g})"
        pb.set_start()

    # physical pressure lives at the gauss points (the hex8 pressure does not
    # carry the hex20 mid-side dofs, so raw nodal averaging is meaningless)
    p_gp = np.asarray(
        asm.get_gp_results(pb.space.variable("PorePressure"), pb.get_dof_solution())
    ).ravel()
    p0_expected = -alpha * M * (dz / L)  # = +1e4 Pa, uniform
    assert np.allclose(p_gp, p0_expected, rtol=0.02), (
        f"undrained gauss-point pressure mean={p_gp.mean():g} std={p_gp.std():g} "
        f"!= -alpha*M*tr(eps) = {p0_expected:g}"
    )


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
