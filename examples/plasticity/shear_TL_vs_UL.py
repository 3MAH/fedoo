"""Plastic simple shear: total Lagrangian against updated Lagrangian.

Both run the same simcoon law (EPICP, J2 plasticity) on the logarithmic strain.

* nlgeom="UL", the rate formulation (default corate "log_r"): the strain is ln V, accumulated
  in the frame that rotates with the material, and the law works on the Kirchhoff stress.
* nlgeom="TL", the total formulation: the strain-space framework of Miehe, Apel and Lambrecht
  (2002). The strain is ln U, a state function of the deformation gradient on the reference
  frame, and the law works on the stress T conjugate to it. There is no objective rate.

The two coincide when the stress is coaxial with the stretch. In shear it is not: the yield
criterion holds on the Kirchhoff stress in one case and on T in the other, so the stresses
differ, and the difference does not vanish with the time step. The result fields
"LogStrainU", "LogStrainV" and "HenckyStress" give either run in the measures of the other.
"""

from time import perf_counter

import fedoo as fd
import numpy as np

E, nu, alpha = 200e3, 0.3, 1e-5
sigma_y, k, m = 300.0, 1000.0, 0.3
gamma = 0.5  # prescribed shear
n_inc = 20


def run(nlgeom, nx=8):
    fd.Assembly.delete_memory()
    fd.ModelingSpace("3D")
    mesh = fd.mesh.box_mesh(nx=nx, ny=nx, nz=nx, elm_type="hex8", name="Domain")
    material = fd.constitutivelaw.Simcoon(
        "EPICP", np.array([E, nu, alpha, sigma_y, k, m]), name="ConstitutiveLaw"
    )
    material.tangent_mode = 2  # algorithmic tangent
    wf = fd.weakform.StressEquilibrium(material, nlgeom=nlgeom)
    assembly = fd.Assembly.create(wf, mesh, name="Assembling")

    pb = fd.problem.NonLinear("Assembling")
    pb.set_nr_criterion("Displacement", err0=1.0, tol=1e-8, max_subiter=20)
    bottom = mesh.find_nodes("Y", mesh.bounding_box.ymin)
    top = mesh.find_nodes("Y", mesh.bounding_box.ymax)
    pb.bc.add("Dirichlet", bottom, "Disp", 0)
    pb.bc.add(
        "Dirichlet",
        top,
        "DispX",
        gamma * (mesh.bounding_box.ymax - mesh.bounding_box.ymin),
    )
    pb.bc.add("Dirichlet", top, "DispY", 0)
    pb.bc.add("Dirichlet", top, "DispZ", 0)

    start = perf_counter()
    pb.nlsolve(dt=1.0 / n_inc, tmax=1.0, update_dt=False, print_info=0)
    elapsed = perf_counter() - start
    stress = assembly.sv["Stress"].asarray()
    return wf.formulation, elapsed, stress.mean(axis=1)


if __name__ == "__main__":
    print(f"EPICP, simple shear gamma = {gamma}, {n_inc} increments")
    print("  nlgeom  formulation   time (s)   mean s_xx   mean s_yy   mean s_xy  (MPa)")
    for nlgeom in ("UL", "TL"):
        formulation, elapsed, s = run(nlgeom)
        print(
            f"  {nlgeom}      {formulation:11s}   {elapsed:8.2f}   {s[0]:9.2f}   {s[1]:9.2f}   {s[3]:9.2f}"
        )
