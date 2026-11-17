r"""
Plastic simple shear: total against updated Lagrangian
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both runs use the same Simcoon law (``EPICP``, J2 plasticity with isotropic
hardening) on a logarithmic strain.

* ``nlgeom="UL"``, rate formulation (default ``corate="log_r"``): the strain
  is :math:`\ln \mathbf{V}`, accumulated in the frame that rotates with the
  material, and the law works on the Kirchhoff stress.
* ``nlgeom="TL"``, total formulation (strain-space framework of Miehe, Apel
  and Lambrecht, 2002): the strain is :math:`\ln \mathbf{U}`, a state
  function of the deformation gradient in the reference frame, and the law
  works on the stress :math:`\mathbf{T}` conjugate to it. No objective rate
  is involved.

The two coincide when the stress is coaxial with the stretch. In simple shear
it is not: the yield criterion holds on the Kirchhoff stress in one case and on
:math:`\mathbf{T}` in the other, so the stresses differ, and the difference
does not vanish with the time step. The printed values are mean Cauchy
stresses. The result fields ``LogStrainU``, ``LogStrainV`` and ``LogStressU``
express either run in the measures of the other.
See :py:attr:`fedoo.weakform.StressEquilibrium.kinematic_form`.
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
    # 0 elastic, 1 continuum (fedoo default), 2 algorithmic (convex cutting
    # plane), 3 closest point projection (simcoon >= 2.2)
    material.tangent_mode = 2
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
    return wf.kinematic_form, elapsed, stress.mean(axis=1)


if __name__ == "__main__":
    print(f"EPICP, simple shear gamma = {gamma}, {n_inc} increments")
    print(
        "  nlgeom  kinematic form  time (s)   mean s_xx   mean s_yy   mean s_xy  (MPa)"
    )
    for nlgeom in ("UL", "TL"):
        kinematic_form, elapsed, s = run(nlgeom)
        print(
            f"  {nlgeom}      {kinematic_form:14s}   {elapsed:8.2f}"
            f"   {s[0]:9.2f}   {s[1]:9.2f}   {s[3]:9.2f}"
        )
