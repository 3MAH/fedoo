"""Not intended for public use, excepted to derive new mechanical constitutivelaw"""

# baseclass
import numpy as np
from simcoon import Rotation
from fedoo.core.base import ConstitutiveLaw


class Mechanical3D(ConstitutiveLaw):
    """Base class for mechanical constitutive laws.

    Strain / stress 6-vector slot ordering
    --------------------------------------
    Strains and stresses are stored as 6-vectors (Voigt-like) at every Gauss
    point. The slot interpretation depends on the active ``ModelingSpace``:

    =========  ===================  ===========================  ===================
    slot       3D                   2Dplane (plane strain)       2Daxi
    =========  ===================  ===========================  ===================
    0          ε_xx                 ε_xx                         ε_rr
    1          ε_yy                 ε_yy                         ε_zz   (z = Y axis)
    2          ε_zz                 0  (unused)                  ε_θθ   (= u_r / R)
    3          γ_xy                 γ_xy                         γ_rz
    4          γ_xz                 0                            0  (γ_rθ = 0 by sym.)
    5          γ_yz                 0                            0  (γ_θz = 0 by sym.)
    =========  ===================  ===========================  ===================

    For ``2Dstress`` (plane stress), the slot layout is the same as
    ``2Dplane``, but the constitutive law internally computes a nonzero
    out-of-plane strain ε_zz so that σ_zz = 0 — the symbolic strain
    operator places 0 in slot 2, but post-processed strain output from
    a constitutive law that relaxes σ_zz may have a nonzero ε_zz that
    is not reflected in slot 2.

    Rationale for the 2Daxi mapping
    -------------------------------
    In axisymmetric kinematics, γ_rθ = γ_θz = 0 by symmetry, so γ_rz is the
    only nonzero shear — exactly as γ_xy is the only nonzero shear in 2D
    plane strain / plane stress. Placing γ_rz in slot 3 keeps slot 3 = "the
    in-plane shear" across all 2D regimes, so element B-matrices, the
    assembly pipeline, and most constitutive-law code are dimension-agnostic
    in their handling of slot 3. The 3D-Voigt slot 2 (normally ε_zz in
    Cartesian) is repurposed to carry ε_θθ; the physical ε_zz of 2Daxi
    lives in slot 1 because z is along the Y axis (z ≡ Y, r ≡ X).

    Convention
    ----------
    For a 2Daxi mesh, ``mesh.nodes[:, 0]`` is the radial coordinate r and
    ``mesh.nodes[:, 1]`` is the axial coordinate z. The symmetry axis is the
    Y axis in 2D; after ``axisymmetric_extrusion`` the symmetry axis becomes
    Z in the resulting 3D mesh.

    Implication for orthotropic / anisotropic constitutive laws
    -----------------------------------------------------------
    The 6×6 tangent matrix produced by a constitutive law is consumed by
    fedoo with the slot mapping above. For a law whose material directions
    are labelled (1, 2, 3): in 2Daxi, axis 1 is r, axis 2 is z, **axis 3 is
    the hoop direction**. Authors of orthotropic / anisotropic laws should
    declare moduli with this convention in mind; otherwise the user-supplied
    "direction 3" stiffness will be silently applied to the hoop response.

    Theory of axisymmetric kinematics
    ---------------------------------
    Label the reference configuration in cylindrical coordinates
    ``(R, Θ, Z)`` and the current configuration ``(r, θ, z)``. An
    axisymmetric motion *without twist* satisfies::

        r = r(R, Z),   z = z(R, Z),   θ = Θ.

    In the orthonormal cylindrical bases ``(e_R, e_Θ, e_Z)`` and
    ``(e_r, e_θ, e_z)`` the deformation gradient takes the block form

    .. math::

        F = \\frac{\\partial r}{\\partial R}\\, e_r \\!\\otimes\\! e_R
          + \\frac{\\partial r}{\\partial Z}\\, e_r \\!\\otimes\\! e_Z
          + \\frac{\\partial z}{\\partial R}\\, e_z \\!\\otimes\\! e_R
          + \\frac{\\partial z}{\\partial Z}\\, e_z \\!\\otimes\\! e_Z
          + \\frac{r}{R}\\, e_θ \\!\\otimes\\! e_Θ.

    Two consequences are load-bearing for the implementation:

    * **Block structure.** All hoop-coupling components vanish:
      ``F_rΘ = F_zΘ = F_θR = F_θZ = 0``. Therefore γ_rθ = γ_θz = 0 in
      every kinematic measure, which is why slot 3 (γ_rz) is the only
      shear component in 2Daxi (slots 4 and 5 are identically zero).

    * **Hoop deformation gradient.** The hoop stretch is
      ``λ_θ = F_θΘ = r/R = 1 + u_r/R`` (where ``u_r = r − R`` is the
      *total* radial displacement and ``R`` is the *reference* radial
      coordinate). At small strain ``r ≈ R`` and this reduces to the
      conventional ``ε_θθ = u_r/r``, which is what fedoo's
      ``_comp_linear_strain`` and the small-strain weakforms compute via
      the symbolic operator ``space.variable("DispX") * (1/r_gp)``. At
      finite strain the two definitions differ; fedoo's UL + 2Daxi
      pipeline therefore divides ``u_r`` by the *reference* radius
      ``R = mesh.nodes[:, 0]`` (captured at problem ``initialize`` as
      ``assembly.sv["_R0_gausspoints"]``), not by ``r_current``.

    See Bonet & Wood, *Nonlinear Continuum Mechanics for Finite Element
    Analysis* (2008), Box 8.3; Holzapfel, *Nonlinear Solid Mechanics*
    (2000), §2.5; Belytschko, Liu & Moran, *Nonlinear Finite Elements
    for Continua and Structures* (2014), §4.5.
    """

    # model of constitutive law for InternalForce Weakform

    _corotational_box_tangent = False
    # True when the law returns the simcoon corotational "box" tangent
    # d(tau_hat)/dD, which the UL weakform must convert to the Lie
    # (Truesdell) spatial tangent (see StressEquilibrium.update_2).
    # Native fedoo laws (e.g. ElasticIsotrop) return a plain engineering
    # tangent and are left unconverted.

    def __init__(self, name="", density=None):
        ConstitutiveLaw.__init__(self, name)
        self.density = density
        self._Lt_from_F = False
        # _Lt_from_F attribute is True if the tangent matrix is related
        # to F instead of log epsilonn, ie for hyper elastic materials

    def set_density(self, density):
        """Set the mass density associated with this mechanical material."""
        self.density = density
        return self

    # def initialize(self, assembly, pb):
    #     pass
    # #function called to initialize the constutive law
    # assembly.sv['Strain'] = 0
    # assembly.sv['Stress'] = 0
    # assembly.sv['DispGradient'] = 0
    # assembly.sv['TangentMatrix'] = self.get_tangent_matrix(assembly)

    # def update(self, assembly, pb):
    #     pass
    # function called to update the state of constitutive law
    # assembly.sv['TangentMatrix'] = self.get_tangent_matrix(assembly)

    # def get_tangent_matrix(self, assembly, dimension=None): #Tangent Matrix in lobal coordinate system (no change of basis)
    #     return NotImplemented

    # def get_H(self, assembly, dimension = None): #Tangent Matrix in global coordinate system (apply change of basis) + account for dimension of the problem
    #     if dimension is None: dimension = assembly.space.get_dimension()
    #     if dimension == "2Dstress":
    #         H = self.get_tangent_matrix_2Dstress()
    #         if H is NotImplemented:
    #             H = self.local2global_H(self.get_tangent_matrix())
    #             return self.get_H_plane_stress(H)
    #         else:
    #             return self.local2global_H(H)

    #     return self.local2global_H(self.get_tangent_matrix())

    def get_H_plane_stress(self, H):
        """
        Convert a full 3D tangent matrix H in an equivalent behavior in 2D with the plane stress assumption.

        Parameters
        ----------
        H : TYPE
            Full 3D tangent matrix

        Returns
        -------
        H_plane_stress

        """
        return [
            [
                H[i][j] - H[i][2] * H[j][2] / H[2][2] if j in [0, 1, 3] else 0
                for j in range(6)
            ]
            if i in [0, 1, 3]
            else [0, 0, 0, 0, 0, 0]
            for i in range(6)
        ]

    def local2global_H(self, H):
        """Rotate stiffness matrix from local material frame to global frame.

        Uses simcoon.Rotation to build the 6x6 Voigt stress rotation matrix
        QS from the local frame, then computes H_global = QS @ H @ QS^T.
        """
        if self.local_frame is not None:
            local_frame = np.asarray(self.local_frame)
            if local_frame.ndim == 2:
                local_frame = local_frame[np.newaxis]
            rot = Rotation.from_matrix(local_frame)
            QS = rot.as_voigt_stress_rotation()  # (N, 6, 6)

            H = np.asarray(H, dtype=float)
            if H.ndim == 3:
                H = np.rollaxis(H, 2, 0)  # (6,6,M) -> (M,6,6)
            H = np.matmul(QS, np.matmul(H, QS.transpose(0, 2, 1)))
            if H.ndim == 3:
                if H.shape[0] == 1:
                    return H[0]
                return np.rollaxis(H, 0, 3)  # (N,6,6) -> (6,6,N)
            return H

        return H
