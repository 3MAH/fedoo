"""Phase-field fracture constitutive law.

Wraps an elastic constitutive law and applies phase-field damage degradation.
Supports AT1 and AT2 regularizations with Bourdin, Amor, or Miehe energy splits.
"""

from fedoo.core.mechanical3d import Mechanical3D
from fedoo.util.voigt_tensors import StressTensorList, StrainTensorList

import numpy as np


class PhaseFieldDamage(Mechanical3D):
    """Phase-field damage constitutive law.

    Wraps an elastic constitutive law and applies degradation g(d) = (1-d)² + k_res
    to the tensile part of the stiffness, following the Francfort-Marigo variational
    approach to fracture.

    Parameters
    ----------
    elastic_cl : ConstitutiveLaw
        Base elastic constitutive law (e.g. ElasticIsotrop).
        Must provide E, nu attributes and get_tangent_matrix method.
    Gc : float
        Critical energy release rate (J/m²).
    l0 : float
        Regularization length (controls diffuse crack width).
    split : str, optional
        Energy split type: 'bourdin' (no split), 'amor' (volumetric-deviatoric),
        or 'miehe' (spectral). Default is 'bourdin'.
    model : str, optional
        Phase-field model: 'AT1' or 'AT2'. Default is 'AT2'.
    k_res : float, optional
        Residual stiffness to avoid singular matrix when d=1. Default is 1e-10.
    name : str, optional
        Name of the constitutive law.
    """

    def __init__(
        self, elastic_cl, Gc, l0, split="bourdin", model="AT2", k_res=1e-10, name=""
    ):
        Mechanical3D.__init__(self, name)

        self.elastic_cl = elastic_cl
        self.Gc = Gc
        self.l0 = l0
        self.split = split.lower()
        self.model = model.upper()
        self.k_res = k_res

        if self.split not in ("bourdin", "amor", "miehe"):
            raise ValueError(
                f"Unknown split '{split}'. Use 'bourdin', 'amor', or 'miehe'."
            )
        if self.model not in ("AT1", "AT2"):
            raise ValueError(f"Unknown model '{model}'. Use 'AT1' or 'AT2'.")

        # History variable H = max_{s<=t} psi_plus(eps(x,s))
        self._H = None  # current history (updated during time step)
        self._H_start = None  # history at start of increment (for to_start)

    def initialize(self, assembly, pb):
        n_gp = assembly.n_gauss_points
        self._H = np.zeros(n_gp)
        self._H_start = np.zeros(n_gp)
        assembly.sv["PhaseFieldHistory"] = self._H
        # Initialize damage to zero if not present
        if "Damage" not in assembly.sv:
            assembly.sv["Damage"] = np.zeros(n_gp)
        # Compute and store elastic tangent matrix
        self._H_elastic = self.elastic_cl.get_tangent_matrix(assembly)
        assembly.sv["TangentMatrix"] = self._H_elastic

    def update(self, assembly, pb):
        strain = assembly.sv["Strain"]
        if np.isscalar(strain) and strain == 0:
            assembly.sv["TangentMatrix"] = self._H_elastic
            assembly.sv["Stress"] = 0
            return

        # Get damage at gauss points
        d = assembly.sv.get("Damage", 0)
        if np.isscalar(d):
            d = np.zeros(assembly.n_gauss_points)

        # Degradation function g(d) = (1-d)^2 + k_res
        g = (1 - d) ** 2 + self.k_res

        # Compute energy split
        E = self.elastic_cl.E
        nu = self.elastic_cl.nu
        dimension = assembly.space.get_dimension()

        if "DStrain" in assembly.sv:
            total_strain = strain + assembly.sv["DStrain"]
        else:
            total_strain = strain

        # Convert strain to numpy array for computations
        eps = np.array(
            [
                total_strain[i]
                if not np.isscalar(total_strain[i])
                else np.full(assembly.n_gauss_points, total_strain[i])
                for i in range(6)
            ]
        )

        if self.split == "bourdin":
            psi_plus, stress_plus, stress_minus, H_plus, H_minus = _split_bourdin(
                E, nu, eps, dimension
            )
        elif self.split == "amor":
            psi_plus, stress_plus, stress_minus, H_plus, H_minus = _split_amor(
                E, nu, eps, dimension
            )
        elif self.split == "miehe":
            psi_plus, stress_plus, stress_minus, H_plus, H_minus = _split_miehe(
                E, nu, eps, dimension
            )

        # Update history variable: H = max(H_old, psi_plus)
        self._H = np.maximum(self._H, psi_plus)
        assembly.sv["PhaseFieldHistory"] = self._H

        # Degraded stress: sigma = g(d) * sigma_plus + sigma_minus
        stress = StressTensorList(
            [g * stress_plus[i] + stress_minus[i] for i in range(6)]
        )
        assembly.sv["Stress"] = stress

        # Degraded tangent matrix: H_deg = g(d) * H_plus + H_minus
        assembly.sv["TangentMatrix"] = [
            [g * H_plus[i][j] + H_minus[i][j] for j in range(6)] for i in range(6)
        ]

    def set_start(self, assembly, pb):
        """Called at the start of a new time increment. Lock history variable."""
        if self._H is not None:
            self._H_start = self._H.copy()

    def to_start(self, assembly, pb):
        """Restart the current time increment."""
        if self._H_start is not None:
            self._H = self._H_start.copy()
            assembly.sv["PhaseFieldHistory"] = self._H

    def reset(self):
        self._H = None
        self._H_start = None

    # ---- Properties for the damage weak form ----

    @property
    def diffusion_coeff(self):
        """Diffusion coefficient k for the damage equation: k * |nabla d|^2."""
        if self.model == "AT1":
            return 0.75 * self.Gc * self.l0
        else:  # AT2
            return self.Gc * self.l0

    def get_reaction_coeff(self, H):
        """Reaction coefficient r(H) for the damage equation: r * d * delta_d.

        Parameters
        ----------
        H : ndarray
            History variable (max tensile energy density).
        """
        if self.model == "AT1":
            return 2.0 * H
        else:  # AT2
            return 2.0 * H + self.Gc / self.l0

    def get_source_coeff(self, H):
        """Source coefficient f(H) for the damage equation RHS: f * delta_d.

        Parameters
        ----------
        H : ndarray
            History variable (max tensile energy density).
        """
        if self.model == "AT1":
            return np.maximum(0.0, 2.0 * H - 3.0 * self.Gc / (8.0 * self.l0))
        else:  # AT2
            return 2.0 * H


# =====================================================================
# Energy split implementations
# =====================================================================


def _get_lame(E, nu):
    """Compute Lame parameters from E, nu."""
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    return lam, mu


def _get_bulk_shear(E, nu):
    """Compute bulk modulus and shear modulus."""
    K = E / (3 * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    return K, mu


def _split_bourdin(E, nu, eps, dimension):
    """No energy split — all elastic energy degrades.

    Returns psi_plus, stress_plus, stress_minus, H_plus, H_minus.
    """
    lam, mu = _get_lame(E, nu)

    # Full stress (Voigt: eps[3:] are engineering shear, sigma[3:] are shear)
    trace_eps = eps[0] + eps[1] + eps[2]
    stress_plus = np.zeros_like(eps)
    for i in range(3):
        stress_plus[i] = lam * trace_eps + 2 * mu * eps[i]
    for i in range(3, 6):
        stress_plus[i] = mu * eps[i]  # sigma_ij = mu * gamma_ij (engineering shear)

    stress_minus = np.zeros_like(eps)

    # Elastic energy density psi = 0.5 * sigma : epsilon
    # With Voigt: psi = 0.5 * sum(sigma_i * eps_i) for i=0..2
    #           + 0.5 * sum(sigma_i * eps_i) for i=3..5
    # Note: eps[3:] are engineering shear = 2*eps_ij, sigma[3:] = mu*eps[3:]
    # so sigma_i * eps_i = mu * gamma^2 = 2 * mu * (2*eps_ij^2)
    psi_plus = 0.5 * (
        sum(stress_plus[i] * eps[i] for i in range(3))
        + sum(stress_plus[i] * eps[i] for i in range(3, 6))
    )

    # Tangent matrices (6x6) — full and zero
    H_full = _elastic_tangent_list(lam, mu, dimension)
    H_zero = [[0] * 6 for _ in range(6)]

    return psi_plus, list(stress_plus), list(stress_minus), H_full, H_zero


def _split_amor(E, nu, eps, dimension):
    """Amor (volumetric-deviatoric) energy split.

    Tensile part: positive volumetric + full deviatoric.
    Compressive part: negative volumetric only.
    """
    K, mu = _get_bulk_shear(E, nu)

    trace_eps = eps[0] + eps[1] + eps[2]
    trace_plus = np.maximum(trace_eps, 0)
    trace_minus = np.minimum(trace_eps, 0)

    # Deviatoric strain
    eps_dev = np.copy(eps)
    for i in range(3):
        eps_dev[i] = eps[i] - trace_eps / 3.0

    # Tensile stress: K * <tr(eps)>_+ * I + 2*mu*eps_dev
    stress_plus = np.zeros_like(eps)
    for i in range(3):
        stress_plus[i] = K * trace_plus + 2 * mu * eps_dev[i]
    for i in range(3, 6):
        stress_plus[i] = mu * eps_dev[i]  # deviatoric shear = full shear

    # Compressive stress: K * <tr(eps)>_- * I
    stress_minus = np.zeros_like(eps)
    for i in range(3):
        stress_minus[i] = K * trace_minus

    # Energy densities
    psi_plus = (
        0.5 * K * trace_plus**2
        + mu * sum(eps_dev[i] ** 2 for i in range(3))
        + 0.5 * mu * sum(eps_dev[i] ** 2 for i in range(3, 6))
    )

    # Tangent matrices
    H_plus = _amor_tangent_plus(K, mu, trace_eps, dimension)
    H_minus = _amor_tangent_minus(K, trace_eps, dimension)

    return psi_plus, list(stress_plus), list(stress_minus), H_plus, H_minus


def _split_miehe(E, nu, eps, dimension):
    """Miehe (spectral) energy split.

    Spectral decomposition of strain tensor. Only positive eigenvalues contribute
    to the tensile part.
    """
    lam, mu = _get_lame(E, nu)

    if np.isscalar(eps[0]):
        eps = np.array([np.atleast_1d(e) for e in eps])
        scalar_input = True
    else:
        scalar_input = False

    n_gp = eps.shape[1]

    # Build symmetric strain tensor for each GP
    # eps is in Voigt engineering notation: [e11, e22, e33, g12, g13, g23]
    # where g_ij = 2*eps_ij
    strain_tensors = np.zeros((n_gp, 3, 3))
    strain_tensors[:, 0, 0] = eps[0]
    strain_tensors[:, 1, 1] = eps[1]
    strain_tensors[:, 2, 2] = eps[2]
    strain_tensors[:, 0, 1] = strain_tensors[:, 1, 0] = 0.5 * eps[3]
    strain_tensors[:, 0, 2] = strain_tensors[:, 2, 0] = 0.5 * eps[4]
    strain_tensors[:, 1, 2] = strain_tensors[:, 2, 1] = 0.5 * eps[5]

    # Spectral decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(strain_tensors)
    # eigenvalues: (n_gp, 3), eigenvectors: (n_gp, 3, 3)

    eig_plus = np.maximum(eigenvalues, 0)
    eig_minus = np.minimum(eigenvalues, 0)

    trace_eps = eigenvalues.sum(axis=1)
    trace_plus = np.maximum(trace_eps, 0)
    trace_minus = np.minimum(trace_eps, 0)

    # Tensile/compressive energy densities
    psi_plus = 0.5 * lam * trace_plus**2 + mu * np.sum(eig_plus**2, axis=1)
    # psi_minus not needed for the algorithm but computed for completeness

    # Reconstruct stress tensors in principal frame then rotate back
    # sigma_plus_principal = lam * <tr(eps)>_+ * I + 2*mu * <eps>_+
    # sigma_minus_principal = lam * <tr(eps)>_- * I + 2*mu * <eps>_-
    sigma_plus_princ = np.zeros_like(eigenvalues)
    sigma_minus_princ = np.zeros_like(eigenvalues)
    for a in range(3):
        sigma_plus_princ[:, a] = lam * trace_plus + 2 * mu * eig_plus[:, a]
        sigma_minus_princ[:, a] = lam * trace_minus + 2 * mu * eig_minus[:, a]

    # Rotate back: sigma = Q * diag(sigma_princ) * Q^T
    stress_plus_tensor = np.einsum(
        "nia,na,nja->nij", eigenvectors, sigma_plus_princ, eigenvectors
    )
    stress_minus_tensor = np.einsum(
        "nia,na,nja->nij", eigenvectors, sigma_minus_princ, eigenvectors
    )

    # Convert to Voigt notation
    stress_plus = [
        stress_plus_tensor[:, 0, 0],
        stress_plus_tensor[:, 1, 1],
        stress_plus_tensor[:, 2, 2],
        stress_plus_tensor[:, 0, 1],
        stress_plus_tensor[:, 0, 2],
        stress_plus_tensor[:, 1, 2],
    ]
    stress_minus = [
        stress_minus_tensor[:, 0, 0],
        stress_minus_tensor[:, 1, 1],
        stress_minus_tensor[:, 2, 2],
        stress_minus_tensor[:, 0, 1],
        stress_minus_tensor[:, 0, 2],
        stress_minus_tensor[:, 1, 2],
    ]

    # Tangent matrices for Miehe split
    H_plus, H_minus = _miehe_tangent(lam, mu, eigenvalues, eigenvectors, dimension)

    if scalar_input:
        psi_plus = psi_plus[0]
        stress_plus = [s[0] for s in stress_plus]
        stress_minus = [s[0] for s in stress_minus]

    return psi_plus, stress_plus, stress_minus, H_plus, H_minus


# =====================================================================
# Tangent matrix helpers
# =====================================================================


def _elastic_tangent_list(lam, mu, dimension):
    """Return full isotropic elastic tangent as a list of lists."""
    H = [[0] * 6 for _ in range(6)]
    H[0][0] = H[1][1] = H[2][2] = lam + 2 * mu
    H[0][1] = H[0][2] = H[1][0] = H[1][2] = H[2][0] = H[2][1] = lam
    H[3][3] = H[4][4] = H[5][5] = mu
    return H


def _amor_tangent_plus(K, mu, trace_eps, dimension):
    """Tangent matrix for the tensile part of the Amor split.

    When trace(eps) > 0: H_plus = K * I_vol + 2*mu * I_dev
    When trace(eps) <= 0: H_plus = 2*mu * I_dev (volumetric part protected)
    """
    # Indicator: 1 where trace_eps > 0, 0 otherwise
    if np.isscalar(trace_eps):
        ind = 1.0 if trace_eps > 0 else 0.0
    else:
        ind = (trace_eps > 0).astype(float)

    K_eff = K * ind  # only contribute when trace > 0

    # H = (K_eff - 2mu/3) * I_x_I + 2*mu * I_sym
    H = [[0] * 6 for _ in range(6)]
    for i in range(3):
        for j in range(3):
            if i == j:
                H[i][j] = K_eff - 2 * mu / 3.0 + 2 * mu
            else:
                H[i][j] = K_eff - 2 * mu / 3.0
    # Note: K_eff - 2mu/3 = effective "lambda" for the tensile part
    H[3][3] = H[4][4] = H[5][5] = mu  # deviatoric shear always degrades
    return H


def _amor_tangent_minus(K, trace_eps, dimension):
    """Tangent matrix for the compressive part of the Amor split.

    Only the volumetric part when trace(eps) < 0.
    """
    if np.isscalar(trace_eps):
        ind = 1.0 if trace_eps <= 0 else 0.0
    else:
        ind = (trace_eps <= 0).astype(float)

    K_eff = K * ind
    H = [[0] * 6 for _ in range(6)]
    # H_minus = K_eff * I_x_I / 3  (purely volumetric)
    # Actually: sigma_minus = K * <tr(eps)>_- * I
    # d(sigma_minus)/d(eps) = K * H(<-tr(eps)>) * I x I / 1
    # In Voigt: d(sigma_minus_i)/d(eps_j) = K_eff for i,j in 0..2
    # But in Voigt notation with engineering shear, the identity is:
    # I_vol[i][j] = 1 for i,j < 3, else 0
    # Since sigma_i = K * tr(eps) and eps_j contribute as: d(tr)/d(eps_j)=1 for j<3
    for i in range(3):
        for j in range(3):
            H[i][j] = K_eff
    return H


def _miehe_tangent(lam, mu, eigenvalues, eigenvectors, dimension):
    """Compute tangent matrices for the Miehe spectral split.

    For simplicity, this returns approximate tangent matrices using
    the same formulas as the stress split (consistent with the split
    at the current strain state). This is sufficient for a staggered
    scheme where each sub-problem is linear.
    """
    n_gp = eigenvalues.shape[0]
    trace_eps = eigenvalues.sum(axis=1)
    trace_plus = np.maximum(trace_eps, 0)
    trace_minus = np.minimum(trace_eps, 0)

    eig_plus = np.maximum(eigenvalues, 0)
    eig_minus = np.minimum(eigenvalues, 0)

    # Build tangent in the rotated frame then transform back
    # For each GP, we need a 6x6 tangent matrix
    # This is expensive but done correctly via numerical perturbation
    # For practical purposes in a staggered scheme, we use a simplified
    # tangent based on the current eigenvalue signs.

    # Determine which eigenvalues are positive
    pos_mask = eigenvalues > 0  # (n_gp, 3)

    # For the "plus" tangent: only directions with positive eigenvalues contribute
    # with the full lam+2mu, while cross terms only get lam if both are positive.
    # For Miehe: H_plus = lam*<tr>_+' * IxI + 2*mu * P_+
    # where P_+ projects onto positive eigenvalue directions.
    # This is complex in general. For the staggered scheme, we use
    # the secant approach: return the full elastic tangent split by
    # the current eigenvalue state.

    # Simplified: use Amor-like split based on current eigenvalue signs
    # This is a common practical approximation
    H_elastic = _elastic_tangent_list(lam, mu, dimension)

    # For staggered scheme, the simplified tangent is acceptable
    # The damage equation is independent, and the mechanical equation
    # just needs a reasonable tangent for the linear solve.
    H_zero = [[0] * 6 for _ in range(6)]

    # Check if all eigenvalues are positive everywhere
    all_positive = np.all(pos_mask)
    all_negative = np.all(~pos_mask)

    if all_positive:
        return H_elastic, H_zero
    elif all_negative:
        return H_zero, H_elastic

    # Mixed case: need per-GP tangent
    # Build H_plus and H_minus as arrays
    H_plus_arr = np.zeros((6, 6, n_gp))
    H_minus_arr = np.zeros((6, 6, n_gp))

    # For each GP, construct the tangent in the rotated frame
    for gp in range(n_gp):
        Q = eigenvectors[gp]  # 3x3 rotation matrix
        eigs = eigenvalues[gp]

        # Build tangent in principal frame
        Hp = np.zeros((6, 6))
        Hm = np.zeros((6, 6))

        tr = eigs.sum()
        tr_p = max(tr, 0)
        tr_m = min(tr, 0)

        for a in range(3):
            for b in range(3):
                if eigs[a] >= 0 and eigs[b] >= 0:
                    val = lam * (1 if tr >= 0 else 0)
                    if a == b:
                        val += 2 * mu
                    Hp[a, b] = val
                elif eigs[a] < 0 and eigs[b] < 0:
                    val = lam * (1 if tr < 0 else 0)
                    if a == b:
                        val += 2 * mu
                    Hm[a, b] = val
                else:
                    # Cross terms: split based on trace sign
                    if tr >= 0:
                        Hp[a, b] = lam
                    else:
                        Hm[a, b] = lam

        # Shear terms in principal frame
        for a in range(3):
            b = (a + 1) % 3
            idx = a + 3  # Voigt shear index
            if eigs[a] >= 0 or eigs[b] >= 0:
                Hp[idx, idx] = mu
            else:
                Hm[idx, idx] = mu

        # Rotate back to global frame using Voigt rotation matrix
        R_eps = _voigt_rotation(Q)
        R_sig_inv = R_eps.T

        Hp_global = R_sig_inv @ Hp @ R_eps
        Hm_global = R_sig_inv @ Hm @ R_eps

        H_plus_arr[:, :, gp] = Hp_global
        H_minus_arr[:, :, gp] = Hm_global

    # Convert to list of lists format
    H_plus = [[H_plus_arr[i, j] for j in range(6)] for i in range(6)]
    H_minus = [[H_minus_arr[i, j] for j in range(6)] for i in range(6)]

    return H_plus, H_minus


def _voigt_rotation(Q):
    """Build the 6x6 Voigt rotation matrix for strain transformation.

    Given a 3x3 rotation matrix Q (columns = principal directions),
    returns R such that eps_global = R @ eps_principal (in Voigt notation).
    """
    R = np.zeros((6, 6))
    # Normal components
    for i in range(3):
        for j in range(3):
            R[i, j] = Q[i, j] ** 2
    # Normal-shear coupling
    for i in range(3):
        R[i, 3] = Q[i, 0] * Q[i, 1]
        R[i, 4] = Q[i, 0] * Q[i, 2]
        R[i, 5] = Q[i, 1] * Q[i, 2]
    # Shear-normal coupling
    for i in range(3):
        j = (i + 1) % 3
        R[i + 3, 0] = 2 * Q[i, 0] * Q[j, 0]
        R[i + 3, 1] = 2 * Q[i, 1] * Q[j, 1]
        R[i + 3, 2] = 2 * Q[i, 2] * Q[j, 2]
    # Shear-shear coupling (corrected for engineering shear)
    for i in range(3):
        j = (i + 1) % 3
        R[i + 3, 3] = Q[i, 0] * Q[j, 1] + Q[i, 1] * Q[j, 0]
        R[i + 3, 4] = Q[i, 0] * Q[j, 2] + Q[i, 2] * Q[j, 0]
        R[i + 3, 5] = Q[i, 1] * Q[j, 2] + Q[i, 2] * Q[j, 1]

    return R
