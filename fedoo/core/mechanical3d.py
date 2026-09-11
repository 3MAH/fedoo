"""Not intended for public use, excepted to derive new mechanical constitutivelaw"""

# baseclass
import numpy as np
from simcoon import Rotation
from fedoo.core.base import ConstitutiveLaw
from fedoo.util.voigt_tensors import StressTensorList


class Mechanical3D(ConstitutiveLaw):
    """Base class for mechanical constitutive laws.

    Attributes
    ----------
    is_isotropic : bool
        Whether the constitutive response is invariant under material-frame
        rotations. Subclasses should set this to ``True`` only when rotating
        the constitutive operator cannot change the response.
    manages_material_frame : bool
        Whether the constitutive law transports its own material frame during
        finite-strain updates. When ``False``, Fedoo maintains the current
        orientation for anisotropic laws.

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

    # Conservative default: a newly derived law is assumed anisotropic until
    # it explicitly states otherwise. The flag only skips transformations of
    # invariant constitutive operators; stresses and state variables remain
    # subject to their normal objective update.
    is_isotropic = False

    # Fedoo-native anisotropic laws use the current material frame maintained
    # by this class. User-material adapters that perform this transport
    # themselves override the flag.
    manages_material_frame = False

    # True when the law returns the simcoon corotational "box" tangent
    # d(tau_hat)/dD, which the UL weakform must convert to the Lie
    # (Truesdell) spatial tangent (see StressEquilibrium.update_2).
    # Native fedoo laws (e.g. ElasticIsotrop) return a plain engineering
    # tangent and are left unconverted.
    _corotational_box_tangent = False

    def __init__(self, density=None, name=""):
        ConstitutiveLaw.__init__(self, name)
        self.density = density
        # True when the tangent is based on deformation-gradient rather than
        # logarithmic-strain kinematics, as for hyperelastic materials.
        self._Lt_from_F = False

    def set_density(self, density):
        """Set the mass density associated with this mechanical material."""
        self.density = density
        return self

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

    # Material-frame resolution

    @staticmethod
    def _data_point_count(data, tensor_order):
        """Return the number of point-wise values stored in a Fedoo array."""
        array = np.asarray(data)
        if tensor_order == 2:
            return 1 if array.ndim < 2 else array.shape[-1]
        return 1 if array.ndim < 3 else array.shape[-1]

    @staticmethod
    def _rotation_from_frames(frames, n_points=1, batch=False):
        frames = np.asarray(frames, dtype=float).reshape(-1, 3, 3)
        if len(frames) == 1 and n_points == 1 and not batch:
            return Rotation.from_matrix(frames[0])
        if len(frames) == 1:
            frames = np.repeat(frames, n_points, axis=0)
        elif n_points != 1 and len(frames) != n_points:
            raise ValueError(f"got {len(frames)} material frames for {n_points} values")
        return Rotation.from_matrix(frames)

    def _initial_material_frame(self, assembly):
        """Return the initial material frame at every integration point."""
        frame = self.get_local_frame(assembly)
        if frame is None:
            frame = np.eye(3)[None, ...]
        frame = np.asarray(frame, dtype=float).reshape(-1, 3, 3)
        if assembly is not None and len(frame) == 1:
            n_points = getattr(
                assembly,
                "n_gauss_points",
                assembly.mesh.n_elements * assembly.n_elm_gp,
            )
            frame = np.repeat(frame, n_points, axis=0)
        return frame

    def get_current_local_frame(self, assembly=None):
        """Return the trial material frame in global coordinates.

        For a small-strain problem this is the user-provided initial frame. In
        finite strain, Fedoo-native anisotropic laws convect the committed
        frame with the current objective rotation increment.  Constitutive
        laws such as Simcoon UMATs that manage this operation internally keep
        using the initial material basis.
        """
        if assembly is None:
            return self.get_local_frame()

        has_user_frame = self.local_frame is not None
        if not getattr(assembly, "_nlgeom", False):
            # Small strain has no evolving material orientation.  With no
            # user frame the material basis is already global, so returning
            # None avoids constructing and applying identity rotations.
            return self.get_local_frame(assembly)
        if self.is_isotropic and not has_user_frame:
            return None

        initial = self._initial_material_frame(assembly)
        if self.manages_material_frame:
            return initial

        start_values = getattr(assembly, "sv_start", {})
        start_frame = np.asarray(
            start_values.get("_MaterialFrame", initial), dtype=float
        ).reshape(-1, 3, 3)
        if "DR" not in assembly.sv:
            current = start_frame
        else:
            increment = np.asarray(assembly.sv["DR"], dtype=float)
            if increment.shape[:2] == (3, 3):
                increment = increment.transpose(2, 0, 1)
            increment = increment.reshape(-1, 3, 3)
            current = increment @ start_frame

        # Replace instead of modifying in place: Assembly.sv_start is a
        # shallow snapshot, so replacement preserves rollback semantics.
        assembly.sv["_MaterialFrame"] = current
        return current

    def get_local_rotation(self, assembly=None, current=True, n_points=1):
        """Return a Simcoon rotation for the requested material frame."""
        if current:
            frames = self.get_current_local_frame(assembly)
        else:
            frames = self.get_local_frame(assembly)
        if frames is None:
            return None
        return self._rotation_from_frames(frames, n_points)

    # Stress, strain, and tangent transformations

    @staticmethod
    def _as_mechanical_array(value):
        if hasattr(value, "asarray"):
            value = value.asarray()
        elif hasattr(value, "array"):
            value = value.array
        return np.asarray(value, dtype=float)

    def _rotate_vector(self, value, assembly, kind, active, current=True):
        value = self._as_mechanical_array(value)
        n_points = self._data_point_count(value, tensor_order=2)
        frames = (
            self.get_current_local_frame(assembly)
            if current
            else self.get_local_frame(assembly)
        )
        if frames is None:
            return value
        frames = np.asarray(frames, dtype=float).reshape(-1, 3, 3)
        if n_points == 1 and len(frames) > 1:
            value = np.repeat(value.reshape(6, 1), len(frames), axis=1)
            n_points = len(frames)
        rotation = self._rotation_from_frames(frames, n_points, batch=value.ndim >= 2)
        apply = rotation.apply_stress if kind == "stress" else rotation.apply_strain
        return apply(value, active=active)

    def local2global_stress(self, stress, assembly=None, current=True):
        """Express stress Voigt vector(s) from material to global axes."""
        return self._rotate_vector(stress, assembly, "stress", True, current)

    def global2local_stress(self, stress, assembly=None, current=True):
        """Express stress Voigt vector(s) from global to material axes."""
        return self._rotate_vector(stress, assembly, "stress", False, current)

    def local2global_strain(self, strain, assembly=None, current=True):
        """Express engineering-strain vector(s) from material to global axes."""
        return self._rotate_vector(strain, assembly, "strain", True, current)

    def global2local_strain(self, strain, assembly=None, current=True):
        """Express engineering-strain vector(s) from global to material axes."""
        return self._rotate_vector(strain, assembly, "strain", False, current)

    def _rotate_H(self, H, assembly, active, current=True):
        H_array = np.asarray(H, dtype=float)
        n_values = self._data_point_count(H_array, tensor_order=4)
        frames = (
            self.get_current_local_frame(assembly)
            if current
            else self.get_local_frame(assembly)
        )
        if frames is None:
            return H
        frames = np.asarray(frames, dtype=float).reshape(-1, 3, 3)
        n_frames = len(frames)
        n_points = max(n_values, n_frames)
        if n_values == 1 and n_points > 1:
            H_array = np.repeat(H_array[:, :, None], n_points, axis=2)
        rotation = self._rotation_from_frames(frames, n_points, batch=H_array.ndim >= 3)
        return rotation.apply_stiffness(H_array, active=active)

    def local2global_H(self, H, assembly=None, current=True):
        """Rotate a material stiffness to global axes using Simcoon."""
        if self.is_isotropic:
            return H
        return self._rotate_H(H, assembly, True, current)

    def global2local_H(self, H, assembly=None, current=True):
        """Rotate a global stiffness to material axes using Simcoon."""
        if self.is_isotropic:
            return H
        return self._rotate_H(H, assembly, False, current)

    def _global2local_matrix_field(self, values, assembly):
        """Change both axes of one or more 3x3 matrices to the initial basis."""
        if self.local_frame is None:
            return values
        frames = self._initial_material_frame(assembly)
        matrices = np.asarray(values, dtype=float)
        single = matrices.ndim == 2
        fedoo_layout = matrices.ndim == 3 and matrices.shape[:2] == (3, 3)
        if single:
            matrices = matrices[None, ...]
        elif fedoo_layout:
            matrices = matrices.transpose(2, 0, 1)
        matrices = matrices.reshape(-1, 3, 3)
        if len(frames) == 1 and len(matrices) > 1:
            frames = np.repeat(frames, len(matrices), axis=0)
        elif len(matrices) == 1 and len(frames) > 1:
            matrices = np.repeat(matrices, len(frames), axis=0)
        elif len(frames) != len(matrices):
            raise ValueError(
                f"got {len(frames)} material frames for "
                f"{len(matrices)} matrix values"
            )
        local = frames.transpose(0, 2, 1) @ matrices @ frames
        if single and len(local) == 1:
            return local[0]
        return local.transpose(1, 2, 0) if fedoo_layout else local

    def global2local_rotation_increment(self, DR, assembly=None):
        """Express a finite rotation increment in the initial material basis."""
        return self._global2local_matrix_field(DR, assembly)

    def global2local_tensor(self, tensor, assembly=None):
        """Change both axes of 3x3 tensor fields to the initial material basis."""
        return self._global2local_matrix_field(tensor, assembly)


class MechanicalUMAT(Mechanical3D):
    """Base class for corotational user-material functions.

    ``MechanicalUMAT`` owns the Fedoo-side constitutive lifecycle: allocation
    of state and work variables, temperature lookup, local-frame changes of
    basis, the UMAT call, and conversion of stress and tangent outputs back to
    global coordinates.  A user material therefore only has to integrate its
    constitutive equations in the supplied corotational material frame.

    Parameters
    ----------
    umat : callable, optional
        Material integration function. Its signature is the same as
        :func:`simcoon.umat` after ``umat_name``::

            umat(strain, dstrain, F0, F1, stress, DR, props,
                 statev_start, time, dtime, wm_start, temperature,
                 *, ndi, tangent_mode)

        It must return ``(stress, statev, wm, tangent)``. Subclasses may omit
        this argument and override :meth:`_call_umat`, as the Simcoon adapter
        does.
    props : array_like, optional
        Material properties. One-dimensional input is stored as a Fortran
        contiguous column, matching the historical Simcoon wrapper behavior.
    n_statev : int, default=0
        Number of private state variables per integration point.
    props_label, statev_label : dict, optional
        Human-readable mappings from names to property/state indices or
        slices. State labels are exposed through ``assembly.sv_component``.
    is_isotropic : bool, default=False
        Whether the material response is invariant under frame rotations.
    tangent_from_F : bool, default=False
        Whether the returned tangent acts on the deformation gradient rather
        than logarithmic strain (used by hyperelastic materials).
    use_elastic_tangent : bool, default=True
        Restore the initialization tangent at the start of every increment.
    tangent_mode : int, default=1
        Tangent selector forwarded unchanged to the UMAT callback.
    density : float, optional
        Material mass density.
    name : str, optional
        Fedoo constitutive-law registration name.

    Notes
    -----
    Private state variables are deliberately opaque to Fedoo. The callback is
    responsible for their objective transport, using the supplied ``DR`` when
    tensor-valued state is present.

    Initial values can be assigned to a particular assembly before problem
    initialization with :meth:`set_initial_statev`. The values are stored on
    the assembly, so one material instance can be reused with different
    initial states.
    """

    manages_material_frame = True
    _corotational_box_tangent = True
    required_corate = None

    def __init__(
        self,
        umat=None,
        props=None,
        n_statev=0,
        *,
        props_label=None,
        statev_label=None,
        is_isotropic=False,
        tangent_from_F=False,
        use_elastic_tangent=True,
        tangent_mode=1,
        density=None,
        name="",
    ):
        super().__init__(name=name, density=density)
        if umat is not None and not callable(umat):
            raise TypeError("umat must be callable")
        if not isinstance(n_statev, (int, np.integer)) or n_statev < 0:
            raise ValueError("n_statev must be a non-negative integer")

        self.umat = umat
        if props is None:
            props = np.array([], dtype=float)
        self.props = np.asfortranarray(np.c_[props])
        self.n_statev = int(n_statev)
        self.props_label = dict(props_label or {})
        self.statev_label = dict(statev_label or {})
        self.is_isotropic = bool(is_isotropic)
        self._Lt_from_F = bool(tangent_from_F)
        # Keep the historical public name used by Simcoon and FE2.
        self.use_elastic_lt = bool(use_elastic_tangent)
        self.tangent_mode = tangent_mode

    def _validate_kinematics(self, assembly):
        """Validate optional objective-rate requirements of a UMAT."""
        if not assembly._nlgeom or self.required_corate is None:
            return
        weakform = getattr(assembly, "weakform", None)
        corate = getattr(weakform, "corate", None)
        if corate is None:
            return
        corate = str(corate).lower()
        required = self.required_corate
        if isinstance(required, str):
            required = (required,)
        required = tuple(value.lower() for value in required)
        if corate not in required:
            choices = " or ".join(repr(value) for value in required)
            raise ValueError(
                f"{type(self).__name__} requires corate={choices} in finite "
                f"strain, got {corate!r}"
            )

    def _call_umat(
        self,
        strain,
        dstrain,
        F0,
        F1,
        stress,
        DR,
        props,
        statev_start,
        time,
        dtime,
        wm_start,
        temperature,
        *,
        ndi,
        tangent_mode,
    ):
        """Invoke the configured user-material function."""
        if self.umat is None:
            raise NotImplementedError(
                "MechanicalUMAT requires a umat callback or a _call_umat override"
            )
        return self.umat(
            strain,
            dstrain,
            F0,
            F1,
            stress,
            DR,
            props,
            statev_start,
            time,
            dtime,
            wm_start,
            temperature,
            ndi=ndi,
            tangent_mode=tangent_mode,
        )

    def _get_ndi(self, assembly):
        if assembly.space.get_dimension() != "2Dstress":
            return 3
        if self._Lt_from_F:
            raise NotImplementedError(
                "UMAT laws with a deformation-gradient tangent are not "
                "compatible with the 2D plane-stress assumption"
            )
        return 2

    def _statev_array(self, assembly):
        """Return an assembly Statev array with the required exact shape."""
        shape = (self.n_statev, assembly.n_gauss_points)
        if "Statev" not in assembly.sv:
            assembly.sv["Statev"] = np.zeros(shape, order="F")
            return assembly.sv["Statev"]

        statev = np.asarray(assembly.sv["Statev"])
        if statev.shape != shape:
            raise ValueError(
                f"Statev has shape {statev.shape}, expected {shape} for "
                f"{type(self).__name__} on this assembly"
            )
        assembly.sv["Statev"] = np.asfortranarray(statev, dtype=float)
        return assembly.sv["Statev"]

    @staticmethod
    def _broadcast_initial_statev(values, n_components, n_points, label):
        """Broadcast labeled initial values to component-by-point layout."""
        values = np.asarray(values, dtype=float)
        shape = (n_components, n_points)

        if values.ndim == 0:
            return np.full(shape, values.item())
        if values.shape == shape:
            return values
        if values.shape == (n_components,) or values.shape == (n_components, 1):
            return np.broadcast_to(values.reshape(n_components, 1), shape)
        if n_components == 1 and values.shape == (n_points,):
            return values.reshape(1, n_points)

        raise ValueError(
            f"initial Statev field {label!r} has shape {values.shape}; expected "
            f"a scalar, ({n_components},), ({n_components}, 1), or {shape}"
        )

    def set_initial_statev(self, assembly, label, value):
        """Set a labeled initial state-variable field on an assembly.

        A scalar is broadcast over all selected components and Gauss points.
        For a multi-component label, a component vector is broadcast over the
        Gauss points. A complete field uses the Fedoo layout
        ``(n_components, n_gauss_points)``. For a scalar label, a vector of
        length ``n_gauss_points`` supplies one value per point.

        This method must be called before the associated problem initializes.
        Repeated calls preserve components initialized previously.

        Parameters
        ----------
        assembly : Assembly
            Assembly that will own the initial state.
        label : str
            Entry in :attr:`statev_label`.
        value : scalar or array_like
            Uniform, component-wise, or Gauss-point initial values.

        Returns
        -------
        MechanicalUMAT
            This material, to allow chained configuration calls.
        """
        if {"DR", "Wm", "TangentMatrix"}.issubset(assembly.sv):
            raise RuntimeError(
                "set_initial_statev() must be called before assembly initialization"
            )
        if label not in self.statev_label:
            available = ", ".join(repr(name) for name in self.statev_label)
            raise KeyError(
                f"unknown state-variable label {label!r}; available labels: "
                f"{available or 'none'}"
            )

        component = self.statev_label[label]
        try:
            indices = np.arange(self.n_statev)[component]
        except (IndexError, TypeError) as error:
            raise ValueError(
                f"state-variable label {label!r} selects invalid component "
                f"{component!r} for n_statev={self.n_statev}"
            ) from error
        indices = np.atleast_1d(indices)
        if len(indices) == 0:
            raise ValueError(f"state-variable label {label!r} selects no components")

        statev = self._statev_array(assembly)
        statev[indices, :] = self._broadcast_initial_statev(
            value,
            len(indices),
            assembly.n_gauss_points,
            label,
        )
        return self

    def initialize(self, assembly, pb):
        self._validate_kinematics(assembly)
        n_points = assembly.n_gauss_points
        had_statev = "Statev" in assembly.sv
        statev = self._statev_array(assembly)
        for label, component in self.statev_label.items():
            assembly.sv_component[label] = ("Statev", component)

        initialized_fields = {"DR", "Wm", "TangentMatrix"}
        if assembly._nlgeom:
            initialized_fields.add("F")
        if had_statev and all(field in assembly.sv for field in initialized_fields):
            self.is_initialized = True
            return

        DR = np.empty((3, 3, n_points), order="F")
        DR[...] = np.eye(3).reshape(3, 3, 1)
        assembly.sv["DR"] = DR

        if assembly._nlgeom:
            F = np.empty((3, 3, n_points), order="F")
            F[...] = np.eye(3).reshape(3, 3, 1)
            assembly.sv["F"] = F
        else:
            F = np.array([])

        assembly.sv["Wm"] = np.zeros((4, n_points), order="F")
        zeros_6 = np.zeros((6, n_points), order="F")
        ndi = self._get_ndi(assembly)

        _, _, _, tangent_local = self._call_umat(
            zeros_6,
            zeros_6,
            F,
            F,
            zeros_6,
            DR,
            self.props,
            statev,
            0,
            0,
            assembly.sv["Wm"],
            self.get_temp_gp(assembly, pb),
            ndi=ndi,
            tangent_mode=self.tangent_mode,
        )
        assembly.sv["TangentMatrix"] = self.local2global_H(
            tangent_local, assembly, current=False
        )
        if ndi == 2:
            assembly.sv["TangentMatrix"] = self.get_tangent_matrix(assembly, "2Dstress")
        if self.use_elastic_lt:
            assembly.sv["ElasticMatrix"] = assembly.sv["TangentMatrix"]
        self.is_initialized = True

    def update(self, assembly, pb):
        if "DStrain" in assembly.sv:
            dstrain = assembly.sv["DStrain"]
        else:
            dstrain = assembly.sv["Strain"] - assembly.sv_start["Strain"]

        if assembly._nlgeom:
            F0 = self.global2local_tensor(assembly.sv_start["F"], assembly)
            F1 = self.global2local_tensor(assembly.sv["F"], assembly)
        else:
            F0 = F1 = np.array([])

        ndi = self._get_ndi(assembly)
        stress_local, statev, wm, tangent_local = self._call_umat(
            self.global2local_strain(
                assembly.sv_start["Strain"], assembly, current=False
            ),
            self.global2local_strain(dstrain, assembly, current=False),
            F0,
            F1,
            self.global2local_stress(
                assembly.sv_start["Stress"], assembly, current=False
            ),
            self.global2local_rotation_increment(assembly.sv["DR"], assembly),
            self.props,
            assembly.sv_start["Statev"],
            pb.time,
            pb.dtime,
            assembly.sv_start["Wm"],
            self.get_temp_gp(assembly, pb),
            ndi=ndi,
            tangent_mode=self.tangent_mode,
        )
        assembly.sv["Statev"] = statev
        assembly.sv["Wm"] = wm
        assembly.sv["TangentMatrix"] = self.local2global_H(
            tangent_local, assembly, current=False
        )
        if ndi == 2:
            assembly.sv["TangentMatrix"] = self.get_tangent_matrix(assembly, "2Dstress")
        assembly.sv["Stress"] = StressTensorList(
            self.local2global_stress(stress_local, assembly, current=False)
        )

    def set_start(self, assembly, pb):
        if self.use_elastic_lt:
            assembly.sv["TangentMatrix"] = assembly.sv["ElasticMatrix"]

    def get_temp_gp(self, assembly, pb):
        """Return the current temperature field at Gauss points, if any."""
        if "Temp" in assembly.sv:
            temperature = assembly.sv["Temp"]
        elif "Temp" in assembly.space.list_variables():
            temperature = assembly.convert_data(
                pb.get_dof_solution("Temp"), "Node", "GaussPoint"
            )
        else:
            return None
        if np.isscalar(temperature) and temperature == 0:
            return None
        return temperature

    def get_tangent_matrix(self, assembly, dimension=None):
        if dimension is None:
            dimension = assembly.space.get_dimension()
        H = assembly.sv["TangentMatrix"]
        if dimension == "2Dstress":
            return self.get_H_plane_stress(H)
        return H
