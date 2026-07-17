"""Small-strain J2 plasticity with isotropic hardening."""

import numpy as np
from simcoon import Rotation as SimRotation

from fedoo.core.mechanical3d import Mechanical3D
from fedoo.util.voigt_tensors import StrainTensorList, StressTensorList


class ElastoPlasticity(Mechanical3D):
    """Elasto-plastic constitutive law with isotropic hardening.

    The stress integration uses a vectorized radial-return algorithm for the
    von Mises yield criterion. In a finite-strain updated-Lagrangian analysis,
    the law operates on the corotated strain increment prepared by
    :class:`fedoo.weakform.StressEquilibrium`.

    The returned elastoplastic tangent is the continuum tangent evaluated at
    the updated stress. It is not the algorithmically consistent tangent of
    the discrete radial-return integration. Consequently, global Newton
    convergence is not guaranteed to be quadratic for finite plastic
    increments or non-proportional loading paths.

    .. warning::

        This Python implementation is intended only for pedagogical use and
        as a readable reference implementation. For computational analyses,
        prefer :class:`fedoo.constitutivelaw.Simcoon`, which provides optimized
        constitutive updates and algorithmically consistent tangent options.

    Parameters
    ----------
    young_modulus : float
        Young modulus.
    poisson_ratio : float
        Poisson ratio.
    yield_stress : float
        Initial yield stress.
    name : str, optional
        Name of the constitutive law.
    """

    def __init__(
        self,
        young_modulus,
        poisson_ratio,
        yield_stress,
        name="",
    ):
        super().__init__(name)
        self.young_modulus = young_modulus
        self.poisson_ratio = poisson_ratio
        self.yield_stress = yield_stress
        self.return_mapping_tolerance = 1e-6
        self.max_return_mapping_iterations = 50

        self._hardening_function = None
        self._hardening_function_derivative = None
        self._current_stress = None
        self._current_plastic_strain = None
        self._current_plasticity = None
        self._current_tangent = None

    @property
    def shear_modulus(self):
        """Shear modulus."""
        return self.young_modulus / (2.0 * (1.0 + self.poisson_ratio))

    def set_return_mapping_tolerance(self, tolerance):
        """Set the absolute tolerance used by the local return mapping."""
        if tolerance <= 0:
            raise ValueError("The return-mapping tolerance must be positive.")
        self.return_mapping_tolerance = tolerance

    def get_elastic_matrix(self, dimension="3D"):
        """Return the isotropic elastic matrix in engineering Voigt form."""
        if dimension == "2Dstress":
            raise NotImplementedError(
                "ElastoPlasticity does not yet implement plane-stress "
                "return mapping."
            )

        young_modulus = self.young_modulus
        poisson_ratio = self.poisson_ratio
        dtype = (
            float
            if np.isscalar(young_modulus) and np.isscalar(poisson_ratio)
            else object
        )
        elastic_matrix = np.zeros((6, 6), dtype=dtype)
        lame_parameter = (
            young_modulus
            * poisson_ratio
            / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio))
        )
        elastic_matrix[:3, :3] = lame_parameter
        elastic_matrix[0, 0] += 2 * self.shear_modulus
        elastic_matrix[1, 1] += 2 * self.shear_modulus
        elastic_matrix[2, 2] += 2 * self.shear_modulus
        elastic_matrix[3, 3] = self.shear_modulus
        elastic_matrix[4, 4] = self.shear_modulus
        elastic_matrix[5, 5] = self.shear_modulus
        return elastic_matrix

    def set_hardening_function(self, function_type, **kwargs):
        """Define the isotropic hardening function.

        ``function_type="power"`` defines ``R(p) = h * p**beta``.
        ``function_type="user"`` accepts ``hardening_function`` and
        ``hardening_function_derivative`` callables.
        """
        function_type = function_type.lower()

        if function_type == "power":
            if "h" not in kwargs:
                raise TypeError("Keyword argument 'h' is required.")
            if "beta" not in kwargs:
                raise TypeError("Keyword argument 'beta' is required.")

            hardening_modulus = kwargs["h"]
            beta = kwargs["beta"]
            if hardening_modulus < 0:
                raise ValueError("The hardening modulus h must be non-negative.")
            if beta <= 0:
                raise ValueError("The hardening exponent beta must be positive.")

            def hardening_function(plasticity):
                return hardening_modulus * np.asarray(plasticity) ** beta

            def hardening_function_derivative(plasticity):
                with np.errstate(divide="ignore", invalid="ignore"):
                    return (
                        beta * hardening_modulus * np.asarray(plasticity) ** (beta - 1)
                    )

        elif function_type == "user":
            hardening_function = kwargs.get("hardening_function")
            hardening_function_derivative = kwargs.get("hardening_function_derivative")
            if hardening_function is None:
                raise TypeError("Keyword argument 'hardening_function' is required.")
            if hardening_function_derivative is None:
                raise TypeError(
                    "Keyword argument 'hardening_function_derivative' " "is required."
                )
        else:
            raise ValueError("function_type must be either 'power' or 'user'.")

        self._hardening_function = hardening_function
        self._hardening_function_derivative = hardening_function_derivative

    def hardening_function(self, plasticity):
        """Evaluate the isotropic hardening stress."""
        if self._hardening_function is None:
            raise RuntimeError(
                "The hardening function has not been defined. Call "
                "set_hardening_function first."
            )
        return self._hardening_function(plasticity)

    def hardening_function_derivative(self, plasticity):
        """Evaluate the derivative of the isotropic hardening stress."""
        if self._hardening_function_derivative is None:
            raise RuntimeError(
                "The hardening function has not been defined. Call "
                "set_hardening_function first."
            )
        return self._hardening_function_derivative(plasticity)

    def yield_function(self, stress, plasticity):
        """Evaluate the von Mises yield function."""
        return (
            stress.von_mises() - self.yield_stress - self.hardening_function(plasticity)
        )

    def yield_function_derivative(self, stress):
        """Differentiate the yield function with respect to stress."""
        equivalent_stress = np.asarray(stress.von_mises())
        if np.any(equivalent_stress == 0):
            raise ZeroDivisionError(
                "The yield direction is undefined at zero deviatoric stress."
            )
        return StressTensorList(
            1.5 * np.asarray(stress.deviatoric()) / equivalent_stress
        ).to_strain()

    @staticmethod
    def _as_point_array(values):
        array = np.asarray(values, dtype=float)
        if array.ndim == 1:
            array = array.reshape(6, 1)
        if array.ndim != 2 or array.shape[0] != 6:
            raise ValueError("Expected a Voigt array with shape (6, n_points).")
        return array

    @staticmethod
    def _deviatoric(stress):
        deviator = stress.copy()
        mean_stress = np.mean(stress[:3], axis=0)
        deviator[:3] -= mean_stress
        return deviator

    @staticmethod
    def _flow_direction(stress):
        stress_tensor = StressTensorList(stress)
        equivalent_stress = np.asarray(stress_tensor.von_mises())
        direction = 1.5 * np.asarray(stress_tensor.deviatoric()) / equivalent_stress
        direction[3:] *= 2.0
        return direction

    def _plastic_increment(self, equivalent_trial_stress, plasticity_old):
        """Solve all active radial-return consistency equations together."""
        equivalent_trial_stress = np.asarray(equivalent_trial_stress, dtype=float)
        plasticity_old = np.asarray(plasticity_old, dtype=float)
        initial_residual = (
            equivalent_trial_stress
            - self.yield_stress
            - self.hardening_function(plasticity_old)
        )
        lower_bound = np.zeros_like(initial_residual)
        upper_bound = initial_residual / (3.0 * self.shear_modulus)

        def residual(plastic_increment):
            return (
                equivalent_trial_stress
                - 3.0 * self.shear_modulus * plastic_increment
                - self.yield_stress
                - self.hardening_function(plasticity_old + plastic_increment)
            )

        for _ in range(self.max_return_mapping_iterations):
            points_to_expand = residual(upper_bound) > 0
            if not np.any(points_to_expand):
                break
            upper_bound[points_to_expand] *= 2.0
        else:
            raise RuntimeError("Unable to bracket every plastic multiplier.")

        plastic_increment = 0.5 * upper_bound
        converged = np.zeros_like(plastic_increment, dtype=bool)

        for _ in range(self.max_return_mapping_iterations):
            value = residual(plastic_increment)
            converged |= np.abs(value) <= self.return_mapping_tolerance
            if np.all(converged):
                return plastic_increment

            active = ~converged
            positive_residual = active & (value > 0)
            negative_residual = active & ~positive_residual
            lower_bound[positive_residual] = plastic_increment[positive_residual]
            upper_bound[negative_residual] = plastic_increment[negative_residual]

            hardening_slope = self.hardening_function_derivative(
                plasticity_old + plastic_increment
            )
            denominator = 3.0 * self.shear_modulus + hardening_slope
            candidate = plastic_increment + value / denominator
            invalid_candidate = (
                ~np.isfinite(candidate)
                | (denominator <= 0)
                | (candidate <= lower_bound)
                | (candidate >= upper_bound)
            )
            candidate[invalid_candidate] = 0.5 * (
                lower_bound[invalid_candidate] + upper_bound[invalid_candidate]
            )
            plastic_increment[active] = candidate[active]

        unconverged_residual = np.max(np.abs(residual(plastic_increment)[~converged]))
        raise RuntimeError(
            "The radial-return algorithm did not converge after "
            f"{self.max_return_mapping_iterations} iterations; maximum "
            f"residual is {unconverged_residual:.6g}."
        )

    def _integrate(self, total_strain, plasticity_old, plastic_strain_old):
        """Integrate all material points from their committed state."""
        strain = self._as_point_array(total_strain)
        plastic_strain_old = self._as_point_array(plastic_strain_old)
        plasticity_old = np.asarray(plasticity_old, dtype=float).reshape(-1)
        n_points = strain.shape[1]
        if plastic_strain_old.shape[1] != n_points or len(plasticity_old) != n_points:
            raise ValueError("Inconsistent number of constitutive-law points.")

        elastic_matrix = np.asarray(self.get_elastic_matrix(), dtype=float)
        trial_stress = elastic_matrix @ (strain - plastic_strain_old)
        equivalent_trial_stress = np.asarray(StressTensorList(trial_stress).von_mises())
        trial_yield = (
            equivalent_trial_stress
            - self.yield_stress
            - self.hardening_function(plasticity_old)
        )
        plastic_points = trial_yield > self.return_mapping_tolerance

        stress = trial_stress.copy()
        plastic_strain = plastic_strain_old.copy()
        plasticity = plasticity_old.copy()
        tangent = np.repeat(elastic_matrix[:, :, None], n_points, axis=2)

        if np.any(plastic_points):
            plastic_increment = self._plastic_increment(
                equivalent_trial_stress[plastic_points],
                plasticity_old[plastic_points],
            )
            plasticity[plastic_points] += plastic_increment

            trial_deviator = self._deviatoric(trial_stress[:, plastic_points])
            radial_factor = (
                1.0
                - 3.0
                * self.shear_modulus
                * plastic_increment
                / equivalent_trial_stress[plastic_points]
            )
            stress[:, plastic_points] = (
                trial_stress[:, plastic_points]
                - trial_deviator
                + radial_factor[None, :] * trial_deviator
            )

            direction = self._flow_direction(stress[:, plastic_points])
            plastic_strain[:, plastic_points] += direction * plastic_increment[None, :]

            hardening_slope = self.hardening_function_derivative(
                plasticity[plastic_points]
            )
            elastic_flow = elastic_matrix @ direction
            denominator = (
                np.einsum("ip,ip->p", direction, elastic_flow) + hardening_slope
            )
            tangent[:, :, plastic_points] -= (
                np.einsum("ip,jp->ijp", elastic_flow, elastic_flow)
                / denominator[None, None, :]
            )

        return (
            StressTensorList(stress),
            StrainTensorList(plastic_strain),
            plasticity,
            tangent,
        )

    def compute_stress(
        self,
        total_strain,
        plasticity_old=None,
        plastic_strain_old=None,
    ):
        """Integrate a material-point state without an assembly.

        This helper is stateless with respect to history: callers advancing
        several increments must pass the previously returned plasticity and
        plastic strain explicitly.
        """
        strain = self._as_point_array(total_strain)
        n_points = strain.shape[1]
        if plasticity_old is None:
            plasticity_old = np.zeros(n_points)
        if plastic_strain_old is None:
            plastic_strain_old = np.zeros((6, n_points))

        (
            self._current_stress,
            self._current_plastic_strain,
            self._current_plasticity,
            self._current_tangent,
        ) = self._integrate(strain, plasticity_old, plastic_strain_old)
        return self._current_stress

    def get_plasticity(self):
        """Return plasticity from the most recent integration."""
        return self._current_plasticity

    def get_plastic_strain(self):
        """Return plastic strain from the most recent integration."""
        return self._current_plastic_strain

    def get_stress(self):
        """Return stress from the most recent integration."""
        return self._current_stress

    def get_tangent_matrix(self, assembly=None, dimension=None):
        """Return the current tangent, or the elastic tangent initially."""
        if assembly is not None:
            if dimension is None:
                dimension = assembly.space.get_dimension()
            if "TangentMatrix" in assembly.sv:
                return assembly.sv["TangentMatrix"]
        if self._current_tangent is not None:
            return self._current_tangent
        return self.get_elastic_matrix(dimension or "3D")

    def initialize(self, assembly, pb):
        """Initialize finite-element state variables."""
        self._dimension = assembly.space.get_dimension()
        elastic_matrix = np.asarray(
            self.get_elastic_matrix(self._dimension), dtype=float
        )
        n_points = assembly.n_gauss_points
        assembly.sv["P"] = np.zeros(n_points)
        assembly.sv["EP"] = StrainTensorList(np.zeros((6, n_points), order="F"))
        assembly.sv["TangentMatrix"] = np.repeat(
            elastic_matrix[:, :, None], n_points, axis=2
        )
        self.is_initialized = True

    def update(self, assembly, pb):
        """Update stress, plastic state, and tangent for the current iterate."""
        if "DStrain" in assembly.sv:
            total_strain = assembly.sv["Strain"] + assembly.sv["DStrain"]
        else:
            total_strain = assembly.sv["Strain"]

        start_state = getattr(assembly, "sv_start", assembly.sv)
        (
            self._current_stress,
            self._current_plastic_strain,
            self._current_plasticity,
            self._current_tangent,
        ) = self._integrate(
            total_strain,
            start_state["P"],
            start_state["EP"],
        )
        assembly.sv["Stress"] = self._current_stress
        assembly.sv["EP"] = self._current_plastic_strain
        assembly.sv["P"] = self._current_plasticity
        assembly.sv["TangentMatrix"] = self._current_tangent

    def set_start(self, assembly, pb):
        """Commit the converged state and prepare the elastic predictor."""
        if assembly._nlgeom and "DR" in assembly.sv:
            rotation = SimRotation.from_matrix(assembly.sv["DR"].transpose(2, 0, 1))
            assembly.sv["EP"] = StrainTensorList(
                rotation.apply_strain(assembly.sv["EP"].asarray())
            )

        elastic_matrix = np.asarray(
            self.get_elastic_matrix(self._dimension), dtype=float
        )
        assembly.sv["TangentMatrix"] = np.repeat(
            elastic_matrix[:, :, None],
            assembly.n_gauss_points,
            axis=2,
        )

    def reset(self):
        """Reset cached results; assembly history is managed by Fedoo."""
        super().reset()
        self._current_stress = None
        self._current_plastic_strain = None
        self._current_plasticity = None
        self._current_tangent = None
