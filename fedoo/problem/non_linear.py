from __future__ import annotations
import numpy as np
from fedoo.core.assembly import Assembly
from fedoo.core.problem import Problem
from fedoo.problem.line_search import line_search, _line_search_manager
import warnings
from typing import Callable


class _NonLinearBase:
    def __init__(self, assembly, nlgeom=False, name="MainProblem"):
        if isinstance(assembly, str):
            assembly = Assembly.get_all()[assembly]

        # A = assembling.current.get_global_matrix() #tangent stiffness matrix
        A = 0  # tangent stiffness matrix - will be initialized only when required
        B = 0
        # D = assembling.get_global_vector() #initial stress vector
        D = 0  # initial stress vector #will be initialized later
        self.print_info = 1  # print info of NR convergence during solve
        self._U = 0  # displacement at the end of the previous converged increment
        self._dU = 0  # displacement increment

        self._err0 = None  # initial error for NR error estimation
        self._alpha = 1  # line search current parameter

        self.nr_parameters = {
            "err0": None,  # default error for NR error estimation
            "criterion": "Force",
            "tol": 5e-3,
            "max_subiter": 10,
            "dt_increase_niter": None,
            "norm_type": 2,
            "adaptive_stiffness": False,
            "assume_cvg_at_max_subiter": False,
            "eigenvalue_shift": False,
            "eigenvalue_shift_factor": 0.1,
        }
        """
        Parameters to set the newton raphson algorithm:
            * 'err0': The reference error.
              Default is None (automatically computed)
            * 'criterion': Type of convergence test in
              ['Displacement', 'Force', 'Work'].
              Default is 'Displacement'.
            * 'tol': Error tolerance for convergence. Default is 1e-3.
            * 'max_subiter': Number of nr iterations before returning a
              convergence error. Default is 6.
            * 'dt_increase_niter': number of nr iterations threshold that
              define an easy convergence. In problem allowing automatic
              convergence, if the Newton–Raphson loop converges in
              fewer iterations, the time step is increased.
              If None, the value is initialized to ``max_subiter // 3``. 
            * 'adaptive_stiffness': bool, to use an adaptative stiffness
              parameter to enhance convergence. Default is False.
            * 'norm_type': define the norm used to test the criterion
              Use numpy.inf for the max value. Default is 2.
            * 'assume_cvg_at_max_subiter': bool, default = False.
              WARNING: This is a dangerous parameter. If True, the Newton-Raphson
              algorithm assumes convergence after max_subiter iterations are reached,
              regardless of whether the tolerance criterion has been satisfied.
              Use only for special cases where you want to force convergence even
              with unconverged solutions. Default False ensures proper convergence
              checking. The error is still computed for informational purposes
              if print_info > 0.
            * 'eigenvalue_shift': bool, default = False.
              If True, adds a shifted identity matrix to improve conditioning:
              A_eff = A + alpha*I
              The shift magnitude is alpha = eigenvalue_shift_factor * R,
              where R is the Rayleigh quotient estimate of the matrix.
            * 'eigenvalue_shift_factor': float, default = 0.1.
              Scaling factor for the eigenvalue shift. The actual shift is:
              alpha = eigenvalue_shift_factor * |Rayleigh quotient|.
              Larger values = stronger stabilization but less accuracy.
              Smaller values = more accurate but weaker stabilization.
        """

        # attributes used for line search algorithm or contact management
        self._step_size_callback = None
        self._ls_callbacks = {}  # dict of line search functions
        self._line_search_update = False  # tag used during line_search update
        self._step_filter_callback = None  # OGC per-vertex filter
        self._nr_min_subiter = (
            0  # SDI: minimum NR sub-iterations before accepting convergence
        )
        self._t_fact_inc = None  # frozen t_fact for the current increment

        self.__assembly = assembly
        super().__init__(A, B, D, assembly.mesh, name, assembly.space)
        self.nlgeom = nlgeom
        self.t0 = 0
        self.tmax = 1
        self.time = 0
        self.dtime = 0
        self.__iter = 0
        self.__compteurOutput = 0

        self.interval_output = -1  # save results every self.interval_output iter or time step if self.save_at_exact_time = True
        self.save_at_exact_time = True
        self.exec_callback_at_each_iter = False
        self.err_num = 1e-8  # numerical error

    @property
    def n_iter(self):
        """Return the number of iterations made to solve the problem."""
        return self.__iter

    def get_disp(self, name="Disp"):
        """Return the displacement components.


        Parameters
        ----------
        name : str, optional
            Name of the variable to return. For instance, if name == 'DispX'
            return only the X component of displacement.

        Returns
        -------
        numpy.ndarray
        """
        if np.isscalar(self._dU) and self._dU == 0:
            return self._get_vect_component(self._U, name)
        return self._get_vect_component(self._U + self._dU, name)

    def get_rot(self, name="Rot"):
        """Return the rotation components.


        Parameters
        ----------
        name : str, optional
            Name of the variable to return. For instance, if name == 'RotX'
            return only the X component of rotation.

        Returns
        -------
        numpy.ndarray
        """
        if np.isscalar(self._dU) and self._dU == 0:
            return self._get_vect_component(self._U, name)
        return self._get_vect_component(self._U + self._dU, name)

    def get_temp(self):
        """Return the nodal temperature field."""
        if np.isscalar(self._dU) and self._dU == 0:
            return self._get_vect_component(self._U, "Temp")
        return self._get_vect_component(self._U + self._dU, "Temp")

    # Return all the dof for every variable under a vector form
    def get_dof_solution(self, name="all"):
        if np.isscalar(self._dU) and self._dU == 0:
            return self._get_vect_component(self._U, name)
        return self._get_vect_component(self._U + self._dU, name)

    def updateA(self):
        # dt not used for static problem
        self.set_A(self.__assembly.current.get_global_matrix())

    def updateD(self, start=False):
        # start not used for static problem
        self.set_D(self.__assembly.current.get_global_vector())

    def initialize(self):
        self.__assembly.initialize(self)
        self.set_A(0)

    def set_start(self, save_results=False, callback=None):
        # dt not used for static problem
        self._nr_min_subiter = 0  # reset SDI for new increment
        if not (np.isscalar(self._dU) and self._dU == 0):
            self._U += self._dU
            self._dU = 0
            self._err0 = self.nr_parameters[
                "err0"
            ]  # initial error for NR error estimation
            self.__assembly.set_start(self)

            # Save results
            if save_results:
                self.save_results(self.__compteurOutput)
                self.__compteurOutput += 1

            if callback is not None:
                if self.exec_callback_at_each_iter or save_results:
                    callback(self)
        else:
            self._err0 = self.nr_parameters[
                "err0"
            ]  # initial error for NR error estimation
            self.__assembly.set_start(self)

    def to_start(self):
        self._dU = 0
        self._nr_min_subiter = 0
        self._t_fact_inc = None
        self._err0 = self.nr_parameters["err0"]  # initial error for NR error estimation
        self.__assembly.to_start(self)

    def update(self, compute="all", updateWeakForm=True):
        """
        Assemble the matrix including the following modification:
            - New initial Stress
            - New initial Displacement
            - Modification of the mesh
            - Change in constitutive law (internal variable)
        Don't Update the problem with the new assembled global matrix and global vector -> use UpdateA and UpdateD method for this purpose
        """
        if updateWeakForm == True:
            self.__assembly.update(self, compute)
        else:
            self.__assembly.current.assemble_global_mat(compute)

        if self.bc._update_during_inc:
            self.update_boundary_conditions()

    def reset(self):
        self.__assembly.reset()

        self.set_A(0)  # tangent stiffness
        self.set_D(0)
        # self.set_A(self.__assembly.current.get_global_matrix()) #tangent stiffness
        # self.set_D(self.__assembly.current.get_global_vector())

        self._U = 0
        self._dU = 0

        self._err0 = self.nr_parameters["err0"]  # initial error for NR error estimation
        self._t_fact_inc = None
        self.t0 = 0
        self.tmax = 1
        self.__iter = 0

    def change_assembly(self, assembling, update=True):
        """
        Modify the assembly associated to the problem and update the problem (see Assembly.update for more information)
        """
        if isinstance(assembling, str):
            assembling = Assembly[assembling]

        self.__assembly = assembling
        if update:
            self.update()

    def _apply_eigenvalue_shift(self, A, shift_factor=None):
        """
        Apply eigenvalue shift to improve matrix conditioning.

        Adds alpha*I to the matrix if eigenvalue_shift is enabled and shift is
        significant. Handles both sparse and dense matrices.

        Parameters
        ----------
        A : sparse or dense matrix
            The matrix to shift
        shif_factor : float, default = 0.01.
            Scaling factor for the eigenvalue shift.
            Larger values = stronger stabilization but less accuracy.
            Smaller values = more accurate but weaker stabilization.

        Returns
        -------
        matrix
            The shifted matrix A + alpha*I, or original A if shift is negligible
        """
        alpha_shift = self._estimate_eigenvalue_shift(A, shift_factor)
        if alpha_shift <= 1e-16:  # Shift too small to apply
            return A

        try:
            from scipy import sparse

            n = A.shape[0]
            if sparse.issparse(A):
                A_shifted = A + alpha_shift * sparse.identity(n, format=A.format)
            else:
                A_shifted = A + alpha_shift * np.eye(n)

            if self.print_info > 1:
                print("          Eigenvalue shift alpha: {:.4e}".format(alpha_shift))
            return A_shifted
        except Exception:
            # If shift fails, return unmodified matrix
            return A

    def _estimate_eigenvalue_shift(self, A, shift_factor=0.01):
        """
        Estimate eigenvalue shift parameter using Rayleigh quotient.

        The shift improves matrix conditioning by adding alpha*I to the stiffness.
        Uses a cheap estimate based on Rayleigh quotient with current displacement.

        Parameters
        ----------
        A : sparse or dense matrix
            The matrix for which to estimate eigenvalue shift (typically KT or KE)
        shift_factor : float, default = 0.01.
            Scaling factor for the eigenvalue shift.
            Larger values = stronger stabilization but less accuracy.
            Smaller values = more accurate but weaker stabilization.

        Returns
        -------
        float
            Estimated shift parameter alpha >= 0
        """
        try:
            # Get current displacement increment to use for Rayleigh quotient
            x = self.get_X()
            if np.isscalar(x) or x.size == 0:
                return 0.0

            # Compute Rayleigh quotient: r = (x^T A x) / (x^T x)
            # This estimates the largest eigenvalue
            x_norm_sq = np.dot(x, x)
            if np.isscalar(A):
                return 0.0

            Ax = A @ x
            rayleigh = np.dot(x, Ax) / (x_norm_sq + 1e-16)

            # Use a fraction of estimated largest eigenvalue as shift
            # For positive definite systems, shift smallest eigenvalue upward
            alpha = shift_factor * abs(rayleigh)

            return max(alpha, 0.0)  # Ensure non-negative shift

        except Exception:
            # If estimation fails, return zero shift
            return 0.0


    def _update_step_size_callback(self):
        """Update the line search callback from 'ls_callbacks' attribute."""
        if not self._ls_callbacks:
            self._step_size_callback = None
            
        elif len(self._ls_callbacks) == 1:
            self._step_size_callback = next(iter(self._ls_callbacks.values()))
            
        else:
            # Multiple constraints exist: use the manager
            self._step_size_callback = _line_search_manager

    def add_line_search(self, method="Quadratic", name=None):
        r"""Add line search algorithm for the Newton-Raphson solver.

        Line search improves global convergence by scaling the displacement
        increment :math:`dX` by a step size :math:`\alpha \in (0, 1]`. This is
        particularly useful for problems with sharp non-linearities or when
        the initial guess is far from the equilibrium.

        Parameters
        ----------
        method : {'Armijo', 'Residual', 'Energy', 'Quadratic'} or callable, default 'Quadratic'
            The strategy used to determine or refine the step size:

            * **'Armijo'**: Ensures a "sufficient decrease" in the residual
              using a least-square assumption. Standard for most nonlinear applications.
            * **'Residual'**: Simple backtracking that accepts any step reducing
              the residual norm. Fast but less robust.
            * **'Energy'**: Minimizes the out-of-balance work (residual projected
              onto the search direction). Ideal for snap-through/buckling.
            * **'Quadratic'**: Performs a parabolic interpolation of the objective
              function to jump directly to the estimated minimum.
            * **callable**: If a function is provided, it must follow the signature 
              ``user_line_search(pb, dX) -> float`` and will be assigned directly 
              as the line search callback.
        name : str, optional
            A unique identifier for the line search. If not provided, it defaults 
            to 'standard' for built-in methods, or the function's name for callables.

        Notes
        -----
        * **Implementation**: This method sets the `_step_size_callback` attribute
          of the problem instance. Parameters like `ls_max_iter` and `ls_method`
          are stored within the `self.nr_parameters` dictionary.
        * **Objective Function**: For 'Armijo' and 'Quadratic' methods, the solver
          minimizes the squared L2-norm of the residual:

          .. math:: \phi(\alpha) = \frac{1}{2} \|R(u + \alpha dX)\|^2

        * **Work Criterion**: The 'Energy' method minimizes the directional
          derivative of the potential energy (the external work).
        * **Safeguards**: To prevent solver stagnation, interpolated values are
          clipped such that :math:`\alpha_{new} \in [0.1\alpha, 0.5\alpha]`.

        Example
        -------
        >>> # Using a built-in method
        >>> my_problem.add_line_search(method="Quadratic")
        >>> # Using a custom function
        >>> def my_ls(pb, dX): return 0.5
        >>> my_problem.add_line_search(method=my_ls)
        """
        if callable(method):
            cb_name = name or getattr(method, "__name__", "custom_ls")
            self._ls_callbacks[cb_name] = method
        elif method not in ["Residual", "Energy", "Armijo", "Quadratic"]:
            raise ValueError(
                "Line search method should be either 'Residual', "
                "'Energy', 'Armijo' or 'Quadratic'"
            )
        else:
            # Remove any existing entry that uses the standard line_search function
            self._ls_callbacks = {k: v for k, v in self._ls_callbacks.items() if v != line_search}

            self.nr_parameters["ls_method"] = method
            self._ls_callbacks[name or "standard"] = line_search

        self._update_step_size_callback()

    def remove_line_search(self, name=None):
        """Remove a line search algorithm by its name.

        If no name provided, remove all defined algorithm.
        """
        if name is None:
            self._ls_callbacks.clear()
        else:
            removed = self._ls_callbacks.pop(name, None)
            if removed is None:
                warnings.warn(f"Line search '{name}' not found. No action taken.", UserWarning)
                
        self._update_step_size_callback()

    def _get_free_dof_residual(self):
        if not hasattr(self, "_MatCB"):
            dof_free = self._dof_free
            if np.isscalar(self.get_D()) and self.get_D() == 0:
                return self.get_B()[dof_free]
            else:
                return self.get_B()[dof_free] + self.get_D()[dof_free]
        else:
            if np.isscalar(self.get_D()) and self.get_D() == 0:
                return self._MatCB.T @ self.get_B()
            else:
                return self._MatCB.T @ (self.get_B() + self.get_D())

    def compute_nr_error(self):
        """Compute the error of the Newton-Raphson algorithm.

        For Force and Work error criterion, the problem must be updated
        (update method).
        """
        norm_type = self.nr_parameters["norm_type"]
        dof_free = self._dof_free
        if len(dof_free) == 0:
            return 0
        if self.nr_parameters["criterion"] == "Displacement":
            if self._err0 is None:  # assess err0 from current state
                # Normalize by the current increment (not the total
                # accumulated displacement) so the criterion does not
                # become progressively looser as loading proceeds.
                err0 = np.linalg.norm(self._dU, norm_type)
                if not np.array_equal(self._U, 0):
                    # to avoid numerical error related to high U values
                    # compared to dU
                    err0 += 0.01 * np.linalg.norm(self._U, norm_type)
                # err0 += 1e-8*np.max(self.mesh.bounding_box.size)
                if err0 == 0:
                    err0 = 1
                    return 1
            else:
                err0 = self._err0
            return np.linalg.norm(self.get_X()[dof_free], norm_type) / (
                err0 * self._alpha
            )
        elif self.nr_parameters["criterion"] == "Force":
            if self._err0 is None:
                # Normalize by external force
                err0 = np.linalg.norm(
                    self.get_ext_forces(include_mpc=False),
                    norm_type,
                )
                # err0 += 1e-8  # to avoid divizion by 0
                if err0 == 0:
                    err0 = 1
                    return 1
            else:
                err0 = self._err0
            return np.linalg.norm(self._get_free_dof_residual(), norm_type) / err0
        else:  # self.nr_parameters['criterion'] == 'Work': work criterion
            # initialize the value of self._err0
            if self._err0 is None:
                self._err0 = 1
                self._err0 = self.compute_nr_error()
                return 1
            else:
                if np.isscalar(self.get_D()) and self.get_D() == 0:
                    return (
                        np.linalg.norm(
                            self.get_X()[dof_free] * self.get_B()[dof_free],
                            norm_type,
                        )
                        / self._err0
                    )
                else:
                    return (
                        np.linalg.norm(
                            self.get_X()[dof_free]
                            * (self.get_B()[dof_free] + self.get_D()[dof_free]),
                            norm_type,
                        )
                        / self._err0
                    )

    def set_nr_criterion(self, criterion="Displacement", **kargs):
        """
        Define the convergence criterion of the newton raphson algorith.
        For a problem pb, the newton raphson parameters can also be directly set in the
        pb.nr_parameters dict.

        Parameter:
            * criterion: str in ['Displacement', 'Force', 'Work'],
              default = "Displacement".
              Type of convergence test.

        Optional parameters that can be set as kargs:
            * 'err0': float or None, default = None.
              The reference error.
              If None (default), err0 is automatically computed.
            * 'tol': float, default is 5e-3.
              Error tolerance for convergence.
            * 'max_subiter': int, default = 10.
              Number of nr iterations before returning a convergence error.
            * 'dt_increase_niter': int or None, default = None.
              Number of nr iterations threshold that define an easy convergence.
              In problem allowing automatic convergence, if the Newton–Raphson
              loop converges in strictly fewer iterations, the time step is increased.
              If None, defaults to max_subiter//3.
            * 'norm_type': int or numpy.inf, default = 2.
              Define the norm used to test the criterion.
            * 'adaptive_stiffness': bool, default = False.
              Enable adaptive stiffness algorithm with xi-blending between safe
              (often elastic) and tangent stiffness matrices to enhance convergence
              robustness. Only works if the weak form provides an elastic stiffness
              matrix at the beginning of the increment, when set_start is triggered
              (e.g. not for contact problems).
            * 'assume_cvg_at_max_subiter': bool, default = False.
              WARNING: DANGEROUS PARAMETER. If True, assumes convergence when
              max_subiter iterations are reached, regardless of tolerance criterion.
              Use only for special cases requiring forced convergence. Skips
              convergence tolerance check when iteration limit is reached.
            * 'eigenvalue_shift': bool, default = False.
              If True, adds a shifted identity matrix to improve conditioning:
              A_eff = A + alpha*I where alpha = eigenvalue_shift_factor * R.
              R is estimated via Rayleigh quotient of the current stiffness.
            * 'eigenvalue_shift_factor': float, default = 0.1.
              Scaling factor multiplied by the estimated eigenvalue (Rayleigh quotient)
              to determine shift magnitude: alpha = eigenvalue_shift_factor * |R|.
              Larger values = stronger stabilization but less accuracy.
              Smaller values = more accurate but weaker stabilization.
        """
        if criterion not in ["Displacement", "Force", "Work"]:
            raise NameError(
                'criterion must be set to "Displacement", "Force" or "Work"'
            )
        self.nr_parameters["criterion"] = criterion

        for key in kargs:
            if key not in [
                "err0",
                "tol",
                "max_subiter",
                "dt_increase_niter",
                "norm_type",
                "adaptive_stiffness",
                "assume_cvg_at_max_subiter",
                "eigenvalue_shift",
                "eigenvalue_shift_factor",
            ]:
                raise NameError(
                    "Newton Raphson parameters should be in "
                    "['err0', 'tol', 'max_subiter', 'dt_increase_niter', "
                    "'adaptive_stiffness', 'norm_type', 'assume_cvg_at_max_subiter', "
                    "'eigenvalue_shift', 'eigenvalue_shift_factor']"
                )

            self.nr_parameters[key] = kargs[key]

    def elastic_prediction(self):
        # update the boundary conditions with the time variation
        self._alpha = 1
        self.apply_boundary_conditions(self.t_fact, self.t_fact_old)

        # For the elastic prediction, it is more efficient to reuse the tangent
        # matrix from the last converged iteration of the previous time step.
        # A new tangent matrix is computed only for the very first increment,
        # where the matrix has its initial value of 0.
        if np.isscalar(self.get_A()) and self.get_A() == 0:
            self.updateA()

        self.updateD(
            start=True
        )  # not modified in principle if dt is not modified, except the very first iteration. May be optimized by testing the change of dt
        self.solve()

        # OGC per-vertex filter or CCD scalar line search
        if self._step_filter_callback is not None:
            dX = self.get_X()
            self._step_filter_callback(self, dX, is_elastic_prediction=True)
        elif self._step_size_callback is not None:
            dX = self.get_X()
            alpha = self._step_size_callback(self, dX)
            if alpha < 1.0:
                # Scale only free DOFs; preserve prescribed Dirichlet values
                self.set_X(dX * alpha + self._Xbc * (1 - alpha))
                self._alpha = alpha

        # set the increment Dirichlet boundray conditions to 0 (i.e. will not change during the NR interations)
        try:
            self._Xbc *= 0
        except:
            self._ProblemPGD__Xbc = 0

        # update displacement increment
        self._dU += self.get_X()

    def solve_nr_increment(self):
        # solve and update total displacement. A and D should up to date
        self.solve()
        if self._step_filter_callback is not None:
            dX = self.get_X()
            self._step_filter_callback(self, dX, is_elastic_prediction=False)
        elif self._step_size_callback is not None:
            dX = self.get_X()
            alpha = self._step_size_callback(self, dX)
            if alpha < 1.0:
                self.set_X(dX * alpha)
        self._dU += self.get_X()

    def solve_time_increment(self, max_subiter=None, tol_nr=None):
        if max_subiter is None:
            max_subiter = self.nr_parameters["max_subiter"]
        if tol_nr is None:
            tol_nr = self.nr_parameters["tol"]
        assume_cvg_at_max_subiter = self.nr_parameters.get(
            "assume_cvg_at_max_subiter", False
        )

        self._t_fact_inc = self.t_fact
        self.elastic_prediction()

        adaptive_stiffness = self.nr_parameters.get("adaptive_stiffness", False)
        eigenvalue_shift = self.nr_parameters.get("eigenvalue_shift", False)

        update_eigenvalue_shift_factor = False
        if eigenvalue_shift:
            if self.nr_parameters.get("eigenvalue_shift_factor", None) is None:
                update_eigenvalue_shift_factor = True
                if not hasattr(self, "_eigenvalue_shift_factor"):
                    self._eigenvalue_shift_factor = 0.01
            else:
                update_eigenvalue_shift_factor = False
                self._eigenvalue_shift_factor = self.nr_parameters.get(
                    "eigenvalue_shift_factor", 0
                )

        if adaptive_stiffness:
            # we take the assembly computed after set_start and before
            # update("matrix"). This should be the elastic or "safe" stiffness.
            # If notadaptive_stiffness algorithm will not work as expected.
            KE = self.get_A().copy()
            xi = 0.0
            xi_increased_this_step = False

        prev_error = float("inf")
        consecutive_decreases = 0
        consecutive_increases = 0
        subiter = 0
        error = 0.0
        while subiter < max_subiter:
            subiter += 1

            # update Stress and initial displacement and Update stiffness matrix
            self.update(compute="vector")  # update the out of balance force vector
            self.updateD()  # required to compute the NR error

            # Check convergence
            error = self.compute_nr_error()
            if error < tol_nr and subiter >= self._nr_min_subiter:
                self._t_fact_inc = None
                return 1, subiter, error

            if subiter >= max_subiter:
                if assume_cvg_at_max_subiter:
                    self._t_fact_inc = None
                    return 1, subiter, error
                break

            # Track convergence trend
            if error > prev_error:
                consecutive_increases += 1
                consecutive_decreases = 0
            else:
                consecutive_increases = 0
                consecutive_decreases += 1

            if self.print_info > 1:
                print_str = "     Subiter {} - Time: {:.5f} - Err: {:.5f}".format(
                    subiter, self.time + self.dtime, error
                )
                if adaptive_stiffness:
                    print_str += " - xi: {:.4f}".format(xi)
                print(print_str)

            if adaptive_stiffness:
                if consecutive_increases >= 2 and xi < 1.0:
                    # Diverging - switch to elastic stiffness
                    xi = 1.0
                    xi_increased_this_step = True
                    if self.print_info > 1:
                        print("     Diverging. Switching to elastic matrix (xi=1.0).")
                    # redo last iteration
                    self._dU -= self.get_X()
                    consecutive_increases -= 1
                    continue
                elif consecutive_decreases > 0:
                    # Improving - try to reduce xi
                    if xi == 1.0:
                        threshold = 2 if xi_increased_this_step else 3
                        if consecutive_decreases >= threshold:
                            xi = 0.25
                            consecutive_decreases = 0
                    elif xi > 0:
                        xi /= 4.0
                        if xi < 0.0156:
                            xi = 0.0

                prev_error = error

            if update_eigenvalue_shift_factor:
                if error > prev_error:
                    self._eigenvalue_shift_factor *= 5
                    if self._eigenvalue_shift_factor >= 1:
                        self._eigenvalue_shift_factor = 1
                    else:
                        if self.print_info > 1:
                            print("     Diverging. Increase eigenvalue_shift_factor.")
                else:
                    # Improving - try to reduce eigenvalue_shift_factor
                    self._eigenvalue_shift_factor *= 0.95

                prev_error = error

            # --------------- Solve --------------------------------------------------------
            self.update(
                compute="matrix", updateWeakForm=False
            )  # assemble the tangeant matrix

            if adaptive_stiffness:
                KT = self.__assembly.current.get_global_matrix()
                A = xi * KE + (1 - xi) * KT
            else:
                A = self.__assembly.current.get_global_matrix()

            # Apply eigenvalue shift if enabled
            if eigenvalue_shift:
                self.set_A(
                    self._apply_eigenvalue_shift(A, self._eigenvalue_shift_factor)
                )
            else:
                self.set_A(A)

            self.solve_nr_increment()

        self._t_fact_inc = None
        return 0, subiter, error

    def nlsolve(
        self,
        dt: float = 0.1,
        update_dt: bool = True,
        tmax: float | None = None,
        t0: float | None = None,
        dt_min: float = 1e-6,
        max_subiter: int | None = None,
        dt_increase_niter: int | None = None,
        tol_nr: float | None = None,
        print_info: int | None = None,
        save_at_exact_time: bool | None = None,
        interval_output: int | float | None = None,
        callback: Callable[[Problem, ...], None] | None = None,
        exec_callback_at_each_iter: bool | None = None,
    ) -> None:
        """Solve the non linear problem using the newton-raphson algorithm.

        Parameters
        ----------
        dt: float, default=0.1
            Initial time increment
        update_dt: bool, default = True
            If True, the time increment may be modified during resolution:
            * decrease if the solver has not converged
            * increase if the solver has converged quickly (see
              ``dt_increase_niter``).
        tmax: float, optional.
            Time at the end of the time step.
            If omitted, the attribute tmax is considered (default = 1.)
            else, the attribute tmax is modified.
        t0: float, optional.
            Time at the start of the time step.
            If omitted, the attribute t0 is considered (default = 0.)
            else, the attribute t0 is modified.
        dt_min: float, default = 1e-6
            Minimal time increment
        max_subiter: int, optional
            Maximal number of newton raphson iteration allowed for each time
            increment, after the initial linear guess.
            If omitted, the 'max_subiter' field in the nr_parameters
            attribute (ie nr_parameters['max_subiter']) is considered
            (default = 10).
        dt_increase_niter: int, optional
            When ``update_dt`` is ``True``, the time increment is multiplied
            by 1.25 if the Newton–Raphson loop converges in fewer
            than ``dt_increase_niter`` iterations. If omitted, the
            'dt_increase_niter' field in the nr_parameters attribute
            (ie nr_parameters['dt_increase_niter']) is considered
            (default = ``max_subiter // 3``).
            For contact problems where NR typically needs several iterations,
            a higher value (e.g. ``max_subiter // 2``) allows dt to recover
            after an earlier reduction.
        tol_nr: float, optional
            Tolerance of the newton-raphson algorithm.
            If omitted, the 'tol' field in the nr_parameters attribute
            (ie nr_parameters['tol']) is considered (default = 5e-3).
        print_info : int, optional
            Level of information printed to console.
            If 0, nothing is printed
            If 1, iterations info are printed
            If 2, iterations and newton-raphson sub iterations info are printed.
            If omitted, the print_info attribute is considered (default = 1).
        save_at_exact_time: bool, optional
            If True, the time increment is modified to stop at times defined by
            interval_output and allow to save results. If omitted, the
            save_at_exact_time attribute is considered (default = True).
            The given value is stored in the save_at_exact_time attribute.
        interval_output: int|float, optional
            Time step for output if save_at_exact_time is True (default) else
            number of iter increments between 2 output. If
            interval_output == -1, the results is saved at each initial
            time_step intervals or each increment depending on the
            save_at_exact_time value. If omitted, the interval_output attribute
            is considred (default -1)
        callback: function, optional
            The callback function is executed automatically during the non
            linear resolution. By default, the callback function is executed
            when output is requested (defined by the interval_output argument).
            If exec_callback_at_each_iter is True, the callback function is
            excuted at each time iteration.
        exec_callback_at_each_iter, bool, default = False
            If True, the callback function is executed after each time
            iteration.
        """
        # parameters
        if tmax is not None:
            self.tmax = tmax
        if t0 is not None:
            self.t0 = t0  # time at the start of the time step
        if max_subiter is None:
            max_subiter = self.nr_parameters["max_subiter"]
        if dt_increase_niter is None:
            dt_increase_niter = self.nr_parameters["dt_increase_niter"]
            if dt_increase_niter is None:
                dt_increase_niter = max_subiter // 3
        if tol_nr is None:
            tol_nr = self.nr_parameters["tol"]
        if print_info is not None:
            self.print_info = print_info
        if save_at_exact_time is not None:
            self.save_at_exact_time = save_at_exact_time
        if exec_callback_at_each_iter is not None:
            self.exec_callback_at_each_iter = exec_callback_at_each_iter
        if interval_output is None:
            interval_output = self.interval_output  # time step for output if save_at_exact_time == 'True' (default) or  number of iter increments between 2 output

        # if kargs: #not empty
        #    raise TypeError(f"{list(kargs)[0]} is an invalid keyword argument for the method nlsolve")

        if interval_output == -1:
            if self.save_at_exact_time:
                interval_output = dt
            else:
                interval_output = 1

        if self.save_at_exact_time:
            next_time = self.t0 + interval_output
        else:
            next_time = self.tmax  # next_time is the next exact time where the algorithm have to stop for output purpose

        self.init_bc_start_value()

        self.time = self.t0  # time at the begining of the iteration

        if np.isscalar(self._U) and self._U == 0:  # Initialize only if 1st step
            self.initialize()

        restart = False  # bool to know if the iteration is another attempt

        while self.time < self.tmax - self.err_num:
            save_results = (self.time == next_time) or (
                self.save_at_exact_time == False and self.__iter % interval_output == 0
            )

            # update next_time
            if self.time == next_time:
                # self.save_at_exact_time should be True
                next_time = next_time + interval_output
                if next_time > self.tmax - self.err_num:
                    next_time = self.tmax

            if (
                self.time + dt > next_time - self.err_num
            ):  # if dt is too high, it is reduced to reach next_time
                self.dtime = next_time - self.time
            else:
                self.dtime = dt

            if restart:
                # reset internal variables, update Stress, initial displacement and assemble global matrix at previous time
                self.to_start()
                restart = False
            else:
                self.set_start(save_results, callback)

            # self.solve_time_increment = Newton Raphson loop
            convergence, nb_nr_iter, error = self.solve_time_increment(
                max_subiter, tol_nr
            )

            if convergence:
                self.time = self.time + self.dtime  # update time value
                self.__iter += 1

                if self.print_info > 0:
                    print(
                        "Iter {} - Time: {:.5f} - dt {:.5f} - NR iter: {} - Err: {:.5f}".format(
                            self.__iter, self.time, dt, nb_nr_iter, error
                        )
                    )

                # Check if dt can be increased
                if update_dt and nb_nr_iter <= dt_increase_niter and dt == self.dtime:
                    dt *= 1.25

            else:
                if update_dt:
                    dt *= 0.25
                    if self.print_info > 0:
                        print(
                            "NR failed to converge (err: {:.5f}) - reduce the time increment to {:.5f}".format(
                                error, dt
                            )
                        )

                    if dt < dt_min:
                        raise NameError(
                            "Current time step is inferior to the specified minimal time step (dt_min)"
                        )

                    restart = True
                    continue
                else:
                    raise NameError(
                        "Newton Raphson iteration has not converged (err: {:.5f})- Reduce the time step or use update_dt = True".format(
                            error
                        )
                    )

        self.set_start(True, callback)

    # def GetElasticEnergy(self): #only work for classical FEM
    #     """
    #     returns : sum (0.5 * U.transposed * K * U)
    #     """

    #     return sum( 0.5*self.get_X().transpose() * self.get_A() * self.get_X() )

    # def GetNodalElasticEnergy(self):
    #     """
    #     returns : 0.5 * K * U . U
    #     """

    #     E = 0.5*self.get_X().transpose() * self.get_A() * self.get_X()

    #     E = np.reshape(E,(3,-1)).T

    #     return E

    @property
    def assembly(self):
        return self.__assembly

    @property
    def t_fact(self):
        """Adimensional time used for boundary conditions.

        Frozen during ``solve_time_increment`` so that modifications to
        ``self.dtime`` (e.g. by a CCD step-size callback) do not alter
        the target time factor mid-increment.
        """
        if self._t_fact_inc is not None:
            return self._t_fact_inc
        return (self.time + self.dtime - self.t0) / (self.tmax - self.t0)

    @property
    def t_fact_old(self):
        """Previous adimensional time for boundary conditions."""
        return (self.time - self.t0) / (self.tmax - self.t0)


class NonLinear(_NonLinearBase, Problem):
    pass
