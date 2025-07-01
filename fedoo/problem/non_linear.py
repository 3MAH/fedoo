from __future__ import annotations
import numpy as np
from fedoo.core.assembly import Assembly
from fedoo.core.problem import Problem
from typing import Callable, Any


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

        self.nr_parameters = {
            "err0": None,  # default error for NR error estimation
            "criterion": "Displacement",
            "tol": 1e-3,
            "max_subiter": 5,
            "norm_type": 2,
        }
        """
        Parameters to set the newton raphson algorithm:
            * 'err0': The reference error.
              Default is None (automatically computed)
            * 'criterion': Type of convergence test in
              ['Displacement', 'Force', 'Work'].
              Default is 'Displacement'.
            * 'tol': Error tolerance for convergence. Default is 1e-3.
            * 'max_subiter': Number of nr iteration before returning a
              convergence error. Default is 5.
            * 'norm_type': define the norm used to test the criterion
              Use numpy.inf for the max value. Default is 2.
        """

        self.__assembly = assembly
        super().__init__(A, B, D, assembly.mesh, name, assembly.space)
        self.nlgeom = nlgeom
        self.t0 = 0
        self.tmax = 1
        self.time = 0
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

    def elastic_prediction(self):
        # update the boundary conditions with the time variation
        self.apply_boundary_conditions(self.t_fact, self.t_fact_old)

        # build and solve the linearized system with elastic rigidty matrix
        self.updateA()  # should be the elastic rigidity matrix
        self.updateD(
            start=True
        )  # not modified in principle if dt is not modified, except the very first iteration. May be optimized by testing the change of dt
        self.solve()

        # set the increment Dirichlet boundray conditions to 0 (i.e. will not change during the NR interations)
        try:
            self._Xbc *= 0
        except:
            self._ProblemPGD__Xbc = 0

        # update displacement increment
        self._dU += self.get_X()

    def set_start(self, save_results=False, callback=None):
        # dt not used for static problem
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
        self._err0 = self.nr_parameters["err0"]  # initial error for NR error estimation
        self.__assembly.to_start(self)

    def NewtonRaphsonIncrement(self):
        # solve and update total displacement. A and D should up to date
        self.solve()
        self._dU += self.get_X()

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

        B = 0
        self._U = 0
        self._dU = 0

        self._err0 = self.nr_parameters["err0"]  # initial error for NR error estimation
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

    def NewtonRaphsonError(self):
        """
        Compute the error of the Newton-Raphson algorithm
        For Force and Work error criterion, the problem must be updated
        (Update method).
        """
        norm_type = self.nr_parameters["norm_type"]
        dof_free = self._dof_free
        if len(dof_free) == 0:
            return 0
        if self._err0 is None:  # if self._err0 is None -> initialize the value of err0
            # if self.nr_parameters["criterion"] == "Displacement":
            #     self._err0 = np.linalg.norm(
            #         (self._U + self._dU)[dof_free], norm_type
            #     )  # Displacement criterion
            #     if self._err0 == 0:
            #         self._err0 = 1
            #         return 1
            #     return np.max(np.abs(self.get_X()[dof_free])) / self._err0
            # else:
            #     self._err0 = 1
            #     self._err0 = self.NewtonRaphsonError()
            #     return 1
            if self.nr_parameters["criterion"] == "Displacement":
                err0 = np.linalg.norm((self._U + self._dU), norm_type)
                if err0 == 0:
                    err0 = 1
                return np.max(np.abs(self.get_X()[dof_free])) / err0
            else:
                self._err0 = 1
                self._err0 = self.NewtonRaphsonError()
                return 1
        else:
            if self.nr_parameters["criterion"] == "Displacement":
                # return np.max(np.abs(self.get_X()[dof_free]))/self._err0  #Displacement criterion
                return (
                    np.linalg.norm(self.get_X()[dof_free], norm_type) / self._err0
                )  # Displacement criterion
            elif self.nr_parameters["criterion"] == "Force":  # Force criterion
                if np.isscalar(self.get_D()) and self.get_D() == 0:
                    return (
                        np.linalg.norm(self.get_B()[dof_free], norm_type) / self._err0
                    )
                else:
                    return (
                        np.linalg.norm(
                            self.get_B()[dof_free] + self.get_D()[dof_free],
                            norm_type,
                        )
                        / self._err0
                    )
            else:  # self.nr_parameters['criterion'] == 'Work': #work criterion
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
            * 'tol': float, default is 1e-3.
              Error tolerance for convergence.
            * 'max_subiter': int, default = 5.
              Number of nr iteration before returning a convergence error.
            * 'norm_type': int or numpy.inf, default = 2.
              Define the norm used to test the criterion
        """
        if criterion not in ["Displacement", "Force", "Work"]:
            raise NameError(
                'criterion must be set to "Displacement", "Force" or "Work"'
            )
        self.nr_parameters["criterion"] = criterion

        for key in kargs:
            if key not in ["err0", "tol", "max_subiter", "norm_type"]:
                raise NameError(
                    "Newton Raphson parameters should be in "
                    "['err0', 'tol', 'max_subiter', 'norm_type']"
                )

            self.nr_parameters[key] = kargs[key]

    def solve_time_increment(self, max_subiter=None, tol_nr=None):
        if max_subiter is None:
            max_subiter = self.nr_parameters["max_subiter"]
        if tol_nr is None:
            tol_nr = self.nr_parameters["tol"]

        self.elastic_prediction()
        for subiter in range(max_subiter):  # newton-raphson iterations
            # update Stress and initial displacement and Update stiffness matrix
            self.update(compute="vector")  # update the out of balance force vector
            self.updateD()  # required to compute the NR error

            # Check convergence
            normRes = self.NewtonRaphsonError()

            if self.print_info > 1:
                print(
                    "     Subiter {} - Time: {:.5f} - Err: {:.5f}".format(
                        subiter, self.time + self.dtime, normRes
                    )
                )

            if normRes < tol_nr:  # convergence of the NR algorithm
                # Initialize the next increment
                return 1, subiter, normRes

            # --------------- Solve --------------------------------------------------------
            # self.__Assembly.current.assemble_global_mat(compute = 'matrix')
            # self.set_A(self.__Assembly.current.get_global_matrix())
            self.update(
                compute="matrix", updateWeakForm=False
            )  # assemble the tangeant matrix
            self.updateA()

            self.NewtonRaphsonIncrement()

        return 0, subiter, normRes

    def nlsolve(
        self,
        dt: float = 0.1,
        update_dt: bool = True,
        tmax: float | None = None,
        t0: float | None = None,
        dt_min: float = 1e-6,
        max_subiter: int | None = None,
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
            * increase if the solver has converged in one iteration.
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
            Maximal number of newton raphson iteration at for each time increment.
            If omitted, the 'max_subiter' field in the nr_parameters attribute
            (ie nr_parameters['max_subiter']) is considered (default = 5).
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
            If True, the time increment is modified to stop at times defined by interval_output and allow to save results.
            If omitted, the save_at_exact_time attribute is considered (default = True).
            The given value is stored in the save_at_exact_time attribute.
        interval_output: int|float, optional
            Time step for output if save_at_exact_time is True (default) else number of iter increments between 2 output
            If interval_output == -1, the results is saved at each initial time_step intervals or each increment depending on the save_at_exact_time value.
            If omitted, the interval_output attribute is considred (default -1)
        callback: function, optional
            The callback function is executed automatically during the non linear resolution.
            By default, the callback function is executed when output is requested
            (defined by the interval_output argument). If
            exec_callback_at_each_iter is True, the callback function is excuted
            at each time iteration.
        exec_callback_at_each_iter, bool, default = False
            If True, the callback function is executed after each time iteration.
        """

        # parameters
        if tmax is not None:
            self.tmax = tmax
        if t0 is not None:
            self.t0 = t0  # time at the start of the time step
        if max_subiter is None:
            max_subiter = self.nr_parameters["max_subiter"]
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
            convergence, nbNRiter, normRes = self.solve_time_increment(
                max_subiter, tol_nr
            )

            if convergence:
                self.time = self.time + self.dtime  # update time value
                self.__iter += 1

                if self.print_info > 0:
                    print(
                        "Iter {} - Time: {:.5f} - dt {:.5f} - NR iter: {} - Err: {:.5f}".format(
                            self.__iter, self.time, dt, nbNRiter, normRes
                        )
                    )

                # Check if dt can be increased
                if update_dt and nbNRiter < 2 and dt == self.dtime:
                    dt *= 1.25
                    # print('Increase the time increment to {:.5f}'.format(dt))

            else:
                if update_dt:
                    dt *= 0.25
                    if self.print_info > 0:
                        print(
                            "NR failed to converge (err: {:.5f}) - reduce the time increment to {:.5f}".format(
                                normRes, dt
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
                            normRes
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
        """Adimensional time used for boundary conditions."""
        return (self.time + self.dtime - self.t0) / (self.tmax - self.t0)

    @property
    def t_fact_old(self):
        """Previous adimensional time for boundary conditions."""
        return (self.time - self.t0) / (self.tmax - self.t0)


class NonLinear(_NonLinearBase, Problem):
    pass
