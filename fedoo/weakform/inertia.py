from fedoo.core.weakform import WeakFormBase


class Inertia(WeakFormBase):
    """
    Weak formulation related to the inertia effect into dynamical simulation.

    Used among others for :mod:`fedoo.problem.Newmark`,  :mod:`fedoo.problem.NonLinearNewmark`
    or :mod:`fedoo.weakform.ArtificialDamping`

    Parameters
    ----------
    density : scalar or arrays of gauss point values.
        Material density. For 1D meshes, this is physically a linear density
        (mass per unit of length) and for 2D meshes an area density (mass per
        unit of surface).
    name : str, optional
        name of the WeakForm
    space : ModelingSpace, optional
        Modeling space used to build the weak form. Default to active one.
    """

    def __init__(self, density, name="", space=None):
        WeakFormBase.__init__(self, name, space)

        self.space.new_variable("DispX")
        self.space.new_variable("DispY")
        if self.space.ndim == 3:
            self.space.new_variable("DispZ")
            self.space.new_vector("Disp", ("DispX", "DispY", "DispZ"))
        else:
            self.space.new_vector("Disp", ("DispX", "DispY"))

        self.density = density

    def get_weak_equation(self, assembly, pb):
        op_dU = self.space.op_disp()  # displacement increment (incremental formulation)
        op_dU_vir = [du.virtual if du != 0 else 0 for du in op_dU]

        return sum([a * b * self.density for (a, b) in zip(op_dU_vir, op_dU)])


class RotaryInertia(WeakFormBase):
    """
    Rotary inertia effect into dynamical simulation.

    Used among others for :mod:`fedoo.problem.Newmark`,  :mod:`fedoo.problem.NonLinearNewmark`
    or :mod:`fedoo.weakform.ArtificialDamping`

    Parameters
    ----------
    rotary_inertia : list or scalar or arrays of gauss point values.
        mass moment of inertia. For 1D meshes, the rotary inertia is understood
        per unit of length and for 2D meshes per unit of surface.
        If list, defines the inertia along each axis (generally using local
        coordinate system), e.g in 3D:
        [x_rot_inertia, y_rot_inertia, z_rot_inertia]
    name : str, optional
        name of the WeakForm
    space : ModelingSpace, optional
        Modeling space used to build the weak form. Default to active one.
    """

    def __init__(self, rotary_inertia, name="", space=None):
        WeakFormBase.__init__(self, name, space)

        if self.space.ndim == 3:
            self.space.new_variable("RotX")  # torsion rotation
            self.space.new_variable("RotY")
            self.space.new_variable("RotZ")
            self.space.new_vector("Rot", ("RotX", "RotY", "RotZ"))
            n_op = 3
        elif self.space.ndim == 2:
            self.space.new_variable("RotZ")
            self.space.variable_alias("Rot", "RotZ")
            n_op = 1

        if not isinstance(rotary_inertia, list):
            rotary_inertia = [rotary_inertia for i in range(n_op)]
        self.rotary_inertia = rotary_inertia

    def get_weak_equation(self, assembly, pb):
        if self.space.ndim == 2:
            op_rot = [self.space.variable("RotZ")]
        else:
            op_rot = [
                self.space.variable("RotX"),
                self.space.variable("RotY"),
                self.space.variable("RotZ"),
            ]
        op_rot_vir = [op.virtual if op != 0 else 0 for op in op_rot]

        return sum(
            [
                a * b * inertia
                for (inertia, a, b) in zip(self.rotary_inertia, op_rot_vir, op_rot)
            ]
        )
