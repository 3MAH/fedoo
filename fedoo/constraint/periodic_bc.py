# from fedoo.core.base   import ProblemBase
import numpy as np
from fedoo.core.boundary_conditions import BCBase, MPC, ListBC
from fedoo.core.base import ProblemBase
from fedoo.core.mesh import MeshBase

USE_SIMCOON = True

if USE_SIMCOON:
    try:
        from simcoon import simmit as sim

        USE_SIMCOON = True
    except:
        USE_SIMCOON = False
        print(
            "WARNING: Simcoon library not found. The simcoon constitutive law is disabled."
        )

    # def DefinePeriodicBoundaryConditionNonPerioMesh(mesh, node_cd, var_cd, dim='3D', tol=1e-8, Problemname = None, nNeighbours = 3, powInter = 1.0):

    #     if Problemname is None: pb = ProblemBase.get_active()
    #     elif isinstance(Problemname, str): pb = ProblemBase.get_all()[Problemname]
    #     elif isinstance(Problemname, ProblemBase): pb = Problemname #assume Problemname is a Problem Object
    #     else: raise NameError('Problemname not understood')

    #     #Definition of the set of nodes for boundary conditions
    #     if isinstance(mesh, str):
    #         mesh = MeshBase.get_all()[mesh]

    #     if isinstance(var_cd, str):
    #         var_cd = [pb.space.variable_rank(v) for v in var_cd]

    #     coords_nodes = mesh.nodes
    #     if isinstance(node_cd[0], np.int64):
    #         node_cd_int32 = [n.item() for n in node_cd]
    #     else:
    #         node_cd_int32 = [n for n in node_cd]

    #     list_nodes = sim.nonperioMPC(coords_nodes, node_cd_int32, nNeighbours, powInter)

    #     for eq_list in list_nodes:
    #         eq = np.array(eq_list)
    #         list_var = tuple(eq[1::3].astype(int)-1)
    #         pb.bc.mpc(list_var, eq[2::3], eq[0::3].astype(int))


class PeriodicBC(BCBase):
    """Class defining periodic boundary conditions"""

    def __init__(
        self, node_cd, var_cd, dim=None, tol=1e-8, meshperio=True, name="Periodicity"
    ):
        """
        Create a perdiodic boundary condition object using several multi-points constraints.
        Some constraint driver (cd) dof  are used to define mean strain or mean displacement gradient.
        The dof associated to a contraint driver is difined by the node indice (defined in node_cd)
        and the associated variable (defined in var_vd).

        The constraint drivers can be defined in several way
            * [Eps_XX] or [[Eps_XX]] -> strain dof for 1D periodicity
            * [Eps_XX, Eps_YY, 2*Eps_XY] -> strain dof for 2D periodicity using Voigt notation for shear components
            * [Eps_XX, Eps_YY, Eps_ZZ, 2*Eps_XY, 2*Eps_XZ, 2*Eps_YZ] -> strain dof for 3D periodicity using Voigt notation for shear components
            * [[DU_XX, DU_XY], [DU_YX, DU_YY]] -> gradient of displacement for 2D periodicity
            * [[DU_XX, DU_XY, DU_XZ], [DU_YX, DU_YY, DU_YZ], [DU_ZX, DU_ZY, DU_ZZ]]
            -> gradient of displacement for 3D periodicity

        Parameters
        ----------
        node_cd : list of nodes, or list of list of nodes
            Nodes used as constraint drivers for each strain component. The dof used as contraint drivers are defined in var_cd.
        var_cd : list of str, or list of list of str.
            Variables used as constraint drivers. The len of lists should be the same as node_cd.
        dim : int in [1,2,3] (default = assess from the constraint drivers dimension)
            Number of dimensions with periodic condition.
            If dim = 1: the periodicity is assumed along the x coordinate
            If dim = 2: the periodicity is assumed along x and y coordinates

        tol : float, optional
            Tolerance for the periodic nodes detection. The default is 1e-8.
        name : str, optional
            Name of the created boundary condition. The default is "Periodicity".


        Notes
        ---------------

        * The boundary condition object needs to be used with a problem associated to
          a periodic mesh.
        * The periodic nodes are automatically detected using the given tolerance (tol).
        * The nodes of the right (x=xmax), top (y=ymax) and front (z=zmax) faces are
          eliminated from the system (slave nodes) and can't be used in another mpc.

        Example
        ---------

        .. code-block:: python

            import fedoo as fd

            mesh = fd.mesh.box_mesh()

            #add nodes not associated to any element for constraint driver
            node_cd = fd.Mesh["Domain2"].add_nodes(crd_center, 3)

            list_strain_nodes = [StrainNodes[0], StrainNodes[0], StrainNodes[0],
                                 StrainNodes[1], StrainNodes[1], StrainNodes[1]]
            list_strain_var = ['DispX', 'DispY', 'DispZ','DispX', 'DispY', 'DispZ']

            # or using the displacement gradient formulation (in this case the shear strain are true strain component):
            # list_strain_nodes = [[StrainNodes[0], StrainNodes[1], StrainNodes[1]],
            #                      [StrainNodes[1], StrainNodes[0], StrainNodes[1]],
            #                      [StrainNodes[1], StrainNodes[1], StrainNodes[0]]]
            # list_strain_var = [['DispX', 'DispX', 'DispY'],
            #                    ['DispX', 'DispY', 'DispZ'],
            #                    ['DispY', 'DispZ', 'DispZ']]

            bc_periodic = fd.homogen.PeriodicBC(list_strain_nodes, list_strain_var)

        """

        self.shear_coef = 1
        if meshperio:
            if np.isscalar(node_cd[0]):
                self.shear_coef = 0.5
                if len(node_cd) == 1:
                    if dim is None:
                        dim = 1
                    var_cd = [var_cd]
                    node_cd = [node_cd]
                elif len(node_cd) == 3:
                    if dim is None:
                        dim = 2
                    node_cd = [[node_cd[0], node_cd[2]], [node_cd[2], node_cd[1]]]
                    var_cd = [[var_cd[0], var_cd[2]], [var_cd[2], var_cd[1]]]

                elif len(node_cd) == 6:
                    if dim is None:
                        dim = 3
                    node_cd = [
                        [node_cd[0], node_cd[3], node_cd[4]],
                        [node_cd[3], node_cd[1], node_cd[5]],
                        [node_cd[4], node_cd[5], node_cd[2]],
                    ]
                    var_cd = [
                        [var_cd[0], var_cd[3], var_cd[4]],
                        [var_cd[3], var_cd[1], var_cd[5]],
                        [var_cd[4], var_cd[5], var_cd[2]],
                    ]
                else:
                    raise NameError("Lenght of node_cd and var_cd should be 1,3 or 6")

            elif dim is None:
                dim = len(node_cd[0])

        else:
            assert np.isscalar(
                node_cd[0]
            ), "Only small strain tensor can be treated with non periodic mesh"

            self.n_neighbours = 3
            self.pow_inter = 1.0

        self.node_cd = node_cd
        self.var_cd = var_cd
        self.dim = dim  # dimension of periodicity (1, 2 or 3)
        self.tol = tol
        self.bc_type = "PeriodicBC"
        BCBase.__init__(self, name)

        self.meshperio = meshperio

    def __repr__(self):
        list_str = ["{}D Periodic Boundary Condition:".format(self.dim)]
        if self.name != "":
            list_str.append("name = '{}'".format(self.name))

        return "\n".join(list_str)

    def initialize(self, problem):
        mesh = problem.mesh
        tol = self.tol
        var_cd = self.var_cd
        node_cd = self.node_cd

        crd = mesh.nodes

        # ==========================================================
        # =========== Non Periodic Mesh using simcoon function =====
        # ==========================================================
        if self.meshperio == False:
            assert USE_SIMCOON, "Simcoon needs to be installed before using Periodic BC with non perio mesh."

            if isinstance(node_cd[0], np.int64):
                node_cd_int32 = [n.item() for n in node_cd]
            else:
                node_cd_int32 = [n for n in node_cd]

            list_nodes = sim.nonperioMPC(
                crd, node_cd_int32, self.n_neighbours, self.pow_inter
            )

            res = ListBC()
            for eq_list in list_nodes:
                eq = np.array(eq_list)
                list_var = tuple(eq[1::3].astype(int) - 1)
                res.append(MPC(eq[0::3].astype(int), list_var, eq[2::3]))

            res.initialize(problem)
            self.list_mpc = res

            return

        # ==========================================================
        # =========== Create set of nodes ==========================
        # ==========================================================

        xmax = np.max(crd[:, 0])
        xmin = np.min(crd[:, 0])
        ymax = np.max(crd[:, 1])
        ymin = np.min(crd[:, 1])
        if self.dim == 3:
            zmax = np.max(crd[:, 2])
            zmin = np.min(crd[:, 2])

        left = np.where(np.abs(crd[:, 0] - xmin) < tol)[0]
        right = np.where(np.abs(crd[:, 0] - xmax) < tol)[0]

        if self.dim > 1:
            bottom = np.where(np.abs(crd[:, 1] - ymin) < tol)[0]
            top = np.where(np.abs(crd[:, 1] - ymax) < tol)[0]

            # extract edges/corners from the intersection of faces
            left_bottom = np.intersect1d(left, bottom, assume_unique=True)
            left_top = np.intersect1d(left, top, assume_unique=True)
            right_bottom = np.intersect1d(right, bottom, assume_unique=True)
            right_top = np.intersect1d(right, top, assume_unique=True)

            if self.dim > 2:  # or dim == 3
                back = np.where(np.abs(crd[:, 2] - zmin) < tol)[0]
                front = np.where(np.abs(crd[:, 2] - zmax) < tol)[0]

                # extract edges/corners from the intersection of faces
                bottom_back = np.intersect1d(bottom, back, assume_unique=True)
                bottom_front = np.intersect1d(bottom, front, assume_unique=True)
                top_back = np.intersect1d(top, back, assume_unique=True)
                top_front = np.intersect1d(top, front, assume_unique=True)

                left_back = np.intersect1d(left, back, assume_unique=True)
                left_front = np.intersect1d(left, front, assume_unique=True)
                right_back = np.intersect1d(right, back, assume_unique=True)
                right_front = np.intersect1d(right, front, assume_unique=True)

                # extract corners from the intersection of edges
                left_bottom_back = np.intersect1d(
                    left_bottom, bottom_back, assume_unique=True
                )
                left_bottom_front = np.intersect1d(
                    left_bottom, bottom_front, assume_unique=True
                )
                left_top_back = np.intersect1d(left_top, top_back, assume_unique=True)
                left_top_front = np.intersect1d(left_top, top_front, assume_unique=True)
                right_bottom_back = np.intersect1d(
                    right_bottom, bottom_back, assume_unique=True
                )
                right_bottom_front = np.intersect1d(
                    right_bottom, bottom_front, assume_unique=True
                )
                right_top_back = np.intersect1d(right_top, top_back, assume_unique=True)
                right_top_front = np.intersect1d(
                    right_top, top_front, assume_unique=True
                )

                # Remove nodes that beloing to several sets
                all_corners = np.hstack(
                    (
                        left_bottom_back,
                        left_bottom_front,
                        left_top_back,
                        left_top_front,
                        right_bottom_back,
                        right_bottom_front,
                        right_top_back,
                        right_top_front,
                    )
                )

                left_bottom = np.setdiff1d(left_bottom, all_corners, assume_unique=True)
                left_top = np.setdiff1d(left_top, all_corners, assume_unique=True)
                right_bottom = np.setdiff1d(
                    right_bottom, all_corners, assume_unique=True
                )
                right_top = np.setdiff1d(right_top, all_corners, assume_unique=True)

                bottom_back = np.setdiff1d(bottom_back, all_corners, assume_unique=True)
                bottom_front = np.setdiff1d(
                    bottom_front, all_corners, assume_unique=True
                )
                top_back = np.setdiff1d(top_back, all_corners, assume_unique=True)
                top_front = np.setdiff1d(top_front, all_corners, assume_unique=True)

                left_back = np.setdiff1d(left_back, all_corners, assume_unique=True)
                left_front = np.setdiff1d(left_front, all_corners, assume_unique=True)
                right_back = np.setdiff1d(right_back, all_corners, assume_unique=True)
                right_front = np.setdiff1d(right_front, all_corners, assume_unique=True)

                all_edges = np.hstack(
                    (
                        left_bottom,
                        left_top,
                        right_bottom,
                        right_top,
                        bottom_back,
                        bottom_front,
                        top_back,
                        top_front,
                        left_back,
                        left_front,
                        right_back,
                        right_front,
                        all_corners,
                    )
                )

            else:  # dim = 2
                all_edges = np.hstack((left_bottom, left_top, right_bottom, right_top))

            left = np.setdiff1d(left, all_edges, assume_unique=True)
            right = np.setdiff1d(right, all_edges, assume_unique=True)
            bottom = np.setdiff1d(bottom, all_edges, assume_unique=True)
            top = np.setdiff1d(top, all_edges, assume_unique=True)

            if mesh.ndim > 2:  # if there is a z coordinate
                # sort edges (required to assign the good pair of nodes)
                left_bottom = left_bottom[np.argsort(crd[left_bottom, 2])]
                left_top = left_top[np.argsort(crd[left_top, 2])]
                right_bottom = right_bottom[np.argsort(crd[right_bottom, 2])]
                right_top = right_top[np.argsort(crd[right_top, 2])]

            if self.dim > 2:
                back = np.setdiff1d(back, all_edges, assume_unique=True)
                front = np.setdiff1d(front, all_edges, assume_unique=True)

                bottom_back = bottom_back[np.argsort(crd[bottom_back, 0])]
                bottom_front = bottom_front[np.argsort(crd[bottom_front, 0])]
                top_back = top_back[np.argsort(crd[top_back, 0])]
                top_front = top_front[np.argsort(crd[top_front, 0])]

                left_back = left_back[np.argsort(crd[left_back, 1])]
                left_front = left_front[np.argsort(crd[left_front, 1])]
                right_back = right_back[np.argsort(crd[right_back, 1])]
                right_front = right_front[np.argsort(crd[right_front, 1])]

        # sort adjacent faces to ensure node correspondance
        if mesh.ndim == 2:
            left = left[np.argsort(crd[left, 1])]
            right = right[np.argsort(crd[right, 1])]
            if self.dim > 1:
                bottom = bottom[np.argsort(crd[bottom, 0])]
                top = top[np.argsort(crd[top, 0])]

        elif mesh.ndim > 2:
            decimal_round = int(-np.log10(tol) - 1)
            left = left[np.lexsort((crd[left, 1], crd[left, 2].round(decimal_round)))]
            right = right[
                np.lexsort((crd[right, 1], crd[right, 2].round(decimal_round)))
            ]
            if self.dim > 1:
                bottom = bottom[
                    np.lexsort((crd[bottom, 0], crd[bottom, 2].round(decimal_round)))
                ]
                top = top[np.lexsort((crd[top, 0], crd[top, 2].round(decimal_round)))]
            if self.dim > 2:
                back = back[
                    np.lexsort((crd[back, 0], crd[back, 1].round(decimal_round)))
                ]
                front = front[
                    np.lexsort((crd[front, 0], crd[front, 1].round(decimal_round)))
                ]

        # ==========================================================
        # =========== build periodic boudary conditions ============
        # ==========================================================

        list_var = (
            problem.space.list_variables()
        )  # list of variable id defined in the active modeling space

        dx = xmax - xmin
        if self.dim > 1:
            dy = ymax - ymin
        if self.dim > 2:
            dz = zmax - zmin

        sc = self.shear_coef

        res = ListBC()

        # Left/right faces (DispX)
        res.append(
            MPC(
                [right, left, np.full_like(right, node_cd[0][0])],
                ["DispX", "DispX", var_cd[0][0]],
                [
                    np.full_like(right, 1),
                    np.full_like(left, -1),
                    np.full_like(right, -dx, dtype=float),
                ],
            )
        )

        if self.dim > 1:
            # Left/right faces (DispY)
            res.append(
                MPC(
                    [right, left, np.full_like(right, node_cd[1][0])],
                    ["DispY", "DispY", var_cd[1][0]],
                    [
                        np.full_like(right, 1),
                        np.full_like(left, -1),
                        np.full_like(right, -sc * dx, dtype=float),
                    ],
                )
            )

            # top/bottom faces (DispX and DispY)
            res.append(
                MPC(
                    [top, bottom, np.full_like(top, node_cd[0][1])],
                    ["DispX", "DispX", var_cd[0][1]],
                    [
                        np.full_like(top, 1),
                        np.full_like(bottom, -1),
                        np.full_like(top, -sc * dy, dtype=float),
                    ],
                )
            )
            res.append(
                MPC(
                    [top, bottom, np.full_like(top, node_cd[1][1])],
                    ["DispY", "DispY", var_cd[1][1]],
                    [
                        np.full_like(top, 1),
                        np.full_like(bottom, -1),
                        np.full_like(top, -dy, dtype=float),
                    ],
                )
            )

            # elimination of DOF from edge left/top -> edge left/bottom (DispX, DispY)
            res.append(
                MPC(
                    [left_top, left_bottom, np.full_like(left_top, node_cd[0][1])],
                    ["DispX", "DispX", var_cd[0][1]],
                    [
                        np.full_like(left_top, 1),
                        np.full_like(left_bottom, -1),
                        np.full_like(left_top, -sc * dy, dtype=float),
                    ],
                )
            )
            res.append(
                MPC(
                    [left_top, left_bottom, np.full_like(left_top, node_cd[1][1])],
                    ["DispY", "DispY", var_cd[1][1]],
                    [
                        np.full_like(left_top, 1),
                        np.full_like(left_bottom, -1),
                        np.full_like(left_top, -dy, dtype=float),
                    ],
                )
            )
            # elimination of DOF from edge right/bottom -> edge left/bottom (DispX, DispY)
            res.append(
                MPC(
                    [right_bottom, left_bottom, np.full_like(left_top, node_cd[0][0])],
                    ["DispX", "DispX", var_cd[0][0]],
                    [
                        np.full_like(right_bottom, 1),
                        np.full_like(left_bottom, -1),
                        np.full_like(right_bottom, -dx, dtype=float),
                    ],
                )
            )
            res.append(
                MPC(
                    [right_bottom, left_bottom, np.full_like(left_top, node_cd[1][0])],
                    ["DispY", "DispY", var_cd[1][0]],
                    [
                        np.full_like(right_bottom, 1),
                        np.full_like(left_bottom, -1),
                        np.full_like(right_bottom, -sc * dx, dtype=float),
                    ],
                )
            )
            # elimination of DOF from edge right/top -> edge left/bottom (DispX, DispY)
            res.append(
                MPC(
                    [
                        right_top,
                        left_bottom,
                        np.full_like(right_top, node_cd[0][0]),
                        np.full_like(right_top, node_cd[0][1]),
                    ],
                    ["DispX", "DispX", var_cd[0][0], var_cd[0][1]],
                    [
                        np.full_like(right_top, 1),
                        np.full_like(left_bottom, -1),
                        np.full_like(right_top, -dx, dtype=float),
                        np.full_like(right_top, -sc * dy, dtype=float),
                    ],
                )
            )
            res.append(
                MPC(
                    [
                        right_top,
                        left_bottom,
                        np.full_like(right_top, node_cd[1][0]),
                        np.full_like(right_top, node_cd[1][1]),
                    ],
                    ["DispY", "DispY", var_cd[1][0], var_cd[1][1]],
                    [
                        np.full_like(right_top, 1),
                        np.full_like(left_bottom, -1),
                        np.full_like(right_top, -sc * dx, dtype=float),
                        np.full_like(right_top, -dy, dtype=float),
                    ],
                )
            )

        if self.dim > 2:
            # DispZ for Left/right faces
            res.append(
                MPC(
                    [right, left, np.full_like(right, node_cd[2][0])],
                    ["DispZ", "DispZ", var_cd[2][0]],
                    [
                        np.full_like(right, 1),
                        np.full_like(left, -1),
                        np.full_like(right, -sc * dx, dtype=float),
                    ],
                )
            )
            # DispZ for top/bottom faces
            res.append(
                MPC(
                    [top, bottom, np.full_like(top, node_cd[2][1])],
                    ["DispZ", "DispZ", var_cd[2][1]],
                    [
                        np.full_like(top, 1),
                        np.full_like(bottom, -1),
                        np.full_like(top, -sc * dy, dtype=float),
                    ],
                )
            )

            # elimination of DOF from edge left/top -> edge left/bottom (DispZ)
            res.append(
                MPC(
                    [left_top, left_bottom, np.full_like(left_top, node_cd[2][1])],
                    ["DispZ", "DispZ", var_cd[2][1]],
                    [
                        np.full_like(left_top, 1),
                        np.full_like(left_bottom, -1),
                        np.full_like(left_top, -sc * dy, dtype=float),
                    ],
                )
            )
            # elimination of DOF from edge right/bottom -> edge left/bottom (DispZ)
            res.append(
                MPC(
                    [right_bottom, left_bottom, np.full_like(left_top, node_cd[2][0])],
                    ["DispZ", "DispZ", var_cd[2][0]],
                    [
                        np.full_like(right_bottom, 1),
                        np.full_like(left_bottom, -1),
                        np.full_like(right_bottom, -sc * dx, dtype=float),
                    ],
                )
            )
            # elimination of DOF from edge right/top -> edge left/bottom (DispZ)
            res.append(
                MPC(
                    [
                        right_top,
                        left_bottom,
                        np.full_like(right_top, node_cd[2][0]),
                        np.full_like(right_top, node_cd[2][1]),
                    ],
                    ["DispZ", "DispZ", var_cd[2][0], var_cd[2][1]],
                    [
                        np.full_like(right_top, 1),
                        np.full_like(left_bottom, -1),
                        np.full_like(right_top, -sc * dx, dtype=float),
                        np.full_like(right_top, -sc * dy, dtype=float),
                    ],
                )
            )

            # front/back faces
            res.append(
                MPC(
                    [front, back, np.full_like(front, node_cd[0][2])],
                    ["DispX", "DispX", var_cd[0][2]],
                    [
                        np.full_like(front, 1),
                        np.full_like(back, -1),
                        np.full_like(front, -sc * dz, dtype=float),
                    ],
                )
            )
            res.append(
                MPC(
                    [front, back, np.full_like(front, node_cd[1][2])],
                    ["DispY", "DispY", var_cd[1][2]],
                    [
                        np.full_like(front, 1),
                        np.full_like(back, -1),
                        np.full_like(front, -sc * dz, dtype=float),
                    ],
                )
            )
            res.append(
                MPC(
                    [front, back, np.full_like(front, node_cd[2][2])],
                    ["DispZ", "DispZ", var_cd[2][2]],
                    [
                        np.full_like(front, 1),
                        np.full_like(back, -1),
                        np.full_like(front, -dz, dtype=float),
                    ],
                )
            )

            # elimination of DOF from edge top/back -> edge bottom/back
            res.append(
                MPC(
                    [top_back, bottom_back, np.full_like(top_back, node_cd[0][1])],
                    ["DispX", "DispX", var_cd[0][1]],
                    [
                        np.full_like(top_back, 1),
                        np.full_like(bottom_back, -1),
                        np.full_like(top_back, -sc * dy, dtype=float),
                    ],
                )
            )
            res.append(
                MPC(
                    [top_back, bottom_back, np.full_like(top_back, node_cd[1][1])],
                    ["DispY", "DispY", var_cd[1][1]],
                    [
                        np.full_like(top_back, 1),
                        np.full_like(bottom_back, -1),
                        np.full_like(top_back, -dy, dtype=float),
                    ],
                )
            )
            res.append(
                MPC(
                    [top_back, bottom_back, np.full_like(top_back, node_cd[2][1])],
                    ["DispZ", "DispZ", var_cd[2][1]],
                    [
                        np.full_like(top_back, 1),
                        np.full_like(bottom_back, -1),
                        np.full_like(top_back, -sc * dy, dtype=float),
                    ],
                )
            )
            # elimination of DOF from edge bottom/front -> edge bottom/back
            res.append(
                MPC(
                    [
                        bottom_front,
                        bottom_back,
                        np.full_like(bottom_front, node_cd[0][2]),
                    ],
                    ["DispX", "DispX", var_cd[0][2]],
                    [
                        np.full_like(bottom_front, 1),
                        np.full_like(bottom_back, -1),
                        np.full_like(bottom_front, -sc * dz, dtype=float),
                    ],
                )
            )
            res.append(
                MPC(
                    [
                        bottom_front,
                        bottom_back,
                        np.full_like(bottom_front, node_cd[1][2]),
                    ],
                    ["DispY", "DispY", var_cd[1][2]],
                    [
                        np.full_like(bottom_front, 1),
                        np.full_like(bottom_back, -1),
                        np.full_like(bottom_front, -sc * dz, dtype=float),
                    ],
                )
            )
            res.append(
                MPC(
                    [
                        bottom_front,
                        bottom_back,
                        np.full_like(bottom_front, node_cd[2][2]),
                    ],
                    ["DispZ", "DispZ", var_cd[2][2]],
                    [
                        np.full_like(bottom_front, 1),
                        np.full_like(bottom_back, -1),
                        np.full_like(bottom_front, -dz, dtype=float),
                    ],
                )
            )
            # elimination of DOF from edge top/front -> edge bottom/back
            res.append(
                MPC(
                    [
                        top_front,
                        bottom_back,
                        np.full_like(top_front, node_cd[0][1]),
                        np.full_like(top_front, node_cd[0][2]),
                    ],
                    ["DispX", "DispX", var_cd[0][1], var_cd[0][2]],
                    [
                        np.full_like(top_front, 1),
                        np.full_like(bottom_back, -1),
                        np.full_like(top_front, -sc * dy, dtype=float),
                        np.full_like(top_front, -sc * dz, dtype=float),
                    ],
                )
            )
            res.append(
                MPC(
                    [
                        top_front,
                        bottom_back,
                        np.full_like(top_front, node_cd[1][1]),
                        np.full_like(top_front, node_cd[1][2]),
                    ],
                    ["DispY", "DispY", var_cd[1][1], var_cd[1][2]],
                    [
                        np.full_like(top_front, 1),
                        np.full_like(bottom_back, -1),
                        np.full_like(top_front, -dy, dtype=float),
                        np.full_like(top_front, -sc * dz, dtype=float),
                    ],
                )
            )
            res.append(
                MPC(
                    [
                        top_front,
                        bottom_back,
                        np.full_like(top_front, node_cd[2][1]),
                        np.full_like(top_front, node_cd[2][2]),
                    ],
                    ["DispZ", "DispZ", var_cd[2][1], var_cd[2][2]],
                    [
                        np.full_like(top_front, 1),
                        np.full_like(bottom_back, -1),
                        np.full_like(top_front, -sc * dy, dtype=float),
                        np.full_like(top_front, -dz, dtype=float),
                    ],
                )
            )

            # elimination of DOF from edge right/back -> edge left/back
            res.append(
                MPC(
                    [right_back, left_back, np.full_like(left_back, node_cd[0][0])],
                    ["DispX", "DispX", var_cd[0][0]],
                    [
                        np.full_like(right_back, 1),
                        np.full_like(left_back, -1),
                        np.full_like(right_back, -dx, dtype=float),
                    ],
                )
            )
            res.append(
                MPC(
                    [right_back, left_back, np.full_like(left_back, node_cd[1][0])],
                    ["DispY", "DispY", var_cd[1][0]],
                    [
                        np.full_like(right_back, 1),
                        np.full_like(left_back, -1),
                        np.full_like(right_back, -sc * dx, dtype=float),
                    ],
                )
            )
            res.append(
                MPC(
                    [right_back, left_back, np.full_like(left_back, node_cd[2][0])],
                    ["DispZ", "DispZ", var_cd[2][0]],
                    [
                        np.full_like(right_back, 1),
                        np.full_like(left_back, -1),
                        np.full_like(right_back, -sc * dx, dtype=float),
                    ],
                )
            )
            # elimination of DOF from edge left/front -> edge left/back
            res.append(
                MPC(
                    [left_front, left_back, np.full_like(left_back, node_cd[0][2])],
                    ["DispX", "DispX", var_cd[0][2]],
                    [
                        np.full_like(left_front, 1),
                        np.full_like(left_back, -1),
                        np.full_like(right_back, -sc * dz, dtype=float),
                    ],
                )
            )
            res.append(
                MPC(
                    [left_front, left_back, np.full_like(left_back, node_cd[1][2])],
                    ["DispY", "DispY", var_cd[1][2]],
                    [
                        np.full_like(left_front, 1),
                        np.full_like(left_back, -1),
                        np.full_like(right_back, -sc * dz, dtype=float),
                    ],
                )
            )
            res.append(
                MPC(
                    [left_front, left_back, np.full_like(left_back, node_cd[2][2])],
                    ["DispZ", "DispZ", var_cd[2][2]],
                    [
                        np.full_like(left_front, 1),
                        np.full_like(left_back, -1),
                        np.full_like(right_back, -dz, dtype=float),
                    ],
                )
            )
            # elimination of DOF from edge right/front -> edge left/back
            res.append(
                MPC(
                    [
                        right_front,
                        left_back,
                        np.full_like(right_front, node_cd[0][0]),
                        np.full_like(right_front, node_cd[0][2]),
                    ],
                    ["DispX", "DispX", var_cd[0][0], var_cd[0][2]],
                    [
                        np.full_like(right_front, 1),
                        np.full_like(left_back, -1),
                        np.full_like(right_front, -dx, dtype=float),
                        np.full_like(right_front, -sc * dz, dtype=float),
                    ],
                )
            )
            res.append(
                MPC(
                    [
                        right_front,
                        left_back,
                        np.full_like(right_front, node_cd[1][0]),
                        np.full_like(right_front, node_cd[1][2]),
                    ],
                    ["DispY", "DispY", var_cd[1][0], var_cd[1][2]],
                    [
                        np.full_like(right_front, 1),
                        np.full_like(left_back, -1),
                        np.full_like(right_front, -sc * dx, dtype=float),
                        np.full_like(right_front, -sc * dz, dtype=float),
                    ],
                )
            )
            res.append(
                MPC(
                    [
                        right_front,
                        left_back,
                        np.full_like(right_front, node_cd[2][0]),
                        np.full_like(right_front, node_cd[2][2]),
                    ],
                    ["DispZ", "DispZ", var_cd[2][0], var_cd[2][2]],
                    [
                        np.full_like(right_front, 1),
                        np.full_like(left_back, -1),
                        np.full_like(right_front, -sc * dx, dtype=float),
                        np.full_like(right_front, -dz, dtype=float),
                    ],
                )
            )

            # #### CORNER ####
            # elimination of DOF from corner right/bottom/back (right_bottom_back) -> corner left/bottom/back (left_bottom_back)
            res.append(
                MPC(
                    [
                        right_bottom_back,
                        left_bottom_back,
                        np.full_like(right_bottom_back, node_cd[0][0]),
                    ],
                    ["DispX", "DispX", var_cd[0][0]],
                    [
                        np.full_like(right_bottom_back, 1),
                        np.full_like(left_bottom_back, -1),
                        np.full_like(right_bottom_back, -dx, dtype=float),
                    ],
                )
            )
            res.append(
                MPC(
                    [
                        right_bottom_back,
                        left_bottom_back,
                        np.full_like(right_bottom_back, node_cd[1][0]),
                    ],
                    ["DispY", "DispY", var_cd[1][0]],
                    [
                        np.full_like(right_bottom_back, 1),
                        np.full_like(left_bottom_back, -1),
                        np.full_like(right_bottom_back, -sc * dx, dtype=float),
                    ],
                )
            )
            res.append(
                MPC(
                    [
                        right_bottom_back,
                        left_bottom_back,
                        np.full_like(right_bottom_back, node_cd[2][0]),
                    ],
                    ["DispZ", "DispZ", var_cd[2][0]],
                    [
                        np.full_like(right_bottom_back, 1),
                        np.full_like(left_bottom_back, -1),
                        np.full_like(right_bottom_back, -sc * dx, dtype=float),
                    ],
                )
            )
            # elimination of DOF from corner left/top/back (left_top_back) -> corner left/bottom/back (left_bottom_back)
            res.append(
                MPC(
                    [
                        left_top_back,
                        left_bottom_back,
                        np.full_like(left_top_back, node_cd[0][1]),
                    ],
                    ["DispX", "DispX", var_cd[0][1]],
                    [
                        np.full_like(left_top_back, 1),
                        np.full_like(left_bottom_back, -1),
                        np.full_like(left_top_back, -sc * dy, dtype=float),
                    ],
                )
            )
            res.append(
                MPC(
                    [
                        left_top_back,
                        left_bottom_back,
                        np.full_like(left_top_back, node_cd[1][1]),
                    ],
                    ["DispY", "DispY", var_cd[1][1]],
                    [
                        np.full_like(left_top_back, 1),
                        np.full_like(left_bottom_back, -1),
                        np.full_like(left_top_back, -dy, dtype=float),
                    ],
                )
            )
            res.append(
                MPC(
                    [
                        left_top_back,
                        left_bottom_back,
                        np.full_like(left_top_back, node_cd[2][1]),
                    ],
                    ["DispZ", "DispZ", var_cd[2][1]],
                    [
                        np.full_like(left_top_back, 1),
                        np.full_like(left_bottom_back, -1),
                        np.full_like(left_top_back, -sc * dy, dtype=float),
                    ],
                )
            )
            # elimination of DOF from corner left/bottom/front (left_bottom_front) -> corner left/bottom/back (left_bottom_back)
            res.append(
                MPC(
                    [
                        left_bottom_front,
                        left_bottom_back,
                        np.full_like(left_bottom_front, node_cd[0][2]),
                    ],
                    ["DispX", "DispX", var_cd[0][2]],
                    [
                        np.full_like(left_bottom_front, 1),
                        np.full_like(left_bottom_back, -1),
                        np.full_like(left_bottom_front, -sc * dz, dtype=float),
                    ],
                )
            )
            res.append(
                MPC(
                    [
                        left_bottom_front,
                        left_bottom_back,
                        np.full_like(left_bottom_front, node_cd[1][2]),
                    ],
                    ["DispY", "DispY", var_cd[1][2]],
                    [
                        np.full_like(left_bottom_front, 1),
                        np.full_like(left_bottom_back, -1),
                        np.full_like(left_bottom_front, -sc * dz, dtype=float),
                    ],
                )
            )
            res.append(
                MPC(
                    [
                        left_bottom_front,
                        left_bottom_back,
                        np.full_like(left_bottom_front, node_cd[2][2]),
                    ],
                    ["DispZ", "DispZ", var_cd[2][2]],
                    [
                        np.full_like(left_bottom_front, 1),
                        np.full_like(left_bottom_back, -1),
                        np.full_like(left_bottom_front, -dz, dtype=float),
                    ],
                )
            )
            # elimination of DOF from corner right/top/back (right_top_back) -> corner left/bottom/back (left_bottom_back)
            res.append(
                MPC(
                    [
                        right_top_back,
                        left_bottom_back,
                        np.full_like(right_top_back, node_cd[0][0]),
                        np.full_like(right_top_back, node_cd[0][1]),
                    ],
                    ["DispX", "DispX", var_cd[0][0], var_cd[0][1]],
                    [
                        np.full_like(right_top_back, 1),
                        np.full_like(left_bottom_back, -1),
                        np.full_like(right_top_back, -dx, dtype=float),
                        np.full_like(right_top_back, -sc * dy, dtype=float),
                    ],
                )
            )
            res.append(
                MPC(
                    [
                        right_top_back,
                        left_bottom_back,
                        np.full_like(right_top_back, node_cd[1][0]),
                        np.full_like(right_top_back, node_cd[1][1]),
                    ],
                    ["DispY", "DispY", var_cd[1][0], var_cd[1][1]],
                    [
                        np.full_like(right_top_back, 1),
                        np.full_like(left_bottom_back, -1),
                        np.full_like(right_top_back, -sc * dx, dtype=float),
                        np.full_like(right_top_back, -dy, dtype=float),
                    ],
                )
            )
            res.append(
                MPC(
                    [
                        right_top_back,
                        left_bottom_back,
                        np.full_like(right_top_back, node_cd[2][0]),
                        np.full_like(right_top_back, node_cd[2][1]),
                    ],
                    ["DispZ", "DispZ", var_cd[2][0], var_cd[2][1]],
                    [
                        np.full_like(right_top_back, 1),
                        np.full_like(left_bottom_back, -1),
                        np.full_like(right_top_back, -sc * dx, dtype=float),
                        np.full_like(right_top_back, -sc * dy, dtype=float),
                    ],
                )
            )
            # elimination of DOF from corner left/top/front (left_top_front) -> corner left/bottom/back (left_bottom_back)
            res.append(
                MPC(
                    [
                        left_top_front,
                        left_bottom_back,
                        np.full_like(left_top_front, node_cd[0][1]),
                        np.full_like(left_top_front, node_cd[0][2]),
                    ],
                    ["DispX", "DispX", var_cd[0][1], var_cd[0][2]],
                    [
                        np.full_like(left_top_front, 1),
                        np.full_like(left_bottom_back, -1),
                        np.full_like(left_top_front, -sc * dy, dtype=float),
                        np.full_like(left_top_front, -sc * dz, dtype=float),
                    ],
                )
            )
            res.append(
                MPC(
                    [
                        left_top_front,
                        left_bottom_back,
                        np.full_like(left_top_front, node_cd[1][1]),
                        np.full_like(left_top_front, node_cd[1][2]),
                    ],
                    ["DispY", "DispY", var_cd[1][1], var_cd[1][2]],
                    [
                        np.full_like(left_top_front, 1),
                        np.full_like(left_bottom_back, -1),
                        np.full_like(left_top_front, -dy, dtype=float),
                        np.full_like(left_top_front, -sc * dz, dtype=float),
                    ],
                )
            )
            res.append(
                MPC(
                    [
                        left_top_front,
                        left_bottom_back,
                        np.full_like(left_top_front, node_cd[2][1]),
                        np.full_like(left_top_front, node_cd[2][2]),
                    ],
                    ["DispZ", "DispZ", var_cd[2][1], var_cd[2][2]],
                    [
                        np.full_like(left_top_front, 1),
                        np.full_like(left_bottom_back, -1),
                        np.full_like(left_top_front, -sc * dy, dtype=float),
                        np.full_like(left_top_front, -dz, dtype=float),
                    ],
                )
            )
            # elimination of DOF from corner right/bottom/front (right_bottom_front) -> corner left/bottom/back (left_bottom_back)
            res.append(
                MPC(
                    [
                        right_bottom_front,
                        left_bottom_back,
                        np.full_like(right_bottom_front, node_cd[0][0]),
                        np.full_like(right_bottom_front, node_cd[0][2]),
                    ],
                    ["DispX", "DispX", var_cd[0][0], var_cd[0][2]],
                    [
                        np.full_like(right_bottom_front, 1),
                        np.full_like(left_bottom_back, -1),
                        np.full_like(right_bottom_front, -dx, dtype=float),
                        np.full_like(right_bottom_front, -sc * dz, dtype=float),
                    ],
                )
            )
            res.append(
                MPC(
                    [
                        right_bottom_front,
                        left_bottom_back,
                        np.full_like(right_bottom_front, node_cd[1][0]),
                        np.full_like(right_bottom_front, node_cd[1][2]),
                    ],
                    ["DispY", "DispY", var_cd[1][0], var_cd[1][2]],
                    [
                        np.full_like(right_bottom_front, 1),
                        np.full_like(left_bottom_back, -1),
                        np.full_like(right_bottom_front, -sc * dx, dtype=float),
                        np.full_like(right_bottom_front, -sc * dz, dtype=float),
                    ],
                )
            )
            res.append(
                MPC(
                    [
                        right_bottom_front,
                        left_bottom_back,
                        np.full_like(right_bottom_front, node_cd[2][0]),
                        np.full_like(right_bottom_front, node_cd[2][2]),
                    ],
                    ["DispZ", "DispZ", var_cd[2][0], var_cd[2][2]],
                    [
                        np.full_like(right_bottom_front, 1),
                        np.full_like(left_bottom_back, -1),
                        np.full_like(right_bottom_front, -sc * dx, dtype=float),
                        np.full_like(right_bottom_front, -dz, dtype=float),
                    ],
                )
            )

            # elimination of DOF from corner right/top/front (right_top_front) -> corner left/bottom/back (left_bottom_back)
            res.append(
                MPC(
                    [
                        right_top_front,
                        left_bottom_back,
                        np.full_like(right_top_front, node_cd[0][0]),
                        np.full_like(right_top_front, node_cd[0][1]),
                        np.full_like(right_top_front, node_cd[0][2]),
                    ],
                    ["DispX", "DispX", var_cd[0][0], var_cd[0][1], var_cd[0][2]],
                    [
                        np.full_like(right_top_front, 1),
                        np.full_like(left_bottom_back, -1),
                        np.full_like(right_top_front, -dx, dtype=float),
                        np.full_like(right_top_front, -sc * dy, dtype=float),
                        np.full_like(right_top_front, -sc * dz, dtype=float),
                    ],
                )
            )
            res.append(
                MPC(
                    [
                        right_top_front,
                        left_bottom_back,
                        np.full_like(right_top_front, node_cd[1][0]),
                        np.full_like(right_top_front, node_cd[1][1]),
                        np.full_like(right_top_front, node_cd[1][2]),
                    ],
                    ["DispY", "DispY", var_cd[1][0], var_cd[1][1], var_cd[1][2]],
                    [
                        np.full_like(right_top_front, 1),
                        np.full_like(left_bottom_back, -1),
                        np.full_like(right_top_front, -sc * dx, dtype=float),
                        np.full_like(right_top_front, -dy, dtype=float),
                        np.full_like(right_top_front, -sc * dz, dtype=float),
                    ],
                )
            )
            res.append(
                MPC(
                    [
                        right_top_front,
                        left_bottom_back,
                        np.full_like(right_top_front, node_cd[2][0]),
                        np.full_like(right_top_front, node_cd[2][1]),
                        np.full_like(right_top_front, node_cd[2][2]),
                    ],
                    ["DispZ", "DispZ", var_cd[2][0], var_cd[2][1], var_cd[2][2]],
                    [
                        np.full_like(right_top_front, 1),
                        np.full_like(left_bottom_back, -1),
                        np.full_like(right_top_front, -sc * dx, dtype=float),
                        np.full_like(right_top_front, -sc * dy, dtype=float),
                        np.full_like(right_top_front, -dz, dtype=float),
                    ],
                )
            )

        # if rot DOF are used, apply continuity of the rotational dof
        list_rot_var = []
        if "RotX" in list_var:
            list_rot_var.append("RotX")
        if "RotY" in list_var:
            list_rot_var.append("RotY")
        if "RotZ" in list_var:
            list_rot_var.append("RotZ")

        # also applied continuity to non used disp component
        if self.dim < 3 and "DispZ" in list_var:
            list_rot_var.append("DispZ")
        if self.dim == 1 and "DispY" in list_var:
            list_rot_var.append("DispY")

        for var in list_rot_var:
            #### FACES ####
            res.append(
                MPC(
                    [right, left],
                    [var, var],
                    [np.full_like(right, 1), np.full_like(left, -1)],
                )
            )
            if self.dim > 1:
                res.append(
                    MPC(
                        [top, bottom],
                        [var, var],
                        [np.full_like(top, 1), np.full_like(bottom, -1)],
                    )
                )
            if self.dim > 2:
                res.append(
                    MPC(
                        [front, back],
                        [var, var],
                        [np.full_like(front, 1), np.full_like(back, -1)],
                    )
                )

            #### EDGES ####
            if self.dim > 1:
                res.append(
                    MPC(
                        [left_top, left_bottom],
                        [var, var],
                        [np.full_like(left_top, 1), np.full_like(left_bottom, -1)],
                    )
                )
                res.append(
                    MPC(
                        [right_bottom, left_bottom],
                        [var, var],
                        [np.full_like(right_bottom, 1), np.full_like(left_bottom, -1)],
                    )
                )
                res.append(
                    MPC(
                        [right_top, left_bottom],
                        [var, var],
                        [np.full_like(right_top, 1), np.full_like(left_bottom, -1)],
                    )
                )

            if self.dim > 2:
                res.append(
                    MPC(
                        [top_back, bottom_back],
                        [var, var],
                        [np.full_like(top_back, 1), np.full_like(bottom_back, -1)],
                    )
                )
                res.append(
                    MPC(
                        [bottom_front, bottom_back],
                        [var, var],
                        [np.full_like(bottom_front, 1), np.full_like(bottom_back, -1)],
                    )
                )
                res.append(
                    MPC(
                        [top_front, bottom_back],
                        [var, var],
                        [np.full_like(top_front, 1), np.full_like(bottom_back, -1)],
                    )
                )

                res.append(
                    MPC(
                        [right_back, left_back],
                        [var, var],
                        [np.full_like(right_back, 1), np.full_like(left_back, -1)],
                    )
                )
                res.append(
                    MPC(
                        [left_front, left_back],
                        [var, var],
                        [np.full_like(left_front, 1), np.full_like(left_back, -1)],
                    )
                )
                res.append(
                    MPC(
                        [right_front, left_back],
                        [var, var],
                        [np.full_like(right_front, 1), np.full_like(left_back, -1)],
                    )
                )

                #### CORNERS ####
                res.append(
                    MPC(
                        [right_bottom_back, left_bottom_back],
                        [var, var],
                        [
                            np.full_like(right_bottom_back, 1),
                            np.full_like(left_bottom_back, -1),
                        ],
                    )
                )
                res.append(
                    MPC(
                        [left_top_back, left_bottom_back],
                        [var, var],
                        [
                            np.full_like(left_top_back, 1),
                            np.full_like(left_bottom_back, -1),
                        ],
                    )
                )
                res.append(
                    MPC(
                        [left_bottom_front, left_bottom_back],
                        [var, var],
                        [
                            np.full_like(left_bottom_front, 1),
                            np.full_like(left_bottom_back, -1),
                        ],
                    )
                )
                res.append(
                    MPC(
                        [right_top_back, left_bottom_back],
                        [var, var],
                        [
                            np.full_like(right_top_back, 1),
                            np.full_like(left_bottom_back, -1),
                        ],
                    )
                )
                res.append(
                    MPC(
                        [left_top_front, left_bottom_back],
                        [var, var],
                        [
                            np.full_like(left_top_front, 1),
                            np.full_like(left_bottom_back, -1),
                        ],
                    )
                )
                res.append(
                    MPC(
                        [right_bottom_front, left_bottom_back],
                        [var, var],
                        [
                            np.full_like(right_bottom_front, 1),
                            np.full_like(left_bottom_back, -1),
                        ],
                    )
                )
                res.append(
                    MPC(
                        [right_top_front, left_bottom_back],
                        [var, var],
                        [
                            np.full_like(right_top_front, 1),
                            np.full_like(left_bottom_back, -1),
                        ],
                    )
                )

        res.initialize(problem)
        self.list_mpc = res

        # if self.dim == 2:
        #     res.append(MPC(['DispX','DispX',var_cd[0][0]], [np.full_like(right,1), np.full_like(left,  -1), np.full_like(right,-(xmax-xmin), dtype=float)], [right,left  ,np.full_like(right,node_cd[0][0])]))
        #     res.append(MPC(['DispY','DispY',var_cd[1][0]], [np.full_like(right,1), np.full_like(left,  -1), np.full_like(right,-(xmax-xmin), dtype=float)], [right,left  ,np.full_like(right,node_cd[1][0])]))
        #     res.append(MPC(['DispX','DispX',var_cd[0][1]], [np.full_like(top  ,1), np.full_like(bottom,-1), np.full_like(top  ,-(ymax-ymin), dtype=float)], [top  ,bottom,np.full_like(top  ,node_cd[0][1])]))
        #     res.append(MPC(['DispY','DispY',var_cd[1][1]], [np.full_like(top  ,1), np.full_like(bottom,-1), np.full_like(top  ,-(ymax-ymin), dtype=float)], [top  ,bottom,np.full_like(top  ,node_cd[1][1])]))

        #     #elimination of DOF from edge left/top -> edge left/bottom
        #     res.append(MPC(['DispY','DispY',var_cd[1][1]], [np.full_like(left_top,1), np.full_like(left_bottom,-1), np.full_like(left_top,-(ymax-ymin), dtype=float)], [left_top, left_bottom, np.full_like(left_top,node_cd[1][1])]))
        #     res.append(MPC(['DispX','DispX',var_cd[0][1]], [np.full_like(left_top,1), np.full_like(left_bottom,-1), np.full_like(left_top,-(ymax-ymin), dtype=float)], [left_top, left_bottom, np.full_like(left_top,node_cd[0][1])]))
        #     #elimination of DOF from edge right/bottom -> edge left/bottom
        #     res.append(MPC(['DispX','DispX',var_cd[0][0]], [np.full_like(right_bottom,1), np.full_like(left_bottom,-1), np.full_like(right_bottom,-(xmax-xmin), dtype=float)], [right_bottom, left_bottom, np.full_like(left_top,node_cd[0][0])]))
        #     res.append(MPC(['DispY','DispY',var_cd[1][0]], [np.full_like(right_bottom,1), np.full_like(left_bottom,-1), np.full_like(right_bottom,-(xmax-xmin), dtype=float)], [right_bottom, left_bottom, np.full_like(left_top,node_cd[1][0])]))
        #     #elimination of DOF from edge right/top -> edge left/bottom
        #     res.append(MPC(['DispX','DispX',var_cd[0][0],var_cd[0][1]], [np.full_like(right_top,1), np.full_like(left_bottom,-1), np.full_like(right_top,-(xmax-xmin), dtype=float), np.full_like(right_top,-(ymax-ymin), dtype=float)], [right_top, left_bottom, np.full_like(right_top,node_cd[0][0]), np.full_like(right_top,node_cd[0][1])]))
        #     res.append(MPC(['DispY','DispY',var_cd[1][0],var_cd[1][1]], [np.full_like(right_top,1), np.full_like(left_bottom,-1), np.full_like(right_top,-(xmax-xmin), dtype=float), np.full_like(right_top,-(ymax-ymin), dtype=float)], [right_top, left_bottom, np.full_like(right_top,node_cd[1][0]), np.full_like(right_top,node_cd[1][1])]))

        #     #if rot DOF are used, apply continuity of the rotational dof on each oposite faces and corner
        #     if 'RotZ' in list_var:
        #         res.append(MPC(['RotZ','RotZ'], [np.full_like(right,1), np.full_like(left,-1)], [right,left]))
        #         res.append(MPC(['RotZ','RotZ'], [np.full_like(top,1), np.full_like(bottom,-1)], [top,bottom]))
        #         res.append(MPC(['RotZ','RotZ'], [np.full_like(left_top,1), np.full_like(left_bottom,-1)], [left_top, left_bottom]))
        #         res.append(MPC(['RotZ','RotZ'], [np.full_like(right_bottom,1), np.full_like(left_bottom,-1)], [right_bottom, left_bottom]))
        #         res.append(MPC(['RotZ','RotZ'], [np.full_like(right_top,1), np.full_like(left_bottom,-1)], [right_top, left_bottom]))
        #     if 'RotY' in list_var:
        #         res.append(MPC(['RotY','RotY'], [np.full_like(right,1), np.full_like(left,-1)], [right,left]))
        #         res.append(MPC(['RotY','RotY'], [np.full_like(top,1), np.full_like(bottom,-1)], [top,bottom]))
        #         res.append(MPC(['RotY','RotY'], [np.full_like(left_top,1), np.full_like(left_bottom,-1)], [left_top, left_bottom]))
        #         res.append(MPC(['RotY','RotY'], [np.full_like(right_bottom,1), np.full_like(left_bottom,-1)], [right_bottom, left_bottom]))
        #         res.append(MPC(['RotY','RotY'], [np.full_like(right_top,1), np.full_like(left_bottom,-1)], [right_top, left_bottom]))
        #     if 'RotX' in list_var:
        #         res.append(MPC(['RotX','RotX'], [np.full_like(right,1), np.full_like(left,-1)], [right,left]))
        #         res.append(MPC(['RotX','RotX'], [np.full_like(top,1), np.full_like(bottom,-1)], [top,bottom]))
        #         res.append(MPC(['RotX','RotX'], [np.full_like(left_top,1), np.full_like(left_bottom,-1)], [left_top, left_bottom]))
        #         res.append(MPC(['RotX','RotX'], [np.full_like(right_bottom,1), np.full_like(left_bottom,-1)], [right_bottom, left_bottom]))
        #         res.append(MPC(['RotX','RotX'], [np.full_like(right_top,1), np.full_like(left_bottom,-1)], [right_top, left_bottom]))

        # if self.dim == 1:
        #     self.node_sets = (left, right)
        # elif self.dim == 2:
        #     self.node_sets = (left, right, bottom, top,
        #                  left_bottom, left_top, right_bottom, right_top)
        # else:
        #     self.node_sets = (left, right, bottom, top, back, front,
        #                  left_bottom, left_top, right_bottom, right_top,
        #                  bottom_back, bottom_front, top_back, top_front,
        #                  left_back, left_front, right_back, right_front,
        #                  left_bottom_back, left_bottom_front, left_top_back, left_top_front,
        #                  right_bottom_back, right_bottom_front, right_top_back, right_top_front)

    def generate(self, problem, t_fact=1, t_fact_old=None):
        return self.list_mpc.generate(problem, t_fact, t_fact_old)
