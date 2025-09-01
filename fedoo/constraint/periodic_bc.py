"""Periodic boundary condition constraint."""

import numpy as np
from scipy.spatial.transform import Rotation
from fedoo.core.boundary_conditions import BCBase, MPC, ListBC


class PeriodicBC(BCBase):
    """Periodic boundary conditions constraint.

    Perdiodic boundary condition object using several multi-points
    constraints.
    Some global dofs are added to define mean strain or mean displacement gradient.
    These dof are defined.
    The added global dof are:

    * For "small_strain" the linearized strain tensor components are introduced. The
      following global variables are therefore accessible:
      'E_xx', 'E_yy', 'E_zz', 'E_xy', 'E_xz', 'E_yz' in 3d (only the in-plane
      component in 2d)
      The whole component can also be extracted using the vector name 'MeanStrain.'
    * For "finite_strain", the displacement gradient is utilized:
      'DU_xx', 'DU_xy', 'DU_xz', 'DU_yx', 'DU_yy', 'DU_yz', 'DU_zx', 'DU_zy', 'DU_zz'
      ie 9 components in 3D and only 4 in-plane component in 2D.
      The whole component can also be extracted using the vector name 'MeanGradDisp'.
    * If off_axis_rotation is used, the local dof are also added for small and
      finite strain respectiveley:
      'E_ij' or 'DU_ij' with i and j taking values in {1,2,3} and the vectors
      'LocalMeanStrain' or 'LocalMeanGradDisp'.

    Parameters
    ----------
    periodicity_type : 'small_strain' or 'finite_strain'.
        The type of periodicity :
        * If small_strain is selected, the constraint is directly expressed in terms
          of linearized strain components.
        * If 'finite_strain' is selected, the constraint is expressed in terms of
          displacement gradient.
    off_axis_rotation : scipy.spatial.transform object Rotation
        Rotation between the reference frame and the off-axis loading frame
        where boundary conditions are applied.
    dim : int in [1,2,3] (default = 3)
        Number of dimensions with periodic condition.
        If dim = 1: the periodicity is assumed along the x coordinate
        If dim = 2: the periodicity is assumed along x and y coordinates
    tol : float, optional
        Tolerance for the periodic nodes detection. The default is 1e-8.
    name : str, optional
        Name of the created boundary condition. The default is "Periodicity".


    Notes
    -----
    * The boundary condition object needs to be used with a problem associated to
      either a a periodic mesh, either
    * The periodic nodes are automatically detected using the given tolerance (tol).
    * The nodes of the Xp (x=xmax), Yp (y=ymax) and Zp (z=zmax) faces are
      eliminated from the system (slave nodes) and can't be used in another mpc.


    Example
    ---------

    .. code-block:: python

        import fedoo as fd

        mesh = fd.mesh.box_mesh()
        bc_periodic = fd.constraint.PeriodicBC(periodicity_type = 'small_strain')
    """

    def __init__(
        self,
        periodicity_type: str = "small_strain",
        off_axis_rotation: Rotation = None,
        dim: int = None,
        meshperio: bool = True,
        tol: float = 1e-8,
        name: str = "Periodicity",
    ):
        if not isinstance(periodicity_type, str):
            raise TypeError("periodicity_type should be a string")
        if periodicity_type == "small_strain":
            self.shear_coef = 0.5
        elif periodicity_type == "finite_strain":
            self.shear_coef = 1.0
        else:
            raise ValueError(
                "periodicity_type should be either 'small_strain' or 'finite_strain'"
            )

        self.off_axis_rotation = off_axis_rotation
        self.dim = dim  # dimension of periodicity (1, 2 or 3)
        self.meshperio = meshperio  # if True, the mesh is periodic
        self.tol = tol
        self.bc_type = "PeriodicBC"
        BCBase.__init__(self, name)
        self.periodicity_type = periodicity_type

    def __repr__(self):
        list_str = ["{}D Periodic Boundary Condition:".format(self.dim)]
        if self.name != "":
            list_str.append("name = '{}'".format(self.name))

        return "\n".join(list_str)

    def _prepare_periodic_lists(self, mesh, tol):
        """
        This function prepares periodic lists, having the list of node coordinates

        :param: self : the PeriodicBC object

        :warning: TO possibly modifiy, the (xmin, xmax, ymin, ymax, zmin, zmax) values are computed from the crd here
                  It might be better to add a parameter that computes it from a BoxMesh object

        :return: A dictionnary containing all the mesh listes (faces, edges, corners)
        """

        d_rve = []
        crd = mesh.nodes

        xmax = np.max(crd[:, 0])
        xmin = np.min(crd[:, 0])
        ymax = np.max(crd[:, 1])
        ymin = np.min(crd[:, 1])
        d_rve.append(xmax - xmin)
        d_rve.append(ymax - ymin)
        if self.dim == 3:
            zmax = np.max(crd[:, 2])
            zmin = np.min(crd[:, 2])
            d_rve.append(zmax - zmin)

        face_Xm = np.where(np.abs(crd[:, 0] - xmin) < tol)[0]
        face_Xp = np.where(np.abs(crd[:, 0] - xmax) < tol)[0]

        if self.dim > 1:
            face_Ym = np.where(np.abs(crd[:, 1] - ymin) < tol)[0]
            face_Yp = np.where(np.abs(crd[:, 1] - ymax) < tol)[0]

            # extract edges/corners from the intersection of faces
            edge_XmYm = np.intersect1d(face_Xm, face_Ym, assume_unique=True)
            edge_XmYp = np.intersect1d(face_Xm, face_Yp, assume_unique=True)
            edge_XpYm = np.intersect1d(face_Xp, face_Ym, assume_unique=True)
            edge_XpYp = np.intersect1d(face_Xp, face_Yp, assume_unique=True)

            if self.dim > 2:  # or dim == 3
                face_Zm = np.where(np.abs(crd[:, 2] - zmin) < tol)[0]
                face_Zp = np.where(np.abs(crd[:, 2] - zmax) < tol)[0]

                # extract edges/corners from the intersection of faces
                edge_YmZm = np.intersect1d(face_Ym, face_Zm, assume_unique=True)
                edge_YmZp = np.intersect1d(face_Ym, face_Zp, assume_unique=True)
                edge_YpZm = np.intersect1d(face_Yp, face_Zm, assume_unique=True)
                edge_YpZp = np.intersect1d(face_Yp, face_Zp, assume_unique=True)

                edge_XmZm = np.intersect1d(face_Xm, face_Zm, assume_unique=True)
                edge_XmZp = np.intersect1d(face_Xm, face_Zp, assume_unique=True)
                edge_XpZm = np.intersect1d(face_Xp, face_Zm, assume_unique=True)
                edge_XpZp = np.intersect1d(face_Xp, face_Zp, assume_unique=True)

                # extract corners from the intersection of edges
                corner_XmYmZm = np.intersect1d(edge_XmYm, edge_YmZm, assume_unique=True)
                corner_XmYmZp = np.intersect1d(edge_XmYm, edge_YmZp, assume_unique=True)
                corner_XmYpZm = np.intersect1d(edge_XmYp, edge_YpZm, assume_unique=True)
                corner_XmYpZp = np.intersect1d(edge_XmYp, edge_YpZp, assume_unique=True)
                corner_XpYmZm = np.intersect1d(edge_XpYm, edge_YmZm, assume_unique=True)
                corner_XpYmZp = np.intersect1d(edge_XpYm, edge_YmZp, assume_unique=True)
                corner_XpYpZm = np.intersect1d(edge_XpYp, edge_YpZm, assume_unique=True)
                corner_XpYpZp = np.intersect1d(edge_XpYp, edge_YpZp, assume_unique=True)

                # Remove nodes that beloing to several sets
                all_corners = np.hstack(
                    (
                        corner_XmYmZm,
                        corner_XmYmZp,
                        corner_XmYpZm,
                        corner_XmYpZp,
                        corner_XpYmZm,
                        corner_XpYmZp,
                        corner_XpYpZm,
                        corner_XpYpZp,
                    )
                )

                edge_XmYm = np.setdiff1d(edge_XmYm, all_corners, assume_unique=True)
                edge_XmYp = np.setdiff1d(edge_XmYp, all_corners, assume_unique=True)
                edge_XpYm = np.setdiff1d(edge_XpYm, all_corners, assume_unique=True)
                edge_XpYp = np.setdiff1d(edge_XpYp, all_corners, assume_unique=True)

                edge_YmZm = np.setdiff1d(edge_YmZm, all_corners, assume_unique=True)
                edge_YmZp = np.setdiff1d(edge_YmZp, all_corners, assume_unique=True)
                edge_YpZm = np.setdiff1d(edge_YpZm, all_corners, assume_unique=True)
                edge_YpZp = np.setdiff1d(edge_YpZp, all_corners, assume_unique=True)

                edge_XmZm = np.setdiff1d(edge_XmZm, all_corners, assume_unique=True)
                edge_XmZp = np.setdiff1d(edge_XmZp, all_corners, assume_unique=True)
                edge_XpZm = np.setdiff1d(edge_XpZm, all_corners, assume_unique=True)
                edge_XpZp = np.setdiff1d(edge_XpZp, all_corners, assume_unique=True)

                all_edges = np.hstack(
                    (
                        edge_XmYm,
                        edge_XmYp,
                        edge_XpYm,
                        edge_XpYp,
                        edge_YmZm,
                        edge_YmZp,
                        edge_YpZm,
                        edge_YpZp,
                        edge_XmZm,
                        edge_XmZp,
                        edge_XpZm,
                        edge_XpZp,
                        all_corners,
                    )
                )

            else:  # dim = 2
                all_edges = np.hstack((edge_XmYm, edge_XmYp, edge_XpYm, edge_XpYp))

            face_Xm = np.setdiff1d(face_Xm, all_edges, assume_unique=True)
            face_Xp = np.setdiff1d(face_Xp, all_edges, assume_unique=True)
            face_Ym = np.setdiff1d(face_Ym, all_edges, assume_unique=True)
            face_Yp = np.setdiff1d(face_Yp, all_edges, assume_unique=True)

            if mesh.ndim > 2:  # if there is a z coordinate
                # sort edges (required to assign the good pair of nodes)
                edge_XmYm = edge_XmYm[np.argsort(crd[edge_XmYm, 2])]
                edge_XmYp = edge_XmYp[np.argsort(crd[edge_XmYp, 2])]
                edge_XpYm = edge_XpYm[np.argsort(crd[edge_XpYm, 2])]
                edge_XpYp = edge_XpYp[np.argsort(crd[edge_XpYp, 2])]

            if self.dim > 2:
                face_Zm = np.setdiff1d(face_Zm, all_edges, assume_unique=True)
                face_Zp = np.setdiff1d(face_Zp, all_edges, assume_unique=True)

                edge_YmZm = edge_YmZm[np.argsort(crd[edge_YmZm, 0])]
                edge_YmZp = edge_YmZp[np.argsort(crd[edge_YmZp, 0])]
                edge_YpZm = edge_YpZm[np.argsort(crd[edge_YpZm, 0])]
                edge_YpZp = edge_YpZp[np.argsort(crd[edge_YpZp, 0])]

                edge_XmZm = edge_XmZm[np.argsort(crd[edge_XmZm, 1])]
                edge_XmZp = edge_XmZp[np.argsort(crd[edge_XmZp, 1])]
                edge_XpZm = edge_XpZm[np.argsort(crd[edge_XpZm, 1])]
                edge_XpZp = edge_XpZp[np.argsort(crd[edge_XpZp, 1])]

        # sort adjacent faces to ensure node correspondance
        if mesh.ndim == 2:
            face_Xm = face_Xm[np.argsort(crd[face_Xm, 1])]
            face_Xp = face_Xp[np.argsort(crd[face_Xp, 1])]
            if self.dim > 1:
                face_Ym = face_Ym[np.argsort(crd[face_Ym, 0])]
                face_Yp = face_Yp[np.argsort(crd[face_Yp, 0])]

        elif mesh.ndim > 2:
            decimal_round = int(-np.log10(tol) - 1)
            face_Xm = face_Xm[
                np.lexsort((crd[face_Xm, 1], crd[face_Xm, 2].round(decimal_round)))
            ]
            face_Xp = face_Xp[
                np.lexsort((crd[face_Xp, 1], crd[face_Xp, 2].round(decimal_round)))
            ]
            if self.dim > 1:
                face_Ym = face_Ym[
                    np.lexsort((crd[face_Ym, 0], crd[face_Ym, 2].round(decimal_round)))
                ]
                face_Yp = face_Yp[
                    np.lexsort((crd[face_Yp, 0], crd[face_Yp, 2].round(decimal_round)))
                ]
            if self.dim > 2:
                face_Zm = face_Zm[
                    np.lexsort((crd[face_Zm, 0], crd[face_Zm, 1].round(decimal_round)))
                ]
                face_Zp = face_Zp[
                    np.lexsort((crd[face_Zp, 0], crd[face_Zp, 1].round(decimal_round)))
                ]

        # save the computated set of nodes as class attributes
        self.face_Xm = face_Xm
        self.face_Xp = face_Xp

        if self.dim > 1:
            self.face_Ym = face_Ym
            self.face_Yp = face_Yp

            self.edge_XmYm = edge_XmYm
            self.edge_XpYm = edge_XpYm
            self.edge_XpYp = edge_XpYp
            self.edge_XmYp = edge_XmYp

        if self.dim > 2:
            self.face_Zm = face_Zm
            self.face_Zp = face_Zp

            self.edge_XmZm = edge_XmZm
            self.edge_YmZm = edge_YmZm
            self.edge_XpZm = edge_XpZm
            self.edge_XpZp = edge_XpZp

            self.edge_XmZp = edge_XmZp
            self.edge_YpZm = edge_YpZm
            self.edge_YpZp = edge_YpZp
            self.edge_YmZp = edge_YmZp

            self.corner_XmYmZm = corner_XmYmZm
            self.corner_XmYmZp = corner_XmYmZp
            self.corner_XmYpZm = corner_XmYpZm
            self.corner_XmYpZp = corner_XmYpZp

            self.corner_XpYmZm = corner_XpYmZm
            self.corner_XpYmZp = corner_XpYmZp
            self.corner_XpYpZm = corner_XpYpZm
            self.corner_XpYpZp = corner_XpYpZp

        self.d_rve = d_rve

    def _list_MPC_rotation(self):
        """
        This function defines the list of MPC constraints for periodic homogenization,
        assuming a periodic mesh

        :param: self : the PeriodicBC object
        :param off_axis_rot_mat: Rotation object from scipy.transform
            Rotation matrix that defines the off_axis loading

        :return: A dictionnary containing the MPC constraints that links constrain drivers of the off_axis to the reference frame.
        """

        node_cd = self.node_cd
        var_cd = self.var_cd
        node_cd_load = self.node_cd_loading
        var_cd_load = self.var_cd_loading
        off_axis_rot_matrix = self.off_axis_rot_matrix

        a, b, c, d, e, f, g, h, i = off_axis_rot_matrix.flatten()

        res = ListBC()

        if self.periodicity_type == "small_strain":
            res.append(
                MPC(
                    [
                        node_cd[0][0],
                        node_cd_load[0][0],
                        node_cd_load[1][1],
                        node_cd_load[2][2],
                        node_cd_load[0][1],
                        node_cd_load[0][2],
                        node_cd_load[1][2],
                    ],
                    [
                        var_cd[0][0],
                        var_cd_load[0][0],
                        var_cd_load[1][1],
                        var_cd_load[2][2],
                        var_cd_load[0][1],
                        var_cd_load[0][2],
                        var_cd_load[1][2],
                    ],
                    [
                        1.0,
                        -(a**2),
                        -(b**2),
                        -(c**2),
                        -(a * b),
                        -(a * c),
                        -(b * c),
                    ],
                )
            )

            res.append(
                MPC(
                    [
                        node_cd[1][1],
                        node_cd_load[0][0],
                        node_cd_load[1][1],
                        node_cd_load[2][2],
                        node_cd_load[0][1],
                        node_cd_load[0][2],
                        node_cd_load[1][2],
                    ],
                    [
                        var_cd[1][1],
                        var_cd_load[0][0],
                        var_cd_load[1][1],
                        var_cd_load[2][2],
                        var_cd_load[0][1],
                        var_cd_load[0][2],
                        var_cd_load[1][2],
                    ],
                    [
                        1.0,
                        -(d**2),
                        -(e**2),
                        -(f**2),
                        -(d * e),
                        -(d * f),
                        -(e * f),
                    ],
                )
            )

            res.append(
                MPC(
                    [
                        node_cd[2][2],
                        node_cd_load[0][0],
                        node_cd_load[1][1],
                        node_cd_load[2][2],
                        node_cd_load[0][1],
                        node_cd_load[0][2],
                        node_cd_load[1][2],
                    ],
                    [
                        var_cd[2][2],
                        var_cd_load[0][0],
                        var_cd_load[1][1],
                        var_cd_load[2][2],
                        var_cd_load[0][1],
                        var_cd_load[0][2],
                        var_cd_load[1][2],
                    ],
                    [
                        1.0,
                        -(g**2),
                        -(h**2),
                        -(i**2),
                        -(g * h),
                        -(g * i),
                        -(h * i),
                    ],
                )
            )

            res.append(
                MPC(
                    [
                        node_cd[0][1],
                        node_cd_load[0][0],
                        node_cd_load[1][1],
                        node_cd_load[2][2],
                        node_cd_load[0][1],
                        node_cd_load[0][2],
                        node_cd_load[1][2],
                    ],
                    [
                        var_cd[0][1],
                        var_cd_load[0][0],
                        var_cd_load[1][1],
                        var_cd_load[2][2],
                        var_cd_load[0][1],
                        var_cd_load[0][2],
                        var_cd_load[1][2],
                    ],
                    [
                        1.0,
                        -2.0 * (a * d),
                        -2.0 * (b * e),
                        -2.0 * (c * f),
                        -(b * d + a * e),
                        -(c * d + a * f),
                        -(c * e + b * f),
                    ],
                )
            )

            res.append(
                MPC(
                    [
                        node_cd[0][2],
                        node_cd_load[0][0],
                        node_cd_load[1][1],
                        node_cd_load[2][2],
                        node_cd_load[0][1],
                        node_cd_load[0][2],
                        node_cd_load[1][2],
                    ],
                    [
                        var_cd[0][2],
                        var_cd_load[0][0],
                        var_cd_load[1][1],
                        var_cd_load[2][2],
                        var_cd_load[0][1],
                        var_cd_load[0][2],
                        var_cd_load[1][2],
                    ],
                    [
                        1.0,
                        -2.0 * (a * g),
                        -2.0 * (b * h),
                        -2.0 * (c * i),
                        -(b * g + a * h),
                        -(c * g + a * i),
                        -(c * h + b * i),
                    ],
                )
            )

            res.append(
                MPC(
                    [
                        node_cd[1][2],
                        node_cd_load[0][0],
                        node_cd_load[1][1],
                        node_cd_load[2][2],
                        node_cd_load[0][1],
                        node_cd_load[0][2],
                        node_cd_load[1][2],
                    ],
                    [
                        var_cd[1][2],
                        var_cd_load[0][0],
                        var_cd_load[1][1],
                        var_cd_load[2][2],
                        var_cd_load[0][1],
                        var_cd_load[0][2],
                        var_cd_load[1][2],
                    ],
                    [
                        1.0,
                        -2.0 * (d * g),
                        -2.0 * (e * h),
                        -2.0 * (f * i),
                        -(e * g + d * h),
                        -(f * g + d * i),
                        -(f * h + e * i),
                    ],
                )
            )
            return res

        if self.periodicity_type == "finite_strain":
            res.append(
                MPC(
                    [
                        node_cd[0][0],
                        node_cd_load[0][0],
                        node_cd_load[0][1],
                        node_cd_load[0][2],
                        node_cd_load[1][0],
                        node_cd_load[1][1],
                        node_cd_load[1][2],
                        node_cd_load[2][0],
                        node_cd_load[2][1],
                        node_cd_load[2][2],
                    ],
                    [
                        var_cd[0][0],
                        var_cd_load[0][0],
                        var_cd_load[0][1],
                        var_cd_load[0][2],
                        var_cd_load[1][0],
                        var_cd_load[1][1],
                        var_cd_load[1][2],
                        var_cd_load[2][0],
                        var_cd_load[2][1],
                        var_cd_load[2][2],
                    ],
                    [
                        1.0,
                        -(a**2),
                        -a * b,
                        -a * c,
                        -b * a,
                        -(b**2),
                        -b * c,
                        -c * a,
                        -c * b,
                        -(c**2),
                    ],
                )
            )

            res.append(
                MPC(
                    [
                        node_cd[0][1],
                        node_cd_load[0][0],
                        node_cd_load[0][1],
                        node_cd_load[0][2],
                        node_cd_load[1][0],
                        node_cd_load[1][1],
                        node_cd_load[1][2],
                        node_cd_load[2][0],
                        node_cd_load[2][1],
                        node_cd_load[2][2],
                    ],
                    [
                        var_cd[0][1],
                        var_cd_load[0][0],
                        var_cd_load[0][1],
                        var_cd_load[0][2],
                        var_cd_load[1][0],
                        var_cd_load[1][1],
                        var_cd_load[1][2],
                        var_cd_load[2][0],
                        var_cd_load[2][1],
                        var_cd_load[2][2],
                    ],
                    [
                        1.0,
                        -a * d,
                        -a * e,
                        -a * f,
                        -b * d,
                        -b * e,
                        -b * f,
                        -c * d,
                        -c * e,
                        -c * f,
                    ],
                )
            )

            res.append(
                MPC(
                    [
                        node_cd[0][2],
                        node_cd_load[0][0],
                        node_cd_load[0][1],
                        node_cd_load[0][2],
                        node_cd_load[1][0],
                        node_cd_load[1][1],
                        node_cd_load[1][2],
                        node_cd_load[2][0],
                        node_cd_load[2][1],
                        node_cd_load[2][2],
                    ],
                    [
                        var_cd[0][2],
                        var_cd_load[0][0],
                        var_cd_load[0][1],
                        var_cd_load[0][2],
                        var_cd_load[1][0],
                        var_cd_load[1][1],
                        var_cd_load[1][2],
                        var_cd_load[2][0],
                        var_cd_load[2][1],
                        var_cd_load[2][2],
                    ],
                    [
                        1.0,
                        -a * g,
                        -a * h,
                        -a * i,
                        -b * g,
                        -b * h,
                        -b * i,
                        -c * g,
                        -c * h,
                        -c * i,
                    ],
                )
            )

            res.append(
                MPC(
                    [
                        node_cd[1][0],
                        node_cd_load[0][0],
                        node_cd_load[0][1],
                        node_cd_load[0][2],
                        node_cd_load[1][0],
                        node_cd_load[1][1],
                        node_cd_load[1][2],
                        node_cd_load[2][0],
                        node_cd_load[2][1],
                        node_cd_load[2][2],
                    ],
                    [
                        var_cd[1][0],
                        var_cd_load[0][0],
                        var_cd_load[0][1],
                        var_cd_load[0][2],
                        var_cd_load[1][0],
                        var_cd_load[1][1],
                        var_cd_load[1][2],
                        var_cd_load[2][0],
                        var_cd_load[2][1],
                        var_cd_load[2][2],
                    ],
                    [
                        1.0,
                        -d * a,
                        -d * b,
                        -d * c,
                        -e * a,
                        -e * b,
                        -e * c,
                        -f * a,
                        -f * b,
                        -f * c,
                    ],
                )
            )

            res.append(
                MPC(
                    [
                        node_cd[1][1],
                        node_cd_load[0][0],
                        node_cd_load[0][1],
                        node_cd_load[0][2],
                        node_cd_load[1][0],
                        node_cd_load[1][1],
                        node_cd_load[1][2],
                        node_cd_load[2][0],
                        node_cd_load[2][1],
                        node_cd_load[2][2],
                    ],
                    [
                        var_cd[1][1],
                        var_cd_load[0][0],
                        var_cd_load[0][1],
                        var_cd_load[0][2],
                        var_cd_load[1][0],
                        var_cd_load[1][1],
                        var_cd_load[1][2],
                        var_cd_load[2][0],
                        var_cd_load[2][1],
                        var_cd_load[2][2],
                    ],
                    [
                        1.0,
                        -(d**2),
                        -d * e,
                        -d * f,
                        -e * d,
                        -(e**2),
                        -e * f,
                        -f * d,
                        -f * e,
                        -(f**2),
                    ],
                )
            )

            res.append(
                MPC(
                    [
                        node_cd[1][2],
                        node_cd_load[0][0],
                        node_cd_load[0][1],
                        node_cd_load[0][2],
                        node_cd_load[1][0],
                        node_cd_load[1][1],
                        node_cd_load[1][2],
                        node_cd_load[2][0],
                        node_cd_load[2][1],
                        node_cd_load[2][2],
                    ],
                    [
                        var_cd[1][2],
                        var_cd_load[0][0],
                        var_cd_load[0][1],
                        var_cd_load[0][2],
                        var_cd_load[1][0],
                        var_cd_load[1][1],
                        var_cd_load[1][2],
                        var_cd_load[2][0],
                        var_cd_load[2][1],
                        var_cd_load[2][2],
                    ],
                    [
                        1.0,
                        -d * g,
                        -d * h,
                        -d * i,
                        -e * g,
                        -e * h,
                        -e * i,
                        -f * g,
                        -f * h,
                        -f * i,
                    ],
                )
            )

            res.append(
                MPC(
                    [
                        node_cd[2][0],
                        node_cd_load[0][0],
                        node_cd_load[0][1],
                        node_cd_load[0][2],
                        node_cd_load[1][0],
                        node_cd_load[1][1],
                        node_cd_load[1][2],
                        node_cd_load[2][0],
                        node_cd_load[2][1],
                        node_cd_load[2][2],
                    ],
                    [
                        var_cd[2][0],
                        var_cd_load[0][0],
                        var_cd_load[0][1],
                        var_cd_load[0][2],
                        var_cd_load[1][0],
                        var_cd_load[1][1],
                        var_cd_load[1][2],
                        var_cd_load[2][0],
                        var_cd_load[2][1],
                        var_cd_load[2][2],
                    ],
                    [
                        1.0,
                        -g * a,
                        -g * b,
                        -g * c,
                        -h * a,
                        -h * b,
                        -h * c,
                        -i * a,
                        -i * b,
                        -i * c,
                    ],
                )
            )

            res.append(
                MPC(
                    [
                        node_cd[2][1],
                        node_cd_load[0][0],
                        node_cd_load[0][1],
                        node_cd_load[0][2],
                        node_cd_load[1][0],
                        node_cd_load[1][1],
                        node_cd_load[1][2],
                        node_cd_load[2][0],
                        node_cd_load[2][1],
                        node_cd_load[2][2],
                    ],
                    [
                        var_cd[2][1],
                        var_cd_load[0][0],
                        var_cd_load[0][1],
                        var_cd_load[0][2],
                        var_cd_load[1][0],
                        var_cd_load[1][1],
                        var_cd_load[1][2],
                        var_cd_load[2][0],
                        var_cd_load[2][1],
                        var_cd_load[2][2],
                    ],
                    [
                        1.0,
                        -g * d,
                        -g * e,
                        -g * f,
                        -h * d,
                        -h * e,
                        -h * f,
                        -i * d,
                        -i * e,
                        -i * f,
                    ],
                )
            )

            res.append(
                MPC(
                    [
                        node_cd[2][2],
                        node_cd_load[0][0],
                        node_cd_load[0][1],
                        node_cd_load[0][2],
                        node_cd_load[1][0],
                        node_cd_load[1][1],
                        node_cd_load[1][2],
                        node_cd_load[2][0],
                        node_cd_load[2][1],
                        node_cd_load[2][2],
                    ],
                    [
                        var_cd[2][2],
                        var_cd_load[0][0],
                        var_cd_load[0][1],
                        var_cd_load[0][2],
                        var_cd_load[1][0],
                        var_cd_load[1][1],
                        var_cd_load[1][2],
                        var_cd_load[2][0],
                        var_cd_load[2][1],
                        var_cd_load[2][2],
                    ],
                    [
                        1.0,
                        -(g**2),
                        -g * h,
                        -g * i,
                        -h * g,
                        -(h**2),
                        -h * i,
                        -i * g,
                        -i * h,
                        -(i**2),
                    ],
                )
            )
            return res

    def _list_MPC_periodic(self):
        """
        This function defines the list of MPC constraints for periodic homogenization,
        assuming a periodic mesh

        :param: self : the PeriodicBC object

        :warning: TO possibly modifiy, the (xmin, xmax, ymin, ymax, zmin, zmax) values are computed from the crd here
                  It might be better to add a parameter that computes it from a BoxMesh object

        :return: A dictionnary containing all the mesh liste (faces, edges, corners)
        """

        node_cd = self.node_cd
        var_cd = self.var_cd

        sc = self.shear_coef

        dx = self.d_rve[0]
        face_Xm = self.face_Xm
        face_Xp = self.face_Xp

        if self.dim > 1:
            dy = self.d_rve[1]

            face_Ym = self.face_Ym
            face_Yp = self.face_Yp

            edge_XmYm = self.edge_XmYm
            edge_XpYm = self.edge_XpYm
            edge_XpYp = self.edge_XpYp
            edge_XmYp = self.edge_XmYp

        if self.dim > 2:
            dz = self.d_rve[2]

            face_Zm = self.face_Zm
            face_Zp = self.face_Zp

            edge_XmZm = self.edge_XmZm
            edge_YmZm = self.edge_YmZm
            edge_XpZm = self.edge_XpZm
            edge_XpZp = self.edge_XpZp
            edge_XmZp = self.edge_XmZp
            edge_YpZm = self.edge_YpZm
            edge_YpZp = self.edge_YpZp
            edge_YmZp = self.edge_YmZp

            corner_XmYmZm = self.corner_XmYmZm
            corner_XmYmZp = self.corner_XmYmZp
            corner_XmYpZm = self.corner_XmYpZm
            corner_XmYpZp = self.corner_XmYpZp
            corner_XpYmZm = self.corner_XpYmZm
            corner_XpYmZp = self.corner_XpYmZp
            corner_XpYpZm = self.corner_XpYpZm
            corner_XpYpZp = self.corner_XpYpZp

        res = ListBC()
        # face_Xm/Xp faces (DispX)
        res.append(
            MPC(
                [face_Xp, face_Xm, np.full_like(face_Xp, node_cd[0][0])],
                ["DispX", "DispX", var_cd[0][0]],
                [
                    np.full_like(face_Xp, 1),
                    np.full_like(face_Xm, -1),
                    np.full_like(face_Xp, -dx, dtype=float),
                ],
            )
        )

        if self.dim > 1:
            # face_Xm/face_Xp faces (DispY)
            res.append(
                MPC(
                    [face_Xp, face_Xm, np.full_like(face_Xp, node_cd[1][0])],
                    ["DispY", "DispY", var_cd[1][0]],
                    [
                        np.full_like(face_Xp, 1),
                        np.full_like(face_Xm, -1),
                        np.full_like(face_Xp, -sc * dx, dtype=float),
                    ],
                )
            )

            # face_Yp/face_Ym faces (DispX and DispY)
            res.append(
                MPC(
                    [face_Yp, face_Ym, np.full_like(face_Yp, node_cd[0][1])],
                    ["DispX", "DispX", var_cd[0][1]],
                    [
                        np.full_like(face_Yp, 1),
                        np.full_like(face_Ym, -1),
                        np.full_like(face_Yp, -sc * dy, dtype=float),
                    ],
                )
            )
            res.append(
                MPC(
                    [face_Yp, face_Ym, np.full_like(face_Yp, node_cd[1][1])],
                    ["DispY", "DispY", var_cd[1][1]],
                    [
                        np.full_like(face_Yp, 1),
                        np.full_like(face_Ym, -1),
                        np.full_like(face_Yp, -dy, dtype=float),
                    ],
                )
            )

            # elimination of DOF from edge Xm/Yp -> edge Xm/Ym (DispX, DispY)
            res.append(
                MPC(
                    [edge_XmYp, edge_XmYm, np.full_like(edge_XmYp, node_cd[0][1])],
                    ["DispX", "DispX", var_cd[0][1]],
                    [
                        np.full_like(edge_XmYp, 1),
                        np.full_like(edge_XmYm, -1),
                        np.full_like(edge_XmYp, -sc * dy, dtype=float),
                    ],
                )
            )
            res.append(
                MPC(
                    [edge_XmYp, edge_XmYm, np.full_like(edge_XmYp, node_cd[1][1])],
                    ["DispY", "DispY", var_cd[1][1]],
                    [
                        np.full_like(edge_XmYp, 1),
                        np.full_like(edge_XmYm, -1),
                        np.full_like(edge_XmYp, -dy, dtype=float),
                    ],
                )
            )
            # elimination of DOF from edge face_Xp/Ym -> edge Xm/Ym (DispX, DispY)
            res.append(
                MPC(
                    [edge_XpYm, edge_XmYm, np.full_like(edge_XmYp, node_cd[0][0])],
                    ["DispX", "DispX", var_cd[0][0]],
                    [
                        np.full_like(edge_XpYm, 1),
                        np.full_like(edge_XmYm, -1),
                        np.full_like(edge_XpYm, -dx, dtype=float),
                    ],
                )
            )
            res.append(
                MPC(
                    [edge_XpYm, edge_XmYm, np.full_like(edge_XmYp, node_cd[1][0])],
                    ["DispY", "DispY", var_cd[1][0]],
                    [
                        np.full_like(edge_XpYm, 1),
                        np.full_like(edge_XmYm, -1),
                        np.full_like(edge_XpYm, -sc * dx, dtype=float),
                    ],
                )
            )
            # elimination of DOF from edge Xp/Yp -> edge Xm/Ym (DispX, DispY)
            res.append(
                MPC(
                    [
                        edge_XpYp,
                        edge_XmYm,
                        np.full_like(edge_XpYp, node_cd[0][0]),
                        np.full_like(edge_XpYp, node_cd[0][1]),
                    ],
                    ["DispX", "DispX", var_cd[0][0], var_cd[0][1]],
                    [
                        np.full_like(edge_XpYp, 1),
                        np.full_like(edge_XmYm, -1),
                        np.full_like(edge_XpYp, -dx, dtype=float),
                        np.full_like(edge_XpYp, -sc * dy, dtype=float),
                    ],
                )
            )
            res.append(
                MPC(
                    [
                        edge_XpYp,
                        edge_XmYm,
                        np.full_like(edge_XpYp, node_cd[1][0]),
                        np.full_like(edge_XpYp, node_cd[1][1]),
                    ],
                    ["DispY", "DispY", var_cd[1][0], var_cd[1][1]],
                    [
                        np.full_like(edge_XpYp, 1),
                        np.full_like(edge_XmYm, -1),
                        np.full_like(edge_XpYp, -sc * dx, dtype=float),
                        np.full_like(edge_XpYp, -dy, dtype=float),
                    ],
                )
            )

        if self.dim > 2:
            # DispZ for Xm/Xp faces
            res.append(
                MPC(
                    [face_Xp, face_Xm, np.full_like(face_Xp, node_cd[2][0])],
                    ["DispZ", "DispZ", var_cd[2][0]],
                    [
                        np.full_like(face_Xp, 1),
                        np.full_like(face_Xm, -1),
                        np.full_like(face_Xp, -sc * dx, dtype=float),
                    ],
                )
            )
            # DispZ for Yp/Ym faces
            res.append(
                MPC(
                    [face_Yp, face_Ym, np.full_like(face_Yp, node_cd[2][1])],
                    ["DispZ", "DispZ", var_cd[2][1]],
                    [
                        np.full_like(face_Yp, 1),
                        np.full_like(face_Ym, -1),
                        np.full_like(face_Yp, -sc * dy, dtype=float),
                    ],
                )
            )

            # elimination of DOF from edge Xm/Yp -> edge Xm/Ym (DispZ)
            res.append(
                MPC(
                    [edge_XmYp, edge_XmYm, np.full_like(edge_XmYp, node_cd[2][1])],
                    ["DispZ", "DispZ", var_cd[2][1]],
                    [
                        np.full_like(edge_XmYp, 1),
                        np.full_like(edge_XmYm, -1),
                        np.full_like(edge_XmYp, -sc * dy, dtype=float),
                    ],
                )
            )
            # elimination of DOF from edge Xp/Ym -> edge Xm/Ym (DispZ)
            res.append(
                MPC(
                    [edge_XpYm, edge_XmYm, np.full_like(edge_XmYp, node_cd[2][0])],
                    ["DispZ", "DispZ", var_cd[2][0]],
                    [
                        np.full_like(edge_XpYm, 1),
                        np.full_like(edge_XmYm, -1),
                        np.full_like(edge_XpYm, -sc * dx, dtype=float),
                    ],
                )
            )
            # elimination of DOF from edge Xp/Yp -> edge Xm/Ym (DispZ)
            res.append(
                MPC(
                    [
                        edge_XpYp,
                        edge_XmYm,
                        np.full_like(edge_XpYp, node_cd[2][0]),
                        np.full_like(edge_XpYp, node_cd[2][1]),
                    ],
                    ["DispZ", "DispZ", var_cd[2][0], var_cd[2][1]],
                    [
                        np.full_like(edge_XpYp, 1),
                        np.full_like(edge_XmYm, -1),
                        np.full_like(edge_XpYp, -sc * dx, dtype=float),
                        np.full_like(edge_XpYp, -sc * dy, dtype=float),
                    ],
                )
            )

            # Zp/Zm faces
            res.append(
                MPC(
                    [face_Zp, face_Zm, np.full_like(face_Zp, node_cd[0][2])],
                    ["DispX", "DispX", var_cd[0][2]],
                    [
                        np.full_like(face_Zp, 1),
                        np.full_like(face_Zm, -1),
                        np.full_like(face_Zp, -sc * dz, dtype=float),
                    ],
                )
            )
            res.append(
                MPC(
                    [face_Zp, face_Zm, np.full_like(face_Zp, node_cd[1][2])],
                    ["DispY", "DispY", var_cd[1][2]],
                    [
                        np.full_like(face_Zp, 1),
                        np.full_like(face_Zm, -1),
                        np.full_like(face_Zp, -sc * dz, dtype=float),
                    ],
                )
            )
            res.append(
                MPC(
                    [face_Zp, face_Zm, np.full_like(face_Zp, node_cd[2][2])],
                    ["DispZ", "DispZ", var_cd[2][2]],
                    [
                        np.full_like(face_Zp, 1),
                        np.full_like(face_Zm, -1),
                        np.full_like(face_Zp, -dz, dtype=float),
                    ],
                )
            )

            # elimination of DOF from edge Yp/Zm -> edge Ym/Zm
            res.append(
                MPC(
                    [edge_YpZm, edge_YmZm, np.full_like(edge_YpZm, node_cd[0][1])],
                    ["DispX", "DispX", var_cd[0][1]],
                    [
                        np.full_like(edge_YpZm, 1),
                        np.full_like(edge_YmZm, -1),
                        np.full_like(edge_YpZm, -sc * dy, dtype=float),
                    ],
                )
            )
            res.append(
                MPC(
                    [edge_YpZm, edge_YmZm, np.full_like(edge_YpZm, node_cd[1][1])],
                    ["DispY", "DispY", var_cd[1][1]],
                    [
                        np.full_like(edge_YpZm, 1),
                        np.full_like(edge_YmZm, -1),
                        np.full_like(edge_YpZm, -dy, dtype=float),
                    ],
                )
            )
            res.append(
                MPC(
                    [edge_YpZm, edge_YmZm, np.full_like(edge_YpZm, node_cd[2][1])],
                    ["DispZ", "DispZ", var_cd[2][1]],
                    [
                        np.full_like(edge_YpZm, 1),
                        np.full_like(edge_YmZm, -1),
                        np.full_like(edge_YpZm, -sc * dy, dtype=float),
                    ],
                )
            )
            # elimination of DOF from edge Ym/Zp -> edge Ym/Zm
            res.append(
                MPC(
                    [edge_YmZp, edge_YmZm, np.full_like(edge_YmZp, node_cd[0][2])],
                    ["DispX", "DispX", var_cd[0][2]],
                    [
                        np.full_like(edge_YmZp, 1),
                        np.full_like(edge_YmZm, -1),
                        np.full_like(edge_YmZp, -sc * dz, dtype=float),
                    ],
                )
            )
            res.append(
                MPC(
                    [edge_YmZp, edge_YmZm, np.full_like(edge_YmZp, node_cd[1][2])],
                    ["DispY", "DispY", var_cd[1][2]],
                    [
                        np.full_like(edge_YmZp, 1),
                        np.full_like(edge_YmZm, -1),
                        np.full_like(edge_YmZp, -sc * dz, dtype=float),
                    ],
                )
            )
            res.append(
                MPC(
                    [edge_YmZp, edge_YmZm, np.full_like(edge_YmZp, node_cd[2][2])],
                    ["DispZ", "DispZ", var_cd[2][2]],
                    [
                        np.full_like(edge_YmZp, 1),
                        np.full_like(edge_YmZm, -1),
                        np.full_like(edge_YmZp, -dz, dtype=float),
                    ],
                )
            )
            # elimination of DOF from edge Yp/Zp -> edge Ym/Zm
            res.append(
                MPC(
                    [
                        edge_YpZp,
                        edge_YmZm,
                        np.full_like(edge_YpZp, node_cd[0][1]),
                        np.full_like(edge_YpZp, node_cd[0][2]),
                    ],
                    ["DispX", "DispX", var_cd[0][1], var_cd[0][2]],
                    [
                        np.full_like(edge_YpZp, 1),
                        np.full_like(edge_YmZm, -1),
                        np.full_like(edge_YpZp, -sc * dy, dtype=float),
                        np.full_like(edge_YpZp, -sc * dz, dtype=float),
                    ],
                )
            )
            res.append(
                MPC(
                    [
                        edge_YpZp,
                        edge_YmZm,
                        np.full_like(edge_YpZp, node_cd[1][1]),
                        np.full_like(edge_YpZp, node_cd[1][2]),
                    ],
                    ["DispY", "DispY", var_cd[1][1], var_cd[1][2]],
                    [
                        np.full_like(edge_YpZp, 1),
                        np.full_like(edge_YmZm, -1),
                        np.full_like(edge_YpZp, -dy, dtype=float),
                        np.full_like(edge_YpZp, -sc * dz, dtype=float),
                    ],
                )
            )
            res.append(
                MPC(
                    [
                        edge_YpZp,
                        edge_YmZm,
                        np.full_like(edge_YpZp, node_cd[2][1]),
                        np.full_like(edge_YpZp, node_cd[2][2]),
                    ],
                    ["DispZ", "DispZ", var_cd[2][1], var_cd[2][2]],
                    [
                        np.full_like(edge_YpZp, 1),
                        np.full_like(edge_YmZm, -1),
                        np.full_like(edge_YpZp, -sc * dy, dtype=float),
                        np.full_like(edge_YpZp, -dz, dtype=float),
                    ],
                )
            )

            # elimination of DOF from edge Xp/Zm -> edge Xm/Zm
            res.append(
                MPC(
                    [edge_XpZm, edge_XmZm, np.full_like(edge_XmZm, node_cd[0][0])],
                    ["DispX", "DispX", var_cd[0][0]],
                    [
                        np.full_like(edge_XpZm, 1),
                        np.full_like(edge_XmZm, -1),
                        np.full_like(edge_XpZm, -dx, dtype=float),
                    ],
                )
            )
            res.append(
                MPC(
                    [edge_XpZm, edge_XmZm, np.full_like(edge_XmZm, node_cd[1][0])],
                    ["DispY", "DispY", var_cd[1][0]],
                    [
                        np.full_like(edge_XpZm, 1),
                        np.full_like(edge_XmZm, -1),
                        np.full_like(edge_XpZm, -sc * dx, dtype=float),
                    ],
                )
            )
            res.append(
                MPC(
                    [edge_XpZm, edge_XmZm, np.full_like(edge_XmZm, node_cd[2][0])],
                    ["DispZ", "DispZ", var_cd[2][0]],
                    [
                        np.full_like(edge_XpZm, 1),
                        np.full_like(edge_XmZm, -1),
                        np.full_like(edge_XpZm, -sc * dx, dtype=float),
                    ],
                )
            )
            # elimination of DOF from edge Xm/Zp -> edge Xm/Zm
            res.append(
                MPC(
                    [edge_XmZp, edge_XmZm, np.full_like(edge_XmZm, node_cd[0][2])],
                    ["DispX", "DispX", var_cd[0][2]],
                    [
                        np.full_like(edge_XmZp, 1),
                        np.full_like(edge_XmZm, -1),
                        np.full_like(edge_XpZm, -sc * dz, dtype=float),
                    ],
                )
            )
            res.append(
                MPC(
                    [edge_XmZp, edge_XmZm, np.full_like(edge_XmZm, node_cd[1][2])],
                    ["DispY", "DispY", var_cd[1][2]],
                    [
                        np.full_like(edge_XmZp, 1),
                        np.full_like(edge_XmZm, -1),
                        np.full_like(edge_XpZm, -sc * dz, dtype=float),
                    ],
                )
            )
            res.append(
                MPC(
                    [edge_XmZp, edge_XmZm, np.full_like(edge_XmZm, node_cd[2][2])],
                    ["DispZ", "DispZ", var_cd[2][2]],
                    [
                        np.full_like(edge_XmZp, 1),
                        np.full_like(edge_XmZm, -1),
                        np.full_like(edge_XpZm, -dz, dtype=float),
                    ],
                )
            )
            # elimination of DOF from edge Xp/Zp -> edge Xm/Zm
            res.append(
                MPC(
                    [
                        edge_XpZp,
                        edge_XmZm,
                        np.full_like(edge_XpZp, node_cd[0][0]),
                        np.full_like(edge_XpZp, node_cd[0][2]),
                    ],
                    ["DispX", "DispX", var_cd[0][0], var_cd[0][2]],
                    [
                        np.full_like(edge_XpZp, 1),
                        np.full_like(edge_XmZm, -1),
                        np.full_like(edge_XpZp, -dx, dtype=float),
                        np.full_like(edge_XpZp, -sc * dz, dtype=float),
                    ],
                )
            )
            res.append(
                MPC(
                    [
                        edge_XpZp,
                        edge_XmZm,
                        np.full_like(edge_XpZp, node_cd[1][0]),
                        np.full_like(edge_XpZp, node_cd[1][2]),
                    ],
                    ["DispY", "DispY", var_cd[1][0], var_cd[1][2]],
                    [
                        np.full_like(edge_XpZp, 1),
                        np.full_like(edge_XmZm, -1),
                        np.full_like(edge_XpZp, -sc * dx, dtype=float),
                        np.full_like(edge_XpZp, -sc * dz, dtype=float),
                    ],
                )
            )
            res.append(
                MPC(
                    [
                        edge_XpZp,
                        edge_XmZm,
                        np.full_like(edge_XpZp, node_cd[2][0]),
                        np.full_like(edge_XpZp, node_cd[2][2]),
                    ],
                    ["DispZ", "DispZ", var_cd[2][0], var_cd[2][2]],
                    [
                        np.full_like(edge_XpZp, 1),
                        np.full_like(edge_XmZm, -1),
                        np.full_like(edge_XpZp, -sc * dx, dtype=float),
                        np.full_like(edge_XpZp, -dz, dtype=float),
                    ],
                )
            )

            # #### CORNER ####
            # elimination of DOF from corner Xp/Ym/Zm (XpYmZm) -> corner Xm/Ym/Zm (XmYmZm)
            res.append(
                MPC(
                    [
                        corner_XpYmZm,
                        corner_XmYmZm,
                        np.full_like(corner_XpYmZm, node_cd[0][0]),
                    ],
                    ["DispX", "DispX", var_cd[0][0]],
                    [
                        np.full_like(corner_XpYmZm, 1),
                        np.full_like(corner_XmYmZm, -1),
                        np.full_like(corner_XpYmZm, -dx, dtype=float),
                    ],
                )
            )
            res.append(
                MPC(
                    [
                        corner_XpYmZm,
                        corner_XmYmZm,
                        np.full_like(corner_XpYmZm, node_cd[1][0]),
                    ],
                    ["DispY", "DispY", var_cd[1][0]],
                    [
                        np.full_like(corner_XpYmZm, 1),
                        np.full_like(corner_XmYmZm, -1),
                        np.full_like(corner_XpYmZm, -sc * dx, dtype=float),
                    ],
                )
            )
            res.append(
                MPC(
                    [
                        corner_XpYmZm,
                        corner_XmYmZm,
                        np.full_like(corner_XpYmZm, node_cd[2][0]),
                    ],
                    ["DispZ", "DispZ", var_cd[2][0]],
                    [
                        np.full_like(corner_XpYmZm, 1),
                        np.full_like(corner_XmYmZm, -1),
                        np.full_like(corner_XpYmZm, -sc * dx, dtype=float),
                    ],
                )
            )
            # elimination of DOF from corner Xm/Yp/Zm (XmYpZm) -> corner Xm/Ym/Zm (XmYmZm)
            res.append(
                MPC(
                    [
                        corner_XmYpZm,
                        corner_XmYmZm,
                        np.full_like(corner_XmYpZm, node_cd[0][1]),
                    ],
                    ["DispX", "DispX", var_cd[0][1]],
                    [
                        np.full_like(corner_XmYpZm, 1),
                        np.full_like(corner_XmYmZm, -1),
                        np.full_like(corner_XmYpZm, -sc * dy, dtype=float),
                    ],
                )
            )
            res.append(
                MPC(
                    [
                        corner_XmYpZm,
                        corner_XmYmZm,
                        np.full_like(corner_XmYpZm, node_cd[1][1]),
                    ],
                    ["DispY", "DispY", var_cd[1][1]],
                    [
                        np.full_like(corner_XmYpZm, 1),
                        np.full_like(corner_XmYmZm, -1),
                        np.full_like(corner_XmYpZm, -dy, dtype=float),
                    ],
                )
            )
            res.append(
                MPC(
                    [
                        corner_XmYpZm,
                        corner_XmYmZm,
                        np.full_like(corner_XmYpZm, node_cd[2][1]),
                    ],
                    ["DispZ", "DispZ", var_cd[2][1]],
                    [
                        np.full_like(corner_XmYpZm, 1),
                        np.full_like(corner_XmYmZm, -1),
                        np.full_like(corner_XmYpZm, -sc * dy, dtype=float),
                    ],
                )
            )
            # elimination of DOF from corner Xm/Ym/Zp (XmYmZp) -> corner Xm/Ym/Zm (XmYmZm)
            res.append(
                MPC(
                    [
                        corner_XmYmZp,
                        corner_XmYmZm,
                        np.full_like(corner_XmYmZp, node_cd[0][2]),
                    ],
                    ["DispX", "DispX", var_cd[0][2]],
                    [
                        np.full_like(corner_XmYmZp, 1),
                        np.full_like(corner_XmYmZm, -1),
                        np.full_like(corner_XmYmZp, -sc * dz, dtype=float),
                    ],
                )
            )
            res.append(
                MPC(
                    [
                        corner_XmYmZp,
                        corner_XmYmZm,
                        np.full_like(corner_XmYmZp, node_cd[1][2]),
                    ],
                    ["DispY", "DispY", var_cd[1][2]],
                    [
                        np.full_like(corner_XmYmZp, 1),
                        np.full_like(corner_XmYmZm, -1),
                        np.full_like(corner_XmYmZp, -sc * dz, dtype=float),
                    ],
                )
            )
            res.append(
                MPC(
                    [
                        corner_XmYmZp,
                        corner_XmYmZm,
                        np.full_like(corner_XmYmZp, node_cd[2][2]),
                    ],
                    ["DispZ", "DispZ", var_cd[2][2]],
                    [
                        np.full_like(corner_XmYmZp, 1),
                        np.full_like(corner_XmYmZm, -1),
                        np.full_like(corner_XmYmZp, -dz, dtype=float),
                    ],
                )
            )
            # elimination of DOF from corner Xp/Yp/Zm (XpYpZm) -> corner Xm/Ym/Zm (XmYmZm)
            res.append(
                MPC(
                    [
                        corner_XpYpZm,
                        corner_XmYmZm,
                        np.full_like(corner_XpYpZm, node_cd[0][0]),
                        np.full_like(corner_XpYpZm, node_cd[0][1]),
                    ],
                    ["DispX", "DispX", var_cd[0][0], var_cd[0][1]],
                    [
                        np.full_like(corner_XpYpZm, 1),
                        np.full_like(corner_XmYmZm, -1),
                        np.full_like(corner_XpYpZm, -dx, dtype=float),
                        np.full_like(corner_XpYpZm, -sc * dy, dtype=float),
                    ],
                )
            )
            res.append(
                MPC(
                    [
                        corner_XpYpZm,
                        corner_XmYmZm,
                        np.full_like(corner_XpYpZm, node_cd[1][0]),
                        np.full_like(corner_XpYpZm, node_cd[1][1]),
                    ],
                    ["DispY", "DispY", var_cd[1][0], var_cd[1][1]],
                    [
                        np.full_like(corner_XpYpZm, 1),
                        np.full_like(corner_XmYmZm, -1),
                        np.full_like(corner_XpYpZm, -sc * dx, dtype=float),
                        np.full_like(corner_XpYpZm, -dy, dtype=float),
                    ],
                )
            )
            res.append(
                MPC(
                    [
                        corner_XpYpZm,
                        corner_XmYmZm,
                        np.full_like(corner_XpYpZm, node_cd[2][0]),
                        np.full_like(corner_XpYpZm, node_cd[2][1]),
                    ],
                    ["DispZ", "DispZ", var_cd[2][0], var_cd[2][1]],
                    [
                        np.full_like(corner_XpYpZm, 1),
                        np.full_like(corner_XmYmZm, -1),
                        np.full_like(corner_XpYpZm, -sc * dx, dtype=float),
                        np.full_like(corner_XpYpZm, -sc * dy, dtype=float),
                    ],
                )
            )
            # elimination of DOF from corner Xm/Yp/Zp (XmYpZp) -> corner Xm/Ym/Zm (XmYmZm)
            res.append(
                MPC(
                    [
                        corner_XmYpZp,
                        corner_XmYmZm,
                        np.full_like(corner_XmYpZp, node_cd[0][1]),
                        np.full_like(corner_XmYpZp, node_cd[0][2]),
                    ],
                    ["DispX", "DispX", var_cd[0][1], var_cd[0][2]],
                    [
                        np.full_like(corner_XmYpZp, 1),
                        np.full_like(corner_XmYmZm, -1),
                        np.full_like(corner_XmYpZp, -sc * dy, dtype=float),
                        np.full_like(corner_XmYpZp, -sc * dz, dtype=float),
                    ],
                )
            )
            res.append(
                MPC(
                    [
                        corner_XmYpZp,
                        corner_XmYmZm,
                        np.full_like(corner_XmYpZp, node_cd[1][1]),
                        np.full_like(corner_XmYpZp, node_cd[1][2]),
                    ],
                    ["DispY", "DispY", var_cd[1][1], var_cd[1][2]],
                    [
                        np.full_like(corner_XmYpZp, 1),
                        np.full_like(corner_XmYmZm, -1),
                        np.full_like(corner_XmYpZp, -dy, dtype=float),
                        np.full_like(corner_XmYpZp, -sc * dz, dtype=float),
                    ],
                )
            )
            res.append(
                MPC(
                    [
                        corner_XmYpZp,
                        corner_XmYmZm,
                        np.full_like(corner_XmYpZp, node_cd[2][1]),
                        np.full_like(corner_XmYpZp, node_cd[2][2]),
                    ],
                    ["DispZ", "DispZ", var_cd[2][1], var_cd[2][2]],
                    [
                        np.full_like(corner_XmYpZp, 1),
                        np.full_like(corner_XmYmZm, -1),
                        np.full_like(corner_XmYpZp, -sc * dy, dtype=float),
                        np.full_like(corner_XmYpZp, -dz, dtype=float),
                    ],
                )
            )
            # elimination of DOF from corner Xp/Ym/Zp (XpYmZp) -> corner Xm/Ym/Zm (XmYmZm)
            res.append(
                MPC(
                    [
                        corner_XpYmZp,
                        corner_XmYmZm,
                        np.full_like(corner_XpYmZp, node_cd[0][0]),
                        np.full_like(corner_XpYmZp, node_cd[0][2]),
                    ],
                    ["DispX", "DispX", var_cd[0][0], var_cd[0][2]],
                    [
                        np.full_like(corner_XpYmZp, 1),
                        np.full_like(corner_XmYmZm, -1),
                        np.full_like(corner_XpYmZp, -dx, dtype=float),
                        np.full_like(corner_XpYmZp, -sc * dz, dtype=float),
                    ],
                )
            )
            res.append(
                MPC(
                    [
                        corner_XpYmZp,
                        corner_XmYmZm,
                        np.full_like(corner_XpYmZp, node_cd[1][0]),
                        np.full_like(corner_XpYmZp, node_cd[1][2]),
                    ],
                    ["DispY", "DispY", var_cd[1][0], var_cd[1][2]],
                    [
                        np.full_like(corner_XpYmZp, 1),
                        np.full_like(corner_XmYmZm, -1),
                        np.full_like(corner_XpYmZp, -sc * dx, dtype=float),
                        np.full_like(corner_XpYmZp, -sc * dz, dtype=float),
                    ],
                )
            )
            res.append(
                MPC(
                    [
                        corner_XpYmZp,
                        corner_XmYmZm,
                        np.full_like(corner_XpYmZp, node_cd[2][0]),
                        np.full_like(corner_XpYmZp, node_cd[2][2]),
                    ],
                    ["DispZ", "DispZ", var_cd[2][0], var_cd[2][2]],
                    [
                        np.full_like(corner_XpYmZp, 1),
                        np.full_like(corner_XmYmZm, -1),
                        np.full_like(corner_XpYmZp, -sc * dx, dtype=float),
                        np.full_like(corner_XpYmZp, -dz, dtype=float),
                    ],
                )
            )

            # elimination of DOF from corner Xp/Yp/Zp (XpYpZp) -> corner Xm/Ym/Zm (XmYmZm)
            res.append(
                MPC(
                    [
                        corner_XpYpZp,
                        corner_XmYmZm,
                        np.full_like(corner_XpYpZp, node_cd[0][0]),
                        np.full_like(corner_XpYpZp, node_cd[0][1]),
                        np.full_like(corner_XpYpZp, node_cd[0][2]),
                    ],
                    ["DispX", "DispX", var_cd[0][0], var_cd[0][1], var_cd[0][2]],
                    [
                        np.full_like(corner_XpYpZp, 1),
                        np.full_like(corner_XmYmZm, -1),
                        np.full_like(corner_XpYpZp, -dx, dtype=float),
                        np.full_like(corner_XpYpZp, -sc * dy, dtype=float),
                        np.full_like(corner_XpYpZp, -sc * dz, dtype=float),
                    ],
                )
            )
            res.append(
                MPC(
                    [
                        corner_XpYpZp,
                        corner_XmYmZm,
                        np.full_like(corner_XpYpZp, node_cd[1][0]),
                        np.full_like(corner_XpYpZp, node_cd[1][1]),
                        np.full_like(corner_XpYpZp, node_cd[1][2]),
                    ],
                    ["DispY", "DispY", var_cd[1][0], var_cd[1][1], var_cd[1][2]],
                    [
                        np.full_like(corner_XpYpZp, 1),
                        np.full_like(corner_XmYmZm, -1),
                        np.full_like(corner_XpYpZp, -sc * dx, dtype=float),
                        np.full_like(corner_XpYpZp, -dy, dtype=float),
                        np.full_like(corner_XpYpZp, -sc * dz, dtype=float),
                    ],
                )
            )
            res.append(
                MPC(
                    [
                        corner_XpYpZp,
                        corner_XmYmZm,
                        np.full_like(corner_XpYpZp, node_cd[2][0]),
                        np.full_like(corner_XpYpZp, node_cd[2][1]),
                        np.full_like(corner_XpYpZp, node_cd[2][2]),
                    ],
                    ["DispZ", "DispZ", var_cd[2][0], var_cd[2][1], var_cd[2][2]],
                    [
                        np.full_like(corner_XpYpZp, 1),
                        np.full_like(corner_XmYmZm, -1),
                        np.full_like(corner_XpYpZp, -sc * dx, dtype=float),
                        np.full_like(corner_XpYpZp, -sc * dy, dtype=float),
                        np.full_like(corner_XpYpZp, -dz, dtype=float),
                    ],
                )
            )

        return res

    def _construct_faces_edges_corners_from_dic_non_periodic_node_distance(
        self, dic_closest_points_on_boundaries, d_rve
    ):
        self.face_Xm = dic_closest_points_on_boundaries("face_Xm")
        self.face_Ym = dic_closest_points_on_boundaries("face_Ym")
        self.face_Zm = dic_closest_points_on_boundaries("face_Zm")
        self.face_Xp = dic_closest_points_on_boundaries("face_Xp")
        self.face_Yp = dic_closest_points_on_boundaries("face_Yp")
        self.face_Zp = dic_closest_points_on_boundaries("face_Zp")
        self.edge_XmYm = dic_closest_points_on_boundaries("edge_XmYm")
        self.edge_XmZm = dic_closest_points_on_boundaries("edge_XmZm")
        self.edge_YmZm = dic_closest_points_on_boundaries("edge_YmZm")
        self.edge_XpYm = dic_closest_points_on_boundaries("edge_XpYm")
        self.edge_XpYp = dic_closest_points_on_boundaries("edge_XpYp")
        self.edge_XmYp = dic_closest_points_on_boundaries("edge_XmYp")
        self.edge_XpZm = dic_closest_points_on_boundaries("edge_XpZm")
        self.edge_XpZp = dic_closest_points_on_boundaries("edge_XpZp")
        self.edge_XmZp = dic_closest_points_on_boundaries("edge_XmZp")
        self.edge_YpZm = dic_closest_points_on_boundaries("edge_YpZm")
        self.edge_YpZp = dic_closest_points_on_boundaries("edge_YpZp")
        self.edge_YmZp = dic_closest_points_on_boundaries("edge_YmZp")
        self.corner_XmYmZm = dic_closest_points_on_boundaries("corner_XmYmZm")
        self.corner_XmYmZp = dic_closest_points_on_boundaries("corner_XmYmZp")
        self.corner_XmYpZm = dic_closest_points_on_boundaries("corner_XmYpZm")
        self.corner_XmYpZp = dic_closest_points_on_boundaries("corner_XmYpZp")
        self.corner_XpYmZm = dic_closest_points_on_boundaries("corner_XpYmZm")
        self.corner_XpYmZp = dic_closest_points_on_boundaries("corner_XpYmZp")
        self.corner_XpYpZm = dic_closest_points_on_boundaries("corner_XpYpZm")
        self.corner_XpYpZp = dic_closest_points_on_boundaries("corner_XpYpZp")

        self.d_rve = d_rve

    def _list_MPC_non_periodic_node_distance(self):
        node_cd = self.node_cd
        var_cd = self.var_cd

        dx = self.d_rve[0]
        dy = self.d_rve[1]
        dz = self.d_rve[2]

        sc = self.shear_coef

        face_Xm = self.face_Xm
        face_Ym = self.face_Ym
        face_Zm = self.face_Zm
        face_Xp = self.face_Xp
        face_Yp = self.face_Yp
        face_Zp = self.face_Zp
        edge_XmYm = self.edge_XmYm
        edge_XmZm = self.edge_XmZm
        edge_YmZm = self.edge_YmZm
        edge_XpYm = self.edge_XpYm
        edge_XpYp = self.edge_XpYp
        edge_XmYp = self.edge_XmYp
        edge_XpZm = self.edge_XpZm
        edge_XpZp = self.edge_XpZp
        edge_XmZp = self.edge_XmZp
        edge_YpZm = self.edge_YpZm
        edge_YpZp = self.edge_YpZp
        edge_YmZp = self.edge_YmZp
        corner_XmYmZm = self.corner_XmYmZm
        corner_XmYmZp = self.corner_XmYmZp
        corner_XmYpZm = self.corner_XmYpZm
        corner_XmYpZp = self.corner_XmYpZp
        corner_XpYmZm = self.corner_XpYmZm
        corner_XpYmZp = self.corner_XpYmZp
        corner_XpYpZm = self.corner_XpYpZm
        corner_XpYpZp = self.corner_XpYpZp

        D_xyz = [
            [-dx, -sc * dx, -sc * dx],
            [-sc * dy, -dy, -sc * dy],
            [-sc * dz, -sc * dz, -dz],
        ]

        list_disp = ["DispX", "DispY", "DispZ"]

        nbDOF = 3

        res = ListBC()

        # face Xm/Xp
        face_Xp_asarray = np.asarray(face_Xp[1])
        dimensions_to_factors_rescaled = face_Xp_asarray / np.sum(
            face_Xp_asarray, axis=1
        ).reshape(-1, 1)
        for i in range(0, nbDOF):
            p1 = 0
            list_node_sets = np.concatenate(
                (
                    face_Xp[0],
                    face_Xm,
                    np.full_like(face_Xm, node_cd[p1][i], dtype=object),
                ),
                axis=1,
            )
            list_variables = (
                [list_disp[i] for i in range(0, face_Xp_asarray.shape[1])]
                + [list_disp[i]]
                + [var_cd[p1][i]]
            )
            list_factors = np.concatenate(
                (
                    dimensions_to_factors_rescaled,
                    np.full_like(face_Xm, -1),
                    np.full_like(face_Xm, D_xyz[p1, i], dtype=float),
                ),
                axis=1,
            )
            res.append(MPC(list_node_sets, list_variables, list_factors))

        # face Ym/Yp
        face_Yp_asarray = np.asarray(face_Yp[1])
        dimensions_to_factors_rescaled = face_Yp_asarray / np.sum(
            face_Yp_asarray, axis=1
        ).reshape(-1, 1)
        for i in range(0, nbDOF):
            p1 = 1
            list_node_sets = np.concatenate(
                (
                    face_Yp[0],
                    face_Ym,
                    np.full_like(face_Ym, node_cd[p1][i], dtype=object),
                ),
                axis=1,
            )
            list_variables = (
                [list_disp[i] for i in range(0, face_Yp_asarray.shape[1])]
                + [list_disp[i]]
                + [var_cd[p1][i]]
            )
            list_factors = np.concatenate(
                (
                    dimensions_to_factors_rescaled,
                    np.full_like(face_Ym, -1),
                    np.full_like(face_Ym, D_xyz[p1, i], dtype=float),
                ),
                axis=1,
            )
            res.append(MPC(list_node_sets, list_variables, list_factors))

        # face Zm/Zp
        face_Zp_asarray = np.asarray(face_Zp[1])
        dimensions_to_factors_rescaled = face_Zp_asarray / np.sum(
            face_Zp_asarray, axis=1
        ).reshape(-1, 1)
        for i in range(0, nbDOF):
            p1 = 2
            list_node_sets = np.concatenate(
                (
                    face_Zp[0],
                    face_Zm,
                    np.full_like(face_Zm, node_cd[p1][i], dtype=object),
                ),
                axis=1,
            )
            list_variables = (
                [list_disp[i] for i in range(0, face_Zp_asarray.shape[1])]
                + [list_disp[i]]
                + [var_cd[p1][i]]
            )
            list_factors = np.concatenate(
                (
                    dimensions_to_factors_rescaled,
                    np.full_like(face_Zm, -1),
                    np.full_like(face_Zm, D_xyz[p1, i], dtype=float),
                ),
                axis=1,
            )
            res.append(MPC(list_node_sets, list_variables, list_factors))

        # edge_XpYm
        edge_XpYm_asarray = np.asarray(edge_XpYm[1])
        dimensions_to_factors_rescaled = edge_XpYm_asarray / np.sum(
            edge_XpYm_asarray, axis=1
        ).reshape(-1, 1)
        for i in range(0, nbDOF):
            p1 = 0
            list_node_sets = np.concatenate(
                (
                    edge_XpYm[0],
                    edge_XmYm,
                    np.full_like(edge_XmYm, node_cd[p1][i], dtype=object),
                ),
                axis=1,
            )
            list_variables = (
                [list_disp[i] for i in range(0, edge_XpYm_asarray.shape[1])]
                + [list_disp[i]]
                + [var_cd[p1][i]]
            )
            list_factors = np.concatenate(
                (
                    dimensions_to_factors_rescaled,
                    np.full_like(edge_XmYm, -1),
                    np.full_like(edge_XmYm, D_xyz[p1, i], dtype=float),
                ),
                axis=1,
            )
            res.append(MPC(list_node_sets, list_variables, list_factors))

        # edge_XpYp
        edge_XpYp_asarray = np.asarray(edge_XpYp[1])
        dimensions_to_factors_rescaled = edge_XpYp_asarray / np.sum(
            edge_XpYp_asarray, axis=1
        ).reshape(-1, 1)
        for i in range(0, nbDOF):
            p1 = 0
            p2 = 1
            list_node_sets = np.concatenate(
                (
                    edge_XpYp[0],
                    edge_XmYm,
                    np.full_like(edge_XmYm, node_cd[p1][i], dtype=object),
                    np.full_like(edge_XmYm, node_cd[p2][i], dtype=object),
                ),
                axis=1,
            )
            list_variables = (
                [list_disp[i] for i in range(0, edge_XpYp_asarray.shape[1])]
                + [list_disp[i]]
                + [var_cd[p1][i]]
                + [var_cd[p2][i]]
            )
            list_factors = np.concatenate(
                (
                    dimensions_to_factors_rescaled,
                    np.full_like(edge_XmYm, -1),
                    np.full_like(edge_XmYm, D_xyz[p1, i], dtype=float),
                    np.full_like(edge_XmYm, D_xyz[p2, i], dtype=float),
                ),
                axis=1,
            )
            res.append(MPC(list_node_sets, list_variables, list_factors))

        # edge_XmYp
        edge_XmYp_asarray = np.asarray(edge_XmYp[1])
        dimensions_to_factors_rescaled = edge_XmYp_asarray / np.sum(
            edge_XmYp_asarray, axis=1
        ).reshape(-1, 1)
        for i in range(0, nbDOF):
            p1 = 1
            list_node_sets = np.concatenate(
                (
                    edge_XmYp[0],
                    edge_XmYm,
                    np.full_like(edge_XmYm, node_cd[p1][i], dtype=object),
                ),
                axis=1,
            )
            list_variables = (
                [list_disp[i] for i in range(0, edge_XmYp_asarray.shape[1])]
                + [list_disp[i]]
                + [var_cd[p1][i]]
            )
            list_factors = np.concatenate(
                (
                    dimensions_to_factors_rescaled,
                    np.full_like(edge_XmYm, -1),
                    np.full_like(edge_XmYm, D_xyz[p1, i], dtype=float),
                ),
                axis=1,
            )
            res.append(MPC(list_node_sets, list_variables, list_factors))

        # edge_XpZm
        edge_XpZm_asarray = np.asarray(edge_XpZm[1])
        dimensions_to_factors_rescaled = edge_XpZm_asarray / np.sum(
            edge_XpZm_asarray, axis=1
        ).reshape(-1, 1)
        for i in range(0, nbDOF):
            p1 = 0
            list_node_sets = np.concatenate(
                (
                    edge_XpZm[0],
                    edge_XmZm,
                    np.full_like(edge_XmZm, node_cd[p1][i], dtype=object),
                ),
                axis=1,
            )
            list_variables = (
                [list_disp[i] for i in range(0, edge_XpZm_asarray.shape[1])]
                + [list_disp[i]]
                + [var_cd[p1][i]]
            )
            list_factors = np.concatenate(
                (
                    dimensions_to_factors_rescaled,
                    np.full_like(edge_XmZm, -1),
                    np.full_like(edge_XmZm, D_xyz[p1, i], dtype=float),
                ),
                axis=1,
            )
            res.append(MPC(list_node_sets, list_variables, list_factors))

        # edge_XpZp
        edge_XpZp_asarray = np.asarray(edge_XpZp[1])
        dimensions_to_factors_rescaled = edge_XpZp_asarray / np.sum(
            edge_XpZp_asarray, axis=1
        ).reshape(-1, 1)
        for i in range(0, nbDOF):
            p1 = 0
            p2 = 2
            list_node_sets = np.concatenate(
                (
                    edge_XpZp[0],
                    edge_XmZm,
                    np.full_like(edge_XmZm, node_cd[p1][i], dtype=object),
                    np.full_like(edge_XmZm, node_cd[p2][i], dtype=object),
                ),
                axis=1,
            )
            list_variables = (
                [list_disp[i] for i in range(0, edge_XpZp_asarray.shape[1])]
                + [list_disp[i]]
                + [var_cd[p1][i]]
                + [var_cd[p2][i]]
            )
            list_factors = np.concatenate(
                (
                    dimensions_to_factors_rescaled,
                    np.full_like(edge_XmZm, -1),
                    np.full_like(edge_XmZm, D_xyz[p1, i], dtype=float),
                    np.full_like(edge_XmZm, D_xyz[p2, i], dtype=float),
                ),
                axis=1,
            )
            res.append(MPC(list_node_sets, list_variables, list_factors))

        # edge_XmZp
        edge_XmZp_asarray = np.asarray(edge_XmZp[1])
        dimensions_to_factors_rescaled = edge_XmZp_asarray / np.sum(
            edge_XmZp_asarray, axis=1
        ).reshape(-1, 1)
        for i in range(0, nbDOF):
            p1 = 2
            list_node_sets = np.concatenate(
                (
                    edge_XmZp[0],
                    edge_XmZm,
                    np.full_like(edge_XmZm, node_cd[p1][i], dtype=object),
                ),
                axis=1,
            )
            list_variables = (
                [list_disp[i] for i in range(0, edge_XmZp_asarray.shape[1])]
                + [list_disp[i]]
                + [var_cd[p1][i]]
            )
            list_factors = np.concatenate(
                (
                    dimensions_to_factors_rescaled,
                    np.full_like(edge_XmZm, -1),
                    np.full_like(edge_XmZm, D_xyz[p1, i], dtZpe=float),
                ),
                axis=1,
            )
            res.append(MPC(list_node_sets, list_variables, list_factors))

        # edge_XpZm
        edge_XpZm_asarray = np.asarray(edge_XpZm[1])
        dimensions_to_factors_rescaled = edge_XpZm_asarray / np.sum(
            edge_XpZm_asarray, axis=1
        ).reshape(-1, 1)
        for i in range(0, nbDOF):
            p1 = 0
            list_node_sets = np.concatenate(
                (
                    edge_XpZm[0],
                    edge_XmZm,
                    np.full_like(edge_XmZm, node_cd[p1][i], dtype=object),
                ),
                axis=1,
            )
            list_variables = (
                [list_disp[i] for i in range(0, edge_XpZm_asarray.shape[1])]
                + [list_disp[i]]
                + [var_cd[p1][i]]
            )
            list_factors = np.concatenate(
                (
                    dimensions_to_factors_rescaled,
                    np.full_like(edge_XmZm, -1),
                    np.full_like(edge_XmZm, D_xyz[p1, i], dtype=float),
                ),
                axis=1,
            )
            res.append(MPC(list_node_sets, list_variables, list_factors))

        # edge_YpZm
        edge_YpZm_asarray = np.asarray(edge_YpZm[1])
        dimensions_to_factors_rescaled = edge_YpZm_asarray / np.sum(
            edge_YpZm_asarray, axis=1
        ).reshape(-1, 1)
        for i in range(0, nbDOF):
            p1 = 1
            list_node_sets = np.concatenate(
                (
                    edge_YpZm[0],
                    edge_YmZm,
                    np.full_like(edge_YmZm, node_cd[p1][i], dtype=object),
                ),
                axis=1,
            )
            list_variables = (
                [list_disp[i] for i in range(0, edge_YpZm_asarray.shape[1])]
                + [list_disp[i]]
                + [var_cd[p1][i]]
            )
            list_factors = np.concatenate(
                (
                    dimensions_to_factors_rescaled,
                    np.full_like(edge_YmZm, -1),
                    np.full_like(edge_YmZm, D_xyz[p1, i], dtype=float),
                ),
                axis=1,
            )
            res.append(MPC(list_node_sets, list_variables, list_factors))

        # edge_YpZp
        edge_YpZp_asarray = np.asarray(edge_YpZp[1])
        dimensions_to_factors_rescaled = edge_YpZp_asarray / np.sum(
            edge_YpZp_asarray, axis=1
        ).reshape(-1, 1)
        for i in range(0, nbDOF):
            p1 = 1
            p2 = 2
            list_node_sets = np.concatenate(
                (
                    edge_YpZp[0],
                    edge_YmZm,
                    np.full_like(edge_YmZm, node_cd[p1][i], dtype=object),
                    np.full_like(edge_YmZm, node_cd[p2][i], dtype=object),
                ),
                axis=1,
            )
            list_variables = (
                [list_disp[i] for i in range(0, edge_YpZp_asarray.shape[1])]
                + [list_disp[i]]
                + [var_cd[p1][i]]
                + [var_cd[p2][i]]
            )
            list_factors = np.concatenate(
                (
                    dimensions_to_factors_rescaled,
                    np.full_like(edge_YmZm, -1),
                    np.full_like(edge_YmZm, D_xyz[p1, i], dtype=float),
                    np.full_like(edge_YmZm, D_xyz[p2, i], dtype=float),
                ),
                axis=1,
            )
            res.append(MPC(list_node_sets, list_variables, list_factors))

        # edge_YmZp
        edge_YmZp_asarray = np.asarray(edge_YmZp[1])
        dimensions_to_factors_rescaled = edge_YmZp_asarray / np.sum(
            edge_YmZp_asarray, axis=1
        ).reshape(-1, 1)
        for i in range(0, nbDOF):
            p1 = 2
            list_node_sets = np.concatenate(
                (
                    edge_YmZp[0],
                    edge_YmZm,
                    np.full_like(edge_YmZm, node_cd[p1][i], dtype=object),
                ),
                axis=1,
            )
            list_variables = (
                [list_disp[i] for i in range(0, edge_YmZp_asarray.shape[1])]
                + [list_disp[i]]
                + [var_cd[p1][i]]
            )
            list_factors = np.concatenate(
                (
                    dimensions_to_factors_rescaled,
                    np.full_like(edge_YmZm, -1),
                    np.full_like(edge_YmZm, D_xyz[p1, i], dtZpe=float),
                ),
                axis=1,
            )
            res.append(MPC(list_node_sets, list_variables, list_factors))

        # Corners
        for i in range(0, nbDOF):
            # elimination of DOF from corner Xp/Ym/Zm (XpYmZm) -> corner Xm/Ym/Zm (XmYmZm)
            res.append(
                MPC(
                    [
                        corner_XpYmZm,
                        corner_XmYmZm,
                        np.full_like(corner_XpYmZm, node_cd[0][i]),
                    ],
                    [list_disp[i], list_disp[i], var_cd[0][i]],
                    [
                        np.full_like(corner_XpYmZm, 1),
                        np.full_like(corner_XmYmZm, -1),
                        np.full_like(corner_XpYmZm, D_xyz[0, i], dtype=float),
                    ],
                )
            )
            # elimination of DOF from corner Xm/Yp/Zm (XmYpZm) -> corner Xm/Ym/Zm (XmYmZm)
            res.append(
                MPC(
                    [
                        corner_XmYpZm,
                        corner_XmYmZm,
                        np.full_like(corner_XmYpZm, node_cd[1][i]),
                    ],
                    [list_disp[i], list_disp[i], var_cd[0][1]],
                    [
                        np.full_like(corner_XmYpZm, 1),
                        np.full_like(corner_XmYmZm, -1),
                        np.full_like(corner_XmYpZm, D_xyz[1, i], dtype=float),
                    ],
                )
            )
            # elimination of DOF from corner Xm/Ym/Zp (XmYmZp) -> corner Xm/Ym/Zm (XmYmZm)
            res.append(
                MPC(
                    [
                        corner_XmYmZp,
                        corner_XmYmZm,
                        np.full_like(corner_XmYmZp, node_cd[2][i]),
                    ],
                    [list_disp[i], list_disp[i], var_cd[0][2]],
                    [
                        np.full_like(corner_XmYmZp, 1),
                        np.full_like(corner_XmYmZm, -1),
                        np.full_like(corner_XmYmZp, D_xyz[2, i], dtype=float),
                    ],
                )
            )
            # elimination of DOF from corner Xp/Yp/Zm (XpYpZm) -> corner Xm/Ym/Zm (XmYmZm)
            res.append(
                MPC(
                    [
                        corner_XpYpZm,
                        corner_XmYmZm,
                        np.full_like(corner_XpYpZm, node_cd[0][i]),
                        np.full_like(corner_XpYpZm, node_cd[1][i]),
                    ],
                    [list_disp[i], list_disp[i], var_cd[0][i], var_cd[1][i]],
                    [
                        np.full_like(corner_XpYpZm, 1),
                        np.full_like(corner_XmYmZm, -1),
                        np.full_like(corner_XpYpZm, D_xyz[0, i], dtype=float),
                        np.full_like(corner_XpYpZm, D_xyz[1, i], dtype=float),
                    ],
                )
            )
            # elimination of DOF from corner Xm/Yp/Zp (XmYpZp) -> corner Xm/Ym/Zm (XmYmZm)
            res.append(
                MPC(
                    [
                        corner_XmYpZp,
                        corner_XmYmZm,
                        np.full_like(corner_XmYpZp, node_cd[1][i]),
                        np.full_like(corner_XmYpZp, node_cd[2][i]),
                    ],
                    [list_disp[i], list_disp[i], var_cd[1][i], var_cd[2][i]],
                    [
                        np.full_like(corner_XmYpZp, 1),
                        np.full_like(corner_XmYmZm, -1),
                        np.full_like(corner_XmYpZp, D_xyz[1, i], dtype=float),
                        np.full_like(corner_XmYpZp, D_xyz[2, i], dtype=float),
                    ],
                )
            )
            # elimination of DOF from corner Xp/Ym/Zp (XpYmZp) -> corner Xm/Ym/Zm (XmYmZm)
            res.append(
                MPC(
                    [
                        corner_XpYmZp,
                        corner_XmYmZm,
                        np.full_like(corner_XpYmZp, node_cd[0][i]),
                        np.full_like(corner_XpYmZp, node_cd[2][i]),
                    ],
                    [list_disp[i], list_disp[i], var_cd[0][i], var_cd[2][i]],
                    [
                        np.full_like(corner_XpYmZp, 1),
                        np.full_like(corner_XmYmZm, -1),
                        np.full_like(corner_XpYmZp, D_xyz[0, i], dtype=float),
                        np.full_like(corner_XpYmZp, D_xyz[2, i], dtype=float),
                    ],
                )
            )
            # elimination of DOF from corner Xp/Yp/Zp (XpYpZp) -> corner Xm/Ym/Zm (XmYmZm)
            res.append(
                MPC(
                    [
                        corner_XpYpZp,
                        corner_XmYmZm,
                        np.full_like(corner_XpYpZp, node_cd[0][i]),
                        np.full_like(corner_XpYpZp, node_cd[1][i]),
                        np.full_like(corner_XpYpZp, node_cd[2][i]),
                    ],
                    [
                        list_disp[i],
                        list_disp[i],
                        var_cd[0][i],
                        var_cd[1][i],
                        var_cd[2][i],
                    ],
                    [
                        np.full_like(corner_XpYpZp, 1),
                        np.full_like(corner_XmYmZm, -1),
                        np.full_like(corner_XpYpZp, D_xyz[0, i], dtype=float),
                        np.full_like(corner_XpYpZp, D_xyz[1, i], dtype=float),
                        np.full_like(corner_XpYpZp, D_xyz[2, i], dtype=float),
                    ],
                )
            )

        return res

    def _add_additional_rot_dof(self, problem, res, dic_faces_edges_periodic):
        face_Xm = dic_faces_edges_periodic("face_Xm")
        face_Ym = dic_faces_edges_periodic("face_Ym")
        face_Zm = dic_faces_edges_periodic("face_Zm")
        face_Xp = dic_faces_edges_periodic("face_Xp")
        face_Yp = dic_faces_edges_periodic("face_Yp")
        face_Zp = dic_faces_edges_periodic("face_Zp")
        edge_XmYm = dic_faces_edges_periodic("edge_XmYm")
        edge_XmZm = dic_faces_edges_periodic("edge_XmZm")
        edge_YmZm = dic_faces_edges_periodic("edge_YmZm")
        edge_XpYm = dic_faces_edges_periodic("edge_XpYm")
        edge_XpYp = dic_faces_edges_periodic("edge_XpYp")
        edge_XmYp = dic_faces_edges_periodic("edge_XmYp")
        edge_XpZm = dic_faces_edges_periodic("edge_XpZm")
        edge_XpZp = dic_faces_edges_periodic("edge_XpZp")
        edge_XmZp = dic_faces_edges_periodic("edge_XmZp")
        edge_YpZm = dic_faces_edges_periodic("edge_YpZm")
        edge_YpZp = dic_faces_edges_periodic("edge_YpZp")
        edge_YmZp = dic_faces_edges_periodic("edge_YmZp")
        corner_XmYmZm = dic_faces_edges_periodic("corner_XmYmZm")
        corner_XmYmZp = dic_faces_edges_periodic("corner_XmYmZp")
        corner_XmYpZm = dic_faces_edges_periodic("corner_XmYpZm")
        corner_XmYpZp = dic_faces_edges_periodic("corner_XmYpZp")
        corner_XpYmZm = dic_faces_edges_periodic("corner_XpYmZm")
        corner_XpYmZp = dic_faces_edges_periodic("corner_XpYmZp")
        corner_XpYpZm = dic_faces_edges_periodic("corner_XpYpZm")
        corner_XpYpZp = dic_faces_edges_periodic("corner_XpYpZp")

        list_var = (
            problem.space.list_variables()
        )  # list of variable id defined in the active modeling space

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
                    [face_Xp, face_Xm],
                    [var, var],
                    [np.full_like(face_Xp, 1), np.full_like(face_Xm, -1)],
                )
            )
            if self.dim > 1:
                res.append(
                    MPC(
                        [face_Yp, face_Ym],
                        [var, var],
                        [np.full_like(face_Yp, 1), np.full_like(face_Ym, -1)],
                    )
                )
            if self.dim > 2:
                res.append(
                    MPC(
                        [face_Zp, face_Zm],
                        [var, var],
                        [np.full_like(face_Zp, 1), np.full_like(face_Zm, -1)],
                    )
                )

            #### EDGES ####
            if self.dim > 1:
                res.append(
                    MPC(
                        [edge_XmYp, edge_XmYm],
                        [var, var],
                        [np.full_like(edge_XmYp, 1), np.full_like(edge_XmYm, -1)],
                    )
                )
                res.append(
                    MPC(
                        [edge_XpYm, edge_XmYm],
                        [var, var],
                        [np.full_like(edge_XpYm, 1), np.full_like(edge_XmYm, -1)],
                    )
                )
                res.append(
                    MPC(
                        [edge_XpYp, edge_XmYm],
                        [var, var],
                        [np.full_like(edge_XpYp, 1), np.full_like(edge_XmYm, -1)],
                    )
                )

            if self.dim > 2:
                res.append(
                    MPC(
                        [edge_YpZm, edge_YmZm],
                        [var, var],
                        [np.full_like(edge_YpZm, 1), np.full_like(edge_YmZm, -1)],
                    )
                )
                res.append(
                    MPC(
                        [edge_YmZp, edge_YmZm],
                        [var, var],
                        [np.full_like(edge_YmZp, 1), np.full_like(edge_YmZm, -1)],
                    )
                )
                res.append(
                    MPC(
                        [edge_YpZp, edge_YmZm],
                        [var, var],
                        [np.full_like(edge_YpZp, 1), np.full_like(edge_YmZm, -1)],
                    )
                )

                res.append(
                    MPC(
                        [edge_XpZm, edge_XmZm],
                        [var, var],
                        [np.full_like(edge_XpZm, 1), np.full_like(edge_XmZm, -1)],
                    )
                )
                res.append(
                    MPC(
                        [edge_XmZp, edge_XmZm],
                        [var, var],
                        [np.full_like(edge_XmZp, 1), np.full_like(edge_XmZm, -1)],
                    )
                )
                res.append(
                    MPC(
                        [edge_XpZp, edge_XmZm],
                        [var, var],
                        [np.full_like(edge_XpZp, 1), np.full_like(edge_XmZm, -1)],
                    )
                )

                #### CORNERS ####
                res.append(
                    MPC(
                        [corner_XpYmZm, corner_XmYmZm],
                        [var, var],
                        [
                            np.full_like(corner_XpYmZm, 1),
                            np.full_like(corner_XmYmZm, -1),
                        ],
                    )
                )
                res.append(
                    MPC(
                        [corner_XmYpZm, corner_XmYmZm],
                        [var, var],
                        [
                            np.full_like(corner_XmYpZm, 1),
                            np.full_like(corner_XmYmZm, -1),
                        ],
                    )
                )
                res.append(
                    MPC(
                        [corner_XmYmZp, corner_XmYmZm],
                        [var, var],
                        [
                            np.full_like(corner_XmYmZp, 1),
                            np.full_like(corner_XmYmZm, -1),
                        ],
                    )
                )
                res.append(
                    MPC(
                        [corner_XpYpZm, corner_XmYmZm],
                        [var, var],
                        [
                            np.full_like(corner_XpYpZm, 1),
                            np.full_like(corner_XmYmZm, -1),
                        ],
                    )
                )
                res.append(
                    MPC(
                        [corner_XmYpZp, corner_XmYmZm],
                        [var, var],
                        [
                            np.full_like(corner_XmYpZp, 1),
                            np.full_like(corner_XmYmZm, -1),
                        ],
                    )
                )
                res.append(
                    MPC(
                        [corner_XpYmZp, corner_XmYmZm],
                        [var, var],
                        [
                            np.full_like(corner_XpYmZp, 1),
                            np.full_like(corner_XmYmZm, -1),
                        ],
                    )
                )
                res.append(
                    MPC(
                        [corner_XpYpZp, corner_XmYmZm],
                        [var, var],
                        [
                            np.full_like(corner_XpYpZp, 1),
                            np.full_like(corner_XmYmZm, -1),
                        ],
                    )
                )

    def initialize(self, problem, dic_closest_points_on_boundaries=None):
        """
        Initialize a periodic boundary condition object using several multi-point constraints. The constraint drivers are constructed as follows, depending on `periodicity_type` and `dim`:
        - For `periodicity_type="small_strain"`:
            - dim=1:
                - `global_dof` is created for ['E_xx'] (1 DOF).
                - `node_cd = [problem.virtual_dof['E_xx']]`
                - `var_cd = ['E_xx']`
            - dim=2:
                - `virtual_dof` is created for ['E_xx', 'E_yy', 'E_xy'] (3 DOF).
                - `node_cd = [[problem.virtual_dof['E_xx'], problem.virtual_dof['E_xy']], [problem.virtual_dof['E_xy'], problem.virtual_dof['E_yy']]]`
                - `var_cd = [['Virtual', 'Virtual'], ['Virtual', 'Virtual']]`
            - dim=3:
                - `virtual_dof` is created for ['E_xx', 'E_yy', 'E_zz', 'E_xy', 'E_xz', 'E_yz'] (6 DOF).
                - `node_cd = [[problem.virtual_dof['E_xx'], problem.virtual_dof['E_xy'], problem.virtual_dof['E_xz']], [problem.virtual_dof['E_xy'], problem.virtual_dof['E_yy'], problem.virtual_dof['E_yz']], [problem.virtual_dof['E_xz'], problem.virtual_dof['E_yz'], problem.virtual_dof['E_zz']]]`
                - `var_cd = [['Virtual', 'Virtual', 'Virtual'], ['Virtual', 'Virtual', 'Virtual'], ['Virtual', 'Virtual', 'Virtual']]`
        - For `periodicity_type="finite_strain"`:
            - dim=1:
                - `virtual_dof` is created for ['DU_xx'] (1 DOF).
                - `node_cd = [problem.virtual_dof['DU_xx']]`
                - `var_cd = ['Virtual']`
            - dim=2:
                - `virtual_dof` is created for ['DU_xx', 'DU_xy', 'DU_yx', 'DU_yy'] (4 DOF).
                - `node_cd = [[problem.virtual_dof['DU_xx'], problem.virtual_dof['DU_xy']], [problem.virtual_dof['DU_yx'], problem.virtual_dof['DU_yy']]]`
                - `var_cd = [['Virtual', 'Virtual'], ['Virtual', 'Virtual']]`
            - dim=3:
                - `virtual_dof` is created for ['DU_xx', 'DU_xy', 'DU_xz', 'DU_yx', 'DU_yy', 'DU_yz', 'DU_zx', 'DU_zy', 'DU_zz'] (9 DOF).
                - `node_cd = [[problem.virtual_dof['DU_xx'], problem.virtual_dof['DU_xy'], problem.virtual_dof['DU_xz']], [problem.virtual_dof['DU_yx'], problem.virtual_dof['DU_yy'], problem.virtual_dof['DU_yz']], [problem.virtual_dof['DU_zx'], problem.virtual_dof['DU_zy'], problem.virtual_dof['DU_zz']]]`
                - `var_cd = [['Virtual', 'Virtual', 'Virtual'], ['Virtual', 'Virtual', 'Virtual'], ['Virtual', 'Virtual', 'Virtual']]`
        The `node_cd` and `var_cd` structures define the virtual DOFs and their associated variables for the constraint drivers, depending on the periodicity type and dimension.

        """
        mesh = problem.mesh
        if self.dim is None:
            self.dim = problem.space.ndim

        res = None

        if self.periodicity_type == "small_strain":
            if self.dim == 1:
                dof_indice = problem.add_global_dof(
                    ["E_xx"], 1, vector_name="MeanStrain"
                )
                var_cd = ["E_xx"]
                node_cd = [dof_indice]
            elif self.dim == 2:
                dof_indice = problem.add_global_dof(
                    ["E_xx", "E_yy", "E_xy"], 1, vector_name="MeanStrain"
                )
                var_cd = [
                    ["E_xx", "E_xy"],
                    ["E_xy", "E_yy"],
                ]
                node_cd = np.full((2, 2, 1), dof_indice)
            elif self.dim == 3:
                dof_indice = problem.add_global_dof(
                    ["E_xx", "E_yy", "E_zz", "E_xy", "E_xz", "E_yz"],
                    1,
                    "MeanStrain",
                )
                var_cd = [
                    ["E_xx", "E_xy", "E_xz"],
                    ["E_xy", "E_yy", "E_yz"],
                    ["E_xz", "E_yz", "E_zz"],
                ]
                node_cd = np.full((3, 3, 1), dof_indice)

        if self.periodicity_type == "finite_strain":
            if self.dim == 1:
                dof_indice = problem.add_global_dof(["DU_xx"], 1, "MeanGradDisp")
                var_cd = ["DU_xx"]
                node_cd = [dof_indice]
            elif self.dim == 2:
                dof_indice = problem.add_global_dof(
                    ["DU_xx", "DU_xy", "DU_yx", "DU_yy"], 1, "MeanGradDisp"
                )
                var_cd = [
                    ["DU_xx", "DU_xy"],
                    ["DU_yx", "DU_yy"],
                ]
                node_cd = np.full((2, 2, 1), dof_indice)
            elif self.dim == 3:
                dof_indice = problem.add_global_dof(
                    [
                        "DU_xx",
                        "DU_xy",
                        "DU_xz",
                        "DU_yx",
                        "DU_yy",
                        "DU_yz",
                        "DU_zx",
                        "DU_zy",
                        "DU_zz",
                    ],
                    1,
                    "MeanGradDisp",
                )
                var_cd = [
                    ["DU_xx", "DU_xy", "DU_xz"],
                    ["DU_yx", "DU_yy", "DU_yz"],
                    ["DU_zx", "DU_zy", "DU_zz"],
                ]
                node_cd = np.full((3, 3, 1), dof_indice)

        self.node_cd = node_cd
        self.var_cd = var_cd

        if self.off_axis_rotation is not None:
            problem.enable_mpc_coupling = True
            if self.dim == 3:
                self.off_axis_rot_matrix = self.off_axis_rotation.as_matrix()
            else:
                raise ValueError("off_axis_rotation is valid in 3D only")

            if self.periodicity_type == "small_strain":
                dof_indice = virtual_dof_loading = problem.add_global_dof(
                    ["E_11", "E_22", "E_33", "E_12", "E_13", "E_23"],
                    1,
                    "LocalMeanStrain",
                )
                var_cd_loading = [
                    ["E_11", "E_12", "E_13"],
                    ["E_12", "E_22", "E_23"],
                    ["E_13", "E_23", "E_33"],
                ]
                node_cd_loading = np.full((3, 3, 1), dof_indice)

            if self.periodicity_type == "fintie_strain":
                dof_indice = problem.add_global_dof(
                    [
                        "DU_11",
                        "DU_12",
                        "DU_13",
                        "DU_21",
                        "DU_22",
                        "DU_23",
                        "DU_31",
                        "DU_32",
                        "DU_33",
                    ],
                    1,
                    "LocalMeanGradDisp",
                )
                var_cd_loading = [
                    ["DU_11", "DU_12", "DU_13"],
                    ["DU_21", "DU_22", "DU_23"],
                    ["DU_31", "DU_32", "DU_33"],
                ]
                node_cd_loading = np.full((3, 3, 1), dof_indice)

            self.node_cd_loading = node_cd_loading
            self.var_cd_loading = var_cd_loading

        periodicity = self.meshperio

        if periodicity:
            self._prepare_periodic_lists(mesh, self.tol)
            if self.off_axis_rotation is None:
                res = self._list_MPC_periodic()
            if self.off_axis_rotation is not None:
                res = ListBC([self._list_MPC_rotation(), self._list_MPC_periodic()])
                # res = ListBC([self._list_MPC_periodic(), self._list_MPC_rotation()])
        else:
            if dic_closest_points_on_boundaries is None:
                raise
            else:
                res = self._list_MPC_non_periodic_node_distance(
                    dic_closest_points_on_boundaries, self.d_rve
                )

        res.initialize(problem)
        self.list_mpc = res

    def generate(self, problem, t_fact=1, t_fact_old=None):
        return self.list_mpc.generate(problem, t_fact, t_fact_old)
