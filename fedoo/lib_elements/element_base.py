import numpy as np


class Element:
    # Lines of xi should contains the coordinates of each point to consider
    def shape_function(self, xi):
        pass  # to be defined in inherited classes

    # def shape_function_derivative(self,xi): pass # defined in inherited classes

    # return an array whose lines are the values of the form functions
    # (one line per point defined in xi)

    def compute_det_jacobian(self, vec_x, vec_xi):
        """
        Compute the Jacobian matrix and its determinent at Gauss points.

        This method assume an isoparametric element, i.e., the same shape functions are
        used for both field and geometry interpolation.

        Parameters
        ----------
        vec_x : 3D numpy.ndarray
            3D array where vec_x[el, i] contains the coordinates of the i-th node
            of the element `el`.
        vec_xi : 2D numpy.ndarray
            A 2D array where each row represents the local coordinates of a point
            (typically Gauss points) in the reference element. These coordinates are
            generally denoted ξ, η, and ζ.


        Notes
        -----
        The Jacobian matrix is computed and stored in the `jacobian_matrix` attribute,
        where `self.jacobian_matrix[el,k]` is the Jacobian of element el at Gauss point
        k, in the form:
            [[dx/dξ, dy/dξ, ...],
             [dx/dη, dy/dη, ...],
             ...]

        The determinant of the Jacobian matrix for the k-th Gauss point of element el
        is also computed and stored in self.detJ[el,k].
        """
        # Shape function derivatives - shape = (n_gp, dim_ref, n_nodes)
        dnn_xi = self.shape_function_derivative(vec_xi)

        # Jacobian matrix
        # gin: Gauss, RefDim, Nodes
        # enj: Element, Nodes, PhysDim
        # Result: (Nel, n_gp, dim_ref, dim_phys)
        n_nodes = dnn_xi[0].shape[-1]
        self.jacobian_matrix = np.einsum("gin,enj->egij", dnn_xi, vec_x[:, :n_nodes])

        # Determinant calculation based on dimensions
        # Last two dims of jacobian_matrix are (dim_ref, dim_phys)
        d_ref = self.jacobian_matrix.shape[-2]
        d_phys = self.jacobian_matrix.shape[-1]

        if d_ref == d_phys:
            # Standard square Jacobian (e.g., 2D element in 2D space)
            self.detJ = np.abs(np.linalg.det(self.jacobian_matrix))

        elif d_ref == 1:
            # 1D Element in 2D or 3D space (Line/Bar)
            # detJ is simply the Euclidean norm of the tangent vector
            self.detJ = np.linalg.norm(self.jacobian_matrix, axis=-1).squeeze(axis=-1)

        elif d_ref == 2 and d_phys == 3:
            # 2D Element in 3D space (Shell/Surface)
            # detJ is the magnitude of the cross product of the two tangent vectors
            # J[..., 0, :] is dX/dxi, J[..., 1, :] is dX/deta
            cross_prod = np.cross(
                self.jacobian_matrix[..., 0, :], self.jacobian_matrix[..., 1, :]
            )
            self.detJ = np.linalg.norm(cross_prod, axis=-1)

        else:
            # General case (e.g., higher dimensions or mixed manifolds)
            # Uses the property: det(J) = sqrt(det(J * J_transpose))
            # This is the most general way to find the "area/volume" scaling factor
            jtj = np.einsum(
                "egij,egik->egjk", self.jacobian_matrix, self.jacobian_matrix
            )
            self.detJ = np.sqrt(np.abs(np.linalg.det(jtj)))

    def compute_jacobian_with_inverse(self, vec_x, vec_xi=None, local_frame=None):
        """
        Compute the Jacobian matrix and its inverse at Gauss points.

        This method assume an isoparametric element, i.e., the same shape functions are
        used for both field and geometry interpolation. The Jacobian matrix and its
        inverse may be computed relatively to a local frame.

        Parameters
        ----------
        vec_x : 3D numpy.ndarray
            3D array where vec_x[el, i] contains the coordinates of the i-th node
            of the element `el`.
        vec_xi : 2D numpy.ndarray
            A 2D array where each row represents the local coordinates of a point
            (typically Gauss points) in the reference element. These coordinates are
            generally denoted ξ, η, and ζ.
        local_frame: numpy.ndarray, optional
            local frames defined at elements or Gauss points. If given, the derivative
            are computed with respect to the local frame.

        Notes
        -----
        The Jacobian matrix and its inverse are computed and stored in the
        `jacobian_matrix` and `inv_jacobian_matrix` attributes,
        where `self.jacobian_matrix[el,k]` is the Jacobian of element el at Gauss point
        k, in the form:
            [[dx/dξ, dy/dξ, ...],
             [dx/dη, dy/dη, ...],
             ...]
        and `self.inv_jacobian_matrix[el,k]` is the inverse of the Jacobian of element
        el at Gauss point k, in the form:
            [[dξ/dx, dη/dx, ...],
             [dξ/dy, dη/dy, ...],
             ...]
        The determinant of the Jacobian matrix for the k-th Gauss point of element el
        is stored in self.detJ[el,k].
        """
        if vec_xi is None:
            vec_xi = self.xi_pg

        self.compute_det_jacobian(vec_x, vec_xi)

        if local_frame is not None:
            # local_frame = self.interpolate_local_frame(
            #     local_frame, vec_xi
            # )  # interpolation from node to gauss points
            self.jacobian_matrix = self.jacobian_matrix @ local_frame.transpose(
                0, 1, 3, 2
            )

        # 3. Inverse Calculation
        J = self.jacobian_matrix

        if J.shape[-2] == J.shape[-1]:
            # Square matrix: (e.g., 2D element in 2D space)
            self.inv_jacobian_matrix = np.linalg.inv(J)
        else:
            # Manifold case: (e.g., 2D shell in 3D space)
            JT = J.transpose(0, 1, 3, 2)
            # Gramian-based pseudo-inverse
            self.inv_jacobian_matrix = JT @ np.linalg.inv(J @ JT)
            # inverseJacobian.shape = (Nel,n_elm_gp, ndim_phy, ndim_ref_elm)

    def interpolate_local_frame(self, local_frame_node, vec_xi=None):
        if vec_xi is None:
            vec_xi = self.xi_pg

        # N shape: (n_gp, n_nodes)
        N = self.shape_function(vec_xi)

        # R_gp shape: (Nel, n_gp, 3, 3)
        R_gp = np.einsum("gn,enij->egij", N, local_frame_node)

        # To keep it a valid rotation matrix (orthonormal):
        # We use a simplified Gram-Schmidt or SVD to "clean" the rotation
        u, s, vh = np.linalg.svd(R_gp)
        R_gp_ortho = u @ vh

        return R_gp_ortho

    def get_local_frame_from_jacobian(self, guide=None):
        n_elements, n_elm_gp = self.inv_jacobian_matrix.shape[:2]
        local_frames = np.zeros((n_elements, n_elm_gp, 3, 3))
        local_frames[..., [0, 1, 2], [0, 1, 2]] = 1.0
        return local_frames

    def get_local_frame(self, vec_x, vec_xi, guide=None):
        self.compute_det_jacobian(vec_x, vec_xi)
        return self.get_local_frame_from_jacobian(guide)


class Element1D(Element):
    def __init__(self, n_elm_gp):
        if n_elm_gp == 0:  # if n_elm_gp == 0, we take the position of the nodes
            self.xi_pg = self.xi_nd
        else:
            self.xi_pg = self.get_gp_elm_coordinates(n_elm_gp)  # = np.c_[xi,eta]
            self.w_pg = self.get_gp_weight(n_elm_gp)

        self.shape_function_gp = self.shape_function(self.xi_pg)

        if hasattr(self, "shape_function_derivative"):
            self.shape_function_derivative_gp = self.shape_function_derivative(
                self.xi_pg
            )

    def get_gp_elm_coordinates(self, n_elm_gp):
        if n_elm_gp == 1:
            return np.array([[0.5]])
        elif n_elm_gp == 2:  # exact order 2
            return np.c_[[0.5 - np.sqrt(3) / 6, 0.5 + np.sqrt(3) / 6]]
        elif n_elm_gp == 3:  # exact order 3
            return np.c_[[0.5 - np.sqrt(0.15), 0.5, 0.5 + np.sqrt(0.15)]]
        elif n_elm_gp == 4:
            a_1 = 0.5 * (1 + np.sqrt((3.0 - 2.0 * np.sqrt(6.0 / 5.0)) / 7.0))
            b_1 = 0.5 * (1 - np.sqrt((3.0 - 2.0 * np.sqrt(6.0 / 5.0)) / 7.0))
            a_2 = 0.5 * (1 + np.sqrt((3.0 + 2.0 * np.sqrt(6.0 / 5.0)) / 7.0))
            b_2 = 0.5 * (1 - np.sqrt((3.0 + 2.0 * np.sqrt(6.0 / 5.0)) / 7.0))
            return np.c_[[b_2, b_1, a_1, a_2]]
        else:
            assert 0, (
                "Number of gauss points "
                + str(n_elm_gp)
                + " unavailable for 1D element"
            )

    def get_gp_weight(self, n_elm_gp):
        if n_elm_gp == 1:
            return np.array([1.0])
        elif n_elm_gp == 2:  # exact order 2
            return np.array([1.0 / 2, 1.0 / 2])
        elif n_elm_gp == 3:  # exact order 3
            return np.array([5.0 / 18, 8.0 / 18, 5.0 / 18])
        elif n_elm_gp == 4:
            w_1 = 0.5 + 1.0 / (6.0 * np.sqrt(6.0 / 5.0))
            w_2 = 0.5 - 1.0 / (6.0 * np.sqrt(6.0 / 5.0))
            return np.array([w_2 / 2, w_1 / 2, w_1 / 2, w_2 / 2])
        else:
            assert 0, (
                "Number of gauss points "
                + str(n_elm_gp)
                + " unavailable for 1D element"
            )

    def compute_det_jacobian(self, vec_x, vec_xi):
        dnn_xi = self.shape_function_derivative(vec_xi)

        # Compute the tangent vector (Jacobian in 1D)
        # Result shape: (Nel, n_gp, 1, dim_phys)
        tangent_vector = np.einsum("gin,enj->egij", dnn_xi, vec_x)

        # Compute the determinant (the length of the tangent vector)
        # self.detJ shape: (Nel, n_gp)
        self.detJ = np.linalg.norm(tangent_vector, axis=-1).squeeze(axis=-1)

        # Store the "Matrix" form if needed by other methods
        self.jacobian_matrix = self.detJ[..., np.newaxis, np.newaxis]

    def compute_jacobian_with_inverse(self, vec_x, vec_xi=None, local_frame=None):
        """Compute the Jacobian matrix and its inverse at Gauss points for 1d element.

        The local frame isn't used
        The jacobian is computed along the axis of the 1D element
        """
        if vec_xi is None:
            vec_xi = self.xi_pg
        self.compute_det_jacobian(vec_x, vec_xi)
        self.inv_jacobian_matrix = 1.0 / self.jacobian_matrix  # dxi_dx

    def _complete_beam_frame(self, x_vec, guide=None, guide_direction="y"):
        """Complete the frame using vectorized cross products.

        x_vec: (Nel, n_gp, 3) - The normalized tangents.
        """
        if x_vec.shape[-1] == 2:  # 2d space
            # guide and guide_direction are ignored.
            # the y direction is set so that the x,y,z local frame is direct
            y_vec = np.stack([-x_vec[..., 1], x_vec[..., 0]], axis=-1)
            return np.stack([x_vec, y_vec], axis=-2)

        # Guide Vector
        if guide is None:
            guide = (
                np.array([0.0, 1.0, 0.0])
                if guide_direction == "y"
                else np.array([0.0, 0.0, 1.0])
            )
        guide = np.array(guide)

        # Compute Orthogonal Vector with Guide
        # We'll compute the cross product for everything first
        v_ortho = (
            np.cross(x_vec, guide) if guide_direction == "y" else np.cross(guide, x_vec)
        )
        norm_v = np.linalg.norm(v_ortho, axis=-1, keepdims=True)

        # Handle Singularity (Parallel cases)
        singular = norm_v[..., 0] < 1e-6
        if np.any(singular):
            # For singular elements, find the global axis least parallel to x_vec
            # We compare absolute components of the tangent
            # If |x| is smallest, Global X is the best fallback, etc.
            x_abs = np.abs(x_vec[singular])
            min_axis = np.argmin(x_abs, axis=-1)  # 0 for X, 1 for Y, 2 for Z

            # Pick the best fallback guide for each singular element
            best_fallbacks = np.eye(3)[min_axis]

            # Recalculate for singular cases
            if guide_direction == "y":
                v_singular = np.cross(x_vec[singular], best_fallbacks)
            else:
                v_singular = np.cross(best_fallbacks, x_vec[singular])

            v_ortho[singular] = v_singular
            norm_v[singular] = np.linalg.norm(v_singular, axis=-1, keepdims=True)

        # Finalize the Orthonormal Set
        v_ortho /= norm_v

        if guide_direction == "y":
            # Primary was y_guide -> v_ortho is z_vec
            z_vec = v_ortho
            y_vec = np.cross(z_vec, x_vec)
        else:
            # Primary was z_guide -> v_ortho is y_vec
            y_vec = v_ortho
            z_vec = np.cross(x_vec, y_vec)

        return np.stack([x_vec, y_vec, z_vec], axis=-2)

    def get_local_frame(self, vec_x, vec_xi, guide=None, guide_direction="y"):
        """Compute the orthonormal local frame for beam elements.

        x_axis is Tangent to the beam (from Jacobian).
        y/z_axis: Derived from the guide vector that approximate y or z direction.
        """
        # Compute Tangent (Local X-axis)
        dnn_xi = self.shape_function_derivative(vec_xi)
        tangent = np.einsum("gin,enj->egj", dnn_xi, vec_x)
        x_vec = tangent / np.linalg.norm(tangent, axis=-1, keepdims=True)
        # returned shape = (Nel, len(listX) = n_elm_gp, dim:listvec, dim:coordinates)
        return self._complete_beam_frame(x_vec, guide, guide_direction)


class Element1DGeom2(Element1D):
    def compute_det_jacobian(self, vec_x, vec_xi):
        """
        Compute the Jacobian matrix and its determinent for 1D linear element.

        Parameters
        ----------
        vec_x : 3D numpy.ndarray
            3D array where vec_x[el, i] contains the coordinates of the i-th node
            of the element `el`.
        vec_xi : 2D numpy.ndarray
            A 2D array where each row represents the local coordinate of a point
            (typically Gauss points) in the reference element. As the element is 1d,
            these local coordinate is a scalar generally denoted ξ.
        """
        x1 = vec_x[:, 0]
        x2 = vec_x[:, 1]
        self.jacobian_matrix = np.linalg.norm(
            x2 - x1, axis=1
        )  # True element length because the ref element length is 1.
        # shape = (vec_x.shape[0] = Nel, len(vec_xi)=n_elm_gp, nb_dir_derivative, vec_x.shape[2] = dim)
        self.detJ = self.jacobian_matrix.reshape(-1, 1) * np.ones(
            len(vec_xi)
        )  # detJ is constant over the element.

    def compute_jacobian_with_inverse(self, vec_x, vec_xi=None, rep_loc=None):
        # rep_loc inutile ici : le repère local élémentaire est utilisé (x : tangeante à l'élément)
        if vec_xi is None:
            vec_xi = self.xi_pg
        self.compute_det_jacobian(vec_x, vec_xi)
        self.inv_jacobian_matrix = (1.0 / self.jacobian_matrix).reshape(
            -1, 1, 1, 1
        )  # dxi/dx -> scalar #shape = (vec_x.shape[0] = Nel, len(vec_xi)=n_elm_gp, nb_dir_derivative, vec_x.shape[2] = dim)

    def get_local_frame(self, vec_x, vec_xi, guide=None, guide_direction="y"):
        """Compute the orthonormal local frame for beam elements.

        x_axis is Tangent to the beam (from Jacobian).
        y/z_axis: Derived from the guide vector that approximate y or z direction.
        """
        # Compute Tangent (Local X-axis)
        if len(vec_x.shape) == 2:
            vec_x = np.array([vec_x])
        tangent = vec_x[:, 1, :] - vec_x[:, 0, :]
        x_vec = tangent / np.linalg.norm(tangent, axis=-1, keepdims=True)

        # Expand to (Nel, 1, 3) to keep consistent with Gauss points
        x_vec = x_vec[:, np.newaxis, :]

        return self._complete_beam_frame(x_vec, guide, guide_direction)


class Element2D(Element):
    def compute_jacobian_with_inverse(self, vec_x, vec_xi=None, local_frame=None):
        if vec_xi is None:
            vec_xi = self.xi_pg

        self.compute_det_jacobian(vec_x, vec_xi)

        if self.jacobian_matrix.shape[-2] == self.jacobian_matrix.shape[-1]:
            # CASE: 2D Elements in 2D Space
            if local_frame is not None:
                # J_local = J_global @ R.T
                self.jacobian_matrix = self.jacobian_matrix @ local_frame.transpose(
                    0, 1, 3, 2
                )
        else:
            # CASE: 2D Elements in 3D Space (Shells)
            # We MUST have a local frame to project into a 2x2 matrix
            if local_frame is None:
                # Fallback if no frame provided: generate one from Jacobian
                local_frame = self.get_local_frame_from_jacobian()

            # Slicing: [..., :2, :] picks the two in-plane basis vectors
            # Resulting local_frame: (Nel, n_gp, 2, 3)
            R_plane = local_frame[..., :2, :]

            # Project: (Nel, n_gp, 2, 3) @ (Nel, n_gp, 3, 2) -> (Nel, n_gp, 2, 2)
            # This makes the Jacobian square and invertible!
            self.jacobian_matrix = self.jacobian_matrix @ R_plane.transpose(0, 1, 3, 2)

        self.inv_jacobian_matrix = np.linalg.inv(self.jacobian_matrix)

    def get_local_frame_from_jacobian(self, guide_x=None):
        """Compute a local orthonormal frame (x, y, z) for shells.

        listZ is the normal. listX is the projection of 'guide_x' onto the plane.
        """
        # Compute Normal (Z-axis)
        # J is (Nel, n_gp, 2, 3). J[..., 0] is dX/dxi, J[..., 1] is dX/deta
        listZ = np.cross(
            self.jacobian_matrix[..., 0, :], self.jacobian_matrix[..., 1, :]
        )
        listZ /= np.linalg.norm(listZ, axis=-1, keepdims=True)

        # Define the Guide Vector (Approximate X)
        if guide_x is None:
            # Default guide is global X
            x_guide = np.array([1.0, 0.0, 0.0])
        else:
            x_guide = np.array(guide_x)

        # Project Guide X onto the tangent plane: X_local = X_guide - (X_guide . Z) * Z
        dot_z_x = np.einsum("...i,i->...", listZ, x_guide)
        listX = x_guide - dot_z_x[..., np.newaxis] * listZ

        # Handle Singularity (if guide_x is parallel to the normal listZ)
        normX = np.linalg.norm(listX, axis=-1)
        mask_singular = normX < 1e-6

        if np.any(mask_singular):
            # Use Global Y as a fallback guide for singular points
            y_fallback = np.array([0.0, 1.0, 0.0])
            dot_z_y = np.einsum("...i,i->...", listZ[mask_singular], y_fallback)
            listX[mask_singular] = (
                y_fallback - dot_z_y[..., np.newaxis] * listZ[mask_singular]
            )
            normX[mask_singular] = np.linalg.norm(listX[mask_singular], axis=-1)

        # Finalize Orthonormal Frame
        listX /= normX[..., np.newaxis]  # Normalize X
        listY = np.cross(listZ, listX)  # Y = Z cross X

        # Return shape (Nel, n_gp, 3, 3)
        # where [..., 0, :] is X, [..., 1, :] is Y, [..., 2, :] is Z
        return np.stack([listX, listY, listZ], axis=-2)
