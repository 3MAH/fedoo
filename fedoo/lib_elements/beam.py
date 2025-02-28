import numpy as np
from fedoo.lib_elements.element_base import Element1D, Element1DGeom2
from fedoo.lib_elements.element_list import CombinedElement


# --------------------------------------
# bernoulliBeam
# --------------------------------------
class BernoulliBeam_disp(Element1D):  # 2 nodes with derivatative dof
    name = "bernoullibeam_disp"
    n_nodes = 2

    def __init__(
        self, n_elm_gp=4, **kargs
    ):  # pour la matrice de masse on est sous-integré (il en faut 6), pour la matrice de rigidite -> reste à voir
        elmGeom = kargs.get("elmGeom", None)
        if elmGeom is not None:
            # if not(isinstance(elmGeom,lin2)):
            #     #TODO if required: for a correct implementation if elmGeom != lin2 we need the derivative of the shape fonction theta_i and theta_j on nodes/x (should be corrected to be = 1) instead of the lenght of the element
            #     print('WARNING: bernoulliBeam element should be associated with lin2 geometrical interpolation')
            self.L = elmGeom.detJ[:, 0]  # element lenght
        else:
            print("Unit lenght assumed")
            self.L = 1

        self.xi_nd = np.c_[[0.0, 1.0]]
        self.n_elm_gp = n_elm_gp
        Element1D.__init__(self, n_elm_gp)

    # Dans les fonctions suivantes, xi doit toujours être une matrice colonne

    def ShapeFunction(self, xi):
        # [(vi,vj,tetai,tetaj)]
        if np.isscalar(self.L) and self.L == 1:  # only for debug purpose
            return np.c_[
                (1 - 3 * xi**2 + 2 * xi**3),
                (3 * xi**2 - 2 * xi**3),
                (xi - 2 * xi**2 + xi**3),
                (-(xi**2) + xi**3),
            ]
        else:
            L = self.L.reshape(1, -1)
            return np.transpose(
                [
                    (1 - 3 * xi**2 + 2 * xi**3) + 0 * L,
                    (3 * xi**2 - 2 * xi**3) + 0 * L,
                    (xi - 2 * xi**2 + xi**3) * L,
                    (-(xi**2) + xi**3) * L,
                ],
                (2, 1, 0),
            )  # shape = (Nel, Nb_pg, Nddl=4)


class BernoulliBeam_rot(Element1D):  # 2 nodes with derivatative dof
    name = "bernoullibeam_rot"
    n_nodes = 2

    def __init__(
        self, n_elm_gp=4, **kargs
    ):  # pour la matrice de masse on est sous-integré (il en faut 6), pour la matrice de rigidite -> reste à voir
        elmGeom = kargs.get("elmGeom", None)
        if elmGeom is not None:
            # if not(isinstance(elmGeom,lin2)):
            #     #TODO if required: for a correct implementation if elmGeom != lin2 we need the derivative of the shape fonction theta_i and theta_j on nodes/x (should be corrected to be = 1) instead of the lenght of the element
            #     print('WARNING: bernoulliBeam element should be associated with lin2 geometrical interpolation')
            self.L = elmGeom.detJ[:, 0]  # element lenght
        else:
            print("Unit lenght assumed")
            self.L = 1

        self.xi_nd = np.c_[[0.0, 1.0]]
        self.n_elm_gp = n_elm_gp
        Element1D.__init__(self, n_elm_gp)

    def ShapeFunction(self, xi):
        # [(tetai,tetaj,vi,vj)]
        if np.isscalar(self.L) and self.L == 1:  # only for debug purpose
            return [
                np.array(
                    [
                        [
                            1 - 4 * x + 3 * x**2,
                            -2 * x + 3 * x**2,
                            -6 * x + 6 * x**2,
                            6 * x - 6 * x**2,
                        ]
                    ]
                )
                for x in xi[:, 0]
            ]
        else:
            L = self.L.reshape(1, -1)
            return np.transpose(
                [
                    (1 - 4 * xi + 3 * xi**2) + 0 * L,
                    (-2 * xi + 3 * xi**2) + 0 * L,
                    (1 / L) * (-6 * xi + 6 * xi**2),
                    (1 / L) * (6 * xi - 6 * xi**2),
                ],
                (2, 1, 0),
            )  # shape = (Nel, Nb_pg, Nddl=4)

    def ShapeFunctionDerivative(self, xi):
        # [(tetai,tetaj,vi,vj)]
        if np.isscalar(self.L) and self.L == 1:  # only for debug purpose
            return [
                np.array([[-4 + 6 * x, -2 + 6 * x, -6 + 12 * x, 6 - 12 * x]])
                for x in xi[:, 0]
            ]
        else:
            L = self.L.reshape(1, 1, -1)
            return np.transpose(
                [
                    (-4 + 6 * xi) + 0 * L,
                    (-2 + 6 * xi) + 0 * L,
                    (1 / L) * (-6 + 12 * xi),
                    (1 / L) * (6 - 12 * xi),
                ],
                (3, 2, 1, 0),
            )  # shape = (Nel, Nb_pg, Nd_deriv=1, Nddl=4)
        # return [np.array([[-4+6*x, -2+6*x, -6+12*x, 6-12*x]]) for x in xi[:,0]]


BernoulliBeam = CombinedElement(
    "bernoullibeam", "lin2", default_n_gp=4, local_csys=True
)
BernoulliBeam.dict_elm_type = {
    "DispY": BernoulliBeam_disp,
    "DispZ": BernoulliBeam_disp,
    "RotY": BernoulliBeam_rot,
    "RotZ": BernoulliBeam_rot,
}
BernoulliBeam.associated_variables = {
    "DispY": (1, "RotZ"),
    "DispZ": (-1, "RotY"),
    "RotY": (-1, "DispZ"),
    "RotZ": (1, "DispY"),
}


# --------------------------------------
# Timoshenko FCQ beam
# --------------------------------------
class BeamFCQ_lin2(Element1DGeom2, Element1D):
    name = "beamfcq_lin2"
    n_nodes = 3

    def __init__(self, n_elm_gp=2, **kargs):
        self.xi_nd = np.c_[[0.0, 1.0, 0.5]]
        self.n_elm_gp = n_elm_gp
        Element1D.__init__(self, n_elm_gp)

    # Dans les fonctions suivantes, xi doit toujours être une matrice colonne
    def ShapeFunction(self, xi):
        return np.c_[(1 - xi), xi, 0 * xi]

    def ShapeFunctionDerivative(self, xi):
        return [np.array([[-1.0, 1.0, 0]]) for x in xi]


class BeamFCQ_rot(Element1D):  # 2 nodes with derivatative dof
    name = "beamfcq_rot"
    n_nodes = 3

    def __init__(
        self, n_elm_gp=4, **kargs
    ):  # pour la matrice de masse on est sous-integré (il en faut 6), pour la matrice de rigidite -> reste à voir
        # elmGeom = kargs.get('elmGeom', None)
        # if elmGeom is not None:
        # if not(isinstance(elmGeom,lin2)):
        #     #TODO if required: for a correct implementation if elmGeom != lin2 we need the derivative of the shape fonction theta_i and theta_j on nodes/x (should be corrected to be = 1) instead of the lenght of the element
        #     print('WARNING: beamFCQM element should be associated with lin2 geometrical interpolation')
        # self.L = elmGeom.detJ[:,0] #element lenght
        # else:
        #     print('Unit lenght assumed')
        #     self.L = np.array([1])

        self.xi_nd = np.c_[[0.0, 1.0, 0.5]]
        self.n_elm_gp = n_elm_gp
        Element1D.__init__(self, n_elm_gp)

    # Dans les fonctions suivantes, xi doit toujours être une matrice colonne
    def ShapeFunction(self, xi):
        # [(tetai,tetaj,tetak)] #tetak -> internal dof without true physical sense

        # see "Ibrahim  Bitar,  St ́ephane  Grange,  Panagiotis  Kotronis,  Nathan  Benkemoun.   Diff ́erentes  for-mulations  ́el ́ements  finis  poutres  multifibres  pour  la  mod ́elisation  des  structures  sous  sollici-tations  statiques  et  sismiques.   9`eme  Colloque  National  de  l’Association  Fran ̧caise  du  G ́enieParasismique (AFPS), Nov 2015,  Marne-la-Vall ́ee,  France.  2015,  9`eme Colloque National del’Association Fran ̧caise du G ́enie Parasismique (AFPS).<hal-01300418 "
        xi = xi.ravel()
        return np.array(
            [(1 - xi) * (1 - 3 * xi), -xi * (2 - 3 * xi), 1 - (1 - 2 * xi) ** 2]
        ).T  # shape = (Nb_pg, Nddl=3)

    def ShapeFunctionDerivative(self, xi):
        return np.transpose(
            [6 * xi - 4, 6 * xi - 2, -8 * xi + 4], (1, 2, 0)
        )  # shape = (Nb_pg, Nd_deriv=1, Nddl=3)


class BeamFCQ_disp(Element1D):  # 2 nodes with derivatative dof
    name = "beamfcq_disp"
    n_nodes = 3

    def __init__(
        self, n_elm_gp=4, **kargs
    ):  # pour la matrice de masse on est sous-integré (il en faut 6), pour la matrice de rigidite -> reste à voir
        #     elmGeom = kargs.get('elmGeom', None)
        #     if elmGeom is not None:
        #     #     if not(isinstance(elmGeom,lin2)):
        #     #         #TODO if required: for a correct implementation if elmGeom != lin2 we need the derivative of the shape fonction theta_i and theta_j on nodes/x (should be corrected to be = 1) instead of the lenght of the element
        #     #         print('WARNING: beamFCQM element should be associated with lin2 geometrical interpolation')
        #         self.L = elmGeom.detJ[:,0] #element lenght
        #     else:
        #         print('Unit lenght assumed')
        #         self.L = np.array([1])

        self.xi_nd = np.c_[[0.0, 1.0, 0.5]]
        self.n_elm_gp = n_elm_gp
        Element1D.__init__(self, n_elm_gp)

    # Dans les fonctions suivantes, xi doit toujours être une matrice colonne
    def ShapeFunction(self, xi):
        # [(vi,vj,vk, 0, 0, vl)] #vk and vl are internal dof without physical sense. vl is taken in a non used internal dof related to another variable (dispx or rotx)

        # see "Ibrahim  Bitar,  St ́ephane  Grange,  Panagiotis  Kotronis,  Nathan  Benkemoun.   Diff ́erentes  for-mulations  ́el ́ements  finis  poutres  multifibres  pour  la  mod ́elisation  des  structures  sous  sollici-tations  statiques  et  sismiques.   9`eme  Colloque  National  de  l’Association  Fran ̧caise  du  G ́enieParasismique (AFPS), Nov 2015,  Marne-la-Vall ́ee,  France.  2015,  9`eme Colloque National del’Association Fran ̧caise du G ́enie Parasismique (AFPS).<hal-01300418 "
        # the shape functions for internal dof have been devedied by 2 to keep derivative = 1 on nodes
        xi = xi.ravel()
        return np.array(
            [
                (1 - xi) ** 2 * (1 + 2 * xi),
                xi**2 * (3 - 2 * xi),
                (1 - xi) ** 2 * xi,
                0 * xi,
                0 * xi,
                -(xi**2) * (1 - xi),
            ]
        ).T  # shape = (Nb_pg, Nddl=6)

    def ShapeFunctionDerivative(self, xi):
        return np.transpose(
            [
                6 * xi**2 - 6 * xi,
                -6 * xi**2 + 6 * xi,
                3 * xi**2 - 4 * xi + 1,
                0 * xi,
                0 * xi,
                3 * xi**2 - 2 * xi,
            ],
            (1, 2, 0),
        )  # shape = (Nb_pg, Nd_deriv=1, Nddl=6)


BeamFCQ = CombinedElement("beamfcq", "lin2", default_n_gp=4, local_csys=True)
BeamFCQ.dict_elm_type = {
    "DispX": BeamFCQ_lin2,
    "DispY": BeamFCQ_disp,
    "DispZ": BeamFCQ_disp,
    "RotX": BeamFCQ_lin2,
    "RotY": BeamFCQ_rot,
    "RotZ": BeamFCQ_rot,
}
BeamFCQ.associated_variables = {"DispY": (1, "DispX"), "DispZ": (1, "RotX")}


# --------------------------------------
# "beam" element
# Timoshenko FCQM beam
# see "Ibrahim  Bitar,  St ́ephane  Grange,  Panagiotis  Kotronis,  Nathan  Benkemoun.   Diff ́erentes  for-mulations  ́el ́ements  finis  poutres  multifibres  pour  la  mod ́elisation  des  structures  sous  sollici-tations  statiques  et  sismiques.   9`eme  Colloque  National  de  l’Association  Fran ̧caise  du  G ́enieParasismique (AFPS), Nov 2015,  Marne-la-Vall ́ee,  France.  2015,  9`eme Colloque National del’Association Fran ̧caise du G ́enie Parasismique (AFPS).<hal-01300418 "
# only work if beam properties dont change (linear problems), because the properties are embedded in the shape functions.
# --------------------------------------


class Beam_rotZ(Element1D):  # 2 nodes with derivatative dof
    name = "beam_rotz"
    n_nodes = 2

    def __init__(
        self, n_elm_gp=4, **kargs
    ):  # pour la matrice de masse on est sous-integré (il en faut 6), pour la matrice de rigidite -> reste à voir
        elmGeom = kargs.get("elmGeom", None)
        if elmGeom is not None:
            # if not(isinstance(elmGeom,lin2)):
            #     #TODO if required: for a correct implementation if elmGeom != lin2 we need the derivative of the shape fonction theta_i and theta_j on nodes/x (should be corrected to be = 1) instead of the lenght of the element
            #     print('WARNING: beamFCQM element should be associated with lin2 geometrical interpolation')
            self.L = elmGeom.detJ[:, 0]  # element lenght
        else:
            print("Unit lenght assumed")
            self.L = np.array([1])

        assembly = kargs.get("assembly", None)
        if assembly is not None:
            self.phi = self._compute_phi(assembly.weakform)
        else:
            self.phi = 0  # no shear effect

        self.xi_nd = np.c_[[0.0, 1.0]]
        self.n_elm_gp = n_elm_gp
        Element1D.__init__(self, n_elm_gp)

    def _compute_phi(self, weakform):
        if weakform is not None:
            try:
                k = weakform.properties.k
            except:
                raise NameError('Weakform not compatible with "beam" element')
            if not (np.isscalar(k) and k == 0):
                # E = weakform.properties.material.E
                nu = weakform.properties.material.nu
                return (
                    24
                    * weakform.properties.Izz
                    * (1 + nu)
                    / (k * weakform.properties.A * self.L**2)
                )
                # or in function of G:
                # self.phi = 12*E*weakform.properties.Izz/(k*G*weakform.properties.A*self.L**2)
        return 0  # no shear effect

    # Dans les fonctions suivantes, xi doit toujours être une matrice colonne
    def ShapeFunction(self, xi):
        # [(tetai,tetaj,vi,vj)]
        L = self.L.reshape(1, -1)
        if not (np.isscalar(self.phi) and self.phi == 0):
            phi = self.phi.reshape(1, -1)
            C = 1 / (1 + phi)
        else:
            phi = 0
            C = np.ones_like(L)

        Nv = (6 * C / L) * (xi**2 - xi)
        return np.transpose(
            [
                C * (3 * xi**2 - (4 + phi) * xi + 1 + phi),
                C * (3 * xi**2 - (2 - phi) * xi),
                Nv,
                -Nv,
            ],
            (2, 1, 0),
        )  # shape = (Nel, Nb_pg, Nddl=4)

    def ShapeFunctionDerivative(self, xi):
        L = self.L.reshape(1, 1, -1)
        if not (np.isscalar(self.phi) and self.phi == 0):
            phi = self.phi.reshape(1, 1, -1)
            C = 1 / (1 + phi)
        else:
            phi = 0
            C = np.ones_like(L)

        Nvprime = (6 * C / L) * (2 * xi - 1)
        return np.transpose(
            [C * (6 * xi - (4 + phi)), C * (6 * xi - (2 - phi)), Nvprime, -Nvprime],
            (3, 2, 1, 0),
        )  # shape = (Nel, Nb_pg, Nd_deriv=1, Nddl=4)


class Beam_dispY(Element1D):  # 2 nodes with derivatative dof
    name = "beam_dispy"
    n_nodes = 2

    _compute_phi = Beam_rotZ._compute_phi

    def __init__(
        self, n_elm_gp=4, **kargs
    ):  # pour la matrice de masse on est sous-integré (il en faut 6), pour la matrice de rigidite -> reste à voir
        elmGeom = kargs.get("elmGeom", None)
        if elmGeom is not None:
            #     if not(isinstance(elmGeom,lin2)):
            #         #TODO if required: for a correct implementation if elmGeom != lin2 we need the derivative of the shape fonction theta_i and theta_j on nodes/x (should be corrected to be = 1) instead of the lenght of the element
            #         print('WARNING: beamFCQM element should be associated with lin2 geometrical interpolation')
            self.L = elmGeom.detJ[:, 0]  # element lenght
        else:
            print("Unit lenght assumed")
            self.L = np.array([1])

        assembly = kargs.get("assembly", None)
        if assembly is not None:
            self.phi = self._compute_phi(assembly.weakform)
        else:
            self.phi = 0  # no shear effect

        self.xi_nd = np.c_[[0.0, 1.0]]
        self.n_elm_gp = n_elm_gp
        Element1D.__init__(self, n_elm_gp)

    # Dans les fonctions suivantes, xi doit toujours être une matrice colonne
    def ShapeFunction(self, xi):
        # [(vi,vj,tetai,tetaj)]
        L = self.L.reshape(1, -1)

        if not (np.isscalar(self.phi) and self.phi == 0):
            phi = self.phi.reshape(1, -1)
            C = 1 / (1 + phi)
        else:
            phi = 0
            C = np.ones_like(L)

        Nv2 = -C * (2 * xi**3 - 3 * xi**2 - phi * xi)
        Nv1 = 1 - Nv2
        Nth1 = C * L * (xi**3 - (2 + phi / 2) * xi**2 + (1 + phi / 2) * xi)
        Nth2 = C * L * (xi**3 - (1 - phi / 2) * xi**2 - (phi / 2) * xi)

        return np.transpose(
            [Nv1, Nv2, Nth1, Nth2], (2, 1, 0)
        )  # shape = (Nel, Nb_pg, Nddl=4)

    def ShapeFunctionDerivative(self, xi):
        L = self.L.reshape(1, 1, -1)

        if not (np.isscalar(self.phi) and self.phi == 0):
            phi = self.phi.reshape(1, 1, -1)
            C = 1 / (1 + phi)
        else:
            phi = 0
            C = np.ones_like(L)

        Nv1prime = C * (6 * xi**2 - 6 * xi - phi)
        Nv2prime = -Nv1prime
        Nth1prime = C * L * (3 * xi**2 - (4 + phi) * xi + (1 + phi / 2))
        Nth2prime = C * L * (3 * xi**2 - (2 - phi) * xi - (phi / 2))

        return np.transpose(
            [Nv1prime, Nv2prime, Nth1prime, Nth2prime], (3, 2, 1, 0)
        )  # shape = (Nel, Nb_pg, Nd_deriv=1, Nddl=4)


class Beam_rotY(Beam_rotZ):
    name = "beam_roty"

    def _compute_phi(self, weakform):
        if weakform is not None:
            try:
                k = weakform.properties.k
            except:
                raise NameError('Weakform not compatible with "beam" element')
            if not (np.isscalar(k) and k == 0):
                # E = weakform.material.E
                nu = weakform.properties.material.nu
                return (
                    24
                    * weakform.properties.Iyy
                    * (1 + nu)
                    / (k * weakform.properties.A * self.L**2)
                )
                # or in function of G:
                # self.phi = 12*E*weakform.properties.Iyy/(k*G*weakform.properties.A*self.L**2)
        return 0  # no shear effect


class Beam_dispZ(Beam_dispY):
    name = "beam_dispz"

    _compute_phi = Beam_rotY._compute_phi


# Beam = {'DispX':['lin2'],
#         'DispY':['beam_dispy', (1, 'RotZ')],
#         'DispZ':['beam_dispz', (-1, 'RotY')],
#         'RotX':['lin2'],
#         'RotY':['beam_roty', (-1, 'DispZ')],
#         'RotZ':['beam_rotz', (1, 'DispY')],
#         '__default':['lin2'],
#         '__local_csys':True}

Beam = CombinedElement("beam", "lin2", default_n_gp=4, local_csys=True)
Beam.dict_elm_type = {
    "DispY": Beam_dispY,
    "DispZ": Beam_dispZ,
    "RotY": Beam_rotY,
    "RotZ": Beam_rotZ,
}
Beam.associated_variables = {
    "DispY": (1, "RotZ"),
    "DispZ": (-1, "RotY"),
    "RotY": (-1, "DispZ"),
    "RotZ": (1, "DispY"),
}


# def SetProperties_Beam(Iyy, Izz, A, nu=None, k=1, E= None, G=None):
#     if np.isscalar(k) and k==0:
#         #no shear effect
#         Beam_rotZ._L2phi = Beam_dispY._L2phi = 0
#         Beam_rotY._L2phi = Beam_dispZ._L2phi = 0
#     elif nu is None:
#         if G is None or E is None: raise NameError('Missing property')
#         Beam_rotZ._L2phi = Beam_dispY._L2phi = 12*E*Izz/(k*G*A)
#         Beam_rotY._L2phi = Beam_dispZ._L2phi = 12*E*Iyy/(k*G*A)
#     else:
#         Beam_rotZ._L2phi = Beam_dispY._L2phi = 24*Izz*(1+nu)/(k*A)
#         Beam_rotY._L2phi = Beam_dispZ._L2phi = 24*Iyy*(1+nu)/(k*A)
