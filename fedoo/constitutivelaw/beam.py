# #derive de ConstitutiveLaw
# #compatible with the simcoon strain and stress notation

from fedoo.core.base import ConstitutiveLaw
# from fedoo.util.voigt_tensors import StressTensorList, StrainTensorList

import numpy as np


class BeamProperties(ConstitutiveLaw):
    def __init__(self, material, A, Jx, Iyy, Izz, k=0, name=""):
        """
        Parameters
        ----------
        material: ConstitutiveLaw name (str) or ConstitutiveLaw object
            Material Constitutive Law used to get the elastic modulus and shear modulus
            The ConstitutiveLaw object should have attributes E and G that gives the young modulus and the shear modulus.
            (as :mod:`fedoo.constitutivelaw.ElasticIsotrop`)
        A: scalar or arrays of gauss point values
            Beam section area
        Jx: scalar or arrays of gauss point values
            Torsion constant
        Iyy: scalar or arrays of gauss point values
            Second moment of area with respect to y (beam local coordinate system)
        Izz:
            Second moment of area with respect to z (beam local coordinate system)
        k=0: scalar or arrays of gauss point values
            Shear shape factor. If k=0 (*default*) the beam use the bernoulli hypothesis
        name: str
            name of the WeakForm
        """
        if isinstance(material, str):
            material = ConstitutiveLaw[material]

        if name == "":
            name = material.name

        self.material = material

        self.A = A
        """Section surface"""

        self.Jx = Jx
        """Torsion constant"""

        self.Iyy = Iyy
        """Second moment of area with respect to y"""

        self.Izz = Izz
        """Second moment of area with respect to z"""

        self.k = k
        """Shear shape factor. if k=0, the bernoulli hypothesis is considered."""

        ConstitutiveLaw.__init__(self, name)  # heritage

    def get_beam_rigidity(self):
        E = self.material.E
        G = self.material.G

        shear_stiffness = self.k * G * self.A

        return [
            E * self.A,
            shear_stiffness,
            shear_stiffness,
            G * self.Jx,
            E * self.Iyy,
            E * self.Izz,
        ]


class BeamCircular(BeamProperties):
    def __init__(self, material, r, k=0.9, name=""):
        """
        Define properties for a beam with circular cross section.

        Parameters
        ----------
        material: ConstitutiveLaw name (str) or ConstitutiveLaw object
            Material Constitutive Law used to get the elastic modulus and shear modulus
            The ConstitutiveLaw object should have attributes E and G that gives the young modulus and the shear modulus.
            (as :mod:`fedoo.constitutivelaw.ElasticIsotrop`)
        r: radius of the beam section
        k: scalar or arrays of gauss point values
            Shear shape factor. If k=0 the beam use the bernoulli hypothesis.
            Default is set to 0.9 (usual value for cylindrical beam)
        name: str
            name of the WeakForm
        """
        self.r = r
        """Radius of the beam section"""
        A = np.pi * r**2
        Jx = np.pi / 2 * r**4
        Iyy = Izz = np.pi / 4 * r**4
        BeamProperties.__init__(self, material, A, Jx, Iyy, Izz, k, name)


class BeamPipe(BeamProperties):
    def __init__(self, material, r_int, r_ext, k=0.5, name=""):
        """
        Define properties for a beam with pipe cross section.

        Parameters
        ----------
        material: ConstitutiveLaw name (str) or ConstitutiveLaw object
            Material Constitutive Law used to get the elastic modulus and shear modulus
            The ConstitutiveLaw object should have attributes E and G that gives the young modulus and the shear modulus.
            (as :mod:`fedoo.constitutivelaw.ElasticIsotrop`)
        r: radius of the beam section
        k: scalar or arrays of gauss point values
            Shear shape factor. If k=0 the beam use the bernoulli hypothesis.
            Default is set to 0.5 (usual value for thin tube)
        name: str
            name of the WeakForm
        """
        self.r_int = r_int
        """Internal radius of the beam section"""

        self.r_ext = r_ext
        """External radius of the beam section"""

        A = np.pi * (r_ext**2 - r_int**2)
        Jx = np.pi / 2 * (r_ext**4 - r_int**4)
        Izz = Iyy = np.pi / 4 * (r_ext**4 - r_int**4)
        BeamProperties.__init__(self, material, A, Jx, Iyy, Izz, k, name)


class BeamRectangular(BeamProperties):
    def __init__(self, material, a, b=None, k=5 / 6, name=""):
        """
        Parameters
        ----------
        material: ConstitutiveLaw name (str) or ConstitutiveLaw object
            Material Constitutive Law used to get the elastic modulus and shear modulus
            The ConstitutiveLaw object should have attributes E and G that gives the young modulus and the shear modulus.
            (as :mod:`fedoo.constitutivelaw.ElasticIsotrop`)
        a: scalar or arrays of gauss point values
            Dimension of the beam section along the y axis
        b: scalar or arrays of gauss point values
            Dimension of the beam section along the z axis
        k: scalar or arrays of gauss point values
            Shear shape factor. If k=0 the beam use the bernoulli hypothesis.
            Default is set to 5/6 (usual value for rectangular beam)
        name: str
            name of the WeakForm
        """
        self.a = a
        """Dimension of the beam section along the y axis"""
        if b is None:
            b = a
        self.b = b
        """Dimension of the beam section along the z axis"""

        A = a * b
        if np.isscalar(a):
            if a > b:
                h = a
                w = b
            else:
                h = b
                w = a
        else:  # assume a and b are numpy arrays
            h = (a > b) * (a - b) + b
            w = (a > b) * (b - a) + a

        Jx = 0.33 * h * w**3 - 0.21 * w**4 + 0.017 * (w**2 / h) ** 4
        Iyy = b * a**3 / 12
        Izz = a * b**3 / 12

        BeamProperties.__init__(self, material, A, Jx, Iyy, Izz, k, name)
