Weak Formulation
=================================

The differential equations related to the problem to solve are written 
in weak formulations. 
Fedoo include a few classical weak formulation. Each weak formulation is a class
deriving from the WeakForm class. There, the developpement 
of new weak formulation is easy by copying and modifying and existing class. 

The WeakForm library contains the following classes: 


* InternalForce(CurrentConstitutiveLaw, ID = "", nlgeom = True)

    * The weak formulation of the mechanical equilibrium equation for solid models (without volume force).
    * May include initial stress depending on the ConstitutiveLaw.
    * Available for geometrical non linearities (with nlgeom = True). In this case the initial displacement is also considered.

* InterfaceForce(CurrentConstitutiveLaw, ID = "")

    * The weak formulation of the interface equilibrium equation.
    * Require an interface constitutive law such as Cohesive Zone law
    * Geometrical non linearities not implemented

* BernoulliBeam(CurrentConstitutiveLaw, Section, Jx, Iyy, Izz, ID = "")

    * The weak formulation of the mechanical equilibrium equation for beams with the bernoulli hypothesis.
    * Section, Jx, Iyy, Izz are the beam parameters and may be constant or defined at point of Gauss.
    * CurrentConstitutiveLaw shoud be isotropic elastic (other laws not implemented for now)
    * Geometrical non linearities not implemented for now

* Inertia(Density, ID = "")

    * The weak formulation related to the inertia effect into dynamical simulation
    * Density may be constant or defined at point of Gauss
    * Geometrical non linearities not implemented for now

* ParametricBernoulliBeam(E=None, nu = None, S=None, Jx=None, Iyy=None, Izz = None, R = None, ID = "")

    * Same has BernoulliBeam but with the possibility to set each parameter as a coordinate of the problem.
    * Mainly usefull for parametric problem with the Proper Orthogonal Decomposition