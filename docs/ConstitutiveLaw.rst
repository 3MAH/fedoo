Constitutive Law
=================================

The constitutive law module inclued several classical mechancical classical 
constitutive laws. This laws are required to create some weak formulations. 

The ConstitutiveLaw library contains the following classes: 


* ElasticIsotrop(YoungModulus, PoissonRatio, ID="")

    * A simple linear elastic isotropic constitutive law defined from a Yound Modulus and a Poisson Ratio.
    * YoungModulus and poisson ratio may be scalars or arrays of gauss point values. 
    * Specific methods: 
        - ElasticIsotrop.GetYoungModulus(): return the Young Modulus 
        - ElasticIsotrop.GetPoissonRatio(): return the Poisson Ratio
    * Associated with :mod:`WeakForm.InternalForce`
    
* ElasticOrthotropic(EX, EY, EZ, GYZ, GXZ, GXY, nuYZ, nuXZ, nuXY, ID="")

    * Linear Orthotropic constitutive law defined from the engineering coefficients in local material coordinates.
    * EX, EY, EZ, GYZ, GXZ, GXY, nuYZ, nuXZ, nuXY may be scalars or arrays of gauss point values. 
    * Specific methods: 
        - ElasticOrthotropic.GetEngineeringConstants(): return a dict containing the engineering constants
    * Associated with :mod:`WeakForm.InternalForce`

* CompositeUD(Vf=0.6, E_f=250000, E_m = 3500, nu_f = 0.33, nu_m = 0.3, angle=0, ID="")

    * Linear Orthotropic constitutive law defined from composites phase parameters (f -> fibers, m -> matrix).
    * angle is the the angle of the fibers relative to the X direction normal to the Z direction (if defined, the local material coordinate is used)
    * Specific methods: 
        - CompositeUD.GetEngineeringConstants(): return a dict containing the engineering constants
    * Associated with :mod:`WeakForm.InternalForce`

* ElasticAnisotropic(H, ID=""))

    * Linear full Anistropic constitutive law defined from the rigidity matrix H.
    * H is a list of list or an array (shape=(3,3)) of scalars or arrays of gauss point values. 
    * Use the material coordinates if defined
    * Associated with :mod:`WeakForm.InternalForce`

* ElastoPlasticity(YoungModulus, PoissonRatio, YieldStress, ID="")
    * Non Linear constitutive Law
    * Associated with :mod:`WeakForm.InternalForce`
    
* CohesiveLaw(GIc=0.3, SImax = 60, KI = 1e4, GIIc = 1.6, SIImax=None, KII=5e4, axis = 2, ID="")
    * Bilinear cohesive Law
    * Associated with :mod:`WeakForm.InterfaceForce`

* Spring(Kx=0, Ky = 0, Kz = 0, ID="")
    * Simple directional spring connector between nodes or surfaces
    * Kx, Ky and Kz are the rigidity along the X, Y and Z directions in material coordinates
    * Associated with :mod:`WeakForm.InterfaceForce`

