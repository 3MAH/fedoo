# derive de ConstitutiveLaw
# compatible with the simcoon strain and stress notation

from fedoo.core.mechanical3d import MechanicalUMAT
import simcoon as sim
import numpy as np


class Simcoon(MechanicalUMAT):
    """Constitutive laws from the simcoon library.

    The constitutive Law should be associated with
    :mod:`fedoo.weakform.StressEquilibrium`

    Parameters
    ----------
    umat_name: str
        Name of the constitutive law.
    props: numpy.array
        The constitive laws properties
    tangent_mode : int, default=1
        Tangent selector forwarded to the Simcoon UMAT.
    name : str
        The name of the constitutive law

    Alternative constructor
    -----------------------
    :meth:`from_modular` builds the same Fedoo wrapper from a
    :class:`simcoon.modular.ModularMaterial`. The configuration supplies its
    own flattened properties and state-variable count::

        from simcoon.modular import elastic_model

        configuration = elastic_model(E=210000.0, nu=0.3)
        material = Simcoon.from_modular(configuration, name="steel")

    The regular constructor is
    ``Simcoon(umat_name, props, tangent_mode=1, name="")``. For backward
    compatibility, a string passed as the third positional argument is still
    interpreted as ``name``.

    Notes
    -----
    UMAT compatibility with the ``2Daxi`` ModelingSpace:

    * **Isotropic UMATs** (e.g. ``ELI``, ``NEOH``, ``MOON``, ``EPICP`` J2
      plasticity): supported. Hooke's response and J2 invariants are
      invariant under the slot remapping fedoo applies in 2Daxi
      (cf. :class:`fedoo.core.mechanical3d.Mechanical3D`).

    * **Orthotropic / anisotropic UMATs** (e.g. ``ELIO``, composite
      Mori-Tanaka, anisotropic damage / plasticity): user-supplied
      material direction "3" is silently the **hoop** direction in
      2Daxi (because slot 2 of the 6-vector carries ε_θθ). Define the
      stiffness / hardening parameters with this convention or the
      response will not match the intended material orientation.

    * Hyperelastic laws are gated on plane stress (see
      ``_Lt_from_F`` branch); they remain compatible with 2Daxi at
      finite strain provided the F[θθ] = r/R fix is in effect (see
      :func:`fedoo.weakform.stress_equilibrium._comp_grad_disp`).

    * A modular material uses the ``MODUL`` UMAT. In finite strain it requires
      the ``log_R`` formulation; set ``weakform.corate = "log_R"`` before
      initializing the problem. Small-strain analyses are unaffected.
    """

    manages_material_frame = True

    _ISOTROPIC_UMATS = {
        "ELISO",
        "EPICP",
        "EPKCP",
        "EPCHA",
        "ZENER",
        "ZENNK",
        "PRONK",
        "NEOHC",
        "MOORI",
        "YEOHH",
        "ISHAH",
        "GETHH",
        "SWANH",
    }

    @classmethod
    def from_modular(cls, modular_material, tangent_mode=1, name=""):
        """Build a Simcoon law from a modular material configuration.

        Parameters
        ----------
        modular_material : simcoon.modular.ModularMaterial
            Composable Simcoon material containing an elasticity block and
            zero or more inelastic mechanisms.
        tangent_mode : int, default=1
            Tangent selector forwarded to the Simcoon ``MODUL`` UMAT.
        name : str, optional
            Fedoo constitutive-law registration name.

        Returns
        -------
        Simcoon
            A law dispatched through Simcoon's ``MODUL`` UMAT.

        Notes
        -----
        Under finite strain, use ``corate="log_R"`` (or ``"log_R_inc"``)
        in the associated stress-equilibrium weakform. This is required by
        Simcoon's modular Hencky formulation.
        """
        if isinstance(tangent_mode, str) and name == "":
            # Historical API: from_modular(configuration, name).
            name = tangent_mode
            tangent_mode = 1

        try:
            from simcoon import modular
        except ImportError as error:
            raise ImportError(
                "Simcoon modular materials require simcoon.modular"
            ) from error

        if not isinstance(modular_material, modular.ModularMaterial):
            raise TypeError(
                "modular_material must be a simcoon.modular.ModularMaterial"
            )

        props_label, statev_label = cls._get_modular_labels(modular_material, modular)
        material = cls.__new__(cls)
        MechanicalUMAT.__init__(
            material,
            props=modular_material.props,
            n_statev=modular_material.nstatev,
            props_label=props_label,
            statev_label=statev_label,
            is_isotropic=cls._is_modular_isotropic(modular_material, modular),
            tangent_mode=tangent_mode,
            name=name,
        )
        material.umat_name = "MODUL"
        material.modular_material = modular_material
        material.required_corate = ("log_r", "log_r_inc")
        return material

    @staticmethod
    def _is_modular_isotropic(modular_material, modular):
        if not isinstance(modular_material.elasticity, modular.IsotropicElasticity):
            return False
        anisotropic_yields = (
            modular.HillYield,
            modular.DFAYield,
            modular.AnisotropicYield,
        )
        return not any(
            isinstance(mechanism, modular.Plasticity)
            and isinstance(mechanism.yield_criterion, anisotropic_yields)
            for mechanism in modular_material.mechanisms
        )

    @staticmethod
    def _get_modular_labels(modular_material, modular):
        """Generate collision-free labels for flattened modular data."""
        props_label = {}
        prop_index = 0

        def add_props(prefix, names):
            nonlocal prop_index
            for label in names:
                props_label[f"{prefix}.{label}"] = prop_index
                prop_index += 1

        add_props("elasticity", ["type"])
        elasticity = modular_material.elasticity
        if isinstance(elasticity, modular.IsotropicElasticity):
            elastic_names = ["convention", "C1", "C2", "alpha"]
        elif isinstance(elasticity, modular.CubicElasticity):
            elastic_names = ["convention", "C1", "C2", "C3", "alpha"]
        elif isinstance(elasticity, modular.TransverseIsotropicElasticity):
            elastic_names = [
                "convention",
                "EL",
                "ET",
                "nuTL",
                "nuTT",
                "GLT",
                "alpha_L",
                "alpha_T",
                "axis",
            ]
        elif isinstance(elasticity, modular.OrthotropicElasticity):
            elastic_names = [
                "convention",
                "C1",
                "C2",
                "C3",
                "C4",
                "C5",
                "C6",
                "C7",
                "C8",
                "C9",
                "alpha1",
                "alpha2",
                "alpha3",
            ]
        else:
            raise TypeError(
                f"unsupported modular elasticity {type(elasticity).__name__!r}"
            )
        add_props("elasticity", elastic_names)
        add_props("mechanisms", ["count"])

        statev_label = {"T_init": 0}
        state_index = 1
        for mechanism_index, mechanism in enumerate(modular_material.mechanisms):
            prefix = f"mechanism_{mechanism_index}"
            add_props(prefix, ["type"])

            if isinstance(mechanism, modular.Plasticity):
                add_props(
                    prefix,
                    [
                        "yield_type",
                        "isotropic_hardening_type",
                        "kinematic_hardening_type",
                        "n_isotropic_terms",
                        "n_kinematic_terms",
                        "sigma_Y",
                    ],
                )
                yield_names = {
                    modular.VonMisesYield: [],
                    modular.TrescaYield: [],
                    modular.DruckerYield: ["b", "n"],
                    modular.HillYield: ["F", "G", "H", "L", "M", "N"],
                    modular.DFAYield: ["F", "G", "H", "L", "M", "N", "K"],
                    modular.AnisotropicYield: [
                        "P11",
                        "P22",
                        "P33",
                        "P12",
                        "P13",
                        "P23",
                        "P44",
                        "P55",
                        "P66",
                    ],
                }
                add_props(
                    f"{prefix}.yield",
                    yield_names[type(mechanism.yield_criterion)],
                )

                hardening = mechanism.isotropic_hardening
                if isinstance(hardening, modular.NoIsotropicHardening):
                    hardening_names = []
                elif isinstance(hardening, modular.LinearIsotropicHardening):
                    hardening_names = ["H"]
                elif isinstance(hardening, modular.PowerLawHardening):
                    hardening_names = ["k", "m"]
                elif isinstance(hardening, modular.VoceHardening):
                    hardening_names = ["Q", "b"]
                elif isinstance(hardening, modular.CombinedVoceHardening):
                    hardening_names = [
                        component
                        for index in range(len(hardening.terms))
                        for component in (f"Q_{index}", f"b_{index}")
                    ]
                else:
                    raise TypeError(
                        "unsupported modular isotropic hardening "
                        f"{type(hardening).__name__!r}"
                    )
                add_props(f"{prefix}.isotropic_hardening", hardening_names)

                hardening = mechanism.kinematic_hardening
                if isinstance(hardening, modular.NoKinematicHardening):
                    hardening_names = []
                elif isinstance(hardening, modular.PragerHardening):
                    hardening_names = ["C"]
                elif isinstance(hardening, modular.ArmstrongFrederickHardening):
                    hardening_names = ["C", "D"]
                elif isinstance(hardening, modular.ChabocheHardening):
                    hardening_names = [
                        component
                        for index in range(len(hardening.terms))
                        for component in (f"C_{index}", f"D_{index}")
                    ]
                else:
                    raise TypeError(
                        "unsupported modular kinematic hardening "
                        f"{type(hardening).__name__!r}"
                    )
                add_props(f"{prefix}.kinematic_hardening", hardening_names)

                statev_label[f"{prefix}.p"] = state_index
                state_index += 1
                statev_label[f"{prefix}.EP"] = slice(state_index, state_index + 6)
                state_index += 6
                for index in range(mechanism.kinematic_hardening.num_backstresses):
                    statev_label[f"{prefix}.a_{index}"] = slice(
                        state_index, state_index + 6
                    )
                    state_index += 6

            elif isinstance(mechanism, modular.Viscoelasticity):
                add_props(prefix, ["n_prony"])
                add_props(
                    f"{prefix}.prony",
                    [
                        component
                        for index in range(len(mechanism.terms))
                        for component in (
                            f"E_{index}",
                            f"nu_{index}",
                            f"etaB_{index}",
                            f"etaS_{index}",
                        )
                    ],
                )
                for index in range(len(mechanism.terms)):
                    statev_label[f"{prefix}.v_{index}"] = state_index
                    state_index += 1
                    statev_label[f"{prefix}.EV_{index}"] = slice(
                        state_index, state_index + 6
                    )
                    state_index += 6

            elif isinstance(mechanism, modular.Damage):
                damage_names = ["damage_type", "Y_0", "Y_c"]
                if mechanism.damage_type == modular.DamageType.EXPONENTIAL:
                    damage_names.append("A")
                elif mechanism.damage_type == modular.DamageType.POWER_LAW:
                    damage_names.append("n")
                elif mechanism.damage_type == modular.DamageType.WEIBULL:
                    damage_names.extend(["A", "n"])
                add_props(prefix, damage_names)
                statev_label[f"{prefix}.D"] = state_index
                state_index += 1
                statev_label[f"{prefix}.Y_max"] = state_index
                state_index += 1
            else:
                raise TypeError(
                    f"unsupported modular mechanism {type(mechanism).__name__!r}"
                )

        if prop_index != modular_material.nprops:
            raise RuntimeError(
                "generated modular property labels do not match the Simcoon "
                f"property layout ({prop_index} != {modular_material.nprops})"
            )
        if state_index != modular_material.nstatev:
            raise RuntimeError(
                "generated modular state labels do not match the Simcoon "
                f"state layout ({state_index} != {modular_material.nstatev})"
            )
        return props_label, statev_label

    def __init__(self, umat_name, props, tangent_mode=1, name=""):
        # props is a nparray containing all the material variables
        # nstatev is a nparray containing all the material variables
        if isinstance(tangent_mode, str) and name == "":
            # Historical API: Simcoon(umat_name, props, name).
            name = tangent_mode
            tangent_mode = 1
        MechanicalUMAT.__init__(self, props=props, tangent_mode=tangent_mode, name=name)
        # self._statev_initial = statev #statev may be an int or an array
        # self.__useElasticModulus = True ??

        # self.__currentGradDisp = self.__initialGradDisp = 0

        # ndi = nshr = 3 #compute the 3D constitutive law even for 2D law
        self.umat_name = umat_name
        self.is_isotropic = umat_name in self._ISOTROPIC_UMATS
        # MechanicalUMAT retains the historical defaults: cache the elastic
        # tangent and request Simcoon's continuum tangent (mode 1).

        # _Lt_from_F attribute is set to True if the tangent matrix is related
        # to F instead of log epsilon, ie for hyper elastic materials
        if umat_name == "ELISO":
            self.n_statev = 1
            self.props_label = {"E": 0, "nu": 1, "alpha": 2}
            self.statev_label = {"T": 0}
        elif umat_name == "ELIST":
            self.n_statev = 1
            self.props_label = {
                "axis": 0,
                "EL": 1,
                "ET": 2,
                "nuTL": 3,
                "nuTT": 4,
                "GLT": 5,
                "alphaL": 6,
                "alphaT": 7,
            }
            self.statev_label = {"T": 0}
        elif umat_name == "ELORT":
            self.n_statev = 1
            self.props_label = {
                "Ex": 0,
                "Ey": 1,
                "Ez": 2,
                "nuxy": 3,
                "nuxz": 4,
                "nuyz": 5,
                "Gxy": 6,
                "Gxz": 7,
                "Gyz": 8,
                "alphax": 9,
                "alphay": 10,
                "alphaz": 11,
            }
            self.statev_label = {"T": 0}
        elif umat_name == "EPICP":
            self.n_statev = 8
            self.props_label = {
                "E": 0,
                "nu": 1,
                "alpha": 2,
                "sigmaY": 3,
                "k": 4,
                "m": 5,
            }  # powerlaw sigma_e = sigmaY + k * eps_p^m
            self.statev_label = {"T": 0, "P": 1, "EP": slice(2, 8)}
        elif umat_name == "EPKCP":
            self.n_statev = 14
            self.props_label = {
                "E": 0,
                "nu": 1,
                "alpha": 2,
                "sigmaY": 3,
                "K": 4,
                "m": 5,
                "h": 6,
            }
            # powerlaw sigma_e = sigmaY + k * eps_p^m -
            # h=linear kinematical hardening
            self.statev_label = {
                "T": 0,
                "P": 1,
                "EP": slice(2, 8),
                "X": slice(8, 14),
            }  # X=backstress
        elif umat_name == "EPCHA":
            self.n_statev = 33
            self.props_label = {
                "E": 0,
                "nu": 1,
                "alpha": 2,
                "sigmaY": 3,
                "Q": 4,
                "b": 5,
                "C_1": 6,
                "D_1": 7,
                "C_2": 8,
                "D_2": 9,
            }
            self.statev_label = {
                "T": 0,
                "P": 1,
                "EP": slice(2, 8),
                "a1": slice(8, 14),
                "a2": slice(14, 20),
                "X1": slice(20, 26),
                "X2": slice(26, 32),
                "Hp": 33,
            }
        elif umat_name == "EPHIL":
            self.n_statev = 8
            self.props_label = {
                "E": 0,
                "nu": 1,
                "alpha": 2,
                "sigmaY": 3,
                "K": 4,
                "m": 5,
                "F_hill": 6,
                "G_hill": 7,
                "H_hill": 8,
                "L_hill": 9,
                "M_hill": 10,
                "N_hill": 11,
            }
            self.statev_label = {"T": 0, "P": 1, "EP": slice(2, 8)}
        elif umat_name == "EPHAC":
            self.n_statev = 33
            self.props_label = {
                "E": 0,
                "nu": 1,
                "G": 2,
                "alpha": 3,
                "sigmaY": 4,
                "Q": 5,
                "b": 6,
                "C_1": 7,
                "D_1": 8,
                "C_2": 9,
                "D_2": 10,
                "F_hill": 11,
                "G_hill": 12,
                "H_hill": 13,
                "L_hill": 14,
                "M_hill": 15,
                "N_hill": 16,
            }
            self.statev_label = {
                "T": 0,
                "P": 1,
                "EP": slice(2, 8),
                "a1": slice(8, 14),
                "a2": slice(14, 20),
                "X1": slice(20, 26),
                "X2": slice(26, 32),
                "Hp": 33,
            }
        elif umat_name == "EPANI":
            self.n_statev = 33
            self.props_label = {
                "E": 0,
                "nu": 1,
                "G": 2,
                "alpha": 3,
                "sigmaY": 4,
                "Q": 5,
                "b": 6,
                "C_1": 7,
                "D_1": 8,
                "C_2": 9,
                "D_2": 10,
                "P11": 11,
                "P22": 12,
                "P33": 13,
                "P12": 14,
                "P13": 15,
                "P23": 16,
                "P44": 17,
                "P55": 18,
                "P66": 19,
            }
            self.statev_label = {
                "T": 0,
                "P": 1,
                "EP": slice(2, 8),
                "a1": slice(8, 14),
                "a2": slice(14, 20),
                "X1": slice(20, 26),
                "X2": slice(26, 32),
                "Hp": 33,
            }
        elif umat_name == "EPDFA":
            self.n_statev = 33
            self.props_label = {
                "E": 0,
                "nu": 1,
                "G": 2,
                "alpha": 3,
                "sigmaY": 4,
                "Q": 5,
                "b": 6,
                "F_dfa": 11,
                "G_dfa": 12,
                "H_dfa": 13,
                "L_dfa": 14,
                "M_dfa": 15,
                "N_dfa": 16,
                "K_dfa": 17,
            }
            self.statev_label = {
                "T": 0,
                "P": 1,
                "EP": slice(2, 8),
                "a1": slice(8, 14),
                "a2": slice(14, 20),
                "X1": slice(20, 26),
                "X2": slice(26, 32),
                "Hp": 33,
            }
        elif umat_name == "EPHIN":
            n_plas = self.props[0, 3]
            # should be the same for all gauss_points. If not, needs several
            # assemblies
            self.n_statev = 7 + n_plas * 7
            self.props_label = {
                "E": 0,
                "nu": 1,
                "alpha": 2,
            }
            # several plastic laws i "sigmaY":4+i*9, "k":4+i*9+1, "m":4+i*9+2,
            # "F_hill":4+i*9+3, "G_hill":4+i*9+4, "H_hill":4+i*9+5,
            # "L_hill":4+i*9+6, "M_hill":4+i*9+7, "N_hill":4+i*9+8
            self.statev_label = {
                "T": 0,
                "EP": slice(1, 7),
            }  # Pi:i*7+7, EPi:slice(i*7+8,i*7+14)
        elif umat_name == "SMADI" or umat_name == "SMAUT":
            self.n_statev = 17
            self.props_label = {
                "flagT": 0,
                "E_A": 1,
                "E_M": 2,
                "nu_A": 3,
                "nu_M": 4,
                "alphaA": 5,
                "alphaM": 6,
                "Hmin": 7,
                "Hmax": 8,
                "k1": 9,
                "sigmacrit": 10,
                "C_A": 11,
                "C_M": 12,
                "Ms0": 13,
                "Mf0": 14,
                "As0": 15,
                "Af0": 16,
                "n1": 17,
                "n2": 18,
                "n3": 19,
                "n4": 20,
                "sigmacaliber": 21,
                "b_prager": 22,
                "n_prager": 23,
                "c_lambda": 24,
                "p0_lambda": 25,
                "n_lambda": 26,
                "alpha_lambda": 27,
            }
            self.statev_label = {
                "T_init": 0,
                "xi": 1,
                "ET": slice(2, 8),
                "xi_F": 8,
                "xi_R": 9,
                "rhoDs0": 10,
                "rhoDE0": 11,
                "D": 12,
                "a1": 13,
                "a2": 14,
                "a3": 15,
                "Y0t": 16,
            }
        elif umat_name == "SMADC":
            self.n_statev = 17
            self.props_label = {
                "flagT": 0,
                "E_A": 1,
                "E_M": 2,
                "nu_A": 3,
                "nu_M": 4,
                "G_A": 5,
                "G_M": 6,
                "alphaA": 7,
                "alphaM": 8,
                "Hmin": 9,
                "Hmax": 10,
                "k1": 11,
                "sigmacrit": 12,
                "C_A": 13,
                "C_M": 14,
                "Ms0": 15,
                "Mf0": 16,
                "As0": 17,
                "Af0": 18,
                "n1": 19,
                "n2": 20,
                "n3": 21,
                "n4": 22,
                "sigmacaliber": 23,
                "b_prager": 24,
                "n_prager": 25,
                "c_lambda": 26,
                "p0_lambda": 27,
                "n_lambda": 28,
                "alpha_lambda": 29,
            }
            self.statev_label = {
                "T_init": 0,
                "xi": 1,
                "ET": slice(2, 8),
                "xi_F": 8,
                "xi_R": 9,
                "rhoDs0": 10,
                "rhoDE0": 11,
                "D": 12,
                "a1": 13,
                "a2": 14,
                "a3": 15,
                "Y0t": 16,
            }
        elif umat_name == "SMAAI" or umat_name == "SMANI":
            self.n_statev = 17
            self.props_label = {
                "flagT": 0,
                "E_A": 1,
                "E_M": 2,
                "nu_A": 3,
                "nu_M": 4,
                "alphaA": 5,
                "alphaM": 6,
                "Hmin": 7,
                "Hmax": 8,
                "k1": 9,
                "sigmacrit": 10,
                "C_A": 11,
                "C_M": 12,
                "Ms0": 13,
                "Mf0": 14,
                "As0": 15,
                "Af0": 16,
                "n1": 17,
                "n2": 18,
                "n3": 19,
                "n4": 20,
                "sigmacaliber": 21,
                "b_prager": 22,
                "n_prager": 23,
                "c_lambda": 24,
                "p0_lambda": 25,
                "n_lambda": 26,
                "alpha_lambda": 27,
                "F_dfa": 28,
                "G_dfa": 29,
                "H_dfa": 30,
                "L_dfa": 31,
                "M_dfa": 32,
                "N_dfa": 33,
                "K_dfa": 34,
            }
            self.statev_label = {
                "T_init": 0,
                "xi": 1,
                "ET": slice(2, 8),
                "xi_F": 8,
                "xi_R": 9,
                "rhoDs0": 10,
                "rhoDE0": 11,
                "D": 12,
                "a1": 13,
                "a2": 14,
                "a3": 15,
                "Y0t": 16,
            }
        elif umat_name == "SMAAC":
            self.n_statev = 17
            self.props_label = {
                "flagT": 0,
                "E_A": 1,
                "E_M": 2,
                "nu_A": 3,
                "nu_M": 4,
                "G_A": 5,
                "G_M": 6,
                "alphaA": 7,
                "alphaM": 8,
                "Hmin": 9,
                "Hmax": 10,
                "k1": 11,
                "sigmacrit": 12,
                "C_A": 13,
                "C_M": 14,
                "Ms0": 15,
                "Mf0": 16,
                "As0": 17,
                "Af0": 18,
                "n1": 19,
                "n2": 20,
                "n3": 21,
                "n4": 22,
                "sigmacaliber": 23,
                "b_prager": 24,
                "n_prager": 25,
                "c_lambda": 26,
                "p0_lambda": 27,
                "n_lambda": 28,
                "alpha_lambda": 29,
                "F_dfa": 30,
                "G_dfa": 31,
                "H_dfa": 32,
                "L_dfa": 33,
                "M_dfa": 34,
                "N_dfa": 35,
                "K_dfa": 36,
            }
            self.statev_label = {
                "T_init": 0,
                "xi": 1,
                "ET": slice(2, 8),
                "xi_F": 8,
                "xi_R": 9,
                "rhoDs0": 10,
                "rhoDE0": 11,
                "D": 12,
                "a1": 13,
                "a2": 14,
                "a3": 15,
                "Y0t": 16,
            }
        elif umat_name == "SMRDI":
            # unified_TR (transformation + reorientation), isotropic elasticity
            self.n_statev = 30
            self.props_label = {
                "flagT": 0,
                "E_A": 1,
                "E_M": 2,
                "nu_A": 3,
                "nu_M": 4,
                "alphaA": 5,
                "alphaM": 6,
                "Hmin": 7,
                "Hmax": 8,
                "k1": 9,
                "sigmacrit": 10,
                "C_A": 11,
                "C_M": 12,
                "Ms0": 13,
                "Mf0": 14,
                "As0": 15,
                "Af0": 16,
                "n1": 17,
                "n2": 18,
                "n3": 19,
                "n4": 20,
                "sigmacaliber": 21,
                "b_prager": 22,
                "n_prager": 23,
                "c_lambda": 24,
                "p0_lambda": 25,
                "n_lambda": 26,
                "alpha_lambda": 27,
                "YReo": 28,
                "HReo": 29,
                "ETRmax": 30,
                "c_lambdaReo": 31,
                "p0_lambdaReo": 32,
                "n_lambdaReo": 33,
                "alpha_lambdaReo": 34,
            }
            self.statev_label = {
                "T_init": 0,
                "xi": 1,
                "ET": slice(2, 8),
                "xi_F": 8,
                "xi_R": 9,
                "rhoDs0": 10,
                "rhoDE0": 11,
                "D": 12,
                "a1": 13,
                "a2": 14,
                "a3": 15,
                "Y0t": 16,
                "pTR": 17,
                "areo": slice(18, 24),
                "EReo": slice(24, 30),
            }
        elif umat_name == "SMRDC":
            # unified_TR (transformation + reorientation), cubic elasticity
            self.n_statev = 30
            self.props_label = {
                "flagT": 0,
                "E_A": 1,
                "E_M": 2,
                "nu_A": 3,
                "nu_M": 4,
                "G_A": 5,
                "G_M": 6,
                "alphaA": 7,
                "alphaM": 8,
                "Hmin": 9,
                "Hmax": 10,
                "k1": 11,
                "sigmacrit": 12,
                "C_A": 13,
                "C_M": 14,
                "Ms0": 15,
                "Mf0": 16,
                "As0": 17,
                "Af0": 18,
                "n1": 19,
                "n2": 20,
                "n3": 21,
                "n4": 22,
                "sigmacaliber": 23,
                "b_prager": 24,
                "n_prager": 25,
                "c_lambda": 26,
                "p0_lambda": 27,
                "n_lambda": 28,
                "alpha_lambda": 29,
                "YReo": 30,
                "HReo": 31,
                "ETRmax": 32,
                "c_lambdaReo": 33,
                "p0_lambdaReo": 34,
                "n_lambdaReo": 35,
                "alpha_lambdaReo": 36,
            }
            self.statev_label = {
                "T_init": 0,
                "xi": 1,
                "ET": slice(2, 8),
                "xi_F": 8,
                "xi_R": 9,
                "rhoDs0": 10,
                "rhoDE0": 11,
                "D": 12,
                "a1": 13,
                "a2": 14,
                "a3": 15,
                "Y0t": 16,
                "pTR": 17,
                "areo": slice(18, 24),
                "EReo": slice(24, 30),
            }
        elif umat_name == "SMRAI":
            # unified_TR (transformation + reorientation), isotropic elasticity + DFA anisotropic criterion
            self.n_statev = 30
            self.props_label = {
                "flagT": 0,
                "E_A": 1,
                "E_M": 2,
                "nu_A": 3,
                "nu_M": 4,
                "alphaA": 5,
                "alphaM": 6,
                "Hmin": 7,
                "Hmax": 8,
                "k1": 9,
                "sigmacrit": 10,
                "C_A": 11,
                "C_M": 12,
                "Ms0": 13,
                "Mf0": 14,
                "As0": 15,
                "Af0": 16,
                "n1": 17,
                "n2": 18,
                "n3": 19,
                "n4": 20,
                "sigmacaliber": 21,
                "b_prager": 22,
                "n_prager": 23,
                "c_lambda": 24,
                "p0_lambda": 25,
                "n_lambda": 26,
                "alpha_lambda": 27,
                "F_dfa": 28,
                "G_dfa": 29,
                "H_dfa": 30,
                "L_dfa": 31,
                "M_dfa": 32,
                "N_dfa": 33,
                "K_dfa": 34,
                "YReo": 35,
                "HReo": 36,
                "ETRmax": 37,
                "c_lambdaReo": 38,
                "p0_lambdaReo": 39,
                "n_lambdaReo": 40,
                "alpha_lambdaReo": 41,
            }
            self.statev_label = {
                "T_init": 0,
                "xi": 1,
                "ET": slice(2, 8),
                "xi_F": 8,
                "xi_R": 9,
                "rhoDs0": 10,
                "rhoDE0": 11,
                "D": 12,
                "a1": 13,
                "a2": 14,
                "a3": 15,
                "Y0t": 16,
                "pTR": 17,
                "areo": slice(18, 24),
                "EReo": slice(24, 30),
            }
        elif umat_name == "SMRAC":
            # unified_TR (transformation + reorientation), cubic elasticity + DFA anisotropic criterion
            self.n_statev = 30
            self.props_label = {
                "flagT": 0,
                "E_A": 1,
                "E_M": 2,
                "nu_A": 3,
                "nu_M": 4,
                "G_A": 5,
                "G_M": 6,
                "alphaA": 7,
                "alphaM": 8,
                "Hmin": 9,
                "Hmax": 10,
                "k1": 11,
                "sigmacrit": 12,
                "C_A": 13,
                "C_M": 14,
                "Ms0": 15,
                "Mf0": 16,
                "As0": 17,
                "Af0": 18,
                "n1": 19,
                "n2": 20,
                "n3": 21,
                "n4": 22,
                "sigmacaliber": 23,
                "b_prager": 24,
                "n_prager": 25,
                "c_lambda": 26,
                "p0_lambda": 27,
                "n_lambda": 28,
                "alpha_lambda": 29,
                "F_dfa": 30,
                "G_dfa": 31,
                "H_dfa": 32,
                "L_dfa": 33,
                "M_dfa": 34,
                "N_dfa": 35,
                "K_dfa": 36,
                "YReo": 37,
                "HReo": 38,
                "ETRmax": 39,
                "c_lambdaReo": 40,
                "p0_lambdaReo": 41,
                "n_lambdaReo": 42,
                "alpha_lambdaReo": 43,
            }
            self.statev_label = {
                "T_init": 0,
                "xi": 1,
                "ET": slice(2, 8),
                "xi_F": 8,
                "xi_R": 9,
                "rhoDs0": 10,
                "rhoDE0": 11,
                "D": 12,
                "a1": 13,
                "a2": 14,
                "a3": 15,
                "Y0t": 16,
                "pTR": 17,
                "areo": slice(18, 24),
                "EReo": slice(24, 30),
            }
        elif umat_name == "LLDM0":
            self.n_statev = 10
            self.props_label = {
                "axis": 0,
                "EL": 1,
                "ET": 2,
                "nuTL": 3,
                "nuTT": 4,
                "GLT": 5,
                "alphaL": 6,
                "alphaT": 7,
            }
            self.statev_label = {
                "T": 0,
                "d_22": 1,
                "d_12": 2,
                "p_ts": 3,
                "EP": slice(4, 10),
            }
        elif umat_name == "ZENER":
            self.n_statev = 8
            self.props_label = {
                "E0": 0,
                "nu0": 1,
                "alpha": 2,
                "E1": 3,
                "nu1": 4,
                "etaB1": 5,
                "etaS1": 6,
            }
            self.statev_label = {"T": 0, "v": 1, "EV": slice(2, 8)}
        elif umat_name == "ZENNK":
            n_kelvin = self.props[0, 3]
            # should be the same for all gauss_points. If not, needs several
            # assemblies
            self.n_statev = 7 + 7 * n_kelvin
            self.props_label = {
                "E0": 0,
                "nu0": 1,
                "alpha": 2,
            }  # Ei":4+i*4,"nui":5+i*4,"etaBi":6+i*4,"etaSi":7+i*4
            self.statev_label = {
                "T": 0,
                "EV": slice(1, 7),
            }  # vi: i*7+7, EVi: slice(i*7+8,i*7+14)
        elif umat_name == "PRONK":
            n_prony = self.props[0, 3]
            # should be the same for all gauss_points. If not, needs several
            # assemblies
            self.n_statev = 7 + 7 * n_prony
            self.props_label = {
                "E0": 0,
                "nu0": 1,
                "alpha": 2,
            }  # Ei":4+i*4,"nui":5+i*4,"etaBi":6+i*4,"etaSi":7+i*4
            self.statev_label = {
                "T": 0,
                "EV_tilde": slice(1, 7),
            }  # vi: i*7+7, EVi: slice(i*7+8,i*7+14)
        elif umat_name == "SMAMO":
            nvariants = self.props[0, 7]
            # should be the same for all gauss_points. If not, needs several
            # assemblies
            self.n_statev = nvariants + 8
            self.props_label = {}
            self.statev_label = {}
        elif umat_name == "SMAMC":
            nvariants = self.props[0, 8]
            # should be the same for all gauss_points. If not, needs several
            # assemblies
            self.n_statev = nvariants + 8
            self.props_label = {}
            self.statev_label = {}
        elif umat_name == "NEOHC":
            self.n_statev = 1
            self.props_label = {
                "mu": 0,
                "kappa": 1,
            }
            self.statev_label = {"T": 0}
            self._Lt_from_F = True
        elif umat_name == "MOORI":
            self.n_statev = 1
            self.props_label = {
                "C_10": 0,
                "C_01": 1,
                "kappa": 2,
            }
            self.statev_label = {"T": 0}
            self._Lt_from_F = True
        elif umat_name == "YEOHH":
            self.n_statev = 1
            self.props_label = {
                "C_10": 0,
                "C_20": 1,
                "C_30": 2,
                "kappa": 3,
            }
            self.statev_label = {"T": 0}
            self._Lt_from_F = True
        elif umat_name == "ISHAH":
            self.n_statev = 1
            self.props_label = {
                "C_10": 0,
                "C_20": 1,
                "C_01": 2,
                "kappa": 3,
            }
            self.statev_label = {"T": 0}
            self._Lt_from_F = True
        elif umat_name == "GETHH":
            self.n_statev = 1
            self.props_label = {
                "C_1": 0,
                "C_2": 1,
                "kappa": 2,
            }
            self.statev_label = {"T": 0}
            self._Lt_from_F = True
        elif umat_name == "SWANH":
            self.n_statev = 1
            self.props_label = {
                "N_Swanson": 0,
                "kappa": 1,
                # Nb of Swanson parameters : 2+i*4, i being the number of
                # Swanson modes
                # (A, B, alpha, beta) are vectors of size N_Swanson
            }
            self.statev_label = {"T": 0}
            self._Lt_from_F = True
        else:
            raise ValueError("Invalid umat_name: Expected a valid 5 char string.")

    def _call_umat(self, *args, **kwargs):
        """Dispatch the generic MechanicalUMAT call to Simcoon."""
        return sim.umat(self.umat_name, *args, **kwargs)
