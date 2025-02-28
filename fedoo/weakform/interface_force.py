from fedoo.core.weakform import WeakFormBase
from fedoo.core.base import ConstitutiveLaw
import numpy as np


class InterfaceForce(WeakFormBase):
    """
    Weak formulation of the interface equilibrium equation.

    * Require an interface constitutive law such as :mod:`fedoo.constitutivelaw.CohesiveLaw` or :mod:`fedoo.constitutivelaw.Spring`
    * Geometrical non linearities not implemented

    Parameters
    ----------
    constitutivelaw: str or ConstitutiveLaw
        Interface constitutive law (ConstitutiveLaw object or name)
        (:mod:`fedoo.constitutivelaw`)
    name: str, optional
        name of the WeakForm
    nlgeom: bool (default = False)
        For future development
        If True, return a NotImplemented Error
    """

    def __init__(self, constitutivelaw, name="", nlgeom=False, space=None):
        if isinstance(constitutivelaw, str):
            constitutivelaw = ConstitutiveLaw[constitutivelaw]

        WeakFormBase.__init__(self, name)

        self.space.new_variable("DispX")
        self.space.new_variable("DispY")
        if self.space.ndim == 3:
            self.space.new_variable("DispZ")
            self.space.new_vector("Disp", ("DispX", "DispY", "DispZ"))
        else:  # 2D assumed
            self.space.new_vector("Disp", ("DispX", "DispY"))

        self.constitutivelaw = constitutivelaw

        self.nlgeom = nlgeom  # geometric non linearities -> False, True, 'UL' or 'TL' (True or 'UL': updated lagrangian - 'TL': total lagrangian)
        """Method used to treat the geometric non linearities. 
            * Set to False if geometric non linarities are ignored (default). 
            * Set to True or 'UL' to use the updated lagrangian method (update the mesh)
            * Set to 'TL' to use the total lagrangian method (base on the initial mesh with initial displacement effet)
        """

        self.assembly_options["assume_sym"] = False  # symetric ?

    def initialize(self, assembly, pb):
        if self.nlgeom:
            if self.nlgeom is True:
                self.nlgeom = "UL"
            elif isinstance(self.nlgeom, str):
                self.nlgeom = self.nlgeom.upper()
                if self.nlgeom != "UL":
                    raise NotImplementedError(
                        f"{self.nlgeom} nlgeom not implemented for Interface force."
                    )
            else:
                raise TypeError("nlgeom should be in {'TL', 'UL', True, False}")

    def update(self, assembly, pb):
        # function called when the problem is updated (NR loop or time increment)
        # Nlgeom implemented only for updated lagragian formulation

        if self.nlgeom == "UL":
            # if updated lagragian method -> update the mesh and recompute elementary op
            assembly.set_disp(pb.get_disp())
            # if assembly.current.mesh in assembly._saved_change_of_basis_mat:
            #     del assembly._saved_change_of_basis_mat[assembly.current.mesh]

            # assembly.current.compute_elementary_operators()

    def to_start(self, assembly, pb):
        if self.nlgeom == "UL":
            # if updated lagragian method -> reset the mesh to the begining of the increment
            assembly.set_disp(pb.get_disp())
            # if assembly.current.mesh in assembly._saved_change_of_basis_mat:
            #     del assembly._saved_change_of_basis_mat[assembly.current.mesh]

            # assembly.current.compute_elementary_operators()

    # def set_start(self, assembly, pb):
    #         if self.nlgeom:
    #             if 'DStrain' in assembly.sv:
    #                 #rotate strain and stress -> need to be checked
    #                 assembly.sv['Strain'] = StrainTensorList(sim.rotate_strain_R(assembly.sv_start['Strain'].asarray(),assembly.sv['DR']) + assembly.sv['DStrain'])
    #                 assembly.sv['DStrain'] = StrainTensorList(np.zeros((6, assembly.n_gauss_points), order='F'))
    # 				#or assembly.sv['DStrain'] = 0 perhaps more efficient to avoid a nul sum

    #             #update cauchy stress
    #             if assembly.sv['DispGradient'] is not 0: #True when the problem have been updated once
    #                 stress = assembly.sv['Stress'].asarray()
    #                 assembly.sv['Stress'] = StressTensorList(sim.rotate_stress_R(stress, assembly.sv['DR']))
    #                 if self.nlgeom == 'TL':
    #                     assembly.sv['PK2'] = assembly.sv['Stress'].cauchy_to_pk2(assembly.sv['F'])

    # def reset(self):
    #     pass

    def get_weak_equation(self, assembly, pb):
        ### Operator for Interface Stress Operator ###
        dim = self.space.ndim
        K = assembly.sv["TangentMatrix"]

        U = self.space.op_disp()  # relative displacement if used with cohesive element
        U_vir = [u.virtual for u in U]
        F = [
            sum([U[j] * K[i][j] for j in range(dim)]) for i in range(dim)
        ]  # Interface stress operator

        diff_op = sum([0 if U[i] == 0 else U_vir[i] * F[i] for i in range(dim)])

        initial_stress = assembly.sv["InterfaceStress"]

        if not (np.array_equal(initial_stress, 0)):
            diff_op = diff_op + sum(
                [
                    0 if U_vir[i] == 0 else U_vir[i] * initial_stress[i]
                    for i in range(dim)
                ]
            )

        return diff_op
