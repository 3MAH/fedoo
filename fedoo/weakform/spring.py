from fedoo.core.base import ConstitutiveLaw
from fedoo.constitutivelaw.beam import BeamProperties
from fedoo.core.weakform import WeakFormBase
from scipy.spatial.transform import Rotation
import numpy as np


class SpringEquilibrium(WeakFormBase):
    """
    Weak formulation of the mechanical equilibrium equation for spring models.

    Geometrical are implemented with the updated lagrangian approach.

    Parameters
    ----------
    K: float, numpy array
        Spring rigidity
        If numpy array, len(K) should be the number of elements in the corresponding mesh.
    nlgeom: bool, 'UL', default = False
        if True or 'UL', activate the geometrical non linearities
    name: str
        name of the WeakForm
    """

    def __init__(
        self,
        K,
        name="",
        nlgeom=False,
        space=None,
    ):
        WeakFormBase.__init__(self, name)

        self.space.new_variable("DispX")
        self.space.new_variable("DispY")
        if self.space.ndim == 3:
            self.space.new_variable("DispZ")
            self.space.new_vector("Disp", ("DispX", "DispY", "DispZ"))
        else:  # 2D assumed
            self.space.new_vector("Disp", ("DispX", "DispY"))

        self.K = K
        self.Kt_factor = None

        self.nlgeom = nlgeom  # geometric non linearities -> False, True, 'UL' or 'TL' (True or 'UL': updated lagrangian - 'TL': total lagrangian)
        """Method used to treat the geometric non linearities. 
            * Set to False if geometric non linarities are ignored (default). 
            * Set to True or 'UL' to use the updated lagrangian method (update the mesh)
            * Set to 'TL' to use the total lagrangian method (base on the initial mesh with initial displacement effet)
        """

        self.assembly_options["assume_sym"] = False

    def initialize(self, assembly, pb):
        # initialize nlgeom value in assembly._nlgeom
        self._initialize_nlgeom(assembly, pb)
        self.nlgeom = assembly._nlgeom

        assembly.sv["Stretch"] = assembly.sv["Fint"] = 0

    def update(self, assembly, pb):
        # function called when the problem is updated (NR loop or time increment)
        # Nlgeom implemented only for updated lagragian formulation
        if self.nlgeom == "UL":
            # if updated lagragian method -> update the mesh and recompute elementary op
            assembly.set_disp(pb.get_disp())

        dof = pb.get_dof_solution()  # displacement and rotation node values
        if dof is 0:
            assembly.sv["Stretch"] = assembly.sv["Fint"] = 0
        else:
            # evaluate Strain
            if self.nlgeom:
                # Compute Beam Strain
                mesh = assembly.current.mesh  # deformed mesh

                if "_InitialLength" in assembly.sv:
                    initial_length = assembly.sv["_InitialLength"]
                else:
                    initial_length = np.linalg.norm(
                        assembly.mesh.nodes[mesh.elements[:, 1]]
                        - assembly.mesh.nodes[mesh.elements[:, 0]]
                        , axis=1
                    )
                    assembly.sv["_InitialLength"] = initial_length

                # coordinates of vector between node 1 and 2 for each element
                element_vectors = (
                    mesh.nodes[mesh.elements[:, 1]]
                    - mesh.nodes[mesh.elements[:, 0]]
                )

                # longitunal displacement in local coordinates
                spring_stretch = np.linalg.norm(element_vectors, axis=1) - initial_length
                assembly.sv["Stretch"] = spring_stretch

            else:
                op_delta = (
                    assembly.space.op_disp()[0]
                )  # relative displacement = disp if used with spring element
                delta = assembly.get_gp_results(op_delta, dof)
                assembly.sv["Stretch"] = delta

            # Compute spring force
            assembly.sv["Fint"] = assembly.sv["Stretch"] * self.K

    def to_start(self, assembly, pb):
        if self.nlgeom == "UL":
            # if updated lagragian method -> reset the mesh to the begining of the increment
            assembly.set_disp(pb.get_disp())

    def get_weak_equation(self, assembly, pb):
        dim = self.space.ndim
        # add a 10% rigididy in the tangent direction to improve 
        # cvg stability
        if self.Kt_factor is None:
            if self.nlgeom:
                Kt = 0.1*self.K
            else: 
                Kt = 0.001*self.K
        else:
            Kt = self.Kt_factor * self.K
        K = [self.K, Kt, Kt]

        op_delta = self.space.op_disp()  # relative displacement if used with cohesive element
        op_F = [
            op_delta[i] * K[i] for i in range(dim)
        ]  # Interface stress operator

        diff_op = sum(
            [
                (
                    0
                    if np.isscalar(K[i]) and K[i] == 0
                    else op_delta[i].virtual * op_delta[i] * K[i]
                )
                for i in range(dim)
            ]
        )

        Fint = assembly.sv["Fint"]

        if Fint is not 0:
            diff_op = diff_op + op_delta[0].virtual * Fint

        return diff_op
