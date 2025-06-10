"""This module contains the AssemblySum class"""

from fedoo.core.base import AssemblyBase

# =============================================================
# Class that build a sum of Assembly
# =============================================================


# need to be modified to include a list of constitutivelaw update.
class AssemblySum(AssemblyBase):
    """
    Build a sum of Assembly objects
    All the Assembly objects should be associated to:
    * meshes based on the same list of nodes.
    * the same modeling space (ie the same space property)

    Parameters
    ----------
    list_assembly: list of Assembly
        list of Assembly objects to sum
    name: str
        name of the Assembly
    assembly_output: Assembly (optional keyword arg)
        Assembly object used to extract output values (using Problem.get_results or Problem.save_results)
    """

    def __init__(self, list_assembly, name="", **kargs):
        AssemblyBase.__init__(self, name)

        for i, assembly in enumerate(list_assembly):
            if isinstance(assembly, str):
                list_assembly[i] = AssemblyBase.get_all()[assembly]
            if list_assembly[i].associated_assembly_sum is None:
                list_assembly[i].associated_assembly_sum = self

        assert (
            len(set([a.space for a in list_assembly])) == 1
        ), "Sum of assembly are possible only if all assembly are associated to the same modeling space"
        assert (
            len(set([a.mesh.n_nodes for a in list_assembly])) == 1
        ), "Sum of assembly are possible only if the two meshes have the same number of Nodes"

        self._list_assembly = list_assembly

        if name == "":
            name = "_".join([assembly.name for assembly in list_assembly])

        self._reload = kargs.pop("reload", "all")

        self.mesh = list_assembly[0].mesh

        self.current = _AssemblySumCurrent(list_assembly)

        # for post-treatment only
        self.__assembly_output = kargs.get("assembly_output", None)
        if self.__assembly_output is not None:
            self.sv = self.__assembly_output.sv  # alias

    def __add__(self, another_assembly):
        if isinstance(another_assembly, AssemblySum):
            return AssemblySum(self.list_assembly + another_assembly.list_assembly)
        else:
            return AssemblySum(self.list_assembly + [another_assembly])

    def __getitem__(self, item):
        return self._list_assembly[item]

    def __repr__(self):
        return f"fedoo.AssemblySum({self._list_assembly})"

    def assemble_global_mat(self, compute="all"):
        if self._reload == "all":
            for assembly in self.list_assembly:
                assembly.assemble_global_mat(compute)
        else:
            for numAssembly in self._reload:
                self.list_assembly[numAssembly].assemble_global_mat(compute)

        if not (compute == "vector"):
            self.global_matrix = sum(
                [assembly.get_global_matrix() for assembly in self.list_assembly]
            )
        if not (compute == "matrix"):
            self.global_vector = sum(
                [assembly.get_global_vector() for assembly in self.list_assembly]
            )

    def update(self, pb, compute="all"):
        """
        Update the associated weak form and assemble the global matrix
        Parameters:
            - pb: a Problem object containing the Dof values
            - time: the current time
        """
        if self._reload == "all" or compute in [
            "vector",
            "none",
        ]:  # if compute == 'vector' or 'none' the reload arg is ignored
            for assembly in self.list_assembly:
                assembly.update(pb, compute)
        else:
            for numAssembly in self._reload:
                self.list_assembly[numAssembly].update(pb, compute)

        if not (compute == "vector"):
            self.current.global_matrix = sum(
                [
                    assembly.current.get_global_matrix()
                    for assembly in self.list_assembly
                ]
            )
        if not (compute == "matrix"):
            self.current.global_vector = sum(
                [
                    assembly.current.get_global_vector()
                    for assembly in self.list_assembly
                ]
            )

    def set_start(self, pb):
        """
        Apply the modification to the constitutive equation required at each new time increment.
        Generally used to increase non reversible internal variable
        Assemble the new global matrix.
        """
        for assembly in self.list_assembly:
            assembly.set_start(pb)

        # Update the global matrix to the trial values (generally elastic tangent matrix)
        self.current.global_matrix = sum(
            [assembly.current.get_global_matrix() for assembly in self.list_assembly]
        )
        # in principle no need to update global vector here because the current state is not modified by set_start. Could be improved by removing the next line
        self.current.global_vector = sum(
            [assembly.current.get_global_vector() for assembly in self.list_assembly]
        )

    def initialize(self, pb):
        """
        reset the current time increment (internal variable in the constitutive equation)
        Doesn't assemble the new global matrix. Use the Update method for that purpose.
        """
        for assembly in self.list_assembly:
            assembly.initialize(pb)

    def to_start(self, pb):
        """
        Reset the current time increment (internal variable in the constitutive equation)
        Doesn't assemble the new global matrix. Use the Update method for that purpose.
        """
        for assembly in self._list_assembly:
            assembly.to_start(pb)
        self.current.global_matrix = sum(
            [assembly.current.get_global_matrix() for assembly in self.list_assembly]
        )
        self.current.global_vector = sum(
            [assembly.current.get_global_vector() for assembly in self.list_assembly]
        )

    def reset(self):
        """
        reset the assembly to it's initial state.
        Internal variable in the constitutive equation are reinitialized
        And stored global matrix and vector are deleted
        """
        for assembly in self.list_assembly:
            assembly.reset()
        self.delete_global_mat()

    @property
    def list_assembly(self):
        return self._list_assembly

    @property
    def assembly_output(self):
        return self.__assembly_output


class _AssemblySumCurrent(
    AssemblySum
):  # same as AssemblySum using the current assemblies and without output function
    def __init__(self, list_assembly, **kargs):
        self._list_assembly = list_assembly
        self.__assembly_output = None
        # self.__assembly_output = kargs.get('assembly_output', None)
        # if self.__assembly_output is not None: self.sv = self.__assembly_output.sv #alias

        self._reload = kargs.pop("reload", "all")
        AssemblyBase.__init__(self, name="")

    @property
    def list_assembly(self):
        return [assembly.current for assembly in self._list_assembly]

    @property
    def mesh(self):
        return self._list_assembly[0].current.mesh


# def Sum(*listAssembly, name ="", **kargs):
#     """
#     Return a new assembly which is a sum of N assembly.
#     Assembly.Sum(assembly1, assembly2, ..., assemblyN, name ="", reload = [1,4] )

#     The N first arguments are the assembly to be summed.
#     name is the name of the created assembly:
#     reload: a list of indices for subassembly that are recomputed at each time the summed assembly
#     is Launched. Default is 'all' (equivalent to all indices).
#     """
#     return AssemblySum(list(listAssembly), name, **kargs)
