# derive de ConstitutiveLaw
# compatible with the simcoon strain and stress notation

from fedoo.core.mechanical3d import Mechanical3D
from fedoo.core.assembly import Assembly
# from copy import deepcopy
# from fedoo.util.voigt_tensors import StressTensorList, StrainTensorList


import numpy as np


class _SubAssembly(Assembly):
    # Assembly with new definition of sv and sv_start that allow maping the global assembly id to the sub_assembly
    def __init__(self, assembly, elset, assembly_id, copied_fields):
        self.assembly = assembly  # base full assembly
        self.elset = elset
        self.copied_fields = copied_fields
        self.assembly_id = assembly_id  # int used for assembly id
        super().__init__(
            assembly.weakform, assembly.mesh.extract_elements(elset), assembly.elm_type
        )

    @property
    def sv(self):
        elset = (
            self.assembly.mesh.element_sets[self.elset]
            if isinstance(self.elset, str)
            else self.elset
        )
        return _SubSV(
            self.assembly,
            self.assembly.sv,
            elset,
            self.assembly_id,
            self.copied_fields,
        )

    @sv.setter
    def sv(self, value):
        pass  # ignored - cl are not supposed to change the sv attribute (only the dict content)

    @property
    def sv_start(self):
        elset = (
            self.assembly.mesh.element_sets[self.elset]
            if isinstance(self.elset, str)
            else self.elset
        )
        return _SubSV(
            self.assembly,
            self.assembly.sv_start,
            elset,
            self.assembly_id,
            set(),
        )

    @sv_start.setter
    def sv_start(self, value):
        pass  # ignored - cl are not supposed to change the sv attribute (only the dict content)


class _SubSV:
    # class just here to map the good id for elset in the global state variable
    def __init__(self, assembly, sv, elset, assembly_id, copied_fields):
        self.sv = sv
        self.elset = elset
        self.assembly = assembly
        self.assembly_id = assembly_id
        self.copied_fields = copied_fields

    def __contains__(self, item):
        return item in self.sv

    def __getitem__(self, k):
        # assume sv values are defined on gauss points.
        # perhaps it may be usefull to allow other definitions
        if k == "Statev":
            # Don't merge constitutive law state variables (ie 'Statev')
            # cause it may have non consistent shape
            return self.sv[f"_Statev_{self.assembly_id}"]

        if np.isscalar(self.sv[k]) and self.sv[k] == 0:
            return 0

        elset = (
            np.array(self.elset)
            + np.c_[
                np.arange(
                    0, self.assembly.n_gauss_points, self.assembly.mesh.n_elements
                )
            ]
        ).reshape(-1)

        if isinstance(self.sv[k], list):
            try:
                return self.sv[k].__class__(self.sv[k].asarray()[..., elset])
            except:
                return self.sv[k].__class__(np.array(self.sv[k])[..., elset])
        else:  # shoud be array
            return self.sv[k][..., elset]  # gp id should be the last axis

    def __setitem__(self, k, v):  # to define property
        # assume sv values are defined on gauss points.
        # perhaps it may be usefull to allow other definitions

        # --- Treat fields with non consistent  ---
        if k == "Statev":  # for now, only statev need a special treatment
            local_key = f"_{k}_{self.assembly_id}"

            # Handle copy-on-write
            if local_key not in self.copied_fields:
                if local_key in self.sv:
                    self.sv[local_key] = self.sv[local_key].copy()
                self.copied_fields.add(local_key)

            # Assign the raw array directly to the base assembly's dictionary
            self.sv[local_key] = v
            return
        # -----------------------------------------------------------------

        if k in self.sv:  # or k in self.copied_fields:
            if np.isscalar(self.sv[k]) and self.sv[k] == 0:
                # force a filled value of sv
                del self.sv[k]
                self.__setitem__(k, v)

            if k not in self.copied_fields:
                self.sv[k] = self.sv[k].copy()
                # self.sv[k] = deepcopy(self.sv[k])
                self.copied_fields.add(k)

            elset = (
                np.array(self.elset)
                + np.c_[
                    np.arange(
                        0, self.assembly.n_gauss_points, self.assembly.mesh.n_elements
                    )
                ]
            ).reshape(-1)

            if hasattr(self.sv[k], "array"):  # ie ListStressTensor or ListStrainTensor
                self.sv[k].array[..., elset] = v
            else:
                if isinstance(v, list):
                    self._set_value_recursively(self.sv[k], v, elset)
                else:  # v should be array or scalar
                    if np.isscalar(v) or self.sv[k].ndim == v.ndim:
                        self.sv[k][..., elset] = v
                    else:
                        # try if scalar values are given for each components
                        self.sv[k][..., elset] = np.expand_dims(v, axis=-1)
        else:
            if isinstance(v, np.ndarray):
                isarray = True
                arr = v
            else:  # maybe a List for instance TensorStressList object or a list of list
                isarray = False
                try:
                    arr = v.asarray()
                except:
                    arr = np.array(v)

            shape = list(arr.shape)
            # treat the special case where TangentMatrix is a 6x6 matrix (each component are scalar for homogeneous materials)
            if k == "TangentMatrix" and len(shape) == 2:
                shape.append(self.assembly.n_gauss_points)
            else:
                shape[-1] = self.assembly.n_gauss_points

            if isarray:
                self.sv[k] = np.zeros(shape)
            else:
                self.sv[k] = v.__class__(np.zeros(shape))

            self.copied_fields.add(k)
            self.__setitem__(k, v)

    def _set_value_recursively(self, target, value, points):
        """Safely navigates nested lists/arrays and assigns them to the target view."""
        # Check if 'value' is a list/tuple OR an object-type array (jagged)
        if isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                # Recurse into the next dimension: target[i] is a view
                self._set_value_recursively(target[i], item, points)
        else:
            # We've reached a 'leaf' (a single array or a scalar)
            # Convert to array only here to avoid 'inhomogeneous' errors earlier
            target[..., points] = value


class _SubSVComponentMapper:
    """Virtual proxy that maps global component requests to sub-assembly fields.

    This class is dedicated to map state variable components from sub assemblies
    - in the context of Heterogeneous constitutive law - to full assembly. This is
    usefull to allow extracting non homogeneous results, ie mainly the 'Statev' field
    created by the simcoon umats.
    """

    def __init__(self, heter_cl, base_assembly, target_dict, base_key):
        self.heter_cl = heter_cl
        self.base_assembly = base_assembly
        self.target_dict = target_dict  # e.g., assembly.sv or assembly.sv_start
        self.base_key = base_key  # e.g., 'Statev'

    def _get_local_key(self, assembly_id):
        return f"_{self.base_key}_{assembly_id}"

    def __getitem__(self, indices):
        sample_data = None
        valid_phases = []

        # 1. Test which sub-assemblies contain these indices
        for i, sub_assemb in enumerate(self.heter_cl.list_assembly):
            local_key = self._get_local_key(i)
            if local_key in self.target_dict:
                local_sv = self.target_dict[local_key]
                try:
                    # Attempt to slice. If this phase has fewer state variables,
                    # it will raise an IndexError and be skipped safely.
                    sliced_data = local_sv[indices]
                    if sample_data is None:
                        sample_data = sliced_data
                    valid_phases.append((i, sub_assemb, sliced_data))
                except IndexError:
                    continue

        if sample_data is None:
            raise IndexError(
                f"Indices {indices} for field '{self.base_key}' not found in any phase."
            )

        # 2. Initialize the global reconstructed array
        global_shape = list(sample_data.shape)
        global_shape[-1] = self.base_assembly.n_gauss_points
        global_data = np.zeros(global_shape)

        # 3. Stitch valid fragments into the global array
        for i, sub_assemb, local_data in valid_phases:
            elset_gp = (
                np.array(sub_assemb.elset)
                + np.c_[
                    np.arange(
                        0,
                        self.base_assembly.n_gauss_points,
                        self.base_assembly.mesh.n_elements,
                    )
                ]
            ).reshape(-1)

            global_data[..., elset_gp] = local_data

        return global_data

    def __setitem__(self, indices, value):
        for i, sub_assemb in enumerate(self.heter_cl.list_assembly):
            local_key = self._get_local_key(i)
            if local_key in self.target_dict:
                local_sv = self.target_dict[local_key]
                try:
                    # Test if the index is valid for this phase
                    _ = local_sv[indices]
                except IndexError:
                    continue

                # Track modified fields for shallow copy management
                if (
                    self.target_dict is self.base_assembly.sv
                    and local_key not in self.heter_cl._copied_fields
                ):
                    if local_key in self.target_dict:
                        self.target_dict[local_key] = self.target_dict[local_key].copy()
                    self.heter_cl._copied_fields.add(local_key)

                elset_gp = (
                    np.array(sub_assemb.elset)
                    + np.c_[
                        np.arange(
                            0,
                            self.base_assembly.n_gauss_points,
                            self.base_assembly.mesh.n_elements,
                        )
                    ]
                ).reshape(-1)

                if np.isscalar(value):
                    self.target_dict[local_key][indices] = value
                else:
                    self.target_dict[local_key][indices] = value[..., elset_gp]


class _SubSVComponentMapper:
    """Virtual proxy that maps global component requests to sub-assembly fields.

    This class is dedicated to map state variable components from sub assemblies
    - in the context of Heterogeneous constitutive law - to full assembly. This is
    usefull to allow extracting non homogeneous results, ie mainly the 'Statev' field
    created by the simcoon umats.
    """

    def __init__(self, heter_cl, base_assembly, target_dict, base_key):
        self.heter_cl = heter_cl  # heterogeneous constitutive law
        self.base_assembly = base_assembly
        self.target_dict = target_dict  # e.g. assembly.sv or assembly.sv_start
        self.base_key = base_key  # e.g., 'Statev' or 'InternalVars'

    def _get_local_key(self, assembly_id):
        return f"_{self.base_key}_{assembly_id}"

    def __getitem__(self, identifier):
        sample_data = None
        valid_phases = []

        # Scenario A: The identifier is a component string name (e.g., 'Damage')
        if isinstance(identifier, str):
            for i, sub_assemb in enumerate(self.heter_cl.list_assembly):
                if (
                    hasattr(sub_assemb, "sv_component")
                    and identifier in sub_assemb.sv_component
                ):
                    sv_name, local_idx = sub_assemb.sv_component[identifier]
                    if sv_name == self.base_key:
                        local_key = self._get_local_key(i)
                        if local_key in self.target_dict:
                            sliced_data = self.target_dict[local_key][local_idx]
                            if sample_data is None:
                                sample_data = sliced_data
                            valid_phases.append((i, sub_assemb, sliced_data))

            if sample_data is None:
                raise KeyError(
                    f"Component '{identifier}' not found in any sub-assembly for field '{self.base_key}'"
                )

        # Scenario B: The identifier is a direct integer/slice (fallback)
        else:
            for i, sub_assemb in enumerate(self.heter_cl.list_assembly):
                local_key = self._get_local_key(i)
                if local_key in self.target_dict:
                    try:
                        sliced_data = self.target_dict[local_key][identifier]
                        if sample_data is None:
                            sample_data = sliced_data
                        valid_phases.append((i, sub_assemb, sliced_data))
                    except IndexError:
                        continue

            if sample_data is None:
                raise IndexError(
                    f"Index {identifier} not found in any phase for field '{self.base_key}'."
                )

        # Reconstruct and stitch the global array
        global_shape = list(sample_data.shape)
        global_shape[-1] = self.base_assembly.n_gauss_points
        global_data = np.zeros(global_shape)

        for i, sub_assemb, local_data in valid_phases:
            elset = (
                self.base_assembly.mesh.element_sets[sub_assemb.elset]
                if isinstance(sub_assemb.elset, str)
                else sub_assemb.elset
            )
            elset_gp = (
                np.array(elset)
                + np.c_[
                    np.arange(
                        0,
                        self.base_assembly.n_gauss_points,
                        self.base_assembly.mesh.n_elements,
                    )
                ]
            ).reshape(-1)

            global_data[..., elset_gp] = local_data

        return global_data

    def __setitem__(self, indices, value):
        for i, sub_assemb in enumerate(self.heter_cl.list_assembly):
            local_key = self._get_local_key(i)
            if local_key in self.target_dict:
                local_sv = self.target_dict[local_key]
                try:
                    # Test if the index is valid for this phase
                    _ = local_sv[indices]
                except IndexError:
                    continue

                # Track modified fields for shallow copy management
                if (
                    self.target_dict is self.base_assembly.sv
                    and local_key not in self.heter_cl._copied_fields
                ):
                    if local_key in self.target_dict:
                        self.target_dict[local_key] = self.target_dict[local_key].copy()
                    self.heter_cl._copied_fields.add(local_key)
                elset = (
                    self.base_assembly.mesh.element_sets[sub_assemb.elset]
                    if isinstance(sub_assemb.elset, str)
                    else sub_assemb.elset
                )
                elset_gp = (
                    np.array(elset)
                    + np.c_[
                        np.arange(
                            0,
                            self.base_assembly.n_gauss_points,
                            self.base_assembly.mesh.n_elements,
                        )
                    ]
                ).reshape(-1)

                if np.isscalar(value):
                    self.target_dict[local_key][indices] = value
                else:
                    self.target_dict[local_key][indices] = value[..., elset_gp]


class Heterogeneous(Mechanical3D):
    """Constitutive Law that allowing to define an heterogeneous constitutive laws.
    
    To define constitutive from a list of phase constitutive laws, and a list of
    element sets.
        
    Parameters
    ----------
    
    tup_cl: tuple or list
        list of constitutive laws for each phase.
    tup_elset: tuple or list
        list of element set that may be given as str (if present in the mesh.element_sets dictionnary)
        or as list of element index.
    name: str
        The name of the heterogeneous constitutive law

    
    Example
    --------
    
    Create a one element mesh from a 2d mesh in a 3d space:
       >>> import fedoo as fd
       >>> import numpy as np
       >>> 
       >>> #Generate a mesh with a spherical inclusion inside
       >>> 
       >>> #matrix 
       >>> mesh = fd.mesh.hole_plate_mesh(nr=11, nt=11, length=100, height=100, radius=20, \
       >>> 	elm_type = 'quad4', name ="Domain")
       >>> mesh.element_sets['matrix'] = np.arange(0,mesh.n_elements)
       >>> 
       >>> #inclusion
       >>> mesh_disk = fd.mesh.disk_mesh(20, 11, 11)
       >>> mesh_disk.element_sets['inclusion'] = np.arange(0,mesh_disk.n_elements)
       >>>         
       >>> #glue the inclusion to the matrix
       >>> mesh = mesh + mesh_disk
       >>> mesh.merge_nodes(np.c_[mesh.node_sets['hole_edge'], mesh.node_sets['boundary']])
       >>> 
       >>> #Define the Modeling Space - Here 2D problem with plane stress assumption.
       >>> fd.ModelingSpace("2Dstress") 
       >>> 
       >>> #define the materials and build the heterogeneous Assembly
       >>> material1 = fd.constitutivelaw.ElasticIsotrop(2e4, 0.3) 
       >>> material2 = fd.constitutivelaw.ElasticIsotrop(1e5, 0.3) 
       >>> 
       >>> material = fd.constitutivelaw.Heterogeneous((material1, material2), ('matrix', 'inclusion'))
       >>> 
       >>> wf = fd.weakform.StressEquilibrium(material) 
       >>> assembly = fd.Assembly.create(wf, mesh)      
       >>> 
       >>> #Define a new static problem
       >>> pb = fd.problem.Linear(assembly)
       >>> 
       >>> #Definition of the set of nodes for boundary conditions
       >>> left = mesh.find_nodes('X',mesh.bounding_box.xmin)
       >>> right = mesh.find_nodes('X',mesh.bounding_box.xmax)
       >>> 
       >>> #Boundary conditions
       >>> pb.bc.add('Dirichlet', left, 'Disp',    0 )     
       >>> pb.bc.add('Dirichlet', right, 'Disp', [20,0] )     
       >>> pb.apply_boundary_conditions()
       >>> 
       >>> #Solve problem
       >>> pb.solve()
       >>> 
       >>> #---------- Post-Treatment ----------    
       >>> res = pb.get_results(assembly, ['Stress','Strain', 'Disp'])
       >>> res.plot('Stress', component='vm')      
    """

    def __init__(self, tup_cl, tup_elset, name=""):
        Mechanical3D.__init__(self, name)  # heritage
        self.list_cl = tup_cl
        self.list_elset = tup_elset

    def initialize(self, assembly, pb):
        # self.list_mesh = [assembly.mesh.extract_elements(elset) for elset in self.list_elset]
        self._copied_fields = set()  # set of field that have already been copied
        # assembly.sv field need to be copied and can't be just modified because
        # it will also modified assembly.sv_start (shallow copy for performance reason)
        # copied_fields is a set that keep in memory the field that have already been copied.
        self.list_assembly = [
            _SubAssembly(assembly, elset, i, self._copied_fields)
            for i, elset in enumerate(self.list_elset)
        ]

        for i, cl in enumerate(self.list_cl):
            cl.initialize(self.list_assembly[i], pb)

        # Fields that should be managed via the Mapper Proxy
        fragmented_fields = ["Statev"]

        # Bubble up sv_component to the base assembly
        for sub_assemb in self.list_assembly:
            if hasattr(sub_assemb, "sv_component"):
                for comp_name, (sv_name, local_idx) in sub_assemb.sv_component.items():
                    if sv_name in fragmented_fields:
                        # Replace the local integer index with the string name
                        # so the script requests: proxy['Damage']
                        assembly.sv_component[comp_name] = (sv_name, comp_name)
                    else:
                        # For global fields, just use the local index directly
                        assembly.sv_component[comp_name] = (sv_name, local_idx)

        # Inject Mappers for each fragmented field directly into the base assembly
        for field in fragmented_fields:
            if field in assembly.sv or any(
                f"_{field}_{i}" in assembly.sv for i in range(len(self.list_cl))
            ):
                assembly.sv[field] = _SubSVComponentMapper(
                    self, assembly, assembly.sv, field
                )
                assembly.sv_start[field] = _SubSVComponentMapper(
                    self, assembly, assembly.sv_start, field
                )

    def update(self, assembly, pb):
        self._copied_fields.clear()  # to force a new copy of each modified fields
        for i, cl in enumerate(self.list_cl):
            cl.update(self.list_assembly[i], pb)

    def set_start(self, assembly, pb):
        self._copied_fields.clear()  # to force a new copy of each modified fields
        for i, cl in enumerate(self.list_cl):
            cl.set_start(self.list_assembly[i], pb)

    def to_start(self, assembly, pb):
        self._copied_fields.clear()  # to force a new copy of each modified fields
        for i, cl in enumerate(self.list_cl):
            cl.to_start(self.list_assembly[i], pb)

    # def get_tangent_matrix(self, assembly, dimension=None): #Tangent Matrix in lobal coordinate system (no change of basis)

    #     if dimension is None: dimension = assembly.space.get_dimension()

    #     # H = self.local2global_H(self._H)
    #     if dimension == "2Dstress":
    #         return self.get_H_plane_stress(assembly.sv['TangentMatrix'])
    #     else:
    #          assembly.sv['TangentMatrix']

    # def get_elastic_matrix(self, dimension = "3D"):
    #     return self.get_tangent_matrix(None,dimension)

    # def ComputeStrain(self, assembly, pb, nlgeom, type_output='GaussPoint'):
    #     displacement = pb.get_dof_solution()
    #     if np.isscalar(displacement) and displacement == 0:
    #         return 0 #if displacement = 0, Strain = 0
    #     else:
    #         return assembly.get_strain(displacement, type_output)
