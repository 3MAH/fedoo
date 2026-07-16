import numpy as np

# from fedoo.core.mesh import *
from fedoo.core.base import AssemblyBase
from fedoo.core.mesh import MultiMesh

# from fedoo.util.ExportData import ExportData
from fedoo.core.dataset import DataSet, MultiFrameDataSet
import os
from zipfile import ZipFile, Path

_available_output = [
    "PKII",
    "PK2",
    "Kirchoff",
    "Kirchhoff",
    "Cauchy",
    "PKII_vm",
    "PK2_vm",
    "Krichoff_vm",
    "Kirchhoff_vm",
    "Cauchy_vm",
    "PKII_pc",
    "PK2_pc",
    "Kirchoff_pc",
    "Kirchhoff_pc",
    "Cauchy_pc",
    "Stress_pc",
    "PKII_pdir1",
    "PK2_pdir1",
    "Kirchoff_pdir1",
    "Kirchhoff_pdir1",
    "Cauchy_pdir1",
    "Stress_pdir1",
    "PKII_pdir2",
    "PK2_pdir2",
    "Kirchoff_pdir2",
    "Kirchhoff_pdir2",
    "Cauchy_pdir2",
    "Stress_pdir2",
    "PKII_pdir3",
    "PK2_pdir3",
    "Kirchoff_pdir3",
    "Kirchhoff_pdir3",
    "Cauchy_pdir3",
    "Stress_pdir3",
    "Disp",
    "Rot",
    "Temp",
    "Strain",
    "Statev",
    "Stress",
    "Stress_vm",
    "Fext",
    "Wm",
    "Fint",
    "Fint_global",
    "BeamStrain",
    "BeamStress",
    "DispGradient",
]

_available_format = [
    "fdz",
    "vtk",
    "msh",
    "npz",
    "csv",
    "xlsx",
    "fdh5",
]


def _unique_assembly_mesh_entries(assemb, element_set=None):
    """Return unique mesh entries for an AssemblySum.

    The returned list keeps the first occurrence order of meshes in
    ``assemb.list_assembly``. If several elementary assemblies are attached to
    the same mesh, they are grouped in the same entry and keep their original
    assembly order. This order is later used to select the first assembly that
    can provide a requested output field.
    """
    entries = []
    mesh_id_to_entry = {}

    for sub_assemb in assemb.list_assembly:
        mesh = sub_assemb.mesh

        if element_set is not None and isinstance(element_set, str):
            if element_set not in mesh.element_sets:
                continue

        mesh_id = id(mesh)
        if mesh_id not in mesh_id_to_entry:
            if element_set is None:
                output_mesh = mesh
            else:
                output_mesh = mesh.extract_elements(element_set)

            mesh_id_to_entry[mesh_id] = len(entries)
            entries.append(
                {
                    "mesh": output_mesh,
                    "assemblies": [],
                }
            )

        entries[mesh_id_to_entry[mesh_id]]["assemblies"].append(sub_assemb)

    return entries


def _output_mesh_for_assembly(assemb, element_set=None):
    """Return the mesh associated with an output request.

    For a regular assembly, this is its mesh, optionally restricted to
    ``element_set``. For an ``AssemblySum``, this builds a ``MultiMesh`` from
    the unique elementary assembly meshes, in
    ``list_assembly`` order. If all elementary assemblies share one mesh, the
    mesh itself is returned.
    """
    if hasattr(assemb, "list_assembly"):
        entries = _unique_assembly_mesh_entries(assemb, element_set)
        meshes = [entry["mesh"] for entry in entries]
        if not meshes:
            raise NameError("No mesh available for the requested element set.")
        if len(meshes) == 1:
            return meshes[0]
        return MultiMesh.from_mesh_list(
            meshes,
            name=getattr(assemb, "name", ""),
            register_name=False,
        )

    if element_set is None:
        return assemb.mesh
    return assemb.mesh.extract_elements(element_set)


def _find_constitutivelaw_with_method(weakform, method_name):
    """Return a constitutive law from a possibly nested weakform."""
    law = getattr(weakform, "constitutivelaw", None)
    if law is not None and hasattr(law, method_name):
        return law

    for child in getattr(weakform, "list_weakform", []):
        law = _find_constitutivelaw_with_method(child, method_name)
        if law is not None:
            return law

    wrapped = getattr(weakform, "weakform", None)
    if wrapped is not None:
        return _find_constitutivelaw_with_method(wrapped, method_name)

    return None


def _dataset_field(data_set, field):
    """Return a field and its data type from a DataSet, or (None, None)."""
    if field in data_set.node_data:
        return data_set.node_data[field], "Node"
    if field in data_set.element_data:
        return data_set.element_data[field], "Element"
    if field in data_set.gausspoint_data:
        return data_set.gausspoint_data[field], "GaussPoint"
    if field in data_set.scalar_data:
        return data_set.scalar_data[field], "Scalar"
    return None, None


def _store_assemblysum_field(result, field, data, data_type, submesh_id, multimesh):
    """Store one field block extracted from an AssemblySum.

    Element and Gauss point fields are stored per submesh when the AssemblySum
    output mesh is a ``MultiMesh``. Node and scalar fields are shared by the
    node list or by the problem and therefore keep the first available value.
    """
    if data_type == "Node":
        if field not in result.node_data:
            result.node_data[field] = data
    elif data_type == "Element":
        if multimesh:
            result.element_data.setdefault(field, {})[submesh_id] = data
        else:
            result.element_data[field] = data
    elif data_type == "GaussPoint":
        if multimesh:
            result.gausspoint_data.setdefault(field, {})[submesh_id] = data
        else:
            result.gausspoint_data[field] = data
    elif data_type == "Scalar":
        if field not in result.scalar_data:
            result.scalar_data[field] = data


def _get_assemblysum_results(
    pb,
    assemb,
    output_list,
    output_type=None,
    position=1,
    element_set=None,
    include_mesh=True,
):
    """Collect output data from an AssemblySum.

    Each requested field is searched independently on the elementary
    assemblies. For each unique mesh, assemblies attached to that mesh are
    tried in their ``list_assembly`` order and the first assembly providing
    the field is used. If no assembly provides a field for a mesh, that mesh is
    simply left without data for the field. If no assembly provides the field
    at all, the field is omitted from the returned dataset.

    When several unique meshes are present, the returned ``DataSet`` is
    associated with a ``MultiMesh`` and element/Gauss point fields are stored
    as per-submesh dictionaries consumable by ``MultiMeshData``.
    """
    entries = _unique_assembly_mesh_entries(assemb, element_set)
    multimesh = len(entries) > 1

    if include_mesh:
        mesh = _output_mesh_for_assembly(assemb, element_set)
        result = DataSet(mesh)
    else:
        result = DataSet()

    for res in output_list:
        field_type = None
        found = False

        for submesh_id, entry in enumerate(entries):
            for sub_assemb in entry["assemblies"]:
                try:
                    sub_result = _get_results(
                        pb,
                        sub_assemb,
                        [res],
                        output_type,
                        position,
                        element_set,
                        False,
                    )
                except NameError:
                    continue

                data, data_type = _dataset_field(sub_result, res)
                if data_type is None:
                    continue
                if field_type is None:
                    field_type = data_type
                elif data_type != field_type:
                    raise NameError(
                        f'Field "{res}" has inconsistent output types in '
                        "the AssemblySum."
                    )

                _store_assemblysum_field(
                    result,
                    res,
                    data,
                    data_type,
                    submesh_id,
                    multimesh,
                )
                found = True
                break

        if not found:
            continue

    if hasattr(pb, "time"):
        result.scalar_data["Time"] = pb.time

    return result


def _get_results(
    pb,
    assemb,
    output_list,
    output_type=None,
    position=1,
    element_set=None,
    include_mesh=True,
):
    if isinstance(output_list, str):
        output_list = [output_list]

    if output_type is not None:
        if output_type.lower() == "node":
            output_type = "Node"
        elif output_type.lower() == "element":
            output_type = "Element"
        elif output_type.lower() == "gausspoint":
            output_type = "GaussPoint"
        else:
            raise NameError(
                "output_type should be either 'Node', 'Element' or 'GaussPoint'"
            )

    if isinstance(assemb, str):
        assemb = AssemblyBase.get_all()[assemb]

    if hasattr(assemb, "list_assembly"):
        return _get_assemblysum_results(
            pb,
            assemb,
            output_list,
            output_type,
            position,
            element_set,
            include_mesh,
        )

    # for i, res in enumerate(output_list):
    # if (
    #     res not in _available_output
    #     and res not in assemb.space.list_variables()
    #     and res not in assemb.space.list_vectors()
    #     and res not in assemb.sv
    #     and res not in assemb.sv_component
    #     and res not in pb._global_dof._variable
    #     and res not in pb._global_dof._vector
    # ):
    #     print("List of available output: ", _available_output)
    #     raise NameError(res, "' doens't match to any available output")

    data_sav = {}  # dict to keep data in memory that may be used more that one time

    sv = assemb.sv  # state variables associated to the assembly

    if include_mesh:
        result = DataSet(_output_mesh_for_assembly(assemb, element_set))
        if element_set is not None and isinstance(element_set, str):
            element_set = assemb.mesh.element_sets[element_set]
    else:
        result = DataSet()

    for res in output_list:
        if res in pb.space.list_variables() or res in pb.space.list_vectors():
            data = pb.get_dof_solution(res)
            data_type = "Node"

        elif res in pb._global_dof._variable or res in pb._global_dof._vector:
            data = pb.get_dof_solution(res)
            data_type = "Scalar"

        elif res == "Fext":
            # Only node dof
            data = pb.get_ext_forces(include_mpc=False)[: pb.n_node_dof].reshape(
                pb.space.nvar, -1
            )
            data_type = "Node"

        elif res[:5] == "Fext(" and res[-1] == ")":
            var = res[5:-1]
            data = pb.get_ext_forces(var)
            if data.shape[-1] == assemb.mesh.n_nodes:
                data_type = "Node"  # if var is a node field variable (or vector)
            else:
                data_type = "Scalar"  # if var is global_dof variable

        elif res in ["PK2", "Kirchhoff", "Strain", "Stress"]:
            if res in data_sav:
                data = data_sav[res]  # avoid a new data conversion
            else:
                if res in sv:
                    data = sv[res]
                else:
                    # attent to compute
                    method_name = "get_strain" if res == "Strain" else "get_stress"
                    law = _find_constitutivelaw_with_method(
                        assemb.weakform, method_name
                    )
                    if law is None:
                        raise NameError('Field "{}" not available'.format(res))
                    try:
                        data = getattr(law, method_name)(assemb, position=position)
                    except Exception as exc:
                        raise NameError('Field "{}" not available'.format(res)) from exc

                # keep data in memory in case it may be used later for vm, pc or pdir stress computation
                data_sav[res] = data

                if output_type is not None and output_type != "GaussPoint":
                    data = data.convert(assemb, None, output_type)
                    data_type = output_type
                else:
                    data_type = "GaussPoint"

            if hasattr(data, "asarray"):
                data = data.asarray()
            else:
                data = np.array(data)

        elif res in ["PK2_vm", "Kirchhoff_vm", "Stress_vm"]:
            if res[:-3] in data_sav:
                data = data_sav[res[:-3]]
            else:
                data = sv[res[:-3]]
                data_sav[res[:-3]] = data

            data = data.von_mises()
            data_type = "GaussPoint"

        elif res in [
            "PK2_pc",
            "Kirchhoff_pc",
            "Cauchy_pc",
            "Stress_pc",
            "PK2_pdir1",
            "Kirchhoff_pdir1",
            "Cauchy_pdir1",
            "Stress_pdir1",
            "PK2_pdir2",
            "Kirchhoff_pdir2",
            "Cauchy_pdir2",
            "Stress_pdir2",
            "PK2_pdir3",
            "Kirchhoff_pdir3",
            "Cauchy_pdir3",
            "Stress_pdir3",
        ]:
            # stress principal component
            if res[-3:] == "_pc":
                measure_type = res[:-3]
            else:
                measure_type = res[:-6]

            if measure_type + "_pc" in data_sav:
                data = data_sav[measure_type + "_pc"]

            elif measure_type in data_sav:
                data = data_sav[measure_type]
                data = data.diagonalize()
                data_sav[measure_type + "_pc"] = data

            else:
                data = sv[measure_type]
                # if measure_type in ['PKII','PK2']:
                #     data = material.get_pk2()
                # elif measure_type == 'Stress':
                #     data = material.get_stress(position = position)
                # elif measure_type == 'Kirchhoff':
                #     data = material.get_kirchhoff()
                # elif measure_type == 'Cauchy':
                #     data = material.get_cauchy()

                data_sav[measure_type] = data
                data = data.diagonalize()
                data_sav[measure_type + "_pc"] = data

            if res[-3:] == "_pc":  # principal component
                data = data[0]  # principal component
            elif res[-6:] == "_pdir1":  # 1st principal direction
                data = data[1][0]
            elif res[-6:] == "_pdir2":  # 2nd principal direction
                data = data[1][1]
            elif res[-6:] == "_pdir3":  # 3rd principal direction
                data = data[1][2]

            data_type = "GaussPoint"

        elif res in sv:
            data = sv[res]
            data_type = assemb.sv_type.get(res, "GaussPoint")
            if isinstance(data, list):
                # try to convert into array
                try:
                    if hasattr(data, "asarray"):
                        data = data.asarray()
                    else:
                        data = np.array(data)
                except ValueError:
                    import warnings

                    warnings.warn(
                        (
                            f"{res} can't be converted into array "
                            "during results extraction."
                        )
                    )

        elif res in assemb.sv_component:
            (sv_name, indices) = assemb.sv_component[res]
            data = assemb.sv[sv_name][indices]
            data_type = assemb.sv_type.get(sv_name, "GaussPoint")

        elif res == "Fint":
            data = assemb.get_int_forces(pb.get_dof_solution(), "local").T
            data_type = "GaussPoint"  # or 'Element' ?

        elif res == "Fint_global":
            data = assemb.get_int_forces(pb.get_dof_solution(), "global").T
            data_type = "GaussPoint"  # or 'Element' ?

        else:
            raise NameError(res, "' doens't match to any available output")

        if (
            (output_type is not None)
            and (output_type != "Scalar")
            and (output_type != data_type)
        ):
            data = assemb.convert_data(data, data_type, output_type)
            data_type = output_type

        if data_type == "Node":
            result.node_data[res] = data
        elif data_type == "Element":
            if element_set is None:
                result.element_data[res] = data
            else:
                result.element_data[res] = data.T[element_set].T
        elif data_type == "GaussPoint":
            if element_set is None:
                result.gausspoint_data[res] = data
            else:
                if data.ndim == 1:
                    data = data.reshape(-1, assemb.mesh.n_elements)
                    result.gausspoint_data[res] = data[:, element_set].ravel()
                else:  # data.ndim ==2
                    data = data.reshape(data.shape[0], -1, assemb.mesh.n_elements)
                    result.gausspoint_data[res] = data[:, :, element_set].reshape(
                        data.shape[0], -1
                    )
        elif data_type == "Scalar":
            result.scalar_data[res] = data

    if hasattr(pb, "time"):
        result.scalar_data["Time"] = pb.time

    return result


class _ProblemOutput:
    def __init__(self):
        self.__list_output = []  # a list containint dictionnary with defined output
        self.data_sets = {}

    def add_output(
        self,
        filename,
        assemb,
        output_list,
        output_type=None,
        file_format="fdh5",
        compressed=False,
        position=1,
        element_set=None,
        save_mesh=True,
    ):
        dirname = os.path.dirname(filename)
        # filename = os.path.basename(filename)
        extension = os.path.splitext(filename)[1]
        if extension == "":
            file_format = file_format.lower()
            if file_format not in ["fdz", "fdh5"]:
                # if no extention -> create a new dir using filename as dirname
                dirname = filename + "/"
                filename = dirname + os.path.basename(filename)
        else:
            # use extension as file format
            file_format = extension[1:].lower()
            filename = os.path.splitext(filename)[
                0
            ]  # remove extension for the base name

        if file_format not in _available_format:
            print(
                "WARNING: '",
                file_format,
                "' doens't match to any available file format",
            )
            print("Specified output ignored")
            print("List of available file format: ", _available_format)

        if output_type is not None and output_type.lower() not in [
            "node",
            "element",
            "gausspoint",
        ]:
            raise NameError(
                "output_type should be either 'Node', 'Element' or 'GaussPoint'"
            )

        for i, res in enumerate(output_list):
            output_list[i] = res

        if isinstance(assemb, str):
            assemb = AssemblyBase.get_all()[assemb]

        mesh = _output_mesh_for_assembly(assemb, element_set)

        if not (os.path.isdir(dirname)) and dirname != "":
            os.mkdir(dirname)

        new_output = {
            "filename": filename,
            "assembly": assemb,
            "type": output_type,
            "list": output_list,
            "file_format": file_format.lower(),
            "position": position,
            "element_set": element_set,
            "compressed": compressed,
        }
        self.__list_output.append(new_output)

        # if file_format in ['npz', 'npz_compressed', 'fdz', 'fdz_compressed']:
        if not (filename in self.data_sets):
            if file_format == "fdz":
                file = ZipFile(filename + ".fdz", "w")  # create a new zip file
                mesh.save("_mesh_")  # create temp '_mesh_.vtk' file

                file.write("_mesh_.vtk")  # add '_mesh_.vtk' to the zip archive
                os.remove("_mesh_.vtk")
                file.close()
            elif save_mesh and (file_format not in ["vtk", "msh", "fdh5"]):
                mesh.save(filename)

            res = MultiFrameDataSet(mesh, [])
            self.data_sets[filename] = res

        else:
            # TODO: use full_filename (with extension) instead of filename
            # to avoid confusion for same file with different extensions
            res = self.data_sets[filename]
        return res

    def save_results(self, pb, comp_output=None):
        list_filename = []
        list_full_filename = []
        list_file_format = []
        list_compressed = []  # True if the file should be compressed
        list_data = []

        for output in self.__list_output:
            filename = output["filename"]
            file_format = output["file_format"].lower()
            output_type = output["type"]  # 'Node', 'Element' or 'GaussPoint'
            position = output["position"]
            element_set = output["element_set"]
            compressed = output["compressed"]

            assemb = output["assembly"]
            # material = assemb.weakform.GetConstitutiveLaw()

            if file_format in _available_format:  # if not ignored
                if (comp_output is None) or (file_format in ["fdz", "fdh5"]):
                    filename_compl = ""
                else:
                    filename_compl = "_" + str(comp_output)

                full_filename = (
                    filename + filename_compl + "." + file_format
                )  # filename including iter number and file format

                if not (full_filename in list_full_filename):
                    # if filename don't exist in the list we create it
                    list_filename.append(filename)
                    list_full_filename.append(full_filename)
                    list_file_format.append(file_format)
                    list_compressed.append(compressed)

                    out = DataSet(_output_mesh_for_assembly(assemb, element_set))
                    list_data.append(out)
                else:
                    # else, the same file is used
                    out = list_data[list_full_filename.index(full_filename)]

                # compute the results
                res = _get_results(
                    pb,
                    assemb,
                    output["list"],
                    output_type,
                    position,
                    element_set,
                    False,
                )
                out.add_data(res)

        for i, out in enumerate(list_data):
            if list_file_format[i] == "fdz":
                out.save("_mesh_.npz", False, list_compressed[i])
                file = ZipFile(list_full_filename[i], "a")
                if comp_output is None:
                    iter_name = "iter_0" + ".npz"
                else:
                    iter_name = "iter_" + str(comp_output) + ".npz"
                file.write("_mesh_.npz", iter_name)
                os.remove("_mesh_.npz")
                file.close()
                self.data_sets[list_filename[i]].list_data.append(
                    Path(list_full_filename[i], iter_name)
                )

            elif list_file_format[i] == "fdh5":
                iteration = 0 if comp_output is None else comp_output
                out.to_fdh5(
                    list_full_filename[i],
                    iteration=iteration,
                    overwrite=(comp_output is None or comp_output == 0),
                )
                self.data_sets[list_filename[i]].list_data.append(
                    ("fdh5", list_full_filename[i], iteration)
                )

            else:
                out.save(list_full_filename[i], compressed=list_compressed[i])
                self.data_sets[list_filename[i]].list_data.append(list_full_filename[i])
