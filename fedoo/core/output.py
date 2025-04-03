import numpy as np

# from fedoo.core.mesh import *
from fedoo.core.base import AssemblyBase

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
]

_label_dict = {
    "pkii": "PK2",
    "pk2": "PK2",
    "kirchoff": "Kirchhoff",
    "kirchhoff": "Kirchhoff",
    "cauchy": "Stress",
    "pkii_vm": "PK2_vm",
    "pk2_vm": "PK2_vm",
    "kirchoff_vm": "Kirchhoff_vm",
    "kirchhoff_vm": "Kirchhoff_vm",
    "cauchy_vm": "Stress_vm",
    "pkii_pc": "PK2_pc",
    "pk2_pc": "PK2_pc",
    "kirchoff_pc": "Kirchhoff_pc",
    "kirchhoff_pc": "Kirchhoff_pc",
    "cauchy_pc": "Stress_pc",
    "stress_pc": "Stress_pc",
    "pkii_pdir1": "PK2_pdir1",
    "pk2_pdir1": "PK2_pdir1",
    "kirchoff_pdir1": "Kirchhoff_pdir1",
    "kirchhoff_pdir1": "Kirchhoff_pdir1",
    "cauchy_pdir1": "Sress_pdir1",
    "stress_pdir1": "Stress_pdir1",
    "pkii_pdir2": "PK2_pdir2",
    "pk2_pdir2": "PK2_pdir2",
    "kirchoff_pdir2": "Kirchhoff_pdir2",
    "kirchhoff_pdir2": "Kirchhoff_pdir2",
    "cauchy_pdir2": "Stress_pdir2",
    "stress_pdir2": "Stress_pdir2",
    "pkii_pdir3": "PK2_pdir3",
    "pk2_pdir3": "PK2_pdir3",
    "kirchoff_pdir3": "Kirchhoff_pdir3",
    "kirchhoff_pdir3": "Kirchhoff_pdir3",
    "cauchy_pdir3": "Stress_pdir1",
    "stress_pdir3": "Stress_pdir3",
    "disp": "Disp",
    "rot": "Rot",
    "temp": "Temp",
    "strain": "Strain",
    "statev": "Statev",
    "stress": "Stress",
    "stress_vm": "Stress_vm",
    "fext": "Fext",
    "wm": "Wm",
    "fint": "Fint",
    "fint_global": "Fint_global",
    "beamstrain": "BeamStrain",
    "beamstress": "BeamStress",
    "dispgradient": "DispGradient",
}


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

    for i, res in enumerate(output_list):
        # output_list[i] = _label_dict[res.lower()] #to allow full lower case str as output
        if (
            res not in _available_output
            and res not in pb.space.list_variables()
            and res not in pb.space.list_vectors()
            and res not in assemb.sv
        ):
            print("List of available output: ", _available_output)
            raise NameError(res, "' doens't match to any available output")

    data_sav = {}  # dict to keep data in memory that may be used more that one time

    if hasattr(assemb, "list_assembly"):  # AssemblySum object
        if assemb.assembly_output is None:
            raise NameError("AssemblySum objects can't be used to extract outputs")
        else:
            assemb = assemb.assembly_output

    sv = assemb.sv  # state variables associated to the assembly

    if include_mesh:
        if element_set is None:
            result = DataSet(assemb.mesh)
        else:
            result = DataSet(assemb.mesh.extract_elements(element_set))
            if isinstance(element_set, str):
                element_set = assemb.mesh.element_sets[element_set]
    else:
        result = DataSet()

    for res in output_list:
        if res in pb.space.list_variables() or res in pb.space.list_vectors():
            data = pb.get_dof_solution(res)
            data_type = "Node"

        elif res == "Fext":
            # data = assemb.get_ext_forces(pb.get_dof_solution())
            data = pb.get_ext_forces().reshape(pb.space.nvar, -1)
            data_type = "Node"

        elif res in ["PK2", "Kirchhoff", "Strain", "Stress"]:
            if res in data_sav:
                data = data_sav[res]  # avoid a new data conversion
            else:
                if res in sv:
                    data = sv[res]
                else:
                    # attent to compute
                    try:
                        if res == "Strain":
                            data = assemb.weakform.constitutivelaw.get_strain(
                                assemb, position=position
                            )
                        elif res == "Stress":
                            data = assemb.weakform.constitutivelaw.get_stress(
                                assemb, position=position
                            )
                        else:
                            assert 0
                    except:
                        raise NameError('Field "{}" not available'.format(res))

                # keep data in memory in case it may be used later for vm, pc or pdir stress computation
                data_sav[res] = data

                if output_type is not None and output_type != "GaussPoint":
                    data = data.convert(assemb, None, output_type)
                    data_type = output_type
                else:
                    data_type = "GaussPoint"

            try:
                data = data.asarray()
            except:
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

        elif res == "Statev":
            data = sv["Statev"]
            # data = material.get_statev()
            # data = assemb.convert_data(data, None, output_type).T
            data_type = "GaussPoint"

        elif res in ["Wm"]:
            data = sv["Wm"]
            # data = material.get_wm()
            # data = assemb.convert_data(data, None, output_type).T
            data_type = "GaussPoint"

        elif res in sv:
            data = sv[res]
            data_type = assemb.sv_type.get(res, "GaussPoint")
            if isinstance(data, list):
                # try to convert into array
                try:
                    data = np.array(data)
                except ValueError:
                    import warnings

                    warnings.warn(
                        (
                            f"{res} can't be converted into array "
                            "during results extraction."
                        )
                    )

        elif res == "Fint":
            data = assemb.get_int_forces(pb.get_dof_solution(), "local").T
            # data = assemb.convert_data(data, None, output_type)
            data_type = "GaussPoint"  # or 'Element' ?

        elif res == "Fint_global":
            data = assemb.get_int_forces(pb.get_dof_solution(), "global").T
            # data = assemb.convert_data(data, None, output_type)
            data_type = "GaussPoint"  # or 'Element' ?

        if output_type is not None and output_type != data_type:
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
        file_format="fdz",
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
            if file_format != "fdz":
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
            output_list[i] = _label_dict[
                res.lower()
            ]  # to allow full lower case str as output
            if res not in _available_output:
                print(
                    "WARNING: '",
                    res,
                    "' doens't match to any available output",
                )
                print("Specified output ignored")
                print("List of available output: ", _available_output)

        if isinstance(assemb, str):
            assemb = AssemblyBase.get_all()[assemb]

        if element_set is None:
            mesh = assemb.mesh
        else:
            mesh = assemb.mesh.extract_elements(element_set)

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
            elif save_mesh and (file_format not in ["vtk", "msh"]):
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
                if (comp_output is None) or (file_format == "fdz"):
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

                    out = DataSet(assemb.mesh)
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

            else:
                out.save(list_full_filename[i], compressed=list_compressed[i])
                self.data_sets[list_filename[i]].list_data.append(list_full_filename[i])
