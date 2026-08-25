import importlib
import inspect
import types

import fedoo as fd
from fedoo.core.assembly import Assembly
from fedoo.core.assembly_sum import AssemblySum
from fedoo.core.weakform import WeakFormSum


def test_legacy_factory_names_are_concrete_weakform_sum_types():
    fd.ModelingSpace("2Dplane")
    material = fd.constitutivelaw.ElasticIsotrop(1.0e6, 0.3)
    fluid = fd.constitutivelaw.PoroFluidProperties(permeability=1.0e-7)

    weakforms = [
        fd.weakform.ImplicitDynamic(material, density=1.0),
        fd.weakform.StressEquilibriumRI(material),
        fd.weakform.PoroMechanics(
            material,
            fluid,
            bulk_modulus=1.0e6,
            nlgeom=False,
        ),
        fd.weakform.PoroMechanicsSimple(material, fluid, nlgeom=False),
    ]

    expected_types = [
        fd.weakform.ImplicitDynamic,
        fd.weakform.StressEquilibriumRI,
        fd.weakform.PoroMechanics,
        fd.weakform.PoroMechanicsSimple,
    ]

    for weakform, expected_type in zip(weakforms, expected_types):
        assert type(weakform) is expected_type
        assert isinstance(weakform, WeakFormSum)


def test_implicit_dynamic_module_is_not_shadowed_by_a_factory_function():
    implicit_dynamic_module = importlib.import_module("fedoo.weakform.implicit_dynamic")

    assert isinstance(implicit_dynamic_module, types.ModuleType)
    assert implicit_dynamic_module.ImplicitDynamic is fd.weakform.ImplicitDynamic


def test_corate_property_is_local_to_stress_equilibrium_ri():
    assert "corate" not in WeakFormSum.__dict__
    assert "corate" in fd.weakform.StressEquilibriumRI.__dict__


def test_public_classes_own_the_factory_documentation():
    classes = [
        fd.weakform.ImplicitDynamic,
        fd.weakform.StressEquilibriumRI,
        fd.weakform.PoroMechanics,
        fd.weakform.PoroMechanicsSimple,
    ]

    for weakform_class in classes:
        doc = inspect.getdoc(weakform_class)
        assert doc is not None
        assert "Parameters\n----------" in doc
        for parameter in inspect.signature(weakform_class).parameters:
            assert f"{parameter} :" in doc


def test_composite_weakforms_preserve_child_assembly_options():
    fd.ModelingSpace("2Dplane")
    material = fd.constitutivelaw.ElasticIsotrop(1.0e6, 0.3)
    fluid = fd.constitutivelaw.PoroFluidProperties(permeability=1.0e-7)

    implicit = fd.weakform.ImplicitDynamic(material, density=1.0)
    assert implicit.assembly_options is None
    assert all(
        wf.assembly_options.get("assume_sym", "quad4", False)
        for wf in implicit.list_weakform
    )

    reduced_integration = fd.weakform.StressEquilibriumRI(material)
    assert reduced_integration.assembly_options is None
    equilibrium, hourglass = reduced_integration.list_weakform
    assert equilibrium.assembly_options.get("n_elm_gp", "quad4") == 1
    assert equilibrium.assembly_options.get("n_elm_gp", "hex8") == 1
    assert hourglass.assembly_options.get("n_elm_gp", "quad4") == 1
    assert hourglass.assembly_options.get("n_elm_gp", "hex8") == 1
    assert hourglass.assembly_options.get("elm_type", "quad4") == "quad4hourglass"
    assert hourglass.assembly_options.get("elm_type", "hex8") == "hex8hourglass"

    poromechanics = [
        fd.weakform.PoroMechanics(
            material,
            fluid,
            bulk_modulus=1.0e6,
            nlgeom=False,
        ),
        fd.weakform.PoroMechanicsSimple(material, fluid, nlgeom=False),
    ]
    for weakform in poromechanics:
        assert weakform.assembly_options is None
        assert (
            weakform.list_weakform[0].assembly_options.get("assume_sym", "quad4", True)
            is False
        )


def test_composite_assembly_options_are_applied_by_assembly_create():
    fd.ModelingSpace("2Dplane")
    mesh = fd.mesh.rectangle_mesh(nx=3, ny=3, elm_type="quad4")
    material = fd.constitutivelaw.ElasticIsotrop(1.0e6, 0.3)
    fluid = fd.constitutivelaw.PoroFluidProperties(permeability=1.0e-7)

    implicit = fd.weakform.ImplicitDynamic(material, density=1.0)
    implicit_assembly = fd.Assembly.create(implicit, mesh)
    assert isinstance(implicit_assembly, Assembly)
    assert implicit_assembly.n_elm_gp == 4
    assert implicit_assembly.assume_sym is True
    assert implicit_assembly.mat_lumping == [False, False]

    reduced_integration = fd.weakform.StressEquilibriumRI(material)
    reduced_assembly = fd.Assembly.create(reduced_integration, mesh)
    assert isinstance(reduced_assembly, AssemblySum)
    assert [assembly.elm_type for assembly in reduced_assembly.list_assembly] == [
        "quad4",
        "quad4hourglass",
    ]
    assert all(assembly.n_elm_gp == 1 for assembly in reduced_assembly.list_assembly)

    poromechanics = fd.weakform.PoroMechanicsSimple(material, fluid, nlgeom=False)
    poromechanics_assembly = fd.Assembly.create(poromechanics, mesh)
    assert isinstance(poromechanics_assembly, Assembly)
    assert poromechanics_assembly.assume_sym is False
    assert poromechanics_assembly.mat_lumping == [False, False, False]
