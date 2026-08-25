import inspect

import numpy as np

import fedoo as fd


def test_linear_newmark_convenience_constructor():
    parameters = inspect.signature(fd.problem.LinearNewmark).parameters
    assert list(parameters) == [
        "assembly",
        "name",
        "time_step",
        "integrator",
    ]

    fd.ModelingSpace("2Dplane")
    mesh = fd.mesh.rectangle_mesh(nx=2, ny=2, elm_type="tri3")
    material = fd.constitutivelaw.ElasticIsotrop(1.0, 0.3)
    material.set_density(1.0)
    weakform = fd.weakform.StressEquilibrium(material)
    assembly = fd.Assembly.create(weakform, mesh)
    problem = fd.problem.LinearNewmark(
        assembly, time_step=0.1, integrator=fd.time.Newmark()
    )
    assert fd.problem.LinearNewmark is fd.problem.Linear
    assert type(problem) is fd.problem.Linear
    assert isinstance(problem.time_integrator, fd.time.Newmark)


def test_explicit_dynamic_constructor_uses_snake_case_keywords():
    parameters = inspect.signature(fd.problem.ExplicitDynamic).parameters
    assert list(parameters) == [
        "assembly",
        "time_step",
        "integrator",
        "mass_lumping",
        "name",
    ]

    fd.ModelingSpace("2Dplane")
    mesh = fd.mesh.rectangle_mesh(nx=2, ny=2, elm_type="tri3")
    material = fd.constitutivelaw.ElasticIsotrop(1.0, 0.3)
    material.set_density(1.0)
    weakform = fd.weakform.StressEquilibrium(material)
    assembly = fd.Assembly.create(weakform, mesh)
    problem = fd.problem.ExplicitDynamic(
        assembly=assembly,
        time_step=0.1,
    )
    assert isinstance(problem.time_integrator, fd.time.CentralDifference)
    problem.initialize()
    assert problem.get_A().ndim == 1
    assert np.all(problem.get_A() > 0.0)
