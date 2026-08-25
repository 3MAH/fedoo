from types import SimpleNamespace

import pytest

import fedoo as fd


class _AssemblyStub:
    def __init__(self):
        self._nlgeom = False
        self.updated_displacement = None

    def set_disp(self, displacement):
        self.updated_displacement = displacement


def _interface_force(nlgeom=None):
    fd.ModelingSpace("3D")
    law = fd.constitutivelaw.Spring(Kx=1.0, Ky=1.0, Kz=1.0)
    return fd.weakform.InterfaceForce(law, nlgeom=nlgeom)


def test_interface_force_inherits_problem_nlgeom():
    weakform = _interface_force()
    assembly = _AssemblyStub()
    displacement = object()
    problem = SimpleNamespace(
        nlgeom=True,
        get_disp=lambda: displacement,
    )

    weakform.initialize(assembly, problem)
    weakform.update(assembly, problem)

    assert assembly._nlgeom == "UL"
    assert weakform.nlgeom == "UL"
    assert assembly.updated_displacement is displacement


def test_interface_force_explicit_false_overrides_problem_nlgeom():
    weakform = _interface_force(nlgeom=False)
    assembly = _AssemblyStub()
    problem = SimpleNamespace(
        nlgeom=True,
        get_disp=lambda: object(),
    )

    weakform.initialize(assembly, problem)
    weakform.update(assembly, problem)

    assert assembly._nlgeom is False
    assert weakform.nlgeom is False
    assert assembly.updated_displacement is None


def test_interface_force_rejects_total_lagrangian_formulation():
    weakform = _interface_force(nlgeom="TL")
    assembly = _AssemblyStub()
    problem = SimpleNamespace(nlgeom=False)

    with pytest.raises(NotImplementedError, match="InterfaceForce"):
        weakform.initialize(assembly, problem)


if __name__ == "__main__":
    pytest.main([__file__])
