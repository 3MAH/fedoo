import numpy as np
import pytest

import fedoo as fd


class _ReferenceBbar(fd.weakform.StressEquilibriumBbar):
    """Direct volumetric/deviatoric split used as an assembly reference."""

    def get_weak_equation(self, assembly, pb):
        eps = self.space.op_strain()
        n_normal_components = self.space.ndim
        eps_vol = sum(eps[:n_normal_components])
        eps_vol_reduced = self.space.derivative("_DispX", "X") + self.space.derivative(
            "_DispY", "Y"
        )
        if n_normal_components == 3:
            eps_vol_reduced += self.space.derivative("_DispZ", "Z")

        correction = (eps_vol_reduced - eps_vol) / n_normal_components
        eps_bar = eps.copy()
        for i in range(n_normal_components):
            eps_bar[i] = eps[i] + correction

        tangent = assembly.sv["TangentMatrix"]
        stress_operator = [
            sum(0 if eps_bar[j] == 0 else eps_bar[j] * tangent[i][j] for j in range(6))
            for i in range(6)
        ]
        return sum(
            0 if eps_bar[i] == 0 else eps_bar[i].virtual * stress_operator[i]
            for i in range(6)
        )


class _ComponentwiseReduced(fd.weakform.StressEquilibriumBbar):
    """Former implementation, retained only to detect the original error."""

    def get_weak_equation(self, assembly, pb):
        eps = self.space.op_strain()
        eps[0] = self.space.derivative("_DispX", "X")
        eps[1] = self.space.derivative("_DispY", "Y")
        if self.space.ndim == 3:
            eps[2] = self.space.derivative("_DispZ", "Z")

        tangent = assembly.sv["TangentMatrix"]
        stress_operator = [
            sum(0 if eps[j] == 0 else eps[j] * tangent[i][j] for j in range(6))
            for i in range(6)
        ]
        return sum(
            0 if eps[i] == 0 else eps[i].virtual * stress_operator[i] for i in range(6)
        )


def _assemble_matrix(dimension, weakform_type):
    fd.ModelingSpace(dimension)
    material = fd.constitutivelaw.ElasticIsotrop(1000.0, 0.499)
    weakform = weakform_type(material)
    if dimension == "3D":
        mesh = fd.mesh.box_mesh(nx=2, ny=2, nz=2, elm_type="hex8")
    else:
        mesh = fd.mesh.rectangle_mesh(nx=2, ny=2, elm_type="quad4")
    assembly = fd.Assembly.create(weakform, mesh)
    fd.problem.Linear(assembly)
    assembly.assemble_global_mat("matrix")
    return assembly.get_global_matrix().toarray()


@pytest.mark.parametrize("dimension", ["2Dplane", "2Dstress", "3D"])
def test_bbar_reduces_only_the_volumetric_strain(dimension):
    matrix = _assemble_matrix(dimension, fd.weakform.StressEquilibriumBbar)
    reference = _assemble_matrix(dimension, _ReferenceBbar)
    componentwise_reduced = _assemble_matrix(dimension, _ComponentwiseReduced)

    assert np.allclose(matrix, reference, rtol=1e-12, atol=1e-12)
    assert not np.allclose(matrix, componentwise_reduced, rtol=1e-8, atol=1e-8)
