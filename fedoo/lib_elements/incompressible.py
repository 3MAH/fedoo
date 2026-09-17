"""Elements to avoid volumetric locking in nearly incompressible analysis."""

import numpy as np

from fedoo.lib_elements.element_base import (
    _evaluate_monomials,
    _monomial_exponents,
)
from fedoo.lib_elements.element_list import CombinedElement, get_element

# selective reduce integration
hex8sri = CombinedElement("hex8sri", "hex8", default_n_gp=8)
hex8sri.set_variable_interpolation("_DispX", "hex8r")
hex8sri.set_variable_interpolation("_DispY", "hex8r")
hex8sri.set_variable_interpolation("_DispZ", "hex8r")

quad4sri = CombinedElement("quad4sri", "quad4", default_n_gp=4)
quad4sri.set_variable_interpolation("_DispX", "quad4r")
quad4sri.set_variable_interpolation("_DispY", "quad4r")
quad4sri.set_variable_interpolation("_DispZ", "quad4r")


# mean dilatation (element-wise discontinuous pressure, statically condensed)
class _MeanDilatationMixin:
    """Interpolation whose derivatives are the deviation from the mean.

    The physical derivatives dN/dx are replaced by ``P(dN/dx) - dN/dx`` where
    ``P`` is the volume weighted L2 projection, over each element, onto the
    polynomial basis of degree ``pressure_degree`` in the reference coordinates
    (``n_pvals`` functions: the discontinuous pressure/dilatation
    interpolation). Applied to the displacement, the sum of the derivatives
    gives the correction ``theta_bar - div(u)`` of the mean dilatation method.

    If the reference gauss point volumes are given in
    ``assembly.sv["_MeanDilatation_dV0"]`` (finite strain, updated lagrangian),
    the projection is the variation ``delta(theta)/theta`` of the dilatation
    ``theta = P0(J)`` projected over the reference configuration.
    """

    pressure_degree = 0

    def __init__(self, n_elm_gp=None, **kargs):
        if n_elm_gp is None:
            n_elm_gp = self.default_n_gp
        super().__init__(n_elm_gp, **kargs)
        elm_geom = kargs.get("elmGeom")
        if elm_geom is None:
            raise ValueError(
                f"'{self.name}' elements require the geometrical interpolation"
                " (elmGeom) to build the mean dilatation operators."
            )

        # basis of the discontinuous dilatation at gauss points: (n_gp, n_pvals)
        basis = _evaluate_monomials(
            self.xi_pg,
            _monomial_exponents(self.xi_pg.shape[1], self.pressure_degree),
        )
        if n_elm_gp <= basis.shape[1]:
            raise ValueError(
                f"The mean dilatation method with {basis.shape[1]} pressure "
                f"value(s) per element requires more than {basis.shape[1]} "
                f"gauss points ({n_elm_gp} used with '{self.name}')."
            )

        # current gauss point volumes, shape = (n_elements, n_elm_gp)
        dv = elm_geom.detJ * elm_geom.w_pg
        dv0 = dv
        assembly = kargs.get("assembly")
        if assembly is not None:
            # reference volumes are stored as (n_elm_gp, n_elements)
            dv0 = assembly.sv.get("_MeanDilatation_dV0", dv.T).T

        # physical derivatives, shape = (n_elements, n_elm_gp, ndim, n_nodes)
        dn_dx = elm_geom.inv_jacobian_matrix @ np.asarray(
            self.shape_function_derivative_gp
        )

        mass = np.einsum("gi,eg,gj->eij", basis, dv0, basis)
        proj = np.einsum(
            "gi,eidn->egdn",
            basis,
            np.linalg.solve(
                mass,
                np.einsum("gi,eg,egdn->eidn", basis, dv, dn_dx).reshape(
                    *mass.shape[:2], -1
                ),
            ).reshape(*mass.shape[:2], *dn_dx.shape[2:]),
        )
        if assembly is not None and "_MeanDilatation_dV0" in assembly.sv:
            # dilatation theta = P0(J) at gauss points
            theta = np.einsum(
                "gi,ei->eg",
                basis,
                np.linalg.solve(mass, np.einsum("gi,eg->ei", basis, dv)[..., None])[
                    ..., 0
                ],
            )
            proj /= theta[:, :, None, None]

        # The assembly computes inv_jacobian_matrix @ shape_function_derivative_gp
        self.shape_function_derivative_gp = elm_geom.jacobian_matrix @ (proj - dn_dx)


def _add_mean_dilatation_element(base_elm, pressure_degree):
    base = get_element(base_elm)
    sub_element = type(
        base.__name__ + "MeanDilatation",
        (_MeanDilatationMixin, base),
        {"name": base.name + "_md", "pressure_degree": pressure_degree},
    )
    element = CombinedElement(base.name + "md", base)
    for variable in ("_DispX", "_DispY", "_DispZ"):
        element.set_variable_interpolation(variable, sub_element)
    return element


# constant pressure for linear elements (Q1/P0) and simplices (P2/P0),
# linear discontinuous pressure for the other quadratic elements (Q2/P1)
for _elm, _degree in [
    ("quad4", 0),
    ("tri6", 0),
    ("hex8", 0),
    ("wed6", 0),
    ("tet10", 0),
    ("quad8", 1),
    ("quad9", 1),
    ("hex20", 1),
    ("wed15", 1),
    ("wed18", 1),
]:
    _add_mean_dilatation_element(_elm, _degree)
