from types import SimpleNamespace

import numpy as np

from fedoo.constitutivelaw import CohesiveLaw


def _matrix_at_material_point(matrix):
    return np.array(
        [[np.asarray(value).reshape(-1)[0] for value in row] for row in matrix],
        dtype=float,
    )


def _evaluate_traction_and_tangent(law, delta, irreversible=0.0):
    """Evaluate one material point with a fixed committed damage state."""
    assembly = SimpleNamespace(
        sv={
            "DamageVariable": irreversible,
            "DamageVariableOpening": irreversible,
            "DamageVariableIrreversible": irreversible,
        }
    )
    delta_arrays = [np.array([component], dtype=float) for component in delta]
    damage_gradient = law._update_damage(assembly, delta_arrays)

    secant = _matrix_at_material_point(law.get_secant_matrix(assembly))
    tangent = _matrix_at_material_point(
        law.get_tangent_matrix(assembly, delta_arrays, damage_gradient)
    )
    traction = secant @ np.asarray(delta, dtype=float)
    return traction, tangent


def _finite_difference_tangent(law, delta, irreversible=0.0, step=1.0e-8):
    delta = np.asarray(delta, dtype=float)
    tangent = np.empty((3, 3))
    for component in range(3):
        perturbation = np.zeros(3)
        perturbation[component] = step
        traction_plus, _ = _evaluate_traction_and_tangent(
            law, delta + perturbation, irreversible
        )
        traction_minus, _ = _evaluate_traction_and_tangent(
            law, delta - perturbation, irreversible
        )
        tangent[:, component] = (traction_plus - traction_minus) / (2.0 * step)
    return tangent


def test_cohesive_law_zero_displacement_update():
    law = CohesiveLaw(KI=1.0e4, KII=2.0e4)
    assembly = SimpleNamespace(sv={})
    problem = SimpleNamespace(get_dof_solution=lambda: 0)

    law.initialize(assembly, problem)
    law.update(assembly, problem)

    assert assembly.sv["InterfaceStress"] == 0
    assert assembly.sv["RelativeDisp"] == 0
    assert np.diag(assembly.sv["TangentMatrix"]).tolist() == [
        2.0e4,
        2.0e4,
        1.0e4,
    ]


def test_cohesive_law_update_uses_current_assembly_local_frame():
    law = CohesiveLaw(KI=1.0e4, KII=2.0e4)
    delta = [
        np.array([0.001]),
        np.array([0.002]),
        np.array([0.003]),
    ]
    operators = [object(), object(), object()]
    current = SimpleNamespace(
        space=SimpleNamespace(op_disp=lambda: operators),
        get_gp_results=lambda operator, displacement: delta[operators.index(operator)],
    )
    assembly = SimpleNamespace(
        sv={
            "DamageVariable": 0,
            "DamageVariableOpening": 0,
            "DamageVariableIrreversible": 0,
        },
        current=current,
    )
    problem = SimpleNamespace(get_dof_solution=lambda: np.ones(1))

    law.update(assembly, problem)

    assert all(
        computed is expected
        for computed, expected in zip(assembly.sv["RelativeDisp"], delta)
    )
    assert np.allclose(
        np.asarray(assembly.sv["InterfaceStress"]).reshape(3),
        [20.0, 40.0, 30.0],
    )


def test_cohesive_law_mode_i_damage_and_compression_contact():
    law = CohesiveLaw(GIc=0.3, SImax=60.0, KI=1.0e4, axis=2)
    assembly = SimpleNamespace(
        sv={
            "DamageVariable": 0,
            "DamageVariableOpening": 0,
            "DamageVariableIrreversible": 0,
        }
    )

    # Damage starts at delta_0 = SImax / KI = 0.006 and reaches one at
    # delta_m = 2 GIc / SImax = 0.01.
    law._update_damage(
        assembly,
        [np.zeros(3), np.zeros(3), np.array([0.0, 0.008, 0.011])],
    )

    assert np.allclose(assembly.sv["DamageVariable"], [0.0, 0.625, 1.0])
    assert np.allclose(
        assembly.sv["DamageVariableOpening"], assembly.sv["DamageVariable"]
    )
    fully_open_stiffness = law.get_tangent_matrix(assembly)
    assert np.allclose(
        [fully_open_stiffness[i][i][2] for i in range(3)],
        0.0,
    )

    law.update_irreversible_damage(assembly)
    law._update_damage(
        assembly,
        [np.zeros(3), np.zeros(3), np.array([-0.001, -0.001, -0.001])],
    )

    # Damage is irreversible, while normal stiffness is restored in
    # compression to provide the cohesive law's soft-contact response.
    assert np.allclose(assembly.sv["DamageVariable"], [0.0, 0.625, 1.0])
    assert np.allclose(assembly.sv["DamageVariableOpening"], 0.0)
    assert np.allclose(
        law.get_tangent_matrix(assembly)[2][2],
        np.full(3, 1.0e4),
    )
    closed_stiffness = law.get_tangent_matrix(assembly)
    assert np.allclose(
        [closed_stiffness[i][i][2] for i in range(3)],
        [0.0, 0.0, 1.0e4],
    )


def test_cohesive_law_consistent_tangent_in_mixed_mode():
    law = CohesiveLaw(tangent_mode="consistent")
    delta = np.array([0.003, 0.002, 0.007])

    _, tangent = _evaluate_traction_and_tangent(law, delta)
    finite_difference = _finite_difference_tangent(law, delta)

    assert np.allclose(tangent, finite_difference, rtol=2.0e-6, atol=2.0e-4)


def test_cohesive_law_consistent_tangent_in_pure_mode_i():
    law = CohesiveLaw(tangent_mode="consistent")
    delta = np.array([0.0, 0.0, 0.008])

    _, tangent = _evaluate_traction_and_tangent(law, delta)
    finite_difference = _finite_difference_tangent(law, delta)

    # The tangential norm has a cusp at exactly zero; its centred finite
    # difference converges one-sidedly and therefore needs a slightly looser
    # tolerance than the smooth mixed-mode point.
    assert np.allclose(tangent, finite_difference, rtol=1.0e-5, atol=2.0e-4)
    assert np.isclose(tangent[2, 2], -1.5e4)


def test_cohesive_law_consistent_tangent_in_mode_ii_compression():
    law = CohesiveLaw(tangent_mode="consistent")
    delta = np.array([0.008, 0.003, -0.001])

    _, tangent = _evaluate_traction_and_tangent(law, delta)
    finite_difference = _finite_difference_tangent(law, delta)

    assert np.allclose(tangent, finite_difference, rtol=2.0e-6, atol=2.0e-4)
    assert tangent[2, 2] == law.parameters["KI"]


def test_cohesive_law_consistent_tangent_uses_secant_during_unloading():
    law = CohesiveLaw(tangent_mode="consistent")
    delta = np.array([0.0, 0.0, 0.007])

    _, tangent = _evaluate_traction_and_tangent(law, delta, irreversible=0.6)
    finite_difference = _finite_difference_tangent(law, delta, irreversible=0.6)

    assert np.allclose(tangent, finite_difference)
    assert np.allclose(np.diag(tangent), [2.0e4, 2.0e4, 4.0e3])


def test_cohesive_law_commits_damage_with_secant_predictor():
    law = CohesiveLaw(tangent_mode="consistent")
    assembly = SimpleNamespace(
        sv={
            "DamageVariable": np.array([0.625]),
            "DamageVariableOpening": np.array([0.625]),
            "DamageVariableIrreversible": 0.0,
            "TangentMatrix": [[0.0, 0.0, 0.0]] * 3,
        }
    )

    law.set_start(assembly, None)

    assert np.allclose(
        np.diag(_matrix_at_material_point(assembly.sv["TangentMatrix"])),
        [1.875e4, 1.875e4, 3.75e3],
    )
    assert np.allclose(assembly.sv["DamageVariableIrreversible"], np.array([0.625]))


def test_cohesive_law_secant_mode_and_validation():
    assert CohesiveLaw().tangent_mode == "secant"

    law = CohesiveLaw(tangent_mode="secant")
    delta = np.array([0.003, 0.002, 0.007])
    _, tangent = _evaluate_traction_and_tangent(law, delta)

    _, consistent_tangent = _evaluate_traction_and_tangent(
        CohesiveLaw(tangent_mode="consistent"), delta
    )
    assert np.allclose(tangent, np.diag(np.diag(tangent)))
    assert not np.allclose(tangent, consistent_tangent)

    with np.testing.assert_raises_regex(ValueError, "tangent_mode"):
        CohesiveLaw(tangent_mode="invalid")


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
