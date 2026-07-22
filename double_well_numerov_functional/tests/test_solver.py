"""Тесты функционального Numerov-решателя."""

from functools import partial

import numpy as np

from double_well_solver import (
    find_eigenvalues,
    harmonic_oscillator,
    make_uniform_grid,
    prepare_problem,
    quartic_double_well,
    solve,
)


def test_harmonic_oscillator_spectrum() -> None:
    """Сравнить четыре уровня с точным спектром E_n=n+1/2."""

    grid = make_uniform_grid(-7.0, 7.0, 2801)
    problem = prepare_problem(
        partial(harmonic_oscillator, omega=1.0),
        grid,
        match_x=0.0,
    )
    energies = find_eigenvalues(
        problem,
        0.1,
        4.2,
        n_states=4,
        scan_points=700,
    )
    exact = np.arange(4, dtype=float) + 0.5

    assert np.allclose(energies, exact, atol=2.0e-6, rtol=0.0)


def test_states_are_normalized() -> None:
    """Проверить нормировку и малый mismatch всех состояний."""

    grid = make_uniform_grid(-4.0, 4.0, 2001)
    problem = prepare_problem(
        partial(quartic_double_well, v0=5.0, alpha=0.2),
        grid,
    )
    states = solve(
        problem,
        -0.3,
        15.0,
        n_states=4,
        scan_points=700,
    )

    integrate = getattr(np, "trapezoid", np.trapz)
    for state in states:
        x = np.asarray(state["x"], dtype=float)
        psi = np.asarray(state["psi"], dtype=float)
        norm = integrate(psi**2, x)

        assert abs(norm - 1.0) < 2.0e-10
        assert float(state["residual"]) < 2.0e-7


def test_high_barrier_keeps_close_tunnelling_pair() -> None:
    """Убедиться, что не пропущен близкий туннельный дублет."""

    grid = make_uniform_grid(-3.5, 3.5, 2601)
    problem = prepare_problem(
        partial(quartic_double_well, v0=20.0, alpha=0.0),
        grid,
    )
    energies = find_eigenvalues(
        problem,
        0.0,
        20.0,
        n_states=2,
        scan_points=700,
    )

    assert energies[1] > energies[0]
    assert energies[1] - energies[0] < 0.2


def test_public_source_contains_no_user_class_definitions() -> None:
    """Зафиксировать требование проекта: никаких пользовательских class."""

    import glob
    import os

    package = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "double_well_solver",
    )
    source_parts = []
    for path in glob.glob(os.path.join(package, "*.py")):
        with open(path, encoding="utf-8") as handle:
            source_parts.append(handle.read())
    source = "\n".join(source_parts)

    assert "\nclass " not in source
    assert "\n@dataclass" not in source
