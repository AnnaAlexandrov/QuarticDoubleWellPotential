"""Автоматические тесты основных физических и численных свойств решателя.

Тесты не доказывают корректность программы абсолютно, но быстро обнаруживают
типичные поломки после редактирования: неверный спектр, потерю нормировки или
пропуск близкого туннельного дублета.
"""

from functools import partial

import numpy as np

from double_well_solver import (
    MatchedNumerovSolver,
    UniformGrid,
    harmonic_oscillator,
    quartic_double_well,
)


def test_harmonic_oscillator_spectrum() -> None:
    """Сравнить численный спектр с точным спектром осциллятора.

    Для `omega=1` и `ℏ=m=1` известно `E_n=n+1/2`. Это наиболее прямой тест
    поиска собственных значений, поскольку правильный ответ не требует
    отдельного численного эталона.
    """

    solver = MatchedNumerovSolver(
        partial(harmonic_oscillator, omega=1.0),
        UniformGrid(-7.0, 7.0, 2801),
        match_x=0.0,
    )

    energies = solver.find_eigenvalues(
        0.1,
        4.2,
        n_states=4,
        scan_points=700,
    )
    exact = np.arange(4, dtype=float) + 0.5

    # rtol=0 отключает относительную погрешность: здесь требуется именно
    # абсолютное совпадение лучше 2e-6 для каждого из четырёх уровней.
    assert np.allclose(energies, exact, atol=2.0e-6, rtol=0.0)


def test_states_are_normalized() -> None:
    """Проверить нормировку и малый mismatch готовых состояний."""

    solver = MatchedNumerovSolver(
        partial(quartic_double_well, v0=5.0, alpha=0.2),
        UniformGrid(-4.0, 4.0, 2001),
    )
    states = solver.solve(
        -0.3,
        15.0,
        n_states=4,
        scan_points=700,
    )

    for state in states:
        # Совместимость со старыми и новыми версиями NumPy.
        integrate = getattr(np, "trapezoid", np.trapz)
        norm = integrate(state.psi**2, state.x)

        assert abs(norm - 1.0) < 2.0e-10
        assert state.residual < 2.0e-7


def test_symmetric_potential_has_nearly_degenerate_low_pair_for_high_barrier() -> None:
    """Убедиться, что алгоритм не пропускает близкий туннельный дублет.

    В высокой симметричной двойной яме основное и первое возбуждённое состояния
    являются почти вырожденной чётно-нечётной парой. Глобальный грубый scan мог
    бы пропустить один из двух близких корней, поэтому этот тест также проверяет
    работу предварительного разбиения через последовательность Штурма.
    """

    solver = MatchedNumerovSolver(
        partial(quartic_double_well, v0=20.0, alpha=0.0),
        UniformGrid(-3.5, 3.5, 2601),
    )
    energies = solver.find_eigenvalues(
        0.0,
        20.0,
        n_states=2,
        scan_points=700,
    )

    # Уровни должны быть различимы и расположены в правильном порядке, но их
    # расщепление должно оставаться малым по сравнению с типичным масштабом.
    assert energies[1] > energies[0]
    assert energies[1] - energies[0] < 0.2
