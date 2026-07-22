"""Функции для проверки сходимости функционального Numerov-решателя."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np

from .numerov import make_uniform_grid
from .solver import Potential, find_eigenvalues, prepare_problem


def grid_convergence(
    potential: Potential,
    *,
    x_min: float,
    x_max: float,
    point_counts: Iterable[int],
    energy_min: float,
    energy_max: float,
    n_states: int,
    match_x: float | None = None,
    scan_points: int = 1200,
) -> list[dict[str, float]]:
    """Пересчитать спектр для нескольких пространственных шагов.

    Для каждого числа узлов создаётся новый словарь сетки, затем новый словарь
    подготовленной задачи. Никакой общий изменяемый решатель не переиспользуется.
    """

    rows: list[dict[str, float]] = []

    for n_points in point_counts:
        grid = make_uniform_grid(x_min, x_max, int(n_points))
        problem = prepare_problem(
            potential,
            grid,
            match_x=match_x,
        )
        energies = find_eigenvalues(
            problem,
            energy_min,
            energy_max,
            n_states=n_states,
            scan_points=scan_points,
        )

        row: dict[str, float] = {
            "n_points": float(n_points),
            "dx": float(grid["step"]),
        }
        row.update({f"E_{i}": float(e) for i, e in enumerate(energies)})
        rows.append(row)

    return rows


def domain_convergence(
    potential_factory: Callable[[], Potential],
    *,
    half_widths: Iterable[float],
    target_step: float,
    energy_min: float,
    energy_max: float,
    n_states: int,
    match_x: float | None = None,
    scan_points: int = 1200,
) -> list[dict[str, float]]:
    """Пересчитать спектр при последовательном удалении границ области."""

    rows: list[dict[str, float]] = []

    for half_width in half_widths:
        n_points = int(round(2.0 * half_width / target_step)) + 1
        if n_points % 2 == 0:
            n_points += 1

        grid = make_uniform_grid(-half_width, half_width, n_points)
        problem = prepare_problem(
            potential_factory(),
            grid,
            match_x=match_x,
        )
        energies = find_eigenvalues(
            problem,
            energy_min,
            energy_max,
            n_states=n_states,
            scan_points=scan_points,
        )

        row: dict[str, float] = {
            "half_width": float(half_width),
            "n_points": float(n_points),
            "dx": float(grid["step"]),
        }
        row.update({f"E_{i}": float(e) for i, e in enumerate(energies)})
        rows.append(row)

    return rows


def rows_to_csv(rows: list[dict[str, float]], path: str) -> None:
    """Сохранить список числовых словарей в CSV без pandas."""

    if not rows:
        raise ValueError("No rows to write")

    keys = list(rows[0].keys())
    matrix = np.array(
        [[row[key] for key in keys] for row in rows],
        dtype=float,
    )

    np.savetxt(
        path,
        matrix,
        delimiter=",",
        header=",".join(keys),
        comments="",
    )
