"""Вспомогательные функции для проверки численной сходимости.

Даже очень малый residual в точке сшивки не гарантирует, что расчётная сетка
достаточно мелкая, а границы достаточно удалены. Поэтому спектр следует
пересчитывать при изменении:

* числа пространственных узлов при фиксированной области;
* размера области при примерно фиксированном шаге.

Функции этого модуля автоматизируют такие проверки и сохраняют результаты в
удобном табличном формате.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np

from .numerov import UniformGrid
from .solver import MatchedNumerovSolver, Potential


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

    Границы области остаются неизменными, а число узлов меняется. Поэтому
    изменяется шаг

        dx = (x_max-x_min)/(N-1).

    Если метод сошёлся по сетке, энергии при увеличении `N` должны
    стабилизироваться.

    Returns
    -------
    list[dict[str, float]]
        Одна строка-словарь на каждый размер сетки. В ней записаны `n_points`,
        `dx` и найденные энергии `E_0`, `E_1`, ...
    """

    rows: list[dict[str, float]] = []

    for n_points in point_counts:
        # Каждый размер сетки требует отдельного объекта solver, потому что
        # массивы x, V(x) и шаг h являются его внутренним состоянием.
        solver = MatchedNumerovSolver(
            potential,
            UniformGrid(x_min, x_max, int(n_points)),
            match_x=match_x,
        )

        energies = solver.find_eigenvalues(
            energy_min,
            energy_max,
            n_states=n_states,
            scan_points=scan_points,
        )

        row: dict[str, float] = {
            "n_points": float(n_points),
            "dx": solver.h,
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
    """Пересчитать спектр при последовательном удалении границ.

    Для каждой полуширины `L` используется область `[-L,L]`. Число узлов
    подбирается так, чтобы шаг оставался как можно ближе к `target_step`.
    Благодаря этому изменение результата в основном отражает влияние границ,
    а не одновременное изменение дискретизации.

    `potential_factory` — функция без аргументов, создающая callable-потенциал.
    Фабрика удобна для общего интерфейса и гарантирует, что каждый независимый
    solver получает собственный объект потенциала.
    """

    rows: list[dict[str, float]] = []

    for half_width in half_widths:
        # Из желаемого шага оцениваем число интервалов, затем добавляем один
        # узел, поскольку N узлов образуют N-1 интервалов.
        n_points = int(round(2.0 * half_width / target_step)) + 1

        # Для симметричной области удобно нечётное N: тогда x=0 является
        # точным узлом сетки, а не серединой между узлами.
        if n_points % 2 == 0:
            n_points += 1

        solver = MatchedNumerovSolver(
            potential_factory(),
            UniformGrid(-half_width, half_width, n_points),
            match_x=match_x,
        )

        energies = solver.find_eigenvalues(
            energy_min,
            energy_max,
            n_states=n_states,
            scan_points=scan_points,
        )

        row: dict[str, float] = {
            "half_width": float(half_width),
            "n_points": float(n_points),
            "dx": solver.h,
        }
        row.update({f"E_{i}": float(e) for i, e in enumerate(energies)})
        rows.append(row)

    return rows


def rows_to_csv(rows: list[dict[str, float]], path: str) -> None:
    """Сохранить список однотипных словарей в CSV без зависимости от pandas.

    Ключи первого словаря задают порядок столбцов. Предполагается, что все
    последующие словари имеют те же ключи; именно такой формат создают функции
    `grid_convergence` и `domain_convergence`.
    """

    if not rows:
        raise ValueError("No rows to write")

    keys = list(rows[0].keys())

    # Преобразуем список словарей в обычную числовую матрицу NumPy.
    matrix = np.array(
        [[row[key] for key in keys] for row in rows],
        dtype=float,
    )

    # `comments=""` не добавляет символ `#` перед строкой заголовка, поэтому
    # полученный файл корректно открывается обычными табличными программами.
    np.savetxt(
        path,
        matrix,
        delimiter=",",
        header=",".join(keys),
        comments="",
    )
