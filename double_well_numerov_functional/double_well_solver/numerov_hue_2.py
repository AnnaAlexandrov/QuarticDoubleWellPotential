"""Низкоуровневые функции метода Нумерова без пользовательских классов.

Модуль решает вспомогательную задачу

    psi''(x) + k(x) psi(x) = 0

на равномерной сетке. Все данные передаются в функции явно: сетка является
обычным словарём, а интеграторы получают массив ``k``, шаг ``h`` и индекс
точки остановки. Никакого скрытого состояния между вызовами нет.
"""

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
Grid = dict[str, object]


def make_uniform_grid(x_min: float, x_max: float, n_points: int) -> Grid:
    """Создать описание равномерной пространственной сетки.

    Вместо объекта класса функция возвращает обычный словарь со значениями
    ``x_min``, ``x_max``, ``n_points``, ``x`` и ``step``. В дальнейшем этот
    словарь только читается и рассматривается как неизменяемая структура
    данных.
    """

    x_min = float(x_min)
    x_max = float(x_max)
    n_points = int(n_points)

    x = np.linspace(x_min, x_max, n_points, dtype=float)
    step = (x_max - x_min) / (n_points - 1)

    return {
        "x_min": x_min,
        "x_max": x_max,
        "n_points": n_points,
        "x": x,
        "step": float(step),
    }


def _rescale_prefix_if_needed(psi: FloatArray, last: int) -> None:
    """Уменьшить уже построенную левую ветвь при угрозе переполнения.

    Эта функция технически изменяет переданный рабочий массив ``y`` на месте.
    Такое локальное изменение не создаёт состояния между вызовами: массив
    создаётся внутри интегратора и наружу передаётся только после завершения.
    Общий масштаб решения физически несущественен для линейного уравнения.
    """

    local_scale = max(abs(psi[last]), abs(psi[last - 1]))
    if local_scale > 1.0e100:
        psi[: last + 1] /= local_scale


def _rescale_suffix_if_needed(psi: FloatArray, first: int) -> None:
    """Уменьшить уже построенную правую ветвь при угрозе переполнения."""

    local_scale = max(abs(psi[first]), abs(psi[first + 1]))
    if local_scale > 1.0e100:
        psi[first:] /= local_scale


def integrate_from_left(
    k: FloatArray,
    h: float,
    i_match: int,
) -> FloatArray:
    """Построить решение от левой границы до точки сшивки и на один шаг за нее.

    Задаются значения ``psi[0]=0`` и ``psi[1]=seed``. Интегрирование
    производится до stop_index + 1, чтобы обеспечить перекрытие для
    вычисления центральной производной.
    """

    n = len(k)
    h = float(h)
    i_match = int(i_match)

    psi = np.zeros(n, dtype=float)
    psi[0] = 0
    psi[1] = 1e-5
    h2_over_12 = h**2 / 12

    # Обратите внимание: stop_index + 1 заставляет цикл сделать лишний шаг
    for i in range(1, i_match + 1):
        denominator = 1 + h2_over_12 * k[i + 1]

        psi[i + 1] = (
            2 * (1 - 5 * h2_over_12 * k[i]) * psi[i]
            - (1 + h2_over_12 * k[i - 1]) * psi[i - 1]
        ) / denominator

        _rescale_prefix_if_needed(psi, i + 1)

    return psi


def integrate_from_right(
    k: FloatArray,
    h: float,
    i_match: int,
) -> FloatArray:
    """Построить решение от правой границы до точки сшивки и на один шаг за нее.

    Граничные значения задаются зеркально: ``psi[-1]=0``, ``psi[-2]=seed``.
    Интегрирование производится до stop_index - 1, чтобы обеспечить перекрытие
    для вычисления центральной производной.
    """

    n = len(k)
    h = float(h)
    i_match = int(i_match)

    psi = np.zeros(n, dtype=float)
    psi[-1] = 0
    psi[-2] = 1e-5
    h2_over_12 = h**2 / 12

    # Обратите внимание: stop_index - 1 заставляет цикл сделать лишний шаг влево
    for i in range(n - 2, i_match - 1, -1):
        denominator = 1 + h2_over_12 * k[i - 1]

        psi[i - 1] = (
            2 * (1 - 5 * h2_over_12 * k[i]) * psi[i]
            - (1 + h2_over_12 * k[i + 1]) * psi[i + 1]
        ) / denominator

        _rescale_suffix_if_needed(psi, i - 1)

    return psi


def derivative_for_mismatch_function(
    psi: FloatArray,
    k: FloatArray,
    i_match: int,
    h: float
) -> float:
    """Оценить производную в точке сшивки по центральной разности с поправкой Нумерова.

    Используется формула O(h^4) на основе значений в соседних узлах:
    y' = (y_{n+1} - y_{n-1}) / (2h) + (h / 12) * (k_{n-1}y_{n-1} - k_{n+1}y_{n+1}).
    """

    i_match = int(i_match)
    h = float(h)

    finite_diff = (psi[i_match + 1] - psi[i_match - 1]) / (2 * h)

    numerov_correction = (h / 12) * (
        k[i_match - 1] * psi[i_match - 1] - k[i_match + 1] * psi[i_match + 1]
    )

    return float(finite_diff + numerov_correction)
