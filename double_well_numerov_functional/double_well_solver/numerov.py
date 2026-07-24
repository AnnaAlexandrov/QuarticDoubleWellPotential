"""Низкоуровневые функции метода Нумерова без пользовательских классов.

Модуль решает вспомогательную задачу

    psi''(x) + k(x) psi(x) = 0

на равномерной сетке. Все данные передаются в функции явно: сетка является
обычным словарём, а интеграторы получают массив ``k``, шаг ``h`` и индекс
точки остановки. Никакого скрытого состояния между вызовами нет.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
Grid = dict[str, object]

# Красивости, в нашем коде её не будет, определяем сетку on the go
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

    if not np.isfinite(x_min) or not np.isfinite(x_max):
        raise ValueError("Grid boundaries must be finite")
    if x_max <= x_min:
        raise ValueError("x_max must be greater than x_min")
    if n_points < 11:
        raise ValueError("At least 11 grid points are required")

    x = np.linspace(x_min, x_max, n_points, dtype=float)
    step = (x_max - x_min) / (n_points - 1)

    return {
        "x_min": x_min,
        "x_max": x_max,
        "n_points": n_points,
        "x": x,
        "step": float(step),
    }

#всё-таки не нужно
def _rescale_prefix_if_needed(y: FloatArray, last: int) -> None:
    """Уменьшить уже построенную левую ветвь при угрозе переполнения.

    Эта функция технически изменяет переданный рабочий массив ``y`` на месте.
    Такое локальное изменение не создаёт состояния между вызовами: массив
    создаётся внутри интегратора и наружу передаётся только после завершения.
    Общий масштаб решения физически несущественен для линейного уравнения.
    """

    local_scale = max(abs(y[last]), abs(y[last - 1]))
    if local_scale > 1.0e100:
        y[: last + 1] /= local_scale

#всё-таки не нужно
def _rescale_suffix_if_needed(y: FloatArray, first: int) -> None:
    """Уменьшить уже построенную правую ветвь при угрозе переполнения."""

    local_scale = max(abs(y[first]), abs(y[first + 1]))
    if local_scale > 1.0e100:
        y[first:] /= local_scale


def integrate_from_left(
    k: FloatArray,
    h: float,
    stop_index: int,
    *,
    seed: float = 1.0e-12,
) -> FloatArray:
    """Построить решение от левой границы до точки сшивки.

    Задаются значения ``psi[0]=0`` и ``psi[1]=seed``. Первое реализует условие
    Дирихле, второе запускает двухшаговую рекурсию. Абсолютная величина seed не
    задаёт физическую нормировку, потому что уравнение однородно.
    """

    n = len(k)
    h = float(h)
    stop_index = int(stop_index)
    seed = float(seed)

    if not 4 <= stop_index < n:
        raise ValueError("stop_index must leave room for derivative evaluation")
    if h <= 0.0:
        raise ValueError("h must be positive")
    if seed == 0.0:
        raise ValueError("seed must be non-zero")

    y = np.zeros(n, dtype=float)
    y[0] = 0.0
    y[1] = seed
    h2_over_12 = h * h / 12.0

    for i in range(1, stop_index):
        denominator = 1.0 + h2_over_12 * k[i + 1]
        #Та ловля ошибки, которую, возможно, добавим
        if abs(denominator) < 1.0e-14:
            raise FloatingPointError(
                "Numerov denominator is nearly zero; reduce the grid step"
            )

        y[i + 1] = (
            2.0 * (1.0 - 5.0 * h2_over_12 * k[i]) * y[i]
            - (1.0 + h2_over_12 * k[i - 1]) * y[i - 1]
        ) / denominator

        _rescale_prefix_if_needed(y, i + 1)

    return y


def integrate_from_right(
    k: FloatArray,
    h: float,
    stop_index: int,
    *,
    seed: float = 1.0e-12,
) -> FloatArray:
    """Построить решение от правой границы до точки сшивки.

    Граничные значения задаются зеркально: ``psi[-1]=0``, ``psi[-2]=seed``.
    Цикл идёт справа налево и выражает очередное значение через два уже
    известных соседних значения.
    """

    n = len(k)
    h = float(h)
    stop_index = int(stop_index)
    seed = float(seed)

    if not 0 <= stop_index <= n - 5:
        raise ValueError("stop_index must leave room for derivative evaluation")
    if h <= 0.0:
        raise ValueError("h must be positive")
    if seed == 0.0:
        raise ValueError("seed must be non-zero")

    y = np.zeros(n, dtype=float)
    y[-1] = 0.0
    y[-2] = seed
    h2_over_12 = h * h / 12.0

    for i in range(n - 2, stop_index, -1):
        denominator = 1.0 + h2_over_12 * k[i - 1]
        if abs(denominator) < 1.0e-14:
            raise FloatingPointError(
                "Numerov denominator is nearly zero; reduce the grid step"
            )

        y[i - 1] = (
            2.0 * (1.0 - 5.0 * h2_over_12 * k[i]) * y[i]
            - (1.0 + h2_over_12 * k[i + 1]) * y[i + 1]
        ) / denominator

        _rescale_suffix_if_needed(y, i - 1)

    return y

#никакой правой и левой производной -- просто пользуемся формулой 5 от Фабиана
def derivative_from_left(y: FloatArray, index: int, h: float) -> float:
    """Оценить производную в точке сшивки по пяти значениям слева.

    Используется односторонняя формула четвёртого порядка точности.
    """

    index = int(index)
    h = float(h)
    if index < 4:
        raise ValueError("index must be at least 4")
    if h <= 0.0:
        raise ValueError("h must be positive")

    return float(
        (
            25.0 * y[index]
            - 48.0 * y[index - 1]
            + 36.0 * y[index - 2]
            - 16.0 * y[index - 3]
            + 3.0 * y[index - 4]
        )
        / (12.0 * h)
    )


def derivative_from_right(y: FloatArray, index: int, h: float) -> float:
    """Оценить производную в точке сшивки по пяти значениям справа."""

    index = int(index)
    h = float(h)
    if index > len(y) - 5:
        raise ValueError("index must be at least four points from the right edge")
    if h <= 0.0:
        raise ValueError("h must be positive")

    return float(
        (
            -25.0 * y[index]
            + 48.0 * y[index + 1]
            - 36.0 * y[index + 2]
            + 16.0 * y[index + 3]
            - 3.0 * y[index + 4]
        )
        / (12.0 * h)
    )
