"""Низкоуровневые функции для интегрирования методом Нумерова.

Этот модуль ничего не знает о конкретном виде потенциала и о поиске
собственных энергий. Он решает более узкую задачу: по уже вычисленному массиву

    k_n = 2(E - V(x_n))

строит дискретное решение уравнения

    psi''(x) + k(x) psi(x) = 0

на равномерной одномерной сетке. Такое разделение полезно: интегратор Нумерова
можно переиспользовать с любым потенциалом и в других задачах второго порядка.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


# Короткий псевдоним для NumPy-массивов вещественных чисел двойной точности.
# Он нужен только для подсказок типов и не влияет на вычисления.
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class UniformGrid:
    """Описание равномерной пространственной сетки.

    Parameters
    ----------
    x_min, x_max:
        Левая и правая границы расчётной области.
    n_points:
        Полное число узлов, включая обе границы.

    Объект сделан неизменяемым (`frozen=True`), чтобы после создания сетки её
    параметры случайно не изменились посреди расчёта.
    """

    x_min: float
    x_max: float
    n_points: int

    def __post_init__(self) -> None:
        """Проверить корректность параметров сразу после создания объекта.

        Метод `__post_init__` автоматически вызывается `dataclass` после
        присваивания полей. Благодаря этому ошибки вроде перепутанных границ
        обнаруживаются до начала численного интегрирования.
        """

        if not np.isfinite(self.x_min) or not np.isfinite(self.x_max):
            raise ValueError("Grid boundaries must be finite")
        if self.x_max <= self.x_min:
            raise ValueError("x_max must be greater than x_min")
        if self.n_points < 11:
            # Для самой рекурсии хватило бы меньшего числа точек, но ниже
            # используются пятиузловые формулы для производной, поэтому
            # оставляем разумный минимальный запас.
            raise ValueError("At least 11 grid points are required")

    @property
    def x(self) -> FloatArray:
        """Вернуть массив координат `x_0, ..., x_{N-1}`.

        Массив создаётся через `linspace`, поэтому обе заданные границы входят
        в сетку, а расстояние между соседними узлами постоянно.
        """

        return np.linspace(self.x_min, self.x_max, self.n_points, dtype=float)

    @property
    def step(self) -> float:
        """Вернуть пространственный шаг `h = (x_max-x_min)/(N-1)`."""

        return (self.x_max - self.x_min) / (self.n_points - 1)


def _rescale_prefix_if_needed(y: FloatArray, last: int) -> None:
    """При необходимости уменьшить уже построенную левую ветвь решения.

    В классически запрещённой области численное решение может содержать
    экспоненциально растущую компоненту. Тогда значения способны достичь
    предела типа `float` задолго до точки сшивки. Поскольку уравнение линейно,
    умножение всей построенной части решения на общий коэффициент не меняет ни
    положение нулей, ни логарифмическую производную, ни условие сшивки.

    Parameters
    ----------
    y:
        Массив решения.
    last:
        Последний уже вычисленный индекс левой ветви.
    """

    local_scale = max(abs(y[last]), abs(y[last - 1]))
    if local_scale > 1.0e100:
        y[: last + 1] /= local_scale


def _rescale_suffix_if_needed(y: FloatArray, first: int) -> None:
    """При необходимости уменьшить уже построенную правую ветвь решения.

    Это зеркальный аналог `_rescale_prefix_if_needed` для интегрирования
    справа налево. Масштабируется только уже вычисленный хвост массива.
    """

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

    Решается дискретизированное уравнение

        psi'' + k(x) psi = 0

    с граничными начальными значениями

        psi[0] = 0,  psi[1] = seed.

    Первое значение реализует условие Дирихле на левой границе. Второе нужно,
    чтобы запустить двухшаговую рекурсию Нумерова. Его абсолютная величина
    несущественна, потому что линейное однородное уравнение определяет решение
    лишь с точностью до общего множителя.

    Parameters
    ----------
    k:
        Значения `k(x_n)=2(E-V(x_n))` во всех узлах сетки.
    h:
        Постоянный шаг сетки.
    stop_index:
        Индекс точки, до которой включительно требуется построить решение.
    seed:
        Малое ненулевое значение во втором узле.

    Returns
    -------
    FloatArray
        Массив длины `len(k)`. Значимая часть лежит от индекса 0 до
        `stop_index`; остальные элементы остаются нулевыми.
    """

    n = len(k)
    if not 4 <= stop_index < n:
        raise ValueError("stop_index must leave room for derivative evaluation")
    if h <= 0:
        raise ValueError("h must be positive")
    if seed == 0:
        raise ValueError("seed must be non-zero")

    y = np.zeros(n, dtype=float)
    y[0] = 0.0
    y[1] = float(seed)

    # Этот коэффициент встречается во всех членах формулы Нумерова, поэтому
    # вычисляем его один раз до цикла.
    h2_over_12 = h * h / 12.0

    for i in range(1, stop_index):
        # Из рекуррентной формулы выражаем psi_{i+1}. Коэффициент при нём
        # переносится в знаменатель.
        denominator = 1.0 + h2_over_12 * k[i + 1]
        if abs(denominator) < 1.0e-14:
            raise FloatingPointError(
                "Numerov denominator is nearly zero; reduce the grid step"
            )

        y[i + 1] = (
            2.0 * (1.0 - 5.0 * h2_over_12 * k[i]) * y[i]
            - (1.0 + h2_over_12 * k[i - 1]) * y[i - 1]
        ) / denominator

        # Защита от переполнения при экспоненциальном росте численной ветви.
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

    Граничные значения теперь задаются зеркально:

        psi[-1] = 0,  psi[-2] = seed.

    Цикл идёт справа налево, а рекуррентная формула выражает `psi[i-1]` через
    два уже известных значения `psi[i]` и `psi[i+1]`.

    Returns
    -------
    FloatArray
        Массив длины `len(k)`. Значимая часть лежит от `stop_index` до правой
        границы; элементы левее остаются нулевыми.
    """

    n = len(k)
    if not 0 <= stop_index <= n - 5:
        raise ValueError("stop_index must leave room for derivative evaluation")
    if h <= 0:
        raise ValueError("h must be positive")
    if seed == 0:
        raise ValueError("seed must be non-zero")

    y = np.zeros(n, dtype=float)
    y[-1] = 0.0
    y[-2] = float(seed)
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


def derivative_from_left(y: FloatArray, index: int, h: float) -> float:
    """Оценить производную в точке сшивки по значениям слева.

    Используется односторонняя пятиузловая формула четвёртого порядка:

        f'(x_i) ≈ [25f_i - 48f_{i-1} + 36f_{i-2}
                   -16f_{i-3} + 3f_{i-4}] / (12h).

    Она удобна тем, что для левой ветви нужны только уже вычисленные точки.
    """

    if index < 4:
        raise ValueError("index must be at least 4")
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
    """Оценить производную в точке сшивки по значениям справа.

    Используется зеркальная односторонняя пятиузловая формула четвёртого
    порядка. Для неё нужны узлы `index, ..., index+4`, то есть только уже
    вычисленная правая ветвь.
    """

    if index > len(y) - 5:
        raise ValueError("index must be at least four points from the right edge")
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
