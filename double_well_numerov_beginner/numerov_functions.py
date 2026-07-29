"""Простые функции для решения одномерного уравнения Шрёдингера.

В этом файле специально нет классов, сложных структур данных и
автоматических оптимизаций. Все функции получают обычные числа и массивы.
"""

import numpy as np


def double_well_potential(x, v0, alpha):
    """Возвращает значения потенциала двойной ямы.

    Формула потенциала:

        V(x) = V0 * (x^2 - 1)^2 + alpha * x

    x может быть как одним числом, так и массивом NumPy.
    """

    return v0 * (x**2 - 1.0) ** 2 + alpha * x


def numerov_from_left(x, energy, potential_function, match_index, epsilon):
    """Строит левую часть волновой функции методом Нумерова.

    Мы начинаем на левой границе:

        psi[0] = 0
        psi[1] = epsilon

    Первое условие соответствует условию Дирихле. Второе маленькое число
    нужно только для запуска рекуррентной формулы.

    Расчёт идёт до точки сшивки и ещё на один узел дальше. Дополнительный
    узел нужен для вычисления производной в точке сшивки.
    """

    h = x[1] - x[0]
    potential = potential_function(x)
    k = 2.0 * (energy - potential)

    psi = np.zeros(len(x))
    psi[0] = 0.0
    psi[1] = epsilon

    # h^2 / 12 много раз встречается в формуле Нумерова,
    # поэтому считаем эту величину один раз.
    h2_div_12 = h * h / 12.0

    # В цикле известны psi[n-1] и psi[n]. По ним находим psi[n+1].
    for n in range(1, match_index + 1):
        numerator = (
            2.0 * (1.0 - 5.0 * h2_div_12 * k[n]) * psi[n]
            - (1.0 + h2_div_12 * k[n - 1]) * psi[n - 1]
        )

        denominator = 1.0 + h2_div_12 * k[n + 1]
        psi[n + 1] = numerator / denominator

    return psi


def numerov_from_right(x, energy, potential_function, match_index, epsilon):
    """Строит правую часть волновой функции методом Нумерова.

    Здесь мы начинаем с правой границы:

        psi[-1] = 0
        psi[-2] = epsilon

    Затем идём по сетке справа налево. Расчёт также продолжается на один
    узел за точку сшивки, чтобы потом можно было посчитать производную.
    """

    h = x[1] - x[0]
    potential = potential_function(x)
    k = 2.0 * (energy - potential)

    psi = np.zeros(len(x))
    psi[-1] = 0.0
    psi[-2] = epsilon

    h2_div_12 = h * h / 12.0

    # Теперь известны psi[n+1] и psi[n]. По ним находим psi[n-1].
    for n in range(len(x) - 2, match_index - 1, -1):
        numerator = (
            2.0 * (1.0 - 5.0 * h2_div_12 * k[n]) * psi[n]
            - (1.0 + h2_div_12 * k[n + 1]) * psi[n + 1]
        )

        denominator = 1.0 + h2_div_12 * k[n - 1]
        psi[n - 1] = numerator / denominator

    return psi


def derivative(psi, index, h):
    """Приближённо вычисляет первую производную.

    Используется самая обычная центральная разность:

        psi'(x_i) = (psi[i+1] - psi[i-1]) / (2h)

    Это не самая точная возможная формула, зато она проста и понятна.
    """

    return (psi[index + 1] - psi[index - 1]) / (2.0 * h)


def wronskian_mismatch(
    x,
    energy,
    potential_function,
    match_index,
    epsilon,
):
    """Вычисляет функцию невязки для пробной энергии.

    Сначала отдельно строятся решение слева и решение справа. Если энергия
    является собственной, обе части соответствуют одной волновой функции.
    Тогда их вронскиан в точке сшивки равен нулю:

        Delta(E) = psi_L' * psi_R - psi_R' * psi_L

    Использование вронскиана разрешено в примечании к заданию. Оно удобно,
    потому что для поиска энергии не нужно заранее масштабировать правую
    часть и делить на значение волновой функции.
    """

    h = x[1] - x[0]

    psi_left = numerov_from_left(
        x,
        energy,
        potential_function,
        match_index,
        epsilon,
    )

    psi_right = numerov_from_right(
        x,
        energy,
        potential_function,
        match_index,
        epsilon,
    )

    derivative_left = derivative(psi_left, match_index, h)
    derivative_right = derivative(psi_right, match_index, h)

    mismatch = (
        derivative_left * psi_right[match_index]
        - derivative_right * psi_left[match_index]
    )

    return mismatch


def bisection(
    x,
    potential_function,
    match_index,
    epsilon,
    energy_left,
    energy_right,
    tolerance,
):
    """Уточняет один корень функции невязки методом половинного деления.

    Перед вызовом функции уже известно, что на концах интервала невязка
    имеет разные знаки. Поэтому внутри интервала находится хотя бы один
    корень.
    """

    mismatch_left = wronskian_mismatch(
        x,
        energy_left,
        potential_function,
        match_index,
        epsilon,
    )

    # Ограничим число шагов, чтобы цикл точно когда-нибудь закончился.
    for step in range(100):
        energy_middle = (energy_left + energy_right) / 2.0

        mismatch_middle = wronskian_mismatch(
            x,
            energy_middle,
            potential_function,
            match_index,
            epsilon,
        )

        # Если интервал уже очень маленький, считаем середину корнем.
        if energy_right - energy_left < tolerance:
            return energy_middle

        # Выбираем ту половину интервала, где знак меняется.
        if mismatch_left * mismatch_middle <= 0.0:
            energy_right = energy_middle
        else:
            energy_left = energy_middle
            mismatch_left = mismatch_middle

    # Обычно программа выходит из цикла раньше. Эта строка оставлена на случай,
    # если 100 шагов оказалось недостаточно.
    return (energy_left + energy_right) / 2.0


def find_eigenvalues(
    x,
    potential_function,
    match_index,
    epsilon,
    energy_min,
    energy_max,
    energy_step,
    number_of_states,
    tolerance,
):
    """Ищет собственные энергии простым просмотром заданного интервала.

    Алгоритм намеренно простой:

    1. Берём энергии energy_min, energy_min + energy_step и так далее.
    2. На каждой энергии считаем функцию невязки.
    3. Если невязка поменяла знак, между двумя энергиями есть корень.
    4. Уточняем этот корень методом бисекции.

    Никакой адаптивной сетки, матричной диагонализации или метода Штурма здесь
    нет. Шаг по энергии задаётся студентом вручную в файле main.py.
    """

    eigenvalues = []

    energy_left = energy_min
    mismatch_left = wronskian_mismatch(
        x,
        energy_left,
        potential_function,
        match_index,
        epsilon,
    )

    energy_right = energy_left + energy_step

    while energy_right <= energy_max:
        mismatch_right = wronskian_mismatch(
            x,
            energy_right,
            potential_function,
            match_index,
            epsilon,
        )

        # Проверяем смену знака. np.isfinite просто не даёт использовать
        # случайные бесконечности, если параметры выбраны совсем неудачно.
        if (
            np.isfinite(mismatch_left)
            and np.isfinite(mismatch_right)
            and mismatch_left * mismatch_right < 0.0
        ):
            root = bisection(
                x,
                potential_function,
                match_index,
                epsilon,
                energy_left,
                energy_right,
                tolerance,
            )

            eigenvalues.append(root)

            # Если найдено нужное число уровней, дальше искать не нужно.
            if len(eigenvalues) == number_of_states:
                break

        energy_left = energy_right
        mismatch_left = mismatch_right
        energy_right = energy_right + energy_step

    return eigenvalues


def build_wavefunction(
    x,
    energy,
    potential_function,
    match_index,
    epsilon,
):
    """Сшивает левую и правую части волновой функции.

    Сначала обе части строятся независимо. Затем правая часть умножается на
    число lambda так, чтобы значения в точке сшивки совпали:

        lambda = psi_L(x_m) / psi_R(x_m)

    После этого из левой части берём точки до x_m, а из правой — точки после
    x_m. В конце волновая функция нормируется.
    """

    psi_left = numerov_from_left(
        x,
        energy,
        potential_function,
        match_index,
        epsilon,
    )

    psi_right = numerov_from_right(
        x,
        energy,
        potential_function,
        match_index,
        epsilon,
    )

    scale = psi_left[match_index] / psi_right[match_index]
    psi_right = scale * psi_right

    psi = np.zeros(len(x))
    psi[: match_index + 1] = psi_left[: match_index + 1]
    psi[match_index + 1 :] = psi_right[match_index + 1 :]

    # Интеграл от |psi|^2 должен быть равен единице.
    norm = np.sqrt(np.trapezoid(psi**2, x))
    psi = psi / norm

    # Знак волновой функции физически не важен. Для единообразия сделаем так,
    # чтобы самый большой по модулю пик был положительным.
    largest_point = np.argmax(np.abs(psi))
    if psi[largest_point] < 0.0:
        psi = -psi

    return psi
