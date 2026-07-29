"""Очень простая проверка зависимости ответа от сетки и границ.

В задании сказано проверить, что энергии слабо меняются при изменении шага
сетки и размера области. Здесь просто повторяется один и тот же расчёт для
нескольких заранее заданных вариантов.
"""

import numpy as np

from numerov_functions import double_well_potential, find_eigenvalues


V0 = 5.0
alpha = 0.25
match_x = 0.2
epsilon = 1.0e-5

energy_min = 0.0
energy_max = 10.0
energy_step = 0.05
number_of_states = 3
energy_tolerance = 1.0e-8


# Каждый элемент списка содержит:
# [левая граница, правая граница, число точек]
tests = [
    [-4.0, 4.0, 801],
    [-4.0, 4.0, 1201],
    [-5.0, 5.0, 1501],
]


for test in tests:
    x_min = test[0]
    x_max = test[1]
    number_of_points = test[2]

    x = np.linspace(x_min, x_max, number_of_points)
    match_index = np.argmin(np.abs(x - match_x))

    def potential(x_values):
        return double_well_potential(x_values, V0, alpha)

    energies = find_eigenvalues(
        x,
        potential,
        match_index,
        epsilon,
        energy_min,
        energy_max,
        energy_step,
        number_of_states,
        energy_tolerance,
    )

    h = x[1] - x[0]

    print()
    print("x_min =", x_min)
    print("x_max =", x_max)
    print("number of points =", number_of_points)
    print("dx =", h)
    print("energies =", energies)
