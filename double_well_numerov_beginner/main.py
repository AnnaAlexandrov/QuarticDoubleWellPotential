"""Главный файл проекта.

Чтобы изменить задачу, достаточно поменять числа в разделе PARAMETERS.
Никакие аргументы командной строки и конфигурационные файлы не используются.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from numerov_functions import (
    build_wavefunction,
    double_well_potential,
    find_eigenvalues,
)


# ================================================================
# PARAMETERS
# Все основные параметры собраны в одном месте.
# ================================================================

V0 = 5.0
alpha = 0.25

x_min = -4.0
x_max = 4.0
number_of_x_points = 1201

# Точку сшивки удобно взять около центрального барьера,
# но не обязательно точно в нуле.
match_x = 0.2

# Маленькое начальное значение для запуска метода Нумерова.
epsilon = 1.0e-5

# Интервал и обычный постоянный шаг для поиска энергий.
energy_min = 0.0
energy_max = 20.0
energy_step = 0.05

number_of_states = 6
energy_tolerance = 1.0e-8


# ================================================================
# PREPARATION OF THE GRID AND THE POTENTIAL
# ================================================================

# Обычная равномерная сетка. Никакого отдельного объекта сетки нет.
x = np.linspace(x_min, x_max, number_of_x_points)

# Находим номер узла, который расположен ближе всего к match_x.
match_index = np.argmin(np.abs(x - match_x))


# Эта функция является Python callable, который передаётся в решатель.
# Благодаря этому численный алгоритм не привязан к одной формуле потенциала.
def potential(x_values):
    return double_well_potential(x_values, V0, alpha)


# ================================================================
# SEARCH FOR EIGENVALUES
# ================================================================

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

print("Found energies:")
for number in range(len(energies)):
    print("E_" + str(number) + " = " + str(energies[number]))

if len(energies) < number_of_states:
    print()
    print("Not all requested states were found.")
    print("Try increasing energy_max or decreasing energy_step.")


# ================================================================
# BUILD THE WAVE FUNCTIONS
# ================================================================

wavefunctions = []

for energy in energies:
    psi = build_wavefunction(
        x,
        energy,
        potential,
        match_index,
        epsilon,
    )
    wavefunctions.append(psi)


# ================================================================
# SAVE THE NUMERICAL RESULTS
# ================================================================

os.makedirs("results", exist_ok=True)

# Сохраняем таблицу энергий.
energy_table = []
for number in range(len(energies)):
    energy_table.append([number, energies[number]])

np.savetxt(
    "results/energies.csv",
    np.array(energy_table),
    delimiter=",",
    header="state,energy",
    comments="",
)

# Сохраняем координату и все найденные волновые функции.
wavefunction_table = [x]
for psi in wavefunctions:
    wavefunction_table.append(psi)

wavefunction_table = np.array(wavefunction_table).T

header = "x"
for number in range(len(wavefunctions)):
    header = header + ",psi_" + str(number)

np.savetxt(
    "results/wavefunctions.csv",
    wavefunction_table,
    delimiter=",",
    header=header,
    comments="",
)


# ================================================================
# DRAW THE POTENTIAL AND THE WAVE FUNCTIONS
# ================================================================

potential_values = potential(x)

plt.figure(figsize=(9, 6))
plt.plot(x, potential_values, label="V(x)")

# Для наглядности каждая волновая функция уменьшается по высоте
# и поднимается на уровень соответствующей энергии.
for number in range(len(wavefunctions)):
    psi = wavefunctions[number]
    energy = energies[number]

    drawing_scale = 0.7 / np.max(np.abs(psi))
    plt.plot(
        x,
        energy + drawing_scale * psi,
        label="psi_" + str(number),
    )

    plt.axhline(energy, linewidth=0.5, linestyle="--")

plt.axvline(x[match_index], linewidth=0.8, linestyle=":", label="match point")
plt.xlim(x_min, x_max)
plt.ylim(-0.5, energy_max)
plt.xlabel("x")
plt.ylabel("Energy")
plt.title("Asymmetric double-well potential")
plt.legend(ncol=2)
plt.tight_layout()
plt.savefig("results/eigenstates.png", dpi=150)
plt.show()
