"""Пример использования решателя с произвольным callable-потенциалом.

Здесь по-прежнему применяется готовая формула двойной ямы, но параметры
фиксируются через `functools.partial`. Точно так же можно передать собственную
функцию `def potential(x): ...` без каких-либо изменений в решателе.
"""

from functools import partial

from double_well_solver import MatchedNumerovSolver, UniformGrid
from double_well_solver.potentials import quartic_double_well


# `partial` заранее подставляет v0 и alpha. Полученный объект принимает только
# один обязательный аргумент x и соответствует интерфейсу Potential.
potential = partial(
    quartic_double_well,
    v0=6.0,
    alpha=0.35,
)

# Создаём решатель на симметричной области. Точка сшивки будет выбрана
# автоматически около центрального барьера.
solver = MatchedNumerovSolver(
    potential,
    UniformGrid(-4.0, 4.0, 2401),
)

# Ищем шесть первых уровней в заданном энергетическом окне и сразу строим
# нормированные волновые функции.
states = solver.solve(
    energy_min=-0.5,
    energy_max=25.0,
    n_states=6,
)

# Печатаем номер уровня, энергию и остаточное рассогласование в точке сшивки.
for state in states:
    print(state.index, state.energy, state.residual)
