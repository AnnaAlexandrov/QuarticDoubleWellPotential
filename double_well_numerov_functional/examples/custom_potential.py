"""Пример использования проекта только через функции."""

from functools import partial
import os
import sys

# При прямом запуске файла добавляем корень проекта в путь импорта.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from double_well_solver import (
    make_uniform_grid,
    prepare_problem,
    quartic_double_well,
    solve,
)


potential = partial(
    quartic_double_well,
    v0=6.0,
    alpha=0.35,
)

grid = make_uniform_grid(-4.0, 4.0, 2401)
problem = prepare_problem(potential, grid)
states = solve(
    problem,
    energy_min=-0.5,
    energy_max=25.0,
    n_states=6,
)

for state in states:
    print(
        int(state["index"]),
        float(state["energy"]),
        float(state["residual"]),
    )
