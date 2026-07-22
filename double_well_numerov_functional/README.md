# Квартичная двойная яма: функциональная реализация метода Нумерова

Проект решает безразмерное стационарное уравнение Шрёдингера

\[
-\frac12\psi''(x)+V(x)\psi(x)=E\psi(x),
\qquad
V(x)=V_0(x^2-1)^2+\alpha x.
\]

В этой версии **нет пользовательских классов, `dataclass`, методов,
конструкторов и свойств**. Численный алгоритм записан как последовательность
обычных функций. Сетка, подготовленная задача и найденные состояния передаются
между функциями как обычные словари.

## Главная идея структуры

```text
make_uniform_grid(...)
        ↓
prepare_problem(potential, grid)
        ↓
find_eigenvalues(problem, ...)
        ↓
build_eigenstate(problem, energy)
        ↓
solve(problem, ...)
```

Данные не прячутся внутри объекта решателя. Например, словарь `problem`
содержит:

```python
{
    "x": ...,             # координатная сетка
    "h": ...,             # шаг сетки
    "v": ...,             # V(x) во всех узлах
    "match_index": ...,   # индекс точки сшивки
    "match_x": ...,       # координата точки сшивки
    "seed": ...,          # стартовый масштаб Numerov-ветвей
}
```

Состояние также является словарём:

```python
{
    "index": 0,
    "energy": 2.57,
    "x": ...,
    "psi": ...,
    "residual": ...,
}
```

Плотность вероятности вычисляется отдельной функцией
`state_probability_density(state)` вместо свойства объекта.

## Установка

Из каталога проекта:

```bash
pip install -e ".[test]"
```

## Запуск готового расчёта

```bash
python run_project.py \
  --v0 5 \
  --alpha 0.25 \
  --xmax 4 \
  --points 2001 \
  --states 6 \
  --energy-max 25 \
  --output results \
  --convergence
```

То же самое после установки пакета:

```bash
python -m double_well_solver --v0 5 --alpha 0.25
```

Справка:

```bash
python run_project.py --help
```

## Использование из Python

```python
from functools import partial

from double_well_solver import (
    make_uniform_grid,
    prepare_problem,
    quartic_double_well,
    solve,
)

potential = partial(
    quartic_double_well,
    v0=5.0,
    alpha=0.25,
)

grid = make_uniform_grid(-4.0, 4.0, 2001)
problem = prepare_problem(potential, grid)

states = solve(
    problem,
    energy_min=-0.5,
    energy_max=25.0,
    n_states=6,
)

for state in states:
    print(state["index"], state["energy"], state["residual"])
```

## Как подставить собственный потенциал

```python
import numpy as np

from double_well_solver import make_uniform_grid, prepare_problem, solve


def my_potential(x):
    x = np.asarray(x, dtype=float)
    return 0.5 * x**2 + 0.02 * x**4


grid = make_uniform_grid(-7.0, 7.0, 2801)
problem = prepare_problem(my_potential, grid, match_x=0.0)
states = solve(problem, 0.0, 10.0, n_states=5)
```

## Файлы проекта

- `double_well_solver/numerov.py` — сетка, рекурсия Нумерова и производные;
- `double_well_solver/solver.py` — сшивка, mismatch, Штурм, бисекция, спектр;
- `double_well_solver/potentials.py` — готовые потенциалы;
- `double_well_solver/analysis.py` — проверки сходимости;
- `double_well_solver/cli.py` — функциональный интерфейс командной строки;
- `examples/custom_potential.py` — минимальный пример;
- `tests/test_solver.py` — автоматические тесты.

## Рекомендуемый порядок чтения

1. `potentials.py` — самые простые функции;
2. `numerov.py` — одна Numerov-рекурсия слева и справа;
3. `solver.py`: `prepare_problem`, `build_branches`, `mismatch`;
4. затем `fd_sturm_count`, `find_eigenvalues`, `build_eigenstate`, `solve`;
5. `cli.py` — только ввод, вывод и графики.

## Что сохраняет программа

В каталоге `--output` создаются:

- `spectrum.csv` — энергии и остаточное рассогласование;
- `wavefunctions.csv` — `x`, `V(x)`, `psi_n(x)` и `|psi_n(x)|²`;
- `eigenstates.png` — потенциал и состояния;
- `mismatch.png` — mismatch-функция;
- таблицы сходимости, если указан `--convergence`.

## Проверка

```bash
pytest -q
```

Тесты проверяют точный спектр гармонического осциллятора, нормировку,
сохранение близкого туннельного дублета и отсутствие пользовательских
объявлений `class`/`dataclass` в пакете.
