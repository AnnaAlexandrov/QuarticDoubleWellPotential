"""Функциональный решатель одномерной задачи на собственные значения.

Здесь нет пользовательских классов и методов. Расчёт устроен как цепочка
обычных функций:

1. ``prepare_problem`` вычисляет сетку, потенциал и точку сшивки;
2. ``mismatch`` строит две Numerov-ветви и возвращает их вронскиан;
3. ``find_eigenvalues`` ищет нули mismatch-функции;
4. ``build_eigenstate`` сшивает и нормирует одну волновую функцию;
5. ``solve`` объединяет поиск энергий и построение состояний.

Все необходимые данные передаются явно через аргументы. Подготовленная задача
и отдельные состояния представлены обычными словарями, которые после создания
только читаются.
"""

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .numerov import (
    Grid,
    derivative_from_left,
    derivative_from_right,
    integrate_from_left,
    integrate_from_right,
)


FloatArray = NDArray[np.float64]
Potential = Callable[[ArrayLike], ArrayLike]
Problem = dict[str, object]
Eigenstate = dict[str, object]


def suggest_match_index(
    x: FloatArray,
    potential_values: FloatArray,
    x_min: float,
    x_max: float,
) -> int:
    """Предложить индекс точки сшивки по форме потенциала.

    Для двойной ямы функция находит два наиболее глубоких локальных минимума и
    выбирает максимум потенциала между ними. Если две ямы распознать не удалось,
    используется ближайший к геометрическому центру узел.
    """

    minima = np.where(
        (potential_values[1:-1] < potential_values[:-2])
        & (potential_values[1:-1] < potential_values[2:])
    )[0] + 1

    if len(minima) >= 2:
        two_lowest = minima[np.argsort(potential_values[minima])[:2]]
        left, right = np.sort(two_lowest)

        if right - left >= 2:
            return int(
                left
                + np.argmax(potential_values[left : right + 1])
            )

    centre = 0.5 * (float(x_min) + float(x_max))
    return int(np.argmin(np.abs(x - centre)))


def prepare_problem(
    potential: Potential,
    grid: Grid,
    *,
    match_x: float | None = None,
    seed: float = 1.0e-12,
) -> Problem:
    """Вычислить все данные, которые не зависят от пробной энергии.

    Возвращаемый словарь содержит массив координат ``x``, шаг ``h``, значения
    потенциала ``v``, индекс и координату точки сшивки, а также seed для запуска
    рекурсии Нумерова. Он заменяет объект решателя из объектно-ориентированной
    версии проекта.
    """

    x = np.asarray(grid["x"], dtype=float)
    h = float(grid["step"])
    seed = float(seed)

    values = np.asarray(potential(x), dtype=float)
    if values.shape != x.shape:
        try:
            values = np.broadcast_to(values, x.shape).astype(float)
        except ValueError as exc:
            raise ValueError(
                "Potential callable returned an incompatible shape"
            ) from exc


    if match_x is None:
        match_index = suggest_match_index(
            x,
            values,
            float(grid["x_min"]),
            float(grid["x_max"]),
        )
    else:
        match_index = int(np.argmin(np.abs(x - float(match_x))))

    return {
        "potential": potential,
        "grid": grid,
        "x": x,
        "h": h,
        "v": values,
        "seed": seed,
        "match_index": match_index,
        "match_x": float(x[match_index]),
    }


def build_branches(
    problem: Problem,
    energy: float,
) -> tuple[FloatArray, FloatArray, float, float]:
    """Построить левую и правую ветви для одной пробной энергии.

    Уравнение Шрёдингера переписывается как

        psi'' + k(x;E) psi = 0,
        k(x;E) = 2(E - V(x)).

    После интегрирования возвращаются обе ветви и две односторонние производные
    в точке сшивки.
    """

    v = np.asarray(problem["v"], dtype=float)
    h = float(problem["h"])
    match_index = int(problem["match_index"])
    seed = float(problem["seed"])

    k = 2.0 * (float(energy) - v)

    left = integrate_from_left(
        k,
        h,
        match_index,
        seed=seed,
    )
    right = integrate_from_right(
        k,
        h,
        match_index,
        seed=seed,
    )

    d_left = derivative_from_left(left, match_index, h)
    d_right = derivative_from_right(right, match_index, h)
    return left, right, d_left, d_right


def mismatch(problem: Problem, energy: float) -> float:
    """Вычислить нормированный вронскиан левой и правой ветвей.

    При собственной энергии две ветви являются частями одного решения, поэтому
    их вронскиан равен нулю. Нормировка на локальные нормы ``(psi, psi')``
    устраняет зависимость от произвольного масштаба seed и не создаёт
    сингулярности в узле волновой функции.
    """

    left, right, d_left, d_right = build_branches(problem, energy)
    match_index = int(problem["match_index"])

    raw = (
        d_left * right[match_index]
        - d_right * left[match_index]
    )
    scale_left = np.hypot(left[match_index], d_left)
    scale_right = np.hypot(right[match_index], d_right)
    denominator = scale_left * scale_right

    if denominator == 0.0 or not np.isfinite(denominator):
        return float("nan")

    return float(raw / denominator)


def bisect_root(
    function: Callable[[float], float],
    a: float,
    b: float,
    *,
    energy_tolerance: float,
    function_tolerance: float,
    max_iterations: int = 200,
) -> float:
    """Уточнить корень скалярной функции методом бисекции."""

    fa = function(a)
    fb = function(b)

    left = float(a)
    right = float(b)
    f_left = float(fa)

    for _ in range(max_iterations):
        middle = 0.5 * (left + right)
        f_middle = function(middle)

        if (
            abs(f_middle) <= function_tolerance
            or 0.5 * (right - left) <= energy_tolerance
        ):
            return float(middle)

        if f_left * f_middle <= 0.0:
            right = middle
        else:
            left = middle
            f_left = float(f_middle)

    return float(0.5 * (left + right))


def fd_sturm_count(problem: Problem, energy: float) -> int:
    """Оценить число конечно-разностных уровней ниже данной энергии.

    Используется последовательность Штурма для трёхдиагонального гамильтониана
    второго порядка. Эта оценка только разделяет близкие уровни; финальные
    энергии всё равно уточняются по Numerov mismatch.
    """

    h = float(problem["h"])
    v = np.asarray(problem["v"], dtype=float)

    diagonal = (1.0 / h**2) + v[1:-1]
    off_diagonal_squared = 1.0 / (4.0 * h**4)
    tiny = 1.0e-14

    pivot = float(diagonal[0] - energy)
    if abs(pivot) < tiny:
        pivot = -tiny
    count = 1 if pivot < 0.0 else 0

    for value in diagonal[1:]:
        pivot = float(value - energy - off_diagonal_squared / pivot)
        if abs(pivot) < tiny:
            pivot = -tiny
        if pivot < 0.0:
            count += 1

    return count


def fd_eigenvalue_estimates(
    problem: Problem,
    energy_min: float,
    energy_max: float,
    n_states: int,
    *,
    tolerance: float = 1.0e-9,
) -> FloatArray:
    """Получить грубые оценки уровней через счётчик Штурма."""

    below_window = fd_sturm_count(problem, energy_min)
    available = (
        fd_sturm_count(problem, energy_max) - below_window
    )

    if available < n_states:
        raise RuntimeError(
            f"The requested window contains only about {available} bound states. "
            "Increase energy_max."
        )

    estimates: list[float] = []

    for target in range(below_window, below_window + n_states):
        left = float(energy_min)
        right = float(energy_max)

        for _ in range(100):
            middle = 0.5 * (left + right)

            if fd_sturm_count(problem, middle) <= target:
                left = middle
            else:
                right = middle

            if right - left <= tolerance:
                break

        estimates.append(0.5 * (left + right))

    return np.asarray(estimates, dtype=float)


def root_near_estimate(
    problem: Problem,
    lower: float,
    upper: float,
    estimate: float,
    *,
    energy_tolerance: float,
    function_tolerance: float,
    local_scan_points: int = 24,
) -> float:
    """Найти Numerov-корень около одной конечно-разностной оценки."""

    mismatch_function = lambda energy: mismatch(problem, energy)
    f_lower = mismatch_function(lower)
    f_upper = mismatch_function(upper)

    if np.isfinite(f_lower) and abs(f_lower) <= function_tolerance:
        return float(lower)
    if np.isfinite(f_upper) and abs(f_upper) <= function_tolerance:
        return float(upper)

    if (
        np.isfinite(f_lower)
        and np.isfinite(f_upper)
        and f_lower * f_upper < 0.0
    ):
        return bisect_root(
            mismatch_function,
            lower,
            upper,
            energy_tolerance=energy_tolerance,
            function_tolerance=function_tolerance,
        )

    mesh = np.linspace(lower, upper, local_scan_points, dtype=float)
    values = np.array(
        [mismatch_function(energy) for energy in mesh],
        dtype=float,
    )
    candidates: list[tuple[float, float, float]] = []

    for i in range(len(mesh) - 1):
        fa = values[i]
        fb = values[i + 1]

        if not np.isfinite(fa) or not np.isfinite(fb):
            continue
        if abs(fa) <= function_tolerance:
            return float(mesh[i])
        if fa * fb < 0.0:
            centre = 0.5 * (mesh[i] + mesh[i + 1])
            candidates.append(
                (abs(centre - estimate), mesh[i], mesh[i + 1])
            )

    if not candidates:
        raise RuntimeError(
            "Could not bracket a Numerov mismatch root near the finite-difference "
            f"estimate E≈{estimate:.12g}. Increase grid resolution or adjust the "
            "energy window."
        )

    _, a, b = min(candidates, key=lambda item: item[0])
    return bisect_root(
        mismatch_function,
        float(a),
        float(b),
        energy_tolerance=energy_tolerance,
        function_tolerance=function_tolerance,
    )


def scan_mismatch(
    problem: Problem,
    energy_min: float,
    energy_max: float,
    *,
    scan_points: int = 1200,
) -> tuple[FloatArray, FloatArray]:
    """Вычислить mismatch на равномерной энергетической сетке."""

    if energy_max <= energy_min:
        raise ValueError("energy_max must be greater than energy_min")
    if scan_points < 20:
        raise ValueError("scan_points must be at least 20")

    energies = np.linspace(
        energy_min,
        energy_max,
        scan_points,
        dtype=float,
    )
    values = np.array(
        [mismatch(problem, energy) for energy in energies],
        dtype=float,
    )
    return energies, values


def find_eigenvalues(
    problem: Problem,
    energy_min: float,
    energy_max: float,
    *,
    n_states: int | None = None,
    scan_points: int = 1200,
    energy_tolerance: float = 1.0e-11,
    function_tolerance: float = 1.0e-10,
) -> FloatArray:
    """Найти собственные энергии как нули Numerov mismatch-функции.

    При заданном ``n_states`` сначала используются оценки Штурма. Это защищает
    поиск от пропуска близких туннельных дублетов. При ``n_states=None``
    выполняется общий энергетический scan и извлекаются все смены знака.
    """

    if energy_max <= energy_min:
        raise ValueError("energy_max must be greater than energy_min")

    if n_states is not None:
        if n_states < 1:
            raise ValueError("n_states must be positive")

        estimates = fd_eigenvalue_estimates(
            problem,
            energy_min,
            energy_max,
            n_states,
        )

        boundaries = np.empty(n_states + 1, dtype=float)
        boundaries[0] = energy_min
        boundaries[-1] = energy_max
        if n_states > 1:
            boundaries[1:-1] = 0.5 * (
                estimates[:-1] + estimates[1:]
            )

        roots = [
            root_near_estimate(
                problem,
                float(boundaries[i]),
                float(boundaries[i + 1]),
                float(estimates[i]),
                energy_tolerance=energy_tolerance,
                function_tolerance=function_tolerance,
            )
            for i in range(n_states)
        ]
        return np.asarray(roots, dtype=float)

    energies, values = scan_mismatch(
        problem,
        energy_min,
        energy_max,
        scan_points=scan_points,
    )

    mismatch_function = lambda energy: mismatch(problem, energy)
    roots: list[float] = []

    for i in range(len(energies) - 1):
        a = float(energies[i])
        b = float(energies[i + 1])
        fa = float(values[i])
        fb = float(values[i + 1])

        if not np.isfinite(fa) or not np.isfinite(fb):
            continue

        if abs(fa) <= function_tolerance:
            root = a
        elif fa * fb < 0.0:
            root = bisect_root(
                mismatch_function,
                a,
                b,
                energy_tolerance=energy_tolerance,
                function_tolerance=function_tolerance,
            )
        else:
            continue

        if not roots or abs(root - roots[-1]) > 10.0 * energy_tolerance:
            roots.append(root)

    return np.asarray(roots, dtype=float)


def build_eigenstate(
    problem: Problem,
    energy: float,
    *,
    index: int = 0,
) -> Eigenstate:
    """Сшить и нормировать волновую функцию для найденной энергии.

    Результат — обычный словарь с ключами ``index``, ``energy``, ``x``, ``psi``
    и ``residual``. Плотность вероятности вычисляется отдельной функцией
    ``state_probability_density``.
    """

    x = np.asarray(problem["x"], dtype=float)
    match_index = int(problem["match_index"])
    left, right, d_left, d_right = build_branches(problem, energy)

    denominator = right[match_index] ** 2 + d_right**2
    if denominator == 0.0:
        raise FloatingPointError(
            "Right branch vanished at the matching point"
        )

    scale = (
        left[match_index] * right[match_index]
        + d_left * d_right
    ) / denominator
    right_scaled = scale * right

    psi = np.empty_like(x)
    psi[: match_index + 1] = left[: match_index + 1]
    psi[match_index + 1 :] = right_scaled[match_index + 1 :]
    psi[match_index] = 0.5 * (
        left[match_index] + right_scaled[match_index]
    )

    integrate = getattr(np, "trapezoid", np.trapz)
    norm2 = float(integrate(psi**2, x))

    if norm2 <= 0.0 or not np.isfinite(norm2):
        raise FloatingPointError(
            "Could not normalize the matched wave function"
        )

    psi = psi / np.sqrt(norm2)

    peak = int(np.argmax(np.abs(psi)))
    if psi[peak] < 0.0:
        psi = -psi

    return {
        "index": int(index),
        "energy": float(energy),
        "x": x.copy(),
        "psi": psi,
        "residual": float(abs(mismatch(problem, energy))),
    }


def state_probability_density(state: Eigenstate) -> FloatArray:
    """Вернуть плотность вероятности ``|psi(x)|²`` для словаря состояния."""

    psi = np.asarray(state["psi"], dtype=float)
    return psi**2


def solve(
    problem: Problem,
    energy_min: float,
    energy_max: float,
    *,
    n_states: int,
    scan_points: int = 1200,
    energy_tolerance: float = 1.0e-11,
    function_tolerance: float = 1.0e-10,
) -> list[Eigenstate]:
    """Выполнить полный расчёт энергий и нормированных состояний."""

    eigenvalues = find_eigenvalues(
        problem,
        energy_min,
        energy_max,
        n_states=n_states,
        scan_points=scan_points,
        energy_tolerance=energy_tolerance,
        function_tolerance=function_tolerance,
    )

    return [
        build_eigenstate(problem, energy, index=index)
        for index, energy in enumerate(eigenvalues)
    ]
