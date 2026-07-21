"""Командный интерфейс для расчёта квартичной двойной ямы.

Модуль связывает численный решатель с пользовательскими аргументами командной
строки. Он отвечает за:

* чтение параметров потенциала и сетки;
* запуск расчёта;
* печать диагностической информации;
* сохранение спектра, волновых функций и графиков;
* необязательные исследования сходимости.

Сам численный алгоритм находится в `solver.py`, поэтому этот файл можно менять,
не затрагивая физическую и математическую часть решателя.
"""

from __future__ import annotations

import argparse
import csv
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .analysis import domain_convergence, grid_convergence, rows_to_csv
from .numerov import UniformGrid
from .potentials import quartic_double_well
from .solver import Eigenstate, MatchedNumerovSolver


def _write_spectrum(states: list[Eigenstate], path: Path) -> None:
    """Сохранить энергии и residual каждого состояния в CSV-файл.

    Функция начинается с подчёркивания, потому что является внутренней частью
    CLI и не считается публичным API библиотеки.
    """

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["state", "energy", "mismatch_residual"])

        for state in states:
            # Энергия сохраняется с высокой точностью, residual — в научной
            # записи, поскольку он обычно очень мал.
            writer.writerow(
                [
                    state.index,
                    f"{state.energy:.15g}",
                    f"{state.residual:.6e}",
                ]
            )


def _write_wavefunctions(
    states: list[Eigenstate],
    potential: np.ndarray,
    path: Path,
) -> None:
    """Сохранить сетку, потенциал, `psi_n` и `|psi_n|²` в один CSV.

    Первые два столбца всегда равны `x` и `V(x)`. Затем для каждого уровня
    добавляются два столбца: волновая функция и плотность вероятности.
    """

    header = ["x", "V"]
    columns = [states[0].x, potential]

    for state in states:
        header.extend(
            [f"psi_{state.index}", f"density_{state.index}"]
        )
        columns.extend([state.psi, state.probability_density])

    # column_stack превращает список одномерных массивов одинаковой длины в
    # матрицу, где каждый исходный массив становится отдельным столбцом.
    matrix = np.column_stack(columns)

    np.savetxt(
        path,
        matrix,
        delimiter=",",
        header=",".join(header),
        comments="",
    )


def _plot_states(
    states: list[Eigenstate],
    potential: np.ndarray,
    output_path: Path,
    *,
    y_max: float | None,
    match_x: float,
) -> None:
    """Построить потенциал, уровни энергии и сдвинутые волновые функции.

    Каждая `psi_n(x)` рисуется не около нуля, а около своего энергетического
    уровня:

        y(x) = E_n + scale * psi_n(x).

    Это стандартный способ одновременно показать форму состояния и его место
    в спектре. Вертикальная штриховая линия отмечает точку сшивки.
    """

    x = states[0].x
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(x, potential, linewidth=2.0, label="V(x)")

    # Коэффициент влияет только на визуальный размер волновых функций и не
    # имеет физического смысла.
    amplitude_scale = 0.45

    for state in states:
        ax.axhline(state.energy, linewidth=0.7, alpha=0.35)
        ax.plot(
            x,
            state.energy + amplitude_scale * state.psi,
            label=fr"$E_{state.index}={state.energy:.6f}$",
        )

    ax.axvline(
        match_x,
        linewidth=0.8,
        alpha=0.35,
        linestyle="--",
        label=fr"$x_m={match_x:.3f}$",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("Energy; shifted wave functions")
    ax.set_title("Quartic double well: matched Numerov eigenstates")
    ax.grid(alpha=0.25)

    if y_max is not None:
        ax.set_ylim(
            bottom=min(float(np.min(potential)), 0.0) - 0.5,
            top=y_max,
        )

    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_mismatch(
    solver: MatchedNumerovSolver,
    energy_min: float,
    energy_max: float,
    states: list[Eigenstate],
    output_path: Path,
    *,
    scan_points: int,
) -> None:
    """Построить график mismatch-функции по энергии.

    Нули кривой соответствуют собственным значениям. Для наглядности найденные
    энергии дополнительно отмечаются вертикальными штриховыми линиями.
    """

    energies, mismatch = solver.scan_mismatch(
        energy_min,
        energy_max,
        scan_points=scan_points,
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(energies, mismatch, linewidth=1.0)
    ax.axhline(0.0, linewidth=0.8)

    for state in states:
        ax.axvline(state.energy, linestyle="--", linewidth=0.8)

    ax.set_xlabel("E")
    ax.set_ylabel("normalized Wronskian mismatch")
    ax.set_title("Root search for the discrete spectrum")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    """Создать и настроить парсер аргументов командной строки.

    Отделение этой функции от `main` удобно для тестирования и для повторного
    использования CLI из другого Python-кода.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Solve the dimensionless 1D Schrödinger equation for "
            "V(x)=V0(x^2-1)^2+alpha*x with matched Numerov propagation."
        )
    )

    # Параметры потенциала.
    parser.add_argument(
        "--v0",
        type=float,
        default=5.0,
        help="barrier scale V0",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.0,
        help="linear asymmetry",
    )

    # Параметры пространственной дискретизации.
    parser.add_argument(
        "--xmax",
        type=float,
        default=4.0,
        help="symmetric half-domain",
    )
    parser.add_argument(
        "--points",
        type=int,
        default=2401,
        help="number of grid points",
    )
    parser.add_argument(
        "--match-x",
        type=float,
        default=None,
        help="manual matching coordinate",
    )

    # Параметры поиска спектра.
    parser.add_argument(
        "--states",
        type=int,
        default=6,
        help="number of states",
    )
    parser.add_argument(
        "--energy-min",
        type=float,
        default=None,
        help="lower scan energy",
    )
    parser.add_argument(
        "--energy-max",
        type=float,
        default=25.0,
        help="upper scan energy",
    )
    parser.add_argument(
        "--scan-points",
        type=int,
        default=1200,
        help="energy scan resolution",
    )

    # Вывод и дополнительные проверки.
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results"),
        help="output directory",
    )
    parser.add_argument(
        "--convergence",
        action="store_true",
        help="also run grid- and domain-convergence studies",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    """Выполнить полный сценарий расчёта из командной строки.

    `argv=None` означает, что argparse читает настоящие аргументы процесса.
    При передаче списка строк функцию можно вызывать программно, например в
    тестах или ноутбуке.
    """

    args = build_parser().parse_args(argv)

    # Простые предварительные проверки предотвращают заведомо ненадёжные
    # расчёты ещё до создания больших массивов.
    if args.xmax <= 1.2:
        raise SystemExit(
            "--xmax should extend well beyond the minima at x=±1"
        )
    if args.points < 101:
        raise SystemExit(
            "--points is too small for a reliable Numerov calculation"
        )

    # Создаём каталог результатов, включая отсутствующие родительские папки.
    args.output.mkdir(parents=True, exist_ok=True)

    # `partial` фиксирует v0 и alpha, превращая общую функцию
    # quartic_double_well(x, v0=..., alpha=...) в callable вида potential(x),
    # который ожидает решатель.
    potential = partial(
        quartic_double_well,
        v0=args.v0,
        alpha=args.alpha,
    )

    grid = UniformGrid(-args.xmax, args.xmax, args.points)
    solver = MatchedNumerovSolver(
        potential,
        grid,
        match_x=args.match_x,
    )

    # Если нижняя граница энергетического окна не указана, начинаем немного
    # выше минимума дискретного потенциала. Малый сдвиг исключает пограничные
    # численные неоднозначности ровно в точке V_min.
    energy_min = (
        float(np.min(solver.v)) + 1.0e-8
        if args.energy_min is None
        else float(args.energy_min)
    )

    states = solver.solve(
        energy_min,
        args.energy_max,
        n_states=args.states,
        scan_points=args.scan_points,
    )

    # Сохраняем численные данные и два диагностических графика.
    _write_spectrum(states, args.output / "spectrum.csv")
    _write_wavefunctions(
        states,
        solver.v,
        args.output / "wavefunctions.csv",
    )
    _plot_states(
        states,
        solver.v,
        args.output / "eigenstates.png",
        y_max=min(
            args.energy_max,
            max(state.energy for state in states) + 3.0,
        ),
        match_x=solver.match_x,
    )
    _plot_mismatch(
        solver,
        energy_min,
        args.energy_max,
        states,
        args.output / "mismatch.png",
        scan_points=min(args.scan_points, 2500),
    )

    # Краткий отчёт в терминал.
    print(f"Matching point: x_m = {solver.match_x:.8f}")
    print(
        f"Grid: [{grid.x_min}, {grid.x_max}], "
        f"N={grid.n_points}, dx={grid.step:.6g}"
    )
    print("\nEigenvalues")

    for state in states:
        print(
            f"  E_{state.index} = {state.energy:.12f}  "
            f"|Delta|={state.residual:.3e}"
        )

    # Проверка удалённости границ. Само условие psi=0 на границе задано
    # искусственно, поэтому смотрим плотность немного внутри области и сравниваем
    # потенциальную энергию на краях с энергией самого высокого уровня.
    edge_offset = max(
        2,
        int(round(0.02 * (len(solver.x) - 1))),
    )
    edge_density = max(
        states[-1].probability_density[edge_offset],
        states[-1].probability_density[-edge_offset - 1],
    )
    forbidden_margin = (
        min(solver.v[0], solver.v[-1]) - states[-1].energy
    )

    print(
        "\nHighest-state density 2% inside the edges: "
        f"{edge_density:.3e}"
    )
    print(
        "Boundary potential minus highest energy: "
        f"{forbidden_margin:.6g}"
    )

    if edge_density > 1.0e-8 or forbidden_margin <= 0.0:
        print(
            "WARNING: move the boundaries farther out and repeat "
            "the calculation."
        )

    if args.convergence:
        # Формируем три размера сетки: примерно половинный, исходный и в 1.5
        # раза больший. Оператор `| 1` делает последнее число нечётным.
        point_counts = sorted(
            {
                max(401, int(round(args.points / 2))),
                args.points,
                int(round(args.points * 1.5)) | 1,
            }
        )

        grid_rows = grid_convergence(
            potential,
            x_min=-args.xmax,
            x_max=args.xmax,
            point_counts=point_counts,
            energy_min=energy_min,
            energy_max=args.energy_max,
            n_states=args.states,
            match_x=args.match_x,
            scan_points=args.scan_points,
        )
        rows_to_csv(
            grid_rows,
            str(args.output / "grid_convergence.csv"),
        )

        # Для проверки границ берём область на 15% уже, исходную и на 15%
        # шире, сохраняя шаг близким к исходному.
        widths = [
            0.85 * args.xmax,
            args.xmax,
            1.15 * args.xmax,
        ]

        def factory():
            """Создать callable потенциала с текущими параметрами CLI."""

            return partial(
                quartic_double_well,
                v0=args.v0,
                alpha=args.alpha,
            )

        domain_rows = domain_convergence(
            factory,
            half_widths=widths,
            target_step=grid.step,
            energy_min=energy_min,
            energy_max=args.energy_max,
            n_states=args.states,
            match_x=args.match_x,
            scan_points=args.scan_points,
        )
        rows_to_csv(
            domain_rows,
            str(args.output / "domain_convergence.csv"),
        )
        print("Convergence tables written to the output directory.")


if __name__ == "__main__":
    # Этот блок срабатывает при прямом запуске файла, но не при импорте модуля.
    main()
