"""Функциональный командный интерфейс для квартичной двойной ямы.

Аргументы разбираются функцией ``getopt.getopt`` и сохраняются в обычном
словаре. Численный расчёт выполняется последовательностью вызовов функций без
создания пользовательских объектов.
"""

from __future__ import annotations

import getopt
import os
import sys
from functools import partial

import matplotlib.pyplot as plt
import numpy as np

from .analysis import domain_convergence, grid_convergence, rows_to_csv
from .numerov import make_uniform_grid
from .potentials import quartic_double_well
from .solver import (
    Eigenstate,
    Problem,
    prepare_problem,
    scan_mismatch,
    solve,
    state_probability_density,
)


def usage() -> str:
    """Вернуть текст справки командной строки."""

    return """Usage: python run_project.py [options]

Options:
  --v0 VALUE             barrier scale V0 (default: 5.0)
  --alpha VALUE          linear asymmetry (default: 0.0)
  --xmax VALUE           symmetric half-domain (default: 4.0)
  --points N             number of spatial grid points (default: 2401)
  --match-x VALUE        manual matching coordinate
  --states N             number of eigenstates (default: 6)
  --energy-min VALUE     lower energy bound; default is min(V)+1e-8
  --energy-max VALUE     upper energy bound (default: 25.0)
  --scan-points N        energy scan resolution (default: 1200)
  --output DIRECTORY     output directory (default: results)
  --convergence          also run convergence studies
  --help                 show this message
"""


def parse_arguments(argv: list[str]) -> dict[str, object]:
    """Разобрать список строк и вернуть словарь параметров.

    Значения по умолчанию сначала записываются в новый словарь. Затем каждая
    опция создаёт его обновлённое содержимое. Никакого объекта Namespace и
    обращения к полям через точку здесь нет.
    """

    options: dict[str, object] = {
        "v0": 5.0,
        "alpha": 0.0,
        "xmax": 4.0,
        "points": 2401,
        "match_x": None,
        "states": 6,
        "energy_min": None,
        "energy_max": 25.0,
        "scan_points": 1200,
        "output": "results",
        "convergence": False,
    }

    try:
        parsed, remaining = getopt.getopt(
            argv,
            "",
            [
                "v0=",
                "alpha=",
                "xmax=",
                "points=",
                "match-x=",
                "states=",
                "energy-min=",
                "energy-max=",
                "scan-points=",
                "output=",
                "convergence",
                "help",
            ],
        )
    except getopt.GetoptError as exc:
        raise SystemExit(f"{exc}\n\n{usage()}") from exc

    if remaining:
        raise SystemExit(
            "Unexpected positional arguments: "
            + " ".join(remaining)
            + "\n\n"
            + usage()
        )

    for name, value in parsed:
        if name == "--help":
            print(usage())
            raise SystemExit(0)
        if name == "--convergence":
            options["convergence"] = True
        elif name == "--v0":
            options["v0"] = float(value)
        elif name == "--alpha":
            options["alpha"] = float(value)
        elif name == "--xmax":
            options["xmax"] = float(value)
        elif name == "--points":
            options["points"] = int(value)
        elif name == "--match-x":
            options["match_x"] = float(value)
        elif name == "--states":
            options["states"] = int(value)
        elif name == "--energy-min":
            options["energy_min"] = float(value)
        elif name == "--energy-max":
            options["energy_max"] = float(value)
        elif name == "--scan-points":
            options["scan_points"] = int(value)
        elif name == "--output":
            options["output"] = value

    return options


def validate_options(options: dict[str, object]) -> None:
    """Проверить параметры до создания больших численных массивов."""

    if float(options["xmax"]) <= 1.2:
        raise SystemExit(
            "--xmax should extend well beyond the minima at x=±1"
        )
    if int(options["points"]) < 101:
        raise SystemExit(
            "--points is too small for a reliable Numerov calculation"
        )
    if int(options["states"]) < 1:
        raise SystemExit("--states must be positive")
    if int(options["scan_points"]) < 20:
        raise SystemExit("--scan-points must be at least 20")


def write_spectrum(states: list[Eigenstate], path: str) -> None:
    """Сохранить номера уровней, энергии и residual в CSV."""

    matrix = np.array(
        [
            [
                int(state["index"]),
                float(state["energy"]),
                float(state["residual"]),
            ]
            for state in states
        ],
        dtype=float,
    )
    np.savetxt(
        path,
        matrix,
        delimiter=",",
        header="state,energy,mismatch_residual",
        comments="",
        fmt=["%.0f", "%.15g", "%.6e"],
    )


def write_wavefunctions(
    states: list[Eigenstate],
    potential_values: np.ndarray,
    path: str,
) -> None:
    """Сохранить сетку, потенциал, волновые функции и плотности в CSV."""

    header = ["x", "V"]
    columns = [
        np.asarray(states[0]["x"], dtype=float),
        potential_values,
    ]

    for state in states:
        index = int(state["index"])
        header.extend([f"psi_{index}", f"density_{index}"])
        columns.extend(
            [
                np.asarray(state["psi"], dtype=float),
                state_probability_density(state),
            ]
        )

    np.savetxt(
        path,
        np.column_stack(columns),
        delimiter=",",
        header=",".join(header),
        comments="",
    )


def plot_states(
    states: list[Eigenstate],
    potential_values: np.ndarray,
    output_path: str,
    *,
    y_max: float | None,
    match_x: float,
) -> None:
    """Построить потенциал, энергетические уровни и сдвинутые состояния."""

    x = np.asarray(states[0]["x"], dtype=float)
    plt.figure(figsize=(9, 6))
    plt.plot(x, potential_values, linewidth=2.0, label="V(x)")

    amplitude_scale = 0.45
    for state in states:
        energy = float(state["energy"])
        index = int(state["index"])
        psi = np.asarray(state["psi"], dtype=float)
        plt.axhline(energy, linewidth=0.7, alpha=0.35)
        plt.plot(
            x,
            energy + amplitude_scale * psi,
            label=fr"$E_{index}={energy:.6f}$",
        )

    plt.axvline(
        match_x,
        linewidth=0.8,
        alpha=0.35,
        linestyle="--",
        label=fr"$x_m={match_x:.3f}$",
    )
    plt.xlabel("x")
    plt.ylabel("Energy; shifted wave functions")
    plt.title("Quartic double well: matched Numerov eigenstates")
    plt.grid(alpha=0.25)

    if y_max is not None:
        plt.ylim(
            min(float(np.min(potential_values)), 0.0) - 0.5,
            y_max,
        )

    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_mismatch(
    problem: Problem,
    energy_min: float,
    energy_max: float,
    states: list[Eigenstate],
    output_path: str,
    *,
    scan_points: int,
) -> None:
    """Построить mismatch-функцию и отметить найденные нули."""

    energies, mismatch_values = scan_mismatch(
        problem,
        energy_min,
        energy_max,
        scan_points=scan_points,
    )

    plt.figure(figsize=(9, 5))
    plt.plot(energies, mismatch_values, linewidth=1.0)
    plt.axhline(0.0, linewidth=0.8)

    for state in states:
        plt.axvline(
            float(state["energy"]),
            linestyle="--",
            linewidth=0.8,
        )

    plt.xlabel("E")
    plt.ylabel("normalized Wronskian mismatch")
    plt.title("Root search for the discrete spectrum")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def run_convergence_studies(
    options: dict[str, object],
    potential,
    grid: dict[str, object],
    energy_min: float,
) -> None:
    """Выполнить две проверки сходимости и сохранить таблицы."""

    points = int(options["points"])
    xmax = float(options["xmax"])
    output = str(options["output"])

    point_counts = sorted(
        {
            max(401, int(round(points / 2))),
            points,
            int(round(points * 1.5)) | 1,
        }
    )

    grid_rows = grid_convergence(
        potential,
        x_min=-xmax,
        x_max=xmax,
        point_counts=point_counts,
        energy_min=energy_min,
        energy_max=float(options["energy_max"]),
        n_states=int(options["states"]),
        match_x=options["match_x"],
        scan_points=int(options["scan_points"]),
    )
    rows_to_csv(
        grid_rows,
        os.path.join(output, "grid_convergence.csv"),
    )

    widths = [0.85 * xmax, xmax, 1.15 * xmax]

    def potential_factory():
        """Вернуть новую функцию потенциала с теми же параметрами."""

        return partial(
            quartic_double_well,
            v0=float(options["v0"]),
            alpha=float(options["alpha"]),
        )

    domain_rows = domain_convergence(
        potential_factory,
        half_widths=widths,
        target_step=float(grid["step"]),
        energy_min=energy_min,
        energy_max=float(options["energy_max"]),
        n_states=int(options["states"]),
        match_x=options["match_x"],
        scan_points=int(options["scan_points"]),
    )
    rows_to_csv(
        domain_rows,
        os.path.join(output, "domain_convergence.csv"),
    )
    print("Convergence tables written to the output directory.")


def main(argv: list[str] | None = None) -> None:
    """Выполнить полный сценарий расчёта из командной строки."""

    actual_argv = sys.argv[1:] if argv is None else argv
    options = parse_arguments(actual_argv)
    validate_options(options)

    output = str(options["output"])
    os.makedirs(output, exist_ok=True)

    potential = partial(
        quartic_double_well,
        v0=float(options["v0"]),
        alpha=float(options["alpha"]),
    )
    xmax = float(options["xmax"])
    grid = make_uniform_grid(
        -xmax,
        xmax,
        int(options["points"]),
    )
    problem = prepare_problem(
        potential,
        grid,
        match_x=options["match_x"],
    )

    potential_values = np.asarray(problem["v"], dtype=float)
    energy_min = (
        float(np.min(potential_values)) + 1.0e-8
        if options["energy_min"] is None
        else float(options["energy_min"])
    )

    states = solve(
        problem,
        energy_min,
        float(options["energy_max"]),
        n_states=int(options["states"]),
        scan_points=int(options["scan_points"]),
    )

    write_spectrum(states, os.path.join(output, "spectrum.csv"))
    write_wavefunctions(
        states,
        potential_values,
        os.path.join(output, "wavefunctions.csv"),
    )
    plot_states(
        states,
        potential_values,
        os.path.join(output, "eigenstates.png"),
        y_max=min(
            float(options["energy_max"]),
            max(float(state["energy"]) for state in states) + 3.0,
        ),
        match_x=float(problem["match_x"]),
    )
    plot_mismatch(
        problem,
        energy_min,
        float(options["energy_max"]),
        states,
        os.path.join(output, "mismatch.png"),
        scan_points=min(int(options["scan_points"]), 2500),
    )

    print(f"Matching point: x_m = {float(problem['match_x']):.8f}")
    print(
        f"Grid: [{float(grid['x_min'])}, {float(grid['x_max'])}], "
        f"N={int(grid['n_points'])}, dx={float(grid['step']):.6g}"
    )
    print("\nEigenvalues")

    for state in states:
        print(
            f"  E_{int(state['index'])} = {float(state['energy']):.12f}  "
            f"|Delta|={float(state['residual']):.3e}"
        )

    x = np.asarray(problem["x"], dtype=float)
    edge_offset = max(2, int(round(0.02 * (len(x) - 1))))
    highest_density = state_probability_density(states[-1])
    edge_density = max(
        highest_density[edge_offset],
        highest_density[-edge_offset - 1],
    )
    forbidden_margin = (
        min(potential_values[0], potential_values[-1])
        - float(states[-1]["energy"])
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

    if bool(options["convergence"]):
        run_convergence_studies(options, potential, grid, energy_min)


if __name__ == "__main__":
    main()
