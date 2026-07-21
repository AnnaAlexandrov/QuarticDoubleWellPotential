"""Публичный интерфейс пакета `double_well_solver`.

Импорт наиболее часто используемых классов и функций сюда позволяет писать

    from double_well_solver import MatchedNumerovSolver, UniformGrid

вместо более длинных импортов из отдельных внутренних модулей.
"""

from .numerov import UniformGrid
from .potentials import harmonic_oscillator, quartic_double_well
from .solver import Eigenstate, MatchedNumerovSolver


# `__all__` явно перечисляет имена, считающиеся публичной частью библиотеки.
# Например, именно они импортируются командой `from double_well_solver import *`.
__all__ = [
    "Eigenstate",
    "MatchedNumerovSolver",
    "UniformGrid",
    "harmonic_oscillator",
    "quartic_double_well",
]
