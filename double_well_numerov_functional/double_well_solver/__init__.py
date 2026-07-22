"""Публичные функции функционального Numerov-проекта."""

from .numerov import make_uniform_grid
from .potentials import harmonic_oscillator, quartic_double_well
from .solver import (
    bisect_root,
    build_branches,
    build_eigenstate,
    fd_eigenvalue_estimates,
    fd_sturm_count,
    find_eigenvalues,
    mismatch,
    prepare_problem,
    root_near_estimate,
    scan_mismatch,
    solve,
    state_probability_density,
    suggest_match_index,
)


__all__ = [
    "bisect_root",
    "build_branches",
    "build_eigenstate",
    "fd_eigenvalue_estimates",
    "fd_sturm_count",
    "find_eigenvalues",
    "harmonic_oscillator",
    "make_uniform_grid",
    "mismatch",
    "prepare_problem",
    "quartic_double_well",
    "root_near_estimate",
    "scan_mismatch",
    "solve",
    "state_probability_density",
    "suggest_match_index",
]
