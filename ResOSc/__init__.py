"""
ResOSc — Resonant Oscillator Simulator for Coupled Systems

A package for simulating coupled oscillator systems, computing normal modes,
forced oscillation response with damping, and optimizing observables.
"""

from .system import CoupledSystem
from .montecarlo import (monte_carlo_optimize, metropolis_optimize,
                         plot_monte_carlo_results, plot_comparison)

__all__ = ["CoupledSystem",
           "monte_carlo_optimize", "metropolis_optimize",
           "plot_monte_carlo_results", "plot_comparison"]
