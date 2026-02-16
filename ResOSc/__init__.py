"""
ResOSc — Resonant Oscillator Simulator for Coupled Systems

A package for simulating coupled oscillator systems, computing normal modes,
forced oscillation response with damping, and optimizing observables.
"""

from .system import CoupledSystem

__all__ = ["CoupledSystem"]
