# ResOSc

A Python package for simulating and analyzing coupled oscillator systems. ResOSc computes normal modes, forced oscillation response with damping, and optimizes observables for detecting system dynamics.

## Features

- **Normal Mode Analysis** — Eigenvalue decomposition to find normal frequencies and mode shapes
- **Forced Oscillation Response** — Complex amplitude response curves with damping
- **Observable Optimization** — Find optimal sensor weights to maximize detection sensitivity across all modes
- **Visualization** — Publication-quality plots of mode shapes, amplitude response, and coupling heatmaps

## Installation

```bash
pip install numpy scipy matplotlib
```

Then clone the repository and import the package:

```python
from ResOSc import CoupledSystem
```

## Quick Start

```python
from ResOSc.system import CoupledSystem

# Define a 3-oscillator system
sys = CoupledSystem(n=3)
sys.set_masses([1.0, 1.0, 1.0])
sys.set_springs(wall=[1.0, 1.0], coupling=[0.5, 0.5])
sys.set_damping(0.01)

# Solve for normal modes
sys.build_H()
sys.solve()

# Compute amplitudes over a frequency range
sys.compute_forces()
sys.compute_amplitudes(omega_range=(0, 3), n_points=1000)
```

## Modules

| Module | Description |
|---|---|
| `system` | Core `CoupledSystem` class — masses, springs, damping, eigenvalue solver |
| `modes` | Mode shape visualization (stem plots and heatmaps) |
| `observables` | Observable analysis, coupling matrices, and sensor optimization |
| `plotting` | Amplitude response curves with LaTeX-rendered labels |

## Dependencies

- NumPy
- SciPy
- Matplotlib

## Demo

See `ResOSc_demo_used.ipynb` for worked examples and visualizations.
