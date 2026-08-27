# ResOSc

A Python package for simulating and analyzing coupled oscillator systems. ResOSc computes normal modes, forced oscillation response with damping, and optimizes observables for detecting system dynamics.

## Features

- **Normal Mode Analysis** — Eigenvalue decomposition to find normal frequencies and mode shapes
- **Forced Oscillation Response** — Complex amplitude response curves with damping
- **Observable Optimization** — Find optimal sensor weights to maximize detection sensitivity across all modes
- **Visualization** — Plots of mode shapes, amplitude response, and coupling heatmaps

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
| `montecarlo` | Monte Carlo and Metropolis searches over array configurations |
| `physical` | Physical (SI) sensitivity pipeline — noise, strain PSD, SNR, horizon distance |
| `plotting` | Amplitude response curves with LaTeX-rendered labels |

## Physical Sensitivity Pipeline

`ResOSc/physical.py` turns an array design into a detector forecast through a
chain of stages, each a strict compression of the previous one:

```
{m_i, k_trap_i, q_i, (w_i)} → {f_n, v_n, mu_n, Gamma_n} → T(f), S_O^th, S_O^ro → S_h(f) → rho → d_max
```

1. **Hardware.** N levitated discs: masses `m_i`, trap stiffnesses `k_trap_i`,
   damping linewidths `gamma_i` (Hz), optional charges `q_i` giving Coulomb
   coupling springs `k_ij ∝ q_i q_j / d_ij^3` between every pair (long-range
   tails included). Masses, traps, and charges are the design variables.
2. **Modal decomposition.** The generalized eigenproblem `K v = ω² M v`
   decouples the array into N damped oscillators — frequencies `f_n`, unit-norm
   shapes `v_n`, modal masses `mu_n`, linewidths `Gamma_n` — each responding
   through a Lorentzian susceptibility `chi_n(f)` with resonant gain
   `Q_n = f_n / Gamma_n`. Design variables enter the chain only here.
3. **Signal coupling.** A gravitational-wave strain `h` displaces each trap
   equilibrium by `h·L`, equivalent to forces `F_i = k_trap_i · L · h`; the
   modal drive is the overlap `B_n = v_nᵀ β`.
4. **Observable.** One scalar `O(t) = Σ w_i x_i(t)` with modal weights
   `N_n = (Vᵀw)_n` and transfer function `T(f) = Σ N_n B_n chi_n(f)`
   (meters per unit strain). Readout A (cavity) fixes `w = g/|g|` in hardware;
   Readout B (imaging) leaves `w` free in software.
5. **Noise budget.** Thermal noise (fluctuation–dissipation theorem) is a
   *force*, filtered by the same `|chi_n|²` as the signal — resonantly peaked.
   Readout shot noise is *imprecision* added after the mechanics — flat
   (Readout B) or cavity-pole shaped (Readout A).
6. **Strain-referred sensitivity.** `S_h(f) = (S_O^th + S_O^ro) / |T(f)|²`.
   Near each resonance the `|chi_n|²` cancels between signal and thermal
   noise, giving flat, weight-independent thermal buckets whose usable width
   is set by the thermal/readout crossover.
7. **Source model.** Planetary-mass primordial-black-hole (PBH) binaries
   (chirp mass 1e-4–1e-2 M☉) inspiral through 10–300 kHz with
   `|h̃(f)|² = A² f^(-7/3)` up to the ISCO cutoff — the natural source for
   high-Q levitated arrays.
8. **Matched-filter SNR → horizon distance.** `ρ² = 4∫ |h̃|²/S_h df`
   accumulates every sensitivity bucket under the source envelope; since
   `ρ ∝ 1/r`, the horizon distance `d_max = r_ref · ρ(r_ref)/8` is the radius
   within which the source is detectable (rate ∝ `d_max³`). The dark-matter
   channel reuses the same machinery with a flat impulse template
   (`min_impulse`).

The full derivation with equation references lives in
`ResOSc_notes_and_findings.tex` (§ "The Physical Sensitivity Chain") and the
companion note `noise_function_notes.pdf`.

## Dependencies

- NumPy
- SciPy
- LAPACK
- Matplotlib

## Demo

See `ResOSc_demo.ipynb` for worked examples and visualizations.
