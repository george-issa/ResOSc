# ResOSc

**Resonant Oscillator Simulator for Coupled Systems** is a Python package for
designing arrays of coupled mechanical oscillators as resonant detectors, with
an application to high-frequency gravitational waves (GW) and dark matter (DM)
using levitated sensors.

ResOSc computes the normal modes and damped forced response of an N-oscillator
array, finds the observable (sensor weighting) that maximizes sensitivity
across all modes, searches the design space with Monte Carlo and Metropolis
optimization, and forecasts physical detector reach (strain noise,
matched-filter SNR, horizon distance) for arrays anchored to the Levitated
Sensor Detector (LSD) parameters.

## Repository layout

```
ResOSc/          Python package
scripts/         Optimization runs and figure scripts (outputs go to results/)
notebooks/       Worked examples
docs/            Notes (LaTeX + PDF), research logs, references
results/
  mc/            Dimensionless-model Monte Carlo and Metropolis runs
  route3/        Parasitic-coupling-floor sweep
  physical/      Physical (SI) model optimization and validation
```

Figures are not tracked: the notebooks write them to `docs/figures/` and the
scripts to `results/`.

## Installation

```bash
git clone https://github.com/george-issa/ResOSc.git
cd ResOSc
conda env create -f environment.yml   # creates `resosc` with the package installed in editable mode
conda activate resosc
```

Or, in an existing environment with Python ≥ 3.7: `pip install -e .`
(dependencies: NumPy, SciPy, Matplotlib).

## Quick start

**Coupled-oscillator model (dimensionless units)**

```python
import numpy as np
from ResOSc import CoupledSystem

sys = CoupledSystem(n=3)
sys.set_masses([1.0, 1.0, 1.0])
sys.set_springs(wall=[1.0, 1.0, 1.0], coupling=[0.5, 0.5])
sys.set_damping(0.01)
sys.solve()                                   # normal modes

sys.compute_forces(force_model='strain')      # GW strain forcing
b, x, w = sys.compute_amplitudes(np.linspace(0.01, 3.0, 1000)**2)
```

**Physical detector forecast (SI units)**

```python
from ResOSc.physical import lsd_array, lsd_readout_A, horizon_distance, KPC

arr = lsd_array(n=10)                  # uncoupled LSD benchmark stack
w, S_ro = lsd_readout_A(arr.n)         # cavity readout: fixed weights + noise
d = horizon_distance(arr, w, S_ro, Mc_solar=1e-3, f_lo=1e4, f_hi=3e5)
print(f"horizon distance: {d / KPC * 1e3:.2f} pc")   # 7.78 pc
```

Run the physical model's self-checks with `python -m ResOSc.physical`.

## Modules

| Module | Description |
|---|---|
| `system` | Core `CoupledSystem` class: masses, springs, damping, eigenvalue solver |
| `modes` | Mode-shape visualization (stem plots and heatmaps) |
| `observables` | Observable analysis, coupling matrices, and sensor optimization |
| `montecarlo` | Monte Carlo and Metropolis searches over array configurations |
| `physical` | Physical (SI) sensitivity pipeline: noise, strain PSD, SNR, horizon distance |
| `physopt` | Metropolis optimization of the physical figure of merit |
| `plotting` | Amplitude response curves with LaTeX-rendered labels |

## Physical sensitivity pipeline

`ResOSc/physical.py` turns an array design into a detector forecast through a
chain of stages, each a strict compression of the previous one:

```
{m_i, k_trap_i, q_i, (w_i)} → {f_n, v_n, mu_n, Gamma_n} → T(f), S_O^th, S_O^ro → S_h(f) → rho → d_max
```

1. **Hardware.** N levitated discs: masses `m_i`, trap stiffnesses `k_trap_i`,
   mechanical linewidths `gamma_i` (Hz, cold-damped), and optional charges
   `q_i` giving Coulomb coupling springs `k_ij ∝ q_i q_j / d_ij^3` between
   every pair (long-range tails included). The force-noise damping (gas
   damping plus photon-recoil heating) is carried separately from the
   linewidth. Masses, traps, and charges are the design variables.
2. **Modal decomposition.** The generalized eigenproblem `K v = ω² M v`
   decouples the array into N damped oscillators (frequencies `f_n`, unit-norm
   shapes `v_n`, modal masses `mu_n`, linewidths `Gamma_n`), each responding
   through a Lorentzian susceptibility `chi_n(f)` with resonant gain
   `Q_n = f_n / Gamma_n`. Design variables enter the chain only here.
3. **Signal coupling.** A gravitational-wave strain `h` inertially drives each
   cavity-locked trap minimum (Arvanitaki & Geraci 2013):
   `F_i = ½ m_i (2π f)² L h`, with `f` the GW frequency and `L` the full
   cavity length. The modal drive is `B_n = ½ (2π f)² L (Vᵀm)_n`; the
   quasi-static form `k_trap_i · L · h` is its on-resonance limit.
4. **Observable.** One scalar `O(t) = Σ w_i x_i(t)` with modal weights
   `N_n = (Vᵀw)_n` and transfer function `T(f) = Σ N_n B_n chi_n(f)`
   (meters per unit strain). Readout A (cavity) fixes `w = g/|g|` in hardware;
   Readout B (imaging) leaves `w` free in software.
5. **Noise budget.** Thermal noise (fluctuation–dissipation theorem, using
   the force-noise damping) is a *force*, filtered by the same `|chi_n|²` as
   the signal, so it is resonantly peaked. Readout shot noise is
   *imprecision* added after the mechanics: flat (Readout B) or cavity-pole
   shaped (Readout A).
6. **Strain-referred sensitivity.** `S_h(f) = (S_O^th + S_O^ro) / |T(f)|²`.
   Near each resonance the `|chi_n|²` cancels between signal and thermal
   noise, giving flat, weight-independent thermal buckets whose usable width
   is set by the thermal/readout crossover.
7. **Source model.** Planetary-mass primordial-black-hole (PBH) binaries
   (chirp mass 1e-4–1e-2 M☉) inspiral through 10–300 kHz with
   `|h̃(f)|² = A² f^(-7/3)` up to the ISCO cutoff, the natural source for
   high-Q levitated arrays.
8. **Matched-filter SNR → horizon distance.** `ρ² = 4∫ |h̃|²/S_h df`
   accumulates every sensitivity bucket under the source envelope; since
   `ρ ∝ 1/r`, the horizon distance `d_max = r_ref · ρ(r_ref)/8` is the radius
   within which the source is detectable (rate ∝ `d_max³`). The dark-matter
   channel reuses the same machinery with a flat impulse template
   (`min_impulse`).

The full derivation is in `docs/anchored_model_notes.pdf`,
`docs/ResOSc_notes_and_findings.pdf` (§ "The Physical Sensitivity Chain, Step by Step"),
and the companion note `docs/noise_function_notes.pdf`.

## Scripts

Scripts resolve paths from the repository root, so they can be run from any
directory, e.g. `python scripts/run_mc.py --help`.

| Script | Purpose | Output |
|---|---|---|
| `run_mc.py`, `plot_mc.py` | Monte Carlo / Metropolis search and its figures | `results/mc/` |
| `validate_swept.py`, `batch_swept_validation.py` | Swept-metric validation of MC winners | `results/mc/` |
| `polish_one_bond.py`, `plot_best_configs.py` | One-bond topology polish (route 2) and best-config figures | `results/mc/` |
| `route3_sweep.py`, `plot_route3.py`, `plot_route3_champion.py` | Sensitivity vs parasitic coupling floor (route 3) | `results/route3/` |
| `run_physopt.py` | Three-way comparison on the physical figure of merit | `results/physical/` |
| `plot_physical_validation.py`, `plot_template_ceiling.py` | Physical-model validation and trap-ceiling figures | `results/physical/` |

## Documentation

| File | Contents |
|---|---|
| `docs/anchored_model_notes.pdf` | Physical model anchored to LSD / AG13: equations, parameters, optimization results |
| `docs/ResOSc_notes_and_findings.pdf` | Full project notes: physics, optimization methods, results |
| `docs/noise_function_notes.pdf` | Companion note: SNR of a multimode mechanical detector |
| `docs/results.md` | Findings log for the dimensionless model |
| `docs/monte_carlo.md` | Monte Carlo run log: settings, seeds, artifacts |
| `docs/references.md` | Papers the physical model is anchored to |

LaTeX sources sit next to their PDFs. To rebuild them, first regenerate the
figures (run the notebooks and the `plot_*` scripts), then
`cd docs && latexmk -pdf anchored_model_notes.tex`.

Results computed before the 2026-08-20 eigenvector fix are superseded and kept
as project history; see the note at the top of `docs/results.md`.

## Notebooks

| Notebook | Contents |
|---|---|
| `notebooks/01_basics.ipynb` | Normal modes, forced response, mode shapes, observables |
| `notebooks/02_forces_and_observables.ipynb` | Force models (GW strain vs DM), generalized forces, frequency spread |
| `notebooks/03_monte_carlo.ipynb` | Monte Carlo optimization of masses and springs |
