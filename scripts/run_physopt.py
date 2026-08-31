"""Three-way comparison on the physical figure of merit (v1).

For each readout (A cavity / B imaging) runs the three cases of the
deliverable:

  1. best uncoupled      — charges absent, trap frequencies (+ weights for B)
                           Metropolis-optimized: the null hypothesis;
  2. naive as-built      — uniform traps at the benchmark frequency, charges
                           at the stray floor: the lab baseline;
  3. optimized coupled   — traps + charges (+ weights for B) optimized.

Objective: PBH horizon distance d_max at the benchmark chirp mass 1e-3 Msun
over the 10-300 kHz band.  Absolute values inherit the TBC placeholder
parameters (powers, g profile, charge bounds); relative statements do not.

Outputs: results-mc/physopt_v1.pkl, physopt_v1_traces.(pdf|png),
physopt_v1_curves.(pdf|png), physopt_v1.log (stdout tee'd by hand).
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)

import pickle
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ResOSc.physopt import (metropolis_physical, naive_as_built,
                            build_array, readout_setup, evaluate_design,
                            MC_BENCH_SOLAR, BAND_HZ, Q_STRAY_E)
from ResOSc.physical import LSD_BENCHMARK, readout_B_observable_noise

AU = 1.496e11
N = 10
N_STEPS = 100000
SEEDS = (42, 43, 44)          # seed 42 chain starts from the incumbent
T_START, T_END = 0.2, 0.002   # v2: colder — v1 (1.0, 0.01) wandered 90% of walk
OUT = 'results-mc/physopt_v5'  # v5 = anchored model (AG13 drive, split
                               # damping, anchored readout A), 1e5 steps

LABELS = {'naive': 'corner system', 'uncoupled-opt': 'uncoupled opt',
          'coupled-opt': 'coupled opt'}


def incumbent_state(readout, coupled):
    """Naive as-built design as a chain seed: uniform traps at the benchmark
    frequency, charges at the stray floor (coupled) or absent (uncoupled),
    uniform weights for Readout B."""
    return {
        'f_traps': np.full(N, LSD_BENCHMARK['f_trap_hz']),
        'q': np.full(N, Q_STRAY_E) if coupled else None,
        'w': (np.full(N, 1.0 / np.sqrt(N)) if readout == 'B' else None),
    }

t_start = time.time()
results = {}

for readout in ('A', 'B'):
    print(f'=== Readout {readout} ===')
    results[readout] = {}

    print('naive as-built (uniform traps, stray-floor charges):')
    r_naive = naive_as_built(n=N, readout=readout)
    print(f'  d_max = {r_naive["best"]["dmax"]/AU:.3f} AU')
    results[readout]['naive'] = r_naive

    for tag, coupled in (('uncoupled-opt', False), ('coupled-opt', True)):
        print(f'{tag}: {len(SEEDS)} seeds x {N_STEPS} steps '
              f'(first seed incumbent-started)')
        runs = []
        for j, seed in enumerate(SEEDS):
            init = incumbent_state(readout, coupled) if j == 0 else None
            r = metropolis_physical(n=N, readout=readout, coupled=coupled,
                                    n_steps=N_STEPS, seed=seed,
                                    T_start=T_START, T_end=T_END,
                                    init_state=init, verbose=False)
            r['init'] = 'incumbent' if j == 0 else 'random'
            print(f'  seed {seed} ({r["init"]:>9}): '
                  f'best {r["best"]["dmax"]/AU:.3f} AU '
                  f'(step {r["best"]["step"]}, acc {r["acceptance_rate"]:.2f})')
            runs.append(r)
        best_run = max(runs, key=lambda r: r['best']['dmax'])
        best_run['all_seed_bests'] = [r['best']['dmax'] for r in runs]
        best_run['all_trajs'] = [r['dmax_traj'] for r in runs]
        results[readout][tag] = best_run
        print(f'  => best of seeds: {best_run["best"]["dmax"]/AU:.3f} AU')

with open(f'{OUT}.pkl', 'wb') as fh:
    pickle.dump(results, fh)
print(f'saved {OUT}.pkl  ({time.time()-t_start:.0f} s)')

# --------------------------------------------------------------------------- #
# Summary table
# --------------------------------------------------------------------------- #
print()
print(f'{"case":<16}{"readout A [AU]":>16}{"readout B [AU]":>16}')
for tag in ('uncoupled-opt', 'naive', 'coupled-opt'):
    row = [results[ro][tag]['best']['dmax'] / AU for ro in ('A', 'B')]
    print(f'{tag:<16}{row[0]:>16.3f}{row[1]:>16.3f}')
for ro in ('A', 'B'):
    dn = results[ro]['naive']['best']['dmax']
    du = results[ro]['uncoupled-opt']['best']['dmax']
    dc = results[ro]['coupled-opt']['best']['dmax']
    print(f'readout {ro}: coupled-opt / naive = {dc/dn:.3f}, '
          f'coupled-opt / uncoupled-opt = {dc/du:.3f}')

# --------------------------------------------------------------------------- #
# Figure 1: convergence traces
# --------------------------------------------------------------------------- #
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
for ax, ro in zip(axes, ('A', 'B')):
    for tag, color in (('uncoupled-opt', 'tab:blue'),
                       ('coupled-opt', 'tab:red')):
        for j, traj in enumerate(results[ro][tag]['all_trajs']):
            ax.plot(np.arange(1, len(traj) + 1), traj / AU, color=color,
                    alpha=0.5, lw=0.8,
                    label=LABELS[tag] if j == 0 else None)
    ax.axhline(results[ro]['naive']['best']['dmax'] / AU, color='k',
               ls='--', lw=1.0, label=LABELS['naive'])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Metropolis step')
    ax.set_title(f'Readout {ro}')
    ax.legend(fontsize=8)
axes[0].set_ylabel(r'$d_{\max}$ of current state [AU]')
fig.suptitle(r'Metropolis on the physical objective '
             r'($\mathcal{M}_c = 10^{-3}\,M_\odot$, 10--300 kHz)')
fig.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(f'{OUT}_traces.{ext}', dpi=200)
plt.close(fig)

# --------------------------------------------------------------------------- #
# Figure 2: sqrt(S_h) of the three best designs per readout
# --------------------------------------------------------------------------- #

def sh_curve(state, readout):
    q = state.get('q')
    arr = build_array(state['f_traps'], q)
    w_fixed, ro_noise = readout_setup(arr.n, readout)
    if readout == 'A':
        w = w_fixed
        f = arr.band_grid(*BAND_HZ)
        S_ro = ro_noise(f)
    else:
        w = state['w'] if state.get('w') is not None \
            else np.full(arr.n, 1.0 / np.sqrt(arr.n))
        w = w / np.linalg.norm(w)
        f = arr.band_grid(*BAND_HZ)
        S_ro = readout_B_observable_noise(w, ro_noise)
    return f, np.sqrt(arr.S_h(f, w, S_ro))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
for ax, ro in zip(axes, ('A', 'B')):
    for tag, color, lw in (('naive', 'k', 0.8),
                           ('uncoupled-opt', 'tab:blue', 0.9),
                           ('coupled-opt', 'tab:red', 0.9)):
        f, sh = sh_curve(results[ro][tag]['best']['state'], ro)
        ax.plot(f * 1e-3, sh, color=color, lw=lw, label=LABELS[tag])
    ax.axhspan(1e-22, 1e-21, color='tab:green', alpha=0.15,
               label='LSD target')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('frequency [kHz]')
    ax.set_title(f'Readout {ro}')
    ax.legend(fontsize=8)
axes[0].set_ylabel(r'$\sqrt{S_h(f)}\ [\mathrm{Hz}^{-1/2}]$')
fig.suptitle('Strain sensitivity of the three designs')
fig.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(f'{OUT}_curves.{ext}', dpi=200)
plt.close(fig)
print(f'figures: {OUT}_traces.pdf/.png, {OUT}_curves.pdf/.png')

# --------------------------------------------------------------------------- #
# Best-design structure dump
# --------------------------------------------------------------------------- #
for ro in ('A', 'B'):
    st = results[ro]['coupled-opt']['best']['state']
    arr = build_array(st['f_traps'], st['q'])
    print(f'\nreadout {ro} coupled-opt best design:')
    print('  f_trap [kHz]:', np.array2string(np.sort(st['f_traps']) * 1e-3,
                                             precision=1))
    print('  q [e]       :', np.array2string(st['q'], precision=2,
                                             formatter={'float_kind': lambda x: f'{x:.2e}'}))
    print('  mode f_n [kHz]:', np.array2string(arr.f_n * 1e-3, precision=1))
    if st.get('w') is not None:
        print('  w           :', np.array2string(st['w'], precision=2))
