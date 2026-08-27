"""Plot the MC configurations that beat the uncoupled bound: their
parameter profiles, and the states (normal modes) of the best one.

Winners are selected by box-normalized sensitivity (> 1) when present,
falling back to the per-config reference ratio for older result files.
An externally polished champion (.npy: 10 masses, 10 walls, k_bond2) can
be overlaid as the highlighted design via --champion.

    python3 plot_best_configs.py \
        --input results-mc/mc_results_v7_resolvable.pkl \
        --stem results-mc/mc_best_configs_v7 \
        --champion results-mc/one_bond_champion_resolvable.npy
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)
import sys, os
sys.path.insert(0, '/Users/gissa/Documents/Nancy/ResOSc')
import argparse
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['text.usetex'] = False
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

p = argparse.ArgumentParser(description='Plot bound-beating MC configurations.')
p.add_argument('--input', default='results-mc/mc_results_v7_resolvable.pkl')
p.add_argument('--stem',  default='results-mc/mc_best_configs_v7',
               help='Output path stem (.pdf and .png are appended)')
p.add_argument('--label', default='v7 constrained run (20000 steps, min gap 3 linewidths)')
p.add_argument('--note',  default='', help='Extra caution/annotation line')
p.add_argument('--champion', default=None,
               help='Optional .npy (10 masses, 10 walls, k_bond2) polished '
                    'one-bond design to highlight instead of the run winner')
p.add_argument('--mass-bounds',     type=float, nargs=2, default=[0.1, 3.0])
p.add_argument('--wall-bounds',     type=float, nargs=2, default=[0.2, 6.0])
p.add_argument('--coupling-bounds', type=float, nargs=2, default=[0.001, 1.0])
args = p.parse_args()

# ---- palette (validated reference instance, light mode) --------------------
BLUE    = '#2a78d6'   # categorical slot 1 — highlighted configuration
ORANGE  = '#eb6834'   # categorical slot 2 — uncoupled reference
INK     = '#0b0b0b'
INK2    = '#52514e'
MUTED   = '#898781'
GRID    = '#e1e0d9'
BASELN  = '#c3c2b7'
SURFACE = '#fcfcfb'
DIV = LinearSegmentedColormap.from_list(
    'blue_gray_red', ['#104281', '#2a78d6', '#86b6ef', '#f0efec',
                      '#f0a5a4', '#e34948', '#8f1f1f'])

plt.rcParams.update({
    'font.family': 'sans-serif',
    'text.color': INK, 'axes.labelcolor': INK2,
    'xtick.color': MUTED, 'ytick.color': MUTED,
    'axes.edgecolor': BASELN, 'axes.linewidth': 0.8,
    'grid.color': GRID, 'grid.linewidth': 0.6,
    'axes.grid': True, 'axes.axisbelow': True,
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
    'font.size': 9, 'axes.titlesize': 10,
})

# ---- data -------------------------------------------------------------------
from ResOSc.montecarlo import _build_and_solve, _box_optimal_uncoupled_scale
from ResOSc.observables import optimize_observable

with open(args.input, 'rb') as f:
    d = pickle.load(f)

refined = d['result_metro']['all_refined']

def score(c):
    v = c.get('box_normalized_min_sensitivity')
    if v is None:
        v = c.get('normalized_min_sensitivity') or 0.0
    return v

winners = [c for c in refined if score(c) > 1.0]
winners.sort(key=score, reverse=True)

if args.champion:
    x = np.load(args.champion)
    kc = np.zeros(9)
    kc[1] = x[20]
    csys = _build_and_solve(10, x[:10], x[10:20], kc, 0.01, 'strain', 1.0, 1.0, None)
    box = _box_optimal_uncoupled_scale(10, tuple(args.mass_bounds),
                                       tuple(args.wall_bounds), 0.01, 'strain')
    opt = optimize_observable(csys)
    cref = csys.reference_system()
    cref.compute_forces(L=1.0, h0=1.0, force_model='strain')
    best = {'masses': x[:10], 'wall': x[10:20], 'coupling': kc,
            'system': csys, 'reference': cref,
            'spread_val': csys.frequency_spread()['relative']}
    ns_best = opt['min_sensitivity'] / box
    best_label = f'one-bond champion ($S/S_{{unc}}$ = {ns_best:.3f})'
    others = winners
    others_label = f'{len(winners)} run winners ($S/S_{{unc}}>1$)'
else:
    best = winners[0]
    ns_best = score(best)
    best_label = f'best ($S/S_{{unc}}$ = {ns_best:.3f})'
    others = winners[1:]
    others_label = f'other {len(others)} with $S/S_{{unc}}>1$'

sysb = best['system']
refb = best['reference']
n    = len(best['masses'])
sites = np.arange(1, n + 1)
bonds = np.arange(1, n)      # bond i sits between sites i and i+1

# normal modes of the highlighted config, sorted ascending in frequency
order = np.argsort(sysb.frequencies)
freqs = sysb.frequencies[order]
modes = sysb.eigenvectors[order]                     # rows = modes, cols = sites
modes = modes / np.linalg.norm(modes, axis=1, keepdims=True)
for i in range(n):                                    # fix sign: largest comp > 0
    if modes[i, np.argmax(np.abs(modes[i]))] < 0:
        modes[i] *= -1
ref_freqs = np.sort(refb.frequencies)

# hand-chosen baseline system for context in the spectrum panel
from ResOSc import CoupledSystem
base = CoupledSystem(n)
base.set_masses(np.linspace(0.45, 0.90, n))
base.set_springs(np.full(n, 1.2), np.full(n - 1, 0.1))
base.set_damping(0.01)
base.solve()
base_freqs = np.sort(base.frequencies)

# ---- figure -----------------------------------------------------------------
fig = plt.figure(figsize=(11.5, 7.2))
gs  = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.30,
                       left=0.065, right=0.965,
                       top=0.81 if args.note else 0.84, bottom=0.08)

fig.suptitle(f'Configurations that beat the universal uncoupled bound '
             f'({len(winners)} found)',
             fontsize=13, color=INK, x=0.065, ha='left', y=0.965)
fig.text(0.065, 0.915,
         f'{args.label}: {len(winners)} of {len(refined)} refined candidates '
         f'score $S/S_{{unc}} > 1$; highlighted design = {ns_best:.3f}.',
         fontsize=9, color=INK2)
fig.text(0.065, 0.888,
         'Top row: parameters. Bottom row: normal-mode states of the highlighted design.',
         fontsize=9, color=INK2)
if args.note:
    fig.text(0.065, 0.861, args.note, fontsize=9, color='#8f1f1f')

def param_panel(ax, x, key, title, ylabel, bound_lo, bound_hi, log=False):
    for c in others:
        y = np.asarray(c[key], dtype=float)
        if log:
            y = np.where(y > 0, y, np.nan)
        ax.plot(x, y, color=BASELN, lw=1.0, zorder=2)
    yb = np.asarray(best[key], dtype=float)
    if log:
        yb = np.where(yb > 0, yb, np.nan)
    ax.plot(x, yb, color=BLUE, lw=2.0, marker='o', ms=4.5, zorder=3)
    ax.axhline(bound_lo, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.axhline(bound_hi, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.set_title(title, color=INK, loc='left')
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    if log:
        ax.set_yscale('log')
        ax.set_ylim(bound_lo * 0.6, bound_hi * 1.8)
    else:
        pad = 0.08 * (bound_hi - bound_lo)
        ax.set_ylim(bound_lo - pad, bound_hi + pad)
    ax.text(x[-1], bound_hi, ' search bound', color=MUTED, fontsize=7.5,
            ha='right', va='bottom')

axA = fig.add_subplot(gs[0, 0])
param_panel(axA, sites, 'masses', 'Masses', r'$m_i$', *args.mass_bounds)
axA.set_xlabel('site $i$')
h_best,  = axA.plot([], [], color=BLUE, lw=2, marker='o', ms=4.5, label=best_label)
h_other, = axA.plot([], [], color=BASELN, lw=1.0, label=others_label)
fig.legend(handles=[h_best, h_other], loc='upper right',
           bbox_to_anchor=(0.965, 0.955), fontsize=8, frameon=False,
           labelcolor=INK2, ncol=1)

axB = fig.add_subplot(gs[0, 1])
param_panel(axB, sites, 'wall', 'Wall springs', r'$k_{ii}$', *args.wall_bounds)
axB.set_xlabel('site $i$')

axC = fig.add_subplot(gs[0, 2])
param_panel(axC, bonds, 'coupling', 'Coupling springs (log)', r'$k_{i,i+1}$',
            *args.coupling_bounds, log=True)
axC.set_xlabel('bond $i$')

# ---- (d) mode shapes of the highlighted configuration -----------------------
axD = fig.add_subplot(gs[1, 0:2])
vmax = np.max(np.abs(modes))
im = axD.imshow(modes, cmap=DIV, norm=TwoSlopeNorm(0, -vmax, vmax),
                aspect='auto', interpolation='nearest')
axD.set_title('Normal-mode states of the highlighted design', color=INK, loc='left')
axD.set_xlabel('site $i$')
axD.set_ylabel('mode (sorted by frequency)')
axD.set_xticks(np.arange(n), sites)
axD.set_yticks(np.arange(n),
               [f'{k+1}   $\\omega$={freqs[k]:.2f}' for k in range(n)],
               fontsize=7.5)
axD.grid(False)
# white spacers between cells
axD.set_xticks(np.arange(-0.5, n, 1), minor=True)
axD.set_yticks(np.arange(-0.5, n, 1), minor=True)
axD.grid(which='minor', color=SURFACE, linewidth=1.6)
axD.tick_params(which='minor', length=0)
cb = fig.colorbar(im, ax=axD, pad=0.008, fraction=0.028)
cb.set_label('eigenvector component $A_{ij}$', color=INK2, fontsize=8)
cb.ax.tick_params(labelsize=7, color=MUTED, labelcolor=MUTED)
cb.outline.set_edgecolor(BASELN)

# ---- (e) frequency spectra ---------------------------------------------------
axE = fig.add_subplot(gs[1, 2])
rows = [('coupled (highlighted)', freqs, BLUE),
        ('uncoupled ref.', ref_freqs, ORANGE),
        ('baseline', base_freqs, BASELN)]
for y, (label, fr, color) in enumerate(rows):
    axE.eventplot(fr, lineoffsets=len(rows) - 1 - y, linelengths=0.55,
                  linewidths=2.0, colors=color)
axE.set_yticks([])
axE.set_ylim(-0.55, len(rows) - 0.35)
axE.set_xlabel(r'normal frequency $\omega$')
axE.set_title('Mode spectra', color=INK, loc='left')
axE.grid(axis='x')
axE.grid(axis='y', visible=False)
x_lo, x_hi = axE.get_xlim()
for y, (label, fr, color) in enumerate(rows):
    span = (fr.max() - fr.min()) / np.exp(np.mean(np.log(fr)))
    axE.text(x_lo + 0.02 * (x_hi - x_lo), len(rows) - 1 - y + 0.33,
             label, color=INK2, fontsize=8, ha='left', va='bottom')
    axE.text(x_hi - 0.02 * (x_hi - x_lo), len(rows) - 1 - y - 0.44,
             f'spread {span:.2f}', color=MUTED, fontsize=7, ha='right', va='bottom')

fig.savefig(args.stem + '.pdf')
fig.savefig(args.stem + '.png', dpi=160)
print(f'saved {args.stem}.pdf and .png')
