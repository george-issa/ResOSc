"""Route-3 summary figure: best achievable S/S_unc vs parasitic coupling floor."""
import sys
sys.path.insert(0, '/Users/gissa/Documents/Nancy/ResOSc')
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['text.usetex'] = False
import matplotlib.pyplot as plt

BLUE, ORANGE = '#2a78d6', '#eb6834'
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#898781'
GRID, BASELN, SURFACE = '#e1e0d9', '#c3c2b7', '#fcfcfb'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'text.color': INK, 'axes.labelcolor': INK2,
    'xtick.color': MUTED, 'ytick.color': MUTED,
    'axes.edgecolor': BASELN, 'axes.linewidth': 0.8,
    'grid.color': GRID, 'grid.linewidth': 0.6,
    'axes.grid': True, 'axes.axisbelow': True,
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
    'font.size': 10,
})

with open('/Users/gissa/Documents/Nancy/ResOSc/results-mc/route3_sweep.pkl', 'rb') as f:
    d = pickle.load(f)
rows = d['rows']
floors = np.array([r['floor'] for r in rows])
iso    = np.array([r['iso'] for r in rows])
swept  = np.array([r['swept'] for r in rows])

fig, ax = plt.subplots(figsize=(8.6, 5.6))
fig.subplots_adjust(left=0.09, right=0.97, top=0.90, bottom=0.12)

fig.suptitle('Best achievable sensitivity vs coupling floor',
             fontsize=13, color=INK, x=0.09, ha='left', y=0.965)

ax.axhline(1.0, color=MUTED, lw=1.2, ls=(0, (5, 3)), zorder=1)
ax.text(floors[0], 1.003, ' uncoupled bound', color=MUTED, fontsize=8, va='bottom')

ax.plot(floors, iso, color=BLUE, lw=2, marker='o', ms=6, zorder=3,
        label='one peak metric')
ax.plot(floors, swept, color=ORANGE, lw=2, marker='o', ms=6, zorder=3,
        label='swept metric (more conservative)')
ax.annotate('one peak', (floors[-1], iso[-1]), xytext=(6, 4),
            textcoords='offset points', color=INK2, fontsize=9)
ax.annotate('swept', (floors[-1], swept[-1]), xytext=(6, -10),
            textcoords='offset points', color=INK2, fontsize=9)

ax.set_xscale('log')
ax.set_xticks(floors, [str(f) for f in floors])
ax.set_xlim(floors[0] * 0.8, floors[-1] * 2.2)
ax.set_xlabel('coupling floor $k_\\mathrm{min}$ (all bonds $\\geq k_\\mathrm{min}$)')
ax.set_ylabel('best $S\\,/\\,S_\\mathrm{unc}$')
ax.legend(loc='upper right', frameon=False, fontsize=9, labelcolor=INK2)

fig.savefig('/Users/gissa/Documents/Nancy/ResOSc/results-mc/route3_curve.pdf')
fig.savefig('/Users/gissa/Documents/Nancy/ResOSc/results-mc/route3_curve.png', dpi=160)
print('saved route3_curve.pdf/.png')
