"""Portrait of the route-3 champion (coupling floor 0.02, swept 1.208):
parameters, normal-mode states, spectra, and the swept observable response
with every resonance above the universal uncoupled bound."""
import sys
sys.path.insert(0, '/Users/gissa/Documents/Nancy/ResOSc')
import pickle
import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['text.usetex'] = False
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from ResOSc.montecarlo import _build_and_solve

BLUE, ORANGE = '#2a78d6', '#eb6834'
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#898781'
GRID, BASELN, SURFACE = '#e1e0d9', '#c3c2b7', '#fcfcfb'
DIV = LinearSegmentedColormap.from_list(
    'bgr', ['#104281', '#2a78d6', '#86b6ef', '#f0efec',
            '#f0a5a4', '#e34948', '#8f1f1f'])
BOUND = 15.8114

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
with open('/Users/gissa/Documents/Nancy/ResOSc/results-mc/route3_sweep.pkl', 'rb') as f:
    d = pickle.load(f)
row = next(r for r in d['rows'] if r['floor'] == 0.02)
x = row['params']
m, kw, kc = x[:10], x[10:20], x[20:29]
s = _build_and_solve(10, m, kw, kc, 0.01, 'strain', 1.0, 1.0, None)
n = 10
sites, bonds = np.arange(1, 11), np.arange(1, 10)

# swept-optimal weights (re-derive so we can draw the response)
def swept_profile(system, w):
    w = np.asarray(w, float)
    S = np.empty(system.n)
    for i, om in enumerate(system.frequencies):
        denom = (system.wr - om**2) + 1j*(2*system.gamma*np.sqrt(system.wr)*om)
        S[i] = np.abs(np.dot(w, (system.q/denom) @ system.eigenvectors))
    return S

rng = np.random.default_rng(1)
best_v, best_w = np.inf, None
def neg(z):
    w = z/np.linalg.norm(z)
    return -np.min(swept_profile(s, w))
for _ in range(50):
    r = minimize(neg, rng.standard_normal(10), method='Nelder-Mead',
                 options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-10})
    if r.fun < best_v:
        best_v, best_w = r.fun, r.x/np.linalg.norm(r.x)
w = best_w
S_res = swept_profile(s, w)
print('min resonance response:', S_res.min().round(3),
      '=', (S_res.min()/BOUND).round(4), 'x bound')

# dense response curve
f_sorted = np.sort(s.frequencies)
om_grid = np.linspace(0.85*f_sorted[0], 1.06*f_sorted[-1], 6000)
b_c = (s.q[None, :] / ((s.wr[None, :] - om_grid[:, None]**2)
       + 1j*(2*s.gamma*np.sqrt(s.wr)[None, :]*om_grid[:, None])))
O = np.abs((b_c @ s.eigenvectors) @ w)

# modes, sorted ascending
order = np.argsort(s.frequencies)
freqs = s.frequencies[order]
modes = s.eigenvectors[order]
modes = modes / np.linalg.norm(modes, axis=1, keepdims=True)
for i in range(n):
    if modes[i, np.argmax(np.abs(modes[i]))] < 0:
        modes[i] *= -1
ref = s.reference_system(); ref.compute_forces(L=1.0, h0=1.0, force_model='strain')
ref_freqs = np.sort(ref.frequencies)

# ---- figure -----------------------------------------------------------------
fig = plt.figure(figsize=(11.5, 10.2))
gs = fig.add_gridspec(3, 3, hspace=0.52, wspace=0.30,
                      left=0.065, right=0.965, top=0.87, bottom=0.06,
                      height_ratios=[1.0, 1.15, 1.0])

fig.suptitle('The route-3 champion: a fully-coupled lab-feasible detector at +21%',
             fontsize=13, color=INK, x=0.065, ha='left', y=0.97)
fig.text(0.065, 0.925,
         f'Coupling floor 0.02 (all nine bonds active). Worst-mode swept response '
         f'{S_res.min():.2f} = {S_res.min()/BOUND:.3f} × the universal uncoupled '
         f'bound ({BOUND:.2f}); all mode gaps ≥ 3 linewidths.',
         fontsize=9, color=INK2)

def param_panel(ax, xv, yv, title, ylabel, lo, hi, log=False):
    ax.plot(xv, yv, color=BLUE, lw=2.0, marker='o', ms=4.5, zorder=3)
    ax.axhline(lo, color=MUTED, lw=0.8, ls=(0, (4, 3)))
    ax.axhline(hi, color=MUTED, lw=0.8, ls=(0, (4, 3)))
    ax.set_title(title, color=INK, loc='left')
    ax.set_ylabel(ylabel)
    ax.set_xticks(xv)
    if log:
        ax.set_yscale('log')
        ax.set_ylim(lo*0.6, hi*1.8)
    else:
        pad = 0.08*(hi-lo)
        ax.set_ylim(lo-pad, hi+pad)
    ax.text(xv[-1], hi, ' search bound', color=MUTED, fontsize=7.5,
            ha='right', va='bottom')

axA = fig.add_subplot(gs[0, 0])
param_panel(axA, sites, m, 'Masses', r'$m_i$', 0.1, 3.0)
axA.set_xlabel('site $i$')
axB = fig.add_subplot(gs[0, 1])
param_panel(axB, sites, kw, 'Wall springs', r'$k_{ii}$', 0.2, 6.0)
axB.set_xlabel('site $i$')
axC = fig.add_subplot(gs[0, 2])
param_panel(axC, bonds, kc, 'Coupling springs (log)', r'$k_{i,i+1}$',
            0.02, 1.0, log=True)
axC.set_xlabel('bond $i$')

# mode-shape heatmap
axD = fig.add_subplot(gs[1, 0:2])
vmax = np.max(np.abs(modes))
im = axD.imshow(modes, cmap=DIV, norm=TwoSlopeNorm(0, -vmax, vmax),
                aspect='auto', interpolation='nearest')
axD.set_title('Normal-mode states', color=INK, loc='left')
axD.set_xlabel('site $i$')
axD.set_ylabel('mode (sorted by frequency)')
axD.set_xticks(np.arange(n), sites)
axD.set_yticks(np.arange(n),
               [f'{k+1}   $\\omega$={freqs[k]:.2f}' for k in range(n)], fontsize=7.5)
axD.grid(False)
axD.set_xticks(np.arange(-0.5, n, 1), minor=True)
axD.set_yticks(np.arange(-0.5, n, 1), minor=True)
axD.grid(which='minor', color=SURFACE, linewidth=1.6)
axD.tick_params(which='minor', length=0)
cb = fig.colorbar(im, ax=axD, pad=0.008, fraction=0.028)
cb.set_label('eigenvector component $A_{ij}$', color=INK2, fontsize=8)
cb.ax.tick_params(labelsize=7, color=MUTED, labelcolor=MUTED)
cb.outline.set_edgecolor(BASELN)

# spectra
axE = fig.add_subplot(gs[1, 2])
rows_sp = [('coupled', np.sort(s.frequencies), BLUE),
           ('uncoupled twin', ref_freqs, ORANGE)]
for y, (label, fr, color) in enumerate(rows_sp):
    axE.eventplot(fr, lineoffsets=len(rows_sp)-1-y, linelengths=0.55,
                  linewidths=2.0, colors=color)
axE.set_yticks([])
axE.set_ylim(-0.55, len(rows_sp)-0.35)
axE.set_xlabel(r'normal frequency $\omega$')
axE.set_title('Mode spectra', color=INK, loc='left')
axE.grid(axis='x'); axE.grid(axis='y', visible=False)
x_lo, x_hi = axE.get_xlim()
for y, (label, fr, color) in enumerate(rows_sp):
    axE.text(x_lo + 0.02*(x_hi-x_lo), len(rows_sp)-1-y+0.33, label,
             color=INK2, fontsize=8, ha='left', va='bottom')

# swept response
axF = fig.add_subplot(gs[2, :])
axF.plot(om_grid, O, color=BLUE, lw=1.6, zorder=3)
axF.axhline(BOUND, color=ORANGE, lw=1.6, ls=(0, (5, 3)), zorder=2)
axF.text(om_grid[-1], BOUND, 'universal uncoupled bound  ',
         color=ORANGE, fontsize=8.5, ha='right', va='bottom')
imin = int(np.argmin(S_res))
axF.scatter(s.frequencies, S_res, s=26, color=BLUE, zorder=4,
            edgecolor=SURFACE, linewidth=1.2)
axF.annotate(f'worst mode: {S_res.min():.1f} = {S_res.min()/BOUND:.3f} × bound',
             (s.frequencies[imin], S_res[imin]), xytext=(10, 14),
             textcoords='offset points', color=INK2, fontsize=8.5,
             arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.8))
axF.set_yscale('log')
axF.set_xlim(om_grid[0], om_grid[-1])
axF.set_xlabel(r'driving frequency $\omega$')
axF.set_ylabel(r'observable response $|O(\omega)|$')
axF.set_title('Swept response of the optimal observable — every resonance clears the bound',
              color=INK, loc='left')

fig.savefig('/Users/gissa/Documents/Nancy/ResOSc/results-mc/route3_champion.pdf')
fig.savefig('/Users/gissa/Documents/Nancy/ResOSc/results-mc/route3_champion.png', dpi=150)
print('saved route3_champion.pdf/.png')
