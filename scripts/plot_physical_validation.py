"""Visualize the four validation tests of ResOSc/physical.py:
1) FDT thermal displacement peak vs the textbook value 4kBTQ/(m w0^3)
2) weight-independence of thermal-limited S_h on resonance
3) numerical matched-filter SNR vs the analytic narrow-linewidth formula
4) benchmark single-disc sqrt(S_h) curve vs the LSD target band
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)
import sys, os
pass  # path handled by _ROOT header
import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['text.usetex'] = False
import matplotlib.pyplot as plt

from ResOSc.physical import (PhysicalArray, LSD_BENCHMARK, lsd_array,
                             readout_B_noise, snr_inspiral, inspiral_A2,
                             KB, KPC)

BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
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
    'font.size': 9, 'axes.titlesize': 9.5,
})

fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.2))
fig.subplots_adjust(left=0.075, right=0.97, top=0.95, bottom=0.08,
                    hspace=0.42, wspace=0.28)

# ---- 1) FDT thermal peak ---------------------------------------------------- #
ax = axes[0, 0]
m, f0, g_hz, T = 1e-10, 1e5, 0.2, 300.0
a = PhysicalArray([m], [m*(2*np.pi*f0)**2], [], [g_hz], T, 1.0)
f = np.linspace(f0 - 6*g_hz, f0 + 6*g_hz, 1200)
Sx = a.S_O_thermal(f, np.array([1.0]))
w0, Q = 2*np.pi*f0, f0/g_hz
Sx_ref = 4*KB*T*Q/(m*w0**3)
ax.semilogy(f - f0, Sx, color=BLUE, lw=1.8, label='computed $S_x(f)$')
ax.scatter([0.0], [Sx_ref], s=70, color=ORANGE, zorder=5, marker='D',
           label=r'analytic $4k_BTQ/(m\omega_0^3)$')
ax.set_title('1)  FDT thermal peak', loc='left', color=INK)
ax.set_xlabel(r'$f - f_0$  [Hz]')
ax.set_ylabel(r'$S_x$  [m$^2$/Hz]')
ax.legend(frameon=False, fontsize=8, labelcolor=INK2)

# ---- 2) weight-independence of thermal S_h on resonance --------------------- #
ax = axes[0, 1]
b = LSD_BENCHMARK
n = 4
mm = np.full(n, b['mass_kg'])
f_traps = np.array([0.8, 0.95, 1.1, 1.25]) * b['f_trap_hz']
kk = mm * (2*np.pi*f_traps)**2
arr = PhysicalArray(mm, kk, np.full(n-1, 0.02*kk.min()),
                    np.full(n, b['gamma_hz']), b['T_kelvin'], b['L_m'])
fn, gn = arr.f_n[1], arr.Gamma_hz[1]
f = np.linspace(fn - 5*gn, fn + 5*gn, 1000)
rng = np.random.default_rng(0)
vals = []
for j in range(4):
    wv = rng.standard_normal(n); wv /= np.linalg.norm(wv)
    Sh = np.sqrt(arr.S_h(f, wv, 0.0))
    vals.append(np.sqrt(arr.S_h(np.array([fn]), wv, 0.0))[0])
    ax.plot(f - fn, Sh * 1e18, lw=1.5, alpha=0.85,
            label=f'random $\\mathbf{{w}}_{{{j+1}}}$')
ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.4f'))
ax.set_title('2)  thermal $\\sqrt{S_h}$ on resonance is weight-independent',
             loc='left', color=INK)
ax.set_xlabel(r'$f - f_n$  [Hz]')
ax.set_ylabel(r'$\sqrt{S_h}$  [$10^{-18}$ Hz$^{-1/2}$]')
ax.legend(frameon=False, fontsize=8, labelcolor=INK2)

# ---- 3) numerical SNR vs analytic narrow-linewidth formula ------------------ #
ax = axes[1, 0]
cold = dict(LSD_BENCHMARK, T_kelvin=1e-9)
a1 = lsd_array(1, benchmark=cold)
S_ro = float(readout_B_noise(b['lam_ro_m'], b['NA'], b['eta_det'], b['P_sc_W']))
Mc, r = 1e-3, 1.0*KPC
fn1, mu1, B1 = a1.f_n[0], a1.mu[0], a1.B[0]
g_ang = 2*np.pi*a1.Gamma_hz[0]
int_chi2 = 1.0/(4.0*mu1**2*(2*np.pi*fn1)**2*g_ang)
rho2_an = 4.0*inspiral_A2(Mc, r)*fn1**(-7.0/3.0)*B1**2*int_chi2/S_ro
# cumulative numerical integral
fgrid = a1.band_grid(0.5*fn1, 1.5*fn1)
Sh = a1.S_h(fgrid, np.array([1.0]), S_ro)
integrand = 4.0*inspiral_A2(Mc, r)*fgrid**(-7.0/3.0)/Sh
cum = np.concatenate([[0.0], np.cumsum(np.diff(fgrid)*0.5*(integrand[1:]+integrand[:-1]))])
ax.plot((fgrid - fn1)/a1.Gamma_hz[0], cum/rho2_an, color=BLUE, lw=1.8,
        label=r'cumulative numerical $\rho^2$')
ax.axhline(1.0, color=ORANGE, lw=1.5, ls=(0, (5, 3)),
           label='analytic per-mode formula (note Eq. 46)')
ax.set_xlim(-30, 30)
ax.set_title(f'3)  matched-filter SNR: numerical/analytic = '
             f'{np.sqrt(cum[-1]/rho2_an):.4f}', loc='left', color=INK)
ax.set_xlabel(r'$(f - f_n)/\Gamma_n$')
ax.set_ylabel(r'$\rho^2 / \rho^2_{\rm analytic}$')
ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc='center left')

# ---- 4) benchmark sqrt(S_h) vs LSD target ----------------------------------- #
ax = axes[1, 1]
a1 = lsd_array(1)
f = a1.band_grid(5e4, 2e5)
Sh = np.sqrt(a1.S_h(f, np.array([1.0]), 0.0))
ax.loglog(f, Sh, color=BLUE, lw=1.8, label='benchmark disc (300 K, thermal)')
ax.axhspan(1e-22, 1e-21, color=AQUA, alpha=0.18, lw=0)
ax.text(5.3e4, 3e-22, 'LSD target band', color=INK2, fontsize=8)
i = np.argmin(Sh)
ax.annotate(f'thermal-limited level: {Sh[i]:.1e} (flat across the band)',
            (f[i], Sh[i]), xytext=(8, -14),
            textcoords='offset points', color=INK2, fontsize=8)
ax.set_title('4)  benchmark $\\sqrt{S_h}$ — gap = unconfirmed parameters',
             loc='left', color=INK)
ax.set_xlabel(r'$f$  [Hz]')
ax.set_ylabel(r'$\sqrt{S_h}$  [Hz$^{-1/2}$]')
ax.legend(frameon=False, fontsize=8, labelcolor=INK2)

fig.savefig('results-mc/phys_sanity.pdf')
fig.savefig('results-mc/phys_sanity.png', dpi=150)
print('saved results-mc/phys_sanity.pdf/.png')
