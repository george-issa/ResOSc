"""Why the trap ceiling was preferred, in one figure.

Left: the inspiral template |h(f)|^2 = A^2 f^(-7/3) for the benchmark
chirp mass, with the ISCO cutoff, the science band, and the trap range
marked — the template is alive across the entire band, so every bucket
placement receives the sweep.

Right: d_max of a 10-disc uncoupled stack as a function of its (common)
trap frequency, computed with the full pipeline for both readouts.  The
curve rises monotonically (~f^(5/6): signal force beta = k_trap*L grows
as f^2, beating the f^(-7/3) source) and is cut off by the published
100 kHz trap ceiling, not by any physical turnover — the corner is a
constraint, not a peak.  Dashed: hypothetical traps beyond the range.

Output: results-mc/template_ceiling.(pdf|png)
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ResOSc.physical import f_isco
from ResOSc.physopt import evaluate_design, MC_BENCH_SOLAR, BAND_HZ

AU = 1.496e11
N = 10
F_CEIL = 1.0e5
FISCO = f_isco(2.0**(6.0/5.0) * MC_BENCH_SOLAR)

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.4))

# ---- left: the template ---------------------------------------------------- #
f = np.geomspace(5e3, 3e6, 800)
h2 = np.where(f <= FISCO, f**(-7.0/3.0), np.nan)
h2 /= np.nanmax(h2)
axL.plot(f * 1e-3, h2, color='tab:purple', lw=1.6)
axL.axvspan(BAND_HZ[0] * 1e-3, BAND_HZ[1] * 1e-3, color='tab:green',
            alpha=0.12, label='science band 10–300 kHz')
axL.axvspan(1e4 * 1e-3, F_CEIL * 1e-3, color='tab:orange', alpha=0.18,
            label='trap range 10–100 kHz')
axL.axvline(FISCO * 1e-3, color='k', ls='--', lw=1.0)
axL.text(FISCO * 1e-3 * 0.88, 3e-6,
         r'$f_{\mathrm{ISCO}}\approx$' + f'{FISCO/1e6:.1f} MHz (merger)',
         rotation=90, va='bottom', ha='right', fontsize=8)
axL.text(3.2e1, 1.1e-2, r'$|\tilde h(f)|^2 \propto f^{-7/3}$',
         fontsize=10, color='tab:purple')
axL.text(1.2e1, 4.0e-6, 'template alive across the whole band:\n'
         'the chirp sweeps through every bucket', fontsize=8)
axL.set_xscale('log'); axL.set_yscale('log')
axL.set_xlabel('frequency [kHz]')
axL.set_ylabel(r'$|\tilde h(f)|^2$ [arb. units]')
axL.set_title(r'The source template ($\mathcal{M}_c = 10^{-3}\,M_\odot$)')
axL.legend(fontsize=8, loc='upper right')

# ---- right: d_max of the stack vs trap frequency --------------------------- #
f_traps = np.geomspace(1e4, 3e5, 25)
w = np.full(N, 1.0 / np.sqrt(N))
curves = {}
for ro in ('A', 'B'):
    curves[ro] = np.array([
        evaluate_design(np.full(N, ft), None, w, ro,
                        band=(BAND_HZ[0], BAND_HZ[1])) / AU
        for ft in f_traps])

for ro, color in (('A', 'tab:blue'), ('B', 'tab:red')):
    inside = f_traps <= F_CEIL
    axR.plot(f_traps[inside] * 1e-3, curves[ro][inside], color=color,
             lw=1.6, label=f'readout {ro} (allowed traps)')
    axR.plot(f_traps[~inside | np.isclose(f_traps, F_CEIL)] * 1e-3,
             curves[ro][~inside | np.isclose(f_traps, F_CEIL)],
             color=color, lw=1.2, ls='--', alpha=0.7,
             label='beyond published range' if ro == 'A' else None)
    i_ceil = np.argmin(np.abs(f_traps - F_CEIL))
    axR.plot(F_CEIL * 1e-3, curves[ro][i_ceil], 'o', color=color, ms=6)
    ratio = curves[ro][i_ceil] / curves[ro][0]
    axR.annotate(f'x{ratio:.0f} from 10 to 100 kHz',
                 xy=(F_CEIL * 1e-3, curves[ro][i_ceil]),
                 xytext=(-95, -14 if ro == 'A' else 10),
                 textcoords='offset points', fontsize=8, color=color)

axR.axvline(F_CEIL * 1e-3, color='k', ls='--', lw=1.0)
axR.text(F_CEIL * 1e-3 * 1.05, curves['A'][0] * 0.6,
         'trap ceiling\n(chosen corner)', fontsize=8)
axR.set_xscale('log'); axR.set_yscale('log')
axR.set_xlabel('common trap frequency of the 10-disc stack [kHz]')
axR.set_ylabel(r'$d_{\max}$ [AU]')
axR.set_title(r'Why the ceiling: $d_{\max}$ rises all the way to it')
axR.legend(fontsize=8, loc='lower right')

fig.suptitle('The falling template loses to the rising trap stiffness '
             r'($\beta = k_{\mathrm{trap}}L \propto f^2$)')
fig.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(f'results-mc/template_ceiling.{ext}', dpi=200)
print('saved results-mc/template_ceiling.pdf/.png')
for ro in ('A', 'B'):
    print(f'readout {ro}: d_max(10 kHz) = {curves[ro][0]:.2f} AU, '
          f'd_max(100 kHz) = {curves[ro][np.argmin(np.abs(f_traps-F_CEIL))]:.2f} AU, '
          f'ratio {curves[ro][np.argmin(np.abs(f_traps-F_CEIL))]/curves[ro][0]:.2f}')
