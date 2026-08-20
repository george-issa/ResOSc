"""Validation: does the >1 normalized sensitivity of the best v2 Metropolis
configuration survive overlapping-mode interference?

The project metric treats each resonance in isolation:
    S_i^iso(w) = |c_i| * |q_i| / (2 gamma omega_i^2),   c = A w

The physical observable at driving frequency omega is the coherent sum
    O(omega) = | sum_j w_j X_j(omega) |,
    X_j(omega) = sum_i A_ij * q_i / (omega_i^2 - omega^2 + 2i gamma omega_i omega)

Here we evaluate the *swept* metric  S_i^swept(w) = O(omega_i*)  at each
resonance, for the coupled winner and its uncoupled reference, both with the
stored/analytic weights and with weights re-optimized for the swept metric
(50-restart Nelder-Mead, same protocol as optimize_observable).
"""
import sys
sys.path.insert(0, '/Users/gissa/Documents/Nancy/ResOSc')
import pickle
import numpy as np
from scipy.optimize import minimize

PKL = sys.argv[1] if len(sys.argv) > 1 else \
    '/Users/gissa/Documents/Nancy/ResOSc/results-mc/mc_results_v2.pkl'
with open(PKL, 'rb') as f:
    d = pickle.load(f)

refined = d['result_metro']['all_refined']
best = max(refined, key=lambda c: c.get('normalized_min_sensitivity') or -np.inf)
sysb, refb = best['system'], best['reference']
w_stored = best['weights']


def swept_profile(system, w):
    """|O(omega_i*)| at each resonance — full complex response, all modes."""
    w = np.asarray(w, dtype=float)
    wr, om_star, gamma = system.wr, system.frequencies, system.gamma
    S = np.empty(system.n)
    for i, om in enumerate(om_star):
        denom = (wr - om**2) + 1j * (2 * gamma * np.sqrt(wr) * om)
        b_c = system.q / denom                 # complex modal amplitudes
        x = b_c @ system.eigenvectors          # spatial response (complex)
        S[i] = np.abs(np.dot(w, x))
    return S


def iso_profile(system, w):
    c = system.eigenvectors @ np.asarray(w, dtype=float)
    return np.abs(c) * np.abs(system.q) / (2 * system.gamma * system.wr)


def optimize_swept(system, seed=42, restarts=50):
    def neg_min(w_raw):
        w = w_raw / np.linalg.norm(w_raw)
        return -np.min(swept_profile(system, w))
    rng = np.random.default_rng(seed)
    best_val, best_w = np.inf, None
    for _ in range(restarts):
        w0 = rng.standard_normal(system.n)
        w0 /= np.linalg.norm(w0)
        res = minimize(neg_min, w0, method='Nelder-Mead',
                       options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-10})
        if res.fun < best_val:
            best_val, best_w = res.fun, res.x / np.linalg.norm(res.x)
    return -best_val, best_w


# ---- coupled winner ----------------------------------------------------------
iso_c   = iso_profile(sysb, w_stored)
swept_c = swept_profile(sysb, w_stored)
opt_c, w_c = optimize_swept(sysb)

# ---- uncoupled reference -----------------------------------------------------
b_ref = np.abs(refb.q) / (2 * refb.gamma * refb.wr)
ref_scale = 1.0 / np.sqrt(np.sum(1.0 / b_ref**2))          # analytic iso optimum
w_ref_analytic = (1.0 / b_ref)
w_ref_analytic /= np.linalg.norm(w_ref_analytic)
swept_r = swept_profile(refb, w_ref_analytic)
opt_r, w_r = optimize_swept(refb)

print('=== isolated-peak metric (what the search optimized) ===')
print(f'coupled  min_i S_i (stored w): {iso_c.min():.4f}')
print(f'reference analytic optimum   : {ref_scale:.4f}')
print(f'normalized (as reported)     : {iso_c.min()/ref_scale:.4f}')
print()
print('=== swept metric (full response with interference) ===')
print(f'coupled  min_i O(w_i*), stored w      : {swept_c.min():.4f}')
print(f'coupled  min_i O(w_i*), re-optimized w: {opt_c:.4f}')
print(f'reference min_i O(w_i*), analytic w   : {swept_r.min():.4f}')
print(f'reference min_i O(w_i*), re-optimized : {opt_r:.4f}')
print(f'normalized under swept metric         : {opt_c/opt_r:.4f}')
print()
print('per-mode profiles (coupled, sorted by frequency):')
order = np.argsort(sysb.frequencies)
print('  omega  :', np.round(sysb.frequencies[order], 4))
print('  iso    :', np.round(iso_c[order], 3))
print('  swept  :', np.round(swept_c[order], 3))
print('  swept* :', np.round(swept_profile(sysb, w_c)[order], 3))
print()
print('reference bare-frequency gaps vs linewidth:')
fr = np.sort(refb.frequencies)
print('  omega  :', np.round(fr, 4))
print('  gap/lw :', np.round(np.diff(fr) / (2 * refb.gamma * fr[:-1]), 1))
