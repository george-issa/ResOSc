"""Route 3: best achievable sensitivity vs parasitic coupling floor.

For each floor K_MIN, ALL nine bonds are constrained to k >= K_MIN (the
lab cannot switch parasitic couplings off).  Pipeline per floor:
  1. constrained Metropolis search (alpha=1, min gap 3 linewidths, step 0.2)
  2. floor-constrained polish of the winner (equalizer objective, 29 params)
  3. swept-metric evaluation of the polished design (conservative number)
Results -> results-mc/route3_sweep.pkl
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)
import sys
sys.path.insert(0, '/Users/gissa/Documents/Nancy/ResOSc')
import os, pickle
import numpy as np
from scipy.optimize import minimize
from ResOSc.montecarlo import (metropolis_optimize, _build_and_solve,
                               _box_optimal_uncoupled_scale)

os.chdir('/Users/gissa/Documents/Nancy/ResOSc')

FLOORS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
RGAP   = 3.0
BOX    = _box_optimal_uncoupled_scale(10, (0.1, 3.0), (0.2, 6.0), 0.01, 'strain')
print(f'box scale = {BOX:.4f}', flush=True)


def equalizer_val(system):
    A = system.eigenvectors
    b = np.abs(system.q) / (2 * system.gamma * system.wr)
    n = system.n
    sigs = np.array([[1 if (s >> i) & 1 else -1 for i in range(n - 1)] + [1]
                     for s in range(2 ** (n - 1))]).T
    W = np.linalg.solve(A, sigs / b[:, None])
    return 1.0 / np.linalg.norm(W, axis=0).min()


def min_gap_ratio(system):
    f = np.sort(system.frequencies)
    return float((np.diff(f) / (2 * system.gamma * f[:-1])).min())


def swept_profile(system, w):
    w = np.asarray(w, float)
    S = np.empty(system.n)
    for i, om in enumerate(system.frequencies):
        denom = (system.wr - om ** 2) + 1j * (2 * system.gamma * np.sqrt(system.wr) * om)
        S[i] = np.abs(np.dot(w, (system.q / denom) @ system.eigenvectors))
    return S


def optimize_swept(system, restarts=40, seed=1):
    rng = np.random.default_rng(seed)
    best = np.inf

    def neg(z):
        w = z / np.linalg.norm(z)
        return -np.min(swept_profile(system, w))

    for _ in range(restarts):
        r = minimize(neg, rng.standard_normal(system.n), method='Nelder-Mead',
                     options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-10})
        best = min(best, r.fun)
    return -best


rows = []
for F in FLOORS:
    print(f'\n===== floor K_MIN = {F} =====', flush=True)
    res = metropolis_optimize(
        n=10, n_samples=20000, n_refine=100, gamma=0.01, force_model='strain',
        wall_bounds=(0.2, 6.0), coupling_bounds=(F, 1.0), mass_bounds=(0.1, 3.0),
        sensitivity_weight=1.0, spread_metric='relative', seed=42,
        step_size=0.2, T_start=3.0, T_end=0.01,
        min_gap_linewidths=RGAP, verbose=True,
    )
    best = max(res['all_refined'],
               key=lambda c: c['box_normalized_min_sensitivity'])
    search_score = best['box_normalized_min_sensitivity']
    x0 = np.concatenate([best['masses'], best['wall'], best['coupling']])

    LO = np.array([0.1] * 10 + [0.2] * 10 + [F] * 9)
    HI = np.array([3.0] * 10 + [6.0] * 10 + [1.0] * 9)

    def obj(logx):
        x = np.clip(np.exp(logx), LO, HI)
        s = _build_and_solve(10, x[:10], x[10:20], x[20:], 0.01,
                             'strain', 1.0, 1.0, None)
        if s is None:
            return 0.0
        try:
            v = equalizer_val(s)
        except Exception:
            return 0.0
        r = min_gap_ratio(s)
        return -v * (1.0 if r >= RGAP else (max(r, 1e-3) / RGAP) ** 2)

    rng = np.random.default_rng(0)
    best_v, best_x = np.inf, None
    for t in range(8):
        z0 = np.log(x0) + (0 if t == 0 else rng.normal(0, 0.12, 29))
        r = minimize(obj, z0, method='Nelder-Mead',
                     options={'maxiter': 20000, 'xatol': 1e-8, 'fatol': 1e-8})
        if r.fun < best_v:
            best_v, best_x = r.fun, np.clip(np.exp(r.x), LO, HI)

    s = _build_and_solve(10, best_x[:10], best_x[10:20], best_x[20:], 0.01,
                         'strain', 1.0, 1.0, None)
    iso = equalizer_val(s)
    gap = min_gap_ratio(s)
    swept = optimize_swept(s)
    rows.append({
        'floor': F,
        'search_score': search_score,
        'iso': iso / BOX,
        'swept': swept / BOX,
        'min_gap_lw': gap,
        'params': best_x,
    })
    print(f'floor {F}: search={search_score:.4f}  polished iso={iso/BOX:.4f}  '
          f'swept={swept/BOX:.4f}  min_gap={gap:.2f} lw', flush=True)

with open('results-mc/route3_sweep.pkl', 'wb') as f:
    pickle.dump({'box_scale': BOX, 'rgap': RGAP, 'rows': rows}, f)

print('\n===== ROUTE 3 SUMMARY =====')
print('floor   search   iso     swept   min_gap')
for r in rows:
    print(f"{r['floor']:<7} {r['search_score']:.4f}  {r['iso']:.4f}  "
          f"{r['swept']:.4f}  {r['min_gap_lw']:.2f}")
