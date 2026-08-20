"""Swept-metric validation of every nominal winner (iso-normalized > 1)
from the v2 and v3 Metropolis runs. For each config, re-optimize weights
under the full swept response for both the coupled system and its uncoupled
reference (identical 50-restart Nelder-Mead), and report the swept-normalized
sensitivity. Results saved to results-mc/swept_validation.pkl.
"""
import sys, os
sys.path.insert(0, '/Users/gissa/Documents/Nancy/ResOSc')
os.chdir('/Users/gissa/Documents/Nancy/ResOSc')
import pickle
import numpy as np
from scipy.optimize import minimize


def swept_profile(system, w):
    w = np.asarray(w, dtype=float)
    wr, om_star, gamma = system.wr, system.frequencies, system.gamma
    S = np.empty(system.n)
    for i, om in enumerate(om_star):
        denom = (wr - om**2) + 1j * (2 * gamma * np.sqrt(wr) * om)
        b_c = system.q / denom
        x = b_c @ system.eigenvectors
        S[i] = np.abs(np.dot(w, x))
    return S


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


rows = []
for run, path in [('v2', 'results-mc/mc_results_v2.pkl'),
                  ('v3', 'results-mc/mc_results_v3_extended.pkl')]:
    with open(path, 'rb') as f:
        d = pickle.load(f)
    for c in d['result_metro']['all_refined']:
        if (c.get('normalized_min_sensitivity') or 0) <= 1.0:
            continue
        opt_c, w_c = optimize_swept(c['system'])
        opt_r, w_r = optimize_swept(c['reference'])
        rows.append({
            'run': run,
            'iso_norm': c['normalized_min_sensitivity'],
            'swept_coupled': opt_c,
            'swept_reference': opt_r,
            'swept_norm': opt_c / opt_r,
            'spread': c['spread_val'],
            'masses': c['masses'], 'wall': c['wall'], 'coupling': c['coupling'],
            'swept_weights': w_c,
        })
        print(f"{run}  iso={rows[-1]['iso_norm']:.4f}  "
              f"swept={rows[-1]['swept_norm']:.4f}  "
              f"({opt_c:.2f} / {opt_r:.2f})  spread={rows[-1]['spread']:.2f}",
              flush=True)

with open('results-mc/swept_validation.pkl', 'wb') as f:
    pickle.dump(rows, f)

survivors = [r for r in rows if r['swept_norm'] > 1.0]
print(f"\n{len(survivors)}/{len(rows)} nominal winners survive the swept metric")
for r in sorted(survivors, key=lambda r: -r['swept_norm']):
    print(f"  {r['run']}  swept_norm={r['swept_norm']:.4f}  spread={r['spread']:.2f}")
