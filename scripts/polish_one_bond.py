"""Local polish of the one-bond topology (route 2).

Starts from the best box-normalized configuration of a saved MC run,
keeps ONLY bond 2 (sites 2-3) as a free coupling, and locally optimizes
all 21 parameters (10 masses, 10 wall springs, k_bond2) in log space
against the sign-equalizer sensitivity objective, with an optional
resolvability penalty (adjacent mode gaps >= --min-gap linewidths).

Reproduces the session results of 2026-08-20:

    python3 polish_one_bond.py --min-gap 0 --out results-mc/one_bond_champion.npy
        -> iso 1.1009 (UNRESOLVABLE doublet 0.05 lw — metric artifact, do not quote)
    python3 polish_one_bond.py --min-gap 3 --out results-mc/one_bond_champion_resolvable.npy
        -> iso 1.0951 / swept 1.0179, min gap 3.00 lw (the quotable design)
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)
import sys, os
pass  # path handled by _ROOT header
import argparse
import pickle
import numpy as np
from scipy.optimize import minimize
from ResOSc.montecarlo import _build_and_solve, _box_optimal_uncoupled_scale

p = argparse.ArgumentParser(description='Polish the one-bond design.')
p.add_argument('--input', default='results-mc/mc_results_v5_sensonly.pkl',
               help='MC result pkl providing the starting configuration')
p.add_argument('--out', default='results-mc/one_bond_champion_resolvable.npy',
               help='Where to save the polished 21-parameter vector')
p.add_argument('--min-gap', type=float, default=3.0,
               help='Resolvability: min adjacent mode gap in linewidths (0 disables)')
p.add_argument('--restarts', type=int, default=12)
p.add_argument('--seed', type=int, default=0)
args = p.parse_args()

BOX = _box_optimal_uncoupled_scale(10, (0.1, 3.0), (0.2, 6.0), 0.01, 'strain')
LO = np.array([0.1] * 10 + [0.2] * 10 + [0.001])
HI = np.array([3.0] * 10 + [6.0] * 10 + [1.0])


def equalizer_val(system):
    """Sign-equalizer minimax: w = A^-1(sigma/b), best sign pattern.
    A feasible construction — every mode attains min_i S_i simultaneously."""
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


with open(args.input, 'rb') as f:
    d = pickle.load(f)
c = max(d['result_metro']['all_refined'],
        key=lambda x: x.get('box_normalized_min_sensitivity')
        or x.get('normalized_min_sensitivity') or 0)
x0 = np.concatenate([c['masses'], c['wall'], [c['coupling'][1]]])


def obj(logx):
    x = np.clip(np.exp(logx), LO, HI)
    kc = np.zeros(9)
    kc[1] = x[20]
    s = _build_and_solve(10, x[:10], x[10:20], kc, 0.01, 'strain', 1.0, 1.0, None)
    if s is None:
        return 0.0
    try:
        v = equalizer_val(s)
    except np.linalg.LinAlgError:
        return 0.0
    if args.min_gap > 0:
        r = min_gap_ratio(s)
        if r < args.min_gap:
            # smooth ramp guides infeasible starts into the feasible region
            v *= (max(r, 1e-3) / args.min_gap) ** 2
    return -v


rng = np.random.default_rng(args.seed)
best_v, best_x = np.inf, None
for t in range(args.restarts):
    z0 = np.log(x0) + (0 if t == 0 else rng.normal(0, 0.15, 21))
    r = minimize(obj, z0, method='Nelder-Mead',
                 options={'maxiter': 20000, 'xatol': 1e-8, 'fatol': 1e-8})
    if r.fun < best_v:
        best_v, best_x = r.fun, np.clip(np.exp(r.x), LO, HI)

kc = np.zeros(9)
kc[1] = best_x[20]
s = _build_and_solve(10, best_x[:10], best_x[10:20], kc, 0.01, 'strain', 1.0, 1.0, None)
v, rg = equalizer_val(s), min_gap_ratio(s)
np.set_printoptions(precision=3, suppress=True)
print(f'polished one-bond: minimax={v:.4f}  box-norm={v / BOX:.4f}  '
      f'min gap={rg:.2f} lw  feasible={args.min_gap <= 0 or rg >= args.min_gap}')
print('masses :', best_x[:10])
print('walls  :', best_x[10:20])
print('k_bond2: %.4f' % best_x[20])
np.save(args.out, best_x)
print('saved', args.out)
