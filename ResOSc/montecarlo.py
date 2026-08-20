"""
Monte Carlo optimization over (masses, wall springs, coupling springs) to
simultaneously maximize minimax sensitivity and frequency spread.

Two strategies are provided and can be compared directly:

Random Monte Carlo (monte_carlo_optimize)
-----------------------------------------
Phase 1 – Screening (fast, n_samples draws):
    For each random configuration, compute a cheap sensitivity proxy
    (harmonic-L2 norm of peak sensitivities) and the frequency spread.
    Score every draw by a weighted combination of both normalized objectives,
    then shortlist the top n_refine candidates.

Phase 2 – Refinement (accurate, n_refine candidates):
    Run the full minimax weight-vector optimization (optimize_observable)
    on each shortlisted candidate to obtain exact normalized sensitivities.
    Extract the Pareto-optimal subset and report the best combined-score
    configuration.

Metropolis Monte Carlo (metropolis_optimize)
---------------------------------------------
Uses the Metropolis–Hastings algorithm with a physics-inspired free-energy
objective to *walk* the parameter space toward high-sensitivity / high-spread
configurations rather than sampling it blindly.

Energy function (inspired by statistical mechanics):
    E = -(α·log(proxy) + β·log(spread))
    where α = sensitivity_weight, β = 1 - sensitivity_weight.

This is analogous to the Helmholtz free energy F = -kT log Z: the chain
minimises E, so it is driven toward configurations where both proxy and
spread are large.  Working in log-space makes the objective scale-invariant
and well-behaved over multiple decades of parameter variation.

Proposals are Gaussian steps in log-space (multiplicative perturbations),
clamped to the parameter bounds.  A simulated-annealing schedule
(geometric cooling T_start → T_end) lets the chain explore broadly early
on and settle into the best region at the end.

After the walk the same Phase 2 refinement is applied to the top n_refine
accepted samples, so both strategies return identically structured dicts
and can be compared with plot_comparison.

Objectives
----------
* Sensitivity  : minimax worst-case sensitivity (raw, same units as Phase 1 proxy).
                 Proxy used in Phase 1 / Metropolis: 1 / sqrt(sum_i 1/b_i^2).
* Spread       : absolute or relative frequency spread of the normal modes.

Usage
-----
    from ResOSc.montecarlo import (monte_carlo_optimize, metropolis_optimize,
                                   plot_monte_carlo_results, plot_comparison)

    result_mc    = monte_carlo_optimize(n=10, n_samples=5000, n_refine=100)
    result_metro = metropolis_optimize( n=10, n_samples=5000, n_refine=100)

    plot_monte_carlo_results(result_mc,    savefig='mc_random.pdf')
    plot_monte_carlo_results(result_metro, savefig='mc_metro.pdf')
    plot_comparison(result_mc, result_metro, savefig='comparison.pdf')
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from .system import CoupledSystem
from .observables import sensitivity_profile, optimize_observable


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #

def _latex_available():
    """Return True if LaTeX + required packages are usable by Matplotlib."""
    import subprocess, shutil
    if not shutil.which('latex'):
        return False
    try:
        import tempfile, os
        with tempfile.TemporaryDirectory() as d:
            tex = os.path.join(d, 'test.tex')
            with open(tex, 'w') as f:
                f.write(r'\documentclass{article}'
                        r'\usepackage{type1ec}'
                        r'\begin{document}x\end{document}')
            r = subprocess.run(
                ['latex', '-interaction=nonstopmode', tex],
                cwd=d, capture_output=True, timeout=10,
            )
            return r.returncode == 0
    except Exception:
        return False


def _ensure_latex():
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica'],
        'text.latex.preamble': r'\usepackage{amsmath} \usepackage{amssymb}',
    })


def _build_and_solve(n, masses, wall, coupling, gamma, force_model, L, h0, force_vec):
    """Build, solve, and compute forces. Returns None if the system is unphysical."""
    sys = CoupledSystem(n)
    sys.set_masses(masses)
    sys.set_springs(wall, coupling)
    sys.set_damping(gamma)
    try:
        sys.build_H()
        sys.solve()
        if np.any(sys.wr <= 0):
            return None
        sys.compute_forces(L=L, h0=h0, force_model=force_model, force_vec=force_vec)
    except Exception:
        return None
    return sys


def _sensitivity_proxy(system):
    """Harmonic-L2 norm of peak sensitivities.

    This equals the exact minimax-optimal sensitivity when the eigenvectors
    are the identity (uncoupled limit) and serves as a fast proxy for the
    coupled case.  Always non-negative.
    """
    b = system.peak_sensitivities()
    if np.any(b <= 0):
        return 0.0
    return 1.0 / np.sqrt(np.sum(1.0 / b ** 2))


def _reference_scale(system):
    """Analytic minimax optimum of the system's uncoupled reference.

    The reference (same masses and wall springs, zero coupling) is diagonal
    with M-orthonormal modes v_i = e_i/sqrt(m_i), so the observable
    sensitivity of mode i is S_i = |w_i| * g_i with

        g_i = |f_i| / (2 gamma k_ii)

    (the mass cancels between the mode normalization and the response),
    and the minimax over unit weight vectors has the closed form
    1/sqrt(sum_i 1/g_i^2) — no eigensolve and no weight optimization
    needed.  Matches system.reference_system() + optimize_observable.
    """
    wall = np.diag(system.K)
    if np.any(wall <= 0):
        return 0.0
    g = np.abs(system.f) / (2.0 * system.gamma * wall)
    if np.any(g <= 0):
        return 0.0
    return 1.0 / np.sqrt(np.sum(1.0 / g ** 2))


def _normalized_proxy(system):
    """Sensitivity proxy divided by the analytic uncoupled-reference optimum.

    Raw peak sensitivities scale as 1/(gamma*omega^2), so the raw proxy grows
    without bound as oscillators get heavier and softer — the search pins at
    any box edge.  The reference scale carries the same global frequency
    factor, so the ratio is scale-invariant and measures only the structural
    gain from coupling: it tends to 1 in the uncoupled limit, and > 1 means
    the couplings genuinely help this layout.
    """
    ref = _reference_scale(system)
    if ref <= 0:
        return 0.0
    return _sensitivity_proxy(system) / ref


def _box_optimal_uncoupled_scale(n, mass_bounds, wall_bounds, gamma,
                                 force_model, L=1.0, h0=1.0, force_vec=None):
    """Minimax sensitivity of the best possible uncoupled design in the box.

    For a diagonal system with M-orthonormal modes the observable
    sensitivity of site i is S_i = |w_i| g_i with g_i = |f_i|/(2γ k_ii)
    (the mass cancels), so the minimax optimum 1/sqrt(sum_i 1/g_i^2) is
    separable and the box optimum follows from maximizing each site's g_i:

        strain  : g_i = L h0 k/(2γ k) = L h0/(2γ)   — parameter-free!
        uniform : g_i = L h0/(2γ k)                  → wall at floor
        custom  : g_i = h0 |force_vec_i|/(2γ k)      → wall at floor

    Under strain forcing every uncoupled design scores the same, so the
    box optimum is the universal constant L h0/(2γ √n).  Unlike the
    per-configuration reference, this is a CONSTANT for a given box and
    force model, so the search cannot raise its score by degrading the
    comparison system.  A coupled configuration scoring > 1 beats the
    best uncoupled design the same search box allows.
    """
    k_lo = wall_bounds[0]
    if force_model == 'strain':
        g = np.full(n, L * h0 / (2.0 * gamma))
    elif force_model == 'uniform':
        g = np.full(n, L * h0 / (2.0 * gamma * k_lo))
    elif force_model == 'custom':
        fv = np.abs(np.asarray(force_vec, dtype=np.float64))
        g = h0 * fv / (2.0 * gamma * k_lo)
    else:
        raise ValueError(f"Unknown force_model '{force_model}'.")
    if np.any(g <= 0):
        raise ValueError(
            "Box-optimal uncoupled scale is zero (a site receives no force); "
            "box normalization is undefined for this force profile.")
    return 1.0 / np.sqrt(np.sum(1.0 / g ** 2))


def _resolvable(system, min_gap_linewidths):
    """True if every adjacent mode gap is at least min_gap_linewidths
    resonance linewidths (2 gamma omega) apart.

    Both sensitivity metrics mis-score overlapping resonances (the
    isolated-peak metric ignores the overlap, the swept metric counts one
    merged response once per mode), so unconstrained searches drift into
    the unresolvable regime where their scores are artifacts.  Requiring
    clearly separated peaks makes the two metrics agree and closes both
    loopholes.
    """
    if min_gap_linewidths <= 0:
        return True
    f = np.sort(system.frequencies)
    gaps = np.diff(f)
    lw = 2.0 * system.gamma * f[:-1]
    return bool(np.all(gaps >= min_gap_linewidths * lw))


def _pareto_front(configs, key_sens, key_spread):
    """Return Pareto-optimal configs (maximize both sensitivity and spread)."""
    def sx(r):
        return r.get(key_sens) or 0.0

    pareto = []
    for r in configs:
        rx, ry = sx(r), r[key_spread]
        dominated = any(
            sx(o) >= rx and o[key_spread] >= ry and (sx(o) > rx or o[key_spread] > ry)
            for o in configs if o is not r
        )
        if not dominated:
            pareto.append(r)
    return pareto


# --------------------------------------------------------------------------- #
# Main function
# --------------------------------------------------------------------------- #

def monte_carlo_optimize(
    n,
    n_samples=5000,
    n_refine=100,
    gamma=0.01,
    force_model='strain',
    wall_bounds=(0.5, 3.0),
    coupling_bounds=(0.01, 1.0),
    mass_bounds=(0.3, 1.5),
    sensitivity_weight=0.5,
    spread_metric='relative',
    L=1.0,
    h0=1.0,
    force_vec=None,
    seed=42,
    min_gap_linewidths=3.0,
    verbose=True,
):
    """Monte Carlo search over (m, k_wall, k_coupling) for joint optimality.

    Finds system configurations that simultaneously achieve high minimax
    sensitivity *and* broad normal-mode frequency coverage.

    Parameters
    ----------
    n : int
        Number of oscillators.
    n_samples : int
        Random draws in the screening phase (Phase 1).
    n_refine : int
        Top candidates from screening to refine with full optimization
        (Phase 2).  Must be <= n_samples.
    gamma : float
        Uniform damping coefficient.
    force_model : {'strain', 'uniform', 'custom'}
        Driving force model (passed to compute_forces).
    wall_bounds : (float, float)
        (min, max) uniform range for each wall spring constant k_ii.
    coupling_bounds : (float, float)
        (min, max) uniform range for each coupling spring constant k_{i,i+1}.
    mass_bounds : (float, float)
        (min, max) uniform range for each oscillator mass m_i.
    sensitivity_weight : float in [0, 1]
        Relative weight given to sensitivity in the combined score.
        0.5 = equal weight; 1.0 = sensitivity only; 0.0 = spread only.
    spread_metric : {'absolute', 'relative'}
        Which frequency-spread metric to use as the second objective:
        'absolute' = omega_max - omega_min,
        'relative' = absolute / geometric-mean frequency.
    L, h0 : float
        Force scale parameters forwarded to compute_forces.
    force_vec : array_like or None
        Required when force_model='custom'.
    seed : int
        RNG seed for reproducibility.
    verbose : bool
        Print phase progress.

    Returns
    -------
    result : dict
        'top_combined'     : dict for the config with the highest combined score.
        'pareto_front'     : list of dicts for all Pareto-optimal configs.
        'all_refined'      : list of all n_refine refined result dicts, sorted
                             by combined score descending.
        'screening_scores' : ndarray, shape (n_valid, 2) — [proxy, spread] for
                             every valid Phase 1 draw (useful for visualizing
                             the full search cloud).
        'screening_params' : list of (masses, wall, coupling) for Phase 1 draws.
        'spread_metric'    : the spread_metric string used.

    Each config dict in 'all_refined' / 'pareto_front' / 'top_combined' has:
        masses, wall, coupling            — parameter arrays
        system, reference                 — solved CoupledSystem objects
        weights                           — optimal weight vector
        sensitivities                     — per-mode sensitivity profile
        min_sensitivity                   — worst-case sensitivity value
        normalized_min_sensitivity        — sensitivity / ref_scale
        ref_scale                         — reference normalization scale
        spread                            — full frequency_spread() dict
        spread_val                        — scalar spread for the chosen metric
        combined_score                    — final normalized combined score [0, 1]
    """
    rng = np.random.default_rng(seed)
    n_refine = min(n_refine, n_samples)
    box_scale = _box_optimal_uncoupled_scale(n, mass_bounds, wall_bounds,
                                             gamma, force_model, L, h0, force_vec)

    # ----------------------------------------------------------------------- #
    # Phase 1: Screening
    # ----------------------------------------------------------------------- #
    if verbose:
        print(f"Box-optimal uncoupled scale: {box_scale:.4f} "
              f"(score 1.0 = best possible uncoupled design)")
        print(f"Phase 1: screening {n_samples} random configurations...")

    screening_scores = []
    screening_params = []

    for _ in range(n_samples):
        masses   = rng.uniform(mass_bounds[0],     mass_bounds[1],     n)
        wall     = rng.uniform(wall_bounds[0],     wall_bounds[1],     n)
        coupling = rng.uniform(coupling_bounds[0], coupling_bounds[1], n - 1)

        sys = _build_and_solve(n, masses, wall, coupling, gamma,
                               force_model, L, h0, force_vec)
        if sys is None:
            continue
        if not _resolvable(sys, min_gap_linewidths):
            continue

        proxy  = _sensitivity_proxy(sys) / box_scale
        spread = sys.frequency_spread()[spread_metric]

        screening_scores.append([proxy, spread])
        screening_params.append((masses.copy(), wall.copy(), coupling.copy()))

    if not screening_scores:
        raise RuntimeError("No valid configurations found in Phase 1.")

    screening_scores = np.array(screening_scores)
    n_valid = len(screening_scores)

    if verbose:
        print(f"  Valid: {n_valid}/{n_samples}")

    # Normalize and combine
    s_range = screening_scores[:, 0].max() - screening_scores[:, 0].min()
    d_range = screening_scores[:, 1].max() - screening_scores[:, 1].min()

    norm_s = (screening_scores[:, 0] - screening_scores[:, 0].min()) / (s_range + 1e-30)
    norm_d = (screening_scores[:, 1] - screening_scores[:, 1].min()) / (d_range + 1e-30)

    combined_screen = sensitivity_weight * norm_s + (1.0 - sensitivity_weight) * norm_d
    top_idx = np.argsort(combined_screen)[::-1][:n_refine]

    # ----------------------------------------------------------------------- #
    # Phase 2: Refinement
    # ----------------------------------------------------------------------- #
    if verbose:
        print(f"Phase 2: refining top {len(top_idx)} candidates "
              f"with full minimax optimization...")

    refined = []
    for rank, idx in enumerate(top_idx):
        masses, wall, coupling = screening_params[idx]
        sys = _build_and_solve(n, masses, wall, coupling, gamma,
                               force_model, L, h0, force_vec)
        if sys is None:
            continue

        ref = sys.reference_system()
        ref.compute_forces(L=L, h0=h0, force_model=force_model, force_vec=force_vec)

        opt    = optimize_observable(sys, reference=ref)
        spread = sys.frequency_spread(reference=ref)

        refined.append({
            'masses':    masses,
            'wall':      wall,
            'coupling':  coupling,
            'system':    sys,
            'reference': ref,
            'weights':          opt['weights'],
            'sensitivities':    opt['sensitivities'],
            'min_sensitivity':  opt['min_sensitivity'],
            'normalized_min_sensitivity': opt.get('normalized_min_sensitivity'),
            'box_normalized_min_sensitivity': opt['min_sensitivity'] / box_scale,
            'ref_scale':        opt.get('ref_scale'),
            'spread':    spread,
            'spread_val': spread[spread_metric],
        })

        if verbose and (rank + 1) % 20 == 0:
            print(f"  Refined {rank + 1}/{len(top_idx)}...")

    if not refined:
        raise RuntimeError("No candidates survived Phase 2 refinement.")

    # Normalize refined objectives and assign combined scores.
    # Ranking uses the box-normalized minimax sensitivity: the yardstick is
    # the best possible uncoupled design in the box (a constant), so the
    # score cannot be gamed by degrading the comparison system.
    r_sens   = np.array([r['box_normalized_min_sensitivity'] for r in refined])
    r_spread = np.array([r['spread_val'] for r in refined])

    rs_range = r_sens.max()   - r_sens.min()
    rd_range = r_spread.max() - r_spread.min()

    norm_rs = (r_sens   - r_sens.min())   / (rs_range + 1e-30)
    norm_rd = (r_spread - r_spread.min()) / (rd_range + 1e-30)

    scores = sensitivity_weight * norm_rs + (1.0 - sensitivity_weight) * norm_rd

    for i, r in enumerate(refined):
        r['norm_sensitivity'] = float(norm_rs[i])
        r['norm_spread']      = float(norm_rd[i])
        r['combined_score']   = float(scores[i])

    refined.sort(key=lambda r: r['combined_score'], reverse=True)

    pareto = _pareto_front(
        refined,
        key_sens='box_normalized_min_sensitivity',
        key_spread='spread_val',
    )

    if verbose:
        best = refined[0]
        print(f"\nBest configuration (combined score = {best['combined_score']:.4f}):")
        print(f"  Box-norm. sensitivity : {best['box_normalized_min_sensitivity']:.4f}"
              f"  (raw {best['min_sensitivity']:.4f}, "
              f"own-ref {best['normalized_min_sensitivity']:.4f})")
        print(f"  Spread ({spread_metric:8s})    : {best['spread_val']:.4f}")
        print(f"  Masses    : {np.round(best['masses'],   3)}")
        print(f"  Wall k    : {np.round(best['wall'],     3)}")
        print(f"  Coupling k: {np.round(best['coupling'], 3)}")
        print(f"  Pareto-front size: {len(pareto)}")

    return {
        'top_combined':      refined[0],
        'pareto_front':      pareto,
        'all_refined':       refined,
        'screening_scores':  screening_scores,
        'screening_params':  screening_params,
        'spread_metric':     spread_metric,
        'objective':         'box-normalized',
        'box_scale':         box_scale,
    }


# --------------------------------------------------------------------------- #
# Visualization
# --------------------------------------------------------------------------- #

def plot_monte_carlo_results(result, savefig=None, use_latex=True):
    """Visualize Monte Carlo results in a two-panel figure.

    Left panel  — Scatter of all Phase 1 screening draws (grey), refined
                  candidates (blue), Pareto front (red line), and the best
                  combined-score point (gold star).
    Right panel — Per-mode sensitivity profile for the best configuration,
                  with the reference scale shown as a dashed line.

    Parameters
    ----------
    result : dict
        Return value of monte_carlo_optimize.
    savefig : str, optional
        If given, save the figure to this path as PDF.

    Returns
    -------
    fig : matplotlib Figure
    """
    _latex_rc = (
        {'text.usetex': True,  'font.family': 'sans-serif',
         'font.sans-serif': ['Helvetica'],
         'text.latex.preamble': r'\usepackage{amsmath} \usepackage{amssymb}'}
        if (use_latex and _latex_available()) else
        {'text.usetex': False}
    )

    with matplotlib.rc_context(_latex_rc):
        refined       = result['all_refined']
        pareto        = result['pareto_front']
        best          = result['top_combined']
        spread_metric = result['spread_metric']
        screen_scores = result['screening_scores']

        # Normalized-objective results store normalized screening scores, so
        # the refined candidates must be plotted on the same axis; older raw
        # results keep the raw axis.
        obj  = result.get('objective')
        skey = {'normalized':     'normalized_min_sensitivity',
                'box-normalized': 'box_normalized_min_sensitivity'
                }.get(obj, 'min_sensitivity')
        norm_obj = obj in ('normalized', 'box-normalized')
        sens_label = {'normalized':     r'Normalized Min.\ Sensitivity',
                      'box-normalized': r'Min.\ Sensitivity / Best Uncoupled'
                      }.get(obj, r'Min.\ Sensitivity')

        ref_x = [r['spread_val'] for r in refined]
        ref_y = [r[skey]         for r in refined]

        p_order = np.argsort([r['spread_val'] for r in pareto])
        px = [pareto[i]['spread_val'] for i in p_order]
        py = [pareto[i][skey]         for i in p_order]

        fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=125)

        # ---- Left: Pareto scatter ---------------------------------------- #
        ax = axes[0]
        ax.scatter(screen_scores[:, 1], screen_scores[:, 0],
                   s=8, alpha=0.25, color='grey', label=r'All Phase 1 draws', zorder=1)
        ax.scatter(ref_x, ref_y,
                   s=40, alpha=0.7, color='steelblue', label=r'Refined candidates', zorder=2)
        ax.plot(px, py, 'o-', color='firebrick', linewidth=2, markersize=8,
                label=r'Pareto front', zorder=3, alpha=0.90)
        ax.scatter([best['spread_val']], [best[skey]],
                   s=200, color='goldenrod', marker='*', zorder=4,
                   label=r'Best combined score')
        if norm_obj:
            ax.axhline(1.0, color='goldenrod', linewidth=1.5, linestyle='--',
                       label=r'Reference threshold', zorder=0)

        spread_label = (r'Frequency Spread (relative)' if spread_metric == 'relative'
                        else r'Frequency Spread (absolute)')
        ax.set_xlabel(spread_label, fontsize=18)
        ax.set_ylabel(sens_label, fontsize=18)
        ax.set_title(r'Monte Carlo: Sensitivity vs.\ Frequency Spread', fontsize=20)
        ax.tick_params(axis='both', direction='in', length=5, top=True, right=True, labelsize=14)
        ax.legend(frameon=False, fontsize=13)

        # ---- Right: sensitivity profile of best config ------------------- #
        ax2 = axes[1]
        freqs = best['system'].frequencies
        n     = best['system'].n
        x_pos = np.arange(n)

        ax2.bar(x_pos, best['sensitivities'], color='firebrick', alpha=0.8,
                label=r'Optimized observable')

        if best.get('ref_scale') is not None:
            ax2.axhline(best['ref_scale'], color='goldenrod', linewidth=2.0,
                        linestyle='--', label=r'Reference scale')

        ax2.set_xlabel(r'Mode', fontsize=18)
        ax2.set_ylabel(r'Sensitivity at Resonance', fontsize=18)
        ax2.set_title(r'Best Configuration: Sensitivity Profile', fontsize=20)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([fr'$\omega^*={freqs[i]:.2f}$' for i in range(n)],
                            fontsize=11, rotation=45, ha='right')
        ax2.tick_params(axis='both', direction='in', length=5, top=True, right=True, labelsize=14)
        ax2.legend(frameon=False, fontsize=13)

        plt.tight_layout()

        if savefig:
            fig.savefig(savefig, format='pdf', bbox_inches='tight', pad_inches=0.10)

        plt.show()
    return fig


# --------------------------------------------------------------------------- #
# Metropolis Monte Carlo
# --------------------------------------------------------------------------- #

def _energy(proxy, spread, sensitivity_weight):
    """Physics-inspired free energy for the Metropolis walk.

    Analogy: Helmholtz free energy F = -kT log Z.
    Both proxy and spread must be large to lower E (more negative = better).
    The log form is scale-invariant and handles multi-decade parameter ranges.

    Parameters
    ----------
    proxy : float
        Sensitivity proxy (_sensitivity_proxy output).
    spread : float
        Frequency spread scalar for the chosen metric.
    sensitivity_weight : float
        Weight α for sensitivity vs (1-α) for spread.

    Returns
    -------
    float — energy value (−∞ → +∞; lower is better).
    """
    if proxy <= 0 or spread <= 0:
        return np.inf
    alpha = sensitivity_weight
    beta  = 1.0 - sensitivity_weight
    return -(alpha * np.log(proxy) + beta * np.log(spread))


def _propose_step(params, bounds_list, step_size, rng):
    """Gaussian proposal in log-space, clamped to parameter bounds.

    A log-space step is equivalent to a multiplicative perturbation:
        p_new = p_old * exp(N(0, step_size))
    which naturally keeps the chain away from zero and treats parameter
    decades uniformly.

    Parameters
    ----------
    params : list of ndarray
        Current parameter arrays [masses, wall, coupling].
    bounds_list : list of (lo, hi)
        Allowed range for each parameter array.
    step_size : float
        Standard deviation of the log-space perturbation (~relative step).
    rng : numpy.random.Generator

    Returns
    -------
    proposed : list of ndarray  (same shapes as params)
    """
    proposed = []
    for p, (lo, hi) in zip(params, bounds_list):
        log_p_new = np.log(p) + rng.normal(0.0, step_size, size=p.shape)
        proposed.append(np.clip(np.exp(log_p_new), lo, hi))
    return proposed


def metropolis_optimize(
    n,
    n_samples=5000,
    n_refine=100,
    gamma=0.01,
    force_model='strain',
    wall_bounds=(0.5, 3.0),
    coupling_bounds=(0.01, 1.0),
    mass_bounds=(0.3, 1.5),
    sensitivity_weight=0.5,
    spread_metric='relative',
    L=1.0,
    h0=1.0,
    force_vec=None,
    seed=42,
    step_size=0.15,
    T_start=2.0,
    T_end=0.05,
    min_gap_linewidths=3.0,
    verbose=True,
):
    """Metropolis Monte Carlo search over (m, k_wall, k_coupling).

    Uses a physics-inspired free-energy objective to walk the parameter
    space toward configurations with high minimax sensitivity *and* broad
    normal-mode frequency coverage, rather than sampling blindly.

    The Metropolis criterion is the standard accept/reject rule:
        accept always  if ΔE ≤ 0  (new config is better)
        accept with prob exp(−ΔE/T)  if ΔE > 0  (uphill move)
    where T decays geometrically from T_start to T_end (simulated annealing).

    After the walk the top n_refine samples (by proxy score) are refined
    with the full minimax optimization — identical to Phase 2 of
    monte_carlo_optimize — so the two results are directly comparable.

    Parameters
    ----------
    n : int
        Number of oscillators.
    n_samples : int
        Total Metropolis steps (accepted + rejected combined).
    n_refine : int
        Number of top chain samples to refine in Phase 2.
    gamma : float
        Uniform damping coefficient.
    force_model : {'strain', 'uniform', 'custom'}
        Driving force model.
    wall_bounds, coupling_bounds, mass_bounds : (float, float)
        Parameter search ranges (same convention as monte_carlo_optimize).
    sensitivity_weight : float in [0, 1]
        α in the energy function E = -(α·log proxy + (1-α)·log spread).
    spread_metric : {'absolute', 'relative'}
        Which frequency-spread scalar to use.
    L, h0 : float
        Force scale parameters forwarded to compute_forces.
    force_vec : array_like or None
        Required when force_model='custom'.
    seed : int
        RNG seed.
    step_size : float
        Log-space Gaussian step size (~relative perturbation per step).
        Typical values: 0.05 (fine) – 0.30 (coarse).
    T_start : float
        Initial temperature (controls early exploration).
    T_end : float
        Final temperature (controls late exploitation).
    verbose : bool
        Print progress and diagnostics.

    Returns
    -------
    result : dict  — same structure as monte_carlo_optimize, with extra keys:
        'method'            : 'metropolis'
        'acceptance_rate'   : fraction of proposed steps that were accepted
        'energy_trajectory' : ndarray, energy at each accepted sample
    """
    rng = np.random.default_rng(seed)
    n_refine = min(n_refine, n_samples)
    bounds_list = [mass_bounds, wall_bounds, coupling_bounds]
    box_scale = _box_optimal_uncoupled_scale(n, mass_bounds, wall_bounds,
                                             gamma, force_model, L, h0, force_vec)

    # ----------------------------------------------------------------------- #
    # Initialise chain: draw random configs until we find a valid one
    # ----------------------------------------------------------------------- #
    current_params = None
    current_sys    = None
    current_E      = np.inf

    for _ in range(10 * n_samples):
        masses   = rng.uniform(mass_bounds[0],     mass_bounds[1],     n)
        wall     = rng.uniform(wall_bounds[0],     wall_bounds[1],     n)
        coupling = rng.uniform(coupling_bounds[0], coupling_bounds[1], n - 1)
        sys = _build_and_solve(n, masses, wall, coupling, gamma,
                               force_model, L, h0, force_vec)
        if sys is None:
            continue
        if not _resolvable(sys, min_gap_linewidths):
            continue
        proxy  = _sensitivity_proxy(sys) / box_scale
        spread = sys.frequency_spread()[spread_metric]
        E      = _energy(proxy, spread, sensitivity_weight)
        if np.isfinite(E):
            current_params = [masses.copy(), wall.copy(), coupling.copy()]
            current_sys    = sys
            current_E      = E
            break

    if current_params is None:
        raise RuntimeError("Could not find a valid initial configuration for the chain.")

    # ----------------------------------------------------------------------- #
    # Metropolis walk with simulated annealing
    # ----------------------------------------------------------------------- #
    if verbose:
        print(f"Box-optimal uncoupled scale: {box_scale:.4f} "
              f"(score 1.0 = best possible uncoupled design)")
        print(f"Metropolis walk: {n_samples} steps, "
              f"T: {T_start} → {T_end}, step_size: {step_size}...")

    screening_scores = []   # [proxy, spread] for every accepted sample
    screening_params = []   # (masses, wall, coupling) for every accepted sample
    energy_traj      = []

    n_accepted  = 0
    n_attempted = 0

    # Record the starting point
    screening_scores.append([_sensitivity_proxy(current_sys) / box_scale,
                              current_sys.frequency_spread()[spread_metric]])
    screening_params.append(tuple(p.copy() for p in current_params))
    energy_traj.append(current_E)

    # Best-ever state: the chain may find its best configuration mid-run and
    # drift away before the walk ends, so track it explicitly and force it
    # into the Phase 2 refinement pool.
    best_E    = current_E
    best_idx  = 0        # index into screening_scores / screening_params
    best_step = 0

    log_T_start = np.log(T_start)
    log_T_end   = np.log(T_end)

    for step in range(n_samples):
        # Anneal temperature geometrically
        frac = step / max(n_samples - 1, 1)
        T    = np.exp(log_T_start + frac * (log_T_end - log_T_start))

        # Propose a new configuration
        proposed = _propose_step(current_params, bounds_list, step_size, rng)
        masses_p, wall_p, coupling_p = proposed

        sys_p = _build_and_solve(n, masses_p, wall_p, coupling_p, gamma,
                                 force_model, L, h0, force_vec)
        n_attempted += 1

        if sys_p is None:
            continue  # invalid physics — reject silently

        if not _resolvable(sys_p, min_gap_linewidths):
            continue  # unresolvable mode pair — reject

        proxy_p  = _sensitivity_proxy(sys_p) / box_scale
        spread_p = sys_p.frequency_spread()[spread_metric]
        E_p      = _energy(proxy_p, spread_p, sensitivity_weight)

        if not np.isfinite(E_p):
            continue

        delta_E = E_p - current_E

        # Metropolis accept/reject
        if delta_E <= 0 or rng.random() < np.exp(-delta_E / T):
            current_params = [masses_p.copy(), wall_p.copy(), coupling_p.copy()]
            current_sys    = sys_p
            current_E      = E_p
            n_accepted    += 1

        # Record current state (accepted or not — for full trajectory)
        # We store the state *after* the decision (chain value)
        screening_scores.append([_sensitivity_proxy(current_sys) / box_scale,
                                  current_sys.frequency_spread()[spread_metric]])
        screening_params.append(tuple(p.copy() for p in current_params))
        energy_traj.append(current_E)

        if current_E < best_E:
            best_E    = current_E
            best_idx  = len(screening_params) - 1
            best_step = step + 1

        if verbose and (step + 1) % (n_samples // 5 or 1) == 0:
            rate = n_accepted / n_attempted if n_attempted else 0.0
            print(f"  Step {step+1}/{n_samples} | T={T:.4f} | "
                  f"E={current_E:.4f} | accept rate so far: {rate:.2%}")

    acceptance_rate = n_accepted / n_attempted if n_attempted else 0.0
    screening_scores = np.array(screening_scores)

    if verbose:
        print(f"  Total accepted: {n_accepted}/{n_attempted} "
              f"({acceptance_rate:.2%})")
        print(f"  Best energy: {best_E:.4f} at step {best_step} "
              f"(final chain energy: {current_E:.4f})")

    # ----------------------------------------------------------------------- #
    # Phase 2: Refinement (identical to random MC)
    # ----------------------------------------------------------------------- #
    if verbose:
        print(f"Phase 2: refining top {n_refine} chain samples "
              f"with full minimax optimization...")

    # Select top candidates by combined proxy score
    s_range = screening_scores[:, 0].max() - screening_scores[:, 0].min()
    d_range = screening_scores[:, 1].max() - screening_scores[:, 1].min()
    norm_s  = (screening_scores[:, 0] - screening_scores[:, 0].min()) / (s_range + 1e-30)
    norm_d  = (screening_scores[:, 1] - screening_scores[:, 1].min()) / (d_range + 1e-30)
    combined_screen = sensitivity_weight * norm_s + (1.0 - sensitivity_weight) * norm_d

    # Rejected steps re-record the current state, so the raw ranking is full
    # of duplicates — keep only the first occurrence of each distinct
    # configuration so the refinement pool is genuinely diverse.
    order = np.argsort(combined_screen)[::-1]
    seen, top_idx = set(), []
    for idx in order:
        key = tuple(p.tobytes() for p in screening_params[idx])
        if key in seen:
            continue
        seen.add(key)
        top_idx.append(idx)
        if len(top_idx) >= n_refine:
            break

    # The best-ever chain state must always be refined, even if its combined
    # proxy score did not rank it in the top n_refine.
    best_key = tuple(p.tobytes() for p in screening_params[best_idx])
    if best_key not in seen:
        if len(top_idx) >= n_refine:
            top_idx[-1] = best_idx
        else:
            top_idx.append(best_idx)

    refined = []
    for rank, idx in enumerate(top_idx):
        masses, wall, coupling = screening_params[idx]
        sys = _build_and_solve(n, masses, wall, coupling, gamma,
                               force_model, L, h0, force_vec)
        if sys is None:
            continue

        ref = sys.reference_system()
        ref.compute_forces(L=L, h0=h0, force_model=force_model, force_vec=force_vec)

        opt    = optimize_observable(sys, reference=ref)
        spread = sys.frequency_spread(reference=ref)

        refined.append({
            'masses':    masses,
            'wall':      wall,
            'coupling':  coupling,
            'system':    sys,
            'reference': ref,
            'weights':          opt['weights'],
            'sensitivities':    opt['sensitivities'],
            'min_sensitivity':  opt['min_sensitivity'],
            'normalized_min_sensitivity': opt.get('normalized_min_sensitivity'),
            'box_normalized_min_sensitivity': opt['min_sensitivity'] / box_scale,
            'ref_scale':        opt.get('ref_scale'),
            'spread':    spread,
            'spread_val': spread[spread_metric],
        })

        if verbose and (rank + 1) % 20 == 0:
            print(f"  Refined {rank+1}/{len(top_idx)}...")

    if not refined:
        raise RuntimeError("No candidates survived Phase 2 refinement.")

    # Ranking uses the box-normalized minimax sensitivity: the yardstick is
    # the best possible uncoupled design in the box (a constant), so the
    # score cannot be gamed by degrading the comparison system.
    r_sens   = np.array([r['box_normalized_min_sensitivity'] for r in refined])
    r_spread = np.array([r['spread_val']       for r in refined])
    rs_range = r_sens.max()   - r_sens.min()
    rd_range = r_spread.max() - r_spread.min()
    norm_rs  = (r_sens   - r_sens.min())   / (rs_range + 1e-30)
    norm_rd  = (r_spread - r_spread.min()) / (rd_range + 1e-30)
    scores   = sensitivity_weight * norm_rs + (1.0 - sensitivity_weight) * norm_rd

    for i, r in enumerate(refined):
        r['norm_sensitivity'] = float(norm_rs[i])
        r['norm_spread']      = float(norm_rd[i])
        r['combined_score']   = float(scores[i])

    refined.sort(key=lambda r: r['combined_score'], reverse=True)

    pareto = _pareto_front(refined, key_sens='box_normalized_min_sensitivity',
                           key_spread='spread_val')

    if verbose:
        best = refined[0]
        print(f"\nBest Metropolis configuration (combined score = {best['combined_score']:.4f}):")
        print(f"  Box-norm. sensitivity : {best['box_normalized_min_sensitivity']:.4f}"
              f"  (raw {best['min_sensitivity']:.4f}, "
              f"own-ref {best['normalized_min_sensitivity']:.4f})")
        print(f"  Spread ({spread_metric:8s})    : {best['spread_val']:.4f}")
        print(f"  Masses    : {np.round(best['masses'],   3)}")
        print(f"  Wall k    : {np.round(best['wall'],     3)}")
        print(f"  Coupling k: {np.round(best['coupling'], 3)}")
        print(f"  Pareto-front size: {len(pareto)}")
        print(f"  Acceptance rate: {acceptance_rate:.2%}")

    return {
        'top_combined':      refined[0],
        'pareto_front':      pareto,
        'all_refined':       refined,
        'screening_scores':  screening_scores,
        'screening_params':  screening_params,
        'spread_metric':     spread_metric,
        'method':            'metropolis',
        'objective':         'box-normalized',
        'box_scale':         box_scale,
        'acceptance_rate':   acceptance_rate,
        'energy_trajectory': np.array(energy_traj),
    }


# --------------------------------------------------------------------------- #
# Comparison visualization
# --------------------------------------------------------------------------- #

def plot_comparison(result_mc, result_metro, savefig=None, use_latex=True):
    """Compare random Monte Carlo vs Metropolis Monte Carlo side-by-side.

    Panel layout
    ------------
    Top-left  : Phase 1 / chain sample clouds for both methods on the same
                sensitivity-vs-spread axes.
    Top-right : Both Pareto fronts and best-score points overlaid.
    Bottom-left : Energy trajectory of the Metropolis chain (E vs step),
                  showing how the walk descends toward good configurations.
    Bottom-right: Sensitivity profiles of the best configuration from each
                  method, plotted side by side per mode.

    Parameters
    ----------
    result_mc : dict
        Return value of monte_carlo_optimize.
    result_metro : dict
        Return value of metropolis_optimize.
    savefig : str, optional
        Save figure to this path as PDF.

    Returns
    -------
    fig : matplotlib Figure
    """
    _latex_rc = (
        {'text.usetex': True,  'font.family': 'sans-serif',
         'font.sans-serif': ['Helvetica'],
         'text.latex.preamble': r'\usepackage{amsmath} \usepackage{amssymb}'}
        if (use_latex and _latex_available()) else
        {'text.usetex': False}
    )

    with matplotlib.rc_context(_latex_rc):
        sm   = result_mc['spread_metric']
        sc   = result_mc['screening_scores']
        sm2  = result_metro['screening_scores']

        best_mc    = result_mc['top_combined']
        best_metro = result_metro['top_combined']
        pareto_mc  = result_mc['pareto_front']
        pareto_mt  = result_metro['pareto_front']

        obj = result_mc.get('objective')
        if obj != result_metro.get('objective'):
            obj = None          # mixed objectives — fall back to raw axes
        skey = {'normalized':     'normalized_min_sensitivity',
                'box-normalized': 'box_normalized_min_sensitivity'
                }.get(obj, 'min_sensitivity')
        norm_obj = obj in ('normalized', 'box-normalized')
        sens_label = {'normalized':     r'Normalized Min.\ Sensitivity',
                      'box-normalized': r'Min.\ Sensitivity / Best Uncoupled'
                      }.get(obj, r'Min.\ Sensitivity')
        proxy_label = {'normalized':     r'Normalized Sensitivity Proxy',
                       'box-normalized': r'Sensitivity Proxy / Best Uncoupled'
                       }.get(obj, r'Sensitivity Proxy')

        def _pareto_xy(pareto):
            order = np.argsort([r['spread_val'] for r in pareto])
            return ([pareto[i]['spread_val'] for i in order],
                    [pareto[i][skey]         for i in order])

        px_mc, py_mc = _pareto_xy(pareto_mc)
        px_mt, py_mt = _pareto_xy(pareto_mt)

        energy_traj = result_metro.get('energy_trajectory', np.array([]))
        acc_rate    = result_metro.get('acceptance_rate', float('nan'))

        fig, axes = plt.subplots(2, 2, figsize=(18, 12), dpi=125)
        spread_label = (r'Frequency Spread (relative)' if sm == 'relative'
                        else r'Frequency Spread (absolute)')

        # ---- Top-left: sample clouds ------------------------------------- #
        ax = axes[0, 0]
        ax.scatter(sc[:, 1],  sc[:, 0],
                   s=6, alpha=0.20, color='grey',
                   label=r'Random MC (Phase 1)', zorder=1)
        ax.scatter(sm2[:, 1], sm2[:, 0],
                   s=6, alpha=0.25, color='steelblue',
                   label=r'Metropolis chain', zorder=2)
        ax.set_xlabel(spread_label, fontsize=16)
        ax.set_ylabel(proxy_label, fontsize=16)
        ax.set_title(r'Sample Clouds: Random vs.\ Metropolis', fontsize=18)
        ax.tick_params(axis='both', direction='in', length=5,
                       top=True, right=True, labelsize=12)
        ax.legend(frameon=False, fontsize=12)

        # ---- Top-right: Pareto fronts ------------------------------------ #
        ax = axes[0, 1]
        ref_x_mc = [r['spread_val'] for r in result_mc['all_refined']]
        ref_y_mc = [r[skey]         for r in result_mc['all_refined']]
        ref_x_mt = [r['spread_val'] for r in result_metro['all_refined']]
        ref_y_mt = [r[skey]         for r in result_metro['all_refined']]

        ax.scatter(ref_x_mc, ref_y_mc,
                   s=30, alpha=0.45, color='grey',      label=r'Random refined', zorder=1)
        ax.scatter(ref_x_mt, ref_y_mt,
                   s=30, alpha=0.45, color='steelblue', label=r'Metropolis refined', zorder=2)
        ax.plot(px_mc, py_mc, 'o-', color='dimgrey',   linewidth=2, markersize=7,
                label=r'Pareto (Random)',     zorder=3, alpha=0.9)
        ax.plot(px_mt, py_mt, 'o-', color='royalblue', linewidth=2, markersize=7,
                label=r'Pareto (Metropolis)', zorder=4, alpha=0.9)
        ax.scatter([best_mc['spread_val']],    [best_mc[skey]],
                   s=220, color='goldenrod', marker='*', zorder=5, label=r'Best (Random)')
        ax.scatter([best_metro['spread_val']], [best_metro[skey]],
                   s=220, color='tomato',    marker='*', zorder=5, label=r'Best (Metropolis)')
        if norm_obj:
            ax.axhline(1.0, color='goldenrod', linewidth=1.5, linestyle='--', zorder=0)
        ax.set_xlabel(spread_label, fontsize=16)
        ax.set_ylabel(sens_label, fontsize=16)
        ax.set_title(r'Pareto Fronts \& Best Configurations', fontsize=18)
        ax.tick_params(axis='both', direction='in', length=5,
                       top=True, right=True, labelsize=12)
        ax.legend(frameon=False, fontsize=11)

        # ---- Bottom-left: energy trajectory ------------------------------ #
        ax = axes[1, 0]
        if energy_traj.size:
            ax.plot(energy_traj, color='steelblue', linewidth=0.8, alpha=0.8)
            ax.set_xlabel(r'Chain Step', fontsize=16)
            ax.set_ylabel(r'Free Energy $E$', fontsize=16)
            ax.set_title(
                r'Metropolis Energy Trajectory'
                fr' (accept rate: {acc_rate:.1%})',
                fontsize=18)
            ax.tick_params(axis='both', direction='in', length=5,
                           top=True, right=True, labelsize=12)
        else:
            ax.set_visible(False)

        # ---- Bottom-right: sensitivity profiles -------------------------- #
        ax = axes[1, 1]
        n_modes = best_mc['system'].n
        x = np.arange(n_modes)
        w = 0.35

        ax.bar(x - w/2, best_mc['sensitivities'],    width=w,
               color='goldenrod', alpha=0.85, label=r'Best (Random)')
        ax.bar(x + w/2, best_metro['sensitivities'], width=w,
               color='tomato',    alpha=0.85, label=r'Best (Metropolis)')

        for rs, lbl, color in [
            (best_mc.get('ref_scale'),    r'Ref.\ scale (Random)',     'goldenrod'),
            (best_metro.get('ref_scale'), r'Ref.\ scale (Metropolis)', 'tomato'),
        ]:
            if rs is not None:
                ax.axhline(rs, color=color, linewidth=1.8, linestyle='--',
                           alpha=0.8, label=lbl)

        freqs = best_mc['system'].frequencies
        ax.set_xticks(x)
        ax.set_xticklabels([fr'$\omega^*={freqs[i]:.2f}$' for i in range(n_modes)],
                           fontsize=10, rotation=45, ha='right')
        ax.set_xlabel(r'Mode', fontsize=16)
        ax.set_ylabel(r'Sensitivity at Resonance', fontsize=16)
        ax.set_title(r'Best-Config Sensitivity Profiles', fontsize=18)
        ax.tick_params(axis='both', direction='in', length=5,
                       top=True, right=True, labelsize=12)
        ax.legend(frameon=False, fontsize=11)

        plt.tight_layout()

        if savefig:
            fig.savefig(savefig, format='pdf', bbox_inches='tight', pad_inches=0.10)

        plt.show()
    return fig
