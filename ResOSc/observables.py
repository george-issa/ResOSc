"""
Observable analysis and optimization for coupled oscillator systems.

An observable is a linear combination O = sum_j w_j x_j of oscillator
displacements. This module analyzes which observables can detect all
normal modes and finds optimal weight vectors that maximize worst-case
sensitivity across all modes.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def _ensure_latex():
    """Enable LaTeX rendering for all text."""
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica'],
        'text.latex.preamble': r'\usepackage{amsmath} \usepackage{amssymb}',
    })


def coupling_matrix(system):
    """Compute the coupling matrix between oscillators and modes.

    For an observable monitoring oscillator j, the coupling to mode i is
    C[j, i] = A[i, j], where A is the eigenvector matrix.

    Parameters
    ----------
    system : CoupledSystem
        A solved coupled oscillator system.

    Returns
    -------
    C : ndarray, shape (n, n)
        Coupling matrix. C[j, i] = coupling of oscillator j to mode i.
    """
    return system.eigenvectors.T.copy()


def coupling_heatmap(system, savefig=None):
    """Plot a heatmap of the coupling matrix |C[j, i]|.

    Parameters
    ----------
    system : CoupledSystem
        A solved coupled oscillator system.
    savefig : str, optional
        If provided, save figure to this path.
    """
    _ensure_latex()
    n = system.n
    C = coupling_matrix(system)
    freqs = system.frequencies

    fig, ax = plt.subplots(figsize=(10, 8), dpi=125)
    im = ax.imshow(np.abs(C), cmap='viridis', aspect='auto')

    ax.set_xlabel(r'Mode', fontsize=22)
    ax.set_ylabel(r'Oscillator', fontsize=22)
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels([fr'$\omega^*={freqs[i]:.2f}$' for i in range(n)],
                       fontsize=14, rotation=45, ha='right')
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels([fr'Osc $\,{j}$' for j in range(n)], fontsize=16)
    ax.tick_params(axis='both', direction='in', length=4, width=1.0, labelsize=16)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(r'$|C_{j,i}|$ (coupling strength)', fontsize=20)
    cbar.ax.tick_params(labelsize=16)

    ax.set_title(r'Oscillator--Mode Coupling Matrix', fontsize=24)
    plt.tight_layout()

    if savefig:
        fig.savefig(savefig, format='pdf', bbox_inches='tight', pad_inches=0.10)

    plt.show()
    return C


def find_minimum_observables(system, threshold=1e-10):
    """Find the minimum number of single-oscillator observables needed
    to detect all normal modes.

    An oscillator j detects mode i if |C[j, i]| > threshold.

    Parameters
    ----------
    system : CoupledSystem
        A solved coupled oscillator system.
    threshold : float
        Minimum coupling strength to consider a mode "detected".

    Returns
    -------
    result : dict
        'single_oscillator_coverage': dict mapping oscillator index to list
            of modes it can detect.
        'full_coverage_oscillators': list of oscillator indices that can
            individually detect all modes.
        'minimum_set': list of oscillator indices forming a minimum covering set.
        'summary': str with readable summary.
    """
    n = system.n
    C = coupling_matrix(system)

    # Coverage of each oscillator
    coverage = {}
    for j in range(n):
        detected = [i for i in range(n) if np.abs(C[j, i]) > threshold]
        coverage[j] = detected

    # Oscillators that see all modes
    full_coverage = [j for j in range(n) if len(coverage[j]) == n]

    # Greedy set cover for minimum set
    remaining_modes = set(range(n))
    minimum_set = []
    while remaining_modes:
        # Pick oscillator covering most remaining modes
        best_j = max(range(n), key=lambda j: len(remaining_modes & set(coverage[j])))
        minimum_set.append(best_j)
        remaining_modes -= set(coverage[best_j])

    summary_lines = []
    summary_lines.append(f"Number of oscillators: {n}")
    summary_lines.append(f"Number of normal modes: {n}")
    summary_lines.append(f"Detection threshold: {threshold:.1e}")
    summary_lines.append("")

    if full_coverage:
        summary_lines.append(
            f"Oscillators that can individually detect ALL modes: {full_coverage}"
        )
        summary_lines.append(
            f"=> A SINGLE observable suffices! Monitor any of these oscillators."
        )
    else:
        summary_lines.append("No single oscillator can detect all modes.")
        summary_lines.append(
            f"Minimum set of oscillators needed: {minimum_set} ({len(minimum_set)} observables)"
        )

    summary_lines.append("")
    summary_lines.append("Coverage per oscillator (number of modes detected):")
    for j in range(n):
        summary_lines.append(f"  Oscillator {j}: {len(coverage[j])}/{n} modes")

    return {
        'single_oscillator_coverage': coverage,
        'full_coverage_oscillators': full_coverage,
        'minimum_set': minimum_set,
        'summary': '\n'.join(summary_lines),
    }


def sensitivity_profile(system, weights):
    """Compute the sensitivity of an observable to each mode at resonance.

    For weight vector w, the sensitivity to mode i is:
        S_i = |c_i| * b_i_max = |sum_j w_j A[i,j]| * |q_i| / (2 gamma omega*_i^2)

    Parameters
    ----------
    system : CoupledSystem
        A solved coupled oscillator system (must have gamma > 0).
    weights : array_like, shape (n,)
        Weight vector defining the observable O = sum_j w_j x_j.

    Returns
    -------
    S : ndarray, shape (n,)
        Sensitivity to each mode at resonance.
    """
    w = np.asarray(weights)
    # Coupling: c_i = sum_j w_j * A[i, j]
    c = system.eigenvectors @ w
    peaks = system.peak_sensitivities()
    return np.abs(c) * peaks


def optimize_observable(system):
    """Find the weight vector that maximizes worst-case sensitivity.

    Solves: max_{||w||=1} min_i S_i(w)

    Parameters
    ----------
    system : CoupledSystem
        A solved coupled oscillator system (must have gamma > 0).

    Returns
    -------
    result : dict
        'weights': optimal weight vector (normalized).
        'sensitivities': sensitivity profile at optimal weights.
        'min_sensitivity': the worst-case sensitivity achieved.
    """
    n = system.n
    peaks = system.peak_sensitivities()

    def neg_min_sensitivity(w_raw):
        w = w_raw / np.linalg.norm(w_raw)
        c = system.eigenvectors @ w
        S = np.abs(c) * peaks
        return -np.min(S)

    # Try multiple random starting points and keep the best
    best_result = None
    best_val = np.inf
    rng = np.random.default_rng(42)

    for _ in range(50):
        w0 = rng.standard_normal(n)
        w0 /= np.linalg.norm(w0)
        res = minimize(neg_min_sensitivity, w0, method='Nelder-Mead',
                       options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-10})
        if res.fun < best_val:
            best_val = res.fun
            best_result = res

    w_opt = best_result.x / np.linalg.norm(best_result.x)
    S_opt = sensitivity_profile(system, w_opt)

    return {
        'weights': w_opt,
        'sensitivities': S_opt,
        'min_sensitivity': np.min(S_opt),
    }


def compare_observables(system, savefig=None):
    """Compare best single-oscillator observable vs optimized weights.

    Plots the sensitivity profile S_i for each mode for:
    1. The best single oscillator (highest worst-case sensitivity)
    2. The optimized linear combination

    Parameters
    ----------
    system : CoupledSystem
        A solved coupled oscillator system (must have gamma > 0).
    savefig : str, optional
        If provided, save figure to this path.

    Returns
    -------
    result : dict
        'best_single_oscillator': index of best single oscillator.
        'best_single_sensitivities': its sensitivity profile.
        'optimal_weights': optimized weight vector.
        'optimal_sensitivities': optimized sensitivity profile.
    """
    n = system.n
    freqs = system.frequencies

    # Find best single oscillator
    best_j = None
    best_single_min = -np.inf
    best_single_S = None
    for j in range(n):
        w = np.zeros(n)
        w[j] = 1.0
        S = sensitivity_profile(system, w)
        if np.min(S) > best_single_min:
            best_single_min = np.min(S)
            best_j = j
            best_single_S = S

    # Optimized observable
    opt = optimize_observable(system)

    # Plot comparison
    _ensure_latex()
    fig, ax = plt.subplots(figsize=(12, 7), dpi=125)
    x_pos = np.arange(n)
    width = 0.35

    bars1 = ax.bar(x_pos - width / 2, best_single_S, width,
                   label=fr'Best single oscillator (Osc {best_j})',
                   color='steelblue', alpha=0.8)
    bars2 = ax.bar(x_pos + width / 2, opt['sensitivities'], width,
                   label=r'Optimized linear combination',
                   color='firebrick', alpha=0.8)

    ax.set_xlabel(r'Mode', fontsize=22)
    ax.set_ylabel(r'Sensitivity at Resonance', fontsize=22)
    ax.set_title(r'Observable Sensitivity Comparison', fontsize=24)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([fr'$\omega^*={freqs[i]:.2f}$' for i in range(n)],
                       fontsize=14, rotation=45, ha='right')
    ax.tick_params(axis='both', direction='in', length=5, width=1.0,
                   top=True, right=True, labelsize=18)
    ax.legend(frameon=False, fontsize=18)
    plt.tight_layout()

    if savefig:
        fig.savefig(savefig, format='pdf', bbox_inches='tight', pad_inches=0.10)

    plt.show()

    return {
        'best_single_oscillator': best_j,
        'best_single_sensitivities': best_single_S,
        'optimal_weights': opt['weights'],
        'optimal_sensitivities': opt['sensitivities'],
    }
