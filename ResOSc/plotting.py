"""
Plotting utilities for forced oscillation amplitude curves.

Provides plots of normal-coordinate and spatial-coordinate amplitudes
vs driving frequency, matching the style of the original notebook.
"""

import numpy as np
import matplotlib.pyplot as plt


def setup_plot_style():
    """Configure matplotlib rcParams to match the project's style."""
    plt.rcParams.update({
        'font.size': 20,
        'axes.labelsize': 22,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 16,
        'axes.titlesize': 24,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica'],
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{amsmath} \usepackage{amssymb}',
    })


def plot_normal_amplitudes(system, w_range=(0.5, 3.5), resolution=250, savefig=None):
    """Plot amplitudes of normal coordinates vs driving frequency.

    Parameters
    ----------
    system : CoupledSystem
        A solved coupled oscillator system with forces computed.
    w_range : tuple of float
        (min, max) for squared driving frequencies.
    resolution : int
        Number of driving frequency points.
    savefig : str, optional
        If provided, save figure to this path.
    """
    n = system.n
    w_squared = np.linspace(w_range[0], w_range[1], resolution)
    b, _, w = system.compute_amplitudes(w_squared)

    colormap = plt.get_cmap('coolwarm', n)
    colors = [colormap(i) for i in range(n)]

    plt.figure(figsize=(10, 6), dpi=125)
    for i in range(n):
        plt.plot(w, b[f'Mode {i}'], '-', color=colors[i], linewidth=1.5,
                 label=fr'$\omega^*={system.frequencies[i]:.2f}$')

    plt.legend(frameon=False, fontsize=16, loc='upper left')
    plt.ylabel(r'Amplitude of Normal Coordinate', fontsize=22)
    plt.xlabel(r'$\omega$', fontsize=22)
    plt.xlim(np.sqrt(w_range[0]), np.sqrt(w_range[1]))
    plt.tick_params(axis='both', direction='in', which='major',
                    top=True, bottom=True, left=True, right=True,
                    length=6, width=1.0, labelsize=18)

    if savefig:
        plt.savefig(savefig, format='pdf', bbox_inches='tight', pad_inches=0.10)

    plt.show()


def plot_spatial_amplitudes(system, w_range=(0.5, 3.5), resolution=250, savefig=None):
    """Plot amplitudes of spatial coordinates (oscillators) vs driving frequency.

    Parameters
    ----------
    system : CoupledSystem
        A solved coupled oscillator system with forces computed.
    w_range : tuple of float
        (min, max) for squared driving frequencies.
    resolution : int
        Number of driving frequency points.
    savefig : str, optional
        If provided, save figure to this path.
    """
    n = system.n
    w_squared = np.linspace(w_range[0], w_range[1], resolution)
    _, x, w = system.compute_amplitudes(w_squared)

    colormap = plt.get_cmap('coolwarm', n)
    colors = [colormap(i) for i in range(n)]

    plt.figure(figsize=(10, 6), dpi=125)
    for i in range(n):
        plt.plot(w, x[f'Oscillator {i}'], '-', color=colors[i], linewidth=1.5,
                 label=f'$m={system.M[i]:.2f}$')

    plt.legend(frameon=False, fontsize=16)
    plt.ylabel(r'Amplitude of Spatial Coordinate', fontsize=22)
    plt.xlabel(r'$\omega$', fontsize=22)
    plt.xlim(np.sqrt(w_range[0]), np.sqrt(w_range[1]))
    plt.tick_params(axis='both', direction='in', which='major',
                    top=True, bottom=True, left=True, right=True,
                    length=6, width=1.0, labelsize=18)

    if savefig:
        plt.savefig(savefig, format='pdf', bbox_inches='tight', pad_inches=0.10)

    plt.show()