"""
Mode shape visualization for coupled oscillator systems.

Provides stem plot grids and heatmaps to visualize the displacement
pattern of each normal mode across all oscillators.
"""

import numpy as np
import matplotlib.pyplot as plt


def _ensure_latex():
    """Enable LaTeX rendering for all text."""
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica'],
        'text.latex.preamble': r'\usepackage{amsmath} \usepackage{amssymb}',
    })


def mode_shapes_stem(system, savefig=None):
    """Plot a grid of stem plots showing each mode's displacement pattern.

    Each subplot shows the eigenvector components A_ij (displacement of
    oscillator j in mode i) as a stem plot vs oscillator index.

    Parameters
    ----------
    system : CoupledSystem
        A solved coupled oscillator system.
    savefig : str, optional
        If provided, save figure to this path.
    """
    _ensure_latex()
    n = system.n
    A = system.eigenvectors
    freqs = system.frequencies

    # Grid layout: aim for roughly 2 rows
    ncols = min(5, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), dpi=125)
    axes = np.atleast_2d(axes)

    colormap = plt.get_cmap('coolwarm', n)

    for i in range(n):
        row, col = divmod(i, ncols)
        ax = axes[row, col]
        osc_indices = np.arange(n)

        markerline, stemlines, baseline = ax.stem(
            osc_indices, A[i, :], linefmt='-', markerfmt='o', basefmt='k-'
        )
        stemlines.set_color(colormap(i))
        markerline.set_color(colormap(i))
        markerline.set_markersize(7)

        ax.set_title(fr'$\omega^*={freqs[i]:.3f}$', fontsize=18)
        ax.set_xlabel(r'Oscillator Index', fontsize=16)
        ax.set_ylabel(r'Displacement', fontsize=16)
        ax.set_xticks(osc_indices)
        ax.tick_params(axis='both', labelsize=14, direction='in',
                       top=True, right=True, length=4, width=1.0)
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.set_xlim(-0.5, n - 0.5)

    # Hide unused subplots
    for i in range(n, nrows * ncols):
        row, col = divmod(i, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle(r'Normal Mode Shapes', fontsize=24, y=1.02)
    plt.tight_layout()

    if savefig:
        fig.savefig(savefig, format='pdf', bbox_inches='tight', pad_inches=0.10)

    plt.show()


def mode_shapes_heatmap(system, savefig=None):
    """Plot a heatmap of the eigenvector matrix.

    Rows correspond to modes (labeled by normal frequency), columns to
    oscillator index. Color represents displacement amplitude (positive
    and negative).

    Parameters
    ----------
    system : CoupledSystem
        A solved coupled oscillator system.
    savefig : str, optional
        If provided, save figure to this path.
    """
    _ensure_latex()
    n = system.n
    A = system.eigenvectors
    freqs = system.frequencies

    fig, ax = plt.subplots(figsize=(10, 8), dpi=125)

    vmax = np.max(np.abs(A))
    im = ax.imshow(A, cmap='coolwarm', aspect='auto', vmin=-vmax, vmax=vmax)

    ax.set_xlabel(r'Oscillator Index', fontsize=22)
    ax.set_ylabel(r'Mode', fontsize=22)
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels([fr'$\omega^*={freqs[i]:.3f}$' for i in range(n)], fontsize=16)
    ax.tick_params(axis='both', direction='in', length=4, width=1.0, labelsize=16)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(r'Displacement', fontsize=20)
    cbar.ax.tick_params(labelsize=16)

    ax.set_title(r'Eigenvector Matrix (Mode Shapes)', fontsize=24)
    plt.tight_layout()

    if savefig:
        fig.savefig(savefig, format='pdf', bbox_inches='tight', pad_inches=0.10)

    plt.show()
