"""
Mode shape visualization for coupled oscillator systems.

Provides stem plot grids and heatmaps to visualize the displacement
pattern of each normal mode across all oscillators.
"""

import numpy as np
import matplotlib.pyplot as plt


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
    n = system.n
    A = system.eigenvectors
    freqs = system.frequencies

    # Grid layout: aim for roughly 2 rows
    ncols = min(5, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows), dpi=125)
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
        markerline.set_markersize(5)

        ax.set_title(fr'$\omega^*={freqs[i]:.3f}$', fontsize=9)
        ax.set_xlabel('Oscillator', fontsize=7)
        ax.set_ylabel('Displacement', fontsize=7)
        ax.set_xticks(osc_indices)
        ax.tick_params(axis='both', labelsize=7, direction='in',
                       top=True, right=True, length=3)
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.set_xlim(-0.5, n - 0.5)

    # Hide unused subplots
    for i in range(n, nrows * ncols):
        row, col = divmod(i, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle('Normal Mode Shapes', fontsize=13, y=1.02)
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
    n = system.n
    A = system.eigenvectors
    freqs = system.frequencies

    fig, ax = plt.subplots(figsize=(8, 6), dpi=125)

    vmax = np.max(np.abs(A))
    im = ax.imshow(A, cmap='coolwarm', aspect='auto', vmin=-vmax, vmax=vmax)

    ax.set_xlabel('Oscillator Index', fontsize=11)
    ax.set_ylabel('Mode', fontsize=11)
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels([fr'$\omega^*={freqs[i]:.3f}$' for i in range(n)], fontsize=8)
    ax.tick_params(axis='both', direction='in', length=3)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Displacement', fontsize=10)

    ax.set_title('Eigenvector Matrix (Mode Shapes)', fontsize=12)
    plt.tight_layout()

    if savefig:
        fig.savefig(savefig, format='pdf', bbox_inches='tight', pad_inches=0.10)

    plt.show()
