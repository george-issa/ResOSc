"""
Core physics for the coupled oscillator system.

Handles system setup, eigenvalue decomposition via LAPACK, damped forced
oscillation amplitudes, and peak sensitivity computation.
"""

import numpy as np
from scipy.linalg import lapack
from scipy.linalg.blas import dgemv, dgemm


class CoupledSystem:
    """A system of n coupled oscillators with optional damping.

    Parameters
    ----------
    n : int
        Number of oscillators.

    Attributes
    ----------
    n : int
        Number of oscillators.
    M : ndarray, shape (n,)
        Mass array.
    K : ndarray, shape (n, n)
        Springs matrix. K[i, i] = wall spring for oscillator i,
        K[i, j] = coupling spring between oscillators i and j.
    gamma : float
        Uniform damping ratio (dimensionless).
    H : ndarray, shape (n, n)
        Dynamical matrix (K/M structure).
    wr : ndarray, shape (n,)
        Squared normal frequencies (eigenvalues of H), sorted descending.
    eigenvectors : ndarray, shape (n, n)
        Eigenvector matrix A. Row i is the eigenvector for mode i.
    frequencies : ndarray, shape (n,)
        Normal frequencies omega*_i = sqrt(wr_i).
    f : ndarray, shape (n,)
        Driving force amplitudes on each oscillator.
    q : ndarray, shape (n,)
        Generalized forces on each normal mode (q = A @ f).
    """

    def __init__(self, n):
        self.n = n
        self.M = np.zeros(n)
        self.K = np.zeros((n, n))
        self.gamma = 0.0
        self.H = None
        self.wr = None
        self.eigenvectors = None
        self.frequencies = None
        self.f = None
        self.q = None

    def set_masses(self, masses):
        """Set the mass array.

        Parameters
        ----------
        masses : array_like, shape (n,)
            Mass of each oscillator.
        """
        self.M = np.array(masses, dtype=np.float64)

    def set_springs(self, wall, coupling):
        """Set spring constants.

        Parameters
        ----------
        wall : array_like, shape (n,)
            Spring constants connecting each oscillator to its wall.
        coupling : array_like, shape (n-1,)
            Spring constants connecting neighboring oscillators.
            coupling[i] = spring between oscillator i and oscillator i+1.
        """
        n = self.n
        self.K = np.zeros((n, n))
        for i in range(n):
            self.K[i, i] = wall[i]
        for i in range(n - 1):
            self.K[i, i + 1] = coupling[i]
            self.K[i + 1, i] = coupling[i]

    def set_damping(self, gamma):
        """Set uniform damping ratio.

        Parameters
        ----------
        gamma : float
            Damping ratio (dimensionless). Typical values: 0.001 to 0.1.
        """
        self.gamma = gamma

    def build_H(self):
        """Build the dynamical matrix H from mass and spring arrays.

        H[i, j] encodes the equation of motion divided by m_i.
        Eigenvalues of H are the squared normal frequencies.
        """
        n = self.n
        M, K = self.M, self.K
        H = np.zeros((n, n), dtype=np.float64)

        for i in range(n):
            # Diagonal: wall spring + coupling springs, divided by mass
            H[i, i] = K[i, i] / M[i]
            if i == 0:
                H[i, i] += K[i, i + 1] / M[i]
            elif i == n - 1:
                H[i, i] += K[i - 1, i] / M[i]
            else:
                H[i, i] += (K[i - 1, i] + K[i, i + 1]) / M[i]

            # Off-diagonal: coupling springs
            for j in range(n):
                if i != j:
                    H[i, j] -= K[i, j] / M[i]

        self.H = H
        return H

    def solve(self):
        """Solve the eigenvalue problem H A = omega*^2 A using LAPACK dgeev.

        Stores eigenvalues (squared frequencies) sorted descending, and
        the corresponding eigenvector matrix (rows = modes).
        """
        if self.H is None:
            self.build_H()

        wr, wi, vr, vi, info = lapack.dgeev(self.H)

        if info != 0:
            raise RuntimeError(f"LAPACK dgeev failed with info = {info}")

        # Eigenvectors come in columns; transpose so rows = modes
        vr = np.transpose(vr)

        # Sort by eigenvalue descending (highest frequency first)
        idx = np.argsort(wr)[::-1]
        self.wr = wr[idx]
        self.eigenvectors = vr[idx]
        self.frequencies = np.sqrt(self.wr)

        return self.wr, self.eigenvectors

    def compute_forces(self, L=1.0, h0=1.0):
        """Compute driving forces and generalized forces on normal modes.

        The driving force on oscillator i is f_i = L * h0 * k_ii.
        The generalized force on mode i is q_i = sum_j A_ij * f_j.

        Parameters
        ----------
        L : float
            Length parameter of the experiment.
        h0 : float
            Strain amplitude parameter.
        """
        n = self.n
        self.f = np.array([L * h0 * self.K[i, i] for i in range(n)])
        self.q = dgemv(alpha=1.0, a=self.eigenvectors, x=self.f)
        return self.q

    def compute_amplitudes(self, w_squared):
        """Compute damped amplitudes for an array of squared driving frequencies.

        Uses complex response b_i = q_i / (w*^2_i - w^2 + 2i*gamma*w*_i*w)
        to properly handle phase through resonance. Returns magnitudes for
        normal coordinates and spatial coordinates.

        Parameters
        ----------
        w_squared : array_like
            Array of squared driving frequencies.

        Returns
        -------
        b : dict
            Normal coordinate amplitude magnitudes. b['Mode i'] is an array.
        x : dict
            Spatial coordinate amplitude magnitudes. x['Oscillator i'] is an array.
        w : ndarray
            Driving frequencies (sqrt of w_squared).
        """
        w_squared = np.asarray(w_squared)
        w = np.sqrt(w_squared)
        n = self.n
        resolution = len(w_squared)
        gamma = self.gamma

        # Complex normal coordinate amplitudes
        b_complex = np.zeros((n, resolution), dtype=complex)
        for i in range(n):
            denom = (self.wr[i] - w_squared) + 1j * (2 * gamma * np.sqrt(self.wr[i]) * w)
            b_complex[i, :] = self.q[i] / denom

        b = {}
        for i in range(n):
            b[f'Mode {i}'] = np.abs(b_complex[i, :])

        # Spatial coordinate amplitudes: X = B @ A (complex)
        B = b_complex.T  # rows = driving freqs, cols = modes
        X = B @ self.eigenvectors

        x = {}
        for i in range(n):
            x[f'Oscillator {i}'] = np.abs(X[:, i])

        return b, x, w

    def peak_sensitivities(self):
        """Compute peak amplitude of each mode at resonance.

        For small damping, the peak amplitude of mode i is:
            b_i_max = |q_i| / (2 * gamma * omega*_i^2)

        Returns
        -------
        peaks : ndarray, shape (n,)
            Peak amplitude of each normal mode.
        """
        if self.gamma == 0:
            raise ValueError("Damping gamma must be > 0 to compute finite peak sensitivities.")
        return np.abs(self.q) / (2 * self.gamma * self.wr)
