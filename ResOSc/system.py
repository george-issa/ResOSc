"""
Core physics for the coupled oscillator system.

Handles system setup, eigenvalue decomposition via LAPACK, damped forced
oscillation amplitudes, and peak sensitivity computation.
"""

import numpy as np
import scipy.linalg
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
        """Solve the generalized eigenvalue problem K x = omega*^2 M x.

        Uses the symmetric-definite formulation (scipy.linalg.eigh) so the
        mode vectors are the physically correct M-orthonormal generalized
        eigenvectors: V^T M V = I.  With rows = modes (self.eigenvectors =
        V^T), the modal force projection q = A f and the spatial
        reconstruction x = b @ A are both exact for unequal masses.

        (History: this previously called LAPACK dgeev on H = M^-1 K and
        unpacked the return as (wr, wi, vr, vi, info) — but dgeev returns
        (wr, wi, vl, vr, info), so the LEFT eigenvectors of the
        non-symmetric H were used as the modal matrix.  For unequal masses
        that basis is neither the right eigenvectors nor M-orthonormal,
        which skewed every force projection and response reconstruction.)

        Stores eigenvalues (squared frequencies) sorted descending, and
        the corresponding eigenvector matrix (rows = modes).
        """
        if self.H is None:
            self.build_H()

        # Stiffness matrix K~ = M H is symmetric; symmetrize away float noise
        K_full = self.M[:, None] * self.H
        K_full = 0.5 * (K_full + K_full.T)

        w, V = scipy.linalg.eigh(K_full, np.diag(self.M))

        # Sort by eigenvalue descending (highest frequency first)
        idx = np.argsort(w)[::-1]
        self.wr = w[idx]
        self.eigenvectors = V[:, idx].T   # rows = modes, V^T M V = I
        self.frequencies = np.sqrt(self.wr)

        return self.wr, self.eigenvectors

    def compute_forces(self, L=1.0, h0=1.0, force_model='strain', force_vec=None):
        """Compute driving forces and generalized forces on normal modes.

        The generalized force on mode i is q_i = sum_j A_ij * f_j.

        Parameters
        ----------
        L : float
            Length parameter of the experiment.
        h0 : float
            Signal amplitude parameter (overall scale factor).
        force_model : {'strain', 'uniform', 'custom'}
            Spatial dependence of the driving force on each oscillator:

            'strain'  — GW / elastic-strain coupling: f_i = h0 * L * k_ii.
                        Force proportional to wall spring, appropriate for
                        gravitational waves where tidal forcing scales with
                        local stiffness.

            'uniform' — Momentum-impulse coupling: f_i = h0 * L for all i.
                        Force identical on all oscillators, appropriate for
                        direct dark matter interactions where the DM imparts
                        a momentum kick independent of local spring stiffness.

            'custom'  — User-defined spatial profile: f_i = h0 * force_vec[i].
                        Provides full control; use for coherent DM waves,
                        position-dependent forces, or any arbitrary coupling.
                        h0 sets the overall amplitude; force_vec carries the
                        spatial profile (e.g. L * cos(k_DM * positions)).

        force_vec : array_like, shape (n,), optional
            Required when force_model='custom'. Spatial profile of the force
            (without the h0 prefactor).
        """
        n = self.n
        if force_model == 'strain':
            self.f = np.array([L * h0 * self.K[i, i] for i in range(n)])
        elif force_model == 'uniform':
            self.f = np.full(n, h0 * L)
        elif force_model == 'custom':
            if force_vec is None:
                raise ValueError("force_vec must be provided for force_model='custom'.")
            self.f = h0 * np.asarray(force_vec, dtype=np.float64)
        else:
            raise ValueError(
                f"Unknown force_model '{force_model}'. "
                "Choose 'strain', 'uniform', or 'custom'."
            )
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

    def reference_system(self):
        """Build and solve the non-interacting (uncoupled) reference system.

        The reference system has the same masses and wall springs as self
        but all coupling springs set to zero.  Its normal modes are the
        individual oscillators with bare frequencies

            omega_i^0 = sqrt(k_ii / m_i)

        and its eigenvector matrix is the identity, so each oscillator is
        its own mode.  Call compute_forces() on the returned object with the
        same parameters used on self to obtain reference peak sensitivities
        for normalization.

        Returns
        -------
        ref : CoupledSystem
            A solved non-interacting system.  Forces are NOT pre-computed;
            call ref.compute_forces(...) with the same arguments before
            accessing peak_sensitivities() or sensitivity_profile().
        """
        ref = CoupledSystem(self.n)
        ref.M = self.M.copy()
        ref.K = np.diag(np.diag(self.K))   # wall springs only, zero coupling
        ref.gamma = self.gamma
        ref.build_H()
        ref.solve()
        return ref

    def frequency_spread(self, reference=None):
        """Compute frequency-spread metrics for the normal modes.

        Parameters
        ----------
        reference : CoupledSystem, optional
            If provided (typically from self.reference_system()), also
            returns comparison metrics showing how much extra spread the
            coupling introduces relative to the bare oscillator frequencies.

        Returns
        -------
        result : dict
            'absolute'          — omega_max - omega_min (bandwidth of the
                                  normal-mode spectrum; the physically
                                  relevant metric for detector bandwidth).
            'relative'          — absolute spread / geometric-mean frequency
                                  (dimensionless bandwidth, useful for
                                  comparing systems at different center freqs).
            'spacing_uniformity'— coefficient of variation (std/mean) of
                                  consecutive mode spacings.  0 = perfectly
                                  uniform spacing; larger = more clustered.
            'spacings'          — array of differences between consecutive
                                  sorted normal frequencies.
            'omega_min'         — lowest normal frequency.
            'omega_max'         — highest normal frequency.

            If reference is provided, also includes:
            'absolute_gain'     — self.absolute - ref.absolute  (extra spread
                                  introduced by the coupling springs).
            'relative_gain'     — self.relative / ref.relative  (ratio of
                                  relative spreads; > 1 means coupling widens
                                  the band beyond the bare-frequency spread).
        """
        freqs = np.sort(self.frequencies)
        omega_min, omega_max = freqs[0], freqs[-1]
        abs_spread = omega_max - omega_min
        geom_mean = np.exp(np.mean(np.log(freqs)))
        rel_spread = abs_spread / geom_mean

        spacings = np.diff(freqs)
        mean_spacing = np.mean(spacings)
        spacing_cv = np.std(spacings) / mean_spacing if mean_spacing > 0 else np.nan

        result = {
            'absolute': abs_spread,
            'relative': rel_spread,
            'spacing_uniformity': spacing_cv,
            'spacings': spacings,
            'omega_min': omega_min,
            'omega_max': omega_max,
        }

        if reference is not None:
            ref_spread = reference.frequency_spread()
            result['absolute_gain'] = abs_spread - ref_spread['absolute']
            ref_rel = ref_spread['relative']
            result['relative_gain'] = (rel_spread / ref_rel) if ref_rel > 0 else np.nan

        return result
