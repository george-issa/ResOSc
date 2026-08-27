"""Physical (SI) sensitivity pipeline for a levitated-sensor array.

Implements the framework of the companion note "SNR of a Multimode
Mechanical Detector" (noise_function_notes.pdf): modal decomposition of a
coupled array, thermal (FDT) + readout noise, strain-referred noise PSD
S_h(f), matched-filter SNR for PBH inspirals (Maggiore's formula), horizon
distance, and a minimum-detectable-impulse figure for dark-matter kicks.

Conventions
-----------
* SI units throughout: masses [kg], frequencies f [Hz], stiffness [N/m],
  temperature [K], lengths [m], forces [N], S_h [1/Hz].
* Per-particle damping gamma_i is a LINEWIDTH IN Hz (as in the LSD paper's
  quoted 0.17-1.7 Hz).  The angular damping rate used in the equations of
  motion and in the FDT is 2*pi*gamma_i.
* Mode shapes V (columns) are unit-Euclidean-norm; modal masses
  mu_n = v_n^T M v_n [kg] are carried explicitly (companion note Eq. 3).
* Readout A (axial cavity probe): weights FIXED, w = g/|g| (note Sec. 8.1).
  Readout B (transverse imaging): weights free (note Sec. 8.2).

Two readouts and both noise regimes (thermal / readout limited) are always
computed; nothing here assumes which one dominates.
"""
import numpy as np
import scipy.linalg

# ---- physical constants (SI) ---------------------------------------------- #
HBAR  = 1.054571817e-34   # J s
KB    = 1.380649e-23      # J/K
C     = 2.99792458e8      # m/s
G     = 6.67430e-11       # m^3 kg^-1 s^-2
MSUN  = 1.98892e30        # kg
KPC   = 3.0856775814913673e19  # m
EPS0  = 8.8541878128e-12  # F/m
QE    = 1.602176634e-19   # C


# --------------------------------------------------------------------------- #
# Mechanics
# --------------------------------------------------------------------------- #

class PhysicalArray:
    """N coupled levitated oscillators in SI units.

    Parameters
    ----------
    masses : (N,) array [kg]
    k_trap : (N,) array [N/m] — optical trap stiffness per particle.
    k_coupling : (N-1,) array [N/m] nearest-neighbour springs, or (N, N)
        symmetric full coupling matrix (zero diagonal) for long-range
        mechanisms such as Coulomb.
    gamma_hz : (N,) array [Hz] — per-particle damping LINEWIDTH.
    T : float [K] — temperature entering the FDT (bath or effective).
    L : float [m] — cavity half-baseline entering the strain force
        beta_i = k_trap,i * L (companion note Eq. 20).
    """

    def __init__(self, masses, k_trap, k_coupling, gamma_hz, T, L):
        self.m = np.asarray(masses, dtype=float)
        self.n = len(self.m)
        self.k_trap = np.asarray(k_trap, dtype=float)
        self.gamma_hz = np.asarray(gamma_hz, dtype=float)
        self.T = float(T)
        self.L = float(L)

        kc = np.asarray(k_coupling, dtype=float)
        K = np.diag(self.k_trap).astype(float)
        if kc.ndim == 1:
            for i in range(self.n - 1):
                K[i, i]     += kc[i]
                K[i+1, i+1] += kc[i]
                K[i, i+1]   -= kc[i]
                K[i+1, i]   -= kc[i]
        else:                       # full symmetric coupling matrix
            off = 0.5 * (kc + kc.T)
            np.fill_diagonal(off, 0.0)
            K += np.diag(off.sum(axis=1)) - off
        self.K = K

        # modal decomposition (note Eqs. 1-7)
        w2, V = scipy.linalg.eigh(self.K, np.diag(self.m))
        idx = np.argsort(w2)
        V = V[:, idx]
        V = V / np.linalg.norm(V, axis=0, keepdims=True)   # unit-norm shapes
        self.f_n = np.sqrt(w2[idx]) / (2.0 * np.pi)        # [Hz]
        self.V = V                                          # (site, mode)
        self.mu = np.einsum('im,i,im->m', V, self.m, V)     # [kg]
        self.Gamma_hz = (np.einsum('im,i,im->m', V, self.m * self.gamma_hz, V)
                         / self.mu)                         # note Eq. 7 [Hz]
        # strain signal couplings (note Eqs. 19-21): beta_i = k_trap,i * L
        self.beta = self.k_trap * self.L                    # [N / strain]
        self.B = V.T @ self.beta                            # [N / strain]

    # ---- responses -------------------------------------------------------- #

    def chi(self, f):
        """Modal susceptibility chi_n(f) [m/N], note Eq. 11. Shape (modes, nf)."""
        f = np.atleast_1d(np.asarray(f, dtype=float))
        w  = 2.0 * np.pi * f[None, :]
        wn = 2.0 * np.pi * self.f_n[:, None]
        gn = 2.0 * np.pi * self.Gamma_hz[:, None]           # angular damping
        return 1.0 / (self.mu[:, None] * (wn**2 - w**2 + 1j * w * gn))

    def transfer(self, f, w_vec):
        """Strain-to-observable transfer T(f) [m/strain], note Eq. 23."""
        N_n = self.V.T @ np.asarray(w_vec, dtype=float)
        return np.einsum('n,n,nf->f', N_n, self.B, self.chi(f))

    # ---- noise at the observable [m^2/Hz] --------------------------------- #

    def S_O_thermal(self, f, w_vec):
        """Thermal noise at the observable, note Eqs. 25-28.
        FDT (one-sided): S_Q_n = 4 kB T mu_n (2 pi Gamma_n) [N^2/Hz]."""
        N_n = self.V.T @ np.asarray(w_vec, dtype=float)
        S_Qn = 4.0 * KB * self.T * self.mu * (2.0 * np.pi * self.Gamma_hz)
        return np.einsum('n,nf->f', N_n**2 * S_Qn, np.abs(self.chi(f))**2)

    # ---- strain-referred noise -------------------------------------------- #

    def S_h(self, f, w_vec, S_O_readout):
        """S_h(f) = (S_O^th + S_O^ro) / |T|^2 [1/Hz], note Eqs. 17-18.
        S_O_readout: scalar or (nf,) array [m^2/Hz]."""
        T2 = np.abs(self.transfer(f, w_vec))**2
        return (self.S_O_thermal(f, w_vec) + S_O_readout) / T2

    # ---- frequency grid resolving every resonance ------------------------- #

    def band_grid(self, f_lo, f_hi, n_broad=2000, n_line=301, halfwidths=30.0):
        """Log-spaced backbone plus a fine window (+- halfwidths*Gamma_n)
        around every resonance inside the band."""
        pieces = [np.geomspace(f_lo, f_hi, n_broad)]
        for fn, gn in zip(self.f_n, self.Gamma_hz):
            lo = max(f_lo, fn - halfwidths * gn)
            hi = min(f_hi, fn + halfwidths * gn)
            if hi > lo:
                pieces.append(np.linspace(lo, hi, n_line))
        return np.unique(np.concatenate(pieces))


# --------------------------------------------------------------------------- #
# Readouts (note Sec. 8)
# --------------------------------------------------------------------------- #

def readout_A(g, lam_probe, P_det, eta_det, kappa):
    """Cavity-probe readout (note Eqs. 30-35).

    g : (N,) optomechanical couplings d(omega_cav)/dx_i [rad s^-1 m^-1]
    kappa : cavity decay rate [rad/s]; cavity pole f_cav = kappa/(2 pi).

    Returns (w_fixed, S_O_ro(f) callable [m^2/Hz]).
    """
    g = np.asarray(g, dtype=float)
    gnorm2 = np.sum(g**2)
    w_fixed = g / np.sqrt(gnorm2)
    omega_L = 2.0 * np.pi * C / lam_probe
    S_phi = HBAR * omega_L / (eta_det * P_det)              # [rad^2/Hz]
    f_cav = kappa / (2.0 * np.pi)

    def S_O_ro(f):
        f = np.atleast_1d(np.asarray(f, dtype=float))
        return (S_phi / gnorm2) * (kappa / 2.0)**2 * (1.0 + (f / f_cav)**2)

    return w_fixed, S_O_ro


def readout_B_noise(lam_ro, NA, eta_det, P_sc):
    """Per-particle imaging shot noise S_i^ro [m^2/Hz] (note Eq. 38)."""
    P_sc = np.asarray(P_sc, dtype=float)
    omega_ro = 2.0 * np.pi * C / lam_ro
    return HBAR * omega_ro * lam_ro**2 / (4.0 * np.pi * eta_det * NA**2 * P_sc)


def readout_B_observable_noise(w_vec, S_i_ro):
    """S_O^ro,B = sum_i w_i^2 S_i^ro (note Eq. 39). Frequency-flat."""
    w_vec = np.asarray(w_vec, dtype=float)
    return float(np.sum(w_vec**2 * np.asarray(S_i_ro)))


# --------------------------------------------------------------------------- #
# Sources and figures of merit
# --------------------------------------------------------------------------- #

def inspiral_A2(Mc_solar, r_m, Theta=1.0):
    """Maggiore inspiral amplitude: |h(f)|^2 = A^2 f^(-7/3) (note Eqs. 44-45)."""
    Mc = Mc_solar * MSUN
    return (5.0 / (24.0 * np.pi**(4.0/3.0))) * (C**2 / r_m**2) \
        * (G * Mc / C**3)**(5.0/3.0) * Theta


def f_isco(Mtot_solar):
    """Innermost-stable-orbit GW frequency, upper cutoff of the inspiral."""
    return C**3 / (6.0**1.5 * np.pi * G * Mtot_solar * MSUN)


def snr_inspiral(arr, w_vec, S_O_readout, Mc_solar, r_m, f_lo, f_hi,
                 Theta=1.0):
    """Matched-filter SNR rho for a PBH inspiral (note Eqs. 15-16, 47).
    S_O_readout: callable f->array, or scalar [m^2/Hz]."""
    f_hi = min(f_hi, f_isco(2.0 * Mc_solar / 2.0**(3.0/5.0)))  # eq-mass Mtot
    f = arr.band_grid(f_lo, f_hi)
    S_ro = S_O_readout(f) if callable(S_O_readout) else S_O_readout
    Sh = arr.S_h(f, w_vec, S_ro)
    A2 = inspiral_A2(Mc_solar, r_m, Theta)
    integrand = A2 * f**(-7.0/3.0) / Sh
    return float(np.sqrt(4.0 * np.trapezoid(integrand, f)))


def horizon_distance(arr, w_vec, S_O_readout, Mc_solar, f_lo, f_hi,
                     rho_threshold=8.0):
    """d_max [m]: distance at which the inspiral reaches rho_threshold.
    (rho scales as 1/r, so one evaluation suffices.)"""
    r_ref = 1.0 * KPC
    rho = snr_inspiral(arr, w_vec, S_O_readout, Mc_solar, r_ref, f_lo, f_hi)
    return r_ref * rho / rho_threshold


def min_impulse(arr, w_vec, S_O_readout, direction, f_lo, f_hi,
                rho_threshold=8.0):
    """Minimum detectable impulse [N s] for a delta-function momentum kick
    with spatial profile `direction` (unit vector over particles):
    f_i(t) = dp * dir_i * delta(t), so |f_tilde| is flat in frequency."""
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    f = arr.band_grid(f_lo, f_hi)
    N_n = arr.V.T @ np.asarray(w_vec, dtype=float)
    D_n = arr.V.T @ d                                       # modal kick coupling
    T_imp = np.einsum('n,n,nf->f', N_n, D_n, arr.chi(f))    # [m/(N s)] per unit dp
    S_ro = S_O_readout(f) if callable(S_O_readout) else S_O_readout
    S_O = arr.S_O_thermal(f, w_vec) + S_ro
    integral = 4.0 * np.trapezoid(np.abs(T_imp)**2 / S_O, f)    # rho^2 per dp^2
    return float(rho_threshold / np.sqrt(integral))


# --------------------------------------------------------------------------- #
# Coulomb coupling helper (first physical mechanism)
# --------------------------------------------------------------------------- #

def coulomb_coupling_matrix(charges_e, spacing):
    """Full (N, N) spring-constant matrix for charges on a line.

    charges_e : (N,) charges in units of the elementary charge.
    spacing : scalar or (N-1,) inter-particle gaps [m].

    k_ij = 2 q_i q_j / (4 pi eps0 d_ij^3)  — the longitudinal spring constant
    of the pair Coulomb force; includes ALL pairs (1/d^3 tails), signed by
    the charge product (stability K > 0 must be checked by the caller).
    """
    q = np.asarray(charges_e, dtype=float) * QE
    n = len(q)
    gaps = np.full(n - 1, spacing, dtype=float) if np.isscalar(spacing) \
        else np.asarray(spacing, dtype=float)
    pos = np.concatenate([[0.0], np.cumsum(gaps)])
    Kc = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = pos[j] - pos[i]
            Kc[i, j] = Kc[j, i] = 2.0 * q[i] * q[j] / (4.0 * np.pi * EPS0 * d**3)
    return Kc


# --------------------------------------------------------------------------- #
# LSD benchmark parameters (PRL 128, 111101 / arXiv:2010.13157 + companion note)
# --------------------------------------------------------------------------- #

LSD_BENCHMARK = {
    # confirmed / published
    'f_trap_hz': 1.0e5,          # 100 kHz benchmark case (10 kHz also published)
    'pressure_torr': 1e-11,
    'T_kelvin': 300.0,           # room temperature (4 K in optimized version)
    'L_m': 10.0,                 # cavity baseline
    'gamma_hz': 0.17 + 0.05,     # gas + photon-recoil linewidth at 100 kHz
    'disc_radius_m': 75e-6,
    'lam_probe_m': 1064e-9,      # readout A probe (companion note)
    'lam_ro_m': 532e-9,          # readout B probe (companion note)
    'NA': 0.1,                   # readout B (companion note)
    'spacing_m': 15.5e-6,        # inter-particle spacing (companion note)
    # ESTIMATED (from disc geometry: SiO2 14.58 um + 2 x 110 nm Si @ r=75 um)
    'mass_kg': 5.8e-10,          # TODO confirm with Nancy
    # UNCONFIRMED — placeholders, flagged for confirmation
    'P_det_W': 1e-3,             # TODO detected probe power (readout A)
    'P_sc_W': 1e-6,              # TODO collected power per particle (readout B)
    'eta_det': 0.8,              # TODO quantum efficiency
    'kappa_rad_s': 2*np.pi*1e6,  # TODO cavity decay rate (sets f_cav = 1 MHz)
}


def lsd_array(n=10, coupling=None, benchmark=LSD_BENCHMARK):
    """Uniform benchmark array of n LSD discs; `coupling` as in PhysicalArray
    (None -> uncoupled)."""
    b = benchmark
    m = np.full(n, b['mass_kg'])
    k = m * (2.0 * np.pi * b['f_trap_hz'])**2
    kc = np.zeros(n - 1) if coupling is None else coupling
    g = np.full(n, b['gamma_hz'])
    return PhysicalArray(m, k, kc, g, b['T_kelvin'], b['L_m'])


# --------------------------------------------------------------------------- #
# Validation (python -m ResOSc.physical or run this file)
# --------------------------------------------------------------------------- #

def _validate(verbose=True):
    ok = True

    # 1) single-oscillator thermal displacement peak: S_x(f0) = 4 kB T Q/(m w0^3)
    m, f0, g_hz, T = 1e-10, 1e5, 0.2, 300.0
    a = PhysicalArray([m], [m*(2*np.pi*f0)**2], [], [g_hz], T, 1.0)
    w = np.array([1.0])
    Sx = a.S_O_thermal(np.array([f0]), w)[0]
    w0 = 2*np.pi*f0
    Q = f0 / g_hz
    Sx_ref = 4*KB*T*Q/(m*w0**3)
    ok &= np.isclose(Sx, Sx_ref, rtol=1e-6)
    if verbose:
        print(f'1) thermal peak S_x: {Sx:.4e} vs 4kBTQ/mw0^3 = {Sx_ref:.4e}  '
              f'{"OK" if np.isclose(Sx, Sx_ref, rtol=1e-6) else "FAIL"}')

    # 2) thermal-limited on-resonance S_h is weight-independent (note Sec. 7)
    n = 4
    rng = np.random.default_rng(0)
    b = LSD_BENCHMARK
    m = np.full(n, b['mass_kg'])
    f_traps = np.array([0.8, 0.95, 1.1, 1.25]) * b['f_trap_hz']  # resolved modes
    k = m * (2*np.pi*f_traps)**2
    arr = PhysicalArray(m, k, np.full(n-1, 0.02*k.min()),
                        np.full(n, b['gamma_hz']), b['T_kelvin'], b['L_m'])
    fn = arr.f_n[1]
    vals = []
    for _ in range(4):
        wv = rng.standard_normal(n); wv /= np.linalg.norm(wv)
        vals.append(arr.S_h(np.array([fn]), wv, 0.0)[0])
    spread = (max(vals) - min(vals)) / min(vals)
    ok &= spread < 1e-2
    if verbose:
        print(f'2) weight-independence of thermal S_h on resonance: '
              f'relative spread {spread:.2e}  {"OK" if spread < 1e-2 else "FAIL"}')

    # 3) narrow-linewidth per-mode SNR (note Eq. 46 & Table 2) vs full integral,
    #    readout-limited, single mode
    cold = dict(LSD_BENCHMARK, T_kelvin=1e-9)   # readout-limited regime
    a1 = lsd_array(1, benchmark=cold)
    b = LSD_BENCHMARK
    S_ro = readout_B_noise(b['lam_ro_m'], b['NA'], b['eta_det'], b['P_sc_W'])
    S_ro_obs = float(S_ro)                                  # w = [1]
    Mc, r = 1e-3, 1.0*KPC
    # analytic: rho_n^2 = A^2 f_n^(-7/3) B^2 / (mu^2 (2pi)^3 f_n^2 Gamma_ang ... )
    # Using note Eq. 46: integral |chi|^2 df = 1/(4 mu^2 (2pi)^3 f_n^2 Gamma_n_ang/(2pi))
    fn, mu, Bn = a1.f_n[0], a1.mu[0], a1.B[0]
    g_ang = 2*np.pi*a1.Gamma_hz[0]
    int_chi2 = 1.0 / (4.0 * mu**2 * (2*np.pi*fn)**2 * g_ang)   # note Eq. 46
    A2 = inspiral_A2(Mc, r)
    rho2_analytic = 4.0 * A2 * fn**(-7.0/3.0) * Bn**2 * int_chi2 / S_ro_obs
    rho_num = snr_inspiral(a1, np.array([1.0]), S_ro_obs, Mc, r,
                           0.5*fn, 1.5*fn)
    ratio = rho_num / np.sqrt(rho2_analytic)
    ok &= abs(ratio - 1.0) < 0.02
    if verbose:
        print(f'3) narrow-linewidth per-mode SNR vs integral: ratio '
              f'{ratio:.4f}  {"OK" if abs(ratio-1)<0.02 else "FAIL"}')

    # 4) benchmark demo: thermal-limited sqrt(S_h) on resonance, N=1
    a1 = lsd_array(1)
    Sh_res = a1.S_h(np.array([a1.f_n[0]]), np.array([1.0]), 0.0)[0]
    if verbose:
        print(f'4) benchmark single disc (100 kHz, 300 K, L=10 m): '
              f'sqrt(S_h) on resonance = {np.sqrt(Sh_res):.2e} 1/sqrt(Hz)')
        print(f'   (LSD target 1e-22-1e-21; gap is set by the UNCONFIRMED '
              f'mass/effective-T/baseline entries in LSD_BENCHMARK)')
    return ok


if __name__ == '__main__':
    print('physical.py validation:')
    print('ALL OK' if _validate() else 'FAILURES PRESENT')
