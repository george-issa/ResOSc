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

    def __init__(self, masses, k_trap, k_coupling, gamma_hz, T, L,
                 gamma_force_hz=None):
        self.m = np.asarray(masses, dtype=float)
        self.n = len(self.m)
        self.k_trap = np.asarray(k_trap, dtype=float)
        self.gamma_hz = np.asarray(gamma_hz, dtype=float)
        # Damping split (anchored to PRL 110,071105 / PRL 128,111101):
        # gamma_hz is the MECHANICAL linewidth (cold-damped Q_eff; noiseless
        # feedback damping), gamma_force_hz the damping entering the FDT
        # force noise (gas gamma_g + recoil-heating equivalent).  They are
        # equal only without feedback cooling; default preserves that.
        self.gamma_force_hz = (self.gamma_hz if gamma_force_hz is None
                               else np.asarray(gamma_force_hz, dtype=float))
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
        self.Gamma_force_hz = (np.einsum('im,i,im->m', V,
                                         self.m * self.gamma_force_hz, V)
                               / self.mu)                   # FDT damping [Hz]
        # Strain signal coupling per AG13 (PRL 110,071105 Eq. 5): the GW
        # inertially drives the cavity-locked trap minimum,
        #   f_i(t) = (1/2) m_i (2 pi f_gw)^2 L h(t),
        # frequency of the WAVE, exact at all f.  The note's Eq. 20
        # (beta_i = k_trap,i L, quasi-static) is its on-resonance value
        # with L the half-baseline; here L is the FULL cavity length.
        self.bm = V.T @ self.m                              # [kg]

    # ---- responses -------------------------------------------------------- #

    def chi(self, f):
        """Modal susceptibility chi_n(f) [m/N], note Eq. 11. Shape (modes, nf)."""
        f = np.atleast_1d(np.asarray(f, dtype=float))
        w  = 2.0 * np.pi * f[None, :]
        wn = 2.0 * np.pi * self.f_n[:, None]
        gn = 2.0 * np.pi * self.Gamma_hz[:, None]           # angular damping
        return 1.0 / (self.mu[:, None] * (wn**2 - w**2 + 1j * w * gn))

    def B_n(self, f):
        """Modal strain drive B_n(f) = 0.5 (2 pi f)^2 L (V^T m)_n
        [N/strain], AG13 inertial coupling.  Shape (modes, nf)."""
        f = np.atleast_1d(np.asarray(f, dtype=float))
        return 0.5 * (2.0 * np.pi * f[None, :])**2 * self.L * self.bm[:, None]

    def transfer(self, f, w_vec):
        """Strain-to-observable transfer T(f) [m/strain], note Eq. 23
        with the frequency-dependent AG13 drive."""
        N_n = self.V.T @ np.asarray(w_vec, dtype=float)
        return np.einsum('n,nf,nf->f', N_n, self.B_n(f), self.chi(f))

    # ---- noise at the observable [m^2/Hz] --------------------------------- #

    def S_O_thermal(self, f, w_vec):
        """Thermal noise at the observable, note Eqs. 25-28.
        FDT (one-sided): S_Q_n = 4 kB T mu_n (2 pi Gamma_force_n) [N^2/Hz].
        Uses the FORCE-noise damping, not the (cold-damped) linewidth."""
        N_n = self.V.T @ np.asarray(w_vec, dtype=float)
        S_Qn = 4.0 * KB * self.T * self.mu * (2.0 * np.pi * self.Gamma_force_hz)
        return np.einsum('n,nf->f', N_n**2 * S_Qn, np.abs(self.chi(f))**2)

    # ---- strain-referred noise -------------------------------------------- #

    def S_h(self, f, w_vec, S_O_readout):
        """S_h(f) = (S_O^th + S_O^ro) / |T|^2 [1/Hz], note Eqs. 17-18.
        S_O_readout: scalar or (nf,) array [m^2/Hz]."""
        T2 = np.abs(self.transfer(f, w_vec))**2
        return (self.S_O_thermal(f, w_vec) + S_O_readout) / T2

    # ---- frequency grid resolving every resonance ------------------------- #

    def band_grid(self, f_lo, f_hi, n_broad=2000, n_line=301, halfwidths=30.0,
                  n_shoulder=150):
        """Log-spaced backbone, a fine linear window (+- halfwidths*Gamma_n)
        around every resonance, plus log-spaced shoulder points from the
        window edge out to the band edges.  The shoulders matter: for a deep
        bucket the thermal/readout crossover sits thousands of linewidths
        from the line, and the backbone alone under-resolves the 1/Delta^2
        wings there (was worth ~15% of d_max for a 10-disc stack)."""
        pieces = [np.geomspace(f_lo, f_hi, n_broad)]
        span = f_hi - f_lo
        for fn, gn in zip(self.f_n, self.Gamma_hz):
            lo = max(f_lo, fn - halfwidths * gn)
            hi = min(f_hi, fn + halfwidths * gn)
            if hi > lo:
                pieces.append(np.linspace(lo, hi, n_line))
            off = np.geomspace(halfwidths * gn, span, n_shoulder)
            for pts in (fn - off, fn + off):
                pieces.append(pts[(pts > f_lo) & (pts < f_hi)])
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
    f_hi = min(f_hi, f_isco(2.0**(6.0/5.0) * Mc_solar))  # eq-mass Mtot = 2^(6/5) Mc
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
    # published (PRL 128,111101 Table I, 100 kHz case, + PRL 110,071105)
    'f_trap_hz': 1.0e5,          # 100 kHz benchmark case (10 kHz also published)
    'pressure_torr': 1e-11,
    'T_kelvin': 300.0,           # room temperature (4 K in optimized 100 m)
    'L_m': 10.0,                 # FULL cavity length (AG13 drive convention)
    'Ni_gamma_g_hz': 0.17,       # occupation x gas damping at 100 kHz, 300 K
    'gamma_sc_hz': 0.05,         # photon-recoil heating rate at 100 kHz
    'Q_eff': 5.4e5,              # cold-damped effective Q (AG13 Table II disc)
    'disc_radius_m': 75e-6,
    'lam_probe_m': 1.55e-6,      # trap/detection wavelength (both papers)
    'waist_m': 75e-6,            # cavity mode waist
    'F_cav': 10.0,               # modest-mirror cavity finesse (AG13)
    'stack_volume_m3': 2.62e-13, # SiO2 spacer 14.58 um + 2 x 110 nm Si caps
    'eps_minus_1': 1.1,          # Re(eps)-1 of the (mostly silica) stack
    'P_det_W': 2e-4,             # detection power (AG13 disc value)
    'lam_ro_m': 532e-9,          # readout B probe (companion note; still TBC)
    'NA': 0.5,                   # imaging aperture; "up to 0.5" per Nancy
                                 # (2026-08-31, via user; was 0.1 placeholder)
    'P_sc_W': 1e-6,              # readout B collected power (still TBC)
    'eta_det': 0.8,              # quantum efficiency (still TBC)
    'spacing_m': 15.5e-6,        # inter-particle spacing (companion note)
    # mass from published stack geometry (matches their h_min bookkeeping)
    'mass_kg': 5.8e-10,
    # anchored strain targets to validate against (PRL 128 Table I)
    'h_min_100k': 1.02e-22,
    'h_min_10k': 7.6e-21,
}


def lsd_gamma_gas_hz(f_trap_hz, benchmark=LSD_BENCHMARK):
    """Gas damping gamma_g [Hz] from the published Ni*gamma_g:
    gamma_g = (hbar omega0 / kB T) * (Ni gamma_g).  Frequency-independent;
    the published 0.17 Hz (100 kHz) / 1.7 Hz (10 kHz) differ only via Ni."""
    b = benchmark
    Ni_100k = KB * b['T_kelvin'] / (HBAR * 2.0 * np.pi * 1.0e5)
    return np.full_like(np.asarray(f_trap_hz, dtype=float),
                        b['Ni_gamma_g_hz'] / Ni_100k)


def lsd_gamma_force_hz(f_trap_hz, benchmark=LSD_BENCHMARK):
    """FDT force-noise damping [Hz]: gas + recoil-heating equivalent.
    S_F = 4 m [kB T (2 pi gamma_g) + hbar omega0 (2 pi gamma_sc)], folded
    into gamma_force = gamma_g + (hbar omega0 / kB T) gamma_sc(f), with
    gamma_sc scaling linearly with f_trap (PRL 128 Table I: 0.005 -> 0.05 Hz
    from 10 to 100 kHz)."""
    b = benchmark
    f = np.asarray(f_trap_hz, dtype=float)
    gamma_sc = b['gamma_sc_hz'] * (f / 1.0e5)
    hw_kt = HBAR * 2.0 * np.pi * f / (KB * b['T_kelvin'])
    return lsd_gamma_gas_hz(f, benchmark) + hw_kt * gamma_sc


def lsd_readout_A(n, benchmark=LSD_BENCHMARK):
    """Anchored cavity readout: dispersive coupling per particle
    G = k_c (V/4V_c)(eps-1) omega_c  [rad s^-1 m^-1]  (AG13 Table I disc),
    kappa = pi c / (F L).  Returns (w_fixed, S_O_ro callable)."""
    b = benchmark
    omega_c = 2.0 * np.pi * C / b['lam_probe_m']
    V_c = np.pi * b['waist_m']**2 * b['L_m'] / 4.0
    G = (2.0 * np.pi / b['lam_probe_m']) \
        * (b['stack_volume_m3'] / (4.0 * V_c)) * b['eps_minus_1'] * omega_c
    kappa = np.pi * C / (b['F_cav'] * b['L_m'])
    return readout_A(np.full(n, G), b['lam_probe_m'], b['P_det_W'],
                     b['eta_det'], kappa)


def lsd_array(n=10, coupling=None, benchmark=LSD_BENCHMARK, f_traps_hz=None):
    """Benchmark array of n LSD discs; `coupling` as in PhysicalArray
    (None -> uncoupled).  f_traps_hz: per-particle trap frequencies
    (default: uniform benchmark value).  Linewidth = f/Q_eff (cold-damped);
    force-noise damping from lsd_gamma_force_hz."""
    b = benchmark
    f = (np.full(n, b['f_trap_hz']) if f_traps_hz is None
         else np.asarray(f_traps_hz, dtype=float))
    m = np.full(n, b['mass_kg'])
    k = m * (2.0 * np.pi * f)**2
    kc = np.zeros(max(n - 1, 0)) if coupling is None else coupling
    return PhysicalArray(m, k, kc, f / b['Q_eff'], b['T_kelvin'], b['L_m'],
                         gamma_force_hz=lsd_gamma_force_hz(f, b))


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
                        np.full(n, 0.2), b['T_kelvin'], b['L_m'])
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
    #    readout-limited, single mode (thermal force noise switched off)
    cold = dict(LSD_BENCHMARK, Ni_gamma_g_hz=1e-30, gamma_sc_hz=1e-30)
    a1 = lsd_array(1, benchmark=cold)
    b = LSD_BENCHMARK
    S_ro = readout_B_noise(b['lam_ro_m'], b['NA'], b['eta_det'], b['P_sc_W'])
    S_ro_obs = float(S_ro)                                  # w = [1]
    Mc, r = 1e-3, 1.0*KPC
    # analytic: per-mode SNR with B evaluated at resonance (AG13 drive)
    # Using note Eq. 46: integral |chi|^2 df = 1/(4 mu^2 (2pi)^2 fn^2 g_ang)
    fn, mu = a1.f_n[0], a1.mu[0]
    Bn = float(a1.B_n(np.array([fn]))[0, 0])
    g_ang = 2*np.pi*a1.Gamma_hz[0]
    int_chi2 = 1.0 / (4.0 * mu**2 * (2*np.pi*fn)**2 * g_ang)   # note Eq. 46
    A2 = inspiral_A2(Mc, r)
    rho2_analytic = 4.0 * A2 * fn**(-7.0/3.0) * Bn**2 * int_chi2 / S_ro_obs
    rho_num = snr_inspiral(a1, np.array([1.0]), S_ro_obs, Mc, r,
                           0.5*fn, 1.5*fn)
    ratio = rho_num / np.sqrt(rho2_analytic)
    ok &= abs(ratio - 1.0) < 0.05
    if verbose:
        print(f'3) narrow-linewidth per-mode SNR vs integral: ratio '
              f'{ratio:.4f}  {"OK" if abs(ratio-1)<0.05 else "FAIL"}')

    # 4/5) anchor: thermal-limited sqrt(S_h) on resonance, N=1, vs the
    # PUBLISHED LSD h_min (PRL 128 Table I).  Tolerance is generous
    # (factor 4) pending the Hz-vs-rad/s convention of the published
    # Ni*gamma_g / gamma_sc values — flagged for Nancy.
    for f0, key in ((1.0e5, 'h_min_100k'), (1.0e4, 'h_min_10k')):
        a1 = lsd_array(1, f_traps_hz=np.array([f0]))
        Sh_res = a1.S_h(np.array([a1.f_n[0]]), np.array([1.0]), 0.0)[0]
        r_pub = np.sqrt(Sh_res) / LSD_BENCHMARK[key]
        ok &= 0.2 < r_pub < 5.0
        if verbose:
            print(f'5) sqrt(S_h) on resonance at {f0/1e3:.0f} kHz: '
                  f'{np.sqrt(Sh_res):.2e} vs published {LSD_BENCHMARK[key]:.2e} '
                  f'-> ratio {r_pub:.2f}  '
                  f'{"OK" if 0.2 < r_pub < 5.0 else "FAIL"}')
    return ok


if __name__ == '__main__':
    print('physical.py validation:')
    print('ALL OK' if _validate() else 'FAILURES PRESENT')
