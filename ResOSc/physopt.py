"""Metropolis optimization of the physical figure of merit.

Ports the directed Metropolis search of ``montecarlo.py`` to the physical
(SI) pipeline of ``physical.py``: the objective is the PBH-inspiral horizon
distance d_max (equivalently the matched-filter SNR at fixed distance), and
the design variables are the quantities a lab can actually set:

* trap frequencies f_trap,i  — laser-power tunable; LSD publishes 10-100 kHz;
* particle charges q_i [e]   — Coulomb coupling k_ij = 2 q_i q_j/(4 pi eps0
  d_ij^3), ALL pairs (1/d^3 tails).  Bounds TBC (flagged parameter 7): the
  upper bound is the controllable charge, the lower bound the stray-charge
  floor (an as-built array can never be exactly uncoupled);
* observable weights w_i     — free for Readout B only; Readout A's weights
  are fixed by the intracavity profile g.

Masses and damping are fixed at the benchmark values (fabricated discs).
All-positive charges keep the stiffness matrix positive definite, so every
proposed design is stable.  The full transfer-function integral treats
overlapping resonances exactly, so no resolvability constraint is needed.

Energy for the walk: E = -log d_max (lower is better), annealed acceptance
exp(-dE/T), log-space proposals for positive variables, tangent-space
Gaussian for the unit-norm weights.  Best-ever state is tracked explicitly.
"""
import numpy as np

from .physical import (PhysicalArray, LSD_BENCHMARK, C,
                       coulomb_coupling_matrix, readout_A, readout_B_noise,
                       readout_B_observable_noise, horizon_distance,
                       lsd_gamma_force_hz, lsd_readout_A)

# TBC placeholder bounds for the controllable charge and the stray floor
# (units of the elementary charge) — flagged parameter 7 for confirmation.
Q_BOUNDS_E = (1.0e3, 1.0e8)
Q_STRAY_E = Q_BOUNDS_E[0]

F_TRAP_BOUNDS_HZ = (1.0e4, 1.0e5)      # published LSD trap range 10-100 kHz
BAND_HZ = (1.0e4, 3.0e5)               # science band of the search
MC_BENCH_SOLAR = 1.0e-3                # benchmark PBH chirp mass [M_sun]


# --------------------------------------------------------------------------- #
# Design -> array -> figure of merit
# --------------------------------------------------------------------------- #

def build_array(f_traps_hz, charges_e=None, benchmark=LSD_BENCHMARK):
    """PhysicalArray for given trap frequencies and (optional) charges.

    charges_e = None -> exactly uncoupled (the theoretical null hypothesis;
    an as-built array sits at the stray-charge floor instead).
    """
    b = benchmark
    f_traps_hz = np.asarray(f_traps_hz, dtype=float)
    n = len(f_traps_hz)
    m = np.full(n, b['mass_kg'])
    k = m * (2.0 * np.pi * f_traps_hz) ** 2
    if charges_e is None:
        kc = np.zeros(max(n - 1, 0))
    else:
        kc = coulomb_coupling_matrix(charges_e, b['spacing_m'])
    # anchored damping split: cold-damped linewidth f/Q_eff (tunable, not
    # sensitivity-critical) vs FDT force damping (gas + recoil, f-dependent)
    return PhysicalArray(m, k, kc, f_traps_hz / b['Q_eff'], b['T_kelvin'],
                         b['L_m'],
                         gamma_force_hz=lsd_gamma_force_hz(f_traps_hz, b))


def readout_setup(n, readout, benchmark=LSD_BENCHMARK):
    """Fixed weights (or None if free) and observable readout noise.

    Returns (w_fixed, S_O_ro) where S_O_ro is a callable f->array for
    Readout A, or the per-particle scalar S_i^ro for Readout B (combine
    with weights via readout_B_observable_noise).
    """
    b = benchmark
    if readout == 'A':
        # anchored dispersive coupling G = k_c (V/4V_c)(eps-1) omega_c and
        # kappa = pi c/(F L)  (AG13 / PRL 128); uniform profile still assumed
        return lsd_readout_A(n, b)
    if readout == 'B':
        S_i = float(readout_B_noise(b['lam_ro_m'], b['NA'], b['eta_det'],
                                    b['P_sc_W']))
        return None, S_i
    raise ValueError(f"unknown readout {readout!r}")


def evaluate_design(f_traps_hz, charges_e, w_vec, readout,
                    Mc_solar=MC_BENCH_SOLAR, band=BAND_HZ,
                    benchmark=LSD_BENCHMARK):
    """Horizon distance [m] of one design under one readout.

    w_vec is ignored for Readout A (hardware-fixed weights).  For Readout B
    it must be a unit vector; S_h is scale-invariant in w, so only the
    direction matters.
    """
    arr = build_array(f_traps_hz, charges_e, benchmark)
    w_fixed, ro = readout_setup(arr.n, readout, benchmark)
    if readout == 'A':
        w, S_ro = w_fixed, ro
    else:
        w = np.asarray(w_vec, dtype=float)
        w = w / np.linalg.norm(w)
        S_ro = readout_B_observable_noise(w, ro)
    return horizon_distance(arr, w, S_ro, Mc_solar, band[0], band[1])


# --------------------------------------------------------------------------- #
# Metropolis walk
# --------------------------------------------------------------------------- #

def _propose(state, rng, step_f, step_q, step_w,
             f_bounds=F_TRAP_BOUNDS_HZ, q_bounds=Q_BOUNDS_E):
    """One proposal: log-space multiplicative steps on f_trap and q
    (clamped to bounds), tangent Gaussian + renormalize on w."""
    new = {}
    f = state['f_traps']
    new['f_traps'] = np.clip(f * np.exp(rng.normal(0.0, step_f, f.shape)),
                             *f_bounds)
    if state.get('q') is not None:
        q = state['q']
        new['q'] = np.clip(q * np.exp(rng.normal(0.0, step_q, q.shape)),
                           *q_bounds)
    else:
        new['q'] = None
    if state.get('w') is not None:
        w = state['w'] + rng.normal(0.0, step_w, state['w'].shape)
        nrm = np.linalg.norm(w)
        new['w'] = w / nrm if nrm > 0 else state['w']
    else:
        new['w'] = None
    return new


def metropolis_physical(
    n=10,
    readout='B',
    coupled=True,
    n_steps=20000,
    Mc_solar=MC_BENCH_SOLAR,
    band=BAND_HZ,
    benchmark=LSD_BENCHMARK,
    f_bounds=F_TRAP_BOUNDS_HZ,
    q_bounds=Q_BOUNDS_E,
    step_f=0.05,
    step_q=0.30,
    step_w=0.15,
    T_start=1.0,
    T_end=0.01,
    seed=42,
    init_state=None,
    verbose=True,
    log_every=2000,
):
    """Annealed Metropolis search maximizing the horizon distance.

    State: trap frequencies (always), charges (if coupled), weights (if
    Readout B).  Energy E = -log d_max; geometric annealing T_start->T_end.
    init_state, when given, seeds the chain (e.g. with the incumbent naive
    design) instead of a random draw.

    Returns a result dict with the best-ever design, its d_max, the d_max
    trajectory (per step, of the CURRENT state), acceptance rate, and the
    run configuration.
    """
    rng = np.random.default_rng(seed)

    if init_state is not None:
        state = {k: (np.asarray(v, dtype=float).copy() if v is not None
                     else None)
                 for k, v in init_state.items()}
    else:
        state = {
            'f_traps': rng.uniform(f_bounds[0], f_bounds[1], n),
            'q': (np.exp(rng.uniform(np.log(q_bounds[0]),
                                     np.log(q_bounds[1]), n))
                  if coupled else None),
            'w': None,
        }
        if readout == 'B':
            w0 = rng.standard_normal(n)
            state['w'] = w0 / np.linalg.norm(w0)

    def dmax_of(s):
        return evaluate_design(s['f_traps'], s['q'], s['w'], readout,
                               Mc_solar, band, benchmark)

    d_cur = dmax_of(state)
    E_cur = -np.log(d_cur)
    best = {'state': {k: (v.copy() if v is not None else None)
                      for k, v in state.items()},
            'dmax': d_cur, 'step': 0}

    cooling = (T_end / T_start) ** (1.0 / max(n_steps - 1, 1))
    T = T_start
    n_accept = 0
    traj = np.empty(n_steps)

    for i in range(n_steps):
        prop = _propose(state, rng, step_f, step_q, step_w, f_bounds, q_bounds)
        d_new = dmax_of(prop)
        E_new = -np.log(d_new)
        dE = E_new - E_cur
        if dE <= 0 or rng.random() < np.exp(-dE / T):
            state, d_cur, E_cur = prop, d_new, E_new
            n_accept += 1
            if d_cur > best['dmax']:
                best = {'state': {k: (v.copy() if v is not None else None)
                                  for k, v in state.items()},
                        'dmax': d_cur, 'step': i + 1}
        traj[i] = d_cur
        T *= cooling
        if verbose and (i + 1) % log_every == 0:
            print(f"  step {i+1:>6d}/{n_steps}  T={T:.3g}  "
                  f"d_max(cur)={d_cur/1.496e11:8.3f} AU  "
                  f"best={best['dmax']/1.496e11:8.3f} AU  "
                  f"acc={n_accept/(i+1):.2f}")

    result = {
        'method': 'metropolis-physical',
        'readout': readout,
        'coupled': coupled,
        'n': n,
        'best': best,
        'dmax_traj': traj,
        'acceptance_rate': n_accept / n_steps,
        'config': dict(n_steps=n_steps, Mc_solar=Mc_solar, band=band,
                       f_bounds=f_bounds, q_bounds=q_bounds, step_f=step_f,
                       step_q=step_q, step_w=step_w, T_start=T_start,
                       T_end=T_end, seed=seed),
    }
    if verbose:
        print(f"  best d_max = {best['dmax']/1.496e11:.3f} AU "
              f"at step {best['step']} (acceptance {result['acceptance_rate']:.2f})")
    return result


# --------------------------------------------------------------------------- #
# The three-way comparison cases
# --------------------------------------------------------------------------- #

def naive_as_built(n=10, readout='B', benchmark=LSD_BENCHMARK,
                   Mc_solar=MC_BENCH_SOLAR, band=BAND_HZ,
                   q_stray_e=Q_STRAY_E):
    """Lab baseline: uniform traps at the benchmark frequency, every charge
    at the stray floor (a real array cannot be built uncoupled), uniform
    weights for Readout B.  Single evaluation, no optimization."""
    f_traps = np.full(n, benchmark['f_trap_hz'])
    q = np.full(n, q_stray_e)
    w = np.full(n, 1.0 / np.sqrt(n))
    d = evaluate_design(f_traps, q, w, readout, Mc_solar, band, benchmark)
    return {'method': 'naive-as-built', 'readout': readout, 'n': n,
            'best': {'state': {'f_traps': f_traps, 'q': q,
                               'w': (w if readout == 'B' else None)},
                     'dmax': d, 'step': 0}}
