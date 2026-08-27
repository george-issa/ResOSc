"""
Run Monte Carlo and Metropolis Monte Carlo optimization for ResOSc.

Edit the variables in the CONFIG section below, then run:

    python3 run_mc.py

Alternatively, any CONFIG variable can be overridden from the command line
without touching this file — command-line flags always win over the defaults
set here.  Run  python3 run_mc.py --help  to see all flags.

================================================================================
EXAMPLES — copy any block into your terminal as-is
================================================================================

-- Quickstart (uses all CONFIG defaults) --------------------------------------

    python3 run_mc.py

-- Choose which method to run ------------------------------------------------

    python3 run_mc.py --method random          # random MC only
    python3 run_mc.py --method metropolis      # Metropolis only
    python3 run_mc.py --method both            # side-by-side comparison (default)

-- Change the system size -----------------------------------------------------

    python3 run_mc.py --n 3                    # 3 oscillators
    python3 run_mc.py --n 5                    # 5 oscillators
    python3 run_mc.py --n 10                   # 10 oscillators

-- Control search budget -------------------------------------------------------

    python3 run_mc.py --samples 1000  --refine 50     # quick test run
    python3 run_mc.py --samples 5000  --refine 100    # default
    python3 run_mc.py --samples 20000 --refine 300    # thorough (workstation)

-- Tune the objective weighting -----------------------------------------------

    python3 run_mc.py --weight 0.5    # equal: sensitivity + spread  (default)
    python3 run_mc.py --weight 1.0    # sensitivity only
    python3 run_mc.py --weight 0.0    # spread only
    python3 run_mc.py --weight 0.7    # lean toward sensitivity

-- Choose frequency-spread metric ---------------------------------------------

    python3 run_mc.py --spread relative    # (omega_max-omega_min)/mean  (default)
    python3 run_mc.py --spread absolute    # omega_max - omega_min

-- Choose force model ---------------------------------------------------------

    python3 run_mc.py --force strain       # strain-gradient drive  (default)
    python3 run_mc.py --force uniform      # uniform drive on all oscillators

-- Change damping --------------------------------------------------------------

    python3 run_mc.py --gamma 0.01         # light damping  (default)
    python3 run_mc.py --gamma 0.05         # moderate damping
    python3 run_mc.py --gamma 0.10         # heavy damping

-- Tune Metropolis-specific parameters ----------------------------------------

    python3 run_mc.py --method metropolis --step 0.05              # fine steps
    python3 run_mc.py --method metropolis --step 0.15              # default
    python3 run_mc.py --method metropolis --step 0.30              # coarse steps
    python3 run_mc.py --method metropolis --T-start 5.0 --T-end 0.01   # wide anneal
    python3 run_mc.py --method metropolis --T-start 1.0 --T-end 0.10   # narrow anneal

-- Reproducibility & output ---------------------------------------------------

    python3 run_mc.py --seed 0                          # different RNG seed
    python3 run_mc.py --savefig results-mc/out.pdf      # save plot to PDF
    python3 run_mc.py --save-results results-mc/mc_results.pkl   # save for later plotting

-- Environment flags ----------------------------------------------------------

    python3 run_mc.py --no-latex    # if LaTeX is not installed on the machine
    python3 run_mc.py --no-show     # headless server (no display)

-- Combined examples ----------------------------------------------------------

    # Large workstation run — save numerical results, no plotting
    python3 run_mc.py --n 8 --samples 20000 --refine 300 --save-results results-mc/mc_results.pkl

    # Quick sensitivity-focused Metropolis test
    python3 run_mc.py --method metropolis --n 5 --samples 3000 --weight 0.8

    # Reproduce a specific result exactly
    python3 run_mc.py --n 5 --samples 5000 --seed 123 --save-results results-mc/seed123.pkl

================================================================================
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)

import argparse
import time


def _fmt(seconds):
    """Format a duration as h m s or m s or s depending on magnitude."""
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s   = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


# ============================================================================ #
# CONFIG — edit these defaults to change behaviour without using CLI flags
# ============================================================================ #

# System
N               = 10       # number of oscillators
GAMMA           = 0.01     # damping coefficient

# Search budget
N_SAMPLES       = 20000    # MC draws / Metropolis steps
N_REFINE        = 100      # top candidates passed to full minimax optimisation

# Objective
SENSITIVITY_WEIGHT  = 0.5           # 0 = spread only, 1 = sensitivity only, 0.5 = equal
SPREAD_METRIC       = 'relative'    # 'relative' or 'absolute'
FORCE_MODEL         = 'strain'      # 'strain' or 'uniform'

# Search bounds (extended 2026-08: the v2 run pinned parameters at the old
# box edges — mass [0.3, 1.5], wall [0.5, 3.0], coupling [0.01, 1.0] — so the
# box was widened ~2x each way; the coupling floor was lowered to probe the
# near-uncoupled limit the winners favor)
MASS_BOUNDS     = (0.1, 3.0)
WALL_BOUNDS     = (0.2, 6.0)
COUPLING_BOUNDS = (0.001, 1.0)

# Resolvability: reject configurations with any adjacent mode gap below
# this many linewidths (2*gamma*omega). Overlapping resonances are
# mis-scored by both sensitivity metrics; 3 linewidths keeps peaks
# clearly separated. Set 0 to disable.
MIN_GAP_LINEWIDTHS = 3.0

# Method
METHOD          = 'metropolis'   # 'random', 'metropolis', or 'both'

# Metropolis tuning
STEP_SIZE       = 0.40     # log-space perturbation per step (~relative change)
T_START         = 3.0      # initial temperature  (high = broad exploration)
T_END           = 0.01     # final temperature    (low  = fine exploitation)

# Reproducibility
SEED            = 42

# Output
# Set SAVE_RESULTS to export numerical results for plotting on another machine.
# Use plot_mc.py on your laptop to load the .pkl and produce LaTeX figures.
SAVE_RESULTS    = 'results-mc/mc_results_latest.pkl'  # set None to skip saving
NO_SHOW         = True     # do not pop up an interactive window (recommended on cluster)

# ============================================================================ #

# Configure matplotlib BEFORE pyplot is imported.
import matplotlib
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument('--no-show', action='store_true', default=NO_SHOW)
_pre_args, _ = _pre.parse_known_args()
if _pre_args.no_show:
    matplotlib.use('Agg')
matplotlib.rcParams['text.usetex'] = False   # no LaTeX on the cluster
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(
        description='Monte Carlo parameter-space search for coupled oscillator systems.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--n',            type=int,   default=N,
                   help='Number of oscillators')
    p.add_argument('--samples',      type=int,   default=N_SAMPLES,
                   help='MC samples / Metropolis steps')
    p.add_argument('--refine',       type=int,   default=N_REFINE,
                   help='Candidates refined with full minimax optimisation')
    p.add_argument('--method',       choices=['random', 'metropolis', 'both'],
                   default=METHOD,   help='Which method to run')
    p.add_argument('--weight',       type=float, default=SENSITIVITY_WEIGHT,
                   help='Sensitivity weight in [0, 1]')
    p.add_argument('--spread',       choices=['relative', 'absolute'],
                   default=SPREAD_METRIC, help='Frequency spread metric')
    p.add_argument('--force',        choices=['strain', 'uniform'],
                   default=FORCE_MODEL,   help='Force model')
    p.add_argument('--gamma',        type=float, default=GAMMA,
                   help='Damping coefficient')
    p.add_argument('--mass-bounds',     type=float, nargs=2, default=list(MASS_BOUNDS),
                   metavar=('LO', 'HI'), help='Mass search range')
    p.add_argument('--wall-bounds',     type=float, nargs=2, default=list(WALL_BOUNDS),
                   metavar=('LO', 'HI'), help='Wall-spring search range')
    p.add_argument('--coupling-bounds', type=float, nargs=2, default=list(COUPLING_BOUNDS),
                   metavar=('LO', 'HI'), help='Coupling-spring search range')
    p.add_argument('--min-gap',         type=float, default=MIN_GAP_LINEWIDTHS,
                   help='Minimum adjacent mode gap in linewidths (0 disables)')
    p.add_argument('--seed',         type=int,   default=SEED,
                   help='RNG seed')
    p.add_argument('--step',         type=float, default=STEP_SIZE,
                   help='[Metropolis] log-space step size')
    p.add_argument('--T-start',      type=float, default=T_START,
                   help='[Metropolis] initial temperature')
    p.add_argument('--T-end',        type=float, default=T_END,
                   help='[Metropolis] final temperature')
    p.add_argument('--save-results', type=str,   default=SAVE_RESULTS,
                   help='Save results to this .pkl for plotting on another machine')
    p.add_argument('--no-show',      action='store_true', default=NO_SHOW,
                   help='Do not open an interactive plot window')
    return p.parse_args()


def main():
    import os, pickle
    args = parse_args()

    from ResOSc import monte_carlo_optimize, metropolis_optimize

    shared = dict(
        n                  = args.n,
        n_samples          = args.samples,
        n_refine           = args.refine,
        gamma              = args.gamma,
        force_model        = args.force,
        sensitivity_weight = args.weight,
        spread_metric      = args.spread,
        mass_bounds        = tuple(args.mass_bounds),
        wall_bounds        = tuple(args.wall_bounds),
        coupling_bounds    = tuple(args.coupling_bounds),
        min_gap_linewidths = args.min_gap,
        seed               = args.seed,
        verbose            = True,
    )

    result_mc    = None
    result_metro = None
    t_total_start = time.perf_counter()

    if args.method in ('random', 'both'):
        print("\n=== Random Monte Carlo ===")
        t0 = time.perf_counter()
        result_mc = monte_carlo_optimize(**shared)
        print(f"  Random MC finished in {_fmt(time.perf_counter() - t0)} ({time.perf_counter() - t0:.1f}s)")

    if args.method in ('metropolis', 'both'):
        print("\n=== Metropolis Monte Carlo ===")
        t0 = time.perf_counter()
        result_metro = metropolis_optimize(
            **shared,
            step_size = args.step,
            T_start   = args.T_start,
            T_end     = args.T_end,
        )
        print(f"  Metropolis MC finished in {_fmt(time.perf_counter() - t0)} ({time.perf_counter() - t0:.1f}s)")

    print(f"\n=== Total wall time: {_fmt(time.perf_counter() - t_total_start)} ===")

    if args.save_results:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_results)), exist_ok=True)
        with open(args.save_results, 'wb') as f:
            pickle.dump({'result_mc': result_mc, 'result_metro': result_metro}, f)
        print(f"\nResults saved to: {os.path.abspath(args.save_results)}")
        print("Transfer this file to your laptop and plot with:")
        print(f"  python plot_mc.py --input {args.save_results}")


if __name__ == '__main__':
    main()
