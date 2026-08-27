"""
Load saved MC results from a .pkl file and produce publication-quality figures
with LaTeX rendering.  Intended to be run on your laptop after transferring
results from the cluster.

Workflow
--------
    # On the cluster:
    python3 run_mc.py --save-results results-mc/mc_results.pkl

    # Transfer to laptop:
    scp user@cluster:~/ResOSc/results-mc/mc_results.pkl results-mc/

    # On your laptop:
    python plot_mc.py

================================================================================
EXAMPLES
================================================================================

-- Quickstart (uses CONFIG defaults) -----------------------------------------

    python plot_mc.py

-- Specify input file explicitly ----------------------------------------------

    python plot_mc.py --input results-mc/mc_results.pkl

-- Control output path --------------------------------------------------------

    python plot_mc.py --savefig results-mc/mc_results.pdf
    python plot_mc.py --savefig figures/comparison.pdf

-- Disable LaTeX (plain fonts) -----------------------------------------------

    python plot_mc.py --no-latex

-- No interactive window (save only) -----------------------------------------

    python plot_mc.py --no-show

-- Plot only one method -------------------------------------------------------

    python plot_mc.py --method random
    python plot_mc.py --method metropolis
    python plot_mc.py --method both        # default: comparison plot

================================================================================
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)

import argparse
import pickle
import os


# ============================================================================ #
# CONFIG — edit these defaults
# ============================================================================ #

INPUT   = 'results-mc/mc_results.pkl'   # .pkl file produced by run_mc.py
SAVEFIG = 'results-mc/mc_results.pdf'   # output PDF path
METHOD  = 'both'     # 'random', 'metropolis', or 'both'
NO_SHOW = True       # set True to suppress the interactive window
NO_LATEX = False     # set True to disable LaTeX rendering

# ============================================================================ #

import matplotlib
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument('--no-show',  action='store_true', default=NO_SHOW)
_pre.add_argument('--no-latex', action='store_true', default=NO_LATEX)
_pre_args, _ = _pre.parse_known_args()
if _pre_args.no_show:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(
        description='Plot ResOSc MC results from a saved .pkl file.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--input',    type=str, default=INPUT,
                   help='Path to the .pkl file produced by run_mc.py')
    p.add_argument('--savefig',  type=str, default=SAVEFIG,
                   help='Save figure to this PDF path')
    p.add_argument('--method',   choices=['random', 'metropolis', 'both'],
                   default=METHOD,
                   help='Which result(s) to plot')
    p.add_argument('--no-show',  action='store_true', default=NO_SHOW,
                   help='Do not open an interactive window')
    p.add_argument('--no-latex', action='store_true', default=NO_LATEX,
                   help='Disable LaTeX rendering (plain fonts)')
    return p.parse_args()


def main():
    args = parse_args()

    # Load
    if not os.path.exists(args.input):
        raise FileNotFoundError(
            f"Results file not found: {args.input}\n"
            f"Run  python3 run_mc.py --save-results {args.input}  on the cluster first."
        )
    print(f"Loading {args.input} ...")
    with open(args.input, 'rb') as f:
        saved = pickle.load(f)

    result_mc    = saved.get('result_mc')
    result_metro = saved.get('result_metro')
    print("Loaded.")

    from ResOSc import plot_monte_carlo_results, plot_comparison

    use_latex = not args.no_latex

    if args.savefig:
        os.makedirs(os.path.dirname(os.path.abspath(args.savefig)), exist_ok=True)

    if args.method == 'both' and result_mc and result_metro:
        plot_comparison(result_mc, result_metro,
                        savefig=args.savefig, use_latex=use_latex)
    elif args.method == 'random' and result_mc:
        plot_monte_carlo_results(result_mc,
                                 savefig=args.savefig, use_latex=use_latex)
    elif args.method == 'metropolis' and result_metro:
        plot_monte_carlo_results(result_metro,
                                 savefig=args.savefig, use_latex=use_latex)
    else:
        # Fall back to whatever is available
        if result_mc and result_metro:
            plot_comparison(result_mc, result_metro,
                            savefig=args.savefig, use_latex=use_latex)
        elif result_mc:
            plot_monte_carlo_results(result_mc,
                                     savefig=args.savefig, use_latex=use_latex)
        elif result_metro:
            plot_monte_carlo_results(result_metro,
                                     savefig=args.savefig, use_latex=use_latex)

    if args.savefig:
        print(f"\nFigure saved to: {os.path.abspath(args.savefig)}")


if __name__ == '__main__':
    main()
