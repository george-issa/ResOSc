# ResOSc — Latest Findings

> **⚠ 2026-08-20 — CRITICAL NOTE**: a left/right eigenvector swap in
> `CoupledSystem.solve()` (LAPACK dgeev unpacking) was discovered and
> fixed on this date. All quantitative results below from **before
> Section 9** were computed with the wrong modal basis and are
> superseded; they are retained as project history. See Section 9 and
> `monte_carlo.md` for the corrected physics and current results.

## Overview

ResOSc is a Python package for simulating and analyzing coupled oscillator systems, with applications to physics experiments such as gravitational wave (GW) detection and dark matter (DM) sensing. The package computes normal modes, forced oscillation responses under damping, and optimizes observable weight vectors to maximize detection sensitivity across a frequency range.

---

## System Configuration

All baseline demonstrations use a 10-oscillator chain with:
- **Masses**: linearly increasing from 0.45 to 0.90
- **Wall springs**: 1.2 (uniform)
- **Coupling springs**: 0.1 between neighbors (weak coupling)
- **Damping**: γ = 0.01
- **Normal mode frequencies**: range from 1.1766 to 1.7583 (relative spread ~40%)

---

## Features & Findings

### 1. Force Model Selection

Three physical driving force models are supported:

| Model | Force Pattern | Physical Context |
|-------|---------------|------------------|
| **Strain** (default) | f_i ∝ k_ii (wall spring stiffness) | Gravitational wave tidal forcing |
| **Uniform** | f_i = constant | Dark matter momentum impulse |
| **Custom** | f_i = h₀ × force_vec[i] | Arbitrary spatial profiles (e.g., coherent DM waves) |

**Key finding**: Different force models produce qualitatively different mode excitation patterns. Uniform forcing concentrates signal in center-of-mass-like (symmetric) modes and suppresses antisymmetric modes — the opposite behavior to GW strain forcing.

---

### 2. Non-Interacting Reference System & Frequency Spread

`reference_system()` and `frequency_spread()` enable quantitative benchmarking against an uncoupled baseline.

**Frequency spread metrics (coupled vs. non-interacting reference):**

| Metric | Coupled | Reference | Difference |
|--------|---------|-----------|------------|
| Absolute spread (ω_max − ω_min) | 0.5816 | 0.4783 | +0.1033 |
| Relative spread | 0.4030 | 0.3545 | +0.0485 |
| Spacing uniformity (CV) | 0.2400 | 0.3021 | −0.0621 |
| Coupling gain (relative) | — | — | ×1.137 |

**Result**: Coupling springs widen the frequency band by ~13.7%. The coupled system also exhibits more uniform mode spacing (lower CV), suggesting better frequency distribution across the detection band.

---

### 3. Sensitivity Normalization Against Reference

The non-interacting system's minimax-optimal sensitivity provides a normalization benchmark:

$$\text{ref\_scale} = \frac{1}{\sqrt{\sum_i \frac{1}{b_{i,\text{ref}}^2}}}$$

**Normalized worst-case sensitivities by force model (baseline system):**

| Force Model | Normalized Sensitivity | Interpretation |
|-------------|------------------------|----------------|
| GW Strain | 0.3495 | Below reference (< 1) |
| DM Uniform | 0.3495 | Below reference (< 1) |
| DM Coherent Wave | 0.7874 | Below reference (< 1) |

**Critical finding**: All three force models yield normalized sensitivities below 1. Even with optimal linear weight vectors, the minimax worst-case sensitivity falls short of the reference. This establishes the motivation for the Monte Carlo parameter search described below.

---

### 4. Monte Carlo Parameter Optimization

#### Motivation

The baseline results (Section 3) show that hand-chosen parameters leave the system below the reference sensitivity threshold. The fixed parameter choices (uniform wall springs, weak coupling, linearly graded masses) were not optimized for either sensitivity or bandwidth. The Monte Carlo search asks: **is there a configuration of masses, wall springs, and coupling springs that simultaneously achieves higher sensitivity and broader frequency coverage?**

#### What the Monte Carlo Does

The search explores the joint parameter space:

| Parameter | Range sampled |
|-----------|---------------|
| Masses m_i | [0.3, 1.5] (each independently) |
| Wall springs k_ii | [0.5, 3.0] (each independently) |
| Coupling springs k_{i,i+1} | [0.01, 1.0] (each independently) |

For a 10-oscillator system this is a **29-dimensional continuous search space** (10 masses + 10 wall springs + 9 coupling springs). Exhaustive search is impossible; random sampling covers it efficiently.

Two objectives are optimized simultaneously:

- **Sensitivity**: the normalized minimax worst-case sensitivity — how well a single linear observable can detect *all* normal modes at once, relative to the uncoupled reference ceiling.
- **Frequency spread**: the relative bandwidth of the normal-mode spectrum — how broadly the modes cover frequency space.

These objectives are in tension: configurations that push modes far apart often do so by creating strong asymmetries that hurt the minimax sensitivity, and vice versa.

#### Two-Phase Strategy

**Phase 1 — Screening** (`n_samples = 5000` random draws, fast)

For each randomly sampled configuration:
1. Build the dynamical matrix H and solve the eigenvalue problem to obtain normal-mode frequencies and eigenvectors.
2. Compute driving forces under the chosen force model and evaluate peak sensitivities b_i at each resonance.
3. Compute a **fast sensitivity proxy** — the harmonic-L2 norm of the peak sensitivities:

$$\text{proxy} = \frac{1}{\sqrt{\sum_i \frac{1}{b_i^2}}}$$

This is the exact minimax-optimal sensitivity for the uncoupled system (where eigenvectors = identity) and a reliable fast lower bound for the coupled case. It requires no weight-vector optimization.

4. Compute the relative frequency spread: (ω_max − ω_min) / geometric-mean(ω).
5. Normalize both metrics to [0, 1] across all valid draws and compute a **combined score**:

$$\text{score} = \alpha \cdot \widetilde{\text{sensitivity}} + (1 - \alpha) \cdot \widetilde{\text{spread}}$$

where α is the `sensitivity_weight` parameter (default 0.5 = equal weight).

6. Retain the top `n_refine = 100` configurations by combined score.

This phase runs in seconds because no weight-vector optimization is needed per draw.

**Phase 2 — Refinement** (top 100 candidates, accurate)

For each shortlisted candidate:
1. Run the full minimax weight-vector optimization (`optimize_observable`): 50-restart Nelder-Mead on the unit sphere to solve max_{‖w‖=1} min_i S_i(w).
2. Build the non-interacting reference system and compute its minimax scale (ref_scale).
3. Compute the **normalized minimax sensitivity** = min_i S_i(w_opt) / ref_scale.
4. Record the full frequency spread metrics via `frequency_spread()`.

After refinement, all candidates are re-ranked by a combined score computed from the accurate (not proxy) objectives.

**Pareto front extraction**

A configuration is **Pareto-optimal** if no other refined candidate is simultaneously better on both sensitivity and spread. The Pareto front is the efficient frontier: it shows every configuration where improving one objective necessarily sacrifices the other. A designer picks a point on this frontier based on the relative importance of sensitivity vs. bandwidth for their experiment.

#### Results

**Balanced search (GW strain, α = 0.5):**

The Monte Carlo finds configurations that are significantly better on both objectives compared to the hand-chosen baseline. The Pareto front reveals a clear trade-off curve — the top end of the sensitivity axis and the top end of the spread axis are occupied by different configurations, confirming the objectives genuinely compete.

**Effect of priority weighting (α):**

| Priority | α | Best norm. sensitivity | Best rel. spread |
|----------|---|------------------------|------------------|
| Sensitivity-first | 0.9 | higher | lower |
| Balanced | 0.5 | intermediate | intermediate |
| Spread-first | 0.1 | lower | higher |

Shifting α moves the selected best configuration along the Pareto front, trading one objective for the other.

**Cross-model comparison (balanced, α = 0.5):**

The optimal parameter region shifts between force models. Under DM uniform forcing, the optimizer finds different mass and spring distributions because uniform forcing loads all modes equally, whereas GW strain forcing loads modes in proportion to wall-spring stiffness. The Pareto front shapes differ between models, reflecting the different coupling between parameter geometry and mode excitation.

**Screening cloud (Phase 1 density)**

The hexbin density plot of all 5000 Phase 1 draws shows that high sensitivity and high spread are anti-correlated in the raw parameter landscape — the upper-right corner of the objective space is sparsely populated. The Pareto front sits at the boundary of this cloud, confirming that the refined candidates are genuinely at the edge of what the parameter space can achieve.

---

### 5. Metropolis v2 Run: Configurations That Beat the Reference (2026-08-19)

After fixing the Metropolis optimizer (best-ever state tracking, deduplicated refinement pool) and retuning it (20000 steps, step size 0.40, T: 3.0 → 0.01, 100 refined candidates), a head-to-head comparison at equal budget gave:

| Metric | Random MC | Metropolis MC |
|--------|-----------|---------------|
| Best raw minimax sensitivity | 11.61 | 16.46 |
| Best **normalized** sensitivity | 0.9439 | **1.0625** |
| Configurations with normalized sensitivity > 1 | 0 / 100 | **10 / 100** |
| Best relative spread on Pareto front | 2.01 | 3.14 |
| Pareto front size | 7 | 9 |
| Final acceptance rate | — | 46.3% |

**Key finding**: Ten Metropolis-refined configurations exceed the uncoupled reference threshold (normalized minimax sensitivity > 1) — the first direct answer to the project's motivating question. Coupled configurations that beat the non-interacting reference do exist, and directed (annealed) search finds them where blind random sampling does not.

The best such configuration reaches normalized sensitivity **1.0625** with relative spread **2.70** (the hand-chosen baseline: 0.35 and 0.40). Its structure is characteristic of all ten: **weak couplings (0.01–0.10) on top of strongly heterogeneous masses and wall springs**, with several parameters pinned at the search bounds (masses at 1.5 and 0.3, one wall spring at 3.0). The bound-pinning indicates the true optimum lies outside the current search box — widening the mass and wall-spring bounds is the natural next experiment.

Chain diagnostics confirm the fixes: the best energy (−1.792) was found at step 19838 of 20000 and the chain froze on it (no drift, unlike the April run), and the final acceptance rate landed at 46.3% (was 71.7%).

**Interference validation**: because the winning configuration has three mode pairs closer than a linewidth, the isolated-peak metric used by the search was checked against the full swept response O(ω) = |Σ_j w_j X_j(ω)| (coherent sum over all modes) evaluated at every resonance, with weights re-optimized under that metric for both the winner and its reference. The result **survives and improves: normalized sensitivity 1.089 under the swept metric** (12.59 coupled vs 11.56 reference). Interference is large with mismatched weights (the stored weights drop to 6.03/2.21), but weights tuned to the true response exploit the coherent overlap, and the coupled system gains more than its reference.

---

### 6. Extended-Bounds Run (v3) and the Validated Champion (2026-08-19)

A Metropolis-only run with an extended search box (masses [0.1, 3.0], wall springs [0.2, 6.0], couplings [0.001, 1.0] — now the default in `run_mc.py`) found 9/100 nominal winners, max iso-normalized 1.095. Two lessons emerged:

1. **The raw-sensitivity objective is scale-unbounded** — winners pin at any box edge (heavy masses + soft springs → low ω → peak response ∝ 1/ω²), so widening the box chases scale rather than structure. The next code improvement is to rank by *normalized* sensitivity, which is equally cheap (reference peaks are analytic).
2. **The isolated-peak metric misranks candidates with overlapping modes.** A batch swept-metric validation of all 19 nominal winners (v2 + v3), with weights re-optimized under the full interference-aware response for each configuration and its reference, shows **11 of 19 survive** (v2: 4/10, v3: 7/9). The v3 *nominal* winner (iso 1.095) fails at swept 0.891, while the v3 runner-up **validates at swept-normalized 1.1226** — the overall champion. Its structure repeats the v2 motif, amplified: near-uncoupled staggered heavy/soft oscillators with a single strong bond (0.23) doing the hybridization. Data: `results-mc/swept_validation.pkl`, `results-mc/mc_results_v3_extended.pkl`, figures `mc_results_v3.pdf`, `mc_best_configs_v3.pdf`.

**Bottom line**: coupled configurations beating their uncoupled reference exist and survive honest interference-aware evaluation, with a validated margin of **+12.3%**; but the best absolute sensitivity in the explored pool still belongs to an uncoupled reference, and the swept check is mandatory before quoting any individual winner.

---

### 7. Normalized-Objective Run (v4): Sympathetic Rescue and the Reference-Sabotage Exploit (2026-08-19)

The search objective was switched from raw to reference-normalized sensitivity (each candidate scored against its own uncoupled twin's analytic optimum). The run behaved as designed in one sense — no more frequency-scale pinning as the driver — but exposed a sharper issue:

- Under strain forcing, the reference peak is analytically **b_ref,i = m_i/(2γ)** (the wall spring cancels), so the reference's minimax depends only on the masses.
- The chain therefore learned to **sabotage its own reference**: every top configuration places one oscillator at the mass floor (m = 0.1), making the uncoupled twin's worst mode terrible, then couples that site so the full system rescues it. 90/100 refined candidates score > 1, max **6.6** — and the top configuration *survives* the swept-interference validation at **6.98**.
- The rescue mechanism is genuine physics (**sympathetic driving**: a nearly-undriven oscillator detected through coupling to strongly-driven neighbors) and is now a documented, validated effect. But the headline ratio measures how bad the uncoupled twin was made, not how good the coupled detector is: in absolute terms the v4 winners (raw ≤ 32.7) fall below v3's (37.3) and well below the analytic best uncoupled design in the box (all masses at the cap: **47.4**).

**Next objective (proposed)**: normalize by the fixed box-optimal uncoupled scale (a constant — 47.43 for strain, mass cap 3.0, n = 10) so the denominator cannot be gamed; a score > 1 would then mean beating the *best possible* uncoupled design, the strongest form of the research claim.

---

### 8. Box-Normalized Runs (v5): The Definitive Comparison (2026-08-20)

The proposed objective was implemented: scores are now sensitivity divided by the analytic box-optimal uncoupled scale (47.4342 — constant, ungameable; the uncoupled minimax is separable per site so the optimum is exact). Two 20000-step Metropolis runs:

| Run | Weighting | Best box-normalized score |
|---|---|---|
| v5 (balanced, α = 0.5) | sensitivity + spread | 0.787 (raw 37.3); 0.873 under swept metric |
| v5b (sensitivity only, α = 1.0) | sensitivity | **0.9974** (raw 47.31); none of 100 configs above 1 |

**Headline: with the objective made honest, coupling *ties* the best possible uncoupled detector (99.7%) but does not beat it** under the isolated-peak metric. The near-tie design is nine heavy oscillators plus one light site (m = 0.25) rescued through its couplings — the sympathetic-driving motif used legitimately. At balanced weighting, coupling's real purchase is bandwidth: the coupled leader spans 2.6 relative spread where the uncoupled optimum is nearly monochromatic, at ~21% sensitivity cost.

The swept metric puts the v5b leader at 1.058 × the iso bar, but that crossing is not quotable: the swept metric has a degeneracy loophole on both sides (an uncoupled stack with all modes inside one linewidth formally scores ~√n × its iso optimum, ≈ 474 here, because one coherent response is counted n times), and the winning config itself contains a 0.23-linewidth pair. **Open problem for the next session: a resolvability-aware figure of merit** (minimum mode spacing, band-integrated response, or Fisher information) — both current metrics mis-score unresolvable modes, in opposite directions.

Artifacts: `mc_results_v5_boxnorm.pkl` / `mc_results_v5.pdf` (balanced), `mc_results_v5_sensonly.pkl` / `mc_results_v5b.pdf` (sensitivity-only), logs `run_v5.log` / `run_v5b.log`.

---

### 9. Corrected Physics: One Coupling Spring Beats Every Uncoupled Detector (2026-08-20)

A verification of Section 8's near-tie uncovered a critical bug present since the package's creation: `solve()` unpacked LAPACK dgeev in the wrong order and used the **left** eigenvectors of the non-symmetric dynamical matrix as the modal basis. For unequal masses this basis is neither the right eigenvectors nor M-orthonormal, so all previous unequal-mass numbers were quantitatively wrong. The fix replaces dgeev with the symmetric-definite generalized eigenproblem `eigh(K, M)`, storing M-orthonormal modes (VᵀMV = I), for which the package's force-projection and response-reconstruction conventions are exactly correct. Verified to machine precision. A sign-equalizer warm start was also added to `optimize_observable` after Nelder-Mead was caught undershooting by ~2%.

**The corrected physics simplifies the theory beautifully**: under strain forcing, an uncoupled oscillator's observable sensitivity is Sᵢ = |wᵢ|·Lh₀/(2γ) — masses and springs cancel — so **every possible uncoupled design scores exactly Lh₀/(2γ√n) = 15.8114**, a universal bound verified numerically.

**And the central claim survives, stronger**: the best known coupled configuration scores **1.073 × the universal uncoupled bound**, and a reduced version with **a single coupling spring** (one bond rescuing one light oscillator, m = 0.25, via sympathetic driving) retains 1.073. This resolves the three publishable routes (isolated-peak metric):

| Route | Status |
|---|---|
| 1. Uncoupled always more sensitive | **Disproven** by explicit construction |
| 2. Minimum coupled oscillators to beat uncoupled | **One bond / two oscillators, +7.3%** |
| 3. Best design under a coupling floor | Re-running under corrected physics (v6+) |

Open items: the corrected-physics search (Run 6) for the true optimum; the coupling-floor sweep for route 3; the swept-metric/resolvability caveat from Section 8 still applies to any final quoted number.

---

### 10. Route 3 Complete: The Coupling-Floor Curve (2026-08-20)

With the resolvability constraint in place, the full route-3 sweep answers the lab's actual question — *given that every bond will have some parasitic coupling, what is the best achievable sensitivity?* All nine bonds were forced ≥ each floor (hard sampling bounds), each floor's winner polished under the same constraints, and every design validated on the conservative swept metric (figure: `results-mc/route3_curve.pdf`; data: `results-mc/route3_sweep.pkl`).

**Headline results** (S/S_unc, swept metric, all modes ≥ 3 linewidths apart):

- **At every floor up to k = 0.2 there exists a lab-feasible design that beats the uncoupled bound** (the curve plots the *optimized best* per floor — a typical random design at any floor scores well below 1; in the v7 run only 27 of the top-100 refined candidates, themselves the best of 20000 chain samples, exceeded the bound). Parasitic coupling in this range is a *feature* when the rest of the design is tuned around it: the best lab-feasible design (floor 0.02) reaches **1.208**, i.e. +21% over the best possible uncoupled detector.
- **Fully-coupled designs beat the sparse one-bond champion** (1.208 vs 1.018): with all bonds active, the optimizer distributes many small sympathetic rescues instead of one.
- **The tolerance limit is k ≈ 0.3**: the swept curve crosses 1 between floors 0.2 and 0.5, marking the maximum parasitic coupling at which a real device still matches the uncoupled ideal.

**All three publishable routes are now resolved** (strain forcing, n = 10, constant-Q damping): (1) "uncoupled always wins" is false; (2) one bond suffices to beat the bound (+1.8% swept); (3) the full feasibility curve above, peaking at +21%. Remaining hardening steps: multi-seed sweep to smooth the curve, optimizing directly under the swept metric (the iso–swept gap is up to 23%), re-running the demo notebooks post-bug-fix, and committing the work.

Artifacts: `results-mc/mc_results_v2.pkl`, `results-mc/mc_results_v2.pdf`, `results-mc/run_v2.log`.

---

## Visualizations

**Baseline figures** (`figures-new/`):
1. **force_model_comparison.pdf** — Generalized forces and peak sensitivities across all three force models.
2. **frequency_spread.pdf** — Mode spacing comparison between coupled and non-interacting systems.
3. **observable_comparison_gw.pdf** — Sensitivity profiles for GW strain forcing.
4. **observable_comparison_dm.pdf** — Sensitivity profiles for DM uniform impulse forcing.

**Monte Carlo figures** (`figures-mc/`):
5. **mc_gw_balanced.pdf** — Pareto scatter + best-config sensitivity profile (GW strain, α = 0.5).
6. **mc_priority_comparison.pdf** — Side-by-side sensitivity profiles for α ∈ {0.9, 0.5, 0.1}.
7. **mc_dm_balanced.pdf** — Pareto scatter + best-config sensitivity profile (DM uniform, α = 0.5).
8. **mc_screening_cloud.pdf** — Hexbin density of all Phase 1 draws with Pareto front overlaid.
9. **mc_best_observable.pdf** — Full compare_observables plot for the GW-strain best configuration.

---

## Summary

| Result | Value |
|--------|-------|
| Coupling-induced bandwidth gain (baseline) | +13.7% |
| Mode spacing uniformity improvement (baseline) | −21% CV |
| Best normalized sensitivity — baseline (coherent DM) | 0.787 |
| All baseline models below reference threshold | Yes |
| MC search space dimensionality | 29 (10 masses + 10 wall k + 9 coupling k) |
| MC Phase 1 draws | 5000 (v1) / 20000 (v2) |
| MC Phase 2 refined candidates | 100 |
| Best normalized sensitivity — Metropolis v2 | **1.0625** (> 1: beats reference) |
| Configurations beating reference — Metropolis v2 | 10 / 100 |

The Monte Carlo search confirms that the baseline parameter choices are suboptimal and that better configurations exist in the parameter space — and the v2 Metropolis run shows that some of them **exceed the uncoupled reference ceiling** (normalized sensitivity > 1). The Pareto front provides a principled menu of trade-offs between sensitivity and bandwidth for experimental design. Future directions include larger search spaces (varying n, damping, force profiles), Bayesian optimization to accelerate convergence, and non-linear observables to push past the reference-scale ceiling.
