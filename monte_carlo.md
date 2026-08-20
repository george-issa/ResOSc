# Monte Carlo Tuning Notes

## Route-3 sweep — 2026-08-20 (coupling floor)

Pipeline per floor K_MIN ∈ {0.01, 0.02, 0.05, 0.1, 0.2, 0.5}: constrained
Metropolis search (all nine bonds ≥ K_MIN as hard sampling bounds, α = 1,
min gap 3 lw, step 0.2) → floor-constrained polish (equalizer objective,
29 params) → swept-metric evaluation. Script preserved via
`results-mc/route3_sweep.log`; data `results-mc/route3_sweep.pkl` (incl.
polished parameters per floor); figure `results-mc/route3_curve.pdf`.

| floor | search | polished iso | swept | min gap |
|---|---|---|---|---|
| 0.01 | 1.035 | 1.125 | 1.038 | 3.00 |
| 0.02 | 1.054 | 1.251 | **1.208** | 3.00 |
| 0.05 | 0.995 | 1.136 | 1.036 | 3.00 |
| 0.1  | 1.128 | **1.326** | 1.081 | 3.00 |
| 0.2  | 1.113 | 1.176 | 1.019 | 3.00 |
| 0.5  | 0.955 | 1.023 | 0.926 | 3.00 |

**Findings**:
1. **Lab-feasible fully-coupled designs beat the uncoupled bound at every
   floor up to 0.2, under both metrics.** Parasitic coupling is a feature,
   not a bug, in this regime.
2. **They beat the sparse one-bond champion too**: best swept 1.208 (floor
   0.02) vs the one-bond design's 1.018. With every bond active, the
   polish can distribute many small sympathetic rescues.
3. The swept curve crosses 1 between floors 0.2 and 0.5 — the maximum
   tolerable parasitic coupling before a real device falls below the
   uncoupled ideal is roughly k ≈ 0.3 (interpolated).
4. The curve is non-monotonic and the iso–swept gap varies (up to 23% at
   floor 0.1) — residual 3-lw peak-tail interference; the swept column is
   the quotable one. Non-monotonicity partly reflects search noise
   (single seed per floor); a multi-seed pass would smooth it.

## Run 7 — 2026-08-20 (resolvability constraint)

Code change: `_resolvable(system, min_gap_linewidths)` in `montecarlo.py`
rejects any configuration whose adjacent mode gaps fall below
`min_gap_linewidths` × (2γω). Enforced at every sampling point in both
optimizers (Phase 1 draws, Metropolis initial draw and proposals).
Default 3.0; CLI flag `--min-gap` (0 disables); CONFIG constant
`MIN_GAP_LINEWIDTHS` in `run_mc.py`.

**Constrained one-bond polish** (equalizer objective + smooth feasibility
ramp, 12 restarts; parameters in
`results-mc/one_bond_champion_resolvable.npy`):

| Quantity | Value |
|---|---|
| iso minimax | 17.3155 = **1.0951 ×** the universal uncoupled bound |
| swept minimax | 16.0947 = **1.0179 ×** |
| min mode gap | 3.00 linewidths (constraint active at the optimum) |
| coupling | single bond, k = 0.067 (not at the floor — genuine) |
| masses | nine heavy (2.6–3.0) + one light m = 0.13 (rescued site) |

Both metrics now agree the design beats every uncoupled detector; the
margin is metric-dependent because at 3-linewidth spacing a neighboring
peak's tail is still ~1/6 of its height, so the isolated-peak
approximation overestimates by ~7%. **Quote the conservative swept
number: +1.8%** (or tighten the constraint / optimize under the swept
objective directly to close the gap — future work).

**Constrained full search** (`mc_results_v7_resolvable.pkl`, `run_v7.log`,
figure `mc_results_v7.pdf`): best 1.0161 (iso), 27/100 candidates above 1,
winner fully resolvable (min gap 4.12 lw). It did not beat the polished
one-bond champion — targeted polish remains the better tool for the final
number, with the MC search serving as motif discovery. Note: the final
acceptance rate fell to 10.4% because the resolvability rejection thins
the feasible region; a future constrained run should use a smaller
step size (~0.2) to compensate.

---

## ⚠ CRITICAL BUG FIX — 2026-08-20: left/right eigenvector swap

`CoupledSystem.solve()` unpacked LAPACK dgeev as
`wr, wi, vr, vi, info = dgeev(H)`, but dgeev returns
`(wr, wi, vl, vr, info)` — so the **left** eigenvectors of the
non-symmetric H = M⁻¹K were used as the modal matrix everywhere (force
projection *and* spatial reconstruction). Proof: stored vectors satisfied
aᵀH = λaᵀ to 1e-14 but ‖Ha − λa‖ ≈ 0.9. For unequal masses this basis is
neither the right eigenvectors nor M-orthonormal, so **every unequal-mass
number produced before this fix is quantitatively wrong** — the baseline
results and MC runs v1–v5 included. (Equal-mass systems were unaffected:
symmetric H ⇒ left = right.)

**Fix**: `solve()` now uses the symmetric-definite generalized
eigenproblem `scipy.linalg.eigh(K, M)` and stores rows = Vᵀ with
VᵀMV = I. With that convention the existing code paths `q = A f` and
`x = b @ A` are both exactly correct. Verified: K V = ω²MV residual
~1e-15; M-orthonormality ~1e-15.

Two consequences of the corrected physics (strain forcing, iso metric):

1. **Universal uncoupled bound**: S_i = |w_i|·Lh₀/(2γ) — masses *and*
   springs cancel — so every uncoupled design scores exactly
   Lh₀/(2γ√n) = **15.8114** (verified numerically on random designs).
   The box-normalization constant is parameter-free.
2. `_reference_scale` / `_box_optimal_uncoupled_scale` were re-derived
   (g_i = |f_i|/(2γk_ii)); `optimize_observable` gained a sign-equalizer
   warm start (w = A⁻¹(σ/b), best sign pattern) after random-restart
   Nelder-Mead was caught undershooting the optimum by ~2%.

**First corrected results**: the (wrong-physics) v5b winner re-evaluated
correctly scores **1.0733 × the universal uncoupled bound** (16.97 vs
15.81), and keeping only its bond 2 — ONE coupling spring, rescuing the
light site m = 0.25 — retains 1.0730. A 300-step smoke search under
corrected physics already finds 1.0367. Full corrected run: Run 6
(`mc_results_v6_fixedphysics.pkl`, `run_v6.log`).

### Run 6 + one-bond polish (2026-08-20, corrected physics)

The corrected-physics search (`mc_results_v6_fixedphysics.pkl`, α = 1.0)
found 1.0136 — it crossed the line but landed below the known champion,
because the Phase 1 proxy (harmonic-L2 of modal peaks) under-values the
mode-leverage the winning motif exploits. Direct local polish of the
one-bond topology (equalizer objective, 21 log-params) then mapped the
landscape:

| Design | min gap (linewidths) | iso score | swept score* |
|---|---|---|---|
| One-bond champion (k = 0.109) | 0.23 | **1.0733** | 1.1333 |
| Polished one-bond (k → 0.001 floor) | 0.05 | 1.1009 | 1.0016 |

*swept score = swept-optimized minimax / 15.8114 (the iso uncoupled bound).

**The polish exposed the decisive issue**: the optimizer tunes two sites
into bare-frequency resonance and drives the coupling to the floor —
resonant hybridization at infinitesimal coupling. The resulting doublet
is unresolvable (0.05 linewidths) and the iso "win" evaporates under the
swept metric (1.0016 ≈ tie). The unpolished champion is marginal (0.23
linewidths) but *gains* under the swept metric (1.133). Meanwhile the
swept uncoupled bar is itself gameable to n× via full degeneracy, so no
cross-metric claim is clean.

**Conclusion: a resolvability constraint (minimum mode spacing ≥ a few
linewidths) must be built into the search and the metric before any final
number is quoted.** With it, both metrics agree and neither loophole
operates. The route-3 coupling-floor sweep should run only after that
constraint exists — its α = 1.0 results would otherwise sit in the same
marginal-overlap territory.

**Standing answers to the three publishable routes** (iso metric):
- Route 1 (uncoupled always wins): **FALSE** — disproven by explicit
  construction.
- Route 2 (minimum coupled oscillators to beat uncoupled): **one bond
  (two coupled oscillators) suffices**, +7.3% over the universal bound.
- Route 3 (best feasible system with a coupling floor): the k_min sweep
  should be re-run under corrected physics after Run 6.

---

## Run 5 — 2026-08-20 (box-normalized objective) — ⚠ INVALIDATED by the bug fix above

Code change: scores are now `sensitivity / box_scale`, where
`_box_optimal_uncoupled_scale()` is the **analytic best possible uncoupled
design in the search box** — a constant (47.4342 for strain, mass cap 3.0,
n = 10: all masses at the cap give b = m/(2γ) = 150 each, minimax
150/√10). The uncoupled minimax is separable per site, so the box optimum
is exact. Score 1.0 = "matches the best uncoupled detector this box
allows"; the denominator cannot be gamed. Results carry
`'objective': 'box-normalized'` + `'box_scale'`; plots label the axis
"Min. Sensitivity / Best Uncoupled" with the win-line at 1. The
per-config own-reference ratio is still recorded/printed as a diagnostic.

Since the divisor is constant, the search dynamics are exactly the raw
search (verified: the v5 chain reproduces v3's trajectory step-for-step at
the same seed, energies shifted by α·log box_scale) — the change is the
yardstick, which is the point.

| Quantity | Value (α = 0.5, balanced) |
|---|---|
| Box scale (strain, this box) | 47.4342 |
| Best box-normalized score | **0.787** (raw 37.33) |
| ... under the swept metric | **0.873** (swept raw 41.39) |
| Pool median box-norm | 0.596 |
| Own-ref ratio of the raw-best config | 0.936 (its couplings *hurt* it) |

Results: `results-mc/mc_results_v5_boxnorm.pkl`, figure
`results-mc/mc_results_v5.pdf`, log `run_v5.log`. A sensitivity-only run
(α = 1.0, `run_v5b.log`, `mc_results_v5_sensonly.pkl`) gives coupling its
best shot at the line — see below.

### Run 5b — sensitivity-only (α = 1.0)

With the spread term removed, the chain gets coupling's best shot at the
line — and lands almost exactly on it: **best box-normalized 0.9974**
(raw 47.31 vs the 47.43 optimum; 6 configs above 0.95, none above 1).
The winning design uses the sympathetic-rescue motif legitimately: nine
heavy sites plus one light site (m = 0.25) fed through its couplings
(own-ref ratio 3.87), with staggered soft/stiff walls.

Under the swept metric the same config reaches raw 50.19 = **1.058 × the
iso box scale** — but this crossing must NOT be quoted as "coupling beats
the best uncoupled detector", because the swept metric has a degeneracy
loophole on both sides: an uncoupled stack with all bare modes inside one
linewidth responds coherently to a single measurement and formally scores
up to ~√n × its iso optimum (≈ 474 here) — the minimax then counts one
giant response n times. The winning coupled config itself contains a
0.23-linewidth pair, i.e. it partly exploits the same loophole. The only
clean comparison is the isolated-peak one, where the verdict is:
**coupling ties the best possible uncoupled design (99.7%) but does not
beat it.**

The open metric-design question this exposes: both metrics mis-score
unresolvable modes (iso ignores overlap; swept double-counts it). A
resolvability-aware figure of merit — e.g. requiring mode spacing above a
linewidth, band-integrated response, or a Fisher-information formulation —
is the right next step before drawing final conclusions.

Files: `results-mc/mc_results_v5_sensonly.pkl`, `run_v5b.log`.

### Interpretation (α = 0.5, balanced)

At balanced weighting, **no coupled configuration reaches the best
possible uncoupled design** — the best gets to 79% (87% with interference
exploitation). Notably, the pool's absolute best performer would improve
by *deleting* its coupling springs (own-ref 0.936 < 1): what coupling
buys in this regime is *spread* (2.6 vs 0 for the all-equal-mass uncoupled
optimum, which parks every mode at nearly the same frequency), at a cost
in worst-case sensitivity.

Caveat on the yardstick: 47.43 is exact for the isolated-peak metric.
Under the swept metric, deliberately near-degenerate uncoupled designs can
exceed their iso optimum (seen repeatedly in the validation batches), so
the swept-metric "best possible uncoupled" bar sits somewhat *above*
47.43 — making 0.873 an upper-bound-flattered number. The negative
conclusion is therefore conservative-safe: coupling has not beaten the
best uncoupled design.

---

## Run 4 — 2026-08-19 (normalized objective)

Code change: the search objective (Phase 1 energy and Phase 2 ranking) now
uses the **reference-normalized** sensitivity — each candidate's proxy /
minimax divided by its own uncoupled twin's analytic optimum
(`_reference_scale` / `_normalized_proxy` in `montecarlo.py`; results carry
`'objective': 'normalized'` and plots switch axes accordingly). Same run
config as Run 3 otherwise.

| Quantity | Value |
|---|---|
| Acceptance rate (final) | 52.2% |
| Best energy | -1.5655 at step 19784 (final -1.5592) |
| Normalized > 1 | 90 / 100, max **6.615** (iso) |
| Swept validation of the top config | **6.98 — survives** |
| Best raw coupled minimax | 32.68 |
| Wall time | 3m 24s |

Results: `results-mc/mc_results_v4_normobj.pkl`, figure
`results-mc/mc_results_v4.pdf`, log `results-mc/run_v4.log`.

### The new exploit: reference sabotage

The normalized objective fixed the frequency-scale pathology but exposed a
new one. Under strain forcing the reference peak has the closed form

    b_ref_i = f_i / (2γ ω⁰ᵢ²) = k_ii / (2γ k_ii/m_i) = m_i / (2γ)

— the wall spring cancels, so the uncoupled reference's minimax depends
**only on the masses**: ref_scale = (1/2γ)·harmonic-L2(m). The chain
discovered that the cheapest way to raise the ratio is to *lower the
denominator*: every top configuration plants one oscillator at the mass
floor (m = 0.1 → b_ref = 5.00 exactly), which cripples its own reference,
then couples that site to the chain so the coupled system rescues it.

The rescue is real physics — **sympathetic driving**: an oscillator that
receives almost no direct force can be driven through its coupling to
strongly-driven neighbors. The 6.6× ratio survives the swept-interference
validation (6.98 with re-optimized weights). But it answers "how much does
coupling help hardware chosen so that no-coupling is terrible?" — not "how
good can a coupled detector be?" In absolute terms the v4 winners (raw
≤ 32.7) sit *below* the v3 winners (37.3), and far below the analytic
best uncoupled design in this box (all masses at the cap: 3.0/(2γ√n)·√n
form → 150/√10 = **47.43**).

### Recommended next objective: fixed best-uncoupled normalization

Normalize by the *box-optimal* uncoupled scale — a constant, e.g. 47.43 for
strain forcing with mass cap 3.0 and n = 10 — instead of each candidate's
own twin. Then the denominator cannot be gamed, the score is scale-safe
(bounded by the mass cap), and score > 1 would mean the coupled design
beats the **best possible** uncoupled design — the strongest version of the
research claim. (For uniform forcing b_ref = m/(2γk) depends on the wall
springs too, so the constant must be recomputed per force model.)

---

## Run 3 — 2026-08-19 (extended bounds, Metropolis only)

New default configuration (now baked into `run_mc.py`): Metropolis-only,
extended search box — masses [0.1, 3.0], wall springs [0.2, 6.0],
couplings [0.001, 1.0] (bounds are also CLI flags now). 20000 steps,
100 refined, step 0.40, T: 3.0 → 0.01.

| Quantity | Value |
|---|---|
| Acceptance rate (final) | 52.3% |
| Best energy | -2.5689 at step 19354 (final -2.5443) |
| Best raw minimax sensitivity | 37.33 |
| Best relative spread | ~5.1 |
| Pareto front size | 20 |
| Nominal (isolated-peak) normalized > 1 | 9 / 100, max **1.0950** |
| Wall time | 3m 38s |

Results: `results-mc/mc_results_v3_extended.pkl`, figure
`results-mc/mc_results_v3.pdf`, log `results-mc/run_v3.log`.

### Findings

1. **The raw-sensitivity objective is scale-unbounded.** The winners pinned
   at the *new* box edges exactly as v2 pinned at the old ones: masses at
   3.0, wall springs at 0.2 (and one at 6.0). Heavy + soft → low ω, and
   b ∝ 1/(2γω²) grows without an interior optimum, so any box will pin.
   Both the Phase 1 energy and the Phase 2 ranking use *raw* sensitivity,
   so widening the box mostly chased overall scale, not coupling structure.
   **Recommendation for the next code change**: rank by *normalized*
   sensitivity instead — the reference peaks are analytic for a diagonal
   system (ω²ᵢ = kᵢᵢ/mᵢ, bᵢ = |fᵢ|/(2γω²ᵢ)), so the normalized proxy is
   just as cheap as the raw one.
2. **Couplings collapsed toward the new floor** (mostly 0.001–0.03): with
   the floor lowered, the chain runs toward the uncoupled limit, where
   normalized → 1 by continuity. Normalized sensitivity is uncorrelated
   with mean coupling strength across the refined pool (r ≈ -0.04).
3. **Batch swept-metric validation of all 19 nominal winners (v2 + v3)**:
   **11 of 19 survive** (v2: 4/10, v3: 7/9); full table in
   `results-mc/swept_validation.pkl`. The v3 *nominal* winner
   (iso 1.0950) fails at swept 0.891 — an artifact of the isolated-peak
   approximation — but the v3 runner-up **validates at swept-normalized
   1.1226**, making it the overall validated champion (ahead of the v2
   winner's 1.089). The iso and swept rankings disagree candidate-by-
   candidate, so the swept check is mandatory before quoting any winner.
   Validated champion (swept 1.1226, iso 1.0912, rel. spread 3.89):

   ```
   masses    : [2.502 1.837 3.    1.134 3.    3.    3.    3.    3.    0.971]
   wall k    : [0.234 0.296 0.898 5.983 0.2   0.736 0.356 0.206 1.896 0.539]
   coupling k: [0.004 0.005 0.001 0.001 0.012 0.01  0.018 0.002 0.228]
   ```

   Same motif as v2, amplified: near-uncoupled staggered oscillators, with
   one strong bond (0.23 on bond 9) doing the hybridization work.
4. Best-vs-best on absolute sensitivity remains uncoupled-favored:
   best coupled raw 37.33 vs best uncoupled reference 39.88 in the pool.

---

## Run 2 — 2026-08-19 (fixes applied)

All recommendations from the baseline run were applied before this run:
best-ever state tracking was added to `metropolis_optimize` (plus
deduplication of repeated chain states before Phase 2 selection, so all
refined candidates are distinct), and the CONFIG was updated to the
recommended values.

| Parameter | Value |
|---|---|
| `N` | 10 |
| `N_SAMPLES` | 20000 |
| `N_REFINE` | 100 |
| `STEP_SIZE` | 0.40 |
| `T_START` | 3.0 |
| `T_END` | 0.01 |
| Acceptance rate (final) | 46.3% |
| Best energy | -1.7922 at step 19838 |
| Energy at end of chain | -1.7922 (no drift — fix works) |
| Pareto front size | 9 (Metropolis), 7 (random) |
| Wall time | 7m 47s (both methods) |

### Diagnostics checklist verdicts

- [ ] Acceptance rate at first milestone (step 4000) below 70% — **87.4%, not met.**
  Partly an artifact of T_START=3.0 (T is still ~0.96 at step 4000, so high
  acceptance is expected early). Per the escalation rule below, try
  STEP_SIZE = 0.6 next run if early mixing matters.
- [x] Acceptance rate at last milestone below 50% — 46.3%
- [x] Energy at end of chain close to (or below) the mid-run minimum —
  best energy was found at step 19838 of 20000; the chain froze on it.
- [x] Pareto front has at least 5 points — 9
- [x] Best Metropolis sensitivity exceeds best random MC —
  raw 16.46 vs 11.61; normalized 1.0625 vs 0.9439

### Headline result

**Ten Metropolis-refined configurations exceed the uncoupled reference
threshold (normalized minimax sensitivity > 1), with a maximum of 1.0625.**
Random MC at the same budget found none (max 0.9439). This is the first
direct evidence for the project's motivating question: coupled
configurations exist that beat the non-interacting reference.

Best normalized-sensitivity configuration (norm. sens. 1.0625,
relative spread 2.70 — vs 0.40 for the hand-chosen baseline):

```
masses    : [1.145 1.181 0.807 1.5   1.061 1.5   0.506 1.5   0.3   1.5  ]
wall k    : [0.5   1.575 0.5   0.691 0.685 1.995 0.5   0.517 3.    0.518]
coupling k: [0.054 0.078 0.033 0.018 0.045 0.102 0.01  0.019 0.018]
```

Structural pattern across the >1 configurations: **weak couplings
(0.01–0.10) on top of strongly heterogeneous masses and wall springs**,
with several parameters pinned at their search bounds (masses at 1.5 and
0.3, one wall spring at 3.0). The bound-pinning suggests the optimum lies
outside the current search box — widening `mass_bounds` / `wall_bounds`
is a natural next experiment.

Results: `results-mc/mc_results_v2.pkl`, figure `results-mc/mc_results_v2.pdf`,
log `results-mc/run_v2.log`. The April baseline is preserved in
`results-mc/mc_results.pkl`.

### Interference validation of the >1 result (2026-08-19)

Concern: the winning configuration has three mode pairs closer than a
linewidth (gaps 0.4–0.5 × 2γω), and the search metric
S_i = |c_i|·|q_i|/(2γω_i²) treats each resonance in isolation. The physical
observable at driving frequency ω is the coherent sum over all modes,
O(ω) = |Σ_j w_j X_j(ω)|, and overlapping resonances interfere.

Check performed: evaluate the *swept* metric min_i O(ω_i*) — full complex
response at every resonance — for the best v2 configuration and its
uncoupled reference, with weights re-optimized for the swept metric on both
sides (identical 50-restart Nelder-Mead protocol).

| Quantity | Isolated-peak metric | Swept metric (re-optimized w) |
|---|---|---|
| Coupled winner | 11.68 | 12.59 |
| Uncoupled reference | 10.99 (analytic) | 11.56 |
| **Normalized** | **1.0625** | **1.0894** |

**The >1 result survives — and improves.** Interference is significant
(with the isolated-metric weights, the swept response drops to 6.03 coupled
/ 2.21 reference), but weights re-tuned to the true response exploit
coherent overlap on both sides, and the coupled system gains more.
Note the reference itself has two near-degenerate bare-frequency pairs
(gaps 0.0 and 0.1 linewidths), so the isolated approximation mismodeled
both systems roughly equally. One caveat: under the swept metric the
reference optimum is numerical rather than analytic, so the 1.089 ratio
compares two optimizer results (same protocol both sides) rather than a
lower bound against an exact ceiling.

---

## Run 1 (baseline)

| Parameter | Value |
|---|---|
| `N` | 10 |
| `N_SAMPLES` | 5000 |
| `N_REFINE` | 20 |
| `STEP_SIZE` | 0.15 |
| `T_START` | 2.0 |
| `T_END` | 0.05 |
| Acceptance rate | 71.7% |
| Energy at start | -0.18 |
| Energy at end | -1.19 |
| Energy minimum (mid-run) | -1.60 |

---

## Issues Identified

### 1. Acceptance rate too high (71.7% vs optimal ~23%)
The step size is too small. The chain takes tiny steps, almost never
rejects a proposal, and fails to explore the landscape. In 29 dimensions
the theoretically optimal acceptance rate for a Gaussian proposal is ~23%.
A rate above ~85% means the chain is essentially doing a slow random walk
with no meaningful direction.

### 2. Best energy found mid-run, not at the end
The chain reached its best energy (-1.60) during the run but ended at
-1.19 — it found a good region and then drifted away. The current
implementation refines the top N_REFINE states by score, which may not
include the actual best states visited. The best-ever state should be
tracked explicitly and forced into the refinement pool.

### 3. N_REFINE = 20 is too low
Only 20 candidates are passed to the full minimax optimization. This gives
a sparse, inaccurate Pareto front. The original default of 100 should be
restored.

### 4. Chain too short for 29 dimensions
5000 steps for a 29-dimensional parameter space is a very sparse sample.
The workstation can handle a much longer chain.

### 5. T_END too high
The chain does not freeze tightly enough at the end. A lower T_END forces
harder exploitation in the final stage and prevents the drift seen in issue 2.

---

## Recommended CONFIG for Next Run — applied 2026-08-19 (see Run 2)

```python
N_SAMPLES          = 20000
N_REFINE           = 100
STEP_SIZE          = 0.40    # was 0.15 — increase to lower acceptance rate
T_START            = 3.0     # was 2.0  — broader early exploration
T_END              = 0.01    # was 0.05 — harder freeze at end
```

Expected effect: acceptance rate should drop to 20–50% by the midpoint
of the run. Check the printed milestone output:

```
Step 4000/20000 | T=... | E=... | accept rate so far: XX%
```

If the rate is still above 60% at step 4000, increase STEP_SIZE further to 0.6.

---

## Recommended Code Fix: Track Best-Ever State — applied 2026-08-19

The chain currently records the state after each accept/reject decision.
If the best configuration is found mid-run and then abandoned, it may not
appear in the top N_REFINE candidates. Add best-state tracking to
`metropolis_optimize` in `ResOSc/montecarlo.py`:

```python
best_E      = current_E
best_params = [p.copy() for p in current_params]

# inside the step loop, after accept/reject:
if current_E < best_E:
    best_E      = current_E
    best_params = [p.copy() for p in current_params]
```

Then insert `best_params` into the refinement pool before selecting
the top N_REFINE candidates by score, so the global best is always refined.

---

## Diagnostics Checklist

After every run, verify:

- [ ] Acceptance rate at first milestone (step N/5) is below 70%
- [ ] Acceptance rate at last milestone (step N) is below 50%
- [ ] Energy at end of chain is close to (or below) the mid-run minimum
- [ ] Pareto front has at least 5 points (increase N_REFINE if not)
- [ ] Best Metropolis sensitivity exceeds best random MC sensitivity
