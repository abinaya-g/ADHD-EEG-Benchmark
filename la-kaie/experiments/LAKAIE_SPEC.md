# LA-KAIE specification (implementation: `lakaie/algorithms/lakaie.py`, `lakaie/components.py`)

Working name: **LA-KAIE — Landscape-Aware Knowledge-Guided Adaptive Information
Exchange** (provisional; see `literature/NOVELTY_ASSESSMENT.md` — the name
prefix collides with LA-BHH, 2026, and the novelty claim is under review).

All symbols below are computed only from information available at the current
evaluation (no future information, no knowledge of f*). Every parameter is in
`configs/controller.yaml`; ablations in `configs/ablation.yaml`.

## 1. Architecture

Two populations P_HBA, P_MPA (N = 30 each). One outer iteration:

1. Phase 1 — HBA update + greedy acceptance (N FEs); MPA update, evaluation,
   marine memory (N FEs). Identical code to MPHB/MPHBS (`hybrid_common.py`).
   Schedules use FE progress: HBA α and MPA CF/phases are evaluated at
   t/T := FE_used/MaxFE with nominal horizon T_nom = MaxFE/(3N+E).
2. FADs stage with evaluation and per-agent greedy acceptance (N FEs); placement
   fixed to *before* the exchange phase for both domains (not tuned).
3. Population statistics, landscape state update (§3); delayed reward of a
   pending A1 decision (§6).
4. Every U = 20 iterations: refit interaction matrix G and groups (§2).
5. Exchange phase: up to E = 24 episodes; in each, the controller observes
   S = (S1..S10), selects an action a ∈ {A1..A8} (§4, §5), the candidate is
   built and evaluated (≤ 1 FE), greedy acceptance into the receiver slot
   (MPA marine memory synchronised), global best updated, reward (§6),
   controller update. Choosing A1 ends the phase: the unused evaluations are
   returned to the backbone (this is how the controller decides **when**
   to exchange).
6. Stop the loop when fewer than 3N FEs remain; the leftover (< 3N) is spent on
   exchange episodes with A1 disabled. The FE counter is checked before every
   evaluation; the budget is used exactly and never exceeded
   (`tests/test_lakaie.py::test_budget_exact_and_never_exceeded`).

## 2. Variable-interaction model

*Estimator (no extra FEs).* Every evaluated point enters a ring buffer
(3000 points). Around the current global best x*, take the
M = ⌈1.5·p⌉ nearest archive points (distance scaled by the box width),
p = 1 + D + D(D+1)/2. Standardise coordinates per variable
z = (x − x*)/s (s = sample std), transform fitness to rank-normal scores
y = Φ⁻¹((rank − ½)/M) and fit a full quadratic model by ridge least squares
(λ = 10⁻³·tr(ΦᵀΦ)/p, intercept unpenalised). With cross coefficients β_ij and
Hessian H (H_ii = 2β_ii, H_ij = β_ij), t_ij = β_ij/SE(β_ij) (OLS standard
error from the residual variance):

    G_ij = min(1, |H_ij| / sqrt(|H_ii||H_jj|)) · 1[|t_ij| ≥ t_crit],  t_crit = 3, G_ii = 0.

Smoothed G ← 0.5·G_old + 0.5·G_new. Rank-based fitting makes G invariant to
monotone transformations of f and to per-coordinate affine rescaling of x.
*Robustness for small D:* p is small, the rule is unchanged; if fewer than
p + 10 points exist, G is not updated (initially G = 0).
*Complexity:* O(M p² + p³) per refit, every U = 20 iterations
(D = 20: p = 231; D = 31: p = 528). This is the "computationally controlled"
version: a local quadratic surrogate refitted periodically instead of
perturbation-based detection (DG), which would cost extra FEs.

*Grouping (deterministic).* Constrained average-linkage agglomeration:
repeatedly merge the two clusters with the highest mean between-cluster G if
it is ≥ τ = 0.3 and the merged size ≤ g_max = 5 (ties: lowest indices).
Weakly interacting variables remain singletons.

*Summary.* S8 = mean_i max_j G_ij. Interaction consistency of a transferred
index set T: IC(T) = (kept − broken)/(kept + broken) over strong pairs
(G_ij ≥ τ) touching T (kept: both in T; broken: exactly one), 0 if none.

*FIR.* The main LA-KAIE uses the same black-box estimator for FIR. The
supplementary gray-box variant `FIRPrior` blends G 50/50 with the exact
coupling of the stopband energy quadratic form MSEs(h) = hᵀQh
(`FIRProblem.stopband_quadratic_form`), which follows from the objective
definition, not from evaluations. It is reported separately and never
compared as if it were black-box.

## 3. Landscape state (all in [0,1])

| | definition |
|---|---|
| S1 diversity | min(1, div_t/div_0); div = mean_j std_j(P_HBA ∪ P_MPA)/(ub_j − lb_j) |
| S2 improvement rate | Δ/(Δ + spread); Δ = f*_{t−10} − f*_t (global best), spread = median(f_pool) − f*_t |
| S3 stagnation | s/(s + 10), s = iterations since the last global-best improvement |
| S4 convergence/progress ratio | FE_used/MaxFE |
| S5 fitness gap | g_t / max(g over last 50 iterations and now), g = mean(f_pool) − f*_t |
| S6 HBA contribution | EMA(0.2) of the fraction of HBA offspring improving their parent |
| S7 MPA contribution | EMA(0.2) of ½(fraction of MPA moves kept by memory + FADs success rate) |
| S8 interaction strength | mean_i max_j G_ij |
| S9 exchange success | EMA(0.2) of exchange success indicators (A2–A8) |
| S10 exploration indicator | step/(step + radius); step = mean length of accepted Phase-1 moves, radius = mean distance of the pooled population to its centroid |

S1, S3, S5 and S9 are recomputed before every episode. All terms use only
comparisons, ranks and ratios of fitness differences → the state is invariant
to additive shifts of f (unit test).

## 4. Controller (ε-greedy linear contextual bandit)

Context x(S): each S_i is discretised into b = 3 equal-width bins, one-hot
encoded, plus a bias (31 features). For each action a a ridge value model
q_a(x) = θ_aᵀx, θ_a = A_a⁻¹b_a, with forgetting λ = 0.99 applied to the chosen
action on update:
A_a ← λA_a + xxᵀ + (1−λ)λ_r I, b_a ← λb_a + r x, λ_r = 1.
Policy: with probability ε(FE) = max(0.02, 0.2·(1 − FE/MaxFE)) a uniformly
random available action, otherwise argmax_a q_a(x) (ties random).
Availability mask: A2 at most once per exchange phase; A1 disabled during
the final leftover phase.
This differs from MPHBS's mediator: MPHBS stores a value per
(dimension, ranked row, source) and learns *which component* to copy; LA-KAIE
learns *which exchange mode* is useful in the *current landscape state*.

## 5. Actions (receiver r, donor d; candidate x; T = transferred index set)

Transfer units are single variables, or the current groups ("group level").
For A5–A8 the level is group if S8 ≥ 0.2 else variable (dynamic granularity).
"choose_sets": each unit is transferred with probability p_x = 0.2, at least one.

| action | construction | FEs |
|---|---|---|
| A1 no exchange | end exchange phase; budget returns to HBA–MPA | 0 |
| A2 elite individual exchange | best of the better population replaces the worst of the other (if better) | 0 |
| A3 dimension-wise cross-population | random receiver population; r, d by binary tournament; units = single variables | 1 |
| A4 interaction-group exchange | as A3 but units = interaction groups | 1 |
| A5 exploration-oriented | r uniform; d = member of the other population farthest from r (normalised) | 1 |
| A6 exploitation-oriented | r = best of the better population; d uniform among top-3 of the other; one unit chosen ∝ |x_d − x_r|₁ on that unit | 1 |
| A7 complementary-source | donor population drawn ∝ (S6, S7); r, d = the two population bests | 1 |
| A8 local refinement | r = global best; one unit u; x_u ← x_u + L z, LLᵀ = covariance of u over the pooled elite (top 25%) | 1 |

Candidates are clipped to the bounds; acceptance is greedy (x replaces r iff
f(x) < f(r)).

## 6. Knowledge-aware reward

    R = w1·FI + w2·DI + w3·SU + w4·IC − w5·COST − w6·S3·(1 − SU)

FI: rank gain = max(0, beat(f_x) − beat(f_r)) with beat(v) the fraction of the
pooled population worse than v; FI = 1 if x is a new global best.
DI: tanh(10·Δdiv/div) of the accepted move (0 if rejected).
SU: 1 if accepted else 0. IC: interaction consistency of T (A2: 1 if any
strong pair exists). COST: FEs used (1 or 0).
A1 (delayed, after the next backbone iteration; per-FE commensurable):
FI = SU = backbone per-offspring success rate, COST = 1, IC = 0, DI over the
iteration. Weights (1.0, 0.2, 0.3, 0.2, 0.2, 0.2) are global, identical for
all functions and both domains. `fitness_only` mode: R = FI.

## 7. Ablations (exactly one mechanism removed each)

Full; NoLandscape (bias-only context); NoInteraction (G replaced by a random
placebo matrix, same pipeline); NoKnowledgeReward (R = FI); DimensionOnly (all
transfers single-variable; G still estimated for S8/IC); RandomExchange
(uniform action); FixedPolicy (always A4); NoAdaptiveController (static
landscape rules, no learning). Supplementary FIR-only gray-box: FIRPrior.

## 8. Complexity

Per iteration: Phase 1 + FADs O(N·D + 3N·C_obj); state O(N·D) per episode;
controller O(k²) per update with k = 10b+1 = 31 (solve); actions O(N·D);
interaction refit O(M p² + p³)/U with p = Θ(D²) → O(D⁶/U) arithmetic, no FEs;
grouping O(D³ g_max). Space O(A·D + D² + p²) with archive size A = 3000.
FEs: exactly MaxFE.

## 9. Differences from MPHBS (design level)

| aspect | MPHBS | LA-KAIE |
|---|---|---|
| decision | which (rank, source) component per dimension | whether / which source / which granularity / exploration vs exploitation |
| state | dimension index | 10-dim landscape vector |
| learning | SARSA-style TD on Q(d,n,ch) | contextual bandit over 8 exchange modes |
| variable coupling | ignored (dimension-wise) | estimated G, group transfer, IC reward |
| reward | sign of improvement | multi-component, per-FE commensurable |
| FE use | 1 + N_sub·N_i per iteration, fixed | 0–24 per iteration, controller-decided |
| proxy screening | K candidates, 1 evaluated | none (each exchange candidate evaluated) |
