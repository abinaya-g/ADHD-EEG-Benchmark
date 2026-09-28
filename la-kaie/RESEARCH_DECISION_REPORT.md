# Research decision report — LA-KAIE after the pilot

Date: 2026-09-28. Status: **paused**. No 30-run experiment has been run, no
manuscript text has been written, and the LA-KAIE code has not been changed
since the pilot. This report only analyses evidence that already exists (the
paper, the literature search, the pilot data) plus a few new *diagnostics* of
the benchmark structure and pilot traces. The diagnostics do not re-run or
tune the method; their scripts are reproduced in §8.

**Evidence limits (read first).**
1. Full texts of arXiv, Springer and Zenodo are blocked by this environment's
   network policy. Apart from MPHBS (read in full, together with its code),
   every prior work below was assessed from its title, abstract and
   search-result snippets. Claims that a prior work *lacks* a feature are
   provisional and must be checked against its full text.
2. The pilot used 5 paired runs per instance. Its performance numbers are
   exploratory. None of the Holm-corrected instance-level tests can be
   significant at this sample size.

---

## 1. Inputs summarised

### 1.1 MPHBS (Flores & Olivares, KBS 352 (2026) 117070), from the full paper and code
* HBA and MPA evolve two populations with their native operators. After
  every iteration, a SARSA-style mediator assembles hybrid vectors dimension
  by dimension from fitness-ranked subsets of both populations. The Q-table is
  indexed by (dimension, rank, source). K proxy-screened candidates are
  generated per episode but only one is evaluated. Reward is the ternary sign
  of the improvement over the currently assembled vector, and acceptance is a
  relaxed simulated-annealing-like rule.
* The paper itself calls the update "a dimension-wise contextual-bandit-style
  update with TD smoothing", explicitly not a sequential MDP (text after Eq. 61).
* Domain-specific final settings: B-C1 for CEC2022 (FADs before the mediator,
  K = 7, N_i = 3) and A-C1 for FIR (FADs after, K = 3, N_i = 9), chosen by
  internal validation on the same benchmarks.
* The paper's own evidence: on CEC2022, B-C1 is statistically
  indistinguishable from MPHB and MPA. On FIR it wins all 8 cases but is not
  significant against MPHB, MPA or EODE after Holm. SARSA beats its matched
  RANDOM mirror on CEC2022 but not on FIR.
* Threats to validity **stated by the authors**: "strong variable coupling may
  reduce the benefit of independent component reuse" (Sec. 3.4), and the
  mechanism "may be less effective in landscapes where meaningful partial
  structures are … highly epistatic" (Sec. 7). This is the documented gap
  LA-KAIE was meant to address.
* Our reproduction: exact FE budgets, and 0/80 distributional differences
  from the authors' published runs (`experiments/baseline_validation.md`).

### 1.2 NOVELTY_ASSESSMENT.md, condensed
* Group-wise exchange driven by learned interactions: RV-GOMEA family,
  RVIntX (DE), VIGPSO, Adaptive Linkage Crossover, EMTO-IVG. **High overlap.**
* Interaction estimated from population data with no extra evaluations:
  VIGPSO and RVIntX. **Medium–high overlap.**
* Landscape-feature contextual bandit or RL operator selection: LA-BHH (2026,
  LinUCB, with a name-prefix collision), RL-HPSDE, DE-DDQN, RLDE-AFL. **High overlap.**
* Not found: a controller deciding *when*, *which source* and *which
  granularity* for exchange between two heterogeneous optimisers. This is at
  most an integration-level claim.

### 1.3 PILOT_REPORT.md, condensed (5 runs)
| | CEC2022 | FIR |
|---|---|---|
| LA-KAIE-Full vs MPHBS, instance W/T/L | 3/0/9 | 3/0/5 |
| avg rank Full vs MPHBS (comparison stage) | 3.42 vs 2.17 | 1.63 vs 1.63 |
| best LA-KAIE variant (all 12 methods) | FixedPolicy (always A4), rank 3.50 | NoKnowledgeReward, rank 3.00 |
| Full's rank among all 12 methods | 7.00 | 4.38 |
| group-level transfers | 15 % | 2 % |

### 1.4 Additional closest works found for the new directions (snippets only)
| work | relevance |
|---|---|
| *Enhancing Differential Evolution Utilizing Eigenvector-Based Crossover Operator* (Guo & Yang, IEEE TEVC) [link](https://ieeexplore.ieee.org/document/6698290/) | crossover in the population-covariance eigenbasis for rotation invariance (single population) |
| *An efficient eigenvector-based crossover for DE: rank-one updates* (AIMS Mathematics 2025) [link](https://www.aimspress.com/article/doi/10.3934/math.2025162?viewType=HTML) | cheaper eigen-crossover |
| *Linkage identification based on epistasis measures* (LIEM, IEEE) [link](https://ieeexplore.ieee.org/document/1004436) | pairwise epistasis measures for linkage groups (binary GA, perturbation-based) |
| *Improved evolutionary optimization from genetically adaptive multimethod search* (AMALGAM, PNAS) [link](https://www.pnas.org/doi/10.1073/pnas.0610471104) | adaptive allocation of offspring among algorithms by reproductive success |
| *Design and Analysis of Schemes for Adapting Migration Intervals in Parallel EAs* (Evol. Comput. 23(4)) [link](https://direct.mit.edu/evco/article-abstract/23/4/559/1023/) | adapting *when* to migrate from improvement signals |
| Extreme-value-based credit and dynamic MAB for AOS [link](https://link.springer.com/chapter/10.1007/978-3-540-87700-4_18); FRRMAB [link](https://www.researchgate.net/publication/256456340) | credit assignment and bandit selection for operators |

---

## A. Parts of LA-KAIE already present in prior work

| LA-KAIE component | Already present in | Assessment |
|---|---|---|
| Two heterogeneous populations exchanging information | MPHBS; multi-population / island models; AMALGAM | not novel |
| Transferring interacting variables together (group-preserving exchange) | RV-GOMEA (FOS mixing), RVIntX (interaction-aware masks), Adaptive Linkage Crossover, EMTO-IVG (group-level transfer) | not novel as a principle |
| Mixing at both univariate and multivariate granularity | GOMEA linkage-tree FOS (singletons plus merged sets) | not novel |
| Interaction estimated from evaluated data without extra FEs | VIGPSO (correlation of updates), RVIntX (online linkage learning) | not novel; our estimator (local quadratic with a t-filter) is a variant |
| Landscape features as controller state | RL-HPSDE (FDC, ruggedness), DE-DDQN (99 features), RLDE-AFL, LA-BHH (landscape + stagnation) | not novel |
| Contextual bandit / ε-greedy operator selection | LA-BHH (LinUCB), FRRMAB (UCB), dynamic MAB | not novel |
| Multi-component reward (improvement, diversity, cost, stagnation) | common in AOS / RL-EA work | not novel |
| Deciding *when* to exchange | adaptive migration-interval schemes (Evol. Comput.) | not novel as a principle |
| Allocating evaluations between search processes | AMALGAM | not novel as a principle |

## B. Genuinely distinguishable parts (provisional)

1. **Joint decision space for one exchange act.** A single controller chooses
   whether to exchange, the source population and the granularity, inside a
   heterogeneous HBA–MPA hybrid. No identical design was found. This is,
   however, a *combination* of known decisions; a reviewer would reasonably
   call it incremental.
2. **Interaction-consistency term in the reward.** This couples the credit of
   an exchange with the estimated dependency structure. No direct match was
   found, but it is a small design element and the pilot gives no evidence
   that it helps (NoKnowledgeReward ≥ Full on both domains).
3. **Positioning relative to MPHBS's own stated limitation** (dimension-wise
   reuse under coupling). This is a legitimate *motivation*, but the pilot
   shows LA-KAIE does not actually address it (see C).

Conclusion: nothing in LA-KAIE is clearly novel at the level of a KBS
methodological contribution. The only distinguishable element is the
integration, and it currently has no empirical support.

## C. Why the group mechanism is rarely activated

Four independent causes, each supported by a diagnostic:

1. **CEC2022 coupling is dense, not block-structured.** The rotation
   matrices of F1–F5 have 80–84 % of off-diagonal entries with |M_ij| > 0.05.
   Hybrid and composition functions (F6–F12, first D×D block) have 13–29 %.
   On a densely rotated function, every variable interacts weakly with every
   other, so no partition into small groups (g_max = 5) is a faithful model.
   A method built on "exchange small interacting groups" has the wrong
   structural prior for most of this benchmark.
2. **The local-quadratic estimator cannot see coupling on rugged landscapes.**
   With ideal isotropic samples around the *true optimum* (a diagnostic the
   algorithm never has), it detects coupling on smooth rotated functions
   (F1 Zakharov: 4–27 % of pairs with G ≥ 0.3), and a little on F2, but
   essentially none on F4 (Rastrigin), F5 (Levy) or F10 (composition).
   There, local ruggedness dominates any quadratic model and the |t| ≥ 3
   filter removes the coefficients.
3. **The estimate decays as the population converges.** In pilot runs S8
   peaks early (CEC mean max 0.13; FIR 0.18) and falls to 0.008 / 0.057 by
   the end. Late in the search the nearest archive points are near-duplicates
   concentrated along a few directions, so the regression design is
   ill-conditioned, standard errors inflate, and the significance filter
   removes everything.
4. **FIR has banded coupling that the black-box estimator misses.** The
   exact stopband quadratic form (gray-box) gives adjacent-coefficient
   coupling ≈ 0.3 for LPF/HPF, and 8–16 groups under the same τ and g_max,
   while the black-box estimate finds ~30.4 singletons out of 31. The
   gray-box FIRPrior variant beat Full on 6/8 cases (5 runs, n.s.), which
   hints that estimation, not group transfer itself, is the bottleneck on FIR.

Because S8 stays below the granularity threshold (0.2), A5–A8 almost always
run at variable level, and A4's groups are mostly singletons, so A4 behaves
like A3.

## D. What caused the pilot failure?

| candidate cause | evidence | verdict |
|---|---|---|
| **Proposed mechanism (premise)** | CEC2022 coupling is dense (C1); removing the group mechanism (DimensionOnly, NoInteraction) does not hurt on CEC | **major**: small-group exchange is the wrong structural prior for densely rotated problems |
| **Mechanism (credit design)** | A2 (copying the best over the worst, 0 FEs) is almost always "successful" (CEC 0.69, FIR 1.00) and gets the highest mean reward (+0.74 / +1.22) while adding no information. Backbone per-offspring success (A1 on CEC: 0.32) far exceeds exchange success (A4: 0.10–0.14), yet the variant that *always* exchanges (FixedPolicy) is best on CEC. The receiver-improvement signal is misaligned with progress of the global best. | **major**: the reward measures local repairs, not contribution to the final result |
| **Implementation** | estimator ill-conditioned near convergence (C3); A2's reward is trivially inflated (arguably a design bug rather than a deliberate choice) | **contributing** |
| **Parameterisation** | only pre-registered defaults were run; the sensitivity study was not run. FixedPolicy vs Full differs in the controller, not in any tuned constant. | **not excluded, but unlikely to be primary**; a parameter sweep cannot fix issues 1–2 |
| **Benchmark characteristics** | dense rotations in CEC2022; banded, weak couplings in FIR (C1, C4) | **major**: the benchmarks do not have the sparse group structure the method assumes |

On FIR, LA-KAIE's advantage over HBA, MPA and MPHB came mostly from **A8
(elite-covariance local refinement, 73 % of decisions)**, not from the
proposed exchange mechanism. A8 is a known kind of operator (covariance-based
sampling).

## E. Abandon, redesign, or retain?

* **Retain as is: no.** The core premise does not match the benchmarks, the
  credit signal is misaligned, the novelty is at most integration-level, and
  there is no pilot advantage over MPHBS.
* **Substantially redesign: possible, but only around a new, clearly stated
  gap.** Tweaking the current design (thresholds, weights, the estimator's
  t_crit) to beat MPHBS would be exactly the pilot-driven reverse-engineering
  that must be avoided.
* **Abandon the "landscape-aware group exchange" framing as the central
  contribution: recommended.** Keep the validated infrastructure: CEC2022 and
  FIR harnesses, the faithful MPHBS/MPHB/HBA/MPA ports, and the statistics,
  runner and validator. Also keep the pilot as a documented negative result
  and diagnostic.

Any new direction below should be developed on the disjoint tuning set only
(CEC2022 D = 10; FIR is not used for tuning), pre-registered, and evaluated
once on the test benchmarks. The present pilot should be reported as
exploratory evidence that motivated the change.

## F. Candidate methodological directions (no direction is selected here)

Notation: X^H, X^M are the populations; an exchange builds candidate x' from
receiver x_r and donor x_d with transfer mask m ∈ {0,1}^D:
x' = (1 − m) ⊙ x_r + m ⊙ x_d.

### F1. Outcome-based co-transfer epistasis learning (OB-CEL)

**Research gap.** Two families of linkage learning exist. Population-statistics
methods (correlations, mutual information, covariance, local surrogates)
measure how variables co-vary under selection or locally around the best;
the pilot shows the local-surrogate version collapses on rugged landscapes.
Perturbation-based methods (DG, LIEM, fitness-based linkage learning) pay
extra evaluations. MPHBS learns only *first-order* per-dimension values. Not
found: learning, at zero extra FE cost, whether transferring variables
*together* between two populations is better or worse than transferring them
separately, directly from the outcomes of the exchange evaluations that are
paid for anyway.

**Formulation.** For exchange e with mask m_e and outcome
y_e = 1[f(x'_e) < f(x_r)] (or a progress-aligned credit, see F3), fit an
online sparse second-order logistic model:

    logit P(y_e = 1 | m_e, s_e) = β_0 + Σ_i β_i m_i + Σ_{i<j} γ_ij m_i m_j,
    loss = Σ_e λ^{t−t_e} ℓ(y_e, ·) + κ_1 ‖γ‖_1 + κ_2 ‖β‖²,

with forgetting λ and source s_e ∈ {H→M, M→H}. γ_ij > 0 means the co-transfer
of i and j is super-additive. Masks are drawn from a design that guarantees
identifiability: pairwise-balanced random masks in an exploration fraction of
episodes. In the remaining episodes masks are sampled from the learned model,
e.g. greedy growth of m maximising β_0 + Σβ_i m_i + Σγ_ij m_i m_j under a size
limit, or Gibbs sampling at temperature T.

**Algorithmic mechanism.** HBA and MPA are kept unchanged; each iteration
spends E exchange evaluations; every outcome updates (β, γ) by online
proximal SGD in O(|m|²); masks follow the explore/exploit design above. No
surrogate of f is fitted.

**Difference from MPHBS.** MPHBS's Q(d, n, ch) is additive over dimensions
(a first-order credit model). OB-CEL contains a first-order model (γ = 0) as
a special case, so MPHBS-style dimension-wise reuse becomes a nested ablation.

**Difference from closest prior work.** RVIntX (online interaction learning
for DE crossover) is the closest. Its learning signal is not visible in the
abstract, and if it already learns from recombination outcomes the overlap is
high: **this must be checked in its full text before committing.** GOMEA and
VIGPSO learn from population statistics, LIEM from perturbations (extra
evaluations), and none of these do so between heterogeneous populations.

**Cost.** Zero extra FEs; O(E·k²) arithmetic per iteration for masks of size
≤ k, O(D²) memory.

**Testable hypotheses.** H1 (validity, not performance): on FIR, the learned
γ correlates with the exact gray-box stopband coupling (Spearman ρ > 0) far
more than a placebo; on the synthetic block-coupled test functions, γ recovers
the true pairs. H2: with equal FE budgets, OB-CEL masks outperform
size-matched random masks and the first-order (γ = 0) model.

**Ablations.** γ = 0 (first-order); random masks with the same size
distribution; γ from population correlation (VIGPSO-style); γ from the
local-quadratic estimator (current LA-KAIE); no forgetting; exploration
fraction 0.

**Risks.** Low signal-to-noise (pilot exchange success is ~10 %);
D(D−1)/2 = 190 / 465 parameters need thousands of outcomes (roughly 40k
exchanges per CEC run are available); dense CEC2022 coupling may again give
no exploitable sparse structure; the RVIntX overlap is unverified.

**KBS suitability.** Good if RVIntX does not already do this: it is a
knowledge-extraction mechanism (learning an interaction model from
optimisation experience) with an interpretable output (γ).

### F2. Frame-adaptive cross-population exchange (FACE)

**Research gap.** MPHBS's dimension-wise reuse, and any raw-coordinate group
exchange, are coordinate-dependent. CEC2022 F1–F5 are densely rotated, so in
raw coordinates no sparse structure exists (C1). Eigenbasis crossover exists
for a single DE population. Not found: exchange *between two heterogeneous
populations* in a jointly estimated frame, with an explicit choice between
the raw frame and a learned frame.

**Formulation.** From the pooled elite E (top q of X^H ∪ X^M), let
μ = mean(E), C = cov(E) + εI = BΛBᵀ. The exchange happens in eigen
coordinates:

    y_r = Bᵀ(x_r − μ),  y_d = Bᵀ(x_d − μ),  y' = (1 − m) ⊙ y_r + m ⊙ y_d,  x' = clip(μ + B y').

The frame is chosen per exchange, a_frame ∈ {raw, eigen}, by a two-armed
(or contextual) bandit with a progress-aligned credit. Eigen-components can be
ranked by λ_k, so the mask can target high- or low-variance directions.

**Algorithmic mechanism.** Refresh B every U iterations (O(D³), no FEs); the
MPHBS-style ranked memory, or plain tournament donors, then operate on the
coordinates y instead of x.

**Difference from MPHBS.** MPHBS reuses raw coordinates dimension by
dimension. FACE reuses components in a data-driven frame, which makes it
invariant to rotations of the search space when the eigen frame is used.

**Difference from closest prior work.** Guo & Yang's eigenvector-based
crossover and its rank-one variant work inside one DE population. FACE
estimates the frame from both heterogeneous populations and uses it for
*inter-population* transfer, with frame choice as a learned decision.
**Overlap risk is medium to high; the novelty may be judged incremental.**

**Cost.** O(D³/U + qD²) arithmetic, no extra FEs.

**Testable hypotheses.** H1: on rotated functions (F1–F5), eigen-frame
exchange beats raw-frame exchange at equal budget, and the gap shrinks on
F6–F12, which have less dense rotations (a structural prediction made before
seeing test results). H2: a random orthogonal frame (placebo) does not help.

**Ablations.** Raw frame only; eigen frame only; random orthogonal frame;
frame estimated from one population; frame from the whole population instead
of the elite; bandit frame choice vs 50/50.

**Risks.** Eigen-frame methods overlap heavily with CMA-ES / EDA ideas; the
frame degenerates as the population converges; clipping in x-space breaks
the frame at the bounds (FIR bounds are active).

**KBS suitability.** Moderate: a clear, mathematically clean mechanism, but
the novelty argument is thin unless the inter-population aspect produces a
distinct, well-explained effect.

### F3. Progress-aligned evaluation allocation between native search and exchange (PEA)

**Research gap.** MPHBS spends a fixed, domain-tuned share of evaluations on
exchange (CEC2022 ≈ 21 %, FIR ≈ 44 % of FEs per iteration). The pilot shows
two things: (i) the share matters (always exchanging was the best LA-KAIE
variant on CEC), and (ii) per-evaluation success rates are misaligned with
progress of the global best (A1/A2 vs A4, section D). AMALGAM allocates
offspring among *algorithms*, and migration-interval schemes adapt *when* to
migrate. Not found: allocating the evaluation budget between *native
operators* and *cross-population exchange* in a hybrid, with a credit defined
by contribution to the best-so-far.

**Formulation.** Per iteration t, choose the exchange share ρ_t ∈ [0, ρ_max]
of a fixed per-iteration budget B_t. Channel c ∈ {native, exchange} has
per-FE utility

    u_c(t) = E[ Δ_q(t) | FE spent on c ],  Δ_q = decrease of the q-quantile (e.g. top 10 %) of the pooled fitness distribution,

estimated with extreme-value or quantile credit over a sliding window, plus
Page–Hinkley change detection. Allocation: ρ_{t+1} = ρ_t + η·(û_exch − û_native)/σ̂
(a mirror-ascent step), or Thompson sampling on the two channels. No
landscape-feature controller is needed; the only state is the two channel
utilities.

**Algorithmic mechanism.** Any exchange operator can be plugged in (MPHBS's
mediator, a simple tournament crossover, F1 or F2). The contribution is the
allocation and credit layer, not the operator.

**Difference from MPHBS.** MPHBS fixes E = N_sub·N_i per iteration and tunes
it per domain (B-C1 vs A-C1). PEA makes the share adaptive and domain-agnostic,
potentially removing the need for domain-specific configurations. That
directly targets the paper's own threat to validity: "the final settings are
domain-specific".

**Difference from closest prior work.** AMALGAM uses reproductive success to
allocate among algorithms (a credit known to reward local repairs); adaptive
migration intervals use best-fitness improvement to set timing. PEA allocates
FEs between two *channels of the same hybrid* using quantile-progress credit.
Overlap risk: medium, since the resource-allocation idea is well known.

**Cost.** Negligible arithmetic, no extra FEs.

**Testable hypotheses.** H1: one PEA configuration matches or beats the
*best domain-specific* fixed share on both CEC2022 and FIR (tested against
fixed shares {0, 10, 20, 40, 60 %}). H2: quantile-progress credit beats
receiver-improvement credit, i.e. it avoids the A1/A2 misalignment observed.

**Ablations.** Fixed shares (grid); receiver-improvement credit vs
quantile credit vs extreme-value credit; no change detection; PEA on top of
the MPHBS mediator vs on top of simple exchange.

**Risks.** Incremental relative to adaptive resource allocation; the effect
may reduce to "exchange more", which the fixed-share grid would expose. That
is a scientifically valid outcome, but a weaker paper.

**KBS suitability.** Moderate. It is practically relevant (removes
domain-specific tuning), but the methodological novelty is modest.

### F4 (non-methodological alternative, for completeness): repositioning

Present the faithful MPHBS reproduction plus the diagnostic findings (dense
coupling vs sparse-group premises, credit misalignment, estimator collapse on
rugged landscapes) as an empirical analysis of *when learned information
exchange helps*. This is low-risk and honest, but **not** a methodological
contribution, and fits KBS poorly (a venue such as a reproducibility track or
an empirical-methods paper would fit better).

### Comparison (to support your decision, not to make it)

| | F1 OB-CEL | F2 FACE | F3 PEA |
|---|---|---|---|
| addresses a gap MPHBS itself states | yes (coupling) | yes (coupling / rotation) | yes (domain-specific settings) |
| closest prior | RVIntX (unverified), LIEM, GOMEA | eigenvector crossover (Guo & Yang) | AMALGAM, migration-interval schemes |
| novelty risk | medium (depends on RVIntX full text) | medium–high | medium |
| extra FEs | 0 | 0 | 0 |
| main technical risk | signal-to-noise | frame degeneration | reduces to "exchange more" |
| can be falsified by a structural test before the performance test | yes (γ vs gray-box FIR coupling) | yes (rotated vs less-rotated functions) | yes (vs fixed-share grid) |

The directions are not mutually exclusive: F3 is an allocation layer that F1
or F2 could sit under. Combining them would make attributing credit harder
and should be decided explicitly.

---

## 8. Diagnostic scripts used in this report (no algorithm re-runs)

* Benchmark structure (rotation density; exact FIR stopband coupling):
  `logs/decision_diagnostics/structure.py`
* Estimator on ideal samples around the true optimum:
  `logs/decision_diagnostics/estimator_ideal_samples.py`
* Per-action success and reward, and S8 trajectories from pilot traces:
  `logs/decision_diagnostics/pilot_actions.py`

## 9. What happens next

Nothing, until you instruct. No code change, no experiment, no manuscript.
