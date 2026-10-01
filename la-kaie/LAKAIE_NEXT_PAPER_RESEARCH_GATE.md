# LA-KAIE NEXT PAPER — RESEARCH GATE (knowledge-guided exchange compatibility)

Date: 2026-10-01.

This is a read-only audit plus a literature gate. No code, specification, seed, or experiment
was created or changed (see §15).

## Evidence tags

| tag | meaning |
|---|---|
| [REPO] | inspected in this repository |
| [S] | search summary or snippet |
| [K] | established knowledge |
| [INF] | my inference |
| [U] | unresolved |

- Full texts are still blocked by the egress policy (arxiv.org, publishers, ResearchGate).
- No prior work below was read in full.
- The kill decision rests on **consistent abstract-level statements across at least six
  independent papers**, each describing the core mechanism directly (§5–§7).

---

## 1. Repository audit

| Component | Existing? | File/module | Reusable? | Why |
|---|---|---|---|---|
| HBA, MPA | yes | `lakaie/algorithms/hba.py`, `mpa.py` | yes | validated ports [REPO] |
| MPHB, MPHBS | yes | `mphb.py`, `mphbs.py` | yes (reference) | 0/80 distributional differences vs. the authors' runs; FE totals exact [REPO] |
| Shared HBA+MPA backbone | yes | `hybrid_common.HybridState` (phase1, `fads_greedy`, memory saving) | yes | sha256-frozen; bit-identical reuse proven in D3 [REPO] |
| LA-KAIE controller | yes | `lakaie/algorithms/lakaie.py`, `lakaie/components.py` | **no** (as a scientific object) | an ε-greedy linear contextual bandit over 8 actions (A1 stop, A2 best→worst, A3 dimension-wise, A4 group-wise, A5 diversity donor, A6 exploit donor, A7 success-weighted donor, A8 elite-covariance refinement) [REPO] |
| Donor/receiver selection | yes | `do_action` | partly | receiver = random/best/tournament per action; donor = tournament/argmax distance/top-k/best [REPO] |
| Candidate generation | yes | `do_action` | partly | coordinate/group copy from donor into receiver (`x[T_idx] = Xd[di][T_idx]`) [REPO] |
| Exchange acceptance | yes | `do_action` | yes | greedy: replace the receiver if f(x) < f(receiver) [REPO] |
| Exchange frequency | yes | `exchange_phase` (E = 24 per iteration; A1 ends the phase) | yes | [REPO] |
| Reward | yes | `reward()` | **no** | immediate, multi-term: fitness-rank gain, diversity change, success, interaction consistency, FE cost, stagnation [REPO] |
| Exchange logging (LA-KAIE pilot) | yes | `lk_ep_*` in `results/raw/pilot/*/LA-KAIE-*/…npz` | **limited** | logs FE, action, reward, immediate success, random flag, n_transfer, group-level flag, interaction consistency, iteration. **It does not log donor or receiver index, candidate or receiver fitness, positions, or any delayed outcome** [REPO] |
| Exchange logging (N1 smoke, arm A7) | yes | `x_*` in `results/raw/n1_smoke/A7/*.npz` | **limited** | logs receiver/donor indices, receiver population, f_r, f_new, immediate y, native-control p0, state, t, FE. **Only the immediate label; N1-specific mapping; frozen study** [REPO] |
| Iteration state | yes | `LandscapeState.vector` (S1–S10: diversity, improvement, stagnation, progress, gap, HBA/MPA success EMA, interaction strength, exchange-success EMA, step/radius) | yes (as features) | [REPO] |
| D3 diagnostics | yes | `src/n1/diagnostics.py`, `results/raw/n1_d3_diagnostic/` | **pattern only** | logs agent identity, raw/standardised Δx, populations, targets (xprey, elite), covariance and eigenstructure. These are **A0 runs with no exchanges at all** [REPO] |
| Native-success control model | yes | `src/n1/native_control.py` | concept only | logistic model of native success (an internal precedent for a counterfactual baseline) [REPO] |
| Benchmarks | yes | `lakaie/benchmarks/cec2022.py` (official C), `fir.py`, `src/n1/synthetic.py` (S1–S8, known Hessian) | yes | validated [REPO] |
| Statistics | yes | `lakaie/analysis/stats.py`, paired common-random-number seed framework (`configs/seeds.json`) | yes | [REPO] |

---

## 2. What the existing implementation already gives us

1. Validated HBA/MPA/MPHBS with exact FE accounting, and a frozen shared backbone.
2. A complete exchange loop: donor and receiver selection, coordinate- or group-wise
   candidates, greedy acceptance, per-exchange logging hooks.
3. A 10-dimensional receiver-and-pooled search-state vector (S1–S10).
4. A paired-seed, Holm-corrected statistics pipeline, plus synthetic landscapes with known
   structure.
5. Passive-logging and trajectory-equivalence testing patterns (D3).

**What LA-KAIE attempted, and why it failed** [REPO: PILOT_REPORT.md,
RESEARCH_DECISION_REPORT.md]:
- LA-KAIE estimated variable interactions from an archive, grouped coupled variables, and
  let a landscape-feature contextual bandit choose exchange actions and granularity.
- In the 5-run pilot:
  - the interaction estimator found almost no couplings (19.9 of 20 and 30.4 of 31 variables
    stayed as singletons);
  - group-level transfers were 15% (CEC) and 2% (FIR);
  - the controller mostly chose A8 local refinement.
- On CEC2022 the **learned controller ranked worse than a fixed "always exchange" policy**.
- Group-wise exchange and landscape-aware bandit selection both had high prior-art overlap
  (RV-GOMEA, RVIntX, VIGPSO; LA-BHH, RL-HPSDE, DE-DDQN).

**What D3 found** [REPO]:
- Successful HBA/MPA/FAD displacements follow population and operator geometry (HBA:
  Δx ≈ b·(xprey − x); FAD is DE-like), not independent landscape structure.
- The N1 landscape-evidence premise was rejected.

**Scientifically obsolete:** interaction-group estimation as an exchange driver; the
landscape-structure evidence premise (N1/E4); LA-KAIE's multi-term immediate reward.

---

## 3. What must NOT be reused

- **The LA-KAIE reward, as a prediction label.** It is immediate, multi-term, and
  ad hoc-weighted.
- **Immediate exchange success (`lk_ep_success`, N1 `x_y`) as a training label.** It is
  greedy-acceptance success, so it is confounded with the receiver's baseline improvement
  rate and is selection-biased (§10).
- **Pilot, N1-smoke and D3 data as training data:**
  - pilot exchanges lack donor/receiver identity, fitness and position;
  - D3 runs contain no exchanges;
  - N1 exchanges use the N1-specific mapping, carry only immediate labels, and belong to
    frozen studies.
- **N1 seeds and the pilot seeds** (test-set contamination).

---

## 4. Proposed scientific question (as evaluated)

> "Can donor–receiver compatibility be learned from previous exchange outcomes and
> receiver/donor search-state information, and can that knowledge predict whether a future
> exchange will produce meaningful subsequent progress?"

| | causal chain |
|---|---|
| Old LA-KAIE | infer landscape/group structure → decide exchange |
| Proposed | observe exchange context and outcome → learn compatibility → predict utility → decide exchange |

The distinction from LA-KAIE is real. The question is whether it is new relative to the
literature.

---

## 5. Literature novelty findings (paper by paper; all [S])

| # | Work | What it does (from the abstract or summary) | Overlap with the proposal |
|---|---|---|---|
| L1 | **EMT-ADT**: "Multifactorial evolutionary algorithm with adaptive transfer strategy based on decision tree", *Complex & Intelligent Systems* (2023), doi 10.1007/s40747-023-01105-4 | Defines the **"transfer ability" of individuals** as a quantity of useful knowledge, builds a **Gini decision tree to predict positive-transferred individuals**, and transfers only those. Calls this "the first attempt … to use the decision tree to enhance positive knowledge transfer". | **HIGH**: an interpretable model, learned from transfer outcomes, predicting before transfer |
| L2 | Gao et al., "Transferring knowledge by budget online learning for multiobjective multitasking optimization", *SWEVO* (2024), S2210650224003031 | "**historical transferred solutions** [are used] as samples to train a classifier that identifies valuable knowledge". The classifier is **updated online (budget online learning) to handle concept drift**. | **HIGH**: learns from historical exchange outcomes and deploys online |
| L3 | EMT with an incrementally trained **naive Bayes** classifier (cited in the [S] summary of the "valuable knowledge" literature; exact paper [U]) | Each candidate gets a "+"/"−" label, and only "+" solutions can be transferred; NB locates valuable areas from positively transferred solutions. | **HIGH** |
| L4 | EMT with a **logistic regression** classifier for valuable transferred solutions (exact paper [U]) | Logistic regression identifies valuable solutions to transfer. | **HIGH**: the model class the brief prefers (§8) |
| L5 | Knowledge-classification-assisted EMT, *IEEE/CAA JAS* (2024), doi 10.1109/JAS.2024.125070 | Trains a classifier on target sub-population levels to pick assistant individuals similar to better-performing receiver individuals. | **MEDIUM–HIGH**: receiver-relative compatibility |
| L6 | Transfer rank plus **KNN classifier** selection of transferred solutions (exact paper [U]) | — | **MEDIUM** |
| L7 | **L2T**: Wu et al., "Learning to Transfer for Evolutionary Multitasking", arXiv 2406.14359 (PubMed 40299733, published 2025) | An RL agent (actor-critic, PPO) decides **when and how** to transfer, from a **state representation of evolution states** with a **reward on convergence and transfer-efficiency gain**. The policy is learned **across problems and deployed**. | **HIGH**: state-feature-based learned transfer controller, trained offline and deployed online |
| L8 | Zhan, Ma, Gong, Tan, "Learning Where, What and How to Transfer: A Multi-Role RL Approach for EMT", arXiv 2511.15199 (2025) | A task-routing agent with **attention-based source–target similarity** picks transfer pairs; a knowledge-control agent sets the elite proportion; strategy agents set transfer strength. | **HIGH**: learned donor–receiver pairing from state |
| L9 | MFEA-II, Bali et al., TEVC (2020) | Learns a **pairwise K×K transfer-probability (rmp) matrix online from data**, capturing non-uniform task-pair synergies. | **MEDIUM–HIGH**: a learned pairwise compatibility |
| L10 | MTEA-AD (TEVC 2022); online transfer with probabilistic outlier detection (*Complex Intell. Syst.* 2025) | Anomaly or outlier models flag likely negative-transfer individuals before transfer. | **MEDIUM**: predicts harm before transfer (unsupervised) |
| L11 | EMT source–target KL/WD/MMD similarity (e.g. TCYB 2023) | Population-distribution similarity gates or weights transfer. | **MEDIUM**: a baseline feature family |
| L12 | Heterogeneous island models with performance-index migration; AMALGAM; PAP | Adaptive migration and allocation among heterogeneous algorithms by observed success. | **MEDIUM**: the single-task heterogeneous setting, with success-adaptive rather than predictive control |
| L13 | MTES-KG (TEVC 2024) [code inspected earlier] | Adapts the number of external samples from their **historical rank success**. | **MEDIUM** |

**Island models.** No paper was found ([S]) that trains a predictor of individual
*migration* benefit in a **single-task heterogeneous island or hybrid** setting. That is a
*setting* gap, not a mechanism gap.

---

## 6. Closest prior-art threats

1. **EMT-ADT (L1).** Interpretable (decision tree), learns "transfer ability" from transfer
   outcomes, and predicts positive transfer of individual candidates *before* transfer. This
   is the proposal's mechanism, including the "knowledge model" framing.
2. **Budget-online-learning classifier (L2).** Learns from historical transferred solutions
   with concept-drift handling. This is the proposal's "learn compatibility from history,
   deploy online".
3. **L2T and the multi-role RL work (L7, L8).** Evolution-state features, learned when, what
   and where to transfer (including source–target pairing), pretrained and then deployed.
   This is the proposal's "receiver + donor state → decision".
4. **MFEA-II (L9).** A learned pairwise donor–receiver compatibility matrix.

---

## 7. Novelty assessment (brief Q1–Q9)

| Q | Answer | Basis |
|---|---|---|
| Q1: compatibility learned from historical transfer outcomes? | **YES**. Individual-level (L1–L4); pairwise (L9) | [S] |
| Q2: benefit/harm predicted before execution? | **YES** (L1, L3, L4, L10) | [S] |
| Q3: receiver + donor state + candidate relation as features? | **PARTIAL to YES**. L5 uses receiver-relative similarity; L7/L8 use evolution-state features of both sides, plus attention-based source–target similarity. The exact feature set is [U] | [S] |
| Q4: trained offline, deployed online? | **YES** (L7 pretrained policy; L8 multi-role RL; a "generalizable meta-policy … pretraining … over an augmented multitask problem distribution") | [S] |
| Q5: out-of-sample *predictive* evaluation (rather than final performance)? | **UNRESOLVED**. The abstracts report optimisation performance; whether any reports classifier AUC or calibration out of sample is [U] | [S][U] |
| Q6: donor-only vs receiver-only vs both vs + relation ablation? | **Not found** [U] | [S] |
| Q7: incremental value beyond fitness, similarity, distance and diversity? | **Not found** [U] | [S] |
| Q8: interpretable knowledge model for the exchange decision? | **YES** (decision tree L1; naive Bayes L3; logistic regression L4) | [S] |
| Q9: essentially the entire proposed study? | **The mechanism, yes.** Learned, interpretable, historically trained, online-deployed transfer-utility prediction exists (L1–L4, L7–L9). Only the *evaluation* elements (Q5–Q7) and the *setting* (single-task heterogeneous hybrid rather than multitask) remain unverified | [S][INF] |

**Assessment.**
- The proposed method, "learn donor–receiver compatibility from exchange history and use it
  to gate exchange", is **established** in evolutionary multitasking.
- Moving it from multitasking to a single-task HBA–MPA hybrid is a change of setting. The
  project's own standard (RESEARCH_DIRECTION_DISCOVERY_GATE §9) classes that as
  insufficient.
- Adding receiver features, or using logistic regression instead of a decision tree, would
  be a superficial modification.

---

## 8. Proposed knowledge representation (recorded; not pursued)

Feature availability for the brief's §7 list:

| features | status |
|---|---|
| R1–R4, R8 | available (S1–S4 state; FE) [REPO] |
| R5–R7, R10, R11 | computable from logged populations, but not in the LA-KAIE pilot logs [REPO] |
| R9 | trivial |
| D1–D8 | require new logging (donor index and fitness are not in the pilot logs; present in N1 A7 only) |
| C1–C8 | require positions of donor and receiver at exchange time, which are not logged in the pilot |
| H1, H5 | available (S9 EMA, exchange counts) |
| H2–H4, H6 | require new logging |

**Model.** Had it passed, the choice would be L2-regularised logistic regression (calibrated,
interpretable, online-cheap), with a shallow tree as a sensitivity check. This is the
**same** model family as L1 and L4 [INF].

---

## 9. Prediction target (recorded for completeness)

- **Recommended if ever pursued:** (C) paired improvement against a matched no-exchange
  control.
  - Y = 1 if the receiver's log best-so-far error after a fixed FE horizon H is lower in the
    exchange branch than in a common-random-number shadow branch that skips the exchange.
- **Why not the others:**
  - (A) absolute and (B) normalised improvement are confounded with phase and baseline
    progress.
  - (D) the log progress ratio is acceptable only within a paired design.
- This target is what distinguishes the idea from the immediate-success labels that L1–L4
  appear to use [INF; label definitions in L1–L4 are U].
- It is an **evaluation-validity** point, not a new mechanism.

---

## 10. Leakage / causal-selection problem

- **Split.** Row-wise splits leak (rows from the same run share trajectory state). The
  primary split would be **leave-problem-out**, nested within leave-run-out.
- **Selection.** Labels exist only for executed exchanges. The smallest defensible fix is a
  **randomised-exploration phase**: exchange decisions are made with a fixed known
  probability, plus **shadow no-exchange branches** under common random numbers for paired
  labels.
  - This doubles the FE cost of data generation; the controller itself is not affected.
- **Recorded, not pursued:** existing EMT classifiers learn only from executed transfers
  (L2: "historical transferred solutions"), so they are exposed to this bias [INF]. This is
  the one scientific observation that survives the gate. It belongs to an
  *evaluation-methodology* question ("are EMT transferability classifiers trained on
  selection-biased, immediate labels?"), not to this proposal. Per the brief's §18 C rule it
  is **not** offered as a rescue.

---

## 11. Minimal experimental design

Not produced. The decision is C.

---

## 12. Baselines

Not produced. The decision is C. For reference, B0–B5 plus EMT-ADT- and L2-style classifiers
would be mandatory, which itself shows the overlap.

---

## 13. KBS suitability

Knowledge representation, learning from experience, and interpretable decision control all
fit KBS in principle. But the specific contribution would be read as "EMT-ADT / budget-online
classifier transplanted into a single-task HBA–MPA hybrid" [INF].
- **Methodological novelty: low.**
- **Generalisation claim: weak.** The predictive gap is unverified, and the setting change is
  not a contribution.

---

## 14. Exact proposed contribution (three statements, tested against the literature)

| # | Statement | Supported by a literature gap? |
|---|---|---|
| 1 | "Exchange utility can be predicted from donor–receiver compatibility features under out-of-sample evaluation." | **No (as a claim of predictability).** L1–L4 already predict transfer utility from features. The *out-of-sample evaluation* part is [U] but is an evaluation detail |
| 2 | "Receiver-state information provides incremental predictive value beyond donor fitness and population similarity." | **Possibly [U].** The feature-group ablation was not found in [S]. On its own it is an analysis of an existing mechanism, not a new principle |
| 3 | "A knowledge-guided controller converts the learned compatibility model into selective information exchange." | **No.** L1, L2, L4, L7 and L8 already do this |

---

## 15. Decision

**C — KILL**

**Reason.** Learning a compatibility or transferability model from historical transfer
outcomes, predicting individual transfer benefit *before* executing it, using interpretable
models (decision tree, naive Bayes, logistic regression), updating it online, and pretraining
state-based transfer policies that are then deployed — all of these are established in
evolutionary multitasking (L1–L4, L7–L9; 2020–2025).
- What remains is (i) a change of setting (single-task heterogeneous hybrid) and (ii)
  evaluation details (out-of-sample AUC, feature-group ablation).
- Neither is a scientifically different hypothesis, and adding them would be the superficial
  rescue the brief forbids.

**Confidence and limits.**
- The kill rests on [S] abstract summaries of at least six independent papers that state the
  mechanism directly. It does not depend on reading between the lines.
- If the full texts of L1–L4 revealed that they never predict before transfer (e.g. they only
  re-weight after the fact), the decision should be revisited. The abstract wording ("predict
  the positive-transferred individuals", "only the solutions whose predictive class labels are
  '+' can be … transferable") makes that unlikely [S].

---

## 16. Implementation specification

Not applicable (the decision is not A).

## Freeze confirmation

| item | changed? |
|---|---|
| source code | NO |
| algorithm files | NO |
| specifications | NO |
| seeds | NO |
| CEC2022 / FIR runs | NO |
| experiments | NO |
| N1 / D3 / MPHBS reproduction | NO |

- This report is the only new file.
- Data inspection was read-only (listing NPZ fields).

## Sources (search results; none read in full)

- https://link.springer.com/article/10.1007/s40747-023-01105-4
- https://www.sciencedirect.com/science/article/abs/pii/S2210650224003031
- https://www.ieee-jas.net/en/article/doi/10.1109/JAS.2024.125070
- https://arxiv.org/abs/2406.14359
- https://pubmed.ncbi.nlm.nih.gov/40299733/
- https://arxiv.org/abs/2511.15199
- https://www.researchgate.net/publication/331729696_Multifactorial_Evolutionary_Algorithm_With_Online_Transfer_Parameter_Estimation_MFEA-II
- https://ieeexplore.ieee.org/document/9385398/
- https://link.springer.com/article/10.1007/s40747-025-02220-0
- https://doi.org/10.1109/tcyb.2023.3234969
- https://www.sciencedirect.com/science/article/abs/pii/S0950705124001655
