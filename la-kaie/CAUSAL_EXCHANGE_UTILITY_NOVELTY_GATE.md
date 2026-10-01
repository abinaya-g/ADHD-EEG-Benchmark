# CAUSAL EXCHANGE UTILITY — KBS NOVELTY GATE

Date: 2026-10-01.

**Scope**
- Read-only audit plus literature search.
- Nothing was implemented, branched, seeded or run.
- This report is the only new file.

**Evidence tags**

| tag | meaning |
|---|---|
| [REPO] | inspected in this repository |
| [S] | search-engine summary or snippet |
| [K] | established methodological knowledge |
| [INF] | my inference |
| [U] | unresolved |

**Access limit**
- Full texts are still blocked by the environment's egress policy: arXiv, ScienceDirect, IEEE, ACM, IGI, ResearchGate.
- **No closest paper was read in full.**
- Every prior-work statement below is [S] unless tagged otherwise.

---

## 1. Executive decision

**C — NOVELTY INSUFFICIENT.** Abandon this as a KBS paper direction.

**What the proposal reduces to:** "predict the counterfactual effect of an information exchange from the pre-exchange state, and exchange only when the predicted effect is positive". Each component is already established:

| component | prior art |
|---|---|
| (a) predicting transfer benefit/harm before transfer, and gating on it | EMT-ADT; MFEA-ML; budget online learning; naive-Bayes and logistic-regression transfer classifiers; L2T (killed in `LAKAIE_NEXT_PAPER_RESEARCH_GATE.md`) |
| (b) causal-ML estimators (causal random forests, doubly robust learning) used *inside* an evolutionary algorithm to drive adaptive strategy switching | the DRCL-driven constrained EA (IJSIR 17(1), 2026) [S] |
| (c) common-random-number paired comparison | a textbook variance-reduction method in stochastic simulation [K] |
| (d) CATE/uplift learners (S-, T-, X-learner, causal forest) | textbook causal ML [K] |

**What remains:**
1. Changing the estimand of an existing gating mechanism from an observational label (immediate success or survival) to a counterfactual label.
2. Applying it to a single-task heterogeneous hybrid.

By the brief's own §12 definitions, that is "different predictor / different label / new application setting / implementation detail" = **C**.

**No D:** no paper was found that estimates the paired counterfactual effect of an *exchange* specifically. "Not found" is not "does not exist", and it does not lift the decision to A or B. The C judgement holds even if that gap is real (§12).

---

## 2. Existing repository assets relevant to the question

| question | answer | evidence |
|---|---|---|
| 1. Exchange mechanisms | LA-KAIE actions A2–A8 (best→worst copy; dimension- or group-wise copy from a donor into a receiver; diversity-, exploit- and success-weighted donors; elite-covariance refinement); the MPHBS SARSA mediator; N1 mapping (HI/HB/HR) with the SPRT-style rule | `lakaie/algorithms/lakaie.py` `do_action`; `mphbs.py`; `src/n1/mapping.py`, `algorithm.py` [REPO] |
| 2. Exchange outcomes logged | LA-KAIE pilot: action, reward, **immediate** success, random flag, n_transfer, group-level, ic, FE, iteration. N1 A7 smoke: f_r, f_new, **immediate** y, native-control p0 | `lk_ep_*`, `x_*` NPZ fields [REPO] |
| 3. Donor identity | **No** in the LA-KAIE pilot; **yes** in N1 (`x_idx_d`, `x_idx_r`, `x_pop_r`) | [REPO] |
| 4. Receiver state | per-iteration S1–S10 (LA-KAIE); `x_state` (N1); full populations only in D3 | [REPO] |
| 5. Candidate state/fitness | **No** (LA-KAIE pilot); fitness only (N1: `x_f_new`); no positions | [REPO] |
| 6. Paired no-exchange controls | **No**. N1 "shadow" means passive *selectors* in A0 runs, not shadow exchange branches. A0 vs A7 are separate whole runs | `src/n1/config.py` `shadows`; spec §6.2 [REPO] |
| 7. CRN / paired branching | **Run-level only.** Seeds are common across arms, and native, exchange and placebo RNG streams are isolated (`spawn_streams`). **No mid-run state cloning or branching exists** (no copy or snapshot facility) | `src/n1/algorithm.py` `spawn_streams` [REPO] |
| 8. Scientifically reusable | backbone (`HybridState`, sha256-frozen); benchmarks (CEC2022, FIR, synthetic S1–S8); paired statistics (`lakaie/analysis/stats.py`); RNG-stream isolation pattern; D3 passive-logging and trajectory-equivalence test pattern | [REPO] |
| 9. Must NOT be reused | LA-KAIE pilot exchange labels (immediate, controller-selected, ε-greedy confounded); N1 A7 `x_y` (immediate; SPRT-selected); D3 data (no exchanges); LA-KAIE reward; all N1/pilot seeds | [REPO][INF] |

**None of the existing labels is a causal effect.** They are outcomes of exchanges that were *selected* by a controller (bandit or SPRT), measured immediately, with no counterfactual branch.

---

## 3. Exact proposed scientific contribution (formalised)

- **State and treatment.** At pre-exchange receiver state X:
  - T = 1: execute a specified exchange.
  - T = 0: skip it.
- **Outcome.** Y(t) is the receiver's progress over a fixed FE horizon H, e.g. log(best-error before / best-error after H).
- **Effect.** τ(X) = Y(1) − Y(0), estimated by paired branches from a cloned state under common random numbers.
- **Hypotheses:**
  - H1: τ is heterogeneous across X.
  - H2: τ ≠ immediate success, fitness, improvement or rank.
  - H3: receiver state moderates τ.
  - H4: τ(X) is predictable out of sample.
  - H5: a τ-predictor adds value beyond transferability, similarity and candidate-quality predictors.
- **Control rule.** Exchange iff the predicted τ̂(X) > 0.

---

## 4. Literature search methodology

- **Channel.** Web search returning summaries; about 15 new queries, plus about 40 from the immediately preceding gate (EMT transferability).
- **Coverage:**
  - EC × causal inference, treatment effects and CATE;
  - counterfactual or causal migration, transfer and negative transfer;
  - counterfactual credit assignment in AOS;
  - off-policy evaluation and dynamic algorithm configuration;
  - uplift-guided optimisation;
  - CRN and coupled runs in EC;
  - rollout or lookahead operator selection;
  - causal analysis of algorithm configuration;
  - EMT transfer-gain quantification;
  - transfer theory.
- **Full texts.** Attempts in earlier gates were blocked (§14).
- **Limit.** No methodology was verified in full text, so absence findings are provisional.

---

## 5. Closest prior work

| # | Paper | Year / venue / ID | What they do (from summaries) | Versus our proposal |
|---|---|---|---|---|
| P1 | **DRCL-driven adaptive evolutionary constrained optimisation** | 2026, *Int. J. Swarm Intelligence Research* 17(1); ScienceDirect S1947926326000070; IGI title 405447 | "Quantifies the causal interaction between objective optimisation and constraint satisfaction through **causal random forests**" and uses **doubly robust causal learning** for "dynamic adaptive strategy-switching" | **Causal-ML-estimated effects already drive an EA's adaptive decision.** Their treatment is a constraint-handling priority strategy, not an exchange. Whether they use counterfactual branches or observational data: [U] |
| P2 | **MFEA-ML** | 2025, *Information Sciences*, S0020025525000404 | "Collects training data by tracing the **survival status** of individuals generated by intertask transfer" and builds an ML model "to guide the transfer … from the perspective of **individual pairs**" | The same online decision (predict whether a transfer helps, then gate it). Its label is observational and near-immediate (survival); no counterfactual [S] |
| P3 | **EMT-ADT** | 2023, *Complex & Intelligent Systems*, doi 10.1007/s40747-023-01105-4 | Decision tree predicts "positive-transferred individuals" from a "transfer ability" indicator | Same gating mechanism; label is observational |
| P4 | **Budget online learning for MO-MTO** (Gao et al.) | 2024, *SWEVO*, S2210650224003031 | Classifier trained on **historical transferred solutions**, updated online for concept drift | Same; observational labels (selection-biased by construction [INF]) |
| P5 | **L2T** (Wu et al.) | arXiv 2406.14359; published 2025 (PubMed 40299733) | PPO agent decides when and how to transfer; reward on "convergence and **transfer efficiency gain**" | Policy learning over transfer decisions with a *delayed* reward. Whether the "gain" is measured against a no-transfer counterfactual is [U] (full text blocked) |
| P6 | **Multi-role RL for EMT** (Zhan, Ma, Gong, Tan) | arXiv 2511.15199 (2025) | Learns where, what and how to transfer from state, with attention-based source–target similarity | State-conditioned transfer policy; no causal estimand reported [S] |
| P7 | **Scott & De Jong, "First Complexity Results for Evolutionary Knowledge Transfer"** | FOGA 2023, pp. 140–151, doi 10.1145/3594805.3607137 | No-free-lunch theorems for transfer; the first asymptotic runtime result for transfer optimisation | Theoretical with/without-transfer comparison; the analytical counterpart of an average treatment effect. Not state-conditional [S] |
| P8 | **Gupta & Ong, "Genetic Transfer or Population Diversification?"** | arXiv 1607.05390 (2016) | Separates transfer from diversification effects in EMT | Effect attribution with controls; aggregate, not state-conditional |
| P9 | **AOS credit-assignment literature** (Fialho thesis; "Credit Assignment in Adaptive EAs", arXiv 0907.0592; historical/ancestral credit assignment) | 2009–2021 | Immediate vs delayed (ancestral) credit for operators | H2-type insight ("immediate success ≠ long-term value") is long established for operators; none counterfactual [S] |
| P10 | **Causal ML methods** (causal forests; S/T/X-learners; doubly robust estimation; uplift modelling; counterfactual off-policy evaluation with SCMs) | standard [K]; e.g. Gumbel-max SCM OPE, arXiv 1905.05824 | Generic CATE and policy-evaluation tools | Supply every estimator the proposal would use |

---

## 6. Detailed overlap analysis (critical prior-art test; brief §5)

| Paper | Setting | Treatment | Control | Outcome horizon | Counterfactual design | ITE/CATE | Pre-treatment prediction | Negative-transfer detection | Online decision use | Single-task heterogeneous? | Key overlap | Remaining difference |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P1 DRCL-EA | single-task constrained EA | strategy (constraint vs objective priority) | [U] | [U] | [U] | **YES** (causal random forest / doubly robust) [S] | YES (strategy chosen from estimated effects) | n/a | **YES** | single-task, homogeneous | causal ML drives EA control | treatment is not exchange; design [U] |
| P2 MFEA-ML | EMT | inter-task transfer | none | survival (short) | NO | NO | **YES** | YES | **YES** | NO | predicts transfer benefit per individual pair; gates transfer | observational label |
| P3 EMT-ADT | EMT | transfer | none | short | NO | NO | **YES** | YES | **YES** | NO | same | observational label |
| P4 Budget OL | MO-EMT | transfer | none | short | NO | NO | **YES** | YES | **YES** | NO | same, plus drift handling | observational label |
| P5 L2T | EMT | when/how to transfer | [U] | delayed (RL return) | [U] | NO | YES (policy) | implicit | **YES** | NO | learned, state-conditioned, delayed-reward transfer control | causal estimand [U] |
| P6 Multi-role RL | EMT | where/what/how | [U] | delayed | [U] | NO | YES | implicit | YES | NO | same | — |
| P7 Scott & De Jong | theory | transfer | no transfer | runtime | analytical | NO (average) | NO | NO | NO | — | with vs without transfer effect | not state-conditional, not empirical |
| P8 Gupta & Ong | EMT | transfer | diversity controls | run-level | controlled comparison | NO | NO | NO | NO | NO | effect attribution | aggregate |
| P9 AOS credit | single-task | operator | none | immediate / ancestral | NO | NO | YES (bandit) | n/a | YES | partly | delayed vs immediate credit | not counterfactual |

---

## 7. Causal/counterfactual prior art (brief §3A–E and §4)

- **Causal ML *for* EA control:** found (P1, 2026).
- **EA *for* causal ML:** found but irrelevant (neuroevolutionary CATE representations; InferenceEvolve, ICML 2026).
- **Causal or counterfactual transfer, migration or exchange:** not found [S].
- **"Causal knowledge transfer":** exists in multi-agent RL (arXiv 2507.13846), not EC [S].
- **Counterfactual credit assignment:** exists in RL (Mesnard et al., ICML 2021) [S]. Not found in AOS [S].
- **Off-policy evaluation of EA parameter-control policies:** not found [S]. Generic OPE and SCM counterfactual tools exist [K].
- **Uplift-guided metaheuristic control:** not found [S]. Uplift with downstream optimisation ("predict-then-optimise") is standard in marketing [K].

---

## 8. Selection-bias prior art

- **Exposure in EMT.** EMT transferability classifiers learn from historically executed transfers (P2–P4), so they are exposed to treatment-selection bias [INF].
- **Recognition or correction** (propensity weighting, randomised assignment, doubly robust correction): **not found** in EMT or migration [S]. Status [U] without full texts.
- **P1** uses doubly robust learning, which is precisely a confounding-robust estimator, inside an EA. So recognising and handling confounding in EA decision data is **not new in EC as a technique**. Only its application to transfer is unreported [S][INF].

---

## 9. Paired-CRN prior art

- **CRN** is a standard variance-reduction method for comparing stochastic systems [K][S].
- Same-seed paired comparison of EA variants is routine. This repository already does it at run level [REPO].
- **State-cloned branching** (fork the same state, apply the action vs not, roll out) is the standard rollout/forward-model mechanism of Monte-Carlo planning and rolling-horizon evolution [K][S].
- **Same-state migration-vs-no-migration branching** in island models was not found [S].
- **Classification: implementation detail / standard methodology**, not a contribution.

---

## 10. Treatment-effect prediction prior art

- **"Predict benefit before intervening, act if positive":** established for transfer (P2–P6). The estimand is observational or reward-based, not counterfactual.
- **"Estimate causal effects with causal forests or doubly robust learning, then switch strategy":** established in an EA (P1).
- **Treatment-effect policy learning, CATE-based decision rules, uplift targeting:** established in causal ML [K].
- **Counterfactual operator selection / heterogeneous operator treatment effects:** not found in EC [S].
- **Conclusion.** The τ(X)-predict-then-gate pipeline is the composition of an established EC gating mechanism and an established causal-ML estimator class. The composition has already been demonstrated in an EA, for a different decision (P1).

---

## 11. Single-task heterogeneous-island gap

- **The setting is new only as an application.** EMT → single-task HBA–MPA hybrid does not change the causal question.
- **The causal contribution is not novel independent of the setting.** The estimand switch and the CRN design apply equally to EMT, and P1 shows causal-ML-driven EA control exists independent of the setting.
- **HBA/MPA are not essential to the question.** Treating their exchange as the object would be the "new optimizer / new application" case of class C.

---

## 12. What is genuinely different, if anything

1. **Estimand.** A paired counterfactual effect of an *exchange*, τ(X), instead of observational survival or immediate success. Not found [S].
2. **Empirical claim it could support.** The observational transfer labels used by P2–P4 disagree systematically with counterfactual exchange effects, i.e. selection or confounding bias in transferability learning.

Assessment:
- (1) is a **label or estimand change** to an existing mechanism: brief §12 C ("different predictor … implementation detail").
- (2) is an **evaluation/critique** finding about other methods. It could be a short methodological note. As a KBS research article it would need to show that the bias *changes decisions and performance*. That is an empirical audit of P2–P4, not a new mechanism.
- Neither (1) nor (2) is a new research question at the level the previous gates required.

---

## 13. Novelty threats

| threat | effect |
|---|---|
| T1 (P1) | causal forests / doubly robust learning already drive an EA's adaptive decisions. Removes "first causal-ML-guided EA control" |
| T2 (P2–P6) | pre-transfer benefit prediction and gating are established. Removes "predict and gate" |
| T3 (P5 L2T) | "transfer efficiency gain" may already be computed against a no-transfer reference [U]. If so, this moves toward D |
| T4 (P7, P8) | with/without-transfer effect analysis exists, theoretical and aggregate |
| T5 | CRN and state-branching rollouts are textbook |
| T6 | H2 (immediate ≠ long-term value) is long known in AOS credit assignment (P9) |

---

## 14. Unresolved full-text papers

These would matter only for moving C → D; none could move C → A.

| paper | question to answer from the full text |
|---|---|
| P1 DRCL-EA (IJSIR 2026) | Is the causal effect estimated from observational run data or from controlled branches? Exactly what is the treatment? |
| P5 L2T | How is "transfer efficiency gain" defined: against a no-transfer counterfactual or rollout? |
| P2 MFEA-ML | Survival horizon and label definition; any control comparison? |
| P7 Scott & De Jong | Do any constructions estimate state-conditional transfer effects? |

**Access attempted (blocked by egress policy):**
- arxiv.org (P5, P8);
- sciencedirect.com (P1, P2, P4);
- igi-global.com (P1);
- dl.acm.org (P7);
- ieeexplore.ieee.org.

No alternative versions were reachable.

---

## 15. Kill conditions

| condition | status |
|---|---|
| D-trigger: a prior paper estimates state-conditional causal effects of transfer or exchange against matched no-exchange counterfactuals and uses them to gate exchange | **not found** [S] |
| C-trigger: the causal framing exists in EC, and the remaining difference is predictor, label, setting or implementation | **met.** Causal-ML-driven EA control (P1); transfer gating (P2–P6); CRN and branching (standard); setting change (single-task) |

---

## 16. Final decision

**C — NOVELTY INSUFFICIENT**

---

## 17. If A or B: minimum defensible research question

Not applicable.

---

## 18. Why it should be abandoned

- The proposal's central mechanism (predict whether an exchange will help, then gate it) was already killed as established (P2–P6).
- The causal-ML layer (CATE via causal forests or doubly robust learning steering an EA) has a 2026 precedent (P1).
- The paired-CRN branching design is standard methodology.
- What remains is a better label for an existing mechanism, in a new single-task setting.
- A KBS reviewer could reasonably summarise it as "EMT-ADT/MFEA-ML with a counterfactual label, in an HBA–MPA hybrid" [INF]. The previous gate killed the mechanism-level idea on exactly this standard.
- **One residual question** (§12, item 2) has not been ruled out by prior work: are observational transferability labels causally biased, and does that bias change decisions? It is an evaluation/critique study of existing EMT methods. Pursuing it would require obtaining P2–P5 in full to check their label definitions. It is recorded, not recommended: it does not follow from the HBA/MPA infrastructure, and per brief §13 it is not invented to rescue this direction.

---

## 19. Recommended next action

1. Do **not** implement causal exchange utility.
2. Stop iterating variants of "learn or predict exchange utility". Three successive gates (N1/R3, compatibility, causal utility) have converged on the same established mechanism.
3. The remaining open thread from the discovery gate is **R6** (behavioural complementarity → hybridisation synergy), still at its second literature gate. Whether to run it, pursue the §18 critique question, or close the LA-KAIE line is the user's decision.
4. Any future gate needs full-text access: allow arxiv.org (and ideally publisher hosts) in the environment network policy, or supply PDFs.

**Freeze confirmation:** code, branches, specifications, seeds, experiments, HBA/MPA parameters and predictors are all unchanged. The repository was inspected read-only (file reads, `grep`, NPZ field listing).

**Sources (search results; none read in full)**
- https://www.sciencedirect.com/org/science/article/pii/S1947926326000070
- https://www.igi-global.com/viewtitle.aspx?titleid=405447
- https://www.sciencedirect.com/science/article/abs/pii/S0020025525000404
- https://link.springer.com/article/10.1007/s40747-023-01105-4
- https://www.sciencedirect.com/science/article/abs/pii/S2210650224003031
- https://arxiv.org/abs/2406.14359
- https://pubmed.ncbi.nlm.nih.gov/40299733/
- https://arxiv.org/abs/2511.15199
- https://dl.acm.org/doi/10.1145/3594805.3607137
- https://arxiv.org/abs/1607.05390
- https://arxiv.org/abs/0907.0592
- https://groups.csail.mit.edu/EVO-DesignOpt/pb/uploads/Site/FialhoThesisDraft.pdf
- https://arxiv.org/html/1905.05824
- https://arxiv.org/pdf/2507.13846
- https://www.sciencedirect.com/science/article/pii/S187775032300114X
- https://arxiv.org/html/2604.04274v1
