# R6 BEHAVIOURAL COMPLEMENTARITY → HYBRIDISATION SYNERGY
## KBS NOVELTY GATE

**Date:** 2026-10-01

**Scope:** read-only audit plus literature search. Nothing was implemented, branched, seeded, tuned or run. This report is the only new file.

**Evidence tags**

| tag | meaning |
|---|---|
| [REPO] | inspected in this repository |
| [S] | search-engine summary or snippet |
| [K] | established knowledge |
| [INF] | my inference |
| [U] | unresolved |

**Access limit:** full texts are still blocked by the environment's egress policy (arXiv, Springer, ACM, ScienceDirect, MDPI not attempted). **No paper below was read in full; every row is FULL TEXT UNVERIFIED.**

---

## 1. Executive decision

**C — NOVELTY INSUFFICIENT.** Do not pursue R6 as a KBS paper.

**Why.** The general question — do relationships between optimizers, measured before combining them, predict the benefit of combining them? — is already addressed:
- **Sequential combination.** Trajectory features from one optimizer's run predict whether switching to a second optimizer is beneficial, out of sample:
  - "To Switch or Not to Switch", EvoApplications 2023;
  - per-run algorithm selection with warm-starting, PPSN 2022.
- **Portfolio combination.**
  - Pairwise algorithm similarity graphs are used to select complementary portfolios (PS-AAS; similarity-based portfolio construction, GECCO 2026).
  - Performance complementarity is standard (instance-space footprints; VBS/SBS; Shapley value of portfolios).
- **Interaction ("synergy") of combined components.**
  - Quantified with functional ANOVA in modular frameworks (2024; PSO-X 2026).
  - A hybrid "Algorithmic Synergy Index" has been proposed.
- **Behavioural characterisation of metaheuristics.**
  - Hayward & Engelbrecht, TEVC 2025 (20 behavioural characteristics; pairwise behavioural similarity).
  - DynamoRep / Opt2Vec trajectory representations.
  - Behaviour-space analysis (2025/26).
- **"Diversity predicts combination benefit".** Long studied and found weak for ensembles (Kuncheva & Whitaker 2003).

**What remains** is the *combination mode*: concurrent exchange-based hybrid vs sequential or portfolio. That is a setting change, together with a choice of descriptor (behavioural vs performance/landscape). The brief's §19 C covers exactly "new descriptor / predictor / metric / application setting".

**Not D:** no paper was found ([S]) that predicts *concurrent hybrid* synergy from *standalone behavioural* descriptors of a pair, validated out of sample. As before, "not found" ≠ "does not exist", and the C judgement does not depend on that gap.

---

## 2. Repository audit

| question | finding |
|---|---|
| 1. Behavioural signals already available | **Standalone HBA and MPA** (baseline-validation and pilot campaigns): per-iteration best, current mean/median, diversity, stagnation, success rate; a 200-point curve (`iter_*`, `curve` in `results/raw/{baseline_validation,pilot}/…/HBA|MPA/*.npz`). **No positions, displacements or step statistics** [REPO] |
| | **D3:** full displacements, targets, covariance and eigenstructure — but from **A0 runs of the coupled `HybridState` backbone** (HBA and MPA populations co-run with interleaved RNG draws; MPA includes FAD). **These are not independent standalone runs** [REPO: `hybrid_common.HybridState`, `phase1`] |
| 2. Requiring new logging | step/displacement distributions, directional persistence, coverage, contraction/expansion, and timing descriptors for **standalone** runs of every optimizer in the pool |
| 3. Computable from independent runs | everything in §4 (A–E), once logged in standalone runs |
| 4. Hybrid-dependent (invalid as pre-hybrid predictors) | exchange success, received-candidate survival, co-run population geometry, anything from MPHB/MPHBS/LA-KAIE/N1 runs, **and D3** (a co-run inside the hybrid backbone) |
| 5. D3 reuse for R6 | **No.** The protocol is coupled, not standalone; there is one optimizer pair; it is a frozen study |
| 6. Invalid datasets | D3; N1 smoke and calib; LA-KAIE pilot and ablations (hybrid outcomes); MPHB/MPHBS runs (they are the *outcome* side, usable only as one data point). Pilot and N1 seeds must not be reused |
| 7. Are HBA vs MPA different enough? | Qualitatively yes: target-directed attraction vs Lévy/Brownian plus a DE-like FAD (D3) [REPO]. **But R6 needs many pairs.** The repository implements only HBA and MPA (plus FAD inside MPA), i.e. **one pair**. An R6 study would require a new pool of roughly 8–10 optimizers and a generic exchange protocol — essentially new infrastructure [REPO][INF] |

---

## 3. Exact proposed scientific contribution

- **Descriptors.** z_A and z_B are behavioural descriptors from standalone runs of A and B on problem f.
- **Complementarity.** C(A,B) = g(z_A, z_B).
- **Synergy.** S(A,B,f) is the hybrid's performance relative to a component baseline, at equal FE (§5).
- **Claim.** C(A,B) predicts S(A,B,f) out of sample (leave-pair-out and leave-problem-out), beyond performance-only complementarity.

---

## 4. Definition of behavioural complementarity

| category | candidate descriptors | measurable pre-hybrid? | independent of hybrid outcome? | used in selection/portfolios? | used to predict combination benefit? | complementarity or only difference? |
|---|---|---|---|---|---|---|
| A. Trajectory | trajectory correlation, visited-region overlap, direction divergence | yes | yes | trajectory features in per-run AS / DynamoRep [S] | **yes, for sequential switching** ("To Switch or Not to Switch") [S] | difference, unless tied to a mechanism |
| B. Distributional | covariance difference, diversity profile, entropy, coverage, contraction | yes | yes | DynamoRep statistics; Hayward & Engelbrecht characteristics [S] | not found for concurrent hybrids [S] | difference |
| C. Temporal | improvement timing, exploration/exploitation phase, stagnation | yes | yes | behaviour-space analysis (exploration, exploitation, convergence, stagnation) [S] | sequential portfolios exploit timing [S] | closest to a complementarity rationale (one optimizer's productive phase covers the other's stagnation) [INF] |
| D. Operator | step-size and success distributions, response to geometry | yes | yes | Hayward & Engelbrecht ("improvement frequency, population turnover, step footprint, step fitness") [S] | not found [S] | difference |
| E. Landscape response | per-function-class performance and behaviour | yes | yes | ISA footprints; ELA-based AS [S] | **yes, portfolio complementarity** (footprints, VBS/SBS, Shapley) [S] | performance complementarity — the strongest existing notion |

**Difference vs complementarity:**
- Every distance in A–D measures difference.
- The only complementarity concept with an established *predictive* role is **landscape/performance complementarity** (E), and it belongs to portfolio theory.
- Kuncheva & Whitaker (2003) showed that pairwise diversity measures relate only weakly to ensemble accuracy [S]. This is a known negative prior for "difference ⇒ benefit".

---

## 5. Definition of hybrid synergy

**Existing notions:**
- **Best-of-components baseline.** "Hybrid better than its best constituent" is the de facto standard in hybrid papers and in PAP ("PAP outperforms its constituent algorithms") [S][K].
- **Algorithmic Synergy Index (ASI).** ASI = P_hybrid / Σ P_individual, with ASI > 1 = synergy (TSP hybrid review, arXiv 2505.18278) [S].
- **Interaction effects.**
  - Functional ANOVA decomposes performance into main and joint module effects (Quantifying Individual and Joint Module Impact, arXiv 2405.11964; PSO-X, arXiv 2601.04100) [S].
  - The regression interaction β₃(A×B) is the parametric version [K].
- **Portfolio gain.** The VBS–SBS gap and Shapley or marginal contribution (Fréchette et al., AAAI 2016) [S].

**Conclusion.** A synergy estimand ("hybrid vs best component at equal FE", with a paired test, or an interaction term) is not a contribution in itself (K6).

---

## 6. Literature search methodology

- **Channel.** Web search returning summaries; about 15 new queries this gate, plus about 15 from the discovery gate on R6.
- **Coverage:** brief §4 A–G — behavioural characterisation, complementarity, synergy, predicting hybrid/portfolio performance, portfolios and ensembles, heterogeneous islands, hyper-heuristics. Plus ELA/trajectory features, fANOVA interactions, and sequential/switching portfolios.
- **Full texts.** Not accessible.

---

## 7. Closest prior art (FULL TEXT UNVERIFIED for all)

| # | Paper | Year / venue / ID | Demonstrated (summary) | Overlap with R6 | Remaining difference |
|---|---|---|---|---|---|
| Q1 | **"To Switch or Not to Switch: Predicting the Benefit of Switching Between Algorithms Based on Trajectory Features"** | 2023, EvoApplications (LNCS), doi 10.1007/978-3-031-30229-9_22 | Predicts from trajectory features whether switching from one algorithm to another during a run is beneficial | **HIGH**: pre-combination trajectory information predicts the benefit of combining two optimizers (sequential hybrid) | sequential, not concurrent; features describe the problem trajectory of A, not the pairwise behavioural relation [U] |
| Q2 | Kostovska, Jankovic, Vermetten et al., **"Per-run Algorithm Selection with Warm-starting using Trajectory-based Features"** | 2022, PPSN XVII; arXiv 2204.09483 | Run A, extract trajectory features, predict the performance of warm-started B for the remainder; beats static per-instance AS | **HIGH**: predicts the performance of an A→B combination out of sample | sequential; per instance |
| Q3 | **"Similarity-based Portfolio Construction for Black-box Optimization"** | 2026, GECCO '26, doi 10.1145/3795095.3805145; arXiv 2604.18196 | Sequential portfolios split budget across algorithms and benefit from "complementarities between algorithms"; kNN finetuning on unseen instances | **HIGH**: pairwise/similarity-based complementarity used to build combinations for unseen problems | instance features, not behavioural descriptors [S] |
| Q4 | **PS-AAS** | 2023, arXiv 2310.10685 | Graph of algorithms by meta-representation similarity (SHAP, performance2vec); selects diverse, non-redundant, complementary portfolios on BBOB | **HIGH**: pairwise algorithm-representation similarity → complementary set | performance-based representations; portfolio, not hybrid |
| Q5 | Hayward & Engelbrecht, **"Determining Metaheuristic Similarity Using Behavioral Analysis"** | 2025, IEEE TEVC 29(1), doi 10.1109/TEVC.2023.3346672 | 20 behavioural characteristics; pairwise behavioural similarity across benchmark functions; most metaheuristics behave alike | **HIGH** for the descriptor layer (pairwise behavioural relation) | no synergy prediction reported [S][U] |
| Q6 | **"Cluster-based framework for metaheuristic empirical similarity"** | 2025, *Soft Computing*, doi 10.1007/s00500-025-11006-y | Performance-driven clustering of PSO instances; links components to performance | MEDIUM | — |
| Q7 | **"Quantifying Individual and Joint Module Impact"** (arXiv 2405.11964); **PSO-X module interactions** (Camacho-Villalón, Nikolikj, Dost, Tuba, Džeroski, Eftimov; arXiv 2601.04100, 2026) | 2024 / 2026 | fANOVA of main and **joint (interaction)** effects of components across problem classes | **HIGH** for the synergy estimand (interaction of combined components) | components within one framework, not two optimizers |
| Q8 | **PAP / EPM-PAP** (Peng, Tang, Chen, Yao; Tang et al., *Inf. Sci.* 2014) | 2010 / 2014 | Population-based portfolios with migration; constituent selection from an estimated performance matrix | **HIGH**: selects which algorithms to combine (with migration, i.e. concurrent) by predicted complementarity | performance-based, not behavioural |
| Q9 | **AMALGAM** (Vrugt & Robinson, PNAS 2007) | 2007 | Concurrent multi-method search with adaptive offspring allocation | MEDIUM: concurrent hybrid that adapts to complementarity online | online, not pre-hybrid prediction |
| Q10 | **Instance Space Analysis / footprints** (Smith-Miles et al.; ACM CSUR 2023, doi 10.1145/3572895) | 2014–2023 | Footprints reveal complementary algorithm strengths; supports selection | MEDIUM–HIGH: complementarity measured before combination | performance/landscape-based |
| Q11 | **DynamoRep** (arXiv 2306.05438); **Opt2Vec** (*Inf. Sci.* 2024, S002002552401048X); meta-feature survey (arXiv 2406.06629) | 2023–2024 | Trajectory-based population-dynamics representations for problem and *algorithm* classification and performance prediction | MEDIUM–HIGH: optimizer behavioural signatures from trajectories | not used for pairwise synergy [S] |
| Q12 | **Behaviour Space Analysis of LLM-driven Meta-heuristic Discovery** (van Stein, Yin, Kononova, Bäck, Ochoa) | 2025/26, IJCCI (CCIS 2828); arXiv 2507.03605 | Exploration, exploitation, convergence and stagnation metrics; links behaviour to performance | MEDIUM | single algorithms |
| Q13 | **Kuncheva & Whitaker**, "Measures of diversity in classifier ensembles…" | 2003, *Machine Learning* 51(2):181–207, doi 10.1023/A:1022859003006 | Pairwise diversity relates only weakly to ensemble accuracy | **HIGH** for the "diversity → combination benefit" hypothesis: an established, largely negative answer in the ensemble analogue | ML ensembles, not optimizers |
| Q14 | **METAFOR** (arXiv 2502.11225); "Hybrid metaheuristics: an automated approach" (ESWA 2019); LTR-generated hybrids (*Algorithms* 18(6) 2025, doi 10.3390/a18060316) | 2019–2025 | Automated hybrid design; METAFOR reports which hybridisation types work best per problem class | MEDIUM–HIGH: data-driven knowledge of which combinations are synergistic | performance search, not a behavioural predictor |
| Q15 | Sroka & Wierzchoń 2025 (arXiv 2509.05445); Hybrid Algorithmic Synergy Index (arXiv 2505.18278) | 2025 | 19 hybrids from one plug-in operator; synergy index | MEDIUM | — |

---

## 8–15. Literature by area (condensed)

- **Behavioural characterisation (§8).** Mature: Q5, Q6, Q11, Q12, plus Search Trajectory Networks. The R6 descriptor layer is not new.
- **Algorithm portfolios (§9).** Pairwise complementarity drives portfolio construction (Q3, Q4, Q8, Q10, Shapley). R6 risks being a portfolio-construction problem with a different combination operator (K3).
- **Hyper-heuristics / adaptive selection (§10).** Online complementarity exploitation is established: AOS bandits, AMALGAM, dynamic portfolios, LCC (GECCO 2026). R6's offline prediction is the pre-hybrid analogue of what these methods learn online.
- **Hybridisation / heterogeneous islands (§11).**
  - Heterogeneous archipelagos outperform homogeneous ones in some studies [S].
  - Knowledge of algorithms' "preferences toward migration" has been used to construct heterogeneous archipelagos [S].
  - Exactly what was done there is [U]; it is a direct threat to the concurrent-hybrid gap.
- **ELA / meta-features (§12).** The ELA-based AS line uses *landscape* features. Trajectory-based features (Q1, Q2, Q11) mix landscape and optimizer behaviour.
  - R6's purely *optimizer-behavioural* framing is a descriptor choice.
  - Moreover, optimizer behaviour is landscape-conditioned, so "behavioural vs landscape features" is not a clean separation [INF].
- **Pairwise complementarity (§13).** Established for portfolios (Q3, Q4, Q10) and for behavioural similarity (Q5).
- **Pre-combination synergy prediction (§14).** Established for sequential combination (Q1, Q2). Not found for concurrent hybrids [S].
- **Out-of-sample (§15).** Leave-instance- and leave-problem-out prediction is standard in these lines (Q2 beats static AS on unseen runs; Q3 on unseen instances; Nikolikj et al. on leave-one-problem-out performance prediction) [S]. Out-of-sample validation would not be a differentiator.

---

## 16. Detailed overlap analysis (vs the strongest candidate claim)

> Candidate claim: *"A behavioural complementarity index computed entirely from independent optimizer trajectories predicts the out-of-sample synergy of subsequently hybridised optimizer pairs."*

| element of claim | status |
|---|---|
| characterise individual optimizer behaviour | established (Q5, Q11, Q12) |
| compute pairwise behavioural similarity or dissimilarity | established (Q5; Q4 for meta-representations) |
| predict combination benefit | established for **sequential** combination (Q1, Q2) and **portfolios** (Q3, Q4, Q8) |
| validate out of sample | established practice (Q2, Q3) |
| combination = concurrent exchange-based hybrid | **not found** [S]. Partially threatened by Q8 (PAP with migration, performance-based selection) and heterogeneous-archipelago construction [U] |

**Gap:** what survives is a combination-mode change (sequential/portfolio → concurrent) plus a descriptor choice (behavioural vs performance/trajectory-of-problem).

---

## 17. Strongest novelty threats

1. **Q1 + Q2.** Benefit-of-combination prediction from trajectories, out of sample. The core question, in the sequential mode.
2. **Q3, Q4, Q8, Q10.** Complementarity-based selection of which algorithms to combine (portfolio mode; PAP even concurrent with migration).
3. **Q5.** Pairwise behavioural similarity of metaheuristics, already measured at scale.
4. **Q7.** Interaction (synergy) estimands already quantified (fANOVA).
5. **Q13.** The ensemble analogue suggests "difference ⇒ benefit" is weak. The likely empirical outcome is a null, and a null would read as "Kuncheva & Whitaker for optimizers" [INF].

---

## 18. Unresolved papers (relevant only to C → D; none could move to A)

| paper | question for the full text |
|---|---|
| Q1 "To Switch or Not to Switch" | Which trajectory features? Do they include features of *both* algorithms? Validation protocol? |
| Q8 PAP / EPM-PAP | Does constituent selection use any behavioural (non-performance) signal? Is pairwise synergy with migration analysed? |
| Q5 Hayward & Engelbrecht | Any link from behavioural similarity to combination or ensemble performance? |
| Heterogeneous-archipelago construction from migration preferences (island-model literature, exact paper [U]) | Is the archipelago composition predicted from per-algorithm behaviour? |
| Q14 METAFOR | Are hybrid-type successes explained by component behaviour? |

---

## 19. Kill conditions

| cond. | status |
|---|---|
| K1: diversity/complementarity already used to predict portfolio/hybrid synergy | **met** for portfolios (Q3, Q4, Q8, Q10) and for sequential hybrids (Q1, Q2) [S] |
| K2: pre-combination descriptors predict whether combining a pair helps | **met** for sequential combination (Q1, Q2) [S]; concurrent [U] |
| K3: equivalent pairwise complementarity-selection in the portfolio literature | **met** (Q3, Q4, Q8) [S] |
| K4: complementarity index = renamed dissimilarity | **likely.** Every behavioural descriptor in §4 A–D is a difference measure; no established mechanism turns difference into complementarity (Q13) [INF] |
| K5: only novelty is HBA+MPA | the repository supports only that one pair; any general study requires new infrastructure [REPO] |
| K6: only a new synergy metric | the synergy estimand already exists (§5) |
| K7: only an empirical pair comparison | the risk is high if prediction fails (Q13 prior) |
| K8: leakage | avoidable by design (standalone runs only); not a novelty issue |
| K9: same problem solved by meta-learning/AS with equivalent features | **largely met** (Q1–Q4, Q11) |

---

## 20. Final decision

**C — NOVELTY INSUFFICIENT**

---

## 21. Minimum defensible research question

Not applicable (decision C).

---

## 22. Exact reason to abandon

- **The question is already studied.** "Does a pre-combination relationship between optimizers predict the benefit of combining them, out of sample?" is studied for sequential combinations (Q1, Q2), and complementarity-driven selection is standard for portfolios (Q3, Q4, Q8, Q10).
- **The building blocks are established.** Behavioural descriptors (Q5, Q11, Q12) and synergy estimands (§5, Q7) both exist.
- **What R6 would add** is a change of combination mode (concurrent exchange hybrid) and descriptor family (behavioural). By the brief's §19 C and §22 that is insufficient for a KBS contribution.
- **The empirical prior is unfavourable.** The ensemble analogue (Q13) found diversity weakly predictive.
- **The repository cannot support it.** It contains one optimizer pair. Testing R6 properly would require building a new multi-optimizer pool and a generic hybridisation protocol — a large investment in a direction already classified C.

---

## 23. Recommended next action

1. **Do not implement R6.**
2. **Close the line.** All directions derived from the N1/D3 observation have now been gated: N1, measurement validity, R1–R10, R3, compatibility, causal exchange utility, R6. Each reached C or worse. I recommend formally closing the LA-KAIE/HBA–MPA hybridisation line as a KBS-novelty source.
3. **Preserve the completed work** as validated infrastructure and a negative-results record: the MPHBS reproduction, the D3 mechanism findings, and the gate reports.
4. **Choose any new direction independently of this codebase**, and gate it with full-text access. That means allowing arxiv.org (and ideally publisher hosts) in the environment network policy, or supplying PDFs. Every gate so far has been limited to search summaries.

**Freeze confirmation:** code, branches, seeds, experiments, HBA/MPA parameters, specifications and benchmark results are all unchanged. Repository inspection was read-only (file reads, NPZ field listing).

**Sources (search results; none read in full)**
- https://link.springer.com/chapter/10.1007/978-3-031-30229-9_22
- https://arxiv.org/abs/2204.09483
- https://arxiv.org/abs/2204.06397
- https://doi.org/10.1145/3795095.3805145
- https://arxiv.org/abs/2604.18196
- https://arxiv.org/abs/2310.10685
- https://arxiv.org/html/2601.16896
- https://ieeexplore.ieee.org/document/10373561/
- https://link.springer.com/article/10.1007/s00500-025-11006-y
- https://arxiv.org/abs/2405.11964
- https://arxiv.org/abs/2601.04100
- https://www.sciencedirect.com/science/article/pii/S0020025514004022
- https://www.pnas.org/doi/10.1073/pnas.0610471104
- https://dl.acm.org/doi/10.1145/3572895
- https://arxiv.org/pdf/2306.05438
- https://www.sciencedirect.com/science/article/pii/S002002552401048X
- https://arxiv.org/html/2406.06629
- https://arxiv.org/abs/2507.03605
- https://dl.acm.org/doi/10.1023/A:1022859003006
- https://arxiv.org/pdf/2502.11225
- https://doi.org/10.3390/a18060316
- https://arxiv.org/pdf/2509.05445
- https://arxiv.org/pdf/2505.18278
- https://www.cs.ubc.ca/~kevinlb/papers/2016-AAAI-portfolio-shapley-extended.pdf
