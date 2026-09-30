# R3 FINAL LITERATURE GATE

Date: 2026-09-30.

This is a literature-only gate. No code, experiments or seeds were produced; see §11.

## Decision

**B — R3 STILL UNRESOLVED**

## One-sentence reason

None of the three critical full texts (Hansen 2011, METAFOR plus Camacho-Villalón's thesis,
and the MTES-KG main paper) could be accessed. Snippet, code and supplement evidence shows
no kill. But it also shows that, for Gaussian-model receivers, the proposed signal is a
monotone function of an already-used quantity: the Mahalanobis distance / likelihood under
the receiver's distribution. So there is no positive evidence that the essential H3.2
contribution is unestablished.

## Access record (unchanged since the previous gate)

**Blocked on 2026-09-30:**
- arxiv.org and export.arxiv.org;
- inria.hal.science;
- ieeexplore.ieee.org;
- ULB repositories (difusion, dipot);
- researchgate;
- openalex, unpaywall;
- scholar.archive.org;
- the GitHub REST API for issues (HTTP 403).

**No PDFs have been added to the repository.**

**Primary material available:**
- `pycma` @ 48a821b [CODE];
- `intLyc/MTES-KG` @ f5f2b5a, with its code [CODE] and the 12-page supplement [SUPP].

**Searches:** 5 new targeted searches ([S]), in addition to the roughly 55 run in earlier
gates.

---

## 1. Hansen 2011

| field | entry |
|-------|-------|
| Paper | N. Hansen, *Injecting External Solutions Into CMA-ES* |
| Year / venue | 2011, INRIA research report RR-7748; arXiv 1110.4181 |
| Full text available? | **NO** [U] |
| What it does | makes CMA-ES accept external solutions (gradient/Newton steps, surrogate optima, best-ever, parallel algorithms) by renormalising too-long injected steps [S] |
| Receiver state involved | **YES**: mean, evolution paths, covariance, step size [S][CODE] |
| Candidate channel involved | **NO** as a separate channel. Injected solutions are not retained; they act only through the update (non-elitist CMA-ES) [CODE][INF] |
| State perturbation measured | **YES, as a control quantity:** the Mahalanobis step length. [S] gives the clipping rule ‖x − m‖_{σ²C} ≤ √n + 2n/(n+2), which "seems to be sufficient" to prevent divergence. The pycma implementation uses `mahalanobis_norm(y) > N**0.5 * CMA_injections_threshold_keep_len` [CODE] |
| Transfer outcome measured | **YES** [S]: "almost n times faster" when the injections are useful ("600 vs 5000 and 2000 vs 70000 evaluations in 10 and 40-D"); "single bad injected solutions … do no significant harm" |
| Harm prediction | **UNRESOLVED.** [S] shows a qualitative mechanism (long steps → divergence), not a predictive model. The full text is needed |
| Candidate-fitness baseline | **UNRESOLVED** |
| Out-of-sample prediction | **UNRESOLVED.** Nothing in [S] suggests it; this cannot be excluded |
| Cross-receiver | **NO**: CMA-ES only [S][CODE] |
| Exact overlap with R3 | **HIGH for CMA-ES** (the same state-perturbation quantity; see §6). LOW for non-distributional receivers |
| Kill condition triggered | **NONE demonstrated.** K2 is *possible* for CMA-ES if the full text links Mahalanobis step length to harm quantitatively |
| Evidence | [S], [CODE]; full text [U]. No page, section or equation numbers are available |

**Question that must be answered from the full text:** does RR-7748 quantify outcome (harm
or divergence) as a function of injected-step Mahalanobis length? If so, does it compare this
with the injected solution's fitness or rank?

---

## 2. METAFOR

| field | entry |
|-------|-------|
| Paper | C. Camacho-Villalón, M. Dorigo, T. Stützle, *METAFOR: A Hybrid Metaheuristics Software Framework for Single-Objective Continuous Optimization Problems* |
| Year / venue | 2025, arXiv 2502.11225 (journal version [U]) |
| Full text available? | **NO** [U] |
| What it does | a modular framework with more than 100 components for PSO, DE, CMA-ES and local search; irace-based automatic design of hybrids. Among six hybrid types, PSO+CMA-ES was best, then DE+CMA-ES, then PSO+DE [S] |
| Receiver state involved | **YES**: the PSO velocity after DE improves a particle ("DE-Recompute Velocity": RV-goBack, RV-random, RV-position, RV-none) [S] |
| Candidate channel involved | **YES**: the DE-improved solution replaces the particle position [S][INF] |
| State perturbation measured | **NO** evidence [S] |
| Transfer outcome measured | performance of configured hybrids [S]. Per-component effect: [U] |
| Harm prediction | **NO** evidence [S]; [U] |
| Candidate-fitness baseline | **NO** evidence [S]; [U] |
| Out-of-sample prediction | **NO** evidence [S]; [U] |
| Cross-receiver | PARTIAL: several hybrid pairs, but state reconciliation is specified for PSO only [S] |
| Exact overlap with R3 | **MEDIUM** for H3.1 (state reconciled vs not, candidate retained, is a half of the factorial for one receiver). **LOW** for H3.2 |
| Kill condition triggered | **NONE demonstrated.** K1 is possible only if the full text reports an RV-level importance analysis framed as channel effects |
| Evidence | [S]; full text [U] |

**Provides vs decomposes.**
- On available evidence, METAFOR **provides** state-reconciliation *options* as tunable
  components. It does **not** (as far as [S] shows) decompose transfer benefit into
  candidate vs receiver-state effects.
- Selection by irace is performance-driven configuration, not a causal decomposition.
- **This cannot be confirmed without the text.**

**Adjacent 2026 work.** A. Nikolikj & C. Camacho-Villalón, *Quantifying the Impact of Modules
and Their Interactions in the PSO-X Framework*, arXiv 2601.04100 [S].
- It is a module-importance and interaction analysis for PSO-X (PSO only, not hybrids with
  exchange).
- It shows the group applies importance and interaction analysis to its frameworks. **An
  equivalent analysis of METAFOR's hybrid components (including RV) is a live K1 risk** [INF].
- Its full text is [U].

---

## 3. Camacho-Villalón thesis

| field | entry |
|-------|-------|
| Paper | C. L. Camacho-Villalón, PhD thesis (ULB/IRIDIA), *Designing new metaheuristic implementations: from metaphors to …* (full title [U]) |
| Full text available? | **NO**: iridia.ulb.ac.be and the ULB repositories are blocked [U] |
| Everything else | **UNRESOLVED.** Whether the thesis contains a METAFOR component analysis beyond the arXiv paper is unknown |
| Kill condition triggered | NONE demonstrated; K1 [U] |

---

## 4. MTES-KG 2024

| field | entry |
|-------|-------|
| Paper | Y. Li, W. Gong, S. Li, *Multitask Evolution Strategy With Knowledge-Guided External Sampling* |
| Year / venue | 2024, IEEE TEVC 28(6):1733–1745, DOI 10.1109/TEVC.2023.3330265 |
| Full text available? | **Main paper: NO** [U]. **Supplement: YES** [SUPP]. **Code: YES** [CODE] |
| What it does | EMT with CMA-ES receivers. Source-task solutions enter the target's update as extra "external samples". **DoS** draws from the source distribution with the step from the target mean **clipped to the target's mean-step length** `mStep`. **SaS** maps a source elite through source whitening into target coordinates. The number of external samples τ is adapted by their **rank success** (`sucExS` = number of external samples in the top μ; threshold 0.5 over `adjGap` generations) [CODE] |
| Receiver state involved | **YES**: target mean, paths, covariance, σ [CODE] |
| Candidate channel involved | PARTIAL: external samples are evaluated and ranked, but not retained beyond the update [CODE] |
| State perturbation measured | **PARTIAL**: used as a *limit* (the DoS clip to `mStep`), not recorded as a variable [CODE] |
| Transfer outcome measured | **YES**: benchmark performance. [SUPP] Table S-I shows naive transfer ("CMA-ES-KT", Alg. S-1) losing to CMA-ES (+/−/= 12/5/1) |
| Harm prediction | **NO** in code or supplement. The main paper is [U] |
| Candidate-fitness baseline | its *own* regulator is candidate rank-based [CODE]; no comparison against a state-based signal in [CODE]/[SUPP] |
| Out-of-sample prediction | **NO** in code or supplement; main paper [U] |
| Cross-receiver | **NO**: CMA-ES only |
| Exact overlap with R3 | **MEDIUM**: the same receiver-state channel and a receiver-relative perturbation limit, but a fitness/rank-based utility signal |
| Kill condition triggered | **NONE demonstrated.** The main-paper ablations (DoS clip on/off, τ fixed/adaptive, DoS vs SaS) and any analysis of "distribution adaptation errors" are [U] |
| Evidence | [CODE] (MTES_KG.m, lines around "Sample external solutions" and "Negative transfer mitigation"); [SUPP] p.1 (Alg. S-1, Table S-I), p.2 (Tables S-II, S-III). [S] for the main-paper claims |

**Brief §2 Q3 items 1–6:**
1. Explains help/harm via distribution-state perturbation: PARTIAL. [S] says external samples
   "can handle the difficulty of distribution adaptation errors"; the analysis itself is [U].
2. Quantifies perturbation magnitude: **not in [CODE]/[SUPP]**; main paper [U].
3. Predicts success/harm from it: **not in [CODE]/[SUPP]**; main paper [U].
4. Compares with fitness, similarity or other utility: **not in [CODE]/[SUPP]**; main paper [U].
5. Ablations: [U] (not in [SUPP]).
6. Establishes a general principle: **no evidence**; [U].

---

## 5. Other closest prior art

| work | what it does | relevance to H3.2 | evidence |
|------|--------------|-------------------|----------|
| **MTEA-AD** (Wang et al., TEVC 2022) | an anomaly-detection model per task marks transferred individuals as outliers, i.e. likely negative transfer, and filters them | **the closest functional threat for population-model receivers.** If its model is a density/likelihood model fitted to the *receiver* population (I believe it is Gaussian-based; **verify**), then "likelihood of the candidate under the receiver model" is already used *predictively* to gate transfer. That is K5 territory for the distributional part of SPI | [S]; model form [K, verify]; full text [U] |
| EMT-PD (two-stage adaptive transfer based on population distribution) | adaptive weight on step size in the first transfer stage, to reduce negative transfer | receiver-relative step scaling, like the Hansen/MTES-KG limits | [S] |
| EMT KL/WD/MMD similarity (e.g. TCYB 2023) | source–target population divergence selects sources or sets intensity | pre-transfer similarity, not post-transfer receiver change; a needed baseline | [S] |
| Online transfer with probabilistic outlier detection (Complex Intell. Syst. 2025) | monitors population distributions and removes detrimental outliers | same family as MTEA-AD | [S] |
| Multi-swarm PSO gbest transfer (PMC4542024) | overwrites receiver gbest (a state-only design) | STATE-ONLY arm as an algorithm; no prediction | [S] |
| Lamarckian vs Baldwinian | write-back vs fitness-only credit | a two-channel precedent; not state | [S] |
| WS-CMA-ES 2020 | initialises receiver distribution state from source data | state-only transfer at t=0; no harm prediction | [S] |
| Sroka & Wierzchoń 2025 | a plug-in hybridisation operator on 19 algorithms; invariance tests | state handling [U] | [S] |
| SHADE memory / archive under migration | no paper found on migration-induced memory contamination | absence ≠ novelty | [S] |

---

## 6. SPI equivalence analysis (mathematical)

**Notation**
- Receiver state θ (for example (m, σ, C) for CMA-ES), with induced proposal density
  p_θ(x).
- Transferred candidate x̃.
- θ' = U(θ; x̃) is the state after the receiver's own update that includes x̃.
- **Proposed generic SPI:** SPI(x̃) = D( p_θ ‖ p_θ' ), for some divergence D such as KL.
  Receiver-specific forms normalise the change in the state variable by the receiver's own
  scale.

**(a) Mahalanobis distance (Hansen 2011; pycma)**
- Definition: d_M(x̃) = ‖x̃ − m‖ in the metric (σ²C)⁻¹.
- For CMA-ES, a single injected solution with recombination weight w contributes:
  - Δm = c_m·w·(x̃ − m);
  - a rank-μ term c_μ·w·(y yᵀ − C), with y = (x̃ − m)/σ.
- For small updates, KL(p_θ ‖ p_θ') ≈ ½ Δmᵀ(σ²C)⁻¹Δm + ¼‖C^{-1/2} ΔC C^{-1/2}‖²_F.
- Both terms are functions of w and of z = C^{-1/2}y, whose squared norm is d_M². So
  KL ≈ f(w, d_M) with f increasing in d_M for fixed w.
- **Verdict:** for CMA-ES, SPI is a **monotone transformation of d_M, modulated by the
  recombination weight w** [INF; standard CMA-ES update algebra]. The weight w is itself a
  function of the candidate's **rank**, i.e. of candidate fitness.
- **So in CMA-ES the "state vs fitness" comparison is confounded by construction:** SPI
  contains the fitness signal. **Special case / monotone transform. Not novel for this
  receiver.**

**(b) Likelihood under the receiver population model (MTEA-AD-type, if Gaussian)**
- Definition: −log p̂_R(x̃), with p̂_R a Gaussian fitted to the receiver population. This
  equals ½ d_M,R(x̃)² + const.
- Adding x̃ to the population changes the fitted Gaussian by amounts that are, to first
  order, functions of d_M,R(x̃).
- **Verdict:** for population-model receivers, SPI is a **monotone function of the anomaly
  score**, and that score is already used *predictively* to gate transfer (MTEA-AD).
  **Special case / monotone transform** [INF; MTEA-AD model form must be verified].

**(c) KL/Wasserstein between source and target populations (EMT)**
- Definition: D(p_S ‖ p_T), computed before transfer.
- **Verdict:** related but distinct. It measures task similarity, not the receiver's change.
  It is a required baseline.

**(d) Mean shift normalised by receiver scale, for target-guided receivers (HBA x_prey, PSO gbest)**
- SPI_target = ‖x_prey' − x_prey‖ / s_pop.
- If x̃ becomes the target, this equals ‖x̃ − x_prey‖ / s_pop: a distance in the receiver's
  scale.
- **Verdict:** closely related to a (non-Mahalanobis) normalised distance of the candidate
  from the receiver's leader. The distinction from "candidate distance" features is thin
  [INF].

**(e) Velocity perturbation (PSO)**
- SPI_v = ‖v' − v‖ / ‖v‖ under the RV rule.
- **Verdict:** genuinely a state quantity. No predictive use was found [S], and none was
  verified [U].

**(f) Success-history memory displacement (SHADE)**
- SPI_M = |M_F' − M_F| + |M_CR' − M_CR| induced by an exchange-caused success.
- **Verdict:** genuinely a state quantity with no geometric analogue. No prior predictive use
  was found [S]; [U].

**Summary**
- SPI is **a generalisation**.
- For **distributional receivers** (CMA-ES, EDAs, population-Gaussian models) it **reduces
  (monotonically) to Mahalanobis distance / likelihood under the receiver**. That quantity is
  already used as a control (Hansen, MTES-KG) and, probably, predictively (MTEA-AD). In
  CMA-ES it is also confounded with candidate rank.
- It is **genuinely different only for non-distributional persistent state**: velocities and
  success-history parameter memories, and marginally targets/leaders.

---

## 7. Critical novelty matrix

| Work | Candidate channel | State channel | Four-arm intervention | Receiver-state perturbation | Harm prediction | Fitness comparison | Out-of-sample prediction | Cross-receiver | Cross-optimizer | Full R3 |
|------|-------------------|---------------|-----------------------|-----------------------------|-----------------|--------------------|--------------------------|----------------|-----------------|---------|
| Hansen 2011 | NO | YES | NO | PARTIAL | UNRESOLVED | UNRESOLVED | UNRESOLVED | NO | NO | NO |
| METAFOR | YES | PARTIAL | UNRESOLVED | NO | UNRESOLVED | UNRESOLVED | UNRESOLVED | PARTIAL | PARTIAL | UNRESOLVED |
| Camacho-Villalón thesis | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED |
| MTES-KG 2024 | PARTIAL | YES | NO | PARTIAL | UNRESOLVED | UNRESOLVED | UNRESOLVED | NO | NO | NO |
| WS-CMA-ES 2020 | NO | PARTIAL | NO | NO | NO | NO | NO | NO | NO | NO |
| Sroka & Wierzchoń 2025 | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | YES | UNRESOLVED |
| MTEA-AD / outlier-based EMT | PARTIAL | NO | NO | PARTIAL | YES | UNRESOLVED | UNRESOLVED | NO | PARTIAL | NO |
| EMT KL/WD similarity | PARTIAL | NO | NO | NO | PARTIAL | NO | UNRESOLVED | NO | PARTIAL | NO |
| Multi-swarm PSO gbest transfer | NO | YES | NO | NO | NO | NO | NO | NO | NO | NO |
| PSO-X module/interaction analysis 2026 | NO | NO | NO | NO | NO | NO | NO | NO | NO | NO |
| **Proposed R3** | YES | YES | YES | YES | YES | YES | YES | YES | YES | YES |

**Evidence behind the cells**

| work | basis |
|------|-------|
| Hansen 2011 | [S], [CODE] |
| METAFOR | [S] |
| Camacho-Villalón thesis | [U] |
| MTES-KG 2024 | [CODE], [SUPP], [S] |
| WS-CMA-ES 2020 | [S] |
| Sroka & Wierzchoń 2025 | [S] |
| MTEA-AD / outlier-based EMT | [S] and [K, verify] |
| EMT KL/WD similarity | [S] |
| Multi-swarm PSO gbest transfer | [S] |
| PSO-X module/interaction analysis 2026 | [S]; the RV component is not in PSO-X |

- **MTEA-AD, "Receiver-state perturbation" = PARTIAL:** it uses the candidate's likelihood
  under the receiver model, which is equivalent for Gaussian receivers (§6b).
- **MTEA-AD, "Harm prediction" = YES:** it predicts which transferred individuals are
  harmful, but not state-based.

---

## 8. Kill-condition audit

- **K1 (equivalent candidate × state factorial):**
  - **Not demonstrated.**
  - The live risk is the METAFOR paper, the thesis, or a METAFOR-level analogue of the 2026
    PSO-X module/interaction study. All [U].
- **K2 (receiver-relative state perturbation shown to predict harm/benefit):**
  - **Not demonstrated.**
  - Live risks:
    - Hansen 2011, for CMA-ES [U];
    - MTEA-AD, *if* its receiver-population likelihood is judged the same quantity (§6b)
      [U].
- **K3 (state perturbation compared with candidate fitness for harm prediction):**
  - **Not demonstrated.**
  - No evidence of such a comparison anywhere in [S]/[CODE]/[SUPP].
  - Note: for CMA-ES this comparison is confounded, because SPI includes rank weights (§6a).
- **K4 (a 2024–2026 paper performing essentially the full study):**
  - **Not demonstrated.** MTES-KG is the closest; its main text is [U].
- **K5 (SPI mathematically equivalent to an established, predictively used criterion):**
  - **Partially supported for distributional receivers**, and **not demonstrated at full-text
    level.**
  - SPI is a monotone function of Mahalanobis distance / receiver likelihood (§6a–b). That
    quantity is used as a control (Hansen [S][CODE]; MTES-KG [CODE]) and probably
    predictively (MTEA-AD [S][K]).
  - If the MTEA-AD full text confirms a Gaussian receiver-likelihood model evaluated as a
    harm filter, **K5 kills the distributional-receiver part of R3.**

---

## 9. What is genuinely new, if anything?

- **Not new:**
  - "receiver state matters";
  - Mahalanobis or likelihood-based compatibility for Gaussian-model receivers;
  - state reconciliation (METAFOR);
  - state-only transfer (gbest overwrite, warm start);
  - two-channel thinking (Lamarckian/Baldwinian);
  - the 2×2 mediator design.
- **Possibly new** (only if the unresolved full texts do not contain it, and **not
  established by positive evidence**): an out-of-sample test of whether perturbation of
  **non-distributional persistent state** predicts exchange harm/benefit with incremental
  value beyond two baselines. "Non-distributional state" means velocities, success-history
  parameter memories, and leader/target variables. The baselines are:
  - candidate fitness/rank/improvement;
  - candidate likelihood under the receiver's population model (the MTEA-AD/Hansen-type
    quantity).
- **This is narrower than R3 as proposed.** It excludes CMA-ES/EDA receivers, where SPI
  collapses to an existing quantity and is confounded with rank.

---

## 10. Final defensible novelty statement (conditional; not established)

Answer to "the smallest scientifically defensible statement that would still be novel if R3
passes":

> "For optimizers whose persistent adaptive state is not a sampling distribution —
> leader/target variables, velocities, and success-history parameter memories — the
> receiver-relative perturbation of that state caused by an incoming foreign candidate
> predicts, out of sample, whether the exchange degrades subsequent progress. It adds
> predictive value beyond (i) the candidate's fitness, rank or improvement and (ii) the
> candidate's likelihood (Mahalanobis distance) under the receiver's population or
> distribution model."

**Status:** unverified.
- Clause (ii) is essential. Without it, the claim collapses into Hansen/MTEA-AD territory.
- The statement is **not** made for CMA-ES/EDA/NES receivers.

---

## 11. Implementation decision

- **IMPLEMENT: NO**
- **Exact unresolved items.** Each needs the full text and has a specific question:
  1. **Hansen 2011 (RR-7748 / arXiv 1110.4181):**
     - Does it quantify harm or divergence as a function of injected-step Mahalanobis
       length?
     - Does it relate this to the injected solution's fitness or rank?
  2. **METAFOR (arXiv 2502.11225):** does it report a component-importance, ablation or
     interaction analysis of DE-Recompute Velocity (RV-goBack/random/position/none) that
     separates the effect of the DE candidate from the effect of the reconciled PSO state?
  3. **Camacho-Villalón PhD thesis (ULB/IRIDIA):** does it contain a hybrid-component
     analysis beyond the paper, e.g. a METAFOR counterpart of the PSO-X module/interaction
     study (arXiv 2601.04100)?
  4. **MTES-KG main paper (TEVC 28(6)):**
     - Does it ablate DoS step clipping, or analyse "distribution adaptation error" as a
       quantified harm mechanism?
     - Does it compare perturbation magnitude with rank-based success?
  5. **MTEA-AD (TEVC 2022; added in this gate as critical for K5):**
     - Is the anomaly-detection model a likelihood/density model fitted to the *receiver*
       population?
     - Is its outlier score evaluated as a predictor of harmful transfer, and against a
       fitness baseline?
- **Access needed:** allow arxiv.org in the environment's network policy (covers 1, 2 and
  the PSO-X paper), or place the PDFs of items 1–5 in the repository.
- R3 remains frozen, and **R6 is not started here.** Moving to R6 is the user's decision,
  whether R3 is later killed or still parked.

---

## 12. Evidence limitations

| tag | what it covers here |
|-----|---------------------|
| [FT] | **none** (no critical paper or thesis was readable) |
| [SUPP] | the MTES-KG supplementary file (12 pp.; Alg. S-1, Tables S-I to S-III) |
| [CODE] | `pycma` `inject()` and its Mahalanobis threshold; `MTES_KG.m` (DoS clip, SaS mapping, `sucExS` rank-based τ adaptation) |
| [ABS] | none read directly; the search tool gives summaries |
| [S] | Hansen 2011 results and clipping bound; METAFOR components and hybrid ranking; PSO-X module analysis (2026); MTEA-AD; EMT-PD; KL/WD similarity EMT; multi-swarm gbest transfer; Lamarckian/Baldwinian; WS-CMA-ES; Sroka & Wierzchoń |
| [K] | CMA-ES update algebra (§6a); MTEA-AD model form (**verify**) |
| [INF] | the SPI equivalence derivations (§6); channel degeneracy for non-elitist ES; the narrowed statement (§10) |
| [U] | all five items in §11 |

- "Not found" in this report means not found in [S]/[CODE]/[SUPP]. **It is not evidence of
  novelty.**

**Freeze:** N1, D3, production code, the specification, experiments and seeds are all
unchanged. This report is the only new repository file. No new scratchpad clones were made in
this gate.

IMPLEMENTATION REMAINS FROZEN PENDING R3 LITERATURE DECISION.

**Sources**
- https://arxiv.org/pdf/1110.4181
- https://www.researchgate.net/publication/51946510_Injecting_External_Solutions_Into_CMA-ES
- https://github.com/CMA-ES/pycma
- https://arxiv.org/pdf/2502.11225
- https://arxiv.org/pdf/2601.04100
- https://ieeexplore.ieee.org/document/10309246/
- https://github.com/intLyc/MTES-KG
- https://ieeexplore.ieee.org/document/9385398/
- https://arxiv.org/pdf/2001.00810
- https://link.springer.com/article/10.1007/s40747-025-02220-0
- https://doi.org/10.1109/tcyb.2023.3234969
- https://pmc.ncbi.nlm.nih.gov/articles/PMC4542024/
