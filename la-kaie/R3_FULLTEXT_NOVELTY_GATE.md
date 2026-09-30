# R3 second-stage novelty gate: exchange channel decomposition

Date: 2026-09-30.

**Scope:** a literature-only gate. Nothing was implemented or run; see §19.

## Evidence tags

| tag | meaning |
|-----|---------|
| [FT] | full text |
| [ABS] | abstract |
| [S] | search summary or snippet |
| [CODE] | authors' public code, or public supplementary material shipped with it |
| [K] | established knowledge |
| [INF] | my inference |
| [U] | unresolved |

**No novelty claim below rests on [S], [K] or [INF] alone.**

---

## 1. Decision (stated first)

**B — R3 PROMISING — FULL-TEXT ACCESS STILL REQUIRED**

**Kill conditions.** None of the kill conditions A–D (§13 of the brief) was met by any
evidence found. But the absence of overlap **cannot be verified**, because the two most
critical papers remain unreadable:
- Hansen 2011;
- METAFOR / Camacho-Villalón's thesis.

**Two conceptual threats (from code, not snippets).** Both must be resolved even if gate
access later clears the literature:

1. **The four-arm design is ill-posed for non-elitist model-based receivers.**
   - In CMA-ES, and in the MTES-KG transfer scheme, a transferred solution is never
     *retained*. Its only effect is through the distribution update (mean, paths, covariance,
     step size) [CODE].
   - For these receivers channel C and channel S coincide, so "candidate-only" and
     "state-only" are not separable.
   - A *cross-receiver* decomposition therefore does not have a uniform definition. This
     weakens the generality claim at the heart of R3 [INF from CODE].
2. **Two-channel decompositions have precedent.**
   - Lamarckian vs Baldwinian learning in memetic algorithms separates "write the improved
     solution back" from "credit only its fitness" [S].
   - The 2×2 factorial with a mediator is the standard controlled-direct-effect design of
     causal inference [K].
   - Neither is the same as R3, but together they make the *method* standard. Only the
     *object* (receiver adaptive state in heterogeneous exchange) and the *predictive
     criterion* (H3.2) could be new.

**Honest assessment** [INF]:
- If the full texts show no overlap, R3 survives only in a narrowed form (§15-pre).
- Its most defensible novel element is H3.2, the out-of-sample harm prediction, not the
  decomposition itself.

---

## 2. Access status

**Network**
- Re-probed on 2026-09-30: arxiv.org, export.arxiv.org, HAL/INRIA, semanticscholar,
  openalex, iridia.ulb.ac.be, cmap.polytechnique.fr and all publishers are still blocked by
  the egress policy.
- No PDFs had been supplied to the repository.

**Primary material obtained legitimately** (public GitHub; cloned read-only into the
scratchpad, not added to the repository)

| source | commit | what it gives |
|--------|--------|---------------|
| `CMA-ES/pycma` (Hansen's reference implementation) | 48a821b (2026-06-09) | the actual injection mechanism referenced to "Hansen (2011)" in the code |
| `intLyc/MTES-KG` (Li, Gong & Li, TEVC 28(6):1733–1745, 2024) | f5f2b5a (2024-12-04) | MATLAB source **and `supp.pdf`**, the paper's 12-page supplementary file. This is [FT] for the *supplement only*, not the main paper |

**Required papers, one by one**

| # | paper | full text | evidence used |
|---|-------|-----------|---------------|
| 1 | Hansen 2011, *Injecting External Solutions into CMA-ES* (INRIA RR-7748; arXiv 1110.4181) | **UNRESOLVED** | [S]; [CODE] pycma |
| 2 | METAFOR (arXiv 2502.11225) and Camacho-Villalón PhD thesis | **UNRESOLVED** | [S] |
| 3 | Sroka & Wierzchoń 2025 (arXiv 2509.05445) | **UNRESOLVED** | [S] (earlier gate) |
| 4 | Warm-starting CMA-ES (arXiv 2012.06932) | **UNRESOLVED** | [S] |
| 5 | Island/distributed DE with migration-induced parameter adaptation | **UNRESOLVED** | [S] |
| 6 | 2024–2026 injection / external solutions / state contamination | partly | MTES-KG supplement [FT-supp] plus code; rest [S] |
| 7 | EMT surveys and knowledge-transfer papers | **UNRESOLVED** | [S] |
| 8 | Terms: "state contamination", "memory contamination", "archive contamination", "migration-induced adaptation", "state-aware transfer" | searched | no matching paper found [S] |

About 20 new queries were run, covering the brief's §6 list and its conceptual equivalents in
EMT, island models, portfolios, memetic algorithms and hybrids.

---

## 3. Paper-by-paper evidence

### P-H11 — Hansen 2011, injecting external solutions into CMA-ES

**Stated result** [S]:
- External solutions may come from a gradient or Newton step, a surrogate optimum, the
  best-ever solution, or "proposals from algorithms running in parallel".
- "Only small modifications … are necessary …: too long steps need to be tightly
  renormalized."
- "Injecting a single bad solution essentially corresponds to decreasing the population size
  by one."
- Always re-injecting the best-ever solution yields an elitist CMA-ES.

**Code** [CODE], `pycma/cma/evolution_strategy.py`:
- `inject()` stores `solution - self.mean` as an injection *direction* unless `force=True`.
- In `ask`, a direction is length-limited if
  `mahalanobis_norm(y) > N**0.5 * opts['CMA_injections_threshold_keep_len']`.
- Docstring: "injected solutions are not used in the 'active' update which would decrease
  variance in the covariance matrix in this direction."
- Changelog: "no negative weights for injected solutions".
- Gradient injection is commented "see Hansen (2011)".

**Assessment against the four R3 elements:**
- **State-perturbation control:** YES. The Mahalanobis length of the injected step,
  relative to the receiver's own distribution, is used to *limit* the state change. This is
  a receiver-relative state-perturbation measure, used as a **control rule**, not as a
  **predictor** [CODE].
- **Candidate vs state separation:** in CMA-ES the candidate is not retained, so there is
  nothing to separate [CODE][INF].
- **Harm prediction:** not found [S]; the full text is [U].
- **Cross-receiver:** NO. CMA-ES only.
- **Kill risk:** MEDIUM.
  - It *nearly* satisfies kill condition B for one receiver: Mahalanobis distance under the
    receiver's model as the compatibility basis for injection.
  - It does not establish a *general* criterion or a *predictive* comparison against
    fitness. That comparison must be checked in the full text.

### P-MF — METAFOR (Camacho-Villalón, Dorigo, Stützle; arXiv 2502.11225)

**Stated content** [S]:
- A component "DE-Recompute Velocity" specifies the velocity update when DE, which has
  precedence, finds a better solution. The reason given: "the velocity vector … may no
  longer correspond to its position".
- Four strategies: RV-goBack, RV-random, RV-position, RV-none.
- A second component, "PSO-only-on-fail".
- Components are selected automatically with irace. The paper reports "the algorithm
  components that contribute to the performance".

**Assessment against R3:**
- It is a *state-reconciliation component with four levels* for one receiver (PSO), chosen by
  automated configuration.
- RV-none vs RV-goBack is effectively state channel "untouched" vs "reconciled". It is a
  **partial state-channel ablation for a single receiver**, not a candidate × state
  factorial, and not a harm predictor [S][INF].
- Whether the paper reports component importance (e.g. an ablation of RV levels): [U].
- **Kill risk:** MEDIUM. The full text and thesis must be checked for a component-importance
  analysis framed as channel effects.

### P-KG — Li, Gong & Li 2024, MTES-KG (TEVC 28(6))

**Supplement** [FT-supp], Algorithm S-1 "CMA-ES-KT":
- A naive transfer samples some of the λ solutions from the source task's distribution,
  then runs a normal distribution update.
- Table S-I shows CMA-ES beating this naive transfer on most CEC17-MTSO cases ("+ / − / =
  12 / 5 / 1"). This is the paper's motivation that plain transfer harms ES.

**Code** [CODE], `MTES_KG.m`:
- External samples are appended to the λ samples.
- **Domain-knowledge sampling (DoS):** draws from the source distribution, but **clips the
  step from the target mean to `mStep`** (the target's mean-step length). This is a
  receiver-relative state-perturbation limit, like Hansen's.
- **Shape-knowledge sampling (SaS):** maps a source elite through the source's whitening
  (`B_k' * D_k^-1`) into the target's coordinates (`B_t * D_t`).
- **Transfer utility:** `sucExS = count(rank(1:mu) > lambda)`, i.e. how many external
  samples reach the top μ. This is a **candidate-fitness (rank) metric**.
- The number of external samples τ is adapted from that ratio (a threshold of 0.5 over
  `adjGap` generations).
- All external samples then enter the mean, path and covariance updates.

**Assessment against R3:**
- It is the **closest 2024 work**. It explicitly treats transferred solutions as external
  samples acting on the receiver's *distribution state*, and it limits their perturbation
  relative to the receiver.
- **But:**
  - its utility and regulation signal is candidate rank (fitness-based);
  - there is no decomposition;
  - there is no harm prediction from state perturbation.
- This is consistent with R3's gap, and it provides a natural *fitness-based baseline*
  (the external-sample success ratio) for H3.2 [CODE][INF].
- **Kill risk:** LOW–MEDIUM. The main paper's text is [U]; the ablations in the main paper
  are unknown.

### P-L/B — Lamarckian vs Baldwinian learning (memetic algorithms)

- **Lamarckian:** "the locally improved individual [is placed] back into the population".
- **Baldwinian:** "only alters the fitness … the improved genotype is not encoded back".
- Comparative studies exist, e.g. "Comparing Lamarckian and Baldwinian Approaches in Memetic
  Optimization", Springer 2023 [S].
- **Assessment:** this is a two-channel decomposition of the effect of externally generated
  improvement (write-back vs fitness credit). It is conceptually adjacent: a precedent for
  channel thinking, but its channels are genotype vs fitness, not candidate vs adaptive state
  [S][INF].
- **Kill risk:** LOW, but it removes any claim that "channel decomposition" is itself new.

### P-MS — Multi-swarm PSO with transfer of the best (PMC4542024) and similar

- "Rather than particles moving, stored global best solutions are modified by replacing them
  with randomly picked solutions from better-performing swarms" [S].
- **Assessment:** this is a *state-only transfer by design*: the receiver's gbest is
  rewritten without the candidate joining the swarm. It is an instance of the STATE-ONLY
  arm as an algorithm, not as a causal comparison [S][INF].
- **Kill risk:** LOW.

### P-EMT — EMT similarity via distribution divergence

- KL divergence or Wasserstein distance between *source and target population
  distributions* is used to select source tasks or control the transfer degree (e.g. TCYB
  2023) [S].
- MTEA-AD flags outlier transferred individuals [S] (earlier gate).
- **Assessment:** these measure *source–target distribution similarity* before transfer. They
  do not measure the *receiver's state change caused by* the transfer. SPI would be distinct
  in definition, but reviewers will see it as a close relative [INF].
- **Kill risk:** MEDIUM for the SPI-novelty sub-claim.

### P-SR — Sroka & Wierzchoń 2025; WS-CMA-ES 2020; island/distributed DE

- Only [S] evidence; nothing found on state channels. [U]

---

## 4. Critical questions

**Q1. Does a four-arm (candidate × state) intervention already exist?**
- **Not found** [S]. **Subsets exist:**
  - **STATE-ONLY as a design:** multi-swarm PSO that rewrites gbest from another swarm [S].
  - **State reconciled vs not, candidate always retained:** METAFOR RV-goBack/…/RV-none, one
    receiver (PSO) [S].
  - **State perturbation limited vs unlimited:** Hansen's injection threshold (CMA-ES)
    [CODE]; MTES-KG's DoS step clipping [CODE].
  - **Genotype write-back vs fitness-only:** Lamarckian/Baldwinian [S].
- No source shows all four arms, or candidate-only vs state-only compared within one receiver.
  This is **UNRESOLVED** until the METAFOR full text is read.

**Q2. Is there a general, receiver-state-based compatibility criterion?**
- **Receiver-specific versions exist:**
  - the Mahalanobis length under the receiver's CMA model (Hansen) [CODE];
  - step-length clipping to the receiver's mean step (MTES-KG) [CODE].
- **Related but different:** source–target distribution divergence (KL/WD/MMD) in EMT [S].
- **A general criterion**, "the divergence of the receiver's proposal distribution caused by
  the transfer", was **not found** [S][U].
- SPI would generalise the Hansen and MTES-KG rules; it is not a renamed existing metric.
  **But** for CMA-ES it may reduce to Mahalanobis length [INF]. That is kill condition E
  territory for that receiver.

**Q3. Is state perturbation → harm compared with fitness → harm out of sample?**
- **Not found** [S]. Existing regulators use candidate fitness or rank (MTES-KG `sucExS`
  [CODE]; EMT success-rate adaptation [S]). Learned transferability predictors (EMT-ADT
  decision trees; L2T) use evolution-state features, not receiver-state perturbation [S].
- **UNRESOLVED** at full-text level. This remains the strongest candidate contribution.

**Q4. Has this been done across receiver classes?**
- **Not found** [S]. Existing work is single-receiver: CMA-ES (Hansen, MTES-KG), PSO
  (METAFOR, multi-swarm).
- **Is generalising meaningful?** Only partly:
  - For **model-based non-elitist receivers** (CMA-ES, NES, EDAs), channels C and S
    coincide, so the decomposition degenerates [CODE][INF].
  - A meaningful cross-class claim must therefore restrict itself to **receivers that retain
    candidates *and* hold separable persistent state**:
    - target/leader-guided (HBA, PSO-gbest);
    - velocity-based (PSO);
    - success-history (SHADE: population plus memory plus archive);
    - elitist or archive-based variants.
  - It must treat model-based receivers as a separate, degenerate class where only SPI
    (H3.2) applies.

---

## 5. Strict novelty matrix

| Prior work | Candidate channel | State channel | Four-arm intervention | State perturbation metric | Harm prediction | Cross-receiver | Cross-optimizer |
|------------|-------------------|---------------|-----------------------|---------------------------|-----------------|----------------|-----------------|
| Hansen 2011 (+ pycma) | NO | YES | NO | PARTIAL | UNRESOLVED | NO | NO |
| METAFOR (DE-Recompute Velocity) | PARTIAL | PARTIAL | UNRESOLVED | NO | UNRESOLVED | NO | PARTIAL |
| MTES-KG 2024 | PARTIAL | YES | NO | PARTIAL | NO | NO | NO |
| Lamarckian vs Baldwinian | YES | NO | NO | NO | NO | UNRESOLVED | PARTIAL |
| Multi-swarm PSO gbest transfer | NO | YES | NO | NO | NO | NO | NO |
| EMT KL/WD/MMD similarity; MTEA-AD | PARTIAL | NO | NO | NO | PARTIAL | NO | PARTIAL |
| Sroka & Wierzchoń 2025 | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | YES |
| WS-CMA-ES 2020 | NO | PARTIAL | NO | NO | NO | NO | NO |
| **Proposed R3** | YES | YES | YES | YES | YES | YES | YES |

**Evidence behind the cells**

| work | basis |
|------|-------|
| Hansen 2011 | [S] + [CODE] |
| MTES-KG 2024 | [CODE] + [FT-supp] |
| METAFOR | [S] |
| Lamarckian vs Baldwinian | [S] |
| multi-swarm PSO | [S] |
| EMT similarity / MTEA-AD | [S] |
| Sroka & Wierzchoń 2025 | [S]; its hybridisation operator's state handling is unknown |
| WS-CMA-ES 2020 | [S]; it initialises the receiver's *state* from source-task data, a state-only transfer at start |

- **Hansen 2011, "Harm prediction" = UNRESOLVED:** the full text might contain an analysis
  of harm vs injected-step length.
- **MTES-KG, "Candidate channel" = PARTIAL:** external samples are ranked and counted, not
  retained.
- **MTES-KG, "State perturbation metric" = PARTIAL:** the DoS step clip uses the mean-step
  length.
- **EMT similarity, "Harm prediction" = PARTIAL:** it identifies harmful transfer, but by
  source–target similarity, not by receiver-state change.

---

## 6. Paper-by-paper overlap table

| Paper | What it actually does | Exact R3 overlap | What remains different | Kill risk |
|-------|------------------------|------------------|------------------------|-----------|
| Hansen 2011 | makes CMA-ES accept external solutions safely; clips injected steps by receiver Mahalanobis length; no negative weights for them [S][CODE] | receiver-relative state-perturbation limit | limit ≠ predictor; single receiver; no channel comparison (degenerate in CMA-ES) | MEDIUM |
| METAFOR | configurable PSO-DE hybrid; four velocity-recompute rules after DE improvement; irace selects components [S] | state reconciliation as an ablatable component | no candidate-only/state-only comparison; no SPI; no harm prediction [U] | MEDIUM |
| MTES-KG | EMT for ES: external samples from source distribution (clipped) or shape-mapped elites; τ adapted by external-sample rank success [CODE][FT-supp] | transfer acts on receiver distribution state; perturbation clipped | fitness/rank-based utility; no decomposition; no state-based prediction | LOW–MEDIUM |
| Lamarckian vs Baldwinian | write-back vs fitness-only credit of local-search improvement [S] | two-channel decomposition idea | different channels; memetic, not heterogeneous exchange | LOW |
| Multi-swarm PSO gbest transfer | overwrites receiver gbest with other swarms' solutions [S] | state-only transfer as design | no comparison, no measurement | LOW |
| EMT similarity (KL/WD/MMD), MTEA-AD | source–target distribution similarity or outlier detection gates transfer [S] | transfer-compatibility criterion | pre-transfer similarity, not post-transfer receiver-state change | MEDIUM (for SPI) |
| Standard mediation / 2×2 factorial | controlled direct/indirect effects [K] | the causal design | nothing new as method | — (limits the claim) |

---

## 7. Kill conditions (brief §13)

| cond. | status | basis |
|-------|--------|-------|
| A: an equivalent four-arm decomposition exists | **not found**; UNRESOLVED (METAFOR full text) | [S] |
| B: receiver-state perturbation is already the basis for compatibility | **partially met for single receivers** (Hansen Mahalanobis clip; MTES-KG step clip); not met as a *general* criterion | [CODE] |
| C: state-perturbation harm prediction is already shown to beat fitness | **not found**; UNRESOLVED | [S] |
| D: a 2024–2026 paper does essentially the full study | **not found** | [S] |
| E: the only novelty is "apply known reconciliation to more optimizers" | **real risk.** Without H3.2 (prediction), the cross-receiver decomposition is largely this, and for model-based receivers it is ill-posed | [INF][CODE] |

---

## 8. Survival criteria (brief §14)

1. **State effects known individually; no general causal decomposition:**
   - TRUE as far as evidence goes: known for CMA-ES [CODE], PSO [S] and gbest transfer [S].
   - Absence is [S]/[U].
2. **No work measures C vs S across receiver classes:** not found [S]/[U].
3. **No general receiver-relative perturbation measure:**
   - Receiver-specific ones exist (Hansen, MTES-KG) [CODE].
   - A general one was not found [S]/[U].
   - SPI is a generalisation, not a rename, but it collapses to them for ES receivers [INF].
4. **No work shows such a measure predicts harm beyond fitness:** not found [S]/[U].
5. **The combination is a genuine question rather than a set of known fixes:**
   - Only if H3.2 is primary: *does receiver-state perturbation predict harm better than
     candidate fitness, out of sample, across receiver classes?*
   - The decomposition (H3.1) alone risks kill condition E, and is ill-posed for non-elitist
     model-based receivers [INF].

Items 1–4 cannot be confirmed without full texts, so the verdict is B, not A.

---

## 15-pre. Required narrowing *if* gate access later clears the literature

This is not the §15 reframing; §15 applies only to decision A.

- **Primary contribution:** H3.2 (the predictive compatibility criterion). Demote H3.1 to a
  supporting analysis.
- **Receiver scope:** restrict the decomposition to receivers where both channels are
  separately defined: PSO (velocity/pbest/gbest), HBA-type target-guided, SHADE (memory plus
  archive), and elitist variants.
- **Model-based receivers** (CMA-ES, EDA/NES) are covered only by SPI, and with explicit
  acknowledgement that SPI then reduces to Hansen-style Mahalanobis length.
- **Mandatory baselines:**
  - MTES-KG's external-sample rank-success ratio;
  - candidate rank/improvement;
  - EMT source–target similarity (KL/MMD), to show that SPI is not a rename of pre-transfer
    similarity.

---

## 16. Decision

**B — R3 PROMISING — FULL-TEXT ACCESS STILL REQUIRED**

**To reach A or C, these must be read in full:**
1. **Hansen 2011** (RR-7748). Does it analyse harm as a function of injected-step Mahalanobis
   length, or compare it with fitness?
2. **METAFOR and Camacho-Villalón's thesis.** Is there an ablation or importance analysis of
   DE-Recompute Velocity that amounts to a state-channel decomposition?
3. **MTES-KG main text.** Does it ablate DoS clipping vs no clipping, and does it analyse
   distribution-adaptation error as a harm mechanism?
4. **One 2024–2026 EMT survey** (e.g. the knowledge-transfer survey, ASOC 2023; the
   methodological overview arXiv 2102.02558), searched for "strategy parameter" or "model"
   transfer and harm prediction.

**Automatic C:** if (1) or (3) already shows state-perturbation-based harm prediction, or
(2) contains a candidate × state decomposition.

**Enabling access:** allow arxiv.org in the environment's network policy (this covers 1, 2
and 4), or place the PDFs in the repository.

R6 is not investigated here, as instructed.

---

## 19. Implementation freeze

| item | status |
|------|--------|
| N1 changed? | **NO** |
| D3 changed? | **NO** |
| Production code changed? | **NO** |
| New experiments? | **NO** |
| New seeds? | **NO** |
| Specification changed? | **NO** |

- This report is the only new repository file.
- Two public repositories (pycma, MTES-KG) were cloned read-only into the session scratchpad
  for inspection.
- A throwaway Python virtual environment with `pypdf` was created there, only to extract the
  MTES-KG supplement text. The system `pdfminer`/`pypdf` failed on a broken `cryptography`
  binding.

IMPLEMENTATION REMAINS FROZEN PENDING R3 LITERATURE DECISION.

---

## Sources

**Search results [S]**
- https://arxiv.org/pdf/1110.4181
- https://arxiv.org/pdf/2502.11225
- https://www.researchgate.net/publication/389089876_METAFOR_A_Hybrid_Metaheuristics_Software_Framework_for_Single-Objective_Continuous_Optimization_Problems
- https://ieeexplore.ieee.org/document/10309246/
- https://dl.acm.org/doi/abs/10.1109/TEVC.2023.3330265
- https://link.springer.com/chapter/10.1007/978-3-031-41774-0_41
- https://pmc.ncbi.nlm.nih.gov/articles/PMC4542024/
- https://doi.org/10.1109/tcyb.2023.3234969
- https://arxiv.org/pdf/2012.06932
- https://arxiv.org/abs/2509.05445
- https://www.sciencedirect.com/science/article/abs/pii/S1568494623002004
- https://arxiv.org/pdf/2406.14359

**Code [CODE]**
- https://github.com/CMA-ES/pycma @ 48a821b
- https://github.com/intLyc/MTES-KG @ f5f2b5a (includes supp.pdf)
