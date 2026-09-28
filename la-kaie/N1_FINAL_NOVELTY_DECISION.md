# N1 final novelty decision

Date: 2026-09-28. Verification stage only: nothing was implemented, run,
tuned or written for the manuscript. Evidence and queries:
`N1_NOVELTY_EVIDENCE_LOG.md` (V1–V5 source-level verifications, S1–S26
searches).

## Final classification: **B — promising, but important unresolved overlap and evidence gaps remain**

Implementation should **not** proceed yet.

The classification is B, not A, for two independent reasons:
1. **genuine conceptual overlap** on E1 and E3 from multitasking and
   model-based EA literature, and
2. **inaccessible full texts** of the closest candidates.

The chance that full-text verification downgrades N1 to C is material (§6).

---

## 1. Element-by-element evidence

Legend: YES / PARTIAL / NO; "?" = cannot be determined without full text.
Evidence type: FT = full text, CODE = source code, SR = search result
(snippet or abstract only, not substantive).

| work (year, venue) | evidence | E1 structure governs inter-optimizer exchange | E2 statistical / model evidence | E3 sequential null test vs native control, suppression | E4 heterogeneous optimizers as corroborating witnesses | overall |
|---|---|---|---|---|---|---|
| MPHBS (2026, KBS) | FT + CODE | NO (fixed dimension-wise) | NO (SARSA-style TD) | NO | NO (sources, not witnesses) | reference |
| **Ensemble knowledge transfer framework for EMT, AIE + MAS** (2023, Swarm Evol. Comput., S2210650223001670) | SR | **PARTIAL**: adaptively selects *domain-adaptation (transfer-representation) strategies* between tasks | NO (multi-armed selection by performance) ? | ? | NO ? | **CLOSE PARTIAL OVERLAP** (E1 in a multitask setting) |
| MFEA-II (2019, IEEE TEVC) | CODE (port of the official code) | NO (transfer *intensity*, not structure) | **YES** for intensity (mixture log-likelihood maximisation) | **PARTIAL** (RMP → 0 suppresses transfer; continuous fit, no sequential test, no native control) | NO (tasks, not optimizers) | CLOSE PARTIAL OVERLAP (E2 + E3 components) |
| LCC (2025, arXiv 2504.17578) | SR | PARTIAL (selects *decomposition strategies* = structural treatments; not inter-optimizer exchange) | NO (DRL policy) ? | ? (whether "no decomposition" is an option is unknown) | NO | CONCEPTUAL PRECEDENT |
| LH-CC (2026, GECCO) | SR | NO (selects an optimizer per subproblem) | NO (meta-agent MDP) | NO ? | NO (heterogeneous optimizers but no structural evidence role) | DISTANT RELATED WORK |
| ACoS (2018, IEEE TCYB) + 2022 ASOC follow-up | SR | PARTIAL (original vs eigen coordinate system for variation, one population) | NO (probability vector updated from offspring) ? | NO | NO | CONCEPTUAL PRECEDENT |
| dd-CMA-ES (2020, Evol. Comput.) | **CODE** | NO (single optimizer's sampling model) | PARTIAL (continuous statistic: conditioning of C scales the diagonal learning rate; no model comparison, no penalty) | NO | NO | CONCEPTUAL PRECEDENT |
| GOMEA / RV-GOMEA library (2023–2025) | **CODE** | PARTIAL (the linkage model governs *mixing* within a population) | NO (model type user-chosen; tree built from MI/NMI; no BIC between types found) | NO | NO (multi-start populations of the same algorithm) | CONCEPTUAL PRECEDENT |
| *Predetermined versus learned linkage models* (2012, GECCO) | SR | PARTIAL (compares model types, apparently offline) | ? | NO ? | NO | CONCEPTUAL PRECEDENT |
| BOA / EBNA (structure by BIC/MDL) | SR | NO (structure of one sampling model) | **YES** (information criterion) | NO | NO | CONCEPTUAL PRECEDENT (for E2's machinery) |
| Gaussian Markov-network EDAs | SR | NO | YES (sparsity selection) | NO | NO | CONCEPTUAL PRECEDENT |
| OKTPO-MFEA (2025, Complex Intell. Syst.) | SR | NO | PARTIAL (probabilistic outlier detection of transfer candidates) | PARTIAL (filters harmful transfers; not a null test of exchange usefulness vs native control) ? | NO | CONCEPTUAL PRECEDENT |
| MGAD (2025, ESWA) | SR | NO | PARTIAL (anomaly detection) | PARTIAL (adaptive transfer probability) | NO | CONCEPTUAL PRECEDENT |
| Bayesian competitive knowledge transfer (arXiv 2510.23407) | SR | NO | YES (transferability as a latent variable, Bayes' rule) | PARTIAL ? | NO | CONCEPTUAL PRECEDENT |
| Gupta & Ong, *Genetic transfer or population diversification?* (2016) | SR | NO | NO | PARTIAL (an **offline** controlled analysis of transfer vs diversification, not an online decision) | NO | CONCEPTUAL PRECEDENT (for the *idea* behind E3's control) |
| Island EDAs with probability-model migration (2004) | SR | NO (model type fixed) | NO (constant or fitness-adaptive weights) | NO | NO | DISTANT RELATED WORK |
| Multi-representation island models (GMU) | SR | NO | NO | NO | PARTIAL (heterogeneous islands, no evidential role) | DISTANT RELATED WORK |
| Adaptive migration intervals (2015, Evol. Comput.) | SR | NO | NO (improvement heuristic) | PARTIAL (timing, not a statistical test) | NO | DISTANT RELATED WORK |
| AMALGAM / MOS / PAP | SR | NO | NO | NO | NO (heterogeneous, allocation only) | DISTANT RELATED WORK |
| RVIntX / GAwLL / HLX / LEL / VIGPSO | CODE (GAwLL) / SR | NO (commit to linkage) | NO | NO | NO | DISTANT RELATED WORK (for N1) |
| KBS 2023–2025 EMT papers (S0950705123006561, S0950705122011200, S0950705124001655) | SR | NO ? | ? | PARTIAL ? (negative-transfer inhibition by knowledge identification) | NO | to verify |

No paper was classified as DIRECT OVERLAP. Two are CLOSE PARTIAL OVERLAP
(ensemble KT framework with AIE + MAS; MFEA-II).

## 2. Answers to the critical novelty questions

| Q | answer | evidence |
|---|---|---|
| **Q1** Has structural-hypothesis selection itself been studied? | **Yes, in related forms.** Online selection of decomposition strategies (LCC, SR); of coordinate systems (ACoS, SR); of transfer/domain-adaptation strategies between tasks (AIE + MAS, SR); continuous separable-vs-full adaptation (dd-CMA, CODE); structure selection within sampling models (BOA/EBNA, SR). | S7, S5, S16, V2, S22 |
| **Q2** Has statistical/model-evidence selection of structural representations been studied? | **Yes for sampling-model structure** (BIC in BOA/EBNA; graphical-model EDAs) and **for transfer intensity** (MFEA-II likelihood, verified in code). **Not verified** for selecting the *exchange representation* between optimizers. | S22, S23, V4 |
| **Q3** Has explicit suppression of exchange based on a statistical test been studied? | **Partially.** Likelihood-driven transfer reduction (MFEA-II, CODE), outlier/anomaly-based transfer filtering (OKTPO-MFEA, MGAD, SR), Bayesian transferability (SR). **No direct precedent was verified in the accessible literature** for a *sequential null test comparing exchange outcomes with a native-search control at matched step length*. Gupta & Ong perform such a comparison **offline** as an analysis. | V4, S8, S10, S12 |
| **Q4** Has structural evidence been obtained jointly from heterogeneous optimizers? | **No direct precedent was identified in the accessible sources searched.** Heterogeneous optimizers appear only as solution sources (MPHBS, AMALGAM, LH-CC) or islands (multi-representation). This is also the least empirically established element: its premise (operator artefacts differ between HBA and MPA while landscape-induced structure is shared) is untested. | S13, S24, V1 |
| **Q5** Has the combination E1 + E2 + E3 + E4 been demonstrated before? | **No direct precedent was identified in the accessible sources searched.** This is **not** a claim that none exists: the closest candidates are unread in full text. | all |
| **Q6** Has an essentially equivalent method been published under different terminology? | **Possibly, in part.** The strongest candidate for "equivalent under other terms" is multitasking transfer control that adaptively chooses the transfer (domain-adaptation) strategy (AIE + MAS) combined with likelihood-controlled transfer intensity (MFEA-II). Re-expressed, "N1 = adaptive choice of transfer representation + transfer on/off control, in a single-task two-population hybrid". What that reading does not cover is E2 as *model-evidence* selection of the representation, E3 as a *native-controlled sequential* test, and E4. Whether AIE + MAS already uses statistical evidence is unknown (SR only). | S16, V4 |
| **Q7** Would an expert reviewer reasonably say "an existing method with renamed components"? | **Plausibly, yes, for E1–E3 taken separately:** "structural hypotheses" ≈ linkage models / coordinate systems / decompositions; "evidence-based selection" ≈ AOS with a different credit; "null test" ≈ negative-transfer control. The defence rests on E4 and on E3's specific native-control design. Both are defensible *in principle*, but E4 is unvalidated and E3 could be dismissed as an implementation choice of transfer control. | synthesis |

## 3. Why N1 remains promising

* No accessible source combines all four elements, and none uses
  heterogeneous optimizers as **corroborating witnesses** of landscape
  structure (E4).
* A **native-controlled sequential test** of exchange informativeness (E3) was
  not found as an online mechanism. The closest items are an offline analysis
  (Gupta & Ong) and continuous likelihood control (MFEA-II).
* The concept addresses a limitation stated in MPHBS itself (dimension-wise
  reuse under strong coupling), and it has falsifiable, non-performance
  predictions (`N1_CONCEPT.md` §11).

## 4. Exact overlaps that remain unresolved

| # | overlap | element(s) threatened | nature of the uncertainty |
|---|---|---|---|
| U1 | Ensemble KT framework with **AIE + MAS** selecting domain-adaptation strategies online | **E1** (and E2 if its selection uses statistical evidence) | **conceptual overlap + inaccessible full text** |
| U2 | **MFEA-II** likelihood-based transfer control | E2, E3 | conceptual overlap (**verified in code**): likelihood-driven transfer control exists; only the *native-controlled sequential test* and the *representation* object differ |
| U3 | **LCC** decomposition-strategy scheduling | E1 | inaccessible full text: the strategy set, whether "no decomposition" is included, and the decision signal |
| U4 | **ACoS** coordinate-system probability update | E1, E2 | inaccessible full text: whether the update uses success counts only or any likelihood/model fit |
| U5 | Negative-transfer detection in EMT (OKTPO-MFEA, MGAD, Bayesian competitive KT, KBS EMT papers) | E3 | inaccessible full texts: whether any uses a formal hypothesis test with a no-transfer control |
| U6 | GOMEA linkage-model *type* selection (e.g. *Predetermined versus learned linkage models*, later GOMEA work) | E1 + E2 in a single population | the library code shows user-chosen types (verified); the literature beyond the library is unverified |
| U7 | Island / multi-representation models | E4 | full texts unread; snippets suggest no evidential role |

## 5. Specific papers requiring full-text verification (priority order)

1. *An ensemble knowledge transfer framework for evolutionary multi-task
   optimization*, Swarm and Evolutionary Computation (2023), ScienceDirect
   PII S2210650223001670: the AIE and MAS mechanism, and the credit used to
   select domain-adaptation strategies. **Highest priority (U1).**
2. Guo, Qiu, Ma, Zhang, Zhang, Gong, *Advancing CMA-ES with Learning-Based
   Cooperative Coevolution for Scalable Optimization*, arXiv 2504.17578
   (2025) (U3).
3. *An Adaptive Framework to Tune the Coordinate Systems in Nature-Inspired
   Optimization Algorithms* (ACoS), IEEE TCYB (2018),
   doi:10.1109/TCYB.2018.2802912, arXiv 1703.06263; and *An adaptive
   framework to select the coordinate systems for evolutionary algorithms*,
   Applied Soft Computing (2022), S156849462200638X (U4).
4. OKTPO-MFEA (Complex & Intelligent Systems 2025, s40747-025-02220-0); MGAD
   (ESWA 2025, S0957417425012217); *Multi-Task Surrogate-Assisted Search with
   Bayesian Competitive Knowledge Transfer* (arXiv 2510.23407) (U5).
5. KBS: S0950705123006561, S0950705122011200, S0950705124001655 (U5).
6. *Predetermined versus learned linkage models*, GECCO 2012,
   doi:10.1145/2330163.2330205 (U6).
7. LH-CC, GECCO 2026, doi:10.1145/3795095.3805054 (U3/E4).
8. Gupta & Ong, arXiv 1607.05390 (E3's control idea).
9. *Migration of Probability Models Instead of Individuals* (Springer
   978-3-540-30217-9_25); *Improving Evolutionary Algorithms with
   Multi-representation Island Models* (GMU) (U7).

## 6. Missing evidence, and what would resolve it

| gap | cause | resolution |
|---|---|---|
| mechanism details of U1, U3, U4, U5 | **inaccessible full texts** (network policy) | supply PDFs in `reference/` (or run this verification from an unrestricted network); read the method sections |
| whether E1 is already established in single-task hybrids under another name | **search coverage** (snippet-level engine; no Scopus/WoS/Google Scholar citation graph) | forward-citation search from ACoS, MFEA-II, AIE + MAS and LCC; Scopus/WoS query "transfer strategy selection" AND ("multi-population" OR "hybrid") 2018–2026, restricted to KBS / TEVC / Swarm Evol. Comput. / ASOC / Inf. Sci. |
| whether E4's premise holds (HBA and MPA operator artefacts differ, landscape structure is shared) | **not a literature question**: an empirical premise, unverified | a mechanism-only diagnostic on known-structure functions *after* novelty is settled (`N1_CONCEPT.md` §12 S1–S3). **Not run.** |
| whether E3 can be dismissed as an implementation variant of transfer control | **actual conceptual proximity** (MFEA-II, negative-transfer detection) | only argument plus ablation (native-controlled SPRT vs likelihood-intensity control vs success-rate control). A reviewer judgement, not resolvable by search |

**Summary of the nature of the uncertainty.** U2 and the E3 proximity are
**actual conceptual overlap** (verified in code for MFEA-II). U1, U3, U4, U5
and U6 are **mainly inaccessible-paper uncertainty**. E4's validity is an
**empirical** uncertainty.

## 7. Decision on implementation

**Do not implement N1 yet.**

* If full-text checks of U1 (AIE + MAS) and U3/U4 (LCC, ACoS) show that
  transfer or structural representations are already selected **by
  statistical evidence**, N1 should be reclassified **C**. The project would
  then move to **R2** (heterogeneous search processes as FE-free structure
  estimators; `N1_CONCEPT.md` §13), which isolates the one element (E4) with
  no identified precedent as its own research question, **without patching N1**.
* If those checks confirm performance-based selection only (bandit /
  success / DRL), and no native-controlled null test exists in U5, N1 could
  be reconsidered for **A**. The contribution claim would then have to rest
  explicitly on E3 + E4, with E1 + E2 acknowledged as building on AIE + MAS,
  ACoS, LCC, BOA and MFEA-II.
