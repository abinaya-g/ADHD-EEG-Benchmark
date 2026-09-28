# Gate 0: literature verification (N1)

Date: 2026-09-28. Scope: spec §16, Gate 0.

This document records verification work only. It contains no
implementation and no manuscript text.

## 1. Outcome

**Gate 0 is NOT resolved.**
- **Full-text verification:** impossible for every outstanding item, because
  the environment's egress policy blocks every publisher, preprint and
  repository host that was tried (§2).
- **Code verification:** possible through GitHub (git protocol) and PyPI.
  It added substantive **new overlap evidence** (§4).
- **Remaining items:** these rest on search-engine snippets only and stay
  **unresolved evidence gaps** (§5).
- **What the spec allows:** Gate 0 can close only in one of two ways:
  1. the remaining gaps are closed from full text; or
  2. the user explicitly accepts them as unresolved.

**Novelty classification: left at B.** It is not upgraded. The new code
evidence **narrows E3 substantially** (§6). It does not show a direct
duplicate of N1 as a whole, so I have not reclassified. Whether the
narrowing warrants a reclassification is a decision for the user.

## 2. Access conditions (this run)

| channel | result |
|---|---|
| `curl` via the egress proxy to arxiv.org, export.arxiv.org, semanticscholar.org, api.openalex.org, api.crossref.org, europepmc, core.ac.uk, scholar.archive.org, web.archive.org, openreview.net, dl.acm.org, sciencedirect.com, link.springer.com, ieeexplore.ieee.org, mdpi.com, pmc.ncbi.nlm.nih.gov, api.unpaywall.org, huggingface.co, researchgate.net, paperswithcode.com, r.jina.ai | **blocked**: `CONNECT tunnel failed, response 403` (organisation egress policy) |
| WebFetch to arxiv.org, sciencedirect.com, pith.science, awesomepapers.io, lacuna.tiptreesystems.com, homepages.cwi.nl, oa.upm.es | **blocked** (`EGRESS_BLOCKED`) |
| Web search engine (titles and snippets) | available; snippets are **leads, not evidence** |
| GitHub via git (clone / ls-remote) | available |
| PyPI (`pip download`) | available |

## 3. Method

Two routes were used:
- For each outstanding item: look for (a) full text and (b) author or
  third-party code, and inspect any code found.
- For "statistical transfer suppression" and KBS 2023–2026 (topic areas
  rather than single papers): search a large, maintained EMT code platform
  for mechanisms that match N1's claimed elements.

N1's claimed elements are:
- **E3:** activation/suppression of exchange relative to a native-search
  control.
- **E4:** selection of the exchange representation (HI/HB/HR) from joint
  structural evidence of heterogeneous optimizers, by BIC.

Code sources:
- **MTO-Platform**, `github.com/intLyc/MTO-Platform`, commit `3ca17b2`
  (2026-09-23), a MATLAB platform for evolutionary multitask optimisation.
  - **Caveat:** its implementations are written by the platform
    maintainers. They are **not necessarily the original authors' code**,
    so they are evidence of the mechanism *as reimplemented*. Details may
    differ from the papers.
- **metaevobox 2.0.2** (PyPI; MetaBox 2.0 MetaBBO library).
- **pypop7** (`github.com/Evolutionary-Intelligence/pypop`).

## 4. New code-level evidence

All file paths are relative to `MTO/Algorithms/` in MTO-Platform `3ca17b2`.

| # | work (citation from the code header) | location | mechanism found | overlap with N1 |
|---|---|---|---|---|
| G1 | **AEMTO**: Xu, Qin, Xia, *Evolutionary Multi-Task Optimization with Adaptive Knowledge Transfer*, IEEE TEVC 2022, doi 10.1109/TEVC.2021.3107435 (stream variant AAEMTO) | `Stream-task/AAEMTO/AAEMTO.m` lines ~149–177 (`acceptBatch`) | transfer probability `p = tsf_lb + Q_other/(Q_other + Q_self)·(tsf_ub − tsf_lb)`, with `tsf_lb` = 0.05 and `tsf_ub` = 0.7. Here Q_self is the exponentially smoothed survival rate of **self-evolution (native) offspring** and Q_other that of **transfer offspring**. Source selection is by smoothed transfer success (probability matching with floor) | **High for E3.** Exchange frequency is regulated by comparing exchange success against native-search success. That is the *concept* of E3. **Differences:** a continuous ratio rule, not a sequential test; no step-length-matched control model; never fully suppressed (lb 0.05); multitask, not heterogeneous optimizers on one task; no structural representation choice |
| G2 | **MMTO-ETA**: Zhang, Huang, Wu, Li, Liu, Li, Gong, *Ensemble of Transfer Techniques via Adaptive Resource Allocation for Multiobjective Multitask Optimization*, IEEE TAI 2026, doi 10.1109/TAI.2026.3689513 | `Multi-objective Multi-task/Multi-population/MMTO-ETA/MMTO_ETA.m` lines ~55–95 (roulette over 3 strategies; reward = offspring survival rate), ~159–190 (strategies), `LPCA` (local PCA per cluster) | three strategies are chosen per offspring by roulette with smoothed survival-rate reward: (1) **eigen-subspace mapping** between tasks, using local-PCA eigenvectors; (2) transfer of a model mean; (3) **no transfer**, plain DE | **High for the E4 mechanism class and for A10.** Adaptive selection among transfer representations, including an eigenbasis representation and a no-transfer option, driven by success rates. **Differences:** success-rate roulette rather than likelihood/BIC over native displacements; no HI/HB/HR hypothesis set; no joint heterogeneous-optimizer evidence; multiobjective multitask |
| G3 | **SSLT**: Yuan, Dai, Peng, Wang, Song, Chen, *Scenario-based self-learning transfer framework for multi-task optimization problems*, **Knowledge-Based Systems** 2025, doi 10.1016/j.knosys.2025.113824 | `Multi-task/Multi-population/SSLT/SSLT_DE.m` lines ~88–180 | a DQN chooses among 4 actions: **no KT**, shape KT, bi-KT, domain KT. The state is convergence rates, Wasserstein distance between populations, dispersion type and phase; the reward is improvement-based | **Medium–high.** A KBS 2025 precedent for adaptively choosing between no transfer and several transfer forms. **Differences:** learned policy (RL), not statistical evidence; no structural (covariance/partition) hypotheses; multitask |
| G4 | **AMT**: Da, Gupta, Ong, *Curbing Negative Influences Online for Seamless Transfer Evolutionary Optimization*, IEEE TCYB 2019, doi 10.1109/TCYB.2018.2864345 | `Multi-objective Multi-task/Multi-population/AMT-NSGA-II/MixtureModel.m` (`createtable` with leave-one-out target model; `EMstacking`); `ProbabilityModel.m` line ~52 (diagonal covariance) | the mixture of source models plus the target's own model is fitted by EM on the target population's likelihood. The target-model weight can absorb all mass, which **suppresses transfer**. Diagonal Gaussians only | **Medium for E3** (likelihood-based negative-transfer suppression with the target's own model as control). **Differences:** density-mixture weights, not a sequential outcome test; no structural hypothesis selection (a fixed diagonal model) |
| G5 | **MTEA-AD**: Wang, Liu, Wu, Wu, *Solving Multi-task Optimization Problems with Adaptive Knowledge Transfer via Anomaly Detection*, IEEE TEVC 2022, doi 10.1109/TEVC.2021.3068157 | `Multi-task/Multi-population/MTEA-AD/MTEA_AD.m` lines ~60–88, 130–150 | a full-covariance Gaussian is fitted on the target offspring; candidates from other tasks are ranked by density. The fraction transferred is set to the previous transfer success rate `epsilon` | **Medium.** Full-covariance Gaussian filtering of transferred individuals, adapted by success. **Differences:** no model-structure selection; no native-control test |
| G6 | **BLKT-DE**: Jiang, Zhan, Tan, Zhang, *Block-Level Knowledge Transfer for Evolutionary Multitask Optimization*, IEEE TCYB 2024, doi 10.1109/TCYB.2023.3273625 | `Multi-task/Multi-population/BLKT-DE/BLKT_DE.m` lines ~45–118 | dimensions are cut into **consecutive blocks of random size** `divD`. Block vectors from all tasks are clustered by **k-means on values**, and DE runs within clusters. `divD` and the cluster count are re-randomised or perturbed on stagnation | **Low–medium for HB.** Block-level transfer exists, but blocks are contiguous and random-sized, not a statistically estimated partition, and there is no hypothesis selection |
| G7 | *(checked, no match)* metaevobox 2.0.2 and pypop7 | whole packages | no LCC, LH-CC or ACoS implementation present | — |
| — | earlier code verifications, still valid (see `N1_NOVELTY_EVIDENCE_LOG.md` V1–V5) | — | MPHBS; dd-CMA (`beta` continuous modulation); GOMEA library (user-chosen linkage type; no BIC between types); MFEA-II (likelihood-fitted RMP); GAwLL | as recorded |

## 5. Item-by-item Gate 0 status

Verification level: FULL = full text; CODE = code inspected; SNIPPET =
search-engine abstract or snippet only.

| outstanding item (spec §16) | level reached | what is known | status |
|---|---|---|---|
| **Ensemble KT framework, AIE + MAS** (Zhou, Rao, Gao; Swarm Evol. Comput. 2023; doi 10.1016/j.swevo.2023.101394) | SNIPPET | three domain-adaptation models (DAE, RBM, ADM) chosen by a bandit mechanism; AIE adjusts the transfer probability and the number of migrated individuals "according to historical search experience"; controls "self-focused refined search or cross-task information exchange" | **UNRESOLVED.** No full text, and no code found (not in MTO-Platform). Whether AIE compares against native search, as AEMTO does (G1), is **unknown** |
| related multitasking transfer-selection methods | CODE (G1–G6) | see §4 | **resolved as overlap evidence** (third-party code); not exhaustive |
| **LCC** (arXiv 2504.17578) | SNIPPET | PPO selects among random, min-variance and max-variance splitting for CC-CMA-ES; 58-dimensional state including "variable interactions within current subgroups"; no extra FE for decomposition | **UNRESOLVED.** No code found; the full text is blocked |
| **LH-CC** (GECCO 2026, doi 10.1145/3795095.3805054) | SNIPPET | MDP meta-agent selects the optimizer per subproblem for heterogeneous LSGO | **UNRESOLVED** |
| **ACoS** (TCYB, doi 10.1109/tcyb.2018.2802912) and its ASOC 2022 follow-up | SNIPPET | eigen vs original coordinate system chosen per individual by a probability vector updated from offspring information | **UNRESOLVED.** No code found. A10 is already named neutrally (spec §11) |
| **GOMEA / FOS structural selection** | CODE (library, V3) + SNIPPET (paper) | library: linkage type chosen by the user; GECCO 2012 paper: *offline experimental comparison* of predetermined vs learned models, not online selection | **Partly resolved** (code); **paper full text UNRESOLVED** |
| **dd-CMA-ES** | CODE (V2) | continuous β modulation | **Partly resolved** (code); **paper full text UNRESOLVED** |
| **Statistical transfer suppression** | CODE (G1, G4, G5; earlier MFEA-II) + SNIPPET | no SPRT-type or sequential-test transfer control found in any code or snippet | **Partly resolved.** OKTPO-MFEA, MGAD and Bayesian competitive KT (arXiv 2510.23407) are **UNRESOLVED** |
| **Island / multi-population EDA model migration** | SNIPPET | delaOssa, Gámez, Puerta (2004); Madera, Alba, Ochoa (2006); UPM report on the "exchanged information" in distributed EDAs | **UNRESOLVED** |
| **KBS 2023–2026** | CODE (G3, SSLT, KBS 2025) + SNIPPET | SSLT is a KBS precedent for choosing no-transfer vs transfer forms. The other KBS items in evidence-log S14 have not been checked | **Partly resolved; UNRESOLVED for the remainder** |

## 6. Impact on N1's claims

This is an assessment, not a novelty claim.

**E3 (exchange informativeness relative to a native control).**
- The *concept* of regulating exchange by comparing exchange outcomes with
  native-search outcomes is **precedented in code**: AEMTO (G1), and
  likelihood-based self-model suppression in AMT (G4) and MFEA-II.
- Explicit **no-transfer actions** chosen adaptively are also precedented:
  MMTO-ETA (G2) and SSLT (G3, KBS 2025).
- What remains *not found* in accessible evidence is narrower:
  - a **sequential likelihood-ratio decision** with full suppression and
    probing;
  - a **step-length-matched logistic native control**;
  - heterogeneous optimizers on **one** task.
- These are differences in *statistical form*. Their significance must be
  argued, not assumed.

**E4 (joint heterogeneous-optimizer structural evidence).**
- Adaptive selection among transfer **representations**, including
  eigenbasis-based transfer and no transfer, is precedented: MMTO-ETA (G2),
  ACoS and LCC (snippets), SSLT, and AIE + MAS (snippet). These methods use
  **success rates, bandits or RL**.
- No accessible code or snippet uses **likelihood/BIC over native
  successful displacements**, and none compares **joint vs pooled vs
  single-optimizer** structural evidence.
- **This is absence of evidence under restricted access, not evidence of
  absence.** The full texts of LCC, ACoS and AIE + MAS remain unverified.

**Consequence for the spec (recommendations only; the spec is unchanged).**
- **R1.** A10 is a probability-matching success-rate control. It is now
  known to resemble MMTO-ETA's allocation rule (G2). The independent
  reviewer may consider citing that correspondence in the A10 description.
  This is a wording change only.
- **R2.** A8 (JOINT without H0) isolates the *effect* of H0. It does not
  compare the SPRT-style rule against the **existing** native-referenced
  frequency rule (AEMTO, G1). An AEMTO-style rule as an additional E3
  comparator would directly test whether the statistical form matters.
  - Adding it would be a **new baseline**, which the user's previous
    instructions forbid without approval.
  - Adding it is **the user's decision** and has not been done.

## 7. What is needed to close Gate 0

Either of the following:
1. **Full text for the unresolved items.** Options:
   - The user places the PDFs in `reference/` (list below).
   - The user allows the relevant hosts in the environment's network
     settings: at least arxiv.org, www.sciencedirect.com, dl.acm.org,
     ieeexplore.ieee.org, link.springer.com.
2. **Explicit user acceptance** of the remaining gaps as unresolved, with
   the E3 narrowing in §6 acknowledged.

PDFs that would close the most important gaps, in priority order:
1. Zhou, Rao, Gao, *An ensemble knowledge transfer framework for
   evolutionary multi-task optimization*, SWEVO 2023,
   doi 10.1016/j.swevo.2023.101394. This is the closest unresolved
   overlap for both E3 and E4.
2. LCC, arXiv 2504.17578.
3. ACoS, doi 10.1109/tcyb.2018.2802912 (arXiv 1703.06263).
4. Xu, Qin, Xia, AEMTO, doi 10.1109/TEVC.2021.3107435. It is
   code-verified, but full text is needed to confirm that the platform
   reimplementation matches the paper.
5. LH-CC, doi 10.1145/3795095.3805054 (arXiv 2604.01241).
6. OKTPO-MFEA (s40747-025-02220-0), MGAD (S0957417425012217), and
   arXiv 2510.23407.
7. delaOssa et al. 2004 (doi 10.1007/978-3-540-30217-9_25).
8. The GECCO 2012 linkage paper (doi 10.1145/2330163.2330205).
9. The dd-CMA paper (EC 28(3), 2020).

## 8. Statements

- No N1 code was written.
- No experiments were run.
- No manuscript text was written.
- The code inspected above is third-party code, read for evidence only. No
  third-party code was copied into the project.
