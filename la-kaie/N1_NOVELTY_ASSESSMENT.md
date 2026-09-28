# N1 novelty assessment — structure-hypothesis selection for information exchange between heterogeneous optimizers

Date: 2026-09-28. Analysis only: no code, experiments or manuscript text.
Companion: `N1_CONCEPT.md` (problem, hypotheses, evidence, test, mapping,
equations, cost, falsifiable predictions, diagnostic benchmark).

## 0. Evidence limits

Full texts remain inaccessible: the network policy blocks arXiv, Springer,
ScienceDirect, IEEE, ACM, Zenodo, ResearchGate and the scholarly APIs, and
only GitHub is reachable. The entries below were assessed from abstracts and
search snippets (**ABS**), except MPHBS (**FT + CODE**) and GAwLL (**CODE**).
"Not found" statements describe this search; they are **not** evidence of
absence. N1 is not called novel merely because no paper with a matching title
turned up.

## 1. Focused searches performed

| query (web search) | most relevant hits |
|---|---|
| "model selection" metaheuristic information exchange | nothing specific (model selection *by* metaheuristics dominates) |
| "structural hypothesis" / "structure selection" + evolutionary / multi-population / adaptive decomposition | decomposition-based MOEAs; overlapping decomposition; nothing on exchange-structure selection |
| "adaptive representation" selection EA | adaptive switching EAs; MAP-based strategy ensembles; *Online Selection of CMA-ES Variants* |
| "linkage model" selection (GOMEA FOS types) | univariate / marginal-product / linkage-tree / full FOS; *Predetermined versus learned linkage models* (GECCO 2012) |
| separable vs full covariance adaptation | **dd-CMA-ES** (*Diagonal Acceleration for CMA-ES*, Evol. Comput. 28(3), 2020) |
| structure learning with an information criterion in EDAs | **BOA / EBNA** (Bayesian-network structure by BIC/MDL); **Gaussian Markov-network EDAs** (graphical-lasso sparsity) |
| adaptive decomposition, learning-based CC | **LCC** (*Advancing CMA-ES with Learning-Based Cooperative Coevolution*, arXiv 2504.17578); **LH-CC** (GECCO 2026, arXiv 2604.01241) |
| "online model selection" / model evidence for surrogates | Bayesian surrogate selection by model evidence; BMA surrogates |
| heterogeneous populations / exchange structure / multi-population structural adaptation | HPCEA (heterogeneous population co-evolution); MPGGA interaction strategies; adaptive inter-subpopulation topology |
| island EDAs | **Migration of probability models instead of individuals** (island-model EDAs); **multi-representation island models** |
| KBS-restricted searches (2023–2026) | QLOBLINFO, RL-LGWO, competitive many-task DE with RL and meta-knowledge transfer, knowledge-transfer DE for nonlinear equation systems, adaptive switching EA (KBS 2022), dynamic multi-knowledge EA |

## 2. Formal novelty table

What each method selects is classified into:
(1) operators, (2) parameters, (3) variable interactions, (4) coordinate
systems, (5) population/task transfer, (6) a structural hypothesis/model itself.

| Existing method | What it selects | Structural hypothesis selection? | Heterogeneous populations? | Exchange representation adaptive? | Difference from N1 | Evidence |
|---|---|---|---|---|---|---|
| **MPHBS** (KBS 2026) | (1)/(5): which (rank, source) component per dimension | **No.** Dimension-wise separability is assumed; FAD placement and K, N_i are fixed per domain offline | yes (HBA, MPA) | no (always dimension-wise) | N1 tests separability against alternatives and can decide that no exchange structure is supported | FT + CODE |
| **BOA / EBNA / hBOA** | (6) within the sampling model: Bayesian-network structure by BIC/MDL | **Yes, but of the generative model of one population** | no | n/a (sampling, not exchange) | Closest *statistical* precedent: information-criterion structure selection. N1 applies structure selection to the *transfer* between two differently biased populations and adds a null "no transferable structure" hypothesis. | ABS |
| **Gaussian Markov-network EDAs** (graphical lasso) | (3)/(6): sparsity pattern of the precision matrix | **Yes (sparse vs dense) within one sampling model** | no | n/a | Same statistical machinery that N1's geometry evidence would use (covariance-structure selection); a different decision object | ABS |
| **GOMEA / RV-GOMEA** (univariate, marginal product, linkage tree, full FOS) | (3)/(6): linkage model (FOS) | **Partly**: FOS type is normally chosen by the designer; *Predetermined versus learned linkage models* compares them offline | no | the FOS governs *mixing* (a form of exchange) inside one population | Closest *conceptual* precedent for "structure governs exchange". Online, evidence-based choice between FOS *types* is not reported in the accessible material (needs full-text check) | ABS |
| **dd-CMA-ES** (Evol. Comput. 2020) | (4)/(6)-like: continuous blend of separable (diagonal) and full adaptation, modulated by the condition number of the correlation matrix | **Implicitly** (separable vs rotated), by a continuous statistic, not by hypothesis selection | no | n/a (sampling distribution) | Direct precedent for exploiting the separable-vs-rotated distinction adaptively. N1 makes it a discrete, testable hypothesis choice for inter-population transfer, with block and null hypotheses | ABS |
| **Online selection of CMA-ES variants** | (1): algorithm variant (possibly incl. separable/full) | no (performance-based algorithm selection) | no | no | selects algorithms by performance or features, not structural models by evidence | ABS |
| **ACoS** (IEEE TCYB 2018) | (4): original vs eigen coordinate system, probability vector updated from offspring | **Partly**: two structures (axis-aligned vs rotated), selected by **offspring success** | no | yes (coordinate frame of crossover) | N1 selects among ≥3 transfer structures + null, by **model evidence** from population geometry and transfer outcomes; inter-population | ABS |
| **DE/eig, CoBiDE, CPI-DE** | (4): eigen frame (random or fixed ratio) | no | no | partly (random frame) | no selection criterion | ABS |
| **MCDE** (multi-population covariance learning DE) | (4): crossover frame from covariance between subpopulations | no | subpopulations of the same DE | fixed eigen frame | no hypothesis selection; homogeneous subpopulations | ABS |
| **HLX** (Inf. Sci. 2015) | (3): linkage matrix via DG → group-wise crossover | no (assumes linkage) | no | group-wise (fixed) | commits to one hypothesis; learns it with extra FEs | ABS |
| **RVIntX / GAwLL** (BRACIS 2025 / GECCO 2023) | (3): empirical VIG → recombination masks | no (assumes linkage) | no | masks from VIG (fixed rule) | commits to linkage; learns it by constructed perturbations (verified in GAwLL code) | ABS + CODE |
| **LEL** (arXiv 2604.04273) | (3): sparse interaction graph + ordering → interval subsets | no | n/s | interval subsets (fixed rule) | commits to (serializable) sparse structure | ABS |
| **LCC** (arXiv 2504.17578) | (6)-like: **decomposition strategy** scheduled by a DRL agent from optimisation-status features | **Partly**: selects among decomposition strategies, i.e. structural treatments of the problem, by **learned policy / performance reward** | no (CC subcomponents, one optimizer) | the decomposition governs how subcomponents are optimised, not inter-population exchange | **Closest in spirit** ("choose the structural treatment online"). Differs in the decision object (decomposition for CC vs transfer representation between heterogeneous optimizers), the criterion (DRL reward vs statistical evidence) and the null hypothesis. **Full text needed.** | ABS |
| **LH-CC** (GECCO 2026, arXiv 2604.01241) | (1): optimizer per subproblem | no | yes (heterogeneous optimizers per subproblem) | no | selects optimizers, not exchange structure | ABS |
| **CBCC family** | (5)/(2): FE allocation by contribution | no | no | no | allocation only | ABS |
| **MFEA-II** (TEVC 2019) | (5): transfer intensity (RMP matrix) by mixture-model likelihood | **No** (transfer amount, not structure) | one population, multiple tasks | no (whole-solution crossover) | Likelihood-based *transfer control* precedent; N1 selects transfer *structure* within one task | ABS |
| **Island EDAs with migration of probability models** | (5): exchange models instead of individuals; combination weights (constant or fitness-adaptive) | no (model *type* fixed) | islands of the same EDA | exchange *medium* is a model, fixed | exchanges structural models but does not *select* the structural hypothesis | ABS |
| **Multi-representation island models** | (5): islands with different representations exchanging individuals | no | yes (different representations) | fixed per island | representation diversity, no selection by evidence | ABS |
| **AMALGAM / MOS** | (1)/(5): offspring share per algorithm | no | yes | no | allocation by reproductive success | ABS |
| **AOS: extreme-value DMAB, FRRMAB, LA-BHH** | (1) | no | no | no | performance credit over operators | ABS |
| **Adaptive migration intervals** (Evol. Comput. 2015) | (5): when to migrate | no | islands | no | timing only; no structure; no statistical null test of exchange utility | ABS |
| **KBS: knowledge-transfer adaptive DE for NES** (S0950705122013107) | (5): transfer intensity between two niching populations from diversity variation | no | two niching techniques (crowding, speciation) | no | adaptive intensity, not structure | ABS |
| **KBS: competitive many-task DE with RL + meta-knowledge transfer** (S0950705125009761) | (1)/(5) via RL | no | task populations | n/s | RL-controlled transfer between tasks | ABS |
| **KBS: QLOBLINFO (2025), RL-LGWO (2025)** | (1)/(2) | no | no | no | RL operator/leader control | ABS |
| **Bayesian surrogate model selection by evidence** | surrogate model class | yes, for a *surrogate of f* | no | n/a | same evidence principle; a different decision object | ABS |

## 3. Classification of the closest methods against N1's defining elements

N1's defining elements:
(E1) the decision object is the **structural hypothesis governing inter-population transfer**;
(E2) the selection is by **statistical evidence** (information criteria / predictive likelihood / sequential tests), not by operator-performance credit;
(E3) there is an explicit **null hypothesis**, "no transferable structure", which suppresses exchange;
(E4) the exchange is between **heterogeneous optimizers**, whose distinct search biases are used as **independent witnesses** of structure (cross-population corroboration).

| element | status | precedents |
|---|---|---|
| E1 | **CONCEPTUAL PRECEDENT**: structure selection exists for sampling models (BOA, GGM-EDA, dd-CMA), linkage models for mixing (GOMEA FOS; offline), decomposition for CC (LCC, online by DRL), and frames for crossover (ACoS, online by success) | not found for inter-population transfer between heterogeneous optimizers (search-level statement) |
| E2 | **CONCEPTUAL PRECEDENT**: information criteria for structure (BOA BIC), likelihood-based transfer control (MFEA-II), evidence-based surrogate selection | not found as the criterion for choosing an *exchange representation* |
| E3 | **PARTIAL OVERLAP**: adaptive migration intervals (when to exchange), MFEA-II (reduce transfer if harmful), allocation methods | a *statistical null-hypothesis test of exchange informativeness against a native-search control* was not found |
| E4 | **Not found** in the accessible material | heterogeneous optimizers exist (AMALGAM, MOS, LH-CC, MPHBS), but none uses them as corroborating witnesses for structural evidence |

## 4. Honest verdict

* The **principle** "select a structural model by statistical evidence" is
  **not novel**: it is standard in model-based EAs (BOA/EBNA BIC, graphical
  EDAs) and implicit in dd-CMA-ES. "Select a structural treatment online" also
  exists (LCC by DRL; ACoS by success).
* What is **not found** is the combination E1 + E2 + E3 + E4. The two
  elements most likely to give a genuine, defensible contribution are **E3**
  (a sequential test of whether cross-population information is transferable
  at all, against a no-extra-FE native control) and **E4** (using the
  *heterogeneity* of the two optimizers as independent evidence, so
  operator-induced artefacts are not mistaken for landscape structure).
  Without E3 and E4, N1 reduces to "BOA/ACoS-style structure selection moved
  to the exchange layer" and would be **incremental**.
* A reviewer familiar with EDAs will ask why the exchange layer needs its own
  structure selection when a model-based optimizer would learn the structure
  anyway. The answer must be empirical and mechanistic: heterogeneous,
  model-free optimizers (HBA, MPA) carry no structural model, and MPHBS's own
  stated limitation is exactly this.

## 5. Decision

**Classification: B — promising but requires another literature check.**

What must be verified (full texts required) before development:

1. **LCC** (arXiv 2504.17578) and **LH-CC** (arXiv 2604.01241): the set of
   decomposition strategies, whether "no decomposition"/rotation-invariant
   treatment is an option, and whether any statistical test (rather than DRL)
   drives the choice. If LCC selects structural treatments by statistical
   evidence, E1 + E2 collapse.
2. **ACoS** (TCYB 2018, arXiv 1703.06263): the exact update of the
   coordinate-system probability vector. If it uses a likelihood or model fit
   rather than offspring success, E2 is weakened.
3. **GOMEA line**: whether any work (e.g. *Predetermined versus learned
   linkage models*, parameterless GOMEA, RV-GOMEA successors) selects the FOS
   *type* (univariate / marginal product / full) **online by evidence**. If
   so, E1 + E2 overlap directly, although not E3/E4.
4. **dd-CMA-ES**: the statistic controlling the diagonal learning rate, to
   state precisely how N1's discrete hypothesis choice differs.
5. **Transfer-or-not tests in evolutionary multitasking**: works that
   *statistically* decide to suppress transfer (beyond MFEA-II's RMP learning,
   e.g. explicit negative-transfer detection). This directly threatens E3.
6. **Island / multi-population EDAs** (model migration; multi-representation
   islands): whether any selects the structural class of the exchanged
   model. This threatens E1/E4.
7. **KBS 2023–2026** with full-text search access, restricted to
   multi-population / hybrid metaheuristics with transfer-control mechanisms.

If items 1, 3 or 5 reveal direct overlap, N1 should be **downgraded to C**.
In that case a *different research question*, not a patched N1, should be
pursued; candidates are listed in `N1_CONCEPT.md` §13.
