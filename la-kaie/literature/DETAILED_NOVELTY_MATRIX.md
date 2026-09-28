# Detailed novelty matrix (2026-09-28)

## Evidence levels (read first)

This environment's network policy blocks arXiv, Springer, ScienceDirect
full text, IEEE Xplore, ACM DL, Zenodo, ResearchGate, OpenAlex, Semantic
Scholar and Crossref. Only GitHub (git protocol) and a web search engine are
reachable. Every row is therefore tagged:

| tag | meaning |
|---|---|
| **FT** | full text read (paper and/or authors' source code) |
| **CODE** | authors' public source code read (mechanism verified) |
| **ABS** | abstract / search-result snippets only; mechanism details may be incomplete |

"n/s" = not stated in the accessible material. That is **not** evidence that
the feature is absent. Classification labels, relative to each candidate:
**DIRECT OVERLAP** (the same mechanism for the same purpose),
**PARTIAL OVERLAP** (the same key idea, a different purpose, estimator or
setting), **CONCEPTUAL PRECEDENT** (the idea exists in a related form),
**DISTINCT**. A different implementation is never counted as novelty.

Candidates: **F1** OB-CEL (co-transfer epistasis learned from exchange
outcomes), **F2** FACE (frame-adaptive cross-population exchange),
**F3** PEA (progress-aligned evaluation allocation), **N1**
structure-hypothesis selection for exchange (proposed in
`NEXT_METHOD_DECISION.md` §6–7).

## A. Reference method

| Paper | Year | Algorithm | Population structure | Information source | Learning mechanism | Interaction learning | Exchange granularity | Exchange timing | Evaluation allocation | Reward / credit | Cross-population exchange | Heterogeneous optimizers | Evidence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Flores & Olivares, *MPHBS*, Knowledge-Based Systems 352:117070 | 2026 | MPHBS | 2 populations (HBA, MPA) | ranked subsets of both populations | SARSA-style TD on Q(dim, rank, source); proxy screening of K candidates | none (dimension-wise; coupling acknowledged as a threat) | single variables within active blocks | every iteration (fixed) | fixed E = N_sub·N_i per iteration, domain-tuned | ternary sign of improvement over the current assembled vector | yes | yes | FT + CODE |

## B. Linkage / interaction learning (relevant to F1, N1)

| Paper | Year | Algorithm | Population structure | Information source | Learning mechanism | Interaction learning | Exchange granularity | Exchange timing | Evaluation allocation | Reward / credit | Cross-pop. | Heterog. | Evidence | F1 | N1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Tinós, Chicano, Przewozniczek, *Variable Importance and Interaction in DE: Online Learning and a New Recombination Operator* (BRACIS 2025, LNCS; replication pkg Zenodo 15782242) | 2025 | DE + online LL + RVIntX | 1 population | evaluated solutions during the search | online empirical linkage learning (details n/s; see GAwLL row for the authors' mechanism) | yes: variable importance + interaction graph | masks from the epistatic structure + variables common to both parents | every recombination | n/s | n/s | no | no | ABS (+ precursor CODE) | **DIRECT/PARTIAL** (same target quantity; see §D) | CONCEPTUAL PRECEDENT |
| Tinós, Przewozniczek, Whitley, Chicano, *Genetic Algorithm with Linkage Learning* (GECCO '23), code github.com/rtinos/GAwLL | 2023 | GAwLL | 1 population | specially constructed mutation offspring x_g, x_h, x_gh of a parent x (evaluated offspring enter the population) | edge (g,h) added when abs(f(x_gh) − f(x_h) − f(x_g) + f(x)) > ε, weights averaged | yes, pairwise empirical VIG, online | used by the GA (feature-selection application) | each generation (fraction of offspring) | a fixed fraction of offspring are LL triplets | n/a | no | no | **CODE** (`gawll.cpp: mutationLL`) | **PARTIAL** (same pairwise non-additivity quantity; different signal source) | CONCEPTUAL PRECEDENT |
| Tinós, Przewozniczek, Whitley, Chicano, *Iterated Local Search with Linkage Learning* (ACM TELO) | 2024 | ILS+LL | single solution | local-search moves | empirical VIG "as a side-effect" of ILS | yes (pairwise) | n/a | online | n/s | n/a | no | no | ABS | PARTIAL | CONCEPTUAL PRECEDENT |
| E. Unlu, *Loop-Extrusion Linkage: Spectral Ordering and Interval-Based Structure Discovery for Continuous Optimization* (arXiv 2604.04273) | 2026 | LEL wrapper | n/s | **successful optimisation steps** | online sparse interaction graph + Fiedler-vector seriation + interval subsets | yes (sparse graph) | overlapping interval subsets | online | n/s | success-based | n/s | n/s | ABS | **PARTIAL, possibly DIRECT** (outcome-based interaction learning; estimator unknown) | CONCEPTUAL PRECEDENT |
| *Using Variable Interaction Graphs to Improve PSO* (VIGPSO, GECCO 2025; arXiv 2509.06985) | 2025 | VIGPSO | 1 swarm | PSO position/velocity updates | Pearson correlation of per-dimension updates | yes (graph) | continuous trajectory modification | every iteration | none extra | n/a | no | no | ABS | PARTIAL | CONCEPTUAL PRECEDENT |
| *Exploiting linkage information … RV-GOMEA* (GECCO 2017); *conditional linkage models* (GECCO 2020); *Fitness-based linkage learning … RV-GOMEA* (IEEE TEVC 2021; arXiv 2402.10757) | 2017–2024 | RV-GOMEA | 1 population | population statistics or fitness-based perturbations | linkage model (FOS); fitness-based LL uses extra evaluations | yes | FOS elements: singletons and multivariate sets | each generation | fitness-based LL costs FEs | n/a | no | no | ABS | PARTIAL (group mixing driven by linkage) | CONCEPTUAL PRECEDENT (fixed linkage hypothesis) |
| *Differential evolution with hybrid linkage crossover* (HLX), Information Sciences | 2015 | DE + HLX | 1 population | perturbations (improved differential grouping) | linkage matrix from DG | yes (pairwise, perturbation-based) | group-wise crossover (GbinX, GorthX) | adaptive | DG evaluations | n/s | no | no | ABS | PARTIAL | CONCEPTUAL PRECEDENT |
| *Linkage identification based on epistasis measures* (LIEM, IEEE) | ~2002 | LIEM | 1 population (binary) | perturbation pairs | pairwise epistasis measure | yes | linkage groups | n/s | extra evaluations | n/a | no | no | ABS | PARTIAL | CONCEPTUAL PRECEDENT |
| Differential grouping family (DG, DG2, extended/enhanced DG; CC) | 2014–2024 | DG + CC | subcomponents | perturbations | nonlinearity check | yes | decomposition (not exchange) | before/during the search | extra FEs | n/a | no | no | ABS | CONCEPTUAL PRECEDENT | CONCEPTUAL PRECEDENT |
| LTGA / GOMEA (mutual-information linkage tree) | 2010– | LTGA/GOMEA | 1 population | population MI | UPGMA linkage tree | yes | FOS (all granularities) | each generation | none | n/a | no | no | ABS | CONCEPTUAL PRECEDENT | CONCEPTUAL PRECEDENT |

## C. Coordinate-frame / eigenbasis operators (relevant to F2, N1)

| Paper | Year | Algorithm | Population structure | Information source | Learning mechanism | Interaction learning | Exchange granularity | Exchange timing | Evaluation allocation | Reward / credit | Cross-pop. | Heterog. | Evidence | F2 | N1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Guo & Yang, *Enhancing DE Utilizing Eigenvector-Based Crossover Operator* (IEEE TEVC) | ~2015 | DE/eig | 1 | population covariance | eigendecomposition; raw/eigen frame chosen at random | implicit (rotation) | crossover in eigen coordinates | every generation | none | n/a | no | no | ABS | **DIRECT** (eigen-frame crossover) | PARTIAL |
| *DE based on covariance matrix learning and bimodal distribution parameter setting* (CoBiDE, Applied Soft Computing) | 2014 | CoBiDE | 1 | top-ps% covariance | eigen-frame crossover (fixed ratio) | implicit | eigen crossover | every generation | none | n/a | no | no | ABS | DIRECT | PARTIAL |
| *Utilizing cumulative population distribution information in DE* (CPI-DE, Applied Soft Computing) | 2016 | CPI-DE | 1 | cumulative distribution (CMA-like) | eigen + original coordinate crossover synergy | implicit | both frames | every generation | none | n/a | no | no | ABS | DIRECT | PARTIAL |
| *An Adaptive Framework to Tune the Coordinate Systems in Nature-Inspired Optimization Algorithms* (ACoS, IEEE TCYB; arXiv 1703.06263) | 2018 | ACoS (DE, PSO) | 1 | cumulative covariance + archive | **selection probability between the original and eigen coordinate system, adapted from offspring information** | implicit | frame choice per individual | every generation | none | offspring success | no | no | ABS | **DIRECT** (adaptive frame selection) | **PARTIAL** (adaptive representation choice by performance) |
| *Multi-populations Covariance Learning DE* (MCDE, JEIT) | ~2019 | MCDE | multiple subpopulations | covariance **between populations** | rotation frame for crossover | implicit | eigen crossover | every generation | none | n/a | partially (shared frame) | same DE family | ABS | **DIRECT** (multi-population shared frame) | PARTIAL |
| *Eigenvector crossover in the efficient jSO* (MENDEL); *efficient eigenvector-based crossover via rank-one updates* (AIMS Math) | 2019–2025 | jSO-eig etc. | 1 | covariance | eigen crossover | implicit | eigen | every generation | none | n/a | no | no | ABS | DIRECT | CONCEPTUAL PRECEDENT |
| EMT subspace alignment / affine transformation (AT-MFEA; OCAT; SETA-MFEA) | 2020–2024 | EMT | one population per task | per-task distributions | PCA subspaces + alignment | implicit | transformed solutions | inter-task transfer | n/s | n/s | yes (between tasks) | n/s | ABS | PARTIAL | CONCEPTUAL PRECEDENT |
| *MODE with dynamic covariance matrix learning for problems with variable linkages*, Knowledge-Based Systems | 2017 | MODE-DCML | 1 | covariance | eigen crossover for linkage | implicit | eigen | every generation | none | n/a | no | no | ABS | DIRECT | CONCEPTUAL PRECEDENT |

## D. Resource / evaluation allocation and credit (relevant to F3, N1)

| Paper | Year | Algorithm | Population structure | Information source | Learning mechanism | Interaction learning | Exchange granularity | Exchange timing | Evaluation allocation | Reward / credit | Cross-pop. | Heterog. | Evidence | F3 | N1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| *Improved evolutionary optimization from genetically adaptive multimethod search* (AMALGAM, PNAS) | 2007 | AMALGAM | shared population, multiple operators | offspring survival | reproductive success sets offspring counts | no | whole offspring | each generation | **adaptive offspring allocation among algorithms** | reproductive success | shared population | yes | ABS | **DIRECT** (adaptive allocation among heterogeneous search processes) | CONCEPTUAL PRECEDENT |
| Multiple Offspring Sampling (MOS), LaTorre et al. | 2009–2013 | MOS | shared, multiple reproductive techniques | quality measures | dynamic participation ratios | no | whole offspring | each step | adaptive | quality of offspring | yes | yes | ABS | DIRECT | CONCEPTUAL PRECEDENT |
| *Population-based algorithm portfolios* (PAP), Peng, Tang, Chen, Yao | 2010 | PAP | subpopulations per algorithm | migration | time budget split + migration | no | individuals | periodic migration | split budget (fixed in the original; adaptive in later work) | n/s | yes | yes | ABS | PARTIAL | CONCEPTUAL PRECEDENT |
| Contribution-based cooperative coevolution (Omidvar et al.; CBCC3; CCFR3; bandit-based CC) | 2011–2022 | CBCC | subcomponents | **contribution to the best overall objective value** | allocate FEs by contribution | via decomposition | subcomponents | adaptive | **adaptive by contribution** | improvement of global best | n/a | no | ABS | **DIRECT** (best-progress credit for FE allocation) | CONCEPTUAL PRECEDENT |
| Memetic algorithms with adaptive local-search intensity (MA-LS-Chains; adaptive LS depth) | 2010–2014 | MA-LS-Chains | EA + LS | per-individual LS history | LS intensity allocation | no | n/a | adaptive | global vs local budget split | improvement | n/a | yes (EA + LS) | ABS | PARTIAL | CONCEPTUAL PRECEDENT |
| *Design and analysis of schemes for adapting migration intervals in parallel EAs* (Evol. Comput. 23(4)) | 2015 | adaptive islands | islands | improvement of best | interval adaptation | no | individuals | **adaptive timing** | communication | best-fitness improvement | yes | no | ABS | PARTIAL (timing) | CONCEPTUAL PRECEDENT |
| Extreme-value credit + dynamic MAB AOS; FRRMAB | 2008–2014 | AOS | 1 | operator improvements | bandit (UCB), Page–Hinkley | no | operator | per offspring | operator shares | extreme / rank-based fitness-improvement rate | no | no | ABS | PARTIAL (credit design) | **PARTIAL** (bandit over representations) |
| *Multifactorial EA with online transfer parameter estimation* (MFEA-II, IEEE TEVC) | 2019 | MFEA-II | one population, multiple tasks | probabilistic models of task populations | **data-driven mixture-model likelihood sets the transfer matrix (RMP)** | no | whole-solution crossover | continuous | n/s | likelihood-based | yes (between tasks) | no | ABS | CONCEPTUAL PRECEDENT | **PARTIAL** (likelihood-driven control of transfer) |

## E. Landscape-aware learned control (relevant to the abandoned LA-KAIE; context)

| Paper | Year | Algorithm | Learning mechanism | Landscape state | Evidence | relation |
|---|---|---|---|---|---|---|
| *Landscape-Aware Bandit Hyper-Heuristics for Online Operator Selection in UAV Inspection Routing* (LA-BHH, arXiv 2605.14620) | 2026 | LA-BHH | LinUCB | static landscape + online search-state features | ABS | DIRECT for LA-KAIE's controller |
| RL-HPSDE (Swarm and Evolutionary Computation) | 2022 | DE | Q-learning | FDC + ruggedness states | ABS | DIRECT for LA-KAIE's controller |
| DE-DDQN (GECCO 2019) | 2019 | DE | double DQN (offline-trained) | 99 features | ABS | PARTIAL |
| RLDE-AFL (arXiv 2503.18061) | 2025 | DE | RL + learned landscape features | learned | ABS | PARTIAL |

## F. Recent Knowledge-Based Systems papers found (2023–2026)

KBS membership inferred from the ScienceDirect PII prefix `S0950705…`
(KBS ISSN 0950-7051); **verify before citing**. Only titles and snippets were
accessible.

| PII / link | Title (as indexed) | Relevance |
|---|---|---|
| [S0950705125006720](https://www.sciencedirect.com/science/article/abs/pii/S0950705125006720) | *Reinforcement and opposition-based learning enhanced weighted mean of vectors algorithm for global optimization and feature selection* (QLOBLINFO) | RL + OBL in a single metaheuristic, evaluated on **CEC2022**; operator-level control, no interaction modelling |
| [S0950705125015102](https://www.sciencedirect.com/science/article/abs/pii/S0950705125015102) | *Enhancing leadership-based metaheuristics using reinforcement learning: A case study in grey wolf optimizer* (RL-LGWO) | RL controls the leader-update mechanism; agents share experience |
| [S0950705125009761](https://www.sciencedirect.com/science/article/abs/pii/S0950705125009761) | *Competitive many-task differential evolution with reinforcement learning and meta-knowledge transfer* | RL-controlled knowledge transfer between task populations |
| [S0950705124008736](https://www.sciencedirect.com/science/article/abs/pii/S0950705124008736) | *A knowledge-learning-and-transfer-aided differential evolution for nonlinear equation systems* | knowledge transfer inside DE |
| [S0950705122013107](https://www.sciencedirect.com/science/article/abs/pii/S0950705122013107) | *A knowledge transfer-based adaptive differential evolution for solving nonlinear equation systems* | diversity-driven transfer intensity between two niching populations |
| [S0950705125018027](https://www.sciencedirect.com/science/article/abs/pii/S0950705125018027) | *Dynamic multi-knowledge evolutionary algorithm for sparse large-scale multi-objective optimization* | multi-knowledge guidance (MO, sparse) |
| KBS 2017 (doi 10.1016/j.knosys.2017.01.020) | *Multi-objective DE with dynamic covariance matrix learning for problems with variable linkages* | eigen crossover for linkage (F2 precedent) |
| KBS 352:117070 (2026) | MPHBS | reference |

No KBS paper found in this search combined (i) structural-hypothesis
testing for information exchange with (ii) two heterogeneous populations.
With snippet-level access, that is a statement about the search, not proof
of absence.

## G. Answers to the RVIntX questions (F1)

Full text: **not accessible** (Springer and the Zenodo replication package
are both blocked). Answers combine the RVIntX abstract with the **verified
source code of the authors' own precursor** (GAwLL, GECCO'23, which the same
group extends to real-valued DE).

| # | question | answer | basis |
|---|---|---|---|
| 1 | how interaction is learned | empirical linkage learning. In the precursor, the pairwise non-additivity f(x_gh) − f(x_h) − f(x_g) + f(x) over constructed perturbations of a parent; RVIntX's real-valued variant is not verified | CODE (precursor) + ABS |
| 2 | online? | yes ("online linkage learning technique for DE") | ABS |
| 3 | additional objective evaluations? | precursor: LL offspring are evaluated and enter the population (no wasted evaluations, but the offspring are *constructed for the test*). RVIntX: **unverified** | CODE (precursor) |
| 4 | recombination outcomes as the learning signal? | precursor: **no**, the signal comes from mutation-based quadruples, not crossover outcomes. RVIntX: **unverified** | CODE (precursor) |
| 5 | explicit pairwise terms? | yes (weighted interaction graph) | CODE + ABS |
| 6 | used to build recombination masks? | yes ("explores the epistatic structure … and variables common to both parents to find good recombination masks") | ABS |
| 7 | between heterogeneous populations? | no evidence; single DE population | ABS |
| 8 | learns co-transfer effects? | **mathematically, yes in substance.** For donor substitution, the co-transfer effect of (i, j) is the second-order non-additivity g(1_i+1_j) − g(1_i) − g(1_j) + g(0) of the transfer function g(m) = f((1 − m)⊙x_r + m⊙x_d), which is the same quantity that empirical LL measures, with perturbation = donor value | derivation |
| 9 | does it make OB-CEL substantially overlapping? | **yes.** OB-CEL would be a different *estimator* (passive regression over random masks, including failures) of an already-targeted quantity, used for inter-population transfer. That is a different implementation and setting, not a new mechanism. Combined with LEL ("interaction graph from successful optimization steps"), F1 is not a defensible core novelty | synthesis |
