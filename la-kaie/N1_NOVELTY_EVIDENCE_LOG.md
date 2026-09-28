# N1 novelty evidence log

Searcher: Claude Code session (cloud container). Date of all searches: **2026-09-28**.

## Access conditions

| channel | status |
|---|---|
| Web search engine (result titles and snippets) | available |
| GitHub over git protocol (clone / ls-remote) | available |
| GitHub web pages and REST API | blocked (HTTP 403) |
| arXiv, export.arxiv, alphaXiv, Bytez mirror | blocked |
| Springer, ScienceDirect full text, IEEE Xplore, ACM DL, MIT Press, ResearchGate | blocked |
| Zenodo, OpenAlex, Semantic Scholar, Crossref, DBLP APIs | blocked |
| PDFs supplied by the user in `reference/` | only MPHBS.pdf present |

Consequence: apart from MPHBS, **no publisher full text was accessible**.
Source-level evidence was obtained only from **code** for four precedents.
Search snippets are recorded as leads, not as substantive evidence.

## A. Source-level verifications (CODE / FULL TEXT)

| # | work | source obtained | exact location | finding |
|---|---|---|---|---|
| V1 | Flores & Olivares, *MPHBS*, Knowledge-Based Systems 352 (2026) 117070, doi:10.1016/j.knosys.2026.117070 | FULL TEXT (PDF) + authors' MATLAB code (github.com/efloresaraya/MPHBS-SARSA-information-exchange) | paper §4.2, Eqs. (37)–(61), Alg. 2; `MPHBS_main.m` | fixed dimension-wise exchange from ranked HBA/MPA memories; SARSA-style Q(d, n, ch); no structural alternative; no null / no-exchange state; the mediator always runs E = N_sub·N_i episodes; domain-specific configurations (B-C1 / A-C1) |
| V2 | Akimoto & Hansen, *Diagonal Acceleration for CMA-ES*, Evolutionary Computation 28(3):405–435 (2020) (reference stated in the code docstring) | CODE: author's gist `ddcma.py` (gist 1180b67b5a0b1265c204cba991fa8518, history up to 2026-08-30) | `DdCma.update`, around lines 210–228 | the diagonal update rate is scaled by `beta = 1 / max(1, max(S)/min(S) − beta_thresh + 1)` (S = sqrt of eigenvalues of the correlation-like matrix C, beta_thresh = 2). This is a **continuous modulation** of the diagonal adaptation by the conditioning of C: no discrete hypothesis choice, no complexity penalty, no null state, a single optimizer |
| V3 | GOMEA library (CWI Evolutionary Intelligence), github.com/CWI-EvolutionaryIntelligence/GOMEA, commit 769fa6e (2025-07-12) | CODE | `gomea/linkage.pyx` (classes Univariate, BlockMarginalProduct, Full, StaticLinkageTree, LinkageTree, Custom; Conditional commented out); `gomea/src/common/linkage_model.cpp` lines ~39–80 (type chosen from user configuration), ~479–486 (similarity: MI, NMI, VIG, TIGHT, RANDOM), ~492–700 (UPGMA linkage tree, optional filtering of elements with similarity ≥ 1−1e−6 or = 0) | the linkage-model **type is chosen by the user** (constructor argument); the learned linkage tree is built from MI/NMI similarity. **No BIC/likelihood comparison between model types was found in the library.** Multiple populations exist (interleaved multi-start of the same algorithm), not heterogeneous optimizers. No no-linkage / no-transfer test |
| V4 | MFEA-II (Bali et al., IEEE TEVC 2019, ieeexplore 8672822) | CODE: Python port github.com/thanhlexyz/mfea-ii, which states it implements the official MATLAB version | `mtsoo/operators.py: learn_rmp` (lines ~102–122) | the transfer probability RMP between tasks k and j is obtained by `fminbound` maximising a mixture log-likelihood of each task's population under both tasks' fitted densities. RMP ∈ [0, 1] can approach 0 (**transfer effectively suppressed**) via continuous likelihood fitting, **not** a sequential test against a native-search control. Multitask (different tasks), not heterogeneous optimizers on one task |
| V5 | Tinós, Przewozniczek, Whitley, Chicano, *Genetic Algorithm with Linkage Learning*, GECCO '23, doi:10.1145/3583131.3590349 | CODE: github.com/rtinos/GAwLL | `gawll.cpp: mutationLL` | empirical linkage from non-additivity of constructed perturbations (recorded in the previous assessment); single population |

## B. Search queries and results (web search engine; snippets only unless noted)

| # | query (abridged) | relevant results | full text accessible? |
|---|---|---|---|
| S1 | GitHub code for LCC ("Advancing CMA-ES with Learning-Based Cooperative Coevolution") | arXiv 2504.17578 (Guo, Qiu, Ma, Zhang, Zhang, Gong, 2025); no code link found; probed GMC-DRL/LCC, LCC-CMAES, LH-CC etc. via git: not found; Awesome-MetaBBO list does not include it | no |
| S2 | dd-CMA source | author's gist found → V2 | code yes |
| S3 | GOMEA library source | CWI repo found → V3 | code yes |
| S4 | ACoS source code | arXiv 1703.06263; TCYB doi 10.1109/tcyb.2018.2802912; follow-up *An adaptive framework to select the coordinate systems for EAs* (Applied Soft Computing, S156849462200638X); no code found | no |
| S5 | ACoS mechanism | snippet: "Eigen coordinate system … coupled with the original coordinate system … selected according to a probability vector … adaptively updated based on the collected information from the offspring" | no |
| S6 | LH-CC | GECCO 2026, doi 10.1145/3795095.3805054, arXiv 2604.01241: snippet "formulating the optimization process as an MDP, where a meta-agent adaptively selects the most suitable optimizer for each subproblem" | no |
| S7 | LCC mechanism | snippet: "dynamically schedules decomposition strategies … decomposition strategy selector parameterized through a neural network that processes optimization status features" | no |
| S8 | EMT negative-transfer detection / statistical tests | OKTPO-MFEA, *EMT with online knowledge transfer and probabilistic outlier detection* (Complex & Intelligent Systems 2025, s40747-025-02220-0); MGAD, *Adaptive EMT based on anomaly detection transfer of multiple similar sources* (ESWA 2025, S0957417425012217); EMT with elbow PCA and negative-transfer optimization (ESWA 2026); Wilcoxon tests used *for evaluation* | no |
| S9 | SPRT + evolutionary transfer / migration | no SPRT-based transfer or migration control found | – |
| S10 | transferability estimation, likelihood ratio / Bayesian | *Multi-Task Surrogate-Assisted Search with Bayesian Competitive Knowledge Transfer* (arXiv 2510.23407): transferability as a latent variable inferred by Bayes' rule; *Bayesian Inverse Transfer in EMO* (arXiv 2312.14713); transfer GPs | no |
| S11 | island model migration decided by statistical test vs control | none found; island strategy switching based on performance feedback | no |
| S12 | transfer vs no-transfer control comparisons in EMT | Gupta & Ong, *Genetic Transfer or Population Diversification? Deciphering the Secret Ingredients of EMT* (arXiv 1607.05390; IEEE 7850038, 2016): **offline analysis** separating transfer from diversification effects; "information transfer success rate prediction function … controlling information exchange between tasks" (source paper not identified in the snippet) | no |
| S13 | heterogeneous optimizers + landscape structure agreement | *On the Structural (Dis)Agreement of Landscape Representations in Black-Box Optimization* (arXiv 2605.28121): agreement between ELA *feature representations*, not optimizers; Heterogeneous DE (PMC3933298): per-individual strategies | partly (PMC not accessed) |
| S14 | KBS negative transfer / EMT 2023–2025 | KBS: *Many-task EA with adaptive knowledge transfer via density-based clustering* (S0950705123006561); *Enhancing EMT by leveraging inter-task knowledge transfers and improved evolutionary operators* (S0950705122011200); *Effective transferred knowledge identified by bipartite graph for MO multitasking* (S0950705124001655); Information Sciences: MFEA-ML (S0020025525000404) | no |
| S15 | KBS landscape / covariance / separability 2023–2025 | no KBS paper on separability-based exchange found; CEC2022 landscape-feature study (arXiv 2402.07654) | no |
| S16 | "exchange representation" / "transfer representation" selection | **Ensemble knowledge transfer framework for EMT** (Swarm and Evolutionary Computation 2023, S2210650223001670): snippet "adaptive information exchange (AIE) strategy in conjunction with multi-arm selection (MAS) … to configure domain adaptation strategies dynamically during the search" | no |
| S17 | BIC / information criterion + migration / CC grouping | none found in evolutionary migration or CC grouping | – |
| S18 | model selection metaheuristic information exchange | none relevant | – |
| S19 | structural hypothesis / structure selection multi-population | none relevant | – |
| S20 | adaptive representation selection EA | *Online Selection of CMA-ES Variants* (arXiv 1904.07801); KBS adaptive switching EA (S0950705122004385; many-objective deletion criteria) | no |
| S21 | linkage-model (FOS) selection | FOS types documented; *Predetermined versus learned linkage models* (GECCO 2012, doi 10.1145/2330163.2330205) | no (library code checked, V3) |
| S22 | BOA / EBNA BIC structure learning | standard: BIC/MDL scores network structures within a single EDA | no |
| S23 | Gaussian graphical-model EDAs | *Continuous EDAs Based on Factorized Gaussian Markov Networks* (Springer chapter 978-3-642-28900-2_10) | no |
| S24 | island EDAs with model migration | *Migration of Probability Models Instead of Individuals: An Alternative When Applying the Island Model to EDAs* (Springer 978-3-540-30217-9_25, 2004): combination weights constant or fitness-adaptive; *A Parallel Island Model for EDAs*; *Improving EAs with Multi-representation Island Models* (GMU) | partly (GMU PDF not fetched: WebFetch not attempted for this host) |
| S25 | separability detection → decomposition switch | LCC; CC convergence analyses (arXiv 2304.05020); DCC (multilevel LM-CMA / CMA-ES) | no |
| S26 | earlier sessions (see `DETAILED_NOVELTY_MATRIX.md`, `N1_NOVELTY_ASSESSMENT.md`) | ACoS, CoBiDE, CPI-DE, DE/eig, MCDE, HLX, RVIntX, LEL, VIGPSO, AMALGAM, MOS, PAP, CBCC, FRRMAB, LA-BHH, RL-HPSDE, MFEA-II, adaptive migration intervals, KBS 2023–2026 list | no (except code items above) |

## C. What remains unverified (full text required)

LCC (arXiv 2504.17578); LH-CC (doi 10.1145/3795095.3805054); ACoS (doi
10.1109/tcyb.2018.2802912) and its 2022 ASOC follow-up; the ensemble
knowledge transfer framework with AIE + MAS (S2210650223001670); OKTPO-MFEA
(s40747-025-02220-0); MGAD (S0957417425012217); the Bayesian competitive
knowledge transfer paper (arXiv 2510.23407); *Predetermined versus learned
linkage models* (GECCO 2012); the island-EDA model-migration papers; the
multi-representation island models paper; Gupta & Ong (arXiv 1607.05390);
the KBS EMT papers listed in S14.
