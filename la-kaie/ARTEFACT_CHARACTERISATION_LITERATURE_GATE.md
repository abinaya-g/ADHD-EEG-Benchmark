# Artefact-characterisation study: literature gate

- Date: 2026-09-30.
- Status: literature review only. No implementation, no experiment, no change to N1 or its
  specification.

**Evidence standard (read first).**
- In this environment, publisher sites, arXiv and scholarly APIs are blocked by the network
  egress policy (confirmed again today: HTTP CONNECT refused for arxiv.org, sciencedirect.com,
  dl.acm.org, link.springer.com, ieeexplore.ieee.org and api.crossref.org).
- The evidence below therefore comes from **search-engine abstracts and snippets**, combined
  with foundational knowledge of the field. **No full text was read.**
- Every mechanism attributed to a paper is marked:
  - **[S]**: stated in an abstract or snippet;
  - **[K]**: my prior knowledge of a well-known result;
  - **[U]**: unresolved, needs the full text.
- DOIs are given only where a search result showed them (as a DOI or a DOI-bearing URL).
  Otherwise the publisher item identifier (PII) is given, or the DOI is marked unknown.
  None was invented.

---

## 1. Research question

> When do successful displacements reveal landscape structure? An empirical characterisation
> of operator-induced geometric bias in metaheuristic optimization.

Candidate hypotheses:
- **H-a:** for HBA and FAD, Cov(Δx) ∝ Cov(X).
- **H-b:** successful-displacement geometry carries landscape structure only to the extent
  that population geometry does.
- **H-c:** block structure present in the population can be merged by significance-based
  linkage partitioning at large n.

## 2. Why the D3 result matters

The exploratory D3 analysis (`N1_D3_OFFLINE_ANALYSIS_REPORT.md`, classification B) found:
- Successful HBA/MPA/FAD displacements track **population** geometry in all cells.
- HBA moves are about 0.85–1.0 × (xprey − x).
- The landscape alignment seen for the MPA population equals the population's own alignment,
  and it disappears under whitening.

This invalidated the N1 premise (E4). It also suggested a possible study of the artefact
itself. The question for this gate is whether such a study would be a contribution, or a
restatement of known results.

## 3. Search strategy

- **Queries:** 26 web-search queries, covering all 20 requested topic areas and the conceptual
  equivalents listed in the brief.
  - DE contour matching / contour fitting;
  - differential-mutation covariance;
  - covariance–Hessian relation in ES;
  - CMA-ES successful-step PCA;
  - Gaussian-EDA covariance misalignment;
  - structural bias (including anisotropy);
  - centre bias;
  - PSO directional and rotation bias;
  - eigenvector crossover (DE);
  - search-trajectory networks;
  - behavioural characterisation of metaheuristics;
  - trajectory-based and sampling-biased ELA;
  - metaphor-algorithm operator decomposition;
  - linkage learning and interaction thresholds;
  - large-n significance;
  - HBA and MPA/FAD analyses;
  - component/ablation methodology.
- **Venues favoured:** IEEE TEVC, Evolutionary Computation, SWEVO, ASOC, Information
  Sciences, KBS, GECCO, PPSN, CEC, TCS.
- **Limitation:** abstracts and snippets only (see the evidence standard).

## 4. Closest prior work

| # | work | venue, year | DOI / id | mechanism (evidence level) |
|---|---|---|---|---|
| P1 | K. Price, R. Storn, J. Lampinen, *Differential Evolution: A Practical Approach to Global Optimization* | Springer book, 2005 | DOI unknown here | **contour matching**: the distribution of difference vectors adapts to the objective's contours, because it is generated from population differences [K][S] |
| P2 | K. R. Opara, J. Arabas, *The contour fitting property of differential mutation* | Swarm and Evolutionary Computation, 2019 | PII S2210650218301470 | analytical model linking differential mutation to adaptation of search range and direction; "the covariance matrix of a population in a stable state is proportional to the covariance matrix of the Gaussian objective function"; proof for a DE/prop/1 operator [S] |
| P3 | K. Opara, J. Arabas, *Differential Mutation Based on Population Covariance Matrix* | PPSN XI, 2010 | 10.1007/978-3-642-15844-5_12 | the sum of many difference vectors converges to a normal distribution with covariance **proportional to the population covariance**, which motivates Gaussian mutation with population covariance [S] |
| P4 | K. R. Opara, J. Arabas, *Differential Evolution: A survey of theoretical analyses* | SWEVO 44 (2019) 546–558 | PII S2210650217304224 | survey of DE theory, including contour fitting [S] |
| P5 | V. Novák, I. Zelinka, *Linear Proposal Operators and Stochastic Search Geometry in SOMA and Differential Evolution* | arXiv preprint, 31 Jul 2026 | arXiv 2607.29228 | operator–selection factorisation separating objective-independent variation from selection; **closed-form proposal mean and covariance**, and the covariance and coordinate dependence induced by binomial crossover, for DE/rand/1/bin and SOMA [S]. Whether it treats *successful* steps or landscape comparison: [U] |
| P6 | O. M. Shir, A. Yehudayoff, *On the covariance–Hessian relation in evolution strategies* | Theoretical Computer Science, 2020 | PII S0304397519305468; arXiv 1806.03674 | the covariance of (1,λ)-**selected** decision vectors becomes proportional to the inverse landscape Hessian as λ grows, from isotropic mutation plus rank selection; proven and numerically validated [S] |
| P7 | N. Hansen, A. Ostermeier, *Completely Derandomized Self-Adaptation in Evolution Strategies* (CMA-ES); N. Hansen, *The CMA Evolution Strategy: A Tutorial* | Evolutionary Computation, 2001; arXiv 1604.00772 | – | CMA performs a PCA of previously **selected (successful) mutation steps** (rank-μ) and uses evolution paths; the covariance tends toward the inverse Hessian on convex quadratics [S][K] |
| P8 | T. Glasmachers, O. Krause, *The Hessian Estimation Evolution Strategy* / *Convergence Analysis of HE-ES* | PPSN 2020 / Evolutionary Computation, 2022 | arXiv 2003.13256; 2009.02732 | covariance converges to a multiple of the inverse Hessian (proof) [S]. Author names: [K], not shown in the snippet |
| P9 | P. A. N. Bosman, J. Grahl, D. Thierens, *Enhancing the Performance of Maximum-Likelihood Gaussian EDAs Using Anticipated Mean Shift* | PPSN X, 2008 | 10.1007/978-3-540-87700-4_14 | fitting an ML Gaussian to the (selected) population on a slope yields density contours "aligned in the worst way": **population geometry misaligned with the useful landscape direction**; also variance shrinkage; remedies AVS / SDR / AMS [S] |
| P10 | A. V. Kononova, D. W. Corne, P. De Wilde, V. Shneer, F. Caraffini, *Structural bias in population-based algorithms* | Information Sciences 298 (2015) 468–490 | 10.1016/j.ins.2014.11.035 | **operator-induced bias** independent of the landscape, detected on an uninformative random objective f0 [S] |
| P11 | D. Vermetten, B. van Stein, F. Caraffini, L. Minku, A. V. Kononova, *BIAS: A Toolbox for Benchmarking Structural Bias in the Continuous Domain* | IEEE TEVC 26(6), 2022 | DOI unknown here | statistical toolbox for structural bias on f0 [S] |
| P12 | *Is there anisotropy in structural bias?* | GECCO Companion, 2021 | 10.1145/3449726.3463218 | directional (anisotropic) structural bias [S]; content [U] |
| P13 | J. Kudela, *A critical problem in benchmarking and analysis of evolutionary computation methods* | Nature Machine Intelligence 4 (2022) 1238–1245 | 10.1038/s42256-022-00579-0 | centre-bias operators, i.e. operator-induced geometric bias, in many recent metaheuristics [S] |
| P14 | M. R. Bonyadi, Z. Michalewicz, *A locally convergent rotationally invariant particle swarm optimization algorithm* (and related PSO analyses) | Swarm Intelligence, 2014 | 10.1007/s11721-014-0095-1 | analytical demonstration that standard PSO's operator **biases search toward coordinate-parallel lines**; rotation invariance [S] |
| P15 | Y. Wang, H.-X. Li, T. Huang, L. Li, *Differential evolution based on covariance matrix learning and bimodal distribution parameter setting* (CoBiDE) | Applied Soft Computing, 2014 | PII S1568494614000581 | uses the population covariance eigenbasis as the crossover coordinate system, *assuming* population geometry reflects the landscape [S] |
| P16 | *A study on rotation invariance in differential evolution* | SWEVO, 2019 | PII S2210650218301482 | rotation dependence of DE operators [S]; authors and content [U] |
| P17 | G. Ochoa, K. M. Malan, C. Blum, *Search trajectory networks: a tool for analysing and visualising the behaviour of metaheuristics* | Applied Soft Computing 109 (2021) 107492 | 10.1016/j.asoc.2021.107492 | compares the **trajectories of heterogeneous metaheuristics on a fixed problem**, graph-based [S] |
| P18 | L. Hayward, A. Engelbrecht, *Determining Metaheuristic Similarity Using Behavioral Analysis* | IEEE TEVC 29(1) (2025) 262–274 | 10.1109/TEVC.2023.3346672 | 20 behavioural characteristics (exploration, exploitation, locality, …) to compare metaheuristics' search behaviour [S] |
| P19 | A. Jankovic, T. Eftimov, C. Doerr, *Towards Feature-Based Performance Regression Using Trajectory Data* | EvoApplications, 2021 | 10.1007/978-3-030-72699-7_38 | trajectory samples are biased toward visited regions and capture algorithm-dependent local properties rather than global landscape characteristics [S] |
| P20 | Q. Renau, C. Doerr, J. Dreo, B. Doerr, *Exploratory Landscape Analysis is Strongly Sensitive to the Sampling Strategy* | PPSN XVI, 2020 | 10.1007/978-3-030-58115-2_10 | landscape-feature estimates depend on the sampling strategy, so a **sample is a biased measurement** of the landscape [S] |
| P21 | C. L. Camacho-Villalón, M. Dorigo, T. Stützle, *Exposing the grey wolf, moth-flame, whale, firefly, bat, and antlion algorithms* | International Transactions in Operational Research, 2023 | 10.1111/itor.13176 | decomposes metaphor-based operators into known components, e.g. attraction toward the best [S]. HBA and MPA are not covered |
| P22 | M. N. Omidvar, M. Yang, Y. Mei, X. Li, X. Yao, *DG2: A Faster and More Accurate Differential Grouping* | IEEE TEVC, 2017 | 10.1109/TEVC.2017.2694221 | interaction-detection thresholds and their error sources (round-off) [S] |
| P23 | F. A. Hashim et al., *Honey Badger Algorithm* | Mathematics and Computers in Simulation, 2022 | PII S0378475421002901 | original HBA [S]. No geometric analysis of HBA moves found |
| P24 | A. Faramarzi et al., *Marine Predators Algorithm* | Expert Systems with Applications, 2020 | PII S0957417420302025 | original MPA including FADs [S]. Reviews and variants found; no geometric analysis of FAD moves found |

## 5. H-a overlap: Cov(Δx) ∝ Cov(X)

- **FAD: HIGH.**
  - FAD's permutation branch (probability 0.8) is a scaled DE difference vector,
    s·(X_a − X_b).
  - That the covariance of difference vectors is proportional to the population covariance
    is established DE theory: P1 (contour matching), P3 (the population-covariance limit of
    difference vectors), P2 (formal contour fitting), and P5 (closed-form finite-population
    moments of differential mutation, 2026).
  - Stating it for FAD is a direct corollary. It is not a new result.
- **HBA: MEDIUM.**
  - For "move toward a shared target" operators, Δx ≈ b·(target − x) implies
    Cov(Δx) ≈ b²·Cov(X). This is elementary algebra.
  - Such operators are well studied as attraction-to-best components (P14 for PSO; P21's
    decomposition of metaphor algorithms).
  - I found **no HBA-specific derivation or measurement** [U: absence of evidence under
    snippet-only access].
  - Applying the known algebra to HBA adds little conceptually.
- **Successful (selected) steps vs all proposals: MEDIUM.**
  - Covariance of *selected* steps is the core of CMA-ES (P7) and of the covariance–Hessian
    theory (P6).
  - Our observation that selection does *not* re-shape HBA/FAD displacements away from the
    population's shape is a specific empirical finding. Its generality is untested.

## 6. H-b overlap: displacements reveal the landscape only through the population

- **Largely established (HIGH–MEDIUM)**, in a closely related form:
  - DE contour fitting (P1, P2): the population, and therefore the differences, adapts to
    the landscape contours "in a stable state".
  - ES theory (P6, P8): selection plus isotropic variation makes selected-point covariance
    ∝ H⁻¹.
  - Gaussian EDAs (P9): population geometry can be misaligned with the landscape, e.g. on
    slopes.
- Together these already say that the population's covariance is the carrier of landscape
  information, with conditions under which it is aligned (near-stationary, near-optimum,
  large λ) or misaligned (slopes, shrinking variance).
- What H-b adds is the *negative* corollary for **displacements used as independent
  evidence**: they bring no information beyond the population's shape. That follows from
  H-a plus contour fitting.
- The empirical demonstration for HBA/MPA, with a known Hessian and whitening, is new in
  detail as far as I found. It is not new in principle.

## 7. H-c overlap: large-n significance merges blocks

- **LOW novelty.**
  - That tiny effects become "significant" at large n is a well-known statistical issue
    (e.g. the "large n" / p-value literature; one recent treatment is *Revisiting the Large
    n (Sample Size) Problem*, Stats 2023, 10.3390/stats6040081 [S]).
  - Linkage-learning literature treats threshold choice as central (P22; filtering in
    GOMEA / linkage trees [K]).
- Applying this to a Bonferroni/Fisher connected-components partition is a methodological
  caution. It is not a research contribution.
- Also, our H-c evidence is indirect: the procedure was not varied.

## 8. Operator-ablation prior art

- **Standard methodology.** Component and ablation studies are routine (e.g. component-study
  design [S]; P21's component decomposition).
- Using a **landscape-free control** to isolate operator-induced bias is the established
  structural-bias method (P10–P12, on f0).
- Using **ablation specifically to separate landscape information from operator geometry
  in the step covariance**: I did not find a direct match [U]. It is a natural extension of
  P10–P12 plus P2/P5.

## 9. Population-vs-landscape prior art

- **Well developed:**
  - contour fitting (P2);
  - covariance–Hessian relation (P6, P8);
  - EDA misalignment (P9);
  - eigenvector crossover built on population covariance (P15).
- **Whitening as a diagnostic:** working in the covariance-whitened ("isotropic")
  coordinates is standard in CMA-ES (P7) [K]. Its use specifically as a diagnostic to remove
  population geometry from displacement structure: not found [U]. It is a routine
  transformation.
- **Sampling-bias framing** ("displacements are a biased measurement of the landscape"):
  closely related to P19 and P20, which already establish that optimizer-generated samples
  are biased landscape measurements for ELA.

## 10. HBA / MPA / FAD prior art

- **HBA:** no analysis of move geometry found; only proposals and variants (P23 and
  improvement papers).
- **MPA:** reviews, variants and parameter-sensitivity notes found (P24; *Marine predators
  algorithm: a comprehensive review*, 2023, PII S2666827023000245 [S]). No analysis of FAD
  step covariance found.
- The FAD permutation branch is DE mutation, so its geometry is covered by P1–P5.
- The metaphor critique (P21; the "bestiary" literature) makes it likely that reviewers would
  treat HBA/MPA-specific results as **instances of known operator classes**.

## 11. Novelty matrix

Evidence levels:
- HIGH = the prior work establishes the item;
- MEDIUM = partial or closely related;
- LOW = tangential;
- NONE = not addressed;
- UNRESOLVED = cannot tell from snippets.

| Prior work | H-a | H-b | H-c | Operator ablation | Population vs landscape | HBA/FAD |
|---|---|---|---|---|---|---|
| P1 Price et al. 2005 (contour matching) | HIGH (DE differences) | MEDIUM | NONE | NONE | MEDIUM | MEDIUM (FAD = DE-type) |
| P2 Opara & Arabas 2019 (contour fitting) | HIGH | HIGH | NONE | NONE | HIGH | MEDIUM (FAD) |
| P3 Opara & Arabas 2010 | HIGH | MEDIUM | NONE | NONE | MEDIUM | MEDIUM (FAD) |
| P5 Novák & Zelinka 2026 (proposal geometry) | HIGH (closed-form moments) | UNRESOLVED | NONE | UNRESOLVED (factorisation separates variation from selection) | UNRESOLVED | MEDIUM (FAD) |
| P6 Shir & Yehudayoff 2020 | MEDIUM (selected-point covariance) | HIGH (when selected points reveal H) | NONE | NONE | HIGH | NONE |
| P7 CMA-ES (Hansen & Ostermeier; tutorial) | MEDIUM | MEDIUM | NONE | NONE | MEDIUM | NONE |
| P8 HE-ES | LOW | MEDIUM | NONE | NONE | MEDIUM | NONE |
| P9 Bosman, Grahl & Thierens 2008 | MEDIUM | HIGH (misalignment) | NONE | LOW | HIGH | NONE |
| P10–P12 structural bias / BIAS / anisotropy | LOW | LOW | NONE | MEDIUM (landscape-free control) | MEDIUM | NONE |
| P13 Kudela 2022 (centre bias) | NONE | LOW | NONE | MEDIUM | LOW | NONE |
| P14 Bonyadi & Michalewicz (PSO direction bias) | MEDIUM (operator-induced direction) | LOW | NONE | LOW | LOW | NONE |
| P15 CoBiDE / eigenvector crossover | LOW | MEDIUM (assumes alignment) | NONE | NONE | MEDIUM | NONE |
| P17 Ochoa et al. 2021 (STNs) | NONE | LOW | NONE | NONE | LOW | NONE |
| P18 Hayward & Engelbrecht 2025 | LOW | LOW | NONE | NONE | LOW | UNRESOLVED |
| P19–P20 trajectory / sampling-biased ELA | NONE | MEDIUM (biased measurement) | NONE | NONE | MEDIUM | NONE |
| P21 Camacho-Villalón et al. 2023 | MEDIUM (attraction-to-best decomposition) | NONE | NONE | LOW | NONE | LOW |
| P22 DG2 | NONE | NONE | MEDIUM (threshold error) | NONE | NONE | NONE |
| large-n significance literature | NONE | NONE | HIGH (principle) | NONE | NONE | NONE |
| P23–P24 HBA / MPA originals and reviews | NONE | NONE | NONE | LOW (parameter sensitivity) | NONE | LOW (definitions only) |

## 12. What is already known
- Difference-vector (DE/FAD-type) step covariance is proportional to population covariance
  (P1–P5).
- Population and selected-point covariance can become aligned with the inverse Hessian under
  identifiable conditions (P2, P6, P8), and can be misaligned in others (P9).
- Operators impose landscape-independent biases. This can be detected with landscape-free
  controls (P10–P14).
- Optimizer-generated samples are biased measurements of landscape features (P19, P20).
- Significance thresholds become uninformative at large n. Interaction-detection thresholds
  are a known difficulty (P22).

## 13. What remains unresolved
- Whether P5 (2026) already analyses **successful/selected** steps and compares them with
  landscape structure. This needs the full text; it is the closest recent risk.
- Whether the anisotropic structural-bias work (P12) already measures step-covariance
  anisotropy.
- An HBA-specific (target-directed plus intensity-term) geometric analysis. None found, but
  access was snippet-only.
- A controlled demonstration, with known Hessians, operator ablation and whitening, that
  **successful displacements add no landscape information beyond population geometry** for
  attraction-to-target and difference-vector operators in general. Not found as such.
- The full texts of P2, P5, P6, P9 and P12, to confirm the depth of overlap. All are
  currently inaccessible.

## 14. Potential contribution (if any)

What is not already covered is narrow:
1. A **negative result about evidence quality**: successful native displacements are not an
   independent measurement of landscape structure. For attraction-to-target and
   difference-vector operators, their structure reduces to population geometry. This is
   shown with:
   - known Hessians (S1–S3);
   - whitening and residualisation;
   - the implication that displacement-based online structure or linkage learning is biased.
2. An **operator-class taxonomy of step-geometry bias**: which operator families add
   information beyond the population (isotropic-mutation + selection, as in P6) and which do
   not (difference vectors, attraction-to-target).
   - This would tie together P2/P5/P6/P9 under one measurement protocol, with ablations as
     interventions.
   - It would need to go beyond HBA/MPA to canonical operators to be of general interest.

Both are **syntheses and empirical confirmations of largely known mechanisms**. The first
has the clearer niche. Its interest depends on the community using displacement or
trajectory statistics as structure evidence, which is plausible given P15, P19 and
linkage-learning practice.

## 15. Risks to novelty
- **Very high:** H-a is essentially DE theory (FAD) plus elementary algebra (HBA). P5 (July
  2026) derives proposal covariances for DE in closed form and may cover more.
- **High:** H-b is implied by contour fitting and the covariance–Hessian relation. Reviewers
  may see the study as re-deriving known facts on metaphor-based algorithms (the P21
  critique).
- **High:** H-c is statistical folklore.
- **Medium:** the "biased measurement" framing overlaps with sampling-bias ELA work (P19,
  P20).
- **Unresolved:** full texts of P2, P5, P6, P9 and P12 were not read. Overlap may be
  greater than assessed.

## 16. Recommended research framing (if pursued at all)
- Do **not** frame the study as a discovery of H-a, H-b or H-c, or as HBA/MPA-specific
  insight.
- If pursued, frame it as a **measurement-validity study**:
  > "Are successful-step statistics valid evidence of landscape structure? A controlled test
  > across operator classes."
  - Build explicitly on contour fitting (P2), proposal geometry (P5), the covariance–Hessian
    relation (P6), EDA misalignment (P9), structural bias (P10–P12) and sampling-biased ELA
    (P19–P20).
  - Test canonical operator classes: isotropic Gaussian + selection; DE/rand/1;
    attraction-to-best (PSO-like); and HBA/MPA only as instances.
  - Use known-Hessian landscapes, landscape-free controls (f0 / sphere), and operator
    ablation as the causal intervention.
- **Standard components** (not contributions): known-Hessian synthetic landscapes, CMA-style
  whitening, ablation, covariance/subspace similarity metrics, temporal binning, residual
  regression.
- The **possibly non-standard element**: the joint use of these components to test the
  validity of displacement-based structure evidence, with the negative-evidence emphasis.
- **Before committing:** read P2, P5, P6, P9 and P12 in full. This needs the PDFs supplied, or
  the hosts allowed in the network policy.

## 17. Verdict

**C. SUBSTANTIAL PRIOR-ART OVERLAP**

- The three candidate hypotheses are, respectively:
  - H-a: established (FAD / DE theory) or elementary (HBA);
  - H-b: largely implied by contour-fitting and covariance–Hessian results;
  - H-c: statistical folklore.
- Operator ablation, landscape-free controls and whitening are standard methodology.
- What remains is a narrow measurement-validity synthesis (§14), whose novelty depends on
  unread full texts (especially P5, 2026) and on generalising beyond HBA/MPA.
- The study should not proceed as proposed. It could only proceed after a reframing (§16)
  **and** a full-text check of P2, P5, P6, P9 and P12.

---

## Sources
- [The contour fitting property of differential mutation (SWEVO)](https://www.sciencedirect.com/science/article/abs/pii/S2210650218301470)
- [Differential Evolution: A survey of theoretical analyses (PDF)](https://www.math.ucdavis.edu/~saito/data/PSO-ACO/opara-arabas_differential-evol-survey.pdf)
- [Differential Mutation Based on Population Covariance Matrix (PPSN 2010)](https://link.springer.com/chapter/10.1007/978-3-642-15844-5_12)
- [Linear Proposal Operators and Stochastic Search Geometry in SOMA and DE (arXiv 2607.29228)](https://arxiv.org/abs/2607.29228)
- [On the covariance–Hessian relation in evolution strategies (arXiv 1806.03674)](https://arxiv.org/abs/1806.03674)
- [On the covariance–Hessian relation in evolution strategies (TCS)](https://www.sciencedirect.com/science/article/pii/S0304397519305468)
- [The CMA Evolution Strategy: A Tutorial (arXiv 1604.00772)](https://arxiv.org/pdf/1604.00772)
- [Completely Derandomized Self-Adaptation in Evolution Strategies](http://www.cmap.polytechnique.fr/~nikolaus.hansen/cmaartic.pdf)
- [The Hessian Estimation Evolution Strategy (arXiv 2003.13256)](https://arxiv.org/pdf/2003.13256)
- [Convergence Analysis of the Hessian Estimation Evolution Strategy (arXiv 2009.02732)](https://arxiv.org/pdf/2009.02732)
- [Enhancing the Performance of ML Gaussian EDAs Using Anticipated Mean Shift (PPSN 2008)](https://link.springer.com/chapter/10.1007/978-3-540-87700-4_14)
- [Structural bias in population-based algorithms (Information Sciences)](https://www.sciencedirect.com/science/article/abs/pii/S0020025514011165)
- [BIAS toolbox (TechRxiv)](https://www.techrxiv.org/doi/full/10.36227/techrxiv.16594880.v1)
- [Is there anisotropy in structural bias? (GECCO 2021)](https://dl.acm.org/doi/10.1145/3449726.3463218)
- [A critical problem in benchmarking and analysis of evolutionary computation methods (NMI 2022)](https://www.nature.com/articles/s42256-022-00579-0)
- [A locally convergent rotationally invariant PSO algorithm (Swarm Intelligence 2014)](https://link.springer.com/article/10.1007/s11721-014-0095-1)
- [DE based on covariance matrix learning and bimodal distribution parameter setting (ASOC 2014)](https://www.sciencedirect.com/science/article/abs/pii/S1568494614000581)
- [A study on rotation invariance in differential evolution (SWEVO)](https://www.sciencedirect.com/science/article/abs/pii/S2210650218301482)
- [Search trajectory networks (ASOC 2021)](https://www.sciencedirect.com/science/article/abs/pii/S1568494621004154)
- [Determining Metaheuristic Similarity Using Behavioral Analysis (IEEE TEVC)](https://ieeexplore.ieee.org/document/10373561/)
- [Towards Feature-Based Performance Regression Using Trajectory Data (arXiv 2102.05370)](https://arxiv.org/pdf/2102.05370)
- [Exploratory Landscape Analysis is Strongly Sensitive to the Sampling Strategy (arXiv 2006.11135)](https://arxiv.org/abs/2006.11135)
- [Exposing the grey wolf, moth-flame, whale, firefly, bat, and antlion algorithms (ITOR)](https://onlinelibrary.wiley.com/doi/10.1111/itor.13176)
- [DG2: A Faster and More Accurate Differential Grouping (IEEE TEVC)](https://dl.acm.org/doi/abs/10.1109/tevc.2017.2694221)
- [Revisiting the Large n (Sample Size) Problem (Stats 2023)](https://doi.org/10.3390/stats6040081)
- [Honey Badger Algorithm (Math. Comput. Simul.)](https://www.sciencedirect.com/science/article/abs/pii/S0378475421002901)
- [Marine Predators Algorithm (ESWA)](https://www.sciencedirect.com/science/article/abs/pii/S0957417420302025)
- [Marine predators algorithm: A comprehensive review](https://www.sciencedirect.com/science/article/pii/S2666827023000245)
