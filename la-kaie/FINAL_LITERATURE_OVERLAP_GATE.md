# Final literature overlap gate: "Are successful-step statistics valid evidence of landscape structure?"

Date: 2026-09-30. Scope: a literature investigation only. Nothing was implemented or run,
and N1, D3 and the specification were not modified (§11).

## Evidence tags

| tag | meaning |
|-----|---------|
| [FT] | confirmed from the paper's full text |
| [ABS] | read from the abstract itself |
| [S] | search-engine snippet or summary. These are paraphrases produced by the search tool, **not** the abstract read directly |
| [CODE] | read from the authors' own public code repository (commit hash given). It shows what the code computes, **not** what the paper claims, and it is not [FT] |
| [K] | established knowledge in the field |
| [INF] | my inference |
| [U] | unresolved |

**No statement in this report is tagged [FT] or [ABS].** None of the five papers could be
opened (§2). This gate uses the following tags only:
- [S] for search summaries;
- [CODE] for two public GitHub repositories;
- [K], [INF] and [U].

No page, section, equation, figure or table numbers are given. None could be verified, and
none are reconstructed from memory.

---

## 1. Executive verdict

**Decision: C — ABANDON THIS DIRECTION.**

- **Full-text status.** All five required papers are **UNRESOLVED** at full-text level. The
  environment's network egress policy blocks every host that serves them (§2).
- **Why C is still the decision.** It is an asymmetric argument, and it does not depend on the
  unread full texts:
  - **Coverage by stated results.** The claims that make up the proposed question already
    exist as *stated results* in the prior work:
    - P2 (population covariance ∝ objective covariance at stationarity) [S];
    - P6 (the covariance of selected decision vectors → the inverse Hessian, from isotropic
      mutation plus rank selection) [S][CODE];
    - P5 (the proposal covariance of DE/SOMA in closed form, with a factorisation that
      separates objective-independent variation from fitness-dependent selection, and Monte
      Carlo quantification of selection effects) [S];
    - the structural-bias line (operator-induced bias identified on a landscape-free f0)
      [S][K].
    - Establishing that such a result *exists* does not need the full text. Establishing that a
      method is *absent* does.
  - **Every tool is standard.** The methodological elements the study would add are:
    - population whitening;
    - target residualisation;
    - subspace and covariance similarity;
    - operator ablation;
    - a landscape-free control.

    Each of these is a standard technique.
  - **What the full texts could change.** Reading them could only reveal *more* overlap (for
    example, P5 already comparing selected steps with landscape geometry). It could not turn a
    combination of known techniques into a non-incremental question [INF]. So the gate's
    criterion C applies: "the remaining difference is mainly a combination of known
    techniques."
- **The largest new risk.** It is not in the five papers themselves. P5's first author has
  public code (2026) for iSOMA-AR, an algorithm that **learns its rotation basis from the
  covariance of strictly improving accepted displacements** [CODE]. The same group is therefore
  already building on the exact statistic in question. Whether it tests that statistic's
  validity is [U].
- **The D3 negative result is largely predictable from existing theory.** In brief: greedy
  one-to-one selection with population-scaled, target-directed steps gives a weak selection
  signal, and under contour fitting the landscape information enters through the population
  [INF, §6 Q6].
  - A paper built on that result would most likely be read as an *empirical confirmation /
    synthesis* rather than a new question (§6 Q4).

---

## 2. Full-text access status

**Attempts**

Every host that could serve the full texts, or a legitimate open copy, was tried. This was
done from the shell (curl via the session proxy) and with the WebFetch tool.

**Probe result on 2026-09-30**

- **Blocked:** every host except GitHub returned a CONNECT refusal (HTTP 403 from the egress
  gateway, recorded as `connect_rejected` in the proxy status) or a WebFetch `EGRESS_BLOCKED`
  error. The blocked hosts:
  - arXiv: arxiv.org, export.arxiv.org and the ar5iv mirrors;
  - publishers: www.sciencedirect.com, dl.acm.org, link.springer.com, doi.org;
  - indexes: api.crossref.org, api.openalex.org, api.semanticscholar.org,
    pdfs.semanticscholar.org, scholar.archive.org, web.archive.org, core.ac.uk,
    citeseerx.ist.psu.edu, europepmc.org, www.base-search.net;
  - repositories and preprint sites: zenodo.org, hal.science, osf.io, www.researchgate.net;
  - author or institutional pages: homepages.cwi.nl (Bosman), www.ibspan.waw.pl (Opara),
    www.math.ucdavis.edu, www.mini.pw.edu.pl;
  - summary and discovery services: pith.science, alphaxiv.org, huggingface.co;
  - search engines.
- **Reachable:** GitHub (public repositories via `git clone` and raw.githubusercontent.com).
- **Correction for the record:** my connectivity probe list also included a shadow-library
  domain. It was blocked, nothing was retrieved from it, and such sources must not be used for
  this work.

**Per-paper status**

| paper | full text | best available evidence |
|-------|-----------|-------------------------|
| P5 Novák & Zelinka 2026, arXiv 2607.29228 | **UNRESOLVED** | [S] (several search summaries of the arXiv abstract page); [CODE] from a *companion* repository of the first author (iSOMA-AR, not P5's own code) |
| P2 Opara & Arabas 2019, Swarm Evol. Comput., PII S2210650218301470 | **UNRESOLVED** | [S] |
| P6 Shir & Yehudayoff 2020, TCS 801, arXiv 1806.03674 | **UNRESOLVED** | [S]; [CODE] github.com/ofersh/invHessian @ 07c9e1f (2019-08-27), README: "python3 source code for producing results" |
| P9 Bosman, Grahl & Thierens 2008, PPSN X, LNCS 5199 | **UNRESOLVED** | [S] |
| P12 Vermetten, Kononova, Caraffini, Wang, Bäck, GECCO'21 Companion, DOI 10.1145/3449726.3463218; arXiv 2105.04480 | **UNRESOLVED** | [S] (the authors and the arXiv id come from search results; the Zenodo data/code record 4725245 is blocked) |

To resolve these papers, the user would need to supply the PDFs (e.g. place them in the
repository) or enable arxiv.org in the environment's network policy.

---

## 3. Paper-by-paper evidence (criteria A–L)

For a single-paper verdict, "YES" is used only where the stated result itself establishes the
point. Everything else is PARTIAL, NO (the stated design excludes it) or U.

### P5 — Novák & Zelinka 2026 (highest priority)

**Sources (all [S], search summaries of the arXiv abstract page)**

- "introduces an operator–selection factorization that separates objective-independent
  variation from boundary repair and fitness-dependent selection".
- "derive closed-form expressions for the proposal mean, covariance, expected squared step
  length, expected squared distance from the leader, active dimensionality, and coordinate
  coverage".
- "For canonical DE/rand/1/bin … derive the finite-population moments of differential
  mutation and characterize the additional covariance and coordinate dependence induced by
  forced-coordinate operations".
- "Exact enumeration and Monte Carlo experiments verify the analytical identities and quantify
  the effects of mask conditioning, boundary repair, and fitness-based selection".
- The analysis is carried out "in leader-relative coordinates".
- The geometry is used to construct "Geometry-Controlled SOMA, Rotation-Aware SOMA" and an
  iSOMA extension, which are evaluated on noiseless BBOB.

| crit. | answer | evidence and interpretation |
|-------|--------|-----------------------------|
| A proposal geometry | **YES** [S] | closed-form proposal mean and covariance (SOMA); finite-population moments (DE/rand/1/bin) |
| B population geometry | PARTIAL [S]/[K] | finite-population DE moments are necessarily functions of the population ([K]: Cov(x_r1 − x_r2) = 2·Cov(X) for distinct random pairs, up to finite-population factors). Whether the paper states this explicitly: [U] |
| C successful/selected | PARTIAL [S] | Monte Carlo experiments "quantify the effects of … fitness-based selection". What "selection" means (one-to-one greedy? the accepted-proposal distribution?) and whether *selected-step* covariance is reported: [U] |
| D landscape geometry | **U** | "orientation", "contraction" and "Rotation-Aware SOMA" suggest the geometry is related to coordinate systems and rotated problems [INF]. No comparison with a Hessian or objective contours is visible in the snippets |
| E operator vs landscape | PARTIAL [S] | the factorisation *is* the separation of objective-independent (operator) geometry from objective-dependent (selection) effects, at the conceptual level. Whether it is measured against a landscape reference: [U] |
| F selection effect | PARTIAL [S] | as C |
| G whitening | U | not in the snippets |
| H residualisation | PARTIAL [S] | "leader-relative coordinates" is a target-centred reparameterisation, the SOMA analogue of our operator-target frame. Explicit residualisation: [U] |
| I ablation | PARTIAL [S] | the effects of PRT, t, F and CR, and of geometry-controlled and rotation-aware variants, are studied with the benchmark held fixed |
| J landscape-free control | PARTIAL [INF] | the objective-independent variation analysis is landscape-free *by construction* (analytic). A flat-objective experiment: [U] |
| K measurement validity | **U** | nothing in the snippets |
| L operator taxonomy | PARTIAL [S] | two operator classes (SOMA, DE) under one moment framework |

**Answers to the ten P5-specific questions**

1. Successful/selected-step analysis — PARTIAL [S]: the effect of selection is quantified; its
   form is [U].
2. Proposal variation vs selection — **YES** [S]: this distinction is the central framing of
   the paper.
3. Successful-step covariance vs population covariance — [U].
4. Comparison with objective/landscape geometry — [U].
5. Hessian or inverse-Hessian comparison — [U]; no snippet mentions it.
6. Whitening — [U].
7. Residualisation — PARTIAL (leader-relative coordinates) [S]; explicit residualisation [U].
8. Operator-induced geometric bias — **YES** [S]: covariance and coordinate dependence
   "induced by" masking and crossover.
9. A conclusion that step statistics can mislead as landscape evidence — [U]. The snippets
   frame the geometry as a *design tool*, not as a measurement warning [S].
10. A controlled protocol resembling ours — **cannot be excluded** [U].

**Distinction required by the brief**

- From the snippets, P5 *derives proposal moments and quantifies selection effects*.
- Whether it addresses **successful-step landscape inference** is **UNRESOLVED**.
- I do **not** claim that it does not.

**Companion work (same first author) — raises the risk. Not one of the five required papers.**

- **Source:** V. Novák, "Adaptive Rotation for iSOMA: Geometry, Benchmarking, and Noise
  Robustness in Variational Quantum Objectives", arXiv 2609.37193 [S].
- **What it does** [S]: iSOMA-AR "learns a basis from successful migration displacements".
- **How the code does it** [CODE] (github.com/VojtechNovak/iSOMA-AR @ 3b3496a, 2026-09-26,
  `isoma_ar.py`):
  - Only strictly improving accepted steps are used (`if new_cost < old_cost`).
  - The step is `s = offspring - Migrant`, where the proposal is
    `Migrant + nstep*mask*(Leader - Migrant)`. This is a target-directed operator, the same
    family as HBA's attraction step.
  - The accumulator is updated as `C = (1-eta) C + eta u u^T` with `u = s/|s|`.
  - The rotation is `Q = eigh(C)`, gated by the off-diagonal ratio ρ.
- **Implication:**
  - The statistic D3 found to be population- and target-dominated is being *used* as landscape
    evidence in 2026 work by the P5 group. This makes a measurement-validity question
    practically relevant.
  - It also means the most qualified group to answer it is active in exactly this space.
  - Whether that paper already tests whether C reflects the landscape: [U].

### P2 — Opara & Arabas 2019, "The contour fitting property of differential mutation"

**Source:** [S] only.

- The paper analyses the Differential Mutation Evolutionary Algorithm (DMEA) on a Gaussian
  objective.
- "a normally distributed population remains normal in consecutive iterations"; the
  parameters "can be updated with explicit algebraic formulas".
- For F below a critical value, the population reaches a stable state in which "the covariance
  matrix of a population in a stable state is proportional to the covariance matrix of the
  Gaussian objective function".
- "contour fitting is formalized in terms of correspondence between the covariance matrix of
  the population and the ellipsoidal objective function".

| crit. | answer | evidence and interpretation |
|-------|--------|-----------------------------|
| A proposal geometry | PARTIAL [S] | the differential-mutation distribution under a Gaussian population is implied by the "explicit algebraic formulas" |
| B population geometry | **YES** [S] | population covariance dynamics and a closed-form stationary covariance |
| C successful/selected | U | which selection DMEA uses, and whether selected steps are analysed, is not in the snippets |
| D landscape geometry | **YES** [S] | population covariance ∝ covariance of the Gaussian objective (for a quadratic log-objective, the inverse Hessian [K]) |
| E operator vs landscape | PARTIAL [INF] | the F-critical-value condition separates an operator regime from a landscape-fitting regime |
| F selection effect | U | |
| G whitening | U | |
| H residualisation | U | |
| I ablation | PARTIAL [S] | dependence on F (the operator parameter), landscape fixed |
| J landscape-free control | U | |
| K measurement validity | NO [S] | positive framing: the population *does* fit the contours |
| L operator taxonomy | NO/U | one operator family |

**Relevance.** In a DE-type population, landscape information is carried *by the population
covariance*. So "Δx is closer to the population than to the landscape" is not, by itself,
evidence against landscape information. This is the core of H-b and is prior art [S][INF].

### P6 — Shir & Yehudayoff 2020, "On the covariance–Hessian relation in evolution strategies"

**Source:** [S]; [CODE].

- **Stated result** [S]:
  - The paper studies ES with isotropic Gaussian mutations on positive quadratic objectives,
    with truncation selection.
  - "the covariance matrix over (1,λ)-selected decision vectors becomes proportional to the
    inverse of the landscape Hessian as the population-size λ increases".
  - This "stems only from the usage of isotropic Gaussian mutations and rank-based selection".
  - It generalises an earlier result that assumed sampling near the optimum.
  - Empirical evidence is given for (μ,λ).
- **Code** [CODE] (`generalPoint_fixedCond.py`, `ellipsoidFunctions.py`, `evalCH.py`):
  - **Samples:** `z = np.random.normal(size=N)` (isotropic, centred at 0). The best of λ by
    f(z) = zᵀHz + zᵀa is kept, repeated 10³ times per λ, with λ up to 10⁴.
  - **Landscapes:** axis-aligned, Hadamard-rotated and plane-rotated ellipsoids of fixed
    condition, plus cigar and discus Hessians.
  - **Linear term:** `a = 1/N`, i.e. a general non-stationary point.
  - **Metrics:** ‖HC/max − CH/max‖_F (commutation), the eigenvalue spread of HC
    (`measureID`), and the diagonal and off-diagonal deviations of HC from a scaled identity
    (`e1`, `e2`).

| crit. | answer | evidence and interpretation |
|-------|--------|-----------------------------|
| A proposal geometry | **YES** [CODE] | isotropic N(0, I) proposals; known by construction |
| B population geometry | NO [CODE] | single centre, no population; the operator geometry is isotropic by design |
| C successful/selected | **YES** [S][CODE] | the (1,λ) winner: the best of λ proposals by f. Since the centre is 0, the winner *is* the selected displacement |
| D landscape geometry | **YES** [S][CODE] | explicit Hessian H, compared through HC ∝ I (i.e. C ∝ H⁻¹) |
| E operator vs landscape | PARTIAL [INF] | operator geometry is removed by design (isotropy), so any anisotropy is landscape plus selection. Not a separation of competing sources |
| F selection effect | **YES** [S] | the result is entirely selection-induced; λ controls the strength of selection |
| G whitening | NO [INF] | not needed: the proposal is already white |
| H residualisation | U | whether C is centred on the winners' mean (removing the gradient-driven shift from `a`) is in scripts I did not trace line by line |
| I ablation | PARTIAL [CODE] | varies λ (selection strength), condition and rotation (landscape); the operator is not varied |
| J landscape-free control | NO/U | |
| K measurement validity | NO [S] | the opposite framing: statistical learning of the landscape is "an inherent characteristic" |
| L operator taxonomy | NO | ES only |

**Relevance.**
- P6 is the **positive control** of the proposed study, already proven: with isotropic
  proposals and rank selection, selected-step covariance *is* valid evidence (C ∝ H⁻¹, for
  large λ).
- The proposed "isotropic Gaussian + selection" arm would replicate P6.

### P9 — Bosman, Grahl & Thierens 2008, "Enhancing the Performance of ML Gaussian EDAs Using Anticipated Mean Shift"

**Source:** [S] only.

- The maximum-likelihood mean and covariance are "estimated from selected solutions".
- AMS "removes this inefficiency".
- The earlier gate recorded the snippet phrase that the ML contours on a slope are "aligned in
  the worst way" [S]. It is not re-verified here.
- LNCS 5199, pp. 133–143 [S].

| crit. | answer | evidence and interpretation |
|-------|--------|-----------------------------|
| A proposal geometry | **YES** [S] | the Gaussian sampling model is the proposal |
| B population geometry | **YES** [S] | the model is the ML estimate of the selected population |
| C successful/selected | **YES** [S] | the estimate is from truncation-selected solutions (selection = best fraction by f [K]) |
| D landscape geometry | PARTIAL [S] | relative to the direction of improvement on a slope. No Hessian comparison is visible |
| E operator vs landscape | U | |
| F selection effect | PARTIAL [S]/[K] | selection on a slope shapes and shrinks the estimated covariance. This is the premise for AMS and variance scaling |
| G whitening | U | |
| H residualisation | PARTIAL [INF] | AMS acts on the mean shift, the analogue of the "target" component |
| I ablation | PARTIAL [S] | with vs without AMS / variance scaling |
| J landscape-free control | U | |
| K measurement validity | PARTIAL [INF] | argues that selected-sample geometry is a *misleading search model* on slopes, a bias argument framed for search efficiency, not measurement |
| L operator taxonomy | NO | Gaussian EDA only |

### P12 — Vermetten, Kononova, Caraffini, Wang, Bäck 2021, "Is there anisotropy in structural bias?"

**Source:** [S] only.

- "an algorithm that is not isotropic would intuitively be considered structurally biased,
  there have been cases where algorithms appear to only show SB in some dimensions".
- "anisotropy is very rare, and even in cases where it is present, there are clear tests for
  SB which do not rely on any assumptions of isotropy".
- A related work by the same line is "Emergence of Structural Bias in Differential Evolution",
  arXiv 2105.04693 [S] (not read).

| crit. | answer | evidence and interpretation |
|-------|--------|-----------------------------|
| A proposal geometry | NO [K] | structural-bias (SB) tests analyse final best positions, not steps |
| B population geometry | NO/U | |
| C successful/selected | PARTIAL [K] | the final best-so-far positions on f0 are selected points, not displacements |
| D landscape geometry | NO [K] | f0 carries no landscape by construction |
| E operator vs landscape | **YES** [K]/[S] | SB is defined as operator-induced bias isolated on a landscape-free objective |
| F selection effect | U | |
| G whitening | NO/U | |
| H residualisation | NO/U | |
| I ablation | U | |
| J landscape-free control | **YES** [K]/[S] | f0 (uniform random objective) is the SB method |
| K measurement validity | PARTIAL [INF] | SB means optimizer outputs carry algorithm-induced structure unrelated to the objective. It is a validity warning for *final positions*, not for step statistics |
| L operator taxonomy | PARTIAL [K] | SB studies test many algorithms under one protocol; P12's exact set is [U] |

---

## 4. Exact overlap with the proposed study

Mapping each component of the proposed study to prior work:

| proposed component | covered by | residual novelty |
|--------------------|------------|------------------|
| isotropic Gaussian + selection arm on known-Hessian quadratics | P6 (proven plus numerics) [S][CODE] | none: this replicates P6's positive result |
| DE/rand/1 arm: Δx geometry ∝ population, population ∝ contours | P2 (stationary theory) [S], P5 (finite-population moments) [S], DE contour fitting [K] | the *selected*-step covariance under one-to-one DE selection vs H⁻¹ may not be tabulated [U] |
| attraction-to-best / HBA arm: Δx ≈ target direction | elementary algebra [K]; P5's leader-relative SOMA analysis is the closest formal treatment [S] | a quantified successful-step version [INF], small |
| MPA/FAD arm | FAD = DE differences (earlier gate) [S][K] | algorithm-specific |
| population whitening | standard statistics [K] | none as a technique |
| target residualisation | standard regression [K]; leader-relative coordinates in P5 [S] | none as a technique |
| subspace / covariance similarity | standard [K]; P6 uses commutator and eigen-spread metrics [CODE] | none |
| operator ablation, landscape fixed | P5 (parameter and variant study) [S]; common practice [K] | none as a technique |
| landscape-free control | SB line (f0), P12 [S][K] | applying it to *step covariance* rather than final positions: [U]/[INF] |
| temporal analysis | common [K] | none |
| "measurement-validity" framing | P9 (a bias framing for search), SB (a bias framing for outputs), ELA sampling sensitivity (arXiv 2006.11135 [S]); trajectory-based ELA (arXiv 2102.05370 [S]) uses optimizer samples as landscape evidence | the framing *for successful-step covariance* specifically: not found [S], but see iSOMA-AR [CODE] |

---

## 5. Strict novelty matrix

Values are YES / PARTIAL / NO / UNRESOLVED. The evidence tag for each value is in §3.

| # | capability | P2 | P5 | P6 | P9 | P12 | Existing D3 | Proposed study |
|---|------------|----|----|----|----|-----|-------------|----------------|
| 1 | population covariance analysis | YES | PARTIAL | NO | YES | NO | YES | YES (planned) |
| 2 | displacement/proposal covariance | PARTIAL | YES | YES | YES | NO | YES | YES (planned) |
| 3 | successful-step analysis | UNRESOLVED | PARTIAL | YES | YES | PARTIAL | YES | YES (planned) |
| 4 | selection-conditioned analysis | UNRESOLVED | PARTIAL | YES | YES | UNRESOLVED | PARTIAL | YES (planned) |
| 5 | known-Hessian landscape | YES | UNRESOLVED | YES | PARTIAL | NO | YES | YES (planned) |
| 6 | population-vs-landscape comparison | YES | UNRESOLVED | NO | PARTIAL | NO | YES | YES (planned) |
| 7 | operator-vs-landscape separation | PARTIAL | PARTIAL | PARTIAL | UNRESOLVED | YES | PARTIAL | YES (planned) |
| 8 | whitening | UNRESOLVED | UNRESOLVED | NO | UNRESOLVED | NO | YES | YES (planned) |
| 9 | residualisation | UNRESOLVED | PARTIAL | UNRESOLVED | PARTIAL | NO | YES | YES (planned) |
| 10 | operator ablation | PARTIAL | PARTIAL | PARTIAL | PARTIAL | UNRESOLVED | NO | YES (planned) |
| 11 | landscape-free control | UNRESOLVED | PARTIAL | NO | UNRESOLVED | YES | NO | YES (planned) |
| 12 | temporal analysis | PARTIAL | UNRESOLVED | NO | UNRESOLVED | NO | YES | YES (planned) |
| 13 | multiple canonical operators | NO | PARTIAL | NO | NO | PARTIAL | NO | YES (planned) |
| 14 | successful-step measurement validity | NO | UNRESOLVED | NO | PARTIAL | NO | PARTIAL | YES (planned) |
| 15 | explicit negative-evidence claim | NO | UNRESOLVED | NO | PARTIAL | PARTIAL | YES | YES (planned) |
| 16 | cross-operator taxonomy | NO | PARTIAL | NO | NO | PARTIAL | NO | YES (planned) |

**Notes on the matrix**

- **P2 row 12:** PARTIAL because the paper tracks the iteration-wise update of the population
  distribution [S].
- **D3 row 4:** PARTIAL because attempts were logged but the analyses used successes.
- **D3 row 7:** PARTIAL because it is correlational only; D3 explicitly states "Causal: not
  established".
- **Proposed-study column:** "YES (planned)" means intended. None of it has been done.

**Reading the matrix**

- Rows 1–7 are each covered by at least one prior paper at YES level.
- Rows 8–12 are standard techniques.
- The only rows where no prior paper reaches YES *and* the column is not a mere technique are:
  - 14 (successful-step measurement validity);
  - 15 (explicit negative claim);
  - 16 (cross-operator taxonomy).
- Of these three, P5 is UNRESOLVED on 14 and 15, and PARTIAL on 16.

---

## 6. What is genuinely new, if anything (Q1–Q6)

**Q1. Is the exact question already answered?**

- **As a single cross-operator study:** not found [S], and not excludable, because P5 is [U].
- **Component by component:**
  - The positive case is answered: with isotropic proposals plus rank selection, successful
    steps are valid evidence (C → H⁻¹) (P6 [S][CODE]).
  - The DE case is largely answered: population ∝ contours, and steps ∝ population (P2 [S],
    P5 [S], [K]).
  - The target-directed case is elementary [K].
- So the question is **mostly answered in pieces** [INF].

**Q2. What exact element is missing?**

A single **decomposition protocol**:
- Take one known-Hessian landscape.
- Compare the covariance of *successful* steps against three references:
  1. the proposal covariance (operator);
  2. the population covariance (state);
  3. H⁻¹ (landscape).
- Measure the *selection increment* in population-whitened coordinates.
- Apply this across operators whose proposals are:
  - isotropic (P6);
  - population-scaled (DE);
  - target-directed (HBA/PSO-like).
- Include a landscape-free control.

As far as the snippets show, no one reports the *selection increment beyond population
geometry* as a validity measure [S][INF].

**Q3. Meaningful, or a combination of standard techniques?**

- It is **mainly a combination of standard techniques**. Each measurement (whitening,
  residualisation, subspace similarity, ablation, f0 control) is textbook.
- The operators' geometry is known in closed form (P5, P2) or elementary.
- The one open quantity is the size of the selection increment under **one-to-one greedy
  selection** (λ = 1 per parent). Its qualitative answer is predictable (Q6) [INF].

**Q4. How would a reviewer most likely describe it?**

- A **synthesis plus empirical confirmation**.
- Possibly a *methodological extension* (a diagnostic protocol).
- Not a new scientific question [INF].
- The specific risk is that a reviewer points to P6 + P2 + P5 and asks what is learned that
  those do not imply.

**Q5. Does generalising from HBA/MPA to canonical operators create novelty?**

- It **makes the paper broader, not newer** [INF].
- The canonical arms are exactly the settings where theory already exists:
  - isotropic → P6;
  - DE → P2/P5;
  - attraction → elementary.
- Generalising therefore moves the work *toward* known results.
- HBA/MPA were the only arms without theory, and results specific to them are the
  "algorithm-specific observation" that the brief itself rules out as the intended claim.

**Q6. Could the negative result be a contribution? What would have to be shown beyond known
theory?**

**What theory already predicts** [INF from P6 and P2 + [K]]:
- (a) With one-to-one greedy selection, each successful step is one draw truncated by
  f(x + Δx) < f(x). For a step small relative to the curvature scale, this is a half-space
  cut along the gradient. It carries gradient (first-order) information, and Hessian
  information only at second order.
  - P6's H⁻¹ result needs large λ, i.e. strong rank selection among many proposals.
- (b) When proposals are population-scaled (DE) or target-directed (HBA), the proposal
  covariance dominates the successful-step covariance unless selection is strong.
- (c) Under contour fitting, landscape information is present, but *through* the population.
  So whitening by the population is expected to remove it.

The D3 findings match (a)–(c).

**To go beyond theory, the work would have to show at least one of:**
- (i) a *quantitative* law for the selection increment, e.g. its dependence on step scale
  relative to curvature, and on λ-equivalent selection strength, that is not a corollary of
  P6;
- (ii) a case where the prediction *fails*, e.g. a successful-step statistic that is
  population-dominated even in P6's regime, or landscape-informative in the greedy regime;
- (iii) evidence that published algorithms which *learn from successful displacements*
  (iSOMA-AR [CODE]; also the rank-μ update in CMA-ES [K]) acquire population/operator
  geometry rather than landscape geometry, with a measurable performance consequence.

(i) is theory work with an uncertain outcome. (ii) is speculative. (iii) is a different
project: an evaluation of a design principle, not a measurement-validity paper. I do not
recommend any of them here.

---

## 7. What is already known [S][K], with gaps marked

- **Selected-step covariance and the Hessian (ES).** With isotropic Gaussian mutation plus
  (1,λ) rank selection on quadratics, selected-step covariance → ∝ H⁻¹ as λ grows. This is
  proven at a general (non-optimal) point and validated numerically [S][CODE] (P6).
  - Its roots are an older hypothesis, and FOCAL (Shir et al., arXiv 1112.4454 [S]) exploits
    it.
- **DE contour fitting.** In DE-type mutation on Gaussian objectives, the population stays
  Gaussian and its stationary covariance ∝ the objective covariance (P2 [S]).
  - Difference vectors inherit population geometry ([K], P5 [S]).
- **DE and SOMA proposal moments.** The proposal mean and covariance, and the coordinate
  dependence induced by masking and crossover, are available in closed form. Their
  separation from fitness-dependent selection is formalised (P5 [S]).
- **Gaussian EDAs.** Selected-sample ML covariance is shaped by selection, and can be
  misaligned with the useful direction on slopes (P9 [S]).
- **Structural bias.** Operator-induced bias is isolated with a landscape-free objective, and
  its anisotropy is rare (P12 and related work [S][K]).
- **ELA sampling.** ELA features are strongly sensitive to the sampling strategy (arXiv
  2006.11135 [S]).
  - Trajectory-based ELA reuses optimizer samples as landscape evidence (arXiv 2102.05370
    [S]).
  - Authors are not listed because the snippets did not reliably give them.

---

## 8. KBS-level assessment

This is not a prediction of any editorial outcome.

| criterion | assessment |
|-----------|------------|
| 1 clear methodological contribution | **Weak.** A decomposition protocol assembled from standard techniques (§6 Q3) |
| 2 generalizable knowledge contribution | **Weak.** The general statements (selection learns H⁻¹; DE fits contours; target steps follow targets) are already published or elementary |
| 3 more than an algorithm-specific observation | **Weak.** The part not covered by theory is the HBA/MPA part, which is algorithm-specific |
| 4 rigorous falsifiability | **Adequate.** Known-Hessian landscapes plus ablations plus f0 controls are falsifiable. But most outcomes are predicted in advance (§6 Q6), so falsification tests known theory more than a new hypothesis |
| 5 useful implication | **Moderate.** It is relevant to algorithms that adapt from successful displacements (iSOMA-AR [CODE], N1's own premise) and to trajectory-based ELA. This is the strongest item, but it points to a different paper (§6 Q6(iii)) |
| 6 novelty beyond existing theory | **Weak.** See §5: rows 14–16 only, with P5 unresolved on them |
| 7 coherent full-article story | **Weak to moderate.** "A statistic can be biased; here is why theory already says so; here is confirmation" is coherent but thin for a full research article [INF] |

Five of seven items are weak. A full article would rest on items 4 and 5.

---

## 9. Final decision

**C — ABANDON THIS DIRECTION**

**Supporting evidence** (strongest first):
1. The positive case is proven: P6 [S][CODE].
2. DE contour fitting: P2 [S].
3. Proposal moments plus the operator–selection separation: P5 [S].
4. Landscape-free operator-bias methodology: P12 and the SB line [S][K].
5. Every remaining element is a standard technique [K].
6. The D3 negative result is predictable from 1–2 plus elementary selection arguments [INF].
7. An active competing use of the statistic by the P5 group: iSOMA-AR [CODE].

**Why not B.**
- B requires "a potentially meaningful gap".
- The unread full texts can only confirm or enlarge the overlap. They cannot create a
  non-incremental question out of a combination of known techniques [INF].
- Reading them would change the confidence in C, not its direction.

**Confidence and caveats.**
- The verdict rests on [S] evidence (search summaries) and [CODE]. No [FT].
- One scenario would weaken C: if the full texts of P2 and P5 showed that the stated results
  hold only under assumptions so narrow that the greedy-selection regime is untouched **and**
  nobody has analysed selected steps under population-scaled proposals.
- Even then, the remaining work is §6 Q6(i): theory with an uncertain outcome, not the
  empirical study proposed.

---

## 10. Reframing

Not applicable. The decision is C, not B.

**Recorded for completeness only, not a recommendation.**

If this line were ever reopened, it would be as a *different* question with its own literature
gate. That question would be §6 Q6(iii):

> "Do optimisers that adapt their search basis from successful displacements (e.g. iSOMA-AR,
> rank-μ CMA-ES updates) learn landscape geometry or their own population/operator geometry?"

Before any design work, that gate would need:
- the full texts of arXiv 2609.37193 and 2607.29228;
- the CMA-ES learning-rate / rank-μ literature.

---

## 11. Implementation status

| item | status |
|------|--------|
| N1 changed? | **NO** |
| D3 changed? | **NO** |
| experiments run? | **NO** |
| new seeds? | **NO** |
| code changed? | **NO** (this report is the only new file in the repository; two public third-party repositories were cloned read-only into the session scratchpad for inspection and not added to the repository) |

**IMPLEMENTATION STATUS: FROZEN PENDING LITERATURE DECISION**

**Sources consulted** (search results and repositories; none read in full text):
- https://arxiv.org/abs/2607.29228 [S]
- https://arxiv.org/abs/2609.37193 [S]
- https://github.com/VojtechNovak/iSOMA-AR [CODE]
- https://arxiv.org/abs/1806.03674 [S]
- https://www.sciencedirect.com/science/article/pii/S0304397519305468 [S]
- https://github.com/ofersh/invHessian [CODE]
- https://www.sciencedirect.com/science/article/abs/pii/S2210650218301470 [S]
- https://link.springer.com/content/pdf/10.1007/978-3-540-87700-4_14.pdf [S]
- https://dl.acm.org/doi/10.1145/3449726.3463218 [S]
- https://arxiv.org/abs/2105.04480 [S]
- https://zenodo.org/records/4725245 [S]
- https://arxiv.org/abs/2105.04693 [S]
- https://arxiv.org/abs/1112.4454 [S]
- https://arxiv.org/html/2006.11135 [S]
- https://arxiv.org/pdf/2102.05370 [S]
