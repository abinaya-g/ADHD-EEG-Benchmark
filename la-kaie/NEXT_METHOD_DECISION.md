# Next-method decision (novelty investigation) — 2026-09-28

Status: **analysis only.** No method has been implemented, no experiment run
and no manuscript text written. LA-KAIE is unchanged and not pursued.
Companion files: `literature/DETAILED_NOVELTY_MATRIX.md` (full matrix and
evidence tags), `RESEARCH_DECISION_REPORT.md`, `results/PILOT_REPORT.md`,
`literature/NOVELTY_ASSESSMENT.md`, `reference/MPHBS_NOTES.md`.

## 0. Evidence limitations — critical

The request was to read full texts. **That was not possible for any paper
except MPHBS.** The network policy blocks arXiv, Springer, ScienceDirect
full text, IEEE Xplore, ACM DL, Zenodo (including RVIntX's replication
package), ResearchGate, OpenAlex, Semantic Scholar and Crossref.
WebFetch mirrors (Bytez, alphaXiv) are blocked too. Only GitHub (git protocol)
and web search are reachable. What was verified at the level of mechanism:

* **MPHBS**: full paper and source code.
* **GAwLL** (Tinós, Przewozniczek, Whitley, Chicano, GECCO 2023): the authors'
  source code (`github.com/rtinos/GAwLL`), which is the published precursor
  of RVIntX's online linkage learning by the same group.

Everything else is from abstracts and search snippets. Where a conclusion
depends on an unverified detail, it is marked **UNVERIFIED**. Before any
submission, the full texts of RVIntX, LEL, ACoS, HLX and MFEA-II must be read
(ideally from a network without these restrictions, or supplied as PDFs).

---

## 1. F1 — Outcome-Based Co-Transfer Epistasis Learning (OB-CEL)

**Closest work.** RVIntX (Tinós, Chicano, Przewozniczek, BRACIS 2025), its
precursor GAwLL (verified in code), ILS with linkage learning, and LEL
(Unlu, arXiv 2604.04273, 2026: "sparse interaction graph from successful
optimization steps").

**Key finding (derivation, independent of full-text access).** For donor
substitution, the co-transfer effect OB-CEL wants to learn is the
second-order non-additivity of the transfer function
g(m) = f((1 − m) ⊙ x_r + m ⊙ x_d):

    γ_ij = g(1_i + 1_j) − g(1_i) − g(1_j) + g(0).

This is exactly the quantity empirical linkage learning measures. GAwLL
computes abs(f(x_gh) − f(x_h) − f(x_g) + f(x)) for constructed perturbations
(verified in `gawll.cpp: mutationLL`). OB-CEL would only replace the
*estimator* (passive regression over random masks, including failures) and
the *setting* (two heterogeneous populations). LEL already learns an
interaction graph from successful steps (estimator UNVERIFIED).

**RVIntX questions 1–9:** answered in `DETAILED_NOVELTY_MATRIX.md` §G. Short
version: online, pairwise, used to build recombination masks, single
population. Its signal in the precursor is constructed perturbations, not
recombination outcomes (RVIntX itself UNVERIFIED). It learns the same
quantity as OB-CEL, so the **overlap is substantial**.

**Verdict:** not a defensible core contribution. A new estimator of an
already-targeted quantity is an implementation difference. The pilot's
structural diagnostics also suggest little sparse pairwise structure to learn
on CEC2022 F1–F5 (dense rotations).

## 2. F2 — Frame-Adaptive Cross-Population Exchange (FACE)

**Already known:**
* Eigen-coordinate crossover: DE/eig (Guo & Yang, TEVC), CoBiDE, CPI-DE,
  eigen-crossover jSO, rank-one eigen crossover, and MODE-DCML (KBS 2017).
* **Adaptive choice between the original and eigen frames**: ACoS (IEEE TCYB
  2018), using a selection-probability vector updated from offspring.
* **Covariance learned across multiple populations to set the crossover
  frame**: MCDE.
* Subspace alignment between populations: EMT (AT-MFEA, SETA-MFEA).

**Verdict:** DIRECT overlap on every defining element: eigen frame,
adaptive frame choice, and a multi-population frame. The only remaining
element is heterogeneous source optimisers, which is a setting, not a
mechanism. **Rejected.**

## 3. F3 — Progress-Aligned Evaluation Allocation (PEA)

**Already known:**
* Adaptive allocation of offspring or evaluations among heterogeneous search
  processes: AMALGAM, MOS, PAP.
* Allocation by **contribution to the best overall objective**:
  contribution-based cooperative coevolution (Omidvar et al.; CBCC3, CCFR3,
  bandit-based CC). This is PEA's credit.
* Budget split between global and local search: MA-LS-Chains, adaptive
  local-search depth.
* Adaptive timing of exchange: migration-interval adaptation (Evol. Comput. 2015).
* Extreme-value and rank-based credit with bandits: AOS, FRRMAB.

**Verdict:** DIRECT overlap on the allocation principle and on the credit
principle. Applying it to "native vs exchange" channels in an HBA–MPA hybrid
is a new *instance*, not a new method. **Rejected** as a core contribution.
It may still be useful as a controlled experimental factor: fixing the
exchange budget removes a confound.

## 4. Closest competing papers (per candidate)

| candidate | closest papers | overlap |
|---|---|---|
| F1 | RVIntX; GAwLL (verified); LEL; HLX; VIGPSO; RV-GOMEA fitness-based LL | DIRECT/PARTIAL |
| F2 | ACoS; MCDE; DE/eig; CoBiDE; CPI-DE | DIRECT |
| F3 | AMALGAM; MOS; CBCC family; migration-interval adaptation; FRRMAB | DIRECT |
| N1 (below) | ACoS (adaptive representation choice); MFEA-II (likelihood-driven transfer control); FRRMAB/dynamic MAB (bandit over options); GOMEA/HLX/RVIntX (fixed linkage hypothesis) | PARTIAL / CONCEPTUAL PRECEDENT |

Recent *Knowledge-Based Systems* papers found (KBS status inferred from the
ScienceDirect PII prefix; verify before citing): QLOBLINFO (2025, CEC2022),
RL-LGWO (2025), competitive many-task DE with RL and meta-knowledge transfer
(2025), knowledge-transfer DE for nonlinear equation systems (2023, 2024),
dynamic multi-knowledge EA (2025), MODE-DCML (2017). None combines
structural-hypothesis testing with heterogeneous-population exchange. With
snippet-level access, that is only a statement about this search.

## 5. Detailed novelty matrix

See `literature/DETAILED_NOVELTY_MATRIX.md`. It covers 30+ methods with all
requested columns, the DIRECT / PARTIAL / CONCEPTUAL PRECEDENT / DISTINCT
labels per candidate, and an evidence tag per row.

Criteria A–H:

| criterion | F1 | F2 | F3 | N1 | N2 | N3 |
|---|---|---|---|---|---|---|
| A distinguishable from MPHBS | yes | yes | yes | **yes** (MPHBS becomes a nested special case) | yes | yes |
| B distinguishable from the closest prior work | **no** (same target quantity as empirical LL) | **no** | **no** | **partially**: rests on evidence-based vs success-based selection (ACoS) and within-task structural hypotheses vs inter-task transfer (MFEA-II); must be verified | weak (mating restriction) | weak (EMT negative transfer) |
| C mathematically meaningful | yes | yes | yes | yes (prequential model comparison) | yes | yes |
| D testable hypothesis | yes | yes | yes | **yes, including a non-performance structural-identification test** | yes | yes |
| E evaluable on CEC2022 and FIR | yes | yes | yes | yes (both have known, different structure) | yes | yes |
| F no unfair extra FEs | yes | yes | yes | yes (0 extra FEs) | yes | yes |
| G suitable for KBS | low | low | low | **moderate** (it produces explicit problem-structure knowledge) | low | low |
| H methodological, not a parameter change | borderline | no | no | **yes** | yes | yes |

## 6. Methodological gap

None of F1–F3 is sufficiently novel. What remains between MPHBS and the
linkage-learning, frame-adaptive and multi-population literature:

> **Every information-exchange method commits to a structural hypothesis
> about *how* information can be transferred without destroying it.**
> MPHBS and binomial crossover assume coordinate separability (independent
> per-dimension reuse). GOMEA, HLX and RVIntX assume a sparse linkage
> structure learned by MI or perturbation. DE/eig, CoBiDE and CPI-DE assume
> a rotated quadratic-like structure (an eigenbasis). Migration assumes
> whole-solution transfer. Methods that adapt between options (ACoS, AOS
> bandits) do so by *success rate*, i.e. by which operator happened to
> produce improvements, not by testing *which structural hypothesis explains
> transfer outcomes*. No method was found that treats these as **competing
> hypotheses**, scores them by **predictive evidence on the same exchange
> data** at zero extra FEs, and returns the posterior as explicit knowledge
> about the problem's transfer structure.

MPHBS states the relevant limitation itself: dimension-wise reuse "may be
less effective in … highly epistatic" landscapes, with "strong variable
coupling" as a threat to validity (Sec. 3.4, Sec. 7). It does not test its
separability assumption. Our structural diagnostics show the two benchmark
domains really differ here. CEC2022 F1–F5 have dense rotations (80–84 %
non-negligible off-diagonal entries). FIR's exact stopband coupling is
banded and sparse. This is a documented property of the benchmark
*definitions*, not of any method's performance numbers.

**Integrity note.** The gap was identified while analysing why the pilot
mechanism was inactive. The structural facts it rests on come from the
benchmark definitions (rotation matrices, the FIR quadratic form), not from
any method's scores, and nothing below was tuned on pilot results.

## 7. Candidate direction (conditional; not a final selection)

### N1 — Evidence-based structural-hypothesis selection for cross-population exchange (working name **SHE**, provisional)

This is the only candidate meeting criteria A and C–H, and B **partially**.
It is offered for your decision, conditional on full-text verification of
ACoS, MFEA-II, LEL and RVIntX. If any of them already scores representations
by predictive evidence on shared exchange data, N1 is not novel either.

Two weaker alternatives, for completeness (not recommended):
* **N2: donor–receiver compatibility learning** (predicting which donor–receiver
  pairs lie in compatible basins). Precedents: mating restriction,
  assortative mating, partition crossover's common-variable hyperplanes.
* **N3: sequential-testing (racing) of transferred information before memory
  injection** to suppress negative transfer. Precedent: MFEA-II and the EMT
  negative-transfer literature.

## 8. Proposed mathematical formulation (N1)

**Exchange event.** Receiver x_r ∈ P^a and donor x_d ∈ P^b, with
a, b ∈ {HBA, MPA} and a ≠ b. A representation h has an invertible coordinate
map T_h and a family of admissible masks 𝓜_h. The candidate is

    x' = clip( T_h⁻¹[ (1 − m) ⊙ T_h(x_r) + m ⊙ T_h(x_d) ] ),  m ∈ 𝓜_h,

costing exactly one FE. The outcome is y = 1[f(x') < f(x_r)], with a
rank-gain variant for ordinal credit.

**Structural hypotheses** H = {h_0, h_1, h_2, h_3}:
* h_0 whole-vector (migration): T = I, m = 1.
* h_1 coordinate-separable (MPHBS's assumption): T = I, per-coordinate masks.
* h_2 sparse linkage: T = I, masks = unions of groups of a linkage graph
  E_t learned from outcomes (below).
* h_3 rotated: T(x) = Bᵀ(x − μ), with B the eigenvectors of the pooled-elite
  covariance of both populations; per-component masks.

**Predictive models on shared data.** Every exchange e (whatever representation
generated it) is recorded as the displacement δ_e = x'_e − x_{r,e}. Each
hypothesis h has a predictive model of the outcome from its own features
z_e^h = S_h⁻¹ T̃_h δ_e (T̃_h the linear part of T_h, S_h per-coordinate scale):

    h_1, h_3:  logit p_h(y=1|δ) = θ_0 + Σ_k [a_k z_k + b_k z_k²]                  (additive in its frame)
    h_2:       logit p_h(y=1|δ) = θ_0 + Σ_k [a_k z_k + b_k z_k²] + Σ_(k,l)∈E_t c_kl z_k z_l
    h_0:       logit p_h(y=1|δ) = θ_0 + a·‖z‖ + b·rankgap(x_d, x_r)

Each is fitted online (L2-regularised logistic regression, forgetting λ).
Covariate shift between the generating representation and h is handled by
importance weights w_e = 1/π_t(h_gen(e)).

**Evidence.** The prequential (predict-then-update) log-loss of h is

    L_h(t) = Σ_{e ≤ t} ρ^{t−e} · w_e · [ −log p_h^{(e−1)}(y_e | δ_e) ],

and the posterior-like weights are

    π_t(h) ∝ π_0(h) · exp(−η L_h(t)).

**Decision.** For each exchange, draw h ~ π_t (Thompson-style) with a floor
π_min per hypothesis to keep the evidence estimable. Build m ∈ 𝓜_h:
binomial over coordinates (h_1, h_3), a random union of groups (h_2), or all
(h_0). Evaluate once, record, update every model.

**Linkage for h_2, learned without extra FEs.** E_t is the set of pairs whose
c_kl in the h_2 model is significant under a sparse (L1) fit on the pooled
outcomes. The linkage estimate is *part of a competing hypothesis* rather
than an assumption.

**Output as knowledge.** The trajectory π_t(h) is an interpretable,
per-instance estimate of which transfer structure explains outcomes. It can be
validated against known ground truth (CEC rotation matrices, the FIR
quadratic form).

## 9. Proposed algorithmic architecture

```
initialise HBA and MPA populations (2N FEs); prior π_0 uniform; models empty
for t = 1 .. until budget:
    Phase 1  (unchanged backbone): HBA move+greedy (N FEs); MPA move, eval, memory (N FEs)
    FADs     (fixed placement, identical for every compared variant)
    every U iterations: refresh B (pooled-elite covariance eigenbasis); refit sparse linkage E_t
    Exchange (fixed E evaluations per iteration, equal to MPHBS's E):
        for e = 1..E:
            choose receiver/donor (fixed rule, e.g. rank-biased, one from each population)
            h ~ π_t (with floor π_min)
            m ~ 𝓜_h ; x' = T_h⁻¹ mix ; evaluate x' (1 FE)
            greedy replacement of x_r (MPA memory synchronised)
            for every hypothesis g in H: accumulate prequential loss on (δ, y), update model g
            update π_t
    record π_t, model coefficients, E_t, diagnostics
```

Design choices made **on purpose**, so that only the novel element varies:
* the exchange budget E per iteration is fixed (no allocation learning, since
  F3 is known), and set equal to MPHBS's E;
* no landscape-feature controller (LA-KAIE's controller is known and did not help);
* the donor/receiver rule is identical across all variants.

## 10. Required ablations

1. **Fixed-hypothesis variants:** h_0 only, h_1 only (an MPHBS-like separable
   exchange with the same budget), h_2 only, h_3 only.
2. **Uniform random hypothesis** (same floor, no evidence).
3. **Success-rate selection (ACoS-like):** π updated by the success rate of
   the exchanges each h *generated*, instead of predictive evidence on shared
   data. This is the key ablation for criterion B.
4. **No data sharing:** evidence computed only from the exchanges each h generated.
5. **Leave-one-hypothesis-out** (drop h_2; drop h_3).
6. **Linkage source for h_2:** outcome-learned (default), random placebo
   graph, gray-box FIR graph (supplementary, clearly labelled).
7. **No importance weighting.**
8. MPHBS (B-C1 / A-C1 as published) and MPHB, HBA, MPA as external references.

## 11. Expected computational cost

* Extra FEs: **0** (one FE per exchange, the same budget as MPHBS's mediator).
* Per exchange: T_h transforms O(D²) for h_3 and O(D) otherwise. Model
  prediction and update for |H| = 4 models: O(D) each for h_0, h_1, h_3 and
  O(D + |E_t|) for h_2.
* Every U iterations: eigendecomposition O(D³) (D = 20 / 31), and an L1 fit
  for E_t in O(n_recent·(D + D²)) with a bounded window.
* Memory O(D² + window·D).
* Expected overhead: comparable to or below the MPHBS mediator, which
  generates K candidates per episode with Python-level loops.

## 12. Risks

1. **Novelty (main risk):** the claim rests on *evidence-based* versus
   *success-based* representation selection. If ACoS or a successor already
   uses predictive or model-based scoring, or if reviewers consider the
   distinction cosmetic, the contribution collapses to "ACoS with more
   options". Ablation 3 must show a measurable *behavioural* difference, not
   just a performance one.
2. **Weak signal:** exchange success was about 10 % in the pilot, so the
   evidence may not separate hypotheses within the budget, and π may stay
   near uniform. This is measurable with the structural-identification test.
3. **Covariate shift and non-stationarity:** hypotheses are evaluated on data
   they did not generate, and the landscape changes as the population
   converges; importance weights may have high variance.
4. **Eigenbasis degeneration** as the population converges (seen in the pilot).
5. **Benchmark predictions may fail:** e.g. FIR may not favour h_2. That is a
   legitimate negative result and must be reported.
6. **No performance guarantee:** the method may not beat MPHBS, and that
   must not trigger post-hoc redesign.

**Testable hypotheses, pre-registered before any implementation:**
* **H-S1 (identification, non-performance):** on synthetic functions with
  known structure (separable ellipsoid, block-coupled, rotated ellipsoid),
  the time-averaged π concentrates on h_1, h_2 and h_3 respectively. On
  CEC2022 F1–F5 (dense rotations), mean π(h_3) > π(h_1).
* **H-S2 (decision relevance):** evidence-based selection differs
  behaviourally from success-rate selection (ablation 3), measured by the
  divergence of the representation-choice distributions, and is not worse
  in final quality.
* **H-P (performance, secondary):** at equal FEs, SHE is not worse than the
  best fixed-hypothesis variant chosen post hoc per domain. It is compared
  with MPHBS and every result is reported regardless of outcome.

## 13. Exact reason why N1 differs from MPHBS

MPHBS **assumes** coordinate separability. Its learned object Q(d, n, ch) is
a per-dimension, additive credit over (rank, source), and it never questions
whether per-dimension transfer is the right unit. N1 makes the transfer
*representation* the object of inference: separability is one hypothesis,
h_1, competing with sparse linkage, a rotated frame and whole-vector
transfer. The winner is decided by predictive evidence on the exchange
outcomes MPHBS would also generate, at zero extra FEs. MPHBS corresponds to
the degenerate case π ≡ δ(h_1) with a different donor-selection rule. N1 also
outputs explicit, checkable structural knowledge, which MPHBS does not.

## 14. Exact reason why N1 differs from the closest prior work

* **ACoS (closest):** chooses between two frames (original vs eigen) for a
  single population, with probabilities updated from **offspring
  success**. N1 differs in (i) the selection criterion: predictive evidence
  of structural models on **shared** outcome data, rather than the success
  rate of the operator that generated the data; (ii) the hypothesis set,
  which includes outcome-learned sparse linkage and whole-vector transfer;
  (iii) inter-population exchange between heterogeneous optimisers.
  **(i) is the load-bearing difference and must be verified against the
  ACoS full text.**
* **RVIntX / GAwLL / HLX / GOMEA:** commit to a linkage hypothesis and learn
  it from constructed perturbations (verified for GAwLL), DG evaluations
  (HLX) or population MI (GOMEA). N1 does not assume linkage; it tests it
  against alternatives using outcomes only.
* **MFEA-II:** uses likelihoods of probabilistic population models to set
  transfer *intensity* between *tasks*. N1 uses predictive likelihood of
  transfer outcomes to choose the transfer *representation* within one task.
* **AOS / FRRMAB / LA-BHH:** select operators by performance-based credit.
  N1's credit is the explanatory power of the hypotheses, not the reward of
  the operator.

---

**No direction is adopted by this document.** Recommended next actions,
awaiting your instruction:
1. Obtain the full texts of ACoS (TCYB 2018 / arXiv 1703.06263), MFEA-II
   (TEVC 2019), LEL (arXiv 2604.04273) and RVIntX (BRACIS 2025), e.g. as PDFs
   you place in `reference/`, and verify §14.
2. Decide between N1, one of the weaker alternatives, or F4 (repositioning
   as an empirical/reproduction study).
3. Only then write a pre-registered specification and implement.
