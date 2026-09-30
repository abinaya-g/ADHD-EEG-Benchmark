# Research-direction discovery gate (after N1 / D3)

Date: 2026-09-30.

**Scope:** literature screening and research design only. Nothing was implemented or run;
see §14.

## Evidence discipline (same tags as FINAL_LITERATURE_OVERLAP_GATE.md)

| tag | meaning |
|-----|---------|
| [S] | search-engine summary or snippet |
| [K] | established knowledge, or my recollection of the literature. Details marked "verify" are not confirmed |
| [INF] | my inference |
| [U] | unresolved |
| [FT] | full text. **Not used anywhere in this report** |

**Access limits**

- The environment's egress policy still blocks arXiv, all publisher hosts (Elsevier, IEEE,
  ACM, Springer, Wiley), Zenodo, ResearchGate and Google Scholar. See §2 of
  FINAL_LITERATURE_OVERLAP_GATE.md.
- The only literature channel was the web-search tool, which returns summaries.
- **Consequence:**
  - No candidate can be graded "A = strong novelty candidate" in this gate.
  - An idea can be *killed* by a snippet that shows an existing paper doing the same thing.
  - An idea cannot be *cleared* without the full texts.

About 35 queries were run. They are grouped in §5, and the prior-art topics 1–40 of the brief
are mapped in §3.

---

## 1. Executive summary

- **Ten directions were generated from the N1/D3 observation.**
- **Two deserve a second literature gate (grade B):**
  - **R3 — Exchange channel decomposition.** Does information exchange act mainly through
    the *candidate* it inserts, or through the *receiver's adaptive state* that the candidate
    rewrites (targets/leaders, archives, success memories, covariance, velocities)?
  - **R6 — Behavioural complementarity as a predictor of hybridisation synergy.** Can
    optimizer-behaviour descriptors, measured on the stand-alone optimizers, predict whether
    two optimizers will benefit from being hybridised?
  - **Neither is certain to be novel:**
    - R3 has partial prior art: Hansen 2011 for CMA-ES injection [S]; METAFOR's
      "DE-Recompute Velocity" component for PSO-DE hybrids [S].
    - R6 has adjacent prior art: behavioural similarity (Hayward & Engelbrecht, TEVC 2025)
      [S]; performance-based portfolio complementarity (PAP / EPM-PAP; Shapley portfolios)
      [S]; automated hybrid design (METAFOR) [S].
- **Graded C (substantial prior-art overlap), five directions:**
  - **R1**, cross-representation translation: evolutionary multitasking (EMT) autoencoding,
    LDA-MFEA, subspace alignment, and multi-representation island models.
  - **R2**, invariance of exchange: Sroka & Wierzchoń 2025, Jian & Zhu 2021, and
    eigen-basis crossover.
  - **R4**, placebo exchange: Gupta & Ong 2016; random-immigrant controls in island models.
  - **R9**, planted-utility benchmarks: TPAM; artificial AOS scenarios.
  - **R10**, support-compatibility filters: Hansen 2011; MTEA-AD.
- **Graded D (insufficient contribution), three directions:**
  - **R5**, direction semantics: collapses into R1/R3, or is a measurement without a
    question.
  - **R7**, exchange-induced structural bias: an existing test applied to a new object.
  - **R8**, deconstruction of MPHBS: already done for other algorithms by
    Camacho-Villalón et al.
- **The central question as posed** ("is information from one search representation
  compatible with another optimizer's representation?"):
  - It is **not new as a question.** It is the core problem of EMT domain adaptation and of
    multi-representation island models [S].
  - For *same-task* heterogeneous hybrids it largely reduces to "the receiver's state must be
    reconciled", which is where R3 sits.
- **Not forced onto MPHBS:**
  - R6 does not need HBA/MPA at all.
  - For R3, HBA is a convenient and informative example of a target-directed receiver, but it
    is not essential.

---

## 2. What N1/D3 taught us (facts, not claims)

1. **Native HBA moves are target-directed:** Δx ≈ b·(x_prey − x). x_prey is shared state, so
   a single foreign candidate that becomes the best redirects every HBA agent (D3 report
   §6–7) [INF from D3].
2. **The MPA-population's successes are mostly FAD (a DE-type difference step).** They
   inherit population geometry. The MPA operator's own successes almost vanish after the
   first third of the run (D3 §6).
3. **No landscape component independent of population/target geometry was found** after
   whitening or residualisation (D3 §7–8).
4. **The literature gate closed the measurement-validity angle** (FINAL_LITERATURE_OVERLAP_GATE
   §9).
5. **Lesson for new directions:** in these optimizers, "information" is not a free-standing
   object. What a transferred candidate *does* depends on which internal state variables of
   the receiver it enters: its target, elite matrix, archive, or success statistics. This is
   the thread R3 follows. It is an observation, not a novelty claim.

---

## 3. Research-space map (brief topics 1–40 → prior-art clusters found)

| cluster | brief topics | representative prior art found [S] |
|---------|--------------|------------------------------------|
| EMT / transfer optimisation | 1–5, 29–32, 35–37 | MFEA and MFEA-II; explicit autoencoding (Feng et al., TCYB 2019); LDA-MFEA (Bali et al. 2017); subspace alignment (2020 onward); MTEA-AD (TEVC 2022); transferability/inter-task distance (arXiv 2305.12807); Learning-to-Transfer (arXiv 2406.14359); Gupta & Ong, "Genetic Transfer or Population Diversification?" (arXiv 1607.05390); Zhao et al. survey (ASOC 145, 2023) |
| migration / islands / multi-population | 6–8, 27–28 | heterogeneous island models (performance-index migration; five-algorithm archipelagos); migration interval/topology studies; multi-representation islands (Skolicki & De Jong, PPSN 2004); random-immigrant controls in island multimemetic algorithms |
| heterogeneous hybrids / portfolios | 9–10, 24, 40 | AMALGAM (Vrugt & Robinson, PNAS 2007); PAP and EPM-PAP (Peng, Tang, Chen, Yao; Tang et al., Inf. Sci. 2014); METAFOR (Camacho-Villalón, Dorigo, Stützle, arXiv 2502.11225); Shapley value of portfolios (AAAI 2016); LCC for heterogeneous LSGO (GECCO 2026) |
| coordinate systems / invariance | 15–19 | Adaptive encoding (Hansen, PPSN 2008); eigenvector crossover (Guo & Yang); affine invariance of metaheuristics (Jian & Zhu, Inf. Sci. 2021); rotation and information exchange in PSO (Santos et al., SWEVO 2019); invariance of hybrids under transformations (Sroka & Wierzchoń, arXiv 2509.05445, 2025); iSOMA-AR (2026) |
| injection / receiver state | 10, 20, 26 | Injecting external solutions into CMA-ES (Hansen, INRIA RR-7748, 2011); METAFOR's DE-Recompute Velocity (RV-goBack / random / position / none); warm-starting CMA-ES (arXiv 2012.06932) |
| AOS / RL / bandits / credit | 21–23, 25–26 | extensive (not re-surveyed; excluded by the brief); TPAM (Tanabe & Fukunaga, GECCO 2017); parameter-control benchmarking (GECCO 2024) |
| behaviour / trajectory analysis | 38–39 | metaheuristic similarity from 20 behavioural characteristics (Hayward & Engelbrecht, TEVC 29(1) 2025); Search Trajectory Networks; trajectory-based ELA and algorithm selection; structural bias (BIAS toolbox; SB review, SWEVO 2024; SB in multi-objective optimisation, 2026) |
| metaphor critique / component analysis | 10 | Camacho-Villalón, Dorigo & Stützle, ITOR 30(6) 2023; Aranha et al., Swarm Intelligence 2022 ("call for action") |
| linkage / cooperative coevolution | 11–12 | not re-surveyed. The brief excludes new linkage-learning methods, and N1 already covered this ground |

---

## 4. Candidate directions (21 fields each)

Field numbers follow the brief's §8 list. Prior-art evidence is summarised in field 8 and
detailed in §5.

### R1 — Cross-representation translation for same-task heterogeneous exchange

1. **Title:** Translating search information between optimizers with different internal
   representations.
2. **Question:** Can a learned mapping (linear, affine, autoencoder) between optimizer A's and
   optimizer B's "representations" make exchanged information more useful?
3. **Why from N1/D3:** HBA, FAD and MPA express moves in different geometric forms: a target
   direction, a difference vector, and Lévy/Brownian steps.
4. **Mechanism:** learn a map from source-population features to the receiver's frame (for
   example, align population covariances) before exchange.
5. **Measured:** receiver improvement per exchanged candidate, with vs without the map.
6. **Controlled:** the landscape, FE budget and seeds; identity map as baseline.
7. **Why it matters:** only if representation mismatch exists within a *single* task.
8. **Closest prior art:**
   - Feng et al. 2019 (EMT via explicit autoencoding; allows GA + DE solvers) [S];
   - Bali et al. 2017 (LDA-MFEA) [S];
   - subspace-alignment EMT (2020 onward) [S];
   - Skolicki & De Jong 2004 (translating individuals between representations during
     migration) [S];
   - EMT transfer-component analysis (Inf. Sci. 2022) [S];
   - manifold-transfer EMT (2026) [S].
9. **Overlap:** a mapping between solution spaces is the core of EMT domain adaptation.
   Heterogeneous-solver EMT is explicitly supported by Feng et al. [S].
10. **Not addressed:** in a *single* task, all optimizers share the same decision space, so a
    "translation" of solutions is the identity. What differs is the operator's *use* of a
    solution, and that is R3, not R1 [INF].
11. **Novelty risk:** HIGH.
12. **Contribution type:** new algorithm (incremental).
13. **KBS suitability:** LOW.
14. **Burden:** MEDIUM.
15. **Reuse:** code yes; science no.
16. **HBA/MPA:** convenient only.
17. **Generalises:** yes, but into EMT, where it is not new.
18. **Falsified by:** the identity map performing as well as the learned map.
19. **Uninteresting if:** gains equal those of any diversity injection.
20. **Strongest objection:** "This is EMT domain adaptation applied to one task."
21. **Answer:** there is none that survives, because the objection is correct.

**Grade: C**

### R2 — Invariance inheritance of exchange mechanisms

1. **Title:** Do exchange mechanisms break the invariances of their constituent optimizers?
2. **Question:** Is a hybrid only as invariant (to rotation, affine or monotone transforms)
   as its least-invariant exchange rule? Are dimension-wise exchanges (as in MPHBS) the weak
   point?
3. **Why from N1/D3:** MPHBS exchanges information dimension-wise, and N1's HI/HB/HR
   distinction was essentially about coordinate dependence.
4. **Mechanism:** an invariance audit. Run the same instance with and without rotation or
   affine transforms and compare with/without exchange.
5. **Measured:** the change in performance under a transform, per component.
6. **Controlled:** instance, seeds and budget.
7. **Why it matters:** invariance is a principled design property [K].
8. **Closest prior art:**
   - Sroka & Wierzchoń 2025 (a plug-and-play hybridisation operator on 19 algorithms; CEC-2017
     under translation, scaling, rotation and constant shift; differential-based hybrids stay
     invariant, while PSO/HHO hybrids degrade) [S];
   - Jian & Zhu 2021 (affine invariance of PSO, DE, OFA vs GWO, SCA, BOA; ignores the
     crossover) [S];
   - Santos et al. 2019 (rotation × information exchange in PSO) [S];
   - Guo & Yang (eigenvector-basis crossover) [S];
   - Hansen 2008 (adaptive encoding, which makes any coordinate-wise search invariant) [S];
   - the principled invariant-operator design paper (arXiv 2105.10657) [S].
9. **Overlap:** the invariance of hybrids under transforms is directly studied (Sroka &
   Wierzchoń). The loss of invariance from coordinate-wise crossover is textbook, and the
   remedy (eigen-basis / adaptive encoding) exists.
10. **Not addressed** [U]: isolating the *exchange rule's* contribution separately from the
    constituents' own invariance. This is a small ablation, not a question.
11. **Novelty risk:** HIGH.
12. **Contribution type:** empirical phenomenon (known).
13. **KBS suitability:** LOW.
14. **Burden:** LOW.
15. **Reuse:** code yes; CEC2022 functions are already rotated/shifted [K].
16. **HBA/MPA:** convenient.
17. **Generalises:** yes.
18. **Falsified by:** exchange showing no transform sensitivity.
19. **Uninteresting if:** it is predictable from "coordinate-wise ⇒ not rotation invariant",
    which it is [K].
20. **Objection:** "Known since the rotational-invariance literature on DE crossover."
21. **Answer:** none.

**Grade: C**

### R3 — Exchange channel decomposition: candidate channel vs receiver-state channel

1. **Title:** Where does exchanged information act? A causal decomposition of transfer into
   candidate and receiver-state channels.
2. **Question:** When a foreign candidate enters a heterogeneous optimizer, is its effect
   (good or bad) carried by:
   - (a) the candidate itself, i.e. being selected into the population; or
   - (b) the receiver's *adaptive state* that the candidate rewrites?

   Examples of state for (b):
   - HBA/PSO targets and leaders (x_prey, gbest);
   - MPA elite matrices;
   - DE archives and success-history memories (SHADE);
   - CMA-ES mean, covariance and paths;
   - PSO velocities and pbest;
   - an RL controller's Q-values.

   A second question: is "compatibility" of transferred information best defined by the
   state perturbation it causes rather than by its fitness?
3. **Why from N1/D3:** D3 showed HBA moves are ≈ b·(x_prey − x), and x_prey is shared state.
   An exchanged candidate that becomes best redirects *every* HBA agent. The candidate
   channel and the state channel are therefore confounded in MPHBS-style exchange, and
   neither N1 nor MPHBS separates them [INF].
4. **Mechanism (a measurement-and-intervention framework, not a new transfer controller):**
   - **Firewall arm:** accept the candidate into the population, but freeze its write access
     to the receiver's state variables.
   - **State-only arm:** apply the state update the candidate would cause, without retaining
     the candidate. Only where this is well-defined.
   - **Full arm and none arm.**
   - The design is factorial: {candidate channel on/off} × {state channel on/off}.
5. **Measured:**
   - the per-arm effect on progress (paired, common random numbers);
   - a *state perturbation* index per exchange, e.g.:
     - the displacement of the target (‖Δx_prey‖, scaled by the population);
     - the KL divergence between the receiver's proposal distributions before and after;
     - the change in a success-history memory;
     - the CMA-ES Mahalanobis step.
6. **Controlled:** landscape, seeds, FE accounting (the firewall costs no FE), exchange
   schedule, candidate source.
7. **Why it matters:**
   - If most exchange effect travels through state, then "transfer quality" metrics based on
     candidate fitness measure the wrong thing. This would affect EMT, islands and hybrids
     alike [INF].
   - It would also give a principled definition of *compatibility*: information is
     compatible if the state change it induces is one the receiver's own dynamics could have
     produced.
8. **Closest prior art:**
   - Hansen 2011, "Injecting External Solutions into CMA-ES": injection works only after
     renormalising overly long steps, because injected solutions otherwise corrupt the update
     [S].
   - METAFOR: after DE improves a particle, the velocity "may no longer correspond to its
     position"; four recompute rules (goBack/random/position/none) are offered as a
     configurable component [S].
   - Warm-starting CMA-ES (arXiv 2012.06932) [S].
   - SHADE/JADE archive and success-history mechanisms (the state that migrants could
     contaminate) [S][K].
   - Island models where migration triggers parameter re-adaptation (adaptive invasion-based
     distributed DE) [S].
   - Gupta & Ong 2016 (transfer vs diversification in EMT) [S].
   - TPAM (quantitative tracking analysis of adaptation state) [S].
9. **Overlap:**
   - The *need* to reconcile receiver state is recognised for specific pairs: CMA-ES (Hansen)
     and PSO-DE velocity (METAFOR).
   - Both treat it as an *engineering fix inside one algorithm*, not as a general causal
     channel with measured effect sizes [S][INF].
10. **Not addressed** [U; to be checked in gate 2]:
    - a cross-optimizer factorial decomposition of exchange effect into candidate vs state
      channels;
    - a state-perturbation-based definition of compatibility;
    - evidence of which channel dominates for target-directed, difference-vector,
      model-based and velocity-based receivers.
11. **Novelty risk:** MEDIUM–HIGH. There is partial prior art, and the full texts of Hansen
    2011, METAFOR and Camacho-Villalón's thesis are needed.
12. **Contribution type:** new methodology (a causal decomposition) plus a new empirical
    phenomenon (if one channel dominates). Possibly a new compatibility criterion.
13. **KBS suitability:** MEDIUM.
14. **Burden:** MEDIUM. About four receiver classes × a handful of landscapes; interventions
    are simple switches.
15. **Reuse:**
    - Code: HybridState, HBA/MPA, the paired seed framework, the D3 logging of targets and
      elites.
    - Science: the D3 runs cannot be reused, because they have no interventions.
16. **HBA/MPA:** HBA is an informative example (a strongly shared target state) but not
    essential. DE-SHADE, PSO and CMA-ES are required receivers.
17. **Generalises:** yes; to any optimizer with persistent adaptive state.
18. **Falsified by:** candidate-only ≈ full, and state-only ≈ none, across receivers, i.e.
    the state channel is negligible.
19. **Uninteresting if:**
    - the dominance is trivially predictable for every receiver, e.g. "global-best-guided
      optimizers are driven by gbest" with no quantitative or cross-class insight; or
    - the effects are all null.
20. **Strongest objection:** "Hansen 2011 and METAFOR already show state must be
    reconciled. You have only generalised an engineering detail."
21. **Answer experimentally:**
    - Show a *measurable* phenomenon that the engineering fixes do not predict. For example,
      that the state-perturbation index predicts exchange harm better than candidate-fitness
      indices do, across receiver classes, on held-out landscapes.
    - If it does not, the objection stands.

**Grade: B**

### R4 — Placebo-controlled exchange: information vs perturbation

1. **Title:** Is information exchange information? Sham-transfer controls for heterogeneous
   hybrids.
2. **Question:** Does the benefit of exchange come from source *information* or from
   perturbation/diversity? Test by replacing transferred content with matched sham content
   (same magnitude distribution, scrambled source).
3. **Why from N1/D3:** D3 found the MPA-population's gains are mostly FAD, i.e.
   diversity-type steps.
4. **Mechanism:** sham arms (shuffled-source, other-instance source, random immigrants
   matched in distance).
5. **Measured:** full vs sham at matched FE.
6. **Controlled:** schedule, budget, seeds.
7. **Why it matters:** it attributes the benefit.
8. **Closest prior art:**
   - Gupta & Ong 2016, "Genetic Transfer or Population Diversification?" (explicitly
     separates transfer from diversification in EMT) [S];
   - random-immigrant controls in island multimemetic algorithms (random immigrants rank
     last, so island gains come from information) [S];
   - EMT studies noting that unrelated tasks help early through diversity [S];
   - Zhao et al. 2023 survey [S].
9. **Overlap:** the attribution question and the sham/random-immigrant control design both
   exist.
10. **Not addressed:** applying the same controls to single-task heterogeneous hybrids. This
    is a new *setting*, not a new question.
11. **Novelty risk:** HIGH.
12. **Contribution type:** methodology (existing).
13. **KBS suitability:** LOW.
14. **Burden:** LOW.
15. **Reuse:** code yes.
16. **HBA/MPA:** convenient.
17. **Generalises:** yes.
18. **Falsified by:** sham ≈ full.
19. **Uninteresting if:** the result replicates the island/EMT finding.
20. **Objection:** "Gupta & Ong did this."
21. **Answer:** none.

**Grade: C.** Worth keeping as a *control* inside R3.

### R5 — Semantic equivalence of transferred directions

1. **Title:** Do two optimizers interpret the same direction differently?
2. **Question:** Measure the mutual information between a transferred direction and the
   receiver's subsequent success, per receiver class.
3. **Why from N1/D3:** a target direction (HBA) and a difference vector (FAD) with the same
   Δx have different generative meanings.
4. **Mechanism:** an information-theoretic transfer measure.
5. **Measured:** MI or transfer entropy.
6. **Controlled:** receiver and landscape.
7. **Why it matters:** weakly. Without an intervention, MI restates D3's correlational
   findings.
8. **Closest prior art:**
   - search-direction transfer in EMT (directions, evolution paths, distribution models as
     "knowledge representations") [S];
   - transferability metrics (MMD/GRA source selection; inter-task distance) [S];
   - D3 itself.
9. **Overlap:** direction-as-knowledge and transferability estimation both exist.
10. **Not addressed:** "semantics" beyond the conditional distribution of Δx given the
    state; that is R3's state channel.
11. **Novelty risk:** HIGH.
12. **Contribution type:** metric (weak).
13. **KBS suitability:** LOW.
14. **Burden:** MEDIUM.
15. **Reuse:** D3 logs could compute it, but the result would be correlational.
16. **HBA/MPA:** essential, so it is algorithm-specific.
17. **Generalises:** weakly.
18. **Falsified by:** MI ≈ 0.
19. **Uninteresting if:** it just re-derives "Δx follows the target".
20. **Objection:** "A new metric with no decision value."
21. **Answer:** none convincing.

**Grade: D**

### R6 — Behavioural complementarity as a predictor of hybridisation synergy

1. **Title:** Can we predict which optimizers should be hybridised from how they search,
   before hybridising them?
2. **Question:** Do behavioural descriptors of *stand-alone* optimizers predict the
   *synergy* of their hybrid, beyond what stand-alone performance complementarity predicts?
   Behavioural descriptors include step geometry, target dependence, operator-induced
   covariance, diversity dynamics and structural bias. Synergy is the hybrid's gain over the
   best constituent at an equal FE budget, under a fixed, generic exchange protocol.
3. **Why from N1/D3:** D3 produced exactly such descriptors: target-directedness,
   population-geometry inheritance, operator success decay. N1's premise, "heterogeneity
   makes exchange valuable", was never tested as a *predictive* hypothesis.
4. **Mechanism:**
   - Measure descriptors on stand-alone runs.
   - Build hybrids for many optimizer pairs with a *fixed, simple* exchange protocol, so the
     result is about the pair and not the controller.
   - Regress or rank synergy on descriptor distance vs performance complementarity.
5. **Measured:** synergy per pair × problem; descriptor distances; performance
   complementarity (e.g. Shapley/marginal contribution).
6. **Controlled:** exchange protocol, budget, problems, seeds. Pairs should include
   same-family negative controls (e.g. DE + DE variant).
7. **Why it matters:** hybrid design today is trial-and-error or automated search (METAFOR).
   A validated predictor would be generalisable design knowledge. If falsified, the
   "complementarity" rationale that hybrid papers routinely assert [S] loses support.
8. **Closest prior art:**
   - Hayward & Engelbrecht 2025, TEVC 29(1) (20 behavioural characteristics; pairwise
     similarity; behavioural novelty) [S];
   - PAP (Peng, Tang, Chen, Yao) and EPM-PAP (Tang et al., Inf. Sci. 2014); constituent
     selection by the *estimated performance matrix* [S];
   - Fréchette et al., AAAI 2016 (Shapley value of algorithm portfolios) [S];
   - METAFOR (automated hybrid design; identifies which hybridisations work per problem
     class) [S];
   - Sroka & Wierzchoń 2025 (19 hybrids built with one plug-in operator, a potential external
     test set) [S];
   - Search Trajectory Networks [S];
   - AMALGAM [S];
   - LCC (GECCO 2026) [S].
9. **Overlap:**
   - Behavioural similarity measurement exists.
   - Portfolio complementarity exists but is performance-based and concerns portfolios
     (separate runs), not hybrids with exchange.
   - Automated hybrid design finds good hybrids but does not provide a *predictor from
     stand-alone behaviour*, as far as the snippets show [S][U].
10. **Not addressed** [U]:
    - a test of whether behavioural *dissimilarity* predicts *hybrid synergy*;
    - whether behaviour adds predictive value beyond performance complementarity.
11. **Novelty risk:** MEDIUM.
    - The main risk: METAFOR or Camacho-Villalón's thesis, or a 2024–2026 automated-design
      paper, may already relate component behaviour to hybrid benefit.
    - A second risk: Hayward & Engelbrecht may include a hybridisation analysis.
12. **Contribution type:** new empirical phenomenon, or a falsified folk principle, plus a
    predictive framework.
13. **KBS suitability:** MEDIUM–HIGH ("knowledge about hybrid design" fits a knowledge-based
    venue, provided the result is predictive and validated out of sample) [INF].
14. **Burden:** HIGH. Many pairs × problems × seeds are needed for a predictive claim. It
    could be reduced by using a fixed exchange protocol and CEC/BBOB subsets.
15. **Reuse:** the paired-statistics framework, benchmark wrappers and the D3-type
    descriptor code. **HybridState/MPHBS are not needed as science**: the fixed protocol must
    be generic, not SARSA.
16. **HBA/MPA:** merely one pair among many.
17. **Generalises:** by design.
18. **Falsified by:** descriptor distance having no out-of-sample predictive value, or no
    value beyond performance complementarity.
19. **Uninteresting if:**
    - synergy is explained by the stand-alone performance gap alone ("hybrid ≈ best
      constituent + noise"); or
    - the predictor works only in-sample.
20. **Strongest objection:** "Synergy depends on the exchange protocol. Your predictor
    predicts one protocol."
21. **Answer experimentally:**
    - Test predictor stability across two or three generic protocols (e.g. best-migration
      islands and a shared-population AMALGAM-style scheme).
    - Report rank correlation of pair synergy across protocols.
    - If synergy is protocol-specific, report that as the finding and narrow the claim.

**Grade: B**

### R7 — Exchange-induced structural bias

1. **Title:** Does information exchange introduce structural bias?
2. **Question:** Run hybrids on f0 with vs without exchange.
3. **Why from N1/D3:** exchange redirects shared targets.
4. **Mechanism:** the BIAS toolbox on hybrids.
5. **Measured:** SB class and strength.
6. **Controlled:** f0, seeds.
7. **Why it matters:** modest.
8. **Closest prior art:**
   - the BIAS toolbox (39 tests plus a random-forest classifier) [S];
   - the SB review (SWEVO 2024) [S];
   - SB in multi-objective optimisation (2026) [S];
   - Deep-BIAS [S];
   - anisotropy in SB (GECCO'21) [S].
9. **Overlap:** the method is complete. Only the object (hybrid ± exchange) is new.
10. **Not addressed:** hybrid-specific SB, a narrow application.
11. **Novelty risk:** MEDIUM (as an application).
12. **Contribution type:** application.
13. **KBS suitability:** LOW.
14. **Burden:** LOW.
15. **Reuse:** yes.
16. **HBA/MPA:** convenient.
17. **Generalises:** yes.
18. **Falsified by:** no SB change.
19. **Uninteresting if:** SB is inherited from the constituents (likely) [INF].
20. **Objection:** "An existing test on a new algorithm", which the brief itself lists as
    insufficient.
21. **Answer:** none.

**Grade: D.** Usable as a control inside R3/R6.

### R8 — Component deconstruction of SARSA-guided hybrids (MPHBS audit)

1. **Title:** What does the mediator contribute? A component-wise deconstruction of
   RL-guided hybrids.
2. **Question:** Does MPHBS's advantage come from SARSA-guided exchange or from
   FAD/DE-equivalent components and extra evaluations?
3. **Why from N1/D3:** about 87% of MPA-population successes come from FAD; mediator effects
   were weak.
4. **Mechanism:** component ablation and equivalence mapping.
5. **Measured:** performance per ablation.
6. **Controlled:** FE budget.
7. **Why it matters:** it is a critique.
8. **Closest prior art:**
   - Camacho-Villalón, Dorigo & Stützle, ITOR 2023 (six metaphor algorithms reduced to known
     components) [S];
   - Aranha et al. 2022 [S];
   - Hayward & Engelbrecht 2025 [S];
   - METAFOR [S].
9. **Overlap:** the method and genre exist.
10. **Not addressed:** this particular algorithm.
11. **Novelty risk:** LOW for the object, HIGH for the method.
12. **Contribution type:** critique / audit.
13. **KBS suitability:** LOW. A critique of a paper in the same venue is a comment-type
    contribution [INF].
14. **Burden:** LOW–MEDIUM.
15. **Reuse:** yes (the MPHBS reproduction exists).
16. **HBA/MPA:** essential, so it is algorithm-specific.
17. **Generalises:** no.
18. **Falsified by:** the SARSA mediator being essential.
19. **Uninteresting if:** "the hybrid is mostly DE", which was expected.
20. **Objection:** "Genre already established; single algorithm."
21. **Answer:** none.

**Grade: D** (C on method).

### R9 — Planted-utility benchmark for transfer controllers

1. **Title:** Do transfer controllers recover a known ground-truth transfer utility?
2. **Question:** Plant a controllable transfer utility and test whether RL/bandit/SPRT
   exchange controllers track it (TPAM-style).
3. **Why from N1/D3:** N1's SPRT calibration needed exactly this.
4. **Mechanism:** simulated reward streams with planted utility.
5. **Measured:** tracking error, regret.
6. **Controlled:** the planted scenario.
7. **Why it matters:** evaluation validity of controllers.
8. **Closest prior art:**
   - TPAM (Tanabe & Fukunaga, GECCO 2017) [S];
   - parameter-control benchmarking (GECCO 2024) [S];
   - the AOS literature's artificial reward scenarios [K, verify (my recollection of
     Fialho-era AOS evaluation)].
9. **Overlap:** the methodology exists; only the target (transfer controllers) differs.
10. **Not addressed:** transfer-controller specifics.
11. **Novelty risk:** HIGH.
12. **Contribution type:** methodology (existing).
13. **KBS suitability:** LOW.
14. **Burden:** LOW.
15. **Reuse:** N1 calibration code.
16. **HBA/MPA:** not needed.
17. **Generalises:** yes.
18. **Falsified by:** —.
19. **Uninteresting if:** it replicates the known bandit tracking results.
20. **Objection:** "TPAM for transfer."
21. **Answer:** none.

**Grade: C**

### R10 — Support-compatibility filter (is the candidate inside the receiver's proposal support?)

1. **Title:** Transfer compatibility as receiver-support membership.
2. **Question:** Accept foreign information only if it is plausible under the receiver's own
   proposal distribution (Mahalanobis / density ratio).
3. **Why from N1/D3:** population-scaled receivers.
4. **Mechanism:** density-ratio gating.
5. **Measured:** harm rate.
6. **Controlled:** schedule.
7. **Why it matters:** modest.
8. **Closest prior art:**
   - Hansen 2011 (renormalise injected steps to the receiver's Mahalanobis scale) [S];
   - MTEA-AD (TEVC 2022; anomaly detection flags outlier transferred individuals) [S];
   - adaptive EMT with anomaly detection (ESWA 2025) [S];
   - EMT-ADT (decision-tree transferability) [S].
9. **Overlap:** direct.
10. **Not addressed:** —.
11. **Novelty risk:** HIGH.
12. **Contribution type:** algorithm (another negative-transfer filter, excluded by the
    brief).
13. **KBS suitability:** LOW.
14. **Burden:** LOW.
15. **Reuse:** yes.
16. **HBA/MPA:** convenient.
17. **Generalises:** yes.
18. **Falsified by:** —.
19. **Uninteresting if:** it performs like MTEA-AD.
20. **Objection:** "MTEA-AD / Hansen."
21. **Answer:** none.

**Grade: C**

---

## 5. Prior-art evidence (queries → decisive hits; all [S])

| candidate | disproving queries run (abridged) | decisive hit(s) |
|-----------|-----------------------------------|-----------------|
| R1 | EMT explicit autoencoding; LDA-MFEA search-space mismatch; subspace alignment EMT; multi-representation island | Feng et al. TCYB 2019; Bali et al. 2017; subspace-alignment EMT; Skolicki & De Jong PPSN 2004 |
| R2 | rotation invariance hybrid exchange; adaptive encoding; affine invariance | Sroka & Wierzchoń arXiv 2509.05445; Jian & Zhu Inf. Sci. 576 (2021); Santos et al. SWEVO 48 (2019); Hansen PPSN 2008 |
| R3 | CMA-ES injection; injection effect on internal state; PSO-DE state transfer; migration × self-adaptation; knowledge transfer disrupts receiver state | Hansen 2011 (partial); METAFOR DE-Recompute Velocity (partial); **no cross-optimizer channel decomposition found** [U] |
| R4 | transfer vs diversity EMT; random immigrants control | Gupta & Ong arXiv 1607.05390; island multimemetic migration analysis |
| R5 | search-direction transfer EMT; transferability estimation | EMT direction/evolution-path transfer; MMD/GRA source selection; inter-task distance |
| R6 | behavioural similarity metaheuristics; complementarity → hybrid; PAP; Shapley portfolios; predicting cooperation between heterogeneous optimizers | Hayward & Engelbrecht TEVC 2025 (adjacent); PAP/EPM-PAP; AAAI 2016 Shapley; METAFOR; **no behaviour→synergy predictor found** [U] |
| R7 | structural bias hybrid / BIAS | BIAS toolbox; SB review SWEVO 2024; SB-MO 2026 |
| R8 | metaphor critique component analysis | Camacho-Villalón et al. ITOR 2023; Aranha et al. 2022 |
| R9 | TPAM; AOS credit counterfactual | TPAM GECCO 2017; parameter-control benchmarking GECCO 2024 |
| R10 | anomaly-detection transfer | MTEA-AD TEVC 2022; Hansen 2011 |

**Absence caveat.** "No X found" is evidence from snippets only. The absence of a hit is not
proof of absence, and it is exactly what gate 2 must test.

---

## 6. Novelty matrix

| Candidate | Core question | Closest prior art | Main overlap | Remaining gap | Novelty risk | Generality | KBS suitability | Class |
|-----------|---------------|-------------------|--------------|---------------|--------------|------------|-----------------|-------|
| R1 translation | map info between optimizer representations | Feng 2019; LDA-MFEA; Skolicki & De Jong 2004 | cross-space mapping | none in single-task setting | HIGH | high | LOW | **C** |
| R2 invariance | does exchange break invariance | Sroka & Wierzchoń 2025; Jian & Zhu 2021 | hybrids under transforms | exchange-rule-only ablation | HIGH | high | LOW | **C** |
| R3 channels | candidate vs receiver-state channel | Hansen 2011; METAFOR | state reconciliation for CMA-ES, PSO-DE | cross-class causal decomposition; state-perturbation compatibility | MED–HIGH | high | MEDIUM | **B** |
| R4 placebo | information vs perturbation | Gupta & Ong 2016; island controls | design identical | new setting only | HIGH | high | LOW | **C** |
| R5 semantics | same direction, different meaning | EMT direction transfer; transferability metrics | direction as knowledge | none beyond R3 | HIGH | low | LOW | **D** |
| R6 complementarity | behaviour predicts hybrid synergy | Hayward & Engelbrecht 2025; PAP/EPM; Shapley; METAFOR | similarity measurement; performance complementarity | behaviour → synergy, out of sample, beyond performance | MEDIUM | high | MED–HIGH | **B** |
| R7 SB | exchange-induced SB | BIAS; SB review | full method | object only | MEDIUM | medium | LOW | **D** |
| R8 audit | deconstruct MPHBS | Camacho-Villalón 2023 | genre and method | this algorithm | HIGH (method) | none | LOW | **D** |
| R9 planted utility | controllers track known utility | TPAM | methodology | target only | HIGH | medium | LOW | **C** |
| R10 support filter | accept only in-support info | MTEA-AD; Hansen 2011 | direct | none | HIGH | high | LOW | **C** |

- No candidate is graded A. With snippet-only evidence, none can be.
- The two B candidates are *screened in*, not ranked against each other.

---

## 7. Directions rejected immediately and why

- **R1:** EMT domain adaptation (autoencoder, LDA, subspace alignment) and multi-representation
  islands already *are* this question. In a single task, the "representations" share one
  decision space.
- **R2:** hybrid invariance under transforms was studied in 2025. Coordinate-wise loss of
  invariance and its remedies are textbook.
- **R4:** transfer-vs-diversification was separated in 2016, and random-immigrant controls
  exist.
- **R5:** a correlational metric with no decision value. Anything real in it belongs to R3.
- **R7, R8:** existing methods applied to a new object. The brief itself classes this as
  insufficient.
- **R9:** TPAM for a different controller.
- **R10:** a negative-transfer filter, which the brief excludes and MTEA-AD and Hansen 2011
  already cover.
- **Excluded by the brief without search** (extensive prior art):
  - new transfer-probability, RL, bandit or intensity controllers;
  - new covariance estimators;
  - new linkage learning.

---

## 8. Candidates for a second-stage literature gate (at most three; two selected)

1. **R3 — Exchange channel decomposition** (candidate vs receiver-state).
2. **R6 — Behavioural complementarity → hybridisation synergy.**

**Why no third.** Every other candidate was killed by a specific prior work (C) or lacks a
scientific question (D). Promoting one would be selecting for ease, which the brief
forbids.

---

## 9. Exact hypotheses

**R3**

- **H3.1 (dominance).** For receivers with *shared adaptive state* (target/leader-guided:
  HBA, PSO-gbest; model-based: CMA-ES), the state channel carries most of the exchange
  effect. That is, |effect(state-only)| > |effect(candidate-only)|, and firewalling the state
  removes most of both benefit and harm.

  For receivers with *weak shared state* (DE/rand/1 without archive), the candidate channel
  dominates.
- **H3.2 (compatibility criterion).** Per exchange event, a receiver-relative
  state-perturbation index predicts subsequent harm (loss vs the no-exchange counterfactual)
  better than candidate-fitness indices (e.g. improvement over the receiver's best, or rank).
  "Better" means higher out-of-sample AUC.
- **Null.** Channel effects are equal, or the state channel is negligible, *and* the
  perturbation index has no predictive value beyond fitness.

**R6**

- **H6.1.** Across optimizer pairs hybridised with a fixed generic exchange protocol,
  stand-alone behavioural distance correlates positively with hybrid synergy (the gain over
  the best constituent at equal FE). The correlation holds out of sample (held-out pairs and
  held-out problems).
- **H6.2.** Behavioural descriptors add predictive value beyond stand-alone performance
  complementarity, measured as incremental R² or rank correlation under nested
  cross-validation.
- **Null.** Synergy is explained by performance complementarity or the performance gap alone,
  or behavioural distance does not generalise across protocols.

---

## 10. Required novelty tests (gate-2 design; nothing to be run now)

### R3

- **A. Hypotheses:** as H3.1 and H3.2.
- **B. What the literature already predicts:**
  - Injected solutions must be renormalised for CMA-ES (Hansen 2011).
  - PSO velocity must be recomputed after foreign improvement (METAFOR).
  - Global-best-guided swarms follow gbest [K].
  - So a state effect *exists* for these specific receivers.
- **C. What would be new:**
  - the *relative size* of the two channels, measured causally and across receiver classes;
  - a receiver-agnostic, state-based compatibility criterion that predicts harm.
- **D. Minimum controls:**
  - the four-arm factorial (none / candidate-only / state-only / full);
  - sham candidates (R4-style) to separate information from perturbation;
  - FE-matched no-exchange runs;
  - common random numbers.
- **E. Minimum landscapes:**
  - sphere and rotated ellipsoid (known geometry);
  - one multimodal function (Rastrigin/Schwefel family);
  - one deceptive/funnel function;
  - D = 10 and 20.
  - A CEC2022 subset only after the synthetic results are clear.
- **F. Receivers (baselines):**
  - PSO (gbest and lbest);
  - DE/rand/1/bin;
  - SHADE (archive plus memory);
  - CMA-ES (injection with and without Hansen's renormalisation);
  - HBA (target-directed).
  - Sources: a fixed donor (e.g. an independent optimizer run) so that source quality is
    controlled.
- **G. Primary metric:** paired log-ratio of best-so-far error at fixed FE checkpoints,
  full/none, decomposed by arm.
- **H. Statistical test:**
  - two-way (candidate × state) analysis of paired outcomes per receiver with a
    Wilcoxon-based aligned-rank transform, or a mixed-effects model on log-error with seed as
    a random effect;
  - Holm correction across receivers;
  - for H3.2, AUC comparison (DeLong) on held-out landscapes.
- **I. Supports:**
  - a state-channel effect ≥ the candidate-channel effect for the shared-state receivers and
    the reverse for DE/rand;
  - perturbation-index AUC significantly > fitness-index AUC out of sample.
- **J. Falsifies:** a negligible state channel everywhere, or a perturbation index no better
  than fitness.
- **K. Non-novel:**
  - The only significant state effects are the two already fixed in the literature (CMA-ES
    step length; PSO velocity).
  - The criterion reduces to Hansen's Mahalanobis renormalisation.
- **L. KBS-level:** a cross-class, causally identified channel structure plus a compatibility
  criterion that transfers to held-out receivers or landscapes.
- **Gate-2 literature to obtain in full text:**
  - Hansen 2011 (RR-7748);
  - METAFOR (arXiv 2502.11225) and Camacho-Villalón's PhD thesis;
  - Sroka & Wierzchoń 2025 (the hybridisation operator's state handling);
  - the WS-CMA-ES paper;
  - the distributed/island DE parameter-adaptation papers;
  - EMT surveys (2021–2026), searched for "state", "parameter", "model" transfer
    contamination;
  - any 2024–2026 "solution injection" or "external solutions" work for DE/PSO/SHADE.
  - **Kill condition:** a paper that already decomposes transfer into solution vs
    strategy-parameter/state channels across optimizer classes.

### R6

- **A. Hypotheses:** as H6.1 and H6.2.
- **B. What the literature already predicts:**
  - Portfolios gain from complementary *performance* (PAP; Shapley).
  - Automated design finds good hybrids per problem class (METAFOR).
  - Behavioural similarity clusters most metaheuristics together (Hayward & Engelbrecht).
- **C. What would be new:** *behavioural* distance of stand-alone optimizers predicting
  *hybrid* synergy (with exchange), out of sample, beyond performance complementarity.
- **D. Minimum controls:**
  - fixed generic exchange protocols (at least two);
  - same-algorithm "hybrid" pairs (the synergy floor);
  - a portfolio without exchange (separates synergy from portfolio effect);
  - FE-matched budgets.
- **E. Minimum problems:** a BBOB or CEC subset stratified by function group (separable,
  ill-conditioned, multimodal-global, multimodal-weak); D = 10 and 20.
- **F. Optimizer pool:** about 8–10 canonical optimizers, giving about 28–45 pairs:
  - DE/rand/1, SHADE, PSO, CMA-ES, (1+1)-ES, a GA (SBX), Nelder–Mead or a local search;
  - one target-directed swarm (HBA or GWO);
  - MPA.
  - Behavioural novelty is the question, so metaphor algorithms serve as *examples*, not
    subjects.
- **G. Primary metric:** synergy = paired log(best constituent error / hybrid error) at equal
  FE; predictor quality = held-out Spearman ρ between predicted and observed synergy.
- **H. Statistical test:**
  - nested cross-validation (leave-pairs-out and leave-problems-out);
  - permutation test for ρ;
  - incremental-R² test for behaviour beyond performance.
- **I. Supports:** significant held-out ρ, and incremental value of behaviour over
  performance, stable across the two protocols.
- **J. Falsifies:** no held-out predictive value, or no incremental value.
- **K. Non-novel:** the full texts of Hayward & Engelbrecht, METAFOR, or a 2024–2026
  automated-hybrid-design paper already report behaviour-based prediction of hybrid benefit.
- **L. KBS-level:** a validated, generalising predictor (or a well-powered refutation of the
  "complementarity" rationale) with a usable design guideline.
- **Gate-2 literature to obtain in full text:**
  - Hayward & Engelbrecht 2025 and the 2025 Swarm Intelligence survey of behaviour
    characteristics;
  - METAFOR and the thesis;
  - PAP (the original TEVC paper; year to verify) and EPM-PAP 2014;
  - Fréchette et al. 2016;
  - Search Trajectory Networks (ASOC 2021; authors to verify);
  - the automated algorithm design literature 2023–2026 (irace-based hybrid design;
    LLM-driven metaheuristic discovery and its "behaviour space" analysis, arXiv 2507.03605).
  - **Kill condition:** a paper already predicting hybrid or cooperation benefit from
    behavioural features.

---

## 11. Reusable infrastructure (code reuse ≠ scientific validity)

| asset | code reusable? | scientifically valid to reuse? |
|-------|----------------|--------------------------------|
| HBA / MPA / HybridState | yes | as *examples of receivers* (R3) or pool members (R6) only. Their MPHBS coupling must not be the object |
| MPHBS / SARSA mediator | yes | **no** for R3/R6. A learned controller confounds channel effects and pair synergy; both need fixed, generic protocols |
| CEC2022 evaluator | yes | yes, as a secondary benchmark after synthetic landscapes |
| FIR evaluator | yes | low relevance |
| common-random-numbers seeds / paired stats | yes | yes, with **new, pre-registered seed masters** (never the n1_* masters) |
| D3 logging (targets, elites, population moments) | yes (pattern) | the *instrumentation pattern* is reusable for R3's state-perturbation index. **The D3 data are not reusable**: they have no interventions and belong to a frozen study |
| N1 SPRT calibration harness | yes | not needed |
| N1 runs, n1_smoke, n1_calib, D3 data | — | **no.** Frozen, and the wrong design for either question |

---

## 12. Major risks

1. **Snippet-only screening.** Both B candidates could be killed in gate 2 by full texts
   (METAFOR and Camacho-Villalón's thesis are the common risk to both).
2. **R3 may be "engineering detail generalised".** The objection in field 20 is strong. The
   contribution rests on H3.2 (a predictive compatibility criterion), not on H3.1 alone.
3. **R6 is costly and statistically demanding.** A predictive claim over about 30–45 pairs
   is modest in sample size. A negative result is likely, and publishable only if well
   powered.
4. **Protocol dependence (R6) and receiver-set dependence (R3)** may make results narrow.
5. **Competing groups.** Automated hybrid design (IRIDIA) and behavioural analysis (Hayward &
   Engelbrecht) are active. Timing risk is real [INF].
6. **MPHBS gravity.** There is a temptation to route either study through the existing
   HBA/MPA/SARSA code. That would reintroduce confounds (§11).

---

## 13. Final recommendation

- Run a **second literature gate** on **R3** and **R6**, with full-text access. This needs
  the PDFs listed in §10 to be supplied, or arxiv.org / publisher hosts to be allowed in the
  environment's network policy.
- Pass either one to design **only** if its kill condition (§10) is not met.
- Do **not** pursue R1, R2, R4, R5, R7, R8, R9 or R10.
- **No implementation is recommended** at this stage.

---

## 14. Implementation freeze confirmation

| item | status |
|------|--------|
| N1 modified? | **NO** |
| D3 modified? | **NO** |
| New experiments? | **NO** |
| New seeds? | **NO** |
| Production code changed? | **NO** |
| Specification changed? | **NO** |

This report is the only new file.

**Status: RESEARCH DIRECTION DISCOVERY ONLY — IMPLEMENTATION FROZEN**

---

## Sources (search results; none read in full text)

**Evolutionary multitasking and transfer**
- https://www.researchgate.net/publication/326133186_Evolutionary_Multitasking_via_Explicit_Autoencoding
- https://ieeexplore.ieee.org/document/7969454/
- https://ieeexplore.ieee.org/document/9385398/
- https://arxiv.org/abs/1607.05390
- https://www.sciencedirect.com/science/article/abs/pii/S156849462300563X
- https://arxiv.org/pdf/2305.12807
- https://arxiv.org/pdf/2406.14359
- https://github.com/xiaofangxd/Multitasking-Optimization

**Injection and receiver state**
- https://arxiv.org/pdf/1110.4181
- https://arxiv.org/pdf/2012.06932

**Invariance and coordinate systems**
- https://link.springer.com/chapter/10.1007/978-3-540-87700-4_21
- https://arxiv.org/abs/2509.05445
- https://www.sciencedirect.com/science/article/abs/pii/S0020025521006538
- https://www.sciencedirect.com/science/article/abs/pii/S2210650218310198
- https://arxiv.org/pdf/2105.10657

**Islands, hybrids and portfolios**
- https://link.springer.com/chapter/10.1007/978-3-540-30217-9_43
- https://www.researchgate.net/publication/265684767_An_Analysis_of_Migration_Strategies_in_Island-Based_Multimemetic_Algorithms
- https://www.pnas.org/doi/10.1073/pnas.0610471104
- https://research.birmingham.ac.uk/en/publications/population-based-algorithm-portfolios-for-numerical-optimization/
- https://www.sciencedirect.com/science/article/pii/S0020025514004022
- https://www.cs.ubc.ca/~kevinlb/papers/2016-AAAI-portfolio-shapley-extended.pdf
- https://arxiv.org/pdf/2502.11225
- https://doi.org/10.1145/3795095.3805054

**Behaviour and trajectory analysis**
- https://ieeexplore.ieee.org/document/10373561/
- https://link.springer.com/article/10.1007/s11721-025-00254-1
- https://www.storre.stir.ac.uk/bitstream/1893/32613/1/stns_asoc_2021.pdf

**Metaphor critique**
- https://onlinelibrary.wiley.com/doi/10.1111/itor.13176
- https://link.springer.com/article/10.1007/s11721-021-00202-9

**Parameter adaptation**
- https://arxiv.org/abs/2010.01877
- https://dl.acm.org/doi/10.1145/3638529.3654019

**Structural bias**
- https://www.sciencedirect.com/science/article/abs/pii/S221065022400350X
- https://arxiv.org/html/2602.06742v1
