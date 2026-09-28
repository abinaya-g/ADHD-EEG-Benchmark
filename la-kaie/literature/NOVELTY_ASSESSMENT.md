# Novelty assessment for LA-KAIE (initial search, 2026-09-28)

## Scope and limitations

* Web search engine only. Full-text access to arXiv, Springer and Zenodo is
  **blocked** by this environment's network policy, so every entry except
  MPHBS was assessed from **titles, abstracts and search-result snippets
  only**. Claims about what a paper does *not* contain are therefore
  provisional and must be verified against the full texts before submission.
* Queries covered: RL-assisted metaheuristics, learning-guided / adaptive
  information exchange, landscape-aware operator selection, variable-interaction
  learning, linkage-aware / group-wise recombination, multi-population
  migration, recent KBS hybrid metaheuristics. The matrix is in
  `novelty_matrix.csv` (11 entries).

## Findings

| LA-KAIE ingredient | Prior art found | Overlap |
|---|---|---|
| Transfer interacting variables together instead of dimension-wise | RV-GOMEA (linkage-model FOS mixing, both singletons and multivariate sets), RVIntX (online interaction learning + interaction-aware recombination masks in DE), Adaptive Linkage Crossover (2000), EMTO-IVG (group-level transfer in multitask) | **High** |
| Interaction estimated from already-evaluated population data, no extra FEs | VIGPSO (correlation of particle updates), RVIntX (online linkage learning) | Medium–high |
| Landscape-feature state + online (contextual) bandit / RL choosing operators | LA-BHH (LinUCB with landscape + stagnation features), RL-HPSDE (Q-learning with FDC / ruggedness states), DE-DDQN, RLDE-AFL | **High** for the controller concept |
| Multi-component reward (improvement + diversity + cost + stagnation) | Common in AOS/RL-EA literature in various forms; no single close match identified in snippets | Medium |
| Controller deciding **when** to exchange, **which source population**, and **dimension-vs-group granularity** between two heterogeneous optimizers (HBA/MPA) | Not found in this search. MPHBS decides only which ranked component per dimension; GOMEA/RVIntX do not choose granularity adaptively and have one population | Not found (provisional) |

## Verdict

The concept stated in the brief as the *main methodological novelty* —
"the ability to exchange individual variables OR interaction groups depending
on the current landscape", with groups learned from variable dependencies —
is **not new as a principle**: linkage-learning EAs (GOMEA family, RVIntX,
VIGPSO) already recombine learned interaction groups, and GOMEA's
linkage-tree FOS already mixes at both univariate and multivariate
granularity. Landscape-aware contextual-bandit operator selection is also
established (LA-BHH, RL-HPSDE, DE-DDQN). The name prefix "LA-" also collides
with LA-BHH (2026).

What remains potentially distinct is the **specific integration**: a
landscape-conditioned controller that jointly decides *whether* to spend
evaluations on exchange at all, *which* heterogeneous source population to
draw from, and at *which granularity* (dimension vs estimated interaction
group) to transfer, inside an HBA–MPA hybrid, with a reward that couples
fitness gain with interaction consistency. That is an incremental
combination claim, and it is weaker than the one the brief anticipated.

This meets STOP condition §31.3 / §26 ("a literature overlap that threatens
novelty"). Following §26, the full 30-run experiment is **not** launched
until the user decides how to position or modify the contribution. Work that
is independent of that decision (benchmarks, baseline reproduction, the
LA-KAIE prototype, unit tests, smoke and pilot runs) proceeds, clearly
labelled as pre-decision.
