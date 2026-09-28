# MPHBS reproduction notes

Source paper: E. Flores, R. Olivares, "MPHBS: A novel hybrid metaheuristic with
SARSA-guided information exchange", *Knowledge-Based Systems* 352 (2026) 117070
(`reference/MPHBS.pdf`, 29 pages, read in full).

Secondary source: the authors' public MATLAB package cited as ref. [74] of the
paper (Zenodo DOI 10.5281/zenodo.21326361). Zenodo is blocked by this
environment's network policy; the same package was obtained from GitHub
`efloresaraya/MPHBS-SARSA-information-exchange` (MIT licence, commit recorded in
`third_party/mphbs_reference/SOURCE_COMMIT.txt`). The key `.m` files and the
authors' per-run result CSVs are vendored in `third_party/mphbs_reference/`.

Conventions in this document

* **[PAPER]** – stated in the PDF (equation numbers refer to the PDF).
* **[CODE]** – taken from the authors' MATLAB code.
* **MISSING FROM PAPER** – required for reproduction but absent from the PDF.
  In every such case the authors' code gives the value, and the code value is
  what our Python port uses.
* **DISCREPANCY** – paper and code disagree. The port follows the code, because
  the paper's own results tables were produced by the code (the FE counts in
  Table 2 can only be reproduced with the code's behaviour, see §14).

---

## 1. HBA formulation (paper §3.1, Eqs. (1)–(12))

* Population `X_HBA ∈ R^{N×D}` initialised uniformly in bounds, Eq. (2).
* Intensity `I_i = r2 · S_i / (4π‖d_i‖² + ε)`, Eq. (3); `d_i = X_best − X_i`,
  Eq. (4); `S_i = ‖X_i − X_{i+1}‖²`, Eq. (5).
  * MISSING FROM PAPER: wrap-around for `i = N` ([CODE] uses `X_1`), and the
    exact placement of ε ([CODE]: `‖X_i − X_best + eps‖²` and
    `‖X_i − X_{i+1} + eps‖²` with MATLAB `eps = 2.22e-16` added element-wise
    inside the norm, and `+eps` again in the denominator). `r2` is drawn per agent.
* Density factor `α = C·exp(−t/t_max)`, Eq. (6), C = 2 [PAPER].
  * [CODE] inside MPHB/MPHBS: `α = C·exp(−(t+1)/max(1,T))` with t starting at 0,
    i.e. identical to the standalone HBA (`t = 1..tmax`).
* Flag F ∈ {−1, +1} with probability 0.5, Eq. (7). [CODE] draws one F and one
  mode switch `r` **per agent** (not per component).
* Digging mode (r < 0.5), Eq. (8):
  `x_new,ij = x_best,j + F·β·I_i·x_best,j + F·r3·α·d_ij·|cos(2πr4)(1 − cos(2πr5))|`
  with r3, r4, r5 drawn per component.
* Honey mode, Eq. (9): `x_new,ij = x_best,j + F·r7·α·d_ij` (r7 per component).
* β: MISSING FROM PAPER (paper says only "a constant"). [CODE] β = 6.
* Bound repair by clipping, Eq. (10); greedy replacement, Eq. (11); best
  refresh, Eq. (12). FE cost: N per iteration.

## 2. MPA formulation (paper §3.2, Eqs. (13)–(30))

* Uniform initialisation, Eq. (14); Elite matrix = replicated best, Eq. (15).
* `CF = (1 − t/t_max)^(2t/t_max)`, Eq. (16).
* Phase 1 (t < t_max/3): Brownian, Eqs. (17)–(18), `P = 0.5` [PAPER].
* Phase 2 (t_max/3 < t < 2t_max/3): first half Lévy (Eqs. 19–20), second half
  Brownian around Elite (Eqs. 21–22).
* Phase 3: Lévy around Elite, Eqs. (23)–(24).
* MISSING FROM PAPER: the Lévy generator. [CODE]: `RL = 0.05·levy(N,D,1.5)`,
  Mantegna's algorithm with β_L = 1.5 (σ_u from the Gamma-function formula).
  `RB ~ N(0,1)`, `R ~ U(0,1)` per component.
* MISSING FROM PAPER: boundary conditions between phases. [CODE] MPHB/MPHBS:
  phase 2 is `T/3 ≤ t < 2T/3`; the original standalone MPA uses
  `Iter > T/3 && Iter < 2T/3` (so `Iter == T/3` falls into phase 3). Each port
  replicates its own source.
* "first half" of the population: [CODE] `i ≤ N/2` is Lévy, `i > N/2` Brownian.
* Clipping, Eq. (25); memory saving, Eqs. (26)–(27) (keep old if strictly better).
* FADs perturbation, Eqs. (28)–(30). FADs value: MISSING FROM PAPER; [CODE]
  FADs = 0.2. `U` is a Bernoulli(FADs) mask; `Z = X_min + rand⊙(X_max − X_min)`;
  the branch `r < FADs` is decided **once for the whole population** [CODE].
* Standalone MPA [CODE] evaluates the population twice per iteration (before
  and after the movement), so its FE cost is 2N per iteration and it performs
  no separate initial evaluation (FE total 199,980 for 3,333 iterations,
  matching the authors' audit file).

## 3. MPHB formulation (internal backbone; paper §5, [CODE] `MPHB_baseline.m`)

The paper describes MPHB only as "the HBA–MPA hybrid without the mediator".
MISSING FROM PAPER: its exact loop. [CODE]:

1. Initialise and evaluate both populations (2N FEs); global best by Eq. (33).
2. Each iteration:
   a. re-evaluate the current MPA population (N FEs; this evaluates the
      FADs-perturbed population of the previous iteration) + memory saving;
   b. HBA and MPA movement interleaved per agent (as in MPHBS Phase 1B),
      HBA greedy acceptance (N FEs);
   c. evaluate moved MPA (N FEs) + memory saving;
   d. global best by Eq. (33);
   e. FADs perturbation **without evaluation / without greedy acceptance**
      (candidate replaces the population, evaluated in step a of the next
      iteration).
3. FE per iteration = 3N; total = 2N + 3N·T (199,950 for CEC2022, T=2221;
   150,000 for FIR, T = 1666) — matches the authors' audit file.

## 4. MPHBS formulation (paper §4, Algorithms 1–2; [CODE] `MPHBS_main.m`)

Outer loop (Algorithm 1):

1. Initialise and evaluate HBA and MPA (2N FEs); Q = zeros(D, N_sub, 2), Eq. (38).
2. Per outer iteration t = 0..T−1:
   * Phase 1A: compute HBA intensity, α, Elite, CF, RL, RB once.
   * Phase 1B: for each agent i: HBA move + MPA move, then HBA clipping and
     greedy acceptance (N FEs).
   * Phase 1C: refresh HBA best.
   * Phase 1D: clip, evaluate MPA (N FEs), memory saving.
   * Phase 1E (only if `fad_before_p2`): FADs stage **with evaluation and
     greedy per-agent acceptance** (N FEs) — DISCREPANCY with Eq. (30), which
     states the perturbed population simply replaces the old one.
   * Global best, Eq. (33) (MPA wins ties: `f_MPA ≤ f_HBA`).
   * Phase 2 (Algorithm 2), §5 below.
   * Reintegrate subsets, refresh bests, keep Pg if not improved.
   * Phase 3F (only if `fad_after_p2`): FADs stage with evaluation + greedy.
   * `Conv(t) = Pg`.

## 5. SARSA mediator (paper §4.2, Eqs. (37)–(61); [CODE])

* Subset size `N_sub = min(N, max(5, round(ρ_sub·N)))`, Eq. (62)
  (MATLAB `round` rounds half away from zero: N=30, ρ=0.25 → 8).
* Elite-plus-random subset: MISSING FROM PAPER (only named). [CODE]:
  `n_elite = floor(N_sub/2)` best by fitness, `n_random = N_sub − n_elite`
  sampled without replacement from the remaining agents; the subset is then
  sorted by fitness (rank 1 = best), for each population separately.
* Episodes per call `E = N_sub·N_i`, Eq. (63).
* Active block size `B = ceil(D/Num_Blocks)`, Num_Blocks = 5, α_lr = 0.10
  [PAPER]; if D < 10, Num_Blocks = 1.
* Dimension order: a random permutation consumed block by block; when the
  pointer passes the end, a new permutation is drawn and the block is completed
  from it (so a block may repeat a dimension across the boundary) [CODE].
* Initial current solution `x_current = P_best` (the global best vector
  itself); its actions are obtained with **ClosestAction**: MISSING FROM PAPER
  (named only). [CODE]: clip the target value to [min, max] of the 2·N_sub
  entries of that column, then pick the entry with minimum absolute difference
  (HBA layer first on ties). `f_current = f(P_best)` costs **one FE** (a
  re-evaluation of an already-known point), Eq. (71)/(85).

## 6. State representation

Paper: the "state" is the dimension index d; the paper explicitly says the
update is "dimension-wise contextual-bandit-style" with TD smoothing and no
sequential MDP (text after Eq. (61)).

## 7. Action representation

Per active dimension d: `a = (n, ch)`, n ∈ {1..N_sub} ranked row, ch ∈ {1=HBA, 2=MPA}
(Eq. (40)). K candidate assemblies per episode use behaviours (Eqs. (41)–(45)):
1 greedy argmax Q; 2 ε-greedy with ε(e); 3 uniform random; 4 second-best Q;
5+ ε-greedy with look-ahead episode ẽ = min(E, e + δ), δ ~ U{1..max(1, ⌊E/10⌋)}
(δ redrawn per dimension [CODE]).

* ε schedule, Eq. (46): `ε(e) = ε_end + (ε_start − ε_end)·exp(−(e−1)/τ_ε)`.
  ε_start, ε_end, τ_ε: MISSING FROM PAPER. [CODE] ε_start = 1.0,
  ε_end = 0.15, τ_ε = max(1, E/4).
* Greedy tie-break [CODE]: max over each layer, HBA preferred if
  `max_HBA ≥ max_MPA`; within a layer MATLAB `max` returns the first index.
* Second-best [CODE]: sort the 2·N_sub values descending (MATLAB stable sort),
  take position 2; HBA layer entries precede MPA entries in the flattened order.

## 8. Tensorized memory

`S_Combined = cat(3, X_HBA_sub, X_MPA_sub) ∈ R^{N_sub×D×2}`, Eq. (37); Q has
the same index structure `D × N_sub × 2`, persistent across outer iterations
(Eq. (38)); rebuilt after each accepted transition.

## 9. Proxy screening (Eqs. (48)–(52))

`Score = −w_rank·mean(n_d) + w_novel·mean|x_d − x_current,d| − w_best·‖x_Dact − P_best,Dact‖₂`,
w_rank = 1.00, w_novel = 0.30 [PAPER]; w_best from the configuration grid.
`n_d` are 1-based ranks [CODE]. The mean rank is taken over the active block.
Arg-max with first-index tie-breaking [CODE].

## 10. Hybrid candidate construction (Eq. (39))

Copy x_current and its action record; overwrite the active dimensions with
`S_Combined(n_d, d, ch_d)`.

## 11. Reward (Eq. (55))

Ternary sign of improvement relative to x_current: +1 / 0 / −1.

## 12. Acceptance / update mechanism (Eqs. (56)–(61))

* `p_g = min(1, t/max(1, T−1))`, `p_l = (e−1)/max(1, E−1)`, Eq. (56) (t is 0-based).
* `Δ_rel = (F_n − F_c)/(|F_c| + eps)`; `a_acc = 0.20(1−p_g)²(0.35 + 0.65(1−p_l))`;
  `p_acc = min(0.25, max(0, a_acc·e^{−4 max(0,Δ_rel)}))`; improving always accepted.
  **Note:** Δ_rel depends on the absolute objective level, so MPHBS behaviour is
  not invariant to additive shifts of f. The authors optimise the CEC2022
  *error* f − f* ([CODE] `cec_error_wrapper`); we do the same for every method.
* Q update, Eqs. (60)–(61), applied to the entry of the **current** action
  `(d, n_cur, ch_cur)`, bootstrapping from the successor action if accepted,
  otherwise from itself. Only active dimensions are updated.
* On acceptance [CODE]: x_current ← x_next; if f_current < max(F_HBA_sub), the
  worst row of the HBA subset is replaced by x_current (same for MPA subset);
  S_Combined rebuilt.
* If `F(x_next) < P_g`: update (P_g, P_best) and inject P_best into a random
  subset position of each population (full-population arrays). [CODE] note:
  because injection targets subset indices and the subsets are written back at
  the end of the mediator call, these injections are overwritten at
  reintegration; the new best still survives through the accepted-candidate
  subset replacement. The port replicates this literally.
* Reintegration of the subsets into the full populations at the end of the
  call; HBA/MPA bests refreshed; P_g kept unless an improvement is found.
* MPA memory (`fit_old`, `X_MPA_old`) is **not** synchronised after Phase 2
  (only after FADs stages) [CODE]; replicated literally.

## 13. Parameter settings

| Parameter | Value | Source |
|---|---|---|
| N | 30 | PAPER |
| HBA C, β | 2, 6 | C: PAPER; β: MISSING FROM PAPER → CODE |
| MPA P, FADs | 0.5, 0.2 | P: PAPER; FADs: MISSING FROM PAPER → CODE |
| Lévy scale / index | 0.05 / 1.5 | MISSING FROM PAPER → CODE |
| Num_Blocks, α_lr | 5, 0.10 | PAPER |
| w_rank, w_novel | 1.00, 0.30 | PAPER |
| ε_start, ε_end, τ_ε | 1.0, 0.15, max(1,E/4) | MISSING FROM PAPER → CODE |
| CEC2022 final B-C1 | K=7, N_i=3, ρ=0.25, w_best=0.050, γ=0.30, FAD **before** | PAPER §5.5 |
| FIR final A-C1 | K=3, N_i=9, ρ=0.25, w_best=0.050, γ=0.30, FAD **after** | PAPER §5.5 |
| Sensitivity grid C1–C4 | (0.050,0.30,0.25), (0.025,0.30,0.25), (0.050,0.20,0.25), (0.050,0.30,0.30) | PAPER Eqs. (94)–(97) |

## 14. CEC2022 protocol (paper §5.1.1, §8)

* 12 functions, D = 20, N = 30, 30 runs, budget 10,000·D = 200,000 FEs,
  bounds [−100,100], official `cec22_test_func`; values reported as error
  `f − f*` (Eq. (93)).
* Outer iterations T are not given in the paper (MISSING FROM PAPER).
  [CODE]: T is calibrated by probing the actual FE counter with T = 1 and T = 2
  on a sphere function, fitting FE(T) = a + bT and taking
  `T = floor((MaxFE − a)/b)`. For B-C1: N_sub = 8, E = 24,
  **FE per iteration = 3N + 1 + E = 115** (2N for HBA+MPA, N for the evaluated
  FADs stage, 1 for f(P_best), E episodes), a = 60 → T = 1738, total FEs =
  199,930 = Table 2. DISCREPANCY: Eq. (85) states `FE_iter = 2N + 1 + E`
  (omits the N FADs evaluations). With Eq. (85) the reported FE totals in
  Table 2 (199,930 / 149,857) cannot be obtained; with the code they are exact.
* Seeds [CODE]: `seed = 20260405 + 1000·function_position + run`, MATLAB
  Mersenne twister; same seed for every method (common random numbers).
  MATLAB RNG streams cannot be reproduced in NumPy, so our reproduction is
  distributional, not bitwise.

## 15. FIR protocol (paper §5.1.2)

* Order 30 → D = 31, h ∈ [−1,1]^31, N = 30, 30 runs, 150,000 FEs, NFFT = 2048,
  grid `w_n = n/(NFFT/2)`, n = 0..1024.
* Eight cases, band sets Eqs. (102)–(105), objective Eqs. (106)–(108),
  Wp = 1, Ws = 100 (Easy/Wide) or 120 (Hard/Narrow). Stored in
  `benchmarks/fir/fir_cases.yaml`; the paper and [CODE] agree exactly.
* MSEs uses |H|² (Eq. 106); MSEp uses (|H| − 1)² (Eq. 107); [CODE] guards
  empty sets with `max(1,·)` (never triggered).
* A-C1: E = 72, FE/iter = 163, T = 919, total 149,857 (= Table 2).

## 16. Statistical analysis (paper §5.2; [CODE] `export_official_block_statistics.m`)

* Per instance: mean, std over 30 runs; mean-based ranks (rank 1 = lowest
  mean); average rank over instances.
* Wilcoxon signed-rank on the 30 seed-paired outcomes (anchor vs competitor);
  [CODE] `signrank(..., 'method','approximate')` (normal approximation), test
  skipped if all differences are zero. Holm correction jointly over all
  anchor–competitor–instance pairs of an analysis stage and domain.
* Friedman test on the matrix of per-instance **means** (instances = blocks);
  Kendall's W with tie correction.
* Post hoc (MISSING FROM PAPER as formula): [CODE]
  `z = (R_anchor − R_j)/sqrt(k(k+1)/(6n))`, two-sided normal p, Holm across
  competitors.
* W/T/L: run-level (per instance, paired runs, tolerance 1e−12) and
  instance-level (per-instance means).

## 17. Computational complexity (paper §4.3)

Time `O(T_max[N log N + N·D + N·C_obj])` when ρ_sub, N_i, K are constants,
≈ `O(T_max·N·C_obj)` when the objective dominates; space `O(N·D)`
(Eqs. (65)–(91)). Per episode `O(K·D + C_obj)`.

## 18. Additional implementation details needed for reproduction

* RANDOM mirror (ablation control, [CODE] `MPHBS_random_mirror.m`): same as
  MPHBS but every component's (n, ch) is uniform random for all K candidates;
  no Q-table; x_current starts at P_best without ClosestAction. (Not part of
  our baseline set; our own RandomExchange ablation is defined for LA-KAIE.)
* External competitors (EODE, RLTLBO, QQLMPA, GWO, FDB-TLABC, FTO) are not
  requested for this study and are not re-implemented.
* Hardware / software of the paper: MATLAB R2024b, Apple M4 Pro, 24 GB RAM,
  macOS 26.3.1. Runtimes are therefore not comparable with ours.
* Paper Eq. (8) writes `β·I_i·X_best,j` — i.e. the intensity term is multiplied
  by the best position component (as in the original HBA code), not by a
  displacement; replicated.

## 19. Summary of paper-vs-code discrepancies and assumptions used by our port

| # | Item | Paper | Code (used) |
|---|---|---|---|
| D1 | FADs stage in MPHBS | replace population (Eq. 30) | evaluate N candidates, greedy per agent |
| D2 | FE per iteration | 2N+1+E (Eq. 85) | 3N+1+E |
| D3 | FADs stage in MPHB | – | not evaluated in place; evaluated next iteration |
| M1 | β (HBA) | – | 6 |
| M2 | FADs | – | 0.2 |
| M3 | Lévy step | – | 0.05·Mantegna(1.5) |
| M4 | ε schedule constants | – | 1.0, 0.15, E/4 |
| M5 | ClosestAction, elite+random subset | named only | defined in §5 |
| M6 | T per budget | – | FE-probe calibration |
| M7 | post hoc formula, Wilcoxon variant | – | z-test on ranks; normal-approx signed-rank |

None of these gaps is critical: the authors' code resolves every one of them,
and the paper's reported FE totals are reproduced exactly by the code's
conventions. Hence no STOP condition (§31.1) was triggered.
