# N1 D3 logging report (measurement only)

- Date: 2026-09-30.
- This pass adds **passive diagnostic logging** and re-runs the existing `n1_smoke` A0
  campaign on S1–S3 with the same seeds and configuration.

What this pass is not:
- It is **not** a new N1 method, a confirmatory experiment, or a parameter search.
- No D3 analysis has been done. Nothing has been whitened, residualised or interpreted.

Unchanged:
- `IMPLEMENTATION_SPEC.md` (rev. 2.1);
- BIC, PLL, the structural selector, W and U;
- all N1 parameters.

`results/raw/n1_smoke/` is untouched: git shows no change under it.

## 1. Fields added

All new fields are stored in each run's NPZ under the `d3_` or `ref_` prefix. Existing fields
are kept unchanged.

| field | shape | content |
|---|---|---|
| `d3_row_t`, `d3_row_pop`, `d3_row_op`, `d3_row_agent` | (rows,) | for every **successful** native displacement: iteration, optimizer (0 = HBA population, 1 = MPA population), operator (0 = HBA, 1 = MPA, 2 = FAD), agent ID |
| `d3_row_raw` | (rows, D) float64 | raw displacement Δx = candidate − parent |
| `d3_row_std` | (rows, D) float64 | the standardized displacement exactly as production computed it, `((Xc − X0) / s)[y]` |
| `d3_row_finite` | (rows,) | the finiteness mask production applied before appending to the windows |
| `d3_it_scale` | (iters, D) | the scale vector s used for standardization in each iteration |
| `d3_att_t`, `d3_att_op`, `d3_att_y` | (3N · iters,) | success indicator for **every** native attempt (successes and failures), with iteration and operator. Order within (t, op) is the agent ID |
| `d3_it_t`, `d3_it_fe`, `d3_it_prog_t`, `d3_T_ref`, `d3_it_CF` | (iters,) | iteration, FE at iteration start, progress variable and horizon passed to `phase1`, and the CF actually returned by `phase1` |
| `d3_it_xprey_H` | (iters, D) | the **HBA target** vector actually passed to the HBA operator (`HybridState.bP_H`, called xprey in the code) |
| `d3_it_elite_M` | (iters, D) | the **MPA elite** vector actually used by the MPA operator (`HybridState.bP_M`, tiled as Elite) |
| `d3_it_Pbest`, `d3_it_Pg`, `d3_it_bF_H`, `d3_it_bF_M` | (iters, D) / (iters,) | global best position and value, and per-population best values, at iteration start |
| `d3_ep_t`, `d3_ep_fe` | (epochs,) | selection-epoch iteration and FE (identical to the production `ep_t`) |
| `d3_ep_XH_pre`, `d3_ep_XM_pre` | (epochs, N, D) | complete HBA / MPA **parent** populations of that iteration (before phase1; MPA = memory) |
| `d3_ep_XH`, `d3_ep_XM`, `d3_ep_FH`, `d3_ep_FM` | (epochs, N, D) / (epochs, N) | complete HBA / MPA populations and fitness **after the native phases** (after FAD) |
| `d3_ep_{H,M}_mean`, `_cov`, `_evals`, `_evecs` | per epoch | population mean, MLE covariance (ddof = 0), eigenvalues and eigenvectors (`numpy.linalg.eigh` of the symmetrised covariance) of the post-native populations |
| `ref_o`, `ref_c` | – | the true optimum shift and conditioning weights |
| `ref_block_label` (S1, S2) | (D,) | block label per coordinate: S1 singletons = diagonal; S2 known blocks |
| `ref_M`, `ref_perm` (S2) | – | block-rotation matrix and coordinate permutation |
| `ref_Q` (S3) | (D, D) | the dense rotation |
| JSON `d3` | – | agent-ID definition, the list of unavailable operator components, seed-source campaign |

**Agent ID** is the population row index. It is the implementation's own identity:
- `HybridState` updates every row in place (HBA greedy, MPA memory, FAD greedy).
- `np.roll` and `rng.permutation` are used only to compute step values, never to reorder rows.
- A0 has no exchange, so no row is ever overwritten by another agent's solution.

IDs were **not** assigned after the fact.

**Naming note.** In this implementation, "xprey" is the HBA target (`bP_H`). MPA uses an elite
(`bP_M`), not an xprey. Both are logged as the vectors actually used.

## 2. Exact logging locations (`src/n1/algorithm.py`, hooks guarded by `if diag is not None`)

| hook | location in the iteration loop | what is recorded |
|---|---|---|
| `iteration_start` | after the plan / stop decision, **before** `phase1` | t, FE, prog_t, T_ref, `bP_H` (HBA target), `bP_M` (MPA elite), Pbest, Pg, best values |
| `cf` | immediately after `phase1` returns | the CF value `phase1` computed and passed on to FAD |
| `scale` | after FAD and `global_from_bests`, at the point where production computes `s` | the scale vector s |
| `native` | inside the production recording loop, right after the window append, once per operator (HBA, MPA, FAD) | success mask, and for successes: agent IDs, raw Δx, the exact standardized rows, the finiteness mask |
| `epoch` | at every `t % U == 0`, before the selectors (inside the zero-FE section) | parent and post-native populations, fitness, mean / covariance / eigen-decomposition |

**Displacement definitions** (exactly those used by production):

| operator | candidate | parent |
|---|---|---|
| HBA | HBA candidate as evaluated (clipped) | HBA row before `phase1` |
| MPA | MPA candidate as evaluated (clipped in `eval_mpa`) | MPA memory row before `phase1` |
| FAD | FAD candidate as evaluated | MPA row after memory saving (before FAD) |

## 3. Trajectory-neutrality mechanism
- `run_n1(..., diag=None)` is the default. With `diag=None` no hook is called, so the
  production code path is the previous one plus five `if` tests.
- The recorder only receives values already computed and stores **copies**. It:
  - draws no random numbers;
  - never calls the objective;
  - never writes to `HybridState`, the windows, the native-control model, the rule or the
    selectors;
  - returns nothing that the loop uses.
- The epoch hook runs inside the existing zero-FE section. Production's hidden-evaluation
  check therefore covers it: `n_hidden_checks == n_iter` in every run.
- The configuration hash is unchanged. Diagnostics are not part of the algorithm
  configuration.

## 4. T-D3-TRAJECTORY result
**Tolerance: none.** Exact equality was required everywhere; it was fixed before comparison.

**(a) Unit test `tests/n1/test_d3_trajectory.py`: PASS**
- Cases: A0 on S1 D10, S2 D20 and S3 D10; A7 (with exchange) on S3 D10.
- Each case runs once with `diag=None` and once with the recorder. The test compares:
  - the FE sequence: batch sizes and total FE;
  - every evaluated point and objective value, bitwise, in call order;
  - the termination point (`n_iter`);
  - the final objective;
  - all Recorder trajectories (best, mean, median, diversity, …);
  - all production logs.
- The recorded HBA and MPA populations are verified to consist of evaluated points.
- Populations are formed only from evaluated points under deterministic acceptance, so an
  identical evaluation sequence implies identical HBA and MPA populations at every
  iteration.
- A second test checks field auditability:
  - `d3_row_std == d3_row_raw / d3_it_scale`, exactly;
  - the logged standardized rows equal the rows production appended to the windows;
  - agent IDs are valid and unique per (t, operator);
  - 3N attempts are logged per iteration;
  - the logged covariance equals `np.cov(ddof=0)`.
- Full suite: **97 passed**.

**(b) Campaign level: PASS**
- Scope: all 30 new runs against the stored `n1_smoke` runs with the same seed.
- **0 of 30 runs with any mismatch** across 72 compared fields per run:
  - JSON: seed, FE, final best, config hash, status, validity, per-phase FE, `n_iter`,
    `n_epochs`, unused FE, window row counts;
  - NPZ: every stored array of the original run, including the convergence curve, all
    per-iteration trajectories, the complete selection log (BIC, likelihood, PLL, labels,
    partitions, digests), epoch pointers, native-attempt summaries, and the displacement
    streams.
- Details: `results/raw/n1_d3_diagnostic/trajectory_equivalence.json`.

## 5. Seed list, FE audit and sizes

- Seeds: `n1_smoke` master (81000000) + SYN offset (200000) + 1000·instance + run, where
  instance = 10·fid + (1 if D = 10 else 2). These are the same seeds as `n1_smoke`.
- Run configuration:
  - git commit: `780bcb399c1afa8ec91a6d87fface8ea6a6f20a6`;
  - configuration hash: `11344a2992950868a0da33753431dcebc10c0c7da62a4a957276f5987fabaeea, 41833a6739f331b8ba01899b29c1888bab2695da2d9bc899b427adf5b6ca088d, 73c114705910927088db0d536584de95721375bb405e97f3034ac49553113745, 766666227c825aaa3976b88421cd284ab10f173cc1eb27eb1c4abf27d98e8ec6, 781d39c61934681bd43e14a064c1a9b8e8b33faa29e8518a06976e22b843a742, bdfcefcfd294c108ac6554eb408bced5cbe729ef784e7cb353e26cdabe06fdd9` (identical to the smoke runs);
  - arm A0; N = 30; MaxFE = 10000·D.

| cell | run | seed | FE | iterations | mismatch vs n1_smoke | NPZ MB |
|---|---|---|---|---|---|---|
| S1_D10 | 01 | 81211001 | 99960 | 1110 | none | 3.6 |
| S1_D10 | 02 | 81211002 | 99960 | 1110 | none | 3.5 |
| S1_D10 | 03 | 81211003 | 99960 | 1110 | none | 3.4 |
| S1_D10 | 04 | 81211004 | 99960 | 1110 | none | 3.5 |
| S1_D10 | 05 | 81211005 | 99960 | 1110 | none | 3.6 |
| S1_D20 | 01 | 81212001 | 199950 | 2221 | none | 20.2 |
| S1_D20 | 02 | 81212002 | 199950 | 2221 | none | 20.0 |
| S1_D20 | 03 | 81212003 | 199950 | 2221 | none | 20.7 |
| S1_D20 | 04 | 81212004 | 199950 | 2221 | none | 19.2 |
| S1_D20 | 05 | 81212005 | 199950 | 2221 | none | 20.0 |
| S2_D10 | 01 | 81221001 | 99960 | 1110 | none | 4.6 |
| S2_D10 | 02 | 81221002 | 99960 | 1110 | none | 4.7 |
| S2_D10 | 03 | 81221003 | 99960 | 1110 | none | 4.6 |
| S2_D10 | 04 | 81221004 | 99960 | 1110 | none | 4.5 |
| S2_D10 | 05 | 81221005 | 99960 | 1110 | none | 4.6 |
| S2_D20 | 01 | 81222001 | 199950 | 2221 | none | 21.2 |
| S2_D20 | 02 | 81222002 | 199950 | 2221 | none | 21.1 |
| S2_D20 | 03 | 81222003 | 199950 | 2221 | none | 21.3 |
| S2_D20 | 04 | 81222004 | 199950 | 2221 | none | 21.6 |
| S2_D20 | 05 | 81222005 | 199950 | 2221 | none | 21.4 |
| S3_D10 | 01 | 81231001 | 99960 | 1110 | none | 3.9 |
| S3_D10 | 02 | 81231002 | 99960 | 1110 | none | 4.0 |
| S3_D10 | 03 | 81231003 | 99960 | 1110 | none | 3.8 |
| S3_D10 | 04 | 81231004 | 99960 | 1110 | none | 3.3 |
| S3_D10 | 05 | 81231005 | 99960 | 1110 | none | 3.8 |
| S3_D20 | 01 | 81232001 | 199950 | 2221 | none | 20.6 |
| S3_D20 | 02 | 81232002 | 199950 | 2221 | none | 21.7 |
| S3_D20 | 03 | 81232003 | 199950 | 2221 | none | 23.9 |
| S3_D20 | 04 | 81232004 | 199950 | 2221 | none | 22.1 |
| S3_D20 | 05 | 81232005 | 199950 | 2221 | none | 21.8 |

**FE audit:**
- per-phase FE sum = counter in all 30 runs (`fe_audit_ok`);
- unused FE < 3N + E_x;
- FE identical to the smoke runs;
- no exchange or probe FE (A0).

**Run count and size:**
- 30 runs, all with status ok;
- 376 MB of NPZ in total (per run 3.3–23.9 MB; D = 20 runs ≈ 20 MB);
- JSON records are small.

## 6. Data completeness

- Every run contains all fields in §1:
  - per-iteration fields for every iteration;
  - attempt masks for 3N · iterations attempts;
  - populations and covariances at every selection epoch (222 at D = 10, 445 at D = 20);
  - raw and standardized rows equal in number to the production window rows.
- The production transformation is audited exactly (`std == raw / s`, bitwise) in all 30
  runs.
- Reference structure:
  - S1 and S2: `ref_block_label`;
  - S2: additionally `ref_M` and `ref_perm`;
  - S3: `ref_Q`. S3 has no block partition by construction, so it has no `ref_block_label`.

## 7. Missing diagnostic fields
These are unavailable **by design**: they cannot be isolated without changing the frozen
operators.
- HBA: intensity I_i, direction flag F_i, dig/honey mode r_i, random factors r3–r7 (drawn
  inside `hba_move`).
- MPA: Brownian RB, Lévy RL, uniform R (drawn inside `mpa_move`). The MPA phase is **not**
  logged as such, but it is a deterministic function of the logged `prog_t` and `T_ref`.
- FAD: branch, mask U, permutations, random factor (drawn inside `fads_candidates`). CF is
  logged.

The operator **target** vectors are logged (HBA `bP_H`, MPA `bP_M`, global `Pbest`), as are
the operator tag and iteration of every displacement. Each displacement can therefore be
matched to the target that directed it. The random operator components above cannot.

None of the fields the task required (agent IDs, raw displacements with scale, populations,
target vectors, population covariance, reference structure) is missing.

## Final status

**READY FOR D3 OFFLINE ANALYSIS**
