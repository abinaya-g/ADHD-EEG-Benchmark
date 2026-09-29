# N1 repository audit

- Date: 2026-09-29.
- Scope: the state of `la-kaie/` before any N1 code was written.
- Authoritative documents:
  - `IMPLEMENTATION_SPEC.md`, rev. 2 (amended to rev. 2.1; see §5);
  - `IMPLEMENTATION_RISK_REGISTER.md`, rev. 2;
  - `SPEC_REVIEW_REPORT.md`;
  - `literature/GATE0_LITERATURE_VERIFICATION.md`;
  - `N1_FINAL_NOVELTY_DECISION.md`.

## 1. Existing modules relevant to N1

| module | content | status |
|---|---|---|
| `lakaie/core.py` | `CountedObjective` (strict FE guard; best-so-far at exact FE resolution; convergence checkpoints; rejects NaN/Inf with `FloatingPointError`), `Recorder`, `levy`, `clip`, `matlab_round`, `calibrate_iterations` | validated; reused unchanged |
| `lakaie/algorithms/hybrid_common.py` | `HybridState`: MPA-then-HBA initialisation from one `rng` (2N FE), `phase1` (HBA N FE, then MPA N FE, with memory saving), `fads_greedy` (N FE), `global_from_bests`, `sync_global_keep`, `memory_saving` | validated backbone; **reused unchanged** |
| `lakaie/algorithms/hba.py`, `mpa.py` | HBA / MPA operators (`hba_move`, `mpa_move`, `fads_candidates`) | validated; used only through `HybridState` |
| `lakaie/algorithms/mphbs.py` | MPHBS port (B-C1 / A-C1 via config) | validated (0/48 CEC and 0/32 FIR distributional differences vs the authors); **A4, unchanged** |
| `lakaie/registry.py` | `run_method`, T calibration by FE probe (`iterations_for`) | reused for A4 |
| `lakaie/experiment.py` | `run_single(task)`: one seeded run producing an immutable JSON + NPZ; `rng = default_rng(seed)` | reused for Stage 0 (A4) |
| `lakaie/benchmarks/cec2022.py`, `fir.py` | official CEC2022 (ctypes) error `f − f*`; FIR objective | reused (Stages 5–6 only) |
| `lakaie/analysis/stats.py` | `holm`, `wilcoxon_p` (normal approximation), Friedman / Kendall W, paired instance tests | reused in Stage 7; the Wilcoxon exact-p choice is to be logged then |
| `scripts/run_experiment.py` | process-pool campaign runner (fork, resumable) | pattern reused |
| `scripts/compare_baseline_validation.py` | Mann–Whitney vs the authors' `task_results.csv`, Holm | logic reused by the Stage 0 script |
| `configs/reproducibility.yaml` | seed formula `master + offset + 1000·instance + run`; `n_workers: 4` | formula reused; file **not modified** (N1 seeds go to `configs/seeds.json`) |
| `configs/cec2022.yaml`, `fir.yaml` | N = 30, budgets, MPHBS B-C1 / A-C1 | reused for A4 |
| `tests/` | 49 tests: baselines (FE totals 199,930 / 149,857; per-iteration FE; determinism), core, stats, LA-KAIE | part of T-MPHBS (must keep passing) |
| `logs/DEVELOPMENT_CHANGES.md` | change log (LA-KAIE DC1–DC3) | N1 entries appended (N1-DCx) |

The FE and seed facts that N1 depends on were verified by reading the
code:

**Per-iteration FE and schedule**
- MPHBS B-C1 costs 3N + 1 + 24 = 115 FE per iteration with T = 1738, for a
  total of 199,930.
- `HybridState` makes three objective calls per iteration, in a fixed
  order:
  1. HBA candidates (N);
  2. MPA candidates (N, clipped in `eval_mpa`);
  3. FAD candidates (N).
- `phase1(prog_t, prog_T)` accepts a float progress. It is used for
  α = 2·exp(−(t+1)/T), CF, and the MPA phase thresholds T/3 and 2T/3.

**Initialisation and seeding**
- A4's generator is `default_rng(seed)`. The first draws are X_M and then
  X_H, inside `HybridState.__init__`.
- An N1 run that also calls `HybridState(obj, …, default_rng(seed))`
  therefore starts from **bit-identical populations** (T-INIT).

**Acceptance rules**
- **MPA memory acceptance** (`memory_saving`): `keep = fit_old < F_new`.
  A move with `F_new == fit_old` is *accepted* by the backbone.
- **HBA and FAD acceptance** are strict (`f_new < f_old`).

## 2. Reusable vs new

**Reused unchanged:**
- `HybridState`;
- `CountedObjective`;
- `Recorder`;
- `clip`;
- MPHBS, the registry and `run_single` (for A4);
- the CEC2022 / FIR problems;
- the statistics helpers.

**New:**
- The N1 data model:
  - native-attempt buffer NA;
  - structural windows Z_H and Z_M (OLD/CUR);
  - the exchange log and selection log;
  - per-phase FE counters;
  - run metadata.
- The structural models HI/HB/HR, with MLE covariance, eigenvalue floor,
  log-likelihood, parameter counts and BIC.
- The Fisher / Bonferroni / connected-components partition (including the
  Stouffer joint version).
- The selectors: HBA-only, MPA-only, JOINT, POOLED and DECOUPLED, plus the
  offline JOINT-HALF.
- Hysteresis.
- The native-control logistic model (L2-IRLS).
- The SPRT-style rule.
- The exchange mapping (HI/HB/HR).
- The N1 iteration loop and arms A0–A3, A5–A8, A10, A11 and P.
- The synthetic suite S1–S8.
- The N1 runner, smoke-gate evaluator and SPRT calibration.

## 3. Compatibility findings and resolutions

| # | finding | resolution | behaviour change to validated code? |
|---|---|---|---|
| F1 | `HybridState.phase1` / `fads_greedy` do not expose per-agent candidates, parents or outcomes. NA needs *all* native attempts, including failures, to compute step lengths | a **recording objective proxy** (`n1.records.RecordingObjective`) wraps the single `CountedObjective` and records each evaluated batch in call order. Parents are snapshotted from `HybridState` immediately before each call. Validated code is not edited | **none**: the proxy forwards the identical array to the identical counter and returns the identical values (tested: T-SHADOW, T-MPHBS regression) |
| F2 | MPA memory accepts ties (`F_new == fit_old`), while spec §3.1 defines `y = 1[f_cand < f_parent]` (strict) | NA `y` and window membership use the **strict** spec definition. Ties are counted and logged (`n_mpa_ties`) | none (backbone unchanged) |
| F3 | spec §8.2 HR mapping uses "the receiver population's Σ̂ from the driving selector". The single-source selectors (A5 HBA-only, A6 MPA-only) and POOLED have **no** covariance for the other population, or no population split at all | **specification ambiguity, resolved by amendment rev. 2.1 (§5):** when the driving selector has no Σ̂ for the receiver population, the selector's only Σ̂ is used for both receivers (single-source: that source; POOLED: the pooled Σ̂). Consequence: A5 and A6 use only their own source's evidence, as their definition requires | n/a (new code) |
| F4 | spec §14.6 (rev. 2) makes E4 an intersection-union rule over C2, C3 and C9. The user's implementation instruction of 2026-09-29 (§14) makes **JOINT > POOLED the primary inferential result**, with C3 supporting and C9 a sensitivity analysis | explicit user instruction; **amendment rev. 2.1** of spec §14.6 and of the header E4 text. Recorded as N1-DC1. No result exists, so this is not post-hoc | no (analysis rule only) |
| F5 | `IMPLEMENTATION_RISK_REGISTER.md` on disk was rev. 3 (the reframing), whereas the spec had been restored to rev. 2 and the user names **rev. 2** of both as authoritative | register restored to its rev. 2 content (commit `cdeb725`) plus a rev. 2.1 note. `SPEC_REVIEW_REPORT.md` Part II is marked "reverted" (not deleted). Recorded as N1-DC2 | no |
| F6 | Gate 0 (spec §16) requires literature resolution **or** explicit user acceptance of the gaps | user statement of 2026-09-29: "The literature gate has been reviewed. Proceed … Remaining inaccessible literature is acknowledged as an unresolved evidence gap." This is taken as **explicit acceptance**; Gate 0 is closed by acceptance, and the novelty classification stays B. Recorded as N1-DC3 | no |
| F7 | stage 3b wording in the user's instruction ("Level 1: false activation / false suppression; Level 2: stopping time") differs from spec §7.2 (Level 1 = semi-synthetic replay, Level 2 = pipeline harness; both report all three quantities) | the spec definition is implemented, and **all three quantities are reported at both levels**. This satisfies both texts | no |
| F8 | the smoke-campaign arms are not enumerated in the spec ("5 runs, synthetic S1–S3 plus control conditions") | smoke = arms **A0** (shadow; E4 gates) and **A7** (full method; FE, log-order and SPRT gates) on S1–S8 × D ∈ {10, 20} × 5 runs, plus A0 with shadows **off** on S1–S3 (T-SHADOW at campaign level). No other arm is run before the gates | n/a |
| F9 | the run-seed instance index for synthetic problems is not defined in the spec | synthetic `instance = 10·fid + (1 if D = 10 else 2)`, offset SYN = 200000 (spec §15.1). This avoids collisions with the CEC/FIR pilot ranges. Calibration campaign master `n1_calib = 83000000` (new). All are recorded in `configs/seeds.json` | n/a |
| F10 | `CountedObjective` raises `FloatingPointError` on a non-finite f, whereas spec §9 treats a non-finite f as a failed candidate | the synthetic functions are finite on the box, and CEC/FIR were validated finite. A non-finite value therefore stays a **hard error**: the run is marked failed and reported, never silently converted. Recorded as a spec note | no |
| F11 | spec §4.1: the scale s is computed "at recording time t" | s is computed once per iteration, after the native phases (step 5), from the current [X_H; X_M]. It is used for all native attempts, displacements and exchanges of iteration t | n/a |

## 4. Files

**Created:**
- `docs/N1_REPOSITORY_AUDIT.md` (this file);
- `configs/seeds.json`;
- `configs/n1.yaml` (hyperparameters from spec §10; arms; campaigns);
- `src/n1/__init__.py`, `config.py`, `records.py`, `structure.py`,
  `selectors.py`, `native_control.py`, `sprt.py`, `mapping.py`,
  `synthetic.py`, `algorithm.py`, `runner.py`;
- `scripts/n1_reproduce_mphbs.py`, `n1_run.py`, `n1_calibrate_sprt.py`,
  `n1_smoke_report.py`;
- `tests/n1/` (conftest plus the test modules for T-FE, T-SHADOW, T-PREQ,
  T-BIC, T-SPRT-UNIT, T-E4-OBS, T-MAP, T-DEGEN, T-MPHBS and T-INIT);
- `MPHBS_REPRODUCTION_REPORT.md`;
- `MECHANISM_SMOKE_REPORT.md`;
- `results/reproduction/MPHBS/…`, `results/raw/n1_smoke/…`,
  `results/n1_calibration/…`.

**Modified:**
- `IMPLEMENTATION_SPEC.md` (rev. 2.1 amendment block: F3, F4, F8–F10);
- `IMPLEMENTATION_RISK_REGISTER.md` (restored to rev. 2, plus a note);
- `SPEC_REVIEW_REPORT.md` (Part II marked reverted);
- `logs/DEVELOPMENT_CHANGES.md` (N1 section);
- `pytest.ini` (adds `src` to `pythonpath`).

**Not modified:** everything under `lakaie/`, `configs/*.yaml` (existing
files), `third_party/` and `benchmarks/`.
