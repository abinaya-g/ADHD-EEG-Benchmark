# LA-KAIE development changes (recorded per research-integrity rule §27)

All changes below were made during development (before any pilot or main
experiment). Only the 20k-FE CEC F1 / 15k-FE FIR1 single-seed development
smoke run (seed 1, not part of any campaign) had been executed with LA-KAIE.

| id | UTC time | component | change | reason | evidence |
|----|----------|-----------|--------|--------|----------|
| DC1 | 2026-09-28T09:53:37Z | interaction estimator | added significance filter: G_ij kept only if the cross-term t-statistic |t| >= t_crit (3.0) | curvature-ratio normalisation produced spurious couplings on a *separable* ellipsoid (mean G 0.35, S8 0.89) | synthetic test (separable / rotated / block-pair / Rastrigin); after the fix separable -> 0, block pairs -> exactly the 10 true pairs. Now covered by tests/test_components.py |
| DC2 | 2026-09-28T09:53:37Z | reward of A1 (no exchange) | A1 credited per-FE: improvement = success = backbone per-offspring success rate, unit cost 1; stagnation penalty uses its expected value S3*(1-success) for all actions | A1 was rewarded per iteration (60-90 FEs) and paid no cost, while each exchange was rewarded per single FE and paid cost w5 -> rewards not commensurable; controller collapsed to "A2 then A1" | development smoke run action counts (logged in PROJECT_LOG) |
| DC3 | 2026-09-28T09:53:37Z | execution | BLAS pinned to 1 thread per process | np.linalg.inv 0.5 s per call due to oversubscription with 4 parallel workers | profile; results unchanged bit-for-bit, runtime 7x lower |

The baseline-validation campaign (HBA/MPA/MPHB/MPHBS) was started from the
working tree before these LA-KAIE changes were committed; none of these
changes touch baseline code (hba.py, mpa.py, mphb.py, mphbs.py, hybrid_common.py).

---

# N1 implementation changes (spec rev. 2.1; change control per user instruction §24)

Scope and status:
- All entries were made before any N1 experimental result existed.
- No file under `lakaie/` (validated backbone, MPHBS, benchmarks) was
  modified.

| id | date | file | change | reason | behaviour change? | associated test |
|----|------|------|--------|--------|-------------------|-----------------|
| N1-DC1 | 2026-09-29 | IMPLEMENTATION_SPEC.md §14.6, header, §17 | E4 decision rule: C2 JOINT > POOLED is primary; C3 supporting; C9 sensitivity (amendment A-1) | explicit user instruction (implementation brief §14) | no (analysis rule) | – |
| N1-DC2 | 2026-09-29 | IMPLEMENTATION_RISK_REGISTER.md, SPEC_REVIEW_REPORT.md | register restored to rev. 2 content (commit cdeb725); review Part II marked reverted | the user named rev. 2 as authoritative after reverting the rev. 3 spec | no | – |
| N1-DC3 | 2026-09-29 | IMPLEMENTATION_SPEC.md §16 | Gate 0 closed by explicit user acceptance; gaps remain gaps; classification B | user statement of 2026-09-29 | no | – |
| N1-DC4 | 2026-09-29 | IMPLEMENTATION_SPEC.md §17 A-2 … A-8 | clarifications: HR basis for single-source/POOLED selectors; strict success for MPA ties; recording proxy; smoke arms; synthetic seed index and n1_calib master; non-finite f stays a hard error; timing of the scale vector | spec ambiguities found in the repository audit (docs/N1_REPOSITORY_AUDIT.md §3) | defines behaviour of new code only | T-MAP, T-FE, T-INIT |
| N1-DC5 | 2026-09-29 | src/n1/* (new) | N1 implementation: records, structure, selectors, native control, SPRT-style rule, mapping, synthetic suite, algorithm, runner | Stage 1 | new code | tests/n1/* |
| N1-DC6 | 2026-09-29 | pytest.ini | `pythonpath = . src` | make `src/n1` importable by tests | no | all |
| N1-DC7 | 2026-09-29 | scripts/n1_reproduce_mphbs.py, n1_run.py, n1_calibrate_sprt.py, n1_smoke_report.py; configs/seeds.json, configs/n1.yaml (new) | stage scripts and configuration | Stages 0, 3a, 3b | no | – |
| N1-DC8 | 2026-09-29 | scripts/n1_calibrate_sprt.py | Level 1 replay uses the A7 smoke p0 sequences instead of A0 | spec §7.2 says "from A0 runs", but A0 has no exchange candidates, so the stated source is impossible; documented deviation | no (calibration only) | – |
| N1-DC9 | 2026-09-29 | tests/n1/test_n1_components.py `_block_cov` | test fixture changed to rotated, ill-conditioned blocks (as in S2) | the old fixture had true within-block correlations of 0.00–0.3, so the block-recovery assertion was unsupportable; the partition code was correct (diagnosed: |ζ| below the Bonferroni threshold 3.69 for the true pairs) | no (test only) | test_partition_recovers_blocks |
| N1-DC10 | 2026-09-29 | tests/n1/test_n1_backbone_frozen.py (new) | sha256 freeze of hybrid_common.py, hba.py, mpa.py, mphbs.py, core.py | T-MPHBS / R-BASE-1 | no | itself |
| N1-DC11 | 2026-09-29 | results/reproduction/MPHBS/AUDIT.md, diagnostic_F10/ | Stage 0 mediator audit and F10 diagnostic (30+30 runs, n1_unit seeds); no code change | P4 failure on CEC F10 | no | – |
| N1-DC12 | 2026-09-29 | MECHANISM_SMOKE_REPORT.md | Stage 3a/3b completed; Gates 5 and 6 FAIL; diagnostic analysis appended; no code or parameter change | hard-stop rule | no | – |
