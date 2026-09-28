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
