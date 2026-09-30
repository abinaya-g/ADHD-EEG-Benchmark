# N1 D3 offline analysis report (EXPLORATORY, offline, zero-FE)

- Date: 2026-09-30.
- Data: `results/raw/n1_d3_diagnostic/` (30 passive A0 runs on the `n1_smoke` seeds).

What was not done:
- no optimizer run, no new seed, no objective evaluation;
- no change to N1, `IMPLEMENTATION_SPEC.md`, BIC, PLL, W, U, the selectors, HBA, MPA, FAD or
  exchange;
- no confirmatory statistics;
- no novelty claim.

Pre-registration:
- The analysis configuration (metrics, whitening, residual models, temporal bins,
  aggregation, figures and the classification rule) was **frozen and committed before any
  metric was computed**: `results/analysis/n1_d3/analysis_config.json`, commit `032164b`.
- The scripts implement it unchanged. Every alternative that was computed is reported:
  - two residual models (R2 primary, R1 secondary);
  - both populations;
  - MPA and FAD separately.

## 1. Data integrity (section 3): PASS

All checks are computed by `d3_analysis.py::integrity()` and stored in `integrity.json`.

| check | result |
|---|---|
| runs present | 30/30 |
| seed equals `n1_smoke` | 30/30 |
| production fields bit-identical (72 fields per run; `trajectory_equivalence.json`) | 30/30 runs, 0 mismatches |
| FE equals `n1_smoke` | 30/30 |
| no NaN/Inf in any `d3_` float field | 30/30 |
| `d3_row_std == d3_row_raw / d3_it_scale` (bitwise) | 30/30 |
| agent identity, HBA: post-phase population = parents + logged moves, at every epoch | max relative error 6.9e-15 |
| agent identity, MPA: parents + logged MPA move + logged FAD move = post population | 0 violations in 300150 agent-epochs |
| logged xprey (HBA target) is a member of the parent HBA population | 10005/10005 epochs |
| logged elite (MPA) is a member of the parent MPA population | 10005/10005 epochs |

## 2. Definitions (frozen)

- **Landscape reference.** Σ_ref = H⁻¹, the inverse Hessian of the quadratic landscape, in
  original coordinates.
  - S1: H = diag(c).
  - S2: H = Pᵀ Mᵀ diag(c) M P (the known block rotation and permutation).
  - S3: H = Qᵀ diag(c) Q.
  - R_ref is the correlation matrix of Σ_ref.
- **Primary metric: `rmse_off`**, the RMS difference of the off-diagonal correlation entries.
  - Lower means more similar.
  - It is invariant to eigenvector sign and ordering, and to per-coordinate scale.
  - Each value is reported with the sampling noise floor 1/√n.
- **Landscape skill** (S2, S3): skill = 1 − rmse_off(R, R_ref) / rmse_off(I, R_ref).
  - 0 means no better than assuming independence; below 0 means worse than independence.
  - For S1 (R_ref = I), rmse_off itself is the distance from the diagonal truth.
- **Representations.**
  - Raw Δx.
  - Target component g = target − parent: HBA xprey, MPA elite. g is exact at epoch
    iterations, where parents are logged.
  - Residual R2: Δx − b·g, with one scalar b per (run, population, bin). This is primary.
  - Residual R1: per-row projection removal. This is secondary.
  - Whitened: C_X^(−1/2) Δx, using the row's own population covariance at the latest
    epoch.
  - Population: the mean per-epoch covariance within the bin.
- **Temporal bins:** 5 equal bins of selection epochs.
- **Aggregation:** mean over bins per run, then median over runs.
- **Table format:** each cell is **HBA / MPA-population**. The MPA population's rows are MPA
  plus FAD moves.

## 3. Primary table (section 12)

| comparison (H / M) | S1 D10 | S1 D20 | S2 D10 | S2 D20 | S3 D10 | S3 D20 |
|---|---:|---:|---:|---:|---:|---:|
| Δx ↔ population (rmse_off ↓) | 0.194 / 0.120 | 0.079 / 0.062 | 0.111 / 0.113 | 0.067 / 0.060 | 0.251 / 0.105 | 0.221 / 0.056 |
| Δx ↔ landscape (rmse_off ↓) | 0.208 / 0.127 | 0.213 / 0.156 | 0.293 / 0.142 | 0.232 / 0.172 | 0.620 / 0.170 | 0.461 / 0.314 |
| population ↔ landscape (rmse_off ↓) | 0.251 / 0.137 | 0.208 / 0.156 | 0.313 / 0.145 | 0.215 / 0.171 | 0.706 / 0.157 | 0.544 / 0.315 |
| target component ↔ Δx (rmse_off ↓) | 0.171 / 0.273 | 0.101 / 0.218 | 0.109 / 0.249 | 0.087 / 0.248 | 0.193 / 0.293 | 0.100 / 0.147 |
| residual R2 ↔ landscape (rmse_off ↓) | 0.244 / 0.147 | 0.195 / 0.188 | 0.302 / 0.327 | 0.215 / 0.265 | 0.541 / 0.536 | 0.447 / 0.417 |
| whitened Δx ↔ landscape (rmse_off ↓) | 0.174 / 0.070 | 0.235 / 0.500 | 0.314 / 0.312 | 0.266 / 0.604 | 0.673 / 0.569 | 0.506 / 0.782 |
| Δx noise floor 1/√n | 0.026 / 0.035 | 0.015 / 0.022 | 0.022 / 0.035 | 0.015 / 0.022 | 0.033 / 0.035 | 0.021 / 0.021 |
| residual noise floor 1/√n | 0.057 / 0.128 | 0.034 / 0.090 | 0.050 / 0.131 | 0.035 / 0.089 | 0.072 / 0.135 | 0.048 / 0.090 |
| independence reference rmse_off(I, R_ref) | 0.000 / 0.000 | 0.000 / 0.000 | 0.291 / 0.291 | 0.120 / 0.120 | 0.542 / 0.542 | 0.397 / 0.397 |

**Landscape skill** (S2/S3; above 0 means closer to R_ref than independence; median over runs):

| comparison (H / M) | S1 D10 | S1 D20 | S2 D10 | S2 D20 | S3 D10 | S3 D20 |
|---|---:|---:|---:|---:|---:|---:|
| Δx skill | (S1: see rmse) | (S1: see rmse) | -0.007 / 0.513 | -0.927 / -0.428 | -0.143 / 0.687 | -0.162 / 0.208 |
| population skill | (S1: see rmse) | (S1: see rmse) | -0.079 / 0.501 | -0.787 / -0.421 | -0.303 / 0.710 | -0.371 / 0.207 |
| target skill | (S1: see rmse) | (S1: see rmse) | -0.124 / -0.151 | -1.134 / -1.174 | -0.158 / -0.133 | -0.226 / -0.067 |
| residual R2 skill | (S1: see rmse) | (S1: see rmse) | -0.040 / -0.126 | -0.788 / -1.203 | 0.002 / 0.012 | -0.126 / -0.052 |
| residual R1 skill | (S1: see rmse) | (S1: see rmse) | -0.213 / 0.000 | -0.539 / -0.258 | -0.191 / 0.000 | -0.101 / -0.030 |
| whitened skill | (S1: see rmse) | (S1: see rmse) | -0.082 / -0.074 | -1.211 / -4.019 | -0.241 / -0.050 | -0.275 / -0.971 |

## 4. Analysis A: displacement vs population geometry

| comparison (H / M) | S1 D10 | S1 D20 | S2 D10 | S2 D20 | S3 D10 | S3 D20 |
|---|---:|---:|---:|---:|---:|---:|
| CS_off(Δx, population) ↑ | 0.695 / 0.648 | 0.895 / 0.907 | 0.921 / 0.922 | 0.919 / 0.937 | 0.919 / 0.979 | 0.883 / 0.985 |
| subspace sim Δx–population ↑ | 0.936 / 0.966 | 0.918 / 0.959 | 0.918 / 0.982 | 0.945 / 0.945 | 0.871 / 0.979 | 0.806 / 0.939 |
| subspace sim Δx–landscape ↑ | 0.768 / 0.896 | 0.655 / 0.831 | 0.587 / 0.963 | 0.657 / 0.832 | 0.618 / 0.901 | 0.352 / 0.668 |
| subspace sim population–landscape ↑ | 0.776 / 0.959 | 0.647 / 0.852 | 0.646 / 0.977 | 0.656 / 0.863 | 0.622 / 0.915 | 0.359 / 0.684 |
| E_pop: Δx variance in top-k population subspace | 0.727 / 0.775 | 0.812 / 0.790 | 0.753 / 0.790 | 0.838 / 0.785 | 0.802 / 0.792 | 0.720 / 0.775 |
| E_land: Δx variance in top-k landscape subspace | 0.695 / 0.769 | 0.664 / 0.749 | 0.616 / 0.782 | 0.612 / 0.755 | 0.646 / 0.776 | 0.371 / 0.672 |
| chance k/D | 0.200 / 0.200 | 0.200 / 0.200 | 0.200 / 0.200 | 0.200 / 0.200 | 0.200 / 0.200 | 0.200 / 0.200 |
| median cos(Δx, target) | 0.970 / 0.435 | 0.947 / 0.409 | 0.973 / 0.549 | 0.951 / 0.415 | 0.971 / 0.392 | 0.929 / 0.420 |
| median ‖Δx‖/‖target‖ | 0.820 / 0.300 | 0.806 / 0.672 | 0.845 / 0.276 | 0.812 / 0.672 | 1.001 / 0.238 | 0.812 / 0.677 |
| fitted b (Δx ≈ b·target) | 0.856 / 0.143 | 0.851 / 0.576 | 0.947 / 0.135 | 0.896 / 0.564 | 0.859 / 0.132 | 1.020 / 0.575 |
| R² of target component | 0.644 / 0.135 | 0.654 / 0.525 | 0.688 / 0.137 | 0.661 / 0.521 | 0.760 / 0.166 | 0.749 / 0.562 |
| subspace sim target–Δx | 0.930 / 0.394 | 0.921 / 0.648 | 0.931 / 0.387 | 0.936 / 0.615 | 0.918 / 0.254 | 0.848 / 0.650 |
| whitened ↔ identity (rmse_off) | 0.174 / 0.070 | 0.235 / 0.500 | 0.099 / 0.066 | 0.218 / 0.569 | 0.323 / 0.077 | 0.271 / 0.613 |
| residual R2 ↔ population (rmse_off) | 0.235 / 0.183 | 0.116 / 0.218 | 0.168 / 0.244 | 0.101 / 0.267 | 0.446 / 0.315 | 0.303 / 0.364 |
| eigen-gap λk/λk+1 of Cov(Δx) | 2.313 / 2.333 | 1.805 / 1.536 | 2.386 / 2.477 | 1.753 / 1.442 | 3.606 / 2.839 | 1.357 / 1.439 |

- In **every cell and both populations**, Δx is closer to the population geometry than to the
  landscape.
  - rmse_off(Δx, X) < rmse_off(Δx, R_ref) in 6/6 cells for HBA and 6/6 for MPA.
- The off-diagonal correlation cosine between Δx and the population is 0.65–0.99, and
  0.88–0.99 at D = 20.
- The top-k principal subspaces of Cov(Δx) and Cov(X) coincide to 0.81–0.98, against a
  chance level of 0.20.
- 72–84% of displacement variance lies in the population's top-k subspace.
- Eigen-gaps λk/λk+1 are 1.4–3.6. The subspace comparisons are therefore not dominated by
  degenerate eigenvalues, although the gap is small at D = 20 (1.4–1.8), which makes top-k
  subspaces less sharply defined.

## 5. Analysis B: displacement vs landscape structure
- **HBA:** Δx is *not* landscape-aligned in any cell.
  - Skill is below 0 on S2 and S3: −0.93 to −0.01, i.e. worse than assuming independence.
  - On S1, rmse_off 0.21 means strong spurious off-diagonal structure, against noise floors
    of 0.015–0.026.
  - The HBA *population* has the same pattern (population skill −0.79 to −0.08).
- **MPA population:** Δx is landscape-aligned on S2 D10 (skill 0.51) and S3 D10 (0.69), and
  weakly on S3 D20 (0.21). It is not aligned on S2 D20 (−0.43).
  - The **MPA population covariance has almost exactly the same skill** (0.50, 0.71, 0.21,
    −0.42).
  - Δx and population agree to rmse_off 0.06–0.11 here, while Δx–landscape is 0.10–0.34.
  - Where MPA displacements look like the landscape, they do so exactly to the extent that
    the MPA population itself does.

## 6. Analysis C: target-direction geometry
- **HBA:** successful displacements are almost parallel to the operator's target direction.
  - median cos(Δx, xprey − x) = 0.93–0.97;
  - fitted Δx ≈ 0.85–1.02 · (xprey − x);
  - the target component explains R² = 0.64–0.76 of displacement energy;
  - target and Δx correlation structures agree (rmse_off 0.09–0.19; subspace similarity
    0.85–0.94).
  - Alignment increases over the run: cosine from 0.85–0.94 in bin 1 to 0.99 in bin 5
    (temporal table).
- **MPA operator:** its own successes are nearly absent after the first third.
  - Example, S2 D20 run 1: attempt success rate 1.5% / 0.0% / 4.8% by third; 1 of 1,415
    MPA success rows falls in the middle third.
  - MPA target and residual metrics are therefore available essentially only in bin 1
    (≈ 55–85 rows) and sometimes bin 5.
  - There: cos(Δx, elite − x) = 0.39–0.55 and R² = 0.13–0.56. The MPA move is only partly
    elite-directed; its Brownian/Lévy component is not logged.
- **FAD:** 87% of the MPA-population success rows (e.g. 9,453 of 10,868 in S2 D20 run 1).
  - FAD has no target vector, so no target component is defined.
- **Mechanism, from the code (a formula-level expectation, not a causal test):**
  - HBA: `hba.py` l.27–29. "honey": new − x = (1 + F·r7·α)(xprey − x). "dig": xprey plus
    terms. The displacement is essentially xprey − x, whose covariance across agents is
    Cov(X), because xprey is shared.
  - FAD: `mpa.py` l.44. The permutation branch (probability 0.8) gives Δx = s·(X_a − X_b),
    whose covariance is 2s²·Cov(X).
  - Both operators are therefore *expected* to produce Cov(Δx) ∝ Cov(X). The data in §4 are
    consistent with that expectation.

## 7. Analysis D: residual geometry
- Removing the target component does **not** reveal landscape structure.
  - Residual R2 skill: HBA −0.79 to 0.00; MPA −1.20 to 0.01.
  - Residual R1 skill: −0.54 to 0.00.
- For the MPA population, removing the elite direction *worsens* landscape alignment on S2
  D10 and S3 D10, from 0.51 and 0.69 to about 0 or below.
  - This residual is computed on bin 1 only (see §6), where alignment is also lower (§9).
    So this comparison is **confounded by time** and is not a clean estimate.
- For HBA, whose residuals are available in all bins, the residual stays about as far from
  the landscape as raw Δx is (S3: 0.54 vs 0.62 at D = 10; 0.45 vs 0.46 at D = 20).

## 8. Analysis E: population whitening
- Whitening by the population covariance **does not reveal landscape structure.** It moves
  the displacements further from the landscape.
  - Whitened skill: HBA −1.21 to −0.08; MPA −4.02 to −0.05.
  - S1 (truth HI): whitened rmse_off 0.17–0.24 (HBA); MPA 0.07 at D = 10 but 0.50 at D = 20.
- Whitened displacements are also not isotropic (rmse_off to I: 0.07–0.61).
- Where the MPA population was landscape-shaped (S2 D10, S3 D10), whitening removes the
  alignment together with the population shape. **No landscape component distinct from the
  population geometry was found.**

## 9. Analysis F: temporal behaviour (pre-specified 5 bins)

| cell / pop | quantity | bin 1 | bin 2 | bin 3 | bin 4 | bin 5 |
|---|---|---:|---:|---:|---:|---:|
| S1 D10 H | Δx↔pop | 0.094 | 0.139 | 0.109 | 0.405 | – |
| S1 D10 H | Δx↔land | 0.205 | 0.185 | 0.219 | 0.220 | – |
| S1 D10 H | res2↔land | 0.216 | 0.229 | 0.276 | 0.289 | – |
| S1 D10 H | cos(Δx,target) | 0.941 | 0.965 | 0.980 | 0.986 | – |
| S1 D10 M | Δx↔pop | 0.069 | 0.112 | 0.099 | 0.139 | 0.123 |
| S1 D10 M | Δx↔land | 0.088 | 0.123 | 0.136 | 0.154 | 0.143 |
| S1 D10 M | res2↔land | 0.147 | – | – | – | – |
| S1 D10 M | cos(Δx,target) | 0.435 | – | – | – | – |
| S1 D20 H | Δx↔pop | 0.094 | 0.089 | 0.079 | 0.061 | 0.062 |
| S1 D20 H | Δx↔land | 0.182 | 0.140 | 0.154 | 0.252 | 0.269 |
| S1 D20 H | res2↔land | 0.208 | 0.177 | 0.153 | 0.238 | 0.256 |
| S1 D20 H | cos(Δx,target) | 0.859 | 0.923 | 0.975 | 0.988 | 0.994 |
| S1 D20 M | Δx↔pop | 0.063 | 0.053 | 0.072 | 0.064 | 0.060 |
| S1 D20 M | Δx↔land | 0.076 | 0.167 | 0.186 | 0.187 | 0.139 |
| S1 D20 M | res2↔land | 0.129 | – | – | – | 0.240 |
| S1 D20 M | cos(Δx,target) | 0.409 | – | – | – | – |
| S2 D10 H | Δx↔pop | 0.099 | 0.131 | 0.138 | 0.067 | 0.085 |
| S2 D10 H | Δx↔land | 0.338 | 0.239 | 0.252 | 0.307 | 0.336 |
| S2 D10 H | res2↔land | 0.373 | 0.230 | 0.264 | 0.284 | 0.318 |
| S2 D10 H | cos(Δx,target) | 0.943 | 0.961 | 0.977 | 0.990 | 0.997 |
| S2 D10 M | Δx↔pop | 0.088 | 0.108 | 0.103 | 0.093 | 0.185 |
| S2 D10 M | Δx↔land | 0.221 | 0.124 | 0.111 | 0.109 | 0.146 |
| S2 D10 M | res2↔land | 0.327 | – | – | – | – |
| S2 D10 M | cos(Δx,target) | 0.549 | – | – | – | – |
| S2 D20 H | Δx↔pop | 0.090 | 0.077 | 0.050 | 0.044 | 0.041 |
| S2 D20 H | Δx↔land | 0.210 | 0.166 | 0.207 | 0.192 | 0.370 |
| S2 D20 H | res2↔land | 0.228 | 0.188 | 0.192 | 0.182 | 0.277 |
| S2 D20 H | cos(Δx,target) | 0.849 | 0.938 | 0.973 | 0.991 | 0.996 |
| S2 D20 M | Δx↔pop | 0.069 | 0.045 | 0.058 | 0.052 | 0.067 |
| S2 D20 M | Δx↔land | 0.126 | 0.159 | 0.195 | 0.203 | 0.164 |
| S2 D20 M | res2↔land | 0.178 | – | – | – | 0.355 |
| S2 D20 M | cos(Δx,target) | 0.415 | – | – | – | – |
| S3 D10 H | Δx↔pop | 0.200 | 0.195 | 0.294 | 0.235 | 0.162 |
| S3 D10 H | Δx↔land | 0.519 | 0.571 | 0.613 | 0.603 | 0.632 |
| S3 D10 H | res2↔land | 0.531 | 0.555 | 0.607 | 0.553 | 0.561 |
| S3 D10 H | cos(Δx,target) | 0.939 | 0.968 | 0.976 | 0.984 | 0.989 |
| S3 D10 M | Δx↔pop | 0.105 | 0.070 | 0.060 | 0.102 | 0.135 |
| S3 D10 M | Δx↔land | 0.437 | 0.089 | 0.083 | 0.081 | 0.112 |
| S3 D10 M | res2↔land | 0.536 | – | – | – | – |
| S3 D10 M | cos(Δx,target) | 0.392 | – | – | – | – |
| S3 D20 H | Δx↔pop | 0.146 | 0.213 | 0.352 | 0.234 | 0.052 |
| S3 D20 H | Δx↔land | 0.474 | 0.473 | 0.467 | 0.474 | 0.433 |
| S3 D20 H | res2↔land | 0.480 | 0.441 | 0.425 | 0.418 | 0.411 |
| S3 D20 H | cos(Δx,target) | 0.852 | 0.894 | 0.955 | 0.977 | 0.987 |
| S3 D20 M | Δx↔pop | 0.099 | 0.039 | 0.033 | 0.043 | 0.067 |
| S3 D20 M | Δx↔land | 0.380 | 0.342 | 0.299 | 0.260 | 0.352 |
| S3 D20 M | res2↔land | 0.425 | – | – | 0.422 | 0.406 |
| S3 D20 M | cos(Δx,target) | 0.400 | – | – | 0.985 | – |

- **HBA:** the structure is **persistent and convergence-dependent**.
  - Δx↔population gets closer over the run on S1/S2 at D = 20 (0.09 → 0.04–0.06).
  - Δx↔landscape does not improve.
  - Target alignment rises monotonically (cosine up to 0.99).
- **MPA population:** alignment with the landscape on S3 D10 appears **after bin 1** (0.44
  → 0.08–0.11) and persists. Δx↔population stays close throughout (0.03–0.14).
  - This is consistent with the MPA population *converging into* the rotated valley, with
    FAD differences inheriting that shape.
  - It is not evidence that displacements carry information beyond the population.

## 10. Analysis G: HBA vs MPA

| comparison (M population: MPA / FAD) | S1 D10 | S1 D20 | S2 D10 | S2 D20 | S3 D10 | S3 D20 |
|---|---:|---:|---:|---:|---:|---:|
| Δx ↔ population | 0.121 / 0.121 | 0.139 / 0.060 | 0.190 / 0.112 | 0.147 / 0.062 | 0.288 / 0.100 | 0.219 / 0.046 |
| Δx ↔ landscape | 0.064 / 0.136 | 0.114 / 0.163 | 0.283 / 0.138 | 0.158 / 0.179 | 0.533 / 0.168 | 0.380 / 0.317 |
| n rows per bin | 290.000 / 792.800 | 677.000 / 1856.600 | 265.000 / 804.600 | 704.500 / 1885.400 | 237.000 / 764.800 | 983.333 / 1958.000 |

| cell | rmse_off(R_HBA, R_MPA) raw | whitened | noise floor | JOINT/POOLED labels raw_std | residual R2 | whitened |
|---|---:|---:|---:|---|---|---|
| S1 D10 | 0.234 | 0.101 | 0.043 | HI/HR:8, HB/HR:4, HR/HR:3, HR/HB:2, HR/HI:1, HI/HB:1, HB/HB:1 | n/a | HR/HR:9, HI/HI:5, HI/HR:3, HI/HB:2, HB/HB:1 |
| S1 D20 | 0.254 | 0.749 | 0.027 | HR/HR:24, HB/HR:1 | HR/HB:1, HB/HR:1 | HR/HR:11, HB/HR:6, HI/HR:4, HI/HB:2, HR/HB:2 |
| S2 D10 | 0.284 | 0.103 | 0.041 | HR/HR:21, HB/HB:2, HB/HR:2 | n/a | HR/HR:10, HI/HI:5, HI/HR:4, HB/HR:2, HB/HI:2, HI/HB:1, HR/HB:1 |
| S2 D20 | 0.260 | 0.770 | 0.027 | HR/HR:24, HB/HR:1 | HR/HR:2, HB/HR:1, HI/HB:1 | HR/HR:15, HI/HB:4, HB/HR:4, HI/HI:2 |
| S3 D10 | 0.591 | 0.285 | 0.050 | HR/HR:24, HR/HB:1 | n/a | HR/HR:19, HR/HB:3, HI/HR:2, HB/HR:1 |
| S3 D20 | 0.467 | 0.711 | 0.029 | HR/HR:25 | HR/HR:5 | HR/HR:25 |

- **HBA and MPA exhibit clearly different displacement geometry.**
  - rmse_off(R_HBA, R_MPA) = 0.23–0.59, against noise floors of 0.027–0.050.
  - The difference is largest on S3 (0.47–0.59).
- The difference is **partly landscape-aligned in one direction.** Where they differ, MPA is
  closer to the landscape on S2 D10, S3 D10 and S3 D20, and HBA never is.
- Each population's displacements track **its own** population geometry (§4). The
  landscape-aligned difference is therefore the difference between the two *populations'*
  shapes. It is not additional structure in the moves.
- Within the MPA population, MPA-operator and FAD rows are both close to the population
  (FAD: 0.05–0.12). The MPA-operator rows are noisier (fewer rows).

## 11. Analysis H: JOINT vs POOLED as a measurement (descriptive only)

See the JOINT/POOLED columns of the table in §10. Label pairs are JOINT/POOLED; counts are
over run × bin.
- On raw standardized rows, JOINT and POOLED both choose HR in most bins on S1 D20, S2 and S3
  (e.g. 24/25 HR/HR on S1 D20 and S2 D20).
- They disagree mainly on S1 D10.
- After whitening, labels spread across HI/HB/HR with **no alignment to the truth**.
- Residual rows are mostly too few for 2W-row windows ("n/a").
- **No structure that JOINT sees and POOLED does not was found that survives population
  adjustment.**
- HBA and MPA do carry different structure (§10), but that difference is population
  geometry.
- The original E4 hypothesis is not supported by these measurements: complementary *landscape*
  information in the moves, beyond population shape. This is descriptive; no test was
  performed.

## 12. Variance / explanation analysis (section 13)

| source | measure | HBA | MPA population | kind of evidence |
|---|---|---|---|---|
| population geometry | E_pop: Δx variance in the top-k population subspace (chance 0.20) | 0.72–0.84 | 0.78–0.79 | association and predictive (population covariance predicts Δx covariance, §4) |
| target direction | R² of the target component | 0.64–0.76 | 0.13–0.56 (MPA operator, bin 1) | association; **mechanistic** by operator formula for HBA and FAD (§6); **not causal** (no intervention) |
| landscape | E_land: Δx variance in the top-k Σ_ref subspace | 0.37–0.70 | 0.67–0.78 | association only; for MPA it equals the population's own landscape alignment |

- **Association:** strong for population and target, weaker for landscape.
- **Predictive:** population geometry predicts displacement geometry better than the
  landscape does, in 12/12 cell × population comparisons.
- **Mechanistic:** the operator formulas imply Cov(Δx) ∝ Cov(X) for HBA-honey and FAD (§6).
- **Causal:** not established. It would need an intervention, e.g. operator ablation or
  controlled population shapes.

## 13. S2 (block structure)
- The known blocks do **not** appear as a recoverable partition in any representation. The
  partition ARI is about 0.00 for raw, residual and whitened, in both populations and at
  both D.
- Within-block share of correlation energy (chance: 0.444 at D = 10, 0.211 at D = 20):

  | representation | D = 10 (HBA / MPA pop.) | D = 20 (HBA / MPA pop.) |
  |---|---|---|
  | raw Δx | 0.59 / 0.88 | 0.25 / 0.38 |
  | population | 0.59 / 0.87 | 0.27 / 0.41 |
  | residual | 0.57 / 0.55 | 0.25 / 0.21 |
  | whitened | 0.46 / 0.51 | 0.22 / 0.27 |

- The MPA population concentrates correlation energy inside the true blocks, well above
  chance. The raw displacements show exactly the population's share.
- The partition procedure nevertheless yields ARI ≈ 0.
  - It connects every significant pair, so the weak cross-block correlations — significant
    at the large per-bin n (≈ 800–1,900 rows) — merge all blocks into one component.
  - The block signal is partial and population-borne, and the connected-components
    partition is not robust to it.
  - This is an inference from the within-block shares and ARI together. The procedure was
    not varied (it is frozen).
- Removing the target direction or whitening pushes the within-block share back toward
  chance.
- **Figure 6:** the landscape shows 4 clean blocks. The population, raw Δx and residual
  correlation matrices all show a diffuse off-block pattern of similar texture, for HBA
  (D = 20, run 1, bin 3).

## 14. S3 (dense rotation)
- **HBA:** raw displacements look dense ("HR-like"), and so does the HBA population. But
  their correlations are *not* the landscape's rotated correlations.
  - Skill −0.14 / −0.16: worse than independence.
  - Target direction: −0.16 / −0.23. Residual: 0.00 / −0.13. Whitened: −0.24 / −0.28.
  - The HR preference seen for HBA on S3 is therefore **dense population / target geometry,
    not landscape rotation.**
- **MPA population:**
  - Displacements *are* aligned with the rotation at D = 10 (skill 0.69) and weakly at
    D = 20 (0.21), but only as much as the MPA population is (0.71 / 0.21).
  - Whitening removes it (−0.05 / −0.97).
  - Residuals (bin 1 only) retain none of it (0.01 / −0.05).
- **Figure 7** shows this for HBA.
- The earlier S3 "HR" selections therefore do not demonstrate detection of the rotation from
  displacement evidence. Where a rotation signal exists, it resides in the MPA population's
  shape.

## 15. Figures (diagnostic, not confirmatory; `results/analysis/n1_d3/figures/`)

| file | content |
|---|---|
| `fig1_spectra.png` | normalised eigen-spectra of Cov(Δx), Cov(X) and Σ_ref (D = 20, run 1, bin 3, HBA) |
| `fig2_similarity_over_time.png` | Δx↔population, Δx↔landscape, residual↔landscape and whitened↔landscape over the 5 bins (D = 20) |
| `fig3_hba_vs_mpa.png` | HBA–MPA correlation-structure difference vs noise floor |
| `fig4_target_vs_actual.png` | median cos(Δx, target − x) per cell and population |
| `fig5_recovery.png` | raw / residual / whitened distance to the landscape vs the independence reference |
| `fig6_S2_blocks.png` | S2 correlation matrices ordered by true block |
| `fig7_S3_rotation.png` | S3 correlation matrices |

Colours: validated reference palette. Tables above are the numeric view.

## 16. Classification (frozen rule, applied mechanically by `d3_report.py`)

- Δx closer to the population than to the landscape: 6/6 cells for HBA and 6/6 for the MPA
  population.
- Landscape criterion met by an adjusted representation (residual R2 or whitened) in any S2
  or S3 cell: **no**. Per-cell details are in `summary.json`.

### **B — OPERATOR/POPULATION GEOMETRY DOMINATES**

Stated honestly:
- The only landscape-aligned structure found in successful displacements occurs for the MPA
  population on S2 D10 and S3 D10 (weakly on S3 D20). It is **fully accounted for by the MPA
  population's own geometry**:
  - equal skill;
  - Δx ≈ population;
  - removed by whitening.
- No landscape component was found that goes beyond population and operator geometry.
- Limitations:
  - The MPA residual analysis is restricted to bin 1 by the MPA operator's near-zero success
    rate in the middle third.
  - Only 5 runs per cell.
  - All results are exploratory.

## 17. Consequences for N1 (no automatic change)

Recommendation, following the B rule: **consider the artefact-characterisation study
instead of continuing the current E4 mechanism.**

Specifically:
1. The current E4 mechanism (structure selection from successful native displacements) is
   **not supported**. On these data, its evidence stream measures population and operator
   geometry.
2. A characterisation study would test, on **new, pre-registered seeds**, the hypotheses this
   exploratory analysis generated:
   - **H-a:** for HBA and FAD, Cov(Δx) ∝ Cov(X), with the stated operator formulas as the
     mechanism.
   - **H-b:** displacement geometry carries landscape structure only to the extent that the
     population geometry does. MPA and FAD on rotated or blocked landscapes are the test
     case.
   - **H-c:** block structure is population-borne but not recoverable by significance-based
     partitioning at large n.
   - Controls: an isotropic sphere; conditioning sweeps; **operator ablations** (e.g. FAD
     off, HBA honey-only) as the interventions needed for causal evidence.
3. The observation that the MPA population's own shape aligns with the landscape (S3 D10) is
   **outside** the current N1 design. Covariance-adaptation methods already exploit
   population geometry, so a novelty check would be needed before building on it.

No N1 code, specification or parameter was changed. STOP.
