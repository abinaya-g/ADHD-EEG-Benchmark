# SHE Stage 1: structure-identification diagnostic report

**Date:** 2026-10-05.

**Inputs:**
- Specification: `experiments/SHE_SPEC.md` (§16–§17 and Amendment 1).
- Code: `2c3ce05`.
- Raw data: `results/raw/she_diag/` (commit `2fa27b8`).
- Analysis: `scripts/she_stage1_analysis.py` → `results/analysis/she_stage1/summary.json`.

**Campaign:**
- Two variants: SHE-NoE3 (primary) and SHE-Uniform.
- Functions S1 (separable), S2 (block-coupled) and S3 (rotated), each at D ∈ {10, 20}.
- 30 paired seeds (master 75000000), MaxFE = 10000·D.
- E = 24, FADs placed before exchange, N = 30, pre-tuning defaults.
- **360/360 runs ok and valid.** For every run: FE = MaxFE, and exactly E evaluated exchanges in every iteration.

## Gate decision: **STOP**

The pre-registered rule (§17) stops if either condition holds:

| condition | result |
|---|---|
| π near uniform (≥ 4 of 6 cells with median max π̄ < 0.35) | **not met** (2 of 6 cells) |
| **H-E4 fails** | **met:** agreement 0/8 significant; JSD criterion failed; artefact criterion held |

**Stages 2–5 are not implemented.** Following the rule, the fallback assessment is in §6.

---

## 1. H-S1 (identification): **does not hold**, in either variant

**Statistic.** For each run, Δ = π̄(truth) − max over the other hypotheses of π̄. On S1 the competitors are h0 and h3; h2 nests h1 and is excluded there. π̄ is computed over records after the first 10% of FEs.

**Test.** One-sided Wilcoxon, Holm over 6 cells, 95% bootstrap CI of the median.

| cell | SHE-NoE3 median Δ [95% CI] | r_rb | p_Holm | median max π̄ | mean π̄ (h0, h1, h2, h3) |
|---|---|---|---|---|---|
| S1 D10 | −0.076 [−0.275, −0.047] | −0.87 | 1 | 0.308 | 0.357, 0.210, 0.210, 0.222 |
| S1 D20 | −0.075 [−0.374, −0.052] | −1.00 | 1 | 0.303 | 0.375, 0.202, 0.202, 0.222 |
| S2 D10 | −0.233 [−0.549, −0.155] | −1.00 | 1 | 0.422 | 0.458, 0.166, 0.166, 0.209 |
| S2 D20 | −0.461 [−0.494, −0.140] | −1.00 | 1 | 0.596 | 0.479, 0.169, 0.169, 0.182 |
| S3 D10 | −0.800 [−0.800, −0.241] | −0.91 | 1 | 0.850 | 0.643, 0.110, 0.110, 0.136 |
| S3 D20 | −0.801 [−0.801, −0.801] | −0.93 | 1 | 0.851 | 0.702, 0.093, 0.093, 0.112 |

- **SHE-NoE3:** 0/6 cells significant, and all medians are negative.
- **SHE-Uniform (passive π):** also 0/6 significant, with medians from −0.11 to −0.80.

**What π actually did.** π did not stay uniform. It concentrated on **h0**, the structure-free model (intercept + log(1 + ‖z‖)). The concentration is strongest where structure is strongest: on S3, π̄(h0) ≈ 0.70 and the other three sit near the floor. Identical h1/h2 values reflect an almost empty learned linkage (next paragraph).

**Linkage recovery on S2** (descriptive). At the last refresh:
- D10: a median of **1** selected pair, precision 1.0.
- D20: a median of **1** pair, precision 0.375.

The true structure has 20 coupled pairs at D10 and 40 at D20. The L1 outcome model recovers essentially no linkage.

**Selection frequencies (SHE-NoE3).** h0 was chosen in 29–43% of attempts. h0 attempts are free (Amendment A-3), with a mean of 9,000–35,000 per run. These are the **FEs saved by h0** relative to h0-evaluated semantics, though in the fixed-E regime they add records rather than saving budget.

**Exchange success rate:** 0.19–0.31.

## 2. H-E4 (premise of heterogeneous corroboration): **fails**

| part | result |
|---|---|
| (a) agreement on S2/S3, per donor population | Δ^(H) and Δ^(M) are negative in all 8 cells; **0/8 significant** |
| | JSD(π̄^(H), π̄^(M)) median [CI]: S2 D10 0.068 [0.019, 0.140]; S2 D20 0.005 [0.001, 0.017]; S3 D10 0.121 [0.003, 0.209]; S3 D20 0.005 [0.000, 0.183] |
| | **The upper CI is < 0.10 only for S2 D20 → criterion fails** |
| (b) operator artefacts differ (S1) | r = rmse_off(Ā^H, Ā^M) / noise floor: D10 median 5.09 [4.88, 5.36]; D20 7.16 [6.81, 7.49]; both p_Holm = 1.9·10⁻⁹ (r > 2) → **holds** |
| | Off-diagonal RMS correlation of successful native steps on the *separable* S1: HBA 0.11 (D10) / 0.07 (D20); MPA population (MPA move + FAD) 0.65 / 0.70 |

**Reading.**
- The heterogeneous optimizers do carry **strongly different operator artefacts.** The MPA population's successful steps are highly correlated even on a separable function. This replicates the D3 geometry finding with a different measurement.
- But the **outcome-based structure estimates from either donor population do not recover the true structure.** Where the two agree (small JSD), they agree on h0.
- The premise "landscape structure is shared and recoverable, while artefacts differ" fails on its first half.

## 3. η/ρ sweep (Amendment A-5): **absent**

- **Method:** offline recomputation on the SHE-Uniform runs. It is exact; at the defaults the maximum absolute difference from the logged π is 8.0·10⁻⁷.
- **Result:** H-S1 holds at **0 of 9** grid points (η ∈ {0.5, 1, 2} × ρ ∈ {0.99, 0.995, 0.999}). Median Δ is negative in every cell at every point (e.g. S3: −0.80 everywhere).
- **Effect of larger ρ:** it only sharpens the concentration (median max π̄ rises), and the concentration is on h0.
- **Classification: absent.** Identification failure is not an evidence-setting artefact within this range. Under rule A-5.1 the sweep cannot change the gate anyway.

## 4. Post-hoc exploratory check (NOT pre-registered; does not affect the gate)

**Question:** is there any structural signal among {h1, h2, h3} once h0 is removed?

**Method:**
- Data: non-h0 records of the SHE-Uniform runs, evidence restricted to {h1, h2, h3}, recomputed from the logged losses.
- Approximation: those losses came from models also fitted on h0 records.
- Script and output: `results/analysis/she_stage1/exploratory_structural_only.{py,json}`.

**Result:**
- h3 receives the largest share in **every** cell: π̄(h3) = 0.38–0.44 on S1, S2 *and* S3.
- On S3 the contrast is positive (p < 0.01, uncorrected).
- But h3 also "wins" on separable S1 and block S2, so it is not identifying structure.
- **Most plausible reading (inference, not tested):** the pooled-elite eigenframe follows population and operator geometry (see §2b), and h3's features gain from that, not from the landscape.

## 5. Interpretation (limited to what the data show)

1. **Step magnitude, not structure.** Exchange outcomes (y = 1[f(x') < f(x_r)]) are predicted better by the step magnitude alone (h0's 2-parameter model) than by any of the 2D+1-parameter structural models. In other words, on these functions, at this budget, outcome data carry no prequentially usable information about transfer structure beyond step size.
2. **Not a tuning problem.** The failure is not sensitive to η or ρ (§3). It is present with and without evidence feedback (SHE-NoE3 vs SHE-Uniform).
3. **Operator artefacts are large, and population-specific** (§2b). This is consistent with D3, which used independent geometry evidence.
4. **Consequence.** SHE's central mechanism, evidence-based identification of transfer structure from exchange outcomes, is **not supported**. Running Stages 2–5 would test a mechanism that does not identify what it is meant to identify.

## 6. Fallback assessment (Amendment A-6): R2 vs F4

### R2: heterogeneous search processes as FE-free structure estimators

**Rule:** the pre-registered §17 rule names this as the pivot.

**Evidence against it:**
- **Empirical (this report).** The two processes' *outcome-based* estimates do not recover structure (§2a). Their *native-step geometries* differ mainly through operator artefacts (§2b), not through the landscape.
- **D3.** The geometry-based version also failed: no independent landscape component after whitening or residualisation.
- **Literature.** The closest formulation ("successful-step statistics as landscape evidence") was rated **C — abandon** in `FINAL_LITERATURE_OVERLAP_GATE.md`. Contour fitting, the covariance–Hessian relation and structural-bias work were judged to cover it. This rests on search-snippet and code evidence only; no full text was read.

**Verdict: not defensible as stated.** Both its empirical premise (recoverability) and its novelty are contradicted by existing evidence.

**The one supported fact** is that heterogeneous optimizers carry strongly different operator artefacts. That is a descriptive observation adjacent to the C-rated literature, not a structure estimator.

### F4: empirical study plus structure diagnostics

**Content:** the faithful MPHBS reproduction (exact FE totals; 0/80 distributional differences vs the authors' runs) together with these negative results. Each was produced under pre-registered rules:
- the LA-KAIE pilot finding (interaction estimator inactive; learned controller not better than fixed exchange);
- D3 (successful displacements follow operator and population geometry);
- this Stage 1 (outcome evidence identifies no transfer structure; operator artefacts differ strongly between HBA and MPA populations).

**Verdict: defensible as an honest empirical / negative-results contribution.** It claims no new method, so it has no method-novelty claim to defend.

**Weaknesses:**
- poor fit for KBS as a methodological venue (as already noted in `RESEARCH_DECISION_REPORT.md`); a reproducibility or empirical-methods venue fits better;
- the pilot used 5 runs, so its performance statements are exploratory only.

### Comparison

| | R2 | F4 |
|---|---|---|
| empirical premise | contradicted (§2a; D3) | its claims *are* the observed results |
| novelty status | rated C (unverified at full text) | no method claim; novelty of the empirical findings unverified |
| new experiments needed | a new estimator design | none essential; optionally the 30-run MPHBS main reproduction |
| **defensible?** | **no** | **yes, as an empirical paper** |

### Unverified for both

- No prior-art full text was read (network policy). Whether comparable negative or empirical analyses already exist is unknown.
- The MPHBS reproduction's single unresolved discrepancy (CEC F10 escape rate, `MPHBS_REPRODUCTION_REPORT.md`) remains open.
- The 30-run main reproduction campaign has never been run.

**Recommendation.** The pre-registered rule names R2. This assessment finds **F4 defensible and R2 not**. The decision is yours; nothing further is implemented.

## 7. What was NOT verified, and implementation notes

- **Synthetic scope only.** Only S1–S3 at D ∈ {10, 20} and one budget (10000·D). Behaviour on CEC2022/FIR, other budgets or other N is untested (not run, per the gate).
- **Defaults only.** All results use the pre-tuning defaults. η and ρ were swept (§3). p_x, π_min, U, the window lengths and the model form were not varied. In particular, a structural model with fewer parameters, or a different outcome definition, might behave differently; this was **not** tested and would be a redesign.
- **Zero-variance coordinates.** Artefact correlation windows with a zero-variance coordinate had those correlations set to 0 (numpy RuntimeWarnings during the campaign). How often this happened was not logged, so its effect on r in §2b is **not quantified**. It cannot create the large MPA-vs-HBA difference by itself, but this is not verified.
- **h0 accounting.** h0 records are free and numerous (~30–45% of records under SHE-NoE3; 25% of attempts under Uniform). The h0 dominance might partly reflect the record mix. SHE-Uniform, with a fixed 25% h0 share, shows the same dominance, but a run *without* h0 (SHE-no-h0, a Stage 4 variant) was **not** run.
- **Implementation deviation (minor).** Receiver population alternation is implemented per exchange attempt (a global counter starting with an MPA receiver), not "slot 1 by iteration parity" as written in spec §3.2. The rule is still identical across variants.
- **Unspecified choices made in code:**
  - `global_from_bests()` is called after each FADs stage;
  - linkage weights are normalised to mean 1 (scale-free; spec §11 allows this, since λ is relative to λ_max).
- **The §4 exploratory check is post hoc and approximate.**

## 8. Runtime

Mean wall time per run (SHE-NoE3): 18–29 s at D = 10 and 67–121 s at D = 20. The whole campaign took 85 min on 4 workers. No MPHBS timing comparison was made.
