# N1 offline evidence diagnostic (EXPLORATORY)

- Date: 2026-09-30.
- **Status: exploratory, zero-FE, offline.**

What was not done:
- no optimizer run, no new seed, no objective evaluation;
- no change to N1 code, to `IMPLEMENTATION_SPEC.md` (rev. 2.1) or to any parameter.

What was used:
- only the existing `n1_smoke` logs: arm A0 (passive shadows), S1–S3 × D ∈ {10, 20} × 5 runs.
- the production selector functions, imported read-only (`n1.selectors.evaluate_sources`,
  `n1.structure.*`). The one non-production estimator (R8, exponential weighting) is defined
  in the script.

Record-keeping:
- The rev. 2.1 smoke result stays recorded as a failure (Gates 5 and 6).
- Nothing here changes that record or the specification.

**Integrity note.**
- All eight representations below, and the separation criterion, were fixed in the analysis
  script *before* it was run once. Every representation tested is reported; none is
  omitted.
- The script is reproduced in full in the appendix.

---

## A. Data available in the existing smoke logs

| # | item | available? | detail |
|---|---|---|---|
| 1 | displacement vectors | **partly** | successful displacements only, **standardized** by the per-coordinate population s.d. at recording time (`streamH_rows`, `streamM_rows`, float32; A0 only). Raw displacements and the scale vector s are **not** logged |
| 2 | success indicators | **partly** | per-iteration and per-operator counts (`na_n`, `na_n_succ`); stream rows are successes by construction. Per-attempt outcomes and failed attempts' displacements are **not** logged (only the mean log-step `na_mean_l`) |
| 3 | iteration index | **yes** | `streamH_t`, `streamM_t`; epoch pointers `ep_t`, `ep_nH`, `ep_nM` |
| 4 | agent ID | **no** | |
| 5 | population positions | **no** | only diversity scalars (`iter_diversity`) |
| 6 | current best / elite / xprey | **no** | only best fitness values (`iter_best`); no positions |
| 7 | optimizer identity | **yes** | separate H / M streams; operator tag HBA / MPA / FAD (`stream*_ops`) |
| 8 | structural-window membership | **yes (reconstructible)** | windows = last W (CUR) and preceding W (OLD) rows at each epoch pointer. The offline reconstruction reproduced the logged JOINT `k_hat` in **1980/1980** sampled epochs |

## Method (fixed before running)

- **Epochs:** every 5th logged selection epoch, per run.
- **Sources:** JOINT (HBA and MPA streams, shared hypothesis, Stouffer partition), as in
  production.
- **Label:** the BIC argmin (`k_hat`, no hysteresis), or the PLL argmax for the PLL rows.
- **Separation criterion:** a representation separates S1/S2/S3 only if, in **all six**
  cells (S1→HI, S2→HB, S3→HR; D = 10, 20), the per-run modal label is correct in
  **≥ 4 of 5 runs**. This is the Gate 5/6 threshold.
- **Table format:** each cell shows `correct runs / 5`, followed by the pooled label shares
  HI/HB/HR.

## G. Scorecard

| Representation | S1 (truth HI) | S2 (truth HB) | S3 (truth HR) | Separates all 3? |
|---|---|---|---|---|
| Raw BIC (production, recomputed) | D10: 4/5 (0.46/0.25/0.29); D20: 0/5 (0.01/0.10/0.89) | D10: 0/5 (0.00/0.14/0.86); D20: 0/5 (0.01/0.08/0.90) | D10: 5/5 (0.00/0.00/1.00); D20: 5/5 (0.00/0.01/0.99) | NO |
| Prequential PLL (logged, lag 1) | D10: 5/5 (0.70/0.25/0.05); D20: 5/5 (0.90/0.06/0.03) | D10: 0/5 (0.49/0.19/0.32); D20: 0/5 (0.91/0.05/0.03) | D10: 3/5 (0.51/0.01/0.48); D20: 0/5 (0.80/0.01/0.18) | NO |
| Agent-thinned (1 row / agent / epoch) | not available (no agent IDs logged) | – | – | not testable |
| Iteration-thinned (1 row / iteration) | D10: 1/5 (0.28/0.22/0.50); D20: 0/5 (0.09/0.03/0.88) | D10: 0/5 (0.00/0.08/0.91); D20: 0/5 (0.08/0.06/0.85) | D10: 5/5 (0.00/0.00/1.00); D20: 5/5 (0.00/0.00/1.00) | NO |
| Iteration-aggregated (mean / iteration) | D10: 1/5 (0.34/0.27/0.39); D20: 0/5 (0.10/0.04/0.86) | D10: 0/5 (0.00/0.09/0.91); D20: 0/5 (0.07/0.06/0.87) | D10: 5/5 (0.00/0.00/1.00); D20: 5/5 (0.00/0.00/1.00) | NO |
| Epoch-aggregated (mean / 5-iteration epoch) | D10: 0/5 (0.08/0.00/0.92); D20: 0/5 (0.14/0.02/0.84) | D10: 0/5 (0.00/0.22/0.78); D20: 0/5 (0.09/0.03/0.88) | D10: 5/5 (0.00/0.00/1.00); D20: 5/5 (0.00/0.00/1.00) | NO |
| Longer window (4W rows) | D10: 0/5 (0.29/0.15/0.56); D20: 0/5 (0.00/0.03/0.97) | D10: 0/5 (0.00/0.00/1.00); D20: 0/5 (0.00/0.03/0.97) | D10: 5/5 (0.00/0.00/1.00); D20: 5/5 (0.00/0.00/1.00) | NO |
| Prequential PLL, lag 2 (gap of W rows) | D10: 5/5 (0.74/0.23/0.03); D20: 5/5 (0.99/0.01/0.00) | D10: 0/5 (0.67/0.17/0.16); D20: 0/5 (0.99/0.01/0.00) | D10: 1/5 (0.64/0.00/0.35); D20: 0/5 (0.92/0.01/0.08) | NO |
| Exponentially weighted accumulation (half-life W) | D10: 5/5 (0.58/0.32/0.10); D20: 0/5 (0.00/0.07/0.92) | D10: 0/5 (0.00/0.06/0.94); D20: 0/5 (0.00/0.09/0.91) | D10: 5/5 (0.00/0.00/1.00); D20: 5/5 (0.00/0.01/0.99) | NO |
| Population-whitened | not available (no population covariance logged) | – | – | not testable |
| Residualized (shared best/elite-directed component removed) | not available (no positions, best, elite or xprey logged) | – | – | not testable |

Supplementary results:
- **Evidence availability** (fraction of sampled epochs with enough rows): Iteration-thinned: S1D10 0.93, S1D20 0.95, S2D10 0.93, S2D20 0.94, S3D10 0.93, S3D20 0.94 ; Epoch-aggregated: S1D10 0.73, S1D20 0.78, S2D10 0.73, S2D20 0.78, S3D10 0.73, S3D20 0.78 ; Longer window: S1D10 0.97, S1D20 0.97, S2D10 0.97, S2D20 0.97, S3D10 0.95, S3D20 0.96 ; Prequential PLL, lag 2: S1D10 0.98, S1D20 0.98, S2D10 0.98, S2D20 0.98, S3D10 0.97, S3D20 0.98. All
  others ≥ 0.97.
- **S2 partition quality** (median adjusted Rand index of the estimated HB partition
  against the true blocks, over epochs where a non-degenerate partition existed):

  | representation | median ARI | n |
  |---|---|---|
  | raw | 0.000 | 350 |
  | iteration-thinned | 0.006 | 477 |
  | long 4W | 0.000 | 37 |

  The true S2 blocks are **not recovered by any representation**.

---

## B. D1 — dependence screen

| representation | tested? | result |
|---|---|---|
| raw rows | yes | as in production |
| one row per agent per epoch | **not possible** (no agent IDs) | – |
| iteration thinning (1 row / iteration) | yes | see below |
| iteration aggregation (mean / iteration) | yes | see below |
| epoch aggregation (mean / epoch) | yes | see below |

**Finding.**
- Removing within-iteration dependence does **not** remove the HR preference.
  - On S1 at D = 20, the HR share stays at 0.84–0.88 under all three reductions (raw 0.89).
  - On S1 at D = 10, the reductions move *further* toward HR (raw 0 correct runs of 5 is
    unchanged or worse).
  - S2 is still labelled HR (0.78–0.91).
- Within-iteration dependence alone therefore does not explain the failure. The spurious
  structure persists *across* iterations, so D1 in its within-iteration form is **not
  sufficient**.
- Across-iteration dependence (the same 30 agents, slowly moving geometry) cannot be
  removed without agent IDs or positions.

## C. D2 — temporal screen

Tested:
- lag-1 prequential PLL (logged);
- lag-2 PLL with a gap of W rows;
- the longer 4W window;
- exponential accumulation.

**Finding.**
- **Structure does not generalise temporally** in a way that matches the landscape.
  - Both PLL variants prefer HI on S1 (correct), but also on S2 and S3 (wrong).
  - The lag-2 PLL is *more* HI-biased than lag-1: S3 at D = 20 gives HI 0.92.
- Whatever correlation exists inside a window does not predict later windows, even on the
  truly rotated S3.
- **Longer or accumulated evidence moves the other way.**
  - The 4W window picks HR on everything, including S1 at D = 10.
  - The exponential weighting does the same, except on S1 at D = 10.
- **No temporal representation tested separates the three.**

## D. D3 — operator / population geometry

- Population positions, population covariance, and best / elite / xprey positions are not
  logged. Raw displacements and the scale vector are not logged either.
- **D3 cannot be isolated from the existing logs.**
- Nothing was reconstructed.

## E. Whitening screen

- Not performed: population covariance is not logged.
- The only population information in the rows is the per-coordinate standardisation that
  was already applied at recording time, which is diagonal only.

## F. Longer-accumulation screen

| window | tested? | result |
|---|---|---|
| current W-row window | yes (R0) | – |
| longer 4W window | yes (R6) | fails |
| exponentially weighted accumulation, half-life W rows | yes (R8) | fails |

- The screen was performed; both fail.
- **Longer accumulation strengthens the HR preference on S1 and S2.** This is consistent with
  a persistent, non-landscape cross-coordinate component in the displacements, which
  averaging does not remove.

---

## Summary of findings

1. **No offline representation that the existing logs support separates S1/S2/S3.**
   - The BIC-type representations (raw, thinned, aggregated, long, exponential) recognise
     S3 but label S1 at D = 20 and S2 as HR.
   - The PLL-type representations recognise S1 but label S2 and S3 as HI.
   - None recovers HB on S2 (ARI ≈ 0).
2. The two families fail in **complementary** directions:
   - in-sample BIC finds cross-coordinate structure everywhere;
   - out-of-sample likelihood finds it nowhere.

   The displacement structure the selectors see is largely unrelated to the known landscape
   structure. This is the pattern expected under D3 (a population- or operator-geometry
   component), but **D3 is not demonstrated**, because it could not be isolated.
3. The leading remaining explanation (D3), and the two remedies that target it (whitening,
   residualization), are **untestable** with the current logs. Agent-level thinning is
   untestable too.

## H. Decision

**D — ADD REQUIRED LOGGING BEFORE FURTHER DIAGNOSIS.**

Justification:
- All representations testable on existing data failed the separation criterion. For
  those, the evidence problem is unresolved (the B-type outcome).
- The critical diagnosis — whether the displacement structure is operator or population
  geometry (D3), and whether whitening or residualization removes it — cannot be performed,
  because population positions or covariance, best / elite / xprey positions, agent IDs
  and raw displacements were not logged.
- A (proceed to a new specification) is **not** justified: no representation separates the
  three structures.
- C (re-scope) is **not yet** justified strongly: the artefact is shown to exist, but its
  cause is not isolated.

Scope of decision D:
- It concerns **logging only**. It does not authorise any change to the N1 mechanism,
  parameters or specification.
- Any added logging, and any new diagnostic runs, would need your approval.
- New runs would have to be passive: A0, trajectory-neutral, with added fields such as
  per-iteration population positions, best, elite and xprey positions, agent IDs, and raw
  displacements with their scale vector.
- Which seeds such runs would use (the existing smoke seeds re-run with extra logging,
  which reproduces identical trajectories, or new ones) is a decision for you.
- This report does not run them.

---

## Appendix: analysis script (run once, as fixed)

```python
"""EXPLORATORY offline zero-FE evidence screen on existing n1_smoke A0 logs.
All representations and the separation criterion are fixed here BEFORE running;
every one is reported. Production N1 code is imported read-only (no changes).

Data: A0 runs, S1/S2/S3 x D in {10,20} x 5 runs (log_windows=True streams).
Epochs: every 5th logged selection epoch (fixed subsample), JOINT sources (H, M).
Label per epoch: k_hat (BIC argmin; no hysteresis) or PLL argmax.

Representations (all per population stream, CUR = last W rows, OLD = preceding W rows):
 R0 raw        : recompute production JOINT BIC from reconstructed windows (sanity vs logged k_hat)
 R1 pll_logged : logged prequential PLL (fit OLD, score CUR) argmax
 R2 agent_thin : NOT AVAILABLE (no agent IDs)
 R3 iter_thin  : first successful row of each iteration (deterministic)
 R4 iter_mean  : mean of the successful rows of each iteration
 R5 epoch_mean : mean of the successful rows of each 5-iteration epoch
 R6 long_4W    : CUR = last 4W raw rows, OLD = preceding 4W rows (production BIC)
 R7 pll_lag2   : PLL fit on rows [n-3W, n-2W), scored on CUR [n-W, n) (gap of W rows)
 R8 exp_weight : exponentially weighted MLE over all rows so far (half-life W rows),
                 BIC with n_eff = (sum w)^2 / sum w^2; HB partition from raw OLD
 R9 pop_whiten, R10 residualised : NOT AVAILABLE (no population covariance / positions / best)
Separation criterion (fixed): a representation separates S1/S2/S3 iff in all 6 cells
(S1->HI, S2->HB, S3->HR; D=10,20) the per-run modal label equals the truth in >= 4/5 runs
(the Gate 5/6 threshold). Also reported: pooled label shares, evidence availability,
median ARI of the HB partition on S2 when available.
"""
import sys, json, glob, math
from collections import Counter
import numpy as np
ROOT = "/home/user/ADHD-EEG-Benchmark/la-kaie"
sys.path[:0] = [ROOT, ROOT + "/src"]
from n1.config import N1Config
from n1.selectors import evaluate_sources
from n1.structure import (scatter, fit_model, loglik_from_scatter, predictive_loglik, bic,
                          stouffer_stat, partition_from_stat, degeneracy)
CFG = N1Config(); TRUTH = {1: "HI", 2: "HB", 3: "HR"}; KS = ("HI", "HB", "HR"); QO = {"HI": 0, "HB": 1, "HR": 2}

def group_rows(R, T, key, how):
    if len(R) == 0: return R
    g = key(T); _, first, inv = np.unique(g, return_index=True, return_inverse=True)
    if how == "first": return R[np.sort(first)]
    sums = np.zeros((first.size, R.shape[1])); np.add.at(sums, inv, R)
    cnt = np.bincount(inv).astype(float)
    order = np.argsort(first)  # chronological
    return (sums / cnt[:, None])[order]

def win(rows, L):
    if rows.shape[0] < L: return None, None
    cur = rows[-L:]; old = rows[-2 * L:-L] if rows.shape[0] >= 2 * L else None
    return cur, old

def bic_label(srcs):
    r = evaluate_sources("X", srcs, "stouffer", CFG)
    return (r.k_hat if r.evidence else None), r

def pll_label(fit_rows, score_rows):
    """fit on fit_rows[p], score on score_rows[p]; partition from fit rows (Stouffer)."""
    if any(v is None for v in list(fit_rows.values()) + list(score_rows.values())): return None
    part = partition_from_stat(stouffer_stat(list(fit_rows.values()), CFG.fisher_clip), CFG.alpha_link)
    ks = ["HI", "HR"] + (["HB"] if degeneracy(part) is None else [])
    best, bv = None, -np.inf
    for k in ks:
        v = 0.0
        for p in fit_rows:
            mu, S = scatter(fit_rows[p]); eps = CFG.eps_rel * np.trace(S) / S.shape[0]
            v += predictive_loglik(score_rows[p], mu, fit_model(S, k, part if k == "HB" else None).Sigma, eps)
        if v > bv: best, bv = k, v
    return best

def exp_label(rows_by_pop, old_by_pop, W):
    if any(r.shape[0] < W for r in rows_by_pop.values()): return None
    part = None
    if all(o is not None for o in old_by_pop.values()):
        part = partition_from_stat(stouffer_stat(list(old_by_pop.values()), CFG.fisher_clip), CFG.alpha_link)
        if degeneracy(part) is not None: part = None
    ks = ["HI", "HR"] + (["HB"] if part is not None else [])
    scores = {}
    for k in ks:
        b = 0.0
        for p, R in rows_by_pop.items():
            n = R.shape[0]; w = 0.5 ** ((n - 1 - np.arange(n)) / W)
            neff = w.sum() ** 2 / (w * w).sum(); wn = w / w.sum()
            mu = wn @ R; C = R - mu; S = (C * wn[:, None]).T @ C
            eps = CFG.eps_rel * np.trace(S) / S.shape[0]
            f = fit_model(S, k, part if k == "HB" else None)
            b += bic(loglik_from_scatter(neff, S, f.Sigma, eps), f.q, neff)
        scores[k] = b
    return min(scores, key=lambda k: (scores[k], QO[k]))

def ari(a, b):
    from math import comb
    n = len(a); ct = Counter(zip(a, b)); sa, sb = Counter(a), Counter(b)
    idx = sum(comb(v, 2) for v in ct.values()); ea = sum(comb(v, 2) for v in sa.values()); eb = sum(comb(v, 2) for v in sb.values())
    exp = ea * eb / comb(n, 2); mx = (ea + eb) / 2
    return 1.0 if mx == exp else (idx - exp) / (mx - exp)

def labels_of(part, D):
    lab = np.zeros(D, int)
    for i, b in enumerate(part): lab[np.asarray(b)] = i
    return lab.tolist()

REPS = ["R0_raw", "R1_pll_logged", "R3_iter_thin", "R4_iter_mean", "R5_epoch_mean", "R6_long_4W", "R7_pll_lag2", "R8_exp_weight"]
out = {}; agree = [0, 0]; aris = {r: [] for r in ("R0_raw", "R3_iter_thin", "R6_long_4W")}
for fid in (1, 2, 3):
    for D in (10, 20):
        cell = {r: {"runs": [], "all": Counter(), "n_ep": 0, "n_ev": 0} for r in REPS}
        for p in sorted(glob.glob(f"{ROOT}/results/raw/n1_smoke/A0/S{fid}_D{D}/run*.npz")):
            z = np.load(p); js = json.load(open(p.replace(".npz", ".json"))); W = js["summary"]["W"]
            tp = js["true_partition"]
            RH, TH = z["streamH_rows"].astype(float), z["streamH_t"]; RM, TM = z["streamM_rows"].astype(float), z["streamM_t"]
            sj = z["s_selector"] == "JOINT"; logged_khat = z["s_k_hat"][sj]; logged_ev = z["s_evidence"][sj]
            P = np.vstack([z["s_pll_HI"][sj], z["s_pll_HB"][sj], z["s_pll_HR"][sj]]).T
            runlab = {r: Counter() for r in REPS}
            for ei in range(0, len(z["ep_t"]), 5):
                nH, nM = int(z["ep_nH"][ei]), int(z["ep_nM"][ei])
                H, M = RH[:nH], RM[:nM]; tH, tM = TH[:nH], TM[:nM]
                lab = {}
                cH, oH = win(H, W); cM, oM = win(M, W)
                if cH is not None and cM is not None:
                    k, r = bic_label({"H": (cH, oH), "M": (cM, oM)}); lab["R0_raw"] = k
                    agree[1] += 1; agree[0] += int(k == logged_khat[ei])
                    if fid == 2 and r.partition is not None and r.degenerate is None and tp:
                        aris["R0_raw"].append(ari(labels_of(r.partition, D), labels_of(tp, D)))
                if logged_ev[ei] and not np.isnan(P[ei, 0]):
                    Pb = np.where(np.isnan(P[ei]), -np.inf, P[ei]); lab["R1_pll_logged"] = KS[int(np.argmax(Pb))]
                for name, key, how in (("R3_iter_thin", lambda t: t, "first"), ("R4_iter_mean", lambda t: t, "mean"),
                                       ("R5_epoch_mean", lambda t: t // CFG.U, "mean")):
                    h = group_rows(H, tH, key, how); m = group_rows(M, tM, key, how)
                    a, b = win(h, W); c, d = win(m, W)
                    if a is not None and c is not None:
                        k, r = bic_label({"H": (a, b), "M": (c, d)}); lab[name] = k
                        if name == "R3_iter_thin" and fid == 2 and r.partition is not None and r.degenerate is None and tp:
                            aris[name].append(ari(labels_of(r.partition, D), labels_of(tp, D)))
                a, b = win(H, 4 * W); c, d = win(M, 4 * W)
                if a is not None and c is not None:
                    k, r = bic_label({"H": (a, b), "M": (c, d)}); lab["R6_long_4W"] = k
                    if fid == 2 and r.partition is not None and r.degenerate is None and tp:
                        aris["R6_long_4W"].append(ari(labels_of(r.partition, D), labels_of(tp, D)))
                if H.shape[0] >= 3 * W and M.shape[0] >= 3 * W:
                    lab["R7_pll_lag2"] = pll_label({"H": H[-3 * W:-2 * W], "M": M[-3 * W:-2 * W]}, {"H": H[-W:], "M": M[-W:]})
                oldH = H[-2 * W:-W] if H.shape[0] >= 2 * W else None; oldM = M[-2 * W:-W] if M.shape[0] >= 2 * W else None
                lab["R8_exp_weight"] = exp_label({"H": H, "M": M}, {"H": oldH, "M": oldM}, W)
                for r in REPS:
                    cell[r]["n_ep"] += 1
                    if lab.get(r):
                        cell[r]["n_ev"] += 1; cell[r]["all"][lab[r]] += 1; runlab[r][lab[r]] += 1
            for r in REPS:
                cell[r]["runs"].append(runlab[r].most_common(1)[0][0] if runlab[r] else "none")
        out[(fid, D)] = cell
res = {"agreement_R0_vs_logged": agree, "aris": {k: [float(np.median(v)) if v else None, len(v)] for k, v in aris.items()}, "cells": {}}
for (fid, D), cell in out.items():
    for r in REPS:
        c = cell[r]; tot = sum(c["all"].values())
        res["cells"][f"S{fid}_D{D}_{r}"] = {"shares": {k: (c["all"][k] / tot if tot else None) for k in KS},
            "runs_modal": c["runs"], "n_correct_runs": sum(x == TRUTH[fid] for x in c["runs"]),
            "evidence_frac": c["n_ev"] / c["n_ep"] if c["n_ep"] else 0}
json.dump(res, open(sys.argv[1], "w"), indent=1)
print("R0 reconstruction agreement with logged k_hat:", agree)
for r in REPS:
    row = []
    ok = True
    for fid in (1, 2, 3):
        for D in (10, 20):
            c = res["cells"][f"S{fid}_D{D}_{r}"]; ok &= c["n_correct_runs"] >= 4
            sh = c["shares"]; row.append(f"S{fid}D{D}:{c['n_correct_runs']}/5 [" + "/".join("-" if sh[k] is None else f"{sh[k]:.2f}" for k in KS) + f"] ev{c['evidence_frac']:.2f}")
    print(r, "SEPARATES" if ok else "no", " | ".join(row))
print("ARI S2 (median, n):", res["aris"])

```
