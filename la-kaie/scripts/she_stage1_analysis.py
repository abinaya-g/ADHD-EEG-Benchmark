"""SHE Stage 1 analysis: pre-registered H-S1, H-E4, gate and eta/rho sweep
(experiments/SHE_SPEC.md §16-§17, Amendment 1 A-5). Reads results/raw/she_diag,
writes results/analysis/she_stage1/{summary.json, tables.md}. No tuning.

  python scripts/she_stage1_analysis.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lakaie.analysis.stats import holm  # noqa: E402
from lakaie.she.config import load_yaml  # noqa: E402
from lakaie.she.evidence import posterior  # noqa: E402

RAW = ROOT / "results" / "raw" / "she_diag"
OUT = ROOT / "results" / "analysis" / "she_stage1"
TRUTH = {1: 1, 2: 2, 3: 3}
ALPHA = 0.05


def contrast(pibar, fid):
    """Delta of §16; on S1 h2 is excluded from the competitor set (nested, tie expected)."""
    t = TRUTH[fid]
    others = [0, 3] if fid == 1 else [h for h in range(4) if h != t]
    return float(pibar[t] - max(pibar[h] for h in others))


def jsd2(p, q):
    p, q = np.asarray(p, float) / np.sum(p), np.asarray(q, float) / np.sum(q)
    m = 0.5 * (p + q)

    def kl(a, b):
        nz = a > 0
        return float(np.sum(a[nz] * np.log2(a[nz] / b[nz])))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def boot_median_ci(x, B=10000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    meds = np.median(x[rng.integers(0, x.size, (B, x.size))], axis=1)
    return [float(np.quantile(meds, 0.025)), float(np.quantile(meds, 0.975))]


def rank_biserial(x):
    x = np.asarray(x, float)
    x = x[x != 0]
    if x.size == 0:
        return 0.0
    r = rankdata(np.abs(x))
    return float((r[x > 0].sum() - r[x < 0].sum()) / r.sum())


def wil_greater(x, mu=0.0):
    x = np.asarray(x, float) - mu
    if np.all(x == 0):
        return 1.0
    return float(wilcoxon(x, alternative="greater", zero_method="wilcox").pvalue)


def load(variant, fid, D):
    out = []
    for jp in sorted((RAW / variant / f"S{fid}_D{D}").glob("run*.json")):
        rec = json.loads(jp.read_text())
        z = np.load(jp.with_suffix(".npz"))
        out.append((rec, z))
    return out


def per_run(rec, z, fid):
    burn = z["r_fe"] > 0.1 * rec["max_fe"]
    pibar = z["r_pi"][burn].mean(axis=0).astype(float)
    res = {"seed": rec["seed"], "valid": rec["valid"], "pibar": pibar.tolist(), "delta": contrast(pibar, fid),
           "maxpi": float(pibar.max()), "h0_attempts": rec["summary"]["h0_attempts"],
           "exchange_fe": rec["summary"]["exchange_fe"], "success": float(z["r_y"].mean()),
           "gen_freq": (np.bincount(z["r_gen"], minlength=4) / z["r_gen"].size).tolist(),
           "final_best": rec["final_best"], "runtime": rec["runtime_sec"]}
    if "r_pis" in z.files:
        for nm, d in (("H", 0), ("M", 1)):
            m = burn & (z["r_don"] == d)
            pb = z["r_pis"][m].mean(axis=0).astype(float)
            res[f"pibar_{nm}"] = pb.tolist()
            res[f"delta_{nm}"] = contrast(pb, fid)
        res["jsd_HM"] = jsd2(res["pibar_H"], res["pibar_M"])
    iu = np.triu_indices(z["A_H"].shape[0], 1)
    nH = float(np.mean(z["A_H_n"])) if z["A_H_n"].size else np.nan
    nM = float(np.mean(z["A_M_n"])) if z["A_M_n"].size else np.nan
    rmse = float(np.sqrt(np.mean((z["A_H"][iu] - z["A_M"][iu]) ** 2)))
    res["artefact_rmse"] = rmse
    res["artefact_r"] = float(rmse / np.sqrt(1 / nH + 1 / nM))
    res["artefact_offdiag_H"] = float(np.sqrt(np.mean(z["A_H"][iu] ** 2)))
    res["artefact_offdiag_M"] = float(np.sqrt(np.mean(z["A_M"][iu] ** 2)))
    if rec.get("true_partition") and fid == 2 and z["link_rows"].size:
        lab = np.zeros(z["A_H"].shape[0], int)
        for k, b in enumerate(rec["true_partition"]):
            lab[b] = k
        last = z["link_rows"][:, 0].max()
        P = z["link_rows"][z["link_rows"][:, 0] == last][:, 1:]
        res["linkage_precision_last"] = float(np.mean(lab[P[:, 0]] == lab[P[:, 1]])) if len(P) else np.nan
        res["linkage_k_last"] = int(len(P))
    return res


def sweep_pi(z, eta, rho, pi_min, hyps=(0, 1, 2, 3)):
    """Exact offline recomputation of the pre-record floored posterior under uniform
    selection (Amendment 1 A-5): records, losses and weights do not depend on eta/rho."""
    ell = z["r_loss"].astype(float)
    w = z["r_w"].astype(float)
    L = np.zeros(4)
    out = np.empty_like(ell)
    for n in range(ell.shape[0]):
        out[n] = posterior(L, hyps, eta, pi_min)
        L = rho * L + w[n] * np.nan_to_num(ell[n])
    return out


def cell_stats(vals):
    vals = np.asarray(vals, float)
    return {"median": float(np.median(vals)), "ci95": boot_median_ci(vals), "rank_biserial": rank_biserial(vals),
            "n": int(vals.size)}


def main():
    y = load_yaml()
    camp = y["campaigns"]["she_diag"]
    cells = [(f, D) for f in camp["functions"] for D in camp["dims"]]
    S = {"cells": [f"S{f}_D{D}" for f, D in cells], "variants": {}}
    runs = {}
    for v in camp["variants"]:
        S["variants"][v] = {}
        for f, D in cells:
            rr = [per_run(rec, z, f) for rec, z in load(v, f, D)]
            runs[(v, f, D)] = rr
            S["variants"][v][f"S{f}_D{D}"] = {
                "n_runs": len(rr), "n_valid": int(sum(r["valid"] for r in rr)),
                "delta": cell_stats([r["delta"] for r in rr]),
                "p_one_sided": wil_greater([r["delta"] for r in rr]),
                "maxpi_median": float(np.median([r["maxpi"] for r in rr])),
                "pibar_mean": np.mean([r["pibar"] for r in rr], axis=0).tolist(),
                "gen_freq_mean": np.mean([r["gen_freq"] for r in rr], axis=0).tolist(),
                "success_mean": float(np.mean([r["success"] for r in rr])),
                "h0_attempts_mean": float(np.mean([r["h0_attempts"] for r in rr])),
                "fe_saved_by_h0_mean": float(np.mean([r["h0_attempts"] for r in rr])),
                "runtime_mean_sec": float(np.mean([r["runtime"] for r in rr])),
                "final_best_median": float(np.median([r["final_best"] for r in rr])),
            }
        keys = [f"S{f}_D{D}" for f, D in cells]
        adj = holm([S["variants"][v][k]["p_one_sided"] for k in keys])
        for k, a in zip(keys, adj):
            S["variants"][v][k]["p_holm"] = float(a)
        nsig = int(sum(a < ALPHA for a in adj))
        allpos = all(S["variants"][v][k]["delta"]["median"] > 0 for k in keys)
        nunif = int(sum(S["variants"][v][k]["maxpi_median"] < 0.35 for k in keys))
        S["variants"][v]["H_S1"] = {"n_significant": nsig, "all_medians_positive": allpos,
                                    "holds": bool(nsig >= 5 and allpos), "n_near_uniform_cells": nunif}

    # ---- H-E4 (primary variant SHE-NoE3) ----
    v = "SHE-NoE3"
    e4 = {"agreement": {}, "artefact": {}}
    pa, labels = [], []
    for f in (2, 3):
        for D in camp["dims"]:
            rr = runs[(v, f, D)]
            for nm in ("H", "M"):
                pa.append(wil_greater([r[f"delta_{nm}"] for r in rr])); labels.append(f"S{f}_D{D}_{nm}")
                e4["agreement"][f"S{f}_D{D}_{nm}"] = {"delta": cell_stats([r[f"delta_{nm}"] for r in rr])}
            e4["agreement"][f"S{f}_D{D}_JSD"] = cell_stats([r["jsd_HM"] for r in rr])
    for lab, a, p in zip(labels, holm(pa), pa):
        e4["agreement"][lab]["p_one_sided"] = p; e4["agreement"][lab]["p_holm"] = float(a)
    n_sig_a = int(sum(a < ALPHA for a in holm(pa)))
    jsd_ok = all(e4["agreement"][f"S{f}_D{D}_JSD"]["ci95"][1] < 0.10 for f in (2, 3) for D in camp["dims"])
    pb = []
    for D in camp["dims"]:
        rr = runs[(v, 1, D)]
        pb.append(wil_greater([r["artefact_r"] for r in rr], mu=2.0))
        e4["artefact"][f"S1_D{D}"] = {"r": cell_stats([r["artefact_r"] for r in rr]),
                                      "offdiag_H": cell_stats([r["artefact_offdiag_H"] for r in rr]),
                                      "offdiag_M": cell_stats([r["artefact_offdiag_M"] for r in rr])}
    for D, a, p in zip(camp["dims"], holm(pb), pb):
        e4["artefact"][f"S1_D{D}"]["p_one_sided_r_gt_2"] = p; e4["artefact"][f"S1_D{D}"]["p_holm"] = float(a)
    art_ok = all(a < ALPHA for a in holm(pb))
    e4["holds"] = bool(n_sig_a >= 7 and jsd_ok and art_ok)
    e4["criteria"] = {"agreement_n_significant_of_8": n_sig_a, "jsd_upper_ci_lt_0.10_all": jsd_ok,
                      "artefact_both_significant": art_ok}
    S["H_E4"] = e4

    # ---- linkage recovery (S2, descriptive) ----
    S["linkage_S2"] = {f"D{D}": {"precision_last_median": float(np.nanmedian(
        [r.get("linkage_precision_last", np.nan) for r in runs[(v, 2, D)]])),
        "k_last_median": float(np.median([r.get("linkage_k_last", 0) for r in runs[(v, 2, D)]]))}
        for D in camp["dims"]}

    # ---- gate (§17) on primary variant at defaults ----
    near_uniform = S["variants"][v]["H_S1"]["n_near_uniform_cells"] >= 4
    S["gate"] = {"near_uniform_stop": bool(near_uniform), "H_E4_fails_stop": not e4["holds"],
                 "STOP": bool(near_uniform or not e4["holds"]),
                 "H_S1_primary_holds": S["variants"][v]["H_S1"]["holds"]}

    # ---- eta/rho sweep on SHE-Uniform passive pi (A-5) ----
    d = y["defaults"]
    sw = {}
    check = []
    for eta in camp["sweep"]["eta"]:
        for rho in camp["sweep"]["rho"]:
            key = f"eta={eta},rho={rho}"
            ps, deltas, maxp = [], {}, {}
            for f, D in cells:
                dl, mp_ = [], []
                for rec, z in load("SHE-Uniform", f, D):
                    pi = sweep_pi(z, eta, rho, d["pi_min"])
                    if eta == d["eta"] and rho == d["rho"]:
                        check.append(float(np.max(np.abs(pi - z["r_pi"].astype(float)))))
                    burn = z["r_fe"] > 0.1 * rec["max_fe"]
                    pb_ = pi[burn].mean(axis=0)
                    dl.append(contrast(pb_, f)); mp_.append(float(pb_.max()))
                deltas[f"S{f}_D{D}"] = float(np.median(dl)); maxp[f"S{f}_D{D}"] = float(np.median(mp_))
                ps.append(wil_greater(dl))
            adj = holm(ps)
            nsig = int(sum(a < ALPHA for a in adj))
            sw[key] = {"median_delta": deltas, "median_maxpi": maxp, "p_holm": [float(a) for a in adj],
                       "n_significant": nsig, "H_S1_holds": bool(nsig >= 5 and all(x > 0 for x in deltas.values()))}
    nh = sum(s["H_S1_holds"] for s in sw.values())
    S["sweep"] = {"grid": sw, "n_holds_of_9": int(nh),
                  "classification": "robust" if nh >= 7 else ("setting-sensitive" if nh >= 1 else "absent"),
                  "recompute_check_max_abs_diff_at_defaults": float(max(check)) if check else None}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(S, indent=1))
    print(json.dumps({"gate": S["gate"], "H_S1": {k: S["variants"][k]["H_S1"] for k in S["variants"]},
                      "H_E4": e4["criteria"], "sweep": {k: S["sweep"][k] for k in ("n_holds_of_9", "classification",
                      "recompute_check_max_abs_diff_at_defaults")}}, indent=1))


if __name__ == "__main__":
    main()
