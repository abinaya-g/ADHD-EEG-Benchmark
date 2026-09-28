"""Statistical analysis following the MPHBS protocol (paper Sec. 5.2, authors'
export_official_block_statistics.m), with instances (functions / FIR cases)
as the independent blocks of all global tests.

* descriptive: mean, std, median, best, worst over paired runs
* ranks: per instance, rank of the mean (1 = lowest); average rank
* Wilcoxon signed-rank on seed-paired runs (normal approximation, zero
  differences discarded, as MATLAB signrank 'approximate'); Holm over the whole
  anchor x competitor x instance family of a stage/domain
* Friedman test on the instance x method matrix of means; Kendall's W with tie
  correction; anchor-vs-competitor post hoc z = (R_a - R_j)/sqrt(k(k+1)/(6n)),
  two-sided normal p, Holm across competitors
* W/T/L: run level (per instance, tolerance 1e-12) and instance level (means)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

TIE_TOL = 1e-12


def load_campaign(raw_root: Path, campaign: str, benchmark: str) -> pd.DataFrame:
    rows = []
    for jp in sorted((Path(raw_root) / campaign / benchmark).glob("*/inst*/run*.json")):
        r = json.loads(jp.read_text())
        rows.append({k: r.get(k) for k in ["method", "benchmark", "instance", "instance_name", "run_id",
                                           "seed", "best_fitness", "FE", "max_fes", "runtime_sec", "status",
                                           "iterations", "final_diversity", "final_stagnation",
                                           "population_size", "dimension"]}
                    | {"summary": r.get("summary", {}), "path": str(jp)})
    return pd.DataFrame(rows)


def holm(p):
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    order = np.argsort(pv, kind="stable")
    m = pv.size
    adj = np.empty(m)
    running = 0.0
    for i, idx in enumerate(order):
        running = max(running, (m - i) * pv[idx])
        adj[idx] = min(1.0, running)
    out[ok] = adj
    return out


def wilcoxon_p(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    if np.all(np.abs(d) <= 0):
        return np.nan
    try:
        return float(stats.wilcoxon(a, b, zero_method="wilcox", method="approx", correction=False).pvalue)
    except ValueError:
        return np.nan


def descriptive(df, value="best_fitness"):
    g = df.groupby(["instance", "method"])[value]
    return g.agg(mean="mean", std="std", median="median", best="min", worst="max", n="count").reset_index()


def mean_matrix(df, methods, value="best_fitness"):
    M = df.pivot_table(index="instance", columns="method", values=value, aggfunc="mean")
    return M[methods]


def rank_matrix(M: pd.DataFrame):
    return M.rank(axis=1, method="average")


def kendall_w(R: np.ndarray) -> float:
    n, k = R.shape
    Rs = R.sum(axis=0)
    ss = np.sum((Rs - n * (k + 1) / 2) ** 2)
    tie = 0.0
    for row in R:
        _, c = np.unique(row, return_counts=True)
        tie += np.sum(c ** 3 - c)
    den = n ** 2 * (k ** 3 - k) - n * tie
    return float(12 * ss / den) if den > 0 else 0.0


def friedman_block_analysis(df, methods, anchor, value="best_fitness"):
    M = mean_matrix(df, methods, value)
    R = rank_matrix(M)
    n, k = R.shape
    chi, p = stats.friedmanchisquare(*[M[m].values for m in methods]) if k >= 3 else (np.nan, np.nan)
    avg = R.mean(axis=0)
    se = np.sqrt(k * (k + 1) / (6 * n))
    rows = []
    for m in methods:
        if m == anchor:
            continue
        z = (avg[anchor] - avg[m]) / se
        w = int(np.sum(M[anchor] < M[m] - TIE_TOL))
        l = int(np.sum(M[anchor] > M[m] + TIE_TOL))
        rows.append({"method": m, "anchor": anchor, "wtl_instances": f"{w}/{n - w - l}/{l}",
                     "anchor_avg_rank": avg[anchor], "method_avg_rank": avg[m],
                     "z": z, "p_raw": 2 * stats.norm.sf(abs(z))})
    post = pd.DataFrame(rows)
    if len(post):
        post["p_holm"] = holm(post["p_raw"].values)
        post["reject_0.05"] = post["p_holm"] < 0.05
    omni = {"n_blocks": n, "k_methods": k, "friedman_chi2": float(chi), "friedman_p": float(p),
            "kendall_w": kendall_w(R.values), "avg_ranks": avg.to_dict()}
    return omni, post, M, R


def paired_instance_tests(df, methods, anchor, value="best_fitness"):
    rows = []
    for inst, g in df.groupby("instance"):
        A = g[g.method == anchor].set_index("run_id")[value]
        for m in methods:
            if m == anchor:
                continue
            B = g[g.method == m].set_index("run_id")[value]
            common = A.index.intersection(B.index)
            a, b = A.loc[common].values, B.loc[common].values
            d = a - b
            rows.append({"instance": inst, "anchor": anchor, "method": m, "n_pairs": len(common),
                         "anchor_wins": int(np.sum(d < -TIE_TOL)), "ties": int(np.sum(np.abs(d) <= TIE_TOL)),
                         "anchor_losses": int(np.sum(d > TIE_TOL)),
                         "anchor_mean": a.mean(), "method_mean": b.mean(),
                         "anchor_median": np.median(a), "method_median": np.median(b),
                         "median_paired_diff": float(np.median(d)) if len(d) else np.nan,
                         "p_raw": wilcoxon_p(a, b)})
    T = pd.DataFrame(rows)
    T["p_holm_domain"] = holm(T["p_raw"].values)
    sig = T["p_holm_domain"] < 0.05
    T["significant_favours"] = np.where(~sig, "none",
                                        np.where(T["median_paired_diff"] < 0, "anchor", "method"))
    return T
