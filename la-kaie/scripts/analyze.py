"""Statistical analysis + tables for one campaign/benchmark.

Two stages (as in the MPHBS paper): the *comparison* stage (anchor vs
baselines) and the *ablation* stage (anchor vs LA-KAIE variants). Each stage
gets its own Holm family. Instances are the blocks of every global test.

  python scripts/analyze.py --campaign pilot --benchmark CEC2022
Outputs: results/statistics/<campaign>/<benchmark>/..., results/tables/<campaign>/<benchmark>/...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lakaie.analysis.stats import (descriptive, friedman_block_analysis, load_campaign,  # noqa: E402
                                   paired_instance_tests)

BASELINES = ["HBA", "MPA", "MPHB", "MPHBS"]
ANCHOR = "LA-KAIE-Full"
ABLATIONS = ["LA-KAIE-NoLandscape", "LA-KAIE-NoInteraction", "LA-KAIE-NoKnowledgeReward",
             "LA-KAIE-DimensionOnly", "LA-KAIE-RandomExchange", "LA-KAIE-FixedPolicy",
             "LA-KAIE-NoAdaptiveController"]


def latex_escape(s):
    return str(s).replace("_", r"\_").replace("%", r"\%")


def sci(x):
    return "--" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.3e}"


def result_table(desc, R, methods, item):
    """Paper-style table: rows Mean/Std/Rank per instance + AvgRank row."""
    rows = []
    for inst in sorted(desc.instance.unique()):
        d = desc[desc.instance == inst].set_index("method")
        for stat in ["mean", "std", "median", "best", "worst"]:
            rows.append({item: inst, "statistic": stat.capitalize(),
                         **{m: d.loc[m, stat] if m in d.index else np.nan for m in methods}})
        rows.append({item: inst, "statistic": "Rank", **{m: R.loc[inst, m] for m in methods}})
    rows.append({item: "AvgRank", "statistic": "Rank", **{m: R[m].mean() for m in methods}})
    return pd.DataFrame(rows)


def to_latex(df, path, caption, label, bold_min_rows=("Mean",)):
    cols = list(df.columns)
    lines = [r"\begin{table*}[t]", r"\centering", r"\scriptsize", f"\\caption{{{caption}}}",
             f"\\label{{{label}}}", r"\begin{tabular}{" + "l" * 2 + "r" * (len(cols) - 2) + "}", r"\hline",
             " & ".join(latex_escape(c) for c in cols) + r" \\", r"\hline"]
    for _, r in df.iterrows():
        vals = [r[c] for c in cols[2:]]
        num = [v for v in vals if isinstance(v, (float, int, np.floating)) and np.isfinite(v)]
        mn = min(num) if (num and r[cols[1]] in bold_min_rows) else None
        cells = [latex_escape(r[cols[0]]), latex_escape(r[cols[1]])]
        for v in vals:
            if isinstance(v, (float, np.floating)):
                s = f"{v:.3f}" if r[cols[1]] == "Rank" else sci(v)
                cells.append(f"\\textbf{{{s}}}" if mn is not None and v == mn else s)
            else:
                cells.append(latex_escape(v))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table*}"]
    Path(path).write_text("\n".join(lines) + "\n")


def plain_latex(df, path, caption, label):
    body = df.to_latex(index=False, float_format=lambda v: f"{v:.3g}", escape=True)
    Path(path).write_text(f"\\begin{{table}}[t]\n\\centering\n\\scriptsize\n\\caption{{{caption}}}\n"
                          f"\\label{{{label}}}\n{body}\\end{{table}}\n")


def stage(df, methods, anchor, name, sdir, tdir, bench, item):
    methods = [m for m in dict.fromkeys(methods) if m in set(df.method)]
    if anchor not in methods or len(methods) < 2:
        return None
    d = df[df.method.isin(methods)]
    desc = descriptive(d)
    omni, post, M, R = friedman_block_analysis(d, methods, anchor)
    pw = paired_instance_tests(d, methods, anchor)
    # summary per competitor
    summ = []
    for m in methods:
        if m == anchor:
            continue
        g = pw[pw.method == m]
        p = post[post.method == m].iloc[0]
        summ.append({"method": m, "role": "ablation" if m.startswith("LA-KAIE") else "baseline",
                     "W/T/L (instances, anchor view)": p.wtl_instances,
                     "anchor avg rank": p.anchor_avg_rank, "method avg rank": p.method_avg_rank,
                     "posthoc p (Holm)": p.p_holm, "reject": bool(p["reject_0.05"]),
                     "sig. instances anchor/method": f"{int((g.significant_favours == 'anchor').sum())}/"
                                                     f"{int((g.significant_favours == 'method').sum())}",
                     "run-level W/T/L (anchor view)": f"{g.anchor_wins.sum()}/{g.ties.sum()}/{g.anchor_losses.sum()}"})
    summ = pd.DataFrame(summ)
    # run-level W/T/L of each method against anchor (method perspective, as paper tables 6/10)
    wtl = pw.assign(cell=lambda x: x.anchor_losses.astype(str) + "/" + x.ties.astype(str) + "/" + x.anchor_wins.astype(str))
    wtl = wtl.pivot(index="instance", columns="method", values="cell").reset_index()

    pre = f"{name}"
    desc.to_csv(sdir / f"{pre}_descriptive.csv", index=False)
    M.to_csv(sdir / f"{pre}_mean_matrix.csv"); R.to_csv(sdir / f"{pre}_rank_matrix.csv")
    pw.to_csv(sdir / f"{pre}_wilcoxon_instances.csv", index=False)
    post.to_csv(sdir / f"{pre}_friedman_posthoc.csv", index=False)
    (sdir / f"{pre}_friedman_omnibus.json").write_text(json.dumps(omni, indent=2, default=float))
    summ.to_csv(tdir / f"{pre}_global_summary.csv", index=False)
    wtl.to_csv(tdir / f"{pre}_wtl_runs.csv", index=False)
    rt = result_table(desc, R, methods, item)
    rt.to_csv(tdir / f"{pre}_results.csv", index=False)
    to_latex(rt[rt.statistic.isin(["Mean", "Std", "Rank"])], tdir / f"{pre}_results.tex",
             f"{bench} {name}: mean, standard deviation and mean-based rank over paired runs "
             f"(lower is better; minimum mean in bold).", f"tab:{bench.lower()}_{name}")
    plain_latex(summ, tdir / f"{pre}_global_summary.tex",
                f"{bench} {name}: instances as blocks. Friedman p = {omni['friedman_p']:.3e}, "
                f"Kendall's W = {omni['kendall_w']:.3f}; post hoc p Holm-adjusted across competitors; "
                f"Wilcoxon instance tests Holm-adjusted over the whole stage.", f"tab:{bench.lower()}_{name}_global")
    plain_latex(wtl, tdir / f"{pre}_wtl_runs.tex",
                f"{bench} {name}: run-level wins/ties/losses of each column method against {anchor}.",
                f"tab:{bench.lower()}_{name}_wtl")
    avg = pd.DataFrame({"method": methods, "avg_rank": [R[m].mean() for m in methods]}).sort_values("avg_rank")
    avg.to_csv(tdir / f"{pre}_average_ranks.csv", index=False)
    plain_latex(avg, tdir / f"{pre}_average_ranks.tex", f"{bench} {name}: average mean-based ranks.",
                f"tab:{bench.lower()}_{name}_ranks")
    return {"omnibus": omni, "summary": summ.to_dict(orient="records")}


def resource_tables(df, tdir, bench):
    g = df.groupby("method")
    res = pd.DataFrame({"mean_runtime_s": g.runtime_sec.mean(), "std_runtime_s": g.runtime_sec.std(),
                        "mean_FE": g.FE.mean(), "min_FE": g.FE.min(), "max_FE": g.FE.max(),
                        "max_fes_budget": g.max_fes.max(), "runs": g.size(),
                        "failed_runs": g.status.apply(lambda s: int((s != "ok").sum()))}).reset_index()
    res.to_csv(tdir / "runtime_fe.csv", index=False)
    plain_latex(res, tdir / "runtime_fe.tex", f"{bench}: wall-clock runtime and objective evaluations per run.",
                f"tab:{bench.lower()}_runtime")
    lk = df[df.method.str.startswith("LA-KAIE")]
    if lk.empty:
        return
    rows = []
    for _, r in lk.iterrows():
        s = r.summary or {}
        if not s:
            continue
        ac = s.get("action_counts", {})
        tot = max(1, sum(ac.values()))
        rows.append({"method": r.method, "instance": r.instance, "run": r.run_id,
                     "exchange_frequency": s.get("exchange_frequency"),
                     "successful_exchange_rate": s.get("successful_exchange_rate"),
                     "controller_entropy": s.get("controller_entropy"),
                     "exploration_exploitation_ratio": s.get("exploration_exploitation_ratio"),
                     "group_level_fraction": s.get("group_level_fraction"),
                     "ic_group": s.get("interaction_consistency_group"),
                     "ic_dim": s.get("interaction_consistency_dim"),
                     "final_S8": s.get("final_S8"), "final_n_groups": s.get("final_n_groups"),
                     "overhead_fraction": s.get("overhead_fraction"),
                     **{f"share_{a}": ac.get(a, 0) / tot for a in ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"]}})
    ex = pd.DataFrame(rows)
    ex.to_csv(tdir / "exchange_statistics_runs.csv", index=False)
    agg = ex.drop(columns=["run"]).groupby(["method", "instance"]).mean(numeric_only=True).reset_index()
    agg.to_csv(tdir / "exchange_statistics.csv", index=False)
    agg2 = ex.drop(columns=["run", "instance"]).groupby("method").mean(numeric_only=True).reset_index()
    agg2.to_csv(tdir / "exchange_statistics_by_method.csv", index=False)
    plain_latex(agg2[["method", "exchange_frequency", "successful_exchange_rate", "controller_entropy",
                      "group_level_fraction", "final_S8", "overhead_fraction"]],
                tdir / "exchange_statistics_by_method.tex",
                f"{bench}: LA-KAIE exchange diagnostics averaged over instances and runs.",
                f"tab:{bench.lower()}_exchange")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--anchor", default=ANCHOR)
    args = ap.parse_args()
    df = load_campaign(ROOT / "results" / "raw", args.campaign, args.benchmark)
    if df.empty:
        sys.exit("no runs")
    sdir = ROOT / "results" / "statistics" / args.campaign / args.benchmark
    tdir = ROOT / "results" / "tables" / args.campaign / args.benchmark
    sdir.mkdir(parents=True, exist_ok=True); tdir.mkdir(parents=True, exist_ok=True)
    ok = df[df.status == "ok"]
    item = "Function" if args.benchmark == "CEC2022" else "Case"
    out = {"n_runs": len(df), "n_failed": int((df.status != "ok").sum())}
    out["comparison"] = stage(ok, [args.anchor] + BASELINES, args.anchor, "comparison", sdir, tdir, args.benchmark, item)
    out["ablation"] = stage(ok, [args.anchor] + ABLATIONS, args.anchor, "ablation", sdir, tdir, args.benchmark, item)
    extra = sorted(set(ok.method) - set([args.anchor] + BASELINES + ABLATIONS))
    if extra:
        out["supplementary"] = stage(ok, [args.anchor] + extra, args.anchor, "supplementary", sdir, tdir,
                                     args.benchmark, item)
    allm = [args.anchor] + [m for m in BASELINES + ABLATIONS + extra if m in set(ok.method)]
    out["all_methods"] = stage(ok, allm, args.anchor, "all_methods", sdir, tdir, args.benchmark, item)
    resource_tables(df, tdir, args.benchmark)
    (sdir / "analysis_summary.json").write_text(json.dumps(out, indent=2, default=str))
    for k in ["comparison", "ablation"]:
        if out.get(k):
            print(f"== {k}: Friedman p={out[k]['omnibus']['friedman_p']:.3g} W={out[k]['omnibus']['kendall_w']:.3f}")
            print(pd.DataFrame(out[k]["summary"]).to_string(index=False))


if __name__ == "__main__":
    main()
