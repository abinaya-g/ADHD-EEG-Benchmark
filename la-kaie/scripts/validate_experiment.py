"""Automatic integrity checks for a campaign (default: main).

Verifies: no missing runs, correct number of runs / functions / FIR cases,
FE budget never exceeded, identical budgets across methods, no NaN/Inf,
seed uniqueness and pairing (same seed for all methods on each instance/run),
raw-result integrity (JSON <-> NPZ consistency, reported best == tracked best,
curve monotone and ending at best), and statistical input integrity (complete
paired matrices).

  python scripts/validate_experiment.py [--campaign main] [--runs 30]
Exit status 1 if any check fails. Report: results/validation/validate_<campaign>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lakaie.analysis.stats import load_campaign  # noqa: E402
from lakaie.experiment import seed_for  # noqa: E402

EXPECT = {"CEC2022": {"instances": list(range(1, 13)), "max_fes": 200000, "dim": 20},
          "FIR": {"instances": list(range(1, 9)), "max_fes": 150000, "dim": 31}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", default="main")
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--max-fes", type=int, help="override expected budget (pilot/smoke)")
    ap.add_argument("--instances", nargs="*", type=int)
    ap.add_argument("--benchmarks", nargs="*", default=list(EXPECT))
    a = ap.parse_args()
    report = {"campaign": a.campaign, "checks": [], "pass": True}

    def check(name, ok, detail=None):
        report["checks"].append({"name": name, "pass": bool(ok), "detail": None if ok else str(detail)[:2000]})
        report["pass"] &= bool(ok)
        print(("PASS " if ok else "FAIL ") + name + ("" if ok or detail is None else f" :: {str(detail)[:400]}"))

    for bench in a.benchmarks:
        exp = EXPECT[bench]
        df = load_campaign(ROOT / "results" / "raw", a.campaign, bench)
        if df.empty:
            check(f"{bench}: runs present", False, "no runs found")
            continue
        insts = a.instances or exp["instances"]
        budget = a.max_fes or exp["max_fes"]
        methods = sorted(df.method.unique())
        report[f"{bench}_methods"] = methods
        # completeness
        missing = []
        for m in methods:
            for i in insts:
                have = set(df[(df.method == m) & (df.instance == i)].run_id)
                miss = sorted(set(range(1, a.runs + 1)) - have)
                if miss:
                    missing.append((m, i, miss))
        check(f"{bench}: no missing runs ({len(methods)} methods x {len(insts)} instances x {a.runs} runs)",
              not missing, missing[:10])
        check(f"{bench}: correct instance set", set(df.instance) == set(insts), sorted(set(df.instance)))
        extra = df.groupby(["method", "instance"]).size()
        check(f"{bench}: exactly {a.runs} runs per method/instance", (extra == a.runs).all(),
              extra[extra != a.runs].to_dict())
        check(f"{bench}: all runs status ok", (df.status == "ok").all(),
              df[df.status != "ok"][["method", "instance", "run_id", "status"]].values.tolist()[:10])
        check(f"{bench}: FE budget never exceeded", (df.FE <= df.max_fes).all() and (df.max_fes == budget).all(),
              df[df.FE > df.max_fes][["method", "instance", "run_id", "FE"]].values.tolist()[:10])
        check(f"{bench}: identical declared budget across methods", df.max_fes.nunique() == 1,
              df.groupby("method").max_fes.unique().to_dict())
        check(f"{bench}: FE utilisation >= 99.5% of budget", (df.FE >= 0.995 * budget).all(),
              df.groupby("method").FE.min().to_dict())
        check(f"{bench}: dimension and population size", (df.dimension == exp["dim"]).all() and
              (df.population_size == 30).all())
        vals = df.best_fitness.astype(float)
        check(f"{bench}: no NaN/Inf results", np.isfinite(vals).all())
        # seeds
        exp_seed = df.apply(lambda r: seed_for(a.campaign, bench, int(r.instance), int(r.run_id)), axis=1)
        check(f"{bench}: seeds follow the common schedule", (exp_seed == df.seed).all())
        per = df.groupby(["instance", "run_id"]).seed.nunique()
        check(f"{bench}: paired seeds (same seed for all methods per instance/run)", (per == 1).all())
        uniq = df.groupby(["instance", "run_id"]).seed.first()
        check(f"{bench}: seed uniqueness across (instance, run)", uniq.is_unique)
        # raw integrity
        bad = []
        for _, r in df.iterrows():
            npz = Path(r.path).with_suffix(".npz")
            if not npz.exists():
                bad.append((r.path, "npz missing")); continue
            with np.load(npz) as z:
                c, cps = z["curve"], z["checkpoints"]
                fin = c[np.isfinite(c)]
                monotone = np.all(np.diff(fin) <= 0)
                ends_at_best = (r.FE < cps[-1]) or (c[-1] == r.best_fitness)
                if not (monotone and ends_at_best):
                    bad.append((r.path, "curve"))
            j = json.loads(Path(r.path).read_text())
            rb = j.get("algorithm_reported_best")
            if rb is None or not np.isclose(rb, r.best_fitness, rtol=1e-12, atol=0):
                bad.append((r.path, f"reported {rb} vs tracked {r.best_fitness}"))
        check(f"{bench}: raw result integrity (npz, curve, reported best == evaluated best)", not bad, bad[:10])
        # statistical input integrity
        piv = df.pivot_table(index=["instance", "run_id"], columns="method", values="best_fitness", aggfunc="count")
        check(f"{bench}: complete paired matrix for statistics", (piv == 1).all().all())
    snaps = list((ROOT / "results" / "raw" / a.campaign).glob("*/config_snapshot_*.yaml"))
    check("configuration snapshots saved", bool(snaps), [str(s) for s in snaps][:5])
    if snaps:
        commits = {yaml.safe_load(s.read_text()).get("git_commit") for s in snaps}
        report["git_commits"] = sorted(commits)
    out = ROOT / "results" / "validation" / f"validate_{a.campaign}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))
    print("OVERALL:", "PASS" if report["pass"] else "FAIL")
    sys.exit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
