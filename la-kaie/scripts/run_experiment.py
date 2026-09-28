"""Run a campaign: every (method, instance, run) is an independent seeded task.

Resumable: completed runs (JSON present) are skipped, never overwritten.
Parallel execution uses a process pool; results do not depend on scheduling
because each task carries its own seed.

Example
  python scripts/run_experiment.py --benchmark CEC2022 --campaign pilot \
      --methods HBA MPA MPHB MPHBS LA-KAIE-Full --runs 5
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")   # one BLAS thread per process (parallel runs, reproducible timing)

import argparse
import datetime as dt
import json
import multiprocessing as mp
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lakaie.experiment import load_yaml, resolve_method, run_single, seed_for  # noqa: E402

BENCH_CFG = {"CEC2022": "cec2022.yaml", "FIR": "fir.yaml"}


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def build_tasks(args):
    bcfg = load_yaml(BENCH_CFG[args.benchmark])
    rep = load_yaml("reproducibility.yaml")
    overrides_by_label = {}
    if args.overrides:
        overrides_by_label = yaml.safe_load(Path(args.overrides).read_text())
    instances = args.instances or bcfg["instances"]
    runs = args.runs or bcfg["runs"]
    max_fes = args.max_fes or bcfg["max_fes"]
    tasks = []
    labels = args.methods + list(overrides_by_label.keys())
    for label in labels:
        base_label = label.split("@")[0]
        fam, cfg = resolve_method(base_label, bcfg, overrides_by_label.get(label))
        for inst in instances:
            for r in range(1, runs + 1):
                tasks.append({
                    "label": label, "family": fam, "cfg": cfg, "benchmark": args.benchmark,
                    "instance": inst, "run": r, "campaign": args.campaign,
                    "seed": seed_for(args.campaign, args.benchmark, inst, r),
                    "max_fes": max_fes, "N": bcfg["population_size"],
                    "dim": args.dim or bcfg["dimension"],
                    "n_checkpoints": rep["convergence_checkpoints"],
                    "out_root": str(ROOT / "results" / "raw"),
                })
    return tasks, {"benchmark_config": bcfg, "reproducibility": rep, "instances": instances,
                   "runs": runs, "max_fes": max_fes, "labels": labels,
                   "overrides": overrides_by_label}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True, choices=list(BENCH_CFG))
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--methods", nargs="*", default=[])
    ap.add_argument("--instances", nargs="*", type=int)
    ap.add_argument("--runs", type=int)
    ap.add_argument("--max-fes", type=int)
    ap.add_argument("--dim", type=int)
    ap.add_argument("--overrides", help="YAML {label@tag: {param overrides}} for sensitivity runs")
    ap.add_argument("--workers", type=int)
    args = ap.parse_args()

    tasks, snapshot = build_tasks(args)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snapdir = ROOT / "results" / "raw" / args.campaign / args.benchmark
    snapdir.mkdir(parents=True, exist_ok=True)
    snapshot.update({"git_commit": git_hash(), "utc": stamp, "argv": sys.argv,
                     "n_tasks": len(tasks),
                     "method_configs": {t["label"]: t["cfg"] for t in tasks}})
    (snapdir / f"config_snapshot_{stamp}.yaml").write_text(yaml.safe_dump(snapshot, sort_keys=False))

    workers = args.workers or snapshot["reproducibility"]["n_workers"]
    log = ROOT / "logs" / f"run_{args.campaign}_{args.benchmark}_{stamp}.log"
    t0 = time.time()
    done = fail = skipped = 0
    # longest families first for better load balance
    tasks.sort(key=lambda t: (t["family"] not in ("MPHBS", "LA-KAIE"), t["instance"], t["run"]))
    with mp.get_context("fork").Pool(workers, maxtasksperchild=50) as pool, log.open("a") as lf:
        for res in pool.imap_unordered(run_single, tasks, chunksize=1):
            if res["status"] == "exists":
                skipped += 1
            elif res["status"] == "ok":
                done += 1
            else:
                fail += 1
            lf.write(dt.datetime.now(dt.timezone.utc).isoformat() + " " + json.dumps(res) + "\n")
            lf.flush()
            n = done + fail + skipped
            if n % 25 == 0 or n == len(tasks):
                print(f"[{time.time()-t0:7.0f}s] {n}/{len(tasks)} ok={done} fail={fail} skipped={skipped}", flush=True)
    print(f"finished: ok={done} fail={fail} skipped={skipped} elapsed={time.time()-t0:.0f}s log={log}")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
