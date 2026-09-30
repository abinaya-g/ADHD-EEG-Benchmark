"""D3 logging pass (measurement only): re-run n1_smoke A0 on S1-S3 x D in {10,20} x 5
runs with the SAME seeds and configuration, plus the passive D3Recorder, into
results/raw/n1_d3_diagnostic/. Then verify, run by run, that every production
field is bit-identical to the stored n1_smoke run (tolerance: none).

  python scripts/n1_d3_diagnostic.py [--workers 4]
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import multiprocessing as mp
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from n1.runner import run_task, seed_for  # noqa: E402
from n1.synthetic import synthetic_instance_id  # noqa: E402

CAMPAIGN, SEED_SOURCE = "n1_d3_diagnostic", "n1_smoke"
RAW = ROOT / "results" / "raw"


def git_hash():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def tasks():
    seeds = json.loads((ROOT / "configs" / "seeds.json").read_text())
    commit = git_hash()
    out = []
    for fid in (1, 2, 3):
        for D in (10, 20):
            for r in range(1, 6):
                s = seed_for(seeds["masters"], SEED_SOURCE, seeds["benchmark_offset"]["SYN"],
                             synthetic_instance_id(fid, D), r)
                out.append({"campaign": CAMPAIGN, "arm": "A0", "fid": fid, "D": D, "run": r, "seed": s,
                            "max_fe": 10000 * D, "git_commit": commit, "out_root": str(RAW),
                            "diag": True, "seed_source_campaign": SEED_SOURCE})
    return out


def verify(t):
    rel = Path("A0") / f"S{t['fid']}_D{t['D']}" / f"run{t['run']:02d}"
    old_j = json.loads((RAW / SEED_SOURCE / rel).with_suffix(".json").read_text())
    new_j = json.loads((RAW / CAMPAIGN / rel).with_suffix(".json").read_text())
    zo = np.load((RAW / SEED_SOURCE / rel).with_suffix(".npz"))
    zn = np.load((RAW / CAMPAIGN / rel).with_suffix(".npz"))
    bad = []
    for k in ("seed", "FE", "final_best", "config_hash", "status", "valid"):
        if old_j[k] != new_j[k]:
            bad.append(k)
    for k in ("fe_by_tag", "n_iter", "n_epochs", "fe_unused", "n_window_rows_H", "n_window_rows_M"):
        if old_j["summary"][k] != new_j["summary"][k]:
            bad.append("summary." + k)
    missing = [k for k in zo.files if k not in zn.files]
    for k in zo.files:
        if k in zn.files and not np.array_equal(zo[k], zn[k], equal_nan=zo[k].dtype.kind == "f"):
            bad.append(k)
    return {"run": str(rel), "seed": new_j["seed"], "FE": new_j["FE"], "n_iter": new_j["summary"]["n_iter"],
            "fields_compared": len(zo.files) + 12, "mismatches": bad, "missing_in_new": missing,
            "config_hash": new_j["config_hash"], "git_commit": new_j["git_commit"],
            "fe_audit_ok": new_j["summary"]["fe_audit_ok"], "npz_bytes": (RAW / CAMPAIGN / rel).with_suffix(".npz").stat().st_size}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    T = tasks()
    with mp.get_context("fork").Pool(args.workers, maxtasksperchild=10) as pool:
        res = list(pool.imap_unordered(run_task, sorted(T, key=lambda t: -t["D"]), chunksize=1))
    ver = [verify(t) for t in T]
    summary = {"runs": len(ver), "status": [r["status"] for r in res],
               "runs_with_mismatch": sum(bool(v["mismatches"] or v["missing_in_new"]) for v in ver),
               "verification": ver}
    (RAW / CAMPAIGN / "trajectory_equivalence.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "verification"}))


if __name__ == "__main__":
    main()
