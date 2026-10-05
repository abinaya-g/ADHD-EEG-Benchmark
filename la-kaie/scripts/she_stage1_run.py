"""SHE Stage 1 synthetic structure-identification diagnostic (experiments/SHE_SPEC.md
§16-§17, Amendment 1 A-5/A-7). Campaign definition: configs/she.yaml (she_diag).

Resumable: existing run JSONs are skipped, never overwritten. Every task carries
its own seed, so results do not depend on scheduling.

  python scripts/she_stage1_run.py [--workers 4]
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import datetime as dt
import json
import multiprocessing as mp
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from lakaie.core import CountedObjective, Recorder, make_checkpoints  # noqa: E402
from lakaie.she.algorithm import run_she  # noqa: E402
from lakaie.she.config import VARIANTS, config_for, config_hash, load_yaml  # noqa: E402
from n1.synthetic import SyntheticProblem, synthetic_instance_id  # noqa: E402

CAMPAIGN = "she_diag"
RAW = ROOT / "results" / "raw" / CAMPAIGN


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def tasks():
    y = load_yaml()
    c = y["campaigns"][CAMPAIGN]
    master, off = y["masters"][CAMPAIGN], y["benchmark_offset"]["SYN"]
    out = []
    for v in c["variants"]:
        for fid in c["functions"]:
            for D in c["dims"]:
                inst = synthetic_instance_id(fid, D)
                for r in range(1, c["runs"] + 1):
                    out.append(dict(variant=v, fid=fid, D=D, run=r, seed=int(master + off + 1000 * inst + r),
                                    max_fe=int(c["budget_per_dim"] * D)))
    return out


def run_task(t: dict) -> dict:
    d = RAW / t["variant"] / f"S{t['fid']}_D{t['D']}"
    jp, npth = d / f"run{t['run']:02d}.json", d / f"run{t['run']:02d}.npz"
    if jp.exists():
        return {"status": "exists"}
    d.mkdir(parents=True, exist_ok=True)
    cfg = config_for("SYN")
    pr = SyntheticProblem(t["fid"], t["D"], run_seed=t["seed"])
    obj = CountedObjective(pr, t["max_fe"], make_checkpoints(t["max_fe"]))
    rec = Recorder(pr.lb, pr.ub)
    t0 = time.perf_counter()
    status, err, out = "ok", "", None
    try:
        out = run_she(pr, VARIANTS[t["variant"]], cfg, t["seed"], obj, t["max_fe"], rec)
    except Exception:  # noqa: BLE001 - recorded, never hidden
        status, err = "error", traceback.format_exc()
    s = out.summary if out else {}
    record = {"campaign": CAMPAIGN, "benchmark": "SYN", "function": f"S{t['fid']}", "fid": t["fid"],
              "label": pr.label, "dimension": t["D"], "run": t["run"], "seed": t["seed"],
              "variant": t["variant"], "max_fe": t["max_fe"], "FE": obj.fe, "final_best": float(obj.best),
              "runtime_sec": time.perf_counter() - t0, "status": status, "error": err,
              "valid": bool(status == "ok" and s.get("fe_audit_ok") and s.get("exact_E_ok")),
              "config": cfg.as_dict(), "config_hash": config_hash(cfg.as_dict()),
              "true_partition": [b.tolist() for b in pr.true_partition] if pr.true_partition is not None else None,
              "summary": {k: (v.item() if hasattr(v, "item") else v) for k, v in s.items()},
              "git_commit": t["git"], "python": platform.python_version(), "numpy": np.__version__}
    arrays = {"checkpoints": obj.checkpoints, "curve": obj.curve}
    arrays.update({f"iter_{k}": v for k, v in rec.as_arrays().items()})
    if out:
        arrays.update(out.arrays)
    tmp = npth.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, npth)
    tj = jp.with_suffix(".tmp")
    tj.write_text(json.dumps(record))
    os.replace(tj, jp)
    return {"status": status, "valid": record["valid"], "rt": record["runtime_sec"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    g = git_hash()
    T = [dict(t, git=g) for t in tasks()]
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log = ROOT / "logs" / f"run_{CAMPAIGN}_{stamp}.log"
    with open(log, "w") as fh:
        fh.write(f"{CAMPAIGN} start {stamp} git={g} tasks={len(T)}\n")
        with mp.get_context("fork").Pool(args.workers, maxtasksperchild=20) as pool:
            for i, r in enumerate(pool.imap_unordered(run_task, sorted(T, key=lambda t: -t["D"]), chunksize=1)):
                fh.write(json.dumps(r) + "\n"); fh.flush()
        fh.write(f"done {dt.datetime.now(dt.timezone.utc).isoformat()}\n")
    print("log:", log)


if __name__ == "__main__":
    main()
