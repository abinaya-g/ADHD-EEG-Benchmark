"""Run an N1 synthetic campaign defined in configs/n1.yaml.

Resumable (existing run JSONs are skipped, never overwritten); every task
carries its own seed, so results do not depend on scheduling.

  python scripts/n1_run.py --campaign n1_smoke [--workers 4]
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

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
sys.path.insert(0, str(ROOT / "src"))
from n1.config import N1Config, config_hash  # noqa: E402
from n1.runner import run_task, seed_for  # noqa: E402
from n1.synthetic import synthetic_instance_id  # noqa: E402

GATED = {"n1_synthetic"}          # require explicit approval (Stage 4+)


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def build_tasks(campaign: str):
    spec = yaml.safe_load((ROOT / "configs" / "n1.yaml").read_text())["campaigns"][campaign]
    seeds = json.loads((ROOT / "configs" / "seeds.json").read_text())
    commit = git_hash()
    out = []
    for g in spec["groups"]:
        for arm in g["arms"]:
            for fid in g["functions"]:
                for D in spec["dims"]:
                    inst = synthetic_instance_id(fid, D)
                    for r in range(1, spec["runs"] + 1):
                        out.append({"campaign": campaign, "arm": arm, "fid": fid, "D": D, "run": r,
                                    "seed": seed_for(seeds["masters"], campaign,
                                                     seeds["benchmark_offset"]["SYN"], inst, r),
                                    "max_fe": spec["budget_per_dim"] * D, "git_commit": commit,
                                    "out_root": str(ROOT / "results" / "raw")})
    return out, spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--approved", action="store_true", help="required for gated campaigns (Stage 4+)")
    args = ap.parse_args()
    if args.campaign in GATED and not args.approved:
        sys.exit(f"{args.campaign} is gated: Gates 1-9 and explicit user approval are required (spec §16).")
    tasks, spec = build_tasks(args.campaign)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    outdir = ROOT / "results" / "raw" / args.campaign
    outdir.mkdir(parents=True, exist_ok=True)
    snap = {"campaign": args.campaign, "spec": spec, "n1_config": N1Config().as_dict(),
            "n1_config_hash": config_hash(N1Config().as_dict()), "git_commit": git_hash(),
            "utc": stamp, "n_tasks": len(tasks)}
    (outdir / f"config_snapshot_{stamp}.yaml").write_text(yaml.safe_dump(snap, sort_keys=False))
    with (outdir / "seeds_used.csv").open("w") as f:
        f.write("arm,function,D,run,seed\n")
        for t in tasks:
            f.write(f"{t['arm']},S{t['fid']},{t['D']},{t['run']},{t['seed']}\n")
    log = ROOT / "logs" / f"run_{args.campaign}_{stamp}.log"
    t0 = time.time()
    n_ok = n_fail = n_skip = n_invalid = 0
    tasks.sort(key=lambda t: (-t["D"], t["fid"] != 8))
    with mp.get_context("fork").Pool(args.workers, maxtasksperchild=20) as pool, log.open("a") as lf:
        for i, res in enumerate(pool.imap_unordered(run_task, tasks, chunksize=1), 1):
            if res["status"] == "exists":
                n_skip += 1
            elif res["status"] == "ok":
                n_ok += 1
                n_invalid += int(not res.get("valid", False))
            else:
                n_fail += 1
            lf.write(dt.datetime.now(dt.timezone.utc).isoformat() + " " + json.dumps(res) + "\n")
            if i % 20 == 0 or i == len(tasks):
                print(f"[{time.time() - t0:6.0f}s] {i}/{len(tasks)} ok={n_ok} fail={n_fail} "
                      f"invalid={n_invalid} skipped={n_skip}", flush=True)
    print(f"finished: ok={n_ok} fail={n_fail} invalid={n_invalid} skipped={n_skip} log={log}")
    sys.exit(1 if (n_fail or n_invalid) else 0)


if __name__ == "__main__":
    main()
