"""One seeded N1 run -> immutable JSON (metadata) + NPZ (logs). Never overwrites."""
from __future__ import annotations

import json
import os
import platform
import time
import traceback
from pathlib import Path

import numpy as np

from lakaie.core import CountedObjective, Recorder, make_checkpoints

from .algorithm import run_n1
from .config import ARMS, N1Config, config_hash
from .diagnostics import UNAVAILABLE_COMPONENTS, D3Recorder
from .synthetic import SyntheticProblem, synthetic_instance_id

ROOT = Path(__file__).resolve().parents[2]


def seed_for(masters: dict, campaign: str, offset: int, instance: int, run: int) -> int:
    return int(masters[campaign] + offset + 1000 * instance + run)


def _j(v):
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def paths(out_root: Path, campaign: str, arm: str, fid: int, D: int, run: int):
    d = out_root / campaign / arm / f"S{fid}_D{D}"
    return d / f"run{run:02d}.json", d / f"run{run:02d}.npz"


def reference_structure(problem) -> dict:
    """Known synthetic structure, stored for OFFLINE comparison only (never used by
    the optimizer or selector)."""
    out = {"ref_o": problem.o, "ref_c": problem.c}
    if problem.true_partition is not None:
        lab = np.zeros(problem.dim, np.int16)
        for i, b in enumerate(problem.true_partition):
            lab[b] = i
        out["ref_block_label"] = lab
    for name in ("Q", "M", "perm"):
        if hasattr(problem, name):
            out[f"ref_{name}"] = getattr(problem, name)
    return out


def run_task(task: dict) -> dict:
    out_root = Path(task["out_root"])
    jpath, npath = paths(out_root, task["campaign"], task["arm"], task["fid"], task["D"], task["run"])
    if jpath.exists():
        return {"status": "exists", "path": str(jpath)}
    jpath.parent.mkdir(parents=True, exist_ok=True)
    cfg = N1Config(**task.get("cfg_overrides", {}))
    arm = ARMS[task["arm"]]
    problem = SyntheticProblem(task["fid"], task["D"], run_seed=task["seed"])
    max_fe = int(task["max_fe"])
    obj = CountedObjective(problem, max_fe, make_checkpoints(max_fe, 200))
    rec = Recorder(problem.lb, problem.ub)
    chash = config_hash({"cfg": cfg.as_dict(), "arm": arm.__dict__, "fid": task["fid"],
                         "D": task["D"], "max_fe": max_fe})
    diag = D3Recorder(cfg.N, task["D"]) if task.get("diag") else None
    t0 = time.perf_counter()
    status, err, out = "ok", "", None
    try:
        out = run_n1(problem, arm, cfg, task["seed"], obj, max_fe, rec, diag=diag)
    except Exception:          # noqa: BLE001 - recorded, never hidden
        status, err = "error", traceback.format_exc()
    runtime = time.perf_counter() - t0
    summary = out.summary if out else {}
    valid = bool(status == "ok" and summary.get("fe_audit_ok") and summary.get("fe_unused_ok"))
    record = {
        "campaign": task["campaign"], "benchmark": "SYN", "function": f"S{task['fid']}",
        "fid": task["fid"], "label": problem.label, "primary": problem.primary,
        "dimension": task["D"], "instance": synthetic_instance_id(task["fid"], task["D"]),
        "instance_seed": problem.seed_inst, "run": task["run"], "seed": task["seed"],
        "arm": task["arm"], "population_size": cfg.N, "max_fe": max_fe, "FE": obj.fe,
        "final_best": float(obj.best), "runtime_sec": runtime, "status": status, "error": err,
        "valid": valid, "git_commit": task.get("git_commit", "unknown"), "config_hash": chash,
        "config": cfg.as_dict(), "true_partition": [b.tolist() for b in problem.true_partition]
        if problem.true_partition is not None else None,
        "summary": {k: _j(v) for k, v in summary.items()},
        "host": platform.node(), "python": platform.python_version(), "numpy": np.__version__,
    }
    arrays = {"checkpoints": obj.checkpoints, "curve": obj.curve}
    arrays.update({f"iter_{k}": v for k, v in rec.as_arrays().items()})
    if out:
        arrays.update(out.arrays)
    if diag is not None:
        arrays.update(diag.arrays())
        arrays.update(reference_structure(problem))
        record["d3"] = {"agent_id": "population row index (stable; backbone updates rows in place)",
                        "unavailable_components": list(UNAVAILABLE_COMPONENTS),
                        "seed_source_campaign": task.get("seed_source_campaign")}
    tmp = npath.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, npath)
    tj = jpath.with_suffix(".tmp")
    tj.write_text(json.dumps(record, default=_j))
    os.replace(tj, jpath)
    return {"status": status, "valid": valid, "path": str(jpath), "runtime": runtime}
