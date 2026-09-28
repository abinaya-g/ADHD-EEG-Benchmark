"""Experiment execution: one seeded run -> immutable raw result files."""
from __future__ import annotations

import copy
import json
import os
import platform
import time
import traceback
from pathlib import Path

import numpy as np
import yaml

from .benchmarks.cec2022 import CEC2022Problem
from .benchmarks.fir import FIRProblem
from .core import BudgetExceeded, CountedObjective, Recorder, make_checkpoints
from .registry import run_method

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs"


def load_yaml(name):
    return yaml.safe_load((CFG / name).read_text())


def deep_update(base: dict, upd: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def lakaie_config(variant: str = "Full", overrides: dict | None = None) -> dict:
    ctrl = load_yaml("controller.yaml")
    abl = load_yaml("ablation.yaml")["variants"]
    if variant not in abl:
        raise KeyError(f"unknown LA-KAIE variant {variant}")
    cfg = deep_update(ctrl, abl[variant] or {})
    cfg = deep_update(cfg, overrides or {})
    cfg["variant"] = variant
    return cfg


def resolve_method(label: str, bench_cfg: dict, overrides: dict | None = None):
    """Returns (family_name, cfg)."""
    if label.startswith("LA-KAIE"):
        variant = label.split("-", 2)[2] if label.count("-") >= 2 else "Full"
        return "LA-KAIE", lakaie_config(variant, overrides)
    return label, copy.deepcopy(bench_cfg["methods"][label])


def make_problem(benchmark: str, instance: int, dim: int | None = None):
    if benchmark == "CEC2022":
        return CEC2022Problem(instance, dim or 20)
    if benchmark == "FIR":
        return FIRProblem(instance)
    raise KeyError(benchmark)


def seed_for(campaign: str, benchmark: str, instance: int, run: int) -> int:
    rep = load_yaml("reproducibility.yaml")
    return int(rep["master_seeds"][campaign] + rep["benchmark_offset"][benchmark] + 1000 * instance + run)


def raw_paths(out_root: Path, campaign, benchmark, label, instance, run):
    d = out_root / campaign / benchmark / label / f"inst{instance:02d}"
    return d / f"run{run:02d}.json", d / f"run{run:02d}.npz"


def _jsonable(v):
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def run_single(task: dict) -> dict:
    """Execute one (method, instance, run). Never overwrites an existing result."""
    out_root = Path(task["out_root"])
    jpath, npath = raw_paths(out_root, task["campaign"], task["benchmark"], task["label"],
                             task["instance"], task["run"])
    if jpath.exists():
        return {"status": "exists", "path": str(jpath)}
    jpath.parent.mkdir(parents=True, exist_ok=True)
    problem = make_problem(task["benchmark"], task["instance"], task.get("dim"))
    max_fe = int(task["max_fes"])
    obj = CountedObjective(problem, max_fe, make_checkpoints(max_fe, task.get("n_checkpoints", 200)))
    rec = Recorder(problem.lb, problem.ub)
    rng = np.random.default_rng(task["seed"])
    t0 = time.perf_counter()
    status, err, out = "ok", "", {}
    try:
        out = run_method(task["family"], task["cfg"], obj, problem, task["N"], rng, rec, max_fe)
    except BudgetExceeded as e:     # hard guard - must never trigger
        status, err = "budget_exceeded", repr(e)
    except Exception:               # noqa: BLE001 - recorded, never hidden
        status, err = "error", traceback.format_exc()
    runtime = time.perf_counter() - t0
    traces = rec.as_arrays()
    extra_traces = out.pop("traces", {}) if isinstance(out, dict) else {}
    summary = out.pop("summary", {}) if isinstance(out, dict) else {}
    record = {
        "method": task["label"], "family": task["family"], "benchmark": task["benchmark"],
        "instance": task["instance"], "instance_name": problem.name, "dimension": problem.dim,
        "population_size": task["N"], "seed": task["seed"], "run_id": task["run"],
        "campaign": task["campaign"], "max_fes": max_fe, "FE": obj.fe,
        "best_fitness": obj.best if np.isfinite(obj.best) else None,
        "algorithm_reported_best": _jsonable(out.get("best")) if out else None,
        "current_fitness": float(traces["current_mean"][-1]) if "current_mean" in traces else None,
        "global_best": _jsonable(obj.best_x) if obj.best_x is not None else None,
        "runtime_sec": runtime, "status": status, "error": err,
        "iterations": _jsonable(out.get("T", len(rec.rows))) if out else len(rec.rows),
        "final_diversity": float(traces["diversity"][-1]) if "diversity" in traces else None,
        "final_stagnation": float(traces["stagnation"][-1]) if "stagnation" in traces else None,
        "summary": {k: _jsonable(v) for k, v in summary.items()},
        "config": task["cfg"], "host": platform.node(), "python": platform.python_version(),
        "numpy": np.__version__,
    }
    arrays = {"checkpoints": obj.checkpoints, "curve": obj.curve}
    arrays.update({f"iter_{k}": v for k, v in traces.items()})
    arrays.update({f"lk_{k}": np.asarray(v) for k, v in extra_traces.items()})
    tmp_n = npath.with_suffix(".tmp.npz")
    np.savez_compressed(tmp_n, **arrays)
    os.replace(tmp_n, npath)
    tmp_j = jpath.with_suffix(".tmp")
    tmp_j.write_text(json.dumps(record, default=_jsonable))
    os.replace(tmp_j, jpath)          # json written last = run complete marker
    return {"status": status, "path": str(jpath), "runtime": runtime, "best": record["best_fitness"]}
