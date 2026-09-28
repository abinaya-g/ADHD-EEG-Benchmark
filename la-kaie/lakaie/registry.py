"""Method registry: maps method names to implementations and configurations."""
from __future__ import annotations

import copy
from functools import lru_cache

import numpy as np

from .algorithms import hba, mpa, mphb, mphbs
from .core import calibrate_iterations

# Iteration-driven baselines: T is calibrated from the FE budget exactly as in
# the authors' runners (FE probe with T=1 and T=2).
ITERATIVE = {"HBA": hba.run, "MPA": mpa.run, "MPHB": mphb.run, "MPHBS": mphbs.run}


def method_family(name: str) -> str:
    if name in ITERATIVE:
        return name
    if name.startswith("LA-KAIE"):
        return "LA-KAIE"
    raise KeyError(name)


@lru_cache(maxsize=None)
def _calib(name: str, cfg_items: tuple, max_fe: int, dim: int, N: int):
    fn = ITERATIVE[name]
    cfg = dict(cfg_items)
    lb, ub = -np.ones(dim), np.ones(dim)

    def run_with_T(obj, T, rng):
        fn(obj, T, lb, ub, dim, N, rng, cfg, None)
    return calibrate_iterations(run_with_T, max_fe, dim, N)


def iterations_for(name: str, cfg: dict, max_fe: int, dim: int, N: int):
    items = tuple(sorted((k, v) for k, v in (cfg or {}).items() if not isinstance(v, (dict, list))))
    return _calib(name, items, max_fe, dim, N)


def run_method(name, cfg, obj, problem, N, rng, rec, max_fe):
    lb, ub, dim = problem.lb, problem.ub, problem.dim
    if name in ITERATIVE:
        T, est = iterations_for(name, cfg, max_fe, dim, N)
        out = ITERATIVE[name](obj, T, lb, ub, dim, N, rng, cfg, rec)
        out["T"] = T
        out["estimated_fe"] = est
        return out
    from .algorithms import lakaie
    return lakaie.run(obj, lb, ub, dim, N, rng, copy.deepcopy(cfg), rec, max_fe, problem=problem)
