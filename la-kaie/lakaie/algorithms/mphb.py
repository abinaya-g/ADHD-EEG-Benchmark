"""MPHB: HBA-MPA backbone without mediator - port of MPHB_baseline.m.

Per iteration: re-evaluate MPA (evaluates last iteration's FADs perturbation)
+ memory saving, Phase-1 movement/evaluation, Eq. (33), FADs perturbation
without in-place evaluation. FE = 2N + 3N*T."""
from __future__ import annotations

import numpy as np

from ..core import Recorder
from .hybrid_common import HybridState
from .mpa import fads_candidates


def run(obj, T, lb, ub, dim, N, rng, cfg=None, rec: Recorder | None = None):
    s = HybridState(obj, lb, ub, dim, N, rng)
    for t in range(T):
        s.eval_mpa(obj)
        s.memory_saving()
        CF = s.phase1(obj, rng, t, T)
        s.global_from_bests()
        if rec is not None:
            rec.iteration(obj.fe, s.Pg, [s.X_H, s.X_M], [s.F_H, s.F_M],
                          hba_success=s.last_hba_success, mpa_success=s.last_mpa_success)
        s.X_M = fads_candidates(rng, s.X_M, lb, ub, CF)
    return {"best": s.Pg, "best_x": s.Pbest}
