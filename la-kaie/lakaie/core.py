"""Shared infrastructure: strict FE counter, convergence tracking, diagnostics."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.special import gamma as gamma_fn


class BudgetExceeded(RuntimeError):
    """Raised if an algorithm requests more objective evaluations than allowed.

    Algorithms are written so that this never happens; it is a hard guard, and
    any occurrence is treated as a bug (the run is marked failed)."""


class CountedObjective:
    """Wraps a vectorised objective f(X)->(m,) and counts every evaluated row.

    * ``max_fe`` is enforced *before* evaluation (no partial overshoot).
    * The best-so-far value is tracked at exact FE resolution and sampled on
      ``checkpoints`` (FE indices) for convergence curves.
    """

    def __init__(self, fn, max_fe: int | None, checkpoints: np.ndarray | None = None):
        self.fn = fn
        self.max_fe = max_fe
        self.fe = 0
        self.best = math.inf
        self.best_x = None
        self.checkpoints = np.asarray(checkpoints if checkpoints is not None else [], dtype=np.int64)
        self._cp_idx = 0
        self.curve = np.full(len(self.checkpoints), np.nan)

    def remaining(self) -> int:
        return (self.max_fe - self.fe) if self.max_fe is not None else 10**12

    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        m = X.shape[0]
        if self.max_fe is not None and self.fe + m > self.max_fe:
            raise BudgetExceeded(f"requested {m} FEs with {self.remaining()} remaining")
        f = np.asarray(self.fn(X), dtype=np.float64).reshape(m)
        if not np.all(np.isfinite(f)):
            raise FloatingPointError("objective returned NaN/Inf")
        cm = np.minimum.accumulate(np.concatenate(([self.best], f)))[1:]
        k = int(np.argmin(f))
        if f[k] < self.best:
            self.best = float(f[k])
            self.best_x = X[k].copy()
        start = self.fe
        self.fe += m
        cps = self.checkpoints
        while self._cp_idx < len(cps) and cps[self._cp_idx] <= self.fe:
            j = cps[self._cp_idx] - start - 1          # index inside this batch
            if j >= 0:
                self.curve[self._cp_idx] = cm[j]
            self._cp_idx += 1
        return f

    def eval1(self, x: np.ndarray) -> float:
        return float(self(x[None, :])[0])


def make_checkpoints(max_fe: int, n: int = 200) -> np.ndarray:
    """Linearly spaced FE checkpoints (inclusive of max_fe) for convergence curves."""
    cps = np.unique(np.round(np.linspace(max_fe / n, max_fe, n)).astype(np.int64))
    return cps


def diversity(pops, lb, ub) -> float:
    """Normalised population diversity: mean over dimensions of the standard
    deviation of the pooled population divided by the box width."""
    X = np.vstack(pops)
    return float(np.mean(X.std(axis=0) / (ub - lb)))


@dataclass
class Recorder:
    """Per-iteration diagnostics shared by all algorithms."""
    lb: np.ndarray
    ub: np.ndarray
    rows: list = field(default_factory=list)
    _last_best: float = math.inf
    _stag: int = 0

    def iteration(self, fe: int, best: float, pops, fits, **extra):
        if best < self._last_best:
            self._stag = 0
            self._last_best = best
        else:
            self._stag += 1
        allf = np.concatenate([np.ravel(f) for f in fits])
        row = {"fe": fe, "best": best, "current_mean": float(np.mean(allf)),
               "current_median": float(np.median(allf)),
               "diversity": diversity(pops, self.lb, self.ub), "stagnation": self._stag}
        row.update(extra)
        self.rows.append(row)

    def as_arrays(self) -> dict:
        if not self.rows:
            return {}
        keys = self.rows[0].keys()
        out = {}
        for k in keys:
            vals = [r.get(k) for r in self.rows]
            try:
                out[k] = np.asarray(vals, dtype=np.float64)
            except (TypeError, ValueError):
                out[k] = np.asarray([str(v) for v in vals])
        return out


# ---------------------------------------------------------------- helpers
def levy(rng: np.random.Generator, n: int, m: int, beta: float = 1.5) -> np.ndarray:
    """Mantegna Levy steps, identical formula to levy.m in the MPA/MPHBS code."""
    num = gamma_fn(1 + beta) * math.sin(math.pi * beta / 2)
    den = gamma_fn((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)
    sigma_u = (num / den) ** (1 / beta)
    u = sigma_u * rng.standard_normal((n, m))
    v = rng.standard_normal((n, m))
    return u / np.abs(v) ** (1 / beta)


def clip(X, lb, ub):
    return np.minimum(np.maximum(X, lb), ub)


def matlab_round(x: float) -> int:
    """MATLAB round: halves away from zero."""
    return int(math.floor(abs(x) + 0.5)) * (1 if x >= 0 else -1)


EPS = np.finfo(float).eps  # MATLAB eps


def calibrate_iterations(run_with_T, max_fe: int, dim: int, n_pop: int) -> tuple[int, int]:
    """Authors' FE-probe calibration (run_*_common_seeds.m, estimate_T_and_FEs):
    count the FEs actually used with T=1 and T=2 on a sphere, fit FE = a + bT,
    and return the largest T with a + bT <= max_fe, and that FE estimate."""
    def probe(T):
        obj = CountedObjective(lambda X: np.sum(X ** 2, axis=1), None)
        run_with_T(obj, T, np.random.default_rng(12345))
        return obj.fe
    fe1, fe2 = probe(1), probe(2)
    b = fe2 - fe1
    a = fe1 - b
    if b <= 0:
        raise ValueError("invalid FE probe")
    T = max(1, (max_fe - a) // b)
    while a + b * T > max_fe and T > 1:
        T -= 1
    return int(T), int(a + b * T)
