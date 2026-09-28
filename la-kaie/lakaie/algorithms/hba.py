"""Honey Badger Algorithm - port of HBA.m used by the MPHBS authors
(Hashim et al., 2021). Per-agent loop vectorised; update semantics unchanged."""
from __future__ import annotations

import numpy as np

from ..core import EPS, Recorder, clip


def intensity_standalone(rng, X, xprey):
    """Intensity() of HBA.m (no eps in the denominator, r2 per agent)."""
    Xn = np.roll(X, -1, axis=0)                     # X(i+1), wraps to X(1)
    di = np.sum((X - xprey + EPS) ** 2, axis=1)
    S = np.sum((X - Xn + EPS) ** 2, axis=1)
    r2 = rng.random(X.shape[0])
    return r2 * S / (4 * np.pi * di)


def hba_move(rng, X, xprey, I, alpha, beta):
    """Eqs. (8)-(9): one flag F and mode r per agent, r3..r7 per component."""
    N, D = X.shape
    r = rng.random(N)
    F = np.where(rng.random(N) < 0.5, 1.0, -1.0)[:, None]   # vec_flag(floor(2*rand+1))
    di = xprey[None, :] - X
    r3, r4, r5 = rng.random((N, D)), rng.random((N, D)), rng.random((N, D))
    r7 = rng.random((N, D))
    dig = xprey + F * beta * I[:, None] * xprey + F * r3 * alpha * di * np.abs(
        np.cos(2 * np.pi * r4) * (1 - np.cos(2 * np.pi * r5)))
    honey = xprey + F * r7 * alpha * di
    return np.where((r < 0.5)[:, None], dig, honey)


def run(obj, T, lb, ub, dim, N, rng, cfg=None, rec: Recorder | None = None):
    beta, C = 6.0, 2.0
    X = rng.random((N, dim)) * (ub - lb) + lb
    fit = obj(X)
    g = int(np.argmin(fit))
    gbest, xprey = float(fit[g]), X[g].copy()
    for t in range(1, T + 1):
        alpha = C * np.exp(-t / T)
        I = intensity_standalone(rng, X, xprey)
        Xnew = clip(hba_move(rng, X, xprey, I, alpha, beta), lb, ub)
        fnew = obj(Xnew)
        imp = fnew < fit
        X[imp], fit[imp] = Xnew[imp], fnew[imp]
        X = clip(X, lb, ub)
        k = int(np.argmin(fit))
        if fit[k] < gbest:
            gbest, xprey = float(fit[k]), X[k].copy()
        if rec is not None:
            rec.iteration(obj.fe, gbest, [X], [fit], hba_success=float(imp.mean()))
    return {"best": gbest, "best_x": xprey}
