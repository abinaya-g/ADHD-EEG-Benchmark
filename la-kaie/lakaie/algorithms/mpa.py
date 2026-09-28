"""Marine Predators Algorithm - port of MPA.m (Faramarzi et al., 2020) as
shipped in the MPHBS package. Vectorised; update semantics unchanged."""
from __future__ import annotations

import numpy as np

from ..core import Recorder, clip, levy

FADS, P = 0.2, 0.5


def mpa_move(rng, X, elite, t, T, CF, phase2_lower_inclusive: bool):
    """Stage-dependent MPA movement (Eqs. (17)-(24)).

    ``phase2_lower_inclusive``: MPHB/MPHBS code uses T/3 <= t < 2T/3,
    the original MPA.m uses T/3 < t < 2T/3 (t == T/3 falls into phase 3)."""
    N, D = X.shape
    RL = 0.05 * levy(rng, N, D, 1.5)
    RB = rng.standard_normal((N, D))
    R = rng.random((N, D))
    if t < T / 3:
        step = RB * (elite - RB * X)
        return X + P * R * step
    in_p2 = (t >= T / 3 if phase2_lower_inclusive else t > T / 3) and t < 2 * T / 3
    if in_p2:
        Xn = np.empty_like(X)
        second = np.arange(N) + 1 > N / 2             # MATLAB i > N/2 (1-based)
        stepB = RB * (RB * elite - X)
        stepL = RL * (elite - RL * X)
        Xn[second] = elite[second] + P * CF * stepB[second]
        Xn[~second] = X[~second] + P * R[~second] * stepL[~second]
        return Xn
    step = RL * (RL * elite - X)
    return elite + P * CF * step


def fads_candidates(rng, X, lb, ub, CF):
    """Eqs. (28)-(29); the branch is drawn once for the whole population."""
    N, D = X.shape
    if rng.random() < FADS:
        U = rng.random((N, D)) < FADS
        return X + CF * ((lb + rng.random((N, D)) * (ub - lb)) * U)
    r = rng.random()
    return X + (FADS * (1 - r) + r) * (X[rng.permutation(N)] - X[rng.permutation(N)])


def run(obj, T, lb, ub, dim, N, rng, cfg=None, rec: Recorder | None = None):
    prey = rng.random((N, dim)) * (ub - lb) + lb
    top_fit, top_pos = np.inf, np.zeros(dim)
    fit_old = prey_old = None
    for it in range(T):
        prey = clip(prey, lb, ub)
        fit = obj(prey)
        k = int(np.argmin(fit))
        if fit[k] < top_fit:
            top_fit, top_pos = float(fit[k]), prey[k].copy()
        if it == 0:
            fit_old, prey_old = fit.copy(), prey.copy()
        keep = fit_old < fit
        prey = np.where(keep[:, None], prey_old, prey)
        fit = np.where(keep, fit_old, fit)
        fit_old, prey_old = fit.copy(), prey.copy()

        elite = np.tile(top_pos, (N, 1))
        CF = (1 - it / T) ** (2 * it / T)
        prey = mpa_move(rng, prey, elite, it, T, CF, phase2_lower_inclusive=False)

        prey = clip(prey, lb, ub)
        fit = obj(prey)
        k = int(np.argmin(fit))
        if fit[k] < top_fit:
            top_fit, top_pos = float(fit[k]), prey[k].copy()
        keep = fit_old < fit
        succ = float(np.mean(~keep))
        prey = np.where(keep[:, None], prey_old, prey)
        fit = np.where(keep, fit_old, fit)
        fit_old, prey_old = fit.copy(), prey.copy()
        if rec is not None:
            rec.iteration(obj.fe, top_fit, [prey], [fit], mpa_success=succ)
        prey = fads_candidates(rng, prey, lb, ub, CF)
    return {"best": top_fit, "best_x": top_pos}
