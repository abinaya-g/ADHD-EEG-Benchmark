"""Phase 1 of MPHB / MPHBS (paper Algorithm 1 lines 10-18; MPHBS_main.m PHASE 1A-1D)."""
from __future__ import annotations

import numpy as np

from ..core import EPS, clip
from .hba import hba_move
from .mpa import FADS, fads_candidates, mpa_move

BETA_HBA, C_HBA = 6.0, 2.0


def intensity_original(rng, X, xprey):
    """Intensity_HBA_Original() of MPHBS_main.m (eps in norm and denominator)."""
    Xn = np.roll(X, -1, axis=0)
    di = np.sum((X - xprey + EPS) ** 2, axis=1)
    S = np.sum((X - Xn + EPS) ** 2, axis=1)
    r2 = rng.random(X.shape[0])
    return r2 * S / (4 * np.pi * di + EPS)


class HybridState:
    """Two-population state (HBA + MPA) with MPA marine memory."""

    def __init__(self, obj, lb, ub, dim, N, rng):
        self.lb, self.ub, self.dim, self.N = lb, ub, dim, N
        # MATLAB order: X_MPA initialised before X_HBA
        self.X_M = clip(rng.random((N, dim)) * (ub - lb) + lb, lb, ub)
        self.X_H = clip(rng.random((N, dim)) * (ub - lb) + lb, lb, ub)
        self.F_M = obj(self.X_M)
        self.F_H = obj(self.X_H)
        k = int(np.argmin(self.F_M)); self.bF_M, self.bP_M = float(self.F_M[k]), self.X_M[k].copy()
        k = int(np.argmin(self.F_H)); self.bF_H, self.bP_H = float(self.F_H[k]), self.X_H[k].copy()
        self.fit_old, self.X_M_old = self.F_M.copy(), self.X_M.copy()
        self.global_from_bests()
        self.last_hba_success = 0.0
        self.last_mpa_success = 0.0
        self.last_step_hba = 0.0
        self.last_step_mpa = 0.0

    def global_from_bests(self):
        """Eq. (33): MPA wins ties."""
        if self.bF_M <= self.bF_H:
            self.Pg, self.Pbest = self.bF_M, self.bP_M.copy()
        else:
            self.Pg, self.Pbest = self.bF_H, self.bP_H.copy()

    def sync_global_keep(self):
        """MPHBS post-phase synchronisation: only replace Pg if improved."""
        if self.bF_M <= self.bF_H:
            if self.bF_M < self.Pg:
                self.Pg, self.Pbest = self.bF_M, self.bP_M.copy()
        else:
            if self.bF_H < self.Pg:
                self.Pg, self.Pbest = self.bF_H, self.bP_H.copy()

    def memory_saving(self):
        keep = self.fit_old < self.F_M
        self.X_M = np.where(keep[:, None], self.X_M_old, self.X_M)
        self.F_M = np.where(keep, self.fit_old, self.F_M)
        self.fit_old, self.X_M_old = self.F_M.copy(), self.X_M.copy()
        return keep

    def eval_mpa(self, obj):
        self.X_M = clip(self.X_M, self.lb, self.ub)
        self.F_M = obj(self.X_M)
        k = int(np.argmin(self.F_M))
        if self.F_M[k] < self.bF_M:
            self.bF_M, self.bP_M = float(self.F_M[k]), self.X_M[k].copy()

    def phase1(self, obj, rng, prog_t: float, prog_T: float):
        """HBA + MPA movement and evaluation (PHASE 1A-1D). ``prog_t`` is the
        0-based iteration counter t and ``prog_T`` the horizon T; LA-KAIE
        passes FE-based equivalents (t/T = FE_used/MaxFE)."""
        t, T = prog_t, max(1.0, prog_T)
        I = intensity_original(rng, self.X_H, self.bP_H)
        alpha = C_HBA * np.exp(-(t + 1) / T)
        elite = np.tile(self.bP_M, (self.N, 1))
        CF = (1 - t / T) ** (2 * t / T)
        Xh_new = clip(hba_move(rng, self.X_H, self.bP_H, I, alpha, BETA_HBA), self.lb, self.ub)
        X_M_prev = self.X_M.copy()
        self.X_M = mpa_move(rng, self.X_M, elite, t, T, CF, phase2_lower_inclusive=True)
        fnew = obj(Xh_new)
        imp = fnew < self.F_H
        self.last_step_hba = float(np.mean(np.linalg.norm(Xh_new[imp] - self.X_H[imp], axis=1))) if imp.any() else 0.0
        self.X_H[imp], self.F_H[imp] = Xh_new[imp], fnew[imp]
        self.last_hba_success = float(imp.mean())
        self.X_H = clip(self.X_H, self.lb, self.ub)
        k = int(np.argmin(self.F_H)); self.bF_H, self.bP_H = float(self.F_H[k]), self.X_H[k].copy()
        self.eval_mpa(obj)
        keep = self.memory_saving()
        self.last_mpa_success = float(np.mean(~keep))
        moved = ~keep
        self.last_step_mpa = float(np.mean(np.linalg.norm(self.X_M[moved] - X_M_prev[moved], axis=1))) if moved.any() else 0.0
        return CF

    def fads_greedy(self, obj, rng, CF):
        """apply_mpa_fads_stage() of MPHBS_main.m: evaluate + per-agent greedy."""
        cand = clip(fads_candidates(rng, self.X_M, self.lb, self.ub, CF), self.lb, self.ub)
        fc = obj(cand)
        imp = fc < self.F_M
        self.X_M[imp], self.F_M[imp] = cand[imp], fc[imp]
        k = int(np.argmin(self.F_M)); self.bF_M, self.bP_M = float(self.F_M[k]), self.X_M[k].copy()
        self.fit_old, self.X_M_old = self.F_M.copy(), self.X_M.copy()
        return float(imp.mean())


__all__ = ["HybridState", "FADS", "intensity_original"]
