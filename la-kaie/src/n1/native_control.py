"""Native-control model (spec §4.7).

logit p_nat(y=1 | l, pop) = th0 + th1 * l + th2 * 1[pop = M],
fitted by L2-regularised IRLS (ridge lambda on th1, th2; none on th0) on the
native-attempt buffer NA. Refitted every iteration from native attempts only.
"""
from __future__ import annotations

import math

import numpy as np

from .config import N1Config


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def logit(p):
    return math.log(p / (1.0 - p))


class NativeControl:
    def __init__(self, cfg: N1Config):
        self.cfg = cfg
        self.theta = np.zeros(3)
        self.separation = False
        self.fitted = False
        self.l_lo = -np.inf
        self.l_hi = np.inf
        self.n_iter = 0

    def fit(self, l: np.ndarray, pop: np.ndarray, y: np.ndarray):
        cfg = self.cfg
        n = len(y)
        if n == 0:
            self.fitted = False
            return
        X = np.column_stack([np.ones(n), l, (pop == 1).astype(float)])
        ys = float(np.sum(y))
        self.l_lo, self.l_hi = (float(np.quantile(l, cfg.support_q[0])),
                                float(np.quantile(l, cfg.support_q[1])))
        if ys == 0 or ys == n:
            # spec §9: intercept-only MLE of the smoothed proportion; flagged
            p = (ys + 0.5) / (n + 1.0)
            self.theta = np.array([logit(p), 0.0, 0.0])
            self.separation, self.fitted, self.n_iter = True, True, 0
            return
        lam = np.array([0.0, cfg.lambda_nat, cfg.lambda_nat])
        th = np.zeros(3)          # cold start each refit: fit depends on the NA buffer only
        yv = y.astype(float)
        it = 0
        for it in range(1, cfg.irls_max_iter + 1):
            p = sigmoid(X @ th)
            w = p * (1.0 - p)
            g = X.T @ (yv - p) - lam * th
            H = (X * w[:, None]).T @ X + np.diag(lam)
            step = np.linalg.solve(H + 1e-12 * np.eye(3), g)
            th = th + step
            if np.max(np.abs(step)) < cfg.irls_tol:
                break
        self.theta, self.separation, self.fitted, self.n_iter = th, False, True, it

    def predict(self, l: float, pop_is_M: bool):
        """Returns (p0, l_clipped, out_of_support)."""
        oos = bool(l < self.l_lo or l > self.l_hi)
        lc = float(min(max(l, self.l_lo), self.l_hi))
        z = self.theta[0] + self.theta[1] * lc + self.theta[2] * float(pop_is_M)
        p = float(sigmoid(z))
        c = self.cfg.p_clip
        return min(max(p, c), 1.0 - c), lc, oos
