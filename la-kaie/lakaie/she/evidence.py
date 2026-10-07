"""Outcome models, prequential evidence, posterior weights and L1 linkage
(experiments/SHE_SPEC.md §5-§9, §11). No objective evaluations, no RNG."""
from __future__ import annotations

import numpy as np

from .config import HYPS, SHEConfig


def sigmoid(u):
    return 0.5 * (1.0 + np.tanh(0.5 * u))


def features(g: int, z1: np.ndarray, z3: np.ndarray, pairs: np.ndarray, nested: bool = False) -> np.ndarray:
    """Design matrix rows phi_g for records (z1, z3: (n, D)); spec §5.
    nested=True (Amendment 2, spec §25 B-1): h1-h3 additionally contain h0's step-size
    term m = log(1+||z1||) right after the intercept, so h0 is nested in every model.
    nested=False is the Stage 1 definition, unchanged."""
    n = z1.shape[0]
    one = np.ones((n, 1))
    if nested and g in (1, 2, 3):
        m = np.log1p(np.linalg.norm(z1, axis=1))[:, None]
        rest = features(g, z1, z3, pairs, nested=False)[:, 1:]
        return np.hstack([one, m, rest])
    if g == 0:
        return np.hstack([one, np.log1p(np.linalg.norm(z1, axis=1))[:, None]])
    if g == 1:
        return np.hstack([one, z1, z1 * z1])
    if g == 2:
        base = np.hstack([one, z1, z1 * z1])
        if pairs.shape[0] == 0:
            return base
        return np.hstack([base, z1[:, pairs[:, 0]] * z1[:, pairs[:, 1]]])
    if g == 3:
        return np.hstack([one, z3, z3 * z3])
    raise KeyError(g)


class LogisticModel:
    """L2-regularised logistic regression (intercept unpenalised), updated by one
    warm-started IRLS (Newton) step per call on a weighted window (spec §6)."""

    def __init__(self, p: int, kappa2: float, y0: float, eps_p: float):
        self.kappa2, self.eps_p = kappa2, eps_p
        self.theta = np.zeros(p)
        self.theta[0] = np.log(y0 / (1.0 - y0))

    def prob(self, Phi: np.ndarray) -> np.ndarray:
        return np.clip(sigmoid(Phi @ self.theta), self.eps_p, 1.0 - self.eps_p)

    def reset_tail(self, p: int, n_base: int):
        """Warm-start mapping after an h2 linkage refit: the n_base base coefficients
        are kept, all interaction coefficients restart at 0 (pairs may have changed)."""
        new = np.zeros(p)
        new[:n_base] = self.theta[:n_base]
        self.theta = new

    def irls_step(self, Phi: np.ndarray, y: np.ndarray, sw: np.ndarray):
        p = sigmoid(Phi @ self.theta)
        pen = np.full(self.theta.size, self.kappa2)
        pen[0] = 0.0
        g = Phi.T @ (sw * (y - p)) - pen * self.theta
        Hm = (Phi * (sw * p * (1.0 - p))[:, None]).T @ Phi + np.diag(pen) + 1e-10 * np.eye(self.theta.size)
        self.theta = self.theta + np.linalg.solve(Hm, g)


class Evidence:
    """Shared-data evidence engine for the hypothesis set H (spec §6-§9).

    ``add`` scores a new record with the CURRENT (pre-update) parameters (prequential),
    updates the discounted losses L_g, and stores the record in the fit window.
    ``refit`` performs the end-of-iteration IRLS step for every model."""

    def __init__(self, cfg: SHEConfig, D: int, hyps=HYPS, nested: bool = False):
        self.cfg, self.D, self.hyps, self.nested = cfg, D, tuple(hyps), bool(nested)
        self.pairs = np.zeros((0, 2), dtype=int)
        self.models = {g: LogisticModel(self._p(g), cfg.kappa2, cfg.y0, cfg.eps_p) for g in self.hyps}
        self.L = np.zeros(4)
        W = cfg.W_x
        self.z1 = np.zeros((W, D)); self.z3 = np.zeros((W, D))
        self.y = np.zeros(W); self.w = np.zeros(W); self.idx = np.zeros(W, dtype=np.int64)
        self.n = 0                                   # records seen

    def _p(self, g):
        return {0: 2, 1: self._nb(), 2: self._nb() + self.pairs.shape[0], 3: self._nb()}[g]

    def _nb(self):
        """Base feature count of h1-h3 (one more under nesting: the h0 step-size term)."""
        return 2 * self.D + 1 + (1 if self.nested else 0)

    def set_pairs(self, pairs: np.ndarray):
        self.pairs = np.asarray(pairs, dtype=int).reshape(-1, 2)
        if 2 in self.models:
            self.models[2].reset_tail(self._p(2), self._nb())

    def losses(self, z1, z3, y: int) -> np.ndarray:
        """Per-hypothesis log-loss -log p_g(y | delta) with pre-update parameters."""
        out = np.full(4, np.nan)
        for g in self.hyps:
            pr = float(self.models[g].prob(features(g, z1[None, :], z3[None, :], self.pairs, self.nested))[0])
            out[g] = -np.log(pr if y == 1 else 1.0 - pr)
        return out

    def add(self, z1, z3, y: int, w: float) -> np.ndarray:
        ell = self.losses(z1, z3, y)
        rho = self.cfg.rho
        for g in self.hyps:
            self.L[g] = rho * self.L[g] + w * ell[g]
        k = self.n % self.cfg.W_x
        self.z1[k], self.z3[k], self.y[k], self.w[k], self.idx[k] = z1, z3, y, w, self.n
        self.n += 1
        return ell

    def window(self):
        m = min(self.n, self.cfg.W_x)
        return self.z1[:m], self.z3[:m], self.y[:m], self.w[:m], self.idx[:m]

    def refit(self):
        if self.n == 0:
            return
        z1, z3, y, w, idx = self.window()
        sw = w * self.cfg.lam_m ** (self.n - 1 - idx)
        for g in self.hyps:
            self.models[g].irls_step(features(g, z1, z3, self.pairs, self.nested), y, sw)

    def pi(self) -> np.ndarray:
        return posterior(self.L, self.hyps, self.cfg.eta, self.cfg.pi_min)


def posterior(L: np.ndarray, hyps, eta: float, pi_min: float) -> np.ndarray:
    """pi(g) ∝ exp(-eta L_g) over hyps (uniform prior), then floored (spec §9)."""
    out = np.zeros(4)
    h = list(hyps)
    a = -eta * L[h]
    a = a - a.max()
    p = np.exp(a)
    p /= p.sum()
    out[h] = pi_min + (1.0 - len(h) * pi_min) * p
    return out


# ------------------------------------------------------------------ L1 linkage
def _soft(x, t):
    return np.sign(x) * np.maximum(np.abs(x) - t, 0.0)


def fit_linkage(z1: np.ndarray, y: np.ndarray, w: np.ndarray, cfg: SHEConfig, nested: bool = False):
    """Sparse pairwise-interaction logistic fit (spec §11): base terms L2, pair
    terms L1; lambda chosen by BIC on a 10-point geometric path; FISTA solver.
    Returns (pairs (k,2), groups list, info dict)."""
    n, D = z1.shape
    if n < 2 * D + 50:
        return np.zeros((0, 2), int), [np.array([j]) for j in range(D)], {"n": n, "skipped": True}
    iu = np.triu_indices(D, 1)
    P = z1[:, iu[0]] * z1[:, iu[1]]
    sd = P.std(axis=0)
    sd[sd < 1e-12] = 1.0
    P = P / sd
    Bm = np.hstack([np.ones((n, 1)), z1, z1 * z1])
    if nested:                     # Amendment 2 B-1: step-size term as an extra L2-penalised base column
        Bm = np.hstack([Bm[:, :1], np.log1p(np.linalg.norm(z1, axis=1))[:, None], Bm[:, 1:]])
    X = np.hstack([Bm, P])
    nb = Bm.shape[1]
    wn = w / w.mean()
    pen2 = np.full(nb, cfg.kappa2)
    pen2[0] = 0.0

    def grad(th):
        p = sigmoid(X @ th)
        g = -(X.T @ (wn * (y - p)))
        g[:nb] += pen2 * th[:nb]
        return g

    def nll(th):
        p = np.clip(sigmoid(X @ th), 1e-12, 1 - 1e-12)
        return -float(np.sum(wn * (y * np.log(p) + (1 - y) * np.log(1 - p))))

    th = np.zeros(X.shape[1])
    ybar = float(np.clip(np.average(y, weights=wn), 1e-3, 1 - 1e-3))
    th[0] = np.log(ybar / (1 - ybar))
    # base-only MAP (Newton) to define lambda_max
    for _ in range(15):
        p = sigmoid(Bm @ th[:nb])
        g = Bm.T @ (wn * (y - p)) - pen2 * th[:nb]
        Hm = (Bm * (wn * p * (1 - p))[:, None]).T @ Bm + np.diag(pen2) + 1e-10 * np.eye(nb)
        th[:nb] += np.linalg.solve(Hm, g)
    lam_max = float(np.max(np.abs(grad(th)[nb:])))
    if lam_max <= 0:
        return np.zeros((0, 2), int), [np.array([j]) for j in range(D)], {"n": n, "lam_max": 0.0}
    Lip = 0.25 * float(np.linalg.norm(X, 2)) ** 2 * float(wn.max()) + cfg.kappa2
    step = 1.0 / Lip
    n_eff = float(wn.sum() / wn.max())
    best = (np.inf, None, None)
    for lam in lam_max * np.geomspace(1.0, 0.05, 10):
        x_prev, yk, tk = th.copy(), th.copy(), 1.0
        for _ in range(300):
            z = yk - step * grad(yk)
            z[nb:] = _soft(z[nb:], step * lam)
            t_new = 0.5 * (1 + np.sqrt(1 + 4 * tk * tk))
            yk = z + ((tk - 1) / t_new) * (z - x_prev)
            rel = np.linalg.norm(z - x_prev) / max(1e-12, np.linalg.norm(x_prev))
            x_prev, tk = z, t_new
            if rel < 1e-6:
                break
        th = x_prev
        k = int(np.count_nonzero(th[nb:]))
        bic = 2.0 * nll(th) + (k + nb) * np.log(n_eff)
        if bic < best[0]:
            best = (bic, lam, th.copy())
    c = best[2][nb:]
    nz = np.flatnonzero(c)
    pairs = np.stack([iu[0][nz], iu[1][nz]], axis=1) if nz.size else np.zeros((0, 2), int)
    groups = group_pairs(D, pairs, np.abs(c[nz]), cfg.g_max)
    return pairs, groups, {"n": n, "lam_max": lam_max, "lam": float(best[1]), "k": int(nz.size)}


def group_pairs(D: int, pairs: np.ndarray, strength: np.ndarray, g_max: int) -> list:
    """Union-find over edges in descending strength, merges capped at g_max (spec §11)."""
    parent = list(range(D))
    size = [1] * D

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for e in np.argsort(-strength, kind="stable"):
        a, b = find(int(pairs[e, 0])), find(int(pairs[e, 1]))
        if a != b and size[a] + size[b] <= g_max:
            parent[b] = a
            size[a] += size[b]
    comp = {}
    for j in range(D):
        comp.setdefault(find(j), []).append(j)
    return sorted((np.array(v) for v in comp.values()), key=lambda v: int(v[0]))
