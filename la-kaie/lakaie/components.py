"""LA-KAIE building blocks.

InteractionModel  - variable-dependency matrix G from already-evaluated points
                    (local quadratic surrogate, rank-transformed fitness), no FEs
group_variables   - deterministic constrained average-linkage clustering of G
LandscapeState    - the ten normalised state variables S1..S10
BanditController  - epsilon-greedy linear contextual bandit on binned state
"""
from __future__ import annotations

import math

import numpy as np
from scipy.special import ndtri

ACTIONS = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"]
ACTION_NAMES = {
    "A1": "no exchange (native HBA-MPA update)",
    "A2": "elite individual exchange",
    "A3": "dimension-wise cross-population exchange",
    "A4": "interaction-group exchange",
    "A5": "exploration-oriented exchange",
    "A6": "exploitation-oriented exchange",
    "A7": "complementary-source exchange",
    "A8": "local refinement exchange",
}


# ----------------------------------------------------------------- archive
class Archive:
    """Ring buffer of evaluated points (x, f)."""

    def __init__(self, size: int, dim: int):
        self.X = np.zeros((size, dim)); self.F = np.zeros(size)
        self.size, self.n, self.ptr = size, 0, 0

    def add(self, X, F):
        X = np.atleast_2d(X); F = np.ravel(F)
        for x, f in zip(X, F):
            self.X[self.ptr] = x; self.F[self.ptr] = f
            self.ptr = (self.ptr + 1) % self.size
            self.n = min(self.n + 1, self.size)

    def data(self):
        return self.X[:self.n], self.F[:self.n]


# ----------------------------------------------------------------- interaction
def quadratic_features(Z: np.ndarray) -> np.ndarray:
    M, D = Z.shape
    iu = np.triu_indices(D)
    quad = Z[:, iu[0]] * Z[:, iu[1]]
    return np.hstack([np.ones((M, 1)), Z, quad])


def estimate_interaction(X, F, center, width, sample_factor=1.5, ridge=1e-3, t_crit=3.0):
    """Local quadratic surrogate around ``center`` fitted to rank-normal scores
    of the nearest archive points. Returns G in [0,1]^{DxD} (zero diagonal)
    or None if too few points. Invariant to monotone transformations of f
    (only ranks are used) and to per-coordinate affine rescaling.

    G_ij = min(1, |H_ij| / sqrt(|H_ii||H_jj|)) if the cross coefficient is
    statistically distinguishable from zero (|t_ij| >= t_crit, OLS standard
    error from the residual variance), else 0."""
    D = X.shape[1]
    p = 1 + D + D * (D + 1) // 2
    M = int(math.ceil(sample_factor * p))
    if X.shape[0] < p + 10:
        return None
    d = np.sum(((X - center) / width) ** 2, axis=1)
    idx = np.argsort(d, kind="stable")[:min(M, X.shape[0])]
    Xs, Fs = X[idx], F[idx]
    s = Xs.std(axis=0)
    s = np.maximum(s, 1e-12 * width)
    Z = (Xs - center) / s
    ranks = np.argsort(np.argsort(Fs, kind="stable"), kind="stable")
    y = ndtri((ranks + 0.5) / len(Fs))
    Phi = quadratic_features(Z)
    A = Phi.T @ Phi
    lam = ridge * np.trace(A) / A.shape[0]
    reg = lam * np.eye(A.shape[0]); reg[0, 0] = 0.0
    Ainv = np.linalg.inv(A + reg)
    beta = Ainv @ (Phi.T @ y)
    dof = max(1, len(y) - Phi.shape[1])
    sigma2 = float(np.sum((y - Phi @ beta) ** 2) / dof)
    se = np.sqrt(np.maximum(sigma2 * np.diag(Ainv), 1e-300))
    H = np.zeros((D, D))
    Tm = np.zeros((D, D))
    iu = np.triu_indices(D)
    H[iu] = beta[1 + D:]
    Tm[iu] = beta[1 + D:] / se[1 + D:]
    H = H + H.T                             # off-diagonal: beta_ij ; diagonal: 2*beta_ii
    Tm = Tm + Tm.T
    diag = np.abs(np.diag(H))
    denom = np.sqrt(np.outer(diag, diag))
    with np.errstate(divide="ignore", invalid="ignore"):
        G = np.where(denom > 0, np.abs(H) / denom, 0.0)
    G = np.clip(np.nan_to_num(G), 0.0, 1.0)
    G[np.abs(Tm) < t_crit] = 0.0
    np.fill_diagonal(G, 0.0)
    return G


def fir_prior(Q: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.abs(np.diag(Q)))
    G = np.abs(Q) / np.outer(d, d)
    np.fill_diagonal(G, 0.0)
    return np.clip(G, 0, 1)


def group_variables(G: np.ndarray, tau: float, gmax: int):
    """Deterministic average-linkage agglomeration: repeatedly merge the pair of
    clusters with the highest mean between-cluster G, provided it is >= tau and
    the merged size <= gmax. Ties broken by smallest (min-index) cluster ids."""
    D = G.shape[0]
    clusters = [[i] for i in range(D)]
    gmax = max(1, min(gmax, D))
    while True:
        best, pair = -1.0, None
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                if len(clusters[a]) + len(clusters[b]) > gmax:
                    continue
                v = float(G[np.ix_(clusters[a], clusters[b])].mean())
                if v > best + 1e-15:
                    best, pair = v, (a, b)
        if pair is None or best < tau:
            break
        a, b = pair
        clusters[a] = sorted(clusters[a] + clusters[b])
        del clusters[b]
        clusters.sort(key=lambda c: c[0])
    return clusters


def interaction_strength(G: np.ndarray) -> float:
    """S8: mean over variables of the strongest coupling (robust for small D)."""
    if G is None or G.shape[0] < 2:
        return 0.0
    return float(np.mean(G.max(axis=1)))


def interaction_consistency(G, tau, transferred: np.ndarray) -> float:
    """(kept - broken)/(kept + broken) over strong pairs touching the transferred
    set; kept = both ends transferred, broken = one end only. 0 if none."""
    if G is None:
        return 0.0
    D = G.shape[0]
    T = np.zeros(D, bool); T[transferred] = True
    strong = np.triu(G >= tau, 1)
    both = strong & np.outer(T, T)
    one = strong & (np.outer(T, ~T) | np.outer(~T, T))
    kept, broken = int(both.sum()), int(one.sum())
    return 0.0 if kept + broken == 0 else (kept - broken) / (kept + broken)


# ----------------------------------------------------------------- state
STATE_NAMES = ["S1_diversity", "S2_improvement", "S3_stagnation", "S4_progress",
               "S5_fitness_gap", "S6_hba_contribution", "S7_mpa_contribution",
               "S8_interaction", "S9_exchange_success", "S10_exploration"]


class LandscapeState:
    """Only information available up to the current evaluation is used."""

    def __init__(self, cfg, lb, ub):
        self.cfg, self.lb, self.ub = cfg, lb, ub
        self.width = ub - lb
        self.div0 = None
        self.best_hist = []            # global best at the end of each iteration
        self.gap_hist = []
        self.stag = 0
        self.s6 = self.s7 = self.s9 = 0.0
        self.step = 0.0
        self.s8 = 0.0

    def diversity(self, pops):
        X = np.vstack(pops)
        return float(np.mean(X.std(axis=0) / self.width))

    def end_iteration(self, best, pooled_f, hba_succ, mpa_succ, step):
        e = self.cfg["ema"]
        if self.best_hist and best < self.best_hist[-1]:
            self.stag = 0
        elif self.best_hist:
            self.stag += 1
        self.best_hist.append(best)
        self.gap_hist.append(float(np.mean(pooled_f) - best))
        self.s6 = (1 - e) * self.s6 + e * hba_succ
        self.s7 = (1 - e) * self.s7 + e * mpa_succ
        self.step = step

    def exchange_outcome(self, success: bool):
        e = self.cfg["ema"]
        self.s9 = (1 - e) * self.s9 + e * float(success)

    def vector(self, pops, pooled_f, best, progress, stag_extra=0):
        div = self.diversity(pops)
        if self.div0 is None:
            self.div0 = max(div, 1e-300)
        s1 = min(1.0, div / self.div0)
        w = self.cfg["improvement_window"]
        if len(self.best_hist) > w:
            delta = self.best_hist[-1 - w] - best
        elif self.best_hist:
            delta = self.best_hist[0] - best
        else:
            delta = 0.0
        spread = float(np.median(pooled_f) - best)
        s2 = delta / (delta + spread) if (delta + spread) > 0 else 0.0
        stag = self.stag + stag_extra
        s3 = stag / (stag + self.cfg["stagnation_scale"])
        gap = float(np.mean(pooled_f) - best)
        gw = self.gap_hist[-self.cfg["gap_window"]:] + [gap]
        gmax = max(gw)
        s5 = gap / gmax if gmax > 0 else 0.0
        X = np.vstack(pops)
        rad = float(np.mean(np.linalg.norm(X - X.mean(axis=0), axis=1)))
        s10 = self.step / (self.step + rad) if (self.step + rad) > 0 else 0.0
        v = np.array([s1, s2, s3, progress, s5, self.s6, self.s7, self.s8, self.s9, s10])
        return np.clip(v, 0.0, 1.0)


# ----------------------------------------------------------------- controller
class BanditController:
    """epsilon-greedy linear contextual bandit.

    Context: each S_i in [0,1] is discretised into ``bins`` equal-width bins and
    one-hot encoded (10*bins features) plus a bias. Per action a, a
    ridge-regression value model q_a(x) = theta_a.x is maintained with
    exponential forgetting of the updated action's statistics (non-stationary
    search). Ties among maximal values are broken uniformly at random."""

    def __init__(self, cfg, rng, n_actions=8):
        self.cfg, self.rng, self.n = cfg, rng, n_actions
        self.bins = int(cfg["bins"])
        self.use_state = bool(cfg.get("use_state", True))
        self.dim = (10 * self.bins + 1) if self.use_state else 1
        lam = float(cfg["ridge_lambda"])
        self.A = np.stack([lam * np.eye(self.dim) for _ in range(n_actions)])
        self.b = np.zeros((n_actions, self.dim))
        self.theta = np.zeros((n_actions, self.dim))
        self.kind = cfg["type"]
        self.last_was_random = False

    def features(self, s: np.ndarray) -> np.ndarray:
        if not self.use_state:
            return np.ones(1)
        x = np.zeros(self.dim)
        idx = np.minimum((s * self.bins).astype(int), self.bins - 1)
        x[np.arange(10) * self.bins + idx] = 1.0
        x[-1] = 1.0
        return x

    def epsilon(self, progress):
        return max(self.cfg["epsilon_min"], self.cfg["epsilon0"] * (1 - progress))

    def values(self, x):
        return self.theta @ x

    def select(self, s, mask, progress):
        avail = np.flatnonzero(mask)
        self.last_was_random = False
        if self.kind == "uniform_random":
            self.last_was_random = True
            return int(self.rng.choice(avail))
        if self.kind == "fixed_action":
            a = ACTIONS.index(self.cfg["fixed_action"])
            return a if mask[a] else int(self.rng.choice(avail))
        if self.kind == "rule_based":
            return self._rule(s, mask)
        if self.rng.random() < self.epsilon(progress):
            self.last_was_random = True
            return int(self.rng.choice(avail))
        q = self.values(self.features(s))[avail]
        best = avail[np.flatnonzero(q >= q.max() - 1e-12)]
        return int(self.rng.choice(best))

    def _rule(self, s, mask):
        """Static expert rules (NoAdaptiveController ablation)."""
        stag, prog, inter = s[2], s[3], s[7]
        if stag >= 0.5:
            a = "A5"
        elif prog > 0.7:
            a = "A8"
        elif inter >= 0.2:
            a = "A4"
        else:
            a = "A3"
        i = ACTIONS.index(a)
        return i if mask[i] else int(np.flatnonzero(mask)[0])

    def update(self, s, a, r):
        if self.kind != "contextual_bandit":
            return
        x = self.features(s)
        lam = self.cfg["forgetting"]
        self.A[a] = lam * self.A[a] + np.outer(x, x) + (1 - lam) * self.cfg["ridge_lambda"] * np.eye(self.dim)
        self.b[a] = lam * self.b[a] + r * x
        self.theta[a] = np.linalg.solve(self.A[a], self.b[a])
