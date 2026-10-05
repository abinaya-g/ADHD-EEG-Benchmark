"""Structural hypotheses h0-h3: transforms, masks and candidate construction
(experiments/SHE_SPEC.md §3-§5)."""
from __future__ import annotations

import numpy as np


def pick_tournament(rng, F: np.ndarray) -> int:
    """Binary tournament of two distinct agents; lower f wins, ties -> lower index.
    Same semantics as src/n1/mapping.pick_receiver (tested for equality)."""
    a, b = rng.choice(F.shape[0], size=2, replace=False)
    a, b = int(a), int(b)
    if F[a] < F[b] or (F[a] == F[b] and a < b):
        return a
    return b


def draw_mask(rng, n_units: int, p_x: float) -> np.ndarray:
    """Each unit with probability p_x; if none is drawn, one uniformly
    (src/n1/mapping.draw_mask semantics)."""
    m = rng.random(n_units) < p_x
    if not m.any():
        m[int(rng.integers(n_units))] = True
    return m


class Frame:
    """Coordinate information shared by all hypotheses at one moment of a run:
    per-coordinate scale s (pooled populations), eigenbasis B and centre mu of the
    pooled elite (h3), eigen-frame scale sB, and linkage groups (h2)."""

    def __init__(self, D: int, lb: np.ndarray, ub: np.ndarray):
        self.D = D
        self.width = ub - lb
        self.s = self.width / 4.0
        self.B = np.eye(D)
        self.mu = (lb + ub) / 2.0
        self.sB = self.s.copy()
        self.groups = [np.array([j]) for j in range(D)]
        self.pairs = np.zeros((0, 2), dtype=int)

    # ---- scales (every iteration) ----
    def update_scale(self, XH: np.ndarray, XM: np.ndarray):
        X = np.vstack([XH, XM])
        floor = 1e-12 * self.width
        self.s = np.maximum(X.std(axis=0), floor)
        self.sB = np.maximum(((X - self.mu) @ self.B).std(axis=0), 1e-12 * float(np.max(self.width)))

    # ---- pooled-elite eigenbasis (every U iterations; spec §4.4) ----
    def refresh_basis(self, XH, FH, XM, FM, q_e: float, lam_LW: float):
        if float(np.max(self.s / self.width)) < 1e-10:
            return False                                   # collapsed: keep previous basis
        n = int(np.ceil(q_e * XH.shape[0]))
        EH = XH[np.argsort(FH, kind="stable")[:n]]
        EM = XM[np.argsort(FM, kind="stable")[:n]]
        Z = np.vstack([EH - EH.mean(axis=0), EM - EM.mean(axis=0)]) / self.s
        C = np.cov(Z, rowvar=False, ddof=1)
        C = (1.0 - lam_LW) * C + lam_LW * (np.trace(C) / self.D) * np.eye(self.D)
        S = np.diag(self.s)
        w, V = np.linalg.eigh(S @ C @ S)
        self.B = V[:, np.argsort(w)[::-1]]
        self.mu = np.vstack([EH, EM]).mean(axis=0)
        return True

    def set_linkage(self, pairs: np.ndarray, groups: list):
        self.pairs = np.asarray(pairs, dtype=int).reshape(-1, 2)
        self.groups = groups


# ---- transforms (affine, invertible) ----
def T(h: int, x: np.ndarray, fr: Frame) -> np.ndarray:
    return fr.B.T @ (x - fr.mu) if h == 3 else x.copy()


def T_inv(h: int, z: np.ndarray, fr: Frame) -> np.ndarray:
    return fr.B @ z + fr.mu if h == 3 else z.copy()


def unit_mask(h: int, rng, fr: Frame, p_x: float) -> np.ndarray:
    """Coordinate-level mask in the frame of h (bool, length D)."""
    D = fr.D
    if h == 0:
        return np.ones(D, dtype=bool)
    if h in (1, 3):
        return draw_mask(rng, D, p_x)
    if h == 2:
        g = draw_mask(rng, len(fr.groups), p_x)
        m = np.zeros(D, dtype=bool)
        for sel, idx in zip(g, fr.groups):
            if sel:
                m[idx] = True
        return m
    raise KeyError(h)


def candidate(h: int, x_r: np.ndarray, x_d: np.ndarray, fr: Frame, rng, p_x: float):
    """x' = T_h^{-1}[(1-m) * T_h(x_r) + m * T_h(x_d)] (no clipping here)."""
    m = unit_mask(h, rng, fr, p_x)
    zr, zd = T(h, x_r, fr), T(h, x_d, fr)
    return T_inv(h, np.where(m, zd, zr), fr), m


def record_features(delta: np.ndarray, fr: Frame):
    """Features fixed at record time (spec §5): z1 = S^-1 delta, z3 = S_B^-1 B^T delta."""
    return delta / fr.s, (fr.B.T @ delta) / fr.sB
