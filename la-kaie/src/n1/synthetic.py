"""Synthetic structural suite S1-S8 (spec §12).

Primary structure-recovery tests: S1 (HI), S2 (HB), S3 (HR).
Secondary / control: S4 (HB, mixed), S5 (weak HR label), S6 (switching,
position-based reference label), S7 (controlled null-transfer scenario),
S8 (stochastic placebo; no structural label).

Instances are generated from the instance seed 91000000 + 100*fid + D.
Every problem returns f >= 0 with f* = 0 (error = f).
"""
from __future__ import annotations

import numpy as np

BOX = (-100.0, 100.0)
LABELS = {1: "HI", 2: "HB", 3: "HR", 4: "HB", 5: "HR", 6: "switch", 7: "null", 8: None}
PRIMARY = (1, 2, 3)
BLOCK = 5


def instance_seed(fid: int, D: int) -> int:
    return 91000000 + 100 * fid + D


def haar(rng, n: int) -> np.ndarray:
    A = rng.standard_normal((n, n))
    Q, R = np.linalg.qr(A)
    return Q * np.sign(np.diag(R))


def cond_weights(D: int) -> np.ndarray:
    return 10.0 ** (4.0 * np.arange(D) / (D - 1))


class SyntheticProblem:
    benchmark = "SYN"

    def __init__(self, fid: int, D: int, run_seed: int | None = None):
        if fid not in LABELS:
            raise KeyError(fid)
        self.fid, self.dim = int(fid), int(D)
        self.name = f"S{fid}"
        self.label = LABELS[fid]
        self.primary = fid in PRIMARY
        self.lb = np.full(D, BOX[0])
        self.ub = np.full(D, BOX[1])
        self.seed_inst = instance_seed(fid, D)
        rng = np.random.default_rng(self.seed_inst)
        self.o = rng.uniform(-80.0, 80.0, D)
        self.c = cond_weights(D)
        self.true_partition = None
        self.init_boxes = None
        self.run_seed = run_seed
        self._fe_index = 0            # rows evaluated so far (S8 rotation stream)
        if fid == 1:
            self.true_partition = [np.array([j]) for j in range(D)]
        elif fid in (2, 4):
            perm = rng.permutation(D)
            n_sep = 0 if fid == 2 else (10 if D == 20 else 5)
            M = np.eye(D)
            blocks = [np.array([p]) for p in perm[:n_sep]]
            for s in range(n_sep, D, BLOCK):
                idx = np.arange(s, min(s + BLOCK, D))
                M[np.ix_(idx, idx)] = haar(rng, idx.size)
                blocks.append(np.sort(perm[idx]))
            self.perm, self.M = perm, M
            self.true_partition = sorted(blocks, key=lambda b: int(b[0]))
        elif fid in (3, 5, 6):
            self.Q = haar(rng, D)
        elif fid == 7:
            while True:
                oA, oB = rng.uniform(-80.0, 80.0, D), rng.uniform(-80.0, 80.0, D)
                if np.linalg.norm(oA - oB) >= 100.0:
                    break
            self.oA, self.oB = oA, oB
            self.QA, self.QB = haar(rng, D), haar(rng, D)
            self.delta7 = 1.0
            # HBA initialised in oA +- 20, MPA in oB +- 20 (clipped to the box)
            self.init_boxes = {"H": (np.maximum(oA - 20, self.lb), np.minimum(oA + 20, self.ub)),
                               "M": (np.maximum(oB - 20, self.lb), np.minimum(oB + 20, self.ub))}
        elif fid == 8:
            if run_seed is None:
                raise ValueError("S8 needs the run seed (rotation stream keyed by run and FE index)")
        self.r0, self.kappa_r = 10.0, 1.0

    # --------------------------------------------------------------- pieces
    def _ell(self, Y):
        return np.sum(self.c * Y * Y, axis=1)

    def w_s6(self, X):
        """S6 blending weight w = sigma((r0 - ||z||) / kappa_r); used offline for labels."""
        r = np.linalg.norm(np.atleast_2d(X) - self.o, axis=1)
        return 1.0 / (1.0 + np.exp(-(self.r0 - r) / self.kappa_r))

    def __call__(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        m = X.shape[0]
        f = self._eval(X)
        self._fe_index += m
        return f

    def _eval(self, X):
        fid = self.fid
        if fid == 1:
            return self._ell(X - self.o)
        if fid in (2, 4):
            Z = (X - self.o)[:, self.perm]
            return self._ell(Z @ self.M.T)
        if fid == 3:
            return self._ell((X - self.o) @ self.Q.T)
        if fid == 5:
            Y = 0.0512 * (X - self.o) @ self.Q.T
            return np.sum(Y * Y - 10.0 * np.cos(2.0 * np.pi * Y) + 10.0, axis=1)
        if fid == 6:
            Z = X - self.o
            w = self.w_s6(X)
            return (1.0 - w) * self._ell(Z) + w * self._ell(Z @ self.Q.T)
        if fid == 7:
            fA = self._ell((X - self.oA) @ self.QA.T)
            fB = self._ell((X - self.oB) @ self.QB.T) + self.delta7
            return np.minimum(fA, fB)
        if fid == 8:
            Z = X - self.o
            out = np.empty(X.shape[0])
            for i in range(X.shape[0]):
                r = np.random.default_rng([self.seed_inst, int(self.run_seed), self._fe_index + i])
                out[i] = float(np.sum(self.c * (haar(r, self.dim) @ Z[i]) ** 2))
            return out
        raise KeyError(fid)


def synthetic_instance_id(fid: int, D: int) -> int:
    """Run-seed instance index (amendment A-6): 10*fid + (1 if D == 10 else 2)."""
    return 10 * fid + (1 if D == 10 else 2)
