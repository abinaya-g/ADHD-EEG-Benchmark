"""Structural models HI / HB / HR, Gaussian log-likelihood and BIC (spec §4.2),
and the prequential block partition (spec §4.3).

Everything here is a pure function of already-recorded displacement
windows: no objective evaluation, no random numbers.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import norm

LOG2PI = math.log(2.0 * math.pi)


class NumericalFailure(RuntimeError):
    """Cholesky failed even after the documented extra ridge (spec §9)."""


def scatter(Z: np.ndarray):
    """Sample mean and MLE scatter matrix S = (1/n) sum (z - zbar)(z - zbar)^T."""
    Z = np.asarray(Z, dtype=np.float64)
    mu = Z.mean(axis=0)
    C = Z - mu
    return mu, (C.T @ C) / Z.shape[0]


def eig_floor(S: np.ndarray, eps: float):
    """Symmetric eigen-decomposition with eigenvalues floored at eps."""
    w, V = np.linalg.eigh((S + S.T) / 2.0)
    floored = bool(np.any(w < eps))
    w = np.maximum(w, eps)
    return (V * w) @ V.T, w, V, floored


@dataclass
class Fit:
    kind: str
    Sigma: np.ndarray
    q: int
    floored: bool


def fit_model(S: np.ndarray, kind: str, partition=None, eps_rel: float = 1e-8) -> Fit:
    """Covariance estimate of model ``kind`` from the MLE scatter S (spec §4.2)."""
    D = S.shape[0]
    tr = float(np.trace(S))
    eps = eps_rel * tr / D if tr > 0 else 0.0
    if kind == "HI":
        d = np.diag(S)
        return Fit("HI", np.diag(np.maximum(d, eps)), 2 * D, bool(np.any(d < eps)))
    if kind == "HR":
        Sig, _, _, fl = eig_floor(S, eps)
        return Fit("HR", Sig, D + D * (D + 1) // 2, fl)
    if kind == "HB":
        if partition is None:
            raise ValueError("HB requires a partition")
        Sig = np.zeros_like(S)
        q, fl = D, False
        for B in partition:
            B = np.asarray(B)
            Sb, _, _, f = eig_floor(S[np.ix_(B, B)], eps)
            Sig[np.ix_(B, B)] = Sb
            q += len(B) * (len(B) + 1) // 2
            fl = fl or f
        return Fit("HB", Sig, q, fl)
    raise KeyError(kind)


def _chol(Sigma: np.ndarray, eps: float) -> np.ndarray:
    try:
        return np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError:
        try:     # spec §9: add 10*eps*I once
            return np.linalg.cholesky(Sigma + 10.0 * max(eps, 1e-300) * np.eye(Sigma.shape[0]))
        except np.linalg.LinAlgError as e:
            raise NumericalFailure(str(e)) from e


def loglik_from_scatter(n: int, S: np.ndarray, Sigma: np.ndarray, eps: float) -> float:
    """l(Z) = -(n/2) [D log 2pi + log det Sigma + tr(Sigma^-1 S)] (spec §4.2)."""
    D = S.shape[0]
    L = _chol(Sigma, eps)
    logdet = 2.0 * float(np.sum(np.log(np.diag(L))))
    Linv_S = np.linalg.solve(L, S)
    tr = float(np.trace(np.linalg.solve(L.T, Linv_S)))
    return -0.5 * n * (D * LOG2PI + logdet + tr)


def bic(ll: float, q: int, n: int) -> float:
    return -2.0 * ll + q * math.log(n)


def predictive_loglik(Z_cur: np.ndarray, mu: np.ndarray, Sigma: np.ndarray, eps: float) -> float:
    """sum_{z in CUR} log N(z; mu_OLD, Sigma_OLD) (spec §4.6)."""
    D = Z_cur.shape[1]
    L = _chol(Sigma, eps)
    R = np.linalg.solve(L, (Z_cur - mu).T)
    logdet = 2.0 * float(np.sum(np.log(np.diag(L))))
    return float(-0.5 * (Z_cur.shape[0] * (D * LOG2PI + logdet) + np.sum(R * R)))


# ------------------------------------------------------------ partition (§4.3)
def correlation(Z: np.ndarray) -> np.ndarray:
    """Pearson correlation; zero-variance columns get r = 0 with all others."""
    Z = np.asarray(Z, dtype=np.float64)
    C = Z - Z.mean(axis=0)
    sd = np.sqrt(np.sum(C * C, axis=0))
    ok = sd > 0
    R = np.zeros((Z.shape[1], Z.shape[1]))
    Cn = np.zeros_like(C)
    Cn[:, ok] = C[:, ok] / sd[ok]
    R = Cn.T @ Cn
    R[~ok, :] = 0.0
    R[:, ~ok] = 0.0
    np.fill_diagonal(R, 1.0)
    return R


def fisher_stat(Z: np.ndarray, clip: float) -> np.ndarray:
    n = Z.shape[0]
    R = np.clip(correlation(Z), -clip, clip)
    return np.arctanh(R) * math.sqrt(max(n - 3, 0))


def stouffer_stat(Z_list, clip: float) -> np.ndarray:
    """Joint statistic: sum_p sqrt(n_p - 3) atanh(r^p) / sqrt(P) (spec §4.3; P = 2)."""
    tot = None
    for Z in Z_list:
        s = fisher_stat(Z, clip)
        tot = s if tot is None else tot + s
    return tot / math.sqrt(len(Z_list))


def edge_threshold(D: int, alpha_link: float) -> float:
    m = D * (D - 1) / 2
    return float(norm.ppf(1.0 - alpha_link / (2.0 * m)))


def partition_from_stat(zeta: np.ndarray, alpha_link: float) -> list[np.ndarray]:
    """Connected components of {(i,j): |zeta_ij| > z_{1 - alpha/(2m)}}. Deterministic;
    blocks are sorted and ordered by their smallest index."""
    D = zeta.shape[0]
    A = np.abs(zeta) > edge_threshold(D, alpha_link)
    np.fill_diagonal(A, False)
    _, lab = connected_components(csr_matrix(A.astype(np.int8)), directed=False)
    blocks = [np.flatnonzero(lab == c) for c in np.unique(lab)]
    blocks.sort(key=lambda b: int(b[0]))
    return blocks


def degeneracy(partition) -> str | None:
    """'HI' if all singletons, 'HR' if one block, else None (spec §4.3)."""
    if partition is None:
        return None
    if all(len(b) == 1 for b in partition):
        return "HI"
    if len(partition) == 1:
        return "HR"
    return None


def partition_str(partition) -> str:
    if partition is None:
        return ""
    return "|".join(",".join(str(int(i)) for i in b) for b in partition)
