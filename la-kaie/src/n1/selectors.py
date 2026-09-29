"""Structural selectors (spec §4.4-4.6).

HBA-only, MPA-only, JOINT (shared hypothesis + shared partition,
population-specific parameters, summed BIC), POOLED (same observations as
JOINT, one common model), DECOUPLED (MPA window column-permuted; placebo) and
the offline-only JOINT-HALF (``evaluate_sources`` on halved windows).

Pure functions of recorded windows. The only randomness is the DECOUPLED
column permutation, drawn from the dedicated ``rng_placebo`` stream.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import HYPOTHESES, N1Config
from .records import rows_digest
from .structure import (NumericalFailure, bic, degeneracy, fisher_stat, fit_model,
                        loglik_from_scatter, partition_from_stat, predictive_loglik,
                        scatter, stouffer_stat)

Q_ORDER = {"HI": 0, "HB": 1, "HR": 2}     # tie-break: fewer parameters first


@dataclass
class SelectorResult:
    name: str
    evidence: bool = False
    available: tuple = ()
    bic: dict = field(default_factory=dict)
    ll: dict = field(default_factory=dict)
    q: dict = field(default_factory=dict)
    pll: dict = field(default_factory=dict)
    floored: dict = field(default_factory=dict)
    partition: list | None = None
    degenerate: str | None = None
    k_hat: str | None = None
    margin: float = float("nan")
    n: dict = field(default_factory=dict)
    basis: dict = field(default_factory=dict)      # pop -> eigenvectors of the HR fit on CUR
    digest: str = ""
    numerical_failure: bool = False
    degenerate_window: bool = False


def _permute_columns(Z: np.ndarray, rng) -> np.ndarray:
    out = np.empty_like(Z)
    for j in range(Z.shape[1]):
        out[:, j] = Z[rng.permutation(Z.shape[0]), j]
    return out


def evaluate_sources(name: str, sources: dict, partition_mode: str, cfg: N1Config) -> SelectorResult:
    """Score HI/HB/HR on the CUR windows of ``sources`` = {pop: (cur, old)}.

    partition_mode: "single" (one source; Fisher statistic on its OLD) or
    "stouffer" (several sources; Stouffer combination of their OLD windows).
    The partition is computed from OLD only (prequential, spec §4.3).
    """
    res = SelectorResult(name)
    if any(cur is None for cur, _ in sources.values()):
        return res                                  # warm-up: no evidence (§6.4)
    res.n = {p: int(cur.shape[0]) for p, (cur, _) in sources.items()}
    if any(float(np.trace(scatter(cur)[1])) <= 0.0 for cur, _ in sources.values()):
        res.degenerate_window = True           # spec §9: keep the previous k*
        return res
    res.evidence = True
    have_old = all(old is not None for _, old in sources.values())
    if have_old:
        olds = [old for _, old in sources.values()]
        zeta = fisher_stat(olds[0], cfg.fisher_clip) if partition_mode == "single" \
            else stouffer_stat(olds, cfg.fisher_clip)
        res.partition = partition_from_stat(zeta, cfg.alpha_link)
        res.degenerate = degeneracy(res.partition)
    cands = ["HI", "HR"] + (["HB"] if have_old and res.degenerate is None else [])
    stats = {p: scatter(cur) for p, (cur, _) in sources.items()}
    try:
        for k in cands:
            ll = q = b = 0.0
            fl = False
            for p, (cur, _) in sources.items():
                mu, S = stats[p]
                fit = fit_model(S, k, res.partition if k == "HB" else None, cfg.eps_rel)
                eps = cfg.eps_rel * float(np.trace(S)) / S.shape[0]
                llp = loglik_from_scatter(cur.shape[0], S, fit.Sigma, eps)
                ll += llp
                q += fit.q
                b += bic(llp, fit.q, cur.shape[0])
                fl = fl or fit.floored
                if k == "HR":
                    res.basis[p] = np.linalg.eigh((fit.Sigma + fit.Sigma.T) / 2.0)[1]
            res.ll[k], res.q[k], res.bic[k], res.floored[k] = ll, int(q), b, fl
            if have_old:
                pll = 0.0
                for p, (cur, old) in sources.items():
                    mu_o, S_o = scatter(old)
                    fit_o = fit_model(S_o, k, res.partition if k == "HB" else None, cfg.eps_rel)
                    eps_o = cfg.eps_rel * float(np.trace(S_o)) / S_o.shape[0]
                    pll += predictive_loglik(cur, mu_o, fit_o.Sigma, eps_o)
                res.pll[k] = pll
    except NumericalFailure:
        res.numerical_failure = True
        res.evidence = False
        return res
    res.available = tuple(sorted(res.bic, key=lambda k: Q_ORDER[k]))
    order = sorted(res.available, key=lambda k: (res.bic[k], Q_ORDER[k]))
    # ties within 1e-9 -> fewer parameters
    best = order[0]
    for k in order[1:]:
        if abs(res.bic[k] - res.bic[best]) <= 1e-9 and Q_ORDER[k] < Q_ORDER[best]:
            best = k
    res.k_hat = best
    if len(order) > 1:
        res.margin = float(res.bic[order[1]] - res.bic[order[0]])
    res.digest = rows_digest(*[cur for cur, _ in sources.values()])
    return res


def evaluate_selector(name: str, win_H, win_M, cfg: N1Config, rng_placebo=None) -> SelectorResult:
    """Build the sources of selector ``name`` from the two structural windows."""
    cH, oH, cM, oM = win_H.cur(), win_H.old(), win_M.cur(), win_M.old()
    if name == "HBA":
        return evaluate_sources(name, {"H": (cH, oH)}, "single", cfg)
    if name == "MPA":
        return evaluate_sources(name, {"M": (cM, oM)}, "single", cfg)
    if name == "JOINT":
        return evaluate_sources(name, {"H": (cH, oH), "M": (cM, oM)}, "stouffer", cfg)
    if name == "POOLED":
        if cH is None or cM is None:
            return evaluate_sources(name, {"P": (None, None)}, "single", cfg)
        oP = np.vstack([oH, oM]) if (oH is not None and oM is not None) else None
        return evaluate_sources(name, {"P": (np.vstack([cH, cM]), oP)}, "single", cfg)
    if name == "DECOUPLED":
        if rng_placebo is None:
            raise ValueError("DECOUPLED requires rng_placebo")
        cMp = _permute_columns(cM, rng_placebo) if cM is not None else None
        oMp = _permute_columns(oM, rng_placebo) if oM is not None else None
        return evaluate_sources(name, {"H": (cH, oH), "M": (cMp, oMp)}, "stouffer", cfg)
    raise KeyError(name)


def joint_half_sources(cur_H, old_H, cur_M, old_M):
    """Offline JOINT-HALF (spec §4.4): newest W/2 rows of each CUR and of each OLD."""
    def half(Z):
        return None if Z is None else Z[-(Z.shape[0] // 2):]
    return {"H": (half(cur_H), half(old_H)), "M": (half(cur_M), half(old_M))}


class Hysteresis:
    """k_cur update of spec §4.5 (starts at HI, the default mapping)."""

    def __init__(self, kappa: float):
        self.kappa = kappa
        self.k_cur = "HI"

    def update(self, res: SelectorResult):
        if not res.evidence or not res.available:
            return self.k_cur, False, False
        if self.k_cur not in res.available:
            switched = self.k_cur != res.k_hat
            self.k_cur = res.k_hat
            return self.k_cur, switched, True
        if res.k_hat != self.k_cur and res.bic[self.k_cur] - res.bic[res.k_hat] > 2.0 * self.kappa:
            self.k_cur = res.k_hat
            return self.k_cur, True, False
        return self.k_cur, False, False


def basis_for(res: SelectorResult | None, pop: str):
    """Eigenbasis used for an HR exchange into population ``pop`` (amendment A-2):
    the receiver's own basis when the selector has one, else the selector's only one."""
    if res is None or not res.basis:
        return None
    if pop in res.basis:
        return res.basis[pop]
    if len(res.basis) == 1:
        return next(iter(res.basis.values()))
    return None


__all__ = ["SelectorResult", "evaluate_sources", "evaluate_selector", "joint_half_sources",
           "Hysteresis", "basis_for", "HYPOTHESES"]
