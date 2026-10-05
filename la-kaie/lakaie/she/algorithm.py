"""SHE run loop (experiments/SHE_SPEC.md §2-§3, §13, Amendment 1).

Stage 1 scope: exchange WITHOUT E3 (fixed-E regime, Amendment 1 A-2) with
evidence-based (SHE-NoE3) or uniform (SHE-Uniform) hypothesis selection, the
shared-data evidence engine, donor-split shadow evidence and native artefact
matrices. E3 is Stage 3 and is not implemented here (Variant.e3 must be False).

The backbone is lakaie.algorithms.hybrid_common.HybridState, used unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..algorithms.hybrid_common import HybridState
from ..core import Recorder, clip
from .config import HYPS, SHEConfig, Variant
from .evidence import Evidence, fit_linkage
from .hypotheses import Frame, candidate, pick_tournament, record_features


@dataclass
class RunOutput:
    best: float
    summary: dict
    arrays: dict = field(default_factory=dict)


def spawn_streams(seed: int):
    """rng_init = default_rng(seed) (as MPHBS); native/exchange/select streams from
    SeedSequence([seed, 2]) (spec §15.3)."""
    rng_init = np.random.default_rng(seed)
    native, exchange, select = [np.random.default_rng(s) for s in np.random.SeedSequence([seed, 2]).spawn(3)]
    return rng_init, native, exchange, select


class _Ring:
    def __init__(self, cap: int, D: int):
        self.buf = np.zeros((cap, D)); self.cap = cap; self.n = 0

    def extend(self, Z):
        for z in Z:
            self.buf[self.n % self.cap] = z
            self.n += 1

    def rows(self):
        return self.buf[:min(self.n, self.cap)]


def _offdiag_corr(Z):
    C = np.corrcoef(Z, rowvar=False)
    C = np.nan_to_num(C, nan=0.0)
    np.fill_diagonal(C, 0.0)
    return C


def run_she(problem, variant: Variant, cfg: SHEConfig, seed: int, obj, max_fe: int,
            rec: Recorder | None = None) -> RunOutput:
    if variant.e3:
        raise NotImplementedError("E3 is Stage 3 (not implemented in Stage 1)")
    N, D, E = cfg.N, problem.dim, cfg.E
    lb, ub = problem.lb, problem.ub
    hyps = tuple(variant.hyps)
    rng_init, rng_nat, rng_ex, rng_sel = spawn_streams(seed)
    st = HybridState(obj, lb, ub, D, N, rng_init)
    fr = Frame(D, lb, ub)
    fr.update_scale(st.X_H, st.X_M)
    ev = Evidence(cfg, D, hyps)
    shadows = {0: Evidence(cfg, D, hyps), 1: Evidence(cfg, D, hyps)} if variant.shadows else {}
    T_nom = max_fe / (3 * N + E)
    burn_fe = 0.1 * max_fe

    # per-record log (columns)
    R = {k: [] for k in ("t", "fe", "gen", "y", "don", "cost", "w")}
    R_pi, R_loss, R_pis = [], [], []
    native = {0: _Ring(cfg.W_A(D), D), 1: _Ring(cfg.W_A(D), D)}
    A_sum = {0: np.zeros((D, D)), 1: np.zeros((D, D))}
    A_cnt = {0: 0, 1: 0}
    A_n = {0: [], 1: []}
    link_log = []
    it_extra = {k: [] for k in ("pi", "piH", "piM", "h0att", "nev", "npairs")}
    counts = dict(h0_attempts=0, exchange_fe=0, iters=0, iters_not_E=0, records=0)
    slot = [0]

    def sel_probs(excl0: bool):
        p = ev.pi() if variant.selection == "evidence" else _uniform(hyps)
        if variant.selection == "fixed":
            p = np.zeros(4); p[variant.fixed_h] = 1.0
        if excl0 and p[0] > 0 and any(h != 0 for h in hyps):
            p = p.copy(); p[0] = 0.0; p /= p.sum()
        return p

    def native_capture(pop: int, X_before, X_after):
        moved = np.any(X_before != X_after, axis=1)
        if moved.any():
            native[pop].extend((X_after[moved] - X_before[moved]) / fr.s)

    def exchange_slots(target: int, it: int):
        evaluated, h0_att = 0, 0
        while evaluated < target and obj.remaining() > 0:
            a = 1 if slot[0] % 2 == 0 else 0          # receiver population (1 = MPA first)
            slot[0] += 1
            Xr, Fr = (st.X_H, st.F_H) if a == 0 else (st.X_M, st.F_M)
            Xd, Fd = (st.X_M, st.F_M) if a == 0 else (st.X_H, st.F_H)
            r = pick_tournament(rng_ex, Fr)
            d = pick_tournament(rng_ex, Fd)
            p = sel_probs(excl0=h0_att >= E)
            h = int(rng_sel.choice(4, p=p))
            x_r, f_r = Xr[r].copy(), float(Fr[r])
            if h == 0 and not variant.h0_eval:
                x_new, f_new, cost = Xd[d].copy(), float(Fd[d]), 0
                h0_att += 1
            else:
                x_new, _ = candidate(h, x_r, Xd[d], fr, rng_ex, cfg.p_x)
                x_new = clip(x_new, lb, ub)
                f_new, cost = obj.eval1(x_new), 1
                evaluated += 1
            y = int(f_new < f_r)
            delta = x_new - x_r
            z1, z3 = record_features(delta, fr)
            w = 1.0 / p[h]
            R_pi.append(ev.pi())
            don = 1 - a                                   # donor population
            if shadows:
                R_pis.append(shadows[don].pi())
                shadows[don].add(z1, z3, y, w)
            ell = ev.add(z1, z3, y, w)
            R_loss.append(ell)
            for k, v in (("t", it), ("fe", obj.fe), ("gen", h), ("y", y), ("don", don),
                         ("cost", cost), ("w", w)):
                R[k].append(v)
            if y:
                Xr[r], Fr[r] = x_new, f_new
                if a == 1:
                    st.X_M_old[r] = x_new; st.fit_old[r] = f_new
                    if f_new < st.bF_M:
                        st.bF_M, st.bP_M = f_new, x_new.copy()
                else:
                    if f_new < st.bF_H:
                        st.bF_H, st.bP_H = f_new, x_new.copy()
                st.global_from_bests()
        counts["h0_attempts"] += h0_att
        counts["exchange_fe"] += evaluated
        counts["records"] = ev.n
        return evaluated, h0_att

    def refresh(it: int):
        fr.refresh_basis(st.X_H, st.F_H, st.X_M, st.F_M, cfg.q_e, cfg.lam_LW)
        engines = [("main", ev)] + [(f"shadow{k}", e) for k, e in shadows.items()]
        for name, e in engines:
            z1, _, y, w, _ = e.window()
            pairs, groups, info = fit_linkage(z1, y, w, cfg)
            e.set_pairs(pairs)
            if name == "main":
                fr.set_linkage(pairs, groups)
                link_log.append((it, pairs.copy()))
        for pop in (0, 1):
            Z = native[pop].rows()
            if Z.shape[0] >= 2 * D:
                A_sum[pop] += _offdiag_corr(Z); A_cnt[pop] += 1; A_n[pop].append(Z.shape[0])

    it = 0
    while obj.remaining() >= 3 * N:
        fr.update_scale(st.X_H, st.X_M)
        if it % cfg.U == 0:
            refresh(it)
        prog = obj.fe / max_fe
        XH0, XM0 = st.X_H.copy(), st.X_M.copy()
        CF = st.phase1(obj, rng_nat, prog * T_nom, T_nom)
        native_capture(0, XH0, st.X_H); native_capture(1, XM0, st.X_M)
        if cfg.fad == "before":
            XM0 = st.X_M.copy(); st.fads_greedy(obj, rng_nat, CF); native_capture(1, XM0, st.X_M)
            st.global_from_bests()
        nev, h0a = exchange_slots(E, it)
        if cfg.fad == "after" and obj.remaining() >= N:
            XM0 = st.X_M.copy(); st.fads_greedy(obj, rng_nat, CF); native_capture(1, XM0, st.X_M)
            st.global_from_bests()
        ev.refit()
        for e in shadows.values():
            e.refit()
        counts["iters"] += 1
        if nev != E and obj.remaining() >= 3 * N:
            counts["iters_not_E"] += 1
        it_extra["pi"].append(ev.pi()); it_extra["h0att"].append(h0a); it_extra["nev"].append(nev)
        it_extra["npairs"].append(fr.pairs.shape[0])
        it_extra["piH"].append(shadows[0].pi() if shadows else np.full(4, np.nan))
        it_extra["piM"].append(shadows[1].pi() if shadows else np.full(4, np.nan))
        if rec is not None:
            rec.iteration(obj.fe, st.Pg, [st.X_H, st.X_M], [st.F_H, st.F_M])
        it += 1
    # leftover (< 3N FEs): exchange slots until the budget is used exactly
    if obj.remaining() > 0:
        exchange_slots(obj.remaining(), it)

    n = len(R["t"])
    arrays = {f"r_{k}": np.asarray(v, dtype=np.float64 if k == "w" else np.int64) for k, v in R.items()}
    arrays["r_pi"] = np.asarray(R_pi, dtype=np.float32).reshape(n, 4)
    arrays["r_loss"] = np.asarray(R_loss, dtype=np.float32).reshape(n, 4)
    if shadows:
        arrays["r_pis"] = np.asarray(R_pis, dtype=np.float32).reshape(n, 4)
    for k, v in it_extra.items():
        arrays[f"it_{k}"] = np.asarray(v, dtype=np.float32)
    for pop, nm in ((0, "H"), (1, "M")):
        arrays[f"A_{nm}"] = A_sum[pop] / max(1, A_cnt[pop])
        arrays[f"A_{nm}_n"] = np.asarray(A_n[pop], dtype=np.int64)
    rows = [np.column_stack([np.full(len(p), i), p]) for i, p in link_log if len(p)]
    arrays["link_rows"] = np.vstack(rows).astype(np.int64) if rows else np.zeros((0, 3), np.int64)
    arrays["link_k"] = np.asarray([[i, len(p)] for i, p in link_log], dtype=np.int64).reshape(-1, 2)
    summary = dict(counts)
    summary.update(fe=obj.fe, fe_audit_ok=bool(obj.fe == max_fe),
                   exact_E_ok=bool(counts["iters_not_E"] == 0),
                   burn_fe=burn_fe, best=float(st.Pg), A_cnt_H=A_cnt[0], A_cnt_M=A_cnt[1])
    return RunOutput(best=float(min(st.Pg, obj.best)), summary=summary, arrays=arrays)


def _uniform(hyps) -> np.ndarray:
    p = np.zeros(4)
    p[list(hyps)] = 1.0 / len(hyps)
    return p


__all__ = ["run_she", "spawn_streams", "RunOutput", "HYPS"]
