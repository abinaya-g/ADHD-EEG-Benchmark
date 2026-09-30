"""N1 iteration loop (spec §1.1) on the unchanged validated backbone.

Per iteration t:
 0 plan      e_t from the rule state of the previous iteration; stop if
             FE_remaining < 3N + e_t
 1 native    HybridState.phase1  (HBA N FE, MPA N FE)
 2 native    HybridState.fads_greedy (N FE)
 3           HybridState.global_from_bests
 4/5 record  scale s; native attempts -> NA; native successes -> Z_H / Z_M;
             refit native control; every U iterations: all selectors (0 FE)
 6 exchange  e_t candidates: map -> predict p0 (logged) -> evaluate -> y ->
             greedy replacement -> SPRT-style update
 7           sync_global_keep; Recorder
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from lakaie.algorithms.hybrid_common import HybridState
from lakaie.core import Recorder, clip

from .config import SELECTORS, Arm, N1Config
from .mapping import map_candidate, pick_receiver, ranks
from .native_control import NativeControl
from .records import (ColumnLog, NativeAttemptBuffer, RecordingObjective, StructuralWindow)
from .selectors import Hysteresis, basis_for, evaluate_selector
from .sprt import ACTIVE, SPRTStyleRule, p1_from_p0


class HiddenEvaluation(RuntimeError):
    """The FE counter changed during a zero-FE section (spec §5 rule 2)."""


class LogOrderViolation(RuntimeError):
    """p0 was not logged before the candidate's evaluation (spec §4.8)."""


def make_state(obj, lb, ub, dim, N, rng_init, init_boxes=None) -> HybridState:
    """Initial HybridState. Without ``init_boxes`` this *is* HybridState.__init__
    (bit-identical to A4 = MPHBS for the same seed, T-INIT). ``init_boxes``
    (S7 only) replicates __init__ with per-population boxes, same draw order."""
    if init_boxes is None:
        return HybridState(obj, lb, ub, dim, N, rng_init)
    s = HybridState.__new__(HybridState)
    s.lb, s.ub, s.dim, s.N = lb, ub, dim, N
    loM, hiM = init_boxes["M"]
    loH, hiH = init_boxes["H"]
    s.X_M = clip(rng_init.random((N, dim)) * (hiM - loM) + loM, lb, ub)
    s.X_H = clip(rng_init.random((N, dim)) * (hiH - loH) + loH, lb, ub)
    s.F_M = obj(s.X_M)
    s.F_H = obj(s.X_H)
    k = int(np.argmin(s.F_M)); s.bF_M, s.bP_M = float(s.F_M[k]), s.X_M[k].copy()
    k = int(np.argmin(s.F_H)); s.bF_H, s.bP_H = float(s.F_H[k]), s.X_H[k].copy()
    s.fit_old, s.X_M_old = s.F_M.copy(), s.X_M.copy()
    s.global_from_bests()
    s.last_hba_success = s.last_mpa_success = 0.0
    s.last_step_hba = s.last_step_mpa = 0.0
    return s


def spawn_streams(seed: int):
    """rng_init = default_rng(seed) (identical to MPHBS); the other streams are
    spawned from an independent SeedSequence([seed, 1]) (spec §15.1)."""
    rng_init = np.random.default_rng(seed)
    native, exchange, placebo = [np.random.default_rng(s) for s in np.random.SeedSequence([seed, 1]).spawn(3)]
    return rng_init, native, exchange, placebo


@dataclass
class RunOutput:
    best: float
    summary: dict
    arrays: dict = field(default_factory=dict)


class SuccessRateControl:
    """A10: neutral success-rate representation-selection control (probability
    matching over HI/HB/HR on exchange success; floor, learning rate)."""

    def __init__(self, floor: float, lr: float):
        self.floor, self.lr = floor, lr
        self.q = np.zeros(3)
        self.p = np.full(3, 1.0 / 3.0)

    def choose(self, rng) -> str:
        return ("HI", "HB", "HR")[int(rng.choice(3, p=self.p))]

    def update(self, k: str, y: int):
        i = ("HI", "HB", "HR").index(k)
        self.q[i] = (1 - self.lr) * self.q[i] + self.lr * y
        tot = self.q.sum()
        self.p = np.full(3, 1.0 / 3.0) if tot <= 0 else self.floor + (1 - 3 * self.floor) * self.q / tot


def run_n1(problem, arm: Arm, cfg: N1Config, seed: int, obj, max_fe: int,
           rec: Recorder | None = None, diag=None) -> RunOutput:
    """One N1 run. ``obj`` is the run's single CountedObjective (budget max_fe).

    ``diag``: optional passive observer (n1.diagnostics.D3Recorder). It only
    receives copies of values already computed here; with diag=None (default)
    it is never called. It must not draw RNG, evaluate, or modify state."""
    N, D = cfg.N, problem.dim
    lb, ub = problem.lb, problem.ub
    W = cfg.W(D)
    rng_init, rng_native, rng_exchange, rng_placebo = spawn_streams(seed)
    proxy = RecordingObjective(obj)
    proxy.begin("init")
    st = make_state(proxy, lb, ub, D, N, rng_init, getattr(problem, "init_boxes", None))
    proxy.take()

    T_ref = max(1, math.floor(max_fe / (3 * N + cfg.E_x)))
    NA = NativeAttemptBuffer(cfg.W_nat())
    ZH = StructuralWindow(D, W, keep_stream=arm.log_windows)
    ZM = StructuralWindow(D, W, keep_stream=arm.log_windows)
    control = NativeControl(cfg)
    rule = SPRTStyleRule(cfg)
    sel_names = SELECTORS if arm.shadows else ((arm.driving,) if arm.driving else ())
    if arm.exchange == "fixed" and "JOINT" not in sel_names:
        sel_names = tuple(sel_names) + ("JOINT",)
    hyst = {s: Hysteresis(cfg.kappa) for s in sel_names}
    latest = {s: None for s in sel_names}
    last_nondeg_joint_partition = None
    a10 = SuccessRateControl(cfg.a10_floor, cfg.a10_lr) if arm.exchange == "success_rate" else None

    xlog, slog, nalog = ColumnLog(), ColumnLog(), ColumnLog()
    epoch_ptr = ColumnLog()
    counts = dict(n_mpa_ties=0, n_nonfinite_disp=0, n_hidden_checks=0, n_out_of_support=0,
                  n_fallback=0, n_iter=0, n_epochs=0, n_separation=0, n_numerical_failure=0)
    width = ub - lb
    t = 0
    s_vec = np.maximum(np.concatenate([st.X_H, st.X_M]).std(axis=0), cfg.s_floor * width)

    while True:
        # ---------------------------------------------------------- 0 plan
        if arm.exchange == "none":
            e_t = 0
        elif arm.always_active():
            e_t = cfg.E_x
        else:
            e_t = cfg.E_x if rule.state == ACTIVE else cfg.r_probe
        if obj.remaining() < 3 * N + e_t:
            break
        prog_t = T_ref * obj.fe / max_fe
        if diag is not None:
            diag.iteration_start(t, obj.fe, prog_t, T_ref, st)

        # ------------------------------------------------------- 1 native
        XH0, FH0 = st.X_H.copy(), st.F_H.copy()
        XM0, FM0 = st.X_M_old.copy(), st.fit_old.copy()      # memory = MPA parents
        proxy.begin("phase1")
        CF = st.phase1(proxy, rng_native, prog_t, T_ref)
        if diag is not None:
            diag.cf(CF)
        b = proxy.take()
        if len(b) != 2 or b[0][0].shape[0] != N or b[1][0].shape[0] != N:
            raise RuntimeError("unexpected phase1 evaluation pattern")
        (XH_c, fH_c), (XM_c, fM_c) = b
        # ------------------------------------------------------- 2 FAD
        XF0, FF0 = st.X_M.copy(), st.F_M.copy()
        proxy.begin("fad")
        st.fads_greedy(proxy, rng_native, CF)
        (XF_c, fF_c), = proxy.take()
        # ------------------------------------------------------- 3
        st.global_from_bests()

        # ------------------------------------------ 4/5 record (0 FE, no native RNG)
        fe_mark = obj.fe
        s_vec = np.maximum(np.concatenate([st.X_H, st.X_M]).std(axis=0), cfg.s_floor * width)
        yH = fH_c < FH0
        yM = fM_c < FM0                                   # strict (amendment A-3)
        counts["n_mpa_ties"] += int(np.sum(fM_c == FM0))
        yF = fF_c < FF0
        if diag is not None:
            diag.scale(s_vec)
        for pop, op, Xc, X0, y in (("H", "HBA", XH_c, XH0, yH), ("M", "MPA", XM_c, XM0, yM),
                                   ("M", "FAD", XF_c, XF0, yF)):
            U_ = (Xc - X0) / s_vec
            l = np.log(np.linalg.norm(U_, axis=1) + cfg.eps_l)
            NA.extend(t, pop, op, l, y.astype(np.int8))
            nalog.add(t=t, op=op, n=int(y.size), n_succ=int(y.sum()), mean_l=float(np.mean(l)))
            if op == "FAD" and not cfg.include_fad:
                continue
            Zs = U_[y]
            fin = np.all(np.isfinite(Zs), axis=1)
            counts["n_nonfinite_disp"] += int(np.sum(~fin))
            (ZH if pop == "H" else ZM).append_native(Zs[fin], op, t)
            if diag is not None:
                diag.native(t, pop, op, Xc, X0, y, Zs, fin)
        l_all, pop_all, y_all = NA.arrays()
        control.fit(l_all, pop_all, y_all)
        counts["n_separation"] += int(control.separation)

        if diag is not None and t % cfg.U == 0:
            diag.epoch(t, obj.fe, st, XH0, XM0)
        if t % cfg.U == 0 and sel_names:
            counts["n_epochs"] += 1
            epoch_ptr.add(t=t, fe=obj.fe, nH=int(ZH.n_total), nM=int(ZM.n_total))
            for sname in sel_names:
                res = evaluate_selector(sname, ZH, ZM, cfg, rng_placebo if sname == "DECOUPLED" else None)
                counts["n_numerical_failure"] += int(res.numerical_failure)
                k_cur, switched, forced = hyst[sname].update(res)
                latest[sname] = res
                if sname == "JOINT" and res.partition is not None and res.degenerate is None:
                    last_nondeg_joint_partition = res.partition
                slog.add(t=t, fe=obj.fe, selector=sname, evidence=bool(res.evidence),
                         nH=int(res.n.get("H", 0)), nM=int(res.n.get("M", 0)), nP=int(res.n.get("P", 0)),
                         avail="".join(res.available),
                         bic_HI=res.bic.get("HI"), bic_HB=res.bic.get("HB"), bic_HR=res.bic.get("HR"),
                         ll_HI=res.ll.get("HI"), ll_HB=res.ll.get("HB"), ll_HR=res.ll.get("HR"),
                         q_HI=res.q.get("HI"), q_HB=res.q.get("HB"), q_HR=res.q.get("HR"),
                         pll_HI=res.pll.get("HI"), pll_HB=res.pll.get("HB"), pll_HR=res.pll.get("HR"),
                         k_hat=res.k_hat or "", k_cur=k_cur, switched=bool(switched), forced=bool(forced),
                         margin=res.margin, degenerate=res.degenerate or "",
                         n_blocks=len(res.partition) if res.partition is not None else 0,
                         partition=";".join(",".join(str(int(i)) for i in blk) for blk in res.partition)
                         if res.partition is not None else "",
                         floored_HR=bool(res.floored.get("HR", False)), digest=res.digest,
                         numerical_failure=bool(res.numerical_failure))
        if obj.fe != fe_mark:
            raise HiddenEvaluation(f"FE changed from {fe_mark} to {obj.fe} in a zero-FE section")
        counts["n_hidden_checks"] += 1

        # ------------------------------------------------------- 6 exchange
        if e_t > 0:
            proxy.begin("exchange" if (arm.always_active() or rule.state == ACTIVE) else "probe")
            orders = {"H": ranks(st.F_H), "M": ranks(st.F_M)}
            pos = {p: np.argsort(o, kind="stable") for p, o in orders.items()}   # agent -> rank
            k_drv = None
            if arm.exchange == "selector":
                k_drv = hyst[arm.driving].k_cur
            elif arm.exchange == "fixed":
                k_drv = arm.fixed_k
            for e in range(1, e_t + 1):
                pr = "M" if e % 2 == 1 else "H"
                pd_ = "H" if pr == "M" else "M"
                Xr, Fr = (st.X_M, st.F_M) if pr == "M" else (st.X_H, st.F_H)
                Xd = st.X_H if pr == "M" else st.X_M
                r = pick_receiver(rng_exchange, Fr)
                d_idx = int(orders[pd_][pos[pr][r]])
                x_r, f_r, x_d = Xr[r].copy(), float(Fr[r]), Xd[d_idx].copy()
                if arm.harness == "positive":
                    x_d = problem.o.copy()
                k = a10.choose(rng_exchange) if a10 is not None else k_drv
                res_d = latest.get(arm.driving if arm.exchange == "selector" else "JOINT")
                part = U = None
                fallback = False
                if k == "HB":
                    if arm.exchange == "selector":       # the driving selector's own partition
                        part = (res_d.partition if res_d is not None and res_d.partition is not None
                                and res_d.degenerate is None else None)
                    else:                                # A2 / A10: latest non-degenerate JOINT partition (§8.4)
                        part = last_nondeg_joint_partition
                    if part is None:
                        fallback = True
                elif k == "HR":
                    U = basis_for(res_d, pr)
                    if U is None:
                        fallback = True
                k_used = "HI" if fallback else k
                counts["n_fallback"] += int(fallback)
                if arm.harness == "null":
                    # native-equivalent candidate: a displacement from this iteration's
                    # native attempts of the receiver population, applied to the receiver
                    src = (XH_c - XH0) if pr == "H" else (XM_c - XM0)
                    x_new = clip(x_r + src[int(rng_exchange.integers(N))], lb, ub)
                    mask_n = 0
                else:
                    x_new, mask, _ = map_candidate(k_used, x_r, x_d, s_vec, rng_exchange, cfg.p_x, part, U)
                    x_new = clip(x_new, lb, ub)
                    mask_n = int(np.sum(mask))
                l_e = float(np.log(np.linalg.norm((x_new - x_r) / s_vec) + cfg.eps_l))
                p0, l_c, oos = control.predict(l_e, pr == "M")
                counts["n_out_of_support"] += int(oos)
                p1 = p1_from_p0(p0, cfg.delta)
                fe_logged = obj.fe                        # p0 and p1 are fixed here
                f_new = proxy.eval1(x_new)
                if not obj.fe > fe_logged:
                    raise LogOrderViolation("evaluation did not follow the p0 log")
                y = int(f_new < f_r)
                if y:
                    Xr[r], Fr[r] = x_new, f_new
                    if pr == "M":
                        st.X_M_old[r], st.fit_old[r] = x_new, f_new
                        if f_new < st.bF_M:
                            st.bF_M, st.bP_M = f_new, x_new.copy()
                    elif f_new < st.bF_H:
                        st.bF_H, st.bP_H = f_new, x_new.copy()
                if a10 is not None:
                    a10.update(k, y)
                decision = None
                if arm.exchange == "selector":           # SPRT-style rule (shadow for A8)
                    decision = rule.update(p0, y, t, obj.fe)
                xlog.add(t=t, e=e, fe_p0=fe_logged, fe_eval=obj.fe, tag=proxy.tag, pop_r=pr,
                         idx_r=r, idx_d=d_idx, k=k or "", k_used=k_used, fallback=fallback,
                         n_units_sel=mask_n, l_e=l_e, l_clip=l_c, oos=oos, p0=p0, p1=p1,
                         f_r=f_r, f_new=f_new, y=y, llr=rule.llr, n_rule=rule.n,
                         state=rule.state, decision=decision or "")
            proxy.take()
        # ------------------------------------------------------- 7
        st.sync_global_keep()
        if rec is not None:
            rec.iteration(obj.fe, obj.best, [st.X_H, st.X_M], [st.F_H, st.F_M],
                          hba_success=float(yH.mean()), mpa_success=float(yM.mean()),
                          fad_success=float(yF.mean()), e_t=e_t,
                          state=1.0 if rule.state == ACTIVE else 0.0,
                          k_drv=float({"HI": 0, "HB": 1, "HR": 2}.get(
                              hyst[arm.driving].k_cur if arm.exchange == "selector" else "", -1)))
        t += 1
    counts["n_iter"] = t

    fe_tags = dict(proxy.fe_by_tag)
    fe_sum = sum(fe_tags.values())
    summary = dict(fe_by_tag=fe_tags, fe_audit_ok=bool(fe_sum == obj.fe), fe_used=obj.fe,
                   fe_unused=max_fe - obj.fe, fe_unused_ok=bool(max_fe - obj.fe < 3 * N + cfg.E_x),
                   T_ref=T_ref, W=W, final_state=rule.state,
                   n_window_rows_H=int(ZH.n_total), n_window_rows_M=int(ZM.n_total), n_decisions=len(rule.decisions),
                   decisions=rule.decisions, **counts)
    arrays = {}
    arrays.update(xlog.as_arrays("x_"))
    arrays.update(slog.as_arrays("s_"))
    arrays.update(nalog.as_arrays("na_"))
    arrays.update(epoch_ptr.as_arrays("ep_"))
    if arm.log_windows:
        for name, Zw in (("H", ZH), ("M", ZM)):
            R, O, T_ = Zw.stream()
            arrays[f"stream{name}_rows"], arrays[f"stream{name}_ops"], arrays[f"stream{name}_t"] = R, O, T_
    return RunOutput(best=float(obj.best), summary=summary, arrays=arrays)
