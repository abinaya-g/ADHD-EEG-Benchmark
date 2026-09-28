"""LA-KAIE: Landscape-Aware Knowledge-Guided Adaptive Information Exchange.

HBA-MPA backbone (identical Phase 1 to MPHB/MPHBS) + an exchange phase in
which a landscape-conditioned contextual bandit decides, episode by episode,
WHETHER to exchange (A1 returns the budget to the backbone), WHICH source
population donates, and at WHICH granularity (single variables or estimated
interaction groups) information is transferred. Each exchange candidate costs
at most one true objective evaluation; the FE counter is checked before every
call so the budget can never be exceeded.

Full specification: experiments/LAKAIE_SPEC.md.
"""
from __future__ import annotations

import math
import time

import numpy as np

from ..core import Recorder, clip
from ..components import (ACTIONS, Archive, BanditController, LandscapeState,
                          estimate_interaction, fir_prior, group_variables,
                          interaction_consistency, interaction_strength)
from .hybrid_common import HybridState


class _ArchivedObjective:
    """Forwards to the counted objective and stores every evaluated point."""

    def __init__(self, obj, archive):
        self.obj, self.archive = obj, archive

    @property
    def fe(self):
        return self.obj.fe

    def __call__(self, X):
        f = self.obj(X)
        self.archive.add(X, f)
        return f

    def eval1(self, x):
        return float(self(x[None, :])[0])


def run(obj, lb, ub, dim, N, rng, cfg, rec: Recorder | None, max_fe: int, problem=None):
    ex, ic_cfg, st_cfg = cfg["exchange"], cfg["interaction"], cfg["state"]
    rw, ctl_cfg = cfg["reward"], cfg["controller"]
    E = int(ex["episodes_per_iteration"])
    width = ub - lb
    t_start = time.perf_counter()
    overhead = {"interaction": 0.0, "controller": 0.0, "state": 0.0}

    archive = Archive(int(ic_cfg["archive_size"]), dim)
    f = _ArchivedObjective(obj, archive)
    s = HybridState(f, lb, ub, dim, N, rng)
    state = LandscapeState(st_cfg, lb, ub)
    ctl = BanditController(ctl_cfg, rng)
    tau, gmax = float(ic_cfg["threshold"]), int(ic_cfg["max_group_size"])
    G = np.zeros((dim, dim))
    groups = [[i] for i in range(dim)]
    G_prior = None
    if ic_cfg["mode"] == "fir_prior_blend":
        if problem is None or not hasattr(problem, "stopband_quadratic_form"):
            raise ValueError("fir_prior_blend requires the FIR problem")
        G_prior = fir_prior(problem.stopband_quadratic_form())

    backbone_cost = 3 * N                     # HBA N + MPA N + FADs N
    T_nom = max(1.0, max_fe / (backbone_cost + E))
    it = 0
    pending_a1 = None                         # (state, div_before, best_before)
    trace = {k: [] for k in ["ep_fe", "ep_action", "ep_reward", "ep_success", "ep_random",
                             "ep_group_level", "ep_n_transfer", "ep_ic", "ep_iter"]}
    it_trace = {k: [] for k in ["it_fe", "it_S8", "it_ngroups", "it_meangroup", "it_epsilon",
                                "it_exchanges", "it_state"]}
    groups_snapshots = []

    def pooled():
        return np.concatenate([s.F_H, s.F_M])

    def refresh_bests():
        k = int(np.argmin(s.F_H)); s.bF_H, s.bP_H = float(s.F_H[k]), s.X_H[k].copy()
        k = int(np.argmin(s.F_M)); s.bF_M, s.bP_M = float(s.F_M[k]), s.X_M[k].copy()
        s.sync_global_keep()

    def current_groups(force_dim=False):
        if force_dim or ex["force_dimension_level"]:
            return [[i] for i in range(dim)], False
        return groups, True

    def choose_sets(unit_groups, p):
        """Each unit (group or variable) transferred w.p. p; at least one."""
        sel = [g for g in unit_groups if rng.random() < p]
        if not sel:
            sel = [unit_groups[int(rng.integers(len(unit_groups)))]]
        return np.array(sorted(i for g in sel for i in g), dtype=int)

    def tournament(F):
        k = int(ex["tournament_size"])
        cand = rng.integers(0, F.size, size=k)
        return int(cand[np.argmin(F[cand])])

    def pop(which):
        return (s.X_H, s.F_H) if which == 0 else (s.X_M, s.F_M)

    def replace(which, i, x, fx):
        X, F = pop(which)
        X[i] = x; F[i] = fx
        if which == 1:                        # keep MPA marine memory consistent
            s.X_M_old[i] = x; s.fit_old[i] = fx

    def beat(fv, Fp):
        return float(np.mean(Fp > fv))

    def granular(S8):
        use_groups = (S8 >= ex["granularity_threshold"]) and not ex["force_dimension_level"]
        return (groups if use_groups else [[i] for i in range(dim)]), use_groups

    # ------------------------------------------------------------ actions
    def do_action(a, S):
        """Returns dict(fe, success, fit_impr, div_change, ic, n_transfer, group_level)."""
        name = ACTIONS[a]
        Fp = pooled()
        div_before = state.diversity([s.X_H, s.X_M])
        gbest_before = s.Pg
        out = {"fe": 1, "success": False, "fit_impr": 0.0, "ic": 0.0, "n_transfer": 0,
               "group_level": False, "div_change": 0.0}
        if name == "A2":
            out["fe"] = 0
            src = 0 if s.bF_H < s.bF_M else 1
            dst = 1 - src
            Xs, Fs = pop(src); Xd, Fd = pop(dst)
            w = int(np.argmax(Fd)); b = int(np.argmin(Fs))
            if Fs[b] < Fd[w]:
                r_before = beat(Fd[w], Fp)
                replace(dst, w, Xs[b].copy(), float(Fs[b]))
                out["success"] = True
                out["fit_impr"] = max(0.0, beat(Fs[b], Fp) - r_before)
            out["n_transfer"] = dim
            out["ic"] = 1.0 if (G >= tau).any() else 0.0
            div_after = state.diversity([s.X_H, s.X_M])
            out["div_change"] = math.tanh(10 * (div_after - div_before) / max(div_before, 1e-300))
            return out

        S8 = S[7]
        if name in ("A3", "A4"):
            recv_pop = int(rng.integers(0, 2)); don_pop = 1 - recv_pop
            Xr, Fr = pop(recv_pop); Xd, Fd = pop(don_pop)
            ri, di = tournament(Fr), tournament(Fd)
            if name == "A3":
                units, gl = [[i] for i in range(dim)], False
            else:
                units, gl = current_groups()
            T_idx = choose_sets(units, ex["transfer_prob"])
            x = Xr[ri].copy(); x[T_idx] = Xd[di][T_idx]
        elif name == "A5":
            recv_pop = int(rng.integers(0, 2)); don_pop = 1 - recv_pop
            Xr, Fr = pop(recv_pop); Xd, Fd = pop(don_pop)
            ri = int(rng.integers(0, Fr.size))
            di = int(np.argmax(np.linalg.norm((Xd - Xr[ri]) / width, axis=1)))
            units, gl = granular(S8)
            T_idx = choose_sets(units, ex["transfer_prob"])
            x = Xr[ri].copy(); x[T_idx] = Xd[di][T_idx]
        elif name == "A6":
            recv_pop = 0 if s.bF_H < s.bF_M else 1; don_pop = 1 - recv_pop
            Xr, Fr = pop(recv_pop); Xd, Fd = pop(don_pop)
            ri = int(np.argmin(Fr))
            top = np.argsort(Fd, kind="stable")[:int(ex["exploit_donor_top"])]
            di = int(top[int(rng.integers(0, top.size))])
            units, gl = granular(S8)
            diff = np.array([np.abs(Xd[di][u] - Xr[ri][u]).sum() for u in units])
            if diff.sum() > 0:
                u = units[int(rng.choice(len(units), p=diff / diff.sum()))]
            else:
                u = units[int(rng.integers(len(units)))]
            T_idx = np.array(sorted(u), dtype=int)
            x = Xr[ri].copy(); x[T_idx] = Xd[di][T_idx]
        elif name == "A7":
            c_h, c_m = S[5], S[6]
            p_h = 0.5 if c_h + c_m <= 0 else c_h / (c_h + c_m)
            don_pop = 0 if rng.random() < p_h else 1; recv_pop = 1 - don_pop
            Xr, Fr = pop(recv_pop); Xd, Fd = pop(don_pop)
            ri, di = int(np.argmin(Fr)), int(np.argmin(Fd))
            units, gl = granular(S8)
            T_idx = choose_sets(units, ex["transfer_prob"])
            x = Xr[ri].copy(); x[T_idx] = Xd[di][T_idx]
        elif name == "A8":
            recv_pop = 0 if s.bF_H < s.bF_M else 1
            Xr, Fr = pop(recv_pop)
            ri = int(np.argmin(Fr))
            units, gl = granular(S8)
            u = np.array(sorted(units[int(rng.integers(len(units)))]), dtype=int)
            Xall = np.vstack([s.X_H, s.X_M]); Fall = pooled()
            ne = max(len(u) + 1, int(math.ceil(ex["elite_fraction"] * Fall.size)))
            el = Xall[np.argsort(Fall, kind="stable")[:ne]][:, u]
            C = np.atleast_2d(np.cov(el, rowvar=False)) + 1e-12 * np.eye(len(u)) * (width[u] ** 2)
            L = np.linalg.cholesky(C)
            x = Xr[ri].copy()
            x[u] = x[u] + L @ rng.standard_normal(len(u))
            T_idx = u
        else:
            raise ValueError(name)

        x = clip(x, lb, ub)
        fx = f.eval1(x)
        out["n_transfer"] = int(T_idx.size)
        out["group_level"] = bool(gl)
        out["ic"] = interaction_consistency(G, tau, T_idx)
        if fx < Fr[ri]:
            r_before = beat(Fr[ri], Fp)
            replace(recv_pop, ri, x, fx)
            out["success"] = True
            out["fit_impr"] = 1.0 if fx < gbest_before else max(0.0, beat(fx, Fp) - r_before)
            refresh_bests()
            div_after = state.diversity([s.X_H, s.X_M])
            out["div_change"] = math.tanh(10 * (div_after - div_before) / max(div_before, 1e-300))
        return out

    def reward(o, S):
        if rw["mode"] == "fitness_only":
            return o["fit_impr"]
        stag_pen = S[2] * (1.0 - float(o["success"]))     # expected value for A1
        return (rw["w1_fitness"] * o["fit_impr"] + rw["w2_diversity"] * o["div_change"]
                + rw["w3_success"] * float(o["success"]) + rw["w4_interaction"] * o["ic"]
                - rw["w5_cost"] * o["fe"] - rw["w6_stagnation"] * stag_pen)

    def update_interaction():
        nonlocal G, groups
        t0 = time.perf_counter()
        if ic_cfg["mode"] == "random_placebo":
            R = rng.random((dim, dim)); Gn = np.triu(R, 1); Gn = Gn + Gn.T
        else:
            Xa, Fa = archive.data()
            Gn = estimate_interaction(Xa, Fa, s.Pbest, width, ic_cfg["sample_factor"], ic_cfg["ridge"],
                                      ic_cfg["t_crit"])
            if Gn is None:
                overhead["interaction"] += time.perf_counter() - t0
                return
            if G_prior is not None:
                Gn = 0.5 * Gn + 0.5 * G_prior
        e = ic_cfg["ema"] if ic_cfg["mode"] != "random_placebo" else 1.0
        G = (1 - e) * G + e * Gn
        groups = group_variables(G, tau, gmax)
        state.s8 = interaction_strength(G)
        overhead["interaction"] += time.perf_counter() - t0

    # ------------------------------------------------------------ main loop
    def exchange_phase(allow_a1: bool):
        nonlocal pending_a1
        n_done = 0
        a2_used = False
        for _ in range(E):
            if f.obj.remaining() < 1:
                break
            t0 = time.perf_counter()
            S = state.vector([s.X_H, s.X_M], pooled(), s.Pg, f.fe / max_fe)
            overhead["state"] += time.perf_counter() - t0
            mask = np.ones(8, bool)
            mask[0] = allow_a1
            mask[1] = not a2_used
            t0 = time.perf_counter()
            a = ctl.select(S, mask, f.fe / max_fe)
            overhead["controller"] += time.perf_counter() - t0
            was_random = ctl.last_was_random
            if a == 0:
                pending_a1 = (S, state.diversity([s.X_H, s.X_M]), s.Pg)
                trace["ep_fe"].append(f.fe); trace["ep_action"].append(0); trace["ep_reward"].append(np.nan)
                trace["ep_success"].append(np.nan); trace["ep_random"].append(was_random)
                trace["ep_group_level"].append(False); trace["ep_n_transfer"].append(0)
                trace["ep_ic"].append(0.0); trace["ep_iter"].append(it)
                break
            if a == 1:
                a2_used = True
            o = do_action(a, S)
            r = reward(o, S)
            state.exchange_outcome(o["success"])
            t0 = time.perf_counter()
            ctl.update(S, a, r)
            overhead["controller"] += time.perf_counter() - t0
            n_done += 1
            trace["ep_fe"].append(f.fe); trace["ep_action"].append(a); trace["ep_reward"].append(r)
            trace["ep_success"].append(float(o["success"])); trace["ep_random"].append(was_random)
            trace["ep_group_level"].append(o["group_level"]); trace["ep_n_transfer"].append(o["n_transfer"])
            trace["ep_ic"].append(o["ic"]); trace["ep_iter"].append(it)
        return n_done

    while f.obj.remaining() >= backbone_cost:
        prog = f.fe / max_fe
        gbest_before = s.Pg
        div_before = state.diversity([s.X_H, s.X_M])
        CF = s.phase1(f, rng, prog * T_nom, T_nom)
        fad_rate = s.fads_greedy(f, rng, CF)
        refresh_bests()
        a1_mpa_rate = 0.5 * (s.last_mpa_success + fad_rate)
        state.end_iteration(s.Pg, pooled(), s.last_hba_success, a1_mpa_rate,
                            0.5 * (s.last_step_hba + s.last_step_mpa))
        if pending_a1 is not None:            # delayed reward for "no exchange"
            # Per-FE comparability: the budget released by A1 is spent by the
            # backbone, so A1 is credited with the backbone's per-evaluation
            # productivity (fraction of backbone offspring that improved their
            # parent) as both improvement and expected success, and pays the
            # same unit cost as one exchange evaluation.
            S_prev, dv_prev, gb_prev = pending_a1
            rate = (s.last_hba_success + a1_mpa_rate) / 2.0
            succ = s.Pg < gb_prev
            dv_now = state.diversity([s.X_H, s.X_M])
            o = {"fe": 1, "success": rate, "fit_impr": rate,
                 "div_change": math.tanh(10 * (dv_now - dv_prev) / max(dv_prev, 1e-300)),
                 "ic": 0.0}
            r = reward(o, S_prev)
            ctl.update(S_prev, 0, r)
            # the A1 decision is the most recent traced episode (nothing is traced in between)
            trace["ep_reward"][-1] = r
            trace["ep_success"][-1] = float(succ)
            pending_a1 = None
        if it % int(ic_cfg["update_every"]) == 0:
            update_interaction()
            if it % (10 * int(ic_cfg["update_every"])) == 0:
                groups_snapshots.append((f.fe, [list(g) for g in groups]))
        n_ex = exchange_phase(allow_a1=True)
        it += 1
        it_trace["it_fe"].append(f.fe); it_trace["it_S8"].append(state.s8)
        it_trace["it_ngroups"].append(len(groups))
        it_trace["it_meangroup"].append(dim / len(groups))
        it_trace["it_epsilon"].append(ctl.epsilon(f.fe / max_fe))
        it_trace["it_exchanges"].append(n_ex)
        it_trace["it_state"].append(state.vector([s.X_H, s.X_M], pooled(), s.Pg, f.fe / max_fe))
        if rec is not None:
            rec.iteration(f.fe, s.Pg, [s.X_H, s.X_M], [s.F_H, s.F_M],
                          hba_success=s.last_hba_success, mpa_success=s.last_mpa_success,
                          fad_success=fad_rate, exchanges=n_ex)
    # leftover budget (< one backbone iteration): exchange episodes only, A1 disabled
    while f.obj.remaining() >= 1:
        if exchange_phase(allow_a1=False) == 0:
            break

    runtime = time.perf_counter() - t_start
    acts = np.asarray(trace["ep_action"], dtype=int)
    succ = np.asarray(trace["ep_success"], dtype=float)
    counts = np.bincount(acts, minlength=8) if acts.size else np.zeros(8, int)
    probs = counts / max(1, counts.sum())
    entropy = float(-(probs[probs > 0] * np.log(probs[probs > 0])).sum() / math.log(8))
    expl = counts[4]; exploit = counts[5] + counts[7]
    ex_mask = acts > 0
    gl = np.asarray(trace["ep_group_level"], dtype=bool)
    icv = np.asarray(trace["ep_ic"], dtype=float)
    summary = {
        "iterations": it,
        "exchange_episodes": int(ex_mask.sum()),
        "exchange_fe": int(np.sum([1 for a in acts if a >= 2])),
        "exchange_frequency": float(ex_mask.sum() / max(1, it)),
        "successful_exchange_rate": float(np.nanmean(succ[ex_mask])) if ex_mask.any() else 0.0,
        "action_counts": {ACTIONS[i]: int(counts[i]) for i in range(8)},
        "action_success": {ACTIONS[i]: (float(np.nanmean(succ[acts == i])) if (acts == i).any() and i > 0 else None)
                           for i in range(8)},
        "controller_entropy": entropy,
        "exploration_exploitation_ratio": float(expl / max(1, expl + exploit)),
        "random_selection_fraction": float(np.mean(trace["ep_random"])) if trace["ep_random"] else 0.0,
        "group_level_fraction": float(gl[ex_mask].mean()) if ex_mask.any() else 0.0,
        "interaction_consistency_group": float(icv[ex_mask & gl].mean()) if (ex_mask & gl).any() else None,
        "interaction_consistency_dim": float(icv[ex_mask & ~gl].mean()) if (ex_mask & ~gl).any() else None,
        "final_S8": state.s8, "final_n_groups": len(groups),
        "final_groups": [list(map(int, g)) for g in groups],
        "final_G_upper": G[np.triu_indices(dim, 1)].round(4).tolist(),
        "overhead_sec": overhead, "runtime_sec": runtime,
        "overhead_fraction": float(sum(overhead.values()) / max(runtime, 1e-12)),
        "groups_snapshots": [(int(fe), g) for fe, g in groups_snapshots],
    }
    traces = {k: np.asarray(v) for k, v in trace.items()}
    traces.update({k: np.asarray(v) for k, v in it_trace.items()})
    return {"best": s.Pg, "best_x": s.Pbest, "T": it, "summary": summary, "traces": traces}
