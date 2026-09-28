"""MPHBS - port of MPHBS_main.m (Flores & Olivares, KBS 2026).

Every mediator step follows the authors' MATLAB code; see
reference/MPHBS_NOTES.md for equation mapping, missing-from-paper items and
paper-vs-code discrepancies (the code is followed). Ranks are kept 1-based where
they enter the proxy score. Differences from MATLAB are limited to the random
number generator (NumPy PCG64 vs MATLAB twister), so reproduction is
distributional, not bitwise.
"""
from __future__ import annotations

import math

import numpy as np

from ..core import EPS, Recorder, matlab_round
from .hybrid_common import HybridState

W_RANK, W_NOVEL = 1.00, 0.30
ALPHA_LR = 0.10
EPS_START, EPS_END = 1.0, 0.15


# ------------------------------------------------------------ action helpers
def greedy(Q, d):
    v1, v2 = Q[d, :, 0], Q[d, :, 1]
    a1, a2 = int(np.argmax(v1)), int(np.argmax(v2))
    return (0, a1) if v1[a1] >= v2[a2] else (1, a2)


def eps_decay(rng, Q, d, episode, total):
    decay = max(1.0, total / 4)
    eps_val = EPS_END + (EPS_START - EPS_END) * math.exp(-(episode - 1) / decay)
    if rng.random() < eps_val:
        return int(rng.integers(0, 2)), int(rng.integers(0, Q.shape[1]))
    return greedy(Q, d)


def second_best(Q, d):
    n = Q.shape[1]
    allv = np.concatenate([Q[d, :, 0], Q[d, :, 1]])
    idx = np.argsort(-allv, kind="stable")
    i2 = idx[1] if allv.size >= 2 else idx[0]
    return (0, int(i2)) if i2 < n else (1, int(i2 - n))


def closest_action(target, result, col):
    n = result.shape[0]
    allv = np.concatenate([result[:, col, 0], result[:, col, 1]])
    tgt = min(max(target, allv.min()), allv.max())
    i = int(np.argmin(np.abs(allv - tgt)))
    return (0, i) if i < n else (1, i - n)


def elite_plus_random(rng, F, n_sub):
    n_elite = n_sub // 2
    n_rand = n_sub - n_elite
    order = np.argsort(F, kind="stable")
    elite = order[:n_elite]
    rest = order[n_elite:]
    rnd = rest if rest.size < n_rand else rest[rng.permutation(rest.size)[:n_rand]]
    idx = np.concatenate([elite, rnd])
    idx = idx[np.argsort(F[idx], kind="stable")]       # rank rows inside subset
    return idx


# ------------------------------------------------------------ main
def run(obj, T, lb, ub, dim, N, rng, cfg, rec: Recorder | None = None):
    K, Ni = int(cfg["K"]), int(cfg["Ni"])
    rho, w_best, gamma = float(cfg["rho_sub"]), float(cfg["w_best"]), float(cfg["gamma"])
    fad_before, fad_after = bool(cfg["fad_before_p2"]), bool(cfg["fad_after_p2"])
    random_mirror = bool(cfg.get("random_actions", False))

    n_blocks = 5 if dim >= 10 else 1
    block = math.ceil(dim / n_blocks)
    n_sub = min(N, max(5, matlab_round(N * rho)))
    Q = np.zeros((dim, n_sub, 2))

    s = HybridState(obj, lb, ub, dim, N, rng)
    for t in range(T):
        prog_global = min(1.0, t / max(1, T - 1))
        CF = s.phase1(obj, rng, t, T)
        fad_rate = np.nan
        if fad_before:
            fad_rate = s.fads_greedy(obj, rng, CF)
        s.global_from_bests()

        # ---------------- PHASE 2
        idxH = elite_plus_random(rng, s.F_H, n_sub)
        idxM = elite_plus_random(rng, s.F_M, n_sub)
        XHs, FHs = s.X_H[idxH].copy(), s.F_H[idxH].copy()
        XMs, FMs = s.X_M[idxM].copy(), s.F_M[idxM].copy()
        result = np.stack([XHs, XMs], axis=2)            # (n_sub, D, 2)

        values = s.Pbest.copy()
        ch = np.zeros(dim, dtype=int); rows = np.zeros(dim, dtype=int)
        for c in range(dim):
            ch[c], rows[c] = closest_action(values[c], result, c)
        f_curr = obj.eval1(values)
        E = n_sub * Ni
        perm = rng.permutation(dim); ptr = 0
        n_acc = n_improve_gbest = 0
        for ep in range(1, E + 1):
            end = ptr + block
            if end > dim:
                part1 = perm[ptr:]
                perm = rng.permutation(dim)
                need = block - part1.size
                active = np.concatenate([part1, perm[:need]])
                ptr = need
            else:
                active = perm[ptr:end]
                ptr = end
                if ptr >= dim:
                    perm = rng.permutation(dim); ptr = 0

            best_score, best_k = -np.inf, 0
            cands = []
            for k in range(1, K + 1):
                v = values.copy(); chn = ch.copy(); rn = rows.copy()
                for c in active:
                    if random_mirror or k == 3:
                        ck, rk = int(rng.integers(0, 2)), int(rng.integers(0, n_sub))
                    elif k == 1:
                        ck, rk = greedy(Q, c)
                    elif k == 2:
                        ck, rk = eps_decay(rng, Q, c, ep, E)
                    elif k == 4:
                        ck, rk = second_best(Q, c)
                    else:
                        ep2 = min(E, ep + int(rng.integers(1, max(1, E // 10) + 1)))
                        ck, rk = eps_decay(rng, Q, c, ep2, E)
                    v[c] = result[rk, c, ck]; chn[c] = ck; rn[c] = rk
                score = (-W_RANK * np.mean(rn[active] + 1)
                         + W_NOVEL * np.mean(np.abs(v[active] - values[active]))
                         - w_best * np.linalg.norm(v[active] - s.Pbest[active]))
                cands.append((v, chn, rn))
                if score > best_score:
                    best_score, best_k = score, k - 1
            v_next, ch_next, rows_next = cands[best_k]

            val = obj.eval1(v_next)
            reward = 1.0 if val < f_curr else (0.0 if val == f_curr else -1.0)
            prog_local = (ep - 1) / max(1, E - 1)
            if val < f_curr:
                accepted = True
            else:
                d_rel = (val - f_curr) / (abs(f_curr) + EPS)
                p = 0.20 * (1 - prog_global) ** 2 * (0.35 + 0.65 * (1 - prog_local)) * math.exp(-4 * max(0.0, d_rel))
                accepted = rng.random() < min(max(p, 0.0), 0.25)

            if not random_mirror:
                for c in active:
                    pred = Q[c, rows[c], ch[c]]
                    nq = Q[c, rows_next[c], ch_next[c]] if accepted else pred
                    Q[c, rows[c], ch[c]] = pred + ALPHA_LR * (reward + gamma * nq - pred)

            if accepted:
                n_acc += 1
                values, f_curr, ch, rows = v_next, val, ch_next, rows_next
                if f_curr < FHs.max():
                    w = int(np.argmax(FHs)); XHs[w] = values; FHs[w] = f_curr
                if f_curr < FMs.max():
                    w = int(np.argmax(FMs)); XMs[w] = values; FMs[w] = f_curr
                result = np.stack([XHs, XMs], axis=2)

            if val < s.Pg:
                n_improve_gbest += 1
                s.Pg, s.Pbest = val, v_next.copy()
                th = idxH[int(rng.integers(0, idxH.size))]
                s.X_H[th] = s.Pbest; s.F_H[th] = s.Pg
                tm = idxM[int(rng.integers(0, idxM.size))]
                s.X_M[tm] = s.Pbest; s.F_M[tm] = s.Pg

        s.X_H[idxH] = XHs; s.F_H[idxH] = FHs
        s.X_M[idxM] = XMs; s.F_M[idxM] = FMs
        k = int(np.argmin(s.F_H)); s.bF_H, s.bP_H = float(s.F_H[k]), s.X_H[k].copy()
        k = int(np.argmin(s.F_M)); s.bF_M, s.bP_M = float(s.F_M[k]), s.X_M[k].copy()
        s.sync_global_keep()

        if fad_after:
            fad_rate = s.fads_greedy(obj, rng, CF)
            s.sync_global_keep()
        if rec is not None:
            rec.iteration(obj.fe, s.Pg, [s.X_H, s.X_M], [s.F_H, s.F_M],
                          hba_success=s.last_hba_success, mpa_success=s.last_mpa_success,
                          fad_success=fad_rate, mediator_accept_rate=n_acc / E,
                          mediator_gbest_improvements=n_improve_gbest)
    return {"best": s.Pg, "best_x": s.Pbest}
