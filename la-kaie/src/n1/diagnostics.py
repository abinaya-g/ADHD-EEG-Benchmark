"""D3 diagnostic recorder (measurement only; spec unchanged).

Passive observer for ``run_n1(..., diag=D3Recorder(...))``. Every method
receives values that the production loop has already computed and stores
*copies*. It never draws random numbers, never evaluates the objective,
never writes to the optimizer state, and returns nothing that the loop uses.
With ``diag=None`` (the default) the loop does not call it at all.

Agent ID = population row index. The backbone (HybridState) updates every
row in place (greedy / memory acceptance write back to the same row), so the
row index is the implementation's own stable agent identity. In A0 there is no
exchange, so no row is ever replaced by another agent's solution.

Operator components that are drawn *inside* the frozen backbone operators
(HBA intensity I, flag F, dig/honey mode r, r3..r7; MPA RB, RL, R; FAD branch,
mask U and permutations) are NOT observable without changing the operators and
are therefore recorded as unavailable.
"""
from __future__ import annotations

import numpy as np

OP_CODE = {"HBA": 0, "MPA": 1, "FAD": 2}
POP_CODE = {"H": 0, "M": 1}

UNAVAILABLE_COMPONENTS = (
    "HBA intensity I_i, direction flag F_i, dig/honey mode r_i, random factors r3-r7 (drawn inside hba_move)",
    "MPA Brownian RB, Levy RL, uniform R matrices (drawn inside mpa_move)",
    "FAD branch choice, FAD mask U, FAD permutations and random factor (drawn inside fads_candidates)",
)


class D3Recorder:
    def __init__(self, N: int, D: int):
        self.N, self.D = N, D
        # per iteration
        self.it_t, self.it_fe, self.it_prog_t, self.it_CF = [], [], [], []
        self.it_xprey_H, self.it_elite_M, self.it_Pbest = [], [], []
        self.it_bF_H, self.it_bF_M, self.it_Pg = [], [], []
        self.it_scale = []
        self.T_ref = None
        # per attempt (all attempts): success masks
        self.att_t, self.att_op, self.att_y = [], [], []
        # per successful displacement row
        self.row_t, self.row_pop, self.row_op, self.row_agent = [], [], [], []
        self.row_raw, self.row_std, self.row_finite = [], [], []
        # per epoch
        self.ep_t, self.ep_fe = [], []
        self.ep_XH_pre, self.ep_XM_pre, self.ep_XH, self.ep_XM, self.ep_FH, self.ep_FM = [], [], [], [], [], []
        self.ep_stats = {f"{p}_{k}": [] for p in ("H", "M") for k in ("mean", "cov", "evals", "evecs")}

    # ------------------------------------------------ hooks (copies only)
    def iteration_start(self, t, fe, prog_t, T_ref, st):
        """Called after the plan step, before phase1. The vectors are exactly the
        ones phase1 hands to the operators: xprey = bP_H (HBA), elite = bP_M (MPA)."""
        self.T_ref = T_ref
        self.it_t.append(t); self.it_fe.append(fe); self.it_prog_t.append(float(prog_t))
        self.it_xprey_H.append(st.bP_H.copy()); self.it_elite_M.append(st.bP_M.copy())
        self.it_Pbest.append(st.Pbest.copy())
        self.it_bF_H.append(float(st.bF_H)); self.it_bF_M.append(float(st.bF_M)); self.it_Pg.append(float(st.Pg))

    def cf(self, CF):
        self.it_CF.append(float(CF))

    def scale(self, s_vec):
        self.it_scale.append(np.array(s_vec, copy=True))

    def native(self, t, pop, op, Xc, X0, y, Zs, fin):
        """Xc: evaluated candidates, X0: parents (same rows), y: strict success mask,
        Zs: the exact standardized rows production computed ((Xc - X0)/s)[y],
        fin: finiteness mask production applied before appending Zs[fin]."""
        y = np.asarray(y, bool)
        self.att_t.append(np.full(y.size, t, np.int32))
        self.att_op.append(np.full(y.size, OP_CODE[op], np.int8))
        self.att_y.append(y.copy())
        agents = np.flatnonzero(y)
        if agents.size == 0:
            return
        self.row_t.append(np.full(agents.size, t, np.int32))
        self.row_pop.append(np.full(agents.size, POP_CODE[pop], np.int8))
        self.row_op.append(np.full(agents.size, OP_CODE[op], np.int8))
        self.row_agent.append(agents.astype(np.int16))
        self.row_raw.append((Xc - X0)[y].copy())
        self.row_std.append(np.array(Zs, copy=True))
        self.row_finite.append(np.array(fin, bool, copy=True))

    def epoch(self, t, fe, st, XH_pre, XM_pre):
        """Called at every selection epoch after the native phases (same time as
        the scale vector). XH_pre / XM_pre are the parents of this iteration."""
        self.ep_t.append(t); self.ep_fe.append(fe)
        self.ep_XH_pre.append(np.array(XH_pre, copy=True)); self.ep_XM_pre.append(np.array(XM_pre, copy=True))
        self.ep_XH.append(st.X_H.copy()); self.ep_XM.append(st.X_M.copy())
        self.ep_FH.append(st.F_H.copy()); self.ep_FM.append(st.F_M.copy())
        for p, X in (("H", st.X_H), ("M", st.X_M)):
            mu = X.mean(axis=0)
            C = (X - mu).T @ (X - mu) / X.shape[0]          # MLE (ddof = 0), deterministic
            w, V = np.linalg.eigh((C + C.T) / 2.0)
            self.ep_stats[f"{p}_mean"].append(mu); self.ep_stats[f"{p}_cov"].append(C)
            self.ep_stats[f"{p}_evals"].append(w); self.ep_stats[f"{p}_evecs"].append(V)

    # ------------------------------------------------ export
    def arrays(self) -> dict:
        out = {
            "d3_it_t": np.asarray(self.it_t, np.int32), "d3_it_fe": np.asarray(self.it_fe, np.int64),
            "d3_it_prog_t": np.asarray(self.it_prog_t), "d3_it_CF": np.asarray(self.it_CF),
            "d3_it_xprey_H": np.asarray(self.it_xprey_H), "d3_it_elite_M": np.asarray(self.it_elite_M),
            "d3_it_Pbest": np.asarray(self.it_Pbest), "d3_it_bF_H": np.asarray(self.it_bF_H),
            "d3_it_bF_M": np.asarray(self.it_bF_M), "d3_it_Pg": np.asarray(self.it_Pg),
            "d3_it_scale": np.asarray(self.it_scale), "d3_T_ref": np.asarray(self.T_ref if self.T_ref else -1),
            "d3_att_t": np.concatenate(self.att_t) if self.att_t else np.zeros(0, np.int32),
            "d3_att_op": np.concatenate(self.att_op) if self.att_op else np.zeros(0, np.int8),
            "d3_att_y": np.concatenate(self.att_y) if self.att_y else np.zeros(0, bool),
            "d3_row_t": np.concatenate(self.row_t) if self.row_t else np.zeros(0, np.int32),
            "d3_row_pop": np.concatenate(self.row_pop) if self.row_pop else np.zeros(0, np.int8),
            "d3_row_op": np.concatenate(self.row_op) if self.row_op else np.zeros(0, np.int8),
            "d3_row_agent": np.concatenate(self.row_agent) if self.row_agent else np.zeros(0, np.int16),
            "d3_row_raw": np.vstack(self.row_raw) if self.row_raw else np.zeros((0, self.D)),
            "d3_row_std": np.vstack(self.row_std) if self.row_std else np.zeros((0, self.D)),
            "d3_row_finite": np.concatenate(self.row_finite) if self.row_finite else np.zeros(0, bool),
            "d3_ep_t": np.asarray(self.ep_t, np.int32), "d3_ep_fe": np.asarray(self.ep_fe, np.int64),
            "d3_ep_XH_pre": np.asarray(self.ep_XH_pre), "d3_ep_XM_pre": np.asarray(self.ep_XM_pre),
            "d3_ep_XH": np.asarray(self.ep_XH), "d3_ep_XM": np.asarray(self.ep_XM),
            "d3_ep_FH": np.asarray(self.ep_FH), "d3_ep_FM": np.asarray(self.ep_FM),
        }
        for k, v in self.ep_stats.items():
            out[f"d3_ep_{k}"] = np.asarray(v)
        return out
