"""D3 offline causal-geometry diagnostic (EXPLORATORY, offline, zero-FE).

Implements results/analysis/n1_d3/analysis_config.json exactly (frozen and
committed before this script was run). Production N1 code is imported
read-only. Outputs: summary.json, metrics_*.csv, figures/*.png.

  python results/analysis/n1_d3/d3_analysis.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from n1.config import N1Config  # noqa: E402
from n1.selectors import evaluate_sources  # noqa: E402
from n1.structure import fisher_stat, partition_from_stat  # noqa: E402

CFG = N1Config()
RAW = ROOT / "results" / "raw"
D3 = RAW / "n1_d3_diagnostic" / "A0"
SMOKE = RAW / "n1_smoke" / "A0"
NBINS = 5
OPS = {0: "HBA", 1: "MPA", 2: "FAD"}


# ------------------------------------------------------------------ helpers
def corr(X):
    X = np.asarray(X, float)
    C = X - X.mean(axis=0)
    sd = np.sqrt((C * C).sum(axis=0))
    ok = sd > 0
    Cn = np.zeros_like(C)
    Cn[:, ok] = C[:, ok] / sd[ok]
    R = Cn.T @ Cn
    R[~ok, :] = 0.0; R[:, ~ok] = 0.0
    np.fill_diagonal(R, 1.0)
    return R


def cov2corr(S):
    d = np.sqrt(np.clip(np.diag(S), 1e-300, None))
    R = S / np.outer(d, d)
    np.fill_diagonal(R, 1.0)
    return R


def cov(X):
    X = np.asarray(X, float)
    C = X - X.mean(axis=0)
    return C.T @ C / X.shape[0]


def rmse_off(A, B):
    D = A.shape[0]
    return float(np.linalg.norm(A - B) / np.sqrt(D * (D - 1)))


def cs_off(A, B):
    m = ~np.eye(A.shape[0], dtype=bool)
    a, b = A[m], B[m]
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / den) if den > 0 else np.nan


def topk(S, k):
    w, V = np.linalg.eigh((S + S.T) / 2)
    o = np.argsort(w)[::-1]
    return V[:, o[:k]], w[o]


def subsim(SA, SB, k):
    UA, _ = topk(SA, k); UB, _ = topk(SB, k)
    return float(np.linalg.norm(UA.T @ UB) ** 2 / k)


def share_in(S_dx, S_basis, k):
    U, _ = topk(S_basis, k)
    return float(np.trace(U.T @ S_dx @ U) / np.trace(S_dx))


def inv_sqrt(C):
    w, V = np.linalg.eigh((C + C.T) / 2)
    w = np.maximum(w, 1e-12 * np.trace(C) / C.shape[0])
    return (V / np.sqrt(w)) @ V.T


def ari(a, b):
    n = len(a); ct = Counter(zip(a, b)); sa, sb = Counter(a), Counter(b)
    idx = sum(comb(v, 2) for v in ct.values()); ea = sum(comb(v, 2) for v in sa.values())
    eb = sum(comb(v, 2) for v in sb.values()); exp = ea * eb / comb(n, 2); mx = (ea + eb) / 2
    return 1.0 if mx == exp else (idx - exp) / (mx - exp)


def part_labels(part, D):
    lab = np.zeros(D, int)
    for i, b in enumerate(part):
        lab[np.asarray(b)] = i
    return lab.tolist()


def landscape_ref(z, fid, D):
    c = z["ref_c"]
    if fid == 1:
        H = np.diag(c)
    elif fid == 2:
        M, perm = z["ref_M"], z["ref_perm"]
        Hp = M.T @ np.diag(c) @ M
        H = np.zeros((D, D)); H[np.ix_(perm, perm)] = Hp
    else:
        Q = z["ref_Q"]; H = Q.T @ np.diag(c) @ Q
    S = np.linalg.inv(H)
    return (S + S.T) / 2


def within_block_share(R, lab):
    D = R.shape[0]; m = ~np.eye(D, dtype=bool); same = (lab[:, None] == lab[None, :]) & m
    tot = (R[m] ** 2).sum()
    return float((R[same] ** 2).sum() / tot) if tot > 0 else np.nan, float(same.sum() / m.sum())


# ------------------------------------------------------------------ integrity (section 3)
def integrity():
    ver = json.loads((RAW / "n1_d3_diagnostic" / "trajectory_equivalence.json").read_text())
    out = {"runs_found": 0, "seed_match": 0, "prod_identical": 0, "fe_match": 0, "nan_free": 0,
           "std_eq_raw_over_scale": 0, "agent_identity_hba_max_err": 0.0, "agent_identity_mpa_violations": 0,
           "agent_identity_mpa_checked": 0, "xprey_member": 0, "elite_member": 0, "epochs_checked": 0,
           "equivalence_file_mismatch_runs": ver["runs_with_mismatch"]}
    vmap = {v["run"]: v for v in ver["verification"]}
    for p in sorted(D3.glob("*/run*.npz")):
        rel = f"A0/{p.parent.name}/{p.stem}"
        out["runs_found"] += 1
        j = json.loads(p.with_suffix(".json").read_text())
        js = json.loads((SMOKE / p.parent.name / p.with_suffix(".json").name).read_text())
        out["seed_match"] += int(j["seed"] == js["seed"])
        out["fe_match"] += int(j["FE"] == js["FE"])
        v = vmap[rel]
        out["prod_identical"] += int(not v["mismatches"] and not v["missing_in_new"])
        z = np.load(p)
        d3 = [k for k in z.files if k.startswith("d3_") and z[k].dtype.kind == "f"]
        out["nan_free"] += int(all(np.all(np.isfinite(z[k])) for k in d3))
        s = z["d3_it_scale"][z["d3_row_t"]]
        out["std_eq_raw_over_scale"] += int(np.array_equal(z["d3_row_std"], z["d3_row_raw"] / s))
        # agent identity: for epoch iterations, post-phase population rows follow from parents + logged moves
        rt, rp, ro, ra, raw = z["d3_row_t"], z["d3_row_pop"], z["d3_row_op"], z["d3_row_agent"], z["d3_row_raw"]
        for e, t in enumerate(z["d3_ep_t"]):
            out["epochs_checked"] += 1
            XHp, XH = z["d3_ep_XH_pre"][e], z["d3_ep_XH"][e]
            exp = XHp.copy()
            m = (rt == t) & (ro == 0)
            exp[ra[m]] = XHp[ra[m]] + raw[m]
            out["agent_identity_hba_max_err"] = max(out["agent_identity_hba_max_err"],
                                                     float(np.max(np.abs(exp - XH) / (1 + np.abs(XH)))))
            # MPA: memory step then FAD; ties accepted by memory are not success rows -> count violations
            XMp, XM = z["d3_ep_XM_pre"][e], z["d3_ep_XM"][e]
            expM = XMp.copy()
            m1 = (rt == t) & (ro == 1)
            expM[ra[m1]] = XMp[ra[m1]] + raw[m1]
            m2 = (rt == t) & (ro == 2)
            expM[ra[m2]] = expM[ra[m2]] + raw[m2]
            bad = np.max(np.abs(expM - XM) / (1 + np.abs(XM)), axis=1) > 1e-9
            out["agent_identity_mpa_violations"] += int(bad.sum()); out["agent_identity_mpa_checked"] += XM.shape[0]
            it = int(np.flatnonzero(z["d3_it_t"] == t)[0])
            out["xprey_member"] += int(np.any(np.all(XHp == z["d3_it_xprey_H"][it], axis=1)))
            out["elite_member"] += int(np.any(np.all(XMp == z["d3_it_elite_M"][it], axis=1)))
    return out


# ------------------------------------------------------------------ per-run analysis
def analyse_run(p):
    z = np.load(p); j = json.loads(p.with_suffix(".json").read_text())
    fid, D, run, W = j["fid"], j["dimension"], j["run"], j["summary"]["W"]
    k = max(2, D // 5)
    Sref = landscape_ref(z, fid, D); Rref = cov2corr(Sref)
    I = np.eye(D)
    null_ref = rmse_off(I, Rref)
    lab = z["ref_block_label"] if "ref_block_label" in z.files else None
    rt, rp, ro, ra, raw, std = (z["d3_row_t"], z["d3_row_pop"], z["d3_row_op"], z["d3_row_agent"],
                                z["d3_row_raw"], z["d3_row_std"])
    ep_t = z["d3_ep_t"]
    bins = np.array_split(np.arange(ep_t.size), NBINS)
    ep_of_t = np.minimum(rt // CFG.U, ep_t.size - 1)
    it_index = {int(t): i for i, t in enumerate(z["d3_it_t"])}
    # target, residual, whitened for every row where defined
    tgt = np.full_like(raw, np.nan)
    isep = (rt % CFG.U == 0)
    for i in np.flatnonzero(isep & (ro != 2)):
        e = int(rt[i] // CFG.U); it = it_index[int(rt[i])]
        if ro[i] == 0:
            tgt[i] = z["d3_it_xprey_H"][it] - z["d3_ep_XH_pre"][e][ra[i]]
        else:
            tgt[i] = z["d3_it_elite_M"][it] - z["d3_ep_XM_pre"][e][ra[i]]
    white = np.empty_like(raw)
    covs = {0: z["d3_ep_H_cov"], 1: z["d3_ep_M_cov"]}
    Wcache = {}
    for i in range(raw.shape[0]):
        key = (int(rp[i]), int(ep_of_t[i]))
        if key not in Wcache:
            Wcache[key] = inv_sqrt(covs[key[0]][key[1]])
        white[i] = Wcache[key] @ raw[i]
    rows, series = [], []
    for b, eps in enumerate(bins):
        inb = np.isin(ep_of_t, eps)
        for pop, pname in ((0, "H"), (1, "M")):
            CX = np.mean(covs[pop][eps], axis=0); RX = cov2corr(CX)
            groups = {"all": inb & (rp == pop)}
            if pop == 1:
                groups["MPA"] = inb & (ro == 1); groups["FAD"] = inb & (ro == 2)
            for gname, m in groups.items():
                n = int(m.sum())
                if n < D + 1:
                    continue
                dx = raw[m]; Cdx = cov(dx); Rdx = corr(dx); Rw = corr(white[m])
                rec = {"fid": fid, "D": D, "run": run, "bin": b, "pop": pname, "group": gname, "n": n,
                       "floor": 1 / np.sqrt(n), "null_ref": null_ref,
                       "dx_pop_rmse": rmse_off(Rdx, RX), "dx_pop_cs": cs_off(Rdx, RX),
                       "dx_pop_sub": subsim(Cdx, CX, k), "dx_land_rmse": rmse_off(Rdx, Rref),
                       "dx_land_sub": subsim(Cdx, Sref, k), "pop_land_rmse": rmse_off(RX, Rref),
                       "pop_land_sub": subsim(CX, Sref, k), "white_land_rmse": rmse_off(Rw, Rref),
                       "white_iso_rmse": rmse_off(Rw, I), "E_pop": share_in(Cdx, CX, k),
                       "E_land": share_in(Cdx, Sref, k), "chance_k_over_D": k / D,
                       "gap_dx": float(topk(Cdx, k)[1][k - 1] / max(topk(Cdx, k)[1][k], 1e-300))}
                if lab is not None and fid == 2:
                    rec["wb_dx"], rec["wb_chance"] = within_block_share(Rdx, lab)
                    rec["wb_pop"], _ = within_block_share(RX, lab)
                    rec["wb_white"], _ = within_block_share(Rw, lab)
                    for name, Z in (("dx", dx), ("white", white[m])):
                        part = partition_from_stat(fisher_stat(Z, CFG.fisher_clip), CFG.alpha_link)
                        rec[f"ari_{name}"] = ari(part_labels(part, D), lab.tolist())
                # target / residual (epoch iterations, operators with a target)
                mt = m & isep & (ro != 2)
                nt = int(mt.sum())
                rec["n_target"] = nt
                if nt >= D + 1:
                    dxe, g = raw[mt], tgt[mt]
                    bcoef = float((dxe * g).sum() / (g * g).sum())
                    res2 = dxe - bcoef * g
                    res1 = dxe - ((dxe * g).sum(1) / (g * g).sum(1))[:, None] * g
                    cosv = (dxe * g).sum(1) / (np.linalg.norm(dxe, axis=1) * np.linalg.norm(g, axis=1))
                    Rg, Rde, Rr2, Rr1 = corr(g), corr(dxe), corr(res2), corr(res1)
                    rec.update({"b": bcoef, "cos_med": float(np.median(cosv)),
                                "norm_ratio_med": float(np.median(np.linalg.norm(dxe, axis=1) / np.linalg.norm(g, axis=1))),
                                "tgt_dx_rmse": rmse_off(Rg, Rde), "tgt_dx_sub": subsim(cov(g), cov(dxe), k),
                                "floor_t": 1 / np.sqrt(nt), "dxep_land_rmse": rmse_off(Rde, Rref),
                                "tgt_land_rmse": rmse_off(Rg, Rref), "res2_land_rmse": rmse_off(Rr2, Rref),
                                "res1_land_rmse": rmse_off(Rr1, Rref), "res2_pop_rmse": rmse_off(Rr2, RX),
                                "R2_target": float(1 - (res2 ** 2).sum() / (dxe ** 2).sum())})
                    if lab is not None and fid == 2:
                        rec["wb_res2"], _ = within_block_share(Rr2, lab)
                        part = partition_from_stat(fisher_stat(res2, CFG.fisher_clip), CFG.alpha_link)
                        rec["ari_res2"] = ari(part_labels(part, D), lab.tolist())
                rows.append(rec)
        # HBA vs MPA direct comparison and JOINT/POOLED descriptive per bin
        mH, mM = inb & (rp == 0), inb & (rp == 1)
        if mH.sum() > D and mM.sum() > D:
            RH, RM = corr(raw[mH]), corr(raw[mM])
            series.append({"fid": fid, "D": D, "run": run, "bin": b, "hm_rmse": rmse_off(RH, RM),
                           "hm_floor": float(np.sqrt(1 / mH.sum() + 1 / mM.sum())),
                           "hm_rmse_white": rmse_off(corr(white[mH]), corr(white[mM]))})
            for rep, Z in (("raw_std", std), ("whitened", white), ("residual_R2", None)):
                if rep == "residual_R2":
                    src = {}
                    for pop in (0, 1):
                        mt = inb & (rp == pop) & isep & (ro != 2)
                        if mt.sum() < D + 1:
                            src = None; break
                        g = tgt[mt]; dxe = raw[mt]
                        src[pop] = dxe - ((dxe * g).sum() / (g * g).sum()) * g
                    if src is None:
                        continue
                    ZH, ZM = src[0], src[1]
                else:
                    ZH, ZM = Z[mH], Z[mM]
                if ZH.shape[0] < 2 * W or ZM.shape[0] < 2 * W:
                    series[-1][f"jp_{rep}"] = "n/a"
                    continue
                jr = evaluate_sources("JOINT", {"H": (ZH[-W:], ZH[-2 * W:-W]), "M": (ZM[-W:], ZM[-2 * W:-W])},
                                      "stouffer", CFG)
                pr = evaluate_sources("POOLED", {"P": (np.vstack([ZH[-W:], ZM[-W:]]),
                                                       np.vstack([ZH[-2 * W:-W], ZM[-2 * W:-W]]))}, "single", CFG)
                series[-1][f"jp_{rep}"] = f"{jr.k_hat}/{pr.k_hat}"
    fig = {"Sref": Sref, "Rref": Rref, "lab": lab}
    return rows, series, fig, z, fid, D, run


def main():
    integ = integrity()
    ok = (integ["runs_found"] == 30 and integ["seed_match"] == 30 and integ["prod_identical"] == 30
          and integ["fe_match"] == 30 and integ["nan_free"] == 30 and integ["std_eq_raw_over_scale"] == 30
          and integ["agent_identity_hba_max_err"] < 1e-9 and integ["xprey_member"] == integ["epochs_checked"]
          and integ["elite_member"] == integ["epochs_checked"])
    integ["ok"] = bool(ok)
    (HERE / "integrity.json").write_text(json.dumps(integ, indent=1))
    print("integrity", json.dumps(integ))
    if not ok:
        print("INTEGRITY FAILED - STOP"); sys.exit(2)
    allrows, allseries, figs = [], [], {}
    for p in sorted(D3.glob("*/run*.npz")):
        rows, series, fig, z, fid, D, run = analyse_run(p)
        allrows += rows; allseries += series
        if run == 1:
            figs[(fid, D)] = (fig, p)
    df = pd.DataFrame(allrows); ds = pd.DataFrame(allseries)
    df.to_csv(HERE / "metrics_bins.csv", index=False); ds.to_csv(HERE / "metrics_hba_mpa_jointpooled.csv", index=False)
    import pickle
    with open(HERE / "_figdata.pkl", "wb") as f:
        pickle.dump({k: (v[0], str(v[1])) for k, v in figs.items()}, f)
    print("rows", len(df), "series", len(ds))


if __name__ == "__main__":
    main()
