"""Tables, frozen classification rule and diagnostic figures for the D3 analysis.
Reads metrics_*.csv / integrity.json produced by d3_analysis.py."""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
DIV = LinearSegmentedColormap.from_list("div", [BLUE, "#f0efec", "#e34948"])
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2})
CELLS = [(f, D) for f in (1, 2, 3) for D in (10, 20)]

df = pd.read_csv(HERE / "metrics_bins.csv")
ds = pd.read_csv(HERE / "metrics_hba_mpa_jointpooled.csv")
integ = json.loads((HERE / "integrity.json").read_text())


def per_run(sub, col):
    return sub.groupby("run")[col].mean()


def med(fid, D, pop, col, group="all"):
    s = df[(df.fid == fid) & (df.D == D) & (df["pop"] == pop) & (df.group == group)]
    if col not in s or s[col].dropna().empty:
        return np.nan
    return float(per_run(s.dropna(subset=[col]), col).median())


def skill(fid, D, pop, col, group="all"):
    s = df[(df.fid == fid) & (df.D == D) & (df["pop"] == pop) & (df.group == group)].dropna(subset=[col])
    if s.empty:
        return pd.Series(dtype=float)
    r = per_run(s, col)
    if fid == 1:
        return r
    null = s.groupby("run")["null_ref"].first()
    return 1 - r / null


# ---------------------------------------------------------------- classification (frozen rule)
def meets_landscape(fid, D, pop, col):
    s = df[(df.fid == fid) & (df.D == D) & (df["pop"] == pop) & (df.group == "all")].dropna(subset=[col])
    if s.empty:
        return False, "no data"
    r = per_run(s, col)
    if fid == 1:
        floor = per_run(s, "floor_t" if col.startswith("res") else "floor")
        ok = (r <= 2 * floor).sum()
        return ok >= 4, f"{ok}/5 runs within 2x noise floor"
    sk = 1 - r / s.groupby("run")["null_ref"].first()
    if fid == 3:
        ok = (sk >= 0.25).sum()
        return ok >= 4, f"{ok}/5 runs skill>=0.25"
    ari_col = "ari_res2" if col.startswith("res") else "ari_white"
    a = per_run(s.dropna(subset=[ari_col]), ari_col) if ari_col in s else pd.Series(dtype=float)
    ok = ((sk >= 0.25) & (a.reindex(sk.index) >= 0.5)).sum()
    return ok >= 4, f"{ok}/5 runs skill>=0.25 & ARI>=0.5"


crit = {}
for rep, col in (("residual_R2", "res2_land_rmse"), ("whitened", "white_land_rmse")):
    for fid, D in CELLS:
        for pop in ("H", "M"):
            crit[(rep, fid, D, pop)] = meets_landscape(fid, D, pop, col)
A = any(all(any(crit[(rep, f, D, p)][0] for p in ("H", "M")) for f, D in CELLS) for rep in ("residual_R2", "whitened"))
closer = {pop: sum(med(f, D, pop, "dx_pop_rmse") < med(f, D, pop, "dx_land_rmse") for f, D in CELLS) for pop in ("H", "M")}
land_any_23 = any(crit[(rep, f, D, p)][0] for rep in ("residual_R2", "whitened") for f, D in CELLS if f in (2, 3)
                  for p in ("H", "M"))
if not integ["ok"]:
    cls = "D"
elif A:
    cls = "A"
elif all(v >= 4 for v in closer.values()) and not land_any_23:
    cls = "B"
elif any(v >= 4 for v in closer.values()) and land_any_23:
    cls = "C"
else:
    cls = "closest: " + ("B" if not land_any_23 else "C")
summary = {"classification": cls, "closer_to_population_cells": closer, "landscape_criterion_S2_S3_any": land_any_23,
           "criterion_detail": {f"{k[0]}|S{k[1]}D{k[2]}|{k[3]}": v[1] for k, v in crit.items()}, "integrity": integ}


# ---------------------------------------------------------------- tables
def cellfmt(fid, D, col, as_skill=False, fmt="{:.3f}"):
    out = []
    for pop in ("H", "M"):
        if as_skill and fid != 1:
            v = skill(fid, D, pop, col).median()
        else:
            v = med(fid, D, pop, col)
        out.append("–" if not np.isfinite(v) else fmt.format(v))
    return " / ".join(out)


rowsdef = [("Δx ↔ population (rmse_off ↓)", "dx_pop_rmse", False),
           ("Δx ↔ landscape (rmse_off ↓)", "dx_land_rmse", False),
           ("population ↔ landscape (rmse_off ↓)", "pop_land_rmse", False),
           ("target component ↔ Δx (rmse_off ↓)", "tgt_dx_rmse", False),
           ("residual R2 ↔ landscape (rmse_off ↓)", "res2_land_rmse", False),
           ("whitened Δx ↔ landscape (rmse_off ↓)", "white_land_rmse", False),
           ("Δx noise floor 1/√n", "floor", False),
           ("residual noise floor 1/√n", "floor_t", False),
           ("independence reference rmse_off(I, R_ref)", "null_ref", False)]
hdr = "| comparison (H / M) | " + " | ".join(f"S{f} D{D}" for f, D in CELLS) + " |"
sep = "|---|" + "---:|" * 6
primary = [hdr, sep] + [f"| {name} | " + " | ".join(cellfmt(f, D, col) for f, D in CELLS) + " |" for name, col, _ in rowsdef]
skillrows = [hdr, sep] + [f"| {name} | " + " | ".join(cellfmt(f, D, col, True) if f != 1 else "(S1: see rmse)" for f, D in CELLS) + " |"
                          for name, col in (("Δx skill", "dx_land_rmse"), ("population skill", "pop_land_rmse"),
                                            ("target skill", "tgt_land_rmse"), ("residual R2 skill", "res2_land_rmse"),
                                            ("residual R1 skill", "res1_land_rmse"), ("whitened skill", "white_land_rmse"))]
extra = [hdr, sep] + [f"| {name} | " + " | ".join(cellfmt(f, D, col) for f, D in CELLS) + " |" for name, col in
                      (("CS_off(Δx, population) ↑", "dx_pop_cs"), ("subspace sim Δx–population ↑", "dx_pop_sub"),
                       ("subspace sim Δx–landscape ↑", "dx_land_sub"), ("subspace sim population–landscape ↑", "pop_land_sub"),
                       ("E_pop: Δx variance in top-k population subspace", "E_pop"),
                       ("E_land: Δx variance in top-k landscape subspace", "E_land"), ("chance k/D", "chance_k_over_D"),
                       ("median cos(Δx, target)", "cos_med"), ("median ‖Δx‖/‖target‖", "norm_ratio_med"),
                       ("fitted b (Δx ≈ b·target)", "b"), ("R² of target component", "R2_target"),
                       ("subspace sim target–Δx", "tgt_dx_sub"), ("whitened ↔ identity (rmse_off)", "white_iso_rmse"),
                       ("residual R2 ↔ population (rmse_off)", "res2_pop_rmse"), ("eigen-gap λk/λk+1 of Cov(Δx)", "gap_dx"))]
ops = [hdr.replace("(H / M)", "(M population: MPA / FAD)"), sep] + [
    f"| {name} | " + " | ".join(" / ".join("–" if not np.isfinite(med(f, D, 'M', col, g)) else f"{med(f, D, 'M', col, g):.3f}"
                                           for g in ("MPA", "FAD")) for f, D in CELLS) + " |"
    for name, col in (("Δx ↔ population", "dx_pop_rmse"), ("Δx ↔ landscape", "dx_land_rmse"), ("n rows per bin", "n"))]
s2 = ["| quantity (H / M), S2 | D10 | D20 |", "|---|---:|---:|"] + [
    f"| {name} | {cellfmt(2, 10, col)} | {cellfmt(2, 20, col)} |" for name, col in
    (("within-block share, Δx", "wb_dx"), ("within-block share, population", "wb_pop"), ("within-block share, residual R2", "wb_res2"),
     ("within-block share, whitened", "wb_white"), ("chance within-block share", "wb_chance"), ("partition ARI, Δx", "ari_dx"),
     ("partition ARI, residual R2", "ari_res2"), ("partition ARI, whitened", "ari_white"))]
hm = ["| cell | rmse_off(R_HBA, R_MPA) raw | whitened | noise floor | JOINT/POOLED labels raw_std | residual R2 | whitened |",
      "|---|---:|---:|---:|---|---|---|"]
for f, D in CELLS:
    s = ds[(ds.fid == f) & (ds.D == D)]
    def lab(c):
        v = s[c].dropna().astype(str) if c in s else pd.Series(dtype=str)
        v = v[v != "n/a"]
        return ", ".join(f"{k}:{n}" for k, n in v.value_counts().items()) or "n/a"
    hm.append(f"| S{f} D{D} | {s.hm_rmse.median():.3f} | {s.hm_rmse_white.median():.3f} | {s.hm_floor.median():.3f} | "
              f"{lab('jp_raw_std')} | {lab('jp_residual_R2')} | {lab('jp_whitened')} |")
temporal = ["| cell / pop | quantity | bin 1 | bin 2 | bin 3 | bin 4 | bin 5 |", "|---|---|---:|---:|---:|---:|---:|"]
for f, D in CELLS:
    for pop in ("H", "M"):
        s = df[(df.fid == f) & (df.D == D) & (df["pop"] == pop) & (df.group == "all")]
        for name, col in (("Δx↔pop", "dx_pop_rmse"), ("Δx↔land", "dx_land_rmse"), ("res2↔land", "res2_land_rmse"),
                          ("cos(Δx,target)", "cos_med")):
            vals = [s[s.bin == b][col].median() for b in range(5)]
            temporal.append(f"| S{f} D{D} {pop} | {name} | " + " | ".join("–" if not np.isfinite(v) else f"{v:.3f}" for v in vals) + " |")
tables = {"primary": primary, "skill": skillrows, "extra": extra, "ops": ops, "s2": s2, "hm": hm, "temporal": temporal}
(HERE / "tables.json").write_text(json.dumps(tables, indent=1))
(HERE / "summary.json").write_text(json.dumps(summary, indent=1, default=str))

# ---------------------------------------------------------------- figures
figd = pickle.load(open(HERE / "_figdata.pkl", "rb"))


def corrm(X):
    X = X - X.mean(0); sd = np.sqrt((X * X).sum(0)); sd[sd == 0] = 1
    R = (X / sd).T @ (X / sd); np.fill_diagonal(R, 1); return R


# 1 spectra (run 1, D=20, middle bin)
fig, axs = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
for ax, f in zip(axs, (1, 2, 3)):
    fd, p = figd[(f, 20)]
    z = np.load(p)
    ep = np.array_split(np.arange(z["d3_ep_t"].size), 5)[2]
    m = np.isin(np.minimum(z["d3_row_t"] // 5, z["d3_ep_t"].size - 1), ep) & (z["d3_row_pop"] == 0)
    for lab_, S, col in (("Cov(Δx), HBA", np.cov(z["d3_row_raw"][m].T), BLUE),
                         ("Cov(X), HBA", z["d3_ep_H_cov"][ep].mean(0), ORANGE), ("Σ_ref = H⁻¹", fd["Sref"], AQUA)):
        w = np.sort(np.linalg.eigvalsh(S))[::-1]; ax.semilogy(np.arange(1, 21), w / w.sum(), color=col, label=lab_)
    ax.set_title(f"S{f}, D=20, run 1, bin 3"); ax.set_xlabel("eigenvalue rank")
axs[0].set_ylabel("normalised eigenvalue"); axs[-1].legend(frameon=False)
fig.tight_layout(); fig.savefig(FIG / "fig1_spectra.png", dpi=150); plt.close(fig)

# 2 similarity over time (D=20)
fig, axs = plt.subplots(2, 3, figsize=(11, 5.4), sharex=True)
for j, f in enumerate((1, 2, 3)):
    for i, pop in enumerate(("H", "M")):
        ax = axs[i, j]; s = df[(df.fid == f) & (df.D == 20) & (df["pop"] == pop) & (df.group == "all")]
        for col, lab_, c in (("dx_pop_rmse", "Δx ↔ population", BLUE), ("dx_land_rmse", "Δx ↔ landscape", ORANGE),
                             ("res2_land_rmse", "residual ↔ landscape", AQUA), ("white_land_rmse", "whitened ↔ landscape", YELLOW)):
            ax.plot(range(1, 6), [s[s.bin == b][col].median() for b in range(5)], marker="o", ms=4, color=c, label=lab_)
        ax.set_title(f"S{f} D=20, {'HBA' if pop == 'H' else 'MPA'} population")
        if j == 0: ax.set_ylabel("rmse_off (lower = more similar)")
        if i == 1: ax.set_xlabel("temporal bin")
axs[0, 2].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "fig2_similarity_over_time.png", dpi=150); plt.close(fig)

# 3 HBA vs MPA
fig, ax = plt.subplots(figsize=(7, 3.2))
x = np.arange(6); w_ = 0.26
for off, col, lab_, c in ((-w_, "hm_rmse", "raw Δx", BLUE), (0, "hm_rmse_white", "whitened Δx", ORANGE), (w_, "hm_floor", "noise floor", AQUA)):
    ax.bar(x + off, [ds[(ds.fid == f) & (ds.D == D)][col].median() for f, D in CELLS], width=w_ - 0.02, color=c, label=lab_)
ax.set_xticks(x, [f"S{f} D{D}" for f, D in CELLS]); ax.set_ylabel("rmse_off(R_HBA, R_MPA)")
ax.set_title("HBA vs MPA displacement correlation structure"); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(FIG / "fig3_hba_vs_mpa.png", dpi=150); plt.close(fig)

# 4 target vs actual: median cos per cell and population
fig, ax = plt.subplots(figsize=(7, 3.2))
for off, pop, c in ((-0.18, "H", BLUE), (0.18, "M", ORANGE)):
    ax.bar(x + off, [med(f, D, pop, "cos_med") for f, D in CELLS], width=0.34, color=c, label="HBA" if pop == "H" else "MPA")
ax.set_xticks(x, [f"S{f} D{D}" for f, D in CELLS]); ax.set_ylabel("median cos(Δx, target − x)")
ax.set_title("Alignment of successful displacements with the operator target direction"); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(FIG / "fig4_target_vs_actual.png", dpi=150); plt.close(fig)

# 5 raw / residual / whitened landscape rmse
fig, axs = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
for ax, pop in zip(axs, ("H", "M")):
    for off, col, lab_, c in ((-0.3, "dx_land_rmse", "raw Δx", BLUE), (-0.1, "res2_land_rmse", "residual R2", ORANGE),
                              (0.1, "white_land_rmse", "whitened", AQUA), (0.3, "null_ref", "independence reference", YELLOW)):
        ax.bar(x + off, [med(f, D, pop, col) for f, D in CELLS], width=0.19, color=c, label=lab_)
    ax.set_xticks(x, [f"S{f} D{D}" for f, D in CELLS]); ax.set_title(f"{'HBA' if pop == 'H' else 'MPA'} population")
axs[0].set_ylabel("rmse_off to landscape R_ref (lower = closer)"); axs[1].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "fig5_recovery.png", dpi=150); plt.close(fig)


# 6/7 heatmaps S2, S3 (D=20, run 1, bin 3, HBA)
def heat(f, name):
    fd, p = figd[(f, 20)]; z = np.load(p)
    ep = np.array_split(np.arange(z["d3_ep_t"].size), 5)[2]
    rt = z["d3_row_t"]; m = np.isin(np.minimum(rt // 5, z["d3_ep_t"].size - 1), ep) & (z["d3_row_pop"] == 0)
    raw = z["d3_row_raw"][m]
    mt = m & (rt % 5 == 0) & (z["d3_row_op"] == 0)
    it = {int(t): i for i, t in enumerate(z["d3_it_t"])}
    g = np.array([z["d3_it_xprey_H"][it[int(t)]] - z["d3_ep_XH_pre"][int(t) // 5][a]
                  for t, a in zip(rt[mt], z["d3_row_agent"][mt])])
    dxe = z["d3_row_raw"][mt]; res = dxe - ((dxe * g).sum() / (g * g).sum()) * g
    order = np.argsort(fd["lab"], kind="stable") if fd["lab"] is not None else np.arange(20)
    mats = [("landscape R_ref", fd["Rref"]), ("population R_X", corrm(np.vstack(z["d3_ep_XH"][ep]) - 0) if False else
            (lambda C: C / np.sqrt(np.outer(np.diag(C), np.diag(C))))(z["d3_ep_H_cov"][ep].mean(0))),
            ("raw Δx", corrm(raw)), ("residual R2", corrm(res))]
    fig, axs = plt.subplots(1, 4, figsize=(12, 3.3))
    for ax, (t_, M) in zip(axs, mats):
        im = ax.imshow(M[np.ix_(order, order)], cmap=DIV, vmin=-1, vmax=1); ax.set_title(t_); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.colorbar(im, ax=axs, shrink=0.8, label="correlation")
    fig.suptitle(f"S{f}, D=20, run 1, bin 3, HBA population" + (" (coordinates ordered by true block)" if fd["lab"] is not None else ""))
    fig.savefig(FIG / name, dpi=150); plt.close(fig)


heat(2, "fig6_S2_blocks.png")
heat(3, "fig7_S3_rotation.png")
print("classification", cls, closer, land_any_23)
