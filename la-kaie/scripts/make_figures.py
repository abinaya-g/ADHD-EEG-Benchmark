"""Publication figures for one campaign/benchmark (PNG 300 dpi + PDF).

Style: one font family/size, fixed method->colour mapping (colour follows the
method, never its rank), thin lines, recessive grid, single y-axis. Log axes
are used only for quantities spanning orders of magnitude; the lower floor
applied to CEC errors before taking logs (1e-8, the conventional CEC
"solved" threshold) is stated in every affected caption/label.

  python scripts/make_figures.py --campaign pilot --benchmark CEC2022
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lakaie.analysis.stats import load_campaign, mean_matrix, rank_matrix  # noqa: E402
from lakaie.components import ACTIONS, STATE_NAMES  # noqa: E402

PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BASE = ["LA-KAIE-Full", "MPHBS", "MPHB", "MPA", "HBA"]
ABL = ["LA-KAIE-Full", "LA-KAIE-NoLandscape", "LA-KAIE-NoInteraction", "LA-KAIE-NoKnowledgeReward",
       "LA-KAIE-DimensionOnly", "LA-KAIE-RandomExchange", "LA-KAIE-FixedPolicy", "LA-KAIE-NoAdaptiveController"]
COLOR = {m: PAL[i] for i, m in enumerate(BASE)}
COLOR.update({m: PAL[i] for i, m in enumerate(ABL)})  # ablation figures use their own fixed order
BASE_COLOR = {m: PAL[i] for i, m in enumerate(BASE)}
ABL_COLOR = {m: PAL[i] for i, m in enumerate(ABL)}
FLOOR = 1e-8
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8,
                     "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e6e3", "grid.linewidth": 0.5, "lines.linewidth": 1.4,
                     "savefig.dpi": 300, "figure.dpi": 100})
ACT_COL = dict(zip(ACTIONS, PAL))


def save(fig, out, name):
    fig.tight_layout()
    fig.savefig(out / f"{name}.png", bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def grid(n):
    c = 4 if n > 8 else (4 if n > 4 else n)
    r = int(np.ceil(n / c))
    return r, c


class _LazyNPZ:
    """Opens each run's .npz on access and closes it immediately (no handle leak)."""

    def __getitem__(self, path):
        with np.load(Path(path).with_suffix(".npz")) as z:
            return {k: z[k] for k in z.files}


def load_npz(df):
    return _LazyNPZ()


def ylabel(bench):
    return f"error (floored at {FLOOR:g})" if bench == "CEC2022" else "fitness F(h)"


def conv_curves(df, npz, methods, colors, bench, out, name, title):
    insts = sorted(df.instance.unique())
    r, c = grid(len(insts))
    fig, axs = plt.subplots(r, c, figsize=(2.3 * c, 1.9 * r), squeeze=False)
    for ax, inst in zip(axs.flat, insts):
        for m in methods:
            g = df[(df.method == m) & (df.instance == inst)]
            if g.empty:
                continue
            curves = np.array([npz[p]["curve"] for p in g.path])
            cps = npz[g.path.iloc[0]]["checkpoints"]
            y = np.nanmean(curves, axis=0)
            if bench == "CEC2022":
                y = np.maximum(y, FLOOR)
            ax.plot(cps, y, color=colors[m], label=m)
        ax.set_yscale("log")
        ax.set_title(f"{'F' if bench == 'CEC2022' else 'FIR'}{inst}")
        ax.set_xlabel("function evaluations")
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    for ax in list(axs.flat)[len(insts):]:
        ax.axis("off")
    axs[0, 0].set_ylabel(f"mean best {ylabel(bench)}")
    h, lab = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=min(len(lab), 5), frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(title, y=1.0)
    fig.subplots_adjust(bottom=0.12)
    save(fig, out, name)


def boxplots(df, methods, bench, out, name):
    insts = sorted(df.instance.unique())
    r, c = grid(len(insts))
    fig, axs = plt.subplots(r, c, figsize=(2.3 * c, 2.0 * r), squeeze=False)
    for ax, inst in zip(axs.flat, insts):
        data, labs = [], []
        for m in methods:
            v = df[(df.method == m) & (df.instance == inst)].best_fitness.values.astype(float)
            if v.size:
                data.append(np.maximum(v, FLOOR) if bench == "CEC2022" else v); labs.append(m)
        bp = ax.boxplot(data, patch_artist=True, widths=0.6, medianprops={"color": "#0b0b0b"},
                        flierprops={"markersize": 2})
        for patch, m in zip(bp["boxes"], labs):
            patch.set_facecolor(BASE_COLOR.get(m, ABL_COLOR.get(m, "#999")))
            patch.set_alpha(0.8)
        ax.set_yscale("log")
        ax.set_xticks(range(1, len(labs) + 1))
        ax.set_xticklabels([lab.replace("LA-KAIE-", "LK-") for lab in labs], rotation=60, ha="right")
        ax.set_title(f"{'F' if bench == 'CEC2022' else 'FIR'}{inst}")
    for ax in list(axs.flat)[len(insts):]:
        ax.axis("off")
    axs[0, 0].set_ylabel(f"final {ylabel(bench)}")
    save(fig, out, name)


def avg_rank_bar(df, methods, out, name, title):
    methods = [m for m in methods if m in set(df.method)]
    R = rank_matrix(mean_matrix(df, methods))
    avg = R.mean().sort_values()
    fig, ax = plt.subplots(figsize=(4.2, 0.28 * len(avg) + 0.8))
    ax.barh(range(len(avg)), avg.values, color="#2a78d6", height=0.6)
    ax.set_yticks(range(len(avg))); ax.set_yticklabels(avg.index)
    for i, v in enumerate(avg.values):
        ax.text(v + 0.05, i, f"{v:.2f}", va="center", fontsize=7, color="#52514e")
    ax.invert_yaxis(); ax.set_xlabel("average mean-based rank (lower is better)")
    ax.set_xlim(0, len(methods) + 0.5); ax.set_title(title)
    save(fig, out, name)


# Nemenyi critical values q_0.05 (Demsar 2006, Table 5a), k = 2..10
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}


def cd_diagram(df, methods, out, name, title):
    methods = [m for m in methods if m in set(df.method)]
    k = len(methods)
    if k not in Q05:
        return
    R = rank_matrix(mean_matrix(df, methods))
    n = R.shape[0]
    avg = R.mean().sort_values()
    cd = Q05[k] * np.sqrt(k * (k + 1) / (6 * n))
    fig, ax = plt.subplots(figsize=(7, 0.25 * k + 1.4))
    ax.set_xlim(0.5, k + 3.2); ax.set_ylim(-1.6, k + 0.8)
    ax.plot([1, k], [k + 0.3, k + 0.3], color="#52514e", lw=0.8)
    for i in range(1, k + 1):
        ax.plot([i, i], [k + 0.3, k + 0.45], color="#52514e", lw=0.8)
        ax.text(i, k + 0.55, str(i), ha="center", fontsize=7)
    for j, (m, v) in enumerate(avg.items()):
        y = k - 1 - j
        ax.plot([v, v, k + 0.6], [k + 0.3, y, y], color="#8f8e88", lw=0.6)
        ax.plot(v, k + 0.3, "o", color="#2a78d6", ms=4)
        ax.text(k + 0.7, y, f"{m} ({v:.2f})", fontsize=7, va="center")
    # cliques: maximal groups whose rank range < CD
    vals = avg.values
    bars = []
    for i in range(k):
        j = i
        while j + 1 < k and vals[j + 1] - vals[i] < cd:
            j += 1
        if j > i and not any(a <= i and j <= b for a, b in bars):
            bars.append((i, j))
    for t, (i, j) in enumerate(bars):
        yy = -0.3 + 0.18 * t
        ax.plot([vals[i], vals[j]], [yy, yy], color="#0b0b0b", lw=2)
    ax.plot([1, 1 + cd], [-1.3, -1.3], color="#e34948", lw=2)
    ax.text(1 + cd + 0.1, -1.3, f"critical difference CD = {cd:.2f}", ha="left", va="center", fontsize=7)
    ax.axis("off")
    ax.set_title(title + "\n(Nemenyi CD, alpha = 0.05, instances as blocks; black bars join methods not separated by CD)", pad=18)
    save(fig, out, name)


def ablation_heatmap(df, out, name, bench):
    methods = [m for m in ABL if m in set(df.method)]
    if len(methods) < 2:
        return
    M = mean_matrix(df, methods)
    if bench == "CEC2022":
        M = M.clip(lower=FLOOR)
    L = np.log10(M.div(M["LA-KAIE-Full"], axis=0)).drop(columns="LA-KAIE-Full")
    lim = max(0.1, np.nanmax(np.abs(L.values)))
    fig, ax = plt.subplots(figsize=(0.55 * L.shape[1] + 2, 0.3 * L.shape[0] + 1.2))
    im = ax.imshow(L.values, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(L.shape[1])); ax.set_xticklabels([c.replace("LA-KAIE-", "") for c in L.columns], rotation=45, ha="right")
    ax.set_yticks(range(L.shape[0])); ax.set_yticklabels([f"{'F' if bench == 'CEC2022' else 'FIR'}{i}" for i in L.index])
    for i in range(L.shape[0]):
        for j in range(L.shape[1]):
            ax.text(j, i, f"{L.values[i, j]:+.2f}", ha="center", va="center", fontsize=6, color="#0b0b0b")
    cb = fig.colorbar(im, ax=ax); cb.set_label("log10(mean variant / mean Full)\n>0: variant worse")
    ax.grid(False); ax.set_title("Ablation: mean final value relative to LA-KAIE-Full")
    save(fig, out, name)


def iter_curves(df, npz, methods, colors, key, out, name, ylab, bench):
    insts = sorted(df.instance.unique())
    r, c = grid(len(insts))
    fig, axs = plt.subplots(r, c, figsize=(2.3 * c, 1.9 * r), squeeze=False)
    grid_fe = None
    for ax, inst in zip(axs.flat, insts):
        for m in methods:
            g = df[(df.method == m) & (df.instance == inst)]
            if g.empty:
                continue
            maxfe = int(g.max_fes.iloc[0])
            grid_fe = np.linspace(0, maxfe, 200)
            ys = []
            for p in g.path:
                z = npz[p]
                if f"iter_{key}" not in z:
                    continue
                ys.append(np.interp(grid_fe, z["iter_fe"], z[f"iter_{key}"]))
            if ys:
                ax.plot(grid_fe, np.mean(ys, axis=0), color=colors[m], label=m)
        ax.set_title(f"{'F' if bench == 'CEC2022' else 'FIR'}{inst}")
        ax.set_xlabel("function evaluations")
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        if key == "diversity":
            ax.set_yscale("log")
    for ax in list(axs.flat)[len(insts):]:
        ax.axis("off")
    axs[0, 0].set_ylabel(ylab)
    h, lab = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=min(len(lab), 5), frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(bottom=0.12)
    save(fig, out, name)


def action_distribution(df, out, name, bench, method="LA-KAIE-Full"):
    g = df[df.method == method]
    if g.empty:
        return
    rows = []
    for inst, gi in g.groupby("instance"):
        tot = np.zeros(8)
        for s in gi.summary:
            tot += np.array([s["action_counts"][a] for a in ACTIONS])
        rows.append(tot / tot.sum())
    S = np.array(rows)
    insts = sorted(g.instance.unique())
    fig, ax = plt.subplots(figsize=(0.45 * len(insts) + 2.5, 2.8))
    bottom = np.zeros(len(insts))
    for j, a in enumerate(ACTIONS):
        ax.bar(range(len(insts)), S[:, j], bottom=bottom, color=ACT_COL[a], label=a, width=0.75,
               edgecolor="white", linewidth=0.8)
        bottom += S[:, j]
    ax.set_xticks(range(len(insts)))
    ax.set_xticklabels([f"{'F' if bench == 'CEC2022' else 'FIR'}{i}" for i in insts])
    ax.set_ylabel("share of controller decisions"); ax.set_ylim(0, 1)
    ax.legend(ncol=1, bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)
    ax.set_title(f"{method}: exchange-action distribution per instance")
    save(fig, out, name)


def controller_behaviour(df, npz, out, name, method="LA-KAIE-Full", nbins=10):
    g = df[df.method == method]
    if g.empty:
        return
    counts = np.zeros((nbins, 8)); rew = [[] for _ in range(nbins)]
    for _, r in g.iterrows():
        z = npz[r.path]
        if "lk_ep_action" not in z:
            continue
        fe = z["lk_ep_fe"] / r.max_fes
        b = np.minimum((fe * nbins).astype(int), nbins - 1)
        for bi, a in zip(b, z["lk_ep_action"]):
            counts[bi, int(a)] += 1
        for bi, rv in zip(b, z["lk_ep_reward"]):
            if np.isfinite(rv):
                rew[bi].append(rv)
    share = counts / np.maximum(1, counts.sum(axis=1, keepdims=True))
    x = (np.arange(nbins) + 0.5) / nbins
    fig, axs = plt.subplots(1, 2, figsize=(7.5, 2.8))
    bottom = np.zeros(nbins)
    for j, a in enumerate(ACTIONS):
        axs[0].bar(x, share[:, j], bottom=bottom, width=0.9 / nbins, color=ACT_COL[a], label=a,
                   edgecolor="white", linewidth=0.6)
        bottom += share[:, j]
    axs[0].set_xlabel("budget used (FE / MaxFE)"); axs[0].set_ylabel("action share"); axs[0].set_ylim(0, 1)
    axs[0].legend(ncol=4, fontsize=6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.25))
    axs[0].set_title("controller decisions over the run")
    mu = [np.mean(v) if v else np.nan for v in rew]
    lo = [np.percentile(v, 25) if v else np.nan for v in rew]
    hi = [np.percentile(v, 75) if v else np.nan for v in rew]
    axs[1].plot(x, mu, color="#2a78d6", marker="o", ms=3, label="mean")
    axs[1].fill_between(x, lo, hi, color="#2a78d6", alpha=0.15, label="interquartile range")
    axs[1].set_xlabel("budget used (FE / MaxFE)"); axs[1].set_ylabel("reward")
    axs[1].set_title("reward evolution"); axs[1].legend(frameon=False)
    save(fig, out, name)


def state_evolution(df, npz, out, name, method="LA-KAIE-Full", nbins=20):
    g = df[df.method == method]
    acc = [[[] for _ in range(nbins)] for _ in range(10)]
    for _, r in g.iterrows():
        z = npz[r.path]
        if "lk_it_state" not in z:
            continue
        fe = z["lk_it_fe"] / r.max_fes
        b = np.minimum((fe * nbins).astype(int), nbins - 1)
        for bi, s in zip(b, z["lk_it_state"]):
            for i in range(10):
                acc[i][bi].append(s[i])
    if not any(acc[0]):
        return
    x = (np.arange(nbins) + 0.5) / nbins
    fig, axs = plt.subplots(2, 5, figsize=(10, 3.6), sharex=True, sharey=True)
    for i, ax in enumerate(axs.flat):
        ax.plot(x, [np.mean(v) if v else np.nan for v in acc[i]], color="#2a78d6")
        ax.set_title(STATE_NAMES[i]); ax.set_ylim(0, 1)
    for ax in axs[1]:
        ax.set_xlabel("FE / MaxFE")
    fig.suptitle(f"{method}: mean landscape-state trajectories (all instances)")
    save(fig, out, name)


def interaction_figure(df, npz, out, name, bench, method="LA-KAIE-Full"):
    g = df[df.method == method]
    if g.empty:
        return
    insts = sorted(g.instance.unique())
    r, c = grid(len(insts))
    fig, axs = plt.subplots(r, c, figsize=(2.2 * c, 2.2 * r), squeeze=False)
    for ax, inst in zip(axs.flat, insts):
        gi = g[g.instance == inst].sort_values("best_fitness")
        rr = gi.iloc[len(gi) // 2]            # median run
        s = rr.summary
        D = int(rr.dimension)
        G = np.zeros((D, D)); G[np.triu_indices(D, 1)] = s["final_G_upper"]; G = G + G.T
        order = [i for grp in s["final_groups"] for i in grp]
        ax.imshow(G[np.ix_(order, order)], cmap="Blues", vmin=0, vmax=1)
        pos = 0
        for grp in s["final_groups"]:
            if len(grp) > 1:
                ax.add_patch(plt.Rectangle((pos - 0.5, pos - 0.5), len(grp), len(grp), fill=False,
                                           edgecolor="#e34948", lw=0.8))
            pos += len(grp)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax.set_title(f"{'F' if bench == 'CEC2022' else 'FIR'}{inst}: S8={s['final_S8']:.2f}, {s['final_n_groups']} groups")
    for ax in list(axs.flat)[len(insts):]:
        ax.axis("off")
    fig.suptitle(f"{method}: final interaction matrix G of the median run (variables ordered by group; red = multi-variable groups)")
    save(fig, out, name)


def overhead(df, out, name):
    g = df.groupby("method").runtime_sec.agg(["mean", "std"]).sort_values("mean")
    fig, axs = plt.subplots(1, 2, figsize=(8, 0.28 * len(g) + 1))
    axs[0].barh(range(len(g)), g["mean"], xerr=g["std"], color="#2a78d6", height=0.6,
                error_kw={"elinewidth": 0.6})
    axs[0].set_yticks(range(len(g))); axs[0].set_yticklabels(g.index); axs[0].set_xlabel("runtime per run (s)")
    axs[0].set_title("wall-clock runtime (mean, sd)")
    lk = df[df.method.str.startswith("LA-KAIE")]
    if not lk.empty:
        rows = []
        for m, gm in lk.groupby("method"):
            ov = np.array([[s["overhead_sec"]["interaction"], s["overhead_sec"]["controller"], s["overhead_sec"]["state"],
                            s["runtime_sec"]] for s in gm.summary])
            rows.append((m, ov.mean(axis=0)))
        names = [r[0] for r in rows]
        vals = np.array([r[1] for r in rows])
        frac = vals[:, :3] / vals[:, 3:4]
        left = np.zeros(len(rows))
        for j, (lab, col) in enumerate(zip(["interaction model", "controller", "state"], PAL)):
            axs[1].barh(range(len(rows)), frac[:, j], left=left, color=col, label=lab, height=0.6,
                        edgecolor="white", linewidth=0.8)
            left += frac[:, j]
        axs[1].set_yticks(range(len(rows))); axs[1].set_yticklabels(names)
        axs[1].set_xlabel("fraction of LA-KAIE runtime"); axs[1].legend(frameon=False, fontsize=6)
        axs[1].set_title("LA-KAIE overhead components")
    else:
        axs[1].axis("off")
    save(fig, out, name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--benchmark", required=True)
    a = ap.parse_args()
    df = load_campaign(ROOT / "results" / "raw", a.campaign, a.benchmark)
    df = df[df.status == "ok"]
    out = ROOT / "results" / "figures" / a.campaign / a.benchmark
    out.mkdir(parents=True, exist_ok=True)
    npz = load_npz(df)
    b = a.benchmark
    base = [m for m in BASE if m in set(df.method)]
    abl = [m for m in ABL if m in set(df.method)]
    conv_curves(df, npz, base, BASE_COLOR, b, out, "01_convergence_comparison",
                f"{b}: mean convergence over paired runs")
    if len(abl) > 1:
        conv_curves(df, npz, abl, ABL_COLOR, b, out, "01b_convergence_ablation", f"{b}: ablation convergence")
    avg_rank_bar(df, base + [m for m in abl if m not in base], out, "03_average_ranks", f"{b}: average ranks (all methods)")
    boxplots(df, base, b, out, "04_boxplots_comparison")
    if len(abl) > 1:
        boxplots(df, abl, b, out, "04b_boxplots_ablation")
    cd_diagram(df, base, out, "05_cd_comparison", f"{b}: comparison stage")
    cd_diagram(df, abl, out, "05b_cd_ablation", f"{b}: ablation stage")
    ablation_heatmap(df, out, "06_ablation_heatmap", b)
    iter_curves(df, npz, base, BASE_COLOR, "diversity", out, "07_diversity", "normalised diversity (log)", b)
    iter_curves(df, npz, base, BASE_COLOR, "stagnation", out, "08_stagnation", "iterations without improvement", b)
    action_distribution(df, out, "09_action_distribution", b)
    interaction_figure(df, npz, out, "10_interaction_groups", b)
    controller_behaviour(df, npz, out, "11_12_controller_and_reward")
    state_evolution(df, npz, out, "11b_state_trajectories")
    overhead(df, out, "13_overhead")
    (out / "README.txt").write_text(json.dumps({"campaign": a.campaign, "benchmark": b,
                                                "cec_error_floor_for_log_axes": FLOOR}, indent=1))
    print("figures ->", out)


if __name__ == "__main__":
    main()
