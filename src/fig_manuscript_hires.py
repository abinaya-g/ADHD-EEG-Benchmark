"""Regenerates the manuscript's data figures at publication quality from the
saved files only (results_final/predictions/all_predictions.csv and
manuscript/TABLES/TABLE_4_CONFIDENCE_INTERVALS.csv).

Sizing rule: figures are placed at ~6.3 in text width, so a figure drawn W in
wide is scaled by 6.3/W; fonts are chosen so printed text stays >= ~7 pt.
All figures 300 dpi. Usage: python -m src.fig_manuscript_hires
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

PRED = "results_final/predictions/all_predictions.csv"
T4 = "manuscript/TABLES/TABLE_4_CONFIDENCE_INTERVALS.csv"
OUT = "manuscript/FIGURES/"
ORDER = ["CNN", "CNN+LR", "CNN+NLSVM", "CNN+RF", "CNN+GNB", "CNN+KNN",
         "CNN+LinearSVM", "EEGNet", "ShallowConvNet", "DeepConvNet"]
BLUE, ORANGE, GREEN = "#1f77b4", "#d95f02", "#2ca02c"
plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 14,
                     "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 11,
                     "axes.linewidth": 1.2})


def save(fig, name):
    fig.savefig(OUT + name, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def subj(d, m, **kw):
    s = d[d.model == m]
    return s.groupby("subject_id").agg(y=("y_true", "first"), p=("y_proba", "mean"))


def fold_ba(d, m, rep=None):
    """Subject-level balanced accuracy for every (repeat, outer_fold)."""
    s = d[d.model == m]
    if rep is not None:
        s = s[s["repeat"] == rep]
    out = []
    for (r, f), g in s.groupby(["repeat", "outer_fold"]):
        a = g.groupby("subject_id").agg(y=("y_true", "first"), p=("y_proba", "mean"))
        yh = (a.p >= .5).astype(int)
        out.append((r, f, 0.5 * ((yh[a.y == 1] == 1).mean() + (yh[a.y == 0] == 0).mean())))
    return pd.DataFrame(out, columns=["repeat", "fold", "ba"])


def boxstyle(ax):
    ax.grid(axis="y", alpha=.3)
    ax.set_axisbelow(True)


def fig_ci(t4):
    t = t4[(t4.preprocessing == "128Hz") & (t4.metric == "balanced_accuracy") & t4.model.isin(ORDER)]
    t = t.set_index("model").loc[ORDER].sort_values("point_estimate")
    fig, ax = plt.subplots(figsize=(8, 5.2))
    y = np.arange(len(t))
    ax.errorbar(t.point_estimate, y, xerr=[t.point_estimate - t.ci_low, t.ci_high - t.point_estimate],
                fmt="o", ms=8, capsize=5, lw=2, color=BLUE)
    ax.axvline(0.5, color="k", ls="--", lw=1, alpha=.6)
    ax.set_yticks(y); ax.set_yticklabels(t.index)
    ax.set_xlabel("Subject-level balanced accuracy (95% bootstrap CI)")
    ax.grid(axis="x", alpha=.3)
    save(fig, "fig4_model_comparison_balanced_accuracy.png")


def fig_roc(d):
    fig, ax = plt.subplots(figsize=(7, 7))
    cmap = plt.get_cmap("tab10")
    for i, m in enumerate(ORDER):
        s = subj(d, m)
        fpr, tpr, _ = roc_curve(s.y, s.p)
        ax.plot(fpr, tpr, lw=2.2, color=cmap(i), label=f"{m} (AUC = {roc_auc_score(s.y, s.p):.2f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=.5, lw=1.2)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_aspect("equal"); ax.grid(alpha=.3)
    ax.legend(loc="lower right", frameon=True)
    save(fig, "fig5_subject_roc.png")


def fig_repeated(d):
    data = [fold_ba(d, m).ba.values for m in ORDER]
    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.boxplot(data, tick_labels=ORDER, showmeans=True, widths=.6,
               medianprops=dict(color=ORANGE, lw=2), boxprops=dict(lw=1.4), whiskerprops=dict(lw=1.4),
               capprops=dict(lw=1.4), meanprops=dict(marker="^", markerfacecolor=GREEN, markeredgecolor=GREEN, ms=9))
    ax.axhline(0.5, color="k", ls="--", lw=1, alpha=.6)
    ax.set_ylabel("Balanced accuracy per outer fold")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    boxstyle(ax)
    save(fig, "fig7_repeated_cv_distribution_balanced_accuracy.png")


def fig_resolution(d, m, name):
    s = d[(d.model == m) & (d["repeat"] == 0)]
    a = fold_ba(s[s.preprocessing == "128Hz"], m).sort_values("fold").ba.values
    b = fold_ba(s[s.preprocessing == "128to512_interp"], m).sort_values("fold").ba.values
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    for u, v in zip(a, b):
        ax.plot([0, 1], [u, v], color="grey", alpha=.6, lw=1.5)
    ax.plot(np.zeros(5), a, "o", color=BLUE, ms=9, mfc="white", mew=2)
    ax.plot(np.ones(5), b, "o", color=BLUE, ms=9, mfc="white", mew=2)
    ax.plot([0, 1], [a.mean(), b.mean()], "^", color=GREEN, ms=12, ls="none", label="Mean of 5 folds")
    ax.set_xticks([0, 1]); ax.set_xticklabels([f"{m}\n128 Hz", f"{m}\n128→512 Hz interp."])
    ax.set_xlim(-.4, 1.4); ax.set_ylabel("Balanced accuracy per outer fold")
    ax.legend(loc="best"); boxstyle(ax)
    save(fig, name)


def fig_coral(d):
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    vals = []
    for m in ("CNN+LR_noCORAL", "CNN+LR_withCORAL"):
        s = d[d.model == m]
        acc = [(g.y_true == g.y_pred).mean() for _, g in s.groupby(["repeat", "outer_fold"])]
        vals.append(acc)
    ax.boxplot(vals, tick_labels=["Without CORAL", "With CORAL"], showmeans=True, widths=.5,
               medianprops=dict(color=ORANGE, lw=2), boxprops=dict(lw=1.4), whiskerprops=dict(lw=1.4),
               capprops=dict(lw=1.4), meanprops=dict(marker="^", markerfacecolor=GREEN, markeredgecolor=GREEN, ms=9))
    ax.set_ylabel("Epoch-level accuracy per outer fold"); boxstyle(ax)
    save(fig, "fig10_coral_covariance.png")


def fig_cv_schematic(outer=5, inner=4):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.set_xlim(0, 10); ax.set_ylim(0.3, outer + 0.7); ax.axis("off")
    for f in range(outer):
        y = outer - f
        ax.add_patch(plt.Rectangle((0, y - .4), 6.9, .8, fill=False, lw=1.5))
        ax.text(3.45, y, f"Outer fold {f+1}: train on outer-train subjects\n(inner {inner}-fold GroupKFold selects training epochs)",
                ha="center", va="center", fontsize=12)
        ax.add_patch(plt.Rectangle((7.2, y - .4), 2.8, .8, fill=False, edgecolor="crimson", lw=1.8))
        ax.text(8.6, y, "outer-test subjects\n(evaluated once)", ha="center", va="center", fontsize=12, color="crimson")
    save(fig, "fig3_nested_cv_schematic.png")


def main():
    d = pd.read_csv(PRED)
    d128 = d[d.preprocessing == "128Hz"]
    fig_ci(pd.read_csv(T4)); fig_roc(d128); fig_repeated(d128)
    fig_resolution(d, "CNN", "fig8_resolution.png"); fig_resolution(d, "EEGNet", "fig8_resolution_EEGNet.png")
    fig_coral(d128); fig_cv_schematic()


if __name__ == "__main__":
    main()
