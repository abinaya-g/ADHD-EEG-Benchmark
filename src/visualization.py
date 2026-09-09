"""
Figure generators (Phase 19). Every function takes already-computed
results (DataFrames from evaluation.py / statistics.py output) and saves a
PNG to config.FIGURES_DIR -- no numbers are computed here that weren't
already computed and saved elsewhere as CSV, so every figure can be
regenerated from the saved tables alone.

Figures 1 (workflow) and 2 (architecture diagrams) are not generated here:
they are conceptual/schematic diagrams best authored directly (e.g. in the
same tool used for the rest of the manuscript's figures) rather than
synthesized from result tables. Figure 3 (nested-CV schematic) is
generated procedurally below since it only needs the fold/subject counts,
not real results.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import config


def _savefig(fig, name):
    path = os.path.join(config.FIGURES_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def fig3_nested_cv_schematic(outer_folds, inner_folds, name="fig3_nested_cv_schematic.png"):
    fig, ax = plt.subplots(figsize=(9, 3 + 0.4 * outer_folds))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, outer_folds + 1)
    ax.axis("off")
    for f in range(outer_folds):
        y = outer_folds - f
        ax.add_patch(plt.Rectangle((0, y - 0.4), 7, 0.8, fill=False))
        ax.text(3.5, y, f"Outer fold {f+1}: train on outer-train subjects\n"
                          f"(inner {inner_folds}-fold GroupKFold selects #epochs)",
                ha="center", va="center", fontsize=8)
        ax.add_patch(plt.Rectangle((7.3, y - 0.4), 2.4, 0.8, fill=False, edgecolor="crimson"))
        ax.text(8.5, y, "outer-test\n(evaluated once)", ha="center", va="center", fontsize=8, color="crimson")
    ax.set_title("Nested subject-independent cross-validation")
    return _savefig(fig, name)


def fig4_model_comparison_ci(summary_df, metric="accuracy", name=None):
    """summary_df: index/column 'model', plus f'{metric}_mean' and CI cols
    as produced by statistics.summarize_repeats applied per model."""
    name = name or f"fig4_model_comparison_{metric}.png"
    fig, ax = plt.subplots(figsize=(8, 4))
    y_pos = np.arange(len(summary_df))
    means = summary_df[f"{metric}_mean"]
    lo = means - summary_df[f"{metric}_ci_low"]
    hi = summary_df[f"{metric}_ci_high"] - means
    ax.errorbar(means, y_pos, xerr=[lo, hi], fmt="o", capsize=4)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(summary_df["model"])
    ax.set_xlabel(metric)
    ax.set_title(f"Model comparison ({metric}, 95% CI)")
    ax.grid(axis="x", alpha=0.3)
    return _savefig(fig, name)


def fig5_subject_level_roc(subject_dfs_by_model, name="fig5_subject_roc.png"):
    from sklearn.metrics import roc_auc_score, roc_curve
    fig, ax = plt.subplots(figsize=(6, 6))
    for model_name, sdf in subject_dfs_by_model.items():
        if sdf["y_true"].nunique() < 2:
            continue
        fpr, tpr, _ = roc_curve(sdf["y_true"], sdf["mean_proba"])
        auc = roc_auc_score(sdf["y_true"], sdf["mean_proba"])
        ax.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.2f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Subject-level ROC")
    ax.legend(fontsize=8)
    return _savefig(fig, name)


def fig6_subject_confusion_matrices(subject_dfs_by_model, pred_col="pred_mean_proba", name="fig6_confusion.png"):
    from sklearn.metrics import confusion_matrix
    n = len(subject_dfs_by_model)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, (model_name, sdf) in zip(axes, subject_dfs_by_model.items()):
        cm = confusion_matrix(sdf["y_true"], sdf[pred_col], labels=[0, 1])
        im = ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Control", "ADHD"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Control", "ADHD"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(model_name)
    fig.suptitle("Subject-level confusion matrices")
    return _savefig(fig, name)


def fig7_repeated_cv_distribution(per_fold_metric_by_model, metric_name="accuracy", name=None):
    name = name or f"fig7_repeated_cv_distribution_{metric_name}.png"
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = list(per_fold_metric_by_model.keys())
    data = [np.asarray(v, dtype=float) for v in per_fold_metric_by_model.values()]
    ax.boxplot(data, labels=labels, showmeans=True)
    ax.set_ylabel(metric_name)
    ax.set_title(f"Distribution across repeated outer folds ({metric_name})")
    plt.xticks(rotation=30, ha="right")
    return _savefig(fig, name)


def fig8_resolution_comparison(df_a, df_b, metric="accuracy", labels=("128 Hz", "128->512 Hz interpolation"), name="fig8_resolution.png"):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot([df_a[metric].values, df_b[metric].values], labels=labels, showmeans=True)
    ax.set_ylabel(metric)
    ax.set_title("Temporal resolution sensitivity experiment")
    return _savefig(fig, name)


def fig9_ablation(ablation_summary_df, metric="accuracy", name=None):
    name = name or f"fig9_ablation_{metric}.png"
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(ablation_summary_df["ablation"], ablation_summary_df[f"{metric}_mean"],
           yerr=ablation_summary_df[f"{metric}_sd"], capsize=4)
    ax.set_ylabel(metric)
    ax.set_title("Ablation study")
    plt.xticks(rotation=30, ha="right")
    return _savefig(fig, name)


def fig10_coral_covariance(coral_df, name="fig10_coral_covariance.png"):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(["Before CORAL", "After CORAL"],
                [coral_df["cov_distance_before"].mean(), coral_df["cov_distance_after"].mean()],
                yerr=[coral_df["cov_distance_before"].std(), coral_df["cov_distance_after"].std()], capsize=4)
    axes[0].set_title("Covariance distance (train vs test features)")
    axes[1].boxplot([coral_df["no_coral_accuracy"].values, coral_df["with_coral_accuracy"].values],
                     labels=["No CORAL", "With CORAL"], showmeans=True)
    axes[1].set_title("Accuracy: with vs without CORAL")
    return _savefig(fig, name)
