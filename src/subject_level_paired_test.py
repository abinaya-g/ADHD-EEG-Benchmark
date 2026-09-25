"""
Subject-level paired model comparison (Table 5).

Replaces an earlier version of this analysis that ran a Wilcoxon
signed-rank test on 25 "paired" observations (5 outer folds x 5
repetitions) at the native 128 Hz condition. Those 25 values are NOT
independent observations: the same 121 subjects are reused, just
reshuffled into different folds, across all 5 repetitions, so treating
them as 25 independent paired samples understates the true uncertainty
(pseudo-replication) and can produce artificially small p-values.

This module instead builds exactly ONE paired observation per subject
(n = 121), using the same per-subject mean-probability aggregation
already used for Table 3 and Table 4 (pooling that subject's predicted
probability across all epochs and all 5 repetitions, then thresholding
at 0.5 -- see evaluation.aggregate_subject_level), and computes:

  - the observed balanced-accuracy difference (model vs. CNN)
  - a paired subject-level bootstrap: resample the 121 subject indices
    with replacement (SAME resampled indices applied to both models'
    predictions, preserving the pairing), recompute both models'
    balanced accuracy on each resample, and take the distribution of
    differences. Reports the bootstrap mean/median difference, the 95%
    percentile CI, and a two-sided bootstrap p-value
    (2 * min(P(diff<=0), P(diff>=0)), capped at 1 -- equivalent to
    inverting the percentile CI).
  - a paired subject-level permutation test: independently for each
    subject, swap which model "produced" that subject's prediction with
    probability 0.5 (testing exchangeability of model identity per
    subject), and compute the two-sided p-value as
    P(|diff_perm| >= |diff_obs|) under the resulting null distribution.
  - a bootstrap-based standardized effect size (mean diff / bootstrap SD).

Only the native 128 Hz condition is compared here, against the plain CNN
as the pre-specified reference model (11 comparisons); Holm-Bonferroni
correction is applied within that 11-comparison family for both the
bootstrap and the permutation p-values, separately from Table 6's
resolution-sensitivity comparisons (a different scientific question,
corrected within its own 2-comparison family).
"""
import numpy as np
import pandas as pd

N_BOOT_DEFAULT = 20000
N_PERM_DEFAULT = 20000
SEED_DEFAULT = 12345


def holm_correction(p_values):
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(n)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (n - rank) * p_values[idx]
        running_max = max(running_max, adj)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted


def build_subject_tables(df128):
    """One row per subject per model: y_true, mean predicted probability
    pooled across all epochs and repetitions, and the thresholded
    prediction -- identical aggregation to Table 3 / Table 4."""
    tables = {}
    for m in sorted(df128["model"].unique()):
        sub = df128[df128["model"] == m]
        g = sub.groupby("subject_id").agg(
            y_true=("y_true", "first"), mean_proba=("y_proba", "mean")
        ).reset_index()
        g["y_pred"] = (g["mean_proba"] >= 0.5).astype(int)
        g = g.sort_values("subject_id").reset_index(drop=True)
        tables[m] = g
    return tables


def balanced_accuracy_vec(y_true_mat, pred_mat):
    """y_true_mat, pred_mat: (B, n) 0/1 arrays -> (B,) balanced accuracy."""
    pos = y_true_mat == 1
    neg = y_true_mat == 0
    correct = pred_mat == y_true_mat
    n_pos = pos.sum(axis=1)
    n_neg = neg.sum(axis=1)
    tpr = np.where(n_pos > 0, (correct & pos).sum(axis=1) / np.maximum(n_pos, 1), np.nan)
    tnr = np.where(n_neg > 0, (correct & neg).sum(axis=1) / np.maximum(n_neg, 1), np.nan)
    return 0.5 * (tpr + tnr)


def balanced_accuracy_scalar(y_true, pred):
    pos = y_true == 1
    neg = y_true == 0
    tpr = (pred[pos] == 1).mean()
    tnr = (pred[neg] == 0).mean()
    return 0.5 * (tpr + tnr)


def compare_all_vs_reference(df128, reference_model="CNN",
                              n_boot=N_BOOT_DEFAULT, n_perm=N_PERM_DEFAULT,
                              seed=SEED_DEFAULT):
    tables = build_subject_tables(df128)
    ref = tables[reference_model]
    y_true = ref["y_true"].values.astype(int)
    pred_ref = ref["y_pred"].values.astype(int)
    n = len(y_true)
    ba_ref_obs = balanced_accuracy_scalar(y_true, pred_ref)

    other_models = [m for m in sorted(tables.keys()) if m != reference_model]
    rows = []
    for m in other_models:
        g = tables[m]
        assert (g["subject_id"].values == ref["subject_id"].values).all(), \
            f"{m} does not share the reference model's subject ordering"
        pred_m = g["y_pred"].values.astype(int)
        ba_m_obs = balanced_accuracy_scalar(y_true, pred_m)
        obs_diff = ba_m_obs - ba_ref_obs

        rng_boot = np.random.default_rng(seed)
        idx = rng_boot.integers(0, n, size=(n_boot, n))
        yt_boot = y_true[idx]
        ba_a_boot = balanced_accuracy_vec(yt_boot, pred_m[idx])
        ba_b_boot = balanced_accuracy_vec(yt_boot, pred_ref[idx])
        boot_diffs = ba_a_boot - ba_b_boot
        boot_diffs = boot_diffs[~np.isnan(boot_diffs)]
        ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])
        boot_p = min(2 * min((boot_diffs <= 0).mean(), (boot_diffs >= 0).mean()), 1.0)
        boot_sd = boot_diffs.std(ddof=1)
        effect_size_std = obs_diff / boot_sd if boot_sd > 0 else np.nan

        rng_perm = np.random.default_rng(seed + 1)
        swap = rng_perm.random((n_perm, n)) < 0.5
        pred_m_tile = np.tile(pred_m, (n_perm, 1))
        pred_ref_tile = np.tile(pred_ref, (n_perm, 1))
        y_true_tile = np.tile(y_true, (n_perm, 1))
        perm_a = np.where(swap, pred_ref_tile, pred_m_tile)
        perm_b = np.where(swap, pred_m_tile, pred_ref_tile)
        perm_diffs = balanced_accuracy_vec(y_true_tile, perm_a) - balanced_accuracy_vec(y_true_tile, perm_b)
        perm_p = (np.abs(perm_diffs) >= abs(obs_diff)).mean()

        rows.append({
            "comparison": f"{reference_model} vs {m}",
            "model_a": m, "model_b": reference_model,
            "n_subjects": n,
            "balanced_accuracy_a": ba_m_obs,
            f"balanced_accuracy_{reference_model.lower()}": ba_ref_obs,
            "mean_diff": obs_diff,
            "boot_mean_diff": boot_diffs.mean(),
            "boot_median_diff": np.median(boot_diffs),
            "diff_ci_low_95": ci_low,
            "diff_ci_high_95": ci_high,
            "effect_size_std_boot": effect_size_std,
            "p_value_bootstrap": boot_p,
            "p_value_permutation": perm_p,
        })

    out = pd.DataFrame(rows)
    out["p_value_bootstrap_holm"] = holm_correction(out["p_value_bootstrap"].values)
    out["p_value_permutation_holm"] = holm_correction(out["p_value_permutation"].values)
    return out.sort_values("p_value_bootstrap").reset_index(drop=True)


if __name__ == "__main__":
    import sys
    raw_path = sys.argv[1] if len(sys.argv) > 1 else "results_final/predictions/all_predictions.csv"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "manuscript/TABLES/TABLE_5_MODEL_COMPARISON_STATISTICS.csv"
    df = pd.read_csv(raw_path)
    df128 = df[df["preprocessing"] == "128Hz"]
    result = compare_all_vs_reference(df128)
    result.to_csv(out_path, index=False)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print(result.round(4).to_string(index=False))
    print(f"\nWrote {out_path}")
