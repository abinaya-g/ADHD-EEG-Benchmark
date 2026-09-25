"""
Subject-level paired model comparison (Table 5, plus the direct CORAL
comparison used in Section 4.7).

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
at 0.5 -- see evaluation.aggregate_subject_level), and computes two
tests per comparison:

  - PRIMARY: a paired subject-level permutation test. Independently for
    each subject, the two models' predictions are swapped with
    probability 0.5 (testing exchangeability of model identity per
    subject); the two-sided p-value is P(|diff_perm| >= |diff_obs|)
    under the resulting null distribution. This is the primary
    inferential test reported (Section 3.17): it makes no distributional
    assumption beyond exchangeability under the null, which is the
    minimal assumption a paired comparison of this kind requires.
  - UNCERTAINTY ESTIMATE (not a second p-value): a paired subject-level
    bootstrap. Resample the 121 subject indices with replacement (the
    SAME resampled indices applied to both models' predictions,
    preserving the pairing), recompute both models' balanced accuracy on
    each resample, and take the distribution of differences. Reports the
    bootstrap mean/median difference, the 95% percentile CI, and a
    bootstrap-based standardized effect size (mean diff / bootstrap SD).
    A bootstrap p-value is also computed and reported for transparency,
    but the permutation p-value is the one Holm-corrected and referred to
    in the manuscript text as "the" p-value for a comparison.

Only the native 128 Hz condition is compared here, against the plain CNN
as the pre-specified reference model. CNN+LR_noCORAL is excluded from
this family: it is numerically identical to CNN+LR (both are the same
L2-penalized logistic regression fit on the same features; Section 3.18
confirms their metrics agree to at least six decimal places), so
including both as separate hypotheses would test the same null twice and
inflate the family size without adding information. The family is
therefore 10 comparisons, not 11; Holm-Bonferroni correction is applied
within that 10-comparison family, separately from Table 6's
resolution-sensitivity comparisons (a different scientific question,
corrected within its own 2-comparison family) and separately from the
single, direct CNN+LR-vs-CNN+LR_withCORAL comparison (Section 4.7),
which is its own one-comparison family.
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


def _one_comparison(y_true, pred_a, pred_b, n, n_boot, n_perm, seed):
    """Shared core for one paired comparison: observed diff, bootstrap CI
    (+ bootstrap p as a secondary, non-primary quantity), and the primary
    permutation p-value."""
    ba_a_obs = balanced_accuracy_scalar(y_true, pred_a)
    ba_b_obs = balanced_accuracy_scalar(y_true, pred_b)
    obs_diff = ba_a_obs - ba_b_obs

    rng_boot = np.random.default_rng(seed)
    idx = rng_boot.integers(0, n, size=(n_boot, n))
    yt_boot = y_true[idx]
    ba_a_boot = balanced_accuracy_vec(yt_boot, pred_a[idx])
    ba_b_boot = balanced_accuracy_vec(yt_boot, pred_b[idx])
    boot_diffs = ba_a_boot - ba_b_boot
    boot_diffs = boot_diffs[~np.isnan(boot_diffs)]
    ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])
    boot_p = min(2 * min((boot_diffs <= 0).mean(), (boot_diffs >= 0).mean()), 1.0)
    boot_sd = boot_diffs.std(ddof=1)
    effect_size_std = obs_diff / boot_sd if boot_sd > 0 else np.nan

    rng_perm = np.random.default_rng(seed + 1)
    swap = rng_perm.random((n_perm, n)) < 0.5
    pred_a_tile = np.tile(pred_a, (n_perm, 1))
    pred_b_tile = np.tile(pred_b, (n_perm, 1))
    y_true_tile = np.tile(y_true, (n_perm, 1))
    perm_a = np.where(swap, pred_b_tile, pred_a_tile)
    perm_b = np.where(swap, pred_a_tile, pred_b_tile)
    perm_diffs = balanced_accuracy_vec(y_true_tile, perm_a) - balanced_accuracy_vec(y_true_tile, perm_b)
    perm_p = (np.abs(perm_diffs) >= abs(obs_diff)).mean()

    return {
        "n_subjects": n,
        "balanced_accuracy_a": ba_a_obs,
        "balanced_accuracy_b": ba_b_obs,
        "mean_diff": obs_diff,
        "boot_mean_diff": boot_diffs.mean(),
        "boot_median_diff": np.median(boot_diffs),
        "diff_ci_low_95": ci_low,
        "diff_ci_high_95": ci_high,
        "effect_size_std_boot": effect_size_std,
        "p_value_permutation": perm_p,
        "p_value_bootstrap": boot_p,
    }


def compare_all_vs_reference(df128, reference_model="CNN",
                              n_boot=N_BOOT_DEFAULT, n_perm=N_PERM_DEFAULT,
                              seed=SEED_DEFAULT, exclude=("CNN+LR_noCORAL",)):
    """The Table 5 family: every model vs. the plain CNN, excluding any
    model in `exclude` (default: CNN+LR_noCORAL, numerically identical to
    CNN+LR -- see module docstring)."""
    tables = build_subject_tables(df128)
    ref = tables[reference_model]
    y_true = ref["y_true"].values.astype(int)
    pred_ref = ref["y_pred"].values.astype(int)
    n = len(y_true)

    other_models = [m for m in sorted(tables.keys()) if m != reference_model and m not in exclude]
    rows = []
    for m in other_models:
        g = tables[m]
        assert (g["subject_id"].values == ref["subject_id"].values).all(), \
            f"{m} does not share the reference model's subject ordering"
        pred_m = g["y_pred"].values.astype(int)
        result = _one_comparison(y_true, pred_m, pred_ref, n, n_boot, n_perm, seed)
        rows.append({
            "comparison": f"{reference_model} vs {m}",
            "model_a": m, "model_b": reference_model,
            "n_subjects": result["n_subjects"],
            "balanced_accuracy_a": result["balanced_accuracy_a"],
            f"balanced_accuracy_{reference_model.lower()}": result["balanced_accuracy_b"],
            "mean_diff": result["mean_diff"],
            "boot_mean_diff": result["boot_mean_diff"],
            "boot_median_diff": result["boot_median_diff"],
            "diff_ci_low_95": result["diff_ci_low_95"],
            "diff_ci_high_95": result["diff_ci_high_95"],
            "effect_size_std_boot": result["effect_size_std_boot"],
            "p_value_permutation": result["p_value_permutation"],
            "p_value_bootstrap": result["p_value_bootstrap"],
        })

    out = pd.DataFrame(rows)
    out["p_value_permutation_holm"] = holm_correction(out["p_value_permutation"].values)
    out["p_value_bootstrap_holm"] = holm_correction(out["p_value_bootstrap"].values)
    return out.sort_values("p_value_permutation").reset_index(drop=True)


def compare_two_models(df128, model_a, model_b, n_boot=N_BOOT_DEFAULT,
                        n_perm=N_PERM_DEFAULT, seed=SEED_DEFAULT):
    """A single, direct, pre-specified paired comparison between two named
    models (e.g. CNN+LR vs. CNN+LR_withCORAL, Section 4.7) -- its own
    one-comparison family, so Holm correction across a family of size 1 is
    a no-op and the raw permutation/bootstrap p-values are reported as-is."""
    tables = build_subject_tables(df128)
    a, b = tables[model_a], tables[model_b]
    assert (a["subject_id"].values == b["subject_id"].values).all(), \
        f"{model_a} and {model_b} do not share the same subject ordering"
    y_true = a["y_true"].values.astype(int)
    n = len(y_true)
    pred_a = a["y_pred"].values.astype(int)
    pred_b = b["y_pred"].values.astype(int)
    result = _one_comparison(y_true, pred_a, pred_b, n, n_boot, n_perm, seed)
    result["comparison"] = f"{model_a} vs {model_b}"
    result["model_a"] = model_a
    result["model_b"] = model_b
    return result


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
    print("=== Table 5: vs. plain CNN (10-comparison family) ===")
    print(result.round(4).to_string(index=False))
    print(f"\nWrote {out_path}")

    coral_out_path = out_path.replace(
        "TABLE_5_MODEL_COMPARISON_STATISTICS.csv", "TABLE_5B_CORAL_DIRECT_COMPARISON.csv"
    )
    coral_result = compare_two_models(df128, "CNN+LR", "CNN+LR_withCORAL")
    pd.DataFrame([coral_result]).to_csv(coral_out_path, index=False)
    print("\n=== Direct comparison: CNN+LR vs. CNN+LR_withCORAL (single comparison) ===")
    for k, v in coral_result.items():
        print(f"  {k}: {v}")
    print(f"\nWrote {coral_out_path}")
