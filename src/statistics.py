"""
Phase 9 (confidence intervals) and Phase 10 (statistical testing).

All bootstrap resampling here is at the SUBJECT level, never the epoch
level: epochs from the same participant are not independent observations,
so epoch-level bootstrapping would understate the true sampling
variability (pseudo-replication). See Phase 9 of the review brief.
"""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def summarize_repeats(values):
    """Mean/median/SD/95% CI (normal approx across repeats)/min/max for a
    metric measured once per outer fold or per repeat (Phase 5 table)."""
    values = np.asarray([v for v in values if not np.isnan(v)], dtype=float)
    if len(values) == 0:
        return {k: np.nan for k in ("mean", "median", "sd", "ci_low", "ci_high", "min", "max", "n")}
    mean = values.mean()
    sd = values.std(ddof=1) if len(values) > 1 else 0.0
    se = sd / np.sqrt(len(values)) if len(values) > 1 else 0.0
    return {
        "mean": mean, "median": float(np.median(values)), "sd": sd,
        "ci_low": mean - 1.96 * se, "ci_high": mean + 1.96 * se,
        "min": float(values.min()), "max": float(values.max()), "n": len(values),
    }


def bootstrap_subject_level_ci(subject_df, metric_fn, n_boot=2000, alpha=0.05, seed=0):
    """
    subject_df: one row per subject with at least y_true and a prediction
        column (see evaluation.aggregate_subject_level output).
    metric_fn: callable(y_true, y_pred_or_proba_array-like-per-row) -> float
        Applied to a resample of SUBJECTS (with replacement), never
        individual epochs.
    Returns dict with point estimate, mean, sd, and percentile 95% CI.
    """
    rng = np.random.default_rng(seed)
    n = len(subject_df)
    point = metric_fn(subject_df)
    boot_vals = np.empty(n_boot)
    idx_all = np.arange(n)
    for b in range(n_boot):
        sample_idx = rng.choice(idx_all, size=n, replace=True)
        boot_vals[b] = metric_fn(subject_df.iloc[sample_idx])
    lo, hi = np.percentile(boot_vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "point_estimate": point, "boot_mean": float(np.nanmean(boot_vals)),
        "boot_sd": float(np.nanstd(boot_vals)), "ci_low": float(lo), "ci_high": float(hi),
        "n_boot": n_boot, "n_subjects": n,
    }


def rank_biserial_effect_size(diffs):
    """Matched-pairs rank-biserial correlation for a Wilcoxon signed-rank
    test: r = (W+ - W-) / (W+ + W-), computed from the signed ranks of the
    nonzero differences. Ranges [-1, 1]; sign follows `diffs`' sign
    convention (diffs = a - b => positive r favors a)."""
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[diffs != 0]
    if len(diffs) == 0:
        return np.nan
    ranks = pd.Series(np.abs(diffs)).rank().values
    w_pos = ranks[diffs > 0].sum()
    w_neg = ranks[diffs < 0].sum()
    total = w_pos + w_neg
    return float((w_pos - w_neg) / total) if total > 0 else np.nan


def paired_model_comparison(values_a, values_b, name_a, name_b):
    """One paired comparison (e.g. per-outer-fold accuracy, CNN vs LR)."""
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    assert len(a) == len(b), "Paired comparison requires equal-length, aligned arrays (same folds/repeats)"
    diffs = a - b
    result = {
        "model_a": name_a, "model_b": name_b, "n_pairs": len(a),
        "mean_diff": float(diffs.mean()), "sd_diff": float(diffs.std(ddof=1)) if len(diffs) > 1 else 0.0,
        "effect_size_rank_biserial": rank_biserial_effect_size(diffs),
    }
    se = result["sd_diff"] / np.sqrt(len(diffs)) if len(diffs) > 1 else 0.0
    result["diff_ci_low"] = result["mean_diff"] - 1.96 * se
    result["diff_ci_high"] = result["mean_diff"] + 1.96 * se
    if np.allclose(diffs, 0):
        result["wilcoxon_stat"], result["p_value"] = np.nan, 1.0
    else:
        try:
            stat, p = wilcoxon(a, b)
            result["wilcoxon_stat"], result["p_value"] = float(stat), float(p)
        except ValueError:
            result["wilcoxon_stat"], result["p_value"] = np.nan, np.nan
    return result


def holm_correction(p_values):
    """Holm-Bonferroni step-down correction. Returns adjusted p-values in
    the SAME order as the input."""
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


def compare_all_models(per_fold_metric_by_model, metric_name="accuracy", reference_model=None):
    """
    per_fold_metric_by_model: dict[model_name] -> array-like of per-fold
        (or per-repeat) values, all aligned to the SAME fold/repeat order.
    reference_model: if given, only compare every other model against it
        (e.g. "CNN"); otherwise all pairwise comparisons are produced.
    Returns a DataFrame with one row per comparison, Holm-corrected
    p-values, and effect sizes. Never states "no significant difference"
    as evidence of equivalence -- callers should phrase results as "no
    statistically significant difference was detected" per Phase 10.
    """
    names = list(per_fold_metric_by_model.keys())
    pairs = []
    if reference_model is not None:
        pairs = [(reference_model, m) for m in names if m != reference_model]
    else:
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                pairs.append((names[i], names[j]))

    rows = [paired_model_comparison(per_fold_metric_by_model[a], per_fold_metric_by_model[b], a, b)
            for a, b in pairs]
    df = pd.DataFrame(rows)
    if len(df) > 0:
        df["p_value_holm"] = holm_correction(df["p_value"].fillna(1.0).values)
        df["metric"] = metric_name
    return df
