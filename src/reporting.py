"""
Table builders for the final manuscript-ready CSVs (review-brief Phase 15:
TABLE_1 through TABLE_7). Kept separate from run_all_experiments.py so the
orchestration script stays readable; every function here only reads
already-computed predictions/fold records -- none of it trains a model or
touches outer-test labels for anything but computing a metric after the
fact.
"""
import numpy as np
import pandas as pd

from . import evaluation, statistics


def table1_dataset_summary(X, y, groups, config, outer_folds, inner_folds, repeats, seeds, max_epochs):
    return pd.DataFrame([{
        "n_subjects": len(set(groups)),
        "n_adhd_subjects": len(set(groups[y == 1])),
        "n_control_subjects": len(set(groups[y == 0])),
        "n_epochs": len(y),
        "n_adhd_epochs": int((y == 1).sum()),
        "n_control_epochs": int((y == 0).sum()),
        "n_channels": X.shape[1],
        "n_timesamples": X.shape[2],
        "assumed_fs_hz": config.FS_ASSUMED_HZ,
        "epoch_seconds": config.EPOCH_SECONDS,
        "outer_folds": outer_folds,
        "inner_folds": inner_folds,
        "repeats": repeats,
        "seeds": ";".join(str(s) for s in seeds),
        "max_epochs": max_epochs,
        "batch_size": config.BATCH_SIZE,
        "early_stopping_patience": config.EARLY_STOPPING_PATIENCE,
        "learning_rate": config.LEARNING_RATE,
    }])


def _per_fold_subject_metric(predictions_df, group_cols, metric_name="balanced_accuracy"):
    """dict[group_key] -> dict[(repeat, outer_fold)] -> subject-level metric,
    where group_key is the (model, preprocessing) tuple. Computed by
    aggregating ONLY that fold's own test-subject epochs to subject level
    first -- this is what makes the resulting values pairable across
    models/preprocessing that share the same outer-fold subject partition
    (same seed), per Phase 12's "paired outer-fold/repetition performance
    estimate" preference over epoch-level pseudo-replication."""
    out = {}
    for key, grp in predictions_df.groupby(group_cols):
        key = key if isinstance(key, tuple) else (key,)
        per_fold = {}
        for (rep, fold), fgrp in grp.groupby(["repeat", "outer_fold"]):
            sdf = evaluation.aggregate_subject_level(fgrp["subject_id"], fgrp["y_true"], fgrp["y_proba"])
            m = evaluation.compute_metrics(sdf["y_true"], sdf["pred_mean_proba"], sdf["mean_proba"])
            per_fold[(rep, fold)] = m[metric_name]
        out[key] = per_fold
    return out


def per_fold_subject_metric(predictions_df, group_cols, metric_name="balanced_accuracy"):
    """Public wrapper around _per_fold_subject_metric, for callers (e.g.
    run_all_experiments.py's figure-generation code) that need the same
    per-(repeat, outer_fold) subject-level metric used by table5/6."""
    return _per_fold_subject_metric(predictions_df, group_cols, metric_name)


def table4_confidence_intervals(predictions_df, group_cols, config, metrics=None):
    """Subject-level bootstrap 95% CI per (model, preprocessing) group,
    pooling that group's own predictions across whatever repeats/outer
    folds it has (each test subject appears exactly once per repeat, so
    pooling repeats before aggregating to subject level is the correct
    per-subject sample, not pseudo-replication)."""
    metrics = metrics or ["accuracy", "balanced_accuracy", "sensitivity", "specificity", "precision", "f1", "mcc", "auc"]
    rows = []
    for key, grp in predictions_df.groupby(group_cols):
        key = key if isinstance(key, tuple) else (key,)
        key_dict = dict(zip(group_cols, key))
        sdf = evaluation.aggregate_subject_level(grp["subject_id"], grp["y_true"], grp["y_proba"])
        for metric_name in metrics:
            def metric_fn(df, _m=metric_name):
                m = evaluation.compute_metrics(df["y_true"], df["pred_mean_proba"], df["mean_proba"])
                return m[_m]
            ci = statistics.bootstrap_subject_level_ci(sdf, metric_fn, n_boot=config.N_BOOTSTRAP, alpha=config.CI_ALPHA, seed=0)
            rows.append({**key_dict, "metric": metric_name, **ci})
    return pd.DataFrame(rows)


def table5_model_comparison(predictions_df, reference_key, group_cols, metric_name="balanced_accuracy"):
    """Paired comparisons (Holm-corrected, rank-biserial effect size)
    against one reference (model, preprocessing) group, using the
    per-outer-fold SUBJECT-level metric as the paired statistic (Phase 12).
    Two groups are only compared using the (repeat, outer_fold) pairs they
    both actually have -- e.g. a single-repeat ablation config is compared
    against only the matching repeat of the primary 5-repeat run, not
    padded with fabricated pairs."""
    per_fold = _per_fold_subject_metric(predictions_df, group_cols, metric_name)
    if reference_key not in per_fold:
        return pd.DataFrame()
    ref_folds = per_fold[reference_key]

    rows = []
    for key, folds in per_fold.items():
        if key == reference_key:
            continue
        common = sorted(set(ref_folds) & set(folds))
        if len(common) < 2:
            continue
        a = [ref_folds[k] for k in common]
        b = [folds[k] for k in common]
        cmp = statistics.paired_model_comparison(a, b, "|".join(map(str, reference_key)), "|".join(map(str, key)))
        cmp["n_paired_folds"] = len(common)
        cmp["metric"] = metric_name
        rows.append(cmp)

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df["p_value_holm"] = statistics.holm_correction(df["p_value"].fillna(1.0).values)
    return df


def table6_resolution_comparison(predictions_df, group_cols, model_name="CNN",
                                  preprocessing_a="128Hz", preprocessing_b="128to512_interp"):
    """128Hz vs 128->512Hz-interpolation, same model, subject-level metrics
    side by side plus a paired comparison where a common (repeat,
    outer_fold) partition exists between the two (true whenever the
    resolution experiment reused the primary run's seed, as
    run_all_experiments.py does by default)."""
    rows = []
    for prep in (preprocessing_a, preprocessing_b):
        sub = predictions_df[(predictions_df["model"] == model_name) & (predictions_df["preprocessing"] == prep)]
        if len(sub) == 0:
            continue
        sdf = evaluation.aggregate_subject_level(sub["subject_id"], sub["y_true"], sub["y_proba"])
        m = evaluation.compute_metrics(sdf["y_true"], sdf["pred_mean_proba"], sdf["mean_proba"])
        rows.append({"model": model_name, "preprocessing": prep, "n_subjects": len(sdf), **m})
    summary_df = pd.DataFrame(rows)

    cmp_df = table5_model_comparison(
        predictions_df[predictions_df["model"] == model_name],
        reference_key=(model_name, preprocessing_a), group_cols=group_cols,
    )
    return summary_df, cmp_df
