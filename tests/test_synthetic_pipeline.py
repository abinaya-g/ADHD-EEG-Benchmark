"""
SYNTHETIC SELF-TEST -- NOT REAL EEG DATA, NOT A MANUSCRIPT RESULT.

This script exists because the sandboxed environment this framework was
developed in has no access to the real ADHD-EEG dataset (it is hosted on
Kaggle at abinayajone/adhd-eeg-dataset and was never downloadable from
here) and no GPU. Every number this script prints is computed from
src.data.make_synthetic_dataset()'s randomly generated surrogate data and
exists ONLY to prove the pipeline's mechanics are correct (no leakage,
correct shapes, correct aggregation, correct statistics) end-to-end before
a user runs run_all_experiments.py on the real dataset. Do not copy any
number from this script's output into a manuscript, table, or figure.

Run: python3 tests/test_synthetic_pipeline.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config, data, evaluation, models, sanity_checks, statistics, coral, explainability

SYN_N_CHANNELS = 19
SYN_N_TIMESAMPLES = 768   # short synthetic epoch, same fs-scaling logic as real pipeline (fs=128)
SYN_FS = 128
OUTER_FOLDS = 3
INNER_FOLDS = 3
REPEATS = 2
SEEDS = [42, 43]
MAX_EPOCHS = 6
BATCH_SIZE = 8
PATIENCE = 2
LR = 1e-3  # higher LR than the real config so the tiny synthetic net moves in a few epochs


def banner(msg):
    print("\n" + "=" * 78)
    print(msg)
    print("=" * 78)


def main():
    t0 = time.time()
    banner("SYNTHETIC SELF-TEST -- generating surrogate dataset (NOT real EEG data)")
    X, y, groups, file_epoch_idx = data.make_synthetic_dataset(
        n_adhd_subjects=15, n_control_subjects=15, min_epochs=3, max_epochs=5,
        n_channels=SYN_N_CHANNELS, n_timesamples=SYN_N_TIMESAMPLES, class_signal=0.9, seed=0,
    )
    print(f"X: {X.shape}  y: {y.shape}  unique subjects: {len(set(groups))}")
    print(f"class balance: ADHD={int((y==1).sum())} Control={int((y==0).sum())}")

    # --- Phase 20 check 14 precondition: verify no duplicate (subject, epoch_idx) pairs
    pair_df = pd.DataFrame({"subject": groups, "epoch_idx": file_epoch_idx})
    assert not pair_df.duplicated().any(), "duplicate (subject, epoch_idx) pairs found in synthetic data"
    print("[OK] no duplicate (subject, epoch_idx) pairs in synthetic data")

    X_norm = data.normalize_all(X)
    before_mean, before_std = X[0, 0, :, 0].mean(), X[0, 0, :, 0].std()
    after_mean, after_std = X_norm[0, 0, :, 0].mean(), X_norm[0, 0, :, 0].std()
    print(f"normalize_epoch sanity: before mean/std=({before_mean:.3f},{before_std:.3f}) "
          f"after=({after_mean:.3f},{after_std:.3f}) [after std should be ~1.0]")
    assert abs(after_std - 1.0) < 1e-3

    all_prediction_rows = []
    all_fold_records = []
    all_coral_rows = []

    banner("Nested + repeated subject-independent CV: CNN (leakage-corrected)")
    for seed in SEEDS:
        repeat_id = SEEDS.index(seed)
        preds, folds, coral_rows = evaluation.run_nested_repeat(
            "CNN", models.build_cnn, X_norm, y, groups,
            outer_folds=OUTER_FOLDS, inner_folds=INNER_FOLDS, seed=seed, repeat_id=repeat_id,
            preprocessing_label="128Hz_synthetic", fs=SYN_FS, n_timesamples=SYN_N_TIMESAMPLES,
            max_epochs=MAX_EPOCHS, batch_size=BATCH_SIZE, patience=PATIENCE, learning_rate=LR,
            run_classifiers=True, run_coral=True,
        )
        all_prediction_rows.extend(preds)
        all_fold_records.extend(folds)
        all_coral_rows.extend(coral_rows)
        print(f"repeat {repeat_id} (seed={seed}): {len(folds)} outer folds, "
              f"selected_epochs={[f['selected_epochs'] for f in folds]}")

    predictions_df = pd.DataFrame(all_prediction_rows)
    coral_df = pd.DataFrame(all_coral_rows)
    predictions_df.to_csv(os.path.join(config.RESULTS_DIR, "SYNTHETIC_predictions.csv"), index=False)
    coral_df.to_csv(os.path.join(config.RESULTS_DIR, "SYNTHETIC_coral.csv"), index=False)
    print(f"saved {len(predictions_df)} prediction rows -> results/SYNTHETIC_predictions.csv")

    banner("Phase 20 sanity checks")
    report = sanity_checks.run_sanity_checks(all_fold_records, predictions_df)
    assert report["passed"].all(), "one or more sanity checks FAILED -- see report above"

    banner("Epoch-level pooled metrics per model (reconstructed from saved predictions)")
    for model_name, grp in predictions_df.groupby("model"):
        m = evaluation.compute_metrics(grp["y_true"], grp["y_pred"], grp["y_proba"])
        print(f"{model_name:20s} acc={m['accuracy']:.3f} bal_acc={m['balanced_accuracy']:.3f} "
              f"sens={m['sensitivity']:.3f} spec={m['specificity']:.3f} mcc={m['mcc']:.3f} auc={m['auc']:.3f}")

    banner("Subject-level aggregation + bootstrap 95% CI (Phase 8/9)")
    cnn_rows = predictions_df[predictions_df["model"] == "CNN"]
    subj_df = evaluation.aggregate_subject_level(cnn_rows["subject_id"], cnn_rows["y_true"], cnn_rows["y_proba"])
    print(subj_df.head())

    def acc_metric_fn(df):
        from sklearn.metrics import accuracy_score
        return accuracy_score(df["y_true"], df["pred_mean_proba"])

    ci = statistics.bootstrap_subject_level_ci(subj_df, acc_metric_fn, n_boot=500, seed=1)
    print(f"CNN subject-level accuracy: point={ci['point_estimate']:.3f} "
          f"95% CI=[{ci['ci_low']:.3f}, {ci['ci_high']:.3f}] (n_subjects={ci['n_subjects']}, n_boot={ci['n_boot']})")

    # Deliberate check: aggregation must raise on inconsistent per-subject labels.
    bad_df = cnn_rows.copy()
    bad_df.iloc[0, bad_df.columns.get_loc("y_true")] = 1 - bad_df.iloc[0]["y_true"]
    try:
        evaluation.aggregate_subject_level(bad_df["subject_id"], bad_df["y_true"], bad_df["y_proba"])
        raise AssertionError("aggregate_subject_level should have raised on inconsistent labels")
    except ValueError:
        print("[OK] aggregate_subject_level correctly rejects inconsistent per-subject labels")

    banner("Statistical comparison: CNN vs LR, per-outer-fold accuracy (Phase 10)")
    per_fold_acc = {}
    for model_name in ["CNN", "CNN+LR", "CNN+NLSVM"]:
        sub = predictions_df[predictions_df["model"] == model_name]
        accs = []
        for (rep, fold), g in sub.groupby(["repeat", "outer_fold"]):
            accs.append(evaluation.compute_metrics(g["y_true"], g["y_pred"])["accuracy"])
        per_fold_acc[model_name] = accs
    cmp_df = statistics.compare_all_models(per_fold_acc, metric_name="accuracy", reference_model="CNN")
    print(cmp_df[["model_a", "model_b", "mean_diff", "p_value", "p_value_holm", "effect_size_rank_biserial"]])

    banner("CORAL experiment (transductive, Phase 13)")
    print(coral_df[["cov_distance_before", "cov_distance_after", "no_coral_accuracy", "with_coral_accuracy"]])
    assert (coral_df["cov_distance_after"] <= coral_df["cov_distance_before"]).all(), \
        "CORAL should reduce train/test feature covariance distance"
    print("[OK] CORAL reduced covariance distance in every fold")

    banner("Repeated CV summary stats (Phase 5)")
    summary = statistics.summarize_repeats(per_fold_acc["CNN"])
    print(f"CNN accuracy across {summary['n']} outer folds: mean={summary['mean']:.3f} "
          f"median={summary['median']:.3f} sd={summary['sd']:.3f} "
          f"95% CI=[{summary['ci_low']:.3f},{summary['ci_high']:.3f}] "
          f"min={summary['min']:.3f} max={summary['max']:.3f}")

    banner("128 -> 512-equivalent interpolation sensitivity experiment (Phase 3) -- shape check only")
    X_interp = data.resample_pipeline_b(X_norm[:6], orig_fs=SYN_FS, target_fs=SYN_FS * 2)
    print(f"Original shape: {X_norm[:6].shape} -> interpolated shape: {X_interp.shape}")
    assert X_interp.shape[2] == SYN_N_TIMESAMPLES * 2
    print("[OK] interpolation produces the expected doubled sample count "
          "(labelled everywhere as interpolation, never as true higher-fs acquisition)")

    banner("Ablation study smoke test (Phase 12): full / no-spatial / no-temporal")
    for ablation_name, cfg in models.ABLATION_CONFIGS.items():
        if ablation_name not in ("A_full", "B_no_spatial", "C_no_temporal"):
            continue
        build_fn = lambda n_channels, n_timesamples, fs, _cfg=cfg: models.build_cnn_ablation(
            n_channels=n_channels, n_timesamples=n_timesamples, fs=fs, **_cfg)
        preds, folds, _ = evaluation.run_nested_repeat(
            f"CNN_{ablation_name}", build_fn, X_norm, y, groups,
            outer_folds=OUTER_FOLDS, inner_folds=INNER_FOLDS, seed=99, repeat_id=0,
            preprocessing_label="128Hz_synthetic", fs=SYN_FS, n_timesamples=SYN_N_TIMESAMPLES,
            max_epochs=MAX_EPOCHS, batch_size=BATCH_SIZE, patience=PATIENCE, learning_rate=LR,
            run_classifiers=False, run_coral=False,
        )
        p_df = pd.DataFrame(preds)
        m = evaluation.compute_metrics(p_df["y_true"], p_df["y_pred"], p_df["y_proba"])
        print(f"{ablation_name:16s} acc={m['accuracy']:.3f} auc={m['auc']:.3f}")

    banner("Alternative architecture smoke test (Phase 6): EEGNet / ShallowConvNet / DeepConvNet")
    for arch_name, build_fn in models.ARCHITECTURE_BUILDERS.items():
        if arch_name == "CNN":
            continue
        preds, folds, _ = evaluation.run_nested_repeat(
            arch_name, build_fn, X_norm, y, groups,
            outer_folds=OUTER_FOLDS, inner_folds=INNER_FOLDS, seed=7, repeat_id=0,
            preprocessing_label="128Hz_synthetic", fs=SYN_FS, n_timesamples=SYN_N_TIMESAMPLES,
            max_epochs=3, batch_size=BATCH_SIZE, patience=1, learning_rate=LR,
            run_classifiers=False, run_coral=False,
        )
        p_df = pd.DataFrame(preds)
        m = evaluation.compute_metrics(p_df["y_true"], p_df["y_pred"], p_df["y_proba"])
        print(f"{arch_name:16s} acc={m['accuracy']:.3f} auc={m['auc']:.3f}  (n_params={build_fn(19, SYN_N_TIMESAMPLES, SYN_FS)[0].count_params()})")

    banner("Explainability smoke test (Phase 15): Integrated Gradients")
    model_final, _ = models.build_cnn(n_channels=SYN_N_CHANNELS, n_timesamples=SYN_N_TIMESAMPLES, fs=SYN_FS)
    import tensorflow as tf
    model_final.compile(optimizer=tf.keras.optimizers.Adam(LR), loss="binary_crossentropy")
    model_final.fit(X_norm[:20], y[:20], epochs=2, batch_size=8, verbose=0)
    ig_result = explainability.batch_channel_temporal_importance(model_final, X_norm[:8], max_samples=8, steps=10)
    print(f"channel_importance_mean shape: {ig_result['channel_importance_mean'].shape} "
          f"(expected ({SYN_N_CHANNELS},))")
    print(f"temporal_importance_mean shape: {ig_result['temporal_importance_mean'].shape}")
    assert ig_result["channel_importance_mean"].shape == (SYN_N_CHANNELS,)

    banner(f"ALL SYNTHETIC SELF-TESTS PASSED in {time.time()-t0:.1f}s "
           "-- reminder: every number above is from synthetic surrogate data, not real EEG")


if __name__ == "__main__":
    main()
