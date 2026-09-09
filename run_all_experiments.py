#!/usr/bin/env python3
"""
Main orchestration script for the ADHD-EEG robust benchmark (Phase 17).

Runs on the REAL dataset by default. Requires:
  - ADHD_EEG_DATA_ROOT environment variable pointing at a local copy of the
    abinayajone/adhd-eeg-dataset folder structure (see README.md), and
  - a GPU is strongly recommended (CNN training here trains one model per
    outer fold, per inner fold, per repeat, per architecture -- see
    README.md "Computational cost" for the exact count).

Usage:
    python run_all_experiments.py                 # full run, real data
    python run_all_experiments.py --smoke-test     # synthetic data, tiny
                                                    # config, just to prove
                                                    # the script runs end to
                                                    # end (NOT real results)
    python run_all_experiments.py --skip-baselines --skip-ablation ...

Every experiment appends rows to results/predictions.csv (epoch-level,
with subject_id/outer_fold/repeat/seed/model/preprocessing columns) so
every table and figure below can be regenerated later from that one file
plus the smaller per-experiment CSVs saved alongside it (Phase 2, Phase 18
requirement: "no manually typed numbers").
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import config, coral as coral_mod, data, evaluation, explainability, models, sanity_checks, statistics, visualization


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke-test", action="store_true",
                    help="Use synthetic surrogate data and a tiny config to verify the script runs end to end. "
                         "Output is NOT a real result and is prefixed SYNTHETIC_ everywhere.")
    p.add_argument("--skip-nested-cnn", action="store_true")
    p.add_argument("--skip-architectures", action="store_true", help="skip EEGNet/ShallowConvNet/DeepConvNet")
    p.add_argument("--skip-ablation", action="store_true")
    p.add_argument("--skip-resolution", action="store_true", help="skip the 128 vs 128->512-interp experiment")
    p.add_argument("--skip-coral", action="store_true")
    p.add_argument("--skip-explainability", action="store_true", help="Integrated Gradients is the most expensive optional step")
    p.add_argument("--outer-folds", type=int, default=None)
    p.add_argument("--inner-folds", type=int, default=None)
    p.add_argument("--repeats", type=int, default=None)
    p.add_argument("--max-epochs", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()
    prefix = "SYNTHETIC_" if args.smoke_test else ""

    outer_folds = args.outer_folds or config.OUTER_FOLDS
    inner_folds = args.inner_folds or config.INNER_FOLDS
    repeats = args.repeats or config.REPEATS
    seeds = config.RANDOM_SEEDS[:repeats]
    max_epochs = args.max_epochs or config.MAX_EPOCHS

    print(f"=== ADHD-EEG robust benchmark {'[SMOKE TEST -- SYNTHETIC DATA]' if args.smoke_test else ''} ===")
    print(f"outer_folds={outer_folds} inner_folds={inner_folds} repeats={repeats} seeds={seeds} max_epochs={max_epochs}")

    # -------------------------------------------------------------------
    # Data
    # -------------------------------------------------------------------
    if args.smoke_test:
        X, y, groups, file_epoch_idx = data.make_synthetic_dataset(
            n_adhd_subjects=15, n_control_subjects=15, min_epochs=3, max_epochs=5,
            n_channels=config.N_CHANNELS, n_timesamples=768, class_signal=0.9, seed=0,
        )
        fs_a, n_time_a = 128, 768
    else:
        X, y, groups, file_epoch_idx = data.load_dataset()
        fs_a, n_time_a = config.FS_ASSUMED_HZ, config.EPOCH_LEN_A

    print(f"Loaded X={X.shape} y={y.shape} unique_subjects={len(set(groups))} "
          f"ADHD_epochs={int((y==1).sum())} Control_epochs={int((y==0).sum())}")

    X_norm = data.normalize_all(X)

    all_predictions = []
    all_fold_records = []
    all_coral_rows = []

    # -------------------------------------------------------------------
    # Phase 4/5: nested + repeated CV, primary CNN, + classical classifiers
    # -------------------------------------------------------------------
    if not args.skip_nested_cnn:
        print("\n--- Nested + repeated subject-independent CV: CNN + classical classifiers ---")
        for i, seed in enumerate(seeds):
            preds, folds, coral_rows = evaluation.run_nested_repeat(
                "CNN", models.build_cnn, X_norm, y, groups,
                outer_folds=outer_folds, inner_folds=inner_folds, seed=seed, repeat_id=i,
                preprocessing_label=f"{fs_a}Hz", fs=fs_a, n_timesamples=n_time_a,
                max_epochs=max_epochs, batch_size=config.BATCH_SIZE,
                patience=config.EARLY_STOPPING_PATIENCE, learning_rate=config.LEARNING_RATE,
                run_classifiers=True, run_coral=not args.skip_coral,
            )
            all_predictions.extend(preds)
            all_fold_records.extend(folds)
            all_coral_rows.extend(coral_rows)
            print(f"  repeat {i} (seed={seed}) done: {len(folds)} outer folds")

    # -------------------------------------------------------------------
    # Phase 6: EEGNet / ShallowConvNet / DeepConvNet, same protocol
    # -------------------------------------------------------------------
    if not args.skip_architectures:
        print("\n--- Alternative EEG architectures (same nested protocol) ---")
        for arch_name, build_fn in models.ARCHITECTURE_BUILDERS.items():
            if arch_name == "CNN":
                continue
            for i, seed in enumerate(seeds):
                preds, folds, _ = evaluation.run_nested_repeat(
                    arch_name, build_fn, X_norm, y, groups,
                    outer_folds=outer_folds, inner_folds=inner_folds, seed=seed, repeat_id=i,
                    preprocessing_label=f"{fs_a}Hz", fs=fs_a, n_timesamples=n_time_a,
                    max_epochs=max_epochs, batch_size=config.BATCH_SIZE,
                    patience=config.EARLY_STOPPING_PATIENCE, learning_rate=config.LEARNING_RATE,
                    run_classifiers=False, run_coral=False,
                )
                all_predictions.extend(preds)
                all_fold_records.extend(folds)
                print(f"  {arch_name} repeat {i} (seed={seed}) done")

    # -------------------------------------------------------------------
    # Phase 12: ablation study (single repeat -- computational budget)
    # -------------------------------------------------------------------
    if not args.skip_ablation:
        print("\n--- Ablation study ---")
        ablation_seed = seeds[0]
        for ablation_name, cfg in models.ABLATION_CONFIGS.items():
            build_fn = lambda n_channels, n_timesamples, fs, _cfg=cfg: models.build_cnn_ablation(
                n_channels=n_channels, n_timesamples=n_timesamples, fs=fs, **_cfg)
            preds, folds, _ = evaluation.run_nested_repeat(
                f"CNN_{ablation_name}", build_fn, X_norm, y, groups,
                outer_folds=outer_folds, inner_folds=inner_folds, seed=ablation_seed, repeat_id=0,
                preprocessing_label=f"{fs_a}Hz", fs=fs_a, n_timesamples=n_time_a,
                max_epochs=max_epochs, batch_size=config.BATCH_SIZE,
                patience=config.EARLY_STOPPING_PATIENCE, learning_rate=config.LEARNING_RATE,
                run_classifiers=True, run_coral=False,
            )
            all_predictions.extend(preds)
            all_fold_records.extend(folds)
            print(f"  {ablation_name} done")

    # -------------------------------------------------------------------
    # Phase 3: 128 Hz vs 128->512 Hz interpolation sensitivity experiment
    # -------------------------------------------------------------------
    if not args.skip_resolution:
        print("\n--- 128Hz vs 128->512Hz-interpolation sensitivity experiment ---")
        print("    NOTE: pipeline B is FFT interpolation of the same 128Hz samples, "
              "NOT the original 512Hz acquisition. See AUDIT_REPORT.md Phase 3.")
        target_fs = fs_a * 4 if args.smoke_test else config.FS_INTERP_HZ
        X_interp = data.resample_pipeline_b(X, orig_fs=fs_a, target_fs=target_fs)
        X_interp_norm = data.normalize_all(X_interp)
        res_seed = seeds[0]
        preds, folds, _ = evaluation.run_nested_repeat(
            "CNN", models.build_cnn, X_interp_norm, y, groups,
            outer_folds=outer_folds, inner_folds=inner_folds, seed=res_seed, repeat_id=0,
            preprocessing_label="128to512_interp", fs=target_fs, n_timesamples=X_interp_norm.shape[2],
            max_epochs=max_epochs, batch_size=config.BATCH_SIZE,
            patience=config.EARLY_STOPPING_PATIENCE, learning_rate=config.LEARNING_RATE,
            run_classifiers=True, run_coral=False,
        )
        all_predictions.extend(preds)
        all_fold_records.extend(folds)
        print("  resolution experiment done")

    # -------------------------------------------------------------------
    # Save raw results (Phase 2 requirement: everything reconstructable)
    # -------------------------------------------------------------------
    predictions_df = pd.DataFrame(all_predictions)
    coral_df = pd.DataFrame(all_coral_rows)
    predictions_df.to_csv(os.path.join(config.RESULTS_DIR, f"{prefix}predictions.csv"), index=False)
    coral_df.to_csv(os.path.join(config.RESULTS_DIR, f"{prefix}coral_results.csv"), index=False)
    pd.DataFrame([{**f, "train_subjects": ";".join(f["train_subjects"]), "test_subjects": ";".join(f["test_subjects"])}
                  for f in all_fold_records]).to_csv(
        os.path.join(config.RESULTS_DIR, f"{prefix}fold_records.csv"), index=False)
    print(f"\nSaved {len(predictions_df)} prediction rows, {len(coral_df)} CORAL fold rows, "
          f"{len(all_fold_records)} fold records -> {config.RESULTS_DIR}/")

    # -------------------------------------------------------------------
    # Phase 20 sanity checks
    # -------------------------------------------------------------------
    print("\n--- Sanity checks ---")
    report = sanity_checks.run_sanity_checks(all_fold_records, predictions_df)
    report.to_csv(os.path.join(config.TABLES_DIR, f"{prefix}table10_leakage_audit_runtime_checks.csv"), index=False)

    # -------------------------------------------------------------------
    # Phase 18 tables (only the ones cheaply derivable generically here;
    # see notebooks/adhd_robust_benchmark.ipynb for the full table set
    # with narrative context per phase)
    # -------------------------------------------------------------------
    print("\n--- Building tables ---")
    rows = []
    for model_name, grp in predictions_df.groupby("model"):
        m = evaluation.compute_metrics(grp["y_true"], grp["y_pred"], grp["y_proba"])
        rows.append({"model": model_name, **m})
    epoch_level_table = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    epoch_level_table.to_csv(os.path.join(config.TABLES_DIR, f"{prefix}table3_epoch_level_performance.csv"), index=False)

    subj_rows = []
    for model_name, grp in predictions_df.groupby("model"):
        sdf = evaluation.aggregate_subject_level(grp["subject_id"], grp["y_true"], grp["y_proba"])
        m = evaluation.compute_metrics(sdf["y_true"], sdf["pred_mean_proba"], sdf["mean_proba"])
        subj_rows.append({"model": model_name, "aggregation": "mean_probability", **m})
        m2 = evaluation.compute_metrics(sdf["y_true"], sdf["pred_majority_vote"], sdf["vote_fraction"])
        subj_rows.append({"model": model_name, "aggregation": "majority_vote", **m2})
    subject_level_table = pd.DataFrame(subj_rows).sort_values(["model", "aggregation"])
    subject_level_table.to_csv(os.path.join(config.TABLES_DIR, f"{prefix}table4_subject_level_performance.csv"), index=False)

    print(epoch_level_table.to_string(index=False))

    if len(coral_df) > 0:
        coral_df.to_csv(os.path.join(config.TABLES_DIR, f"{prefix}table9_coral_vs_no_coral.csv"), index=False)

    print(f"\nDone in {time.time()-t0:.1f}s. Tables -> {config.TABLES_DIR}/, "
          f"raw results -> {config.RESULTS_DIR}/, figures -> {config.FIGURES_DIR}/")
    if args.smoke_test:
        print("\nREMINDER: this was --smoke-test with SYNTHETIC surrogate data. "
              "No number above is a real result. Re-run without --smoke-test on the "
              "real dataset (set ADHD_EEG_DATA_ROOT) to get manuscript-usable numbers.")


if __name__ == "__main__":
    main()
