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
    python run_all_experiments.py --skip-nested-cnn --skip-ablation ...

Checkpointing / resuming after a crash
---------------------------------------
Each phase writes its own results file (results/predictions_<phase>.csv,
plus a matching fold_records_<phase>.csv), and WITHIN a phase, results are
appended incrementally as each unit of work finishes (each repeat for the
nested-CNN and architecture phases, each individual config for the
ablation phase) -- not held in memory until the end. If the process dies
partway through a phase, everything that phase completed before the crash
is already on disk.

A phase's file is only overwritten if that phase actually runs in this
invocation (i.e. you did not pass --skip-<phase>). So after a crash, you
can add --skip-<phase-that-already-finished> flags and re-run: finished
phases' files are left untouched, and the crashed phase starts that one
file fresh and continues appending. At the end, ALL phase files present in
results/ (from this run and any previous run) are combined into
results/predictions.csv / results/fold_records.csv for the tables/sanity
checks below -- so a resumed run still produces a complete combined output
as long as every phase has completed at least once across your runs.
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import config, coral as coral_mod, data, evaluation, explainability, models, sanity_checks, statistics, visualization


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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


def _fold_records_to_df(fold_records):
    return pd.DataFrame([
        {**f, "train_subjects": ";".join(f["train_subjects"]), "test_subjects": ";".join(f["test_subjects"])}
        for f in fold_records
    ])


class PhaseCheckpoint:
    """Writes one phase's predictions/fold_records/coral rows to their own
    file, truncated once at phase start and appended to as units of work
    (repeats / architectures / ablation configs) complete -- see module
    docstring "Checkpointing / resuming after a crash"."""

    def __init__(self, phase_name, prefix, results_dir):
        self.pred_path = os.path.join(results_dir, f"{prefix}predictions_{phase_name}.csv")
        self.fold_path = os.path.join(results_dir, f"{prefix}fold_records_{phase_name}.csv")
        self.coral_path = os.path.join(results_dir, f"{prefix}coral_results_{phase_name}.csv")
        self._started = False

    def append(self, preds=None, folds=None, coral_rows=None):
        mode = "w" if not self._started else "a"
        header = not self._started
        if preds:
            pd.DataFrame(preds).to_csv(self.pred_path, mode=mode, header=header, index=False)
        if folds:
            _fold_records_to_df(folds).to_csv(self.fold_path, mode=mode, header=header, index=False)
        if coral_rows:
            pd.DataFrame(coral_rows).to_csv(self.coral_path, mode=mode, header=header, index=False)
        self._started = True


def combine_results(prefix, results_dir):
    """Reads every per-phase file present in results_dir (from this run
    and/or previous runs) and concatenates them into the combined
    predictions.csv / fold_records.csv / coral_results.csv used for
    tables/sanity checks below."""
    pred_files = sorted(glob.glob(os.path.join(results_dir, f"{prefix}predictions_*.csv")))
    fold_files = sorted(glob.glob(os.path.join(results_dir, f"{prefix}fold_records_*.csv")))
    coral_files = sorted(glob.glob(os.path.join(results_dir, f"{prefix}coral_results_*.csv")))

    predictions_df = pd.concat([pd.read_csv(f) for f in pred_files], ignore_index=True) if pred_files else pd.DataFrame()
    fold_records_df = pd.concat([pd.read_csv(f) for f in fold_files], ignore_index=True) if fold_files else pd.DataFrame()
    coral_df = pd.concat([pd.read_csv(f) for f in coral_files], ignore_index=True) if coral_files else pd.DataFrame()

    predictions_df.to_csv(os.path.join(results_dir, f"{prefix}predictions.csv"), index=False)
    fold_records_df.to_csv(os.path.join(results_dir, f"{prefix}fold_records.csv"), index=False)
    if len(coral_df) > 0:
        coral_df.to_csv(os.path.join(results_dir, f"{prefix}coral_results.csv"), index=False)

    print(f"Combined {len(pred_files)} prediction file(s) -> {len(predictions_df)} rows, "
          f"{len(fold_files)} fold-record file(s) -> {len(fold_records_df)} rows, "
          f"{len(coral_files)} CORAL file(s) -> {len(coral_df)} rows")
    return predictions_df, fold_records_df, coral_df


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
        # n_timesamples=1152, not 768: the resolution experiment below
        # doubles fs (and n_timesamples) for build_cnn's fs-scaled temporal
        # kernels, and 768 leaves too little margin -- verified to raise
        # "Computed output size would be zero or negative" in
        # AveragePooling1D at 2x. See tests/test_synthetic_pipeline.py for
        # the same fix and margin calculation.
        X, y, groups, file_epoch_idx = data.make_synthetic_dataset(
            n_adhd_subjects=15, n_control_subjects=15, min_epochs=3, max_epochs=5,
            n_channels=config.N_CHANNELS, n_timesamples=1152, class_signal=0.9, seed=0,
        )
        fs_a, n_time_a = 128, 1152
    else:
        X, y, groups, file_epoch_idx = data.load_dataset()
        fs_a, n_time_a = config.FS_ASSUMED_HZ, config.EPOCH_LEN_A

    print(f"Loaded X={X.shape} y={y.shape} unique_subjects={len(set(groups))} "
          f"ADHD_epochs={int((y==1).sum())} Control_epochs={int((y==0).sum())}")

    X_norm = data.normalize_all(X)

    # -------------------------------------------------------------------
    # Phase 4/5: nested + repeated CV, primary CNN, + classical classifiers
    # -------------------------------------------------------------------
    if not args.skip_nested_cnn:
        print("\n--- Nested + repeated subject-independent CV: CNN + classical classifiers ---")
        ckpt = PhaseCheckpoint("nested_cnn", prefix, config.RESULTS_DIR)
        for i, seed in enumerate(seeds):
            preds, folds, coral_rows = evaluation.run_nested_repeat(
                "CNN", models.build_cnn, X_norm, y, groups,
                outer_folds=outer_folds, inner_folds=inner_folds, seed=seed, repeat_id=i,
                preprocessing_label=f"{fs_a}Hz", fs=fs_a, n_timesamples=n_time_a,
                max_epochs=max_epochs, batch_size=config.BATCH_SIZE,
                patience=config.EARLY_STOPPING_PATIENCE, learning_rate=config.LEARNING_RATE,
                run_classifiers=True, run_coral=not args.skip_coral,
            )
            ckpt.append(preds, folds, coral_rows)
            print(f"  repeat {i} (seed={seed}) done and saved: {len(folds)} outer folds")

    # -------------------------------------------------------------------
    # Phase 6: EEGNet / ShallowConvNet / DeepConvNet, same protocol
    # -------------------------------------------------------------------
    if not args.skip_architectures:
        print("\n--- Alternative EEG architectures (same nested protocol) ---")
        ckpt = PhaseCheckpoint("architectures", prefix, config.RESULTS_DIR)
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
                ckpt.append(preds, folds)
                print(f"  {arch_name} repeat {i} (seed={seed}) done and saved")

    # -------------------------------------------------------------------
    # Phase 12: ablation study (single repeat -- computational budget)
    # -------------------------------------------------------------------
    if not args.skip_ablation:
        print("\n--- Ablation study ---")
        ckpt = PhaseCheckpoint("ablation", prefix, config.RESULTS_DIR)
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
            ckpt.append(preds, folds)
            print(f"  {ablation_name} done and saved")

    # -------------------------------------------------------------------
    # Phase 3: 128 Hz vs 128->512 Hz interpolation sensitivity experiment
    # -------------------------------------------------------------------
    if not args.skip_resolution:
        print("\n--- 128Hz vs 128->512Hz-interpolation sensitivity experiment ---")
        print("    NOTE: pipeline B is FFT interpolation of the same 128Hz samples, "
              "NOT the original 512Hz acquisition. See AUDIT_REPORT.md Phase 3.")
        ckpt = PhaseCheckpoint("resolution", prefix, config.RESULTS_DIR)
        # 2x for smoke-test (not the real 4x 128->512 ratio): the synthetic
        # epoch is short, and this experiment only needs to prove the code
        # path works, not replicate the real resolution ratio.
        target_fs = fs_a * 2 if args.smoke_test else config.FS_INTERP_HZ
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
        ckpt.append(preds, folds)
        print("  resolution experiment done and saved")

    # -------------------------------------------------------------------
    # Combine every phase file present on disk (this run + any previous
    # run) into the single predictions.csv / fold_records.csv used below.
    # -------------------------------------------------------------------
    print("\n--- Combining phase results ---")
    predictions_df, fold_records_df, coral_df = combine_results(prefix, config.RESULTS_DIR)

    if len(predictions_df) == 0:
        print("No results on disk yet (every phase was --skip-ped and no prior run left files behind). Nothing to report.")
        return

    # -------------------------------------------------------------------
    # Phase 20 sanity checks (reconstructs fold_records as list[dict] with
    # subjects split back into lists for the checks that need them)
    # -------------------------------------------------------------------
    print("\n--- Sanity checks ---")
    fold_records = []
    for _, row in fold_records_df.iterrows():
        d = row.to_dict()
        d["train_subjects"] = d["train_subjects"].split(";") if d["train_subjects"] else []
        d["test_subjects"] = d["test_subjects"].split(";") if d["test_subjects"] else []
        fold_records.append(d)
    report = sanity_checks.run_sanity_checks(fold_records, predictions_df)
    report.to_csv(os.path.join(config.TABLES_DIR, f"{prefix}table10_leakage_audit_runtime_checks.csv"), index=False)

    # -------------------------------------------------------------------
    # Phase 18 tables (only the ones cheaply derivable generically here;
    # see notebooks/adhd_robust_benchmark.ipynb for the full table set
    # with narrative context per phase)
    # -------------------------------------------------------------------
    print("\n--- Building tables ---")
    # Group by (model, preprocessing), NOT model alone: the same model name
    # (e.g. "CNN", "CNN+LR") is reused across different preprocessing
    # pipelines (128Hz primary vs 128->512 interpolation sensitivity
    # experiment) and across ablation configs' shared preprocessing label.
    # Grouping by model alone silently pools rows from unrelated
    # experiments into one misleading averaged accuracy -- caught when a
    # real run showed "CNN" at n=1016 (508 native-128Hz rows + 508
    # interpolated-512Hz rows averaged together instead of compared).
    table_group_cols = [c for c in ("model", "preprocessing") if c in predictions_df.columns]

    rows = []
    for key, grp in predictions_df.groupby(table_group_cols):
        key = key if isinstance(key, tuple) else (key,)
        m = evaluation.compute_metrics(grp["y_true"], grp["y_pred"], grp["y_proba"])
        rows.append({**dict(zip(table_group_cols, key)), **m})
    epoch_level_table = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    epoch_level_table.to_csv(os.path.join(config.TABLES_DIR, f"{prefix}table3_epoch_level_performance.csv"), index=False)

    subj_rows = []
    for key, grp in predictions_df.groupby(table_group_cols):
        key = key if isinstance(key, tuple) else (key,)
        key_dict = dict(zip(table_group_cols, key))
        sdf = evaluation.aggregate_subject_level(grp["subject_id"], grp["y_true"], grp["y_proba"])
        m = evaluation.compute_metrics(sdf["y_true"], sdf["pred_mean_proba"], sdf["mean_proba"])
        subj_rows.append({**key_dict, "aggregation": "mean_probability", **m})
        m2 = evaluation.compute_metrics(sdf["y_true"], sdf["pred_majority_vote"], sdf["vote_fraction"])
        subj_rows.append({**key_dict, "aggregation": "majority_vote", **m2})
    subject_level_table = pd.DataFrame(subj_rows).sort_values(table_group_cols + ["aggregation"])
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
