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

from src import config, coral as coral_mod, data, evaluation, explainability, models, reporting, sanity_checks, statistics, visualization


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
    p.add_argument("--only-architecture", type=str, default=None,
                    help="Restrict the architectures phase to one named architecture (e.g. DeepConvNet). "
                         "For resuming a partially-completed phase (e.g. a crashed/interrupted run) without "
                         "redoing architectures that already finished. Implies appending to the existing "
                         "predictions_architectures.csv rather than truncating it.")
    p.add_argument("--repeat-indices", type=str, default=None,
                    help="Comma-separated 0-based repeat indices to run in the architectures phase "
                         "(e.g. '4' to run only the 5th repeat). Default: all repeats. Combine with "
                         "--only-architecture to resume exactly the missing (architecture, repeat) work.")
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

    def __init__(self, phase_name, prefix, results_dir, append_existing=False):
        self.pred_path = os.path.join(results_dir, f"{prefix}predictions_{phase_name}.csv")
        self.fold_path = os.path.join(results_dir, f"{prefix}fold_records_{phase_name}.csv")
        self.coral_path = os.path.join(results_dir, f"{prefix}coral_results_{phase_name}.csv")
        # append_existing=True: resuming a narrowed subset of this phase
        # (e.g. --only-architecture/--repeat-indices) -- append to whatever
        # is already on disk instead of truncating it, since the untouched
        # rest of the phase (other architectures/repeats) must survive.
        self._started = append_existing and os.path.exists(self.pred_path)

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
        resuming = args.only_architecture is not None or args.repeat_indices is not None
        ckpt = PhaseCheckpoint("architectures", prefix, config.RESULTS_DIR, append_existing=resuming)
        repeat_indices = [int(x) for x in args.repeat_indices.split(",")] if args.repeat_indices else list(range(repeats))
        if resuming:
            print(f"  RESUMING (appending to existing file): architecture={args.only_architecture or 'all'}, "
                  f"repeat_indices={repeat_indices}")
        for arch_name, build_fn in models.ARCHITECTURE_BUILDERS.items():
            if arch_name == "CNN":
                continue
            if args.only_architecture and arch_name != args.only_architecture:
                continue
            for i in repeat_indices:
                seed = seeds[i]
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
        # CNN (+ its classical classifiers) is the primary comparison;
        # EEGNet is added as the strongest deep-learning baseline, per the
        # review brief's "if computationally feasible, also test EEGNet".
        # Both reuse repeat_id=0/res_seed -- the SAME subject partition as
        # the primary 128Hz run's first repeat -- so the two conditions
        # are validly paired per outer fold (see reporting.table5/6).
        for arch_name, build_fn, run_clf in (("CNN", models.build_cnn, True),
                                              ("EEGNet", models.build_eegnet, False)):
            preds, folds, _ = evaluation.run_nested_repeat(
                arch_name, build_fn, X_interp_norm, y, groups,
                outer_folds=outer_folds, inner_folds=inner_folds, seed=res_seed, repeat_id=0,
                preprocessing_label="128to512_interp", fs=target_fs, n_timesamples=X_interp_norm.shape[2],
                max_epochs=max_epochs, batch_size=config.BATCH_SIZE,
                patience=config.EARLY_STOPPING_PATIENCE, learning_rate=config.LEARNING_RATE,
                run_classifiers=run_clf, run_coral=False,
            )
            ckpt.append(preds, folds)
            print(f"  {arch_name} resolution experiment done and saved")

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
    primary_preprocessing = f"{fs_a}Hz"
    final_dir = os.path.join(config.RESULTS_DIR, "..", "results_final") if not args.smoke_test else None

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

    # -------------------------------------------------------------------
    # TABLE_1-7 (review-brief Phase 15 exact filenames), plus subject-level
    # bootstrap CIs and paired statistical comparisons that the tables
    # above don't compute. See src/reporting.py for how each is built.
    # -------------------------------------------------------------------
    print("\n--- Building TABLE_1-7 (manuscript deliverables) ---")
    table1 = reporting.table1_dataset_summary(X, y, groups, config, outer_folds, inner_folds, repeats, seeds, max_epochs)
    table1.to_csv(os.path.join(config.TABLES_DIR, f"{prefix}TABLE_1_DATASET_SUMMARY.csv"), index=False)

    epoch_level_table.to_csv(os.path.join(config.TABLES_DIR, f"{prefix}TABLE_2_EPOCH_LEVEL_RESULTS.csv"), index=False)
    subject_level_table.to_csv(os.path.join(config.TABLES_DIR, f"{prefix}TABLE_3_SUBJECT_LEVEL_RESULTS.csv"), index=False)

    table4 = reporting.table4_confidence_intervals(predictions_df, table_group_cols, config)
    table4.to_csv(os.path.join(config.TABLES_DIR, f"{prefix}TABLE_4_CONFIDENCE_INTERVALS.csv"), index=False)
    print(f"  TABLE_4: {len(table4)} (group x metric) rows, subject-level bootstrap, n_boot={config.N_BOOTSTRAP}")

    reference_key = ("CNN", primary_preprocessing)
    table5 = reporting.table5_model_comparison(predictions_df, reference_key, table_group_cols)
    table5.to_csv(os.path.join(config.TABLES_DIR, f"{prefix}TABLE_5_MODEL_COMPARISON_STATISTICS.csv"), index=False)
    print(f"  TABLE_5: {len(table5)} paired comparisons vs {reference_key}, Holm-corrected")

    # Built from whatever "128to512_interp" data actually exists on disk,
    # not from whether THIS invocation ran the resolution phase -- a
    # --skip-resolution combine-only rerun must still rebuild TABLE_6 from
    # a previous run's saved data (same reasoning as TABLE_7/coral_df
    # below, which checks len(coral_df) rather than args.skip_coral).
    if "128to512_interp" in set(predictions_df.get("preprocessing", [])):
        resolution_models = [m for m in ("CNN", "EEGNet") if
                              ((predictions_df["model"] == m) & (predictions_df["preprocessing"] == "128to512_interp")).any()]
        t6_summaries, t6_cmps = [], []
        for m in resolution_models:
            s, c = reporting.table6_resolution_comparison(predictions_df, table_group_cols, model_name=m)
            t6_summaries.append(s)
            t6_cmps.append(c)
        table6_summary = pd.concat(t6_summaries, ignore_index=True) if t6_summaries else pd.DataFrame()
        table6_cmp = pd.concat(t6_cmps, ignore_index=True) if t6_cmps else pd.DataFrame()
        table6_summary.to_csv(os.path.join(config.TABLES_DIR, f"{prefix}TABLE_6_128HZ_VS_128TO512.csv"), index=False)
        table6_cmp.to_csv(os.path.join(config.TABLES_DIR, f"{prefix}TABLE_6_128HZ_VS_128TO512_paired_test.csv"), index=False)
        print(f"  TABLE_6: models={resolution_models}, {len(table6_summary)} summary rows, {len(table6_cmp)} paired-test rows")
    else:
        resolution_models = []

    if len(coral_df) > 0:
        coral_df.to_csv(os.path.join(config.TABLES_DIR, f"{prefix}TABLE_7_CORAL_RESULTS.csv"), index=False)

    # -------------------------------------------------------------------
    # Explainability (Integrated Gradients) -- Phase 15/11. Trained on ALL
    # epochs purely for attribution visualization, mirroring the ORIGINAL
    # notebook's own precedent (adhd-coral.ipynb cell-7 trains on the full
    # dataset "purely to extract filter weights for visualization... NOT a
    # performance evaluation"). This is not a held-out evaluation and no
    # accuracy from this model should ever be cited; see AUDIT_REPORT.md.
    # -------------------------------------------------------------------
    if not args.skip_explainability:
        print("\n--- Explainability (Integrated Gradients, visualization-only model) ---")
        import tensorflow as tf
        models.set_all_seeds(seeds[0])
        ig_model, _ = models.build_cnn(n_channels=X_norm.shape[1], n_timesamples=X_norm.shape[2], fs=fs_a)
        ig_model.compile(optimizer=tf.keras.optimizers.Adam(config.LEARNING_RATE), loss="binary_crossentropy")
        ig_model.fit(X_norm, y, epochs=min(max_epochs, 20), batch_size=config.BATCH_SIZE, verbose=0)
        ig_result = explainability.batch_channel_temporal_importance(ig_model, X_norm, max_samples=30, steps=50, seed=0)
        ig_channel_df = pd.DataFrame({
            "channel_index": range(len(ig_result["channel_importance_mean"])),
            "importance_mean": ig_result["channel_importance_mean"],
            "importance_sd": ig_result["channel_importance_sd"],
        })
        ig_temporal_df = pd.DataFrame({
            "time_bin": range(len(ig_result["temporal_importance_mean"])),
            "importance_mean": ig_result["temporal_importance_mean"],
            "importance_sd": ig_result["temporal_importance_sd"],
        })
        ig_channel_df.to_csv(os.path.join(config.TABLES_DIR, f"{prefix}table_explainability_channel_importance.csv"), index=False)
        ig_temporal_df.to_csv(os.path.join(config.TABLES_DIR, f"{prefix}table_explainability_temporal_importance.csv"), index=False)
        print(f"  saved channel/temporal importance ({len(ig_result['sample_indices'])} sampled epochs, "
              f"NOT a performance claim -- see AUDIT_REPORT.md)")

    # -------------------------------------------------------------------
    # Figures (Phase 19). visualization.py was implemented and imported
    # here but never actually called -- a real gap found the same way as
    # the --skip-explainability dead flag: importing a module is not
    # evidence it runs. Figures 1/2 (workflow/architecture diagrams) are
    # not generated here by design (see src/visualization.py docstring);
    # everything else is built from data already computed above, filtered
    # to a readable "core" model set rather than all ~50 model x
    # preprocessing x ablation combinations.
    # -------------------------------------------------------------------
    print("\n--- Figures ---")
    core_models = [m for m in ("CNN", "CNN+LR", "CNN+NLSVM", "CNN+RF", "CNN+GNB", "CNN+KNN",
                                "CNN+LinearSVM", "EEGNet", "ShallowConvNet", "DeepConvNet")
                   if m in set(predictions_df["model"])]
    per_fold_bal_acc = reporting.per_fold_subject_metric(predictions_df, table_group_cols, "balanced_accuracy")

    visualization.fig3_nested_cv_schematic(outer_folds, inner_folds)

    fig4_df = table4[(table4["preprocessing"] == primary_preprocessing)
                      & (table4["metric"] == "balanced_accuracy")
                      & (table4["model"].isin(core_models))].rename(columns={
        "point_estimate": "balanced_accuracy_mean", "ci_low": "balanced_accuracy_ci_low",
        "ci_high": "balanced_accuracy_ci_high"})
    if len(fig4_df) > 0:
        visualization.fig4_model_comparison_ci(fig4_df, metric="balanced_accuracy")

    subject_dfs = {}
    for m in core_models:
        sub = predictions_df[(predictions_df["model"] == m) & (predictions_df["preprocessing"] == primary_preprocessing)]
        if len(sub) > 0:
            subject_dfs[m] = evaluation.aggregate_subject_level(sub["subject_id"], sub["y_true"], sub["y_proba"])
    if len(subject_dfs) > 0:
        visualization.fig5_subject_level_roc(subject_dfs)
        visualization.fig6_subject_confusion_matrices(subject_dfs)

    per_fold_fig7 = {m: list(per_fold_bal_acc.get((m, primary_preprocessing), {}).values()) for m in core_models}
    per_fold_fig7 = {m: v for m, v in per_fold_fig7.items() if len(v) > 0}
    if len(per_fold_fig7) > 0:
        visualization.fig7_repeated_cv_distribution(per_fold_fig7, metric_name="balanced_accuracy")

    # Built from data presence, not args.skip_resolution/skip_ablation --
    # same reasoning as TABLE_6/TABLE_7 above (a combine-only rerun must
    # still regenerate figures from a previous run's saved data).
    for m in resolution_models:
        vals_a = list(per_fold_bal_acc.get((m, primary_preprocessing), {}).values())
        vals_b = list(per_fold_bal_acc.get((m, "128to512_interp"), {}).values())
        if vals_a and vals_b:
            fig_name = "fig8_resolution.png" if m == "CNN" else f"fig8_resolution_{m}.png"
            visualization.fig8_resolution_comparison(
                pd.DataFrame({"balanced_accuracy": vals_a}), pd.DataFrame({"balanced_accuracy": vals_b}),
                metric="balanced_accuracy", labels=(f"{m} 128 Hz", f"{m} 128->512 Hz interpolation"), name=fig_name)

    ablation_models_present = sorted(m for m in predictions_df["model"].unique() if m.startswith("CNN_"))
    if ablation_models_present:
        ablation_rows = []
        for m in ablation_models_present:
            vals = list(per_fold_bal_acc.get((m, primary_preprocessing), {}).values())
            if vals:
                ablation_rows.append({"ablation": m, "balanced_accuracy_mean": float(np.mean(vals)),
                                       "balanced_accuracy_sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0})
        if ablation_rows:
            visualization.fig9_ablation(pd.DataFrame(ablation_rows), metric="balanced_accuracy")

    if len(coral_df) > 0:
        visualization.fig10_coral_covariance(coral_df)

    print(f"  figures written -> {config.FIGURES_DIR}/")

    # -------------------------------------------------------------------
    # Mirror the polished deliverables into a NEW results_final/ tree
    # (review-brief Phase 14) without touching the working results/tables/
    # figures/ checkpoint directories used for resuming. Real-data runs
    # only -- smoke-test output never lands here.
    # -------------------------------------------------------------------
    if final_dir is not None:
        import shutil
        for sub in ("audit", "configs", "fold_assignments", "predictions", "subject_level",
                    "epoch_level", "confidence_intervals", "statistics", "figures", "tables", "logs"):
            os.makedirs(os.path.join(final_dir, sub), exist_ok=True)

        predictions_df.to_csv(os.path.join(final_dir, "predictions", "all_predictions.csv"), index=False)
        fold_records_df.to_csv(os.path.join(final_dir, "fold_assignments", "fold_assignments.csv"), index=False)
        epoch_level_table.to_csv(os.path.join(final_dir, "epoch_level", "TABLE_2_EPOCH_LEVEL_RESULTS.csv"), index=False)
        subject_level_table.to_csv(os.path.join(final_dir, "subject_level", "TABLE_3_SUBJECT_LEVEL_RESULTS.csv"), index=False)
        table4.to_csv(os.path.join(final_dir, "confidence_intervals", "TABLE_4_CONFIDENCE_INTERVALS.csv"), index=False)
        table5.to_csv(os.path.join(final_dir, "statistics", "TABLE_5_MODEL_COMPARISON_STATISTICS.csv"), index=False)
        table1.to_csv(os.path.join(final_dir, "tables", "TABLE_1_DATASET_SUMMARY.csv"), index=False)
        report.to_csv(os.path.join(final_dir, "audit", "leakage_sanity_checks.csv"), index=False)
        for fname in os.listdir(config.FIGURES_DIR):
            src_path = os.path.join(config.FIGURES_DIR, fname)
            if os.path.isfile(src_path):
                shutil.copy2(src_path, os.path.join(final_dir, "figures", fname))
        with open(os.path.join(final_dir, "configs", "run_config.txt"), "w") as fh:
            fh.write(f"outer_folds={outer_folds}\ninner_folds={inner_folds}\nrepeats={repeats}\n"
                     f"seeds={seeds}\nmax_epochs={max_epochs}\nbatch_size={config.BATCH_SIZE}\n"
                     f"learning_rate={config.LEARNING_RATE}\npatience={config.EARLY_STOPPING_PATIENCE}\n")
        print(f"\nMirrored final deliverables -> {os.path.normpath(final_dir)}/")

    print(f"\nDone in {time.time()-t0:.1f}s. Tables -> {config.TABLES_DIR}/, "
          f"raw results -> {config.RESULTS_DIR}/, figures -> {config.FIGURES_DIR}/")
    if args.smoke_test:
        print("\nREMINDER: this was --smoke-test with SYNTHETIC surrogate data. "
              "No number above is a real result. Re-run without --smoke-test on the "
              "real dataset (set ADHD_EEG_DATA_ROOT) to get manuscript-usable numbers.")


if __name__ == "__main__":
    main()
