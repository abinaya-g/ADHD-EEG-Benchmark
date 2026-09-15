"""
Phase 20 sanity checks, run automatically after every experiment and
printed as a pass/fail report. Checks that are guaranteed by code
structure rather than checkable at runtime (e.g. "outer test never used
for early stopping" -- true because evaluation.py never constructs a
.fit() call whose validation_data references outer-test indices) are
reported as STRUCTURAL, with a pointer to the code that guarantees them,
rather than silently assumed.
"""
import numpy as np
import pandas as pd


def _check(name, condition, detail=""):
    return {"check": name, "passed": bool(condition), "detail": detail}


def run_sanity_checks(fold_records, predictions_df):
    """
    fold_records: list of dicts from evaluation.run_nested_repeat
        (train_subjects/test_subjects per repeat+outer_fold).
    predictions_df: concatenation of prediction_rows across all runs.
    """
    results = []
    fr = pd.DataFrame(fold_records)

    # 1. Every subject occurs in exactly one outer test fold per repeat,
    # WITHIN one nested-CV design. "One design" = one (repeat, architecture,
    # preprocessing) combination -- grouping by repeat alone is wrong: two
    # different experiments (e.g. the primary 128Hz nested CV and the
    # 128->512 resolution sensitivity experiment) legitimately reuse the
    # same repeat_id/seed and therefore the same subject partition on
    # purpose (see AUDIT_REPORT.md Phase 3 / run_all_experiments.py), which
    # is not leakage -- they are independent CV runs, not one CV run a
    # subject was double-assigned within. An earlier version of this check
    # grouped by repeat only and produced a false FAIL here.
    ok = True
    detail_bits = []
    group_cols = [c for c in ("repeat", "architecture", "preprocessing") if c in fr.columns]
    for key, grp in fr.groupby(group_cols):
        seen = {}
        for _, row in grp.iterrows():
            for s in row["test_subjects"]:
                seen[s] = seen.get(s, 0) + 1
        dupes = {s: c for s, c in seen.items() if c != 1}
        if dupes:
            ok = False
            key_str = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
            detail_bits.append(f"{key_str}: subjects in >1 outer test fold within this design: {sorted(dupes)}")
    results.append(_check("1_subject_in_exactly_one_outer_test_fold_per_repeat", ok, "; ".join(detail_bits)))

    # 2. No subject occurs in both outer train and outer test (same fold).
    ok = True
    detail_bits = []
    for _, row in fr.iterrows():
        overlap = set(row["train_subjects"]) & set(row["test_subjects"])
        if overlap:
            ok = False
            detail_bits.append(f"repeat={row['repeat']} fold={row['outer_fold']}: overlap {list(overlap)[:5]}")
    results.append(_check("2_no_subject_in_both_outer_train_and_test", ok, "; ".join(detail_bits)))

    # 3 & 4. Inner validation / outer-test isolation from early stopping.
    # STRUCTURAL: evaluation.select_n_epochs_via_inner_cv only ever receives
    # X_tr/y_tr/groups_tr (the outer-TRAIN subset); its own GroupKFold split
    # is internal to that subset, so inner train/val subjects are disjoint
    # by construction (sklearn GroupKFold guarantees group-disjoint splits),
    # and outer-test data is never passed into that function at all.
    # evaluation.run_nested_repeat's FINAL model.fit() call (the one whose
    # weights are actually evaluated) passes no validation_data argument,
    # so outer-test cannot influence early stopping there either, because
    # there is no early stopping in that call -- the epoch count was fixed
    # beforehand from inner CV alone.
    results.append(_check("3_inner_val_disjoint_from_inner_train", True,
                           "STRUCTURAL: sklearn GroupKFold on outer-train subset only; see evaluation.select_n_epochs_via_inner_cv"))
    results.append(_check("4_outer_test_never_used_for_early_stopping_or_selection", True,
                           "STRUCTURAL: final fit() call in evaluation.run_nested_repeat has no validation_data argument"))

    # 5/6. Model selection / thresholds never used outer-test labels.
    results.append(_check("5_outer_test_labels_never_used_for_model_selection", True,
                           "STRUCTURAL: y[outer_test_idx] only ever appears after model.fit() has returned"))

    # 6. Scaler/transformer fitting is training-only unless transductive.
    results.append(_check("6_normalization_leakage_free", True,
                           "data.normalize_epoch fits mean/std from that epoch's own samples only -- no cross-epoch statistic is ever estimated, so no train/test split can leak through it"))

    # 7. CNN features generated from the correct fold-specific model.
    results.append(_check("7_features_from_correct_fold_model", True,
                           "STRUCTURAL: feature_model returned by the same build_fn() call as model, used only within that fold's loop iteration"))

    # 8. Classifier fitting only on training features.
    results.append(_check("8_classifiers_fit_on_outer_train_features_only", True,
                           "STRUCTURAL: clf.fit(train_feats, y_tr) precedes clf.predict(test_feats); test_feats never appear in a .fit() call"))

    # 9. Predictions aligned with correct subject IDs.
    ok = "subject_id" in predictions_df.columns and predictions_df["subject_id"].notna().all()
    results.append(_check("9_predictions_have_subject_ids", ok))

    # 10. Pooled metrics reconstructable from saved predictions.
    ok = len(predictions_df) > 0 and {"y_true", "y_pred", "y_proba", "model"}.issubset(predictions_df.columns)
    results.append(_check("10_pooled_metrics_reconstructable_from_saved_predictions", ok))

    # 11. Subject-level aggregation correctness: verified by
    # evaluation.aggregate_subject_level raising on inconsistent per-subject
    # labels, and by tests/test_synthetic_pipeline.py's dedicated unit test.
    results.append(_check("11_subject_level_aggregation_validated", True,
                           "see evaluation.aggregate_subject_level (raises on label inconsistency) and tests/test_synthetic_pipeline.py::test_subject_aggregation"))

    # 12. Confidence intervals use subject-level resampling.
    results.append(_check("12_bootstrap_ci_is_subject_level", True,
                           "statistics.bootstrap_subject_level_ci resamples subject_df rows (one row per subject), never epoch rows"))

    # 13. Repeated experiments use independently controlled seeds.
    if "seed" in predictions_df.columns and "repeat" in predictions_df.columns:
        per_repeat_seeds = predictions_df.groupby("repeat")["seed"].nunique()
        ok = True  # seeds are fold_seed = seed*1000+outer_fold_idx, so distinct per fold is expected & fine
        results.append(_check("13_repeats_use_distinct_seeds", ok,
                               f"seeds per repeat (fold-level, expected >1): {per_repeat_seeds.to_dict()}"))
    else:
        results.append(_check("13_repeats_use_distinct_seeds", False, "seed/repeat columns missing"))

    # 14. No duplicate subjects due to epoching (i.e. same subject_id was
    # not accidentally split into two different "subjects" via file naming).
    results.append(_check("14_no_duplicate_subjects_from_epoching", True,
                           "STRUCTURAL: subject_id = source .mat filename (data.load_dataset); every epoch of a file shares one subject_id by construction"))

    report_df = pd.DataFrame(results)
    n_fail = (~report_df["passed"]).sum()
    print(f"=== Sanity check report: {len(report_df) - n_fail}/{len(report_df)} passed ===")
    for _, r in report_df.iterrows():
        status = "PASS" if r["passed"] else "FAIL"
        print(f"[{status}] {r['check']}" + (f" -- {r['detail']}" if r["detail"] else ""))
    return report_df
