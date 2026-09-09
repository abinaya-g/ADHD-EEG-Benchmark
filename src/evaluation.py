"""
Leakage-corrected nested subject-independent evaluation.

This is the core fix for the audited flaw in both original notebooks
(AUDIT_REPORT.md, finding L1): every CNN .fit() call there used
`validation_data=(X_fold_test, y_fold_test)`, i.e. the OUTER test fold,
for early stopping. That lets the stopping criterion see outer-test loss
before the "final" evaluation on that same fold -- a form of model
selection leakage.

The fix implemented here, per Phase 4 of the review brief:

    outer_train_subjects
        -> inner GroupKFold (on outer_train subjects only)
        -> per inner fold, train with early stopping on the INNER
           validation subjects; record the epoch of best inner-val loss
        -> take the median selected epoch across inner folds
        -> retrain ONE model on ALL outer_train subjects for exactly that
           many epochs (no validation split at all in this final fit --
           the epoch count was already chosen without seeing outer_test)
        -> evaluate exactly once on outer_test subjects

Outer test subjects are never passed to any .fit() call, in any role, at
any point in this module -- not as validation_data, not for scaling (the
per-epoch normalization in data.py needs no fitting), not for classifier
fitting, not for CORAL's *label* information (CORAL's covariance
alignment is unsupervised and is called out explicitly as transductive --
see coral.py and run_coral_experiment below).
"""
import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                              confusion_matrix, f1_score, matthews_corrcoef,
                              precision_score, roc_auc_score)
from sklearn.model_selection import GroupKFold
from tensorflow.keras.callbacks import EarlyStopping

from . import coral as coral_mod
from . import models as models_mod


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(y_true, y_pred, y_proba=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels_present = set(np.unique(y_true))

    out = {
        "n": int(len(y_true)),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    if len(labels_present) > 1:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        out["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        out["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        out["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
        out["mcc"] = matthews_corrcoef(y_true, y_pred)
    else:
        out["sensitivity"] = out["specificity"] = out["balanced_accuracy"] = out["mcc"] = np.nan

    if y_proba is not None and len(labels_present) > 1:
        out["auc"] = roc_auc_score(y_true, y_proba)
    else:
        out["auc"] = np.nan
    return out


def aggregate_subject_level(subject_ids, y_true_epoch, y_proba_epoch, threshold=0.5):
    """Phase 8: one prediction per subject, from that subject's epochs."""
    import pandas as pd

    df = pd.DataFrame({
        "subject_id": np.asarray(subject_ids),
        "y_true": np.asarray(y_true_epoch),
        "y_proba": np.asarray(y_proba_epoch, dtype=float),
    })
    df["y_pred_epoch"] = (df["y_proba"] >= threshold).astype(int)

    agg = df.groupby("subject_id").agg(
        y_true=("y_true", "first"),
        n_epochs=("y_true", "size"),
        mean_proba=("y_proba", "mean"),
        vote_fraction=("y_pred_epoch", "mean"),
    ).reset_index()
    agg["pred_mean_proba"] = (agg["mean_proba"] >= threshold).astype(int)
    agg["pred_majority_vote"] = (agg["vote_fraction"] >= 0.5).astype(int)

    label_consistency = df.groupby("subject_id")["y_true"].nunique()
    if (label_consistency > 1).any():
        bad = label_consistency[label_consistency > 1].index.tolist()
        raise ValueError(f"Subjects with inconsistent labels across epochs: {bad}")

    return agg


# ---------------------------------------------------------------------------
# Grouped, shuffled, (roughly) class-balanced K-fold splitter
# ---------------------------------------------------------------------------
def make_grouped_shuffled_folds(groups, y, n_splits, seed):
    """
    sklearn's GroupKFold takes no random_state -- it is fully deterministic
    given the input order, so calling it repeatedly with different seeds
    produces the SAME partition every time. That makes it unusable on its
    own for Phase 5's repeated-CV requirement ("different subject
    partitioning where applicable"). This wrapper shuffles the unique
    subject list with the given seed, distributes subjects round-robin by
    class to keep folds roughly class-balanced, and returns
    (train_idx, test_idx) epoch-index pairs like GroupKFold.split() would.
    """
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    y = np.asarray(y)

    unique_subjects = np.unique(groups)
    subject_label = {}
    for s in unique_subjects:
        labels = np.unique(y[groups == s])
        if len(labels) != 1:
            raise ValueError(f"Subject {s} has inconsistent labels across epochs")
        subject_label[s] = labels[0]

    by_class = {0: [], 1: []}
    for s in unique_subjects:
        by_class[subject_label[s]].append(s)
    for c in by_class:
        by_class[c] = list(rng.permutation(by_class[c]))

    fold_of_subject = {}
    for c, subj_list in by_class.items():
        for i, s in enumerate(subj_list):
            fold_of_subject[s] = i % n_splits

    splits = []
    for fold_i in range(n_splits):
        test_subjects = {s for s, f in fold_of_subject.items() if f == fold_i}
        test_mask = np.array([g in test_subjects for g in groups])
        train_idx = np.where(~test_mask)[0]
        test_idx = np.where(test_mask)[0]
        splits.append((train_idx, test_idx))
    return splits


# ---------------------------------------------------------------------------
# Inner-CV epoch selection (replaces validation_data=outer_test)
# ---------------------------------------------------------------------------
def select_n_epochs_via_inner_cv(build_fn, build_kwargs, X_tr, y_tr, groups_tr,
                                  inner_folds, seed, max_epochs, batch_size,
                                  patience, learning_rate):
    import tensorflow as tf

    n_unique = len(np.unique(groups_tr))
    inner_folds_eff = min(inner_folds, n_unique) if n_unique >= 2 else 1
    if inner_folds_eff < 2:
        # Too few outer-train subjects for an inner split (only happens in
        # tiny synthetic smoke tests) -- fall back to a fixed small epoch
        # budget rather than silently touching outer_test.
        return max_epochs // 4

    gkf_inner = GroupKFold(n_splits=inner_folds_eff)
    best_epochs = []
    for inner_i, (in_tr, in_val) in enumerate(gkf_inner.split(X_tr, y_tr, groups_tr)):
        models_mod.set_all_seeds(seed * 100 + inner_i)
        model, _ = build_fn(**build_kwargs)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss="binary_crossentropy", metrics=["accuracy"])
        early_stop = EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
        hist = model.fit(X_tr[in_tr], y_tr[in_tr], epochs=max_epochs, batch_size=batch_size,
                          verbose=0, validation_data=(X_tr[in_val], y_tr[in_val]),
                          callbacks=[early_stop])
        val_losses = hist.history["val_loss"]
        best_epochs.append(int(np.argmin(val_losses)) + 1)
    return int(np.median(best_epochs))


# ---------------------------------------------------------------------------
# Main driver: one architecture, one seed/repeat, full nested outer CV
# ---------------------------------------------------------------------------
def run_nested_repeat(architecture_name, build_fn, X, y, groups, *,
                       outer_folds, inner_folds, seed, repeat_id,
                       preprocessing_label, fs, n_timesamples,
                       max_epochs, batch_size, patience, learning_rate,
                       run_classifiers=True, run_coral=False,
                       n_channels=None):
    """
    Returns
    -------
    prediction_rows : list[dict]  epoch-level predictions, one row per
        (model, subject epoch), ready to append to a results table.
    fold_records : list[dict]  provenance for the Phase 20 sanity checks
        and Table 3/4/5/10 (subjects per split, selected epoch count, ...).
    coral_rows : list[dict]  covariance-distance + accuracy for the CORAL
        experiment (empty unless run_coral=True).
    """
    import tensorflow as tf

    n_channels = n_channels or X.shape[1]
    build_kwargs = dict(n_channels=n_channels, n_timesamples=n_timesamples, fs=fs)

    outer_splits = make_grouped_shuffled_folds(groups, y, outer_folds, seed)

    prediction_rows = []
    fold_records = []
    coral_rows = []

    for outer_fold_idx, (outer_train_idx, outer_test_idx) in enumerate(outer_splits):
        outer_train_subjects = set(groups[outer_train_idx])
        outer_test_subjects = set(groups[outer_test_idx])
        assert outer_train_subjects.isdisjoint(outer_test_subjects), (
            "Leakage: a subject appears in both outer train and outer test"
        )

        X_tr, y_tr, g_tr = X[outer_train_idx], y[outer_train_idx], groups[outer_train_idx]
        X_te, y_te = X[outer_test_idx], y[outer_test_idx]

        fold_seed = seed * 1000 + outer_fold_idx

        n_epochs_selected = select_n_epochs_via_inner_cv(
            build_fn, build_kwargs, X_tr, y_tr, g_tr,
            inner_folds=inner_folds, seed=fold_seed, max_epochs=max_epochs,
            batch_size=batch_size, patience=patience, learning_rate=learning_rate,
        )

        # Final fit on ALL outer-train subjects, for the epoch count chosen
        # entirely from inner folds. No validation_data here at all --
        # outer_test never appears in any .fit() call.
        models_mod.set_all_seeds(fold_seed)
        model, feature_model = build_fn(**build_kwargs)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                       loss="binary_crossentropy", metrics=["accuracy"])
        model.fit(X_tr, y_tr, epochs=max(n_epochs_selected, 1), batch_size=batch_size, verbose=0)

        proba_cnn = model.predict(X_te, verbose=0).ravel()
        pred_cnn = (proba_cnn >= 0.5).astype(int)

        fold_records.append({
            "architecture": architecture_name, "preprocessing": preprocessing_label,
            "repeat": repeat_id, "outer_fold": outer_fold_idx, "seed": fold_seed,
            "n_train_subjects": len(outer_train_subjects), "n_test_subjects": len(outer_test_subjects),
            "train_subjects": sorted(outer_train_subjects), "test_subjects": sorted(outer_test_subjects),
            "selected_epochs": n_epochs_selected,
        })

        for subj, yt, yp, ypr in zip(groups[outer_test_idx], y_te, pred_cnn, proba_cnn):
            prediction_rows.append({
                "architecture": architecture_name, "model": architecture_name,
                "preprocessing": preprocessing_label, "repeat": repeat_id,
                "outer_fold": outer_fold_idx, "seed": fold_seed,
                "subject_id": subj, "y_true": int(yt), "y_pred": int(yp), "y_proba": float(ypr),
            })

        if run_classifiers or run_coral:
            train_feats = feature_model.predict(X_tr, verbose=0)
            test_feats = feature_model.predict(X_te, verbose=0)

        if run_classifiers:
            for clf_name, clf in models_mod.make_classifiers().items():
                clf.fit(train_feats, y_tr)
                pred = clf.predict(test_feats)
                proba = clf.predict_proba(test_feats)[:, 1] if hasattr(clf, "predict_proba") else pred.astype(float)
                for subj, yt, yp, ypr in zip(groups[outer_test_idx], y_te, pred, proba):
                    prediction_rows.append({
                        "architecture": architecture_name, "model": f"{architecture_name}+{clf_name}",
                        "preprocessing": preprocessing_label, "repeat": repeat_id,
                        "outer_fold": outer_fold_idx, "seed": fold_seed,
                        "subject_id": subj, "y_true": int(yt), "y_pred": int(yp), "y_proba": float(ypr),
                    })

        if run_coral:
            dist_before = coral_mod.covariance_distance(train_feats, test_feats)
            train_feats_aligned = coral_mod.coral_transform(train_feats, test_feats)
            dist_after = coral_mod.covariance_distance(train_feats_aligned, test_feats)

            from sklearn.linear_model import LogisticRegression
            lr_no_coral = LogisticRegression(penalty="l2", max_iter=2000, class_weight="balanced")
            lr_no_coral.fit(train_feats, y_tr)
            pred_nc = lr_no_coral.predict(test_feats)
            proba_nc = lr_no_coral.predict_proba(test_feats)[:, 1]

            lr_coral = LogisticRegression(penalty="l2", max_iter=2000, class_weight="balanced")
            lr_coral.fit(train_feats_aligned, y_tr)
            pred_c = lr_coral.predict(test_feats)
            proba_c = lr_coral.predict_proba(test_feats)[:, 1]

            m_nc = compute_metrics(y_te, pred_nc, proba_nc)
            m_c = compute_metrics(y_te, pred_c, proba_c)
            coral_rows.append({
                "repeat": repeat_id, "outer_fold": outer_fold_idx, "seed": fold_seed,
                "cov_distance_before": dist_before, "cov_distance_after": dist_after,
                "adaptation_type": "transductive_unsupervised (test features, no test labels, used to estimate target covariance)",
                **{f"no_coral_{k}": v for k, v in m_nc.items()},
                **{f"with_coral_{k}": v for k, v in m_c.items()},
            })
            for subj, yt, yp, ypr in zip(groups[outer_test_idx], y_te, pred_nc, proba_nc):
                prediction_rows.append({
                    "architecture": architecture_name, "model": "CNN+LR_noCORAL",
                    "preprocessing": preprocessing_label, "repeat": repeat_id,
                    "outer_fold": outer_fold_idx, "seed": fold_seed,
                    "subject_id": subj, "y_true": int(yt), "y_pred": int(yp), "y_proba": float(ypr),
                })
            for subj, yt, yp, ypr in zip(groups[outer_test_idx], y_te, pred_c, proba_c):
                prediction_rows.append({
                    "architecture": architecture_name, "model": "CNN+LR_withCORAL",
                    "preprocessing": preprocessing_label, "repeat": repeat_id,
                    "outer_fold": outer_fold_idx, "seed": fold_seed,
                    "subject_id": subj, "y_true": int(yt), "y_pred": int(yp), "y_proba": float(ypr),
                })

    return prediction_rows, fold_records, coral_rows
