"""Summarises the outputs of run_revision_deep.py (leakage comparison, CORAL
lambda sweep, resolution extension). Usage after the Kaggle runs:
    python -m src.revision_deep_analysis --dir results_revision
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def ba(y, yh):
    y, yh = np.asarray(y), np.asarray(yh)
    return 0.5 * ((yh[y == 1] == 1).mean() + (yh[y == 0] == 0).mean())


def level_metrics(d):
    ep = dict(epoch_ba=ba(d.y_true, d.y_pred), epoch_auc=roc_auc_score(d.y_true, d.y_proba))
    s = d.groupby("subject_id").agg(y=("y_true", "first"), p=("y_proba", "mean"))
    return {**ep, "subject_ba": ba(s.y, s.p >= .5), "subject_auc": roc_auc_score(s.y, s.p)}


def leakage(dir_, pred):
    leaky = pd.read_csv(f"{dir_}/leakage_epoch_level_predictions.csv")
    base = pd.read_csv(pred)
    base = base[base.preprocessing == "128Hz"]
    rows = []
    for m in sorted(leaky.model.unique()):
        for r in sorted(leaky["repeat"].unique()):
            a = leaky[(leaky.model == m) & (leaky["repeat"] == r)]
            b = base[(base.model == m) & (base["repeat"] == r)]
            rows.append(dict(model=m, repeat=r, split="subject_grouped (paper)", **level_metrics(b)))
            rows.append(dict(model=m, repeat=r, split="epoch_level (leakage-permitting)", **level_metrics(a)))
    t = pd.DataFrame(rows)
    t.to_csv(f"{dir_}/E3_leakage_comparison.csv", index=False)
    return t


def coral(dir_):
    c = pd.read_csv(f"{dir_}/coral_lambda_sweep_folds.csv")
    t = (c.groupby("coral_lambda")
          .agg(cov_before=("cov_distance_before", "mean"), cov_after=("cov_distance_after", "mean"),
               acc_no_coral=("no_coral_accuracy", "mean"), acc_with_coral=("with_coral_accuracy", "mean"),
               ba_no_coral=("no_coral_balanced_accuracy", "mean"), ba_with_coral=("with_coral_balanced_accuracy", "mean"))
          .reset_index())
    t.to_csv(f"{dir_}/E4_coral_lambda_sweep_summary.csv", index=False)
    return t


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results_revision")
    ap.add_argument("--pred", default="results_final/predictions/all_predictions.csv")
    a = ap.parse_args()
    pd.set_option("display.width", 220, "display.max_columns", 30)
    if os.path.exists(f"{a.dir}/leakage_epoch_level_predictions.csv"):
        print(leakage(a.dir, a.pred).round(3).to_string(index=False))
    if os.path.exists(f"{a.dir}/coral_lambda_sweep_folds.csv"):
        print(coral(a.dir).round(4).to_string(index=False))
