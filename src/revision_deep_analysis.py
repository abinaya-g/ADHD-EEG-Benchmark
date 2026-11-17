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
    import glob
    leaky = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{dir_}/leakage_epoch_level_predictions_*.csv"))])
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


def _holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    out = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        out[i] = min(1.0, running)
    return out


def _subject_table(d):
    return d.groupby("subject_id").agg(y=("y_true", "first"), p=("y_proba", "mean"))


def _paired_perm(y, pa, pb, n=20000, seed=12345):
    """Two-sided paired permutation test on the balanced-accuracy difference (subject-level)."""
    rng = np.random.default_rng(seed)
    ya = (pa >= .5).astype(int)
    yb = (pb >= .5).astype(int)
    obs = ba(y, ya) - ba(y, yb)
    cnt = 0
    for _ in range(n):
        sw = rng.random(len(y)) < .5
        a2, b2 = np.where(sw, yb, ya), np.where(sw, ya, yb)
        cnt += abs(ba(y, a2) - ba(y, b2)) >= abs(obs) - 1e-12
    return obs, (1 + cnt) / (n + 1)


def matched_head(dir_, pred):
    """Plain CNN trunk + regularised linear head (matched control) vs plain CNN and CNN+LR."""
    h = pd.read_csv(f"{dir_}/matched_head_predictions.csv")
    base = pd.read_csv(pred)
    base = base[base.preprocessing == "128Hz"]
    rows = []
    for name, d in (("CNN_linear_head (matched control)", h), ("CNN", base[base.model == "CNN"]), ("CNN+LR", base[base.model == "CNN+LR"])):
        d = d[d["repeat"].isin(sorted(h["repeat"].unique()))]
        per = []
        for r in sorted(d["repeat"].unique()):
            s = _subject_table(d[d["repeat"] == r])
            per.append((ba(s.y, s.p >= .5), roc_auc_score(s.y, s.p)))
        pool = _subject_table(d)
        rows.append(dict(model=name, n_repeats=len(per), single_run_ba_mean=np.mean([a for a, _ in per]), single_run_ba_sd=np.std([a for a, _ in per], ddof=1) if len(per) > 1 else np.nan,
                         single_run_auc_mean=np.mean([b for _, b in per]), pooled_ba=ba(pool.y, pool.p >= .5), pooled_auc=roc_auc_score(pool.y, pool.p),
                         per_repeat_ba=" ".join(f"{a:.3f}" for a, _ in per)))
    t = pd.DataFrame(rows)
    t.to_csv(f"{dir_}/E5_matched_head_summary.csv", index=False)
    hp = _subject_table(h)
    tests = []
    for other in ("CNN", "CNN+LR"):
        o = _subject_table(base[(base.model == other) & base["repeat"].isin(sorted(h["repeat"].unique()))])
        j = hp.join(o, lsuffix="_h", rsuffix="_o", how="inner")
        d, p = _paired_perm(j.y_h.values, j.p_h.values, j.p_o.values)
        tests.append(dict(comparison=f"CNN_linear_head - {other}", pooled_ba_difference=d, p_permutation=p, n_subjects=len(j)))
    tt = pd.DataFrame(tests)
    tt["p_holm"] = _holm(tt.p_permutation.values)
    tt.to_csv(f"{dir_}/E5_matched_head_tests.csv", index=False)
    return t, tt


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results_revision")
    ap.add_argument("--pred", default="results_final/predictions/all_predictions.csv")
    a = ap.parse_args()
    pd.set_option("display.width", 220, "display.max_columns", 30)
    import glob
    if glob.glob(f"{a.dir}/leakage_epoch_level_predictions_*.csv"):
        print(leakage(a.dir, a.pred).round(3).to_string(index=False))
    if os.path.exists(f"{a.dir}/coral_lambda_sweep_folds.csv"):
        print(coral(a.dir).round(4).to_string(index=False))
    if os.path.exists(f"{a.dir}/matched_head_predictions.csv"):
        mh, mt = matched_head(a.dir, a.pred)
        print(mh.round(3).to_string(index=False))
        print(mt.round(4).to_string(index=False))
