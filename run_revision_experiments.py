"""
Experiments that need the RAW EEG: a classical band-power baseline, a leakage
comparison on an identical pipeline, and a retrain-on-permuted-labels null.
sklearn/scipy only -- no TensorFlow, CPU, minutes not hours.

  1. Classical feature baseline: log band power (delta/theta/alpha/beta/gamma
     per channel) + theta/beta ratio -> L2 logistic regression, with the
     regularisation strength chosen by an INNER subject-grouped CV.
  2. Leakage comparison on an otherwise IDENTICAL pipeline: the same features
     and classifier evaluated with (a) subject-grouped outer folds (as in the
     primary analysis) and (b) epoch-level random outer folds that let one child's epochs
     appear in both train and test. Reported at epoch level (as leaky studies
     do) and at subject level.
  3. Retrain-on-permuted-labels null for the classical baseline (subject
     labels permuted, whole nested pipeline re-run) -> true chance distribution.

The same split idea is applied to the deep models in run_revision_deep.py
(--task leakage), which needs TensorFlow and, ideally, a GPU.

Usage (real data):  ADHD_EEG_DATA_ROOT=... python run_revision_experiments.py
       (self-test):  python run_revision_experiments.py --synthetic
"""
import argparse
import json
import os
import platform

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}
FS = 128  # ASSUMED (unverified) sampling rate, as in the paper


def band_features(X, fs=FS):
    """X: (n_epochs, n_channels, n_samples, 1) -> (n_epochs, 5*C + C) features:
    log relative band power per band/channel + log theta/beta ratio per channel."""
    X = X[..., 0]
    f, P = welch(X, fs=fs, nperseg=min(X.shape[-1], fs * 4), axis=-1)
    tot = P[..., (f >= 1) & (f <= 45)].sum(-1) + 1e-12
    feats, bp = [], {}
    for name, (lo, hi) in BANDS.items():
        bp[name] = P[..., (f >= lo) & (f < hi)].sum(-1)
        feats.append(np.log(bp[name] / tot + 1e-12))
    tbr = np.log((bp["theta"] + 1e-12) / (bp["beta"] + 1e-12))
    return np.concatenate(feats + [tbr], axis=1)


def fit_predict(Ftr, ytr, gtr, Fte, seed):
    """Inner subject-grouped CV picks C; refit on all outer-train; predict test."""
    best_c, best = None, -1
    gkf = GroupKFold(n_splits=min(4, len(np.unique(gtr))))
    for c in (0.001, 0.01, 0.1, 1.0):
        sc = []
        for a, b in gkf.split(Ftr, ytr, gtr):
            if len(np.unique(ytr[a])) < 2 or len(np.unique(ytr[b])) < 2:
                continue
            m = make_pipeline(StandardScaler(), LogisticRegression(C=c, class_weight="balanced", max_iter=2000))
            m.fit(Ftr[a], ytr[a])
            sc.append(roc_auc_score(ytr[b], m.predict_proba(Ftr[b])[:, 1]))
        if sc and np.mean(sc) > best:
            best, best_c = np.mean(sc), c
    m = make_pipeline(StandardScaler(), LogisticRegression(C=best_c or 0.01, class_weight="balanced", max_iter=2000))
    m.fit(Ftr, ytr)
    return m.predict_proba(Fte)[:, 1]


def outer_splits(groups, y, mode, seed, k=5):
    n = len(y)
    if mode == "subject":  # class-balanced subject-grouped folds
        rng = np.random.default_rng(seed)
        subj = np.unique(groups)
        lab = {s: y[groups == s][0] for s in subj}
        fold = {}
        for c in (0, 1):
            ss = list(rng.permutation([s for s in subj if lab[s] == c]))
            for i, s in enumerate(ss):
                fold[s] = i % k
        fa = np.array([fold[g] for g in groups])
        return [(np.where(fa != f)[0], np.where(fa == f)[0]) for f in range(k)]
    return list(StratifiedKFold(k, shuffle=True, random_state=seed).split(np.zeros(n), y))  # epoch-level (leaky)


def run_cv(F, y, groups, mode, seed):
    prob = np.full(len(y), np.nan)
    for tr, te in outer_splits(groups, y, mode, seed):
        prob[te] = fit_predict(F[tr], y[tr], groups[tr], F[te], seed)
    return prob


def summarise(y, groups, prob):
    ep = dict(epoch_ba=balanced_accuracy_score(y, prob >= .5), epoch_auc=roc_auc_score(y, prob))
    d = pd.DataFrame(dict(g=groups, y=y, p=prob)).groupby("g").agg(y=("y", "first"), p=("p", "mean"))
    sj = dict(subject_ba=balanced_accuracy_score(d.y, d.p >= .5), subject_auc=roc_auc_score(d.y, d.p))
    return {**ep, **sj}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--n-perm", type=int, default=200, help="retrain-on-permuted-labels repetitions")
    ap.add_argument("--out", default="manuscript/TABLES/revision")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.synthetic:
        from src import data
        X, y, groups, _ = data.make_synthetic_dataset(n_adhd_subjects=30, n_control_subjects=30, n_timesamples=512)
        prefix = "SYNTHETIC_"
    else:
        from src import data
        X, y, groups, _ = data.load_dataset()
        X = data.normalize_all(X) if hasattr(data, "normalize_all") else X
        prefix = ""
    F = band_features(X)
    rows = []
    for r in range(a.repeats):
        for mode in ("subject", "epoch"):
            rows.append(dict(repeat=r, split=mode, **summarise(y, groups, run_cv(F, y, groups, mode, 42 + r))))
    res = pd.DataFrame(rows)
    res.to_csv(f"{a.out}/{prefix}E1_bandpower_leakage_comparison.csv", index=False)
    print(res.groupby("split").mean(numeric_only=True).round(3))

    # retrain-on-permuted-labels null (subject-grouped pipeline, subject-level BA)
    rng = np.random.default_rng(2026)
    subj = np.unique(groups)
    lab = {s: y[groups == s][0] for s in subj}
    null = []
    for i in range(a.n_perm):
        perm = dict(zip(subj, rng.permutation([lab[s] for s in subj])))
        yp = np.array([perm[g] for g in groups])
        null.append(summarise(yp, groups, run_cv(F, yp, groups, "subject", 42))["subject_ba"])
    obs = res[res.split == "subject"].subject_ba.iloc[0]
    p = (1 + (np.array(null) >= obs).sum()) / (len(null) + 1)
    pd.DataFrame(dict(null_subject_ba=null)).to_csv(f"{a.out}/{prefix}E2_bandpower_permutation_null.csv", index=False)
    print(f"observed subject BA (repeat 0) {obs:.3f}; null 95th pct {np.quantile(null, .95):.3f}; p={p:.4f}")

    json.dump(dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                   sklearn=sklearn.__version__, pandas=pd.__version__, seeds=[42 + r for r in range(a.repeats)],
                   assumed_fs_hz=FS, bands=BANDS), open(f"{a.out}/{prefix}E_versions.json", "w"), indent=1)


if __name__ == "__main__":
    main()
