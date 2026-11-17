"""
Artefact analysis for the ADHD-EEG benchmark (needs the raw EEG, CPU only).

The preprocessing in the paper is per-epoch z-scoring only: no filtering,
ocular/muscular/movement artefact rejection or correction. This script
quantifies how much artefact-like activity there is, whether it differs between
groups, whether it relates to the models' outputs, and whether a classical
baseline changes when the worst epochs are removed. It makes no claim to
identify the artefact type; the measures are generic proxies.

Per-epoch proxies (computed on the mean-removed raw epoch, then summarised over
the 19 channels; sampling rate assumed 128 Hz as in the paper):
  p2p_median   median over channels of peak-to-peak amplitude (raw units)
  kurt_mean    mean over channels of excess kurtosis (spiky/blink-like activity)
  emg_share    median over channels of 30-45 Hz power / 0.25-45 Hz power
  drift_share  median over channels of 0.25-1 Hz power / 0.25-45 Hz power

Usage (from the repository root):
    ADHD_EEG_DATA_ROOT=/path/to/adhd-eeg-dataset python -m src.artefact_analysis
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
from scipy.stats import kurtosis, mannwhitneyu, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import data
from run_revision_experiments import band_features, run_cv, summarise

FS = 128
PROXIES = ["p2p_median", "kurt_mean", "emg_share", "drift_share"]
MAIN_MODELS = ["EEGNet", "ShallowConvNet", "CNN+LR", "CNN+LR_withCORAL", "CNN+NLSVM",
               "CNN+KNN", "CNN+RF", "CNN+LinearSVM", "DeepConvNet", "CNN", "CNN+GNB"]


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    out = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        out[i] = min(1.0, running)
    return out


def epoch_proxies(X, fs=FS):
    x = X[..., 0].astype(float)
    xc = x - x.mean(-1, keepdims=True)
    p2p = np.ptp(xc, axis=-1)
    kurt = kurtosis(xc, axis=-1)
    f, P = welch(xc, fs=fs, nperseg=fs * 4, axis=-1)
    tot = P[..., (f >= 0.25) & (f <= 45)].sum(-1) + 1e-12
    emg = P[..., (f >= 30) & (f <= 45)].sum(-1) / tot
    drift = P[..., (f >= 0.25) & (f < 1)].sum(-1) / tot
    return pd.DataFrame(dict(p2p_median=np.median(p2p, 1), kurt_mean=kurt.mean(1),
                             emg_share=np.median(emg, 1), drift_share=np.median(drift, 1)))


def group_comparison(sub):
    rows = []
    for k in PROXIES:
        a, b = sub.loc[sub.y == 1, k], sub.loc[sub.y == 0, k]
        u = mannwhitneyu(a, b, alternative="two-sided")
        rows.append(dict(proxy=k, adhd_median=a.median(), adhd_q1=a.quantile(.25), adhd_q3=a.quantile(.75),
                         control_median=b.median(), control_q1=b.quantile(.25), control_q3=b.quantile(.75),
                         auc_for_adhd=roc_auc_score(sub.y, sub[k]), p=u.pvalue))
    t = pd.DataFrame(rows)
    t["p_holm"] = holm(t.p)
    return t


def artefact_only_classifier(sub, n_perm=1000, seed=0):
    """Can the artefact proxies alone separate ADHD from controls (5-fold CV over children)?"""
    X, y = sub[PROXIES].values, sub.y.values

    def cv_auc(yv, s):
        prob = np.zeros(len(yv))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X, yv):
            m = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000))
            m.fit(X[tr], yv[tr])
            prob[te] = m.predict_proba(X[te])[:, 1]
        return roc_auc_score(yv, prob), balanced_accuracy_score(yv, prob >= .5)

    obs = [cv_auc(y, s) for s in range(20)]
    auc, ba = float(np.mean([o[0] for o in obs])), float(np.mean([o[1] for o in obs]))
    rng = np.random.default_rng(seed)
    null = [cv_auc(rng.permutation(y), 0)[0] for _ in range(n_perm)]
    p = (1 + sum(n >= auc for n in null)) / (n_perm + 1)
    return dict(cv_auc=auc, cv_balanced_accuracy=ba, perm_p=p, n_perm=n_perm, null_95th=float(np.percentile(null, 95)))


def model_correlations(sub, pred_path):
    d = pd.read_csv(pred_path)
    d = d[d.preprocessing == "128Hz"]
    rows = []
    for m in MAIN_MODELS:
        s = d[d.model == m].groupby("subject_id").y_proba.mean()
        j = sub.set_index("subject_id").join(s.rename("p"), how="inner")
        for k in PROXIES:
            r = spearmanr(j[k], j.p)
            rows.append(dict(model=m, proxy=k, spearman_rho=r.statistic, p=r.pvalue, n=len(j)))
    t = pd.DataFrame(rows)
    t["p_holm"] = holm(t.p)
    return t


def far_out(v):
    q1, q3 = np.percentile(v, [25, 75])
    return q1 - 3 * (q3 - q1), q3 + 3 * (q3 - q1)


def bandpower_sensitivity(X, y, g, prox, repeats=5):
    """Subject-grouped band-power baseline with and without removing artefact-flagged epochs."""
    lo_p, hi_p = far_out(prox.p2p_median)
    lo_k, hi_k = far_out(prox.kurt_mean)
    flag = ((prox.p2p_median > hi_p) | (prox.kurt_mean > hi_k)).values
    F_all = band_features(X)
    rows, counts = [], []
    for name, keep in (("no rejection (as in the paper)", np.ones(len(y), bool)),
                       ("far-out epochs removed (p2p or kurtosis)", ~flag)):
        yk, gk, Fk = y[keep], g[keep], F_all[keep]
        counts.append(dict(variant=name, epochs=int(keep.sum()), subjects=int(len(set(gk))),
                           adhd_epochs=int((yk == 1).sum()), control_epochs=int((yk == 0).sum()),
                           removed_adhd=int(((y == 1) & ~keep).sum()), removed_control=int(((y == 0) & ~keep).sum())))
        for r in range(repeats):
            prob = run_cv(Fk, yk, gk, "subject", 42 + r)
            rows.append(dict(variant=name, repeat=r, **summarise(yk, gk, prob)))
    t = pd.DataFrame(rows)
    s = t.groupby("variant")[["epoch_ba", "epoch_auc", "subject_ba", "subject_auc"]].agg(["mean", "std"])
    s.columns = ["_".join(c) for c in s.columns]
    return t, s.reset_index(), pd.DataFrame(counts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="results_final/predictions/all_predictions.csv")
    ap.add_argument("--out", default="results_revision")
    ap.add_argument("--n-perm", type=int, default=1000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    X, y, g, _ = data.load_dataset()
    prox = epoch_proxies(X)
    prox["y"], prox["subject_id"] = y, g
    sub = prox.groupby("subject_id").agg(y=("y", "first"), **{k: (k, "mean") for k in PROXIES}).reset_index()
    sub["n_epochs"] = pd.Series(g).value_counts().reindex(sub.subject_id).values

    gc = group_comparison(sub); gc.to_csv(f"{a.out}/artefact_group_comparison.csv", index=False)
    ao = artefact_only_classifier(sub, a.n_perm); json.dump(ao, open(f"{a.out}/artefact_only_classifier.json", "w"), indent=1)
    mc = model_correlations(sub, a.pred); mc.to_csv(f"{a.out}/artefact_model_correlations.csv", index=False)
    per, summ, counts = bandpower_sensitivity(X, y, g, prox)
    per.to_csv(f"{a.out}/artefact_bandpower_sensitivity_per_repeat.csv", index=False)
    summ.to_csv(f"{a.out}/artefact_bandpower_sensitivity.csv", index=False)
    counts.to_csv(f"{a.out}/artefact_rejection_counts.csv", index=False)
    json.dump(dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                   sklearn=sklearn.__version__, pandas=pd.__version__, fs_assumed_hz=FS),
              open(f"{a.out}/versions_artefact.json", "w"), indent=1)
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print("\n== group comparison (subject level)"); print(gc.round(4).to_string(index=False))
    print("\n== artefact-only classifier"); print(ao)
    print("\n== model correlations: max |rho| %.3f; Holm p<0.05: %d of %d" % (mc.spearman_rho.abs().max(), int((mc.p_holm < .05).sum()), len(mc)))
    print(mc.sort_values("p").head(6).round(4).to_string(index=False))
    print("\n== band-power sensitivity"); print(summ.round(3).to_string(index=False)); print(counts.to_string(index=False))


if __name__ == "__main__":
    main()
