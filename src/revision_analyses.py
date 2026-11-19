"""
Analyses that need ONLY the saved predictions
(results_final/predictions/all_predictions.csv) -- no raw EEG, no GPU.

Contents:
  - per-repetition results; all-pairs comparisons among the leading models
    with Holm correction; label-permutation chance tests; confidence intervals.
  - EEGNet vs ShallowConvNet vs CNN+LR direct comparisons.
  - 128 Hz vs 128->512 Hz compared on the SAME repetition (repeat 0), the only
    repetition that has both conditions in the primary results.

IMPORTANT LIMITATION of the permutation test implemented here: it permutes
subject labels against FIXED saved predictions. It tests whether the final
subject-level predictions are associated with the labels beyond chance. It is
NOT the retrain-on-permuted-labels null (Ojala & Garwood 2010) that a full
nested-CV permutation test would give; that requires the raw data and is in
run_revision_experiments.py (--permutation).

Usage:  python -m src.revision_analyses [--pred PATH] [--out DIR]
"""
import argparse
import itertools
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

N_PERM = 10000
N_BOOT = 10000
RNG_SEED = 20260101
MAIN_MODELS = ["EEGNet", "ShallowConvNet", "CNN+LR", "CNN+LR_withCORAL", "CNN+NLSVM",
               "CNN+KNN", "CNN+RF", "CNN+LinearSVM", "DeepConvNet", "CNN", "CNN+GNB"]
LEADERS = ["EEGNet", "ShallowConvNet", "CNN+LR"]


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    out = np.empty_like(p)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i])
        out[i] = min(run, 1.0)
    return out


def subject_table(df):
    """One row per subject: mean probability over its epochs (and over any
    repetitions present in df), thresholded at 0.5."""
    g = df.groupby("subject_id").agg(y=("y_true", "first"), p=("y_proba", "mean"))
    g["yhat"] = (g["p"] >= 0.5).astype(int)
    return g


def ba(y, yhat):
    """Balanced accuracy (numpy; much faster than sklearn inside permutation loops)."""
    y, yhat = np.asarray(y), np.asarray(yhat)
    pos, neg = y == 1, y == 0
    return 0.5 * ((yhat[pos] == 1).mean() + (yhat[neg] == 0).mean())


def auc_fast(y, p):
    """Mann-Whitney AUC with tie handling."""
    from scipy.stats import rankdata
    y = np.asarray(y)
    r = rankdata(p)
    n1, n0 = (y == 1).sum(), (y == 0).sum()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def per_repetition(df128):
    rows = []
    for m in MAIN_MODELS:
        d = df128[df128.model == m]
        for r, dr in d.groupby("repeat"):
            s = subject_table(dr)
            rows.append(dict(model=m, repeat=int(r), n_subjects=len(s),
                             balanced_accuracy=ba(s.y, s.yhat),
                             auc=roc_auc_score(s.y, s.p)))
    t = pd.DataFrame(rows)
    summ = (t.groupby("model")
              .agg(ba_mean=("balanced_accuracy", "mean"), ba_sd=("balanced_accuracy", "std"),
                   ba_min=("balanced_accuracy", "min"), ba_max=("balanced_accuracy", "max"),
                   auc_mean=("auc", "mean"), auc_sd=("auc", "std"))
              .reindex(MAIN_MODELS).reset_index())
    return t, summ


def label_permutation(df128, rng):
    """Chance test on fixed subject-level predictions (see module docstring)."""
    rows = []
    for m in MAIN_MODELS:
        s = subject_table(df128[df128.model == m])
        y, yhat, p = s.y.to_numpy(), s.yhat.to_numpy(), s.p.to_numpy()
        obs_ba, obs_auc = ba(y, yhat), roc_auc_score(y, p)
        nb = np.empty(N_PERM)
        na = np.empty(N_PERM)
        for i in range(N_PERM):
            yp = rng.permutation(y)
            nb[i] = ba(yp, yhat)
            na[i] = auc_fast(yp, p)
        rows.append(dict(model=m, observed_ba=obs_ba, null_ba_mean=nb.mean(),
                         null_ba_95th=np.quantile(nb, .95),
                         p_ba=(1 + (nb >= obs_ba).sum()) / (N_PERM + 1),
                         observed_auc=obs_auc, null_auc_95th=np.quantile(na, .95),
                         p_auc=(1 + (na >= obs_auc).sum()) / (N_PERM + 1)))
    t = pd.DataFrame(rows)
    t["p_ba_holm"] = holm(t.p_ba)
    t["p_auc_holm"] = holm(t.p_auc)
    return t


def paired(sa, sb, rng):
    """Paired subject-level permutation test + bootstrap CI for BA difference."""
    idx = sa.index.intersection(sb.index)
    y = sa.loc[idx, "y"].to_numpy()
    a, b = sa.loc[idx, "yhat"].to_numpy(), sb.loc[idx, "yhat"].to_numpy()
    obs = ba(y, a) - ba(y, b)
    n = len(y)
    perm = np.empty(N_PERM)
    for i in range(N_PERM):
        sw = rng.random(n) < 0.5
        a2, b2 = np.where(sw, b, a), np.where(sw, a, b)
        perm[i] = ba(y, a2) - ba(y, b2)
    p = (1 + (np.abs(perm) >= abs(obs) - 1e-12).sum()) / (N_PERM + 1)
    bs = np.empty(N_BOOT)
    for i in range(N_BOOT):
        j = rng.integers(0, n, n)
        if len(np.unique(y[j])) < 2:
            bs[i] = np.nan
            continue
        bs[i] = ba(y[j], a[j]) - ba(y[j], b[j])
    lo, hi = np.nanquantile(bs, [.025, .975])
    return dict(n=n, ba_a=ba(y, a), ba_b=ba(y, b), diff=obs, ci_low=lo, ci_high=hi,
                p_perm=p, n_discordant=int((a != b).sum()))


def pairwise(df128, rng, models, label):
    tabs = {m: subject_table(df128[df128.model == m]) for m in models}
    rows = []
    for a, b in itertools.combinations(models, 2):
        r = paired(tabs[a], tabs[b], rng)
        rows.append(dict(model_a=a, model_b=b, **r))
    t = pd.DataFrame(rows)
    t["p_holm"] = holm(t.p_perm)
    t["family"] = label
    return t


def resolution_matched(df, rng):
    """128 Hz vs 128->512 Hz, repeat 0 only (the only repetition run at 512 Hz)."""
    rows = []
    for m in ["CNN", "EEGNet"]:
        sub = df[(df.model == m) & (df["repeat"] == 0)]
        a = subject_table(sub[sub.preprocessing == "128Hz"])
        b = subject_table(sub[sub.preprocessing == "128to512_interp"])
        r = paired(a, b, rng)
        # outer-fold level (n = 5 disjoint folds) as in the original Table 6
        fold_diffs = []
        for f, sf in sub.groupby("outer_fold"):
            fa = subject_table(sf[sf.preprocessing == "128Hz"])
            fb = subject_table(sf[sf.preprocessing == "128to512_interp"])
            fold_diffs.append(ba(fa.y, fa.yhat) - ba(fb.y, fb.yhat))
        rows.append(dict(model=m, ba_128=r["ba_a"], ba_interp=r["ba_b"], diff=r["diff"],
                         ci_low=r["ci_low"], ci_high=r["ci_high"], p_perm=r["p_perm"],
                         fold_diffs=";".join(f"{x:.3f}" for x in fold_diffs),
                         n_subjects=r["n"], repeats_used="repeat 0 only (both conditions)"))
    t = pd.DataFrame(rows)
    t["p_holm"] = holm(t.p_perm)
    return t


def confound_epochs(d128, rng):
    """Recording length (n 30-s epochs per subject) vs diagnosis, and whether
    each model's subject-level probability tracks recording length.
    Age / sex / artefact load are NOT in the released files (see plan)."""
    from scipy.stats import spearmanr, mannwhitneyu
    base = d128[(d128.model == "EEGNet") & (d128["repeat"] == 0)]
    n_ep = base.groupby("subject_id").size().rename("n_epochs")
    y = base.groupby("subject_id")["y_true"].first()
    t = pd.concat([n_ep, y], axis=1)
    out = [dict(item="n_epochs ADHD median (IQR)",
                value=f"{t[t.y_true==1].n_epochs.median():.0f} ({t[t.y_true==1].n_epochs.quantile(.25):.0f}-{t[t.y_true==1].n_epochs.quantile(.75):.0f})"),
           dict(item="n_epochs Control median (IQR)",
                value=f"{t[t.y_true==0].n_epochs.median():.0f} ({t[t.y_true==0].n_epochs.quantile(.25):.0f}-{t[t.y_true==0].n_epochs.quantile(.75):.0f})"),
           dict(item="Mann-Whitney p (n_epochs ADHD vs Control)",
                value=f"{mannwhitneyu(t[t.y_true==1].n_epochs, t[t.y_true==0].n_epochs).pvalue:.4g}"),
           dict(item="AUC of n_epochs alone for diagnosis", value=f"{auc_fast(t.y_true, t.n_epochs):.3f}")]
    for m in MAIN_MODELS:
        s = subject_table(d128[d128.model == m]).join(n_ep)
        rho, p = spearmanr(s.p, s.n_epochs)
        out.append(dict(item=f"{m}: Spearman(subject prob, n_epochs)", value=f"rho={rho:.3f}, p={p:.3g}"))
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="results_final/predictions/all_predictions.csv")
    ap.add_argument("--out", default="manuscript/TABLES/revision")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)
    df = pd.read_csv(a.pred)
    # CNN+LR_noCORAL is numerically identical to CNN+LR (manuscript Sec. 3.18)
    d128 = df[df.preprocessing == "128Hz"]

    rep, summ = per_repetition(d128)
    rep.to_csv(f"{a.out}/R1_per_repetition_results.csv", index=False)
    summ.to_csv(f"{a.out}/R1_per_repetition_summary.csv", index=False)

    perm = label_permutation(d128, rng)
    perm.to_csv(f"{a.out}/R2_label_permutation_chance_tests.csv", index=False)

    lead = pairwise(d128, rng, LEADERS, "leaders (3 pairs)")
    lead.to_csv(f"{a.out}/R3_leader_pairwise.csv", index=False)
    allp = pairwise(d128, rng, MAIN_MODELS, "all 11 models (55 pairs)")
    allp.to_csv(f"{a.out}/R3_all_pairs.csv", index=False)

    res = resolution_matched(df, rng)
    res.to_csv(f"{a.out}/R4_resolution_matched_repeat0.csv", index=False)

    conf = confound_epochs(d128, rng)
    conf.to_csv(f"{a.out}/R5_recording_length_confound.csv", index=False)
    print(conf.to_string(index=False)); print()

    pd.set_option("display.width", 200, "display.max_columns", 30)
    print(summ.round(3).to_string(index=False)); print()
    print(perm.round(4).to_string(index=False)); print()
    print(lead.round(4).to_string(index=False)); print()
    print(allp.sort_values("p_perm").head(12).round(4).to_string(index=False)); print()
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
