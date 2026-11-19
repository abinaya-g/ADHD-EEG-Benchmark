"""
128 Hz versus 128->512 Hz interpolation over all five repetitions.

The primary results contain the interpolated condition for repetition 0 only
(results_final/predictions/all_predictions.csv). The additional repetitions 1-4
come from run_revision_deep.py --task resolution (results_revision/
resolution_interp_predictions.csv). Both conditions then cover the same five
repetitions and the same 121 children.

Usage: python -m src.resolution_five_repetitions
"""
import argparse

import numpy as np
import pandas as pd

from src.revision_analyses import RNG_SEED, ba, holm, paired, subject_table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="results_final/predictions/all_predictions.csv")
    ap.add_argument("--extra", default="results_revision/resolution_interp_predictions.csv")
    ap.add_argument("--out", default="results_revision")
    a = ap.parse_args()
    base = pd.read_csv(a.pred)
    extra = pd.read_csv(a.extra)
    rng = np.random.default_rng(RNG_SEED)
    per_rows, pool_rows = [], []
    for m in ["CNN", "EEGNet"]:
        d128 = base[(base.model == m) & (base.preprocessing == "128Hz")]
        i0 = base[(base.model == m) & (base.preprocessing == "128to512_interp") & (base["repeat"] == 0)]
        ix = extra[extra.model == m]
        dint = pd.concat([i0, ix], ignore_index=True)
        reps = sorted(set(d128["repeat"]) & set(dint["repeat"]))
        assert reps == [0, 1, 2, 3, 4], reps
        for r in reps:
            s1 = subject_table(d128[d128["repeat"] == r])
            s2 = subject_table(dint[dint["repeat"] == r])
            per_rows.append(dict(model=m, repeat=r, ba_128=ba(s1.y, s1.yhat), ba_interp=ba(s2.y, s2.yhat)))
        res = paired(subject_table(d128[d128["repeat"].isin(reps)]), subject_table(dint[dint["repeat"].isin(reps)]), rng)
        pr = pd.DataFrame([x for x in per_rows if x["model"] == m])
        pr["d"] = pr.ba_128 - pr.ba_interp
        pool_rows.append(dict(model=m, pooled_ba_128=res["ba_a"], pooled_ba_interp=res["ba_b"], diff=res["diff"],
                              ci_low=res["ci_low"], ci_high=res["ci_high"], p_perm=res["p_perm"], n_subjects=res["n"],
                              n_discordant=res["n_discordant"],
                              mean_single_run_128=pr.ba_128.mean(), sd_single_run_128=pr.ba_128.std(ddof=1),
                              mean_single_run_interp=pr.ba_interp.mean(), sd_single_run_interp=pr.ba_interp.std(ddof=1),
                              mean_diff_single_run=pr.d.mean(), sd_diff_single_run=pr.d.std(ddof=1),
                              reps_interp_higher=int((pr.d < 0).sum())))
    per = pd.DataFrame(per_rows)
    pool = pd.DataFrame(pool_rows)
    pool["p_holm"] = holm(pool.p_perm)
    per.to_csv(f"{a.out}/E6_resolution_per_repetition.csv", index=False)
    pool.to_csv(f"{a.out}/E6_resolution_five_repetitions.csv", index=False)
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print(per.round(3).to_string(index=False))
    print(pool.round(4).T.to_string())


if __name__ == "__main__":
    main()
