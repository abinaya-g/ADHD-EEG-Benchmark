import numpy as np
import pandas as pd

from lakaie.analysis.stats import friedman_block_analysis, holm, kendall_w, paired_instance_tests


def test_holm_known():
    adj = holm([0.01, 0.04, 0.03, 0.005])
    assert np.allclose(adj, [0.03, 0.06, 0.06, 0.02])


def test_kendall_w_perfect_and_null():
    R = np.tile(np.arange(1, 5), (6, 1)).astype(float)
    assert abs(kendall_w(R) - 1.0) < 1e-12
    R2 = np.array([[1, 2, 3], [3, 2, 1]], float)
    assert abs(kendall_w(R2)) < 1e-12


def _df():
    rng = np.random.default_rng(0)
    rows = []
    for inst in range(1, 7):
        for r in range(1, 21):
            for m, shift in [("A", 0.0), ("B", 1.0), ("C", 2.0)]:
                rows.append({"instance": inst, "run_id": r, "method": m,
                             "best_fitness": shift + rng.normal(0, 0.1)})
    return pd.DataFrame(rows)


def test_friedman_and_posthoc():
    omni, post, M, R = friedman_block_analysis(_df(), ["A", "B", "C"], "A")
    assert omni["kendall_w"] == 1.0 and omni["friedman_p"] < 0.01
    assert omni["avg_ranks"]["A"] == 1.0
    assert set(post.method) == {"B", "C"} and (post["p_holm"] >= post["p_raw"]).all()


def test_paired_tests_direction():
    T = paired_instance_tests(_df(), ["A", "B", "C"], "A")
    assert (T.anchor_wins == 20).all() and (T.significant_favours == "anchor").all()
