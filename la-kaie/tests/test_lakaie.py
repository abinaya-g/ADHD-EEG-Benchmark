import numpy as np
import pytest

from lakaie.algorithms import lakaie
from lakaie.benchmarks.fir import FIRProblem
from lakaie.components import (ACTIONS, BanditController, LandscapeState, estimate_interaction,
                               group_variables, interaction_consistency)
from lakaie.core import CountedObjective, Recorder
from lakaie.experiment import lakaie_config, load_yaml

VARIANTS = [v for v in load_yaml("ablation.yaml")["variants"] if v != "FIRPrior"]


def _run(fn, dim=10, max_fe=3000, variant="Full", seed=5, problem=None, overrides=None):
    lb, ub = -5 * np.ones(dim), 5 * np.ones(dim)
    if problem is not None:
        lb, ub = problem.lb, problem.ub
    obj = CountedObjective(fn, max_fe)
    rec = Recorder(lb, ub)
    out = lakaie.run(obj, lb, ub, dim, 30, np.random.default_rng(seed),
                     lakaie_config(variant, overrides), rec, max_fe, problem=problem)
    return obj, out, rec


def rot_ellipsoid(X):
    D = X.shape[1]
    R = np.linalg.qr(np.random.default_rng(0).normal(size=(D, D)))[0]
    return np.sum(10 ** np.linspace(0, 3, D) * ((X - 1) @ R.T) ** 2, axis=1)


@pytest.mark.parametrize("max_fe", [2000, 2917, 5003])
def test_budget_exact_and_never_exceeded(max_fe):
    obj, out, _ = _run(rot_ellipsoid, max_fe=max_fe)
    assert obj.fe == max_fe
    assert out["best"] == pytest.approx(obj.best)


@pytest.mark.parametrize("variant", VARIANTS)
def test_all_variants_run(variant):
    obj, out, rec = _run(rot_ellipsoid, variant=variant, max_fe=2500)
    sm = out["summary"]
    assert obj.fe == 2500 and np.isfinite(out["best"])
    assert sum(sm["action_counts"].values()) == len(out["traces"]["ep_action"])
    if variant == "FixedPolicy":
        assert set(np.unique(out["traces"]["ep_action"])) <= {ACTIONS.index("A4"), ACTIONS.index("A2")}
    if variant == "RandomExchange":
        assert sm["random_selection_fraction"] == 1.0
    if variant == "DimensionOnly":
        assert sm["group_level_fraction"] == 0.0


def test_determinism():
    a = _run(rot_ellipsoid, seed=9)[1]
    b = _run(rot_ellipsoid, seed=9)[1]
    assert a["best"] == b["best"]
    assert np.array_equal(a["traces"]["ep_action"], b["traces"]["ep_action"])


def test_shift_invariance():
    """LA-KAIE uses only comparisons, ranks and ratios of differences, so an
    additive shift of f (e.g. the CEC optimum F*) cannot change its behaviour.
    Dyadic objective values make the shifted arithmetic exact."""
    def g(X):
        return np.round(np.sum((X - 1) ** 2, axis=1) * 1024) / 1024
    a = _run(g, seed=4)[1]
    b = _run(lambda X: g(X) + 4096.0, seed=4)[1]
    assert np.array_equal(a["traces"]["ep_action"], b["traces"]["ep_action"])
    assert b["best"] - 4096.0 == a["best"]


def test_estimator_separable_vs_coupled():
    rng = np.random.default_rng(1)
    D = 10; c = np.zeros(D); w = np.full(D, 10.0)
    X = rng.normal(0, 1, (200, D))
    sep = np.sum(np.arange(1, D + 1) * X ** 2, 1)
    G = estimate_interaction(X, sep, c, w)
    # separable: at most a few spurious pairs survive the |t|>=3 filter (45 pairs tested)
    assert np.count_nonzero(np.triu(G, 1)) <= 2
    blk = np.sum(X ** 2, 1) + 3 * X[:, 0] * X[:, 1]
    G = estimate_interaction(X, blk, c, w)
    assert G[0, 1] > 0.5 and G[0, 1] == G.max() and np.count_nonzero(np.triu(G, 1)) <= 2
    assert any({0, 1} <= set(g) for g in group_variables(G, 0.3, 5))
    assert estimate_interaction(X[:20], blk[:20], c, w) is None     # too few points


def test_grouping_deterministic_and_bounded():
    rng = np.random.default_rng(2)
    R = rng.random((12, 12)); G = np.triu(R, 1); G = G + G.T
    g1, g2 = group_variables(G, 0.3, 4), group_variables(G, 0.3, 4)
    assert g1 == g2
    assert max(len(g) for g in g1) <= 4
    assert sorted(i for g in g1 for i in g) == list(range(12))


def test_interaction_consistency():
    G = np.zeros((4, 4)); G[0, 1] = G[1, 0] = 0.9
    assert interaction_consistency(G, 0.3, np.array([0, 1])) == 1.0
    assert interaction_consistency(G, 0.3, np.array([0])) == -1.0
    assert interaction_consistency(G, 0.3, np.array([2])) == 0.0


def test_state_in_unit_interval_and_no_future_info():
    lb, ub = -np.ones(5), np.ones(5)
    st = LandscapeState(lakaie_config()["state"], lb, ub)
    rng = np.random.default_rng(0)
    for k in range(30):
        pops = [rng.uniform(-1, 1, (30, 5)) * (0.9 ** k) for _ in range(2)]
        f = rng.random(60) + 10 - k * 0.1
        st.end_iteration(float(f.min()), f, 0.3, 0.2, 0.1)
        v = st.vector(pops, f, float(f.min()), k / 30)
        assert v.shape == (10,) and np.all(v >= 0) and np.all(v <= 1)


def test_controller_features_and_learning():
    cfg = lakaie_config()["controller"]
    ctl = BanditController(dict(cfg, epsilon0=0.0, epsilon_min=0.0), np.random.default_rng(0))
    s = np.full(10, 0.5)
    x = ctl.features(s)
    assert x.sum() == 11 and x.size == 10 * cfg["bins"] + 1
    for _ in range(50):
        ctl.update(s, 3, 1.0)
        ctl.update(s, 5, -1.0)
    mask = np.ones(8, bool)
    assert ctl.select(s, mask, 0.5) == 3
    mask[3] = False
    assert ctl.select(s, mask, 0.5) != 3


def test_fir_prior_variant_runs():
    p = FIRProblem(2)
    obj, out, _ = _run(p, dim=31, variant="FIRPrior", problem=p, max_fe=4000)
    assert obj.fe == 4000
