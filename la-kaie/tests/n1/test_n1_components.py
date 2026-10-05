"""Component tests: T-BIC, T-PREQ (partition), T-E4-OBS, T-MAP, T-DEGEN,
T-SPRT-UNIT, native control, determinism."""
import math

import numpy as np
import pytest
from scipy.stats import multivariate_normal

from n1.config import N1Config
from n1.mapping import apply_mask, draw_mask, pick_receiver
from n1.native_control import NativeControl
from n1.records import StructuralWindow, rows_digest
from n1.selectors import Hysteresis, evaluate_selector, evaluate_sources, joint_half_sources
from n1.sprt import INFORMATIVE, UNINFORMATIVE, SPRTStyleRule, p1_from_p0
from n1.structure import (degeneracy, fit_model, loglik_from_scatter, partition_from_stat,
                          fisher_stat, scatter)

CFG = N1Config()


def _block_cov(D, blocks, rng):
    """Block-diagonal covariance with a random rotation and condition number 1e3
    inside each block (as in S2), so that within-block correlations are strong."""
    C = np.eye(D)
    for b in blocks:
        k = len(b)
        Q = np.linalg.qr(rng.standard_normal((k, k)))[0]
        C[np.ix_(b, b)] = Q @ np.diag(10.0 ** np.linspace(0, 3, k)) @ Q.T
    return C


def _windows(D, W, cov_H, cov_M, rng, n_rows=None):
    wH, wM = StructuralWindow(D, W), StructuralWindow(D, W)
    n = n_rows or 2 * W
    wH.append_native(rng.multivariate_normal(np.zeros(D), cov_H, n), "HBA", 0)
    wM.append_native(rng.multivariate_normal(np.zeros(D), cov_M, n), "MPA", 0)
    return wH, wM


# ------------------------------------------------------------------ T-BIC
def test_loglik_matches_scipy():
    rng = np.random.default_rng(0)
    Z = rng.standard_normal((80, 6)) @ rng.standard_normal((6, 6))
    mu, S = scatter(Z)
    for kind, part in (("HI", None), ("HR", None), ("HB", [np.arange(3), np.arange(3, 6)])):
        fit = fit_model(S, kind, part)
        ll = loglik_from_scatter(Z.shape[0], S, fit.Sigma, 1e-12)
        ref = multivariate_normal(mu, fit.Sigma).logpdf(Z).sum()
        assert ll == pytest.approx(ref, rel=1e-9, abs=1e-6)


def test_parameter_counts():
    D = 20
    S = np.eye(D)
    assert fit_model(S, "HI").q == 40
    assert fit_model(S, "HR").q == 20 + 210
    part = [np.arange(0, 5), np.arange(5, 10), np.arange(10, 20)]
    assert fit_model(S, "HB", part).q == 20 + 15 + 15 + 55


def test_bic_selection_rates_reported(capsys):
    """T-BIC: selection rates on Gaussian data at n = W are REPORTED, not asserted
    (spec §13.1); only the implementation is asserted elsewhere."""
    rng = np.random.default_rng(1)
    out = {}
    for D in (10, 20):
        W = CFG.W(D)
        blocks = [np.arange(i, min(i + 5, D)) for i in range(0, D, 5)]
        Q = np.linalg.qr(rng.standard_normal((D, D)))[0]
        c = 10 ** (np.linspace(0, 2, D))
        covs = {"HI": np.diag(c), "HB": _block_cov(D, blocks, rng), "HR": Q @ np.diag(c) @ Q.T}
        for truth, C in covs.items():
            hits = 0
            for _ in range(20):
                wH, wM = _windows(D, W, C, C, rng)
                hits += evaluate_selector("JOINT", wH, wM, CFG).k_hat == truth
            out[(D, truth)] = hits / 20
    print("T-BIC JOINT selection rate at n=W:", out)
    assert set(out) == {(10, "HI"), (10, "HB"), (10, "HR"), (20, "HI"), (20, "HB"), (20, "HR")}


# ------------------------------------------------------------------ T-PREQ
def test_old_cur_split_is_disjoint_and_ordered():
    D, W = 3, 4
    w = StructuralWindow(D, W)
    w.append_native(np.arange(3 * W * D, dtype=float).reshape(3 * W, D), "HBA", 0)
    old, cur = w.old(), w.cur()
    assert old[-1, 0] < cur[0, 0]                      # OLD strictly earlier than CUR
    np.testing.assert_array_equal(cur[-1], np.arange(3 * W * D).reshape(3 * W, D)[-1])


def test_partition_invariant_to_cur():
    rng = np.random.default_rng(3)
    D, W = 10, 60
    C = _block_cov(D, [np.arange(5), np.arange(5, 10)], rng)
    old_H = rng.multivariate_normal(np.zeros(D), C, W)
    old_M = rng.multivariate_normal(np.zeros(D), C, W)
    base = {}
    for trial in range(3):
        cur_H = rng.standard_normal((W, D)) * (trial + 1)
        cur_M = rng.standard_normal((W, D)) * (trial + 1)
        res = evaluate_sources("JOINT", {"H": (cur_H, old_H), "M": (cur_M, old_M)}, "stouffer", CFG)
        base[trial] = [b.tolist() for b in res.partition]
    assert base[0] == base[1] == base[2]


# ------------------------------------------------------------------ T-E4-OBS
def test_joint_pooled_identical_observations():
    rng = np.random.default_rng(4)
    D, W = 10, 60
    wH, wM = _windows(D, W, np.eye(D), 2 * np.eye(D), rng)
    j = evaluate_selector("JOINT", wH, wM, CFG)
    p = evaluate_selector("POOLED", wH, wM, CFG)
    assert j.digest == p.digest and j.digest == rows_digest(wH.cur(), wM.cur())
    assert p.n["P"] == j.n["H"] + j.n["M"] == 2 * W
    half = joint_half_sources(wH.cur(), wH.old(), wM.cur(), wM.old())
    assert sum(c.shape[0] for c, _ in half.values()) == W


def test_selector_determinism():
    rng = np.random.default_rng(5)
    wH, wM = _windows(10, 60, np.eye(10), np.eye(10), rng)
    for name in ("HBA", "MPA", "JOINT", "POOLED"):
        a, b = evaluate_selector(name, wH, wM, CFG), evaluate_selector(name, wH, wM, CFG)
        assert a.bic == b.bic and a.k_hat == b.k_hat
    a = evaluate_selector("DECOUPLED", wH, wM, CFG, np.random.default_rng(9))
    b = evaluate_selector("DECOUPLED", wH, wM, CFG, np.random.default_rng(9))
    assert a.bic == b.bic


# ------------------------------------------------------------------ T-DEGEN
def test_degenerate_hb_removed():
    assert degeneracy([np.array([i]) for i in range(4)]) == "HI"
    assert degeneracy([np.arange(4)]) == "HR"
    assert degeneracy([np.arange(2), np.arange(2, 4)]) is None
    rng = np.random.default_rng(6)
    D, W = 10, 60
    wH, wM = _windows(D, W, np.eye(D), np.eye(D), rng)            # independent -> singletons
    r = evaluate_selector("JOINT", wH, wM, CFG)
    assert r.degenerate == "HI" and "HB" not in r.available
    Q = np.linalg.qr(rng.standard_normal((D, D)))[0]
    C = Q @ np.diag(10 ** np.linspace(0, 3, D)) @ Q.T
    wH, wM = _windows(D, W, C, C, rng)                              # dense -> one block
    r = evaluate_selector("JOINT", wH, wM, CFG)
    assert r.degenerate == "HR" and "HB" not in r.available


def test_zero_variance_window():
    D, W = 5, 60
    wH, wM = StructuralWindow(D, W), StructuralWindow(D, W)
    wH.append_native(np.ones((2 * W, D)), "HBA", 0)
    wM.append_native(np.ones((2 * W, D)), "MPA", 0)
    r = evaluate_selector("JOINT", wH, wM, CFG)
    assert r.degenerate_window and not r.evidence
    h = Hysteresis(3.0)
    assert h.update(r) == ("HI", False, False)


def test_warmup_availability():
    rng = np.random.default_rng(7)
    D, W = 10, 60
    wH, wM = _windows(D, W, np.eye(D), np.eye(D), rng, n_rows=W - 1)
    assert not evaluate_selector("JOINT", wH, wM, CFG).evidence
    wH, wM = _windows(D, W, np.eye(D), np.eye(D), rng, n_rows=W + 10)
    r = evaluate_selector("JOINT", wH, wM, CFG)
    assert r.evidence and set(r.available) == {"HI", "HR"} and r.partition is None


def test_hysteresis():
    h = Hysteresis(3.0)
    from n1.selectors import SelectorResult
    r = SelectorResult("x", evidence=True, available=("HI", "HR"), bic={"HI": 100.0, "HR": 95.0}, k_hat="HR")
    assert h.update(r)[0] == "HI"              # 5 < 2*kappa = 6
    r.bic["HR"] = 93.0
    assert h.update(r)[0] == "HR"              # 7 > 6
    r2 = SelectorResult("x", evidence=True, available=("HI", "HB"), bic={"HI": 1.0, "HB": 2.0}, k_hat="HI")
    assert h.update(r2) == ("HI", True, True)  # forced: HR unavailable


def test_partition_recovers_blocks():
    rng = np.random.default_rng(8)
    D = 10
    C = _block_cov(D, [np.arange(5), np.arange(5, 10)], rng)
    Z = rng.multivariate_normal(np.zeros(D), C, 400)
    part = partition_from_stat(fisher_stat(Z, CFG.fisher_clip), CFG.alpha_link)
    assert [b.tolist() for b in part] == [list(range(5)), list(range(5, 10))]


# ------------------------------------------------------------------ T-MAP
def test_mapping_full_and_empty_masks():
    rng = np.random.default_rng(9)
    D = 8
    x_r, x_d = rng.uniform(-5, 5, D), rng.uniform(-5, 5, D)
    s = rng.uniform(0.5, 2, D)
    U = np.linalg.qr(rng.standard_normal((D, D)))[0]
    part = [np.arange(3), np.arange(3, 8)]
    for kind, k, kw in (("HI", D, {}), ("HR", D, {"U": U}), ("HB", 2, {"partition": part})):
        full = np.ones(k, bool) if kind == "HB" else np.ones(k)
        empty = np.zeros(k, bool) if kind == "HB" else np.zeros(k)
        np.testing.assert_allclose(apply_mask(kind, x_r, x_d, s, full, **kw), x_d, atol=1e-12)
        np.testing.assert_allclose(apply_mask(kind, x_r, x_d, s, empty, **kw), x_r, atol=1e-12)
    # HR with identity basis equals HI for the same mask
    m = (rng.random(D) < 0.5).astype(float)
    np.testing.assert_allclose(apply_mask("HR", x_r, x_d, s, m, U=np.eye(D)),
                               apply_mask("HI", x_r, x_d, s, m), atol=1e-12)


def test_mask_nonempty_and_tournament():
    rng = np.random.default_rng(10)
    for _ in range(200):
        assert draw_mask(rng, 3, 0.0).sum() == 1
    F = np.array([3.0, 1.0, 1.0, 5.0])
    for _ in range(200):
        r = pick_receiver(rng, F)
        assert r in (0, 1, 2, 3)


# ------------------------------------------------------------------ T-SPRT-UNIT
@pytest.mark.parametrize("p0", [0.1, 0.3])
def test_sprt_unit_idealised(p0):
    """Implementation check under idealised Wald assumptions (iid, known p0, no
    truncation/reset in effect): first-decision error rates <= nominal + 3 SE."""
    cfg = N1Config(n_min=1, n_max=10 ** 9)
    rng = np.random.default_rng(11)
    reps = 2000

    def first_decision(p_true):
        rule = SPRTStyleRule(cfg)
        while True:
            d = rule.update(p0, int(rng.random() < p_true))
            if d is not None:
                return d
    p1 = p1_from_p0(p0, cfg.delta)
    fa = np.mean([first_decision(p0) == INFORMATIVE for _ in range(reps)])
    fs = np.mean([first_decision(p1) == UNINFORMATIVE for _ in range(reps)])
    se = math.sqrt(0.05 * 0.95 / reps)
    assert fa <= 0.05 + 3 * se and fs <= 0.05 + 3 * se


def test_sprt_min_samples_truncation_reset():
    cfg = N1Config()
    rule = SPRTStyleRule(cfg)
    for i in range(cfg.n_min - 1):
        assert rule.update(0.2, 0) is None
    rule = SPRTStyleRule(cfg)
    # alternate outcomes at p = p0: stays between the bounds until n_max
    d = None
    for i in range(cfg.n_max):
        d = rule.update(0.5, i % 2)
        if d:
            break
    assert d is not None and rule.decisions[-1]["n"] <= cfg.n_max
    assert rule.llr == 0.0 and rule.n == 0


# ------------------------------------------------------------------ native control
def test_native_control_fit_and_separation():
    cfg = N1Config()
    rng = np.random.default_rng(12)
    n = 900
    l = rng.normal(0, 1, n)
    pop = rng.integers(0, 2, n)
    p = 1 / (1 + np.exp(-(-1.0 - 1.5 * l + 0.5 * pop)))
    y = (rng.random(n) < p).astype(int)
    nc = NativeControl(cfg)
    nc.fit(l, pop, y)
    assert nc.theta[1] == pytest.approx(-1.5, abs=0.35)
    p0, lc, oos = nc.predict(100.0, True)
    assert oos and lc == nc.l_hi and cfg.p_clip <= p0 <= 1 - cfg.p_clip
    nc.fit(l, pop, np.zeros(n, int))
    assert nc.separation and nc.predict(0.0, False)[0] == pytest.approx(0.5 / (n + 1), rel=1e-6)
